"""
Agent 主循环 —— 把 LLM、工具、上下文三者编排起来。

⚠️ 这段逻辑和 v1 的 Agent.run() 几乎一模一样，v2、v3 都刻意没动结构。
   历次版本只在循环里插了两个钩子：
     · 请求前调用 context.prepare() 瘦身（v2 剪枝，v3 压缩）
     · 工具结果超长就地截断（v2）

   其余部分保持不变 —— 让每一版的「新概念」成为唯一的变量，
   一旦出问题你能立刻分清是重构引入的还是新功能引入的。
"""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.text import Text

import structured
from config import MAX_ROUNDS, MAX_TOOL_RESULT_CHARS
from context import ContextManager
from llm import LLMClient
from tools import TOOL_IMPLS

console = Console()


def _assistant_message(message: Any) -> dict[str, Any]:
    """把 SDK 返回的 assistant 消息转成可以回灌给接口的 dict。

    两个坑：
      · tool_calls 里的 arguments 必须是字符串且不能为空，空串要补成 "{}"
      · 带 tool_calls 时 content 允许为 null；不带 tool_calls 时不能为 null，
        否则下一轮请求会被接口拒绝
    """
    payload: dict[str, Any] = {"role": "assistant", "content": message.content}
    if not message.tool_calls and not payload["content"]:
        payload["content"] = ""
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments or "{}",
                },
            }
            for call in message.tool_calls
        ]
    return payload


class Agent:
    """一个带上下文管理 + 记忆压缩的最小 Agent。"""

    def __init__(self, client: LLMClient, context: ContextManager) -> None:
        self.client = client
        self.context = context

    def reset(self) -> None:
        self.context.reset()

    # ── 核心：主循环 ────────────────────────────────────────────────

    def run(self, user_input: str) -> str:
        self.context.append({"role": "user", "content": user_input})

        for _ in range(MAX_ROUNDS):
            # ★ 上下文钩子：每次请求模型之前先瘦身（剪枝 + 压缩）
            #
            #   压缩要调一次模型，可能要等几秒，所以给它一个「正在干活」的提示 ——
            #   否则用户只会觉得界面卡住了。
            report = self.context.prepare(
                on_compact=lambda: console.print(
                    "  [magenta]🗜 正在压缩旧对话…（这一步要额外调一次模型）[/magenta]"
                )
            )
            # ⚠️ 用 should_report 而不是 did_trim：
            #   「超了预算但什么都没动（因为没东西可动）」同样要说清楚，
            #    否则用户只会看到它悄悄超预算。
            if report.should_report:
                for line in report.describe_lines():
                    # ⚠️ 用 Text 包一层：报告里有方括号，直接交给 rich 会被当成样式标记吃掉
                    console.print(Text(f"  {line}", style="yellow"))

            reply = self.client.complete(self.context.messages, use_tools=True)
            message = reply.message
            self.context.append(_assistant_message(message))
            console.print(f"  [dim]· 本次请求 {reply.prompt_tokens:,} tokens[/dim]")

            # ★ 循环的终止条件：模型没要求调工具，说明它认为可以直接回答了
            if not message.tool_calls:
                return message.content or "（模型返回了空回答）"

            for call in message.tool_calls:
                self.context.append(self._execute_tool_call(call))

            # ★ v8：结构化结果一旦通过校验，直接收工 ——
            #   再问下去只会多烧一次调用（该说的都已经在 submit_result 里了）。
            #   运行器不关心这里返回的字符串，它要的是 structured.STATE.result。
            if structured.STATE.result is not None:
                return "（结构化结果已通过 submit_result 提交）"

        # 兜底：轮数用尽说明模型一直停不下来。
        # 撤掉 tools 再问最后一次 —— 没有工具可用，它就只能用手上的资料作答。
        console.print(f"[yellow]⚠ 已达最大轮数 {MAX_ROUNDS}，撤掉工具强制生成最终回答[/yellow]")
        self.context.prepare()
        reply = self.client.complete(self.context.messages, use_tools=False)
        return reply.message.content or "（超出最大轮数，未能得出结论）"

    # ── 辅助：执行一次工具调用 ──────────────────────────────────────

    def _execute_tool_call(self, call: Any) -> dict[str, Any]:
        """执行工具，并把结果包装成 role="tool" 消息。

        工具报错不能让程序崩掉，而是把错误当成「工具的输出」交回给模型 ——
        它读到错误往往能自己修正参数重试。Agent 的自我纠错能力就来自这里。
        """
        name = call.function.name
        raw_args = call.function.arguments or "{}"

        # v8：submit_result 的参数可能是几百字的 JSON —— 日志仍要自证清白，
        # 但没必要把屏幕刷爆。截断的只是**显示**，执行用的还是原文。
        shown = raw_args if len(raw_args) <= 220 else raw_args[:220] + "…（显示截断）"
        console.print(f"  [cyan]🔧 调用[/cyan] [bold]{name}[/bold]({shown})")

        try:
            args = json.loads(raw_args)
            impl = TOOL_IMPLS.get(name)
            if impl is None:
                raise ValueError(f"不存在名为 {name} 的工具")
            result = str(impl(**args))
        except structured.StructuredGiveUp:
            # v8：结构化提交超限 = 这次运行已判死刑。
            # 它不是「工具的一次失败」（那种应回灌给模型），而是放弃信号 ——
            # 放行，让它穿到运行器（main.py）那里收尸。
            raise
        except Exception as exc:
            result = f"错误：{type(exc).__name__}: {exc}"
            console.print(f"  [red]✗ {result}[/red]")
        else:
            console.print(f"  [green]← {result}[/green]")

        # ★ 上下文管理的第一道闸：工具结果过长就地截断。
        #   与其等它进了上下文再剪，不如一开始就别让它进来。
        if len(result) > MAX_TOOL_RESULT_CHARS:
            original_len = len(result)
            result = result[:MAX_TOOL_RESULT_CHARS] + f"\n…[结果过长已截断，原长 {original_len} 字符]"
            console.print(f"  [yellow]✂ 工具结果超长，已截断到 {MAX_TOOL_RESULT_CHARS} 字符[/yellow]")

        return {
            "role": "tool",
            "tool_call_id": call.id,  # 必须带上，用来和上面那条 assistant 配对
            "name": name,
            "content": result,
        }
