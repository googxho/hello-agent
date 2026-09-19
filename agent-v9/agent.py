"""
Agent 主循环 —— 骨架和 v1 一字不差，只加了一件东西：

    ⭐ 每一步都留下记录（trace）

v1 的循环把过程打印在屏幕上就完事；可一旦要「自动判定、批量跑分」，
屏幕上的字就不够用了 —— 程序需要拿到结构化的过程：

    这次提问调用了哪些工具？参数是什么？成功了吗？花了几次调用、多少 token？

这就是「可观测」的第一层：**把发生过的事，变成可查询的数据。**

（再往上一层是 trace 平台干的事 —— 见 ECOSYSTEM.md 的 07 节。）
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from config import MAX_ROUNDS
from tools import TOOL_IMPLS, openai_schemas


class ReplyLike(Protocol):
    """agent 只要求 client 有 complete()，返回带 .message/.prompt_tokens 的对象。"""

    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any: ...


# ══════════════════════════════════════════════════════════════════════
# 记录结构：一次提问攒下的全部事实
# ══════════════════════════════════════════════════════════════════════


@dataclass
class ToolTrace:
    """一次工具调用的痕迹。"""

    name: str
    args_text: str    # 模型给的原始参数（字符串，可能不合法的 JSON 也先留着）
    result_text: str  # 工具的输出；失败时是「错误：……」信息
    ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "args": self.args_text, "result": self.result_text, "ok": self.ok}


@dataclass
class AgentRun:
    """一次提问的完整记录 —— 评估器的输入之一。"""

    prompt: str
    answer: str = ""
    tools: list[ToolTrace] = field(default_factory=list)
    rounds: int = 0               # 实际调了几次模型
    forced_finish: bool = False   # 是否被 MAX_ROUNDS 兜底「强制收尾」
    elapsed_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self.tools]

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _assistant_message(message: Any) -> dict[str, Any]:
    """把 SDK 返回的 assistant 消息转成可以回灌给接口的 dict（v1 原样）。

    两个坑：
    · tool_calls 里的 arguments 必须是字符串且不能为空，有的模型给空串，
      直接回灌会被接口拒绝，所以补成 "{}"。
    · 带 tool_calls 的消息 content 允许为 null；不带 tool_calls 的
      assistant 消息 content 不能为 null。
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


# ══════════════════════════════════════════════════════════════════════
# Agent
# ══════════════════════════════════════════════════════════════════════


class Agent:
    """和 v1 同一个 Agent：messages 是唯一状态，循环跑到模型不再调工具为止。

    和 v1 的两点区别（都为评估服务）：
      · 构造时注入 system_prompt（提示词现在来自文件，可 A/B 换着跑）
      · on_tool 回调：REPL 用它实时打印；评估**不传** —— 免得屏幕被工具日志淹没
    """

    def __init__(
        self,
        client: ReplyLike,
        system_prompt: str,
        on_tool: Callable[[ToolTrace], None] | None = None,
    ) -> None:
        self.client = client
        self.system_prompt = system_prompt
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        # 构造那一刻取工具清单 —— 被关掉的工具根本不在这里
        self.tools = openai_schemas()
        self.on_tool = on_tool

    def reset(self) -> None:
        """清空对话历史，但保留 system 提示词。"""
        self.messages = [{"role": "system", "content": self.system_prompt}]

    # ── 核心：主循环（v1 的 40 行 + trace 记录）─────────────────────
    def ask(self, user_input: str) -> AgentRun:
        started = time.perf_counter()
        run = AgentRun(prompt=user_input)
        self.messages.append({"role": "user", "content": user_input})

        for _ in range(MAX_ROUNDS):
            reply = self.client.complete(self.messages, use_tools=True)
            run.rounds += 1
            run.prompt_tokens += reply.prompt_tokens
            run.completion_tokens += reply.completion_tokens
            message = reply.message

            self.messages.append(_assistant_message(message))

            # ★ 终止条件：这一轮没有 tool_calls —— 它自己决定收工了
            if not message.tool_calls:
                run.answer = message.content or "（模型返回了空回答）"
                run.elapsed_ms = (time.perf_counter() - started) * 1000
                return run

            # 有工具调用 → 替它执行，把结果和「痕迹」都记下来
            for call in message.tool_calls:
                tool_message, trace = self._execute_tool_call(call)
                self.messages.append(tool_message)
                run.tools.append(trace)
                if self.on_tool:
                    self.on_tool(trace)

        # 兜底：轮数用尽还在调工具，就撤掉工具再问最后一次（v1 老保险丝）
        reply = self.client.complete(self.messages, use_tools=False)
        run.rounds += 1
        run.prompt_tokens += reply.prompt_tokens
        run.completion_tokens += reply.completion_tokens
        run.forced_finish = True
        run.answer = reply.message.content or "（超出最大轮数，未能得出结论）"
        run.elapsed_ms = (time.perf_counter() - started) * 1000
        return run

    # ── 辅助：执行一次工具调用（v1 原逻辑 + 痕迹）───────────────────
    def _execute_tool_call(self, call: Any) -> tuple[dict[str, Any], ToolTrace]:
        """执行模型指定的工具，返回 (给接口的 tool 消息, 痕迹)。

        关键设计（v1 就有的）：工具执行失败时绝不崩程序 ——
        把错误当成「工具的输出」交回给模型，它往往会自己补参数重试。
        Agent 的自我纠错能力，很大一部分就来自这几行。
        """
        name = call.function.name
        raw_args = call.function.arguments or "{}"
        ok = True

        try:
            args = json.loads(raw_args)
            impl = TOOL_IMPLS.get(name)
            if impl is None:
                raise ValueError(f"不存在名为 {name} 的工具")
            result = str(impl(**args))
        except Exception as exc:
            result = f"错误：{type(exc).__name__}: {exc}"
            ok = False

        trace = ToolTrace(name=name, args_text=raw_args, result_text=result, ok=ok)
        tool_message = {
            "role": "tool",
            "tool_call_id": call.id,  # 必须带上，和上面那条 assistant 消息配对
            "name": name,
            "content": result,
        }
        return tool_message, trace
