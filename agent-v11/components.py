"""
组件层巡礼 —— v11 的核心讲练文件。

七个 demo，每个对照一段我们手写过的历史（v1~v10）：

    demo_messages    消息类型      ↔ v1 手写的 {...} dict          （离线可跑）
    demo_tools       @tool 自动 schema ↔ v1~v10 手写的 ToolDefinition（离线可跑）
    demo_tool_loop   一次工具回环  ↔ v1 的 _execute_tool_call + 配对
    demo_usage       usage 账本    ↔ v9 手写的 token 账本
    demo_lcel        `|` 管道组装  ↔ 手写循环里拼 messages
    demo_structured  结构化输出    ↔ v8 手写「工具即 schema + 重试」
    demo_agent       create_agent  ↔ v1 手写的 40 行主循环（结尾揭盖子：内核是 LangGraph）

每个 demo 的排版约定（铁律 7）：
    🎯 目的 → 过程（带真实数据）→ 💡 面试点（一句话能说出去的那种）
"""

from __future__ import annotations

import ast
import json
import operator
import re
from datetime import datetime
from pathlib import Path

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from rich.console import Console
from rich.panel import Panel

import config

console = Console()


def _head(title: str, purpose: str) -> None:
    console.print()
    console.print(Panel(purpose, title=f"demo · {title}", border_style="cyan", title_align="left"))


def _point(text: str) -> None:
    console.print(f"[yellow]💡 面试点：{text}[/yellow]")


def _fail_note(text: str) -> None:
    console.print(f"[red]✗ {text}[/red]")


# ══════════════════════════════════════════════════════════════════════
# 三个工具 —— 这次用 @tool 写（对照 v1~v10 的手写 ToolDefinition）
# ══════════════════════════════════════════════════════════════════════
#
# ⚠️ 注意 docstring 的写法 —— 它会被自动提取成工具的 description，
#    也就是「写给模型的提示词」（v1 实验 10 的老教训，框架不替你想文案）。

_ALLOWED_OPS: dict[type, object] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST):
    """递归求值一个算术表达式 AST（白名单机制，绝不用裸 eval）——沿用 v1 的实现。"""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"不支持的常量：{node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"不支持的表达式：{ast.dump(node)}")


@tool
def calculator(expression: str) -> str:
    """计算一个数学表达式并返回精确结果。凡是涉及数字运算的问题都必须调用本工具，
    不要自己心算，否则很容易算错。支持 + - * / // % ** 以及括号。"""
    result = _safe_eval(ast.parse(expression, mode="eval"))
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


_WEEKDAYS = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


@tool
def get_current_time() -> str:
    """获取当前的本地日期和时间。凡是涉及『今天』『现在』『当前时间』的问题都必须调用
    本工具，不要凭记忆或猜测作答。"""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S") + f"（{_WEEKDAYS[now.weekday()]}）"


def _slugify(title: str) -> str:
    """标题 → 安全文件名（和 v1 起的实现一致）。"""
    return re.sub(r"[^\w\u4e00-\u9fff-]+", "_", title).strip("_") or "untitled"


@tool
def save_note(title: str, content: str) -> str:
    """把一条笔记写入本地文件长期保存。当用户说『记下来』『记一下』『保存』『备忘』时
    调用本工具。title 会被用作文件名。"""
    path = config.NOTES_DIR / f"{_slugify(title)}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{content}\n", encoding="utf-8")
    return f"已保存到 {path.name}"


TOOLS = [calculator, get_current_time, save_note]
TOOL_MAP = {t.name: t for t in TOOLS}


def tool_schema(t) -> dict:
    """把一个 @tool 转成 OpenAI 的 schema dict（demo 和 selftest 共用）。"""
    from langchain_core.utils.function_calling import convert_to_openai_tool

    return convert_to_openai_tool(t)


def parse_maybe_fenced_json(text: str) -> dict:
    """从模型输出里抠出 JSON（容忍 ```json 围栏和前后废话）。

    为什么需要它？—— 结构化输出 demo 的最土路线（纯提示词 + 自己解析），
    这正是 v8 手写时代每天的日常，也是「框架救不了你」时的安全网。
    """
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # 再退一步：找第一个 { 到最后一个 }
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        return json.loads(cleaned[start : end + 1])
    raise ValueError(f"抠不出 JSON ｜原文开头：{cleaned[:80]!r}")


# ══════════════════════════════════════════════════════════════════════
# demo 1：消息类型（离线）
# ══════════════════════════════════════════════════════════════════════


def demo_messages() -> None:
    _head("messages", "LangChain 的消息类型 —— 对照 v1 手写的 {'role': ..., 'content': ...}")
    console.print("  手写时代（v1~v10 里到处可见）：")
    console.print('    [dim]{"role": "user", "content": "今天几号？"}[/dim]')
    console.print()
    console.print("  组件时代：")
    msgs = [
        SystemMessage("你是一个简洁的助手。"),
        HumanMessage("今天几号？"),
        AIMessage("", tool_calls=[{"name": "get_current_time", "args": {}, "id": "call_1"}]),
        ToolMessage(content="2026-10-03 16:00:00（星期六）", tool_call_id="call_1"),
    ]
    for m in msgs:
        extra = ""
        if isinstance(m, AIMessage) and m.tool_calls:
            # ⚠️ 别用方括号包名字 —— rich 会把它当样式标记吃掉（v2 踩过的坑）
            extra = f" tool_calls=«{m.tool_calls[0]['name']}…»"
        if isinstance(m, ToolMessage):
            extra = f" tool_call_id={m.tool_call_id}"
        console.print(f"    {type(m).__name__:<15} type={m.type:<8} content={str(m.content)[:32]!r}{extra}")
    _point(
        "消息类型是「协议的对象化」：手写时代靠「字典里放对 key」，现在类型把协议固化、"
        "把校验提前。但注意 —— tool_call_id 配对、空 content 规则，一个都没消失，"
        "只是从「你的纪律」变成了「类型的约束」。"
    )


# ══════════════════════════════════════════════════════════════════════
# demo 2：@tool 自动 schema（离线）
# ══════════════════════════════════════════════════════════════════════


def demo_tools() -> None:
    _head("tools", "@tool —— docstring + 签名，自动推导 schema（离线，不花钱）")
    console.print("  手写时代（v1~v10）：每个工具一段 ToolDefinition + ToolParameter，约 25 行；")
    console.print("  组件时代：")
    console.print('    [dim]@tool[/dim]')
    console.print('    [dim]def calculator(expression: str) -> str:[/dim]')
    console.print('    [dim]    """计算一个数学表达式……（这段 docstring 就是 description）"""[/dim]')
    console.print()
    console.print("  自动生成的 OpenAI schema（calculator）：")
    schema = tool_schema(calculator)
    console.print_json(json.dumps(schema["function"], ensure_ascii=False, indent=2))
    console.print()
    console.print("  三个工具的 schema 一览：")
    for t in TOOLS:
        s = tool_schema(t)["function"]
        params = list(s["parameters"].get("properties", {}).keys())
        console.print(f"    {t.name:<18} 参数={params or '（无）'}")
    _point(
        "schema 自动推导 = 文档即接口。但 v1 的实验 10 照样成立 —— "
        "**工具描述就是写给模型的提示词**：docstring 写得烂，工具就没人会用。框架不替你想文案。"
    )


# ══════════════════════════════════════════════════════════════════════
# demo 3：一次完整的工具回环（真实调用）
# ══════════════════════════════════════════════════════════════════════


def demo_tool_loop() -> None:
    _head("tool-loop", "一次完整的工具回环：模型要工具 → 你执行 → 结果回灌 → 收工")
    llm = config.get_chat_model(temperature=0)
    llm_tools = llm.bind_tools(TOOLS)

    question = "请用工具算一下 12 × 34，再看看现在的时间。"
    messages = [SystemMessage("你需要计算或知道时间时，必须调用对应工具。"), HumanMessage(question)]
    console.print(f"  用户：{question}")

    ai = llm_tools.invoke(messages)
    console.print()
    console.print("  ① 模型的第一反应（AIMessage）：")
    if not ai.tool_calls:
        console.print("     [yellow]（这次它没有调用工具，直接答了——模型行为有波动，再跑一次试试）[/yellow]")
        console.print(f"     content={ai.content[:60]!r}")
        return
    for call in ai.tool_calls:
        console.print(f"     tool_calls: {call['name']}({call['args']})  «call_id={call['id'][:12]}…»")

    messages.append(ai)
    console.print()
    console.print("  ② 我方执行工具 + 用 ToolMessage 回灌（注意 tool_call_id 必须配对）：")
    for call in ai.tool_calls:
        result = TOOL_MAP[call["name"]].invoke(call["args"])
        console.print(f"     {call['name']} → {result}")
        messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    final = llm_tools.invoke(messages)
    console.print()
    console.print(f"  ③ 模型的最终回答：{final.content[:100]}")
    console.print(f"  [dim]账本：input {final.usage_metadata['input_tokens']} / output {final.usage_metadata['output_tokens']} tokens[/dim]")
    _point(
        "协议和你 v1 手写的**一模一样**：tool_calls、call_id 配对、结果回灌。"
        "区别只是「有人替你定义了类型和解析」。框架没有魔法 —— 是协议的对象化 + 标准化。"
    )


# ══════════════════════════════════════════════════════════════════════
# demo 4：usage 账本（真实调用）
# ══════════════════════════════════════════════════════════════════════


def demo_usage() -> None:
    _head("usage", "token 账本 —— usage_metadata（对照 v9 手写账本）")
    llm = config.get_chat_model(temperature=0)
    resp = llm.invoke([HumanMessage("用一句话解释什么是 token。")])
    u = resp.usage_metadata or {}
    console.print(f"  input_tokens  : {u.get('input_tokens')}   {u.get('input_token_details')}")
    console.print(f"  output_tokens : {u.get('output_tokens')}   {u.get('output_token_details')}")
    console.print(f"  total_tokens  : {u.get('total_tokens')}")
    console.print()
    console.print(f"  [dim]回答开头：{str(resp.content)[:60]}[/dim]")
    _point(
        "账本里已经带上了「缓存命中（cache_read）」和「推理 token（reasoning）」—— "
        "成本优化的两个抓手（缓存、推理预算）都能从这里监控。你 v9 手写的账本，"
        "在组件时代就是 usage_metadata 一个字段。"
    )


# ══════════════════════════════════════════════════════════════════════
# demo 5：LCEL 管道（真实调用）
# ══════════════════════════════════════════════════════════════════════


def demo_lcel() -> None:
    _head("lcel", "LCEL：`prompt | model | parser` —— 组装即代码，invoke/batch/stream 一套协议")
    llm = config.get_chat_model(temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你只输出结果数字，不要任何其他文字。"),
            ("human", "{expr} 等于几？"),
        ]
    )
    chain = prompt | llm | StrOutputParser()

    console.print("  ① invoke（单次）：")
    console.print(f"     chain.invoke({{'expr': '7 * 8'}}) → {chain.invoke({'expr': '7 * 8'})!r}")

    console.print("  ② batch（批量的『免费』来自统一的 Runnable 协议）：")
    outs = chain.batch([{"expr": "7 * 8"}, {"expr": "16 + 2"}, {"expr": "33 * 3"}])
    console.print(f"     {outs}")

    console.print("  ③ stream（逐段到达）：")
    chunks: list[str] = []
    shown = 0
    for chunk in chain.stream({"expr": "123 * 456"}):
        chunks.append(chunk)
        if chunk and shown < 6:  # 跳过空分片再显示（服务的头部可能给出空片）
            shown += 1
            console.print(f"     分片{shown}: {chunk!r}")
    console.print(f"     {len(chunks)} 个分片拼起来 → {''.join(chunks)!r}")
    _point(
        "LCEL 的真正卖点不是「少写代码」，是**组合的标准化**："
        "任何 Runnable 都自带 invoke/batch/stream，所以批量、流式、重试、回退"
        "这些能力可以一行挂上去。代价：复杂控制流写不出来 —— 管道是数据流（DAG），"
        "而 Agent 需要的是「循环 + 状态」（v12 的正题）。"
    )


# ══════════════════════════════════════════════════════════════════════
# demo 6：结构化输出四种尝试（真实调用，重点是「看它们怎么翻车」）
# ══════════════════════════════════════════════════════════════════════


def demo_structured() -> None:
    _head("structured", "with_structured_output 四种姿势 × DeepSeek —— 重点是「看它们怎么翻车」")
    from pydantic import BaseModel, Field

    class NoteDraft(BaseModel):
        """一条笔记的草稿。"""

        title: str = Field(description="笔记标题")
        keywords: list[str] = Field(description="两到三个关键词")

    llm = config.get_chat_model(temperature=0)
    task = "把『周五交作业，负责人老王』整理成一条笔记草稿，以 json 输出。"

    # 姿势 1：json_schema（framework 默认）
    console.print("  ① json_schema 模式（框架默认姿势）：")
    try:
        out = llm.with_structured_output(NoteDraft).invoke(task)
        console.print(f"     [green]OK: {out}[/green]")
    except Exception as exc:  # noqa: BLE001
        _fail_note(f"{type(exc).__name__}: {str(exc)[:110]}")
        console.print("     [dim]→ DeepSeek 的服务端不支持 response_format=json_schema[/dim]")

    # 姿势 2：function_calling
    console.print("  ② function_calling 模式（强制 tool_choice）：")
    try:
        out = llm.with_structured_output(NoteDraft, method="function_calling").invoke(task)
        console.print(f"     [green]OK: {out}[/green]")
    except Exception as exc:  # noqa: BLE001
        _fail_note(f"{type(exc).__name__}: {str(exc)[:110]}")
        console.print("     [dim]→ DeepSeek 的 thinking 模式拒绝强制 tool_choice（v8 时代就发现了）[/dim]")

    # 姿势 3：json_mode（裸用）
    console.print("  ③ json_mode 模式（裸用，没告诉它字段名）：")
    try:
        out = llm.with_structured_output(NoteDraft, method="json_mode").invoke(task)
        console.print(f"     [green]OK: {out}[/green]")
    except Exception as exc:  # noqa: BLE001
        _fail_note(f"{type(exc).__name__}: {str(exc)[:110]}")
        console.print("     [dim]→ 只保证了「是 JSON」，不保证「键名对」——模型自由发挥[/dim]")

    # 姿势 4：json_mode 的正确用法 = prompt 里自己带字段说明
    console.print("  ④ json_mode + 把字段说明写进提示词（回退到「说清楚」的自觉）：")
    try:
        structured = llm.with_structured_output(NoteDraft, method="json_mode")
        ask = (
            "以 json 格式输出，字段要求：\n"
            '- title：字符串，笔记标题\n- keywords：字符串数组，两到三个关键词\n\n'
            "内容：把『周五交作业，负责人老王』整理成一条笔记草稿。"
        )
        out = structured.invoke(ask)
        console.print(f"     [green]OK: {out}[/green]")
    except Exception as exc:  # noqa: BLE001
        _fail_note(f"{type(exc).__name__}: {str(exc)[:110]}")

    # 姿势 5（终极安全网）：纯提示词 + 自己解析 —— v8 手写时代的日常
    console.print("  ⑤ 纯提示词 + 自己解析（最土也最稳的兜底）：")
    raw = llm.invoke(
        "以 json 输出，字段：title（标题）、keywords（关键词数组）。"
        "不要输出 json 以外的任何文字。\n内容：把『周五交作业，负责人老王』整理成笔记草稿。"
    )
    try:
        parsed = parse_maybe_fenced_json(str(raw.content))
        console.print(f"     [green]解析成功: {parsed}[/green]")
    except Exception as exc:  # noqa: BLE001
        _fail_note(f"解析失败: {exc}")

    _point(
        "框架给的三种姿势，在 DeepSeek 上全部不顺手（json_schema 不支持 / "
        "function_calling 被拒 / json_mode 键名漂移）。**这不是 LangChain 的错** —— "
        "框架抽象的是「接口形状」，而服务端的约束凌驾其上。所以："
        "① 你必须知道抽象**为什么**失效（协议细节）；② v8 手写「工具即 schema + 校验重试」"
        "在 DeepSeek 上反而是最稳的姿势 —— 手写教会你的判断力，框架给不了。"
    )


# ══════════════════════════════════════════════════════════════════════
# demo 7：create_agent 门面（真实调用 + 揭盖子）
# ══════════════════════════════════════════════════════════════════════


def demo_agent() -> None:
    _head("agent", "create_agent：一行代码得到会调工具的 agent —— 然后揭开它的盖子")
    from langchain.agents import create_agent

    llm = config.get_chat_model(temperature=0)
    agent = create_agent(llm, TOOLS, system_prompt="你需要计算或知道时间时，必须调用对应工具。")

    question = "帮我算 88 × 11，再看一眼现在几点。"
    console.print(f"  用户：{question}")
    result = agent.invoke({"messages": [HumanMessage(question)]})
    console.print(f"  回答：{str(result['messages'][-1].content)[:120]}")

    console.print()
    console.print("  它的内部结构（agent.get_graph()）——节点列表：")
    for node in agent.get_graph().nodes:
        console.print(f"     node: {node}")
    console.print()
    console.print(f"  它的类型定义在：[bold]{type(agent).__module__}[/bold]  ← 注意这个名字")
    _point(
        "LangChain 的 agent 门面，内核就是一张 **LangGraph 的图**（state / model / tools 节点 + "
        "条件边）。这是 2026 年的重要事实：**『组件层的尽头是编排层』** —— "
        "当你想改这张图（加检查点、加人工审批、加重规划），LangChain 的界面就不够了。"
        "那就是 v12（LangGraph）要拆开的东西。"
    )


# ══════════════════════════════════════════════════════════════════════
# demo 调度表（main.py 和 selftest 共用）
# ══════════════════════════════════════════════════════════════════════

DEMOS: dict[str, tuple[str, object, bool]] = {
    # 名字 -> (一句话说明, 函数, 是否需要 API Key)
    "messages": ("消息类型巡礼", demo_messages, False),
    "tools": ("@tool 自动 schema", demo_tools, False),
    "tool-loop": ("一次工具回环", demo_tool_loop, True),
    "usage": ("token 账本 usage_metadata", demo_usage, True),
    "lcel": ("LCEL 管道 invoke/batch/stream", demo_lcel, True),
    "structured": ("结构化输出四种姿势的翻车现场", demo_structured, True),
    "agent": ("create_agent 门面 + 揭盖子", demo_agent, True),
}
