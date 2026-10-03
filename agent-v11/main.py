"""
命令行入口 —— v11 的「零件巡礼」调度台。

    python main.py --demo messages     # 消息类型巡礼（离线，不花钱）
    python main.py --demo tools        # @tool 自动 schema（离线，不花钱）
    python main.py --demo tool-loop    # 一次工具回环
    python main.py --demo usage        # token 账本
    python main.py --demo lcel         # LCEL 管道 invoke/batch/stream
    python main.py --demo structured   # 结构化输出四种姿势（看它们怎么翻车）
    python main.py --demo agent        # create_agent 门面 + 揭盖子（通往 v12）
    python main.py --demo all          # 按顺序全跑一遍
    python main.py --selftest          # 离线自检（不需要 API Key，不花钱）
    python main.py --config            # 看配置从哪来
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

import config
import components
from components import (
    DEMOS,
    TOOL_MAP,
    TOOLS,
    _slugify,
    calculator,
    get_current_time,
    parse_maybe_fenced_json,
    save_note,
    tool_schema,
)

console = Console()


# ══════════════════════════════════════════════════════════════════════
# demo 调度
# ══════════════════════════════════════════════════════════════════════


def run_demo(name: str) -> int:
    if name == "all":
        order = [n for n in DEMOS if not DEMOS[n][2]] + [n for n in DEMOS if DEMOS[n][2]]
        worst = 0
        for n in order:
            if not config.API_KEY and DEMOS[n][2]:
                console.print(f"\n[dim]⏭ 跳过 demo「{n}」：它要真实调用模型，但没有 API Key（--demo all 会跳过这类）[/dim]")
                continue
            worst = max(worst, run_demo(n))
        return worst

    if name not in DEMOS:
        known = "、".join(DEMOS)
        console.print(f"[red]✗ 不认识 demo「{name}」[/red]（可用的有：{known}、all）")
        return 2

    description, fn, needs_key = DEMOS[name]
    if needs_key and not config.API_KEY:
        console.print(
            Panel(
                f"[red]demo「{name}」（{description}）要真实调用模型，没有 Key 跑不了[/red]\n\n"
                "先把仓库根 .env 里的 LLM_API_KEY 填上。\n\n"
                "[dim]先玩两个离线 demo（不花钱）：--demo messages / --demo tools[/dim]",
                border_style="red",
                title="缺少配置",
            )
        )
        return 1

    try:
        fn()
        return 0
    except Exception as exc:  # noqa: BLE001
        console.print(f"\n[red]✗ demo 跑挂了：{type(exc).__name__}: {exc}[/red]")
        _print_error_hint(exc)
        return 1


def _print_error_hint(exc: Exception) -> None:
    text = str(exc).lower()
    if "401" in text or "authentication" in text or "api key" in text:
        console.print("[yellow]💡 看起来是 API Key 无效。检查仓库根 .env 里的 LLM_API_KEY。[/yellow]")
    elif "connect" in text or "timeout" in text:
        console.print("[yellow]💡 连接不上模型服务，检查网络和 .env 里的 LLM_BASE_URL。[/yellow]")
    elif "tool" in text and "support" in text:
        console.print("[yellow]💡 当前模型可能不支持 tool calling。换 deepseek-chat / gpt-4o-mini / qwen-plus。[/yellow]")


# ══════════════════════════════════════════════════════════════════════
# 离线自检
# ══════════════════════════════════════════════════════════════════════


class _Check:
    """极简断言收集器：不因第一个失败就中断，能一次看到所有问题。"""

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def ok(self, condition: bool, label: str) -> None:
        if condition:
            self.passed += 1
            console.print(f"  [green]✓[/green] {label}")
        else:
            self.failed += 1
            console.print(f"  [red]✗ {label}[/red]")

    def report(self) -> int:
        total = self.passed + self.failed
        if self.failed:
            console.print(f"\n[red]✗ {self.failed}/{total} 项断言失败[/red]")
            return 1
        console.print(f"\n[green]✓ 全部 {total} 项断言通过[/green]")
        return 0


def selftest() -> int:
    """离线把组件层测穿：schema 推导 / 消息类型 / 解析容错 / 配置。"""
    check = _Check()
    console.print("[bold]离线自检（不调用真实模型、不花一分钱）[/bold]\n")

    # ── ① @tool 自动推导的 schema ───────────────────────────────────
    console.print("[bold]① @tool：schema 推导[/bold]")
    check.ok(len(TOOLS) == 3, "三个工具都在（calculator / get_current_time / save_note）")
    check.ok(set(TOOL_MAP) == {"calculator", "get_current_time", "save_note"}, "工具名映射正确")

    schema = tool_schema(calculator)["function"]
    check.ok(schema["name"] == "calculator", "schema 名字正确")
    check.ok("计算" in schema["description"], "schema 描述来自 docstring（写给模型的提示词）")
    check.ok(list(schema["parameters"]["properties"]) == ["expression"], "calculator 参数从签名推导")
    check.ok("expression" in schema["parameters"]["required"], "必填参数自动标记 required")

    s2 = tool_schema(get_current_time)["function"]
    check.ok("parameters" not in s2 or not s2["parameters"].get("properties"), "无参工具：schema 里没有参数")

    s3 = tool_schema(save_note)["function"]
    check.ok(set(s3["parameters"]["properties"]) == {"title", "content"}, "save_note 两个参数都推出来")

    # 对照：结构字段和 v1~v10 手写版对齐（type/function/name/parameters/properties/required）
    check.ok(
        schema["parameters"]["type"] == "object"
        and isinstance(schema["parameters"]["properties"], dict),
        "schema 结构与手写版（ToolDefinition）同宗：type/function/parameters/properties",
    )

    # ── ② 消息类型 ──────────────────────────────────────────────────
    console.print("\n[bold]② 消息类型：协议的对象化[/bold]")
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    check.ok(SystemMessage("x").type == "system", "SystemMessage → type=system")
    check.ok(HumanMessage("x").type == "human", "HumanMessage → type=human")
    check.ok(AIMessage("x").type == "ai", "AIMessage → type=ai")
    check.ok(ToolMessage(content="x", tool_call_id="c1").type == "tool", "ToolMessage → type=tool")

    ai = AIMessage("", tool_calls=[{"name": "calculator", "args": {"expression": "1+1"}, "id": "c1"}])
    check.ok(ai.tool_calls[0]["name"] == "calculator" and ai.content == "",
             "带 tool_calls 的 AI 消息：content 允许为空（v1 的老坑，类型替你兜了）")

    try:
        ToolMessage(content="x")  # 缺 tool_call_id
        check.ok(False, "ToolMessage 缺 tool_call_id 应该报错")
    except Exception:
        check.ok(True, "ToolMessage 缺 tool_call_id：构造直接报错（协议约束从纪律变成类型）")

    dumped = SystemMessage("你好").model_dump()
    check.ok(dumped.get("type") == "system" and dumped.get("content") == "你好",
             "消息可序列化回 dict（协议还是那个协议，只是有了类型）")

    # ── ③ 工具执行行为（安全性与 v1 一致）───────────────────────────
    console.print("\n[bold]③ 工具执行：白名单安全性（和 v1 同款）[/bold]")
    check.ok(calculator.invoke({"expression": "2 * (3 + 4)"}) == "14", "calculator 正常计算")
    check.ok(calculator.invoke({"expression": "7 / 2"}) == "3.5", "calculator 浮点计算")
    try:
        calculator.invoke({"expression": "__import__('os')"})
        check.ok(False, "恶意表达式应该被白名单拒绝")
    except Exception:
        check.ok(True, "恶意表达式（__import__）：白名单拒绝，绝不用裸 eval")

    check.ok(str(get_current_time.invoke({})).count("星期") == 1, "get_current_time 返回带星期的格式")

    with tempfile.TemporaryDirectory() as tmp:
        old_notes = config.NOTES_DIR
        config.NOTES_DIR = Path(tmp)
        try:
            out = save_note.invoke({"title": "测试 笔记", "content": "内容"})
            check.ok("已保存" in out and (Path(tmp) / "测试_笔记.md").is_file(),
                     "save_note 写文件 + slug 处理（空格→下划线）")
        finally:
            config.NOTES_DIR = old_notes

    # ── ④ 解析容错（结构化输出的安全网）─────────────────────────────
    console.print("\n[bold]④ parse_maybe_fenced_json：抠 JSON 的容错[/bold]")
    check.ok(parse_maybe_fenced_json('{"a": 1}') == {"a": 1}, "裸 JSON：直接解析")
    check.ok(parse_maybe_fenced_json('```json\n{"a": 1}\n```') == {"a": 1}, "```json 围栏：剥掉")
    check.ok(parse_maybe_fenced_json('好的！\n{"a": 1}\n以上。') == {"a": 1}, "前后有废话：抠出来")
    try:
        parse_maybe_fenced_json("完全没有 JSON")
        check.ok(False, "没有 JSON 应该报错")
    except ValueError as exc:
        check.ok("抠不出 JSON" in str(exc) and "原文开头" in str(exc), "抠不出：报错带原文开头（v8 的日志纪律）")

    # ── ⑤ slug 细节 ─────────────────────────────────────────────────
    console.print("\n[bold]⑤ slug 细节[/bold]")
    check.ok(_slugify("周五交作业") == "周五交作业", "中文标题原样保留")
    check.ok(_slugify("a b/c!") == "a_b_c", "特殊字符换成下划线")
    check.ok(_slugify("!!!") == "untitled", "全非法字符：回退 untitled")

    # ── ⑥ 配置与模型工厂 ────────────────────────────────────────────
    console.print("\n[bold]⑥ 配置与模型工厂[/bold]")
    llm = config.get_chat_model()
    check.ok(llm.model_name == config.MODEL, "模型工厂：model 名跟随配置")
    check.ok(str(llm.openai_api_base) == config.BASE_URL, "模型工厂：base_url 跟随配置（换家=改 .env）")
    check.ok(abs(float(llm.temperature) - config.TEMPERATURE) < 1e-9, "温度默认走共享层")
    check.ok(abs(float(config.get_chat_model(temperature=0).temperature)) < 1e-9, "温度可被单次覆盖（评估要温度 0）")

    # ── ⑦ demo 调度表 ───────────────────────────────────────────────
    console.print("\n[bold]⑦ demo 调度表[/bold]")
    check.ok(len(DEMOS) == 7, f"七个 demo 都在（{len(DEMOS)}）")
    check.ok(DEMOS["messages"][2] is False and DEMOS["tools"][2] is False,
             "前两个 demo 标为「离线可跑」（没有 Key 也能体验）")
    check.ok(all(isinstance(v[0], str) and callable(v[1]) for v in DEMOS.values()),
             "调度表条目结构正确（说明 + 函数 + 是否要 Key）")

    return check.report()


# ══════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════


def _get_arg(argv: list[str], flag: str) -> str | None:
    if flag not in argv:
        return None
    idx = argv.index(flag)
    if idx + 1 < len(argv) and not argv[idx + 1].startswith("--"):
        return argv[idx + 1]
    return ""


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        return selftest()

    if "--config" in argv:
        console.print(Panel(Text(config.describe_config()), border_style="cyan", title="当前配置"))
        return 0

    if "--help" in argv or "-h" in argv:
        console.print(__doc__)
        return 0

    if "--demo" in argv:
        name = _get_arg(argv, "--demo")
        if not name:
            known = "、".join(DEMOS)
            console.print(f"[red]✗ --demo 后面要跟名字[/red]（可用的有：{known}、all）")
            return 2
        return run_demo(name)

    console.print(__doc__)
    console.print("[dim]（先试试：python main.py --demo messages —— 离线可跑）[/dim]")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
