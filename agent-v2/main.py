"""
命令行入口 —— 交互界面、/context 命令，以及不需要 API Key 的自检。

    python main.py              # 进入交互对话
    python main.py --selftest   # 测工具 + 测上下文剪枝（不需要 API Key）
    python main.py --config     # 看配置从哪来、当前生效的值是什么
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from agent import Agent
from config import (
    API_KEY,
    BASE_URL,
    DISABLED_TOOLS,
    KEEP_RECENT_MESSAGES,
    MAX_CONTEXT_TOKENS,
    MODEL,
    SYSTEM_PROMPT,
    describe_config,
)
from context import (
    ContextManager,
    SNIP_PLACEHOLDER,
    check_invariants,
    estimate_tokens,
)
from llm import LLMClient
from tools import TOOL_DEFS, TOOL_IMPLS

console = Console()


# ══════════════════════════════════════════════════════════════════════
# 界面辅助
# ══════════════════════════════════════════════════════════════════════


def _error_hint(exc: Exception) -> str:
    """把常见的报错翻译成人话，省得初学者一头雾水。"""
    text = str(exc).lower()
    if "401" in text or "authentication" in text or "api key" in text:
        return "看起来是 API Key 无效。检查 .env 里的 LLM_API_KEY 是否填对了。"
    if "context length" in text or "maximum context" in text or "too long" in text:
        return (
            "上下文还是超了模型窗口。把 .env 里的 LLM_MAX_CONTEXT_TOKENS 调小一些，"
            "或者把 LLM_KEEP_RECENT 调小。"
        )
    if "tool" in text and ("not support" in text or "unsupported" in text or "invalid" in text):
        return "当前模型可能不支持 tool calling。换 deepseek-chat / gpt-4o-mini / qwen-plus。"
    if "404" in text or ("model" in text and "not found" in text):
        return "模型名可能写错了，检查 .env 里的 LLM_MODEL。"
    if "connect" in text or "timeout" in text:
        return "连接不上模型服务，检查网络和 .env 里的 LLM_BASE_URL。"
    return ""


def _banner() -> Panel:
    # 只列【当前启用】的工具。关掉的那些单独用红字提醒，
    # 否则你很容易忘了自己还开着实验开关，然后奇怪它为什么变笨了。
    active = [name for name in TOOL_DEFS if name not in DISABLED_TOOLS]
    if DISABLED_TOOLS:
        disabled_line = (
            f"[bold red]⚠ 已关闭工具：{', '.join(sorted(DISABLED_TOOLS))}[/bold red]\n"
        )
    else:
        disabled_line = ""

    return Panel(
        "[bold]hello-agent[/bold] v2 · 带上下文管理的最小 Agent\n"
        f"[dim]模型 {MODEL}  @  {BASE_URL}[/dim]\n"
        f"[dim]工具 {', '.join(active) or '（全部被关闭）'}[/dim]\n"
        f"[dim]上下文预算 {MAX_CONTEXT_TOKENS:,} tokens · 最近 {KEEP_RECENT_MESSAGES} 条受保护[/dim]\n"
        f"{disabled_line}\n"
        "[dim]/context 查看占用    /reset 清空历史    /exit 退出[/dim]",
        border_style="blue",
        title="Agent",
    )


# ══════════════════════════════════════════════════════════════════════
# 自检 —— 不需要 API Key
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


def _make_fake_history(n_rounds: int, tool_blob_chars: int) -> list[dict]:
    """造一段假的超长对话历史，用来测剪枝。

    每轮的形状和真实对话完全一致：
        user → assistant(tool_calls) → tool → assistant
    """
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for r in range(n_rounds):
        cid = f"call_{r}"
        messages.append({"role": "user", "content": f"第 {r} 轮问题"})
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": cid,
                        "type": "function",
                        "function": {"name": "calculator", "arguments": '{"expression":"1+1"}'},
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": cid,
                "name": "calculator",
                # 故意塞很大一块，模拟「工具返回了一个巨型结果」
                "content": "X" * tool_blob_chars,
            }
        )
        messages.append({"role": "assistant", "content": f"第 {r} 轮回答"})
    return messages


def _test_tools(check: _Check) -> None:
    """工具层测试 —— 与 v1 完全相同的用例。

    这一组同时充当「拆文件是纯重构」的回归网：
    同样的输入必须给出同样的输出。
    """
    console.print("\n[bold]① 工具层测试（与 v1 相同的用例，验证重构无副作用）[/bold]\n")
    cases: list[tuple[str, dict]] = [
        ("calculator", {"expression": "(1234 * 5678) + 90"}),
        ("calculator", {"expression": "2 ** 10"}),
        ("calculator", {"expression": "1 / 0"}),  # 故意制造错误，看兜底
        ("get_current_time", {}),
        ("save_note", {"title": "selftest", "content": "该文件由 --selftest 生成，可删。"}),
    ]
    for name, args in cases:
        impl = TOOL_IMPLS[name]
        arg_text = ", ".join(f"{k}={v!r}" for k, v in args.items())
        try:
            result = impl(**args)
        except Exception as exc:
            console.print(
                f"  [cyan]{name}[/cyan]({arg_text})\n"
                f"    [yellow]✗ {type(exc).__name__}: {exc}[/yellow]"
            )
        else:
            console.print(f"  [cyan]{name}[/cyan]({arg_text})\n    [green]← {result}[/green]")

    check.ok(
        TOOL_IMPLS["calculator"](expression="(1234 * 5678) + 90") == "7006742",
        "calculator 结果与 v1 一致",
    )
    check.ok(len(TOOL_DEFS) == 3, f"工具数量与 v1 一致（{len(TOOL_DEFS)} 个）")


def _test_token_estimate(check: _Check) -> None:
    console.print("\n[bold]② token 估算器测试[/bold]\n")
    check.ok(estimate_tokens("") == 0, "空文本 → 0 tokens")
    check.ok(estimate_tokens("你好世界") == 4, "4 个汉字 → 4 tokens")
    check.ok(estimate_tokens("abcdefgh") == 2, "8 个英文字符 → 2 tokens")
    check.ok(estimate_tokens("X" * 4000) == 1000, "4000 个英文字符 → 1000 tokens")


def _test_context(check: _Check) -> None:
    console.print("\n[bold]③ 上下文剪枝测试（构造假历史，不调用 LLM）[/bold]\n")

    # ── 场景 A：只触发第一级 ────────────────────────────────────────
    #    预算刚好需要剪掉工具结果，但还不需要丢整轮
    cm1 = ContextManager(SYSTEM_PROMPT, max_tokens=12_000, keep_recent=0)
    cm1.messages = _make_fake_history(n_rounds=20, tool_blob_chars=4000)  # 约 20k tokens
    n_before = len(cm1.messages)
    r1 = cm1.prepare()

    console.print(f"  [dim]场景 A（只剪内容）：{r1.describe()}[/dim]")
    check.ok(r1.snipped_tool_results > 0, f"第一级剪掉了 {r1.snipped_tool_results} 条工具结果")
    check.ok(r1.dropped_rounds == 0, "第一级够用，没有丢整轮")
    check.ok(len(cm1.messages) == n_before, "⭐ 第一级只换 content、不删消息，条数不变")
    check.ok(not check_invariants(cm1.messages), "⭐ 剪枝后无孤儿 tool 消息，结构健康")
    placeholders_a = [m for m in cm1.messages if m.get("role") == "tool" and m["content"] == SNIP_PLACEHOLDER]
    check.ok(
        len(placeholders_a) == r1.snipped_tool_results,
        "⭐ 剪掉的工具结果原样保留在上下文里，没有被丢弃",
    )

    # ── 场景 B：预判「剪不够」→ 先丢整轮 ────────────────────────────
    cm2 = ContextManager(SYSTEM_PROMPT, max_tokens=2000, keep_recent=6)
    cm2.messages = _make_fake_history(n_rounds=20, tool_blob_chars=4000)

    tail_idx = max(i for i, m in enumerate(cm2.messages) if m["role"] == "user")
    tail_before = [dict(m) for m in cm2.messages[tail_idx:]]
    before = cm2.estimate_tokens()

    r2 = cm2.prepare()
    after = cm2.estimate_tokens()

    console.print(f"  [dim]场景 B（预判剪不够 → 先丢整轮）：{r2.describe()}[/dim]")
    check.ok(r2.did_trim, "确实发生了剪枝")
    check.ok(after < before, f"token 下降（{before:,} → {after:,}）")

    budget_ok = after <= cm2.max_tokens
    check.ok(
        budget_ok or r2.hit_floor,
        f"⭐ 要么回到预算内、要么如实标记「已顶到保护区」（实际 {after:,}，预算 {cm2.max_tokens:,}）",
    )
    # 这个场景是刻意压到极限的：预算 2000 相对内容太小，必然撞上保护区地板。
    # 重点不是「必须达标」，而是「达不了标要如实说，不能假装成功」。
    check.ok(r2.hit_floor, "⭐ 顶到保护区时如实标记 hit_floor（不假装成功）")
    check.ok(not check_invariants(cm2.messages), "⭐ 丢整轮后仍无孤儿 tool 消息")

    # 这条断言直接检验「白忙一场」的 bug 有没有复发
    placeholders_b = [m for m in cm2.messages if m.get("role") == "tool" and m["content"] == SNIP_PLACEHOLDER]
    check.ok(
        len(placeholders_b) == r2.snipped_tool_results,
        f"⭐ 没有白忙一场：剪了 {r2.snipped_tool_results} 条，上下文里正好剩 {len(placeholders_b)} 条",
    )

    # 当前轮一字未动（注意：下标会因为前面删了整轮而左移，所以重新定位）
    tail_idx_after = max(i for i, m in enumerate(cm2.messages) if m["role"] == "user")
    check.ok(
        cm2.messages[tail_idx_after:] == tail_before,
        "⭐ 最后一条 user 之后的内容一字未动（当前轮受保护）",
    )
    check.ok(cm2.messages[0]["role"] == "system", "⭐ 第 0 条仍是 system 消息")

    # ── 场景 C：不超预算时完全不动（不误伤）────────────────────────
    cm3 = ContextManager(SYSTEM_PROMPT, max_tokens=100_000, keep_recent=8)
    cm3.append({"role": "user", "content": "你好"})
    snapshot = [dict(m) for m in cm3.messages]
    r3 = cm3.prepare()
    check.ok(not r3.did_trim and cm3.messages == snapshot, "未超预算时完全不动（不误伤）")


def selftest() -> int:
    check = _Check()
    _test_tools(check)
    _test_token_estimate(check)
    _test_context(check)
    return check.report()


# ══════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        return selftest()

    if "--config" in argv:
        # ⚠️ 这里必须用 Text 而不是直接传字符串：
        #    describe_config() 里有 "[v2 独有]" 这种方括号，直接交给 rich
        #    会被当成样式标记解析掉 —— 标记内容会凭空消失。
        #    （"[共享]" 侥幸没事，是因为 rich 的标记必须以 ASCII 字母开头。）
        console.print(Panel(Text(describe_config()), border_style="cyan", title="当前配置"))
        return 0

    if "--help" in argv or "-h" in argv:
        console.print(__doc__)
        return 0

    if not API_KEY:
        console.print(
            Panel(
                "[red]没有找到 API Key[/red]\n\n"
                "把 v1 的配置复制过来、填上密钥即可：\n\n"
                "  [bold]cp ../agent-v1/.env .env[/bold]\n\n"
                "（或者 cp .env.example .env 从模板开始填）\n\n"
                "[dim]没有密钥也可以先跑自检：python main.py --selftest[/dim]",
                border_style="red",
                title="缺少配置",
            )
        )
        return 1

    client = LLMClient(API_KEY, BASE_URL, MODEL)
    context = ContextManager(
        SYSTEM_PROMPT,
        max_tokens=MAX_CONTEXT_TOKENS,
        keep_recent=KEEP_RECENT_MESSAGES,
    )
    agent = Agent(client, context)

    console.print(_banner())

    while True:
        try:
            user_input = console.input("\n[bold green]你 > [/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n再见 👋")
            return 0

        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            console.print("再见 👋")
            return 0
        if user_input == "/reset":
            agent.reset()
            console.print("[dim]已清空对话历史[/dim]")
            continue
        if user_input == "/context":
            console.print(Panel(context.describe(), border_style="magenta", title="上下文状态"))
            continue

        try:
            reply = agent.run(user_input)
        except Exception as exc:
            console.print(f"\n[red]请求失败：{type(exc).__name__}: {exc}[/red]")
            hint = _error_hint(exc)
            if hint:
                console.print(f"[yellow]💡 {hint}[/yellow]")
            continue

        console.print(Panel(reply, border_style="blue", title="Agent"))
        console.print(f"[dim]{context.summary_line()}[/dim]")
        console.print(
            f"[dim]累计 {client.calls} 次调用 · "
            f"{client.total_prompt_tokens:,} 输入 / "
            f"{client.total_completion_tokens:,} 输出 tokens[/dim]"
        )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
