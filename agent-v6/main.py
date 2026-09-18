"""
命令行入口 —— 交互界面、/summary 与 /session 命令，以及不需要 API Key 的自检。

    python main.py              # 进入交互对话（会自动恢复上一次的会话）
    python main.py --selftest   # 测工具 + 剪枝 + 压缩 + 持久化（不需要 API Key，不花钱）
    python main.py --config     # 看配置从哪来、当前生效的值是什么
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from agent import Agent
from compaction import LLMSummarizer, SummarizeResult
from embedding import EmbeddingCache, EmbedStats
from config import (
    API_KEY,
    BASE_URL,
    DISABLED_TOOLS,
    KEEP_RECENT_MESSAGES,
    MAX_CONTEXT_TOKENS,
    MIN_COMPACT_TOKENS,
    MODEL,
    NOTES_DIR,
    PERSIST_ENABLED,
    RETRIEVAL_ENABLED,
    RETRIEVAL_MODE,
    RETRIEVAL_TOP_K,
    SESSION_FILE,
    SUMMARY_ENABLED,
    SUMMARY_MAX_TOKENS,
    SUMMARY_MODEL,
    SYSTEM_PROMPT,
    describe_config,
)
from context import (
    ContextManager,
    SNIP_PLACEHOLDER,
    SUMMARY_BLOCK_HEADER,
    check_invariants,
    estimate_tokens,
)
from llm import LLMClient
from retrieval import (
    Chunk,
    KnowledgeBase,
    build_index,
    split_markdown,
    tokenize,
)
from storage import (
    delete_session,
    describe_age,
    human_size,
    load_session,
    save_session,
)
from tools import KNOWLEDGE, TOOL_DEFS, TOOL_IMPLS

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
        return "模型名可能写错了，检查 .env 里的 LLM_MODEL（和 LLM_SUMMARY_MODEL）。"
    if "connect" in text or "timeout" in text:
        return "连接不上模型服务，检查网络和 .env 里的 LLM_BASE_URL。"
    return ""


def _banner() -> Panel:
    # 只列【当前启用】的工具。关掉的那些单独用红字提醒，
    # 否则你很容易忘了自己还开着实验开关，然后奇怪它为什么变笨了。
    active = [name for name in TOOL_DEFS if name not in DISABLED_TOOLS]
    lines = [
        "[bold]hello-agent[/bold] v5 · 会去翻你笔记的最小 Agent",
        f"[dim]模型 {MODEL}  @  {BASE_URL}[/dim]",
        f"[dim]工具 {', '.join(active) or '（全部被关闭）'}[/dim]",
        f"[dim]上下文预算 {MAX_CONTEXT_TOKENS:,} tokens · 最近 {KEEP_RECENT_MESSAGES} 条受保护[/dim]",
    ]
    if SUMMARY_ENABLED:
        lines.append(
            f"[dim]记忆压缩 [green]开[/green] · 摘要模型 {SUMMARY_MODEL} · "
            f"摘要上限 {SUMMARY_MAX_TOKENS} tokens[/dim]"
        )
    else:
        lines.append("[bold red]记忆压缩 已关闭 —— 退回 v2 的「丢弃」行为[/bold red]")
    if PERSIST_ENABLED:
        lines.append(
            f"[dim]持久化 [green]开[/green] · 每轮自动存盘 → {SESSION_FILE.name}[/dim]"
        )
    else:
        lines.append("[bold red]持久化 已关闭 —— 退出就忘光[/bold red]")
    if RETRIEVAL_ENABLED:
        lines.append(
            f"[dim]本地检索 [green]开[/green] · 模式 [bold]{RETRIEVAL_MODE}[/bold]"
            f" · 资料目录 {NOTES_DIR} · 每次带 {RETRIEVAL_TOP_K} 段回来[/dim]"
        )
    else:
        lines.append("[bold red]本地检索 已关闭 —— 它读不到你的笔记[/bold red]")
    if DISABLED_TOOLS:
        lines.append(f"[bold red]⚠ 已关闭工具：{', '.join(sorted(DISABLED_TOOLS))}[/bold red]")

    lines += [
        "",
        "[dim]/context 看上下文    /summary 看摘要    /session 看会话文件    "
        "/knowledge 看资料库    /reset 清空    /exit 退出[/dim]",
    ]
    return Panel("\n".join(lines), border_style="blue", title="Agent")


def _session_report(context: ContextManager) -> str:
    """`/session` 命令：把磁盘上那个文件的真实情况摆出来。

    ⚠️ 这里特意把「磁盘上的内容」和「内存里的内容」分开讲 ——
       持久化最容易让人困惑的一点就是：**有两个状态源，它们可能不一致**。
    """
    if not PERSIST_ENABLED:
        return (
            "持久化已关闭（LLM_PERSIST_ENABLED=0）\n\n"
            "这一版什么都不写盘 —— 现在聊得再好，退出就蒸发。"
        )

    path = SESSION_FILE
    in_memory = f"内存里当前有 {len(context.messages)} 条消息"
    if context.has_summary:
        in_memory += f"（含摘要 {context.summary_tokens:,} tokens）"

    if not path.is_file():
        return (
            f"文件路径  {path}\n"
            "状态      文件还不存在 —— 第一条消息答完之后就会出现\n\n"
            f"（{in_memory}）"
        )

    stat = path.stat()
    age = time.time() - stat.st_mtime
    saved_messages, saved_summary = context.export()
    summary_line = (
        f"摘要 {estimate_tokens(saved_summary):,} tokens" if saved_summary else "还没有摘要"
    )
    return (
        f"文件路径  {path}\n"
        f"大小      {human_size(stat.st_size)}\n"
        f"最后写入  {datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}"
        f"（{describe_age(age)}）\n"
        f"内容      {len(saved_messages)} 条消息 · {summary_line}\n\n"
        f"（{in_memory} —— 下一轮结束时会再存一次）"
    )


# ══════════════════════════════════════════════════════════════════════
# 自检 —— 不需要 API Key，也不花钱
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


def _make_fake_conversation(
    n_rounds: int,
    *,
    opening: str | None = None,
    tool_blob_chars: int = 0,
) -> list[dict]:
    """造一段假的对话历史，形状和真实对话完全一致：

        user → assistant(tool_calls) → tool → assistant

    opening 是用来埋「暗号」的：它是整段历史的第一条 user 消息，
    压缩之后还能不能在上下文里找到它，就是压缩有没有用的判据。
    """
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if opening:
        messages.append({"role": "user", "content": opening})
        messages.append({"role": "assistant", "content": "好的。"})

    for r in range(n_rounds):
        cid = f"call_{r}"
        messages.append({"role": "user", "content": f"第 {r} 轮问题：帮我算一下 {r} + {r} 等于几。"})
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": cid,
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": json.dumps({"expression": f"{r}+{r}"}),
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": cid,
                "name": "calculator",
                # 故意塞一块大的，模拟「工具返回了一个巨型结果」
                "content": f"结果是 {2 * r}。" + "X" * tool_blob_chars,
            }
        )
        messages.append({"role": "assistant", "content": f"第 {r} 轮的答案是 {2 * r}。"})
    return messages


def _context_contains(messages: list[dict], needle: str) -> bool:
    """上下文里（不管藏在哪条消息里）还能不能找到这段文字。"""
    return needle in json.dumps(messages, ensure_ascii=False, default=str)


def _tail_snapshot(context: ContextManager) -> list[dict]:
    """记下「最后一条 user 消息之后」的原文，用来验证当前轮没被动过。"""
    last_user = max(i for i, m in enumerate(context.messages) if m.get("role") == "user")
    return [dict(m) for m in context.messages[last_user:]]


def _tail_unchanged(context: ContextManager, snapshot: list[dict]) -> bool:
    last_user = max(i for i, m in enumerate(context.messages) if m.get("role") == "user")
    return context.messages[last_user:] == snapshot


def _print_report(report) -> None:
    """把一次剪枝/压缩的报告打出来（用 Text 包一层，避免方括号被 rich 吃掉）。"""
    for line in report.describe_lines():
        console.print(Text(f"    {line}", style="dim"))


class _FakeSummarizer:
    """离线测试专用的「假摘要器」。

    真实压缩要调 LLM；自检不能联网、不能花钱，所以塞一个确定性的替身。

    它模拟的是**抽取式摘要**：每条对话消息保留开头若干个字符，再拼起来。
    这个替身和真实摘要器是「同构」的 ——

      · 有损：内容明显变短了
      · 但不是清零：开头的信息（比如用户交代的暗号）会保住

    正好够验证 v3 的全部逻辑：压缩流水线、摘要的位置、滚动合并、失败回退……
    唯一不验证的就是「真实模型写摘要写得好不好」——
    那件事只能在 EXPERIMENTS.md 里用真模型肉眼看。
    """

    def __init__(self, *, fail: bool = False, keep_chars: int = 36) -> None:
        self.fail = fail
        self.keep_chars = keep_chars
        self.calls = 0

    def summarize(
        self,
        messages: list[dict],
        previous_summary: str | None,
    ) -> SummarizeResult:
        self.calls += 1
        if self.fail:
            raise RuntimeError("模拟压缩失败（就当是网络断了）")

        parts: list[str] = []
        if previous_summary:
            # 真实摘要器也是这么干的：把旧摘要一起吞进来，产出更新的摘要。
            # 少了这一步，每次压缩都会把上辈子的记忆冲掉。
            parts.append(previous_summary)
        for msg in messages:
            if msg.get("role") == "tool":
                continue  # 真实摘要器也不会逐条抄工具原始输出
            content = str(msg.get("content") or "").strip().replace("\n", " ")
            if content:
                parts.append(content[: self.keep_chars])

        return SummarizeResult(
            text=" ｜ ".join(parts),
            prompt_tokens=300,
            completion_tokens=120,
            elapsed=0.01,
        )


# ── ① 工具层 ─────────────────────────────────────────────────────────


def _test_tools(check: _Check) -> None:
    """工具层测试 —— 与 v1/v2 完全相同的用例。

    这一组同时充当「拆文件 + 加压缩没有碰坏老东西」的回归网。
    """
    console.print("\n[bold]① 工具层测试（与 v1 相同的用例，验证老功能没被碰坏）[/bold]\n")
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
    check.ok(
        len(TOOL_DEFS) == 4,
        f"工具数 = 3 个老工具 + 1 个检索工具（{len(TOOL_DEFS)} 个）",
    )


# ── ② token 估算器 ───────────────────────────────────────────────────


def _test_token_estimate(check: _Check) -> None:
    console.print("\n[bold]② token 估算器测试[/bold]\n")
    check.ok(estimate_tokens("") == 0, "空文本 → 0 tokens")
    check.ok(estimate_tokens("你好世界") == 4, "4 个汉字 → 4 tokens")
    check.ok(estimate_tokens("abcdefgh") == 2, "8 个英文字符 → 2 tokens")
    check.ok(estimate_tokens("X" * 4000) == 1000, "4000 个英文字符 → 1000 tokens")


# ── ③ 剪枝（v2 的老本事，回归验证）───────────────────────────────────


def _test_snipping(check: _Check) -> None:
    console.print("\n[bold]③ 剪枝测试（v2 的老本事，确认 v3 没碰坏它）[/bold]\n")

    # ── 场景 A：预算只需剪工具结果就够 ──────────────────────────────
    cm = ContextManager(SYSTEM_PROMPT, max_tokens=12_000, keep_recent=0, summarizer=None)
    cm.messages = _make_fake_conversation(20, tool_blob_chars=4000)
    n_before = len(cm.messages)
    report = cm.prepare()
    _print_report(report)

    check.ok(report.snipped_tool_results > 0, f"第一级剪掉了 {report.snipped_tool_results} 条工具结果")
    check.ok(report.compacted_rounds == 0, "剪一剪就够了，不惊动压缩器")
    check.ok(len(cm.messages) == n_before, "⭐ 剪枝只换 content、不删消息，条数不变")
    check.ok(not check_invariants(cm.messages), "⭐ 剪枝后无孤儿 tool 消息，结构健康")
    placeholders = [
        m for m in cm.messages if m.get("role") == "tool" and m["content"] == SNIP_PLACEHOLDER
    ]
    check.ok(
        len(placeholders) == report.snipped_tool_results,
        "⭐ 剪掉的工具结果原样保留在上下文里，没有被丢弃",
    )

    # ── 场景 B：不超预算时完全不动（不误伤）────────────────────────
    cm2 = ContextManager(SYSTEM_PROMPT, max_tokens=100_000, keep_recent=8, summarizer=None)
    cm2.append({"role": "user", "content": "你好"})
    snapshot = [dict(m) for m in cm2.messages]
    report2 = cm2.prepare()
    check.ok(not report2.did_trim and cm2.messages == snapshot, "未超预算时完全不动（不误伤）")


# ── ④ 上下文压缩（v3 的核心）─────────────────────────────────────────

_SECRET = "紫色河马"

# 压缩测试的「标准场景」参数：
# 20 轮工具对话 + 开头一句暗号 ≈ 2,500 tokens，而预算只有 1,000 ——
# 超出的部分（约 1,500）**光靠剪工具结果救不回来**（剪光也只能省 700 左右），
# 于是必然走到「压缩」这条路上。这样每条断言都发生在真实超标的状态下。
_CASE_ROUNDS = 20
_CASE_BLOB = 200
_CASE_BUDGET = 1_000


def _secret_line() -> str:
    return f"请记住这个暗号：{_SECRET}。回复「好的」就行。"


def _test_compaction(check: _Check) -> None:
    console.print("\n[bold]④ 上下文压缩测试（v3 的核心）[/bold]\n")

    # ── D1：超预算 → 压缩（v3 的主路）──────────────────────────────
    console.print("  [bold]D1 · 旧对话被「压成摘要」，而不是丢掉[/bold]")
    fake = _FakeSummarizer()
    cm = ContextManager(
        SYSTEM_PROMPT, max_tokens=_CASE_BUDGET, keep_recent=6, summarizer=fake
    )
    cm.messages = _make_fake_conversation(
        _CASE_ROUNDS, opening=_secret_line(), tool_blob_chars=_CASE_BLOB
    )
    n_before = len(cm.messages)
    tail = _tail_snapshot(cm)
    report = cm.prepare()
    _print_report(report)

    check.ok(fake.calls == 1, "压缩器被调用了一次（超预算才动手）")
    check.ok(report.compacted_rounds >= 1, f"压缩掉了 {report.compacted_rounds} 轮旧对话")
    check.ok(cm.has_summary, "上下文里多了一条摘要（这就是「长期记忆」）")
    check.ok(
        cm.messages[1].get("role") == "system"
        and SUMMARY_BLOCK_HEADER in cm.messages[1]["content"],
        "摘要以 system 消息的身份插在 index 1（system 提示词之后）",
    )
    check.ok(_SECRET in (cm.summary_text or ""), f"⭐ 暗号「{_SECRET}」被摘要保住了")
    check.ok(
        0 < report.compaction_summary_tokens < report.compaction_input_tokens,
        f"⭐ 压缩确实变短了：{report.compaction_input_tokens:,} → "
        f"{report.compaction_summary_tokens:,} tokens",
    )
    check.ok(len(cm.messages) < n_before, "被压缩的原文已经删掉（不是复制一份）")
    check.ok(not check_invariants(cm.messages), "⭐ 压缩后无孤儿 tool 消息，结构健康")
    check.ok(
        report.tokens_after <= cm.max_tokens or report.hit_floor,
        f"要么回到预算内、要么如实标记 hit_floor"
        f"（实际 {report.tokens_after:,} / 预算 {cm.max_tokens:,}）",
    )
    check.ok(_tail_unchanged(cm, tail), "⭐ 最后一条 user 之后的内容一字未动（当前轮受保护）")
    check.ok(cm.messages[0]["role"] == "system", "第 0 条仍是 system 提示词")

    # ── D2：对照组 —— 没有压缩器时，退回 v2 的丢弃 ──────────────────
    console.print("\n  [bold]D2 · 对照组：关掉压缩，看 v2 是怎么失忆的[/bold]")
    cm_v2 = ContextManager(
        SYSTEM_PROMPT, max_tokens=_CASE_BUDGET, keep_recent=6, summarizer=None
    )
    cm_v2.messages = _make_fake_conversation(
        _CASE_ROUNDS, opening=_secret_line(), tool_blob_chars=_CASE_BLOB
    )
    report_v2 = cm_v2.prepare()
    _print_report(report_v2)

    check.ok(
        report_v2.compacted_rounds == 0 and report_v2.dropped_rounds > 0,
        "没有压缩器 → 走 v2 的丢弃路径",
    )
    check.ok(
        report_v2.compaction_skipped is not None and "关闭" in report_v2.compaction_skipped,
        "⭐ 日志里写明了「压缩被关闭」，而不是默默丢弃",
    )
    check.ok(
        not _context_contains(cm_v2.messages, _SECRET),
        f"⭐ 对照：v2 把暗号「{_SECRET}」彻底弄丢了（不是变模糊，是不存在了）",
    )
    check.ok(
        _context_contains(cm.messages, _SECRET),
        "⭐ 对比：v3 的上下文里还能找到它 —— 这就是这一版的全部意义",
    )

    # ── D3：压第二次 —— 旧摘要要被合并，不能被冲掉 ──────────────────
    console.print("\n  [bold]D3 · 压了又压：两次压缩后，两次的记忆都还在[/bold]")
    fake3 = _FakeSummarizer()
    cm3 = ContextManager(
        SYSTEM_PROMPT, max_tokens=_CASE_BUDGET, keep_recent=6, summarizer=fake3
    )
    cm3.messages = _make_fake_conversation(
        _CASE_ROUNDS,
        opening="请记住这个暗号：紫色河马。回复「好的」就行。",
        tool_blob_chars=_CASE_BLOB,
    )
    cm3.prepare()
    cm3.messages.extend(
        _make_fake_conversation(
            _CASE_ROUNDS,
            opening="第二个暗号是：绿色骆驼。回复「好的」就行。",
            tool_blob_chars=_CASE_BLOB,
        )[1:]  # 去掉重复的 system
    )
    report3 = cm3.prepare()
    _print_report(report3)

    check.ok(fake3.calls == 2, "第二次超预算，又压了一次")
    check.ok(
        _context_contains(cm3.messages, "紫色河马") and _context_contains(cm3.messages, "绿色骆驼"),
        "⭐ 两个暗号都还在 —— 旧摘要被并进新摘要，没有被冲掉",
    )
    check.ok(
        sum(1 for m in cm3.messages if m.get("role") == "system") == 2,
        "摘要始终只占一条 system 消息（不会越压越多）",
    )
    check.ok(not check_invariants(cm3.messages), "反复压缩后结构依然健康")

    # ── D4：压缩失败 → 自动回退，绝不连累对话 ───────────────────────
    console.print("\n  [bold]D4 · 压缩失败：不崩、不卡，自动回退到丢弃[/bold]")
    fake4 = _FakeSummarizer(fail=True)
    cm4 = ContextManager(
        SYSTEM_PROMPT, max_tokens=_CASE_BUDGET, keep_recent=6, summarizer=fake4
    )
    cm4.messages = _make_fake_conversation(
        _CASE_ROUNDS, opening=_secret_line(), tool_blob_chars=_CASE_BLOB
    )
    report4 = cm4.prepare()  # ← 这一行本身就是断言：异常不该抛到这里
    _print_report(report4)

    check.ok(report4.compaction_error is not None, "压缩异常被接住，并记在报告里")
    check.ok(report4.dropped_rounds > 0, "自动回退到 v2 的「丢弃」")
    check.ok(cm4.compaction_failures == 1 and cm4.compactions == 0, "账本记得没错：成功 0 次、失败 1 次")
    check.ok(not check_invariants(cm4.messages), "回退之后结构依然健康")
    check.ok(
        report4.tokens_after <= cm4.max_tokens or report4.hit_floor,
        "回退之后照样把上下文弄回预算内",
    )

    # ── D5：值不值得压，是一笔账 ────────────────────────────────────
    console.print("\n  [bold]D5 · 「值不值得压」是一笔账，阈值真的在管事[/bold]")
    fake5 = _FakeSummarizer()
    cm5 = ContextManager(SYSTEM_PROMPT, max_tokens=60, keep_recent=0, summarizer=fake5)
    cm5.messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "你" * 20},
        {"role": "assistant", "content": "嗯"},
        {"role": "user", "content": "再" * 20},
        {"role": "assistant", "content": "好"},
    ]
    report5 = cm5.prepare()
    _print_report(report5)

    check.ok(fake5.calls == 0, "⭐ 内容太少，一次都没调压缩器 —— 这笔账算不过来，不如直接丢")
    check.ok(report5.dropped_rounds > 0, "走的还是 v2 的丢弃路径")

    # 同一个场景，只把阈值调高 —— 本来该压的也不压了
    fake5b = _FakeSummarizer()
    cm5b = ContextManager(
        SYSTEM_PROMPT,
        max_tokens=_CASE_BUDGET,
        keep_recent=6,
        summarizer=fake5b,
        min_tokens_to_compact=100_000,  # 大到永远凑不够
    )
    cm5b.messages = _make_fake_conversation(
        _CASE_ROUNDS, opening=_secret_line(), tool_blob_chars=_CASE_BLOB
    )
    report5b = cm5b.prepare()
    check.ok(fake5b.calls == 0, "阈值调大之后，该压的也不压了（阈值真的在管事）")
    check.ok(
        report5b.compaction_skipped is not None and "阈值" in report5b.compaction_skipped,
        "⭐ 而且日志把「为什么没压」写清楚了（差值多少、该改哪个参数）",
    )
    check.ok(
        report5b.dropped_rounds > 0 and not _context_contains(cm5b.messages, _SECRET),
        "副作用：暗号又丢了 —— 阈值调过头，v3 就在悄悄退回成 v2",
    )

    # ── D6：压一次就够，不会反复烧钱 ────────────────────────────────
    console.print("\n  [bold]D6 · 压一次就够：预算内不会反复调压缩器[/bold]")
    fake6 = _FakeSummarizer()
    cm6 = ContextManager(
        SYSTEM_PROMPT, max_tokens=_CASE_BUDGET, keep_recent=6, summarizer=fake6
    )
    cm6.messages = _make_fake_conversation(
        _CASE_ROUNDS, opening=_secret_line(), tool_blob_chars=_CASE_BLOB
    )
    cm6.prepare()
    calls_after_first = fake6.calls
    cm6.prepare()
    cm6.prepare()
    check.ok(
        fake6.calls == calls_after_first,
        f"⭐ 连续 prepare 三次，压缩器调用次数没变（始终 {fake6.calls} 次）",
    )

    # ── D7：超了预算但没东西可动，也要说清楚 ────────────────────────
    console.print("\n  [bold]D7 · 没东西可动时，别沉默[/bold]")
    cm7 = ContextManager(
        SYSTEM_PROMPT, max_tokens=60, keep_recent=8, summarizer=_FakeSummarizer()
    )
    cm7.messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好，有什么可以帮你的？"},
    ]
    report7 = cm7.prepare()
    _print_report(report7)

    check.ok(not report7.did_trim, "确实什么都没动（能动的都受保护）")
    check.ok(
        report7.compaction_skipped is not None and "保护区" in report7.compaction_skipped,
        "⭐ 但日志没有沉默：说出来「没有可压缩的完整轮次」",
    )
    check.ok(report7.should_report, "并且它知道这种情况值得汇报（should_report 为真）")


# ── ⑤ 会话持久化（v4 的核心）─────────────────────────────────────────


def _test_storage(check: _Check, tmp_dir: Path) -> None:
    """全部在临时目录里测 —— 绝不碰你真实的 session.json。"""
    console.print("\n[bold]⑤ 会话持久化测试（v4 的核心）[/bold]\n")

    path = tmp_dir / "session.json"

    # ── E1：存进去，读回来 ──────────────────────────────────────────
    console.print("  [bold]E1 · 存盘 → 读回：会话还在吗[/bold]")
    cm = ContextManager(
        SYSTEM_PROMPT, max_tokens=_CASE_BUDGET, keep_recent=6, summarizer=_FakeSummarizer()
    )
    cm.messages = _make_fake_conversation(
        _CASE_ROUNDS, opening=_secret_line(), tool_blob_chars=_CASE_BLOB
    )
    cm.prepare()  # 顺手压一次，让摘要也参与存盘

    messages, summary = cm.export()
    save = save_session(path, messages=messages, summary=summary)
    console.print(
        Text(
            f"    💾 {save.message_count} 条消息 · {human_size(save.bytes_written)}"
            f" · {save.elapsed_ms:.1f} ms",
            style="dim",
        )
    )

    check.ok(path.is_file(), "会话文件已经落盘")
    check.ok(
        not path.with_name(path.name + ".tmp").exists(),
        "⭐ 临时文件没有残留（原子写收尾干净）",
    )
    raw = path.read_text(encoding="utf-8")
    check.ok(SYSTEM_PROMPT not in raw, "⭐ 提示词没被存进文件（它属于代码，不属于会话）")
    check.ok(SUMMARY_BLOCK_HEADER not in raw, "摘要是以纯文本单独存的（头尾标记由代码现拼）")

    loaded = load_session(path)
    check.ok(loaded.status == "restored", "读回来的状态是 restored")
    check.ok(loaded.messages == messages, "⭐ 消息一字不差地回来了（含 tool_calls）")
    check.ok(loaded.summary == summary, "摘要也回来了")

    # ── E2：真的能「接着聊」吗 ──────────────────────────────────────
    console.print("\n  [bold]E2 · 换个新实例恢复：它还记得暗号吗[/bold]")
    cm2 = ContextManager(
        SYSTEM_PROMPT, max_tokens=_CASE_BUDGET, keep_recent=6, summarizer=_FakeSummarizer()
    )
    problems = cm2.restore(loaded.messages, loaded.summary)

    check.ok(not problems, f"恢复时结构校验通过（问题数 {len(problems)}）")
    check.ok(cm2.has_summary, "摘要被重新装回上下文（长期记忆回来了）")
    check.ok(_SECRET in (cm2.summary_text or ""), f"⭐ 暗号「{_SECRET}」跨实例活下来了")
    check.ok(cm2.messages[0]["content"] == SYSTEM_PROMPT, "第 0 条是当前代码里的 system 提示词")
    check.ok(
        len(cm2.messages) == len(messages) + 2,
        f"消息数 = system + 摘要 + 会话消息（实际 {len(cm2.messages)}）",
    )
    check.ok(not check_invariants(cm2.messages), "⭐ 恢复后结构健康（无孤儿 tool 消息）")

    # ── E3~E5：三种坏情况，一个都不能崩 ─────────────────────────────
    console.print("\n  [bold]E3 · 各种坏情况：磁盘上什么都可能发生[/bold]")
    missing = load_session(tmp_dir / "not-there.json")
    check.ok(missing.status == "missing", "文件不存在 → missing（第一次运行很正常）")

    bad = tmp_dir / "bad.json"
    bad.write_text("{ 这不是合法的 JSON", encoding="utf-8")
    corrupt = load_session(bad)
    check.ok(corrupt.status == "corrupt", "坏文件 → corrupt，异常没有上抛")
    check.ok(
        bad.with_name(bad.name + ".corrupt").is_file(),
        "⭐ 坏文件被改名备份（程序没资格替你删数据）",
    )
    check.ok("从零开始" in corrupt.note, "note 里讲清楚了这次会从零开始")

    future = tmp_dir / "future.json"
    future.write_text(
        json.dumps({"version": 999, "messages": [], "summary": None}), encoding="utf-8"
    )
    fut = load_session(future)
    check.ok(fut.status == "future_version", "更新的格式 → future_version，不敢乱读")
    check.ok(future.is_file(), "⭐ 而且原文件保持原样（不当作坏文件处理）")

    # ── E6：坏结构拦在门口 ─────────────────────────────────────────
    console.print("\n  [bold]E4 · 结构坏了：拦在门口，而不是等到请求模型时才炸[/bold]")
    orphan = [
        {"role": "user", "content": "你好"},
        {"role": "tool", "tool_call_id": "找不到的 id", "name": "calculator", "content": "1"},
    ]
    cm3 = ContextManager(SYSTEM_PROMPT, max_tokens=_CASE_BUDGET, keep_recent=6)
    cm3.append({"role": "user", "content": "我正在说的话"})
    before = [dict(m) for m in cm3.messages]
    problems3 = cm3.restore(orphan, None)
    check.ok(problems3, f"坏结构被拦住了（{len(problems3)} 个问题）")
    check.ok(cm3.messages == before, "⭐ 拒绝恢复时，当前状态一字未动")

    # ── E7：删除 ───────────────────────────────────────────────────
    check.ok(delete_session(path), "删除会话文件 → True")
    check.ok(not path.is_file(), "文件真的没了")
    check.ok(not delete_session(path), "再删一次 → False（本来就没有）")


# ── ⑥ 本地检索（v5 的核心）───────────────────────────────────────────

_DOC_PROJECT = """# 晨曦项目

本周五要交付第一版。负责人是老王，验收方式写在下面。

## 交付物

- 一个命令行工具
- 一份使用说明

## 风险

第三方接口可能延期，要提前准备备选方案。
"""

_DOC_GARDEN = """# 园艺笔记

番茄需要每天浇水。阳台朝向会影响光照时长。

## 堆肥

厨余垃圾可以堆肥，但要避免油腻的东西。
"""


def _test_retrieval(check: _Check, tmp_dir: Path) -> None:
    """全部在临时目录里跑 —— 不碰你真实的 notes/。"""
    console.print("\n[bold]⑥ 本地检索测试（v5 的核心）[/bold]\n")

    kb_dir = tmp_dir / "notes"
    kb_dir.mkdir(parents=True, exist_ok=True)
    (kb_dir / "chenxi.md").write_text(_DOC_PROJECT, encoding="utf-8")
    (kb_dir / "garden.md").write_text(_DOC_GARDEN, encoding="utf-8")

    # ── F1：切分 ────────────────────────────────────────────────────
    console.print("  [bold]F1 · 切分：按标题和段落切，不按字数硬切[/bold]")
    chunks = split_markdown(_DOC_PROJECT, "chenxi.md")
    console.print(
        Text("    " + " ｜ ".join(f"「{c.heading}」L{c.line}" for c in chunks), style="dim")
    )

    check.ok(len(chunks) == 3, f"一篇三小节的笔记，切成了 {len(chunks)} 块")
    check.ok(chunks[0].heading == "晨曦项目", "每块还记得自己属于哪个小节")
    check.ok(any(c.heading == "交付物" for c in chunks), "`##` 小节标题被认出来了")
    check.ok(all(c.source == "chenxi.md" for c in chunks), "每块还记得自己来自哪个文件")
    check.ok(all(c.line >= 1 for c in chunks), "每块还记得自己的行号")

    # ── F2：分词 ────────────────────────────────────────────────────
    console.print("\n  [bold]F2 · 分词：中文没有空格，用 bigram 切[/bold]")
    terms = tokenize("性能优化")
    console.print(Text(f"    「性能优化」→ {terms}", style="dim"))

    check.ok("性能" in terms and "优化" in terms, "「性能优化」被切成 性能/能优/优化")
    check.ok("python" in tokenize("Python 代码"), "英文转小写后保留")
    check.ok("the" not in tokenize("the python"), "英文停用词被丢掉（它们没有区分度）")

    # ── F3：IDF ────────────────────────────────────────────────────
    tiny = build_index(
        [
            Chunk(source="a", line=1, heading="", text="苹果 苹果 苹果"),
            Chunk(source="b", line=1, heading="", text="苹果 香蕉"),
        ]
    )
    check.ok(
        tiny.idf("香蕉") > tiny.idf("苹果"),
        "⭐ IDF 生效：只出现在一块里的词，权重高于到处都是的词",
    )

    # ── F4：检索 ────────────────────────────────────────────────────
    console.print("\n  [bold]F4 · 检索：问对的问题，能不能找到对的那段[/bold]")
    kb = KnowledgeBase(kb_dir, chunk_max_chars=600)
    outcome = kb.search("交付物是什么", top_k=3)
    for hit in outcome.hits:
        console.print(Text(f"    {hit.score:.2f}  {hit.chunk.citation()}", style="dim"))

    check.ok(bool(outcome.hits), "检索到了内容")
    top = outcome.hits[0]
    check.ok(
        top.chunk.source == "chenxi.md",
        f"⭐ 排第一的来自晨曦项目，没有串到园艺笔记（{top.chunk.citation()}）",
    )
    check.ok(
        "交付" in top.chunk.heading or "交付" in top.chunk.text,
        "而且确实是最相关的那一块",
    )
    check.ok(":" in top.chunk.citation(), "结果带出处（文件:行号），能回溯到原文")
    check.ok(bool(top.matched), f"结果里记着命中了哪些词（{top.matched[:3]}）")

    # ── F5：找不到就是找不到 ────────────────────────────────────────
    miss = kb.search("网球拍穿线磅数", top_k=3)
    check.ok(miss.is_empty, "⭐ 问资料里没有的东西 → 0 命中，而不是硬凑一段")

    # ── F6：资料变了，索引要跟上 ────────────────────────────────────
    console.print("\n  [bold]F5 · 新写的笔记，下一次检索就能看到[/bold]")
    before = kb.chunk_count
    (kb_dir / "new.md").write_text(
        "# 新笔记\n\n这里记了关于量子纠缠的一点想法。\n", encoding="utf-8"
    )
    fresh = kb.search("量子纠缠")
    check.ok(bool(fresh.hits), "⭐ 刚写进去的文件，马上就能检索到（索引自动重建）")
    check.ok(kb.chunk_count > before, f"块数跟着涨了（{before} → {kb.chunk_count}）")
    check.ok(kb.rebuilds >= 2, f"索引重建过 {kb.rebuilds} 次")

    # ── F7：坏文件不能拖垮检索 ──────────────────────────────────────
    (kb_dir / "broken.md").write_bytes(b"\xff\xfe\x00\x00not utf-8")
    kb.search("随便找找")
    check.ok(kb.skipped_files >= 1, "读不了的文件被跳过，检索没有崩（同 v4 的容错精神）")

    # ── F8：目录不存在 ──────────────────────────────────────────────
    empty = KnowledgeBase(tmp_dir / "不存在的目录")
    outcome_empty = empty.search("任意")
    check.ok(
        outcome_empty.is_empty and empty.chunk_count == 0,
        "资料目录不存在 → 返回空，而不是崩",
    )


# ── ⑦ 向量检索（v6 的核心）──────────────────────────────────────────

_VEC_DIM = 64


class _FakeEmbedder:
    """离线测试用的假向量器：**词袋向量**。

    真实向量器要联网、要花钱、结果还不可复现（同一句话每次未必完全一样）。
    这个替身把每个「词」哈希到固定维度上 —— **词重叠越多，向量越像**。

    ⚠️ 它模拟不了「语义相似」（那是真模型的活，只能在 EXPERIMENTS 里肉眼看）。
       但整条流水线 —— 缓存、门槛、排序、降级 —— 都能用它测穿，
       而且一分钱不花、每次结果都一样。
    """

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0
        self.stats = EmbedStats()

    @property
    def model_id(self) -> str:
        return "fake-bag-v1"

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.stats.calls += 1
        self.stats.texts += len(texts)
        if self.fail:
            raise RuntimeError("模拟向量服务挂掉")
        return [self._vector(t) for t in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        vec = [0.0] * _VEC_DIM
        for term in tokenize(text):
            idx = int(hashlib.sha256(term.encode()).hexdigest()[:8], 16) % _VEC_DIM
            vec[idx] += 1.0
        return vec


_DOC_PERF = """# 性能

性能优化有三个方向：加缓存、减少重复计算、把串行改并行。
"""


def _test_vector_retrieval(check: _Check, tmp_dir: Path) -> None:
    """全部离线 + 全部在临时目录：不联网、不花钱、不碰你的资料和缓存。"""
    console.print("\n[bold]⑦ 向量检索测试（v6 的核心）[/bold]\n")

    kb_dir = tmp_dir / "vecnotes"
    kb_dir.mkdir(parents=True, exist_ok=True)
    (kb_dir / "perf.md").write_text(_DOC_PERF, encoding="utf-8")

    fake = _FakeEmbedder()
    cache = EmbeddingCache(tmp_dir / "vec-cache.json")
    cache.load(fake.model_id)
    kb = KnowledgeBase(
        kb_dir,
        mode="semantic",
        embedder=fake,
        cache=cache,
        min_similarity=0.3,
    )

    # ── G1：向量这条路真的通 ────────────────────────────────────────
    console.print("  [bold]G1 · 向量模式：能找到相关内容[/bold]")
    out = kb.search("性能优化", top_k=3)
    console.print(Text(f"    模式={out.mode} 命中={len(out.hits)} 最高分={out.best_score:.3f}", style="dim"))

    check.ok(out.mode == "semantic", "这次走的是向量，不是关键词")
    check.ok(bool(out.hits), "向量检索找到了内容")
    check.ok(fake.calls >= 1, f"确实调用了向量服务（{fake.calls} 次）")

    # ── G2：门槛把不相关的挡在外面 ──────────────────────────────────
    console.print("\n  [bold]G2 · 门槛：不相关的挡在门外，而不是硬塞给你[/bold]")
    miss = kb.search("番茄需要浇水", top_k=3)
    check.ok(miss.is_empty, "⭐ 无关查询 → 0 命中（不设门槛的话 top_k 会硬塞）")
    check.ok(
        miss.best_score >= 0,
        f"而且报告里给出了「最高只有 {miss.best_score:.3f}」—— 能解释为什么是 0",
    )

    # ── G3：缓存 —— 第二次不花钱 ────────────────────────────────────
    console.print("\n  [bold]G3 · 缓存：同样的内容，第二次绝不重算[/bold]")
    fake2 = _FakeEmbedder()
    cache2 = EmbeddingCache(tmp_dir / "vec-cache.json")
    cache2.load(fake2.model_id)
    check.ok(cache2.status == "loaded", f"从磁盘读回了缓存（{cache2.size} 个向量）")

    kb2 = KnowledgeBase(kb_dir, mode="semantic", embedder=fake2, cache=cache2, min_similarity=0.3)
    kb2.search("性能优化")
    check.ok(
        fake2.stats.cache_hits >= 1,
        f"⭐ 块向量全部命中缓存（{fake2.stats.cache_hits} 段没花钱）",
    )

    # ── G4：换模型 → 缓存作废，不是硬用 ─────────────────────────────
    console.print("\n  [bold]G4 · 换向量模型：旧缓存作废（而不是硬用错的向量）[/bold]")
    other = EmbeddingCache(tmp_dir / "vec-cache.json")
    other.load("另一个模型")
    check.ok(
        other.status == "discarded" and "对不上" in other.note,
        "⭐ 模型对不上 → 丢弃缓存并说明原因（混用不同模型的向量 = 一堆噪音）",
    )

    # ── G5：向量服务挂了 → 降级，而不是整个检索失效 ─────────────────
    console.print("\n  [bold]G5 · 向量失败了怎么办：降级，而不是瘫掉[/bold]")
    broken = _FakeEmbedder(fail=True)
    cache3 = EmbeddingCache(tmp_dir / "vec-cache-broken.json")
    cache3.load(broken.model_id)
    kb3 = KnowledgeBase(kb_dir, mode="semantic", embedder=broken, cache=cache3, min_similarity=0.3)
    degraded = kb3.search("性能优化")

    check.ok(degraded.mode.startswith("keyword"), f"降级到关键词检索（mode={degraded.mode}）")
    check.ok(degraded.note is not None, "而且把原因记在 note 里（不能偷偷降级）")
    check.ok(bool(degraded.hits), "降级之后照样能搜到东西（功能没丢）")

    # ── G6：缓存文件坏了 → 当作空的，不崩 ───────────────────────────
    bad_cache = tmp_dir / "bad-cache.json"
    bad_cache.write_text("{ 这不是 JSON", encoding="utf-8")
    broken_cache = EmbeddingCache(bad_cache)
    broken_cache.load("whatever")
    check.ok(
        broken_cache.status == "discarded",
        "缓存文件坏了 → 丢掉重来（大不了重算，绝不能崩）",
    )


def selftest() -> int:
    check = _Check()
    _test_tools(check)
    _test_token_estimate(check)
    _test_snipping(check)
    _test_compaction(check)
    # 持久化与检索的测试都在临时目录里跑，不会碰你真实的 session.json / notes /
    with tempfile.TemporaryDirectory() as tmp:
        _test_storage(check, Path(tmp))
        _test_retrieval(check, Path(tmp))
        _test_vector_retrieval(check, Path(tmp))
    return check.report()


# ══════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        return selftest()

    if "--config" in argv:
        # ⚠️ 这里必须用 Text 而不是直接传字符串：
        #    describe_config() 里有 "[v3 新增]" 这种方括号，直接交给 rich
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
                "在仓库根目录创建共享层配置、填上密钥即可：\n\n"
                "  [bold]cp ../.env.example ../.env[/bold]\n\n"
                "[dim]没有密钥也可以先跑自检：python main.py --selftest[/dim]",
                border_style="red",
                title="缺少配置",
            )
        )
        return 1

    client = LLMClient(API_KEY, BASE_URL, MODEL)

    # 压缩器只在开启时才装配 —— 关掉之后 ContextManager 收到的就是 None，
    # 它会自动退回 v2 的「丢弃」路径。这正是实验 3 要感受的东西。
    summarizer = (
        LLMSummarizer(client, model=SUMMARY_MODEL, max_tokens=SUMMARY_MAX_TOKENS)
        if SUMMARY_ENABLED
        else None
    )
    context = ContextManager(
        SYSTEM_PROMPT,
        max_tokens=MAX_CONTEXT_TOKENS,
        keep_recent=KEEP_RECENT_MESSAGES,
        summarizer=summarizer,
        min_tokens_to_compact=MIN_COMPACT_TOKENS,
    )
    agent = Agent(client, context)

    console.print(_banner())

    # ── 恢复上一次的会话（v4 的核心）──────────────────────────────
    save_notice_shown = False
    if PERSIST_ENABLED:
        loaded = load_session(SESSION_FILE)
        if loaded.restored:
            problems = context.restore(loaded.messages, loaded.summary)
            if problems:
                # 读到了、但结构不合法 —— 宁可这次从零开始，
                # 也不要带着坏结构去请求模型（那时报错更难懂）
                console.print(
                    f"[yellow]⚠ 会话文件的结构不合法，这次从零开始：{problems[0]}[/yellow]"
                )
            else:
                bits = [f"{len(loaded.messages)} 条消息"]
                if context.has_summary:
                    bits.append(f"含摘要 {context.summary_tokens:,} tokens")
                if loaded.age_seconds is not None:
                    bits.append(f"文件保存于 {describe_age(loaded.age_seconds)}")
                console.print(f"[cyan]📂 已恢复上次会话[/cyan] [dim]· {' · '.join(bits)}[/dim]")
        elif loaded.status == "missing":
            # 第一次运行，完全正常 —— 但也要说一句，别让用户以为坏了
            console.print(f"[dim]📂 {loaded.note}[/dim]")
        else:
            # corrupt / future_version —— 不行就不行，但必须讲清楚
            console.print(f"[yellow]📂 {loaded.note}[/yellow]")
    else:
        console.print(
            "[dim]📂 持久化已关闭（LLM_PERSIST_ENABLED=0）：这次聊的不会写盘，退出即蒸发[/dim]"
        )

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
            # ⚠️ 内存清了、磁盘没清 = 重启之后它又全想起来了。
            #   持久化带来的第一个坑就是「两个状态源会打架」。
            if PERSIST_ENABLED:
                removed = delete_session(SESSION_FILE)
                note = "磁盘上的会话文件也删掉了" if removed else "磁盘上本来就没有会话文件"
                console.print(f"[dim]已清空对话历史（摘要一起清掉）· {note}[/dim]")
            else:
                console.print("[dim]已清空对话历史（摘要也一起清掉了）[/dim]")
            continue
        if user_input == "/session":
            console.print(
                Panel(Text(_session_report(context)), border_style="cyan", title="会话文件")
            )
            continue
        if user_input == "/knowledge":
            body = KNOWLEDGE.describe() if KNOWLEDGE else "检索模块不可用"
            if not RETRIEVAL_ENABLED:
                body += "\n\n⚠ 但检索开关是关的（LLM_RETRIEVAL_ENABLED=0）—— 模型看不到 search_notes 这个工具"
            console.print(Panel(Text(body), border_style="green", title="资料库"))
            continue
        if user_input == "/context":
            console.print(Panel(Text(context.describe()), border_style="magenta", title="上下文状态"))
            continue
        if user_input == "/summary":
            if context.summary_text:
                console.print(
                    Panel(
                        Text(context.summary_text),
                        border_style="magenta",
                        title=f"当前摘要（{context.summary_tokens:,} tokens）",
                    )
                )
            else:
                console.print(
                    "[dim]还没有摘要 —— 等上下文第一次超出预算，压缩发生之后就有了。[/dim]"
                )
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
        console.print(f"[dim]{client.describe()}[/dim]")

        # ── 存盘（v4 的核心）────────────────────────────────────────
        # 每轮都存，而不是「退出时存」—— 防的是「聊了两小时，
        # 一不小心 Ctrl+C / 断网 / 笔记本没电，全部蒸发」。
        # 写盘是毫秒级的，这点开销换来「随时可以拍屁股走人」，很值。
        if PERSIST_ENABLED:
            messages, summary = context.export()
            saved = save_session(SESSION_FILE, messages=messages, summary=summary)
            if not save_notice_shown:
                console.print(
                    f"[dim]💾 已存盘 → {saved.path.name}"
                    f"（{human_size(saved.bytes_written)} · {saved.elapsed_ms:.1f} ms）"
                    f" —— 之后每轮都会自动存，不再重复提示；/session 可随时查看[/dim]"
                )
                save_notice_shown = True


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
