"""
命令行入口 —— 交互对话、评估、以及不需要 API Key 的自检。

    python main.py                                    # 交互对话（不带记忆，退出即忘）
    python main.py --eval                             # 跑评估：固定用例 → 分数 + 运行记录
    python main.py --eval --baseline latest           # ⭐ 和上一次对比：变好还是变坏
    python main.py --eval --only calc-big             # 只跑一条（失败时调试它）
    python main.py --eval --prompt prompts/system-degraded.md
    python main.py --eval --baseline latest --tag degraded   # 给这次运行起个名字
    python main.py --selftest                         # 离线自检（不需要 API Key，不花钱）
    python main.py --config                           # 看配置从哪来、当前生效的值是什么

--eval 的退出码约定（方便以后挂 CI）：
    0 = 全部用例通过 · 1 = 有用例没过 · 2 = 根本跑不起来（缺 Key / 用例集坏了）
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

import config
import evaluator
import tools as tools_module
from agent import Agent, ToolTrace
from evaluator import (
    CaseResult,
    CheckResult,
    CheckSpec,
    EvalCase,
    EvalError,
    RunReport,
    build_judge_messages,
    compare_runs,
    find_latest_run,
    format_case_line,
    format_comparison,
    format_failures,
    format_header,
    format_judge_notes,
    format_no_baseline_note,
    format_summary,
    load_cases,
    load_run,
    parse_judge_reply,
    run_all,
    run_check,
    run_judge,
    save_run,
    Submission,
)
from llm import LLMClient

console = Console()

KNOWN_FLAGS = {
    "--eval",
    "--baseline",
    "--only",
    "--tag",
    "--prompt",
    "--selftest",
    "--config",
    "--help",
    "-h",
}


# ══════════════════════════════════════════════════════════════════════
# 小工具
# ══════════════════════════════════════════════════════════════════════


def _get_arg(argv: list[str], flag: str) -> str | None:
    """取 --flag 的值。没给这个旗子返回 None；给了旗子但没给值返回 ""。"""
    if flag not in argv:
        return None
    idx = argv.index(flag)
    if idx + 1 < len(argv) and not argv[idx + 1].startswith("--"):
        return argv[idx + 1]
    return ""


def _unknown_flag(argv: list[str]) -> str | None:
    for token in argv:
        if token.startswith("--") and token not in KNOWN_FLAGS:
            return token
    return None


def _display_path(path: Path) -> str:
    """尽量显示相对路径（在 v9 目录下的显示成 runs/xxx.json）。"""
    try:
        return str(Path(path).resolve().relative_to(Path(config.__file__).resolve().parent))
    except ValueError:
        return str(path)


def _error_hint(exc: Exception) -> str:
    """把常见的报错翻译成人话（v1 同款）。"""
    text = str(exc).lower()
    if "401" in text or "authentication" in text or "api key" in text:
        return "看起来是 API Key 无效。检查仓库根 .env 里的 LLM_API_KEY 是否填对了。"
    if "tool" in text and ("not support" in text or "unsupported" in text or "invalid" in text):
        return "当前模型可能不支持 tool calling。换 deepseek-chat / gpt-4o-mini / qwen-plus。"
    if "404" in text or ("model" in text and "not found" in text):
        return "模型名可能写错了，检查 .env 里的 LLM_MODEL（还有 LLM_JUDGE_MODEL）。"
    if "connect" in text or "timeout" in text:
        return "连接不上模型服务，检查网络和 .env 里的 LLM_BASE_URL。"
    return ""


def _print_tool_trace(trace: ToolTrace) -> None:
    """交互模式下的工具实况（评估模式不打印 —— 屏幕留给成绩单）。"""
    args = trace.args_text
    if len(args) > 120:
        args = args[:120] + "…"
    if trace.ok:
        result = trace.result_text
        if len(result) > 160:
            result = result[:160] + "…"
        console.print(f"  [cyan]🔧[/cyan] [bold]{trace.name}[/bold]({args}) → [green]{result}[/green]")
    else:
        console.print(f"  [cyan]🔧[/cyan] [bold]{trace.name}[/bold]({args}) → [red]{trace.result_text}[/red]")


def _banner(prompt_label: str) -> Panel:
    active = tools_module.active_tool_names()
    lines = [
        "[bold]hello-agent[/bold] v9 · 评估与可观测（基于 v1 的最小 Agent）",
        f"[dim]模型 {config.MODEL}  @  {config.BASE_URL}[/dim]",
        f"[dim]工具 {', '.join(active) or '（全部被关闭）'} · 温度 {config.TEMPERATURE}[/dim]",
        f"[dim]提示词 {prompt_label}[/dim]",
        "",
        "[dim]本版故意不带：持久化记忆 / 检索 / 结构化输出 —— 那是 v2~v8 的课题，[/dim]",
        "[dim]这一版只专注一件事：让「好不好」变成能测量的数字。[/dim]",
        "",
        "[dim]评估    python main.py --eval [--baseline latest] · 自检 --selftest · 配置 --config[/dim]",
        "[dim]对话命令 /reset 清空（只在内存里，退出就忘）   /exit 退出[/dim]",
    ]
    if config.DISABLED_TOOLS:
        lines.insert(2, f"[bold red]⚠ 已关闭工具：{', '.join(sorted(config.DISABLED_TOOLS))}[/bold red]")
    return Panel("\n".join(lines), border_style="blue", title="Agent")


# ══════════════════════════════════════════════════════════════════════
# 交互对话
# ══════════════════════════════════════════════════════════════════════


def run_repl(argv: list[str]) -> int:
    unknown = _unknown_flag(argv)
    if unknown:
        console.print(f"[red]✗ 不认识的参数：{unknown}[/red]（可用：--eval/--selftest/--config/--help）")
        return 2

    prompt_arg = _get_arg(argv, "--prompt")
    if prompt_arg == "":
        console.print("[red]✗ --prompt 后面要跟文件路径，例如 --prompt prompts/system-degraded.md[/red]")
        return 2

    system_prompt, prompt_label, _ = config.load_system_prompt(
        Path(prompt_arg) if prompt_arg else None
    )
    if prompt_label.startswith("⚠"):
        console.print(f"[yellow]{prompt_label}[/yellow]")

    if not config.API_KEY:
        console.print(
            Panel(
                "[red]没有找到 API Key[/red]\n\n"
                "请先复制一份配置文件，然后填入你的密钥：\n\n"
                "  [bold]cp ../.env.example ../.env[/bold]\n\n"
                "编辑仓库根的 .env，把 LLM_API_KEY 填上即可。\n\n"
                "[dim]没有密钥也能先跑离线自检：python main.py --selftest[/dim]",
                border_style="red",
                title="缺少配置",
            )
        )
        return 1

    client = LLMClient(config.API_KEY, config.BASE_URL, config.MODEL)
    agent = Agent(client, system_prompt, on_tool=_print_tool_trace)

    console.print(_banner(prompt_label))

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
            console.print("[dim]已清空对话历史（内存里那份）[/dim]")
            continue

        try:
            run = agent.ask(user_input)
        except Exception as exc:
            console.print(f"\n[red]请求失败：{type(exc).__name__}: {exc}[/red]")
            hint = _error_hint(exc)
            if hint:
                console.print(f"[yellow]💡 {hint}[/yellow]")
            continue

        console.print(Panel(run.answer, border_style="blue", title="Agent"))
        # 可观测：每次回答都顺手报账（花了几次、多久、多少 token）
        console.print(
            f"[dim]↳ {run.rounds} 次模型调用 · {run.elapsed_ms / 1000:.1f}s"
            f" · {run.total_tokens:,} tokens[/dim]"
        )


# ══════════════════════════════════════════════════════════════════════
# 评估（本版的正题）
# ══════════════════════════════════════════════════════════════════════


def run_eval(argv: list[str]) -> int:
    baseline_arg = _get_arg(argv, "--baseline")
    prompt_arg = _get_arg(argv, "--prompt")
    only = _get_arg(argv, "--only")
    tag = _get_arg(argv, "--tag")

    for flag, value in (("--baseline", baseline_arg), ("--prompt", prompt_arg), ("--only", only), ("--tag", tag)):
        if value == "":
            console.print(
                f"[red]✗ {flag} 后面要跟一个值[/red]"
                f"（例如 {flag} latest / {flag} prompts/system-degraded.md）"
            )
            return 2

    if not config.API_KEY:
        console.print(
            Panel(
                "[red]评估要真的调用模型，没有 Key 跑不了[/red]\n\n"
                "先把仓库根 .env 里的 LLM_API_KEY 填上。\n\n"
                "[dim]想先看看评估器本身长什么样（不花钱）：python main.py --selftest[/dim]",
                border_style="red",
                title="缺少配置",
            )
        )
        return 2

    # ── 1. 读用例集 ─────────────────────────────────────────────────
    try:
        cases = load_cases(config.EVAL_CASES_FILE)
    except EvalError as exc:
        console.print(f"[red]✗ 用例集读不进来：[/red]\n{exc}")
        return 2

    # --only 的校验要放到表头之前 —— 否则先印一屏「只跑 1 条」再报错，
    # 学习者会以为跑过了（铁律 7：输出不能误导）
    if only and only not in {c.id for c in cases}:
        known = "、".join(c.id for c in cases)
        console.print(f"[red]✗ --only {only} 没有匹配到任何用例[/red]（现有 id：{known}）")
        return 2

    # ── 2. 解析基线（必须在写新的运行记录之前！）───────────────────
    # 不然「latest」找到的就是你刚写的那份，等于和自己比。
    baseline_path: Path | None = None
    if baseline_arg == "latest":
        baseline_path = find_latest_run(config.EVAL_RUNS_DIR)
    elif baseline_arg:
        baseline_path = Path(baseline_arg)

    # ── 3. 提示词 + 账本客户端 ──────────────────────────────────────
    system_prompt, prompt_label, prompt_sha = config.load_system_prompt(
        Path(prompt_arg) if prompt_arg else None
    )
    if prompt_label.startswith("⚠"):
        console.print(f"[yellow]{prompt_label}[/yellow]")

    client = LLMClient(config.API_KEY, config.BASE_URL, config.MODEL)

    meta: dict[str, Any] = {
        "model": config.MODEL,
        "temperature": config.EVAL_TEMPERATURE,
        "prompt_label": prompt_label,
        "prompt_sha": prompt_sha,
        "cases_file": str(config.EVAL_CASES_FILE),
        "disabled_tools": sorted(config.DISABLED_TOOLS),
        "tag": tag or "run",
    }

    console.print(Panel(format_header({**meta, "selected": len(cases), "cases_total": len(cases), "only": only}),
                        border_style="cyan", title="评估开始"))

    width = max((len(c.id) for c in cases), default=0)

    try:
        report = run_all(
            cases,
            make_agent=lambda: Agent(client, system_prompt),  # 每条用例都新建 —— 互不污染
            llm=client,
            notes_dir=config.NOTES_DIR,
            meta=meta,
            only=only,
            on_case_done=lambda r: console.print(format_case_line(r, width)),
        )
    except EvalError as exc:
        console.print(f"[red]✗ {exc}[/red]")
        return 2

    # ── 4. 成绩单 ───────────────────────────────────────────────────
    console.print()
    judge_notes = format_judge_notes(report)
    if judge_notes:
        console.print(judge_notes)
    failures = format_failures(report)
    if failures:
        console.print(failures)
        console.print()
    console.print(format_summary(report))

    run_path = save_run(report, config.EVAL_RUNS_DIR)
    console.print(f"📄 本次记录已写入 [bold]{_display_path(run_path)}[/bold]")

    # ── 5. 对比 ─────────────────────────────────────────────────────
    if baseline_path is not None:
        try:
            baseline = load_run(baseline_path)
        except EvalError as exc:
            console.print(f"[red]✗ 基线读不了，跳过对比：[/red]\n{exc}")
        else:
            console.print()
            console.print(format_comparison(compare_runs(report, baseline, _display_path(baseline_path))))
    else:
        console.print(format_no_baseline_note(_display_path(config.EVAL_RUNS_DIR)))

    # ── 6. 退出码：全过 0，有没过 1 ─────────────────────────────────
    return 0 if report.passed_count == report.total else 1


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


def _fake_tool_call(call_id: str, name: str, **args: Any) -> SimpleNamespace:
    """造一个和 SDK 形状一致的 tool_call 对象（剧本用）。"""
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(args, ensure_ascii=False)),
    )


def _step(content: str | None = None, tool_calls: list | None = None,
          prompt_tokens: int = 100, completion_tokens: int = 20,
          finish_reason: str | None = None) -> dict[str, Any]:
    return {
        "content": content,
        "tool_calls": tool_calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
    }


class _ScriptedLLM:
    """剧本客户端：每次 complete() 按剧本给下一个回复。

    （假替身第四次出场：v3 假摘要器 / v6 假向量器 / v8 剧本客户端。）
    真实模型不可能被你指挥着「先调错工具、再改对」——剧本可以。
    """

    def __init__(self, steps: list[dict[str, Any]]) -> None:
        self.steps = list(steps)
        self.calls: list[dict[str, Any]] = []
        self._calls = 0
        self._in = 0
        self._out = 0

    def snapshot(self) -> tuple[int, int, int]:
        return self._calls, self._in, self._out

    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> SimpleNamespace:
        self.calls.append({"messages": [dict(m) for m in messages], **kwargs})
        if not self.steps:
            raise AssertionError("剧本用完了 —— 测试少写了一步？")
        step = self.steps.pop(0)
        self._calls += 1
        self._in += step["prompt_tokens"]
        self._out += step["completion_tokens"]
        message = SimpleNamespace(content=step["content"], tool_calls=step["tool_calls"])
        return SimpleNamespace(
            message=message,
            prompt_tokens=step["prompt_tokens"],
            completion_tokens=step["completion_tokens"],
            finish_reason=step.get("finish_reason"),
        )


def _cs(check_type: str, **params: Any) -> CheckSpec:
    return CheckSpec(type=check_type, params=params)


def _mk_report(passed: dict[str, bool], sha: str = "aaaaaaaa", model: str = "m1",
               temp: str = "0.0") -> RunReport:
    return RunReport(
        meta={"prompt_sha": sha, "model": model, "temperature": temp},
        results=[
            CaseResult(id=i, prompt="（测试）", passed=p, checks=[], tool_names=[],
                       answer="", elapsed_ms=1.0)
            for i, p in passed.items()
        ],
    )


def selftest() -> int:
    """离线把评估器整条流水线测穿：用例加载 → 判定 → 跑分 → 存档 → 对比。"""
    check = _Check()
    console.print("[bold]离线自检（不调用真实模型、不花一分钱）[/bold]\n")

    # ── ① 用例集加载 ────────────────────────────────────────────────
    console.print("[bold]① 用例集加载[/bold]")
    real_cases = load_cases(config.EVAL_CASES_FILE)
    check.ok(len(real_cases) == 9, f"真实用例集能加载，共 {len(real_cases)} 条")
    check.ok(len({c.id for c in real_cases}) == len(real_cases), "用例 id 不重复")
    check.ok(all(c.prompt and c.checks for c in real_cases), "每条用例都有 prompt 和 checks")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        def _write_cases(text: str) -> Path:
            p = tmp_path / "cases.jsonl"
            p.write_text(text, encoding="utf-8")
            return p

        good_line = '{"id": "a", "prompt": "1+1?", "checks": [{"type": "no_tools"}]}'

        try:
            load_cases(_write_cases("# 注释行\n\n" + good_line + "\n"))
            check.ok(True, "注释行和空行会被跳过")
        except EvalError:
            check.ok(False, "注释行和空行会被跳过")

        try:
            load_cases(_write_cases(good_line + "\n{坏 JSON\n"))
            check.ok(False, "坏 JSON 行应该报错")
        except EvalError as exc:
            check.ok("第 2 行" in str(exc) and "JSON" in str(exc), "坏 JSON：报错带行号")

        try:
            load_cases(_write_cases('{"id": "a", "prompt": "x", "checks": [{"type": "没见过的类型"}]}'))
            check.ok(False, "未知检查类型应该报错")
        except EvalError as exc:
            check.ok("不认识的检查类型" in str(exc) and "没见过的类型" in str(exc),
                     "未知检查类型：报错带上类型名和可用清单")

        try:
            load_cases(_write_cases(good_line + "\n" + good_line))
            check.ok(False, "重复 id 应该报错")
        except EvalError as exc:
            check.ok("重复" in str(exc), "重复 id：报错说明会「对不上账」")

        try:
            load_cases(_write_cases('{"id": "a", "checks": [{"type": "no_tools"}]}'))
            check.ok(False, "缺 prompt 应该报错")
        except EvalError as exc:
            check.ok("prompt" in str(exc), "缺 prompt：报错点名缺什么")

        try:
            load_cases(_write_cases('{"id": "a", "prompt": "x", "checks": [{"type": "tool_called"}]}'))
            check.ok(False, "检查缺参数应该报错")
        except EvalError as exc:
            check.ok("缺少参数" in str(exc) and "name" in str(exc), "检查缺参数：报错点名参数")

        try:
            load_cases(_write_cases("# 只有注释\n"))
            check.ok(False, "空用例集应该报错")
        except EvalError as exc:
            check.ok("空的" in str(exc), "空用例集：报错说明「至少要有一个用例」")

        try:
            load_cases(tmp_path / "不存在.jsonl")
            check.ok(False, "文件不存在应该报错")
        except EvalError as exc:
            check.ok("不存在" in str(exc), "用例集文件不存在：报错带路径")

    # ── ② 确定性检查 ────────────────────────────────────────────────
    console.print("\n[bold]② 确定性检查（规则判定）[/bold]")
    sub = Submission(answer="答案是 7,006,742 ，没错。", tool_names=["calculator"], notes_dir=Path("/tmp"))

    r = run_check(_cs("tool_called", name="calculator"), sub)
    check.ok(r.ok and "calculator" in r.detail, "tool_called 通过：写明调用了谁")

    r = run_check(_cs("tool_called", name="get_current_time"), sub)
    check.ok(not r.ok and "没有调用" in r.detail and "calculator" in r.detail,
             "tool_called 不过：写明实际调了什么")

    r = run_check(_cs("tool_called", name="get_current_time"),
                  Submission(answer="x", tool_names=[], notes_dir=Path("/tmp")))
    check.ok(not r.ok and "一个都没有" in r.detail, "tool_called 不过且没调任何工具：也说得出来")

    r = run_check(_cs("no_tools"), Submission(answer="x", tool_names=[], notes_dir=Path("/tmp")))
    check.ok(r.ok, "no_tools 通过")

    r = run_check(_cs("no_tools"), sub)
    check.ok(not r.ok and "calculator" in r.detail, "no_tools 不过：点名乱调的工具")

    r = run_check(_cs("answer_contains", value="7006742"), sub)
    check.ok(r.ok, "answer_contains：对千分位逗号免疫（7,006,742 认作 7006742）")

    r = run_check(_cs("answer_contains", value="等于2"), Submission(answer="结果 等于 2 哦", tool_names=[], notes_dir=Path("/tmp")))
    check.ok(r.ok, "answer_contains：对空格免疫")

    r = run_check(_cs("answer_contains", value="巴黎"), Submission(answer="", tool_names=[], notes_dir=Path("/tmp")))
    check.ok(not r.ok and "没有" in r.detail and "（空回答）" in r.detail,
             "answer_contains 不过：如实说明是空回答")

    r = run_check(_cs("answer_not_contains", value="巴黎"), sub)
    check.ok(r.ok, "answer_not_contains 通过")

    r = run_check(_cs("answer_not_contains", value="7,006,742"), Submission(answer="是 7006742", tool_names=[], notes_dir=Path("/tmp")))
    check.ok(not r.ok and "不该出现" in r.detail, "answer_not_contains 不过：说明「不该出现」")

    r = run_check(_cs("answer_regex", value=r"23\.7"), Submission(answer="等于 23.70", tool_names=[], notes_dir=Path("/tmp")))
    check.ok(r.ok and "命中" in r.detail, "answer_regex 通过：展示命中的片段")

    r = run_check(_cs("answer_regex", value=r"星期[一二三四五六日天]"), Submission(answer="今天是 9 月 19 日星期五", tool_names=[], notes_dir=Path("/tmp")))
    check.ok(r.ok, "answer_regex：中文星期也能匹配")

    r = run_check(_cs("answer_regex", value="abc"), Submission(answer="xyz", tool_names=[], notes_dir=Path("/tmp")))
    check.ok(not r.ok and "不匹配" in r.detail and "回答开头" in r.detail, "answer_regex 不过：给出回答开头")

    r = run_check(_cs("answer_regex", value="[坏正则"), sub)
    check.ok(not r.ok and "用例里的正则写错了" in r.detail, "正则写错：点名这是「用例的问题，不是模型的问题」")

    with tempfile.TemporaryDirectory() as tmp:
        notes = Path(tmp)
        (notes / "homework.md").write_text("# 作业\n", encoding="utf-8")
        r = run_check(_cs("file_exists", value="homework.md"),
                      Submission(answer="", tool_names=["save_note"], notes_dir=notes))
        check.ok(r.ok and "字节" in r.detail and "本次运行新写入" in r.detail,
                 "file_exists 通过：报文件大小，并注明「本次新写入」")

        r = run_check(_cs("file_exists", value="break.md"),
                      Submission(answer="", tool_names=["save_note"], notes_dir=notes))
        check.ok(not r.ok and "homework.md" in r.detail, "file_exists 不过：列出目录里现有文件")

        st = (notes / "homework.md").stat()
        r = run_check(_cs("file_exists", value="homework.md"),
                      Submission(answer="", tool_names=[], notes_dir=notes,
                                 notes_before={"homework.md": (st.st_mtime_ns, st.st_size)}))
        check.ok(not r.ok and "不是这次写的" in r.detail,
                 "残留文件拦截：和跑之前一模一样 → 判定「不是这次写的」（防假通过）")

        empty = notes / "空的"
        empty.mkdir()
        r = run_check(_cs("file_exists", value="x.md"),
                      Submission(answer="", tool_names=[], notes_dir=empty))
        check.ok(not r.ok and "目录是空的" in r.detail, "file_exists 不过且目录为空：也说得出来")

    # ── ③ 裁判判定 ──────────────────────────────────────────────────
    console.print("\n[bold]③ 裁判判定（LLM-as-judge）[/bold]")
    parsed = parse_judge_reply("PASS\n解释很通俗，有比喻。")
    check.ok(parsed == (True, "解释很通俗，有比喻。"), "解析 PASS + 理由")
    parsed = parse_judge_reply("**FAIL**\n全是术语。")
    check.ok(parsed is not None and parsed[0] is False, "解析 FAIL：markdown 装饰也能容错")
    check.ok(parse_judge_reply("哈？") is None, "乱输出 → 解析失败（返回 None，不猜）")

    judge_spec = _cs("llm_judge", criterion="解释通俗")
    messages = build_judge_messages(judge_spec, "token 是什么", "就是一小块文字")
    joined = messages[0]["content"]
    check.ok("解释通俗" in joined and "token 是什么" in joined and "一小块文字" in joined,
             "裁判提示词里带上了标准 / 问题 / 待评回答")

    judge_reply = _step(content="PASS\n通俗易懂。")
    llm = _ScriptedLLM([judge_reply])
    r = run_judge(judge_spec, "token 是什么", sub, llm)
    check.ok(r.ok and "PASS" in r.detail and "通俗易懂" in r.detail, "裁判 PASS：理由被带进结果里")
    kwargs = llm.calls[0]
    check.ok(kwargs["use_tools"] is False, "裁判调用不带工具（它是评委，不是选手）")
    check.ok(kwargs["temperature"] == config.EVAL_TEMPERATURE, "裁判用评估温度（默认 0）")
    check.ok(kwargs.get("max_tokens") == config.JUDGE_MAX_TOKENS,
             f"裁判输出有上限（当前 {config.JUDGE_MAX_TOKENS} tokens）")

    llm = _ScriptedLLM([_step(content="FAIL\n太多术语。")])
    r = run_judge(judge_spec, "q", sub, llm)
    check.ok(not r.ok and "FAIL" in r.detail, "裁判 FAIL 如实记录")

    llm = _ScriptedLLM([_step(content="我也说不清")])
    r = run_judge(judge_spec, "q", sub, llm)
    check.ok(not r.ok and "无法解析" in r.detail and "按未通过记" in r.detail,
             "裁判输出解析不了：按未通过记，并留下原文")

    # ⭐ 实测坑（2026-09-19）：裁判被 max_tokens 截断 → 空内容 → 假红
    llm = _ScriptedLLM([_step(content="", finish_reason="length")])
    r = run_judge(judge_spec, "q", sub, llm)
    check.ok(not r.ok and "截断" in r.detail and "LLM_JUDGE_MAX_TOKENS" in r.detail,
             "裁判被截断：说清「这是判定环节故障，不是模型答得不好」+ 给出旋钮名")

    llm = _ScriptedLLM([_step(content="")])
    r = run_judge(judge_spec, "q", sub, llm)
    check.ok(not r.ok and "空内容" in r.detail and "判定环节" in r.detail,
             "裁判空回复：同样和「回答的问题」区分开")

    class _BrokenLLM:
        def complete(self, messages: list[dict], **kwargs: Any) -> Any:
            raise RuntimeError("网络断了")

    r = run_judge(judge_spec, "q", sub, _BrokenLLM())
    check.ok(not r.ok and "裁判调用失败" in r.detail, "裁判调用失败：不假装判过，说明修好再跑")

    # ── ④ Agent 循环（剧本重放，不联网）────────────────────────────
    console.print("\n[bold]④ Agent 循环：对轨迹的记录[/bold]")
    llm = _ScriptedLLM([
        _step(tool_calls=[_fake_tool_call("c1", "calculator", expression="1+1")]),
        _step(content="等于 2。"),
    ])
    events: list[ToolTrace] = []
    agent = Agent(llm, "测试提示词", on_tool=events.append)
    run = agent.ask("算 1+1")

    check.ok(run.answer == "等于 2。", "循环正常收工，拿到回答")
    check.ok(run.tool_names == ["calculator"], "轨迹记录了调用的工具名")
    check.ok(run.tools[0].ok and "1+1" in run.tools[0].args_text, "轨迹记录了参数原文")
    check.ok(run.tools[0].result_text == "2", "轨迹记录了工具结果")
    check.ok(len(run.tools) == len(events), "on_tool 回调次数 == 工具调用次数")
    check.ok(len(agent.messages) == 5, "messages 形状：system/user/assistant/tool/assistant")
    check.ok(run.prompt_tokens == 200 and run.completion_tokens == 40, "token 账记到了这一次运行上")
    check.ok(run.rounds == 2, "rounds = 实际模型调用次数")
    check.ok(llm.calls[0]["use_tools"] is True, "正常循环带工具")
    check.ok(llm.calls[1]["messages"][-1]["role"] == "tool", "工具结果以 role=tool 回灌")

    llm = _ScriptedLLM([
        _step(tool_calls=[_fake_tool_call("c1", "不存在的工具")]),
        _step(content="好吧，我直接回答。"),
    ])
    agent = Agent(llm, "测试提示词")
    run = agent.ask("试试坏工具")
    check.ok(not run.tools[0].ok and "错误" in run.tools[0].result_text, "工具报错：痕迹标记 ok=False")
    check.ok("错误" in llm.calls[1]["messages"][-1]["content"], "报错被当作工具输出回灌给模型")
    check.ok(run.answer == "好吧，我直接回答。", "报错不崩循环，它还能继续工作")

    llm = _ScriptedLLM(
        [_step(tool_calls=[_fake_tool_call(f"c{i}", "calculator", expression="1+1")]) for i in range(8)]
        + [_step(content="（兜底）用已有资料回答。")]
    )
    agent = Agent(llm, "测试提示词")
    run = agent.ask("一直在调工具")
    check.ok(run.forced_finish, "轮数用尽：标记 forced_finish（不是静默失败）")
    check.ok(llm.calls[-1]["use_tools"] is False, "兜底那次请求撤掉了工具（强制出答案）")
    check.ok(run.rounds == 9 and run.answer.startswith("（兜底）"), "兜底调用也计入 rounds，回答拿到手")

    # 两条用例 ⇒ 两个全新 Agent（互不污染）
    cases = [
        EvalCase(id="a", prompt="算 1+1", checks=[_cs("tool_called", name="calculator")]),
        EvalCase(id="b", prompt="法国的首都", checks=[_cs("answer_contains", value="巴黎")]),
    ]
    llm = _ScriptedLLM([
        _step(tool_calls=[_fake_tool_call("c1", "calculator", expression="1+1")]),
        _step(content="等于 2。"),
        _step(content="巴黎。"),
    ])
    with tempfile.TemporaryDirectory() as tmp:
        report = run_all(cases, make_agent=lambda: Agent(llm, "测试提示词"), llm=llm,
                         notes_dir=Path(tmp))
    check.ok(report.passed_count == report.total == 2, "两条用例都通过（端到端跑通）")
    check.ok(len(llm.calls[0]["messages"]) == 2 and len(llm.calls[2]["messages"]) == 2,
             "每条用例都从「system+user」开始 —— 上一条的对话不会污染下一条")

    # ── ⑤ 报告与运行记录 ────────────────────────────────────────────
    console.print("\n[bold]⑤ 报告与运行记录[/bold]")
    tool_check = CheckResult(type="tool_called", category="工具行为", ok=True, detail="调用了 calculator")
    content_check = CheckResult(type="answer_contains", category="回答内容", ok=False, detail="回答里没有「巴黎」")
    judge_check = CheckResult(type="llm_judge", category="模型判定", ok=True, detail="裁判 PASS：通俗")
    report = RunReport(
        meta={"tag": "测试", "prompt_sha": "aaaa1111", "model": "m1", "temperature": 0.0},
        results=[
            CaseResult(id="ok", prompt="p1", passed=True, checks=[tool_check], tool_names=["calculator"],
                       answer="2", elapsed_ms=1000.0),
            CaseResult(id="bad", prompt="p2", passed=False, checks=[tool_check, content_check],
                       tool_names=["calculator"], answer="伦敦", elapsed_ms=3000.0),
            CaseResult(id="judged", prompt="p3", passed=True, checks=[judge_check], tool_names=[],
                       answer="…", elapsed_ms=2000.0),
        ],
        calls=5, prompt_tokens=1000, completion_tokens=200,
    )
    check.ok(report.total == 3 and report.passed_count == 2, "汇总：3 条用例过 2 条")
    check.ok(abs(report.pass_rate - 66.7) < 0.1, "通过率算对（66.7%）")
    check.ok(abs(report.avg_ms - 2000.0) < 1, "平均耗时算对")
    stats = report.category_stats()
    check.ok(stats["工具行为"] == (2, 2) and stats["回答内容"] == (0, 1) and stats["模型判定"] == (1, 1),
             "分类统计（检查项口径）算对")
    check.ok(report.failed_ids == ["bad"], "失败清单指得清是谁")
    check.ok(report.judge_count == 1, "裁判次数被单独数出来")

    with tempfile.TemporaryDirectory() as tmp:
        runs_dir = Path(tmp)
        path = save_run(report, runs_dir)
        check.ok(path.is_file() and path.name.endswith("-测试.json"), "运行记录已写入（文件名带 tag）")
        loaded = load_run(path)
        check.ok(loaded.passed_count == 2 and loaded.meta["prompt_sha"] == "aaaa1111",
                 "读回：成绩和提示词指纹都在")

        bad_file = runs_dir / "坏的.json"
        bad_file.write_text("{不是 json", encoding="utf-8")
        try:
            load_run(bad_file)
            check.ok(False, "坏运行记录应该报错")
        except EvalError as exc:
            check.ok("读不进来" in str(exc) and "删掉" in str(exc), "坏运行记录：报错给出去路（删或换一份）")

        wrong_file = runs_dir / "别的.json"
        wrong_file.write_text('{"a": 1}', encoding="utf-8")
        try:
            load_run(wrong_file)
            check.ok(False, "缺 results 应该报错")
        except EvalError as exc:
            check.ok("格式不对" in str(exc), "格式不对：报错点名缺 results")

        # 造两份长着不同年龄的记录，验证「找最新」和「排除自己」
        older = runs_dir / "20200101-000000-old.json"
        older.write_text("{}", encoding="utf-8")
        now = time.time()
        os.utime(bad_file, (now - 300, now - 300))
        os.utime(wrong_file, (now - 200, now - 200))
        os.utime(older, (now - 100, now - 100))
        check.ok(find_latest_run(runs_dir) == path, "find_latest_run 找到最新的一份")
        check.ok(find_latest_run(runs_dir, exclude=path) == older,
                 "find_latest_run 能排除「刚写的自己」（否则就成了和自己比）")

    # ── ⑥ 对比 ──────────────────────────────────────────────────────
    console.print("\n[bold]⑥ 对比：变好还是变坏[/bold]")
    baseline = _mk_report({"a": False, "b": True})
    current = _mk_report({"a": True, "b": True, "c": False})
    cmp = compare_runs(current, baseline, "runs/旧.json")
    check.ok(cmp.new_pass == ["a"] and cmp.same_pass == ["b"], "新通过 / 持平 分得清")
    check.ok(cmp.delta == 1, "通过率差异算对（1/2 → 2/2，+1）")
    check.ok(cmp.only_current == ["c"] and cmp.only_baseline == [], "用例集差异被标记出来")
    text = format_comparison(cmp)
    check.ok("对比基线 runs/旧.json" in text and "+1" in text and "✅ 新通过   a" in text,
             "对比面板：基线路径、差值、新通过都在")

    worse = _mk_report({"a": False, "b": True}, sha="bbbb2222")
    text = format_comparison(compare_runs(worse, current, "x"))
    check.ok("-1" in text and "❌ 新失败   a" in text, "分数掉了也说得清楚（-1，新失败列出）")

    same_sha = compare_runs(_mk_report({"a": True}, sha="zzzz"), _mk_report({"a": False}, sha="zzzz"), "x")
    text = format_comparison(same_sha)
    check.ok("没变" in text and "波动" in text and "LLM_DISABLED_TOOLS" not in text,
             "什么都没变时说清「分数变化只可能来自波动」——不给假因果")

    tools_diff = compare_runs(
        _mk_report({"a": True}, sha="zzzz"),  # 这次工具全开
        RunReport(meta={"prompt_sha": "zzzz", "model": "m1", "temperature": "0.0",
                        "disabled_tools": ["calculator"]},
                 results=[CaseResult(id="a", prompt="", passed=False, checks=[], tool_names=[],
                                     answer="", elapsed_ms=1.0)]),
        "x",
    )
    text = format_comparison(tools_diff)
    check.ok("工具开关" in text and "LLM_DISABLED_TOOLS" in text and "波动" not in text,
             "分数变化是「关了工具」造成的：面板说工具开关，不甩锅给波动")

    changed = compare_runs(_mk_report({"a": True}, sha="2222"), _mk_report({"a": True}, sha="1111"), "x")
    check.ok("变了" in format_comparison(changed), "提示词变了：并排显示指纹")

    no_common = compare_runs(_mk_report({"x": True}), _mk_report({"y": True}), "x")
    check.ok("基线不适用" in format_comparison(no_common), "没有共同用例：直接说基线不适用")

    many_ids = compare_runs(_mk_report({"x": True}),
                            _mk_report({"x": True, "y": True, "z": True, "w": True,
                                        "v": True, "u": True}), "x")
    check.ok("等 5 条" in format_comparison(many_ids),
             "用例集差异列表太长时截断显示（面板不刷屏）")

    note = format_no_baseline_note(Path("runs"))
    check.ok("还没有上一份运行记录" in note and "--baseline latest" in note,
             "没有基线：既说「为什么没有」，也说「怎么才有」")

    # ── ⑦ 日志质量（铁律 7）─────────────────────────────────────────
    console.print("\n[bold]⑦ 日志质量：每条「没做 X」都说清「因为 Y」[/bold]")
    ok_line = format_case_line(report.results[0], width=6)
    bad_line = format_case_line(report.results[1], width=6)
    check.ok(ok_line.strip().startswith("✓") and "工具: calculator" in ok_line, "实况行：过的带 ✓ 和工具清单")
    check.ok(bad_line.strip().startswith("✗") and "1/2 项检查过" in bad_line, "实况行：没过的带 ✗ 和检查计数")

    failures = format_failures(report)
    check.ok("p2" in failures and "回答里没有" in failures and "回答开头" in failures,
             "失败明细：问题原文 + 具体哪条没过 + 实际回答")

    summary = format_summary(report)
    check.ok("通过 2/3" in summary and "66.7%" in summary, "分数板：通过率和百分比")
    check.ok("分类" in summary and "模型判定 1/1" in summary, "分数板：分类统计")
    check.ok("裁判也是模型" in summary, "用了裁判：提醒它也要校准")

    no_judge = _mk_report({"a": True})
    check.ok("零额外调用" in format_summary(no_judge),
             "没用裁判：也说一句「为什么没有」（而不是沉默）")

    judge_notes = format_judge_notes(report)
    check.ok("judged" in judge_notes and "裁判 PASS" in judge_notes,
             "过了的裁判理由也会显示出来（判定理由不该只藏在 run 文件里）")

    header = format_header({"selected": 1, "cases_total": 9, "model": "m1", "temperature": 0.0,
                            "prompt_label": "prompts/system.md（sha aaaa）", "cases_file": "evals/cases.jsonl",
                            "only": "calc-big", "disabled_tools": ["calculator"]})
    check.ok("--only calc-big" in header and "别当整卷用" in header, "只跑一条：提醒这不是整卷成绩")
    check.ok("已关闭工具：calculator" in header and "这正是你要看的" in header,
             "关了工具：报出关了谁、以及这是你要求的")

    old_disabled = config.DISABLED_TOOLS
    try:
        config.DISABLED_TOOLS = {"calculator"}
        names = tools_module.active_tool_names()
        check.ok("calculator" not in names, "关掉的工具不会出现在给模型的清单里（等于不存在）")
        llm = _ScriptedLLM([_step(content="好。")])
        agent = Agent(llm, "测试提示词")
        names_from_agent = [s["function"]["name"] for s in agent.tools]
        check.ok("calculator" not in names_from_agent, "Agent 构造时也拿不到被关的工具")
        with tempfile.TemporaryDirectory() as tmp:
            one_case = [EvalCase(id="t", prompt="q", checks=[_cs("no_tools")])]
            rep = run_all(one_case, make_agent=lambda: Agent(llm, "测试提示词"),
                          llm=llm, notes_dir=Path(tmp))
        check.ok(rep.meta["disabled_tools"] == ["calculator"], "关闭名单会写进运行记录（可追溯）")
    finally:
        config.DISABLED_TOOLS = old_disabled

    return check.report()


# ══════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        return selftest()

    if "--config" in argv:
        # 用 Text 而不是字符串：配置报告里有 "[共享]" 这种方括号，
        # 直接交给 rich 会被当成样式标记解析掉（v2 踩过的坑）。
        console.print(Panel(Text(config.describe_config()), border_style="cyan", title="当前配置"))
        return 0

    if "--help" in argv or "-h" in argv:
        console.print(__doc__)
        return 0

    if "--eval" in argv:
        return run_eval(argv)

    return run_repl(argv)


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv[1:]))
