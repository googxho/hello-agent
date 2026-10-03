"""
命令行入口 —— v12 的「带检查点的任务图」。

    # 跑一个任务（新线程）
    python main.py --run "先算 27 × 37，再查现在时间，最后写进标题 result 的笔记"

    # 指定线程名（检查点的钥匙；恢复时要用同一个名字）
    python main.py --run "..." --thread demo1

    # 跳过人工审批（自动化用）——默认会为 save_note 停下来等你批准
    python main.py --run "..." --yes

    # 从断点继续（本版的核心能力：进程死过也能接上）
    python main.py --resume demo1

    # 看一个线程的检查点历史（每一步的快照都躺在磁盘里）
    python main.py --inspect demo1

    # 演示开关
    python main.py --run "..." --pause-at 1          # 第 1 步后模拟进程猝死（os._exit）
    python main.py --run "..." --sabotage-step 2     # 第 2 步工具全部报错（触发重规划）

    # 离线自检（不需要 API Key）/ 配置
    python main.py --selftest
    python main.py --config
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from langgraph.graph.message import add_messages  # noqa: E402 —— 供 selftest 的假图注解使用（必须模块级）

import config
import graph_agent
import planner
from graph_agent import (
    build_graph,
    compose_result,
    format_plan_log,
    initial_state,
    is_step_failed,
    make_checkpointer,
    memory_checkpointer,
    route_after_advance,
    route_after_step,
    thread_config,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

console = Console()


def _get_arg(argv: list[str], flag: str) -> str | None:
    if flag not in argv:
        return None
    idx = argv.index(flag)
    if idx + 1 < len(argv) and not argv[idx + 1].startswith("--"):
        return argv[idx + 1]
    return ""


def _apply_switches(argv: list[str]) -> None:
    """把命令行开关写进 config（节点里读）。"""
    config.AUTO_APPROVE = "--yes" in argv
    pause = _get_arg(argv, "--pause-at")
    config.PAUSE_AT = int(pause) if pause else 0
    sabotage = _get_arg(argv, "--sabotage-step")
    config.SABOTAGE_STEP = int(sabotage) if sabotage else 0


def _handle_interrupts(graph, result, cfg) -> dict:
    """审批交互循环：图一停下就问你，回答完继续跑。"""
    rounds = 0
    while isinstance(result, dict) and "__interrupt__" in result and rounds < 10:
        intr = result["__interrupt__"][0]
        payload = intr.value if hasattr(intr, "value") else {}
        question = payload.get("question", payload) if isinstance(payload, dict) else payload
        console.print()
        console.print(
            Panel(
                f"⏸ [bold]需要你批准[/bold]\n\n{question}",
                border_style="yellow",
                title="人机协同（interrupt）",
            )
        )
        if config.AUTO_APPROVE:
            answer = "approve"
            console.print("  [dim]（--yes：自动批准）[/dim]")
        else:
            try:
                raw = console.input("批准吗？[y/N] > ").strip().lower()
            except EOFError:
                # 没有输入可读（管道用完 / 自动化场景）——保守处理：不批准任何写入
                console.print("  [yellow]（读不到输入——按「拒绝」处理，不写入任何文件）[/yellow]")
                raw = ""
            answer = "approve" if raw in {"y", "yes", "批准", "同意"} else "reject"
        result = graph.invoke(Command(resume=answer), cfg, durability=DURABILITY)
        rounds += 1
    return result


# ⚠️ 为什么显式用 "sync"：LangGraph 默认是 "async"（每个节点跑完后台异步落盘）。
# 实测（假图 50 行实验，见 EXPERIMENTS.md 实验 2）：async 下 os._exit 猝死时，
# 最近 1~2 张还在途的快照会丢（甚至出现只写一半的残张）——恢复点变得「看运气」。
# 教学演示需要确定的故事：sync 保证每个节点结束同步落盘后才继续，
# 「重跑哪个节点」完全可预测。生产里 async 是性能默认，代价就是这里的不确定性。
DURABILITY = "sync"


def _print_result(result: dict) -> None:
    console.print()
    console.print(Panel(str(result.get("result", "（没有产出）")), border_style="blue", title="任务报告"))


def cmd_run(argv: list[str]) -> int:
    task = _get_arg(argv, "--run")
    if not task:
        console.print('[red]✗ --run 后面要跟任务文本[/red]（例如 --run "算 27×37 并写进笔记"）')
        return 2
    thread = _get_arg(argv, "--thread") or f"task-{datetime.now():%H%M%S}"
    _apply_switches(argv)

    if config.PAUSE_AT:
        console.print(f"[red]🧪 演示模式：第 {config.PAUSE_AT} 步完成后会模拟进程猝死（os._exit）[/red]")
    if config.SABOTAGE_STEP:
        console.print(f"[red]🧪 演示模式：第 {config.SABOTAGE_STEP} 步的工具会全部报错[/red]")

    console.print(f"🧵 线程 [bold]{thread}[/bold] · 检查点 → {config.CHECKPOINT_DB.name}")
    graph = build_graph(checkpointer=make_checkpointer())
    cfg = thread_config(thread)

    result = graph.invoke(initial_state(task), cfg, durability=DURABILITY)
    result = _handle_interrupts(graph, result, cfg)
    _print_result(result)
    console.print(f'[dim]想看过程快照：python main.py --inspect {thread}[/dim]')
    return 0


def cmd_resume(argv: list[str]) -> int:
    thread = _get_arg(argv, "--resume")
    if not thread:
        console.print("[red]✗ --resume 后面要跟线程名[/red]（就是你 --run 时用的 --thread）")
        return 2
    _apply_switches(argv)

    graph = build_graph(checkpointer=make_checkpointer())
    cfg = thread_config(thread)
    snapshot = graph.get_state(cfg)
    if not snapshot.values:
        console.print(f"[red]✗ 找不到线程「{thread}」[/red]（检查拼写？跑过的线程都在 {config.CHECKPOINT_DB.name} 里）")
        return 2

    idx = snapshot.values.get("step_index", 0)
    total = len(snapshot.values.get("plan", []))
    console.print(f"🧵 线程 [bold]{thread}[/bold] · 上次停在：第 {idx + 1}/{total or '?'} 步")
    console.print(f"  下一个要跑的节点：{', '.join(snapshot.next) or '（已经结束）'}")

    result = graph.invoke(None, cfg, durability=DURABILITY)
    result = _handle_interrupts(graph, result, cfg)
    _print_result(result)
    return 0


def cmd_inspect(argv: list[str]) -> int:
    thread = _get_arg(argv, "--inspect")
    if not thread:
        console.print("[red]✗ --inspect 后面要跟线程名[/red]")
        return 2
    graph = build_graph(checkpointer=make_checkpointer())
    cfg = thread_config(thread)

    console.print(f"🧵 线程 [bold]{thread}[/bold] 的检查点历史（从最新往回看）：")
    count = 0
    for snap in graph.get_state_history(cfg):
        v = snap.values
        meta = snap.metadata or {}
        nexts = ", ".join(snap.next) or "（结束）"
        waiting = any(getattr(t, "interrupts", None) for t in (snap.tasks or []))
        mark = " ⏸等你批准" if waiting else ""
        total = len(v.get("plan", []))
        done = min(v.get("step_index", 0), total)
        console.print(
            f"  #{count:<2} step={meta.get('step', '?'):<3} 下一节点={nexts:<10}"
            f" 计划={total}步 进度={done}/{total}"
            f" 重规划={v.get('replans', 0)}{mark}"
        )
        count += 1
        if count >= 20:
            console.print("  …（更早的略）")
            break
    if count == 0:
        console.print("  （这个线程还没有任何检查点）")
    return 0


# ══════════════════════════════════════════════════════════════════════
# 离线自检
# ══════════════════════════════════════════════════════════════════════


class _Check:
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
    """离线把图整条测穿：结构 / 路由 / 状态机制 / 解析 / 日志。"""
    check = _Check()
    console.print("[bold]离线自检（不调用真实模型、不花一分钱）[/bold]\n")

    # ── ① 图结构 ────────────────────────────────────────────────────
    console.print("[bold]① 图结构（编译 + 节点 + 边）[/bold]")
    graph = build_graph(checkpointer=memory_checkpointer())
    nodes = {n for n in graph.get_graph().nodes}
    expected = {"__start__", "plan", "step", "approve", "tools", "advance", "replan", "finish", "__end__"}
    check.ok(expected <= nodes, f"七个节点齐了（{len(expected - {'__start__', '__end__'}) - 1} 个业务节点 + 起止）")

    edges = {(e.source, e.target) for e in graph.get_graph().edges}
    for src, dst in [
        ("__start__", "plan"),
        ("plan", "step"),
        ("approve", "tools"),
        ("tools", "step"),
        ("replan", "step"),
        ("finish", "__end__"),
    ]:
        check.ok((src, dst) in edges, f"边存在：{src} → {dst}")

    # ── ② 路由（纯函数）─────────────────────────────────────────────
    console.print("\n[bold]② 路由：谁来决定下一站[/bold]")

    def make_state(**kw):
        base = {
            "task": "t", "plan": ["a", "b"], "step_index": 0, "step_errors": 0,
            "step_failed": False, "step_outputs": [], "messages": [], "approved": False,
            "replans": 0, "result": "",
        }
        base.update(kw)
        return base

    old_auto = config.AUTO_APPROVE
    try:
        config.AUTO_APPROVE = False
        s_step = make_state(messages=[AIMessage("说完了")])
        check.ok(route_after_step(s_step) == "advance", "无工具调用 → advance（本步收尾）")

        s_tool = make_state(messages=[AIMessage("", tool_calls=[{"name": "calculator", "args": {}, "id": "c1"}])])
        check.ok(route_after_step(s_tool) == "tools", "普通工具调用 → tools")

        s_save = make_state(messages=[AIMessage("", tool_calls=[{"name": "save_note", "args": {}, "id": "c1"}])])
        check.ok(route_after_step(s_save) == "approve", "save_note → approve（先过审批门）")

        config.AUTO_APPROVE = True
        check.ok(route_after_step(s_save) == "tools", "--yes（自动批准）：跳过审批门")
    finally:
        config.AUTO_APPROVE = old_auto

    check.ok(route_after_advance(make_state(step_index=2)) == "finish", "步骤跑完 → finish")
    check.ok(route_after_advance(make_state(step_index=1)) == "step", "还有步骤 → step")
    check.ok(
        route_after_advance(make_state(step_failed=True, step_errors=2)) == "replan",
        "本步失败且还能改 → replan",
    )
    check.ok(
        route_after_advance(make_state(step_failed=True, step_errors=2, replans=config.MAX_REPLANS)) == "step",
        "重规划次数用尽：不再打转，继续往前走（留给收尾如实汇报）",
    )

    # ── ③ 失败判定与收尾口径 ────────────────────────────────────────
    console.print("\n[bold]③ 失败判定与收尾口径[/bold]")
    check.ok(is_step_failed(make_state(step_errors=config.STEP_ERROR_LIMIT)), "工具错误到阈值 → 本步失败")
    check.ok(
        is_step_failed(make_state(messages=[AIMessage("本步无法完成：工具一直报错")])),
        "模型自己说「本步无法完成」 → 也算失败",
    )
    check.ok(not is_step_failed(make_state(messages=[AIMessage("本步完成：结果是 42")])), "正常完成 → 不算失败")

    ok_text = compose_result(["第 1 步：OK"], failed=False)
    check.ok("✅ 全部步骤完成" in ok_text, "收尾报告：成功口径")
    bad_text = compose_result(["第 1 步：本步无法完成：…"], failed=True)
    check.ok("未能完成" in bad_text and "不装成功" in bad_text, "收尾报告：失败必须如实（v8 的老纪律）")

    log = format_plan_log(["步骤一", "步骤二"], degrade_reason="拆计划失败（坏 JSON）—— 已降级")
    check.ok("计划建成（2 步）" in "\n".join(log) and "降级" in "\n".join(log), "计划日志：步数 + 降级原因都写清楚")

    # ── ④ 计划解析与提示词 ──────────────────────────────────────────
    console.print("\n[bold]④ 计划解析 / 提示词（纯函数）[/bold]")
    check.ok(planner.parse_plan('["a", "b"]') == ["a", "b"], "数组：直接解析")
    check.ok(planner.parse_plan('```json\n["a"]\n```') == ["a"], "围栏：剥掉")
    check.ok(planner.parse_plan('{"steps": ["a", "b"]}') == ["a", "b"], "对象带 steps 键：也能吃")
    check.ok(planner.parse_plan('["一", "二", "三", "四"]', max_steps=2) == ["一", "二"], "超过上限：裁掉多余的")
    try:
        planner.parse_plan("完全没有 JSON")
        check.ok(False, "坏输出应报错")
    except planner.PlanError as exc:
        check.ok("抠不出 JSON" in str(exc), "坏输出：报错带原文开头")

    steps, reason = planner.safe_parse_plan("乱说")
    check.ok(steps == ["完成整个任务"] and "降级" in reason, "safe 版：失败降级为单步，不连累任务")

    text = planner.build_step_text("总任务例句", ["甲", "乙"], 1)
    check.ok("总任务例句" in text and "▶ 2. 乙" in text and "第 2 步" in text, "步骤指令：带任务/计划（▶ 标当前）/步号")
    check.ok("以「本步无法完成」开头" in planner.STEP_ROLE, "角色设定里教了「做不成怎么说」（失败可判定）")

    # ── ⑤ 状态机制：reducer / 检查点 / interrupt（假图，不联网）────
    console.print("\n[bold]⑤ 状态机制：reducer / 检查点 / interrupt[/bold]")
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import interrupt as lg_interrupt

    class MechState(TypedDict, total=False):
        n: Annotated[int, lambda a, b: (a or 0) + b]  # 自定义 reducer：累加
        log: Annotated[list, add_messages]
        answer: str

    def bump(s):
        return {"n": 1, "log": [AIMessage(f"第 {s.get('n', 0) + 1} 次")]}

    def bump2(s):
        return {"n": 1}

    def gate(s):
        a = lg_interrupt("继续吗？")
        return {"answer": str(a)}

    mb = StateGraph(MechState)
    mb.add_node("bump", bump)
    mb.add_node("bump2", bump2)
    mb.add_node("gate", gate)
    mb.add_edge(START, "bump")
    mb.add_edge("bump", "bump2")
    mb.add_edge("bump2", "gate")
    mb.add_edge("gate", END)
    mech = mb.compile(checkpointer=memory_checkpointer())
    mcfg = {"configurable": {"thread_id": "mech-1"}}

    r = mech.invoke({}, mcfg)
    check.ok("__interrupt__" in r, "interrupt：图在 gate 节点停下（返回里带 __interrupt__）")
    r2 = mech.invoke(Command(resume="继续！"), mcfg)
    check.ok(r2.get("answer") == "继续！", "Command(resume)：暂停点拿到外面的回答")
    check.ok(r2.get("n") == 2, "自定义 reducer：两次 +1 累加为 2（不是覆盖）")
    check.ok(len(r2.get("log", [])) == 1, "add_messages：消息列表自动合并")

    history = list(mech.get_state_history(mcfg))
    check.ok(len(history) >= 3, f"检查点历史：每步一张快照（拿到 {len(history)} 张）")
    check.ok(any(s.next and "gate" in s.next for s in history), "历史里能找到「停在 gate 等审批」的那个瞬间")

    # ── ⑥ 工具（沿用三件套）─────────────────────────────────────────
    console.print("\n[bold]⑥ 工具三件套[/bold]")
    from tools import TOOL_MAP, calculator, get_current_time, save_note

    check.ok(set(TOOL_MAP) == {"calculator", "get_current_time", "save_note"}, "三个工具都在")
    check.ok(calculator.invoke({"expression": "27 * 37"}) == "999", "calculator：27×37=999")
    check.ok("星期" in str(get_current_time.invoke({})), "get_current_time：带星期")
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        old_notes = config.NOTES_DIR
        config.NOTES_DIR = Path(tmp)
        try:
            out = save_note.invoke({"title": "结果 草稿", "content": "27×37=999"})
            check.ok("已保存" in out and (Path(tmp) / "结果_草稿.md").is_file(), "save_note：写文件 + slug")
        finally:
            config.NOTES_DIR = old_notes

    return check.report()


# ══════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        return selftest()
    if "--config" in argv:
        console.print(Panel(Text(config.describe_config()), border_style="cyan", title="当前配置"))
        return 0
    if "--help" in argv or "-h" in argv:
        console.print(__doc__)
        return 0
    if "--run" in argv:
        return cmd_run(argv)
    if "--resume" in argv:
        return cmd_resume(argv)
    if "--inspect" in argv:
        return cmd_inspect(argv)

    console.print(__doc__)
    console.print("[dim]（先试试：python main.py --selftest —— 离线不花钱）[/dim]")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
