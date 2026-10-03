"""
图版 Agent —— v12 的核心：把 Agent 的流程画成**显式的状态图**。

═══════════════════════════════════════════════════════════════════════
图的形状（先看全貌，再看代码）
═══════════════════════════════════════════════════════════════════════

                   ┌──────────────────────────────────────────┐
                   ▼                                          │
  START ──► [plan] ──► [step] ──┬─（要用 save_note）─► [approve] ──┐
                            ▲   ├─（普通工具调用）───────────────► [tools]
                            │   └─（本步说完了）───────┐          │
                            │                          ▼          │
                            │                     [advance] ◄──────┘（工具结果回灌 step）
                            │                          │
                            │        ┌─────────────────┼──────────────┐
                            │        │ 还有步骤         │ 失败且还能改   │ 全完成
                            │        ▼                 ▼              ▼
                            └───── [step]          [replan] ──►     [finish] ──► END

v10 手写版对比：那是一个 40 行的大循环，模型自由决定「调工具还是收工」。
v12 把控制权从模型手里拿回来一部分：
  · 「第几步、什么时候算完成」由图决定（step / advance 节点）；
  · 「危险操作要不要放行」由图插一道门（approve 节点）；
  · 「这步做不成怎么办」由图给条出路（replan 节点）。

═══════════════════════════════════════════════════════════════════════
三个关键机制（也都是面试考点）
═══════════════════════════════════════════════════════════════════════

1. **状态 + reducer**：节点返回「增量」，框架按 reducer 合并——
   默认是覆盖（plan / step_index），想「追加」就得显式声明
   （step_outputs 用 operator.add，messages 用内置的 add_messages）。

2. **检查点（checkpointer）**：每个节点执行后，状态快照落盘。
   所以进程猝死（kill -9 / os._exit）后，resume 能从上一次的
   完整状态继续 —— v10「断在第 5 步就永远断了」的答案。

3. **interrupt（人机协同）**：节点里调用 interrupt() 会让图暂停，
   把「问题」交给外面；外面用 Command(resume=答案) 恢复。
   ⚠️ 恢复时**整个节点会从头重跑**（interrupt 那行返回答案）——
   所以节点里 interrupt 之前的代码必须幂等/无副作用。
   （我们把审批做成了独立的 approve 节点，就是为这个原因：
    approve 节点里 interrupt 之前只是读数据，重跑无害。）
"""

from __future__ import annotations

import operator
import os
import sys
from typing import Annotated, Any, TypedDict

import config
import planner
from tools import TOOL_MAP, TOOLS

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from rich.console import Console

console = Console()


# ══════════════════════════════════════════════════════════════════════
# 状态定义 —— 图里跑的不是 messages 一个袋子，而是显式的任务状态
# ══════════════════════════════════════════════════════════════════════


class TaskState(TypedDict):
    task: str                                          # 原始任务
    plan: list[str]                                    # 计划步骤（重规划会改它）
    step_index: int                                    # 当前执行到第几步（0-based）
    step_errors: int                                   # 本步累计工具错误数
    step_failed: bool                                  # 本步是否判定失败（advance 写入）
    step_outputs: Annotated[list[str], operator.add]   # 每步产出 ★ 自定义 reducer：追加
    messages: Annotated[list, add_messages]            # 工具回环消息 ★ 内置 reducer
    approved: bool                                     # 最近一次人工审批的结果
    replans: int                                       # 重规划次数（防震荡）
    result: str                                        # 最终报告


def initial_state(task: str) -> dict[str, Any]:
    """新任务的初始状态（所有字段都给全，避免 reducer 拿到 undefined）。"""
    return {
        "task": task,
        "plan": [],
        "step_index": 0,
        "step_errors": 0,
        "step_failed": False,
        "step_outputs": [],
        "messages": [],
        "approved": False,
        "replans": 0,
        "result": "",
    }


# ══════════════════════════════════════════════════════════════════════
# 纯函数：日志格式 + 失败判定（离线可测 —— 铁律 7 的延续）
# ══════════════════════════════════════════════════════════════════════


def format_plan_log(steps: list[str], degrade_reason: str = "") -> list[str]:
    """计划建成时的日志行（降级原因必须说清楚，不许沉默）。"""
    lines = [f"📋 计划建成（{len(steps)} 步）："]
    lines += [f"   {i}. {s}" for i, s in enumerate(steps, start=1)]
    if degrade_reason:
        lines.append(f"   ⚠ {degrade_reason}")
    return lines


def compose_result(step_outputs: list[str], failed: bool) -> str:
    """收尾报告（纯函数）：成功 / 有失败残留，两种口径说得明明白白。"""
    lines = ["任务执行报告："]
    lines += [f"- {out}" for out in step_outputs]
    if failed or any("本步无法完成" in out for out in step_outputs):
        lines.append("⚠ 有步骤未能完成（重规划次数已用尽或没有更好的拆法）——不装成功。")
    else:
        lines.append("✅ 全部步骤完成。")
    return "\n".join(lines)


def _latest_ai_text(messages: list) -> str:
    """本步的产出 = 最近一条「没有工具调用的 AI 文本」。"""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
            text = str(m.content or "").strip()
            if text:
                return text
    return "（本步没有留下文字产出）"


def is_step_failed(state: TaskState) -> bool:
    """本步是否失败：工具连续报错到阈值，或模型自己说「本步无法完成」。"""
    if state.get("step_errors", 0) >= config.STEP_ERROR_LIMIT:
        return True
    return _latest_ai_text(state["messages"]).startswith("本步无法完成")


# ══════════════════════════════════════════════════════════════════════
# 路由（纯函数 —— selftest 的主战场）
# ══════════════════════════════════════════════════════════════════════


def route_after_step(state: TaskState) -> str:
    """step 之后去哪：approve（要审批）/ tools（直接跑）/ advance（本步收尾）。"""
    last = state["messages"][-1]
    calls = getattr(last, "tool_calls", None) or []
    if not calls:
        return "advance"
    if not config.AUTO_APPROVE and any(c["name"] == "save_note" for c in calls):
        return "approve"
    return "tools"


def route_after_advance(state: TaskState) -> str:
    """advance 之后去哪：replan（失败且还能改）/ step（继续）/ finish（收尾）。

    ⚠️ 注意数据流：advance 已经把 step_index 推进过了，
    所以「失败的步」= step_index - 1（replan 节点会回退指针重做它）。
    """
    if state.get("step_failed") and state.get("replans", 0) < config.MAX_REPLANS:
        return "replan"
    if state["step_index"] >= len(state["plan"]):
        return "finish"
    return "step"


# ══════════════════════════════════════════════════════════════════════
# 七个节点 —— 每个节点就是个「状态进、增量出」的函数
# ══════════════════════════════════════════════════════════════════════


def plan_node(state: TaskState) -> dict[str, Any]:
    """调一次模型，把任务拆成步骤清单（拆不出 → 降级为单步）。"""
    llm = config.get_chat_model(temperature=0)
    reply = llm.invoke(planner.build_plan_prompt(state["task"]))
    steps, degrade = planner.safe_parse_plan(str(reply.content))
    for line in format_plan_log(steps, degrade):
        console.print(line)
    return {"plan": steps, "step_index": 0}


def step_node(state: TaskState) -> dict[str, Any]:
    """执行当前步：一次带工具的模型调用（要么调工具，要么汇报本步结果）。

    ⚠️ 消息序列的组装纪律（踩过的坑，详见 planner.build_step_text 的注释）：
    新的一步以 **HumanMessage 指令**开场/推进；工具回环中（尾部是 ToolMessage）
    则不再插指令 —— 保证请求永远是标准的「人机对话」形态。
    （之前把指令做成第二条 system、让请求以 [system, assistant] 收尾，
     在带 tools 的 thinking 服务端上触发了一个误导性的 400——查了很久。）
    """
    current = state["step_index"]
    plan = state["plan"]
    step_text = plan[current] if current < len(plan) else "（计划已到末尾）"

    llm = config.get_chat_model(temperature=0)
    llm_tools = llm.bind_tools(TOOLS)
    history = list(state["messages"])
    fresh: list = []
    if not history or not isinstance(history[-1], ToolMessage):
        # 只在本步的「第一轮」打印（工具回环里不重复打）
        console.print(f"▶ 第 {current + 1}/{len(plan)} 步：{step_text}")
        fresh.append(HumanMessage(planner.build_step_text(state["task"], plan, current)))
    ai = llm_tools.invoke([SystemMessage(planner.STEP_ROLE), *history, *fresh])
    return {"messages": [*fresh, ai]}


def approve_node(state: TaskState) -> dict[str, Any]:
    """人工审批门（interrupt）。

    ⚠️ 恢复时本节点从头重跑 —— 所以这里 interrupt 之前只做「读」，
    没有任何副作用。「副作用放到审批之后」是 HITL 的第一纪律。
    """
    last = state["messages"][-1]
    save_call = next(c for c in last.tool_calls if c["name"] == "save_note")
    decision = interrupt(
        {
            "question": f"模型要写文件（save_note：{save_call['args'].get('title', '?')}）——批准这次写入吗？",
            "tool": "save_note",
            "args": save_call["args"],
        }
    )
    approved = str(decision).strip().lower() in {"approve", "y", "yes", "批准", "同意"}
    console.print(f"  🖐 人工审批：{'✅ 批准' if approved else '⛔ 拒绝'}")
    return {"approved": approved}


def tools_node(state: TaskState) -> dict[str, Any]:
    """执行本轮工具调用（含两个演示开关：审批结果、sabotage）。"""
    last = state["messages"][-1]
    sabotaged = config.SABOTAGE_STEP == state["step_index"] + 1
    results: list[ToolMessage] = []
    errors = 0

    for call in last.tool_calls:
        name = call["name"]
        refused = False
        if sabotaged:
            out = f"错误：演示用故障（--sabotage-step {config.SABOTAGE_STEP}）"
            ok = False
        elif name == "save_note" and not config.AUTO_APPROVE and not state.get("approved"):
            out = (
                "用户拒绝了这次写入（未批准）。这是最终决定：不要重试、不要换其它方式写入，"
                "直接汇报本步现状（说明写入被拒绝）即可。"
            )
            ok = False
            refused = True
        else:
            try:
                out = str(TOOL_MAP[name].invoke(call["args"]))
                ok = True
            except Exception as exc:  # noqa: BLE001
                out = f"错误：{type(exc).__name__}: {exc}"
                ok = False
        # ⚠️ 「用户拒绝」是决策、不是故障 —— 不计入失败计数（否则会误触发重规划）
        if not ok and not refused:
            errors += 1
        style = "green" if ok else ("yellow" if refused else "red")
        console.print(f"  🔧 {name}({call['args']}) → [{style}]{out[:80]}[/{style}]")
        results.append(ToolMessage(content=out, tool_call_id=call["id"]))

    return {"messages": results, "step_errors": state.get("step_errors", 0) + errors}


def advance_node(state: TaskState) -> dict[str, Any]:
    """一步的收尾：判定成败、记录产出、推进指针。

    ⚠️ 演示开关 --pause-at N：在第 N 步收尾时模拟进程猝死（os._exit，等价 kill -9）。
       猝死发生在「节点返回之前」——这一步的状态不会落盘；
       resume 时本节点重跑（幂等，无副作用），所以演示是安全的。
    """
    current = state["step_index"]
    failed = is_step_failed(state)
    output = _latest_ai_text(state["messages"])
    mark = "✗" if failed else "✅"
    console.print(f"  {mark} 第 {current + 1} 步{'失败' if failed else '完成'}：{output[:60]}")

    if config.PAUSE_AT and current + 1 == config.PAUSE_AT:
        console.print(
            f"  [bold red]🧪 演示模式：第 {current + 1} 步完成后模拟进程猝死（os._exit）"
            f"—— 刚才的状态已落盘，试试用 --resume 找回[/bold red]"
        )
        sys.stdout.flush()
        os._exit(1)

    return {
        "step_index": current + 1,
        "step_failed": failed,
        "step_errors": 0,
        "step_outputs": [f"第 {current + 1} 步（{state['plan'][current][:24]}…）：{output}"],
    }


def replan_node(state: TaskState) -> dict[str, Any]:
    """重规划：回到失败的那一步，只重拆「它和之后的部分」（最小修订，防震荡）。

    数据流提示：进入本节点时 advance 已把指针推进过，
    所以失败步 = step_index - 1；重拆后把指针**回退**到那里重做。
    """
    failed_idx = max(0, min(state["step_index"] - 1, len(state["plan"]) - 1))
    reason = f"第 {failed_idx + 1} 步「{state['plan'][failed_idx]}」反复失败"
    console.print(f"  🔁 重规划（第 {state['replans'] + 1}/{config.MAX_REPLANS} 次）：{reason}")

    llm = config.get_chat_model(temperature=0)
    reply = llm.invoke(
        planner.build_replan_prompt(state["task"], state["plan"], failed_idx, reason)
    )
    remaining, degrade = planner.safe_parse_plan(str(reply.content))
    if degrade:
        console.print(f"   [yellow]⚠ {degrade}[/yellow]")
    new_plan = state["plan"][:failed_idx] + remaining
    console.print("   新计划（剩余部分）：")
    for i, s in enumerate(remaining, start=failed_idx + 1):
        console.print(f"     {i}. {s}")
    return {
        "plan": new_plan,
        "step_index": failed_idx,  # ← 回退指针：重做失败的那一步
        "replans": state["replans"] + 1,
        "step_failed": False,
        "step_errors": 0,
    }


def finish_node(state: TaskState) -> dict[str, Any]:
    """收尾：汇总每步产出，如实汇报（失败不装成功 —— v8 的老纪律）。"""
    result = compose_result(state["step_outputs"], state.get("step_failed", False))
    console.print(f"🏁 {result}")
    return {"result": result}


# ══════════════════════════════════════════════════════════════════════
# 组装图
# ══════════════════════════════════════════════════════════════════════


def build_graph(checkpointer: Any | None = None):
    """把七个节点和边接成图，编译成可执行对象。"""
    builder = StateGraph(TaskState)
    builder.add_node("plan", plan_node)
    builder.add_node("step", step_node)
    builder.add_node("approve", approve_node)
    builder.add_node("tools", tools_node)
    builder.add_node("advance", advance_node)
    builder.add_node("replan", replan_node)
    builder.add_node("finish", finish_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "step")
    builder.add_conditional_edges(
        "step",
        route_after_step,
        {"approve": "approve", "tools": "tools", "advance": "advance"},
    )
    builder.add_edge("approve", "tools")
    builder.add_edge("tools", "step")
    builder.add_conditional_edges(
        "advance",
        route_after_advance,
        {"step": "step", "replan": "replan", "finish": "finish"},
    )
    builder.add_edge("replan", "step")
    builder.add_edge("finish", END)
    return builder.compile(checkpointer=checkpointer)


def make_checkpointer(db_path: Any | None = None):
    """SqliteSaver：检查点落盘（跨进程恢复的关键）。

    ⚠️ check_same_thread=False 是必须的 —— LangGraph 会在别的线程里
    执行节点并写检查点，SQLite 默认拒绝跨线程使用连接（我们踩过）。
    """
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    path = db_path or config.CHECKPOINT_DB
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn)


def memory_checkpointer():
    """内存版检查点（selftest / 演示用，退出即消失）。"""
    return MemorySaver()


def thread_config(thread_id: str) -> dict[str, Any]:
    """一个线程（会话）的配置：thread_id 是「这是哪个任务的检查点序列」的钥匙。"""
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": config.RECURSION_LIMIT,
    }
