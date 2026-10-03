"""
Agent 主循环 —— v9 的循环骨架一字保留，前面加了一段「规划阶段」。

    v9 ：提问 → [执行循环] → 回答（走一步看一步）
    v10：提问 → 📋 规划（先让模型交一份步骤清单）→ [执行循环，每轮带着计划] → 回答

改动只有四处，全都在为「计划」让路：
  1. ask() 开头多了一段规划阶段（_make_plan）；规划失败**降级**为无规划执行；
  2. 执行循环每轮多带一个 update_plan 工具（只在有活跃计划时提供）；
  3. 计划消息挂在任务消息之后，进度更新时**原位替换**（v2 的老手艺：改内容、不增删消息）；
  4. trace 里照旧记录一切 —— 现在包括 update_plan 的每一次标记。

v9 的「每一步都留下记录」原封不动：
    这次提问调用了哪些工具？参数是什么？成功了吗？花了几次调用、多少 token？
—— 评估靠的就是它。规划让循环变长了，但观测口径一点没变。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import config
from config import MAX_ROUNDS
from planner import (
    SUBMIT_PLAN_SCHEMA,
    UPDATE_PLAN_SCHEMA,
    Plan,
    PlanError,
    PlanStep,
    apply_sabotage,
    build_planner_messages,
    parse_submit_plan,
    render_plan,
    tools_cheat_sheet,
)
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
    """一次提问的完整记录 —— 评估器的输入之一。

    v10 加了一组「规划账本」：计划本体 + 规划阶段的调用与 token。
    注意口径：prompt_tokens / completion_tokens 只算**执行阶段**；
    规划开销单独记（plan_* 字段）—— 这样能分清「任务本身花了多少」
    和「规划多花了多少」，回答「多花的 token 值不值」。
    """

    prompt: str
    answer: str = ""
    tools: list[ToolTrace] = field(default_factory=list)
    rounds: int = 0               # 实际调了几次模型（不含规划阶段）
    forced_finish: bool = False   # 是否被 MAX_ROUNDS 兜底「强制收尾」
    elapsed_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # ── 规划账本（v10 新增）─────────────────────────────────────────
    plan: Plan | None = None      # 本次任务实际执行的计划（没规划就是 None）
    plan_error: str = ""          # 规划失败的原因（成功时为空）
    plan_calls: int = 0           # 规划阶段调了几次模型（含重试）
    plan_prompt_tokens: int = 0
    plan_completion_tokens: int = 0

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self.tools]

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def plan_tokens(self) -> int:
        return self.plan_prompt_tokens + self.plan_completion_tokens

    @property
    def plan_progress(self) -> str:
        """计划完成度的一句话（日志用）：「4 步：3 完成 / 1 未标记」。"""
        if self.plan is None:
            return ""
        parts = [f"{self.plan.done_count} 完成"]
        if self.plan.failed_count:
            parts.append(f"{self.plan.failed_count} 失败")
        if self.plan.pending_count:
            parts.append(f"{self.plan.pending_count} 未标记")
        return f"{len(self.plan.steps)} 步：{' / '.join(parts)}"


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
        on_plan: Callable[[Plan | None, str], None] | None = None,
        planning_enabled: bool | None = None,
    ) -> None:
        self.client = client
        self.system_prompt = system_prompt
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        # 构造那一刻取工具清单 —— 被关掉的工具根本不在这里
        self.tools = openai_schemas()
        self.on_tool = on_tool
        # on_plan 在规划阶段结束时回调一次：(计划, 失败原因)。
        # REPL 用它打印 📋 计划；评估不传 —— 屏幕留给成绩单（v9 老规矩）。
        self.on_plan = on_plan
        # None = 跟随全局配置；测试里显式传 False/True 控制
        self.planning_enabled = (
            config.PLANNING_ENABLED if planning_enabled is None else planning_enabled
        )
        self._plan: Plan | None = None       # 活跃计划（跟着当前任务走）
        self._plan_index: int | None = None  # 计划消息在 messages 里的位置

    def reset(self) -> None:
        """清空对话历史，但保留 system 提示词。"""
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self._plan = None
        self._plan_index = None

    # ── 核心：主循环（v9 的骨架 + 前面的规划阶段）──────────────────
    def ask(self, user_input: str) -> AgentRun:
        started = time.perf_counter()
        run = AgentRun(prompt=user_input)

        # 多轮对话：上一个任务的计划已经完成使命，先撤下它的消息。
        # （撤下的只是「计划消息」；上一轮聊过什么、做过什么，都还在 messages 里）
        self._clear_plan_message()
        self.messages.append({"role": "user", "content": user_input})

        # ── ① 规划阶段：先想完，再做 ────────────────────────────────
        if self.planning_enabled:
            plan, error = self._make_plan(user_input, run)
            if plan is not None:
                self._plan = plan
                self._insert_plan_message()
                run.plan = plan
                if self.on_plan:
                    self.on_plan(plan, "")
            else:
                run.plan_error = error
                if self.on_plan:
                    self.on_plan(None, error)
        # 规划关闭时不走这里 —— 横幅已写明「规划 已关闭」，不再逐句唠叨

        # ── ② 执行阶段：循环骨架和 v1 一字不差，只是每轮都带着计划 ──
        for _ in range(MAX_ROUNDS):
            reply = self.client.complete(
                self.messages,
                use_tools=True,
                tools_override=self._executor_tools(),
            )
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
            if name == "update_plan":
                # v10：Agent 自己的工具（更新计划进度）—— 不进全局注册表
                result = self._handle_update_plan(args)
            else:
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

    # ── 规划阶段的新零件（v10）───────────────────────────────────────

    def _make_plan(self, task: str, run: AgentRun) -> tuple[Plan | None, str]:
        """让规划器把任务拆成步骤清单。返回 (计划, 失败原因)。

        降级路径（v3 起的老哲学：增强功能坏了，不能连累主流程）：
        · 规划成功        → 计划注入上下文，执行阶段每轮都能看到；
        · 模型不交卷/计划不合法 → 重试（LLM_PLAN_MAX_ATTEMPTS 次上限）；
        · 重试也不行/网络错 → **降级为无规划执行**，理由如实记进 run.plan_error。

        失败原因写「人话」：学习者照着日志就知道发生了什么、该拧哪个旋钮。
        """
        prompt_text, _, _ = config.load_planner_prompt()
        messages = build_planner_messages(
            task=task,
            history=self._history_for_planner(),
            tools_text=tools_cheat_sheet(),
            prompt_text=prompt_text,
            max_steps=config.PLAN_MAX_STEPS,
        )

        last_error = "（没有尝试）"
        for _ in range(max(1, config.PLAN_MAX_ATTEMPTS)):
            try:
                reply = self.client.complete(
                    messages,
                    use_tools=True,
                    tools_override=[SUBMIT_PLAN_SCHEMA],
                    model=config.PLANNER_MODEL,
                )
            except Exception as exc:
                return None, (
                    f"规划调用失败：{type(exc).__name__}: {exc} —— 已退回无规划执行"
                    f"（检查 LLM_PLANNER_MODEL / 网络？）"
                )
            run.plan_calls += 1
            run.plan_prompt_tokens += reply.prompt_tokens
            run.plan_completion_tokens += reply.completion_tokens

            call = next(
                (c for c in (reply.message.tool_calls or []) if c.function.name == "submit_plan"),
                None,
            )
            if call is None:
                spoken = (reply.message.content or "").strip()
                last_error = (
                    f"规划器没有调用 submit_plan，而是直接说了一段话"
                    f"（开头：{spoken[:60] or '（空）'}）"
                )
                continue

            try:
                steps = parse_submit_plan(call.function.arguments or "", config.PLAN_MAX_STEPS)
            except PlanError as exc:
                last_error = str(exc)
                continue

            steps, sabotage_note = apply_sabotage(steps)
            return Plan(
                steps=[PlanStep(text=s) for s in steps], sabotage_note=sabotage_note
            ), ""

        return None, (
            f"规划尝试 {max(1, config.PLAN_MAX_ATTEMPTS)} 次都没成功"
            f"（最后一次原因：{last_error}）—— 已退回无规划执行"
        )

    def _history_for_planner(self) -> list[dict[str, Any]]:
        """给规划器看的对话脉络：只要「用户说了什么、助手回答了什么」。

        不含工具细节（对「要做什么」没帮助）、不含 system（规划器有自己的），
        也会去掉刚追加的当前问题 —— 它作为 task 单独传（避免重复）。
        """
        history: list[dict[str, Any]] = []
        for message in self.messages:
            if message.get("role") not in ("user", "assistant"):
                continue
            content = (message.get("content") or "").strip()
            if not content:
                continue
            history.append({"role": message["role"], "content": content})
        if history and history[-1]["role"] == "user":
            history = history[:-1]  # 当前问题单传
        return history[-6:]         # 只留最近几轮：够理解「刚才的结果」这类指代

    def _executor_tools(self) -> list[dict[str, Any]] | None:
        """执行阶段的工具清单：常规工具 + （有活跃计划时）update_plan。

        返回 None 表示「走 client 的默认清单」——
        「update_plan 只在你真的有计划时才存在」，模型看不到用不上的工具。
        """
        if self._plan is None:
            return None
        return openai_schemas() + [UPDATE_PLAN_SCHEMA]

    def _insert_plan_message(self) -> None:
        """把计划作为一条 system 消息挂在「任务消息之后、执行之前」。

        为什么不放最上面？—— 注意力对「离得远」的内容会变淡
        （Lost in the Middle）。挂在任务旁边，它才真正「在场」。
        """
        assert self._plan is not None
        self.messages.append({"role": "system", "content": render_plan(self._plan)})
        self._plan_index = len(self.messages) - 1

    def _clear_plan_message(self) -> None:
        """撤下上一个任务的计划消息（多轮对话里它已经过时了）。

        只删「我们自己插的那条」（以【任务计划开头才算数），
        不碰其它任何消息 —— 它不在 tool_call 配对里，删除是安全的。
        """
        if self._plan_index is not None and self._plan_index < len(self.messages):
            message = self.messages[self._plan_index]
            content = str(message.get("content", ""))
            if message.get("role") == "system" and content.startswith("【任务计划"):
                del self.messages[self._plan_index]
        self._plan_index = None
        self._plan = None

    def _refresh_plan_message(self) -> None:
        """进度更新后，原位替换计划消息的内容（不新增消息）。"""
        if self._plan is None or self._plan_index is None:
            return
        if self._plan_index < len(self.messages):
            self.messages[self._plan_index]["content"] = render_plan(self._plan)

    def _handle_update_plan(self, args: dict[str, Any]) -> str:
        """执行 update_plan 调用：校验 → 标记 → 刷新上下文里的计划文本。"""
        if self._plan is None:
            raise ValueError("当前没有活跃的任务计划，update_plan 不可用")
        message = self._plan.mark(
            args.get("step"), args.get("status"), str(args.get("note") or "")
        )
        self._refresh_plan_message()
        return message
