"""
规划器 —— v10 的新东西：把「先做什么、后做什么」从模型的脑子里请到纸面上。

═══════════════════════════════════════════════════════════════════════
为什么要有这个文件？
═══════════════════════════════════════════════════════════════════════

v9 的尺子量出了第一个系统性弱点：任务一长就丢步骤（`multi-step` 最先趴下）。
根源在于 —— **模型每一步只看眼前**，任务没有全局视野。

规划的做法：动手之前，先让模型交一份「步骤清单」。这份清单会被写进
上下文，模型执行到每一步时都能看到「我在哪、还有几步」。

    没有计划：任务 → [黑盒里走一步看一步] → 结果（中途丢了什么不知道）
    有计划　：任务 → 📋 计划（可见！）→ 每一步都带着 [任务 + 计划 + 进度] → 结果

**关键**：规划不改变模型的能力，它改变的是「每一刻模型能看到什么」。
本质是上下文工程，和 v2~v8 的上下文管理知识块是同一族技术。

═══════════════════════════════════════════════════════════════════════
计划是怎么「落地」的？（两个工具协议）
═══════════════════════════════════════════════════════════════════════

  submit_plan（规划阶段）   规划器调用它「交计划」——
                            把步骤列表作为参数交回来。工具即 schema（v8 的老思想）。
                            它只在规划那一次调用里可用，不注册进全局工具表。

  update_plan（执行阶段）   执行 Agent 调用它「报进度」——
                            每完成一步就标记一次。它是个**实例级工具**：
                            不属于「世界」（不写文件、不算数），只更新 Agent
                            自己脑子里的计划状态，所以不进 TOOL_DEFS 注册表。

═══════════════════════════════════════════════════════════════════════
这个文件里的东西全部是纯函数 / 纯数据（除了两个 schema 常量）
═══════════════════════════════════════════════════════════════════════

解析（parse）、校验（validate）、标记（mark）、渲染（render）、
实验开关（apply_sabotage）都是「数据进、数据出」——
不需要 API Key、不花钱，selftest 里可以逐条喂假数据测穿。
（v3 假摘要器 / v6 假向量器 / v8 剧本客户端 / v9 判定器：同一个套路。）

⚠️ 一个容易踩的坑：**注入给模型看的计划文本，和给日志看的真相要分开**。
`sabotage_note`（实验模式说明）只进日志，绝不进模型上下文 ——
否则模型知道「计划被砍过」就会警惕，实验就测不准了。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import config
from tools import TOOL_DEFS, ToolDefinition, ToolParameter


class PlanError(Exception):
    """计划相关的坏数据 —— 消息要能直接指导修复（谁看到这句话，谁就知道怎么改）。"""


# ══════════════════════════════════════════════════════════════════════
# 工具协议之一：submit_plan —— 规划器用它「交计划」
# ══════════════════════════════════════════════════════════════════════
#
# 为什么它不注册进 TOOL_DEFS？
#   它不是给「执行阶段的 Agent」用的，只在规划那一次调用里临时提供给规划器。
#   注册进全局，执行循环里就会多出一个不该出现的工具（模型会乱调）。

SUBMIT_PLAN_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_plan",
        "description": (
            "提交任务计划：把任务拆成按顺序执行的步骤清单。"
            "开工之前必须先交计划 —— 这是你的唯一职责，不要执行任务本身。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "按执行顺序排列的步骤清单；每一步一句话，"
                        "尽量对应一个工具调用；要覆盖任务的全部要求"
                    ),
                },
            },
            "required": ["steps"],
        },
    },
}


# ══════════════════════════════════════════════════════════════════════
# 工具协议之二：update_plan —— 执行 Agent 用它「报进度」
# ══════════════════════════════════════════════════════════════════════

UPDATE_PLAN_TOOL = ToolDefinition(
    name="update_plan",
    description=(
        "更新任务计划的进度。每完成计划中的一步，立刻调用本工具标记；"
        "某一步反复尝试仍做不成时，也调用并标记为 failed（写明原因）。"
        "不要用它做别的事。"
    ),
    parameters=[
        ToolParameter(
            name="step",
            type="integer",
            description="第几步（从 1 开始，对应计划里的编号）",
        ),
        ToolParameter(
            name="status",
            type="string",
            description="这一步的结果",
            choices=["done", "failed"],
        ),
        ToolParameter(
            name="note",
            type="string",
            description="（可选）一句话说明；failed 时要写清楚是什么原因",
            required=False,
        ),
    ],
)

UPDATE_PLAN_SCHEMA: dict[str, Any] = UPDATE_PLAN_TOOL.to_openai_schema()


# ══════════════════════════════════════════════════════════════════════
# 数据模型：Plan / PlanStep
# ══════════════════════════════════════════════════════════════════════

_STATUS_CN = {"done": "完成", "failed": "失败", "pending": "未开始"}
_ICONS = {"done": "✅", "failed": "❌", "pending": "☐"}


@dataclass
class PlanStep:
    """计划里的一步。"""

    text: str
    status: str = "pending"  # pending / done / failed
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "status": self.status, "note": self.note}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PlanStep":
        return cls(
            text=str(d.get("text", "")),
            status=str(d.get("status", "pending")),
            note=str(d.get("note", "")),
        )


@dataclass
class Plan:
    """一份任务计划 = 有序的步骤清单 + 每步的进度。"""

    steps: list[PlanStep] = field(default_factory=list)
    # ⚠️ 实验模式的说明 —— 只进日志，永远不要渲染进给模型看的文本
    sabotage_note: str = ""

    # ── 统计 ────────────────────────────────────────────────────────
    def count(self, status: str) -> int:
        return sum(1 for s in self.steps if s.status == status)

    @property
    def done_count(self) -> int:
        return self.count("done")

    @property
    def failed_count(self) -> int:
        return self.count("failed")

    @property
    def pending_count(self) -> int:
        return self.count("pending")

    def first_pending(self) -> int | None:
        """第一个还没开始的步骤编号（给「下一步：第 N 步」用）。"""
        for i, s in enumerate(self.steps, start=1):
            if s.status == "pending":
                return i
        return None

    def progress_line(self) -> str:
        base = f"进度：✅{self.done_count} ❌{self.failed_count} ☐{self.pending_count}"
        nxt = self.first_pending()
        if self.pending_count == 0 and self.failed_count == 0:
            return base + "｜全部完成"
        if nxt is not None:
            return base + f"｜下一步：第 {nxt} 步"
        return base + f"｜没有未开始的步骤（{self.failed_count} 步失败）"

    # ── 标记进度（update_plan 的落点）──────────────────────────────
    def mark(self, step_no: Any, status: Any, note: str = "") -> str:
        """标记某一步的结果，返回给模型的【人话】回执。

        校验不过一律 raise PlanError —— agent 会把它包装成工具错误回灌，
        模型能看懂并自己纠正（v1 的老机制，没有新循环）。
        """
        try:
            index = int(step_no)
        except (TypeError, ValueError):
            raise PlanError(f"step 必须是整数（从 1 开始），收到了 {step_no!r}") from None
        if index < 1 or index > len(self.steps):
            raise PlanError(
                f"计划的步骤编号是 1~{len(self.steps)}，没有第 {index} 步 —— 编号以计划为准"
            )
        status = str(status or "").strip().lower()
        if status not in ("done", "failed"):
            raise PlanError(f"status 只能是 done 或 failed，收到了 {status!r}")

        step = self.steps[index - 1]
        note = (note or "").strip()
        old = step.status
        suffix = f"（{note}）" if note else ""

        if old == status:
            return (
                f"第 {index} 步之前已经是「{_STATUS_CN[status]}」（本次重复标记，已忽略）"
                f"｜{self.progress_line()}"
            )
        step.status = status
        step.note = note
        # ⚠️ 进度必须在更新【之后】算 —— 先算会显示旧数字（selftest 抓过一次）
        progress = f"｜{self.progress_line()}"
        if old == "pending":
            return f"第 {index} 步已标记「{_STATUS_CN[status]}」{suffix}{progress}"
        return (
            f"第 {index} 步从「{_STATUS_CN[old]}」改为「{_STATUS_CN[status]}」{suffix}{progress}"
        )

    # ── 序列化（进 runs/*.json，对比时能看）────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "sabotage_note": self.sabotage_note,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Plan":
        return cls(
            steps=[PlanStep.from_dict(s) for s in d.get("steps", [])],
            sabotage_note=str(d.get("sabotage_note", "")),
        )


# ══════════════════════════════════════════════════════════════════════
# 解析与校验（纯函数 —— selftest 的主战场）
# ══════════════════════════════════════════════════════════════════════


def parse_submit_plan(arguments_text: str, max_steps: int) -> list[str]:
    """从 submit_plan 调用的原始参数里解析出步骤列表。

    任何坏数据都抛 PlanError（带原文开头）——
    调用方（agent）会把失败原因记下来并决定重试/降级，绝不猜。
    """
    raw = arguments_text or ""
    try:
        args = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PlanError(
            f"submit_plan 的参数不是合法 JSON（{exc.msg}）｜原文开头：{raw[:80]!r}"
        ) from exc
    if not isinstance(args, dict):
        raise PlanError(f"submit_plan 的参数应该是一个 JSON 对象，收到了 {type(args).__name__}")
    return validate_steps(args.get("steps"), max_steps)


def validate_steps(raw_steps: Any, max_steps: int) -> list[str]:
    """校验步骤列表。规则都是照着「这张计划能不能真正指导执行」定的。"""
    if not isinstance(raw_steps, list) or not raw_steps:
        raise PlanError("计划里没有任何步骤（steps 必须是一个非空数组）")

    steps: list[str] = []
    for i, item in enumerate(raw_steps, start=1):
        if not isinstance(item, str) or not item.strip():
            raise PlanError(f"第 {i} 个步骤不是一段文字（每个步骤必须是字符串）")
        steps.append(item.strip())

    if len(steps) > max_steps:
        raise PlanError(
            f"计划有 {len(steps)} 步，超过上限 {max_steps}（LLM_PLAN_MAX_STEPS）——"
            f"请把细碎的动作合并成更粗的步骤"
        )

    seen: dict[str, int] = {}
    for i, step in enumerate(steps, start=1):
        if step in seen:
            raise PlanError(
                f"第 {i} 步和第 {seen[step]} 步完全重复（「{step[:30]}…」）——"
                f"重复的步骤对执行没有指导意义，请合并"
            )
        seen[step] = i

    return steps


# ══════════════════════════════════════════════════════════════════════
# 渲染：计划 → 注入上下文的文本
# ══════════════════════════════════════════════════════════════════════


def render_plan(plan: Plan) -> str:
    """把计划渲染成一条消息的正文（每次进度更新后重新渲染，原位替换）。

    两条设计：
    · 渲染文本里带着「怎么配合」的规则 —— 模型每轮都会读到；
    · 状态图标一眼可辨（✅❌☐），日志和上下文同款 —— 你看到的 = 模型看到的。
    """
    lines = [
        "【任务计划 · 执行时按顺序推进】",
        "每完成一步，立刻调用 update_plan 标记"
        '（step=编号，status="done"）；某步做不了就标记 status="failed" 并写明原因。',
        "",
    ]
    for i, step in enumerate(plan.steps, start=1):
        icon = _ICONS.get(step.status, "?")
        note = f"（{step.note}）" if step.note else ""
        lines.append(f"  {i}. {icon} {step.text}{note}")
    lines.append("")
    lines.append(plan.progress_line())
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# 实验开关：亲手把计划写坏（ROADMAP 要求的「反例验证」）
# ══════════════════════════════════════════════════════════════════════


def apply_sabotage(steps: list[str]) -> tuple[list[str], str]:
    """按 LLM_PLAN_SABOTAGE 人为破坏计划。返回 (处理后的步骤, 日志说明)。

    两种破坏方式：
        drop_last —— 砍掉最后一步（测「缺一步」的后果）
        reverse   —— 颠倒顺序（测「计划和任务原文冲突时，谁说了算」）

    ⚠️ 返回的说明**只能进日志**：模型看到「计划被砍过」就会警惕，实验就废了。
    """
    mode = config.PLAN_SABOTAGE
    if not mode:
        return steps, ""
    if mode == "drop_last":
        if len(steps) < 2:
            return steps, f"⚠ 实验模式（drop_last）：计划只有 {len(steps)} 步，没有可砍的 —— 保持原样"
        dropped = steps[-1]
        return (
            steps[:-1],
            f"⚠ 实验模式（drop_last）：计划被人为砍掉最后一步「{dropped}」——模型不会知道少了一步",
        )
    if mode == "reverse":
        if len(steps) < 2:
            return steps, f"⚠ 实验模式（reverse）：计划只有 {len(steps)} 步，颠倒没有意义 —— 保持原样"
        return (
            list(reversed(steps)),
            f"⚠ 实验模式（reverse）：计划顺序被人为颠倒（首尾互换）——模型不会知道顺序错了",
        )
    return steps, f"⚠ 不认识的 LLM_PLAN_SABOTAGE 值「{mode}」（可用：drop_last / reverse）——按不处理"


# ══════════════════════════════════════════════════════════════════════
# 规划提示词的组装
# ══════════════════════════════════════════════════════════════════════


def tools_cheat_sheet() -> str:
    """给规划器看的「可用工具清单」（名字 + 第一句描述）。

    为什么不直接把工具 schema 给规划器？
    因为规划器**不该执行任务** —— 给了完整 schema，它可能顺手调用工具
    （规划阶段产生副作用就乱了）。只让它「知道有什么手可用」，
    拆出来的步骤才会对得上执行时会用的工具。
    """
    rows = []
    for name, definition in TOOL_DEFS.items():
        if name in config.DISABLED_TOOLS:
            continue
        first_sentence = definition.description.split("。")[0].strip()
        rows.append(f"- {name}：{first_sentence}")
    return "\n".join(rows) if rows else "（当前没有可用工具）"


def build_planner_messages(
    task: str,
    history: list[dict[str, Any]],
    tools_text: str,
    prompt_text: str,
    max_steps: int,
) -> list[dict[str, str]]:
    """拼规划调用的一次性对话（规划器有自己的 system、自己的视野）。

    history 是「用户和助手的对话脉络」（不含 system 和工具细节）——
    多轮对话里用户说「把刚才的结果存起来」，规划器得先知道「刚才的结果」是什么。
    """
    system = prompt_text.replace("{tools}", tools_text).replace("{max_steps}", str(max_steps))

    parts: list[str] = []
    if history:
        lines = []
        for message in history:
            who = "用户" if message["role"] == "user" else "助手"
            lines.append(f"{who}：{message['content']}")
        parts.append("【之前的对话（供参考）】\n" + "\n".join(lines))
    parts.append(f"【本次任务】\n{task}")

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
