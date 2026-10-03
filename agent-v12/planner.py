"""
规划器（精简版）—— 图的第一步：任务 → 步骤清单。

v10 的规划是「全套机制」（submit_plan 工具交卷 + 校验 + 重试 + 降级）。
v12 只保留最核心的一件事：**拆出步骤**。路线选了最稳的那条 ——
「要求 JSON 数组 + 兜底解析」（v11 实验 5 的教训：在这套服务端上，
 花式结构化输出姿势全灭，土办法最稳）。

降级哲学照旧（v3 起的老规矩）：拆不出计划？**不要连累任务** ——
退化成「单步计划：完成整个任务」，图照样跑。
"""

from __future__ import annotations

import json
import re
from typing import Any

import config


class PlanError(Exception):
    """拆计划失败（消息是人话，用来解释为什么降级）。"""


# ══════════════════════════════════════════════════════════════════════
# 解析：从模型输出里抠出步骤数组（纯函数，selftest 主战场）
# ══════════════════════════════════════════════════════════════════════


def _extract_json_value(text: str) -> Any:
    """宽容地从文本里抠出第一个 JSON 值（数组或对象）。

    三种尝试：原文直接解析 → 剥 ``` 围栏 → 抠第一个 [ / { 到最后一个 ] / }。
    """
    cleaned = (text or "").strip()
    for candidate in (cleaned,):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass

    starts = [i for i in (cleaned.find("["), cleaned.find("{")) if i != -1]
    if starts:
        start = min(starts)
        closer = "]" if cleaned[start] == "[" else "}"
        end = cleaned.rfind(closer)
        if end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                pass
    raise PlanError(f"抠不出 JSON ｜原文开头：{cleaned[:80]!r}")


def parse_plan(text: str, max_steps: int | None = None) -> list[str]:
    """解析模型输出的步骤清单。坏数据一律 PlanError（带原文开头）。"""
    limit = max_steps or config.PLAN_MAX_STEPS
    payload = _extract_json_value(text)

    if isinstance(payload, dict):
        for key in ("steps", "plan", "步骤", "计划"):
            if key in payload:
                payload = payload[key]
                break

    if not isinstance(payload, list) or not payload:
        raise PlanError(f"计划必须是一个非空数组，收到了 {type(payload).__name__}")

    steps: list[str] = []
    for i, item in enumerate(payload, start=1):
        if isinstance(item, str) and item.strip():
            steps.append(item.strip())
        else:
            raise PlanError(f"第 {i} 个步骤不是一段文字（{item!r}）")
    if not steps:
        raise PlanError("计划里没有任何有效步骤")

    if len(steps) > limit:
        steps = steps[:limit]  # 裁到上限（宽松处理：多余的丢掉，日志里说明）

    return steps


def safe_parse_plan(text: str) -> tuple[list[str], str]:
    """永不抛的版本：失败了就降级成「单步计划」。返回 (步骤, 降级原因)。

    —— 为什么敢降级？图的价值在「流程控制」，不该被「拆计划」卡死。
    （v3 起的老哲学：增强功能坏了，主流程照走。）
    """
    try:
        return parse_plan(text), ""
    except PlanError as exc:
        return ["完成整个任务"], f"拆计划失败（{exc}）—— 已降级为「单步计划」照常执行"


# ══════════════════════════════════════════════════════════════════════
# 提示词构造（纯函数，方便 selftest 和调试时肉眼检查）
# ══════════════════════════════════════════════════════════════════════


def build_plan_prompt(task: str, max_steps: int | None = None) -> list[dict[str, str]]:
    limit = max_steps or config.PLAN_MAX_STEPS
    return [
        {
            "role": "system",
            "content": (
                "你是任务规划器。把用户任务拆成可独立执行的步骤清单。"
                "只输出一个 JSON 字符串数组（例如 [\"步骤一\", \"步骤二\"]），不要任何其他文字。"
                "每个步骤一句话、以动词开头。\n"
                "注意粒度（很重要）：能一步做完的动作不要拆成两步；"
                "对同一个文件/笔记的多次写入应当合并为一步（一次写完）；"
                "不要在同一件事上重复列步骤。"
            ),
        },
        {
            "role": "user",
            "content": f"最多拆 {limit} 步。任务：{task}",
        },
    ]


def render_plan_lines(plan: list[str], current: int) -> str:
    """把计划渲染成带状态标记的清单给执行者看（✅ 已完成 / ▶ 当前 / ☐ 未开始）。"""
    lines = []
    for i, step in enumerate(plan):
        if i < current:
            mark = "✅"
        elif i == current:
            mark = "▶"
        else:
            mark = "☐"
        lines.append(f"  {mark} {i + 1}. {step}")
    return "\n".join(lines)


# 执行阶段的角色设定（固定不变）
STEP_ROLE = (
    "你是一个执行 Agent，正在按计划完成一个任务。\n"
    "工作规则：\n"
    "- 需要计算 / 查询时间 / 保存内容时，调用对应工具；\n"
    "- 本步完成后，直接用一句话汇报结果（不用提第几步，直接说做了什么、得到什么），不要再等待指令；\n"
    "- 如果这一步反复尝试仍做不成，直接以「本步无法完成」开头说明原因。"
)


def build_step_text(task: str, plan: list[str], current: int) -> str:
    """当前步的指令正文 —— ⚠️ 它会被包成 **HumanMessage** 发出去（不是 system）。

    为什么？——消息序列必须是标准的「人机对话」形态：以 user 开场/推进。
    实测教训（2026-10）：把指令做成第二条 system、让请求以 [system, assistant] 收尾，
    在「带 tools + thinking」的服务端上会触发一个**误导性报错**
    （The `reasoning_content` … must be passed back）——查了半天包，真因是序列形态。
    「节点里怎么组装消息，是节点作者的职责」——别一出错就怪库。
    """
    step_text = plan[current] if 0 <= current < len(plan) else "（计划已到末尾）"
    return (
        f"【总任务】{task}\n"
        f"【完整计划】\n{render_plan_lines(plan, current)}\n"
        f"【本次指令】完成计划中的第 {current + 1} 步：{step_text}"
    )


def build_replan_prompt(
    task: str, plan: list[str], current: int, failure_reason: str
) -> list[dict[str, str]]:
    """重规划提示：只重拆「还没完成的部分」——已完成的不许动（防震荡的最小修订）。"""
    done = "\n".join(f"  ✅ {s}" for s in plan[:current]) or "（无）"
    remaining = "\n".join(f"  ☐ {s}" for s in plan[current:]) or "（无）"
    return [
        {
            "role": "system",
            "content": (
                "你是任务规划器。执行 Agent 在某个步骤上失败了，请**最小修订**原计划：\n"
                "只重新规划「还没完成的部分」，已完成的步骤不许改动、不许重复。\n"
                "只输出一个 JSON 字符串数组（重新规划的剩余步骤），不要任何其他文字。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"【总任务】{task}\n"
                f"【已完成（不许动）】\n{done}\n"
                f"【未完成 / 失败的部分】\n{remaining}\n"
                f"【失败原因】{failure_reason}\n\n"
                "请给出修订后的「剩余步骤」JSON 数组。"
            ),
        },
    ]
