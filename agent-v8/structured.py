"""
结构化输出 —— v8 的核心。

═══════════════════════════════════════════════════════════════════════
这一版要解决的问题
═══════════════════════════════════════════════════════════════════════

v7 之前，整个项目都在教模型「读懂机器」：
检索报告有名次、有分数、有出处 —— 全是代码算出来的结构化数据。

但反过来一直不成立：**机器看不懂模型的话。**
它最后那段回答是自由文本，除了给人看，程序什么都做不了：

  · 想知道「它到底引用了哪个文件」？得靠正则去猜
  · 想批量跑 20 个问题、自动判定对错？没法判定
  · 想让它和别的程序对接？对方只收到一段散文

v7 的收尾就卡在这里：评估（知识块 07）的第一步是「判定要程序化」，
而判定对象（模型的输出）根本进不了程序。

═══════════════════════════════════════════════════════════════════════
这一版引入的最小概念
═══════════════════════════════════════════════════════════════════════

    给模型的输出也定一份 schema —— 用你早就在用的机制：**工具调用**。

从 v1 起，工具的**参数**就是 JSON Schema（见 tools.py 的 ToolDefinition）。
这一版把「最终回答」也变成一组工具参数：

    模型不再是「说完就完了」，
    而是必须调用 `submit_result(...)`，把结果**交上来**：

        {
          "answer":     "给用户看的答案",
          "found":      true/false,        ← 答案有没有笔记依据
          "sources":    ["perf.md"],       ← 引用了哪些文件
          "confidence": 0.8                ← 把握几成
        }

于是「模型说了什么」从一段散文变成一个**字典** —— 程序可以先校验、再消费。

（外面还有两条路线：模型原生的 Structured Outputs / JSON 模式。
 三条路的取舍见 README 和 ECOSYSTEM.md —— 这里选的「工具即 schema」
 是对任何支持 tool calling 的模型都通用的那条。）

═══════════════════════════════════════════════════════════════════════
⚠️ schema 约束 ≠ 保证 —— 所以还要「校验 + 失败重试」
═══════════════════════════════════════════════════════════════════════

接口层能保证「长得像 JSON」，保证不了「说得对」。所以要有第二层：校验。

    · 语法/类型：字段全不全、类型对不对
    · 语义规则：answer 不能为空、confidence 在 0~1、
      found 和 sources 必须自洽（说找到了依据，就得给出来处）
    · ⭐ **可交叉核对的规则**：sources 引用了「本次根本没检索到过的文件」
      —— 编造出处！这条靠的是运行**轨迹**：
      search_notes 每送回一段，就记一笔「这个文件模型见过」。

校验不过 → 把「哪里错了」**回灌**给模型让它改 → 改不好再来 ——
这就是**失败重试**（工具报错本来就是回灌机制，v1 就有了）。
重试有上限（默认 3 次）：改不好就放弃，如实汇报失败 ——
**评估最怕的不是失败，是一堆「假装成功」的数据。**

═══════════════════════════════════════════════════════════════════════
三个想清楚了的决定
═══════════════════════════════════════════════════════════════════════

**① 校验器是纯函数**（数据进、问题列表出）—— 不碰网络、不碰模型，
   所以能徒手测穿（selftest ⑨ 组 J2 一行行喂坏数据）。

**② 「见过」= 真实轨迹，不是模型自述** —— 它说自己见过不算数，
   只有 search_notes 真返回过才算（轨迹在 tools.py 的 SEEN_SOURCES）。

**③ 放弃也是一种结果**：试满次数就抛 `StructuredGiveUp`，
   运行器输出 `ok=false` + 原因，退出码 2。
   宁可让调用方看到「这次不行」，也不给一份看似成功的脏数据。
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ══════════════════════════════════════════════════════════════════════
# 运行状态（每次结构化运行开始时由运行器重置）
# ══════════════════════════════════════════════════════════════════════


@dataclass
class SubmissionState:
    """一次结构化运行里，「提交」这件事的全部记录。"""

    active: bool = False      # 结构化模式开着吗（openai_schemas 靠它过滤工具）
    max_attempts: int = 3     # 提交上限：第 N 次仍不合规 → 放弃
    attempts: int = 0         # 已经提交过几次（含不合规的）
    failures: list[str] = field(default_factory=list)  # 每次被拦下的原因（人话）
    result: dict | None = None  # 最后一次**通过校验**的提交


# ⚠️ 刻意用「原地修改」而不是「重新赋值」：外面（tools.py）import 了这个对象，
#    如果哪天有人把 begin_run 改成 `STATE = SubmissionState(...)`，
#    别人手里的还是旧对象 —— 这类 bug 查起来极其痛苦。
STATE = SubmissionState()


class StructuredGiveUp(Exception):
    """提交次数超限 —— 放弃，不再陪模型磨。

    它刻意**穿透** Agent 循环（agent.py 里放行它）：
    它不代表「工具的一次失败」（那种应该回灌给模型），
    而是「这次运行已判死刑」—— 交给运行器收尸。
    """


def begin_run(max_attempts: int = 3) -> None:
    """开始一次结构化运行：状态清零、开关打开。"""
    STATE.active = True
    STATE.max_attempts = max(1, max_attempts)
    STATE.attempts = 0
    STATE.failures.clear()
    STATE.result = None


def finish_run() -> None:
    """运行结束：关掉开关。

    ⚠️ 必须关 —— 否则下一次普通对话里，模型会看到 submit_result 这个
       莫名其妙的工具（它不该参与自由聊天）。
    """
    STATE.active = False


# ══════════════════════════════════════════════════════════════════════
# 校验（纯函数）
# ══════════════════════════════════════════════════════════════════════


def validate_submission(data: dict, seen_sources: set[str]) -> list[str]:
    """检查一次提交，返回**问题列表**（空列表 = 通过）。

    ⚠️ 刻意不做「遇到第一个问题就返回」：一次把所有问题列全，
    模型一轮就能全部改掉 —— 少一轮往返，少一次 API 调用。
    每条问题都按铁律 7 的格式写：「哪里错、现在是什么、该是什么」。
    """
    problems: list[str] = []

    answer = data.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        problems.append(f"answer 必须是非空字符串（现在拿到的是：{answer!r}）")

    found = data.get("found")
    if not isinstance(found, bool):
        problems.append(f"found 必须是 true 或 false（现在拿到的是：{found!r}）")

    sources = data.get("sources")
    sources_ok = isinstance(sources, list) and all(isinstance(s, str) for s in sources)
    if not sources_ok:
        problems.append(f"sources 必须是字符串数组（现在拿到的是：{sources!r}）")
    else:
        unknown = [s for s in sources if s not in seen_sources]
        if unknown:
            allowed = "、".join(sorted(seen_sources)) or "（无 —— 本次没有检索到任何内容）"
            problems.append(
                f"sources 里的 {unknown} 不是本次检索见过的文件（见过的只有：{allowed}）"
                " —— 出处必须是真实检索结果，不能编造"
            )

    confidence = data.get("confidence")
    # ⚠️ bool 要挡掉：Python 里 True 也是 int（isinstance(True, int) 为真），
    #    不挡的话 confidence=true 会被当成 1.0 混过去。
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not (0.0 <= float(confidence) <= 1.0)
    ):
        problems.append(f"confidence 必须是 0~1 之间的数字（现在拿到的是：{confidence!r}）")

    # ── 自洽性：found 和 sources 必须对得上 ─────────────────────────
    if isinstance(found, bool) and sources_ok:
        if found and not sources:
            problems.append("found=true 却给不出 sources —— 说找到了依据，就要给出处")
        if not found and sources:
            problems.append("found=false 却带了 sources —— 说没有依据，就不该列出来源")

    return problems


# ══════════════════════════════════════════════════════════════════════
# 系统提示词续章 —— 结构化模式的「玩法说明」
# ══════════════════════════════════════════════════════════════════════
#
# 普通对话**不加**这一段 —— 这就是 ROADMAP 里说的「什么时候不该用」：
# 给人看的闲聊不该被塞进 JSON。只有程序入口（--ask）才拼上它。

SYSTEM_ADDENDUM = """
【结构化输出模式】
你现在以「结构化输出」方式运行：最终结果**必须调用 submit_result 工具提交**，
不要用普通文字作答。要求：
1. 提交之前先完成必要的工作（比如先用 search_notes 检索笔记）。
2. sources 只能填本次运行中 search_notes 真正返回过的文件；
   没有依据就填 []，并把 found 设为 false。出处造假会被校验拦下。
3. 如果 submit_result 返回错误，按错误提示修正字段后重新提交。
"""
