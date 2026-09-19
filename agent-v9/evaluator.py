"""
评估器 —— v9 的核心：把「感觉」变成「数字」。

═══════════════════════════════════════════════════════════════════════
一次评估，就是一条回路：
═══════════════════════════════════════════════════════════════════════

    用例集（固定问题 + 每条可自动判定的期望）
        │
        ▼
    挨个跑一遍 Agent ──→ 一组「回答 + 工具轨迹（trace）」
        │
        ▼
    判定：每条期望 → 过 / 不过 + 一句人话理由（为什么不过、实际是什么）
        │
        ▼
    分数 + 运行记录（runs/*.json）──对比──→ 上一次的记录
                                                 │
                                            「变好还是变坏」有答案了

═══════════════════════════════════════════════════════════════════════
三个刻意的设计决定（每个都吃过亏）
═══════════════════════════════════════════════════════════════════════

1. **判定逻辑是纯函数**（数据进、结论出）—— 离线可测，不花一分钱。
   （v3 的假摘要器 / v6 的假向量器 / v8 的剧本客户端：同一个套路第四次出场）

2. **判定要对「无害的格式差异」免疫** —— `7,006,742` 和 `7006742`
   是同一个答案。判定太死板，红一片，全是假警报，你就再也不信分数了。

3. **失败必须说清楚「预期什么、实际什么、哪里能看」** —— 铁律 7。
   一句「不通过」是废话；「没调用 calculator，实际一个工具都没调；回答开头：…」
   才是能拿去做决定的信息。

═══════════════════════════════════════════════════════════════════════
两种判定方式的取舍（这是这一版最重要的判断力）
═══════════════════════════════════════════════════════════════════════

    规则判定（确定性）              模型判定（LLM-as-judge）
    ─────────────────────          ─────────────────────
    · 便宜、离线、逐位可复现        · 贵、要联网、含义级判断
    · 只能判「能写成条件的东西」    · 能判「语气友好」「解释通俗」
    · 死板：写错一个字符就假红      · 模糊：它自己也会飘、要校准

    → 优先规则；规则写不出来的，才请裁判。别拿裁判干规则的活，反之亦然。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import config


class EvalError(Exception):
    """用例集 / 运行记录读不进来时抛这个 —— 消息要能直接指导修复。"""


# ══════════════════════════════════════════════════════════════════════
# 检查类型登记表
# ══════════════════════════════════════════════════════════════════════
#
# 每种检查声明：属于哪一类（报告里分类统计用）+ 必须带哪些参数。
# 加新检查类型 = 在这里加一行 + 在 run_check() 里加一段。

CHECK_TYPES: dict[str, tuple[str, tuple[str, ...]]] = {
    # type              (类别,        必需参数)
    "tool_called": ("工具行为", ("name",)),
    "no_tools": ("工具行为", ()),
    "answer_contains": ("回答内容", ("value",)),
    "answer_not_contains": ("回答内容", ("value",)),
    "answer_regex": ("回答内容", ("value",)),
    "file_exists": ("副作用", ("value",)),
    "llm_judge": ("模型判定", ("criterion",)),
}


# ══════════════════════════════════════════════════════════════════════
# 用例集
# ══════════════════════════════════════════════════════════════════════


@dataclass
class CheckSpec:
    """一条期望。"""

    type: str
    params: dict[str, Any]

    @property
    def category(self) -> str:
        return CHECK_TYPES[self.type][0]

    def label(self) -> str:
        """给人看的一行标题，失败明细里用。"""
        p = self.params
        if self.type == "tool_called":
            return f"调用 {p['name']}"
        if self.type == "no_tools":
            return "不调用任何工具"
        if self.type == "answer_contains":
            return f"回答包含「{p['value']}」"
        if self.type == "answer_not_contains":
            return f"回答不含「{p['value']}」"
        if self.type == "answer_regex":
            return f"回答匹配 /{p['value']}/"
        if self.type == "file_exists":
            return f"生成文件 {p['value']}"
        if self.type == "llm_judge":
            return f"裁判评判：{p['criterion']}"
        return self.type


@dataclass
class EvalCase:
    """一条用例 = 问什么 + 怎么算答对。"""

    id: str
    prompt: str
    checks: list[CheckSpec]
    note: str = ""


def load_cases(path: Path) -> list[EvalCase]:
    """读 JSONL 用例集。每一行一个用例。

    所有出错情况都抛 EvalError，而且消息里要带**文件:行号 + 怎么修** ——
    用例集是你要经常改的文件，报错含糊等于逼你回来读源码。
    """
    path = Path(path)
    if not path.is_file():
        raise EvalError(
            f"用例集文件不存在：{path}\n"
            f"（默认在 evals/cases.jsonl；也可以用 LLM_EVAL_CASES 指到别处）"
        )
    raw = path.read_text(encoding="utf-8")

    cases: list[EvalCase] = []
    seen_ids: dict[str, int] = {}
    for lineno, line in enumerate(raw.splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            raise EvalError(
                f"{path.name} 第 {lineno} 行不是合法 JSON：{exc.msg}"
                f"（每行必须是一个完整用例，注意引号和逗号）"
            ) from exc
        if not isinstance(obj, dict):
            raise EvalError(f"{path.name} 第 {lineno} 行不是 JSON 对象，而是一行「{text[:40]}…」")

        cid = str(obj.get("id") or "").strip()
        if not cid:
            raise EvalError(f"{path.name} 第 {lineno} 行缺少 id —— 每条用例要有名字，失败清单才指得清是谁")
        if cid in seen_ids:
            raise EvalError(
                f"{path.name} 第 {lineno} 行的 id「{cid}」和第 {seen_ids[cid]} 行重复 —— "
                f"对比基线时会对不上账"
            )
        seen_ids[cid] = lineno

        prompt = str(obj.get("prompt") or "").strip()
        if not prompt:
            raise EvalError(f"{path.name} 第 {lineno} 行（{cid}）缺少 prompt —— 要问它什么？")

        raw_checks = obj.get("checks")
        if not isinstance(raw_checks, list) or not raw_checks:
            raise EvalError(
                f"{path.name} 第 {lineno} 行（{cid}）没有 checks —— 没有期望，就没法判定对错"
            )

        checks: list[CheckSpec] = []
        for raw_check in raw_checks:
            if not isinstance(raw_check, dict) or not raw_check.get("type"):
                raise EvalError(f"{path.name} 第 {lineno} 行（{cid}）里有一条 checks 项缺少 type")
            ctype = str(raw_check["type"])
            if ctype not in CHECK_TYPES:
                valid = "、".join(CHECK_TYPES)
                raise EvalError(
                    f"{path.name} 第 {lineno} 行（{cid}）用了不认识的检查类型「{ctype}」——"
                    f"可用的有：{valid}"
                )
            missing = [k for k in CHECK_TYPES[ctype][1] if not raw_check.get(k)]
            if missing:
                raise EvalError(
                    f"{path.name} 第 {lineno} 行（{cid}）的 {ctype} 检查缺少参数："
                    f"{'、'.join(missing)}"
                )
            checks.append(
                CheckSpec(type=ctype, params={k: v for k, v in raw_check.items() if k != "type"})
            )

        cases.append(EvalCase(id=cid, prompt=prompt, checks=checks, note=str(obj.get("note") or "")))

    if not cases:
        raise EvalError(f"用例集是空的：{path}（至少要有一个用例才算一份卷子）")
    return cases


# ══════════════════════════════════════════════════════════════════════
# 判定：规则部分（纯函数，离线可测）
# ══════════════════════════════════════════════════════════════════════


@dataclass
class Submission:
    """判定需要知道的、关于「这次回答」的全部事实。

    notes_before 是跑这条用例**之前**的资料目录快照（文件名 -> (mtime_ns, 大小)）。
    为什么需要它？见 run_check() 里 file_exists 那段 —— 一句话：
    **残留文件会让检查假通过，而假通过比假红更毒。**
    """

    answer: str
    tool_names: list[str]
    notes_dir: Path
    notes_before: dict[str, tuple[int, int]] = field(default_factory=dict)


@dataclass
class CheckResult:
    type: str
    category: str
    ok: bool
    detail: str
    label: str = ""  # 「期望什么」的人话标题（从 CheckSpec 抄一份，日志/明细直接用）

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "category": self.category,
            "ok": self.ok,
            "detail": self.detail,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CheckResult":
        return cls(
            type=str(d.get("type", "")),
            category=str(d.get("category", "")),
            ok=bool(d.get("ok")),
            detail=str(d.get("detail", "")),
            label=str(d.get("label", "")),
        )


def _normalize(text: str) -> str:
    """去掉空白 + 去掉数字里的千分位逗号。

    '7,006,742' 和 '7006742' 是同一个答案；空格不该影响子串匹配。
    判定必须对「无害的格式差异」免疫，否则红一片假警报。
    """
    compact = re.sub(r"\s+", "", text)
    return re.sub(r"(?<=\d),(?=\d)", "", compact)


def _excerpt(text: str, limit: int = 60) -> str:
    """把一段文本压成一行、截短 —— 给日志用（方便你一眼看到「实际是什么」）。"""
    flat = re.sub(r"\s+", " ", text or "").strip()
    if not flat:
        return "（空回答）"
    return flat[:limit] + ("…" if len(flat) > limit else "")


def run_check(spec: CheckSpec, submission: Submission) -> CheckResult:
    """执行一条【确定性】检查 —— 不联网、不花钱、逐位可复现。"""
    ctype = spec.type
    category = spec.category
    p = spec.params
    answer = submission.answer or ""
    normalized = _normalize(answer)
    tools = submission.tool_names

    def ok(detail: str) -> CheckResult:
        return CheckResult(type=ctype, category=category, ok=True, detail=detail, label=spec.label())

    def fail(detail: str) -> CheckResult:
        return CheckResult(type=ctype, category=category, ok=False, detail=detail, label=spec.label())

    if ctype == "tool_called":
        name = str(p["name"])
        if name in tools:
            return ok(f"调用了 {name}")
        actual = "、".join(tools) if tools else "一个都没有"
        return fail(f"没有调用 {name}（这次实际调用：{actual}）")

    if ctype == "no_tools":
        if not tools:
            return ok("没有调用任何工具")
        return fail(f"要求不调用工具，实际调用了：{'、'.join(tools)}")

    if ctype == "answer_contains":
        value = _normalize(str(p["value"]))
        if value in normalized:
            return ok(f"回答里出现了「{p['value']}」")
        return fail(f"回答里没有「{p['value']}」｜回答开头：{_excerpt(answer)}")

    if ctype == "answer_not_contains":
        value = _normalize(str(p["value"]))
        if value not in normalized:
            return ok(f"回答里没有出现「{p['value']}」")
        return fail(f"回答里不该出现「{p['value']}」，但它出现了｜回答开头：{_excerpt(answer)}")

    if ctype == "answer_regex":
        pattern = str(p["value"])
        try:
            matched = re.search(pattern, normalized)
        except re.error as exc:
            return fail(f"用例里的正则写错了：{exc} —— 这是用例的问题，不是模型的问题")
        if matched:
            return ok(f"回答匹配 /{pattern}/（命中「{matched.group(0)}」）")
        return fail(f"回答不匹配 /{pattern}/｜回答开头：{_excerpt(answer)}")

    if ctype == "file_exists":
        rel = str(p["value"])
        notes_dir = Path(submission.notes_dir)
        target = notes_dir / rel
        if target.is_file():
            st = target.stat()
            before = submission.notes_before.get(rel)
            if before == (st.st_mtime_ns, st.st_size):
                # ⭐ 文件在，但和跑之前一模一样 —— 这不是这次写出来的。
                # 评估里最毒的错就是这种「假通过」：分数给了你，但什么都没发生。
                return fail(
                    f"{rel} 存在，但不是这次写的（和跑之前一模一样）—— 很可能是上一次评估留下的残留。"
                    f"先清空资料目录（rm -rf {notes_dir}）再跑，这条才代表真实能力"
                )
            return ok(f"生成了 {rel}（{st.st_size} 字节，本次运行新写入）")

        listing = (
            sorted(x.name for x in notes_dir.iterdir() if x.is_file())
            if notes_dir.is_dir()
            else []
        )
        shown = "、".join(listing) if listing else "（目录是空的）"
        return fail(f"没有生成 {rel}｜资料目录里现在有：{shown}")

    raise EvalError(f"内部错误：检查类型 {ctype} 没有实现")


# ══════════════════════════════════════════════════════════════════════
# 判定：裁判部分（LLM-as-judge）
# ══════════════════════════════════════════════════════════════════════

JUDGE_PROMPT = """你是一位严格的评审员。你会拿到一条评审标准、一条用户问题、一份待评回答。
请**只根据这条标准**评判，不要加入你自己的偏好；答非所问、含糊其辞都应判 FAIL。

评审标准：
{criterion}

用户问题：
{prompt}

待评回答：
{answer}

输出格式（不要输出其他任何内容）：
第一行：PASS 或 FAIL
第二行：一句话理由（中文）"""


def build_judge_messages(spec: CheckSpec, case_prompt: str, answer: str) -> list[dict[str, str]]:
    """拼裁判的一次性对话（无历史、无工具）。"""
    content = JUDGE_PROMPT.format(
        criterion=spec.params["criterion"],
        prompt=case_prompt,
        answer=answer or "（空回答）",
    )
    return [{"role": "user", "content": content}]


def parse_judge_reply(text: str) -> tuple[bool, str] | None:
    """解析裁判输出：第一行 PASS/FAIL，第二行理由。

    容错：去掉 markdown 装饰（**PASS**、`FAIL`）；解析不了就返回 None ——
    调用方会把「解析不了」记成未通过，而不是假装通过。
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return None
    first = lines[0].strip("*#` ").upper()
    reason = lines[1].strip() if len(lines) > 1 else "（裁判没写理由）"
    if first.startswith("PASS"):
        return True, reason
    if first.startswith("FAIL"):
        return False, reason
    return None


def run_judge(
    spec: CheckSpec,
    case_prompt: str,
    submission: Submission,
    llm: Any,
    model: str | None = None,
) -> CheckResult:
    """请裁判判一条标准。裁判调用失败 / 输出解析不了 → 按未通过记。"""
    category = spec.category
    messages = build_judge_messages(spec, case_prompt, submission.answer)
    try:
        reply = llm.complete(
            messages,
            use_tools=False,
            temperature=config.EVAL_TEMPERATURE,
            model=model or config.JUDGE_MODEL,
            max_tokens=config.JUDGE_MAX_TOKENS,
            purpose="judge",
        )
    except Exception as exc:  # 裁判挂了不该连累整场评估 —— 但也不能装作判过
        return CheckResult(
            type=spec.type,
            category=category,
            ok=False,
            detail=f"裁判调用失败：{type(exc).__name__}: {exc}（这条按未通过记；修好后再跑）",
            label=spec.label(),
        )

    raw_reply = (getattr(reply.message, "content", "") or "").strip()
    finish_reason = getattr(reply, "finish_reason", None)
    parsed = parse_judge_reply(raw_reply)
    if parsed is None:
        # ⚠️ 判定环节自己也会坏。坏法必须和「模型答得不好」区分开，
        #    否则你的失败清单会指向错误的嫌疑人。
        if not raw_reply and finish_reason == "length":
            detail = (
                f"裁判的回答被截断：{config.JUDGE_MAX_TOKENS} tokens 用光了还没写出结论"
                f"（finish_reason=length）—— 这是判定环节的故障，不是模型答得不好。"
                f"先按未通过记；把 LLM_JUDGE_MAX_TOKENS 调大再跑"
            )
        elif not raw_reply:
            detail = "裁判返回了空内容（按未通过记）—— 这是判定环节的故障，不是回答的问题"
        else:
            detail = f"裁判输出无法解析（按未通过记）｜原样开头：{_excerpt(raw_reply, 80)}"
        return CheckResult(
            type=spec.type,
            category=category,
            ok=False,
            detail=detail,
            label=spec.label(),
        )
    ok_flag, reason = parsed
    return CheckResult(
        type=spec.type,
        category=category,
        ok=ok_flag,
        detail=f"裁判 {'PASS' if ok_flag else 'FAIL'}：{reason}",
        label=spec.label(),
    )


# ══════════════════════════════════════════════════════════════════════
# 跑一场评估
# ══════════════════════════════════════════════════════════════════════


@dataclass
class CaseResult:
    """一条用例的成绩单。"""

    id: str
    prompt: str
    passed: bool
    checks: list[CheckResult]
    tool_names: list[str]
    answer: str
    elapsed_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None

    def to_dict(self, answer_limit: int = 1000) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "passed": self.passed,
            "tools": self.tool_names,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "answer": self.answer[:answer_limit],
            "error": self.error,
            "checks": [c.to_dict() for c in self.checks],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CaseResult":
        return cls(
            id=str(d.get("id", "?")),
            prompt=str(d.get("prompt", "")),
            passed=bool(d.get("passed")),
            checks=[CheckResult.from_dict(c) for c in d.get("checks", [])],
            tool_names=[str(t) for t in d.get("tools", [])],
            answer=str(d.get("answer", "")),
            elapsed_ms=float(d.get("elapsed_ms", 0.0)),
            prompt_tokens=int(d.get("prompt_tokens", 0)),
            completion_tokens=int(d.get("completion_tokens", 0)),
            error=d.get("error"),
        )


def _notes_snapshot(notes_dir: Path) -> dict[str, tuple[int, int]]:
    """拍一张资料目录的快照（文件名 -> (mtime_ns, 大小)）。

    给 file_exists 检查当「跑之前的底片」用：跑完之后再对一次，
    分得清「这是刚才写出来的」和「这是上次留下的」。
    """
    notes_dir = Path(notes_dir)
    if not notes_dir.is_dir():
        return {}
    snapshot: dict[str, tuple[int, int]] = {}
    for f in notes_dir.iterdir():
        if f.is_file():
            st = f.stat()
            snapshot[f.name] = (st.st_mtime_ns, st.st_size)
    return snapshot


def run_case(
    case: EvalCase,
    make_agent: Callable[[], Any],
    llm: Any,
    notes_dir: Path,
) -> CaseResult:
    """跑一条用例：**新建一个 Agent**（没有历史污染）→ 问 → 逐条判定。

    为什么每条用例都要新 Agent？
      评估要的是「同样条件下重复可测」。如果第一条的对话历史影响了第二条，
      分数就没法比较 —— 这也是 v5/v6 实验手册那个「上一个实验污染下一个」
      教训在评估里的版本。
    """
    started = time.perf_counter()
    notes_before = _notes_snapshot(Path(notes_dir))  # ⭐ 跑之前的底片
    try:
        agent = make_agent()
        run = agent.ask(case.prompt)
    except Exception as exc:
        return CaseResult(
            id=case.id,
            prompt=case.prompt,
            passed=False,
            checks=[],
            tool_names=[],
            answer="",
            elapsed_ms=(time.perf_counter() - started) * 1000,
            error=f"{type(exc).__name__}: {exc}",
        )

    submission = Submission(
        answer=run.answer,
        tool_names=run.tool_names,
        notes_dir=Path(notes_dir),
        notes_before=notes_before,
    )
    check_results: list[CheckResult] = []
    for spec in case.checks:
        if spec.type == "llm_judge":
            check_results.append(run_judge(spec, case.prompt, submission, llm))
        else:
            check_results.append(run_check(spec, submission))

    return CaseResult(
        id=case.id,
        prompt=case.prompt,
        passed=all(c.ok for c in check_results),
        checks=check_results,
        tool_names=run.tool_names,
        answer=run.answer,
        elapsed_ms=run.elapsed_ms,
        prompt_tokens=run.prompt_tokens,
        completion_tokens=run.completion_tokens,
    )


@dataclass
class RunReport:
    """一场评估的完整记录（就是写进 runs/*.json 的东西）。"""

    meta: dict[str, Any] = field(default_factory=dict)
    results: list[CaseResult] = field(default_factory=list)
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    started_at: str = ""
    finished_at: str = ""

    # ── 汇总指标 ────────────────────────────────────────────────────
    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def pass_rate(self) -> float:
        return (self.passed_count / self.total * 100) if self.total else 0.0

    @property
    def avg_ms(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.elapsed_ms for r in self.results) / len(self.results)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def failed_ids(self) -> list[str]:
        return [r.id for r in self.results if not r.passed]

    @property
    def judge_count(self) -> int:
        return sum(1 for r in self.results for c in r.checks if c.type == "llm_judge")

    def category_stats(self) -> dict[str, tuple[int, int]]:
        """分类统计（检查项维度，不是用例维度）：类别 -> (过, 共)。"""
        stats: dict[str, list[int]] = {}
        for r in self.results:
            for c in r.checks:
                row = stats.setdefault(c.category, [0, 0])
                row[1] += 1
                if c.ok:
                    row[0] += 1
        return {k: (v[0], v[1]) for k, v in stats.items()}

    def pass_map(self) -> dict[str, bool]:
        return {r.id: r.passed for r in self.results}

    # ── 序列化 ──────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,  # 版本号从第一天就有（v4 的教训）
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "meta": self.meta,
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "summary": {
                "passed": self.passed_count,
                "total": self.total,
                "pass_rate": round(self.pass_rate, 1),
            },
            "results": [r.to_dict() for r in self.results],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunReport":
        return cls(
            meta=dict(d.get("meta") or {}),
            results=[CaseResult.from_dict(r) for r in d.get("results", [])],
            calls=int(d.get("calls", 0)),
            prompt_tokens=int(d.get("prompt_tokens", 0)),
            completion_tokens=int(d.get("completion_tokens", 0)),
            started_at=str(d.get("started_at", "")),
            finished_at=str(d.get("finished_at", "")),
        )


def run_all(
    cases: list[EvalCase],
    *,
    make_agent: Callable[[], Any],
    llm: Any,
    notes_dir: Path | None = None,
    meta: dict[str, Any] | None = None,
    only: str | None = None,
    on_case_done: Callable[[CaseResult], None] | None = None,
) -> RunReport:
    """跑一整场评估。`--only <id>` 只跑一条（调试用）。"""
    selected = cases
    if only:
        selected = [c for c in cases if c.id == only]
        if not selected:
            known = "、".join(c.id for c in cases)
            raise EvalError(f"--only {only} 没有匹配到任何用例（现有 id：{known}）")

    report = RunReport(
        meta={
            **(meta or {}),
            "only": only,
            "cases_total": len(cases),
            "selected": len(selected),
            "notes_dir": str(notes_dir or config.NOTES_DIR),
            "disabled_tools": sorted(config.DISABLED_TOOLS),
        },
        started_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    calls_before, in_before, out_before = _snapshot(llm)
    for case in selected:
        result = run_case(case, make_agent, llm, notes_dir or config.NOTES_DIR)
        report.results.append(result)
        if on_case_done:
            on_case_done(result)
    calls_after, in_after, out_after = _snapshot(llm)

    report.calls = calls_after - calls_before
    report.prompt_tokens = in_after - in_before
    report.completion_tokens = out_after - out_before
    report.finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return report


def _snapshot(llm: Any) -> tuple[int, int, int]:
    """读 llm 的账本（没有账本就记 0 —— 替身客户端可以不带）。"""
    if hasattr(llm, "snapshot"):
        return llm.snapshot()
    return 0, 0, 0


# ══════════════════════════════════════════════════════════════════════
# 运行记录：存盘 / 读回 / 找最新的
# ══════════════════════════════════════════════════════════════════════


def save_run(report: RunReport, runs_dir: Path) -> Path:
    """把这次评估写进 runs/。原子写（v4 的老规矩：临时文件 + 替换）。"""
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = re.sub(r"[^\w-]+", "_", str(report.meta.get("tag") or "run")) or "run"
    path = runs_dir / f"{stamp}-{tag}.json"
    n = 2
    while path.exists():
        path = runs_dir / f"{stamp}-{tag}-{n}.json"
        n += 1

    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def load_run(path: Path) -> RunReport:
    """读一份运行记录。坏了要说清楚坏在哪、怎么办。"""
    path = Path(path)
    if not path.is_file():
        raise EvalError(
            f"运行记录不存在：{path}\n（先用 --eval 跑一次；或加 --baseline latest 让它自动找最新的一份）"
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise EvalError(
            f"运行记录读不进来：{path}（{type(exc).__name__}: {exc}）\n"
            f"—— 它可能被写坏了。删掉这个文件，或换一份 --baseline"
        ) from exc
    if not isinstance(data, dict) or "results" not in data:
        raise EvalError(
            f"运行记录格式不对：{path}（缺少 results 字段）—— 这不是 --eval 写出来的文件？"
        )
    return RunReport.from_dict(data)


def find_latest_run(runs_dir: Path, exclude: Path | None = None) -> Path | None:
    """找 runs/ 里最新的一份（按修改时间）。

    exclude 用来排除「本次刚写的那份」—— 对比的对象必须是**上一次**，
    不是自己（这个坑不加参数就必踩：跑完立刻保存，latest 就是自己）。
    """
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return None
    exclude_resolved = Path(exclude).resolve() if exclude else None
    candidates = [
        p
        for p in runs_dir.glob("*.json")
        if p.is_file() and (exclude_resolved is None or p.resolve() != exclude_resolved)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


# ══════════════════════════════════════════════════════════════════════
# 对比：回答「变好还是变坏」
# ══════════════════════════════════════════════════════════════════════


@dataclass
class Comparison:
    baseline_path: str
    common: list[str]
    new_pass: list[str]      # 基线没过、这次过了
    new_fail: list[str]      # 基线过了、这次没过
    same_pass: list[str]
    same_fail: list[str]
    only_baseline: list[str]  # 只在基线里有的用例 id
    only_current: list[str]
    baseline_pass_common: int
    current_pass_common: int
    prompt_baseline: str
    prompt_current: str
    prompt_changed: bool
    model_baseline: str
    model_current: str
    temp_baseline: str
    temp_current: str
    tools_baseline: str    # 被 LLM_DISABLED_TOOLS 关掉的工具（“无” = 全开）
    tools_current: str

    @property
    def delta(self) -> int:
        return self.current_pass_common - self.baseline_pass_common

    @property
    def anything_changed(self) -> bool:
        """提示词 / 模型 / 温度 / 工具开关里有任何一样变了没有？"""
        return (
            self.prompt_changed
            or self.model_baseline != self.model_current
            or self.temp_baseline != self.temp_current
            or self.tools_baseline != self.tools_current
        )


def _fmt_disabled_tools(meta: dict[str, Any]) -> str:
    tools = meta.get("disabled_tools") or []
    return "、".join(tools) if tools else "无"


def compare_runs(current: RunReport, baseline: RunReport, baseline_path: str = "") -> Comparison:
    """按用例 id 对齐，算清「新通过 / 新失败 / 持平」。

    ⚠️ 只对比两次都有的用例（共同集）—— --only 调试版和整卷对比时，
    拿「1 条 vs 9 条」的总分做差是自欺欺人。
    """
    cur_map = current.pass_map()
    base_map = baseline.pass_map()
    common = [i for i in cur_map if i in base_map]
    only_current = [i for i in cur_map if i not in base_map]
    only_baseline = [i for i in base_map if i not in cur_map]

    new_pass = [i for i in common if cur_map[i] and not base_map[i]]
    new_fail = [i for i in common if not cur_map[i] and base_map[i]]
    same_pass = [i for i in common if cur_map[i] and base_map[i]]
    same_fail = [i for i in common if not cur_map[i] and not base_map[i]]

    p_base = str(baseline.meta.get("prompt_sha", "?"))
    p_cur = str(current.meta.get("prompt_sha", "?"))
    m_base = str(baseline.meta.get("model", "?"))
    m_cur = str(current.meta.get("model", "?"))
    t_base = str(baseline.meta.get("temperature", "?"))
    t_cur = str(current.meta.get("temperature", "?"))

    return Comparison(
        baseline_path=baseline_path,
        common=common,
        new_pass=new_pass,
        new_fail=new_fail,
        same_pass=same_pass,
        same_fail=same_fail,
        only_baseline=only_baseline,
        only_current=only_current,
        baseline_pass_common=sum(1 for i in common if base_map[i]),
        current_pass_common=sum(1 for i in common if cur_map[i]),
        prompt_baseline=p_base,
        prompt_current=p_cur,
        prompt_changed=(p_base != p_cur),
        model_baseline=m_base,
        model_current=m_cur,
        temp_baseline=t_base,
        temp_current=t_cur,
        tools_baseline=_fmt_disabled_tools(baseline.meta),
        tools_current=_fmt_disabled_tools(current.meta),
    )


# ══════════════════════════════════════════════════════════════════════
# 输出格式化 —— 全部做成「字符串进、字符串出」，这样日志质量也能被测
# ══════════════════════════════════════════════════════════════════════
#
# 铁律 7：学习者看不到你的代码，只能看到终端滚过的这几行。
# 所以每一条「我没做 X」，都要说清楚「因为 Y」——下面是落实。


def format_header(meta: dict[str, Any]) -> str:
    """开跑之前先交代清楚：用什么卷子、什么配置、什么提示词。"""
    lines = [
        f"评估：{meta.get('selected', '?')}/{meta.get('cases_total', '?')} 条用例"
        f" · 模型 {meta.get('model', '?')} · 评估温度 {meta.get('temperature', '?')}",
        f"提示词：{meta.get('prompt_label', '?')}",
        f"用例集：{meta.get('cases_file', '?')}",
    ]
    if meta.get("only"):
        lines.append(
            f"⚠ 只跑 1 条（--only {meta['only']}）—— 这是调试版成绩，别当整卷用"
        )
    disabled = meta.get("disabled_tools") or []
    if disabled:
        lines.append(
            f"⚠ 已关闭工具：{'、'.join(disabled)}（LLM_DISABLED_TOOLS 指定）"
            f"—— 依赖它们的用例会失败，这正是你要看的"
        )
    return "\n".join(lines)


def format_case_line(result: CaseResult, width: int = 0) -> str:
    """一条用例跑完的实况行（边跑边打，学习者能看着分数一条条出来）。"""
    mark = "✓" if result.passed else "✗"
    ms = f"{result.elapsed_ms / 1000:.1f}s"
    if result.error:
        return f"  {mark} {result.id.ljust(width)} 运行出错 · {ms}"
    tools = "、".join(result.tool_names) if result.tool_names else "无"
    checks_ok = sum(1 for c in result.checks if c.ok)
    return (
        f"  {mark} {result.id.ljust(width)} 工具: {tools}"
        f" · {checks_ok}/{len(result.checks)} 项检查过 · {ms}"
    )


def format_failures(report: RunReport) -> str:
    """失败明细：每条都写清「问题是什么、哪条期望没过、实际是什么」。"""
    failed = [r for r in report.results if not r.passed]
    if not failed:
        return ""
    lines = [f"✗ 失败明细（{len(failed)} 条）：", ""]
    for r in failed:
        lines.append(f"  {r.id}（问题：{r.prompt}）")
        if r.error:
            lines.append(f"    ✗ 运行出错：{r.error}")
        for c in r.checks:
            if not c.ok:
                lines.append(f"    ✗ {c.label or c.type} —— {c.detail}")
        if not r.error:
            lines.append(f"    回答开头：{_excerpt(r.answer, 80)}")
        lines.append("")
    return "\n".join(lines).rstrip()


def format_judge_notes(report: RunReport) -> str:
    """把「过了的裁判判定」的理由也拉出来给你看。

    ⭐ 判定理由不该只藏在 run 文件里：裁判说 PASS / FAIL 的理由，
    既是「这条为什么算过」的解释，也是你校准它的第一手材料。
    （没过的那些已经出现在失败明细里了，这里只补上过的。）
    """
    lines = [
        f"  🧑‍⚖️ {r.id}：{c.detail}"
        for r in report.results
        for c in r.checks
        if c.type == "llm_judge" and c.ok
    ]
    return "\n".join(lines)


def format_summary(report: RunReport) -> str:
    """收尾的分数板 —— 数字 + 分类 + 记账 + 下一步。"""
    lines = ["─" * 56]
    lines.append(
        f"通过 {report.passed_count}/{report.total}（{report.pass_rate:.1f}%）"
        f" · 平均 {report.avg_ms / 1000:.1f}s/条"
        f" · {report.calls} 次模型调用 · {report.total_tokens:,} tokens"
    )
    stats = report.category_stats()
    if stats:
        parts = [f"{cat} {ok}/{total}" for cat, (ok, total) in stats.items()]
        lines.append("分类（检查项口径）：" + " · ".join(parts))
    if report.judge_count:
        lines.append(
            f"其中裁判判定 {report.judge_count} 次 —— 裁判也是模型，要校准（见 README「裁判也需要校准」）"
        )
    else:
        lines.append("⏭ 没有用例需要模型判定 —— 全部检查离线完成，零额外调用")
    return "\n".join(lines)


def _join_ids(ids: list[str], limit: int = 4) -> str:
    """把 id 列表拼成人话；太长了就截断 —— 面板不该刷屏。"""
    if len(ids) <= limit:
        return "、".join(ids)
    return "、".join(ids[:limit]) + f" 等 {len(ids)} 条"


def format_comparison(cmp: Comparison) -> str:
    """对比面板 —— 这一版的正题就是这几行。"""
    lines = [f"对比基线 {cmp.baseline_path}"]
    if not cmp.common:
        lines.append("  ⚠ 两次没有共同用例（id 全对不上）—— 基线不适用，换一份吧")
        return "\n".join(lines)

    lines.append(
        f"  通过率    {cmp.baseline_pass_common}/{len(cmp.common)} → "
        f"{cmp.current_pass_common}/{len(cmp.common)}（{cmp.delta:+d}）"
    )

    # 提示词指纹是第一嫌疑人；模型 / 温度 / 工具开关也要摊开来讲。
    # ⚠️ 教训（实验 4）：只改 prompt 的时候就盯着 prompt，
    #    一旦成因在别处（关了个工具），面板就必须说对 —— 不给假因果。
    if cmp.prompt_changed:
        lines.append(f"  提示词    {cmp.prompt_baseline} → {cmp.prompt_current}（变了）")
    else:
        lines.append(f"  提示词    {cmp.prompt_current}（没变）")
    if cmp.model_baseline != cmp.model_current:
        lines.append(f"  模型      {cmp.model_baseline} → {cmp.model_current}（变了）")
    if cmp.temp_baseline != cmp.temp_current:
        lines.append(f"  温度      {cmp.temp_baseline} → {cmp.temp_current}（变了）")
    if cmp.tools_baseline != cmp.tools_current:
        lines.append(
            f"  工具开关  {cmp.tools_baseline} → {cmp.tools_current}"
            f"（变了，LLM_DISABLED_TOOLS）"
        )
    if not cmp.anything_changed:
        lines.append(
            "  ⏭ 提示词 / 模型 / 温度 / 工具开关都没变 —— 分数变化只可能来自模型侧随机波动"
            "（温度 0 也不保证逐位复现）"
        )

    lines.append(f"  ✅ 新通过   {'、'.join(cmp.new_pass) if cmp.new_pass else '无'}")
    lines.append(f"  ❌ 新失败   {'、'.join(cmp.new_fail) if cmp.new_fail else '无'}")
    lines.append(
        f"  ➖ 持平     {len(cmp.same_pass) + len(cmp.same_fail)} 条"
        f"（{len(cmp.same_pass)} 通过 / {len(cmp.same_fail)} 失败）"
    )

    if cmp.only_baseline or cmp.only_current:
        bits = []
        if cmp.only_baseline:
            bits.append(f"基线里有、这次没有：{_join_ids(cmp.only_baseline)}")
        if cmp.only_current:
            bits.append(f"这次有、基线里没有：{_join_ids(cmp.only_current)}")
        lines.append(f"  ⚠ 用例集不同（只对比了共同有的 {len(cmp.common)} 条）—— " + "；".join(bits))
    return "\n".join(lines)


def format_no_baseline_note(runs_dir: Path) -> str:
    """没有基线时的说明 —— 不是沉默，要告诉学习者「为什么没有」和「怎么才有」。"""
    return (
        f"⏭ 没有对比基线：这次只记录分数，不做对比。\n"
        f"   原因：{runs_dir} 里还没有上一份运行记录 —— 这次就是第一份基线。\n"
        f"   下一步：改完提示词再跑一次，并加 --baseline latest，就能看到「变好还是变坏」"
    )
