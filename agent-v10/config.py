"""
配置层 —— v10 的所有可调参数，集中在这里。

═══════════════════════════════════════════════════════════════════════
本版说明：v10 = v9 的评估底盘 + 规划与任务拆解
═══════════════════════════════════════════════════════════════════════

v10 只加一个知识块：**规划与任务拆解**（Plan-and-Execute）。

底盘不动：依然是最小 Agent + 评估回路（v9 的全部能力还在 ——
量新东西，靠的就是那把旧尺子）。依然**不带** v2~v8 的
上下文预算 / 压缩 / 持久化 / 检索 / 结构化输出 ——「被测对象越简单，
结论越干净」这条老规矩继续生效：

    这一版要量的问题是「先出计划再执行，到底值不值」——
    它和压缩、检索这些变量混在一起，就说不清是谁的功劳了。

新参数都在下面的「规划」一节（LLM_PLANNING_ENABLED 总开关、
LLM_PLANNER_MODEL 规划模型、LLM_PLAN_SABOTAGE 实验开关……）。

═══════════════════════════════════════════════════════════════════════
配置文件放在哪？
═══════════════════════════════════════════════════════════════════════

**仓库根目录的 .env 是全项目共用的** —— 你只需要维护那一份。
加载优先级（靠前的不会被后面的覆盖）：

    1. 真实环境变量      LLM_EVAL_TEMPERATURE=0.3 python main.py --eval
    2. 本目录的 .env     （可选）只放这个版本想覆盖的项
    3. 仓库根的 .env     ← 平时只动这一个
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv

_HERE = Path(__file__).resolve().parent


def _find_repo_root() -> Path:
    """从本文件所在目录往上找，直到找到含 .git 的仓库根。"""
    for candidate in _HERE.parents:
        if (candidate / ".git").exists():
            return candidate
    return _HERE.parent  # 找不到就退回上一级，尽力而为


_ROOT = _find_repo_root()

# 两次都用默认的 override=False，这一条很关键：
#   · 真实环境变量永远不会被文件覆盖
#   · 仓库根的 .env 也不会覆盖本目录的 .env
load_dotenv(_HERE / ".env")
load_dotenv(_ROOT / ".env")


# ══════════════════════════════════════════════════════════════════════
# 读取工具（和 v2~v8 同一套写法：配置写错就用默认值，不因为手滑就崩）
# ══════════════════════════════════════════════════════════════════════


def _read_str(name: str, default: str) -> str:
    return (os.getenv(name) or "").strip() or default


def _read_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _read_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# ══════════════════════════════════════════════════════════════════════
# LLM 连接（共享层）
# ══════════════════════════════════════════════════════════════════════


def _read_llm_config() -> tuple[str, str, str]:
    api_key = (os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    base_url = _read_str("LLM_BASE_URL", "https://api.deepseek.com")
    model = _read_str("LLM_MODEL", "deepseek-chat")
    return api_key, base_url, model


API_KEY, BASE_URL, MODEL = _read_llm_config()

# ── 循环保护（v1 同款保险丝）────────────────────────────────────────
# 单次提问内最多循环几轮，防止 Agent 死循环烧光额度。
# 用尽会撤掉工具再问最后一次，强制它用已有资料作答。
MAX_ROUNDS = 8

# ── 采样温度 ─────────────────────────────────────────────────────────
# 对话用 0.3（v6 起的口径：Agent 要的是稳定，不是文采）。
TEMPERATURE = _read_float("LLM_TEMPERATURE", 0.3)

# ⭐ 评估专用温度：默认 0。
#
# 为什么评估要单独一个温度？
#   评估是一把**尺子** —— 尺子自己必须尽量不晃。
#   对话可以有点随机性（像人），但打分必须尽量可复现，
#   否则「到底是我改的 prompt 起作用了，还是这次抽签抽得好？」
#   永远说不清。
#
# ⚠️ 注意：温度 0 也只是「尽量」确定 —— 服务端仍然有浮点误差、
#    版本更新等噪声。所以结论要看**多次运行的趋势**，别把单次
#    0.1 的抖动当圣旨（见 EXPERIMENTS.md 实验 7）。
EVAL_TEMPERATURE = _read_float("LLM_EVAL_TEMPERATURE", 0.0)

# 裁判模型（LLM-as-judge）用哪个。默认跟随主模型。
# 单开一个口子是为了实验：换一个小模型当裁判，看看判定会不会变。
JUDGE_MODEL = _read_str("LLM_JUDGE_MODEL", MODEL)

# 裁判回复的 token 上限。
#
# ⚠️ 这个值不是拍脑袋定的 —— 实测（2026-09-19，deepseek-flash）：
#    设 200 时，模型把 200 个 token 全花在了内部思考上，可见内容一个字都没写出来
#    （finish_reason=length，content 为空）→ 解析失败 → 一个「假红」。
#    调到 600 之后稳定输出。
#
# 这条教训值得记住：**裁判也是一条会坏的链路** —— 它的参数要像主链路一样被照料，
# 而且失败时必须和「模型答得不好」区分开（见 evaluator.run_judge）。
JUDGE_MAX_TOKENS = _read_int("LLM_JUDGE_MAX_TOKENS", 600)


# ══════════════════════════════════════════════════════════════════════
# 系统提示词 —— 从 v9 起它是一个**文件**，不是代码里的常量
# ══════════════════════════════════════════════════════════════════════
#
# 为什么改成文件？
#   这一版的正题是「改了 prompt，不知道是变好还是变坏」。
#   要让「改 prompt」变成一件随手可做的事（改文件、跑评估、看分数），
#   而不是「打开 Python 源码改字符串」。
#
#   再配一个 --prompt 参数，你就能拿两份提示词 A/B 对比了：
#
#       python main.py --eval                                    # 默认提示词
#       python main.py --eval --prompt prompts/system-degraded.md  # 故意写坏的
#
SYSTEM_PROMPT_FILE = Path(
    _read_str("LLM_SYSTEM_PROMPT_FILE", str(_HERE / "prompts" / "system.md"))
)

# 内置备用提示词：只有文件读不到时才用，并且会明确告诉你「用的是备用」
DEFAULT_SYSTEM_PROMPT = """你是一个可以通过调用工具来完成任务的助手。

工作规则：
1. 需要计算、需要知道当前时间、需要保存内容时，必须调用对应工具，
   不要凭记忆或心算作答。
2. 不需要工具时就直接回答，不要为了显得勤快而乱调工具。
3. 工具返回结果后，用自然语言把你的结论清楚地告诉用户。
4. 如果工具返回了错误信息，请读懂错误并修正参数后重试。
"""


def load_system_prompt(path: Path | None = None) -> tuple[str, str, str]:
    """读提示词文件，返回 (正文, 显示用的来源说明, sha8 指纹)。

    指纹的用途：对比两次评估时并排显示「a1b2c3d4 → 77b0e2f1（变了）」，
    让「分数变化是不是提示词带来的」一眼可查。

    文件读不到就退回内置版本 —— 但必须在说明里讲清楚，
    绝不假装读到了（铁律 7：没做这个动作，要说清楚为什么）。
    """
    target = Path(path) if path else SYSTEM_PROMPT_FILE
    if target.is_file():
        text = target.read_text(encoding="utf-8").strip()
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        # 显示相对路径更友好；不在本目录下的就显示全路径
        try:
            shown = str(target.relative_to(_HERE))
        except ValueError:
            shown = str(target)
        return text, f"{shown}（sha {digest}）", digest

    fallback = DEFAULT_SYSTEM_PROMPT
    digest = hashlib.sha256(fallback.encode("utf-8")).hexdigest()[:8]
    label = (
        f"⚠ 没找到 {target} —— 已退回内置备用提示词"
        f"（sha {digest}；它比默认提示词短，行为可能不同）"
    )
    return fallback, label, digest


# ══════════════════════════════════════════════════════════════════════
# 评估相关（v9 核心）
# ══════════════════════════════════════════════════════════════════════

# 用例集文件：一行一个用例（JSONL 格式）。
#
# 你自己的用例也可以放别的地方：
#     LLM_EVAL_CASES=/tmp/我的用例.jsonl python main.py --eval
EVAL_CASES_FILE = Path(_read_str("LLM_EVAL_CASES", str(_HERE / "evals" / "cases.jsonl")))

# 每次评估的运行记录往哪写（一场评估一个 JSON 文件）。
# 下一节要讲「对比」时，比的就是这些文件。
EVAL_RUNS_DIR = Path(_read_str("LLM_EVAL_RUNS_DIR", str(_HERE / "runs")))

# ══════════════════════════════════════════════════════════════════════
# 规划（v10 的核心）—— Plan-and-Execute 的旋钮
# ══════════════════════════════════════════════════════════════════════
#
# 规划要做的事：动手之前，先让模型交一份「步骤清单」，写进上下文。
# 这样模型执行到每一步时都能看到全局 —— 而不是走一步看一步。

# 总开关。关掉 = 退回 v9 的「走一步看一步」（评估里当对照组用）：
#     LLM_PLANNING_ENABLED=0 python main.py --eval --tag no-plan
PLANNING_ENABLED = _read_str("LLM_PLANNING_ENABLED", "1").lower() not in {
    "0", "false", "no", "off",
}

# 计划最多拆几步。上限不是防模型，是防「过度规划」——
# 步骤越细，执行时越容易被现实推翻（概念课的口诀：粗计划、细执行）。
# 超上限的计划会被判成坏计划（重试，再不行降级为无规划执行）。
PLAN_MAX_STEPS = _read_int("LLM_PLAN_MAX_STEPS", 8)

# 规划最多尝试几次：第一次没交好计划（比如没调 submit_plan）再给一次机会，
# 还是不行就降级（见 agent._make_plan 的降级路径：规划失败不连累对话）。
PLAN_MAX_ATTEMPTS = _read_int("LLM_PLAN_MAX_ATTEMPTS", 2)

# 规划用哪个模型。默认跟随主模型；单开一个口子的两个用途：
#   · 实验：把它写成不存在的模型名（如 no-such-model）→ 观察「规划失败」的样子
#   · 现实：规划可以换更便宜的模型（它只产出一段清单，不干重活）
PLANNER_MODEL = _read_str("LLM_PLANNER_MODEL", MODEL)

# 规划提示词文件（v9 把系统提示词文件化，规划提示词同款待遇）：
# 想让规划器换个拆法，改文件、重跑评估、看分数 —— 不用动代码。
PLANNER_PROMPT_FILE = Path(
    _read_str("LLM_PLANNER_PROMPT_FILE", str(_HERE / "prompts" / "planner.md"))
)

# ⚠️ 实验开关：人为把计划写坏（ROADMAP 要求的「反例验证」）。
# 目前支持两种坏法：
#     LLM_PLAN_SABOTAGE=drop_last   砍掉计划的最后一步（模型不会知道）
#     LLM_PLAN_SABOTAGE=reverse     颠倒计划顺序（测「计划 vs 任务原文谁说了算」）
# 回答的问题：「计划质量真的决定执行质量吗？弄坏计划，分数会不会掉？」
# （说明只会出现在日志里 —— 模型不该知道自己被做了实验。）
PLAN_SABOTAGE = _read_str("LLM_PLAN_SABOTAGE", "")

# 内置备用规划提示词：只有文件读不到时才用，并且会明确告诉你
DEFAULT_PLANNER_PROMPT = """你是任务规划器。把【本次任务】拆成按顺序执行的步骤清单，
调用 submit_plan 提交。

要求：
- 只交计划，不执行任务；
- 步骤覆盖任务的全部要求（数字、文件名、格式都要写进步骤里）；
- 每步一句话，尽量对应一个工具调用；最多 {max_steps} 步。

可用工具：
{tools}
"""


def load_planner_prompt(path: Path | None = None) -> tuple[str, str, str]:
    """读规划提示词文件，返回 (正文, 显示用的来源说明, sha8 指纹)。

    指纹要进评估 meta —— 对比面板得跟踪「规划器提示词变没变」。
    （v9 实验 4 的老教训：面板不许给假因果。规划提示词不是「波动」。）
    """
    target = Path(path) if path else PLANNER_PROMPT_FILE
    if target.is_file():
        text = target.read_text(encoding="utf-8").strip()
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        try:
            shown = str(target.relative_to(_HERE))
        except ValueError:
            shown = str(target)
        return text, f"{shown}（sha {digest}）", digest

    fallback = DEFAULT_PLANNER_PROMPT
    digest = hashlib.sha256(fallback.encode("utf-8")).hexdigest()[:8]
    label = f"⚠ 没找到 {target} —— 已退回内置备用规划提示词（sha {digest}）"
    return fallback, label, digest


# ══════════════════════════════════════════════════════════════════════
# 工具与资料目录
# ══════════════════════════════════════════════════════════════════════

# save_note 写文件的目标目录（v1 同款）。
NOTES_DIR = Path(_read_str("LLM_NOTES_DIR", str(_HERE / "notes")))

# ── 实验开关：临时「关掉」某些工具 ───────────────────────────────────
# 逗号分隔的工具名。被关掉的工具**不会出现在给模型的 tools 清单里** ——
# 等于模型根本不知道它存在。
#
#     LLM_DISABLED_TOOLS=calculator python main.py --eval --baseline latest
#
# v9 起用它做一件新事情：**给评估制造「能力的损失」** ——
# 看分数是不是真的会掉。不会掉，就说明评估没测到东西。
DISABLED_TOOLS: set[str] = {
    name.strip()
    for name in (os.getenv("LLM_DISABLED_TOOLS") or "").split(",")
    if name.strip()
}


# ══════════════════════════════════════════════════════════════════════
# 配置清单 —— 「这个版本到底读哪些环境变量」
# ══════════════════════════════════════════════════════════════════════

CONFIG_KEYS: list[tuple[str, str, str]] = [
    # (环境变量名, 当前生效值, 归属)
    ("LLM_API_KEY", "(已设置)" if API_KEY else "(未设置 ✗)", "共享"),
    ("LLM_BASE_URL", BASE_URL, "共享"),
    ("LLM_MODEL", MODEL, "共享"),
    ("LLM_TEMPERATURE", str(TEMPERATURE), "v9 · 对话温度（沿用 v6 口径）"),
    ("LLM_EVAL_TEMPERATURE", str(EVAL_TEMPERATURE), "v9 新增强 · 评估温度"),
    ("LLM_JUDGE_MODEL", JUDGE_MODEL, "v9 新增"),
    ("LLM_JUDGE_MAX_TOKENS", str(JUDGE_MAX_TOKENS), "v9 新增"),
    ("LLM_SYSTEM_PROMPT_FILE", str(SYSTEM_PROMPT_FILE), "v9 新增（提示词文件化）"),
    ("LLM_EVAL_CASES", str(EVAL_CASES_FILE), "v9 新增"),
    ("LLM_EVAL_RUNS_DIR", str(EVAL_RUNS_DIR), "v9 新增"),
    ("LLM_NOTES_DIR", str(NOTES_DIR), "v1 沿用"),
    ("LLM_DISABLED_TOOLS", ", ".join(sorted(DISABLED_TOOLS)) or "(无，全部启用)", "实验开关"),
    ("LLM_PLANNING_ENABLED", "开" if PLANNING_ENABLED else "关（对照组模式）", "v10 新增 · 规划总开关"),
    ("LLM_PLAN_MAX_STEPS", str(PLAN_MAX_STEPS), "v10 新增"),
    ("LLM_PLAN_MAX_ATTEMPTS", str(PLAN_MAX_ATTEMPTS), "v10 新增"),
    ("LLM_PLANNER_MODEL", PLANNER_MODEL, "v10 新增"),
    ("LLM_PLANNER_PROMPT_FILE", str(PLANNER_PROMPT_FILE), "v10 新增"),
    ("LLM_PLAN_SABOTAGE", PLAN_SABOTAGE or "(无 —— 不做实验时保持空)", "实验开关"),
]


def describe_config() -> str:
    """给 `--config` 用：配置从哪来、当前是什么、谁在用。"""
    lines = ["配置文件（优先级从低到高）："]

    for path, note in (
        (_ROOT / ".env", "共享层 · 全项目一份，放密钥等公共配置"),
        (_HERE / ".env", "本版本层 · 可选，放这个版本独有的调参"),
    ):
        exists = path.is_file()
        mark = "✓" if exists else "✗"
        suffix = "" if exists else "   （不存在，这是正常的）"
        lines.append(f"  {mark}  {path}{suffix}")
        lines.append(f"       {note}")

    lines += ["", "本版本读取的配置项：", ""]
    for name, value, owner in CONFIG_KEYS:
        lines.append(f"  {name:<26} {value:<30} [{owner}]")

    lines += [
        "",
        "（真实环境变量优先级最高，可以临时覆盖任何一项：）",
        "    LLM_PLANNING_ENABLED=0 python main.py --eval   （对照：退回走一步看一步）",
        "    LLM_PLAN_SABOTAGE=drop_last python main.py      （实验：砍掉计划最后一步）",
        "    LLM_PLANNER_MODEL=no-such-model python main.py  （实验：让规划失败）",
        "    LLM_EVAL_TEMPERATURE=0.3 python main.py --eval",
        "    LLM_DISABLED_TOOLS=calculator python main.py --eval",
        "    LLM_SYSTEM_PROMPT_FILE=prompts/system-degraded.md python main.py",
        "",
        "注意：v10 不带 v2~v8 的记忆 / 压缩 / 检索 / 结构化输出 ——",
        "它们的老参数（LLM_MAX_CONTEXT_TOKENS、LLM_PERSIST_ENABLED…）这一版不读。",
    ]
    return "\n".join(lines)
