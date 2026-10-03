"""
配置层 —— v12（LangGraph 编排层）的配置。

═══════════════════════════════════════════════════════════════════════
v12 是什么？
═══════════════════════════════════════════════════════════════════════

v11 装了「零件」（LangChain：消息/工具/模型/LCEL）。
v12 装「流程的控制权」（LangGraph：状态图 + 检查点 + 人机协同）。

它要治的是 v10 留下的两个真实伤口（都在 v10 的实验里亲眼见过）：

    ① 「断在第 5 步就永远断了」——8 步挑战被 DSML 中断，
       「8 步：4 完成 / 4 未标记」就停在磁盘上，没有任何恢复手段；
    ② 「危险操作没有把关」——模型想写文件就写文件，
       没有任何一层能说「等一下，先让我确认」。

v12 的答案：把 Agent 的流程画成**显式的状态图**——
每一步的状态都落盘（检查点），关键动作前可以停下来等人（interrupt），
步骤走不下去还能触发重规划（条件边）。

═══════════════════════════════════════════════════════════════════════
本版独有的实验开关（都是「演示用」，配合 EXPERIMENTS.md 使用）
═══════════════════════════════════════════════════════════════════════

    --pause-at N      第 N 步完成后模拟进程猝死（os._exit，等价 kill -9）
                      ——用来演示「检查点恢复」：resume 能从断点继续
    --sabotage-step N 第 N 步的所有工具调用强制报错
                      ——用来演示「条件边重规划」：失败会触发 replan
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_HERE = Path(__file__).resolve().parent


def _find_repo_root() -> Path:
    for candidate in _HERE.parents:
        if (candidate / ".git").exists():
            return candidate
    return _HERE.parent


_ROOT = _find_repo_root()
load_dotenv(_HERE / ".env")
load_dotenv(_ROOT / ".env")


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
# LLM 连接（共享层，和 v1~v11 同一套）
# ══════════════════════════════════════════════════════════════════════


def _read_llm_config() -> tuple[str, str, str]:
    api_key = (os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    base_url = _read_str("LLM_BASE_URL", "https://api.deepseek.com")
    model = _read_str("LLM_MODEL", "deepseek-chat")
    return api_key, base_url, model


API_KEY, BASE_URL, MODEL = _read_llm_config()
TEMPERATURE = _read_float("LLM_TEMPERATURE", 0.3)
NOTES_DIR = Path(_read_str("LLM_NOTES_DIR", str(_HERE / "notes")))

# 检查点落盘位置（SqliteSaver）——「断在哪」的证据就在这个文件里
CHECKPOINT_DB = Path(_read_str("LLM_CHECKPOINT_DB", str(_HERE / "checkpoints.sqlite")))


def get_chat_model(temperature: float | None = None):
    """ChatOpenAI 工厂（和 v11 同款，换供应商 = 改 .env）。"""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=MODEL,
        base_url=BASE_URL,
        api_key=API_KEY or "missing-key",
        temperature=TEMPERATURE if temperature is None else temperature,
    )


# ══════════════════════════════════════════════════════════════════════
# 图的参数（本版核心旋钮）
# ══════════════════════════════════════════════════════════════════════

# 图循环的兜底保险丝 = LangGraph 的 recursion_limit
# （对应 v1 手写循环里的 MAX_ROUNDS：防死循环烧钱）
RECURSION_LIMIT = _read_int("LLM_GRAPH_RECURSION_LIMIT", 60)

# 计划最多拆几步
PLAN_MAX_STEPS = _read_int("LLM_PLAN_MAX_STEPS", 6)

# 一个步骤里工具连续报错到这个数 → 判定「本步做不成」→ 触发重规划
STEP_ERROR_LIMIT = _read_int("LLM_STEP_ERROR_LIMIT", 2)

# 重规划最多几次（防「改来改去原地打转」——概念课的「防震荡」）
MAX_REPLANS = _read_int("LLM_MAX_REPLANS", 2)

# ── 运行期开关（CLI 会改这些，节点里读）───────────────────────────
AUTO_APPROVE = False   # True = --yes：跳过人工审批（自动化/评估用）
PAUSE_AT = 0           # N>0 = 第 N 步的 advance 节点里模拟进程猝死
SABOTAGE_STEP = 0      # N>0 = 第 N 步的工具全部强制报错


# ══════════════════════════════════════════════════════════════════════
# 配置清单（--config 用）
# ══════════════════════════════════════════════════════════════════════

CONFIG_KEYS: list[tuple[str, str, str]] = [
    ("LLM_API_KEY", "(已设置)" if API_KEY else "(未设置 ✗)", "共享"),
    ("LLM_BASE_URL", BASE_URL, "共享"),
    ("LLM_MODEL", MODEL, "共享"),
    ("LLM_TEMPERATURE", str(TEMPERATURE), "共享"),
    ("LLM_NOTES_DIR", str(NOTES_DIR), "沿用 v1"),
    ("LLM_CHECKPOINT_DB", str(CHECKPOINT_DB), "v12 新增 · 检查点落盘位置"),
    ("LLM_GRAPH_RECURSION_LIMIT", str(RECURSION_LIMIT), "v12 新增 · 图的保险丝"),
    ("LLM_PLAN_MAX_STEPS", str(PLAN_MAX_STEPS), "v12 新增"),
    ("LLM_STEP_ERROR_LIMIT", str(STEP_ERROR_LIMIT), "v12 新增 · 失败判定阈值"),
    ("LLM_MAX_REPLANS", str(MAX_REPLANS), "v12 新增 · 防震荡"),
]


def describe_config() -> str:
    lines = ["配置文件（优先级从低到高）："]
    for path, note in (
        (_ROOT / ".env", "共享层 · 全项目一份"),
        (_HERE / ".env", "本版本层 · 可选"),
    ):
        exists = path.is_file()
        mark = "✓" if exists else "✗"
        suffix = "" if exists else "   （不存在，这是正常的）"
        lines.append(f"  {mark}  {path}{suffix}")
        lines.append(f"       {note}")

    lines += ["", "本版本读取的配置项：", ""]
    for name, value, owner in CONFIG_KEYS:
        lines.append(f"  {name:<26} {value:<40} [{owner}]")

    lines += [
        "",
        "（真实环境变量优先级最高；演示开关走命令行参数：）",
        "    python main.py --run \"任务\" --pause-at 2        # 模拟猝死（演示恢复）",
        "    python main.py --run \"任务\" --sabotage-step 2   # 制造失败（演示重规划）",
        "    python main.py --resume <thread_id>            # 从断点继续",
    ]
    return "\n".join(lines)
