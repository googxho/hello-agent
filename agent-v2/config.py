"""
配置层 —— 所有可调参数集中在这里。

═══════════════════════════════════════════════════════════════════════
配置文件放在哪？
═══════════════════════════════════════════════════════════════════════

**仓库根目录的 .env 是全项目共用的** —— 你只需要维护那一份，
不必每个版本目录都复制一遍。

加载优先级（靠前的不会被后面的覆盖）：

    1. 真实环境变量      LLM_MAX_CONTEXT_TOKENS=800 python main.py
    2. 本目录的 .env     （可选）只放这个版本想覆盖的项
    3. 仓库根的 .env     ← 平时只动这一个

这样做的好处：

  · 平时改密钥、换模型，只改根目录那一份
  · 某个版本想临时用别的模型，在它自己目录下放个 .env 写一行就行
  · 一次性实验根本不用改文件：LLM_MODEL=xxx python main.py
"""

from __future__ import annotations

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
load_dotenv(_find_repo_root() / ".env")


def _read_llm_config() -> tuple[str, str, str]:
    """从环境变量读出 (api_key, base_url, model)。

    显式读取、显式传给客户端，而不是依赖 SDK 自动读环境变量 ——
    让你一眼看见密钥和地址的去向。
    """
    api_key = (os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    base_url = (os.getenv("LLM_BASE_URL") or "https://api.deepseek.com").strip()
    model = (os.getenv("LLM_MODEL") or "deepseek-chat").strip()
    return api_key, base_url, model


def _read_int(name: str, default: int) -> int:
    """读一个整数配置。写错了就用默认值，不因为配置手滑就崩掉。"""
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


API_KEY, BASE_URL, MODEL = _read_llm_config()

# ── 循环保护 ──────────────────────────────────────────────────────────
# 单次提问内最多循环几轮，防止 Agent 死循环烧光额度。
MAX_ROUNDS = 8

# ── 上下文管理（v2 的核心，新增）───────────────────────────────────────
# messages 列表的 token 预算。一旦超出就开始「剪枝」。
#
# ⚠️ 故意设得很小（4000），这样你随便聊几句就能看到剪枝在工作。
#    生产环境请改成「模型真实上下文窗口 × 0.7」，比如 64K 的模型设 45000。
MAX_CONTEXT_TOKENS = _read_int("LLM_MAX_CONTEXT_TOKENS", 4000)

# 最近这么多条消息永不剪枝（当前轮和上一轮最相关，动不得）。
KEEP_RECENT_MESSAGES = _read_int("LLM_KEEP_RECENT", 8)

# 单个工具返回值最多保留多少字符 —— 上下文管理的第一道闸。
# 与其事后剪，不如从源头就限流。
MAX_TOOL_RESULT_CHARS = _read_int("LLM_MAX_TOOL_RESULT_CHARS", 2000)

# ── 实验开关：临时「关掉」某些工具 ───────────────────────────────────
# 逗号分隔的工具名。被关掉的工具**不会出现在给模型的 tools 清单里** ——
# 等于模型根本不知道它存在。
#
#     LLM_DISABLED_TOOLS=calculator,get_current_time python main.py
#
# 用途：亲手感受「没有工具时，模型有多弱」。
# （见 EXPERIMENTS.md 的实验 4 和 5 —— 那是最有冲击力的两个实验。）
#
# 平时留空即可。
DISABLED_TOOLS: set[str] = {
    name.strip()
    for name in (os.getenv("LLM_DISABLED_TOOLS") or "").split(",")
    if name.strip()
}

# ── 路径 ─────────────────────────────────────────────────────────────
NOTES_DIR = Path(__file__).parent / "notes"

# ── 系统提示词 ───────────────────────────────────────────────────────
SYSTEM_PROMPT = """你是一个可以通过调用工具来完成任务的助手。

工作规则：
1. 需要计算、需要知道当前时间、需要保存内容时，必须调用对应工具，
   不要凭记忆或心算作答。
2. 不需要工具时就直接回答，不要为了显得勤快而乱调工具。
3. 工具返回结果后，用自然语言把你的结论清楚地告诉用户。
4. 如果工具返回了错误信息，请读懂错误并修正参数后重试。
"""


# ══════════════════════════════════════════════════════════════════════
# 配置清单 —— 「这个版本到底读哪些环境变量」
# ══════════════════════════════════════════════════════════════════════
#
# 这张表有两个用途：
#   1. 给 `python main.py --config` 当数据源
#   2. 给未来的你（和 v3）当文档 —— 一眼看出哪些是共享的、哪些是 v2 独有的
#
# ⚠️ 以后新增配置项，请一并在这里登记，否则 --config 就看不到它了。

CONFIG_KEYS: list[tuple[str, str, str]] = [
    # (环境变量名, 当前生效值, 归属)
    ("LLM_API_KEY", "(已设置)" if API_KEY else "(未设置 ✗)", "共享"),
    ("LLM_BASE_URL", BASE_URL, "共享"),
    ("LLM_MODEL", MODEL, "共享"),
    ("LLM_MAX_CONTEXT_TOKENS", str(MAX_CONTEXT_TOKENS), "v2 独有"),
    ("LLM_KEEP_RECENT", str(KEEP_RECENT_MESSAGES), "v2 独有"),
    ("LLM_MAX_TOOL_RESULT_CHARS", str(MAX_TOOL_RESULT_CHARS), "v2 独有"),
    ("LLM_DISABLED_TOOLS", ", ".join(sorted(DISABLED_TOOLS)) or "(无，全部启用)", "实验开关"),
]


def describe_config() -> str:
    """给 `--config` 用：配置从哪来、当前是什么、谁在用。"""
    lines = ["配置文件（优先级从低到高）："]

    for path, note in (
        (_ROOT / ".env", "共享层 · 全项目一份，放密钥等公共配置"),
        (_HERE / ".env", "本版本层 · 可选，放 v2 独有的调参"),
    ):
        exists = path.is_file()
        mark = "✓" if exists else "✗"
        suffix = "" if exists else "   （不存在，这是正常的）"
        lines.append(f"  {mark}  {path}{suffix}")
        lines.append(f"       {note}")

    lines += ["", "本版本读取的配置项：", ""]
    for name, value, owner in CONFIG_KEYS:
        lines.append(f"  {name:<26} {value:<24} [{owner}]")

    lines += [
        "",
        "（真实环境变量优先级最高，可以临时覆盖任何一项：）",
        "    LLM_MAX_CONTEXT_TOKENS=800 python main.py",
    ]
    return "\n".join(lines)
