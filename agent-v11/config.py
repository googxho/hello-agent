"""
配置层 —— v11（LangChain 组件层）的配置 + 两个工厂。

═══════════════════════════════════════════════════════════════════════
v11 是什么？（先交代全图景）
═══════════════════════════════════════════════════════════════════════

v1 ~ v10：我们**手写**了一个 Agent 的每一层 ——
工具协议、消息拼装、模型调用、token 账本、结构化输出、评估器……
每一层都能跑，每一层也都带着「又造了一遍轮子」的味道。

v11 是「框架化」的上半场：**LangChain = 组件层** ——
把「和模型打交道」这件事标准化：

    消息 → Message 类（不再是 dict）
    模型 → ChatModel 协议（invoke / batch / stream 统一）
    工具 → @tool（docstring + 签名 → 自动推导 schema）
    组装 → LCEL `|` 管道（一切皆 Runnable）
    Agent → create_agent（一行得到 ReAct 循环）

v12 是下半场：**LangGraph = 编排层** —— 状态机、检查点、人机协同。
（你可能已经听说：LangChain 的 agent 门面，内核就是 LangGraph 的图 ——
 这会是 v11 最后一个 demo 要亲眼验证的事。）

═══════════════════════════════════════════════════════════════════════
配置文件放在哪？（和 v2~v10 同一套规矩）
═══════════════════════════════════════════════════════════════════════

**仓库根目录的 .env 是全项目共用的** —— 你只需要维护那一份。
加载优先级（靠前的不会被后面的覆盖）：

    1. 真实环境变量       LLM_TEMPERATURE=0 python main.py --demo lcel
    2. 本目录的 .env      （可选）只放这个版本想覆盖的项 —— v11 目前没有独有项
    3. 仓库根的 .env      ← 平时只动这一个

v11 **没有版本独有参数**（所以也没有 .env.example）——
组件层要演示的东西，全部跟着共享层走。
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
    return _HERE.parent


_ROOT = _find_repo_root()

# 两次都用默认 override=False（和 v2~v10 一样的优先级设计）
load_dotenv(_HERE / ".env")
load_dotenv(_ROOT / ".env")


# ══════════════════════════════════════════════════════════════════════
# 读取工具（沿用 v2 起的习惯：配置写错就用默认值，不因为手滑就崩）
# ══════════════════════════════════════════════════════════════════════


def _read_str(name: str, default: str) -> str:
    return (os.getenv(name) or "").strip() or default


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

TEMPERATURE = _read_float("LLM_TEMPERATURE", 0.3)

# save_note 写文件的目标目录（和其他版本同款；运行时才创建）
NOTES_DIR = Path(_read_str("LLM_NOTES_DIR", str(_HERE / "notes")))


# ══════════════════════════════════════════════════════════════════════
# 工厂一：ChatModel
# ══════════════════════════════════════════════════════════════════════


def get_chat_model(temperature: float | None = None):
    """造一个 LangChain 的 ChatOpenAI —— 这就是「模型协议」的入口。

    面试点：ChatOpenAI 不是「OpenAI 专用」—— 只要你走 OpenAI 兼容接口
    （DeepSeek / 通义 / 月之暗面 / 本地 Ollama 全兼容），换个 base_url
    和 model 名就是换一家供应商。**统一接口，是组件层的第一价值。**
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=MODEL,
        base_url=BASE_URL,
        api_key=API_KEY or "missing-key",
        temperature=TEMPERATURE if temperature is None else temperature,
    )


# ══════════════════════════════════════════════════════════════════════
# 配置清单（--config 用）
# ══════════════════════════════════════════════════════════════════════

CONFIG_KEYS: list[tuple[str, str, str]] = [
    ("LLM_API_KEY", "(已设置)" if API_KEY else "(未设置 ✗)", "共享"),
    ("LLM_BASE_URL", BASE_URL, "共享"),
    ("LLM_MODEL", MODEL, "共享"),
    ("LLM_TEMPERATURE", str(TEMPERATURE), "共享"),
    ("LLM_NOTES_DIR", str(NOTES_DIR), "沿用 v1"),
]


def describe_config() -> str:
    """给 `--config` 用：配置从哪来、当前是什么、谁在用。"""
    lines = ["配置文件（优先级从低到高）："]
    for path, note in (
        (_ROOT / ".env", "共享层 · 全项目一份，放密钥等公共配置"),
        (_HERE / ".env", "本版本层 · v11 没有独有参数（存在也只会覆盖上面几项）"),
    ):
        exists = path.is_file()
        mark = "✓" if exists else "✗"
        suffix = "" if exists else "   （不存在，这是正常的）"
        lines.append(f"  {mark}  {path}{suffix}")
        lines.append(f"       {note}")

    lines += ["", "本版本读取的配置项：", ""]
    for name, value, owner in CONFIG_KEYS:
        lines.append(f"  {name:<24} {value:<36} [{owner}]")

    lines += [
        "",
        "（真实环境变量优先级最高，可以临时覆盖任何一项：）",
        "    LLM_TEMPERATURE=0 python main.py --demo lcel",
        "",
        "注意：v11 是「组件层演示」——不带 v2~v10 的记忆 / 压缩 / 评估，",
        "也没有版本独有参数（连 .env.example 都不需要）。",
    ]
    return "\n".join(lines)
