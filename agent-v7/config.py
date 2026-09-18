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


def _read_bool(name: str, default: bool) -> bool:
    """读一个开关。1/true/yes/on 为真，0/false/no/off 为假，认不出来就用默认值。"""
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on", "y"}:
        return True
    if raw in {"0", "false", "no", "off", "n"}:
        return False
    return default


def _read_float(name: str, default: float) -> float:
    """读一个小数配置。写错了就用默认值。"""
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


API_KEY, BASE_URL, MODEL = _read_llm_config()

# ── 循环保护 ──────────────────────────────────────────────────────────
# 单次提问内最多循环几轮，防止 Agent 死循环烧光额度。
MAX_ROUNDS = 8

# ── 上下文管理（v2 引入）──────────────────────────────────────────────
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

# ── 上下文压缩（v3 的核心，新增）─────────────────────────────────────
# 超预算、且光剪工具结果救不回来时，v2 的做法是「直接丢弃」旧对话；
# v3 改成「先让 LLM 把它概括成一条摘要，再用摘要换掉原文」。
#
# 一句话区别：
#
#     v2 丢弃：删掉 → 信息量归零
#     v3 压缩：概括 → 用 5% 的 token 保住 80% 的关键信息
#
# 关掉它就能亲手对比两种行为：
#
#     LLM_SUMMARY_ENABLED=0 python main.py
#
SUMMARY_ENABLED = _read_bool("LLM_SUMMARY_ENABLED", True)

# 压缩用哪个模型。默认跟随 LLM_MODEL。
#
# 为什么单开一个口子？因为「压缩」和「干活」是两类完全不同的任务：
#
#   · 干活（调用工具、完成任务）需要最强的模型
#   · 压缩（把一段话读短）本质上是个整理归纳的活儿，便宜的小模型往往就够
#
# 真实产品里这常常是省钱的第一个开关。顺便它还是个绝佳的实验入口 ——
# 填一个不存在的模型名，就能安全地观察「压缩失败」长什么样（见 EXPERIMENTS.md 实验 7）。
SUMMARY_MODEL = (os.getenv("LLM_SUMMARY_MODEL") or "").strip() or MODEL

# 摘要最多输出多少 token。
#
# 这是「压缩比 vs 保真度」的旋钮：
#   · 太小 → 摘要短得装不下关键信息，压了等于没压
#   · 太大 → 摘要不再是摘要，省不下 token，压缩就白做了
#
# 500 大致够写下「目标 + 关键信息 + 结论 + 待办」四类内容。
SUMMARY_MAX_TOKENS = _read_int("LLM_SUMMARY_MAX_TOKENS", 500)

# 可压缩内容少于这个 token 数时，懒得压，直接丢。
#
# 为什么？算一笔账：压缩一次的固定开销大致是
#     「压缩指令（约 250 tokens）+ 摘要输出（最多 500 tokens）」
# 而被压缩的内容本身如果只有一两百 token，那就不是省钱，是倒贴。
#
# ⚠️ 但把它调得太大，你会在预算很小时看到 v3 退回 v2 的失忆行为 ——
#    因为「永远凑不够要压的量」。
#
#    好在这种事不用猜：没压的话，日志里会多出一行
#    「⏭ 没有走压缩：可压内容只有 X tokens，低于阈值 Y」，
#    直接把原因和数字摆出来。
MIN_COMPACT_TOKENS = _read_int("LLM_MIN_COMPACT_TOKENS", 200)
# ── 持久化记忆（v4 的核心，新增）─────────────────────────────────────
# v3 解决了「聊久了会忘」，但状态全在内存里 —— 关掉终端就蒸发。
# v4 把会话（消息 + 摘要）写进磁盘，启动时读回来。
#
# 关掉它就能亲手对比两种行为（重启后还记得 vs 又变成陌生人）：
#
#     LLM_PERSIST_ENABLED=0 python main.py
#
PERSIST_ENABLED = _read_bool("LLM_PERSIST_ENABLED", True)

# 会话存到哪个文件。
#
# 默认就放在版本目录下，**一场会话一个文件** —— 这是能做到最小的方案。
# 想同时保留多场会话？把 session.json 复制一份就是一次「存档」，
# 想切回去时用 LLM_SESSION_FILE 指过去：
#
#     LLM_SESSION_FILE=~/我的存档.json python main.py
#
# （会话目录 / 列表 / 切换是下一层的事，这一版先不做 —— 见 README 的「动手练习」。）
SESSION_FILE = Path((os.getenv("LLM_SESSION_FILE") or "").strip() or (_HERE / "session.json"))
# ── 实验开关：临时「关掉」某些工具 ───────────────────────────────────
# 逗号分隔的工具名。被关掉的工具**不会出现在给模型的 tools 清单里** ——
# 等于模型根本不知道它存在。
#
#     LLM_DISABLED_TOOLS=calculator,get_current_time python main.py
#
# 用途：亲手感受「没有工具时，模型有多弱」。
# （见 EXPERIMENTS.md 的实验 —— 那是最有冲击力的两个实验。）
#
# 平时留空即可。
DISABLED_TOOLS: set[str] = {
    name.strip()
    for name in (os.getenv("LLM_DISABLED_TOOLS") or "").split(",")
    if name.strip()
}

# ── 路径 ─────────────────────────────────────────────────────────────
# 资料目录 —— 它要读的「图书馆」在哪（v5 起可配置）。
#
# 默认就是本版本的 notes/（save_note 写进去的那个目录）。
# 想让它读你自己的文档？指过去就行：
#
#     LLM_NOTES_DIR=~/我的文档 python main.py
NOTES_DIR = Path((os.getenv("LLM_NOTES_DIR") or "").strip() or (_HERE / "notes"))

# ── 本地检索（v5 的核心，新增）─────────────────────────────────────
# v4 让 Agent 记住了「你告诉过它的事」，但它对**你写在磁盘上的资料**一无所知。
# v5 给它一把钥匙：先去资料里找，再把找到的塞进上下文。
#
# 关掉它，就退回 v4（它连你有笔记这件事都不知道）：
#
#     LLM_RETRIEVAL_ENABLED=0 python main.py
RETRIEVAL_ENABLED = _read_bool("LLM_RETRIEVAL_ENABLED", True)

# 每次检索带回来几段。
#
# ⚠️ 这是「找得全」和「别把上下文塞满」之间的平衡：
#    太小 → 该找的没找到；太大 → 一堆无关内容挤占预算（v2 的老问题又回来了）
RETRIEVAL_TOP_K = _read_int("LLM_RETRIEVAL_TOP_K", 5)

# 一块最大多少字符（切分粒度）。
#
# 太小 → 一段话被切碎，上下文丢了；太大 → 一块里混着好几件事，检索不准。
# 这是检索系统里最典型的「旋钮」之一 —— 见 EXPERIMENTS.md。
CHUNK_MAX_CHARS = _read_int("LLM_CHUNK_MAX_CHARS", 600)

# ── 向量检索（v6 的核心，新增）─────────────────────────────────────
# v5 的关键词检索**认字不认意思**：笔记写「性能优化」，你问「怎么让它更快」→ 0 命中。
# v6 把文字变成向量（一串数字），意思是相近的，向量也相近 ——
# 「找相关内容」从「字面对不对」变成了「夹角小不小」。
#
# 检索模式：hybrid（v7 默认：两路都跑）/ semantic（v6：只比意思）/ keyword（v5：只比字面）。
# 三种模式随时可切 —— 「并排对照实验」就靠它：
#
#     LLM_RETRIEVAL_MODE=keyword python main.py
#     LLM_RETRIEVAL_MODE=semantic python main.py
#
# ⚠️ 值写错（比如手滑成 hybird）不会崩，但会悄悄退回 hybrid ——
#    发现行为不对劲时，先看 `python main.py --config` 里的这一行。
RETRIEVAL_MODE = (os.getenv("LLM_RETRIEVAL_MODE") or "hybrid").strip().lower()
if RETRIEVAL_MODE not in {"keyword", "semantic", "hybrid"}:
    RETRIEVAL_MODE = "hybrid"

# 向量模型 —— ⚠️ 它和对话模型是**两家服务**，密钥各是各的。
# 都在仓库根的 .env 里（见 .env.example 的说明）。
# 默认值：SiliconFlow 的 BAAI/bge-m3（1024 维，中文效果好）。
EMBEDDING_API_KEY = (os.getenv("EMBEDDING_API_KEY") or "").strip()
EMBEDDING_BASE_URL = (
    os.getenv("EMBEDDING_BASE_URL") or "https://api.siliconflow.cn/v1"
).strip()
EMBEDDING_MODEL = (os.getenv("EMBEDDING_MODEL") or "BAAI/bge-m3").strip()

# 向量缓存文件 —— ⚠️ 这个文件不是可选项，是**省钱的关键**。
#
# embedding 是按 token 计费的：一个几百块的资料库，
# 每次启动都重算一遍 = 每次启动都重新烧钱。
# 缓存让「第一次贵，之后免费」，而且只重算**变过的那几块**。
EMBEDDING_CACHE_FILE = Path(
    (os.getenv("LLM_EMBEDDING_CACHE") or "").strip() or (_HERE / "embedding-cache.json")
)

# 向量检索的「相关性门槛」（余弦相似度）。低于它的一律不算命中。
#
# ⚠️ 这个值**不能拍脑袋定**：不同向量模型的相似度分布完全不同 ——
#    在有些模型里，「相关」和「无关」的内容都落在 0.4 ~ 0.8 之间。
#
#    本项目实测（bge-m3）：
#      「性能优化…」↔「怎么让程序跑得更快」 0.65  ← 相关
#      「性能优化…」↔「番茄需要浇多少水」   0.36  ← 无关
#    所以 0.5 在这里能把它们分开。
#
#    **换模型、换资料，这个值必须重新校准。** 想要「宁滥毋缺」就调低，
#    想要「宁缺毋滥」就调高 —— 没有普适值。
MIN_SIMILARITY = _read_float("LLM_MIN_SIMILARITY", 0.5)

# ── 混合检索（v7 的核心，新增）─────────────────────────────────────
# v6 的实验 6 / 7 摆清了事实：两把尺子各有盲区，而且盲区几乎不重叠 ——
#
#     「怎么让程序跑得更快」  关键词 0 命中 ✗      向量 0.651 ✓
#     「ERR_5002」            关键词稳稳命中 ✓     向量 0.556（门槛一抬就没）✗
#
# 那为什么还要二选一？v7 两个都跑，再用 RRF（倒数排名融合）合并名次。
#
# ⭐ RRF 只看**名次**，不看分数 —— 因为两边的分数量纲完全不同：
#     关键词是 0.2~2.x 的 BM25 打分，向量是 0~1 的余弦。直接相加是灾难。
#     而「第 1 名」是跨检索器的通用语言。
#
#         融合分(d) = Σ 1 / (k + 名次_r(d))
#
#     被一边提名拿一份分，两边都提名拿两份 —— **共识是真金白银**。
#
# k 是「共识的含金量」旋钮（默认 60，Cormack 等 2009 的经验值）：
#     k 大 → 两边都提名 > 单边第一名；k 小 → 单边霸权抬头。
#     （想亲手看它翻转：见 EXPERIMENTS.md 实验 5）
RRF_K = _read_int("LLM_RRF_K", 60)

# 每一路最多提名几段进入融合（depth）。
#
# ⚠️ 名单之外 = 没有名次 = 贡献 0 分。
#     取太小 → 另一路的好结果没机会吃到「共识加成」；
#     取太大 → 低质量的提名会稀释前排。
HYBRID_DEPTH = _read_int("LLM_HYBRID_DEPTH", 10)

# ── 采样温度（v6 修正 —— v1~v5 一直没设过）────────────────────────
# 0 = 每次挑「最可能的下一个词」，1 = 更随机。不设就吃 API 默认值（DeepSeek 是 1.0）。
#
# ⚠️ 为什么到 v6 才补上？因为一直没撞到痛处：
#    v1~v5 的实验里，模型「调不调工具」的波动不太显眼；
#    到了这一版，实测同一句「我想让程序跑得更快」问两次 ——
#    一次它调用了 search_notes，一次它凭自己的知识答了一大篇，
#    实验根本没法复现。
#
# 写文章要的是文采，Agent 要的是**稳定**：同样的问题，应该走同样的路。
TEMPERATURE = _read_float("LLM_TEMPERATURE", 0.3)

# ── 系统提示词 ───────────────────────────────────────────────────────
SYSTEM_PROMPT = """你是一个可以通过调用工具来完成任务的助手。

工作规则：
1. 需要计算、需要知道当前时间、需要保存内容时，必须调用对应工具，
   不要凭记忆或心算作答。
2. **回答之前先想一下：这个问题的答案会不会在用户的笔记库里？**
   只要有可能（哪怕只是「可能」），就先用 search_notes 查一次再回答。
   你不可能记得用户在笔记里写过什么，而查一次很便宜、猜错了很贵。
3. 不需要工具时就直接回答，不要为了显得勤快而乱调工具。
4. 工具返回结果后，用自然语言把你的结论清楚地告诉用户。
5. 如果工具返回了错误信息，请读懂错误并修正参数后重试。
"""


# ══════════════════════════════════════════════════════════════════════
# 配置清单 —— 「这个版本到底读哪些环境变量」
# ══════════════════════════════════════════════════════════════════════
#
# 这张表有两个用途：
#   1. 给 `python main.py --config` 当数据源
#   2. 给未来的你（和 v4）当文档 —— 一眼看出哪些是共享的、哪些是 v3 独有的
#
# ⚠️ 以后新增配置项，请一并在这里登记，否则 --config 就看不到它了。

CONFIG_KEYS: list[tuple[str, str, str]] = [
    # (环境变量名, 当前生效值, 归属)
    ("LLM_API_KEY", "(已设置)" if API_KEY else "(未设置 ✗)", "共享"),
    ("LLM_BASE_URL", BASE_URL, "共享"),
    ("LLM_MODEL", MODEL, "共享"),
    ("LLM_MAX_CONTEXT_TOKENS", str(MAX_CONTEXT_TOKENS), "v2 引入"),
    ("LLM_KEEP_RECENT", str(KEEP_RECENT_MESSAGES), "v2 引入"),
    ("LLM_MAX_TOOL_RESULT_CHARS", str(MAX_TOOL_RESULT_CHARS), "v2 引入"),
    ("LLM_SUMMARY_ENABLED", "开" if SUMMARY_ENABLED else "关（退回 v2 的丢弃）", "v3 新增"),
    ("LLM_SUMMARY_MODEL", SUMMARY_MODEL, "v3 新增"),
    ("LLM_SUMMARY_MAX_TOKENS", str(SUMMARY_MAX_TOKENS), "v3 新增"),
    ("LLM_MIN_COMPACT_TOKENS", str(MIN_COMPACT_TOKENS), "v3 引入"),
    ("LLM_PERSIST_ENABLED", "开" if PERSIST_ENABLED else "关（退回 v3：重启就忘）", "v4 引入"),
    ("LLM_SESSION_FILE", str(SESSION_FILE), "v4 引入"),
    ("LLM_NOTES_DIR", str(NOTES_DIR), "v5 新增"),
    ("LLM_RETRIEVAL_ENABLED", "开" if RETRIEVAL_ENABLED else "关（退回 v4）", "v5 新增"),
    ("LLM_RETRIEVAL_TOP_K", str(RETRIEVAL_TOP_K), "v5 引入"),
    ("LLM_CHUNK_MAX_CHARS", str(CHUNK_MAX_CHARS), "v5 引入"),
    ("LLM_RETRIEVAL_MODE", RETRIEVAL_MODE, "v6 新增 · v7 换默认"),
    ("EMBEDDING_API_KEY", "(已设置)" if EMBEDDING_API_KEY else "(未设置 ✗)", "共享"),
    ("EMBEDDING_BASE_URL", EMBEDDING_BASE_URL, "共享"),
    ("EMBEDDING_MODEL", EMBEDDING_MODEL, "共享"),
    ("LLM_EMBEDDING_CACHE", str(EMBEDDING_CACHE_FILE), "v6 新增"),
    ("LLM_MIN_SIMILARITY", str(MIN_SIMILARITY), "v6 新增"),
    ("LLM_RRF_K", str(RRF_K), "v7 新增"),
    ("LLM_HYBRID_DEPTH", str(HYBRID_DEPTH), "v7 新增"),
    ("LLM_TEMPERATURE", str(TEMPERATURE), "v6 修正（v1~v5 未设置）"),
    ("LLM_DISABLED_TOOLS", ", ".join(sorted(DISABLED_TOOLS)) or "(无，全部启用)", "实验开关"),
]


def describe_config() -> str:
    """给 `--config` 用：配置从哪来、当前是什么、谁在用。"""
    lines = ["配置文件（优先级从低到高）："]

    for path, note in (
        (_ROOT / ".env", "共享层 · 全项目一份，放密钥等公共配置"),
        (_HERE / ".env", "本版本层 · 可选，放 v3 独有的调参"),
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
        "    LLM_SUMMARY_ENABLED=0 python main.py",
        "    LLM_PERSIST_ENABLED=0 python main.py",
        "    LLM_SESSION_FILE=/tmp/我的存档.json python main.py",
        "    LLM_RETRIEVAL_ENABLED=0 python main.py",
        "    LLM_NOTES_DIR=~/我的文档 python main.py",
        "    LLM_RETRIEVAL_MODE=keyword python main.py",
        "    LLM_RETRIEVAL_MODE=semantic python main.py",
        "    LLM_RRF_K=0 python main.py",
    ]
    return "\n".join(lines)
