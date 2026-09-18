"""
工具层 —— 工具协议、注册表，以及四个内置工具。

⚠️ 从 v2 到 v4，这个文件一行没动 —— 加新能力只碰该碰的地方。
v5 破了一次例，但只加了一个工具：**模型读不了你的磁盘**，
这是第四个「LLM 做不到、但工具能做到」的能力。

四个工具各自演示一种能力边界：

    calculator        算不准         —— 模型不会算术
    get_current_time  不知道「现在」 —— 模型的知识有截止日期
    save_note         改不了磁盘     —— 模型没有副作用能力
    search_notes      读不了你的资料 —— 模型不知道你有什么（v5 新增）
"""

from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from rich.console import Console

from config import (
    CHUNK_MAX_CHARS,
    DISABLED_TOOLS,
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_CACHE_FILE,
    EMBEDDING_MODEL,
    MIN_SIMILARITY,
    NOTES_DIR,
    RETRIEVAL_ENABLED,
    RETRIEVAL_MODE,
    RETRIEVAL_TOP_K,
)
from embedding import EmbeddingCache, RemoteEmbedder
from retrieval import KnowledgeBase

# ══════════════════════════════════════════════════════════════════════
# 工具协议 —— 怎么把一个 Python 函数「介绍」给 LLM
# ══════════════════════════════════════════════════════════════════════
#
# 这是整个 Agent 里最容易被低估的部分。
#
# LLM 看不到你的代码，它只能读到一段 JSON Schema 文本。它完全靠这段文本
# 来判断：什么时候该用这个工具、参数该怎么填。所以 ——
#
#     description 和参数说明，本质上就是写给 LLM 的提示词。


@dataclass
class ToolParameter:
    """描述工具的一个参数。"""

    name: str
    type: str  # JSON Schema 类型：string / integer / number / boolean / array
    description: str
    required: bool = True

    def to_schema(self) -> dict[str, Any]:
        return {"type": self.type, "description": self.description}


@dataclass
class ToolDefinition:
    """描述一个工具的完整信息。"""

    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)

    def to_openai_schema(self) -> dict[str, Any]:
        """拼成 OpenAI / DeepSeek 等接口要求的 tools 数组元素格式。"""
        properties = {p.name: p.to_schema() for p in self.parameters}
        required = [p.name for p in self.parameters if p.required]
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


# 所有工具定义的注册表：  工具名 -> ToolDefinition
TOOL_DEFS: dict[str, ToolDefinition] = {}
# 所有工具实现的注册表：  工具名 -> 普通 Python 函数
TOOL_IMPLS: dict[str, Callable[..., str]] = {}


def tool(definition: ToolDefinition) -> Callable[[Callable[..., str]], Callable[..., str]]:
    """装饰器：把「工具描述」和「工具实现」一起登记进注册表。"""

    def wrapper(fn: Callable[..., str]) -> Callable[..., str]:
        TOOL_DEFS[definition.name] = definition
        TOOL_IMPLS[definition.name] = fn
        return fn

    return wrapper


def openai_schemas() -> list[dict[str, Any]]:
    """把注册表里【当前可用】的工具转成接口要的 tools 数组。

    被 LLM_DISABLED_TOOLS 关掉的工具会在这里被过滤掉 —— 模型根本看不到它，
    自然不会去调。这是 EXPERIMENTS.md 里做「能力边界」实验的开关。
    """
    disabled = set(DISABLED_TOOLS)
    if not RETRIEVAL_ENABLED:
        # 关掉检索 = 让 search_notes 从模型视野里消失
        # （而不是「它调了、但被拒绝」—— 后者会让模型困惑，还多烧一次调用）
        disabled.add("search_notes")
    return [
        d.to_openai_schema()
        for name, d in TOOL_DEFS.items()
        if name not in disabled
    ]


# ══════════════════════════════════════════════════════════════════════
# 工具一：calculator —— 能力边界（模型算不准乘法）
# ══════════════════════════════════════════════════════════════════════

# 只允许这些运算，白名单机制 —— 绝对不要用裸 eval()。
_ALLOWED_OPS: dict[type, Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> Any:
    """递归求值一个被解析过的算术表达式 AST，遇到不允许的东西直接报错。"""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"不支持的常量：{node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"不支持的表达式：{ast.dump(node)}")


@tool(
    ToolDefinition(
        name="calculator",
        description=(
            "计算一个数学表达式并返回精确结果。"
            "凡是涉及数字运算的问题都必须调用本工具，不要自己心算，否则很容易算错。"
            "支持 + - * / // % ** 以及括号。"
        ),
        parameters=[
            ToolParameter(
                name="expression",
                type="string",
                description="要计算的数学表达式，例如 '(1234 * 5678) + 90'",
            ),
        ],
    )
)
def calculator(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")
    result = _safe_eval(tree)
    # 把 7006742.0 这种浮点结果整理成 7006742，看着更自然
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


# ══════════════════════════════════════════════════════════════════════
# 工具二：get_current_time —— 知识边界（模型不知道「现在」）
# ══════════════════════════════════════════════════════════════════════

_WEEKDAYS = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


@tool(
    ToolDefinition(
        name="get_current_time",
        description=(
            "获取当前的本地日期和时间。"
            "凡是涉及『今天』『现在』『当前时间』『今年是哪年』的问题都必须调用本工具，"
            "不要凭训练数据里的记忆猜测。"
        ),
        parameters=[],
    )
)
def get_current_time() -> str:
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S") + f"（{_WEEKDAYS[now.weekday()]}）"


# ══════════════════════════════════════════════════════════════════════
# 工具三：save_note —— 副作用能力（模型改不了磁盘）
# ══════════════════════════════════════════════════════════════════════


@tool(
    ToolDefinition(
        name="save_note",
        description=(
            "把一条笔记写入本地文件长期保存。"
            "当用户说『记下来』『记一下』『保存』『备忘』时调用本工具。"
            "注意：用户如果提到了日期，请先用 get_current_time 工具拿到今天的日期，再填进 content。"
        ),
        parameters=[
            ToolParameter(name="title", type="string", description="笔记标题，会被用作文件名"),
            ToolParameter(name="content", type="string", description="笔记正文内容，可以是多行文本"),
        ],
    )
)
def save_note(title: str, content: str) -> str:
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    # 清洗文件名：只保留字母、数字、下划线、中文和连字符。
    # 这一步同时也是安全措施 —— 冒号和斜杠会被替换掉，模型就无法用
    # "../../etc/passwd" 这种标题写到目录外面去。
    slug = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", title).strip("_") or "untitled"
    path = NOTES_DIR / f"{slug}.md"
    path.write_text(f"# {title}\n\n{content}\n", encoding="utf-8")
    return f"已保存到 {path.name}"


# ══════════════════════════════════════════════════════════════════════
# 工具四：search_notes —— 检索能力（模型不知道你有什么资料）
# ══════════════════════════════════════════════════════════════════════
#
# 这是 v5 唯一的“新工具”，但它和第 1~3 个不太一样：
# 前三个是「原子能力」（算个数、看个时间），这一个背后是一条流水线
# （扫目录 → 切块 → 索引 → 打分 → 排序）。

# ⚠️ 这是整个项目里**唯一会在工具内部 print 的工具**。
#
#    理由：检索过程全发生在工具内部（扫描、打分、排序），
#    agent.py 只看得到最后那段给模型看的文本。不打印出来，
#    用户根本不知道「它找了什么、找到几段、得分多少、有没有落空」。
#
#    铁律 7：过程要能自证清白 —— 这条在“黑盒工具”身上尤其重要。
console = Console()

# ── 装配知识库（v6：带上向量能力）────────────────────────────────────
#
# 这里要处理一个现实问题：**配置不全也得能用。**
# 选择了 semantic 模式、但没配 EMBEDDING_API_KEY → 退回关键词检索，
# 而且要说清楚原因（而不是默默变笨）。
_embedder = None
_embed_note = f"未使用（当前是 {RETRIEVAL_MODE} 模式）"
if RETRIEVAL_MODE == "semantic":
    if EMBEDDING_API_KEY:
        _embedder = RemoteEmbedder(EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL)
        _embed_note = f"{EMBEDDING_MODEL} @ {EMBEDDING_BASE_URL}"
    else:
        _embed_note = "没有配置 EMBEDDING_API_KEY —— 退回关键词检索（见仓库根 .env.example）"

_embed_cache = EmbeddingCache(EMBEDDING_CACHE_FILE)
if _embedder is not None:
    # 读一次缓存（坏文件/换模型都不怕，load() 内部自己容错）
    _embed_cache.load(_embedder.model_id)

KNOWLEDGE = KnowledgeBase(
    NOTES_DIR,
    chunk_max_chars=CHUNK_MAX_CHARS,
    mode=RETRIEVAL_MODE if _embedder is not None else "keyword",
    embedder=_embedder,
    cache=_embed_cache,
    min_similarity=MIN_SIMILARITY,
)
EMBEDDING_NOTE = _embed_note


@tool(
    ToolDefinition(
        name="search_notes",
        description=(
            "在用户本地的笔记库（notes/ 目录）里检索内容。"
            "当用户问「我记过什么」「我的笔记里有没有…」，"
            "或者问题可能涉及用户的个人资料时，必须先调用本工具检索，不要凭空回答。"
            "若检索结果里没有相关信息，就如实告诉用户笔记里没有 —— 不要编造。"
        ),
        parameters=[
            ToolParameter(
                name="query",
                type="string",
                description="要检索的关键词或问题描述，例如「项目交付时间」",
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="最多返回几段，默认 5",
                required=False,
            ),
        ],
    )
)
def search_notes(query: str, limit: int = RETRIEVAL_TOP_K) -> str:
    """检索笔记库。

    返回的字符串会原样交给模型 —— 所以里面**要有出处**
    （文件名 + 行号），否则模型只能含糊其辞，你也没法核对。
    """
    try:
        limit = max(1, min(int(limit), 20))
    except (TypeError, ValueError):
        limit = RETRIEVAL_TOP_K

    outcome = KNOWLEDGE.search(query, top_k=limit)

    # ── 给用户看的（不占模型上下文）────────────────────────────
    # 第一件事：说清楚「这次走的是哪条路」。
    # 向量检索和关键词检索的分数含义完全不同（相似度 vs 打分），
    # 混在一起看只会把人看糊涂 —— 而这些数字正是你判断检索好坏的主要依据。
    is_semantic = outcome.mode.startswith("semantic")
    mode_tag = "向量" if is_semantic else "关键词"
    head = f"  [cyan]🔍 检索[/cyan] [bold]{query}[/bold] [dim]（{mode_tag}）[/dim] → "

    if outcome.hits:
        if is_semantic:
            detail = "相似度 " + " / ".join(f"{h.score:.3f}" for h in outcome.hits[:3])
        else:
            detail = "最高分 " + " / ".join(f"{h.score:.2f}" for h in outcome.hits[:3])
        console.print(
            head
            + f"[green]{len(outcome.hits)} 段命中[/green]"
            + f" [dim]（{detail} · 扫描 {outcome.scanned_chunks} 块 · "
            f"{outcome.elapsed_ms:.1f} ms）[/dim]"
        )
        for hit in outcome.hits:
            if is_semantic:
                extra = f"相似度 {hit.score:.3f}"
            else:
                extra = "命中 " + ("、".join(hit.matched[:4]) or "（标题命中）")
            console.print(f"      [dim]{hit.chunk.citation()} · {extra}[/dim]")
    else:
        # 0 命中也要解释清楚「为什么」——
        # 向量检索和关键词检索「没找到」的原因完全不同：
        #   向量：相似度不够高（离门槛差多少？调门槛有用吗？）
        #   关键词：一个字都没对上（换个说法再试）
        if is_semantic and outcome.best_score:
            why = f"最高相似度 {outcome.best_score:.3f} < 门槛 {MIN_SIMILARITY}"
        else:
            why = "没有任何词命中"
        console.print(
            head
            + "[yellow]0 段命中[/yellow]"
            + f" [dim]（{why} · 扫描 {outcome.scanned_chunks} 块 · "
            f"{outcome.elapsed_ms:.1f} ms）[/dim]"
        )
        # 这一行是给用户的提醒，也是检索最怕的失效模式：
        # 空手而归时，模型很可能开始编 —— 提醒用户盯住它。
        console.print(
            "      [yellow]⚠ 资料里没有相关内容 —— 它接下来应该说「不知道」，"
            "而不是编一个〔盯一下〕[/yellow]"
        )

    if outcome.note:
        # 降级之类的意外必须说出来（铁律 7）：
        # 你以为在用向量检索，实际走的是关键词 —— 不说你就永远发现不了
        console.print(f"      [yellow]⚠ {outcome.note}[/yellow]")

    # ── 给模型看的（会进上下文）──────────────────────────────
    if not outcome.hits:
        return (
            f"在资料库里没有找到与「{query}」相关的内容"
            f"（已扫描 {outcome.scanned_chunks} 块）。\n"
            "如果需要这些资料，请如实告诉用户「你的笔记里没有相关内容」，不要编造。"
        )

    lines = [f"在资料库里找到 {len(outcome.hits)} 段相关内容（按相关度从高到低）："]
    for i, hit in enumerate(outcome.hits, 1):
        lines.append("")
        lines.append(f"[{i}] 出处：{hit.chunk.citation()}")
        lines.append(hit.chunk.text)
    return "\n".join(lines)
