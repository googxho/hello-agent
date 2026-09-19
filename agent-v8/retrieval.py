"""
检索层 —— v7 的核心（融合）。

═══════════════════════════════════════════════════════════════════════
这一版要解决的问题
═══════════════════════════════════════════════════════════════════════

v6 的实验 6 / 7 摆出了两个同样真实、方向相反的事实（都是实测）：

    「怎么让程序跑得更快」  关键词 0 命中 ✗        向量 0.651 ✓
    「ERR_5002」            关键词稳稳命中 ✓       向量 0.556（门槛一抬就死）✗

两把尺子，盲区几乎不重叠 —— 那为什么还要二选一？

═══════════════════════════════════════════════════════════════════════
这一版引入的最小概念
═══════════════════════════════════════════════════════════════════════

    两个都跑，然后融合排名 —— RRF（Reciprocal Rank Fusion）。

        查询 ──┬─→ 关键词检索 ─→ 名单 A ─┐
               └─→ 向量检索   ─→ 名单 B ─┴─→ RRF ─→ 前 N 段 ─→ 塞进上下文

RRF 的核心顿悟只有一句话：

    **分数不可比，但名次是通用语。**

    BM25 的分是 0.2~2.x，余弦是 0~1 —— 直接相加是灾难（谁量大谁说话）。
    可「第 1 名」跟「第 1 名」是可比的。于是：

        融合分(d) = Σ 1 / (k + 名次_r(d))      （没被提名的一路贡献 0）

    被一边提名拿一份分，被两边都提名拿两份 —— **共识是真金白银**：
    「两边都看中」的段，天然压过「一边的第一名」。

    k 是「共识的含金量」旋钮：k 大 → 共识更值钱；k 小 → 单边冠军抬头。
    （默认 60，Cormack 等 2009 的经验值。手算例子见 rrf_fuse 的 docstring。）

═══════════════════════════════════════════════════════════════════════
三个想清楚了的决定
═══════════════════════════════════════════════════════════════════════

**① 每一路按自己的老标准发言。** 关键词侧「有词命中」才算、向量侧过了
`LLM_MIN_SIMILARITY` 才算 —— 各自的命中规则一个没改，RRF 只合并**命中项**。
于是 v5 的「宁缺毋滥」和 v6 的门槛行为原样保留；0 命中的查询也不会
因为「融合」就被硬塞 —— **RRF 没有无中生有的本事。**

**② 合并按「内容」去重。** 同一段被两路同时提名 → 并成一条、分数相加。
身份用 (文件, 行号, 内容) —— 内容才是「同一段」的判据，文件名不是。

**③ 每条结果都带两边的名次。** 比如 `[关键词#1 + 向量#3] → 融合 0.032`。
结果不对劲时，这一行直接告诉你「是关键词的锅还是向量的锅」（铁律 7）。

═══════════════════════════════════════════════════════════════════════
地基（v5 / v6 定的，这版一行都没动）
═══════════════════════════════════════════════════════════════════════

    · 切分：按 Markdown 标题 + 空行切段落，每块记得「文件:行号 + 小节」
      （出处可回溯 —— 不然你没法核对它有没有在编）
    · 关键词打分：IDF（越罕见的词越值钱）× 饱和词频（BM25 那一套），
      中文用字符 bigram 补「没有空格」的缺口
    · 深度讲解都在 v5 / v6 的 README 里，这里不重复

═══════════════════════════════════════════════════════════════════════
⚠️ 这一版的天花板（留给下一版的痛）
═══════════════════════════════════════════════════════════════════════

旋钮多起来了：门槛、k、depth、top_k —— 调它们目前全靠手工实验。
想回答「k 取多少最好」「depth 取几合适」，需要一批「问题 + 期望命中」
的评估集：这活儿没法靠感觉，也没法靠一次运行说了算。
（见 README 的「下一版」。）
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from embedding import Embedder, EmbeddingCache, cosine_to_all, embed_with_cache

# ══════════════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════════════


@dataclass
class Chunk:
    """一段可被检索的文字。

    ⚠️ source / line / heading 不是装饰品 ——
       它们是「检索结果能被回溯」的保证：用户看到的不只是答案，
       还有「它是从哪个文件哪一行找出来的」，可以自己去核对。
    """

    source: str        # 文件名（相对资料目录）
    line: int          # 在文件里的起始行（从 1 数）
    heading: str       # 所属的小节标题（没有就是整篇的标题）
    text: str

    def citation(self) -> str:
        """给人和模型看的出处标注。"""
        where = f"{self.source}:{self.line}"
        return f"{where}（{self.heading}）" if self.heading else where


@dataclass
class SearchHit:
    """一条检索结果 = 文段 + 得分 + **命中了哪些词**（+ 融合时的两路名次）。

    命中词是刻意留的：它让「为什么这段被选中」变成可查的 ——
    没有它，检索就是个黑盒，调不动也信不过。

    v7 起，score 的含义随模式走：
      · keyword  → BM25 打分
      · semantic → 余弦相似度
      · hybrid   → RRF 融合分（⚠️ 这时别看它的大小，看 ranks —— 名次才是信息）
    """

    chunk: Chunk
    score: float
    matched: list[str] = field(default_factory=list)
    # v7 新增：只在 hybrid 模式下填充。
    #   ranks["keyword"] = 1  → 关键词侧把它排第 1（没被提名的一路不出现）。
    #   raw_scores 保留两路各自的原始分（0.42 / 0.651）——
    #   它们【不可相加】，摆在一起只是为了让「量纲不同」肉眼可见。
    ranks: dict[str, int] = field(default_factory=dict)
    raw_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class SearchOutcome:
    """一次检索的全貌，给上层打印用。"""

    query: str
    hits: list[SearchHit]
    scanned_chunks: int
    elapsed_ms: float
    # 这次实际走的是哪条路 —— keyword（v5）/ semantic（v6）/ hybrid（v7）/ 降级
    mode: str = "keyword"
    note: str | None = None      # 降级原因之类的补充说明
    best_score: float = 0.0      # 过滤前的最高分（能解释「为什么是 0 命中」）
    # v7 新增：只在 hybrid 模式有意义 —— 「两路各提名了几段」。
    # 0 命中时排查全靠这两个数：是两路都空，还是某一侧哑火了？
    keyword_count: int = 0
    semantic_count: int = 0
    rrf_k: int = 0               # 这次融合用的 k（0 = 没走融合）

    @property
    def is_empty(self) -> bool:
        return not self.hits


# ══════════════════════════════════════════════════════════════════════
# 分词
# ══════════════════════════════════════════════════════════════════════

_CJK = re.compile(r"[\u4e00-\u9fff]+")
_LATIN = re.compile(r"[a-z0-9_]+")

# 英文里最常见的一批词，检索时必须丢掉 —— 它们不携带任何区分度，
# 留着只会让「the」「is」这种词把结果带偏。
_STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "with", "that", "this", "you",
    "your", "not", "but", "can", "will", "has", "have", "from", "what", "how",
    "when", "which", "there", "their", "它", "的", "了", "是", "在", "和",
}


def tokenize(text: str) -> list[str]:
    """把一段文字切成可检索的「词」。

    规则：
      · 英文/数字：按非字母数字切开，转小写，丢掉停用词和单字母
      · 中文：连续汉字串 → 字符 bigram（「性能优化」→ 性能/能优/优化）
            单独一个字（比如「我」）也保留，否则短查询会什么都匹配不到
    """
    lowered = text.lower()
    tokens: list[str] = []

    for word in _LATIN.findall(lowered):
        if len(word) >= 2 and word not in _STOPWORDS:
            tokens.append(word)

    for run in _CJK.findall(lowered):
        if len(run) == 1:
            tokens.append(run)
            continue
        for i in range(len(run) - 1):
            tokens.append(run[i : i + 2])

    return tokens


# ══════════════════════════════════════════════════════════════════════
# 切分
# ══════════════════════════════════════════════════════════════════════

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def split_markdown(text: str, source: str, *, max_chars: int = 600) -> list[Chunk]:
    """把一篇 Markdown 切成若干块。

    两遍处理：
      1. **按标题分段** —— 认 `#` 开头的行，它开启一个新的小节
      2. **小节内按空行切段落** —— 段落是语义的自然边界

    超过 max_chars 的段落会被进一步切 —— 但**只在段落内部切**，
    这样至少不会把不同话题搅在一起。
    """
    chunks: list[Chunk] = []
    heading = ""
    buffer: list[str] = []
    buffer_line = 1

    def flush(line: int) -> None:
        if not buffer:
            return
        body = "\n".join(buffer).strip()
        buffer.clear()
        if not body:
            return
        # 太长的段落再切一刀（按字符数，尽量在换行处断开）
        if len(body) <= max_chars:
            chunks.append(Chunk(source=source, line=line, heading=heading, text=body))
            return
        start = 0
        while start < len(body):
            piece = body[start : start + max_chars]
            if len(body) - start > max_chars:
                cut = piece.rfind("\n")
                if cut > max_chars // 2:
                    piece = piece[:cut]
            chunks.append(Chunk(source=source, line=line, heading=heading, text=piece.strip()))
            start += len(piece)

    for lineno, raw in enumerate(text.splitlines(), start=1):
        match = _HEADING.match(raw.strip())
        if match:
            # 新小节开始：先把上一段收掉
            flush(buffer_line)
            heading = match.group(2).strip()
            buffer_line = lineno
            continue
        if raw.strip():
            if not buffer:
                buffer_line = lineno
            buffer.append(raw)
        else:
            flush(buffer_line)

    flush(buffer_line)
    return [c for c in chunks if c.text]


# ══════════════════════════════════════════════════════════════════════
# 索引 + 检索
# ══════════════════════════════════════════════════════════════════════

# BM25 的饱和系数：词频越大，每一次新增的边际收益越小。
# 1.5 是文献里的常用值，这里没做调优 —— 它不影响「能不能找到」，
# 只影响「找到几段」的排序细节。
_TF_K = 1.5


@dataclass
class Index:
    """倒排索引：词 → 出现在哪些块里。

    ⚠️ 只存「有哪些块」（用于算 df / idf），不存词频 ——
       因为块很小，打分时现算 tf 的成本可以忽略，
       而少维护一份数据就少一类不一致的 bug。
    """

    chunks: list[Chunk]
    postings: dict[str, set[int]] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.chunks)

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        if df == 0:
            return 0.0
        n = max(self.size, 1)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))


def build_index(chunks: list[Chunk]) -> Index:
    """给一堆文段建倒排索引。"""
    index = Index(chunks=list(chunks))
    for i, chunk in enumerate(index.chunks):
        # 标题也参与检索 —— 「项目 A」这种关键词往往只出现在标题里
        haystack = f"{chunk.heading}\n{chunk.text}"
        for term in set(tokenize(haystack)):
            index.postings.setdefault(term, set()).add(i)
    return index


def search(index: Index, query: str, *, top_k: int = 5) -> list[SearchHit]:
    """在索引里找最相关的 top_k 段。"""
    terms = tokenize(query)
    if not terms or not index.chunks:
        return []

    # 同一个词在查询里出现几次，就记几次权重（查询侧的词频）
    query_tf: dict[str, int] = {}
    for term in terms:
        query_tf[term] = query_tf.get(term, 0) + 1

    scored: list[SearchHit] = []
    for i, chunk in enumerate(index.chunks):
        # 标题权重加倍：能命中标题的词，相关度天然更高
        chunk_terms = tokenize(chunk.text)
        heading_terms = tokenize(chunk.heading)
        tf: dict[str, int] = {}
        for term in chunk_terms:
            tf[term] = tf.get(term, 0) + 1
        for term in heading_terms:
            tf[term] = tf.get(term, 0) + 2

        score = 0.0
        matched: list[str] = []
        for term, qtf in query_tf.items():
            freq = tf.get(term)
            if not freq:
                continue
            score += index.idf(term) * qtf * (freq / (freq + _TF_K))
            matched.append(term)

        if score > 0:
            scored.append(SearchHit(chunk=chunk, score=score, matched=matched))

    scored.sort(key=lambda hit: hit.score, reverse=True)
    return scored[:top_k]


# ══════════════════════════════════════════════════════════════════════
# RRF 融合（v7 的核心）
# ══════════════════════════════════════════════════════════════════════
#
# 为什么不是「把两个分数加起来」？因为两边的分数量纲完全不同：
#
#     关键词给 A 打 2.4、给 B 打 0.3      （BM25：0 到几不等）
#     向量给   A 打 0.65、给 B 打 0.61     （余弦：0~1，还挤在一起）
#
# 直接相加 = 谁量大谁说了算，而不是谁说得对谁说了算。
# 归一化之后再加？敏感、还得调权重 —— 是另一条路（见 README 的对比表）。
#
# RRF 绕开了整件事：**扔掉分数，只看名次。**
#
#     融合分(d) = Σ  1 / (k + 名次_r(d))
#                 r∈提了d名的那些路
#
# 手算一遍（k=60，两份名单）：
#
#     关键词：[B#1, C#2, D#3]      向量：[A#1, E#2, B#3, C#4]
#
#     B = 1/61 + 1/63 ≈ 0.0323   ← 两边都提名，第一
#     C = 1/62 + 1/64 ≈ 0.0318   ← 两边都提名，第二
#     A = 1/61        ≈ 0.0164   ← 向量侧第一，但只有一边提名
#     E = 1/62        ≈ 0.0161
#     D = 1/63        ≈ 0.0159
#
# 看出灵魂了吗：**A 在向量那边是第一，却输给了名次平平的 B、C** ——
# 因为 B、C 被两条独立的路同时看中。「共识」比「单边的第一名」更值钱。
#
# k 调的就是这份「共识的含金量」：
#     · k=0  → A=1.0 反超 C=0.75 —— 单边冠军抬头（小组赛只看头名）
#     · k=60 → C 反超 A  —— 两边都提名更吃香（委员会投票）
#
# 顺带一提：RRF 是 2009 年 Cormack 等人提出的，现在是工程界默认做法 ——
# Elasticsearch / Azure AI Search / Milvus 里的 "RRF" 说的都是它。


def rrf_fuse(
    ranked_lists: list[tuple[str, list[SearchHit]]],
    *,
    k: int = 60,
    depth: int = 0,
) -> list[SearchHit]:
    """把多路检索结果融合成一个排名。

    ranked_lists：[(路的名字, 该路的命中列表), ...]，列表本身要已按相关度排好序。
    k：     平滑常数（默认 60）。越小越放大「第一名」的优势。
    depth： 每一路只算前几名的分（0 = 不限）。名单之外 = 没有名次 = 贡献 0 ——
            这是融合的隐藏门槛，EXPERIMENTS.md 里会拧它。

    ⚠️ 合并按【内容】去重：(文件, 行号, 内容) 相同的算同一段，分数相加。
       同一段被两路都提名是好事，不该在结果里出现两次。
    """
    if k < 0:
        k = 0
    merged: dict[tuple[str, int, str], dict] = {}

    for source, hits in ranked_lists:
        for i, hit in enumerate(hits):
            if depth > 0 and i >= depth:
                break
            rank = i + 1
            key = (hit.chunk.source, hit.chunk.line, hit.chunk.text)
            entry = merged.get(key)
            if entry is None:
                entry = {
                    "chunk": hit.chunk,
                    "score": 0.0,
                    "matched": [],
                    "ranks": {},
                    "raw_scores": {},
                }
                merged[key] = entry
            entry["score"] += 1.0 / (k + rank)
            entry["ranks"][source] = rank
            entry["raw_scores"][source] = hit.score
            if hit.matched:
                # 命中词只有关键词侧才有（向量侧没有「命中了哪个词」这回事）
                entry["matched"] = hit.matched

    fused = [
        SearchHit(
            chunk=entry["chunk"],
            score=entry["score"],
            matched=entry["matched"],
            ranks=entry["ranks"],
            raw_scores=entry["raw_scores"],
        )
        for entry in merged.values()
    ]
    # 分数降序；并列时两路名次更好的在前（再并列则保持先出现的顺序 —— 稳定排序）
    fused.sort(key=lambda hit: (-hit.score, min(hit.ranks.values())))
    return fused


# ══════════════════════════════════════════════════════════════════════
# 资料库：把「读文件 → 切分 → 建索引」打包，并管好缓存
# ══════════════════════════════════════════════════════════════════════


class KnowledgeBase:
    """一个目录 + 它的索引。

    ⚠️ 这里有个绕不开的矛盾：
         · 每次检索都重新读一遍文件 → 永远最新，但**慢**（几十个文件就够呛）
         · 只在启动时读一次        → 快，但**你新写的笔记它看不到**

    这一版的选择：**懒加载 + 失效检查**。

      · 第一次真正用到时才建索引（启动不变慢）
      · 之后每次检索前，比一下目录的「指纹」（文件数 + 最新修改时间）
      · 指纹变了就重建 —— 便宜，而且不会漏掉新增的笔记
    """

    def __init__(
        self,
        root: Path,
        *,
        chunk_max_chars: int = 600,
        mode: str = "keyword",
        embedder: Embedder | None = None,
        cache: EmbeddingCache | None = None,
        min_similarity: float = 0.0,
        rrf_k: int = 60,
        hybrid_depth: int = 10,
    ) -> None:
        self.root = root
        self.chunk_max_chars = chunk_max_chars
        # 检索模式："keyword"（v5 字面）| "semantic"（v6 向量）| "hybrid"（v7 两个都跑）
        # ⚠️ 带向量的模式（semantic / hybrid）里如果向量算不出来，会**降级**成 keyword ——
        #    降级不是「偷偷换一种」，而是会在日志里说明（铁律 7）。
        self.mode = mode
        self.embedder = embedder
        self.cache = cache
        # 向量检索的相关性门槛。低于它的不算命中 ——
        # 不设这道坎，top_k 会把「最不相关的」也硬塞给你（实测里它真的会）。
        self.min_similarity = min_similarity
        # v7：融合的两个旋钮（只在 hybrid 模式下用）
        self.rrf_k = rrf_k
        self.hybrid_depth = hybrid_depth

        self._index: Index | None = None
        self._vectors: list[list[float]] = []   # 和 chunks 一一对应的向量
        self._fingerprint: tuple[int, float] | None = None
        self.rebuilds = 0        # 重建过几次
        self.skipped_files = 0   # 读不了的文件数（容错，同 v4 的精神）
        self.vectors_built = 0   # 算过几次全库向量
        self.last_error: str | None = None  # 上一次降级的原因

    # ── 指纹：用来判断「资料变了吗」────────────────────────────────

    def _scan(self) -> tuple[tuple[int, float], list[Path]]:
        files: list[Path] = []
        mtime_max = 0.0
        if self.root.is_dir():
            for path in sorted(self.root.rglob("*")):
                if not path.is_file() or path.name.startswith("."):
                    continue
                if path.suffix.lower() not in {".md", ".txt", ".markdown"}:
                    continue
                files.append(path)
                try:
                    mtime_max = max(mtime_max, path.stat().st_mtime)
                except OSError:
                    pass
        return (len(files), mtime_max), files

    def refresh(self) -> None:
        """如果资料变了，重建索引。"""
        fingerprint, files = self._scan()
        if self._index is not None and fingerprint == self._fingerprint:
            return

        chunks: list[Chunk] = []
        self.skipped_files = 0
        for path in files:
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                # 单个文件读不了，不该让整次检索失败 —— 跳过它，记个数
                self.skipped_files += 1
                continue
            rel = str(path.relative_to(self.root))
            chunks.extend(split_markdown(text, rel, max_chars=self.chunk_max_chars))

        self._index = build_index(chunks)
        self._fingerprint = fingerprint
        self.rebuilds += 1
        # 索引重建 = 块变了，之前算好的向量必须作废重来。
        # ⚠️ 别担心成本：缓存是按**内容指纹**命中的，
        #    真正会重算的只有「变了的那几块」（而不是整个库）。
        self._vectors = []

    # ── 查询接口 ───────────────────────────────────────────────────

    @property
    def chunk_count(self) -> int:
        return len(self._index.chunks) if self._index else 0

    def file_count(self) -> int:
        return self._scan()[0][0]

    def search(self, query: str, *, top_k: int = 5) -> SearchOutcome:
        started = time.perf_counter()
        self.refresh()
        self.last_error = None

        hits: list[SearchHit] = []
        mode_used = "keyword"
        note: str | None = None
        best_score = 0.0
        use_keyword = True
        keyword_count = 0
        semantic_count = 0
        rrf_k_used = 0

        # 需要向量的模式：semantic（v6）/ hybrid（v7）
        wants_vector = self.mode in {"semantic", "hybrid"}
        if wants_vector and self.embedder is not None and self.cache is not None:
            try:
                if self.mode == "hybrid":
                    # ── 两路都跑（v7）──────────────────────────────
                    # 每一路先出各自的「提名名单」（top hybrid_depth 段），
                    # RRF 融合之后，再截出真正要用的 top_k。
                    vec_hits, best_score = self._semantic_hits(query, self.hybrid_depth)
                    semantic_count = len(vec_hits)
                    kw_hits = (
                        search(self._index, query, top_k=self.hybrid_depth)
                        if self._index is not None
                        else []
                    )
                    keyword_count = len(kw_hits)
                    fused = rrf_fuse(
                        [("keyword", kw_hits), ("semantic", vec_hits)],
                        k=self.rrf_k,
                        depth=self.hybrid_depth,
                    )
                    hits = fused[:top_k]
                    mode_used = "hybrid"
                    rrf_k_used = self.rrf_k
                else:
                    # ── v6 的老路：只看向量 ─────────────────────────
                    hits, best_score = self._semantic_hits(query, top_k)
                    semantic_count = len(hits)
                    mode_used = "semantic"
                use_keyword = False
            except Exception as exc:
                # ⚠️ 降级路径：向量算不出来（断网 / 配额 / 模型名写错），
                #    不能让检索整个失效 —— 退回 v5 的关键词检索。
                #    代价是「同义词又搜不到了」，但功能还在。
                self.last_error = f"{type(exc).__name__}: {exc}"
                mode_used = "keyword（向量失败，已降级）"
                note = self.last_error
                use_keyword = True

        # ⚠️ 用 bool 控制流程，不要拿 mode 字符串去比 ——
        #    mode 里带着给人看的中文后缀（「已降级」），
        #    拿它当开关，迟早会踩坑（这里就踩过一次）。
        if use_keyword and self._index is not None:
            hits = search(self._index, query, top_k=top_k)
            best_score = hits[0].score if hits else 0.0
        # 注意：带向量的模式下「0 命中」是**合法结果**，不偷偷换成关键词 ——
        # 否则「换个说法能不能找到」这个对比实验就做不成了（你分不清走的哪条路）。

        elapsed_ms = (time.perf_counter() - started) * 1000
        return SearchOutcome(
            query=query,
            hits=hits,
            scanned_chunks=self.chunk_count,
            elapsed_ms=elapsed_ms,
            mode=mode_used,
            note=note,
            best_score=best_score,
            keyword_count=keyword_count,
            semantic_count=semantic_count,
            rrf_k=rrf_k_used,
        )

    # ── 向量检索（v6 新增）──────────────────────────────────────────

    def _semantic_hits(self, query: str, top_k: int) -> tuple[list[SearchHit], float]:
        """把查询也变成向量，找夹角最小的几块。返回 (命中的块, 过滤前的最高分)。"""
        if self._index is None:
            return [], 0.0
        self._ensure_vectors()
        # 查询词不上缓存（store=False）：它每次都不一样，存了只会把缓存撑大
        query_vector = embed_with_cache(
            self.embedder, self.cache, [query], store=False
        )[0]
        scores = cosine_to_all(query_vector, self._vectors)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        best = scores[order[0]] if order else 0.0

        hits: list[SearchHit] = []
        for i in order[:top_k]:
            if scores[i] < self.min_similarity:
                # 已经按分数降序，后面的只会更低 —— 到此为止
                break
            # matched 留空：向量检索没有「命中了哪个词」这回事 ——
            # 「为什么选中它」的答案就是相似度本身（已经在 score 里）
            hits.append(SearchHit(chunk=self._index.chunks[i], score=scores[i], matched=[]))
        return hits, best

    def _ensure_vectors(self) -> None:
        """确保每个块都有对应的向量（懒加载 + 走缓存）。

        ⚠️ 这是 v6 里**唯一会花钱**的地方：
           第一次跑要把所有块送去算向量；之后全靠缓存，免费。
        """
        if self._vectors or self._index is None:
            return
        before = self.cache.size
        self._vectors = embed_with_cache(
            self.embedder, self.cache, [c.text for c in self._index.chunks]
        )
        self.vectors_built += 1
        if self.cache.size > before:
            # 有新算出的向量 → 马上落盘。
            # 不存的话，下次启动又得花钱重算一遍 —— 这正是缓存存在的意义。
            self.cache.save()

    def describe(self) -> str:
        """给 /knowledge 命令用的多行状态。"""
        if not self.root.is_dir():
            return f"资料目录不存在：{self.root}（用 LLM_NOTES_DIR 指到别处）"
        files, _ = self._scan()

        mode_line = f"检索模式：{self.mode}"
        if self.mode == "hybrid" and self.embedder is not None:
            mode_line += f"（关键词 + {self.embedder.model_id} → RRF k={self.rrf_k}，每边取前 {self.hybrid_depth} 段）"
        elif self.mode == "semantic" and self.embedder is not None:
            mode_line += f"（模型 {self.embedder.model_id}）"

        if self._index is None:
            return (
                f"{self.root}\n"
                f"{files[0]} 个文件 · {mode_line}\n"
                "索引还没建（第一次检索时才会建，这样启动不拖慢）"
            )

        lines = [
            f"{self.root}",
            f"{files[0]} 个文件 / {self.chunk_count} 块 · 索引重建过 {self.rebuilds} 次",
            mode_line,
        ]
        if self.mode in {"semantic", "hybrid"} and self.cache is not None and self.embedder is not None:
            lines.append(
                f"向量：已算 {len(self._vectors)} 块（全库算过 {self.vectors_built} 次）"
                f" · 缓存里 {self.cache.size} 个"
            )
            lines.append(f"缓存状态：{self.cache.note}")
            lines.append(f"向量账本：{self.embedder.stats.describe()}")
        if self.skipped_files:
            lines.append(f"跳过 {self.skipped_files} 个读不了的文件")
        if self.last_error:
            lines.append(f"⚠ 上一次检索降级了：{self.last_error}")
        return "\n".join(lines)
