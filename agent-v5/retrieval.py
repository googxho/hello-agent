"""
检索层 —— v5 的核心。

═══════════════════════════════════════════════════════════════════════
这一版要解决的问题
═══════════════════════════════════════════════════════════════════════

v4 让 Agent 记住了「你告诉过它的事」，但它对**你已经写在磁盘上的东西**一无所知。

你问「我上次记的那个项目叫什么」，它说没听过 ——
尽管答案就躺在 `notes/` 里，离它只有几厘米。

原因很朴素：**没人告诉过它那里有东西，更没人给它一把钥匙。**

═══════════════════════════════════════════════════════════════════════
这一版引入的最小概念
═══════════════════════════════════════════════════════════════════════

    资料太多，塞不进上下文 —— 那就**先找出来，再塞进去**。

    问题 ──→ 检索器 ──→ 最相关的 N 段 ──→ 塞进上下文 ──→ 模型回答

就这么一条流水线。但里面藏着**三个必须做的决定**，它们才是这一版的知识量。

═══════════════════════════════════════════════════════════════════════
决定一：怎么切（chunking）
═══════════════════════════════════════════════════════════════════════

文件不能整篇塞进去（太长），也不能按固定字数硬切（会把一句话劈成两半）。
这里按**结构**切：先认 Markdown 的标题，再按空行切段落。

    一篇笔记
      ├─ 小节 A → 段落 1、段落 2
      └─ 小节 B → 段落 3

每块都记住「我来自哪个文件、哪一行、属于哪个小节」——
因为**检索结果必须能被人回溯到原文**（不然你没法核对它有没有在编）。

═══════════════════════════════════════════════════════════════════════
决定二：怎么找（打分）
═══════════════════════════════════════════════════════════════════════

朴素的做法是「查到了就得分」。它有两个毛病：

  · 常见词（「的」「是」「项目」）到处都是，命中它们毫无信息量
  · 一段文字里出现 10 次 ≠ 比出现 1 次相关 10 倍（该饱和）

所以用 IDF（越罕见的词越值钱）+ 饱和的词频（BM25 那一套的核心）：

    score(查询, 文段) = Σ   idf(词) × tf / (tf + 1.5)
                       词∈查询

    idf(词) = log(1 + (总块数 − 含该词的块数 + 0.5) / (含该词的块数 + 0.5))

═══════════════════════════════════════════════════════════════════════
决定三：中文怎么办（分词）
═══════════════════════════════════════════════════════════════════════

英文有空格，`split()` 就完事。中文没有 —— 而装一个分词器（jieba 之类）
就要引入新依赖，还要维护词典。

所以这里用**字符 bigram**：把连续的中文切成两字一组。

    「性能优化」 →  性能 / 能优 / 优化

土，但有效：只要查询里的词在文段里出现过，就至少有一个 bigram 命中。
代价是「能优」这种无意义的组合也进了索引 —— 下一节会看到它的副作用。

═══════════════════════════════════════════════════════════════════════
⚠️ 这一版的天花板（留给下一版的痛）
═══════════════════════════════════════════════════════════════════════

**关键词检索靠的是「字面相同」。** 所以：

    笔记里写「性能优化」    →   你问「怎么让它更快」    →   检索不到 ✗

同义词、近义表达、跨语言的语义相似 —— 这一版全部无能为力。
这不是 bug，是这类方法的**天花板**，也正是 embedding（向量检索）
要解决的问题。

（亲手撞一次：见 EXPERIMENTS.md 的实验 3。）
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

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
    """一条检索结果 = 文段 + 得分 + **命中了哪些词**。

    命中词是刻意留的：它让「为什么这段被选中」变成可查的 ——
    没有它，检索就是个黑盒，调不动也信不过。
    """

    chunk: Chunk
    score: float
    matched: list[str] = field(default_factory=list)


@dataclass
class SearchOutcome:
    """一次检索的全貌，给上层打印用。"""

    query: str
    hits: list[SearchHit]
    scanned_chunks: int
    elapsed_ms: float

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

    def __init__(self, root: Path, *, chunk_max_chars: int = 600) -> None:
        self.root = root
        self.chunk_max_chars = chunk_max_chars
        self._index: Index | None = None
        self._fingerprint: tuple[int, float] | None = None
        self.rebuilds = 0        # 重建过几次（给界面看的）
        self.skipped_files = 0   # 读不了的文件数（容错，同 v4 的精神）

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

    # ── 查询接口 ───────────────────────────────────────────────────

    @property
    def chunk_count(self) -> int:
        return len(self._index.chunks) if self._index else 0

    def file_count(self) -> int:
        return self._scan()[0][0]

    def search(self, query: str, *, top_k: int = 5) -> SearchOutcome:
        import time

        started = time.perf_counter()
        self.refresh()
        hits = search(self._index, query, top_k=top_k) if self._index else []
        elapsed_ms = (time.perf_counter() - started) * 1000
        return SearchOutcome(
            query=query,
            hits=hits,
            scanned_chunks=self.chunk_count,
            elapsed_ms=elapsed_ms,
        )

    def describe(self) -> str:
        """给 /knowledge 命令用的一句话状态。"""
        if not self.root.is_dir():
            return f"资料目录不存在：{self.root}（用 LLM_NOTES_DIR 指到别处）"
        files, _ = self._scan()
        if self._index is None:
            return (
                f"{self.root} · 扫到 {files[0]} 个文件 —— "
                "索引还没建（第一次检索时才会建，这样启动不拖慢）"
            )
        extra = f" · 跳过 {self.skipped_files} 个读不了的文件" if self.skipped_files else ""
        return (
            f"{self.root} · {files[0]} 个文件 / {self.chunk_count} 块"
            f" · 索引重建过 {self.rebuilds} 次{extra}"
        )
