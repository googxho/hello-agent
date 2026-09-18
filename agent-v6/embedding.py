"""
向量层 —— v6（05b）的核心。

═══════════════════════════════════════════════════════════════════════
这一版要解决的问题
═══════════════════════════════════════════════════════════════════════

v5 的关键词检索**认字不认意思**：

    笔记里写「性能优化」  →  你问「怎么让它更快」  →  0 命中 ✗

这不是 bug，是「字面匹配」这类方法的物理上限。而用户根本不知道
你库里是怎么写的 —— 所以这个坎必须跨过去。

═══════════════════════════════════════════════════════════════════════
这一版引入的最小概念
═══════════════════════════════════════════════════════════════════════

    把文字变成一串数字（向量）。意思相近的文字，向量也相近。

    「性能优化…」        → [ 0.12, -0.03, … , 0.44]    ┐ 方向接近
    「怎么让它更快」      → [ 0.11, -0.02, … , 0.41]    ┘ 夹角小
    「番茄需要浇水」      → [-0.31,  0.22, … ,-0.08]     方向差很远

于是「找相关内容」变成了「找夹角最小的向量」—— **一个纯数学问题，
跟「字面上一样不一样」彻底无关。**

实测（就用这个项目的例子，bge-m3）：

    「性能优化…」 ↔ 「怎么让程序跑得更快」  余弦 0.6565
    「性能优化…」 ↔ 「番茄需要浇水」        余弦 0.4157

═══════════════════════════════════════════════════════════════════════
两个必须想清楚的问题
═══════════════════════════════════════════════════════════════════════

**① 向量从哪来？** —— 这次是远程服务（SiliconFlow 的 BAAI/bge-m3，1024 维）。
把它抽成一个 `Embedder` 接口，理由和 v3 的 `Summarizer` 一样：
真实实现要联网、要花钱；测试里塞个确定性的替身，整条流水线就能离线跑穿。

**② 算好的向量放哪？** —— ⚠️ **必须存盘。**

    embedding 是**按 token 计费**的。
    一个几百块的资料库，每次启动都重算一遍 = 每次启动都烧钱。

    缓存让「第一次贵，之后免费」。而缓存的**失效**靠「内容指纹」：
    哪一段变了，只重算那一段 —— 而不是整个库重来。
    （这是 v4「目录指纹」的精确版：从「整个目录」精确到「每一块」。）
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from openai import OpenAI

# 缓存文件的格式版本 —— 结构变了就 +1，老缓存会被安全地忽略。
CACHE_FORMAT_VERSION = 1


# ══════════════════════════════════════════════════════════════════════
# 指纹
# ══════════════════════════════════════════════════════════════════════


def content_key(text: str) -> str:
    """一段文字的指纹：内容没变，指纹就不变。

    缓存失效全靠它：内容一变 → 指纹一变 → 缓存查不到 → 只重算这一块。
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ══════════════════════════════════════════════════════════════════════
# 接口：把文字变成向量
# ══════════════════════════════════════════════════════════════════════


class Embedder(Protocol):
    """把文字变成向量。

    抽成接口的理由（和 v3 的 `Summarizer` 一模一样）：**可测性**。
    """

    @property
    def model_id(self) -> str:
        """模型标识。换模型 = 所有旧向量作废（缓存按它分区）。"""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量转换。失败时**抛异常**，由上层决定要不要降级。"""
        ...


@dataclass
class EmbedStats:
    """这一版的钱账 —— 向量是要花钱的，花了多少必须看得见。"""

    calls: int = 0          # 调了几次接口
    texts: int = 0          # 真正送去算的文字段数
    tokens: int = 0         # API 报的 token 数
    cache_hits: int = 0     # 有多少段直接命中缓存（= 省下的那部分）
    elapsed: float = 0.0

    def describe(self) -> str:
        if not self.calls and not self.cache_hits:
            return "还没算过向量"
        parts = []
        if self.cache_hits:
            parts.append(f"缓存命中 {self.cache_hits} 段（没花钱）")
        if self.calls:
            parts.append(
                f"调用 {self.calls} 次 · {self.texts} 段 · {self.tokens:,} tokens"
                f" · {self.elapsed:.1f}s"
            )
        return " · ".join(parts)


def _normalize_base(raw: str) -> str:
    """容错：把 `…/v1/embeddings` 这种「完整端点」削成 SDK 要的根路径 `…/v1`。

    坑：很多服务商文档给的是完整端点，而 SDK 会自己在后面拼 `/embeddings` ——
    照抄文档就会拼成 `/v1/embeddings/embeddings`，报 404。
    两种写法都接受，省得你踩这个坑。
    """
    url = (raw or "").strip().rstrip("/")
    if url.endswith("/embeddings"):
        url = url[: -len("/embeddings")]
    return url


class RemoteEmbedder:
    """调 OpenAI 兼容的 `/embeddings` 接口。"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        *,
        timeout: float = 30.0,
    ) -> None:
        self._client = OpenAI(
            api_key=api_key, base_url=_normalize_base(base_url), timeout=timeout
        )
        self._model = model
        self.stats = EmbedStats()

    @property
    def model_id(self) -> str:
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        started = time.perf_counter()
        # 一次请求里批量送 —— 比一条条送快得多，也更省
        response = self._client.embeddings.create(model=self._model, input=texts)
        elapsed = time.perf_counter() - started

        usage = getattr(response, "usage", None)
        self.stats.calls += 1
        self.stats.texts += len(texts)
        self.stats.tokens += getattr(usage, "total_tokens", 0) or 0
        self.stats.elapsed += elapsed

        vectors = [item.embedding for item in response.data]
        if len(vectors) != len(texts):
            # 服务端返回的数量对不上 —— 宁可报错，也不要把向量和文字错位对应
            raise RuntimeError(
                f"向量数量不对：送去 {len(texts)} 段，回来 {len(vectors)} 段"
            )
        return vectors


# ══════════════════════════════════════════════════════════════════════
# 缓存
# ══════════════════════════════════════════════════════════════════════


class EmbeddingCache:
    """向量缓存：按「内容指纹」索引，存在磁盘上。

    ⚠️ 两个容易忽略的细节：

      · **按模型分区**：换个 embedding 模型，同一段文字的向量完全不同 ——
        混在一起算相似度就是噪音。所以缓存文件里存了 model 名，
        对不上就整份作废（而不是硬用）。
      · **原子写**：同 v4 —— 写 `.tmp` → fsync → `os.replace()`。
        向量是花钱算出来的，写到一半崩了不该全丢。
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.model_id: str | None = None
        self.vectors: dict[str, list[float]] = {}
        # 读盘的状态，给界面看（铁律 7：没读到也要说清楚为什么）
        self.status: str = "empty"      # empty / loaded / discarded
        self.note: str = "还没读过缓存"

    # ── 读 ─────────────────────────────────────────────────────────

    def load(self, model_id: str) -> None:
        self.model_id = model_id
        if not self.path.is_file():
            self.status, self.note = "empty", "第一次用，还没有缓存"
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.status = "discarded"
            self.note = f"缓存文件读不了（{type(exc).__name__}），这次重新算"
            return

        if not isinstance(data, dict) or data.get("version") != CACHE_FORMAT_VERSION:
            self.status, self.note = "discarded", "缓存格式版本不符，这次重新算"
            return
        if data.get("model") != model_id:
            self.status = "discarded"
            self.note = (
                f"缓存是 {data.get('model')!r} 算的、现在用的是 {model_id!r}，"
                f"对不上，这次重新算"
            )
            return
        vectors = data.get("vectors")
        if not isinstance(vectors, dict):
            self.status, self.note = "discarded", "缓存里的 vectors 结构不对，这次重新算"
            return

        self.vectors = {k: v for k, v in vectors.items() if isinstance(v, list) and v}
        self.status = "loaded"
        self.note = f"从磁盘读回 {len(self.vectors)} 个向量"

    # ── 写 ─────────────────────────────────────────────────────────

    def save(self) -> None:
        """原子写。任何写盘失败都**不抛异常** —— 缓存只该加速，不该拖垮主流程。"""
        payload = {
            "version": CACHE_FORMAT_VERSION,
            "model": self.model_id,
            "count": len(self.vectors),
            "vectors": self.vectors,
        }
        text = json.dumps(payload, ensure_ascii=False)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except OSError as exc:
            self.note = f"缓存写盘失败（{type(exc).__name__}），下次还得重算"
            return
        self.note = f"缓存已存盘（{len(self.vectors)} 个向量）"

    # ── 查 / 存 ────────────────────────────────────────────────────

    def get(self, key: str) -> list[float] | None:
        return self.vectors.get(key)

    def put(self, key: str, vector: list[float]) -> None:
        self.vectors[key] = vector

    @property
    def size(self) -> int:
        return len(self.vectors)


# ══════════════════════════════════════════════════════════════════════
# 高层：带缓存的批量转换
# ══════════════════════════════════════════════════════════════════════


def embed_with_cache(
    embedder: Embedder,
    cache: EmbeddingCache,
    texts: list[str],
    *,
    batch_size: int = 32,
    store: bool = True,
) -> list[list[float]]:
    """把一批文字转成向量，能命中缓存就绝不重算。

    ⚠️ 返回的顺序和输入**严格一一对应** —— 调用方不用自己维护对应关系
       （这种「顺序错位」是向量检索里最容易出、又最难查的 bug）。

    ⚠️ store=False 时不把结果写进缓存。
       **查询词就该这么用**：用户每次问的都不一样，
       存进去只会把缓存越撑越大，而命中率极低。
       缓存是给「资料块」用的 —— 它们反复要用，而且总量有限。

    没命中的部分会**分批**送出去：一次几百段容易被接口限流，
    而且失败时重试的代价太大。
    """
    keys = [content_key(t) for t in texts]
    result: list[list[float] | None] = []
    missing: list[int] = []

    for i, key in enumerate(keys):
        cached = cache.get(key)
        result.append(cached)
        if cached is None:
            missing.append(i)
        else:
            embedder.stats.cache_hits += 1  # 命中缓存 = 省下的一次调用

    for start in range(0, len(missing), batch_size):
        chunk_idx = missing[start : start + batch_size]
        vectors = embedder.embed([texts[i] for i in chunk_idx])
        for i, vector in zip(chunk_idx, vectors):
            result[i] = vector
            if store:
                cache.put(keys[i], vector)

    return [v for v in result if v is not None]


# ══════════════════════════════════════════════════════════════════════
# 相似度
# ══════════════════════════════════════════════════════════════════════


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度：两个向量夹角的余弦，越接近 1 越相似。

    为什么不用欧氏距离？因为余弦**对向量的长度不敏感** ——
    一篇长文档和一个短句子的向量长度差很多，但「方向」（也就是意思）
    可以很接近。检索要的正是方向。
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def cosine_to_all(query: list[float], vectors: list[list[float]]) -> list[float]:
    """一次算出 query 跟所有向量的相似度。

    这里是整个检索里唯一「随规模变慢」的地方（N 块就是 N 次计算）。
    几百块毫无感觉；几万块就该换专业向量库了 —— 见 README 的「下一步」。
    """
    return [cosine(query, vector) for vector in vectors]


@dataclass
class VectorStats:
    """给界面看的一句话账本。"""

    embedder: EmbedStats = field(default_factory=EmbedStats)
    cache_size: int = 0
