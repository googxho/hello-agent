"""
会话存储层 —— v4 的核心。

═══════════════════════════════════════════════════════════════════════
这一版要解决的问题
═══════════════════════════════════════════════════════════════════════

v3 把「聊久了会忘」解决了，但解决得不彻底：

    messages 和 summary_text 全都躺在 Python 变量里。

关掉终端 → 变量消失 → 下次启动，你对它来说是个陌生人。

    换句话说：v3 的记忆有「保质期」，但这个保质期的单位是**进程**。

═══════════════════════════════════════════════════════════════════════
这一版引入的最小概念
═══════════════════════════════════════════════════════════════════════

    把会话（消息 + 摘要）写进磁盘，启动时读回来。

    内存里的状态 ──写─→ session.json ──读─→ 内存里的状态

概念就这么简单。**真正花心思的是下面三件事 —— 它们才是这一版的知识量。**

═══════════════════════════════════════════════════════════════════════
坑一：写到一半崩了怎么办（原子写）
═══════════════════════════════════════════════════════════════════════

最直觉的写法是「打开文件、写入」，但它在真实世界里会出事：

    session.json ← 写到一半，进程被 kill / 断电 / 磁盘满
    → 文件只剩半截 JSON → 下次启动解析失败 → **整场会话没了**

解法是「原子写」—— 三步，全靠操作系统保证：

    1. 写一个临时文件  session.json.tmp
    2. flush + fsync（确保真的落到磁盘，而不是还在系统缓存里）
    3. os.replace(tmp, target)  ← 原子替换

新文件要么完整存在，要么完全不存在；**永远不会被读者看到「半截」的状态**。

═══════════════════════════════════════════════════════════════════════
坑二：文件坏了，程序不能跟着坏（容错读）
═══════════════════════════════════════════════════════════════════════

磁盘上的东西你无法控制：可能是手改坏的、可能是旧版本留下的、
可能是上一版程序写了一半的。

所以读的时候，**所有异常都不许上抛**，而是降级成一个明确的「状态」：

    "restored"        读到了，内容可用
    "missing"         文件不存在（第一次运行，完全正常）
    "corrupt"         文件坏了（解析失败 / 结构不对）→ 改名备份，从零开始
    "future_version"  文件是更新版本写的 → 不敢乱读，从零开始

每一种状态都带一句**给人看的说明**（note），启动时原样打出来 ——
学习者必须一眼知道自己处在哪一种情况里。

═══════════════════════════════════════════════════════════════════════
坑三：结构变了怎么办（版本号）
═══════════════════════════════════════════════════════════════════════

文件里存了 "version"。以后格式变了，靠它判断能不能读。
现在只有一个版本，但这个字段必须从第一天就有 ——
**等你需要它的时候再加，老文件里就没有它了。**
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# 存储格式的版本号。改动文件结构时 +1。
SESSION_FORMAT_VERSION = 1

# 备份后缀：坏文件会被改名成 session.json.corrupt，而不是删掉。
# 为什么不删？因为里面可能是用户聊了一下午的内容 ——
# 程序没资格替人做主扔掉它。
CORRUPT_SUFFIX = ".corrupt"


@dataclass
class SaveResult:
    """一次存盘的结果（含耗时，让你能看到写盘到底贵不贵）。"""

    path: Path
    bytes_written: int
    elapsed_ms: float
    message_count: int
    summary_chars: int


@dataclass
class LoadResult:
    """一次读取的结果。

    status 只有四种（见文件头的说明），note 是**给人看的一句话** ——
    启动时直接打印，学习者一眼就知道自己处在什么状态。
    """

    status: str                     # restored / missing / corrupt / future_version
    note: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    summary: str | None = None
    saved_at: str | None = None
    age_seconds: float | None = None

    @property
    def restored(self) -> bool:
        return self.status == "restored"


# ══════════════════════════════════════════════════════════════════════
# 写
# ══════════════════════════════════════════════════════════════════════


def save_session(
    path: Path,
    *,
    messages: list[dict[str, Any]],
    summary: str | None,
) -> SaveResult:
    """把会话原子地写进磁盘。

    ⚠️ 这里绝不能用「直接 open(path, "w")」—— 见文件头的「坑一」。
    """
    started = time.perf_counter()

    payload = {
        "version": SESSION_FORMAT_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "messages": messages,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")

    # 第 1 步：写临时文件
    with open(tmp_path, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        # 第 2 步：fsync —— 逼操作系统真的把数据落到磁盘。
        #   没有这一步，断电时数据可能还躺在系统缓存里，
        #   于是「原子替换」保住的是一个空文件。
        os.fsync(fh.fileno())

    # 第 3 步：原子替换。在这一瞬间之前，读者看到的还是旧文件；
    #   在这一瞬间之后，看到的是完整的新文件。中间的半成品状态不存在。
    os.replace(tmp_path, path)

    elapsed_ms = (time.perf_counter() - started) * 1000
    return SaveResult(
        path=path,
        bytes_written=len(text.encode("utf-8")),
        elapsed_ms=elapsed_ms,
        message_count=len(messages),
        summary_chars=len(summary or ""),
    )


# ══════════════════════════════════════════════════════════════════════
# 读
# ══════════════════════════════════════════════════════════════════════


def load_session(path: Path) -> LoadResult:
    """读会话文件。**任何异常都不上抛**，而是降级成一个明确的状态。

    ⚠️ 这是这一层最重要的设计：磁盘上的东西不可信。
       程序必须能对着一堆烂数据说「我读不了，但我还能跑」。
    """
    if not path.is_file():
        return LoadResult(
            status="missing",
            note=f"还没有会话文件（第一次运行就会有）→ {path}",
        )

    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        backup = _quarantine(path)
        return LoadResult(
            status="corrupt",
            note=(
                f"会话文件读不了（{type(exc).__name__}），已把它改名备份到 {backup.name}，"
                f"这次从零开始"
            ),
        )

    if not isinstance(data, dict):
        backup = _quarantine(path)
        return LoadResult(
            status="corrupt",
            note=f"会话文件的结构不对（顶层不是对象），已备份到 {backup.name}，这次从零开始",
        )

    version = data.get("version")
    if not isinstance(version, int):
        backup = _quarantine(path)
        return LoadResult(
            status="corrupt",
            note=f"会话文件缺少版本号，已备份到 {backup.name}，这次从零开始",
        )

    if version > SESSION_FORMAT_VERSION:
        # 未来的格式，不敢乱读 —— 但也**不能**把它当坏文件改名，
        # 因为那是「更新版本的程序」留下的东西，用户可能还在用。
        return LoadResult(
            status="future_version",
            note=(
                f"会话文件是更新的格式（v{version} > 本程序认识的 v{SESSION_FORMAT_VERSION}），"
                f"为了不把它读坏，这次从零开始（文件保持原样）"
            ),
        )

    messages = data.get("messages")
    summary = data.get("summary")
    if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
        backup = _quarantine(path)
        return LoadResult(
            status="corrupt",
            note=f"会话文件里的 messages 不是消息列表，已备份到 {backup.name}，这次从零开始",
        )
    if summary is not None and not isinstance(summary, str):
        backup = _quarantine(path)
        return LoadResult(
            status="corrupt",
            note=f"会话文件里的 summary 不是文本，已备份到 {backup.name}，这次从零开始",
        )

    saved_at = data.get("saved_at")
    age: float | None = None
    if isinstance(saved_at, str):
        try:
            age = (datetime.now() - datetime.fromisoformat(saved_at)).total_seconds()
        except ValueError:
            age = None

    return LoadResult(
        status="restored",
        note="会话已恢复",
        messages=messages,
        summary=summary,
        saved_at=saved_at if isinstance(saved_at, str) else None,
        age_seconds=age,
    )


def delete_session(path: Path) -> bool:
    """删掉会话文件（/reset 用）。返回是否真的删了东西。"""
    if path.is_file():
        path.unlink()
        return True
    return False


def _quarantine(path: Path) -> Path:
    """把一个坏文件改名备份，返回新路径。

    为什么是「改名」而不是「删除」：里面可能是用户聊了一下午的内容，
    程序没有资格替人做主扔掉它。改名之后，人还能自己去看一眼。
    """
    backup = path.with_name(path.name + CORRUPT_SUFFIX)
    try:
        os.replace(path, backup)
    except OSError:
        pass  # 连改名都失败（比如权限问题），那就保持原样，别把程序搞崩
    return backup


# ══════════════════════════════════════════════════════════════════════
# 给界面用的小工具
# ══════════════════════════════════════════════════════════════════════


def human_size(num_bytes: int) -> str:
    """把字节数写成人话。"""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / 1024 / 1024:.1f} MB"


def describe_age(seconds: float | None) -> str:
    """把「多少秒之前」写成人话。"""
    if seconds is None:
        return "未知"
    if seconds < 60:
        return f"{seconds:.0f} 秒前"
    if seconds < 3600:
        return f"{seconds / 60:.0f} 分钟前"
    if seconds < 86400:
        return f"{seconds / 3600:.1f} 小时前"
    return f"{seconds / 86400:.1f} 天前"
