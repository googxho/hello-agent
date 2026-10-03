"""零成本实验：LangGraph 的 durability（持久化时机）三档，猝死时到底保住多少检查点？

    python durability-lab.py          # 三档对照（每档都在子进程里真猝死 os._exit）
    python durability-lab.py sync     # 只看一档：async / sync / exit

不需要任何 API Key —— 图里就是三个假节点 a → b → c。
唯一目的：搞清「检查点在什么时候落盘」——这是 checkpointer（断点恢复）
背后最容易被忽略、也最容易造成误解的一层。

背景（实测 + 源码，langgraph 1.2.12）：
    durability 默认值是 "async" —— 每个节点结束，状态快照是
    「排队交给后台线程写」。平时看不出差别；但当进程猝死（kill -9 /
    os._exit）时，还在途的快照可能没写完甚至只写一半。

预期结果：
    async：只剩 1~2 张（a、b 完成的快照几乎全丢——假图跑得太快）
    sync ：4 张全在（每步同步落盘后才继续——恢复点精确到「c 待执行」）
    exit ：0 张（只在图正常退出/中断时写——猝死 = 白干）
"""

import os
import sqlite3
import sys
from typing import Annotated, TypedDict
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

DB = "durability-lab.sqlite"


def _clean_db() -> None:
    # ⚠️ SQLite 除了主文件还可能有 -wal / -shm 辅助文件；
    # 只删主文件会让下个子进程撞上「旧 WAL + 新主文件」→ disk I/O error。
    for suffix in ("", "-wal", "-shm"):
        path = DB + suffix
        if os.path.exists(path):
            os.remove(path)


class S(TypedDict):
    n: Annotated[list, operator.add]


def a(state):  # noqa: ARG001
    return {"n": ["a"]}


def b(state):  # noqa: ARG001
    return {"n": ["b"]}


def c(state):  # noqa: ARG001
    print("  [子进程] c 节点执行中 → os._exit(1)（模拟猝死：等价 kill -9）", flush=True)
    os._exit(1)


def build():
    g = StateGraph(S)
    g.add_node("a", a)
    g.add_node("b", b)
    g.add_node("c", c)
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", END)
    return g


def child(mode: str) -> None:
    conn = sqlite3.connect(DB, check_same_thread=False)
    app = build().compile(checkpointer=SqliteSaver(conn))
    app.invoke({"n": []}, {"configurable": {"thread_id": "t1"}}, durability=mode)


def parent(modes: list[str]) -> None:
    print("图： START → a → b → c(猝死)")
    print("问题：c 猝死的那一刻，盘上保住了哪些检查点？\n")
    for mode in modes:
        _clean_db()
        os.system(f"{sys.executable} {__file__} child-{mode}")
        conn = sqlite3.connect(DB, check_same_thread=False)
        app = build().compile(checkpointer=SqliteSaver(conn))
        cfg = {"configurable": {"thread_id": "t1"}}
        hist = list(app.get_state_history(cfg))
        print(f"══ durability={mode!r}：猝死后盘上有 {len(hist)} 张检查点")
        for i, snap in enumerate(hist):
            print(
                f"    #{i} next={snap.next} 状态 n={snap.values.get('n')}"
                f" meta.step={snap.metadata.get('step')}"
            )
        print()
        conn.close()
    print("结论：")
    print("  async（默认）——后台异步写，猝死时「在途」的快照会丢（哪怕平时看起来一切正常）")
    print("  sync        ——每个超步同步落盘后才继续：最稳，代价是每步多一次写盘等待")
    print("  exit        ——只在正常退出/中断时写：进程猝死 = 这个线程什么都没留下")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg.startswith("child-"):
        child(arg.removeprefix("child-"))
    elif arg in {"async", "sync", "exit"}:
        parent([arg])
    else:
        parent(["async", "sync", "exit"])
