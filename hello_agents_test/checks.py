"""极简断言器 —— 只做一件事：把「哪一项通过、为什么、失败在哪」讲清楚。

设计原则（照抄 AGENTS.md 铁律 7「自证清白」）：
- 每一项都打 ✓ / ✗，不留沉默；
- 失败时必须给出「实际拿到了什么」，而不是只说「断言失败」；
- 结尾汇总：通过数 / 失败数 / 逐条失败清单。
"""

import contextlib
import io
from typing import Iterator


@contextlib.contextmanager
def quiet() -> Iterator[io.StringIO]:
    """把被测代码的 print 吞进缓冲区。

    为什么需要它：ReAct / Reflection 这些 Agent 会往终端打大量日志，
    直接跑会把测试报告淹掉。要检查日志内容时，用 `with quiet() as buf:` 之后再读 buf。
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


class Check:
    def __init__(self, title: str = ""):
        self.title = title
        self.passed = 0
        self.failed = 0
        self.failures: list[str] = []
        self.known: list[str] = []      # 已知缺陷：单独记，不算失败
        self._section = "(未分组)"

    # ---------- 输出 ----------
    def section(self, title: str) -> None:
        self._section = title
        print(f"\n{'─' * 68}")
        print(f"▶ {title}")
        print("─" * 68)

    def ok(self, cond, msg: str, detail: str = "") -> bool:
        cond = bool(cond)
        if cond:
            self.passed += 1
            print(f"  ✓ {msg}")
        else:
            self.failed += 1
            line = f"[{self._section}] {msg}"
            self.failures.append(line + (f" —— {detail}" if detail else ""))
            print(f"  ✗ {msg}")
            if detail:
                print(f"      ↳ 实际：{detail}")
        return cond

    def eq(self, actual, expected, msg: str) -> bool:
        return self.ok(actual == expected, msg, f"得到 {actual!r}，期望 {expected!r}")

    def known_defect(self, msg: str, detail: str = "") -> None:
        """记录一个「已确认存在、但本次不修」的缺陷。

        为什么要单独记？因为「跳过」和「通过」是两回事 ——
        沉默会让人以为一切正常（AGENTS.md 铁律 7）。
        """
        self.known.append(msg + (f" —— {detail}" if detail else ""))
        print(f"  ⚠ 已知缺陷：{msg}")
        if detail:
            print(f"      ↳ {detail}")

    def skip(self, msg: str, reason: str) -> None:
        print(f"  ⏭ 跳过：{msg}")
        print(f"      ↳ 原因：{reason}")

    # ---------- 汇总 ----------
    def summary(self) -> int:
        print(f"\n{'=' * 68}")
        print(f"汇总：通过 {self.passed} 项，失败 {self.failed} 项，已知缺陷 {len(self.known)} 项")
        if self.known:
            print(f"\n⚠ 已知缺陷清单（本项目当前状态，非本次引入）：")
            for i, k in enumerate(self.known, 1):
                print(f"   {i}. {k}")
        if self.failures:
            print(f"\n✗ 失败清单：")
            for i, f in enumerate(self.failures, 1):
                print(f"   {i}. {f}")
            print(f"\n结论：有 {self.failed} 项不符合预期。")
        else:
            print(f"\n结论：全部符合预期{'（含上述已知缺陷）' if self.known else ''}。")
        print("=" * 68)
        return 1 if self.failed else 0


def expect_raise(exc_type, func, *args, **kwargs):
    """调用 func，期待它抛出 exc_type；返回异常实例，没抛则返回 None。

    用途：验证「抽象类不能实例化」这类**靠异常表达**的规则。
    """
    try:
        func(*args, **kwargs)
    except exc_type as e:
        return e
    except Exception as e:                      # 抛了别的异常 = 规则不对
        return ("WRONG_TYPE", e)
    return None
