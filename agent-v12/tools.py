"""
三个工具 —— 和 v11 里**同一个定义**。

这本身就是组件标准化的意义：零件可以跨项目搬。
（v11 的 components.py 里演示过「25 行手写 → 3 行 @tool」；
 这里直接把 3 行的版本搬过来用。）

⚠️ docstring 依旧是「写给模型的提示词」——这个事实不因框架而改变。
"""

from __future__ import annotations

import ast
import operator
import re
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

import config

_ALLOWED_OPS: dict[type, object] = {
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


def _safe_eval(node: ast.AST):
    """白名单求值（v1 的老手艺，绝不用裸 eval）。"""
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


@tool
def calculator(expression: str) -> str:
    """计算一个数学表达式并返回精确结果。凡是涉及数字运算的问题都必须调用本工具，
    不要自己心算，否则很容易算错。支持 + - * / // % ** 以及括号。"""
    result = _safe_eval(ast.parse(expression, mode="eval"))
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


_WEEKDAYS = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


@tool
def get_current_time() -> str:
    """获取当前的本地日期和时间。凡是涉及『今天』『现在』『当前时间』的问题都必须调用
    本工具，不要凭记忆或猜测作答。"""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S") + f"（{_WEEKDAYS[now.weekday()]}）"


def slugify(title: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff-]+", "_", title).strip("_") or "untitled"


@tool
def save_note(title: str, content: str) -> str:
    """把一条笔记写入本地文件长期保存。当用户说『记下来』『记一下』『保存』『备忘』时
    调用本工具。title 会被用作文件名。"""
    path = config.NOTES_DIR / f"{slugify(title)}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{content}\n", encoding="utf-8")
    return f"已保存到 {path.name}"


TOOLS = [calculator, get_current_time, save_note]
TOOL_MAP = {t.name: t for t in TOOLS}
