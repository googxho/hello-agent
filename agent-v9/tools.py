"""
三个工具 —— 原样来自 v1，每个演示一种「LLM 做不到，但工具能做到」的能力：

    calculator        → 能力边界（它算乘法其实是在猜）
    get_current_time  → 知识边界（它不知道「现在」）
    save_note         → 副作用能力（它能改变外部世界）

⚠️ v9 对这里只做了两处结构性调整（都是给评估让路，不加新工具）：
    1. NOTES_DIR / DISABLED_TOOLS 改为**调用时**从 config 读 ——
       不然测试里没法把笔记目录换到临时目录，会污染真实 notes/
    2. 加 openai_schemas() 作为「给模型的工具清单」的唯一来源 ——
       关掉工具 = 从清单里消失（v2 留下的老机制，评估制造「能力损失」要用）
"""

from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import config


# ══════════════════════════════════════════════════════════════════════
# 工具协议（v1 §2，逐字保留）
# ══════════════════════════════════════════════════════════════════════
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


# 注册表：  工具名 -> ToolDefinition / 工具名 -> 实现函数
TOOL_DEFS: dict[str, ToolDefinition] = {}
TOOL_IMPLS: dict[str, Callable[..., str]] = {}


def tool(definition: ToolDefinition) -> Callable[[Callable[..., str]], Callable[..., str]]:
    """装饰器：把「工具描述」和「工具实现」一起登记进注册表。"""

    def wrapper(fn: Callable[..., str]) -> Callable[..., str]:
        TOOL_DEFS[definition.name] = definition
        TOOL_IMPLS[definition.name] = fn
        return fn

    return wrapper


def openai_schemas() -> list[dict[str, Any]]:
    """给模型的工具清单 —— ⚠️ 这是唯一来源。

    被 LLM_DISABLED_TOOLS 点名的工具**根本不会出现在这里**，
    模型压根不知道它存在（v2 留下的机制）。
    """
    return [
        d.to_openai_schema()
        for name, d in TOOL_DEFS.items()
        if name not in config.DISABLED_TOOLS
    ]


def active_tool_names() -> list[str]:
    return [d["function"]["name"] for d in openai_schemas()]


# ══════════════════════════════════════════════════════════════════════
# 工具一：calculator（v1 逐字保留）
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
# 工具二：get_current_time（v1 逐字保留）
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
# 工具三：save_note（v1 逐字保留，只有目录改为动态读取）
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
    notes_dir = Path(config.NOTES_DIR)  # 调用时才读 —— 测试可以把目录换到临时位置
    notes_dir.mkdir(parents=True, exist_ok=True)
    # 清洗文件名：只保留字母、数字、下划线、中文和连字符。
    # 这一步同时也是安全措施 —— 冒号和斜杠会被替换掉，模型就无法用
    # "../../etc/passwd" 这种标题写到目录外面去。
    slug = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", title).strip("_") or "untitled"
    path = notes_dir / f"{slug}.md"
    path.write_text(f"# {title}\n\n{content}\n", encoding="utf-8")
    return f"已保存到 {path.name}"
