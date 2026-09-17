#!/usr/bin/env python3
"""
hello-agent v1 —— 最小可运行的 Agent
================================================

一句话理解 Agent：

    Agent = LLM + 工具 + 循环 + 状态

LLM 本身只是一个「文本进、文本出」的函数。它不会算术、不知道今天几号、
更没有办法把东西写进磁盘。所谓 Agent 框架，做的就是：

    给这个函数配上几只手（工具），
    然后让它在循环里反复调用自己，直到它说「我不需要工具了」。

这份代码共分五段，建议从上往下依次读：

    §1 配置       —— 密钥、地址、模型名
    §2 工具协议   —— 怎么向 LLM「描述」一个工具
    §3 三个工具   —— 每个演示一个 LLM 做不到的能力
    §4 主循环     —— 全篇核心，只有约 40 行
    §5 命令行     —— 交互界面

运行方式：

    python agent.py              # 进入交互对话
    python agent.py --selftest   # 只测工具，不调 LLM（不需要 API Key）
    python agent.py --config     # 看配置从哪来、当前生效的值是什么
"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════════════
# §1 配置
# ══════════════════════════════════════════════════════════════════════

import ast
import json
import operator
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# ── 配置从哪来 ────────────────────────────────────────────────────────
# 密钥和模型配置统一放在【仓库根目录】的 .env 里，所有版本共用一份，
# 不必每个版本目录都复制一遍。
#
# 加载优先级（靠前的不会被后面的覆盖）：
#     1. 真实环境变量       LLM_MODEL=xxx python agent.py
#     2. 本目录的 .env      （可选）只放这个版本想覆盖的项
#     3. 仓库根的 .env      ← 平时只动这一个
_HERE = Path(__file__).resolve().parent


def _find_repo_root() -> Path:
    """从本文件所在目录往上找，直到找到含 .git 的仓库根。"""
    for candidate in _HERE.parents:
        if (candidate / ".git").exists():
            return candidate
    return _HERE.parent  # 找不到就退回上一级，尽力而为


# 两次都用默认的 override=False，这一条很关键：
#   · 真实环境变量永远不会被文件覆盖
#   · 仓库根的 .env 也不会覆盖本目录的 .env
load_dotenv(_HERE / ".env")
load_dotenv(_find_repo_root() / ".env")

# 单次提问内，最多允许「调用工具 → 回灌结果 → 再问一轮」循环多少次。
# 这是防止 Agent 陷入死循环、把你的 API 额度烧光的保险丝。
# 8 是业界常用的经验值，够复杂任务用，又不至于失控。
MAX_ROUNDS = 8

# save_note 工具写文件的目标目录
NOTES_DIR = Path(__file__).parent / "notes"

# 所有工具定义的注册表：  工具名 -> ToolDefinition
TOOL_DEFS: dict[str, "ToolDefinition"] = {}
# 所有工具实现的注册表：  工具名 -> 普通 Python 函数
TOOL_IMPLS: dict[str, Callable[..., str]] = {}

console = Console()


def _read_config() -> tuple[str, str, str]:
    """从环境变量里读出 (api_key, base_url, model)。

    这里故意显式读取、显式传给 OpenAI() 客户端，而不是依赖 SDK 自动读
    OPENAI_API_KEY 之类的环境变量 —— 让你一眼就能看见密钥和地址的去向。
    """
    api_key = os.getenv("LLM_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("LLM_BASE_URL", "").strip() or "https://api.deepseek.com"
    model = os.getenv("LLM_MODEL", "").strip() or "deepseek-chat"
    return api_key, base_url, model


API_KEY, BASE_URL, MODEL = _read_config()


# ══════════════════════════════════════════════════════════════════════
# §2 工具协议 —— 怎么把一个 Python 函数「介绍」给 LLM
# ══════════════════════════════════════════════════════════════════════
#
# 这是整个 Agent 里最容易被低估的部分。
#
# LLM 看不到你的代码，它只能读到一段 JSON Schema 文本。它完全靠这段文本
# 来判断：什么时候该用这个工具、参数该怎么填。所以 ——
#
#     description 和参数说明，本质上就是写给 LLM 的提示词。
#
# 写得含糊，模型就会该调用时不调用、或者填错参数。
# 写得清楚，模型的表现会立刻上一个台阶。


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
        """拼成 OpenAI / DeepSeek 等接口要求的 tools 数组元素格式。

        生成出来大概长这样：

            {
              "type": "function",
              "function": {
                "name": "calculator",
                "description": "计算一个数学表达式……",
                "parameters": {
                  "type": "object",
                  "properties": {
                    "expression": {"type": "string", "description": "……"}
                  },
                  "required": ["expression"]
                }
              }
            }
        """
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


def tool(definition: ToolDefinition) -> Callable[[Callable[..., str]], Callable[..., str]]:
    """装饰器：把「工具描述」和「工具实现」一起登记进注册表。

    这样写的好处是定义紧挨着实现，读代码时不会两头跳。
    注意 schema 仍然是手写的，没有任何自动生成 —— 因为这个项目就是要让你
    亲手体会「工具描述写得清不清楚」对 Agent 表现的影响。
    """

    def wrapper(fn: Callable[..., str]) -> Callable[..., str]:
        TOOL_DEFS[definition.name] = definition
        TOOL_IMPLS[definition.name] = fn
        return fn

    return wrapper


# ══════════════════════════════════════════════════════════════════════
# §3 三个工具 —— 每个演示一种「LLM 做不到，但工具能做到」的能力
# ══════════════════════════════════════════════════════════════════════

# ── 工具一：calculator ────────────────────────────────────────────────
#
# 演示的是【能力边界】。
# LLM 是概率模型，它「算」乘法其实是在猜下一个 token，长数字必然出错。
# 别指望用提示词治好，正确做法是给它一个计算器。

# 只允许这些运算，白名单机制 —— 绝对不要用裸 eval()，那等于把整个机器
# 交给模型为所欲为。
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


# ── 工具二：get_current_time ─────────────────────────────────────────
#
# 演示的是【知识边界】。
# 模型的训练数据有截止日期，它根本不知道「现在」是几点，也不知道今天星期几。
# 任何涉及当下时间的问题，都必须靠工具去问操作系统。

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


# ── 工具三：save_note ────────────────────────────────────────────────
#
# 演示的是【副作用能力】。
# 模型只能生成文本，它没有任何办法改变外部世界。工具就是它的「一双手」。
# 这是 Agent 从「聊天机器人」变成「数字员工」的分水岭 —— 它开始能干活了。


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
    return f"已保存到 notes/{slug}.md"


# ══════════════════════════════════════════════════════════════════════
# §4 主循环 —— 整个 Agent 的心脏，请重点读这一段
# ══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """你是一个可以通过调用工具来完成任务的助手。

工作规则：
1. 需要计算、需要知道当前时间、需要保存内容时，必须调用对应工具，
   不要凭记忆或心算作答。
2. 不需要工具时就直接回答，不要为了显得勤快而乱调工具。
3. 工具返回结果后，用自然语言把你的结论清楚地告诉用户。
4. 如果工具返回了错误信息，请读懂错误并修正参数后重试。
"""


def _assistant_message(message: Any) -> dict[str, Any]:
    """把 SDK 返回的 assistant 消息转成可以回灌给接口的 dict。

    有个坑：tool_calls 里的 arguments 必须是字符串，且不能为空。
    有些模型返回空串，直接回灌会被接口拒绝，所以这里补成 "{}"。

    另一个坑：带 tool_calls 的消息 content 允许为 null，但不带 tool_calls 的
    assistant 消息 content 不能为 null，否则下一轮请求会被接口拒绝。
    """
    payload: dict[str, Any] = {"role": "assistant", "content": message.content}
    if not message.tool_calls and not payload["content"]:
        payload["content"] = ""
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments or "{}",
                },
            }
            for call in message.tool_calls
        ]
    return payload


class Agent:
    """一个最小的 Agent。

    它需要维护的唯一状态就是 `self.messages` —— 一条不断变长的消息列表，
    也就是我们常说的「对话历史」或「上下文」。

    messages 的结构是：

        [0]  system    系统提示词，固定不变，永远在开头
        [1]  user      用户说的第一句话
        [2]  assistant 模型的回复（可能带 tool_calls）
        [3]  tool      工具执行的结果（必须紧跟在对应的 assistant 后面）
        [4]  assistant 模型看到工具结果后的第二轮回复
        ...  依此类推

    整部 Agent 的历史，就是这一条列表。
    """

    def __init__(self, client: OpenAI, model: str = MODEL) -> None:
        self.client = client
        self.model = model
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        # 把 §2 里登记的工具定义转成接口要的格式，每次请求都带上
        self.tools = [d.to_openai_schema() for d in TOOL_DEFS.values()]

    def reset(self) -> None:
        """清空对话历史，但保留 system 提示词。"""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # ── 核心：主循环 ────────────────────────────────────────────────
    def run(self, user_input: str) -> str:
        # 第一步：把用户输入追加进历史
        self.messages.append({"role": "user", "content": user_input})

        # 第二步：进入循环。每一轮都是「问模型一次」
        for _ in range(MAX_ROUNDS):
            message = self._call_llm(use_tools=True)

            # 模型这一轮的回复，无论有没有工具调用，都要记进历史
            self.messages.append(_assistant_message(message))

            # ★ 循环的终止条件：模型这一轮没有要求调用任何工具，
            #   说明它认为可以直接回答了。这就是「它自己决定何时收工」。
            if not message.tool_calls:
                return message.content or "（模型返回了空回答）"

            # 模型要求调工具 → 我们替它执行，把结果塞回历史，然后继续循环
            for call in message.tool_calls:
                self.messages.append(self._execute_tool_call(call))

        # 第三步：兜底。轮数用尽说明模型一直在调工具停不下来，
        # 这时把 tools 撤掉再问最后一次 —— 没有工具可用，它就只能用
        # 已经拿到手的资料给出答案，而不是继续死循环烧钱。
        console.print(f"[yellow]⚠ 已达最大轮数 {MAX_ROUNDS}，撤掉工具强制生成最终回答[/yellow]")
        message = self._call_llm(use_tools=False)
        return message.content or "（超出最大轮数，未能得出结论）"

    # ── 辅助：向模型发起一次请求 ────────────────────────────────────
    def _call_llm(self, *, use_tools: bool) -> Any:
        """调用模型一次，返回它的回复消息。

        use_tools=False 时请求里不带 tools 字段，模型就失去了调用工具的能力，
        被迫只能用文字回答。这是上面那个兜底逻辑的关键。
        """
        kwargs: dict[str, Any] = {"model": self.model, "messages": self.messages}
        if use_tools:
            kwargs["tools"] = self.tools
            kwargs["tool_choice"] = "auto"  # 让模型自己决定要不要用工具
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message

    # ── 辅助：执行一次工具调用 ──────────────────────────────────────
    def _execute_tool_call(self, call: Any) -> dict[str, Any]:
        """执行模型指定的那个工具，并把结果包装成 role="tool" 消息。

        注意这里最关键的一个设计：工具执行失败时，绝不能让程序崩掉，
        而是要把错误信息当成「工具的输出」交回给模型。

        因为对模型来说，报错也是一种信息 —— 它读到 "TypeError: 缺少参数
        content" 之后，往往会自己补上参数重试。Agent 的自我纠错能力，
        很大一部分就来自这 3 行代码。
        """
        name = call.function.name
        raw_args = call.function.arguments or "{}"

        console.print(f"  [cyan]🔧 调用[/cyan] [bold]{name}[/bold]({raw_args})")

        try:
            args = json.loads(raw_args)
            impl = TOOL_IMPLS.get(name)
            if impl is None:
                raise ValueError(f"不存在名为 {name} 的工具")
            result = str(impl(**args))
        except Exception as exc:
            result = f"错误：{type(exc).__name__}: {exc}"
            console.print(f"  [red]✗ {result}[/red]")
        else:
            console.print(f"  [green]← {result}[/green]")

        return {
            "role": "tool",
            "tool_call_id": call.id,  # 必须带上，用来和上面那条 assistant 消息配对
            "name": name,
            "content": result,
        }


# ══════════════════════════════════════════════════════════════════════
# §5 命令行界面
# ══════════════════════════════════════════════════════════════════════


def _error_hint(exc: Exception) -> str:
    """把常见的报错翻译成人话，省得初学者一头雾水。"""
    text = str(exc).lower()
    if "401" in text or "authentication" in text or "api key" in text:
        return "看起来是 API Key 无效。检查 .env 里的 LLM_API_KEY 是否填对了。"
    if "tool" in text and ("not support" in text or "unsupported" in text or "invalid" in text):
        return "当前模型可能不支持 tool calling。换一个支持函数调用的模型，例如 deepseek-chat / gpt-4o-mini / qwen-plus。"
    if "404" in text or ("model" in text and "not found" in text):
        return "模型名可能写错了，检查 .env 里的 LLM_MODEL。"
    if "connect" in text or "timeout" in text:
        return "连接不上模型服务，检查网络和 .env 里的 LLM_BASE_URL。"
    return ""


def _describe_config() -> str:
    """给 --config 用：配置从哪来、当前生效的值是什么。

    排查「为什么我的配置没生效」时，先看这里。
    """
    lines = ["配置文件（优先级从低到高）："]
    for path, note in (
        (_find_repo_root() / ".env", "共享层 · 全项目一份，放密钥等公共配置"),
        (_HERE / ".env", "本版本层 · 可选，放 v1 想单独覆盖的项"),
    ):
        exists = path.is_file()
        suffix = "" if exists else "   （不存在，这是正常的）"
        lines.append(f"  {'✓' if exists else '✗'}  {path}{suffix}")
        lines.append(f"       {note}")

    lines += [
        "",
        "本版本读取的配置项：",
        "",
        f"  {'LLM_API_KEY':<22} {'(已设置)' if API_KEY else '(未设置 ✗)':<18} [共享]",
        f"  {'LLM_BASE_URL':<22} {BASE_URL:<18} [共享]",
        f"  {'LLM_MODEL':<22} {MODEL:<18} [共享]",
        "",
        "v1 只读上面这三项。上下文管理等参数是 v2 才引入的能力，v1 不读。",
    ]
    return "\n".join(lines)


def _banner() -> Panel:
    return Panel(
        "[bold]hello-agent[/bold] v1 · 最小可运行 Agent\n"
        f"[dim]模型 {MODEL}  @  {BASE_URL}[/dim]\n"
        f"[dim]工具 {', '.join(TOOL_DEFS)}[/dim]\n\n"
        "[dim]/reset 清空对话历史    /exit 退出[/dim]",
        border_style="blue",
        title="Agent",
    )


def selftest() -> int:
    """不调用 LLM，直接执行每个工具，用来验证工具层本身有没有 bug。

    这是很有用的习惯：把「工具层」和「模型层」分开调试。
    工具本身都跑不通的时候，去怀疑模型是白费力气。
    """
    console.print("[bold]工具层自检（不调用 LLM）[/bold]\n")
    cases: list[tuple[str, dict[str, Any]]] = [
        ("calculator", {"expression": "(1234 * 5678) + 90"}),
        ("calculator", {"expression": "2 ** 10"}),
        ("calculator", {"expression": "1 / 0"}),  # 故意制造错误，看兜底是否生效
        ("get_current_time", {}),
        ("save_note", {"title": "selftest", "content": "这条笔记由 --selftest 生成，可以删掉。"}),
    ]
    for name, args in cases:
        impl = TOOL_IMPLS[name]
        arg_text = ", ".join(f"{k}={v!r}" for k, v in args.items())
        try:
            result = impl(**args)
        except Exception as exc:
            console.print(f"[cyan]{name}[/cyan]({arg_text})\n  [yellow]✗ {type(exc).__name__}: {exc}[/yellow]")
        else:
            console.print(f"[cyan]{name}[/cyan]({arg_text})\n  [green]← {result}[/green]")
    console.print("\n[green]✓ 工具层自检结束[/green]")
    return 0


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        return selftest()

    if "--config" in argv:
        # 用 Text 而不是字符串：配置报告里有 "[共享]" 这种方括号，
        # 直接交给 rich 会被当成样式标记解析掉。
        console.print(Panel(Text(_describe_config()), border_style="cyan", title="当前配置"))
        return 0

    if "--help" in argv or "-h" in argv:
        console.print(__doc__)
        return 0

    if not API_KEY:
        console.print(
            Panel(
                "[red]没有找到 API Key[/red]\n\n"
                "请先复制一份配置文件，然后填入你的密钥：\n\n"
                "  [bold]cp .env.example .env[/bold]\n\n"
                "编辑 .env，把 LLM_API_KEY 填上即可。\n\n"
                "[dim]没有密钥也可以先跑工具层自检：python agent.py --selftest[/dim]",
                border_style="red",
                title="缺少配置",
            )
        )
        return 1

    # 这里才是真正的 Agent 起点：一个指向模型服务的客户端
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    agent = Agent(client)

    console.print(_banner())

    while True:
        try:
            user_input = console.input("\n[bold green]你 > [/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n再见 👋")
            return 0

        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            console.print("再见 👋")
            return 0
        if user_input == "/reset":
            agent.reset()
            console.print("[dim]已清空对话历史[/dim]")
            continue

        try:
            reply = agent.run(user_input)
        except Exception as exc:
            console.print(f"\n[red]请求失败：{type(exc).__name__}: {exc}[/red]")
            hint = _error_hint(exc)
            if hint:
                console.print(f"[yellow]💡 {hint}[/yellow]")
            continue

        console.print(Panel(reply, border_style="blue", title="Agent"))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
