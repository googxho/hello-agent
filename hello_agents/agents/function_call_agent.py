# ===== 文件导读 =====
# [职责] FunctionCallAgent：用 **OpenAI 原生 function calling** 调工具的 Agent，是 simple_agent.py 的对照版本。
# [位置] 继承 core/agent.py 的 Agent（实现 run / stream_run）；工具清单来自 tools/registry.py；
#        调模型时不走 core/llm.py 的 think / invoke，而是直接摸底层 self.llm._client 发原始请求（见 _invoke_with_tools）。
# [阅读顺序] ① _map_parameter_type + _build_tool_schemas（Python 工具怎么翻译成 JSON Schema）
#            → ② _invoke_with_tools（tools= / tool_choice= 这两个参数是什么意思）
#            → ③ _parse_function_call_arguments（模型回的 arguments 是「字符串形式的 JSON」）
#            → ④ run（多轮循环 + tool_call_id 配对）→ ⑤ 其余方法可快速跳读。
# [一句话] 和 SimpleAgent 的根本差别：那边靠「提示词约定文本格式 + 正则解析」，
#          这边把工具结构交给**协议本身**——模型直接返回结构化的 tool_calls，不用正则去猜。
# ===================
"""FunctionCallAgent - 使用OpenAI函数调用范式的Agent实现"""

# [语法] `from __future__ import annotations` 让本文件所有类型注解「只存成字符串、运行时不求值」，
#        所以下面 Optional["ToolRegistry"] 的引号其实可以不写；少了这一行，运行时求值这个注解就会 NameError。
from __future__ import annotations

import json
from typing import Iterator, Optional, Union, TYPE_CHECKING, Any, Dict

from ..core.agent import Agent
from ..core.config import Config
from ..core.llm import HelloAgentsLLM
from ..core.message import Message

# [语法] TYPE_CHECKING 运行时恒为 False（只在 mypy / pyright 里为真），用来写「只给类型检查器看」的导入。
if TYPE_CHECKING:
    from ..tools.registry import ToolRegistry


# [作用] 把工具自己声明的参数类型（"int" / "str" / "float" / 其它任何字符串）归一化成 JSON Schema 认识的那 6 种。
# [概念] JSON Schema 的类型是**封闭枚举**：只有 string / number / integer / boolean / array / object。
#        它不是 Python 类型系统，而是「发给模型看的数据形状说明书」，所以必须先翻译成它认得的那几个词。
def _map_parameter_type(param_type: str) -> str:
    """将工具参数类型映射为JSON Schema允许的类型"""
    # [易错] `param_type or ""` 是为了挡住 None；后面的 .lower() 才做大小写归一 —— 少了它 "String" 会落到兜底分支。
    normalized = (param_type or "").lower()
    # [概念] 白名单校验：命中就原样放行，不命中就返回 "string"。
    # [为什么] 这里宁可「猜错成 string」也不抛异常 —— schema 只是给模型看的描述，翻译失败不该让整个 Agent 挂掉。
    if normalized in {"string", "number", "integer", "boolean", "array", "object"}:
        return normalized
    # [实测] 这是个**静默降级**：ToolParameter(type="int") 实测会被翻译成 "string"
    #        （因为白名单里只有 "integer"，没有 "int"），模型因此看到的是「这个参数是字符串」。
    # [易错] 更别扭的是后面 _convert_parameter_types 认 "int" 并会把它转成 int —— 两处对类型名的认知并不一致。
    return "string"


# [作用] 函数调用版 Agent：构造时存好工具表、tool_choice 默认值和循环轮数上限。
# [概念] 它和 SimpleAgent 的区别不在「能不能调工具」，而在**用什么当接口**：
#        SimpleAgent 传的是自然语言的格式说明，本类传的是 tools=[JSON Schema]。
class FunctionCallAgent(Agent):
    """基于OpenAI原生函数调用机制的Agent"""

    # [参数] tool_choice 默认 "auto"（让模型自己决定调不调）；max_tool_iterations 是 run 里 while 的安全阀。
    # [易错] enable_tool_calling 又被 tool_registry 与了一次（和 SimpleAgent 同一处设计）：
    #        只传 enable_tool_calling=True 却忘了传 registry，工具会**静默失效**，不报错。
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: Optional["ToolRegistry"] = None,
        enable_tool_calling: bool = True,
        default_tool_choice: Union[str, dict] = "auto",
        max_tool_iterations: int = 3,
    ):
        # [机制] super().__init__ 负责建好 self.name / self.llm / self.config / self._history；
        #        子类重写 __init__ 时最容易漏掉这一句，漏了则父类属性全不存在。
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        # [语法] and 短路求值：左边为假就直接返回左边，所以没给 registry 时这行必定是 False，不会拿着 None 去调工具。
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None
        self.default_tool_choice = default_tool_choice
        self.max_tool_iterations = max_tool_iterations

    # [作用] 拼出 system 消息：这里写工具清单，是给模型看的**自然语言补充说明**，不是协议要求。
    # [概念] 原生 function calling 有两条互不依赖的通道：tools= 里的 JSON Schema 才是机器读的「工具定义」，
    #        这段提示词只是补充语境 —— 把它删掉，工具照样能被调用（对比 SimpleAgent：那边没有这段提示词就彻底调不动）。
    def _get_system_prompt(self) -> str:
        """构建系统提示词，注入工具描述"""
        base_prompt = self.system_prompt or "你是一个可靠的AI助理，能够在需要时调用工具完成任务。"

        if not self.enable_tool_calling or not self.tool_registry:
            return base_prompt

        # [概念] get_tools_description() 会把函数式工具也列进来；本文件后面的 schema 也覆盖了它们，
        #        所以「提示词里说到的工具」和「真正能被调用的工具」这里是对得上的。
        tools_description = self.tool_registry.get_tools_description()
        if not tools_description or tools_description == "暂无可用工具":
            return base_prompt

        prompt = base_prompt + "\n\n## 可用工具\n"
        prompt += "当你判断需要外部信息或执行动作时，可以直接通过函数调用使用以下工具：\n"
        prompt += tools_description + "\n"
        prompt += "\n请主动决定是否调用工具，合理利用多次调用来获得完备答案。"
        return prompt

    # [作用] ⭐ 本文件的核心：把注册表里的每个工具，翻译成 OpenAI 要求的 JSON Schema 列表（就是请求体里的 tools=）。
    # [概念] 「结构化输出」的关键一步：给模型的不是「你可以用 add(a, b)」这句话，而是一份带类型的形状说明。
    # [机制] 这份 schema 由服务端塞进模型的对话模板（各家的模板不同，通常是特殊 token 包起来的一段工具定义），
    #        模型是在「已经看到工具定义」的上下文里，被训练成直接吐出 tool_calls 结构；
    #        至于它是靠采样约束还是纯靠提示，由服务商实现决定 —— 别把它当成本地的参数校验器。
    def _build_tool_schemas(self) -> list[dict[str, Any]]:
        # [易错] 工具被关掉、或压根没注册表时返回空列表；run() 靠 `if not tool_schemas` 决定要不要降级成普通聊天。
        if not self.enable_tool_calling or not self.tool_registry:
            return []

        # [语法] 变量注解（annotation）：只声明类型、不赋值 —— Python 不会因此真建出一个 list，下面照样要赋值。
        schemas: list[dict[str, Any]] = []

        # Tool对象
        # [概念] 工具来自两个地方：Tool 对象（get_all_tools）和 register_function 注册的裸函数（内部字典 _functions）。
        #        两者形状完全不同，所以下面分两段、用两套写法翻译。
        for tool in self.tool_registry.get_all_tools():
            # [概念] properties 是「参数名 → 这个参数长什么样」的字典，required 是「哪些参数必填」的列表；
            #        两个键合起来才描述得清一个函数的入参 —— 少写 required，几乎所有服务端都会当成「全部可选」。
            properties: Dict[str, Any] = {}
            required: list[str] = []

            # [易错] get_parameters() 是各工具自己的实现，可能抛异常；这里吞掉异常、当作「没有参数」，
            #        属于有意的容错：一个工具的元数据坏了，不该让整个 Agent 起不来。
            try:
                parameters = tool.get_parameters()
            except Exception:
                parameters = []

            for param in parameters:
                properties[param.name] = {
                    # [概念] type 必须走 _map_parameter_type 归一化；description 是写给模型看的，直接影响它填什么值。
                    "type": _map_parameter_type(param.type),
                    "description": param.description or ""
                }
                # [概念] JSON Schema 的 "default" **只是文档**：服务端不会拿它替你补值，模型可能参考它，
                #        真正兜底的是工具函数自己的默认参数。
                if param.default is not None:
                    properties[param.name]["default"] = param.default
                # [实测] ToolParameter.required 的默认值本来就是 True，所以「没显式写 required 的参数」全都进 required 列表
                #        （实测：只声明 required=False 的 note 才被排除）——「有默认值」并不会自动变成可选，别和上一句搞混。
                if getattr(param, "required", True):
                    required.append(param.name)

            # [概念] 整个 schema 是三层：type=function → function.name / description → function.parameters（一个 JSON Schema 对象）。
            #        这层外壳由 OpenAI 规定死，写错层级服务端会直接 400。
            schema: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": {
                        "type": "object",
                        "properties": properties
                    }
                }
            }
            # [概念] 这个键是「有才加」：若一个工具的参数全部 required=False，发出去的 schema 里就没有 required 键，
            #        对模型来说等于「这些参数都可选」——和 JSON Schema 的默认语义正好对得上，不会漏掉必填信息。
            if required:
                schema["function"]["parameters"]["required"] = required
            schemas.append(schema)

        # register_function 注册的工具（直接访问内部结构）
        # [易错] `_functions` 是 ToolRegistry 的私有属性，这里用 getattr 兜底成 {}：上游一旦改名不会崩，
        #        但这些函数式工具会**静默消失**（模型再也看不到它们，而且没有任何日志）。
        function_map = getattr(self.tool_registry, "_functions", {})
        for name, info in function_map.items():
            # [概念] 函数式工具统一按「一个叫 input 的字符串参数」来描述，这是 registry 层定下的约定
            #        （ToolRegistry 本身就是按 func(input_text) 调的），不是这里临时猜的。
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": info.get("description", ""),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "input": {
                                    "type": "string",
                                    "description": "输入文本"
                                }
                            },
                            "required": ["input"]
                        }
                    }
                }
            )

        return schemas

    # [作用] 从 message.content 里安全抠出文本。不同服务商给的东西形状不一样，可能是 None、字符串，也可能是「内容块」列表。
    # [概念] 这个防御层本身就是证据：所谓「OpenAI 兼容接口」，各家返回的**形状**并不完全一致，写 Agent 必须做防御解析。
    @staticmethod
    def _extract_message_content(raw_content: Any) -> str:
        """从OpenAI响应的message.content中安全提取文本"""
        # [易错] 模型只调工具、不写正文时，content 就是 None；直接拿去做字符串拼接会 TypeError，所以先归一成 ""。
        if raw_content is None:
            return ""
        if isinstance(raw_content, str):
            return raw_content
        if isinstance(raw_content, list):
            # [概念] 多模态 / 新版接口下 content 是一个「块」列表（如 [{"type": "text", "text": "..."}]），
            #        需要把每块的文本拼起来；这是「同一份协议在不同版本里长得不同」的典型例子。
            parts: list[str] = []
            for item in raw_content:
                # [语法] getattr(x, "text", None)：属性不存在时返回默认值而不是抛 AttributeError —— 鸭子类型的常用手法。
                text = getattr(item, "text", None)
                # [概念] 块既可能是对象（新版 SDK 的 pydantic 模型），也可能是普通 dict（老接口），所以两条路都要试。
                if text is None and isinstance(item, dict):
                    text = item.get("text")
                if text:
                    parts.append(text)
            return "".join(parts)
        return str(raw_content)

    # [作用] ⭐ 把模型给的 arguments 变成真正能用的 dict —— 本文件最高频的踩坑点就在这里。
    # [易错] 看参数类型就明白了：arguments 是 Optional[**str**]，它是**字符串形式的 JSON**，
    #        形如 '{"a": 12, "b": "x"}'，而不是 dict。直接当字典用（arguments["a"]、arguments.get(...)）必炸。
    #        这是协议决定的：OpenAI 接口里 function.arguments 本来就是个字符串字段。
    @staticmethod
    def _parse_function_call_arguments(arguments: Optional[str]) -> dict[str, Any]:
        """解析模型返回的JSON字符串参数"""
        # [实测] 三种空值都返回 {} 而不是 None：arguments=None → {}，arguments=""（无参函数）→ {}，
        #        这样下游才敢直接 .get("input") 而不必先判空。
        if not arguments:
            return {}

        try:
            parsed = json.loads(arguments)
            # [易错] 模型偶尔会返回数组或裸字符串（'[]'、'"abc"'），json.loads 照样成功但类型不对；
            #        这里只认 dict，其它一律当「没有参数」——否则 _execute_tool_call 会以更奇怪的方式炸。
            return parsed if isinstance(parsed, dict) else {}
        # [易错] 模型可能生成**不合法**的 JSON（少引号、尾逗号、被长度截断）；实测 '{"a": 1,}' 会抛 JSONDecodeError。
        #        这里吞掉并返回 {} —— 后果是「工具被用空参数调了一次」，失败现场转移到工具内部，比较难查。
        except json.JSONDecodeError:
            return {}

    # [作用] 按工具声明的类型，把参数值从 JSON 原生类型转成工具真正想要的类型（float / int / bool）。
    # [概念] 为什么必须转：JSON 里 12 和 "12" 是两种东西，模型给你哪一种并不保证，而工具内部要拿它算数。
    #        这类「在边界处做类型转换」是解析层的标准职责（SimpleAgent 里也有一个同名方法，思路一致）。
    def _convert_parameter_types(self, tool_name: str, param_dict: dict[str, Any]) -> dict[str, Any]:
        """根据工具定义尽可能转换参数类型"""
        if not self.tool_registry:
            return param_dict

        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return param_dict

        try:
            tool_params = tool.get_parameters()
        except Exception:
            return param_dict

        # [语法] 字典推导式：{键表达式: 值表达式 for 元素 in 可迭代对象}，这里把参数列表压成「名字 → 类型」的查找表。
        type_mapping = {param.name: param.type for param in tool_params}
        converted: dict[str, Any] = {}

        for key, value in param_dict.items():
            # [易错] 这里不写 `if not param_type` 会怎样：`param_type.lower()` 对 None 会 AttributeError。
            #        另外它只是「放行」，不做剔除 —— 工具不认识的参数会原样传下去，可能引发 TypeError。
            param_type = type_mapping.get(key)
            if not param_type:
                converted[key] = value
                continue

            try:
                normalized = param_type.lower()
                if normalized in {"number", "float"}:
                    converted[key] = float(value)
                elif normalized in {"integer", "int"}:
                    converted[key] = int(value)
                elif normalized in {"boolean", "bool"}:
                    # [概念] 布尔要分四种来源处理：真正的 bool、0/1 这类数字、字符串、其它对象。
                    # [易错] 字符串判断用白名单 `in {"true", "1", "yes"}`，**绝不能**写成 bool(value)：
                    #        非空字符串恒为真，模型回 "false" 时 bool("false") 会得到 True —— 最经典的布尔坑。
                    if isinstance(value, bool):
                        converted[key] = value
                    elif isinstance(value, (int, float)):
                        converted[key] = bool(value)
                    elif isinstance(value, str):
                        converted[key] = value.lower() in {"true", "1", "yes"}
                    else:
                        converted[key] = bool(value)
                else:
                    converted[key] = value
            # [易错] 转不动就静默保留原值 —— 错误被推迟到工具内部才爆出来，排查时看不到「其实是转换失败」这一层。
            except (TypeError, ValueError):
                converted[key] = value

        return converted

    # [作用] 执行一次工具调用：先按 Tool 对象查，查不到再按函数式工具查，两路都失败才报「未找到」。
    # [返回] 永远返回字符串（连错误也是字符串），从不抛异常 —— 每个 except 都被转成了 ❌ 开头的结果。
    def _execute_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """执行工具调用并返回字符串结果"""
        if not self.tool_registry:
            return "❌ 错误：未配置工具注册表"

        tool = self.tool_registry.get_tool(tool_name)
        if tool:
            try:
                # [概念] 顺序很重要：先用工具的元数据把类型转好，再交给 tool.run —— 工具本身假定拿到的是正确类型。
                typed_arguments = self._convert_parameter_types(tool_name, arguments)
                return tool.run(typed_arguments)
            except Exception as exc:
                return f"❌ 工具调用失败：{exc}"

        # [概念] 这一路是相对 SimpleAgent 的实打实修复：那边只查 get_tool()，函数式工具会「提示词里有、
        #        真调用时说找不到」；这里补上了 get_function()，两类工具都能执行。
        func = self.tool_registry.get_function(tool_name)
        if func:
            try:
                # [概念] 函数式工具的约定是「只吃一个字符串」，和上面 _build_tool_schemas 里只声明 input 是对上的。
                input_text = arguments.get("input", "")
                return func(input_text)
            except Exception as exc:
                return f"❌ 工具调用失败：{exc}"

        # [为什么] 错误用字符串返回而不是抛异常：循环不会断，这句错误会被当成 tool 消息喂回模型，
        #        模型有机会自己换个说法重试；代价是它和真实结果长得一样，只能靠 ❌ 前缀区分。
        return f"❌ 错误：未找到工具 '{tool_name}'"

    # [作用] ⭐ 真正发请求的地方：它绕开 HelloAgentsLLM 的 invoke / think，直接拿底层 _client 调 create()。
    # [为什么] 因为 HelloAgentsLLM 的两个入口都没有留传 tools / tool_choice 的位置 —— 想发原生 function calling，
    #          只能自己补这两个参数；代价是跳过了那层封装（它打印的调用日志、异常包装都不会再出现）。
    # [参数] tools 是上一步拼好的 JSON Schema 列表；tool_choice 控制模型「可以自己决定 / 不许调 / 必须调」。
    # [机制] 取私有属性 _client 属于「穿透封装」：HelloAgentsLLM 若改了内部结构，这里会以 AttributeError
    #        或下面那句 RuntimeError 的形式炸掉 —— 这是这种写法的固有风险，本文件兜不住。
    def _invoke_with_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], tool_choice: Union[str, dict], **kwargs):
        """调用底层OpenAI客户端执行函数调用"""
        client = getattr(self.llm, "_client", None)
        if client is None:
            raise RuntimeError("HelloAgentsLLM 未正确初始化客户端，无法执行函数调用。")

        # [语法] dict(kwargs) 先复制一份：字典是可变对象、按引用传递，直接改 kwargs 会污染调用方的字典。
        client_kwargs = dict(kwargs)
        # [机制] setdefault 只在「键不存在」时写入：调用方显式传了 temperature 就用他的，没传才回落到 llm 上的默认值。
        client_kwargs.setdefault("temperature", self.llm.temperature)
        if self.llm.max_tokens is not None:
            client_kwargs.setdefault("max_tokens", self.llm.max_tokens)

        return client.chat.completions.create(
            model=self.llm.model,
            messages=messages,
            # [概念] tools= 告诉模型「有这些函数可用」，值就是 _build_tool_schemas 拼出来的那串 JSON Schema。
            tools=tools,
            # [概念] tool_choice 的三档语义（字符串写法）：
            #        "auto" = 模型自己决定调不调；"none" = 禁止调用，只能输出正文；"required" = 这一轮必须调工具。
            # [概念] 也可以传 dict（如 {"type": "function", "function": {"name": "calculator_multiply"}}）
            #        强制指定某一个函数，所以签名上写的是 Union[str, dict]。
            # [易错] "required" 是 OpenAI 原生才有的档位，不少兼容实现不认，会直接报参数错误 —— 换模型时要留意。
            tool_choice=tool_choice,
            **client_kwargs,
        )

    # [作用] ⭐ 主循环：反复「问模型 → 若它要调工具就执行并把结果回灌 → 再问」，直到它给出纯文本回答。
    # [参数] input_text 用户输入；max_tool_iterations 本次调用可覆盖构造时的轮数上限（None = 用默认值）；
    #        tool_choice 本次调用可覆盖默认的 "auto"。
    # [语法] 参数列表里那个单独的 `*` 表示「它后面的参数只能按关键字传」——
    #        防止调用方写成 run("你好", 5) 这种顺序错误，强迫写出 max_tool_iterations=5。
    # [返回] str 最终回答；[副作用] 会把本轮 user / assistant 消息写进 self._history。
    # [易错] 和 SimpleAgent.run 一样，这里是**每轮都重发全部历史**：轮数越多请求越大越贵。
    def run(
        self,
        input_text: str,
        *,
        max_tool_iterations: Optional[int] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        **kwargs,
    ) -> str:
        """
        执行函数调用范式的对话流程
        """
        messages: list[dict[str, Any]] = []
        system_prompt = self._get_system_prompt()
        messages.append({"role": "system", "content": system_prompt})

        # [概念] 模型本身不记事：所谓「记得上下文」就是把历史重发一遍 —— 这也是上下文会越来越贵的原因。
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": input_text})

        tool_schemas = self._build_tool_schemas()
        # [作用] 一个工具都没有（没注册 / 被关掉）时走最简路径：退回普通的 llm.invoke，一次拿完答案。
        # [易错] 这条分支里 while 根本不执行，所以 max_tool_iterations 和 tool_choice 在这条路上是摆设。
        if not tool_schemas:
            response_text = self.llm.invoke(messages, **kwargs)
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(response_text, "assistant"))
            return response_text

        # [语法] `A if 条件 else B` 三元表达式：这里表达的是「本次传了就用本次的，否则用构造时存的默认值」。
        iterations_limit = max_tool_iterations if max_tool_iterations is not None else self.max_tool_iterations
        # [概念] 变量名里的 effective 就是在点明优先级：**单次调用 > 构造参数 > 类里的默认值 "auto"**。
        effective_tool_choice: Union[str, dict] = tool_choice if tool_choice is not None else self.default_tool_choice

        current_iteration = 0
        final_response = ""

        # [作用] 每一圈循环 = 一次独立的模型调用。iterations_limit 是硬保险，防止模型陷在「调工具 → 再调工具」里烧钱。
        # [易错] 轮数用满 ≠ 任务完成：下面那个收尾分支只在「一次正文都没拿到」时才兜底再问一次。
        while current_iteration < iterations_limit:
            response = self._invoke_with_tools(
                messages,
                tools=tool_schemas,
                tool_choice=effective_tool_choice,
                **kwargs,
            )

            # [易错] 这里只看 choices[0]：请求里没传 n，正常只会有一个候选，其余候选（如果服务端给了）会被直接丢掉。
            choice = response.choices[0]
            assistant_message = choice.message
            content = self._extract_message_content(assistant_message.content)
            # [语法] `X or []`：字段是 None（这一轮没调工具）时用空列表顶上，下面 `if tool_calls:` 和 for 才安全；
            #        list(...) 只是把 SDK 给的序列再拷成普通列表。
            tool_calls = list(assistant_message.tool_calls or [])

            if tool_calls:
                # [概念] ⭐ 从这里开始消息流「结构化」了：把 SDK 对象**手工降级成 dict**，因为下一轮要原样发回去。
                # [为什么] assistant 这条消息承载着 tool_calls 的声明。不把它塞回 messages，后面那些 role="tool"
                #          的消息就成了「没有问题的答案」，服务端会直接报错。
                assistant_payload: dict[str, Any] = {"role": "assistant", "content": content}
                assistant_payload["tool_calls"] = []

                for tool_call in tool_calls:
                    assistant_payload["tool_calls"].append(
                        {
                            # [易错] id 是配对的唯一钥匙，必须原封不动带回去；漏掉或改写它，后面那条 tool 消息就对不上号。
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.function.name,
                                # [易错] arguments 原样就是**字符串**，回传时也必须是字符串 ——
                                #        千万别在这里先 json.loads 再塞回去，那会变成「对象套对象」被服务端拒绝。
                                "arguments": tool_call.function.arguments,
                            },
                        }
                    )
                messages.append(assistant_payload)

                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    # [易错] ⭐ 到这一步 arguments 仍然只是字符串，必须先过 _parse_function_call_arguments
                    #        （内部 json.loads）变成 dict，才能交给 _execute_tool_call。
                    arguments = self._parse_function_call_arguments(tool_call.function.arguments)
                    result = self._execute_tool_call(tool_name, arguments)
                    # [概念] 两个 for 是分开的：先把整条 assistant（含全部 tool_calls）放进对话，再逐条追加结果。
                    #        这不是风格问题，是协议要求——每条 tool 消息都必须能对上前面某条 tool_calls。
                    messages.append(
                        {
                            # [概念] role="tool" 是协议里的**第四种角色**（system / user / assistant / tool），
                            #        专门承载工具返回值 —— 这正是它比 SimpleAgent「把结果塞进 user 消息」更不容易混淆的地方。
                            "role": "tool",
                            # [易错] ⭐ tool_call_id 必须与本轮 assistant.tool_calls 里的 id **一一配对**：
                            #        少一条、多一条、或 id 对不上，服务端都会 400（大意是「tool 消息必须回应前面的 tool_calls」）。
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": result,
                        }
                    )

                # [概念] 一次工具调用算一轮，continue 回到 while 顶部再问模型 —— 是**这段代码**在替模型排队：
                #        模型是无状态的，它不会「等着」工具跑完，每次调用对它来说都是全新的。
                current_iteration += 1
                continue

            # [作用] 这一轮模型没要求调工具 → 它认为可以直接回答了 → 收工。
            final_response = content
            messages.append({"role": "assistant", "content": final_response})
            break

        # [易错] 兜底分支：循环因「轮数用满」退出、且始终没拿到正文时，再问一次 —— 但这次 tool_choice="none"
        #        明确禁止它再调工具，逼它输出正文。代价是多花一次 API 调用。
        if current_iteration >= iterations_limit and not final_response:
            final_choice = self._invoke_with_tools(
                messages,
                tools=tool_schemas,
                # [概念] 这里就是 tool_choice="none" 的典型用法：tools= 照样传（schema 是现成的），
                #        但这一轮明确禁止再调用工具，逼模型把已经拿到的东西整理成正文。
                tool_choice="none",
                **kwargs,
            )
            final_response = self._extract_message_content(final_choice.choices[0].message.content)
            messages.append({"role": "assistant", "content": final_response})

        # [概念] 只有最终的 user 和 assistant 两条进历史；中间那一串 assistant.tool_calls / role="tool" 消息
        #        **不进** _history —— 下一轮对话模型看不到上次调过什么工具，只记得结论。
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_response, "assistant"))
        return final_response

    # [作用] 便利方法：没注册表就顺手建一个；遇到可展开的工具（典型是 MCP）先展开成多个子工具再注册。
    def add_tool(self, tool) -> None:
        """便捷方法：将工具注册到当前Agent"""
        if not self.tool_registry:
            from ..tools.registry import ToolRegistry

            self.tool_registry = ToolRegistry()
            # [易错] 只建了注册表就顺手把开关打开：构造时因为没传 registry 而被压成 False 的 enable_tool_calling，
            #        在这里被重新打开 —— 忘了传注册表这件事，代价被推迟到「以后加工具」时才还。
            self.enable_tool_calling = True

        # [概念] 鸭子类型判断：不查 isinstance，只问「你有没有 auto_expand 且为真」——
        #        于是任何实现了 get_expanded_tools() 的类都能参与展开，不需要继承某个基类。
        if hasattr(tool, "auto_expand") and getattr(tool, "auto_expand"):
            # [概念] 展开 = 把「一个带 action 参数的大工具」拆成「多个名字不同的小工具」，
            #        好处是模型的 JSON Schema 更简单：不必猜 action 该填什么字符串。
            expanded_tools = tool.get_expanded_tools()
            if expanded_tools:
                for expanded_tool in expanded_tools:
                    self.tool_registry.register_tool(expanded_tool)
                print(f"✅ MCP工具 '{tool.name}' 已展开为 {len(expanded_tools)} 个独立工具")
                return

        self.tool_registry.register_tool(tool)

    def remove_tool(self, tool_name: str) -> bool:
        if self.tool_registry:
            # [概念] 靠「注销前后的名字集合」对比来判断是否真的删掉了：因为 registry.unregister 本身不返回任何值
            #        （返回 None），直接拿它的返回值判断会永远得到假。
            before = set(self.tool_registry.list_tools())
            self.tool_registry.unregister(tool_name)
            after = set(self.tool_registry.list_tools())
            return tool_name in before and tool_name not in after
        return False

    def list_tools(self) -> list[str]:
        if self.tool_registry:
            return self.tool_registry.list_tools()
        return []

    def has_tools(self) -> bool:
        return self.enable_tool_calling and self.tool_registry is not None

    # [作用] run() 的流式版本 —— 但它**其实不流式**：先完整跑完 run()，再把整段结果一次性 yield 出去。
    # [语法] 函数体里有 yield，所以它是生成器函数：调用 stream_run(...) 不会立刻执行 run()，
    #        要等外面开始迭代（for / next）才真正发请求 —— 这一点和其他生成器完全一样。
    # [易错] 指望「一边生成一边看到字」的调用方会失望：实际是等到最后「啪」一下全出来。
    #        接口形状对了，体验是假的 —— 这也是它和 SimpleAgent.stream_run（真的逐片转发）的差距。
    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """流式调用暂未实现，直接回退到一次性调用"""
        result = self.run(input_text, **kwargs)
        yield result
