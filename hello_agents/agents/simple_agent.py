# ===== 文件导读 =====
# [职责] 最基础的 Agent 范式：一轮对话 + 可选的工具调用循环（用「文本协议」实现）。
# [位置] 继承 core/agent.py 的 Agent（实现抽象方法 run）；又被 ToolAwareSimpleAgent 继承（二级继承）；
#        依赖 core 三件套和 tools/registry.py。
# [阅读顺序] ① __init__（两个工具开关）→ ② _get_enhanced_system_prompt（工具怎么「告诉」模型）
#            → ③ _parse_tool_calls / _parse_tool_parameters（模型的话怎么变成函数参数）
#            → ④ run（主循环）→ ⑤ stream_run（生成器版本）。
# [一句话] 本文件的本质：**用提示词约定一个文本格式，再用正则把它解析回函数调用**。
# ===================
"""简单Agent实现 - 基于OpenAI原生API"""

# [语法] TYPE_CHECKING 是 typing 里的一个常量，运行时恒为 False —— 专门用来写「只在类型检查时才执行」的导入。
from typing import Optional, Iterator, TYPE_CHECKING
import re

# [语法] `..core.agent` 里两个点表示「上一层包」：本文件在 hello_agents.agents，上一层就是 hello_agents。
from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message

# [概念] 把导入放进 if TYPE_CHECKING: 里，运行时就不会真的加载 ToolRegistry，只有 mypy 这类工具看得到它。
# [语法] 配合它，下面的注解必须写成字符串 'ToolRegistry'（前向引用），否则运行时求值会 NameError。
# [机制] 实测 tools/ 并没有反向 import agents，所以这里不是「非躲不可」的循环导入，属于防御性写法。
if TYPE_CHECKING:
    from ..tools.registry import ToolRegistry

# [作用] 基础 Agent：支持多轮对话 + 可选的工具调用。
# [概念] 继承与履约：父类 Agent 用 @abstractmethod 记了账（__abstractmethods__ == {'run'}），
#        本类必须实现 run，否则实例化时就会抛 TypeError。
class SimpleAgent(Agent):
    """简单的对话Agent，支持可选的工具调用"""
    
    # [作用] 构造：比父类多两个工具相关参数。
    # [机制] 前 4 个参数（name/llm/system_prompt/config）原样转交给父类的 __init__，见下面 super().__init__。
    # [易错] tool_registry 是可选的：不传就是「纯聊天 Agent」，一个工具都不会有。
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: Optional['ToolRegistry'] = None,
        enable_tool_calling: bool = True
    ):
        """
        初始化SimpleAgent
        
        Args:
            name: Agent名称
            llm: LLM实例
            system_prompt: 系统提示词
            config: 配置对象
            tool_registry: 工具注册表（可选，如果提供则启用工具调用）
            enable_tool_calling: 是否启用工具调用（只有在提供tool_registry时生效）
        """
        # [机制] super().__init__(...) 调用父类 Agent 的初始化 —— 父类会在那里建好 self.name / self.llm /
        #        self.config / self._history。**子类重写 __init__ 时最容易忘的就是这一句**，忘了则父类属性全不存在。
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        # [语法] `A and B` 短路求值：A 为假就返回 A，不再看 B。所以 tool_registry 没给时，
        #        enable_tool_calling 一定被压成 False —— 这是个「自动关闸」，防止后面拿着 None 去调工具。
        # [易错] 用户传 enable_tool_calling=True 但忘了传 registry，结果静默变成不启用工具（不报错，只是不生效）。
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None
    
    # [作用] 构造最终发给模型的 system 消息：把「可用工具清单 + 调用格式说明」拼在原 system_prompt 后面。
    # [概念] 这就是 Agent 框架里最重要的一句话：**工具描述本身就是提示词**。
    #        模型不是「知道」有哪些工具，而是你在提示词里告诉它有哪些工具。
    def _get_enhanced_system_prompt(self) -> str:
        """构建增强的系统提示词，包含工具信息"""
        base_prompt = self.system_prompt or "你是一个有用的AI助手。"
        
        if not self.enable_tool_calling or not self.tool_registry:
            return base_prompt
        
        # 获取工具描述
        # [易错] 注意 get_tools_description() 会把「函数式工具」（register_function 注册的）也列进来，
        #        但下面 _execute_tool_call 只查 Tool 对象、查不到它们 —— 实测会出现「模型被告知有个工具，
        #        调用却报未找到」的错配。这里只报告，不改代码。
        tools_description = self.tool_registry.get_tools_description()
        if not tools_description or tools_description == "暂无可用工具":
            return base_prompt
        
        tools_section = "\n\n## 可用工具\n"
        tools_section += "你可以使用以下工具来帮助回答问题：\n"
        tools_section += tools_description + "\n"

        tools_section += "\n## 工具调用格式\n"
        tools_section += "当需要使用工具时，请使用以下格式：\n"
        tools_section += "`[TOOL_CALL:{tool_name}:{parameters}]`\n\n"

        tools_section += "### 参数格式说明\n"
        tools_section += "1. **多个参数**：使用 `key=value` 格式，用逗号分隔\n"
        tools_section += "   示例：`[TOOL_CALL:calculator_multiply:a=12,b=8]`\n"
        tools_section += "   示例：`[TOOL_CALL:filesystem_read_file:path=README.md]`\n\n"
        tools_section += "2. **单个参数**：直接使用 `key=value`\n"
        tools_section += "   示例：`[TOOL_CALL:search:query=Python编程]`\n\n"
        tools_section += "3. **简单查询**：可以直接传入文本\n"
        tools_section += "   示例：`[TOOL_CALL:search:Python编程]`\n\n"

        tools_section += "### 重要提示\n"
        tools_section += "- 参数名必须与工具定义的参数名完全匹配\n"
        tools_section += "- 数字参数直接写数字，不需要引号：`a=12` 而不是 `a=\"12\"`\n"
        tools_section += "- 文件路径等字符串参数直接写：`path=README.md`\n"
        tools_section += "- 工具调用结果会自动插入到对话中，然后你可以基于结果继续回答\n"

        return base_prompt + tools_section
    
    # [作用] 用正则从模型回复里抠出所有 [TOOL_CALL:工具名:参数] 标记。
    # [语法] re.findall 返回「元组列表」：模式里有几个捕获组，每个元组就有几项，
    #        所以下面 `for tool_name, parameters in matches` 是顺手解包。
    # [概念] 这是典型的「文本协议」：靠约定格式通信，解析全靠字符串处理，没有类型保障。
    def _parse_tool_calls(self, text: str) -> list:
        """解析文本中的工具调用"""
        # [语法] 正则 `\[TOOL_CALL:([^:]+):([^\]]+)\]`：\[ 转义方括号；[^:]+ 表示「一个或多个非冒号字符」；
        #        [^\]]+ 表示「一个或多个非右方括号字符」—— 用「排除法」表达参数里不能再出现 ]。
        # [易错] 因此参数里只要含有 ]（比如 JSON 数组）就会被截断 —— 这是这种格式天生的限制。
        pattern = r'\[TOOL_CALL:([^:]+):([^\]]+)\]'
        matches = re.findall(pattern, text)
        
        tool_calls = []
        for tool_name, parameters in matches:
            tool_calls.append({
                'tool_name': tool_name.strip(),
                'parameters': parameters.strip(),
                'original': f'[TOOL_CALL:{tool_name}:{parameters}]'
            })
        
        return tool_calls
    
    # [作用] 执行单个工具调用：查工具 → 解析参数 → 调用 → 包一层结果文本。
    # [易错] 它用的是 self.tool_registry.get_tool()（只查 Tool 对象），而不是 registry.execute_tool()
    #        （那个才会兜到函数式工具）。这就是上面那个错配的根因（实测第 2 个真 bug）。
    def _execute_tool_call(self, tool_name: str, parameters: str) -> str:
        """执行工具调用"""
        if not self.tool_registry:
            return f"❌ 错误：未配置工具注册表"

        try:
            # 获取Tool对象
            # [易错] 查不到就返回一条「错误字符串」而不是抛异常 —— 好处是循环不会断，坏处是错误被当成正常内容
            #        喂给了模型（模型只能自己读懂并道歉，实测它确实会说「工具不可用」）。
            tool = self.tool_registry.get_tool(tool_name)
            if not tool:
                return f"❌ 错误：未找到工具 '{tool_name}'"

            # 智能参数解析
            param_dict = self._parse_tool_parameters(tool_name, parameters)

            # 调用工具
            result = tool.run(param_dict)
            return f"🔧 工具 {tool_name} 执行结果：\n{result}"

        except Exception as e:
            return f"❌ 工具调用失败：{str(e)}"

    # [作用] 把模型写的一串参数，变成 dict。按优先级试三种格式：
    #        ① JSON（以 { 开头）② key=value（多个用逗号分隔）③ 纯文本（交给 _infer_simple_parameters）。
    # [概念] 这个「一个输入、三种解析策略、按顺序兜底」的写法，就是容错式解析的常见形态。
    def _parse_tool_parameters(self, tool_name: str, parameters: str) -> dict:
        """智能解析工具参数"""
        # [语法] 在函数体内部写 import（局部导入）：每次调用都会查一遍模块缓存。
        #        好处是把依赖推迟到真正用到时（省启动时间 / 避免模块级循环导入）；坏处是可能被重复执行到。
        import json
        param_dict = {}

        # 尝试解析JSON格式
        if parameters.strip().startswith('{'):
            try:
                param_dict = json.loads(parameters)
                # JSON解析成功，进行类型转换
                param_dict = self._convert_parameter_types(tool_name, param_dict)
                return param_dict
            except json.JSONDecodeError:
                # JSON解析失败，继续使用其他方式
                pass

        if '=' in parameters:
            # 格式: key=value 或 action=search,query=Python
            if ',' in parameters:
                # 多个参数：action=search,query=Python,limit=3
                pairs = parameters.split(',')
                for pair in pairs:
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        param_dict[key.strip()] = value.strip()
            else:
                # 单个参数：key=value
                key, value = parameters.split('=', 1)
                param_dict[key.strip()] = value.strip()

            # 类型转换
            param_dict = self._convert_parameter_types(tool_name, param_dict)

            # 智能推断action（如果没有指定）
            if 'action' not in param_dict:
                param_dict = self._infer_action(tool_name, param_dict)
        # [易错] 走到这个 else 的分支，参数就会变成 {'input': 纯文本}（见 _infer_simple_parameters）。
        #        于是「工具方法的形参名」必须正好叫 input 才对得上；内置工具的参数都叫 content/query/memory_id，
        #        所以它们只能靠模型写 key=value 才能工作（实测第 3 个真 bug）。
        else:
            # 直接传入参数，根据工具类型智能推断
            param_dict = self._infer_simple_parameters(tool_name, parameters)

        return param_dict

    # [作用] 按工具声明的参数类型，把字符串转成真正的类型（数字/布尔）。
    # [概念] 为什么必须转：模型写出来的一切都是文本，`a=12` 到这儿还是字符串 "12"，
    #        而工具内部要拿它做算术。这类「边界处做类型转换」是解析层的标准职责。
    def _convert_parameter_types(self, tool_name: str, param_dict: dict) -> dict:
        """
        根据工具的参数定义转换参数类型

        Args:
            tool_name: 工具名称
            param_dict: 参数字典

        Returns:
            类型转换后的参数字典
        """
        if not self.tool_registry:
            return param_dict

        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return param_dict

        # 获取工具的参数定义
        try:
            tool_params = tool.get_parameters()
        # [易错] 裸 except（不写异常类型）会连 KeyboardInterrupt 之类的都吞掉，还掩盖真实错误。
        #        这里安全是因为只是「拿不到参数定义就算了」，但生产代码里应写 except Exception。
        except:
            return param_dict

        # 创建参数类型映射
        param_types = {}
        for param in tool_params:
            param_types[param.name] = param.type

        # 转换参数类型
        converted_dict = {}
        for key, value in param_dict.items():
            if key in param_types:
                param_type = param_types[key]
                try:
                    # [作用] 数字和布尔各转各的：boolean 用 `value.lower() in ('true', '1', 'yes')` 判，
                    #        避开了 bool("false") == True 那个坑（字符串非空即为真）。
                    # [易错] 转换失败时（第 205 行）静默保持原字符串，错误会推迟到工具内部才爆出来。
                    if param_type == 'number' or param_type == 'integer':
                        # 转换为数字
                        if isinstance(value, str):
                            converted_dict[key] = float(value) if param_type == 'number' else int(value)
                        else:
                            converted_dict[key] = value
                    elif param_type == 'boolean':
                        # 转换为布尔值
                        if isinstance(value, str):
                            converted_dict[key] = value.lower() in ('true', '1', 'yes')
                        else:
                            converted_dict[key] = bool(value)
                    else:
                        converted_dict[key] = value
                except (ValueError, TypeError):
                    # 转换失败，保持原值
                    converted_dict[key] = value
            else:
                converted_dict[key] = value

        return converted_dict

    # [易错] 这里把工具名 'memory' / 'rag' 硬编码进了 Agent —— 说明「工具的参数推断」本不该由 Agent 负责。
    #        代价：每加一个需要推断的工具，都得回来改这个 if。这是典型的坏味道（知识放错了层）。
    def _infer_action(self, tool_name: str, param_dict: dict) -> dict:
        """根据工具类型和参数推断action"""
        if tool_name == 'memory':
            if 'recall' in param_dict:
                param_dict['action'] = 'search'
                param_dict['query'] = param_dict.pop('recall')
            elif 'store' in param_dict:
                param_dict['action'] = 'add'
                param_dict['content'] = param_dict.pop('store')
            elif 'query' in param_dict:
                param_dict['action'] = 'search'
            elif 'content' in param_dict:
                param_dict['action'] = 'add'
        elif tool_name == 'rag':
            if 'search' in param_dict:
                param_dict['action'] = 'search'
                param_dict['query'] = param_dict.pop('search')
            elif 'query' in param_dict:
                param_dict['action'] = 'search'
            elif 'text' in param_dict:
                param_dict['action'] = 'add_text'

        return param_dict

    # [作用] 纯文本参数的兜底推断：rag/memory 猜成搜索，其它工具统一塞进 {'input': 文本}。
    # [易错] 这个 {'input': ...} 是硬约定：工具方法想吃到纯文本，形参就必须命名成 input。
    def _infer_simple_parameters(self, tool_name: str, parameters: str) -> dict:
        """为简单参数推断完整的参数字典"""
        if tool_name == 'rag':
            return {'action': 'search', 'query': parameters}
        elif tool_name == 'memory':
            return {'action': 'search', 'query': parameters}
        else:
            return {'input': parameters}

    # [作用] 主入口：组装消息 → 调模型 → 若模型要求用工具就执行并把结果回灌 → 再调模型，直到它不再要求。
    # [参数] input_text 用户输入；max_tool_iterations 最多循环几轮（默认 3，防止模型反复调工具烧钱）。
    # [返回] str（最终回答）。注意同族的 stream_run() 返回的是生成器 —— 两者返回类型不同。
    # [副作用] 会把这一轮的问答追加进 self._history。
    def run(self, input_text: str, max_tool_iterations: int = 3, **kwargs) -> str:
        """
        运行SimpleAgent，支持可选的工具调用
        
        Args:
            input_text: 用户输入
            max_tool_iterations: 最大工具调用迭代次数（仅在启用工具时有效）
            **kwargs: 其他参数
            
        Returns:
            Agent响应
        """
        # [作用] 每次调用都重新拼一遍 messages：system（含工具说明）+ 全部历史 + 本次输入。
        # [概念] 模型本身不记事，所谓「记住上下文」就是每次都把历史重新发一遍 —— 这也是上下文会越来越贵的原因。
        # 构建消息列表
        messages = []
        
        # 添加系统消息（可能包含工具信息）
        enhanced_system_prompt = self._get_enhanced_system_prompt()
        messages.append({"role": "system", "content": enhanced_system_prompt})
        
        # 添加历史消息
        # [概念] Message 对象在这里被「降级」成普通 dict：说明模型层只认 role/content 两个键（对应 message.py 的 to_dict）。
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})
        
        # 添加当前用户消息
        messages.append({"role": "user", "content": input_text})
        
        # [作用] 没启用工具时走最简路径：一次调用、存历史、返回。
        # [易错] 这条分支里没有 while 循环，所以 max_tool_iterations 参数完全不生效。
        # 如果没有启用工具调用，使用原有逻辑
        if not self.enable_tool_calling:
            response = self.llm.invoke(messages, **kwargs)
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(response, "assistant"))
            return response
        
        # 迭代处理，支持多轮工具调用
        current_iteration = 0
        final_response = ""

        # [作用] 工具调用循环：每一轮都问一次模型，直到它给出不需要工具的最终回答。
        # [易错] 这是「花多次钱」的地方：模型每轮都要重新读完整对话，所以轮数越多越贵。
        while current_iteration < max_tool_iterations:
            # 调用LLM
            # [概念] 注意循环里是一问一答：先把模型的话解析成工具调用，执行，再带着结果问下一次。
            #        模型自己不会「等着」工具跑完 —— 是这段代码在替它排队。
            response = self.llm.invoke(messages, **kwargs)

            # 检查是否有工具调用
            # [概念] 判定「模型想用工具」靠的是正则匹配，而不是结构化字段（对比 function_call_agent.py 用原生 tool_calls）。
            tool_calls = self._parse_tool_calls(response)

            if tool_calls:
                # 执行所有工具调用并收集结果
                tool_results = []
                clean_response = response

                # 构建包含工具结果的消息
                # [概念] 把模型的原始回复（含 [TOOL_CALL:...] 原文）作为 assistant 消息存进对话，再追加工具结果。
                # [为什么] OpenAI 格式要求角色交替：assistant 说了什么、工具返回了什么，都得留在消息流里，
                #        模型才能把「我请求了工具」和「工具回了什么」对应起来。
                messages.append({"role": "assistant", "content": clean_response})

                for call in tool_calls:
                    result = self._execute_tool_call(call['tool_name'], call['parameters'])
                    tool_results.append(result)
                    # 从响应中移除工具调用标记
                    # [易错] clean_response 只出现在三处：上面赋值为 response、被 append 进对话、
                    #        以及下面那一句被削掉标记 —— 然后**再也没被用过**。也就是说这次「清理」是死代码，
                    #        对话里留下的仍是带 [TOOL_CALL:...] 的原文（可能诱导模型下一轮重复调用同一工具）。
                    clean_response = clean_response.replace(call['original'], "")



                # [作用] ⭐ 关键一步：把工具返回值包装成 **user 消息**塞回对话，并在末尾加一句「请基于这些结果给出完整的回答」。
                # [为什么] 这是「把工具结果喂回模型」的唯一通道 —— 模型只能通过消息流看到工具干了什么。
                # [易错] 用 user 角色而不是 tool 角色（OpenAI 原生 function calling 用的是 tool 角色），
                #        这是文本协议范式的将就之处，会让模型偶尔分不清「用户说的」和「工具说的」。
                # 添加工具结果
                tool_results_text = "\n\n".join(tool_results)
                messages.append({"role": "user", "content": f"工具执行结果：\n{tool_results_text}\n\n请基于这些结果给出完整的回答。"})

                current_iteration += 1
                continue

            # [作用] 模型这一轮没提工具 → 说明它认为可以直接回答了 → 收工。
            # 没有工具调用，这是最终回答
            final_response = response
            break

        # 如果超过最大迭代次数，获取最后一次回答
        # [易错] 兜底逻辑：如果循环用满了还没拿到 final_response，就再问一次模型要答案。
        #        这意味着「工具调用到上限」时，会额外多花一次 API 调用。
        if current_iteration >= max_tool_iterations and not final_response:
            final_response = self.llm.invoke(messages, **kwargs)
        
        # 保存到历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_response, "assistant"))

        return final_response

    # [作用] 便利方法：没建注册表就顺手建一个（注意这里的局部导入，见前文 TYPE_CHECKING 的说明）。
    def add_tool(self, tool, auto_expand: bool = True) -> None:
        """
        添加工具到Agent（便利方法）

        Args:
            tool: Tool对象
            auto_expand: 是否自动展开可展开的工具（默认True）

        如果工具是可展开的（expandable=True），会自动展开为多个独立工具
        """
        if not self.tool_registry:
            from ..tools.registry import ToolRegistry
            self.tool_registry = ToolRegistry()
            self.enable_tool_calling = True

        # 直接使用 ToolRegistry 的 register_tool 方法
        # ToolRegistry 会自动处理工具展开
        self.tool_registry.register_tool(tool, auto_expand=auto_expand)

    def remove_tool(self, tool_name: str) -> bool:
        """移除工具（便利方法）"""
        if self.tool_registry:
            return self.tool_registry.unregister_tool(tool_name)
        return False

    def list_tools(self) -> list:
        """列出所有可用工具"""
        if self.tool_registry:
            return self.tool_registry.list_tools()
        return []

    def has_tools(self) -> bool:
        """检查是否有可用工具"""
        return self.enable_tool_calling and self.tool_registry is not None

    # [作用] run() 的流式版本：一边收一个字一边往外 yield，用户能看到「打字机效果」。
    # [语法] 函数体里有 yield → 它也是生成器函数；调用它不会立刻发请求，要等外面开始迭代（见 core/llm.py 的说明）。
    # [易错] 它和 run() 是两份几乎重复的代码，但返回类型完全不同：一个 str、一个生成器。
    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """
        流式运行Agent
        
        Args:
            input_text: 用户输入
            **kwargs: 其他参数
            
        Yields:
            Agent响应片段
        """
        # 构建消息列表
        messages = []
        
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})
        
        messages.append({"role": "user", "content": input_text})
        
        # 流式调用LLM
        full_response = ""
        # [作用] stream_invoke 内部 yield from think()（再委托给 core/llm.py 的流式调用），逐片转交给调用方。
        # [概念] 累加 full_response 是为了收完后存历史：流式对外是碎片，对内仍要留下完整记录。
        for chunk in self.llm.stream_invoke(messages, **kwargs):
            full_response += chunk
            yield chunk
        
        # 保存完整对话到历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(full_response, "assistant"))
