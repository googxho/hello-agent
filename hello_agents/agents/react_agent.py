# ===== 文件导读 =====
# [职责] ReAct（Reasoning + Acting）范式的 Agent：让模型先写 `Thought:` 再写 `Action:`，
#        代码执行 Action 拿到 `Observation:`，再拼回提示词进入下一轮 —— 边想边做、做完再看。
# [位置] 继承 core/agent.py 的 Agent（实现抽象方法 run）；与 simple_agent.py 是同门不同流派的对照组：
#        本文件用**自维护的字符串历史** + 每轮重新 format 提示词，SimpleAgent 用 messages 列表累积对话。
# [阅读顺序] ① 模块常量 DEFAULT_REACT_PROMPT（先看懂协议长什么样）→ ② __init__（四个字段）
#            → ③ run（主循环：format → invoke → 解析 → 执行 → 回灌）
#            → ④ 三个解析方法 _parse_output / _parse_action / _parse_action_input。
# [一句话] 本文件的本质：**用提示词约定 Thought / Action / Observation 三段文本，再用正则把它们解析回来**。
# ===================
"""ReAct Agent实现 - 推理与行动结合的智能体"""

# [语法] 全文件只用到一个标准库模块 re：三个解析方法全靠正则，没有 JSON、也没有用原生 tool_calls 字段。
import re
# [语法] List[str] / Tuple[Optional[str], Optional[str]] 都只是**类型标注**：写在变量和返回值位置上，
#        运行时既不校验也不拦错；真正约束行为的只有代码本身。
from typing import Optional, List, Tuple
# [语法] 相对导入：开头的点表示「上一层包」——本文件在 hello_agents.agents，
#        所以 ..core.agent 指向 hello_agents/core/agent.py，而不是 site-packages 里可能同名的包。
from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message
# [概念] 注意这里的 ToolRegistry 是**运行时真导入**（simple_agent.py 里是塞进 TYPE_CHECKING 躲着的）：
#        因为 __init__ 要 `ToolRegistry()` 建空注册表，躲不掉。
from ..tools.registry import ToolRegistry

# [职责] 这一整块字符串就是 ReAct 的「协议说明书」：把输出格式（Thought / Action / Finish）写死在提示词里；
#        下面 run() 里的解析代码是这份协议的「读方」—— 两边必须严格对齐，改一边就得改另一边。
# 默认ReAct提示词模板
# [概念] ReAct = Reasoning + Acting：模型每轮只做两件事 —— 先 Thought（想），再 Action（做）。
# [概念] 它属于「文本协议」流派：不依赖任何原生 function calling 字段，格式全靠提示词约定 + 字符串解析。
# [概念] 三个占位符由下面 run() 的 str.format 填：{tools} 工具清单 / {question} 用户问题 / {history} 执行历史。
# [语法] 模板里的 `{{tool_name}}` 是**双花括号转义**：str.format 把 `{{` 还原成一个 `{`，
#        所以模型最终看到的是 `{tool_name}[{tool_input}]`（要字面花括号就得写两层，写一层会被当成占位符）。
# [易错] 实测：模板里出现未配对的单花括号会当场 KeyError —— 自定义提示词写 `{"answer": {question}}`，
#        报的是 KeyError: '"answer"'（它把 `"answer"` 整个当成了参数名去取值）。
DEFAULT_REACT_PROMPT = """你是一个具备推理和行动能力的AI助手。你可以通过思考分析问题，然后调用合适的工具来获取信息，最终给出准确的答案。

## 可用工具
{tools}

## 工作流程
请严格按照以下格式进行回应，每次只能执行一个步骤：

Thought: 分析问题，确定需要什么信息，制定研究策略。
Action: 选择合适的工具获取信息，格式为：
- `{{tool_name}}[{{tool_input}}]`：调用工具获取信息。
- `Finish[研究结论]`：当你有足够信息得出结论时。

## 重要提醒
1. 每次回应必须包含Thought和Action两部分
2. 工具调用的格式必须严格遵循：工具名[参数]
3. 只有当你确信有足够信息回答问题时，才使用Finish
4. 如果工具返回的信息不够，继续使用其他工具或相同工具的不同参数

## 当前任务
**Question:** {question}

## 执行历史
{history}

现在开始你的推理和行动："""

# [作用] ReAct Agent：反复「想一步 → 做一步 → 看结果」，直到模型自己喊 Finish 或步数用尽。
# [概念] 继承关系同 SimpleAgent：父类 Agent 用 @abstractmethod 记了 run 这笔账，本类必须实现，
#        否则实例化时抛 TypeError（父类负责建好 self.name / self.llm / self.config / self._history）。
class ReActAgent(Agent):
    """
    ReAct (Reasoning and Acting) Agent
    
    结合推理和行动的智能体，能够：
    1. 分析问题并制定行动计划
    2. 调用外部工具获取信息
    3. 基于观察结果进行推理
    4. 迭代执行直到得出最终答案
    
    这是一个经典的Agent范式，特别适合需要外部信息的任务。
    """
    
    # [作用] 构造：除转交父类的四个参数外，本类只多三个字段 —— 工具注册表 / 步数上限 / 提示词模板。
    # [参数] tool_registry 省略时**自动建一个空注册表**（与 simple_agent.py 不同：那边不传就是「没有工具」）。
    # [参数] custom_prompt 是整套提示词的**替换品**而不是追加件 —— 换掉它就得自己保留那三个占位符。
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 5,
        custom_prompt: Optional[str] = None
    ):
        """
        初始化ReActAgent

        Args:
            name: Agent名称
            llm: LLM实例
            tool_registry: 工具注册表（可选，如果不提供则创建空的工具注册表）
            system_prompt: 系统提示词
            config: 配置对象
            max_steps: 最大执行步数
            custom_prompt: 自定义提示词模板
        """
        # [机制] super().__init__ 先跑父类构造函数 —— 子类重写 __init__ 时最容易漏的一句，漏了父类属性全不存在。
        super().__init__(name, llm, system_prompt, config)

        # [概念] 「没传就自己建一个空注册表」是**空对象模式**：让 self.tool_registry 永远不是 None，
        #        后面 get_tools_description() / execute_tool() 就不用到处写判空。
        # [易错] 代价是「一个工具都没有」这件事被藏起来了：清单渲染成「暂无可用工具」，模型只能瞎猜或硬答。
        # 如果没有提供tool_registry，创建一个空的
        if tool_registry is None:
            self.tool_registry = ToolRegistry()
        else:
            self.tool_registry = tool_registry

        # [概念] 步数上限就是**成本熔断器**：每轮都要发一次完整提示词，步数越大越贵越慢
        #        （对应 simple_agent.py 里的 max_tool_iterations）。
        self.max_steps = max_steps
        # [概念] ⭐ 本文件最关键的字段：历史是**一串字符串**，不是 messages 列表。
        # [概念] 对照 simple_agent.py：那边把 {"role": ..., "content": ...} 攒进 messages，模型能分清「谁说的」；
        #        这边全拼进一条提示词，模型只能从 `Action:` / `Observation:` 前缀自己辨认 —— 省事，但更脆。
        self.current_history: List[str] = []

        # 设置提示词模板：用户自定义优先，否则使用默认模板
        # [语法] 条件表达式 `A if 条件 else B`：custom_prompt 为真就用它，否则退回默认模板。
        # [易错] 它判的是真假而不是「是否为 None」：传 custom_prompt="" 也会被当成没传，静默用回默认模板。
        self.prompt_template = custom_prompt if custom_prompt else DEFAULT_REACT_PROMPT

    # [作用] 注册工具；对「可展开的 MCP 工具」做特殊处理：把一个大工具拆成一堆小工具再逐个注册。
    def add_tool(self, tool):
        """
        添加工具到工具注册表
        支持MCP工具的自动展开

        Args:
            tool: 工具实例(可以是普通Tool或MCPTool)
        """
        # 检查是否是MCP工具
        # [语法] hasattr(obj, 'x') 是**运行时鸭子类型检查**：不问类型，只问「你有没有这个名字的属性」。
        # [易错] 属性名是字符串，写错不报错、只是永远为 False —— 把错误从「写代码时」推迟到了「运行时」。
        if hasattr(tool, 'auto_expand') and tool.auto_expand:
            # MCP工具会自动展开为多个工具
            if hasattr(tool, '_available_tools') and tool._available_tools:
                for mcp_tool in tool._available_tools:
                    # 创建包装工具
                    from ..tools.base import Tool
                    # [机制] 注意 func 的两个默认参数 `t=tool, tn=mcp_tool['name']`：lambda 里的循环变量
                    #        要等它被**调用时**才求值，不在这里用默认参数「当场拍照」，
                    #        所有 lambda 都会看到循环结束时最后一个值（经典的闭包晚绑定坑）。
                    wrapped_tool = Tool(
                        name=f"{tool.name}_{mcp_tool['name']}",
                        description=mcp_tool.get('description', ''),
                        func=lambda input_text, t=tool, tn=mcp_tool['name']: t.run({
                            "action": "call_tool",
                            "tool_name": tn,
                            "arguments": {"input": input_text}
                        })
                    )
                    self.tool_registry.register_tool(wrapped_tool)
                print(f"✅ MCP工具 '{tool.name}' 已展开为 {len(tool._available_tools)} 个独立工具")
            else:
                self.tool_registry.register_tool(tool)
        else:
            self.tool_registry.register_tool(tool)

    # [作用] 主入口：反复「拼提示词 → 问模型 → 解析 → 执行工具 → 把观察结果写回历史」，直到 Finish 或步数耗尽。
    # [参数] input_text 是用户问题；**kwargs 原样转交给 self.llm.invoke（比如 temperature），本方法自己不读它。
    # [返回] str 最终答案。注意步数用尽时返回的是一句**道歉话术**、不是抛异常 —— 调用方从返回值看不出失败。
    # [副作用] 一进来就清空 self.current_history（历史不跨问题累积），结束时才把一问一答写进父类的 _history。
    def run(self, input_text: str, **kwargs) -> str:
        """
        运行ReAct Agent
        
        Args:
            input_text: 用户问题
            **kwargs: 其他参数
            
        Returns:
            最终答案
        """
        # [概念] ⭐ 与 simple_agent.py 的根本差别在这里：那边的 history 是「对话消息」，要跨轮累积；
        #        这边只是「本轮问题的执行轨迹」，所以每次 run 都先清空 —— 上一个问题不会污染这一个。
        self.current_history = []
        current_step = 0
        
        print(f"\n🤖 {self.name} 开始处理问题: {input_text}")
        
        # [语法] while 条件成立就循环；current_step 在循环体第一行自增，所以 max_steps=5 最多问模型 5 次。
        # [概念] ⭐ 每转一圈都要**重新 format 一次提示词**：历史变了，提示词就必须重拼（模型自己不记事）。
        #        对比 simple_agent：那边是往 messages 里 append，消息只拼一次、随对话自然变长。
        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")
            
            # 构建提示词
            # [概念] 工具清单每轮重新取一次（而不是构造时缓存一次）：运行中注册的新工具下一轮马上能用。
            tools_desc = self.tool_registry.get_tools_description()
            history_str = "\n".join(self.current_history)
            # [语法] history_str 是给 {history} 准备的值：`"\n".join(字符串列表)` 用换行把它们粘成一段文本。
            # [语法] str.format(**关键字参数)：按名字把参数填进模板的 {} 里，返回新字符串（模板本身不变）。
            # [易错] 多余的关键字参数会被**静默忽略**（模板没写 {history} 也不报错）；反过来模板写了参数表里没有的
            #        名字就 KeyError —— 报错方向是单向的，两个方向都实测过。
            # [实测] 观察结果里带的 `{` `}` 是安全的：format 只解析**模板本身**，填进去的值不会再被解析一遍。
            prompt = self.prompt_template.format(
                tools=tools_desc,
                question=input_text,
                history=history_str
            )
            
            # 调用LLM
            # [概念] 每轮只有一条 user 消息，而且每次都新建 messages —— 对话历史完全靠 prompt 文本里的 {history} 承载。
            # [易错] 提示词是以 **user** 身份发出去的、不是 system：构造时收下的 system_prompt 在 run() 里根本没用上。
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm.invoke(messages, **kwargs)
            
            # [易错] 空回复只 print 一句就 break，随后落到循环外的兜底话术 —— 但那里打印的是「已达到最大步数」，与真实原因不符。
            if not response_text:
                print("❌ 错误：LLM未能返回有效响应。")
                break
            
            # 解析输出
            # [语法] 元组解包：_parse_output 返回 (thought, action) 两个值，一行拆成两个变量。
            thought, action = self._parse_output(response_text)
            
            if thought:
                print(f"🤔 思考: {thought}")
            
            # [易错] 解析失败 = **直接终止**（而不是把错误当成 Observation 回灌让模型自我纠正）：
            #        模型偶尔漏写一行 Action，整个任务就白跑了。对比下面 tool_name 无效时走的是 continue。
            if not action:
                print("⚠️ 警告：未能解析出有效的Action，流程终止。")
                break
            
            # 检查是否完成
            # [概念] 「结束信号」不靠结构化字段，只靠字符串前缀：Action 以 "Finish" 开头就当模型宣布收工。
            # [易错] startswith 是**大小写敏感**的前缀匹配：模型写 "finish[...]" 或 LangChain 风格的
            #        "Final Answer: ..." 都不会被识别，会一路走到「未解析出 Action」而终止。
            if action.startswith("Finish"):
                # [概念] 这里特意换了个取值方法：Finish[结论] 的结论里可能带空格、方括号，用专门的正则取括号内内容更稳。
                final_answer = self._parse_action_input(action)
                print(f"🎉 最终答案: {final_answer}")
                
                # 保存到历史记录
                self.add_message(Message(input_text, "user"))
                self.add_message(Message(final_answer, "assistant"))
                
                return final_answer
            
            # 执行工具调用
            # [概念] Action 的约定格式是 `工具名[参数]`，_parse_action 把这两半拆开，再分别交给注册表。
            tool_name, tool_input = self._parse_action(action)
            # [易错] tool_input 判的是 `is None` 而不是假值：参数为空串 "" 时仍算合法（虽然工具多半会报错）。
            # [概念] ⭐ 这里是唯一的「重试」通道：解析失败不当致命错误，而是转成一条 Observation 回灌给模型，
            #        让它下一轮自己改写（代价是白烧一轮 API）。
            if not tool_name or tool_input is None:
                self.current_history.append("Observation: 无效的Action格式，请检查。")
                continue
            
            print(f"🎬 行动: {tool_name}[{tool_input}]")
            
            # 调用工具
            # [概念] ⭐ 执行工具只传一个**字符串**：不解析 key=value，也没有类型转换（对比 simple_agent 的 _execute_tool_call）。
            # [机制] execute_tool 内部固定包成 {"input": input_text} 再调 tool.run() —— 所以工具的形参必须正好叫 input。
            # [实测] 形参叫 content 的工具在这条路径上会返回「错误：执行工具 'echo_content' 时发生异常: 'content'」，
            #        但因为是返回错误字符串而不是抛异常，循环不会断，模型只能自己读懂并换招。
            # [机制] 它同时兜住两类工具：Tool 对象（self._tools）和 register_function 注册的函数（self._functions）；
            #        而 simple_agent 走的是 get_tool()，只查 Tool 对象 —— 函数式工具在那边必然「未找到」。
            observation = self.tool_registry.execute_tool(tool_name, tool_input)
            print(f"👀 观察: {observation}")
            
            # 更新历史
            # [概念] ⭐ Observation 回灌：把刚做的动作和刚看到的结果追加进字符串历史，下一轮 format 时它们就出现在 {history} 里。
            #        这正是 ReAct 的闭环 —— 模型下一轮能「看到」上一轮行动的后果。
            # [易错] 只回灌 Action 和 Observation，**Thought 被丢掉了**：模型下一轮看不到自己上一轮想了什么。
            self.current_history.append(f"Action: {action}")
            self.current_history.append(f"Observation: {observation}")
        
        # [易错] 这行在三种情况下都会打印：步数真的用尽 / 模型没输出 Action / 模型回复为空 ——
        #        后两种属于提前失败，日志却说成「已达到最大步数」，会把人往错误方向带（只报告，不改代码）。
        print("⏰ 已达到最大步数，流程终止。")
        final_answer = "抱歉，我无法在限定步数内完成这个任务。"
        
        # 保存到历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))
        
        return final_answer
    
    # [作用] 从模型回复里抠出 Thought 和 Action 两段文本（不做语义校验，纯字符串查找）。
    # [返回] (thought, action) 元组，找不到的那项是 None —— 调用方靠 `if not action:` 判断该不该终止。
    # [机制] 用的是 re.search（在整段文本里**找第一次出现**），不是 re.match（只从开头比），
    #        所以模型前面多写几句客套话也不影响解析。
    # [易错] 正则里的 `.` 默认**不匹配换行**，而 `.*` 又是贪婪的 —— 于是 Action 必须待在一行内，
    #        同一行若出现两次 "Action: "，后面那截会被一起吞进来（实测 'Action: a[1] Action: b[2]' → 'a[1] Action: b[2]'）。
    def _parse_output(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """解析LLM输出，提取思考和行动"""
        # [语法] `r"..."` 是原始字符串：`(.*)` 里的括号是**捕获组**，group(1) 取的就是括号里那一段。
        # [易错] 前缀 `Thought: ` 必须完全一致（冒号后跟一个空格）：模型写成 `Thought:` 没空格就匹配不上。
        thought_match = re.search(r"Thought: (.*)", text)
        action_match = re.search(r"Action: (.*)", text)
        
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        
        return thought, action
    
    def _parse_action(self, action_text: str) -> Tuple[Optional[str], Optional[str]]:
        """解析行动文本，提取工具名称和输入"""
        # [语法] re.match 只从字符串**开头**匹配：` search[x]`（前面多一个空格）会直接失败返回 None（实测）。
        # [机制] 贪婪 `.*` 会一路吃到**最后一个** `]`：实测 `search[Python[3.12] 新特性]` 解析出的参数是
        #        `Python[3.12] 新特性`（参数里带方括号反而没事）；代价是模型漏写 `]` 时整条 Action 就废了。
        match = re.match(r"(\w+)\[(.*)\]", action_text)
        if match:
            return match.group(1), match.group(2)
        return None, None
    
    def _parse_action_input(self, action_text: str) -> str:
        """解析行动输入"""
        # [语法] 同样的模式，但只取括号里的内容 —— 这是 Finish 专用的取值口。
        # [易错] 匹配不到时返回空串 ""（不是 None）：调用方会拿一个空答案当最终答案，从返回值上看不出出错。
        match = re.match(r"\w+\[(.*)\]", action_text)
        return match.group(1) if match else ""
