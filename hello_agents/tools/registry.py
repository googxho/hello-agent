# ===== 文件导读 =====
# [职责] 工具注册表：一个「工具箱」。谁想用工具，先往这里登记；Agent 运行时只跟它打交道。
# [位置] tools/ 包的中枢：core/agent.py 和 agents/*.py 都持有它的实例；
#        文件末尾的 global_registry 是全局唯一的那一个（各 Agent 默认共用）。
# [阅读顺序] ① __init__ 里两个字典（本文件最重要的设计决定）
#            → ② register_tool（含「自动展开」）
#            → ③ execute_tool（唯一的执行入口，重点看参数怎么传）
#            → ④ get_tools_description（工具描述就是提示词）
#            → ⑤ 文件末尾的 global_registry。
# [本文件新概念] ① 双字典按类型分流　② 覆盖式注册　③ 描述即提示词　④ 模块级单例　⑤ 错误也是上下文。
# ===================
"""工具注册表 - HelloAgents原生工具系统"""

# [语法] 这一行导入的是「类型注解用的工具」，运行时几乎不参与逻辑，只为人和类型检查器服务。
from typing import Optional, Any, Callable
# [语法] 相对导入：开头的点表示「当前所在这个包」（hello_agents.tools），
#        所以它指向同目录下的 base.py。依赖是单向的：base.py 完全不认识 registry。
from .base import Tool

# [作用] 工具的容器 + 调度中心，只做四件事：登记、查找、执行、列清单。
# [为什么] Agent 不直接持有一个个工具对象，而是通过注册表访问：
#        这样加工具 / 换工具都不用动 Agent 的代码，多个 Agent 也能共享同一个工具箱。
# [概念] 本文件是「一个类管两类东西」的典型：Tool 对象和普通函数分开存，见下面两个字典。
class ToolRegistry:
    """
    HelloAgents工具注册表

    提供工具的注册、管理和执行功能。
    支持两种工具注册方式：
    1. Tool对象注册（推荐）
    2. 函数直接注册（简便）
    """

    # [作用] 只建两个空字典 —— 注册表初始是空的，这里没有任何内置工具。
    # [概念] 双字典 = 按「注册方式」分流：
    #        _tools     存 Tool 子类实例（功能强，能展开、能出 OpenAI schema）
    #        _functions 存裸函数（只收一个字符串、只回一个字符串，最省事）
    # [为什么] 不合成一个字典：两者的调用方式根本不同（一个 run(dict)，一个 func(str)），
    #        分开存，查找和调用都不需要先判断类型。
    def __init__(self):
        # [语法] 变量注解：dict[str, Tool] 只写给类型检查器看，运行时不校验、也不影响性能。
        # [易错] 两个字典的键**互不相通**：同名的 Tool 和函数可以同时存在，
        #        而 execute_tool 优先走 _tools —— 「到底调用了谁」不会报错，只会悄悄走错分支。
        self._tools: dict[str, Tool] = {}
        self._functions: dict[str, dict[str, Any]] = {}

    # [作用] 登记一个 Tool 实例；若它声明自己可展开，就登记它展开出来的那批子工具。
    # [参数] auto_expand=True 表示「能展开就自动展开」—— 此时父工具本体**不会**被注册，
    #        只有子工具进去，避免「父 + 子」重复出现在提示词里。
    # [副作用] 会 print 成功 / 覆盖警告 —— 这是本模块唯一的反馈渠道（没有日志、没有返回值）。
    # [概念] 覆盖式注册：同名工具不报错、直接替换。方便迭代，但名字写错时会静默顶掉别人。
    def register_tool(self, tool: Tool, auto_expand: bool = True):
        """
        注册Tool对象

        Args:
            tool: Tool实例
            auto_expand: 是否自动展开可展开的工具（默认True）
        """
        # 检查工具是否可展开
        # [语法] 三个 and 依次短路：先看要不要展开，再看这个对象有没有 expandable 属性
        #        （hasattr 是鸭子类型式的检查），最后才看它到底是真值还是假值。
        # [易错] 这里用 hasattr 而不是 isinstance(tool, Tool)：说明作者不假定传进来的
        #        一定是 Tool 子类 —— 防御性写法，也暗示注册表对「工具」的定义很宽松。
        if auto_expand and hasattr(tool, 'expandable') and tool.expandable:
            # [机制] 展开动作发生在**注册这一刻**，之后注册表里就只剩子工具了。
            #        所以「父工具是谁」这条线索在注册表里会丢掉，只留在子工具的 parent 属性上。
            expanded_tools = tool.get_expanded_tools()
            if expanded_tools:
                # 注册所有展开的子工具
                for sub_tool in expanded_tools:
                    # [易错] 覆盖警告只对比 _tools，不对比 _functions：
                    #        先注册了一个同名函数式工具的话，这里不会有任何提醒。
                    if sub_tool.name in self._tools:
                        print(f"⚠️ 警告：工具 '{sub_tool.name}' 已存在，将被覆盖。")
                    self._tools[sub_tool.name] = sub_tool
                # [机制] 展开之后直接 return —— 父工具本体不注册。
                # [实测] 这就是为什么注册一个可展开工具之后，list_tools() 里看到的是一串
                #        形如 demo_echo 的子工具名，而 get_tool('demo') 返回的是 None。
                print(f"✅ 工具 '{tool.name}' 已展开为 {len(expanded_tools)} 个独立工具")
                return

        # 普通工具或不展开的工具
        # [易错] 能走到这里，说明「要么不可展开，要么一个标签方法都没找到」。
        #        后一种情况很容易被误解成「工具没注册上」，其实是 get_expanded_tools() 返回了 None。
        if tool.name in self._tools:
            print(f"⚠️ 警告：工具 '{tool.name}' 已存在，将被覆盖。")

        # [语法] 字典赋值即「新增或覆盖」。这里没做任何校验（名字是否为空、run 是否可调用都不查）。
        self._tools[tool.name] = tool
        print(f"✅ 工具 '{tool.name}' 已注册。")

    # [作用] 最省事的注册方式：直接扔一个函数进来，不写类、不写参数说明。
    # [参数] func 的签名被约定成「收一个字符串、返回一个字符串」——
    #        也就是说这条路只能表达**单参数**工具。
    # [易错] description 完全靠人手写，没有内省兜底；
    #        而且存进 _functions 后就丢了参数信息，to_openai_schema() 那一套用不上。
    # [易错] 用本方法注册的工具，SimpleAgent 的 _execute_tool_call 找不到它（那条路只查 get_tool），
    #        结果就是「工具描述列得出来，执行时却说未找到」—— 实测确认过的框架缺陷。
    def register_function(self, name: str, description: str, func: Callable[[str], str]):
        """
        直接注册函数作为工具（简便方式）

        Args:
            name: 工具名称
            description: 工具描述
            func: 工具函数，接受字符串参数，返回字符串结果
        """
        if name in self._functions:
            print(f"⚠️ 警告：工具 '{name}' 已存在，将被覆盖。")

        # [概念] 这里存的是 dict 而不是函数本身：给以后加字段留了位置（现在只有 description 和 func）。
        self._functions[name] = {
            "description": description,
            "func": func
        }
        print(f"✅ 工具 '{name}' 已注册。")

    # [作用] 按名字删工具，两个字典都找一遍。
    # [易错] 只删第一个命中的那个：如果两个字典里同名都有，函数版会留下来。
    # [易错] 这里的名字是 unregister，而 agents/simple_agent.py 里调的是 unregister_tool ——
    #        本类根本没有那个方法，走到那一行会抛 AttributeError。实测确认过。
    def unregister(self, name: str):
        """注销工具"""
        if name in self._tools:
            del self._tools[name]
            print(f"🗑️ 工具 '{name}' 已注销。")
        elif name in self._functions:
            del self._functions[name]
            print(f"🗑️ 工具 '{name}' 已注销。")
        else:
            print(f"⚠️ 工具 '{name}' 不存在。")

    # [作用] 两个查找入口，各查各的字典，互不穿透 —— 找 Tool 的方法不会返回函数。
    # [语法] dict.get(key) 找不到时返回 None（对比 d[key] 会抛 KeyError）。
    #        返回类型写成 Optional[Tool] 就是在说「可能是 None」，调用方必须先判空。
    def get_tool(self, name: str) -> Optional[Tool]:
        """获取Tool对象"""
        return self._tools.get(name)

    def get_function(self, name: str) -> Optional[Callable]:
        """获取工具函数"""
        # [机制] 先取出整个 dict 再判断真假：空 dict 是假值，None 也是假值，
        #        所以下一行一个 if 就同时挡住了「没这个工具」和「有但内容为空」两种情况。
        func_info = self._functions.get(name)
        # [语法] 条件表达式（三元）：找到就取 func，没找到就返回 None。
        return func_info["func"] if func_info else None

    # [作用] 唯一的执行入口，也是本文件最值得读的一段。
    # [参数] input_text 是**一个字符串** —— 注意这里没有「参数字典」这个概念。
    # [返回] 永远返回字符串；出错时也把错误「翻译」成字符串返回，而不是抛出异常。
    # [为什么] 不抛异常：工具结果要拼回对话交给模型，抛出去会中断整个 Agent 循环。
    #        写成字符串，模型就有机会看到错误、换工具或改参数 —— 这叫「错误也是上下文」。
    # [本文件最大的一处设计约束] 因为手上只有一个字符串，Tool 的调用被硬编码成
    #        tool.run({"input": input_text}) —— 参数名必须叫 input，多参数工具走不通这条路。
    def execute_tool(self, name: str, input_text: str) -> str:
        """
        执行工具

        Args:
            name: 工具名称
            input_text: 输入参数

        Returns:
            工具执行结果
        """
        # 优先查找Tool对象
        # [概念] 按注册方式分流，_tools 优先。两条分支结构几乎一样，只有调用方式不同。
        if name in self._tools:
            tool = self._tools[name]
            try:
                # 简化参数传递，直接传入字符串
                # [机制] 就是这一行把「参数字典」简化成了固定的 {"input": ...}。
                # [实测] 所以被 @tool_action 装饰的方法，形参必须叫 input 才能从这条路被调用；
                #        builtin 里那些形参叫 content / query 的方法，得写成 key=value 才走得通。
                return tool.run({"input": input_text})
            # [易错] except Exception 吞掉一切异常（包括打错字引起的 AttributeError）。
            #        好处是 Agent 不会崩；坏处是真正的 bug 会被包装成一句「工具执行异常」，
            #        而且堆栈在这里就被丢掉了 —— 排查时要记得回来看这一处。
            except Exception as e:
                return f"错误：执行工具 '{name}' 时发生异常: {str(e)}"

        # 查找函数工具
        elif name in self._functions:
            func = self._functions[name]["func"]
            try:
                return func(input_text)
            except Exception as e:
                return f"错误：执行工具 '{name}' 时发生异常: {str(e)}"

        # [机制] 两条分支都没命中才走到这里。注意这是 if / elif / else 结构，
        #        所以「Tool 执行时抛异常」不会掉进这里，而是被上面的 except 接住并变成错误字符串。
        else:
            return f"错误：未找到名为 '{name}' 的工具。"

    # [作用] 把所有工具拼成一段文本，交给 Agent 塞进提示词。
    # [概念] 描述即提示词：模型看到的「工具清单」就是这里生成的字符串。
    #        它写得不清楚，模型就会选错工具、或者编造参数名。
    # [易错] 格式只有「- 名字: 描述」—— 没有参数名、没有类型、没有必填标记。
    #        可 SimpleAgent 的提示词却要求模型按「名字:参数」输出，参数名从哪来？
    #        只能靠模型从描述文字里猜 —— 这是实测确认过的第二个框架缺陷。
    # [为什么] 对比 to_openai_schema()：那边参数信息一应俱全，因为 JSON Schema 有地方放；
    #        而这里的纯文本格式压根没给参数留位置。
    def get_tools_description(self) -> str:
        """
        获取所有可用工具的格式化描述字符串

        Returns:
            工具描述字符串，用于构建提示词
        """
        # [语法] 先攒成一个列表、最后用 "\n".join(...) 拼：比在循环里反复用 + 拼字符串省内存、
        #        也更快（字符串不可变，每次 + 都要整份复制一遍）。
        descriptions = []

        # Tool对象描述
        for tool in self._tools.values():
            descriptions.append(f"- {tool.name}: {tool.description}")

        # 函数工具描述
        # [易错] Tool 工具和函数工具被拼进同一段文本、格式完全一样，
        #        模型（以及读日志的人）都分不出哪些是可展开的、哪些只是裸函数。
        for name, info in self._functions.items():
            descriptions.append(f"- {name}: {info['description']}")

        return "\n".join(descriptions) if descriptions else "暂无可用工具"

    # [作用] 两个只读出口：一个只要名字（合并两个字典），一个只要 Tool 对象。
    # [机制] 都返回**新列表**（list(...) 做了一次拷贝），调用方随便改也动不到注册表内部 —— 故意的防御。
    # [易错] 顺序不保证是注册顺序（尤其先注册函数、再注册 Tool 时），别依赖它做展示排序。
    def list_tools(self) -> list[str]:
        """列出所有工具名称"""
        return list(self._tools.keys()) + list(self._functions.keys())

    def get_all_tools(self) -> list[Tool]:
        """获取所有Tool对象"""
        return list(self._tools.values())

    # [作用] 清空。作用域是**这一个实例** —— 但如果那个实例正好是下面的 global_registry，
    #        效果就是全局清空：别的 Agent 手里攥着的工具也会一起没掉。
    def clear(self):
        """清空所有工具"""
        self._tools.clear()
        self._functions.clear()
        print("🧹 所有工具已清空。")

# 全局工具注册表
# [概念] 模块级单例：这一行在模块第一次被导入时执行，之后再 import 拿到的都是同一个对象
#        （Python 会缓存已导入的模块，不会重复执行模块体）。
# [机制] 注意它是「模块级单例」而不是「类级单例」：ToolRegistry() 本身可以随便 new 很多个，
#        全局性完全来自这一个变量。
# [易错] 于是它有隐蔽的串味风险：A 测试往里注册了工具却没清掉，B 测试就会莫名多出一个工具。
#        写测试时要么用独立实例，要么记得 clear()。
global_registry = ToolRegistry()
