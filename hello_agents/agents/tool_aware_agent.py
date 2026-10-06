# ===== 文件导读 =====
# [职责] 父类 SimpleAgent 的「工具调用增强版」：重写解析 / 执行 / 流式三个方法，
#        让工具调用能扛住嵌套括号，并把每次调用通过回调播报给外部。
# [位置] agents/tool_aware_agent.py；继承 agents/simple_agent.py 的 SimpleAgent（→ core/agent.py 的 Agent），
#        是全项目唯一的二级继承；上层把它当作「能看见工具在干什么」的 Agent 使用。
# [阅读顺序] ① __init__（*args/**kwargs 透传 + 回调）→ ② _parse_tool_calls（为什么要重写父类）
#            → ③ _find_tool_call_end（引号 + 括号状态机，本文件重点）
#            → ④ _execute_tool_call（重写点：执行完通知监听器）
#            → ⑤ _sanitize_parameters / _normalize_string（清洗模型给的脏参数）
#            → ⑥ stream_run（流式 + 工具调用）→ ⑦ 末尾 _coerce_sequence（工具函数）。
# [一句话] 父类靠正则解析，参数里一出现 ] 就被截断；本文件把「正则」换成「手工状态机」——
#          这是方法重写（override）最典型的一次动机。
# ===================
"""Wrapper around HelloAgents SimpleAgent that records tool calls."""

# [概念] 这条 import 让「类型注解」变成惰性字符串：下面写 ToolRegistry | None 时运行时不会真的求值，
#        所以本文件在还不支持 `X | Y` 这种写法的老解释器上也能正常导入。
from __future__ import annotations

import ast
import json
import logging
from collections.abc import Iterator
# [语法] Any = 「放弃类型检查」的逃生舱；Callable / Optional 的读法见下面 __init__ 的注释。
from typing import Any, Callable, Optional

from .simple_agent import SimpleAgent
from ..core.message import Message
# [概念] 和父类 simple_agent.py 不同：那里用 TYPE_CHECKING 懒加载 ToolRegistry，这里直接 import。
# [实测] 本文件里 ToolRegistry 只出现在 attach_registry 的参数注解里，而注解已被 __future__ 变成字符串，
#        所以这个导入运行时其实用不到（属于「省事换启动成本」的取舍，不是 bug）。
from ..tools import ToolRegistry

# [语法] __name__ 是当前模块的完整名字（hello_agents.agents.tool_aware_agent），
#        用它建 logger，日志里就能看出是哪一层打的；本文件只在监听器抛异常时用它。
logger = logging.getLogger(__name__)


# [位置] 二级继承：ToolAwareSimpleAgent → SimpleAgent → Agent（core/agent.py）。
#        本类没有再实现 run()，因为抽象方法的账在父类 SimpleAgent 那里就结清了。
# [概念] 重写的本质是「同名方法在子类里被换掉」：外部调用 self._execute_tool_call(...) 时，
#        Python 从实例的类型开始沿 MRO 往上找，先找到谁就用谁 —— 这就是多态。
class ToolAwareSimpleAgent(SimpleAgent):
    """SimpleAgent 子类，记录工具调用情况。

    ToolAwareSimpleAgent 扩展了 SimpleAgent，增加了工具调用监听功能。
    这使得外部系统可以追踪和记录智能体的工具调用行为，用于日志记录、
    调试、性能分析等场景。

    主要特性：
    - 工具调用监听：通过回调函数记录每次工具调用的详细信息
    - 增强的工具调用解析：支持复杂的嵌套参数和字符串处理
    - 流式工具调用：在流式输出中支持工具调用
    - 参数清理：自动清理和规范化工具参数

    示例：
        >>> def tool_listener(call_info):
        ...     print(f"工具调用: {call_info['tool_name']}")
        ...     print(f"参数: {call_info['parsed_parameters']}")
        ...     print(f"结果: {call_info['result']}")
        >>>
        >>> agent = ToolAwareSimpleAgent(
        ...     name="研究助手",
        ...     system_prompt="你是一个研究助手",
        ...     llm=llm,
        ...     tool_call_listener=tool_listener
        ... )
        >>> agent.run("搜索最新的AI研究")
    """

    # [作用] 只做父类没做的两件事：把参数原样转交给父类，再存下回调函数。
    # [参数] *args 把「所有位置参数」收成元组、**kwargs 把「所有关键字参数」收成 dict；
    #        这里是**原样透传**——父类将来加参数，本类一行都不用改（代价是签名看不出父类要什么）。
    # [语法] Optional[Callable[[dict[str, Any]], None]] 的读法：Optional[X] = 「X 或 None」；
    #        Callable[[参数类型列表], 返回类型]，所以它表示「收一个 dict、返回 None 的函数」
    #        —— 返回 None 意味着「只看不管返回值」，调用方不会用它的返回结果。
    def __init__(
        self,
        *args: Any,
        tool_call_listener: Optional[Callable[[dict[str, Any]], None]] = None,
        **kwargs: Any,
    ) -> None:
        """初始化 ToolAwareSimpleAgent。

        Args:
            *args: 传递给 SimpleAgent 的位置参数
            tool_call_listener: 工具调用监听器回调函数，接收包含工具调用信息的字典
            **kwargs: 传递给 SimpleAgent 的关键字参数
        """
        # [机制] *args / **kwargs 在这里被「拆包」回位置参数和关键字参数，等于把刚收到的东西
        #        原封不动又传了一遍；父类 __init__ 的 name / llm / system_prompt 就靠这个对上号。
        # [易错] 忘了这一句，父类的 self.name / self.llm / self._history 全都不存在 —— 重写 __init__ 的头号坑。
        super().__init__(*args, **kwargs)
        # [概念] 回调（callback）= 把一个函数存起来，等事件发生（这里是一次工具调用结束）时再调它。
        #        属性名带下划线表示「外部别直接读」，想接收通知就自己传一个函数进来。
        self._tool_call_listener = tool_call_listener

    # [位置] 重写父类 SimpleAgent._execute_tool_call：流程照抄，只在「执行完」之后多一步通知监听器。
    # [作用] 查工具 → 解析并清洗参数 → 执行 → 格式化结果 → 播报给 listener。
    # [语法] 行尾 `# type: ignore[override]` 是给类型检查器（mypy / pyright）看的抑制注释：
    #        子类覆盖父类方法时若签名对不上，检查器会报 override 不兼容，这行就是「我知道，别报了」；
    #        它是注释，运行时完全不生效（本机 myenv 里没装 mypy，未实测具体报错）。
    def _execute_tool_call(self, tool_name: str, parameters: str) -> str:  # type: ignore[override]
        """执行工具调用并通知监听器。

        Args:
            tool_name: 工具名称
            parameters: 工具参数（字符串格式）

        Returns:
            工具执行结果的格式化字符串
        """
        if not self.tool_registry:
            return "❌ 错误：未配置工具注册表"

        try:
            tool = self.tool_registry.get_tool(tool_name)
            if not tool:
                return f"❌ 错误：未找到工具 '{tool_name}'"

            # [作用] 先用父类的 _parse_tool_parameters 把字符串变 dict（它自己会挑 JSON / key=value / 纯文本），
            #        再过一道本类的 _sanitize_parameters，把模型写脏的值（多余引号、字符串化的列表）洗一遍。
            parsed_parameters = self._parse_tool_parameters(tool_name, parameters)
            parsed_parameters = self._sanitize_parameters(parsed_parameters)

            result = tool.run(parsed_parameters)
            formatted_result = f"🔧 工具 {tool_name} 执行结果：\n{result}"
        # [机制] 这里除了兜住错误，还必须把 parsed_parameters 初始化成 {}：只要上面任何一步抛异常，
        #        这个变量就从未被赋值，而下面通知监听器时一定会读它 —— 不初始化就是 UnboundLocalError。
        except Exception as exc:  # pragma: no cover - tool failures回退
            parsed_parameters = {}
            formatted_result = f"❌ 工具调用失败：{exc}"

        # 通知监听器
        # [概念] 可观测性：Agent 内部干了什么，外部默认看不见。回调让它主动播报一次结构化事件
        #        （agent_name / tool_name / 原始参数 / 解析后参数 / 结果），上层可以拿去落日志、计费或画时间线。
        if self._tool_call_listener:
            try:
                self._tool_call_listener(
                    {
                        "agent_name": self.name,
                        "tool_name": tool_name,
                        "raw_parameters": parameters,
                        "parsed_parameters": parsed_parameters,
                        "result": formatted_result,
                    }
                )
            # [为什么] 监听器是外部代码（可能写错），不能让它把工具结果搞丢：这里吞掉异常只记一条日志，
            #        保证 return 出去的永远是「工具真实干了什么」。
            except Exception:  # pragma: no cover - 防御性兜底
                logger.exception("Tool call listener failed")

        return formatted_result

    # [位置] 重写父类的同名方法 —— 这是本文件存在的首要理由。
    # [为什么] 父类用正则 [^\]]+ 抓参数，参数里一出现 ] 就被截断；本方法改成「从 [TOOL_CALL: 往后
    #        一个字符一个字符地扫，数方括号 + 认引号」，所以能处理嵌套。
    # [实测] 输入 [TOOL_CALL:note:title=会议,content=数组 [1, 2, 3]] —— 父类正则得到
    #        'title=会议,content=数组 [1, 2, 3'（末尾被截断），本方法得到完整的 'title=会议,content=数组 [1, 2, 3]'。
    def _parse_tool_calls(self, text: str) -> list:  # type: ignore[override]
        """解析文本中的工具调用。

        支持格式：[TOOL_CALL:tool_name:parameters]

        Args:
            text: 包含工具调用的文本

        Returns:
            工具调用列表，每个元素包含 tool_name、parameters 和 original
        """
        # [易错] 这个标记必须和父类 _get_enhanced_system_prompt 里教给模型的格式一字不差；
        #        两边各写死一份，改一处忘一处就会「模型按新格式写、解析按老格式找」，全部漏掉。
        marker = "[TOOL_CALL:"
        calls: list = []
        start = 0

        # [语法] str.find(子串, start) 从下标 start 开始找，找不到返回 -1（不抛异常）。
        #        外层靠「找不到就 break」收尾，这是手写扫描器最常见的骨架。
        while True:
            begin = text.find(marker, start)
            if begin == -1:
                break

            tool_start = begin + len(marker)
            colon = text.find(":", tool_start)
            if colon == -1:
                break

            tool_name = text[tool_start:colon].strip()
            body_start = colon + 1
            pos = body_start
            # [概念] 三个变量组成一个极小的状态机：depth 记「还欠几个右方括号」，
            #        in_string 记「此刻是否在引号里」，string_quote 记住引号是单引号还是双引号。
            # [为什么] 有了引号状态，b="a]b" 里的 ] 不再被当成结束符 —— 这正是正则做不到的地方。
            depth = 0
            in_string = False
            string_quote = ""

            # [机制] 主循环每一步只看一个字符：不在引号里就数括号（[ 加一、] 减一），
            #        遇到 depth == 0 的 ] 说明配对结束，那一位就是这次工具调用的真实结尾 —— 这就是括号配对。
            while pos < len(text):
                char = text[pos]

                if char in {'"', "'"}:
                    if not in_string:
                        in_string = True
                        string_quote = char
                    # [易错] 反斜杠判断太粗糙：只看前一个字符是不是 \，分不清「转义引号」和「以反斜杠结尾的 Windows 路径」。
                    # [实测] [TOOL_CALL:a:b="C:\path\"] → 末尾的 " 被当成转义，引号永不闭合，
                    #        整个调用被静默丢弃（返回 []，不报错）；引号内含 ] 的 b="a]b" 则能正确包住。
                    elif string_quote == char and text[pos - 1] != "\\":
                        in_string = False

                if not in_string:
                    if char == '[':
                        depth += 1
                    elif char == ']':
                        if depth == 0:
                            body = text[body_start:pos].strip()
                            original = text[begin : pos + 1]
                            calls.append(
                                {
                                    "tool_name": tool_name,
                                    "parameters": body,
                                    "original": original,
                                }
                            )
                            start = pos + 1
                            break
                        else:
                            depth -= 1

                pos += 1
            # [语法] while / for 后面可以跟 else：循环「正常跑完」（没被 break）时才执行 else。
            #        这里的意思是：扫到文本末尾都没等到配对的 ] → 这个调用不完整 → 放弃整个解析。
            # [易错] while...else 不是 if-else 的那个「否则」，极易误读。
            else:
                break

        return calls

    # [概念] @staticmethod：不接收 self / cls，只是「挂在类命名空间里的普通函数」——
    #        靠类名或实例都能调，但拿不到实例状态，所以它必须是纯函数式的「输入 → 输出」。
    # [易错] 装饰器必须紧贴 def，中间不能夹注释；本文件的注释一律写在 @staticmethod 之上。
    # [位置] stream_run 的流式解析会调用它；它和上面 _parse_tool_calls 的内层循环几乎一样，
    #        区别只是「返回结尾下标」而不是「直接往列表里塞」。
    @staticmethod
    def _find_tool_call_end(text: str, start_index: int) -> int:
        """查找工具调用的结束位置。

        Args:
            text: 文本内容
            start_index: 工具调用的起始位置

        Returns:
            工具调用结束位置的索引，如果未找到返回 -1
        """
        marker = "[TOOL_CALL:"
        tool_start = start_index + len(marker)
        colon = text.find(":", tool_start)
        if colon == -1:
            return -1

        body_start = colon + 1
        pos = body_start
        depth = 0
        in_string = False
        string_quote = ""

        while pos < len(text):
            char = text[pos]

            if char in {'"', "'"}:
                if not in_string:
                    in_string = True
                    string_quote = char
                elif string_quote == char and text[pos - 1] != "\\":
                    in_string = False

            if not in_string:
                if char == '[':
                    depth += 1
                elif char == ']':
                    if depth == 0:
                        return pos
                    depth -= 1

            pos += 1

        # [返回] -1 = 「扫到末尾也没等到配对的 ]」；调用方必须显式判断这个哨兵值（见 stream_run）。
        return -1

    # [作用] 便利方法：把外部建好的 registry 挂到 agent 上，并顺手打开工具开关。
    # [易错] 名字里有 static 却不是纯函数 —— 它改的是传进来那个 agent 对象的属性（副作用）。
    # [实测] 全仓库 grep 不到任何调用点，只有这一处定义，属于预留接口。
    @staticmethod
    def attach_registry(agent: "ToolAwareSimpleAgent", registry: ToolRegistry | None) -> None:
        """Helper to attach a tool registry if provided.

        Args:
            agent: ToolAwareSimpleAgent 实例
            registry: 工具注册表
        """
        if registry:
            agent.tool_registry = registry
            agent.enable_tool_calling = True

    # [作用] 把 _parse_tool_parameters 吐出来的脏参数洗干净再交给工具。
    # [为什么] 模型写出来的参数都是文本：tags="[a,b]"、task_id="12"、title='"标题"' 都算「看起来对」，
    #        但工具拿到后可能直接崩；清洗层的职责就是在调用工具前把这些形态收敛掉。
    @staticmethod
    def _sanitize_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
        """清理和规范化工具参数。

        Args:
            parameters: 原始参数字典

        Returns:
            清理后的参数字典
        """
        sanitized: dict[str, Any] = {}
        for key, value in parameters.items():
            # [易错] 已经是数字 / 布尔 / 列表 / 字典的值原样放行、不再加工。
            #        注意 bool 是 int 的子类，所以 True 走的就是这一支（不是 bug，但类型思维上要知道）。
            if isinstance(value, (int, float, bool, list, dict)):
                sanitized[key] = value
                continue

            if isinstance(value, str):
                normalized = ToolAwareSimpleAgent._normalize_string(value)

                # [易错] 这里出现了 'task_id' / 'tags' / 'note_type' 这些具体字段名 —— 通用解析层里混进了
                #        某个工具（笔记类）的领域知识。加一个新工具就得回来改这里，是典型的「知识放错层」。
                if key == "task_id":
                    try:
                        sanitized[key] = int(normalized)
                        continue
                    except ValueError:
                        pass

                if key == "tags":
                    parsed_tags = ToolAwareSimpleAgent._coerce_sequence(normalized)
                    if isinstance(parsed_tags, list):
                        sanitized[key] = parsed_tags
                        continue
                    if normalized:
                        sanitized[key] = [item.strip() for item in normalized.split(",") if item.strip()]
                        continue

                # [易错] 这个分支做的事（sanitized[key] = normalized; continue）和这个 if 之外的兜底完全一样，
                #        也就是说这一整个 if 是冗余的 —— 只报告，不改代码。
                if key in {"note_type", "action", "title", "content", "note_id"}:
                    sanitized[key] = normalized
                    continue

                sanitized[key] = normalized
                continue

            # [作用] 剩下的类型（None、其它对象）一概不加工，原样透传。
            sanitized[key] = value

        return sanitized

    # [作用] 清洗的第一步：把字符串两端「多余的引号」和「缺一半的括号」修掉。
    # [易错] 它只做模式匹配、不管语义：'"[1, 2, 3"' 会变成 '[1, 2, 3]'，
    #        看着像列表其实还是字符串 —— 想变真列表得靠下面的 _coerce_sequence。
    @staticmethod
    def _normalize_string(value: str) -> str:
        """规范化字符串值，移除多余的引号和括号。

        Args:
            value: 原始字符串

        Returns:
            规范化后的字符串
        """
        trimmed = value.strip()

        # [语法] trimmed.count(trimmed[0]) == 1 读作「这个引号在整个串里只出现一次」——
        #        只出现一次才算落单的引号，才敢删；成对的留给第三个 if 统一处理。
        if trimmed and trimmed[0] in {'"', "'"} and trimmed.count(trimmed[0]) == 1:
            trimmed = trimmed[1:]
        if trimmed and trimmed[-1] in {'"', "'"} and trimmed.count(trimmed[-1]) == 1:
            trimmed = trimmed[:-1]

        # [机制] 三条判断依次收口：先删落单的左引号、再删落单的右引号、最后处理成对的首尾引号。
        if trimmed and trimmed[0] in {'"', "'"} and trimmed[-1] == trimmed[0]:
            trimmed = trimmed[1:-1]

        # [机制] 只补不删：'[1, 2, 3' 这种「开了没关」的串会被补成 '[1, 2, 3]'，
        #        目的就是让下一步 ast.literal_eval / json.loads 能解析成功。
        if trimmed and trimmed[0] in {'[', '('} and trimmed[-1] not in {']', ')'}:
            closing = ']' if trimmed[0] == '[' else ')'
            trimmed = f"{trimmed}{closing}"

        return trimmed.strip()

    # [位置] 重写父类 SimpleAgent.stream_run（父类那版只转发文本，完全不认工具调用）。
    # [作用] 边流边吐字，同时用 residual 缓冲区识别可能被拆成两半的 [TOOL_CALL:...]：
    #        把标记和参数从正文里剔掉、执行工具、把结果回灌后再问一轮。
    # [易错] 循环结构和 run() 同构，但多一层麻烦：标记可能横跨两个字片，不能见到就下判断。
    def stream_run(self, input_text: str, max_tool_iterations: int = 3, **kwargs: Any) -> Iterator[str]:  # type: ignore[override]
        """Stream assistant output while supporting tool calls mid-generation.

        流式运行智能体，支持在生成过程中调用工具。

        Args:
            input_text: 用户输入文本
            max_tool_iterations: 最大工具调用迭代次数
            **kwargs: 传递给 LLM 的额外参数

        Yields:
            生成的文本片段
        """
        messages: list[dict[str, Any]] = []
        enhanced_system_prompt = self._get_enhanced_system_prompt()
        messages.append({"role": "system", "content": enhanced_system_prompt})

        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": input_text})

        # [作用] final_segments 收集所有已经 yield 出去的正文（最后要拼成历史记录），
        #        final_response_text 只存「不带工具调用的那一轮」的正文。
        final_segments: list[str] = []
        final_response_text = ""
        current_iteration = 0

        marker = "[TOOL_CALL:"

        # [机制] 每一轮都重新开一个 residual（残留缓冲）和 segments_this_round（本轮正文）：
        #        工具结果回灌后模型会重新开始说话，上一轮的缓冲不能混进来。
        while current_iteration < max_tool_iterations:
            residual = ""
            segments_this_round: list[str] = []
            tool_call_texts: list[str] = []

            # [机制] 函数体里定义的嵌套函数 + nonlocal：residual 是外层 stream_run 的局部变量，
            #        内层读它不用声明，但要**改**它就必须写 nonlocal，否则赋值会被当成新建一个内层局部变量。
            # [语法] 它体内有 yield，所以 process_residual 也是生成器函数：调用它只拿到生成器，不执行函数体。
            def process_residual(final_pass: bool = False) -> Iterator[str]:
                nonlocal residual
                while True:
                    start = residual.find(marker)
                    if start == -1:
                        # [机制] 保守吐出：尾部留 (len(marker) - 1) 个字符先不发，因为那几个字符可能正好是
                        #        下一个 [TOOL_CALL: 的前半截，等下一片到了再一起判断。
                        safe_len = len(residual) if final_pass else max(0, len(residual) - (len(marker) - 1))
                        if safe_len > 0:
                            segment = residual[:safe_len]
                            residual = residual[safe_len:]
                            yield segment
                        break

                    if start > 0:
                        segment = residual[:start]
                        residual = residual[start:]
                        if segment:
                            yield segment
                        continue

                    # [位置] 这里用上前面那个静态方法：拿到本次工具调用的结尾下标；-1 表示「还没收完」，先等着。
                    end = self._find_tool_call_end(residual, 0)
                    if end == -1:
                        break

                    tool_call_texts.append(residual[: end + 1])
                    residual = residual[end + 1 :]

            # [概念] 流式 + 工具调用的组合拳：模型吐字是碎片，而工具调用必须拿到完整标记才能执行，
            #        所以代码只能「先攒、再判断、判断成立才动手」。
            for chunk in self.llm.stream_invoke(messages, **kwargs):
                if not chunk:
                    continue

                residual += chunk

                for segment in process_residual():
                    if not segment:
                        continue
                    segments_this_round.append(segment)
                    final_segments.append(segment)
                    yield segment

            # [作用] 流结束后再跑一次收尾（final_pass=True）：这时不再保守，把压在 residual 里的尾巴全吐出来。
            for segment in process_residual(final_pass=True):
                if not segment:
                    continue
                segments_this_round.append(segment)
                final_segments.append(segment)
                yield segment

            # [易错] 存进对话的是 clean_response（只有正文、已剔掉工具标记），而不是模型原话；
            #        这是它和父类 run() 的一处实质差别 —— 父类会把带 [TOOL_CALL:...] 的原文塞回对话。
            clean_response = "".join(segments_this_round)
            tool_calls: list[dict[str, Any]] = []

            # [机制] 二次解析：process_residual 已按「引号 + 括号」把每个工具调用切成整块文本存进 tool_call_texts，
            #        这里再交给 _parse_tool_calls 把每一块拆成 {tool_name, parameters, original}。
            for call_text in tool_call_texts:
                tool_calls.extend(self._parse_tool_calls(call_text))

            # [概念] 这一段和父类 run() 里的工具循环是同一个套路：assistant 说原话 → user 发工具结果 → 再问一轮。
            if tool_calls:
                messages.append({"role": "assistant", "content": clean_response})

                tool_results = []
                for call in tool_calls:
                    result = self._execute_tool_call(call["tool_name"], call["parameters"])
                    tool_results.append(result)

                tool_results_text = "\n\n".join(tool_results)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "工具执行结果：\n"
                            f"{tool_results_text}\n\n"
                            "请基于这些结果给出完整的回答。"
                        ),
                    }
                )

                current_iteration += 1
                continue

            final_response_text = clean_response
            break

        # [易错] 迭代次数用满还没拿到最终回答时，会再多花一次 API 调用（父类 run() 有同样的兜底）；
        #        而且这次兜底走的是非流式 invoke，会一次性 yield 一大段 —— 前端的打字机效果到这里会卡一下。
        if current_iteration >= max_tool_iterations and not final_response_text:
            fallback_response = self.llm.invoke(messages, **kwargs)
            final_segments.append(fallback_response)
            final_response_text = fallback_response
            yield fallback_response

        # [机制] 存历史时「最终回答」优先；如果全程只有工具调用、一句正文都没有，
        #        就退化成把已经 yield 出去的片段拼起来，至少不存空字符串。
        stored_response = final_response_text or "".join(final_segments)

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(stored_response, "assistant"))

    # [作用] 尽力把字符串变成真正的 list：先补上可能缺失的右括号，再拿两个解析器轮流试。
    # [为什么] 参数可能是 JSON 数组、Python 字面量，甚至只有半截 —— 这层就是「都试一遍，谁成算谁」。
    @staticmethod
    def _coerce_sequence(value: str) -> Any:
        """尝试将字符串转换为列表。

        Args:
            value: 字符串值

        Returns:
            解析后的列表，如果解析失败返回 None
        """
        if not value:
            return None

        candidates = [value]
        if value.startswith("[") and not value.endswith("]"):
            candidates.append(f"{value}]")
        if value.startswith("(") and not value.endswith(")"):
            candidates.append(f"{value})")

        for candidate in candidates:
            # [概念] 函数是一等对象：json.loads 和 ast.literal_eval 本身可以放进元组里循环调用，
            #        想加第三个解析器只要往元组里加一项。
            # [易错] 两个都失败就 continue，最终返回 None（不抛异常）—— 调用方必须自己判 None。
            for loader in (json.loads, ast.literal_eval):
                try:
                    parsed = loader(candidate)
                except Exception:
                    continue
                if isinstance(parsed, list):
                    return parsed

        return None

