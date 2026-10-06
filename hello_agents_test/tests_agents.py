"""第 3 组：六个 Agent 范式的行为测试（agents/ 全部文件）。

关键手法：用 `FakeLLM` 顶替真模型客户端 —— 因为 Agent 运行时只读 `self.llm` 的
几个方法，Python 不做类型检查（鸭子类型），所以假对象完全能顶替真客户端。
好处是：不花钱、不受网络影响、**回复可预测**（才能断言「第几轮该返回什么」）。

对应的教学进度：agents/simple_agent.py · function_call_agent.py · react_agent.py
                reflection_agent.py · plan_solve_agent.py · tool_aware_agent.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from checks import Check, quiet
from fake_llm import FakeLLM


def _empty_registry():
    from hello_agents.tools.registry import ToolRegistry
    return ToolRegistry()


def _calc_registry():
    """一个能用的工具注册表（参数名必须是 input，否则纯文本调用会失败）。"""
    from hello_agents.tools.base import Tool, ToolParameter, tool_action
    from hello_agents.tools.registry import ToolRegistry

    class CharTool(Tool):
        def __init__(self):
            super().__init__(name="char", description="字符统计", expandable=True)

        def run(self, parameters):
            return "请调用子工具"

        def get_parameters(self):
            return []

        @tool_action("char_count", "统计文本字符数")
        def _count(self, input: str) -> str:
            return f"{len(input)} 个字符"

    reg = ToolRegistry()
    with quiet():
        reg.register_tool(CharTool(), auto_expand=True)
    return reg


def run(c: Check) -> None:
    from hello_agents.agents.simple_agent import SimpleAgent
    from hello_agents.agents.function_call_agent import FunctionCallAgent
    from hello_agents.agents.react_agent import ReActAgent
    from hello_agents.agents.reflection_agent import ReflectionAgent
    from hello_agents.agents.plan_solve_agent import PlanAndSolveAgent
    from hello_agents.agents.tool_aware_agent import ToolAwareSimpleAgent
    from hello_agents.core.agent import Agent

    # ────────────────────────────────────────────────────────────
    c.section("A. 继承契约：一个基类，六个子类")

    direct = [SimpleAgent, FunctionCallAgent, ReActAgent, ReflectionAgent, PlanAndSolveAgent]
    for cls in direct:
        c.ok(issubclass(cls, Agent), f"{cls.__name__} 直接继承 Agent", f"实际父类 {cls.__bases__!r}")

    c.ok(issubclass(ToolAwareSimpleAgent, SimpleAgent),
         "ToolAwareSimpleAgent 继承的是 SimpleAgent（全项目唯一的二级继承）")
    mro = [k.__name__ for k in ToolAwareSimpleAgent.__mro__[:3]]
    c.eq(mro, ["ToolAwareSimpleAgent", "SimpleAgent", "Agent"],
         "它的 MRO 是 ToolAware → Simple → Agent（三级链）")
    c.eq(ToolAwareSimpleAgent.run.__qualname__, "SimpleAgent.run",
         "机制验证：ToolAwareSimpleAgent 自己不写 run，直接复用父类的实现")

    for cls in [SimpleAgent, FunctionCallAgent, ReActAgent, ReflectionAgent,
                PlanAndSolveAgent, ToolAwareSimpleAgent]:
        c.eq(cls.__abstractmethods__, frozenset(),
              f"{cls.__name__} 的抽象账本已清空（契约已履行，能实例化）")

    stream = {cls.__name__: hasattr(cls, "stream_run")
              for cls in [SimpleAgent, FunctionCallAgent, ReActAgent,
                          ReflectionAgent, PlanAndSolveAgent, ToolAwareSimpleAgent]}
    c.eq([k for k, v in stream.items() if v],
         ["SimpleAgent", "FunctionCallAgent", "ToolAwareSimpleAgent"],
         "只有 3/6 个范式提供 stream_run")

    # ────────────────────────────────────────────────────────────
    c.section("B. SimpleAgent —— 无工具时的最小闭环")

    llm = FakeLLM(replies=["你好，我是假模型。"])
    agent = SimpleAgent(name="闲聊", llm=llm, system_prompt="你是助手")
    with quiet() as buf:
        answer = agent.run("你好")

    c.eq(answer, "你好，我是假模型。", "run() 原样返回模型的回复")
    c.eq(llm.call_count, 1, "无工具时只调用 1 次模型（不进入工具循环）")
    first_msgs = llm.calls[0]
    c.eq([m["role"] for m in first_msgs], ["system", "user"],
         "发给模型的消息是 [system, user] 两条")
    c.eq(len(agent.get_history()), 2, "这一轮问答被写进历史（user + assistant 各一条）")
    c.ok(not agent.has_tools(), "没传 tool_registry → has_tools() 为 False")

    # ────────────────────────────────────────────────────────────
    c.section("C. SimpleAgent —— 工具循环（文本协议 + 回灌）")

    reg = _calc_registry()
    llm2 = FakeLLM(replies=[
        "[TOOL_CALL:char_count:深度学习]",       # 第 1 轮：模型要求用工具
        "『深度学习』一共 4 个字符。",              # 第 2 轮：拿到工具结果后的最终回答
    ])
    agent2 = SimpleAgent(name="会算的", llm=llm2, tool_registry=reg)
    with quiet():
        answer2 = agent2.run("『深度学习』有几个字？")

    c.eq(answer2, "『深度学习』一共 4 个字符。", "拿到最终回答")
    c.eq(llm2.call_count, 2, "工具循环让模型被调用了 2 次（第 1 轮要工具、第 2 轮收尾）")
    c.ok("[TOOL_CALL:char_count:深度学习]" in llm2.flat_text(),
       "第 1 轮的原始回复（含工具调用标记）被作为 assistant 消息留在了对话里")
    c.ok("工具执行结果" in llm2.flat_text(),
       "⭐ 工具结果被作为【user 消息】回灌进对话（这是模型能看见工具干了什么的唯一通道）")
    c.ok("4 个字符" in llm2.flat_text(), "回灌的内容里确实带着工具的真实返回值")

    c.ok(SimpleAgent(name="x", llm=FakeLLM(), tool_registry=reg,
                     enable_tool_calling=True).enable_tool_calling,
         "传了 registry 且 enable_tool_calling=True → 工具开关为真")
    c.ok(not SimpleAgent(name="y", llm=FakeLLM(),
                         enable_tool_calling=True).enable_tool_calling,
         "⚠ 只传 enable_tool_calling=True 但没传 registry → 被 and 短路压成 False（静默不启用）")

    # ────────────────────────────────────────────────────────────
    c.section("D. ReActAgent —— Thought / Action / Finish")

    react_llm = FakeLLM(replies=["Thought: 这题我直接就知道\nAction: Finish[42]"])
    react = ReActAgent(name="react", llm=react_llm, tool_registry=_empty_registry())
    with quiet():
        react_answer = react.run("生命的意义是什么")

    c.eq(react_answer, "42", "识别 Finish[...] 并把方括号里的内容作为最终答案")
    c.eq(react_llm.call_count, 1, "直接给出 Finish 时只需 1 次模型调用")

    react_llm2 = FakeLLM(replies=["Thought: 我需要查一下\nAction: missing_tool[参数]"])
    react2 = ReActAgent(name="react2", llm=react_llm2, tool_registry=_empty_registry())
    with quiet() as buf2:
        react2.run("随便问")
    log2 = buf2.getvalue()
    c.ok("未找到" in log2, "工具不存在时，错误信息被打印成「观察」",
         f"日志里没找到「未找到」字样")
    c.ok("Observation" in react_llm2.flat_text(),
         "⭐ 错误被当作 Observation 回灌进下一轮提示词（模型因此有机会自我纠正）")
    c.eq(react_llm2.call_count, 5, "模型反复要同一个不存在的工具 → 跑满 max_steps=5 次才熔断")

    # 同一个日志陷阱：解析失败时也会打「已达到最大步数」
    react_llm3 = FakeLLM(replies=["这句话里既没有 Thought 也没有 Action"])
    react3 = ReActAgent(name="react3", llm=react_llm3, tool_registry=_empty_registry())
    with quiet() as buf3:
        react3.run("随便问")
    log3 = buf3.getvalue()
    c.ok(
        react_llm3.call_count == 1 and "已达到最大步数" in log3,
        "⚠ 只调用了 1 次（步数根本没跑满）却同样打印「已达到最大步数」—— 日志归因错误",
        f"调用了 {react_llm3.call_count} 次；日志含该句={('已达到最大步数' in log3)}",
    )

    # ────────────────────────────────────────────────────────────
    c.section("E. ReflectionAgent —— 初稿 → 反思 → 提前停止")

    ref_llm = FakeLLM(replies=["这是初稿。", "无需改进"])
    ref = ReflectionAgent(name="reflect", llm=ref_llm)
    with quiet():
        ref_answer = ref.run("写一段介绍")

    c.eq(ref_answer, "这是初稿。", "反思认为「无需改进」时，返回的是最后那版执行结果")
    c.eq(ref_llm.call_count, 2, "调用次数 = 1 次初稿 + 1 次反思（命中停止条件，没有进入优化）")
    c.ok("无需改进" in ref_llm.flat_text(), "停止条件确实是靠【字符串匹配】判断的")

    ref_llm2 = FakeLLM(replies=["初稿", "还可以更好", "第二稿", "还可以更好", "第三稿", "还可以更好", "第四稿"])
    ref2 = ReflectionAgent(name="reflect2", llm=ref_llm2, max_iterations=2)
    with quiet():
        ref2.run("写一段介绍")
    c.eq(ref_llm2.call_count, 5, "反思一直不认可时：1 次初稿 + 2 轮 × (反思 + 优化) = 5 次（靠轮数上限熔断）")

    # ────────────────────────────────────────────────────────────
    c.section("F. PlanAndSolveAgent —— 先规划，再逐步执行")

    ps_llm = FakeLLM(replies=[
        '```python\n["第一步：查资料", "第二步：写摘要"]\n```',   # Planner 拿到的计划
        "资料查完了",                                              # 执行第 1 步
        "摘要写好了",                                              # 执行第 2 步
    ])
    ps = PlanAndSolveAgent(name="plan", llm=ps_llm)
    with quiet():
        ps_answer = ps.run("帮我做一份摘要")

    c.eq(ps_answer, "摘要写好了", "返回最后一步的执行结果")
    c.eq(ps_llm.call_count, 3, "调用次数 = 1 次规划 + 2 步执行")
    c.ok("第一步：查资料" in ps_llm.flat_text(), "计划的步骤被逐条送进了执行提示词")

    ps_llm2 = FakeLLM(replies=["我不会写计划"])          # 模型没按 ```python 格式回答
    ps2 = PlanAndSolveAgent(name="plan2", llm=ps_llm2)
    with quiet():
        ps_answer2 = ps2.run("随便问")
    c.ok("无法生成有效的行动计划" in ps_answer2,
       "⚠ 计划解析失败时会退化成「任务终止」（而不是抛异常，所以调用方看不出是解析失败）",
       f"实际返回 {ps_answer2!r}")

    # ────────────────────────────────────────────────────────────
    c.section("G. ToolAwareSimpleAgent —— 重写父类解析：括号状态机 vs 正则")

    tricky = "[TOOL_CALL:search:Python[3.12] 新特性]"
    plain = SimpleAgent(name="p", llm=FakeLLM())
    aware = ToolAwareSimpleAgent(name="a", llm=FakeLLM())

    p_calls = plain._parse_tool_calls(tricky)
    a_calls = aware._parse_tool_calls(tricky)

    c.eq(len(p_calls), 1, "父类正则能匹配到这条调用")
    c.eq(p_calls[0]["parameters"], "Python[3.12",
         "⚠ 父类的 [^\\]]+ 在第一个 ] 就收尾 → 参数被截断")
    c.eq(a_calls[0]["parameters"], "Python[3.12] 新特性",
         "⭐ 子类用括号配对状态机 → 参数完整保留（这就是重写父类方法的动机）")

    # ────────────────────────────────────────────────────────────
    c.section("H. 流式的真与假（同名接口，体验不同）")

    sa_llm = FakeLLM(replies=["答案"])
    sa = SimpleAgent(name="s", llm=sa_llm)
    with quiet():
        sa_chunks = list(sa.stream_run("问"))

    fc_llm = FakeLLM(replies=["答案"])
    fc = FunctionCallAgent(name="f", llm=fc_llm)      # 不传 registry → 降级为普通聊天
    with quiet():
        fc_chunks = list(fc.stream_run("问"))

    c.eq(len(sa_chunks), 2, "SimpleAgent.stream_run 是【真流式】：回复有 2 个字符就 yield 2 片")
    c.eq(sa_chunks, ["答", "案"], "而且顺序就是逐字符的")
    c.eq(len(fc_chunks), 1,
         "⚠ FunctionCallAgent.stream_run 是【假流式】：先跑完 run() 再一次性 yield（只有 1 片）")
    c.known_defect(
        "FunctionCallAgent.stream_run 名不副实",
        "它的实现是 `result = self.run(...); yield result`（docstring 也自认「流式调用暂未实现」）。"
        "所以 hasattr(obj,'stream_run') 为真 ≠ 真的流式 —— 判断能力必须读实现或跑时序",
    )
