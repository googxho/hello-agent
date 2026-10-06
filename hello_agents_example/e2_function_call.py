"""② FunctionCallAgent —— 会议安排助手

场景：用户说「下周三下午两点和张三李四开一小时的项目会」，要变成一条结构化日程。

为什么选它：工具参数是结构化的 —— 时长是数字、参与人是**列表**。
           在 SimpleAgent 那种「key=value 逗号切分」的文本协议里，
           `attendees=["张三","李四"]` 这种结构解析不了。
           原生 function calling 由协议保证 JSON 结构，不用正则猜、也不用自己转类型。

不选别的：SimpleAgent 处理不了数组；ReAct 不是探索任务；PlanSolve 不需要计划。

运行：python e2_function_call.py
"""
from typing import List

import _env

_env.setup()

from hello_agents import HelloAgentsLLM
from hello_agents.agents.function_call_agent import FunctionCallAgent
from hello_agents.tools.base import Tool, ToolParameter, tool_action
from hello_agents.tools.registry import ToolRegistry


class ScheduleTool(Tool):
    def __init__(self):
        super().__init__(name="schedule", description="日程工具", expandable=True)

    def run(self, parameters):
        return "请调用 schedule_create"

    def get_parameters(self):
        return []

    @tool_action("schedule_create", "创建日程。title 标题，when 时间，"
                                    "duration_minutes 时长（分钟），attendees 参与人列表")
    def _create(self, title: str, when: str, duration_minutes: int,
                attendees: List[str]) -> str:
        # 到这里的 duration_minutes 已经是 int、attendees 已经是 list —— 框架按声明的类型转好了
        people = "、".join(attendees) if isinstance(attendees, list) else str(attendees)
        return f"已创建：{title}｜{when}｜{duration_minutes} 分钟｜参与人 {people}"


registry = ToolRegistry()
registry.register_tool(ScheduleTool(), auto_expand=True)

agent = FunctionCallAgent(name="会议助手", llm=HelloAgentsLLM(), tool_registry=registry)

question = "下周三下午两点，和张三、李四开一个小时的项目会"
print(f"👤 {question}\n")
print(f"🤖 {agent.run(question)}")
