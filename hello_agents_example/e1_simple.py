"""① SimpleAgent —— 电商客服

场景：用户问「我的订单什么时候到」，Agent 调一个工具查一下再回答。

为什么选它：单步任务、一问一答。无工具时只花 1 次模型调用，是最省的一种；
           而且它走「提示词约定格式 + 正则解析」，不要求模型支持 function calling，
           换成本地小模型也能跑。

不选别的：ReAct 要「边查边想」太重；FunctionCall 的 JSON Schema 在这里用不上；
         PlanSolve 不需要计划。

运行：python e1_simple.py
"""
import _env

_env.setup()

from hello_agents import HelloAgentsLLM
from hello_agents.agents.simple_agent import SimpleAgent
from hello_agents.tools.base import Tool, ToolParameter, tool_action
from hello_agents.tools.registry import ToolRegistry


class OrderTool(Tool):
    def __init__(self):
        super().__init__(name="order", description="订单查询工具", expandable=True)

    def run(self, parameters):
        return "请调用 order_query"

    def get_parameters(self):
        return []

    @tool_action("order_query", "查询订单状态。参数是订单号")
    def _query(self, input: str) -> str:          # 形参必须叫 input（框架的约定）
        return f"订单 {input}：已发货，预计明天下午送达"


registry = ToolRegistry()
registry.register_tool(OrderTool(), auto_expand=True)

agent = SimpleAgent(
    name="客服小助手",
    llm=HelloAgentsLLM(),
    # 工具说明里不带参数名，所以这里把调用格式写清楚（给个示例值最不容易误解）
    system_prompt="需要查订单时只输出一行：[TOOL_CALL:order_query:ORD-1001]"
                  "（把 ORD-1001 换成要查的订单号）",
    tool_registry=registry,
)

question = "我的订单 ORD-1001 什么时候到？"
print(f"👤 {question}\n")
print(f"🤖 {agent.run(question)}")
