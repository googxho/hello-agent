"""⑥ ToolAwareSimpleAgent —— 需要留痕的内部助手

场景：公司内部助手，员工用它查客户数据。合规要求：每次工具调用都要留痕。

为什么选它：需求本质还是「一轮对话 + 调工具」，只多一条「我要看见每次调用」。
           ToolAwareSimpleAgent 就是 SimpleAgent 加了一个 tool_call_listener 回调 ——
           不用换范式，更不用自己写一个 Agent 子类。

不选别的：SimpleAgent 没有观测点，跑完终端上看不见工具被调用过什么；
         自己写子类是为一个回调重写整套循环，没必要。

运行：python e6_tool_aware.py
"""
import _env

_env.setup()

from hello_agents import HelloAgentsLLM
from hello_agents.agents.tool_aware_agent import ToolAwareSimpleAgent
from hello_agents.tools.base import Tool, ToolParameter, tool_action
from hello_agents.tools.registry import ToolRegistry

audit_log = []


def audit(info: dict) -> None:
    """每次工具调用都会走这里 —— 真实项目里就写审计库、发监控、或者拦下来要审批。"""
    audit_log.append(info)
    print(f"   📝 审计留痕：tool={info['tool_name']} 参数={info['raw_parameters']!r}")


class CrmTool(Tool):
    def __init__(self):
        super().__init__(name="crm", description="客户数据查询", expandable=True)

    def run(self, parameters):
        return "请调用 crm_query"

    def get_parameters(self):
        return []

    @tool_action("crm_query", "查询客户信息。参数是客户编号")
    def _query(self, input: str) -> str:
        return f"客户 {input}：合同金额 128,000 元"


registry = ToolRegistry()
registry.register_tool(CrmTool(), auto_expand=True)

agent = ToolAwareSimpleAgent(
    name="内部助手",
    llm=HelloAgentsLLM(),
    system_prompt="查客户数据时只输出一行：[TOOL_CALL:crm_query:C-88]（把 C-88 换成要查的编号）",
    tool_registry=registry,
    tool_call_listener=audit,
)

question = "帮我查一下客户 C-88 的合同金额"
print(f"👤 {question}\n")
print(f"🤖 {agent.run(question)}")
print(f"\n本次共留痕 {len(audit_log)} 条")
