"""③ ReActAgent —— 线上故障排查

场景：服务报警了。先查监控看是哪个接口出问题，再查那个服务的日志 ——
      **下一步查什么，取决于上一步看到了什么。**

为什么选它：事先根本写不出完整步骤，只能「看一眼 → 决定下一步」。
           ReAct 的 Thought / Action / Observation 循环就是干这个的；
           而且每步的 Thought 都打出来，人能看懂它为什么查这个、不查那个。

不选别的：PlanSolve 事先规划只能瞎编（不知道要查几步）；SimpleAgent 只有一轮，
         查完监控就结束了；Reflection 拿不到监控数据这些外部事实。

运行：python e3_react.py
"""
import _env

_env.setup()

from hello_agents import HelloAgentsLLM
from hello_agents.agents.react_agent import ReActAgent
from hello_agents.tools.base import Tool, ToolParameter, tool_action
from hello_agents.tools.registry import ToolRegistry


class MonitorTool(Tool):
    def __init__(self):
        super().__init__(name="monitor", description="监控与日志查询", expandable=True)

    def run(self, parameters):
        return "请调用 check_metric / check_log"

    def get_parameters(self):
        return []

    @tool_action("check_metric", "查询指标。参数是指标名")
    def _metric(self, input: str) -> str:
        # 注意：工具必须对「模型可能问的名字」都能给出**有用的观察**，尤其要明确告诉它
        # 「没有更多数据了」—— 否则模型会一直换名字重试，直到步数用尽（这个坑我踩过）
        name = input or ""
        if "错误" in name or "error" in name.lower():
            return "各服务错误率：gateway 0.2% / order-service 0.5% / user-service 18%"
        if "连接" in name or "池" in name.lower():
            return "user-service 数据库连接池：活跃 20/20（已打满），等待队列 37"
        return f"指标「{name}」不存在。已有信息足以定位根因，请直接给结论。"

    @tool_action("check_log", "查询某个服务的日志。参数是服务名")
    def _log(self, input: str) -> str:
        if "user" in (input or ""):
            return "user-service 日志：HikariPool 连接不可用，等待超时 30000ms（连接池 max=20）"
        return f"{input} 日志：无异常"


registry = ToolRegistry()
registry.register_tool(MonitorTool(), auto_expand=True)

agent = ReActAgent(name="排障助手", llm=HelloAgentsLLM(), tool_registry=registry)

question = "线上错误率报警了，帮我排查根因"
print(f"👤 {question}\n")
print(f"🤖 {agent.run(question)}")
