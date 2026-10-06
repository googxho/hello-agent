"""⑤ PlanAndSolveAgent —— 搬家预算规划

场景：给出一份搬家计划和预算。

为什么选它：这件事**事先就能拆成有序步骤** —— 清点物品 → 询价 → 定日期 → 汇总预算。
           先一次性规划出来，全局一致、不容易漏项，而且计划可以拿给人看或审批。

不选别的：ReAct 走一步看一步，容易只顾眼前、漏掉全局；
         SimpleAgent 一轮问答给不出有条理的清单。

运行：python e5_plan_solve.py
"""
import _env

_env.setup()

from hello_agents import HelloAgentsLLM
from hello_agents.agents.plan_solve_agent import PlanAndSolveAgent

agent = PlanAndSolveAgent(name="搬家规划师", llm=HelloAgentsLLM())

question = "帮我做一份搬家计划，并给出预算"
print(f"👤 {question}\n")
print(f"🤖 {agent.run(question)}")
