"""④ ReflectionAgent —— 对外文案打磨

场景：写一条产品发布文案。质量比速度重要，允许改几轮。

为什么选它：问题的关键是「写得够不够好」，而判断好不好**不需要外部信息** ——
           让模型自己看一遍、批评、再改就行。这正是「初稿 → 反思 → 优化」的用武之地。

不选别的：ReAct 的循环是「行动 → 观察外部世界」，这里没有外部世界可观察；
         PlanSolve 拆不出步骤（拆了也只是「写、改、再改」）；
         SimpleAgent 一次成稿，没有改进的机会。

代价：默认 1 + 2×3 = 7 次模型调用，是六个例子里最贵的。

运行：python e4_reflection.py
"""
import _env

_env.setup()

from hello_agents import HelloAgentsLLM
from hello_agents.agents.reflection_agent import ReflectionAgent

agent = ReflectionAgent(
    name="文案助手",
    llm=HelloAgentsLLM(),
    max_iterations=2,          # 改两轮就停，省点钱
)

question = "给我们的新产品「智能笔记」写一条 30 字以内的发布文案"
print(f"👤 {question}\n")
print(f"🤖 {agent.run(question)}")
