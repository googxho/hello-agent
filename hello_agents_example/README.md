# hello_agents_example

六种 Agent 范式，每种一个最小可跑的 example —— **用真实 API，直接跑**。

```bash
cd hello_agents_example
python e1_simple.py          # 换成 e2 / e3 / e4 / e5 / e6 就是别的范式
```

> 用仓库的虚拟环境：`/Users/kuki/work/claude-code/jupyterlab/myenv/bin/python`
> API Key 从仓库根的 `.env` 读取（`_env.py` 负责加载）。

---

## 六个例子的选择理由

| 文件 | 范式 | 场景 | 为什么选它 |
|---|---|---|---|
| `e1_simple.py` | `SimpleAgent` | 电商客服查订单 | 单步任务、一问一答；最省，且不要求模型支持 function calling |
| `e2_function_call.py` | `FunctionCallAgent` | 会议安排 | 参数多且强类型（时长是数字、参与人是**列表**），文本协议表达不了 |
| `e3_react.py` | `ReActAgent` | 线上故障排查 | **下一步取决于上一步看到什么**，事先写不出完整步骤 |
| `e4_reflection.py` | `ReflectionAgent` | 产品文案打磨 | 好坏不需要外部信息，只要**多改几轮** |
| `e5_plan_solve.py` | `PlanAndSolveAgent` | 搬家预算规划 | 任务**可事先拆成有序步骤**，全局一致、可拿给人看 |
| `e6_tool_aware.py` | `ToolAwareSimpleAgent` | 需要留痕的内部助手 | 需求同 e1，只多一条「**每次工具调用都要看得见**」 |

**一句话判据**：加能力 → 写 `Tool`；换策略 → 换范式；都不合用才继承 `Agent`。
（这六个例子**没有一个**继承 `Agent` —— 全是挑现成范式装上去。）

---

## 结构

```
_env.py        10 行公共准备：让 import 找到源码 + 读 .env（仅此而已）
e1 … e6.py     每个 40~70 行，自成一体，复制就能改成你自己的
```

每个文件的格式都一样：

```
文件头    场景 / 为什么选它 / 不选别的 / 怎么跑
工具      继承 Tool + @tool_action（只有需要工具的例子才有）
装配      一行 new 出框架的 Agent
运行      agent.run("...") 真实调用
```

---

## 两个小提醒

1. **会花 API 额度**：e4（反思）最贵，默认 7 次调用，例子里已调到 `max_iterations=2`。
2. `@tool_action` 方法的**形参必须叫 `input`**（框架的注册表固定传 `{"input": ...}`）；
   而且工具说明里不带参数名，所以需要工具的例子里都用 `system_prompt` 把格式钉死。
