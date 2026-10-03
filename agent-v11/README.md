# hello-agent v11

> 「框架化」拆成了上下两场 —— 这一场是**组件层**：
>
> **LangChain：把「和模型打交道」标准化成组件协议 —— 零件都给你备好了。**
>
> 下一场（v12）是 LangGraph：**流程的控制权**。两个版本合起来回答一个面试高频题：
> 「LangChain 和 LangGraph 到底什么区别、怎么选？」

---

## 🧪 先做实验，再看代码

**[EXPERIMENTS.md](EXPERIMENTS.md)** —— 7 个实验、每个 1~2 分钟，**全部为面试服务**：
做完一个实验，你就多得一段「亲测证据」可以讲。

不想等的话，两条命令先感受一下：

| 命令 | 你会看到 |
|---|---|
| `python main.py --demo tools` | 25 行手写 schema → 一个 `@tool` 装饰器（离线、不花钱） |
| `python main.py --demo structured` | ⭐ 框架给的结构化输出三种姿势，在 DeepSeek 上**全部翻车** —— 最有面试谈资的一条 |

---

## 每一版都回答四个问题

### ① 上一版遇到了什么具体问题？

**手写账本的利息，v10 时已经很高了。** 数一数 v1~v10 里「重造过的轮子」：

| 你手写过的东西 | 在哪一版 | 现在它长什么样 |
|---|---|---|
| 工具协议（ToolDefinition + ToolParameter） | v1 起 | 每个工具 ~25 行描述 |
| 消息拼装（dict + tool_call_id 手工配对） | v1 起 | 纪律全靠自己守 |
| 模型调用封装 + token 账本 | v9 | 手写 LLMClient + snapshot 差分 |
| 结构化输出（submit + 校验 + 重试） | v8 | 整整一个版本 |
| 评估回路 | v9 | 1044 行 evaluator.py |

每一层都能跑、都教会了我们东西 —— 但**换一家模型、开一个新项目，这些全部重来一遍**。
而现实是：这些零件在 2026 年的生态里**早就被标准化了**。

### ② 这一版引入什么最小概念？

> **LangChain = 组件层。把「和模型打交道」的五件事变成协议：**

```
消息  →  Message 类型（不再是手拼 dict）
模型  →  ChatModel 协议（ChatOpenAI：invoke / batch / stream / bind_tools）
工具  →  @tool（schema 从函数签名 + docstring 自动推导）
组装  →  LCEL  `prompt | model | parser`（一切皆 Runnable）
Agent →  create_agent（一行得到 ReAct 循环 —— ⚠️ 内核是 LangGraph，见实验 6）
```

一张对照表（你的手写史 ↔ 组件时代）：

| 能力 | 你手写的 | 组件时代的 |
|---|---|---|
| 工具定义 | `ToolDefinition` + 参数说明 ~25 行/个 | `@tool` + 一段 docstring |
| 消息 | `{"role": ..., "tool_call_id": ...}` | `ToolMessage(content=…, tool_call_id=…)`（**缺 id 直接构造失败**） |
| 模型调用 | `LLMClient.complete()` | `ChatOpenAI(...)`——换家只改 `.env` |
| 账本 | snapshot 差分 | `usage_metadata`（还带「缓存命中」「推理 token」明细） |
| 结构化输出 | v8 的 submit_result + 校验 + 重试 | `with_structured_output`（⚠️ 见实验 5：在 DeepSeek 上全灭） |
| 批量 / 流式 | 自己写循环 / 自己拼分片 | `batch()` / `stream()`——协议自带 |

**「最小」的边界**：v11 只做组件层演示（七个 demo、一条命令一个），
**不做**「用 LangChain 重写前面所有版本」——那是重复劳动。
要对照的，是「同一个零件，两种写法」。

### ③ 如何验证它真的解决了？

**离线验证**（`--selftest`，**35 项断言**，零 API Key、零花费）：

- `@tool` 推导出的 schema 与手写版**同宗同构**（type/function/parameters/properties/required 一个不少）
- 消息类型把 v1 的老坑变成了「构造即报错」（缺 `tool_call_id` 直接拒收）
- 白名单计算器、slug、JSON 抠取容错——全部照旧成立

**真实运行验证**（7 个 demo，全部跑过捕获）：

| demo | 结论 |
|---|---|
| `tools` | 25 行手写 → 3 行装饰器，schema 一分不差 |
| `tool-loop` | 回环协议与 v1 手写**一模一样**（call_id 配对、结果回灌）——框架没魔法 |
| `usage` | 账本升级：`cache_read` 和 `reasoning` 明细（实测一次回答 output 146 tokens 里 **111 是推理**） |
| `lcel` | invoke/batch/stream 一套协议 —— 批量和流式是「免费」的 |
| `structured` | ⭐ **三种官方姿势在 DeepSeek 上全灭**，两种兜底能work（详见下） |
| `agent` | `create_agent` 一行跑通；**盖子揭开：`langgraph.graph.state`** |

⭐ 最有价值的一条实测——框架不是万能的（这三条失败都真实发生了）：

```
① json_schema 模式      → 400：DeepSeek 不支持 response_format=json_schema
② function_calling 模式 → 400：thinking 模式拒绝强制 tool_choice
③ json_mode 裸用        → 解析失败：模型自由发挥键名（自创字段、漏字段）
④ json_mode + 提示词里写清字段 → 成功（但要你回到「说清楚」的自觉）
⑤ 纯提示词 + 自己解析          → 成功（v8 手写时代的日常）
```

**结论（面试可直接说）**：框架抽象的是「接口形状」，而**服务端的约束凌驾其上**。
在 DeepSeek 这类服务端上，v8 手写的「工具即 schema + 校验重试」反而是最稳的姿势——
「手写教会你的判断力，框架给不了」。

### ④ 下一版的新痛点是什么？

组件层解决了「零件」，但**流程的控制权没有解决**——两条实测线索：

1. **LCEL 是数据流（DAG）**：`prompt | model | parser` 表达不了「循环 + 状态」。
   而 Agent 的本质就是循环 + 状态。（实验 4 的面试点）
2. **agent 门面把循环封装成了黑盒**：`create_agent` 一行很方便，
   但它内部那张图（`state → model → tools → …`）你**看不见、改不了**——
   想加「检查点 / 人工审批 / 重规划」，LangChain 的界面就不够了。

而揭开盖子后那个模块名已经给了答案：**`langgraph.graph.state`**。

**→ 下一版（v12）：LangGraph —— 把那张图拆开，自己画。**
（v10 留下的两个真实伤口「断在第 5 步就永远断了」「危险操作没人把关」，
将在 v12 里第一次有药可治。）

---

## ⭐ 面试问答（这一版的隐藏主菜）

> 每个问题给「60 秒答题骨架 + 本版亲测证据」。**先自己答一遍再往下看。**

### Q1 「LangChain 是什么？」

**骨架**：LLM 应用的**组件标准层**——统一消息、模型、工具、组装方式。
核心抽象是 **Runnable 协议**（一切组件都自带 invoke/batch/stream）。
一句话定位：「它不让你变聪明，它让你不用重新造零件。」

**本版证据**：`--demo tools`——25 行手写 `ToolDefinition` 变成 3 行 `@tool`；
`--demo tool-loop`——协议（call_id 配对、结果回灌）和你手写的一模一样。

**追问「那它解决什么、不解决什么？」**：
不解决编排（LCEL 是 DAG，表达不了循环+状态）、不替你想提示词
（docstring 写得烂，工具就没人会用——v1 实验 10 的教训），
也不替你和服务端博弈（见 Q6）。

### Q2 「LCEL 是什么？」

**骨架**：用 `|` 把组件串成管道（`prompt | model | parser`）。
本质不是语法糖，是 **Runnable 协议**：任何组件都统一暴露
invoke / batch / stream / ainvoke——所以**批量、流式、组合可以「免费」挂上**。

**本版证据**：`--demo lcel`——同一个 chain，三种调用方式各一行；
修 bug 时只改链上某一段（这是组合的复利）。

### Q3 「LangChain 抽象太重 / 老被吐槽，你怎么看？」

**骨架**（三分法）：
① 批评成立的部分：过度封装的黑盒、版本剧变（0.x → LCEL → 1.x）、
   出问题时要「穿透抽象」调试；
② 但也有实在价值：标准化让团队协作和换供应商变便宜；
③ 正确姿势：**知道抽象底下的协议**——会用的人不怕抽象，不会用的人被抽象坑。

**本版证据**：`--demo structured`——官方三种「舒适姿势」在 DeepSeek 上全部翻车。
**光会调框架 API 不够，你必须知道底下发生了什么**（response_format 是什么、
tool_choice 为什么被拒），这才是「抽象税」的真实样子。

### Q4 「什么时候不该上 LangChain？」

**骨架**：三-种情况——① 学习阶段（手写一遍才能建立心智模型：tool_call_id、
消息协议、error 回灌，都是手写时才学扎实的）；② 单点小需求（为了一个小功能
引入一棵依赖树不值得）；③ 需要非标准控制流（马上撞上它的边界）。

**本版证据**：这个仓库的 v1~v10 就是「手写存 → 组件取」的过程；
v1 的 40 行主循环到现在都能跑。看看 `pip show langchain` 的依赖列表，
再想想 v1 的工具定义其实只要 25 行——**平衡点自己称**。

### Q5 「LangChain 和 LangGraph 什么关系？」

**骨架**：两代问题，两层职责——
- **LangChain（组件层）**：把和模型打交道的零件标准化（v11 的主题）；
- **LangGraph（编排层）**：把 agent 的流程变成显式的状态图——
  循环、分支、检查点、人机协同（v12 的主题）；
- 关系：同门（LangChain 团队），LangGraph 建立在 langchain-core 之上；
  **LangChain 1.x 的 agent 门面（create_agent），内核就是 LangGraph 的图**——
  「组件层的尽头是编排层」。

**本版证据**：`--demo agent`——`type(agent).__module__` 打印出
`langgraph.graph.state`，眼见为实。

### Q6 「框架和模型不兼容怎么办？」（高频实战题）

**骨架**（四步）：① 最小复现锁定是哪一层坏了（框架层？服务端约束？）；
② 读报错里的**服务端原文**（「unavailable now」「does not support」是服务端说的，
不是框架说的）；③ 找到该换的模式/旋钮（换个 method、降级到通用姿势）；
④ 兜底：**退回到协议本身手写**（我们的 v8 就是这么活的）。

**本版证据**：`--demo structured` 的四连——三次失败 + 两次兜底，
把「框架层 vs 服务端」的分界线用真实报错画出来了。

---

## 快速开始

```bash
myenv
cd agent-v11

# 0) 离线自检 —— 不需要任何 Key，不花钱
python main.py --selftest          # 35 项断言

# 1) 两个离线 demo（无 Key 也能玩）
python main.py --demo messages     # 消息类型巡礼
python main.py --demo tools        # @tool 自动 schema

# 2) 五个真实 demo（要 Key）
python main.py --demo tool-loop    # 一次工具回环
python main.py --demo usage        # token 账本
python main.py --demo lcel         # LCEL invoke/batch/stream
python main.py --demo structured   # 结构化输出翻车现场 ⭐
python main.py --demo agent        # create_agent 揭盖子 ⭐

# 3) 一次全跑
python main.py --demo all

# 4) 看配置
python main.py --config
```

---

## 诚实的话（这一版自己承认的）

1. **demo 不覆盖 LangChain 的全部**：RAG 组件（Document/Splitter/Retriever）、
   记忆、回调系统都没展开——挑的是「和你 v1~v10 手写史能对上」的零件。
   （RAG 部分对照已在 concepts 与 ECOSYSTEM 里有）
2. **「25 行 → 3 行」的对比要诚实看**：省的是「你自己写 schema 推导」，
   没省「想清楚工具怎么描述」——那部分反而更值钱了。
3. **实测数据跟着版本走**：DeepSeek 的约束（不支持 json_schema 等）
   是 2026-10 的现场；半年后再跑 `--demo structured`，结果可能不一样——
   **这本身就是这堂课的一部分：服务端在变，抽象跟着变，判断力不变。**

## 文件地图

```
agent-v11/
├── config.py       配置层 + ChatModel 工厂（换供应商 = 改 .env）
├── components.py   ⭐ 七个 demo 的正文（三个工具 + 各层巡礼 + 面试点注释）
├── main.py         CLI：--demo / --selftest / --config
├── requirements.txt 第一次引入框架依赖（langchain / langchain-core / langchain-openai / langgraph）
├── README.md       你正在看的（含面试问答）
└── EXPERIMENTS.md  7 个实验（面试导向）
```

> 下一场：**v12 · LangGraph** —— 用同一批零件，给 v10 的伤口上药：
> 「断在第 5 步」怎么救、危险操作怎么审批、计划死了怎么重规划。
