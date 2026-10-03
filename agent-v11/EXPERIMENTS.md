# agent-v11 实验手册

> **7 个实验，每个 1~2 分钟 —— 全部为面试服务。**
> 做完一个实验，你就多得一段「亲测证据」可以讲；
> 最后用实验 7 把它们组装成六个答案。
>
> ⚠️ token 数、措辞每次都会浮动 —— **看特征，别抠数字**。
> 本手册里的数字是 2026-10-03 那次真跑的值（deepseek-flash）。

---

## 开始之前（2 分钟）

**① 开终端、进环境、进目录**

```bash
myenv
cd /Users/kuki/homework/hello-agent/agent-v11
```

✅ 提示符变成 `(myenv) ... agent-v11 %`

❓ `command not found: myenv`？换成全路径：
`source /Users/kuki/work/claude-code/jupyterlab/myenv/bin/activate`

**② 先跑离线自检（不联网、不花钱）**

```bash
python main.py --selftest
```

✅ 最后一行是 `✓ 全部 35 项断言通过`

**③ 确认依赖和 Key**

```bash
pip show langchain langgraph | grep -E '^(Name|Version)'
python main.py --config | head -12
```

✅ 应该看到 `langchain 1.x` / `langgraph 1.x`，以及 `LLM_API_KEY (已设置)`。

**④ 花费预告**

两个离线 demo 不花钱；五个真实 demo 每个约 1~3k tokens，全跑一遍约 10k tokens。
（把整个手册做完大约 15~20k tokens —— 比 v10 的实验便宜，因为这一版是「零件级」的。）

**⑤ 起点状态**

v11 没有持久化概念，每次启动都是干净的。唯一会留痕的是 `notes/`——
如果哪个 demo 给你写了笔记（大部分不会），删掉就好，不影响任何实验。

---

## 实验 1：消息巡礼 —— 「dict 到类型」 🆕

**🎯 要感受什么**

协议的对象化：手写时代靠「字典里放对 key」，组件时代靠类型约束 —— **但协议本身一个字没变**。

**📋 步骤**

```bash
python main.py --demo messages
```

✅ **你应该看到**（离线的，结果稳定）：

```
    SystemMessage   type=system   content='你是一个简洁的助手。'
    HumanMessage    type=human    content='今天几号？'
    AIMessage       type=ai       content='' tool_calls=«get_current_time…»
    ToolMessage     type=tool     content='…（星期六）' tool_call_id=call_1
```

💡 **所以呢（面试可讲）**

- 记住一句话：**「消息类型 = 协议的对象化；纪律变成了类型」**。
  举 v1 的例子：tool 消息必须带 `tool_call_id`（否则服务端拒绝）——
  手写时代这是「你记得就没事」，现在 `ToolMessage` 缺 id **构造直接失败**。
- 追问防御：类型不是新协议，**同一个 OpenAI 协议的两种表达**——
  从「字典 + 纪律」到「类 + 校验」。

---

## 实验 2：@tool —— 白拿的 schema（含互动） 🆕

**🎯 要感受什么**

从「手写 25 行 ToolDefinition」到「3 行装饰器」；以及——**docstring 的地位**。

**📋 步骤**

```bash
python main.py --demo tools
```

✅ **你应该看到**：calculator 的完整 JSON schema 打印出来（name/description/parameters 全齐）。

**🛠 互动：亲手改一段 docstring，看 schema 怎么变**

打开 `components.py`，找到 `get_current_time`，把它的 docstring 从

```python
"""获取当前的本地日期和时间。凡是涉及『今天』『现在』『当前时间』的问题都必须调用
本工具，不要凭记忆或猜测作答。"""
```

改成一句冷淡的：

```python
"""获取时间。"""
```

再跑 `python main.py --demo tools`。

✅ **你应该看到**：schema 的 description 跟着变了 —— 模型看到的就是它。

❓ 觉得没差别？**这正是重点**：在 v1 的实验 10 里我们验证过
「工具描述就是提示词」——描述写得烂，模型就不会在正确的时候用它。
**框架把 schema 自动化了，但没把「写清楚」自动化。**

💡 **所以呢（面试可讲）**

- 「@tool 省的到底是什么？」——省的是**重复的机械劳动**（JSON Schema 拼装），
  省不掉**设计劳动**（这个工具该在什么时候被用、描述怎么写）。
- 改完记得把 docstring 改回去（或留着，感受你自己的项目会怎样）。

---

## 实验 3：工具回环 —— 协议还是那个协议（含破坏） 🆕

**🎯 要感受什么**

一次真实的「模型要工具 → 你执行 → 回灌 → 收工」；然后**亲手把它改坏**，
看服务端的守卫有多严。

**📋 步骤**

```bash
python main.py --demo tool-loop
```

✅ **你应该看到**（数字/措辞会浮动）：

```
  ① 模型的第一反应（AIMessage）：
     tool_calls: calculator({'expression': '12 * 34'})  «call_id=call_00_…»
     tool_calls: get_current_time({})  «call_id=…»
  ② 我方执行工具 + 用 ToolMessage 回灌（注意 tool_call_id 必须配对）：
     calculator → 408
     get_current_time → 2026-10-03 16:27:40（星期六）
  ③ 模型的最终回答：……
```

**🛠 破坏性互动：把 call_id 配对改坏**

打开 `components.py`，在 `demo_tool_loop` 里找到这一行：

```python
messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
```

把 `call["id"]` 改成 `"wrong-id"`，保存，再跑：

```bash
python main.py --demo tool-loop
```

✅ **你应该看到**（真跑的捕获）：跑挂了，服务端直接点名：

```
✗ demo 跑挂了：OpenAIInvalidRequestError: Error code: 400 - {'error': {'message':
"An assistant message with 'tool_calls' must be followed by tool messages responding
to each 'tool_call_id', The following tool_call_ids did not have response messages:
wrong-id …"}}
```

❓ **为什么要做这个破坏？** ——为了亲眼确认：**框架没有魔法**。
它帮你把消息变成了类型，但**配对规则是服务端的**，框架只是提前拦你（或让你
挂得容易懂）。这就是「抽象之下的协议」。

改回 `call["id"]` 再跑一遍，确认恢复正常。

💡 **所以呢（面试可讲）**

- 「用框架还需要懂协议吗？」——**需要**。得意忘形的地方就是事故现场：
  你 v1 手写时踩过的坑（配对、空 content），一个都没消失。

---

## 实验 4：LCEL —— invoke / batch / stream 一套协议 🆕

**🎯 要感受什么**

「组合的标准化」：批量、流式为什么在 LCEL 里是「免费」的。

**📋 步骤**

```bash
python main.py --demo lcel
```

✅ **你应该看到**：

```
  ① invoke（单次）： → '56'
  ② batch：['56', '18', '99']          ← 三问并发，你一行没写
  ③ stream：分片逐个到达，拼起来 → '56088'
```

❓ 分片数量/内容每次不同（服务端的切法），拼起来的结果是对的就行。

💡 **所以呢（面试可讲）**

- 一句话：「LCEL 的真正卖点是 **Runnable 协议**——invoke/batch/stream
  是同一套接口，所以组装出来的链自动继承这些能力」。
- 追问「那它不能做什么？」——**它是数据流（DAG）**：
  `a | b | c` 有方向、无环。而 Agent 需要的是**循环 + 状态**
  （这正好接上 v12 的动机）。

---

## 实验 5：结构化输出 —— 翻车现场 ⭐

**🎯 要感受什么**

框架「舒适区」的边缘：**同一份代码，三种官方姿势，在 DeepSeek 上全部翻车**；
以及两种能work的兜底。这是全版最有面试谈资的实验。

**📋 步骤**

```bash
python main.py --demo structured
```

✅ **你应该看到**（我们真跑的结果）：

```
  ① json_schema 模式：      ✗ 400 …response_format type is unavailable now…
  ② function_calling 模式： ✗ 400 …Thinking mode does not support this tool_choice…
  ③ json_mode 裸用：        ✗ 解析失败：模型自创键名（title/type/content…）
  ④ json_mode + 字段写进提示词： OK
  ⑤ 纯提示词 + 自己解析：        OK
```

**🤔 思考题**：「明天让你在 DeepSeek 的项目里做结构化输出，你选哪条路？为什么？」

💡 **所以呢（面试可讲）**

- 骨架回答：「框架抽象的是**接口形状**，服务端约束凌驾其上。
  这个 demo 里三种官方姿势都因为服务端的限制失败，最后靠『把要求说清楚 + 兜底解析』
  和『工具即 schema』（我们 v8 手写的路子）跑通。
  **所以用框架也要知道协议细节** —— 不然你连为什么坏、该换哪个旋钮都不知道。」
- 追问「这不是框架的锅吗？」——一半一半：框架没法凭空造出服务端不支持的
  能力；但它的错误信息确实把服务端原文藏在里面——**会读报错的人赢**。

---

## 实验 6：揭盖子 —— create_agent 的内核 ⭐

**🎯 要感受什么**

「一行得到 agent」很方便 —— 然后看它**里面是什么**。

**📋 步骤**

```bash
python main.py --demo agent
```

✅ **你应该看到**：

```
  回答：88 × 11 = 968，现在时间……
  它的内部结构（agent.get_graph()）——节点列表：
     node: __start__ / model / tools / __end__
  它的类型定义在：langgraph.graph.state  ← 注意这个名字
```

💡 **所以呢（面试可讲）**

- 这段就是 Q5「LangChain 和 LangGraph 什么关系」的**实物证据**：
  「LangChain 1.x 的 agent 门面，内核就是一张 LangGraph 的图——
  `create_agent` 返回的对象，模块名就是 `langgraph.graph.state`。」
- 再补一刀：「所以当你要加**检查点、人工审批、重规划**这些控制能力时，
  LangChain 的界面就不够了——**得把这张图拆开自己画**。这就是 v12 要做的。」

---

## 实验 7：模拟面试 —— 把六个实验变成六个答案 🆕⭐

**🎯 要感受什么**

前面六个实验攒的证据，能不能**在 60 秒内说出口**。面试考的不是你跑过什么，
而是你能不能把跑过的东西讲成判断力。

**📋 步骤**

1. 打开 [README.md](README.md) 的「面试问答」一节，**先不要看答案**；
2. 对下面每题自己说 60 秒（出声或写三行要点）：
   - Q1 LangChain 是什么？
   - Q2 LCEL 是什么？
   - Q3 抽象太重怎么看？
   - Q4 什么时候不该上？
   - Q5 和 LangGraph 什么关系？
   - Q6 框架和模型不兼容怎么办？
3. 再对答案，检查你的版本里有没有**至少一个本版实证**（demo 名 + 看到了什么）。

✅ **合格线**：每题都能说出「一个定位 + 一个实证」。
比如 Q5 的实证是「`--demo agent` 打印了 `langgraph.graph.state`」。

❓ 卡壳了？回到对应实验重跑一遍 —— **证据是讲出来的底气**。

---

# 实验记录表

| # | 实验 | 我看到的（关键输出） | 面试时我要讲的点 |
|---|---|---|---|
| 1 | 消息巡礼 | | 协议的对象化 = |
| 2 | @tool | 改 docstring 后 description | 省了 __、省不掉 __ |
| 3 | 工具回环（破坏） | 改坏 call_id 的报错： | 框架没有魔法 = |
| 4 | LCEL | batch/stream 的表现 | 卖点是 __；做不了 __ |
| 5 | 结构化翻车 | 哪种姿势work了？ | 框架抽象 vs 服务端约束 = |
| 6 | 揭盖子 | agent 的模块名是 | 两框架关系 = |
| 7 | 模拟面试 | 卡壳的题： | （空缺的回去补实验） |

---

**附：这一版没做实验、但面试可能被问的（去 [concepts/](../concepts/README.md) 补）**

- LangChain 的 **RAG 组件**（Document/Splitter/Retriever）——和 v5~v7 手写检索对照；
- **回调系统 / LangSmith**（观测）——和 v9 评估对照；
- **Middleware / 中间件**——想给 create_agent 加「非标准逻辑」的官方口子。
