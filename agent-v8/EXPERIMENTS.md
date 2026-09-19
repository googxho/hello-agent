# agent-v8 实验手册

> **7 个实验，每个 1~2 分钟。一个实验只测一个点。**
>
> 实验 1~3 是这一版的主线：**交卷 → 两个出口 → 批量消费**。
> 做完实验 3，你会第一次看到「程序读懂模型的话」的完整样子 ——
> 那也是下一版（评估）的雏形。
>
> ⚠️ 本手册里所有「你应该看到」都是**真跑出来的输出**。
> 答案措辞、耗时、confidence 每次都会浮动 —— **看特征，别抠数字**。

---

## 开始之前（3 分钟）

**① 开终端、进环境、进目录**

```bash
myenv
cd /Users/kuki/homework/hello-agent/agent-v8
```

✅ 提示符变成 `(myenv) ... agent-v8 %`

❓ `command not found: myenv`？换成全路径：
`source /Users/kuki/work/claude-code/jupyterlab/myenv/bin/activate`

**② 确认环境没问题**

```bash
python main.py --selftest
```

✅ 最后一行是 `✓ 全部 152 项断言通过`
（这一步**不联网、不花钱** —— 重试/放弃用剧本客户端重放）

**③ 确认配置和向量服务**

```bash
python main.py --config | grep -E 'STRUCTURED|EMBEDDING_API'
```

✅ 应该看到：

```
EMBEDDING_API_KEY             (已设置)               [共享]
LLM_STRUCTURED_MAX_ATTEMPTS   3                      [v8 新增]
```

**④ 关于「记忆」** —— 本版的新说明

这一版的实验大多走 **`--ask`（单发模式）**：**不读会话文件、也不写** ——
每个问题独立，天然不会有「上一个实验污染下一个」的问题。

所以：**只有交互对话类实验**才需要 `LLM_PERSIST_ENABLED=0`（本手册里是实验 1 附带的对照）。
开始之前**不用**背这条规则 —— 但要记住它的原因：
**凡是「退出换配置再进来」的步骤，都要问一句「这次重启会把哪些旧信息带回来？」**

**⑤ 把资料目录清干净，埋两条笔记**

```bash
rm -rf notes && mkdir -p notes
cat > notes/perf.md <<'EOF'
# 性能

性能优化有三个方向：加缓存、减少重复计算、把串行改并行。
EOF
cat > notes/model.md <<'EOF'
# 设备

设备型号 X1000-3 的故障代码是 ERR_5002，解决办法是重启电源模块。
EOF
```

✅ `ls notes` 看到两个文件

---

# 第一组 · 交卷、出口、批量

## 实验 1：第一次让程序「读懂」它 ⭐⭐⭐

**🎯** 让模型**交卷**而不是「说完就完」：它调用 `submit_result`，
把回答变成四个可校验的字段 —— 你第一次看到「回答」被拆成结构。

**⏱ 2 分钟**

**起点** 🆕 单发模式（`--ask`，不读会话，天然干净）

**📋 步骤**

**第 1 步** —— 跑这一句：

```bash
python main.py --ask "我想让程序跑得更快，有什么建议？"
```

**👀 你应该看到** —— 结尾长这样（真实输出，中间有省略）：

```
  🔧 调用 search_notes({"query": "程序性能优化 加速"})
  🔍 检索 程序性能优化 加速 （混合） → 1 段命中 （关键词提名 1 段 + 向量提名 1 段 → RRF k=60 · 扫描 2 块 · 216.4 ms）
      perf.md:3（性能） · 关键词#1（打分 1.02） + 向量#1（余弦 0.735） → 融合 0.0328
  ← 在资料库里找到 1 段相关内容（按相关度从高到低）：
  …
  🔧 调用 submit_result({"answer": "你的笔记里正好有一条性能优化的记录（perf.md）…（显示截断）
  ← 已收到结构化结果，校验通过。本次任务结束。
╭────────────── 结构化结果 · 第 1 次提交就通过 · 4228 ms ──────────────╮
│ answer                                                                 │
│ 你的笔记里正好有一条性能优化的记录（perf.md），里面总结的三个方向可以   │
│ 直接用：                                                               │
│ 1. **加缓存** —— 把重复请求/重复计算的结果存下来，下次直接取。          │
│ 2. **减少重复计算** —— 检查循环里有没有可以提到循环外的运算……          │
│ 3. **把串行改并行** —— 相互独立的任务并发执行，而不是一个等一个。       │
│ found       True                                                       │
│ sources     perf.md                                                    │
│ confidence  0.9                                                        │
╰────────────────────────────────────────────────────────────────────────╯
```

> 💡 两次实测的对照（帮你建立「什么会变、什么不会变」的直觉）：
> 另一次运行里，第一条 search 的 query 是「**程序 性能 优化 加速**」，
> 中途它**又查了一次**「项目 技术栈 瓶颈 慢」（想找你项目的具体信息），
> `confidence` 报的是 **0.55**。变量是：改写的搜索词、查几次、措辞、耗时、confidence；
> **不变的是**：`submit_result` 必被调用、`sources` 指向 `perf.md`、卡片格式一致。

⚠️ **数字和措辞每次都不一样**。你要确认的是这五样特征：

| 确认什么 | 它在告诉你 |
|---|---|
| `🔧 调用 submit_result(...)` | 它**交卷了**（而不是用文字作答）|
| `…（显示截断）` | 长参数日志只截**显示**，执行用的是原文（不然屏幕被 JSON 刷爆）|
| `← 已收到结构化结果，校验通过` | 一次通过，没有重试 |
| 卡片上的 `found / sources / confidence` | **回答变成了字段** —— 程序现在能读了 |
| 标题里的 `第 1 次提交就通过` | 重试账本也在明面上 |

**第 2 步（对照）** —— 交互对话**没变**，还是自由文本：

```bash
LLM_PERSIST_ENABLED=0 python main.py
```

问同样的问题。✅ 它会正常回答，但**不会**出现 `submit_result` ——
为什么？把横幅里的工具清单对比一下：交互模式只有
`calculator, get_current_time, save_note, search_notes`。

> 💡 这就是「什么时候**不该**用结构化」：给人看的聊天，不该被塞进 JSON。
> `submit_result` 只在 `--ask` 模式才出现在模型眼前（selftest J6 盯着这条）。

**💡 所以呢**

```
之前：模型的输出 = 一段散文        → 想知道出处？靠猜
现在：模型的输出 = 一个可校验的字典 → 想读哪个字段读哪个字段
```

---

## 实验 2：两个出口 —— 给人看 vs 给程序吃 ⭐⭐⭐

**🎯** 同一件事，两种输出契约。重点看 `--json` 的**分流**：
stdout 只留 JSON，日志全部去 stderr。

**⏱ 2 分钟**

**起点** 🆕 单发模式

**📋 步骤**

**第 1 步** —— 把两个流分开存：

```bash
python main.py --ask "笔记里 ERR_5002 这个故障代码是什么情况？" --json > /tmp/ask_out.json 2> /tmp/ask_err.log
echo "退出码 exit=$?"
```

✅ 应该看到 `退出码 exit=0`

**第 2 步** —— 看 stdout（要求：**只有**一个 JSON 文档）：

```bash
python -m json.tool --no-ensure-ascii /tmp/ask_out.json
```

**👀 你应该看到**（真实输出，answer 有换行）：

```json
{
    "question": "笔记里 ERR_5002 这个故障代码是什么情况？",
    "ok": true,
    "result": {
        "answer": "你的笔记里确实有关于 ERR_5002 的记载（出处：model.md）：\n\n- 该故障代码对应的是**设备型号 X1000-3**；\n- 处理办法是**重启电源模块**。…",
        "found": true,
        "sources": ["model.md"],
        "confidence": 0.95
    },
    "meta": {
        "attempts": 1,
        "failures": [],
        "sources_seen": ["model.md"],
        "error": null,
        "elapsed_ms": 3021.1
    }
}
```

**第 3 步** —— 看 stderr（日志还是原来那个样子，还是给人看的）：

```bash
head -14 /tmp/ask_err.log
```

**👀 你应该看到**（真实输出，前 14 行）：

```
📦 结构化单发模式 · 模型 deepseek-flash · 提交上限 3 次
询问：笔记里 ERR_5002 这个故障代码是什么情况？
  · 本次请求 1,193 tokens
  🔧 调用 search_notes({"query": "ERR_5002"})
  🔍 检索 ERR_5002 （混合） → 1 段命中 （关键词提名 1 段 + 向量提名 1 段 → RRF k=60 · 扫描 2 块 · 625.8 ms）
      model.md:3（设备） · 关键词#1（打分 0.28） + 向量#1（余弦 0.556） → 融合 0.0328
  ← 在资料库里找到 1 段相关内容（按相关度从高到低）：
  …
```

**注意 `meta` 里那句分工**：

- `result` 是**模型交的**（过了校验）
- `meta` 是**程序记的**（几次提交、见过哪些文件、报没报错）

两层刻意分开 —— 将来做评估时，「模型说的」和「机器看到的」绝不混在一起。

**💡 所以呢**

```
日志 → stderr（给人看）     JSON → stdout（给程序吃）
```

这是所有 CLI 工具的通用契约。以后你写任何「LLM 给我干活」的脚本，
第一件事就是**把这两个流分开** —— 否则日志会把 JSON 弄脏，下游解析器直接爆炸。

---

## 实验 3：批量 —— 程序第一次消费了模型的话 ⭐⭐⭐

**🎯** 这一版存在的理由，浓缩成一张表：**三个问题一次跑完，程序自己读结果**。
看第三个问题 —— 它不知道的事，它**没编**。

**⏱ 2 分钟（其中大半是等模型）**

**起点** 🆕 单发模式（每个问题独立）

**📋 步骤**

**第 1 步** —— **整段复制粘贴**：

```bash
python - <<'PY'
import json, subprocess

qs = ["怎么让程序跑得更快", "ERR_5002 是什么故障", "我的护照号码是多少"]
print(f"{'':2} {'found':<6} {'sources':<12} 问题")
print("─" * 58)
for q in qs:
    out = subprocess.run(["python", "main.py", "--ask", q, "--json"],
                         capture_output=True, text=True, timeout=180)
    d = json.loads(out.stdout)
    r = d["result"] or {}
    mark = "✓" if d["ok"] else "✗"
    print(f"{mark:2} {str(r.get('found')):<6} {','.join(r.get('sources') or []) or '—':<12} {q}")
    print(f"     confidence={r.get('confidence')} · attempts={d['meta']['attempts']} · {d['meta']['elapsed_ms']:.0f}ms")
PY
```

**👀 你应该看到**（真实输出）：

```
   found  sources      问题
──────────────────────────────────────────────────────────
✓  True   perf.md      怎么让程序跑得更快
     confidence=0.9 · attempts=1 · 4071ms
✓  True   model.md     ERR_5002 是什么故障
     confidence=0.95 · attempts=1 · 2675ms
✓  False  —            我的护照号码是多少
     confidence=0.9 · attempts=1 · 5921ms
```

一行一行读：

- 前两行：`found=True`、`sources` 指向**正确的文件** —— 程序自动核对了「答哪了」
- 第三行：**`found=False`、`sources` 空** —— 笔记里没有护照号码，它**如实说没有**，
  而且这个「没有」是**机器可判定**的（不用人去读答案判断）

**💡 所以呢**

这就是 v7 收尾时说的「评估的第一步」：

```
人肉模式（v7 之前）：跑 20 个问题 → 人眼逐条读 → 人才知道对错
机器模式（v8 之后）：跑 20 个问题 → 程序读 found/sources/ok → 机器就能筛出「不对劲」的
```

⚠️ 但注意：**这张表还只是「能读」，不是「会判」** ——
「期望命中什么」还没定，「答得对不对」还没打分。
那是下一版（07 · 评估）的正题 —— 你现在手里的是它的地基。

---

# 第二组 · 把校验和重试拆开看

## 实验 4：校验器动手玩 ⭐⭐

**🎯** 校验器是个**纯函数**（数据进、问题列表出）—— 不用模型、不用网络，
你可以一行行喂坏数据，看它逐条拦下。

**⏱ 1 分钟**

**起点** ↩️ 不用启动任何东西

**📋 步骤**

**第 1 步** —— **整段复制粘贴**：

```bash
python - <<'PY'
import structured

seen = {"perf.md"}   # 假装本次检索「见过」perf.md
good = {"answer": "性能优化有三个方向…", "found": True, "sources": ["perf.md"], "confidence": 0.9}

cases = [
    ("完全合规", good),
    ("answer 是空的", {**good, "answer": ""}),
    ("引用了没见过的文件", {**good, "sources": ["幽灵文件.md"]}),
    ("found 和 sources 打架", {**good, "found": False}),
    ("confidence 超范围", {**good, "confidence": 1.5}),
]
for label, payload in cases:
    problems = structured.validate_submission(payload, seen)
    verdict = "✓ 通过" if not problems else f"✗ 拦下：{problems[0]}"
    print(f"  {label:<16} {verdict}")
PY
```

**👀 你应该看到**（真实输出）：

```
  完全合规             ✓ 通过
  answer 是空的       ✗ 拦下：answer 必须是非空字符串（现在拿到的是：''）
  引用了没见过的文件        ✗ 拦下：sources 里的 ['幽灵文件.md'] 不是本次检索见过的文件（见过的只有：perf.md） —— 出处必须是真实检索结果，不能编造
  found 和 sources 打架 ✗ 拦下：found=false 却带了 sources —— 说没有依据，就不该列出来源
  confidence 超范围   ✗ 拦下：confidence 必须是 0~1 之间的数字（现在拿到的是：1.5）
```

**注意每条错误的写法**：哪里错、现在是什么、该是什么、怎么改 ——
这不是给你看的，是**回灌给模型看的**（下一轮它照着修，一次全改完）。

**💡 所以呢**

**「出处必须是真实检索结果」这条是这一版的灵魂** ——
它防的是 RAG 最阴险的一种错误：**格式全对，出处是编的**。
防线不在提示词里，在**程序**里：

```
模型见过的文件 ← search_notes 每返回一段就记一笔（轨迹）
模型提交的 sources ← 必须 ⊆ 轨迹   ← 交叉核对，编造必被抓
```

---

## 实验 5：重试现场 —— 坏到底会怎样 ⭐⭐

**🎯** 真实模型不可能被你指挥着「先交错的、再交对的」—— 所以用**剧本客户端**
把两个剧本各演一遍：坏→改好，和坏到底→放弃。

**⏱ 2 分钟**

**起点** ↩️ 不用启动 Agent（剧本 + 假客户端，全离线）

**📋 步骤**

**第 1 步** —— **整段复制粘贴**：

```bash
python - <<'PY' 2>&1 | grep -vE '^\s*·|^\s*$'
import structured
from agent import Agent
from context import ContextManager
from config import SYSTEM_PROMPT
from main import _ScriptedLLMClient, _submit_call
from tools import SEEN_SOURCES, reset_seen_sources

# 手工造一份「轨迹」：假装刚才检索到了 perf.md（省去真检索，聚焦重试机制）
reset_seen_sources()
SEEN_SOURCES.add("perf.md")

print("── 剧本 A：坏 → 好 ──")
structured.begin_run(max_attempts=3)
script = [
    _submit_call("c1", {"answer": "…", "found": True, "sources": ["幽灵文件.md"], "confidence": 0.8}),
    _submit_call("c2", {"answer": "…", "found": True, "sources": ["perf.md"], "confidence": 0.8}),
]
ctx = ContextManager(SYSTEM_PROMPT + structured.SYSTEM_ADDENDUM, max_tokens=100_000, keep_recent=8)
Agent(_ScriptedLLMClient(script), ctx).run("演示问题")
state = structured.STATE
print(f"提交次数={state.attempts} · 最终结果={'拿到了' if state.result else '没有'}")
structured.finish_run()

print()
print("── 剧本 B：坏 → 坏 → 放弃（上限 2）──")
structured.begin_run(max_attempts=2)
script = [
    _submit_call("c1", {"answer": "…", "found": True, "sources": ["幽灵文件.md"], "confidence": 0.8}),
    _submit_call("c2", {"answer": "…", "found": True, "sources": ["幽灵文件.md"], "confidence": 0.8}),
]
ctx = ContextManager(SYSTEM_PROMPT, max_tokens=100_000, keep_recent=8)
try:
    Agent(_ScriptedLLMClient(script), ctx).run("演示问题")
except structured.StructuredGiveUp as exc:
    print(f"StructuredGiveUp 穿透出来：{exc}")
print(f"状态：result={structured.STATE.result} · attempts={structured.STATE.attempts}")
structured.finish_run()
PY
```

**👀 你应该看到**（真实输出，长行有截断）：

```
── 剧本 A：坏 → 好 ──
  🔧 调用 submit_result({"answer": "…", "found": true, "sources": ["幽灵文件.md"]…
  ✗ 错误：ValueError: 第 1 次提交不合规（还剩 2 次机会）：sources 里的 ['幽灵文件.md']
   不是本次检索见过的文件（见过的只有：perf.md） —— 出处必须是真实检索结果，不能编造。
   请修正后重新调用 submit_result。
  🔧 调用 submit_result({"answer": "…", "found": true, "sources": ["perf.md"]…
  ← 已收到结构化结果，校验通过。本次任务结束。
提交次数=2 · 最终结果=拿到了

── 剧本 B：坏 → 坏 → 放弃（上限 2）──
  🔧 调用 submit_result({"answer": "…", "found": true, "sources": ["幽灵文件.md"]…
  ✗ 错误：ValueError: 第 1 次提交不合规（还剩 1 次机会）：sources 里的 ['幽灵文件.md'] …
  🔧 调用 submit_result({"answer": "…", "found": true, "sources": ["幽灵文件.md"]…
StructuredGiveUp 穿透出来：连续 2 次提交不合规，放弃。最后一次的问题：sources 里的
['幽灵文件.md'] 不是本次检索见过的文件（见过的只有：perf.md） —— 出处必须是真实检索结果，不能编造
状态：result=None · attempts=2
```

**三个看点**：

1. **重试不需要新机制** —— 校验失败抛 ValueError，agent 把异常包成
   「工具错误」回灌（v1 就有的行为）—— 模型看到错、改完再交
2. **「还剩 N 次机会」** —— 上限（`LLM_STRUCTURED_MAX_ATTEMPTS`）在管着，
   日志把余额也写出来了
3. **放弃会「穿透」**：`StructuredGiveUp` 不是可回灌的工具错误，
   而是「这次运行已判死刑」—— 它穿过 agent 循环，交给运行器收尸，
   **状态如实：result=None**（绝不把最后一次坏数据当成功）

**💡 所以呢**

```
失败重试 = 校验器 + v1 的工具回灌机制（没写一行循环代码）
放弃     = 一个专门的异常 + 状态清零（不装成功）
```

**评估最怕的不是失败，是一堆「假装成功」的数据** —— 所以「放弃」和「成功」
一样，都是**一等公民的结局**。

---

# 第三组 · 坏消息的样子

## 实验 6：失败面 —— 退出码和信封 ⭐

**🎯** 给程序吃的东西，报错也得是程序能读的。看一次完整的失败：

**⏱ 1 分钟**

**起点** 🆕 单发模式

**📋 步骤**

**第 1 步** —— 故意把模型名写错：

```bash
LLM_MODEL=不存在的模型 python main.py --ask "你好" --json > /tmp/ask_fail.json 2>/dev/null
echo "退出码 exit=$?"
```

✅ 应该看到 `退出码 exit=2` —— **失败也有稳定契约**（0 = 成功，2 = 结构化失败）

**第 2 步** —— 看失败信封：

```bash
python -m json.tool --no-ensure-ascii /tmp/ask_fail.json
```

**👀 你应该看到**（真实输出，错误详情有省略）：

```json
{
    "question": "你好",
    "ok": false,
    "result": null,
    "meta": {
        "attempts": 0,
        "failures": [],
        "sources_seen": [],
        "error": "BadRequestError: Error code: 400 - {'error': {'message': 'The supported API model names are deepseek-flash, … but you passed 不存在的模型.', …}}",
        "elapsed_ms": 632.0
    }
}
```

**💡 所以呢**

三种结局、三种姿态（都是机器可读的）：

| 结局 | 信封 | 退出码 |
|---|---|---|
| 拿到合规结果 | `ok=true` + `result` | 0 |
| 放弃 / 报错 | `ok=false` + `meta.error` + `meta.failures` | 2 |
| 模型压根没交卷 | `ok=false` + `error` 写明「它可能直接输出了文字」| 2 |

调用方（脚本、流水线、未来的评估器）可以**只看 `ok` 字段**决定下一步 ——
不用解析人类语言。

---

# 第四组 · 想一想

## 实验 7：思考题 —— 什么时候**不该**用结构化 ⭐

**🎯** 新玩具不是万能药。想清楚它的边界，比会用它更重要。

**⏱ 2 分钟（对着前面实验想，不用动手）**

**问题一** —— 为什么**交互对话**没改成结构化？（提示：回想实验 1 第 2 步的横幅）

**问题二** —— 实验 4 里「sources 必须 ⊆ 本次检索轨迹」这条规则，
如果改成「sources 必须是文件系统里存在的文件」，会漏掉什么坏情况？
（提示：文件存在 ≠ 这次答案的依据是它 —— 模型可能引用一个**它压根没查过**的文件，
格式上毫无破绽）

**问题三** —— 三条实现路线（原生 Structured Outputs / JSON 模式 / 工具即 schema），
如果你做的产品必须支持「换模型不改代码」，你会选哪条？各自赌的是什么？

> 想系统了解外面怎么做：看 [ECOSYSTEM.md](../ECOSYSTEM.md) 的 06 一节
> （OpenAI Structured Outputs、Instructor、Pydantic… 各自走的是哪条路线）。

**💡 所以呢**

- **给人看的，别塞 JSON；给程序吃的，别留散文** —— 判断标准就一句：下一站是人眼还是代码
- 校验能保「形式 + 可核对的事实」，保不了「内容对不对」—— 后者要**评估**（下一版）
- 路线选择是**成本与绑定**的取舍，没有免费午餐

---

# 实验记录表

| 我想知道 | 我的答案 | 实验 |
|---|---|---|
| 交卷（submit_result）和普通回答，界面差在哪？| | 1 |
| `--json` 时，stdout 和 stderr 各装了什么？退出码几？| | 2 |
| 批量三问里，哪一问是 `found=False`？它是怎么「没编」的？| | 3 |
| 校验器拦下了哪五种坏数据？错误信息是写给谁看的？| | 4 |
| 「还剩 N 次机会」里的 N 由什么决定？试满后 `result` 是什么？| | 5 |
| 失败时退出码是几？信封里哪个字段让程序知道「别用这个结果」？| | 6 |
| 什么时候不该用结构化？为什么 sources 要对「轨迹」而不是对「文件」？| | 7 |
| ⚠️ 为什么本版的实验大多不用管 `LLM_PERSIST_ENABLED`？| | 开始之前 |

---

# 做完之后

你应该对下面这些有了一手的判断：

- ✅ **结构化输出 = 交卷**：用 v1 的工具机制，把「最终回答」变成可校验的字典
- ✅ **schema 约束 ≠ 保证**：类型 / 语义 / **交叉核对**（轨迹 ⊆ sources）三层校验
- ✅ **重试长在回灌机制上**：校验失败抛错 → agent 回灌 → 模型改 —— 没写新循环
- ✅ ⭐ **放弃也是结果**：试满上限 → `ok=false` + 退出码 2，绝不假装成功
- ✅ **两个出口**：stdout 给程序，stderr 给人 —— 这是 LLM 脚本化的第一课
- ✅ **程序第一次消费了模型的话**（实验 3）—— 评估的地基，但不是评估本身
- ✅ **边界感**：给人看的聊天不要塞 JSON；confidence 是自报的，别当真理

## 下一步

| 你的感受 | 下一步做什么 |
|---|---|
| 「有输出了，可我**不知道答得对不对**」 | → **评估**：给一批问题配上「期望命中」（知识块 07）—— 下一版 |
| 「confidence 说是 0.95，我凭什么信」 | → 拿评估数据校准它（还是 07）|
| 「想给输出加字段」 | → 改 `SUBMIT_RESULT_DEF` + `validate_submission` 两处，跑 selftest J 组 |
| 「提交总失败，模型不听话」 | → 调大 `LLM_STRUCTURED_MAX_ATTEMPTS`；或看 `meta.failures` 对症下药 |
| 「想让它输出更复杂的东西（嵌套数组/表格）」 | → 从给 schema 加一层嵌套开始 —— 先感受到『模型填嵌套填不对』的痛，再想办法 |

**先感受到痛，再动手解决。** 每一版都回答四个问题：上一版痛在哪、
这一版用什么最小概念解决、怎么验证解决了、下一版的新痛是什么。
