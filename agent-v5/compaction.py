"""
上下文压缩层 —— v3 的核心。

═══════════════════════════════════════════════════════════════════════
这一版要解决的问题
═══════════════════════════════════════════════════════════════════════

v2 的上下文管理是**丢弃**：超预算就把最老的整轮对话删掉。

实测下来（v2 实验 7）后果是这样的：你让它记住「紫色河马」，聊 8 轮无关的，
再回头问 —— 它答不上来，而且**没有任何报错**：

    ✂ 丢弃 N 轮旧对话（20,826 → 2,214 tokens）

占位符里什么信息都没有。暗号不是变模糊了，是**彻底不存在了**。

═══════════════════════════════════════════════════════════════════════
这一版引入的最小概念
═══════════════════════════════════════════════════════════════════════

    丢弃之前，先让 LLM 把它概括成一条摘要 —— 用少量 token 保住关键信息。

    v2:  旧对话 ──────────────────────────→ （删除，信息量 = 0）
    v3:  旧对话 ─→ LLM 概括 ─→ 摘要文本 ─→ （插入上下文，信息量 > 0）

于是上下文有了**分辨率差异**，这就是「记忆分层」：

    ┌─ 长期记忆：一条摘要 system 消息     ← 很省 token，保留关键结论
    └─ 短期记忆：user / assistant 原文    ← 完整精确，但很贵

最近的看得清，久远的只剩素描。**这不是「上下文变短了」，
而是「上下文按重要性分了档」。**

═══════════════════════════════════════════════════════════════════════
代价：压缩要花钱、花时间
═══════════════════════════════════════════════════════════════════════

压缩一次 = 一次额外的 LLM 调用（读进被压内容 + 吐出摘要）。

所以这一版真正的工程问题不是「怎么写摘要」，而是**这笔账划不划算**：

    成本：summary 指令（约 250 tokens）+ 被压缩内容 + 摘要输出
    收益：（被压缩内容 - 摘要）的 token，而且这个收益**每次请求都在兑现**

规则是「能剪就不压，要压就压一次管很久」——
宁可一次多压点，也不要每轮压一点点（每次调用都有固定开销）。

═══════════════════════════════════════════════════════════════════════
这一层对外提供什么
═══════════════════════════════════════════════════════════════════════

    Summarizer      压缩器的「接口」（一个 Protocol）
    LLMSummarizer   真实实现：调模型
    render_history()  把 messages 渲染成给压缩模型看的纯文本
    SummarizeResult   一次压缩的结果（文本 + 账单）

⚠️ 为什么要把 Summarizer 抽成一个接口？两个理由：

  1. **可离线测试**。自检不能联网、不能花钱，所以注入一个假的压缩器
     （见 main.py 里的 _FakeSummarizer），就能把整条压缩流水线测穿。
  2. **可替换**。真实产品里压缩器可能就是另一个 Agent、或者一个抽取式算法。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from llm import LLMClient

# ══════════════════════════════════════════════════════════════════════
# 摘要指令 —— 这是写给「压缩模型」的提示词
# ══════════════════════════════════════════════════════════════════════
#
# 这里值得多花点心思：**「摘要写什么」直接决定压缩有没有用。**
#
# 如果只说「请概括一下」，模型会给你一段读起来很顺、但没用的散文 ——
# 用户交代的「暗号是紫色河马」很可能被它当成无关细节丢掉，
# 而这恰恰是唯一重要的东西。
#
# 所以这里用**结构化模板**，把「必须保留」的类别一条条列死。

COMPACTION_INSTRUCTION = """你是一个对话压缩器。你的任务是把一段对话历史压缩成一条简短的摘要，
供同一个助手在后续对话中继续使用 —— 就像一个人记笔记：不抄全文，但关键的一条都不能漏。

必须遵守：
1. 只写历史里真实出现过的信息。不要推测、不要补充、不要美化、不要总结出历史里没有的结论。
2. 下面这些信息【必须】保留，一条都不能漏：
   · 用户交代过的关键信息：代号、暗号、名称、数字、日期、地址、偏好、限制条件
   · 用户的目标和需求
   · 已经得到的结论、算出的结果
   · 尚未完成、答应过要做的待办
3. 下面这些可以丢弃：寒暄、开场白、重复的解释、已经被推翻的中间过程。
4. 用第三人称陈述（「用户要求…」「助手算出…」），不要写成对话体。
5. **必须简短** —— 压缩的全部意义就在于短。宁可漏掉细节，也不要写成流水账。

严格按下面的格式输出（某一条确实没有内容就写「无」）：

【对话摘要】
目标：……
关键信息：……
已确认的事实：……
未完成：……"""


# ══════════════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════════════


@dataclass
class SummarizeResult:
    """一次压缩的产物，外加它的账单。

    prompt_tokens / completion_tokens 是这次压缩调用**真实花掉**的 token
    （来自 API 的 usage），elapsed 是真实耗时 —— 这三个数让你能回答
    「压缩到底值不值」这个问题。
    """

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed: float = 0.0


class Summarizer(Protocol):
    """压缩器的接口。

    真实实现要调 LLM；测试里可以塞一个假的。
    ContextManager 只依赖这个形状，不关心背后是谁。
    """

    def summarize(
        self,
        messages: list[dict[str, Any]],
        previous_summary: str | None,
    ) -> SummarizeResult:
        """把一段消息压缩成摘要文本。

        previous_summary 不为 None 时，表示之前已经压过一次 ——
        这时要输出「旧摘要 + 新内容」**合并后**的完整摘要，
        而不是只总结新内容。（否则每次压缩都会把上辈子的记忆冲掉。）
        """
        ...


# ══════════════════════════════════════════════════════════════════════
# 把消息渲染成文本
# ══════════════════════════════════════════════════════════════════════


def render_history(messages: list[dict[str, Any]]) -> str:
    """把 messages 渲染成一段纯文本，交给压缩模型。

    为什么不直接把 messages 数组丢过去？

      · 压缩模型只做「读 → 写」，用不到 tool calling 那套协议
      · 渲染时可以把**工具调用的过程**也写进去 —— 直接丢数组的话，
        「它调过 calculator（参数是 1234*5678）」这条信息会丢失，
        而摘要里恰恰可能需要它
    """
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "user":
            lines.append(f"[用户] {content}")
        elif role == "assistant":
            for call in msg.get("tool_calls") or []:
                fn = call.get("function") or {}
                lines.append(f"[助手 → 调用工具 {fn.get('name')}] 参数：{fn.get('arguments')}")
            if content:
                lines.append(f"[助手] {content}")
        elif role == "tool":
            lines.append(f"[工具 {msg.get('name')} 返回] {content}")
        elif role == "system":
            lines.append(f"[系统] {content}")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# 真实实现：调 LLM 来压缩
# ══════════════════════════════════════════════════════════════════════


class LLMSummarizer:
    """用一次 LLM 调用，把历史压成摘要。"""

    def __init__(self, client: LLMClient, *, model: str, max_tokens: int) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def summarize(
        self,
        messages: list[dict[str, Any]],
        previous_summary: str | None,
    ) -> SummarizeResult:
        history = render_history(messages)

        if previous_summary:
            # ⚠️ 这里必须说清楚「输出的是合并结果」。
            #    否则模型很容易只总结新增的那一段，把之前的记忆整个丢掉 ——
            #    那就不是压缩，是换一次方式失忆。
            user_content = (
                "【已有摘要】（这是更早之前压缩出来的内容，请把它和下面的新对话"
                "合并成一条完整的摘要，不要重复、也不要遗漏）\n"
                f"{previous_summary}\n\n"
                "【新增的对话历史】\n"
                f"{history}\n\n"
                "请输出合并后的完整摘要（保持同样的格式）。"
            )
        else:
            user_content = f"【需要压缩的对话历史】\n{history}\n\n请输出摘要。"

        started = time.time()
        reply = self.client.complete(
            [
                {"role": "system", "content": COMPACTION_INSTRUCTION},
                {"role": "user", "content": user_content},
            ],
            use_tools=False,          # 压缩不需要工具
            purpose="summarize",      # 单独记账，方便算「压缩花了多少钱」
            model=self.model,
            max_tokens=self.max_tokens,
        )
        elapsed = time.time() - started

        text = (reply.message.content or "").strip()
        if not text:
            # 空摘要等于「压缩失败」。抛出去让上层走回退路径，
            # 而不是往上下文里塞一条空消息假装成功。
            raise ValueError("压缩模型返回了空内容")

        return SummarizeResult(
            text=text,
            prompt_tokens=reply.prompt_tokens,
            completion_tokens=reply.completion_tokens,
            elapsed=elapsed,
        )
