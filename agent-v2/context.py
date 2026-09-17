"""
上下文管理层 —— v2 的核心。

═══════════════════════════════════════════════════════════════════════
这一版要解决的问题
═══════════════════════════════════════════════════════════════════════

v1 的 self.messages 只增不减：每轮对话都往里 append，工具返回的大块内容
也永远躺在里面。于是 token 随对话线性增长，最终必然撞上模型的上下文窗口，
API 直接报错 —— 而 v1 对此毫无准备。

（注意：v1 的 MAX_ROUNDS 只保护「单次提问内部」的循环，
  对「跨轮次累积」完全无能为力。）

═══════════════════════════════════════════════════════════════════════
这一版引入的最小概念
═══════════════════════════════════════════════════════════════════════

    给上下文设一个 token 预算，超预算时优雅地「忘记」最不重要的部分。

两级剪枝：

  第一级：把最早的 role="tool" 消息的 content 换成一句占位符
            —— 只丢内容，不丢结构，伤害最小

  第二级：如果还超，按「整轮」丢弃最老的对话
            —— 从一条 user 消息开始，到下一条 user 之前为止

通常先做第一级（伤害更小），但**动手前会先算一笔账**（look-ahead）：
如果预判光剪工具结果根本不够，就直接跳到第二级 —— 因为那些轮次反正
马上要被整轮删掉，先剪一遍纯属白忙。丢完之后再回头补剪，榨干最后一点。

    （这个 look-ahead 是实测逼出来的：最早的版本无条件「先剪再丢」，
     结果出现过「剪枝 18 条工具结果」紧接着「丢弃 18 轮旧对话」的场面 ——
     剪掉的恰好就是随后被删掉的那批。剪枝报告还会因此撒谎，
     显示剪了 18 条，实际一条都没留下。）

为什么是「替换内容」而不是「删除消息」？这是本版最关键的一点：

    OpenAI 协议要求每条 role="tool" 消息都能对应上前面某条 assistant
    消息里的 tool_call_id。你把 tool 消息删掉，配对就断了，API 会报：

        messages with role 'tool' must be a response to a preceding
        message with 'tool_calls'

    而只把 content 换掉，消息本身还在原位，配对关系天然保持完好，
    token 却照样省下来了。

两条护栏：

  · 当前轮永不动  —— 最后一条 user 消息之后的内容一律不剪，
                     因为那正是模型下一步要用的
  · 源头限流      —— 单个工具返回值超长就地截断（见 agent.py）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# 被剪掉的工具结果会变成这句话。
# 特意写明「已剪枝」而不是留空 —— 让模型知道这里原来有内容，
# 而不是以为工具当初就返回了个空字符串。
#
# ⚠️ 它必须**足够短**：这个字符串会在上下文里出现很多次，占位符越长，
#    剪枝省下的 token 就越少。所以刻意砍到 10 个 token 左右。
SNIP_PLACEHOLDER = "[早期工具结果已剪枝]"

# 每条消息本身有固定开销（role、name、分隔符等），估算时加上。
_PER_MESSAGE_OVERHEAD = 4


def estimate_tokens(text: str) -> int:
    """粗略估算一段文本的 token 数。

    规则来自经验：
      · 中日韩字符 ≈ 1 字 1 token
      · 英文/数字/符号 ≈ 4 字符 1 token（向上取整）

    误差大约 ±20%，**只用来做前瞻决策**（要不要剪）。
    真实用量以 API 返回的 usage.prompt_tokens 为准，那个才是事实。
    两个数字在 CLI 里都会显示出来，你可以自己对照误差有多大。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff" or "\u3040" <= ch <= "\u30ff")
    other = len(text) - cjk
    return cjk + (other + 3) // 4  # 向上取整


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """估算整个 messages 列表的 token 数。"""
    total = 0
    for msg in messages:
        total += _PER_MESSAGE_OVERHEAD
        content = msg.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content)
        # tool_calls 里的函数名和参数也是要占 token 的
        for call in msg.get("tool_calls") or []:
            fn = call.get("function") or {}
            total += estimate_tokens(str(fn.get("name") or ""))
            total += estimate_tokens(str(fn.get("arguments") or ""))
    return total


@dataclass
class TrimReport:
    """一次剪枝的结果，用来向用户汇报「刚才动了什么手脚」。"""

    snipped_tool_results: int = 0
    dropped_rounds: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    # 已经两级都用尽、却依然超预算 —— 说明顶到了「最近消息保护区」这块地板。
    # 这不是 bug，而是我们主动选择保护最近上下文的结果。
    hit_floor: bool = False

    @property
    def did_trim(self) -> bool:
        return self.snipped_tool_results > 0 or self.dropped_rounds > 0

    def describe(self) -> str:
        if not self.did_trim:
            return "未剪枝"
        parts = []
        if self.snipped_tool_results:
            parts.append(f"剪枝 {self.snipped_tool_results} 条工具结果")
        if self.dropped_rounds:
            parts.append(f"丢弃 {self.dropped_rounds} 轮旧对话")
        text = "，".join(parts) + f"（{self.tokens_before:,} → {self.tokens_after:,} tokens）"
        if self.hit_floor:
            text += " ⚠ 已顶到保护区，无枝可剪"
        return text


class ContextManager:
    """管理 messages 列表，并在超出预算时剪枝。"""

    def __init__(self, system_prompt: str, *, max_tokens: int, keep_recent: int) -> None:
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.keep_recent = keep_recent
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        # 历史累计剪掉过多少条，用于统计
        self.total_snipped = 0

    # ── 读写 ─────────────────────────────────────────────────────────

    def append(self, message: dict[str, Any]) -> None:
        self.messages.append(message)

    def reset(self) -> None:
        """清空历史，但保留 system 提示词。"""
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def estimate_tokens(self) -> int:
        return estimate_messages_tokens(self.messages)

    def summary_line(self) -> str:
        """一行摘要，给交互界面每轮结束时显示。"""
        used = self.estimate_tokens()
        pct = (used / self.max_tokens * 100) if self.max_tokens else 0
        return f"上下文 约 {used:,} / {self.max_tokens:,} tokens（{pct:.0f}%）· 共 {len(self.messages)} 条消息"

    def describe(self) -> str:
        """多行详情，给 /context 命令显示。"""
        problems = check_invariants(self.messages)
        health = "健康" if not problems else "异常：" + "；".join(problems)
        return (
            f"{self.summary_line()}\n"
            f"历史累计剪枝  {self.total_snipped} 条工具结果\n"
            f"结构自检      {health}"
        )

    # ── 核心：剪枝 ───────────────────────────────────────────────────

    def prepare(self) -> TrimReport:
        """每次调用模型之前调用一次。

        如果超出预算，就地剪枝，并返回一份「刚才动了什么」的报告。
        """
        before = self.estimate_tokens()
        if before <= self.max_tokens:
            return TrimReport(tokens_before=before, tokens_after=before)

        need = before - self.max_tokens

        # ★ 先算一笔账（look-ahead）：光靠「剪工具结果」够不够？
        #
        #   如果不够，说明迟早得丢整轮。那就别先剪了 —— 反正那些轮次
        #   马上会被整体删掉，剪了也是白剪。
        #
        #   （这不是纸上谈兵：最早写成「无条件先剪再丢」时，实测出现过
        #     「剪枝 18 条工具结果」紧接着「丢弃 18 轮旧对话」的场面，
        #     剪掉的恰好就是随后被删掉的那批，白忙一场。）
        if self._max_snip_savings() < need:
            dropped = self._drop_oldest_rounds()
            snipped = self._snip_old_tool_results()  # 丢完再补剪，榨干最后一点
        else:
            snipped = self._snip_old_tool_results()
            dropped = self._drop_oldest_rounds()

        after = self.estimate_tokens()
        self.total_snipped += snipped
        return TrimReport(
            snipped_tool_results=snipped,
            dropped_rounds=dropped,
            tokens_before=before,
            tokens_after=after,
            hit_floor=after > self.max_tokens,
        )

    def _max_snip_savings(self) -> int:
        """算账：把当前所有「可剪的工具结果」全剪光，最多能省多少 token。"""
        boundary = self._snippable_range_end()
        placeholder_tokens = estimate_tokens(SNIP_PLACEHOLDER)
        savings = 0
        for i in range(1, boundary):
            msg = self.messages[i]
            if msg.get("role") != "tool":
                continue
            content = msg.get("content")
            if not isinstance(content, str) or content == SNIP_PLACEHOLDER:
                continue
            delta = estimate_tokens(content) - placeholder_tokens
            if delta > 0:
                savings += delta
        return savings

    def _snippable_range_end(self) -> int:
        """返回「允许被改动」的消息下标上限（不含）。

        两道红线，取更严格的那个：
          1. 最后一条 user 消息之后的全部内容 —— 当前轮，模型马上要用，绝不碰
          2. 最后 keep_recent 条 —— 最近的对话通常最相关
        """
        last_user = 0
        for i, msg in enumerate(self.messages):
            if msg.get("role") == "user":
                last_user = i
        recent_boundary = max(1, len(self.messages) - self.keep_recent)
        return min(last_user, recent_boundary)

    def _snip_old_tool_results(self) -> int:
        """第一级：把可以动的、最早的 role="tool" 消息的 content 换成占位符。

        返回剪掉了几条。**只改 content，不删消息** —— 这样 tool_call_id
        的配对关系天然保持完好。
        """
        snipped = 0
        running = self.estimate_tokens()
        boundary = self._snippable_range_end()
        placeholder_tokens = estimate_tokens(SNIP_PLACEHOLDER)

        # 从最早的一条开始剪（下标 1 起，跳过 system）
        for i in range(1, boundary):
            if running <= self.max_tokens:
                break
            msg = self.messages[i]
            if msg.get("role") != "tool":
                continue
            content = msg.get("content")
            if not isinstance(content, str) or content == SNIP_PLACEHOLDER:
                continue  # 已经是占位符了

            content_tokens = estimate_tokens(content)
            if content_tokens <= placeholder_tokens:
                # 本来就很短，剪了反而更长（占位符也是要占 token 的），不值得
                continue

            msg["content"] = SNIP_PLACEHOLDER
            running -= content_tokens - placeholder_tokens
            snipped += 1

        return snipped

    def _drop_oldest_rounds(self) -> int:
        """第二级：剪完工具结果还超预算，就按「整轮」丢弃最老的对话。

        一轮 = 从一条 user 消息开始，到下一条 user 消息之前为止。

        为什么必须整轮丢？因为 assistant(tool_calls) 和随后的 tool 消息
        是一组，拆散任何一半都会产生孤儿消息，API 会拒绝。
        """
        dropped = 0
        while self.estimate_tokens() > self.max_tokens:
            indices = [i for i, m in enumerate(self.messages) if m.get("role") == "user"]
            # 至少要留下一条 user（也就是当前这个还没回答的问题）
            if len(indices) <= 1:
                break
            start, end = indices[0], indices[1]
            # 不要吃到最近保护区内
            if end > self._snippable_range_end():
                break
            del self.messages[start:end]
            dropped += 1
        return dropped


def check_invariants(messages: list[dict[str, Any]]) -> list[str]:
    """检查 messages 是否满足接口要求的几条硬性规则。

    返回违规描述列表，空列表表示健康。

    ⚠️ 这几条 API 自己也会检查，但报错信息极难懂。与其等它报错，
       不如每次剪枝后自己先验一遍 —— 能省下大量调试时间。
    """
    problems: list[str] = []

    if not messages:
        return ["messages 是空的"]
    if messages[0].get("role") != "system":
        problems.append("第 0 条不是 system 消息")

    # 收集所有 assistant 声明过的 tool_call_id
    known_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") == "assistant":
            for call in msg.get("tool_calls") or []:
                cid = call.get("id")
                if cid:
                    known_ids.add(cid)

    # 每条 tool 消息都必须能配上号
    for i, msg in enumerate(messages):
        if msg.get("role") != "tool":
            continue
        cid = msg.get("tool_call_id")
        if not cid:
            problems.append(f"第 {i} 条 tool 消息缺少 tool_call_id")
        elif cid not in known_ids:
            problems.append(f"第 {i} 条 tool 消息是孤儿（找不到 id={cid} 的 assistant）")

    return problems
