"""
上下文管理层 —— v2 引入，v3 把最狠的一刀从「丢弃」换成了「压缩」。

═══════════════════════════════════════════════════════════════════════
v2 的做法（也是 v3 的基础，完整保留）
═══════════════════════════════════════════════════════════════════════

给上下文设一个 token 预算，超了就分两级瘦身：

  第一级：把最早的 role="tool" 消息的 content 换成一句占位符
           —— 只丢内容，不丢结构，伤害最小

  第二级：还超的话，处理最老的整轮对话
           —— 从一条 user 消息开始，到下一条 user 之前为止

两条护栏：
  · 当前轮永不动  最后一条 user 之后的内容一律不碰
  · 源头限流      单个工具返回值超长就地截断（见 agent.py）

还有那个实测逼出来的 look-ahead 算账：如果预判「光剪工具结果不够」，
就直接跳到第二级，免得先剪一遍、随后又整轮删掉，白忙一场。

═══════════════════════════════════════════════════════════════════════
v3 只改了一处：第二级从「丢弃」变成「压缩」
═══════════════════════════════════════════════════════════════════════

    丢弃：del messages[start:end]                    信息量 = 0
    压缩：summarize(start:end) → del → 插入摘要       信息量 > 0

被压掉的内容会以一条 **system 消息**的形式回到上下文里（固定在 index 1，
system 提示词之后）。所以压缩之后，上下文变成了两层：

    messages[0]  system 提示词        ← 永远第一条，不动
    messages[1]  摘要（长期记忆）      ← 压缩产物，很省 token
    messages[2+] 原始对话（短期记忆）  ← 完整精确，但贵

═══════════════════════════════════════════════════════════════════════
三条容易踩的坑，都写在代码里了
═══════════════════════════════════════════════════════════════════════

  1. **替换 content 而不是删消息**（v2 的老经验，剪枝那级继续遵守）
     —— 删 tool 消息会破坏 tool_call_id 配对，API 直接报错。

  2. **必须是完整的轮次才能压** —— 切一半会切出孤儿 tool 消息。
     所以切点永远落在某条 user 消息上。

  3. **压缩失败绝不能连累整次请求** —— 网络抖一下、模型名写错，
     都不能让用户的对话挂掉。失败就回退到 v2 的「丢弃」，游戏继续。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from compaction import Summarizer

# 被剪掉的工具结果会变成这句话。
# 特意写明「已剪枝」而不是留空 —— 让模型知道这里原来有内容，
# 而不是以为工具当初就返回了个空字符串。
#
# ⚠️ 它必须**足够短**：这个字符串会在上下文里出现很多次，占位符越长，
#    剪枝省下的 token 就越少。所以刻意砍到 10 个 token 左右。
SNIP_PLACEHOLDER = "[早期工具结果已剪枝]"

# 摘要消息的头尾标记。
# 有这两行，模型才能分清「哪些是压缩过的旧记忆、哪些是现在的对话」——
# 没有边界的摘要很容易被它当成一条新的用户输入来处理。
SUMMARY_BLOCK_HEADER = "【早前对话摘要 · 由系统自动压缩生成，不是用户的新输入】"
SUMMARY_BLOCK_FOOTER = "【早前摘要结束，以下是本次对话的后续内容】"

# 可压缩内容少于这个 token 数时，就不值得为它调一次 LLM —— 直接丢弃。
#
# 这笔账很好算：压缩一次至少要付出
#     「指令（约 250 tokens）+ 被压缩内容 + 摘要输出（最多 500 tokens）」
# 而收益只是「被压缩内容 − 摘要」。
# 内容本身太短时，收益还抵不上调用成本，那就别压 —— 这时候丢弃才是对的。
#
# ⚠️ 这个值**不是拍脑袋定的**，而且和预算一样会改变 Agent 的行为：
#    实测（预算 400 的小实验）里它曾经卡在 400，结果「永远凑不够要压的量」，
#    于是终端里一直出现的是「✂ 丢弃 1 轮旧对话」——
#    v3 悄悄退回成了 v2。想观察这个现象，把它调大即可。
DEFAULT_MIN_TOKENS_TO_COMPACT = 200

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
    """一次 prepare() 动了什么手脚，用来向用户汇报。"""

    # 第一级：剪枝
    snipped_tool_results: int = 0

    # 第二级 · 路线 A：压缩（v3 的主路）
    compacted_rounds: int = 0
    compacted_messages: int = 0
    compaction_ordinal: int = 0           # 这是本会话的第几次压缩（从 1 数起）
    compaction_input_tokens: int = 0      # 被压掉的那些消息，值多少 token（估算）
    compaction_summary_tokens: int = 0    # 换来的摘要占多少 token（估算）
    compaction_prompt_tokens: int = 0     # 压缩调用真实花掉的输入 token
    compaction_completion_tokens: int = 0 # 压缩调用真实花掉的输出 token
    compaction_elapsed: float = 0.0
    compaction_error: str | None = None   # 压缩失败的原因（失败时会回退到丢弃）

    # ★ 没压缩的话，为什么没压？
    #
    #   这个字段是被一次真实的困惑逼出来的：日志里只有一句「✂ 丢弃 1 轮旧对话」，
    #   看的人根本分不清是「压缩坏了」「压缩被关了」还是「压根没打算压」。
    #   把原因写下来，日志才能自证清白。
    compaction_skipped: str | None = None

    # 第二级 · 路线 B：丢弃（压缩关闭、不值得压、或压缩失败时的回退）
    dropped_rounds: int = 0

    tokens_before: int = 0
    tokens_after: int = 0
    # 已经两级都用尽、却依然超预算 —— 顶到了「最近消息保护区」这块地板。
    # 这不是 bug，而是我们主动选择保护最近上下文的结果。
    hit_floor: bool = False

    @property
    def did_trim(self) -> bool:
        return (
            self.snipped_tool_results > 0
            or self.dropped_rounds > 0
            or self.compacted_rounds > 0
        )

    @property
    def did_compact(self) -> bool:
        return self.compacted_rounds > 0

    @property
    def should_report(self) -> bool:
        """这次 prepare 有没有值得讲给用户听的事。

        注意它不只是 did_trim —— 「超了预算但什么都没动，因为没东西可动」
        同样需要说清楚，否则用户只会看到它悄悄超预算。
        """
        return self.did_trim or self.compaction_skipped is not None

    def describe_lines(self) -> list[str]:
        """把这次动的手脚组织成人话（1~4 行）。

        每一行都在回答一个具体问题：
          · 动了什么？      「🗜 第 N 次压缩 …」/「✂ 丢弃 …」/「✂ 剪枝 …」
          · 为什么没压？    「⏭ 没有走压缩：…」（只在真的没压时出现）
          · 现在多少了？    「上下文 A → B tokens」
          · 花了多少钱？    「压缩调用：…」

        ⚠️ 注意：这里面会出现方括号。调用方用 rich 打印时要用 Text 包一层，
           否则 rich 会把它们当成样式标记吃掉。
        """
        head: list[str] = []
        if self.compacted_rounds:
            pct = (
                self.compaction_summary_tokens / self.compaction_input_tokens * 100
                if self.compaction_input_tokens
                else 0.0
            )
            head.append(
                f"🗜 第 {self.compaction_ordinal} 次压缩："
                f"{self.compacted_rounds} 轮旧对话（{self.compacted_messages} 条消息）"
                f" → 摘要 {self.compaction_summary_tokens:,} tokens"
                f"，压缩比 {pct:.0f}%"
            )
        if self.dropped_rounds:
            head.append(f"✂ 丢弃 {self.dropped_rounds} 轮旧对话（内容不再保留）")
        if self.snipped_tool_results:
            head.append(f"✂ 剪枝 {self.snipped_tool_results} 条工具结果")

        lines = ["  ".join(head) if head else "本次没有瘦身动作"]

        if self.compaction_skipped:
            lines.append(f"⏭ 没有走压缩：{self.compaction_skipped}")

        lines.append(
            f"上下文 {self.tokens_before:,} → {self.tokens_after:,} tokens"
            + ("   ⚠ 已顶到保护区，没东西可动了" if self.hit_floor else "")
        )
        if self.compacted_rounds:
            lines.append(
                f"压缩调用：{self.compaction_prompt_tokens:,} 输入 / "
                f"{self.compaction_completion_tokens:,} 输出 tokens"
                f" · 耗时 {self.compaction_elapsed:.1f}s"
            )
        if self.compaction_error:
            lines.append(f"⚠ 压缩失败：{self.compaction_error}（已回退到「丢弃」）")
        return lines


class ContextManager:
    """管理 messages 列表：超预算时先剪枝，再压缩。"""

    def __init__(
        self,
        system_prompt: str,
        *,
        max_tokens: int,
        keep_recent: int,
        summarizer: Summarizer | None = None,
        min_tokens_to_compact: int = DEFAULT_MIN_TOKENS_TO_COMPACT,
    ) -> None:
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.keep_recent = keep_recent
        # summarizer 为 None 时，行为完全退回 v2 的「丢弃」——
        # 这既是「关掉压缩」的实现，也是压缩失败时的逃生通道。
        self.summarizer = summarizer
        # 少于这么多 token 的可压内容就不压了（见常量处的账）
        self.min_tokens_to_compact = min_tokens_to_compact

        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

        # 长期记忆：压缩出来的摘要文本（None = 还没压过）
        self.summary_text: str | None = None
        # 摘要那条消息对象本身。用它来判断摘要是否已挂在 messages 里 ——
        # 比比较字符串可靠（内容里有任何改动都不会误判）。
        self._summary_message: dict[str, Any] | None = None

        # ── 统计（整个会话累计）─────────────────────────────────────
        self.total_snipped = 0
        self.compactions = 0                  # 成功压缩了几次
        self.compaction_failures = 0          # 失败几次
        self.compacted_input_tokens = 0       # 累计压掉多少 token 的原文
        self.compaction_prompt_tokens = 0     # 压缩调用累计输入 token
        self.compaction_completion_tokens = 0 # 压缩调用累计输出 token

    # ── 读写 ─────────────────────────────────────────────────────────

    def append(self, message: dict[str, Any]) -> None:
        self.messages.append(message)

    def reset(self) -> None:
        """清空历史，但保留 system 提示词。"""
        self.messages = [{"role": "system", "content": self.system_prompt}]
        # 摘要属于历史的一部分，一并清掉
        self.summary_text = None
        self._summary_message = None

    def estimate_tokens(self) -> int:
        return estimate_messages_tokens(self.messages)

    @property
    def has_summary(self) -> bool:
        return self.summary_text is not None

    @property
    def summary_tokens(self) -> int:
        """当前摘要连带头尾标记一共占多少 token。"""
        return estimate_tokens(self._summary_content()) if self.summary_text else 0

    def summary_line(self) -> str:
        """一行摘要，给交互界面每轮结束时显示。"""
        used = self.estimate_tokens()
        pct = (used / self.max_tokens * 100) if self.max_tokens else 0
        memory = f" · 摘要 {self.summary_tokens:,} tokens" if self.has_summary else ""
        return (
            f"上下文 约 {used:,} / {self.max_tokens:,} tokens（{pct:.0f}%）"
            f" · 共 {len(self.messages)} 条消息{memory}"
        )

    def describe(self) -> str:
        """多行详情，给 /context 命令显示。"""
        problems = check_invariants(self.messages)
        health = "健康" if not problems else "异常：" + "；".join(problems)

        if self.has_summary:
            memory_line = (
                f"长期记忆      摘要 {self.summary_tokens:,} tokens"
                f"（占上下文 {self.summary_tokens / max(1, self.estimate_tokens()) * 100:.0f}%）"
            )
        else:
            memory_line = "长期记忆      （还没有 —— 第一次压缩发生后才会出现）"

        lines = [self.summary_line(), memory_line]

        if self.compactions or self.compaction_failures:
            parts = [f"{self.compactions} 次成功"]
            if self.compaction_failures:
                parts.append(f"{self.compaction_failures} 次失败")
            lines.append(f"压缩记录      {' / '.join(parts)}")
            lines.append(
                f"压缩开销      输入 {self.compaction_prompt_tokens:,} / "
                f"输出 {self.compaction_completion_tokens:,} tokens"
                f"（累计压掉原文约 {self.compacted_input_tokens:,} tokens）"
            )
        else:
            lines.append("压缩记录      （还没有压缩过）")

        lines.append(f"历史累计剪枝  {self.total_snipped} 条工具结果")
        lines.append(f"结构自检      {health}")
        return "\n".join(lines)

    # ── 核心：瘦身 ───────────────────────────────────────────────────

    def prepare(self, on_compact: Callable[[], None] | None = None) -> TrimReport:
        """每次调用模型之前调用一次。

        如果超出预算，就地剪枝 / 压缩，并返回一份「刚才动了什么」的报告。

        on_compact: 压缩真正要开跑之前的回调（用来在界面上提示「正在压缩…」）。
                    压缩是一次额外的模型调用，可能要等几秒 ——
                    没有提示的话，用户只会觉得界面卡住了。
        """
        before = self.estimate_tokens()
        if before <= self.max_tokens:
            return TrimReport(tokens_before=before, tokens_after=before)

        need = before - self.max_tokens
        report = TrimReport(tokens_before=before)

        # ★ look-ahead 算账（v2 的老经验，继续用）：
        #   光靠「剪工具结果」够不够？不够的话就别白剪 —— 反正那些轮次
        #   马上就要被整个压掉，先剪一遍纯属白忙。
        if self._max_snip_savings() < need:
            # 走贵的那条路：压缩旧轮次（v2 在这里是「直接丢弃」）
            self._compact_or_drop(report, on_compact)
            report.snipped_tool_results = self._snip_old_tool_results()  # 补剪，榨干最后一点
        else:
            report.snipped_tool_results = self._snip_old_tool_results()
            if self.estimate_tokens() > self.max_tokens:
                self._compact_or_drop(report, on_compact)

        report.tokens_after = self.estimate_tokens()
        report.hit_floor = report.tokens_after > self.max_tokens
        self.total_snipped += report.snipped_tool_results
        return report

    # ── 第二级 · 路线 A：压缩 ────────────────────────────────────────

    def _compact_or_drop(self, report: TrimReport, on_compact: Callable[[], None] | None = None) -> None:
        """把可动的旧轮次压成摘要；压不了就退回 v2 的「丢弃」。

        三条岔路（都在这里处理）：
          1. 没有可压的完整轮次        → 直接返回（全在保护区里）
          2. 内容太少 or 压缩被关闭    → 丢弃，不浪费时间调 LLM
          3. 压缩调用抛异常            → 记下原因，回退到丢弃，绝不上抛

        ⚠️ 每一条岔路都必须往 report 里写清楚**原因**。
           否则日志里只剩一句「✂ 丢弃 1 轮旧对话」，看的人根本分不清
           是压缩坏了、压缩被关了、还是压根没打算压 ——
           这个字段就是被一次真实的困惑逼出来的。
        """
        span = self._compactable_span()
        if span is None:
            # 岔路 1：能动的区间是空的 —— 剩下的内容全在保护区内。
            # 这时连丢弃都无从下手（丢谁都不行），所以什么都不做。
            report.compaction_skipped = (
                f"没有可压缩的完整轮次（可动的只剩保护区：最近 {self.keep_recent} 条 + 当前轮）"
            )
            return

        start, end = span
        chunk = self.messages[start:end]
        chunk_tokens = estimate_messages_tokens(chunk)

        # 岔路 2a：压缩被关掉了。
        if self.summarizer is None:
            report.compaction_skipped = "记忆压缩被关闭了（LLM_SUMMARY_ENABLED=0）"
            report.dropped_rounds += self._drop_oldest_rounds()
            return

        # 岔路 2b：这笔账算不过来，不压。
        #   调一次 LLM 的成本 ≈ 把这段内容再读一遍 + 写出一段摘要；
        #   内容本身还没这个开销大时，丢弃更省。
        #
        #   ⚠️ 但要知道这个选择的代价：丢弃 = 这段内容**永久消失**。
        #      所以这条阈值不该调大，也不该随便动默认值。
        if chunk_tokens < self.min_tokens_to_compact:
            report.compaction_skipped = (
                f"可压内容只有 {chunk_tokens:,} tokens，低于阈值 {self.min_tokens_to_compact:,}"
                f"（调小 LLM_MIN_COMPACT_TOKENS 就会压）"
            )
            report.dropped_rounds += self._drop_oldest_rounds()
            return

        if on_compact is not None:
            on_compact()

        try:
            result = self.summarizer.summarize(chunk, self.summary_text)
        except Exception as exc:
            # 岔路 3：压缩失败不能连累整次请求 —— 这是工程底线。
            #   降级到 v2 的老办法（丢弃），对话照常继续，
            #   只是这一刻它又回到「会失忆」的状态。
            report.compaction_error = f"{type(exc).__name__}: {exc}"
            self.compaction_failures += 1
            report.dropped_rounds += self._drop_oldest_rounds()
            return

        # 成功：原文删掉，摘要顶上
        rounds = sum(1 for m in chunk if m.get("role") == "user")
        del self.messages[start:end]
        self._install_summary(result.text)

        self.compactions += 1  # 先自增，下面的序号才是「这是第几次」
        report.compacted_rounds = rounds
        report.compacted_messages = len(chunk)
        report.compaction_ordinal = self.compactions
        report.compaction_input_tokens = chunk_tokens
        report.compaction_summary_tokens = self.summary_tokens
        report.compaction_prompt_tokens = result.prompt_tokens
        report.compaction_completion_tokens = result.completion_tokens
        report.compaction_elapsed = result.elapsed

        self.compacted_input_tokens += chunk_tokens
        self.compaction_prompt_tokens += result.prompt_tokens
        self.compaction_completion_tokens += result.completion_tokens

    def _compactable_span(self) -> tuple[int, int] | None:
        """找出一段「可以拿去压缩」的连续区间 [start, end)。

        约束有两条，都是为了不切出孤儿消息：

          · 起点必须是某条 user 消息 —— 轮次的分界线永远在 user 上
          · 终点也必须是某条 user 消息（或当前轮的起点）——
            这样切走的一定是整数个**完整的**轮次

        另外还要绕开两条护栏（当前轮、最近 keep_recent 条），
        所以终点取「最后一个仍落在护栏之前」的 user 消息。
        """
        boundary = self._snippable_range_end()
        user_indices = [i for i, m in enumerate(self.messages) if m.get("role") == "user"]
        if not user_indices:
            return None

        start = user_indices[0]
        if start >= boundary:
            return None  # 连第一条 user 都在保护区内，没得压

        # 取最后一个 <= boundary 的 user 下标作为切点
        end = None
        for idx in user_indices:
            if start < idx <= boundary:
                end = idx
        if end is None:
            return None  # 找不到第二轮的起点，那就连一轮都切不出来
        return start, end

    def _install_summary(self, text: str) -> None:
        """把摘要装进上下文：第一次插在 index 1，之后就地更新。"""
        self.summary_text = text
        content = self._summary_content()

        if self._summary_message is not None and any(
            m is self._summary_message for m in self.messages
        ):
            self._summary_message["content"] = content  # 滚动更新（合并后的新摘要）
        else:
            self._summary_message = {"role": "system", "content": content}
            self.messages.insert(1, self._summary_message)

    def _summary_content(self) -> str:
        """拼出摘要消息的完整正文（带头尾标记）。"""
        return f"{SUMMARY_BLOCK_HEADER}\n{self.summary_text}\n{SUMMARY_BLOCK_FOOTER}"

    # ── 第一级：剪工具结果（v2 原样保留）────────────────────────────

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

    # ── 第二级 · 路线 B：丢弃（v2 原样保留，现在是兜底手段）──────────

    def _drop_oldest_rounds(self) -> int:
        """按「整轮」丢弃最老的对话。

        v3 里它退居二线，只在三种情况下出手：
        压缩被关掉 / 内容太少不值得压 / 压缩失败。
        但正因为它还在，v3 永远不会出现「压缩一失败，对话就崩」的场面。

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
