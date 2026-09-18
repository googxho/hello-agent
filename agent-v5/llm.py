"""
LLM 调用层 —— 把「发一次请求」封装起来。

⚠️ 为什么要有这一层？
   因为上下文管理必须知道「这次请求到底吃了多少 token」。
   API 会在 response.usage 里给出真实数字 —— 这是我们唯一的权威数据来源
   （自己估算总会有误差）。把它收在这里，上层就不用关心。

v3 给它加了两个口子（都是为了压缩）：

  · max_tokens —— 给摘要输出设上限，否则摘要可能长得跟原文差不多
  · purpose    —— 区分「对话调用」和「压缩调用」，
                  这样你才能回答「压缩到底花了多少钱」这个问题
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from tools import openai_schemas


@dataclass
class LLMReply:
    """一次模型调用的结果。"""

    message: Any        # 原始回复消息（含 content / tool_calls）
    prompt_tokens: int  # 本次请求的真实输入 token 数（来自 API）
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient:
    """对 OpenAI 兼容接口的薄封装。"""

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        # 显式传参，让你一眼看见密钥和地址的去向
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

        # ── 累计用量（跨所有轮次、所有用途）─────────────────────────
        self.calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        # ── 其中「压缩」单独记一笔 ──────────────────────────────────
        # 这是 v3 的新账本。压缩是额外开销，不记清楚就不知道它值不值。
        self.summarize_calls = 0
        self.summarize_prompt_tokens = 0
        self.summarize_completion_tokens = 0

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        use_tools: bool,
        model: str | None = None,
        max_tokens: int | None = None,
        purpose: str = "chat",
    ) -> LLMReply:
        """调用模型一次。

        use_tools=False 时请求里不带 tools 字段，模型就失去了调用工具的能力，
        被迫只能用文字回答 —— Agent 主循环用它做「强制收尾」。

        model / max_tokens 留空就跟随默认。压缩调用用这两个参数走一条
        「更便宜、更短」的支路。
        """
        kwargs: dict[str, Any] = {"model": model or self.model, "messages": messages}
        if use_tools:
            kwargs["tools"] = openai_schemas()
            kwargs["tool_choice"] = "auto"
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        response = self._client.chat.completions.create(**kwargs)
        self.calls += 1

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

        if purpose == "summarize":
            self.summarize_calls += 1
            self.summarize_prompt_tokens += prompt_tokens
            self.summarize_completion_tokens += completion_tokens

        return LLMReply(
            message=response.choices[0].message,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    # ── 给界面用的一句总结 ──────────────────────────────────────────

    def describe(self) -> str:
        chat_calls = self.calls - self.summarize_calls
        parts = [
            f"累计 {chat_calls} 次对话调用 · "
            f"{self.total_prompt_tokens - self.summarize_prompt_tokens:,} 输入 / "
            f"{self.total_completion_tokens - self.summarize_completion_tokens:,} 输出 tokens"
        ]
        if self.summarize_calls:
            parts.append(
                f"压缩 {self.summarize_calls} 次 · "
                f"{self.summarize_prompt_tokens:,} 输入 / "
                f"{self.summarize_completion_tokens:,} 输出 tokens"
            )
        return " · ".join(parts)
