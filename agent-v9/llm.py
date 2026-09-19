"""
LLM 调用层 —— 把「发一次请求」封装起来。

这一层比 v1 厚一点点，多了两件评估需要的东西：

  · **账本**：这次运行一共调了几次模型、烧了多少 token
    —— 「没做可观测」的第一条就是「花了多少钱不知道」
  · **参数可覆盖**：评估时要用自己的温度（默认 0）、
    裁判调用要用自己的 max_tokens，又不能污染全局配置

⚠️ 为什么要有 Protocol（LLMClientLike）？
   和 v3 的 Summarizer、v6 的 Embedder 是同一个套路：
   真实客户端要联网、要花钱、行为还飘；
   测试里塞一个「剧本替身」，整条评估流水线就能离线测穿。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from openai import OpenAI

from config import EVAL_TEMPERATURE, TEMPERATURE
from tools import openai_schemas


@dataclass
class LLMReply:
    """一次模型调用的结果。"""

    message: Any        # 原始回复消息（含 content / tool_calls）
    prompt_tokens: int  # 本次请求的真实输入 token 数（来自 API）
    completion_tokens: int
    finish_reason: str | None = None  # "stop" = 正常说完；"length" = 被 max_tokens 截断

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClientLike(Protocol):
    """测试替身要满足的最小接口（鸭子类型）。"""

    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> LLMReply: ...


class LLMClient:
    """对 OpenAI 兼容接口的薄封装 + 一本账。"""

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        # 显式传参，让你一眼看见密钥和地址的去向
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

        # ── 账本（跨所有调用累计）────────────────────────────────
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        use_tools: bool = True,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        purpose: str = "chat",
    ) -> LLMReply:
        """调用模型一次。

        purpose 只用于记账口吻（"chat" / "judge"），不传也无妨 ——
        评估报告里说「其中 X 次是裁判」就靠它。
        """
        kwargs: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            # 温度永远显式传：不传就吃 API 默认值（可能 1.0），
            # 模型会在「调不调工具」这种决策上左右横跳，评估没法复现。
            "temperature": TEMPERATURE if temperature is None else temperature,
        }
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
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

        choice = response.choices[0]
        return LLMReply(
            message=choice.message,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=getattr(choice, "finish_reason", None),
        )

    # ── 给界面用的一句总结 ──────────────────────────────────────────

    def snapshot(self) -> tuple[int, int, int]:
        """账本的即时快照（次数, 输入 tokens, 输出 tokens）。

        「一次评估花了多少」= 跑前快照和跑后快照的差 —— 比拆东墙补西墙的
        分项统计可靠（也顺便告诉学习者：账本要用差分读）。
        """
        return self.calls, self.prompt_tokens, self.completion_tokens

    def describe(self) -> str:
        return (
            f"累计 {self.calls} 次调用 · "
            f"{self.prompt_tokens:,} 输入 / {self.completion_tokens:,} 输出 tokens"
        )


def default_eval_temperature() -> float:
    """评估默认温度的单一来源（main / 测试都从这里取）。"""
    return EVAL_TEMPERATURE
