"""
LLM 调用层 —— 把「发一次请求」封装起来。

⚠️ 为什么要有这一层？
   因为 v2 要管上下文，就必须知道「这次请求到底吃了多少 token」。
   API 会在 response.usage 里给出真实数字 —— 这是我们唯一的权威数据来源
   （自己估算总会有误差）。把它收在这里，上层就不用关心。
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

        # 累计用量（跨所有轮次）
        self.calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def complete(self, messages: list[dict[str, Any]], *, use_tools: bool) -> LLMReply:
        """调用模型一次。

        use_tools=False 时请求里不带 tools 字段，模型就失去了调用工具的能力，
        被迫只能用文字回答 —— Agent 主循环用它做「强制收尾」。
        """
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if use_tools:
            kwargs["tools"] = openai_schemas()
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        self.calls += 1

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

        return LLMReply(
            message=response.choices[0].message,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
