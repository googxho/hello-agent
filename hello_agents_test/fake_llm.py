"""离线测试用的「假模型客户端」。

## 为什么能用假的？

因为 `Agent` 只在**运行时**读 `self.llm` 的几个属性和方法，而 Python 不做类型检查
（鸭子类型）—— 只要对象「长得像」`HelloAgentsLLM`，就能顶替它。
这条在 core/agent.py 里被验证过：`self.llm: HelloAgentsLLM` 只是注解，不参与运行。

## 它怎么工作？

按脚本吐回复：`replies` 是一个列表，每次 `invoke` 取一条；
列表只剩最后一条时会**一直重复**它（这样「最后一轮给最终答案」的循环能自然收敛）。

## 它比真客户端强在哪？

1. 不需要 API Key、不花一分钱、不受网络影响（可以当回归测试反复跑）；
2. **可预测**：真实模型每次回答都不一样，没法断言「第几轮该返回什么」；
3. **可观测**：`calls` 把每次收到的 messages 都记下来，能精确检查
   「工具结果有没有被回灌进对话」这种内部行为。
"""
from typing import Iterator, Optional


class FakeLLM:
    def __init__(
        self,
        replies: Optional[list[str]] = None,
        provider: str = "fake",
        model: str = "fake-model",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ):
        self.replies: list[str] = list(replies or [])
        self.provider = provider          # Agent.__str__ 会读它
        self.model = model                # function_call_agent 会读它
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None               # 真客户端才有；这里故意留 None
        self.calls: list[list[dict]] = []  # 每次 invoke 收到的完整 messages

    # ---------- 被 Agent 调用的三个方法 ----------
    def invoke(self, messages, **kwargs) -> str:
        self.calls.append(list(messages))
        if not self.replies:
            return "(FakeLLM 没有更多预设回复了)"
        # 只剩一条时重复使用；多条时按顺序消费
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]

    def think(self, messages, temperature: Optional[float] = None) -> Iterator[str]:
        """真·逐片流式：每个字符一片，方便观察「流式到底流不流」。"""
        text = self.invoke(messages)
        for ch in text:
            yield ch

    def stream_invoke(self, messages, **kwargs) -> Iterator[str]:
        yield from self.think(messages, kwargs.get("temperature"))

    # ---------- 给测试用的便捷读数 ----------
    @property
    def call_count(self) -> int:
        return len(self.calls)

    def last_messages(self) -> list[dict]:
        return self.calls[-1] if self.calls else []

    def flat_text(self) -> str:
        """把所有轮次收到的消息拍平成一个字符串，方便 grep 关键词。"""
        return "\n".join(m.get("content") or "" for msgs in self.calls for m in msgs)
