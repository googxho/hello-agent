"""第 1 组：core 三件套（Message / Config / Agent 契约）。

不需要 API Key —— 这一组只验证「数据结构」和「抽象基类规则」。
对应的教学进度：core/message.py · core/config.py · core/agent.py
"""
import pathlib
import sys
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from checks import Check, expect_raise


class _CompleteAgent:
    """给下面的测试当样板用（不继承 Agent，避免污染真正的继承链）。"""


def run(c: Check) -> None:
    # ────────────────────────────────────────────────────────────
    c.section("A. Message —— 统一消息格式（core/message.py）")
    from hello_agents.core.message import Message

    m = Message("你好", "user")
    c.eq(m.role, "user", "位置参数构造 Message('你好','user') 可用（因为该类覆盖了 __init__）")
    c.eq(str(m), "[user] 你好", "__str__ 输出格式为 [role] content")
    c.eq(m.to_dict(), {"role": "user", "content": "你好"}, "to_dict() 只保留 role 与 content")
    c.ok(m.timestamp is not None, "timestamp 被自动填成当前时间（不是 None）")
    c.eq(m.metadata, {}, "metadata 默认为空 dict")

    bad = expect_raise(Exception, Message, "hi", "boss")
    c.ok(
        bad is not None and type(bad).__name__ == "ValidationError",
        "Literal 写在 Pydantic 字段上会【运行时真的校验】：非法 role 抛 ValidationError",
        f"实际得到 {type(bad).__name__ if bad else '没有抛异常'}",
    )

    plain = Message(content="关键字", role="system")
    c.eq(plain.content, "关键字", "关键字写法 Message(content=..., role=...) 也能用")

    # ────────────────────────────────────────────────────────────
    c.section("B. Config —— 配置中心（core/config.py）")
    from hello_agents.core.config import Config

    cfg = Config()
    c.eq(cfg.temperature, 0.7, "默认 temperature = 0.7")
    c.eq(cfg.log_level, "INFO", "默认 log_level = 'INFO'")
    c.ok(cfg.max_tokens is None, "默认 max_tokens = None（表示不设上限）")

    custom = Config(temperature=0.1)
    c.eq(custom.temperature, 0.1, "可以只覆盖其中一个字段：Config(temperature=0.1)")
    c.eq(Config().temperature, 0.7, "改一个实例不影响别的实例（Pydantic 字段是每实例一份）")

    c.ok(
        callable(getattr(Config, "from_env", None)),
        "Config.from_env 作为 classmethod 存在（不需要先造实例就能调用）",
    )
    c.ok(
        isinstance(Config.__dict__["from_env"], classmethod),
        "机制验证：类字典里存的确实是一个 classmethod 对象（描述符）",
        f"实际是 {type(Config.__dict__['from_env']).__name__}",
    )

    d = cfg.to_dict()
    c.ok(isinstance(d, dict) and d["temperature"] == 0.7, "to_dict() 返回普通 dict")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg.to_dict()
    dep = [w for w in caught if "deprecated" in str(w.message).lower()]
    if dep:
        c.known_defect(
            "Config.to_dict() 内部用的是 Pydantic v1 的 self.dict()",
            f"实测会发 {dep[0].category.__name__} 警告，官方建议换 model_dump()",
        )
    else:
        c.skip("to_dict() 的废弃警告", "本次运行没有触发警告（可能 Pydantic 版本已移除该 API）")

    # ────────────────────────────────────────────────────────────
    c.section("C. Agent —— 抽象基类契约（core/agent.py）")
    from hello_agents.core.agent import Agent
    from hello_agents.core.message import Message

    err = expect_raise(TypeError, Agent, "裸的", None)
    c.ok(
        err is not None and not isinstance(err, tuple),
        "抽象类不能直接实例化：Agent('裸的', None) 抛 TypeError",
        f"实际得到 {err!r}",
    )
    c.ok(
        isinstance(err, TypeError) and "abstract" in str(err),
        "错误信息明确点名了未实现的抽象方法",
        f"原文：{err}",
    )

    c.eq(Agent.__abstractmethods__, frozenset({"run"}), "机制验证：Agent 的「账本」是 frozenset({'run'})")

    class HalfAgent(Agent):
        pass

    half = expect_raise(TypeError, HalfAgent, "半成品", None)
    c.ok(
        half is not None and not isinstance(half, tuple),
        "子类漏实现 run 也实例化不了（账本原样继承下来）",
        f"实际得到 {half!r}",
    )
    c.eq(HalfAgent.__abstractmethods__, frozenset({"run"}), "HalfAgent 的账本仍是 {'run'}")

    from fake_llm import FakeLLM

    class MyAgent(Agent):
        def run(self, input_text: str, **kwargs) -> str:
            return f"跑了：{input_text}"

    a = MyAgent("测试员", FakeLLM())
    c.eq(MyAgent.__abstractmethods__, frozenset(), "实现了 run 之后，账本变成空集 → 可以实例化")
    c.eq(a.run("你好"), "跑了：你好", "子类的 run 被正确调用")
    c.ok("测试员" in str(a), "__str__ 里带上了 name（鸭子类型读到了 llm.provider）")

    a.add_message(Message("第一条", "user"))
    a.add_message(Message("第二条", "assistant"))
    c.eq(len(a.get_history()), 2, "add_message 累加历史")
    snapshot = a.get_history()
    snapshot.clear()
    c.eq(len(a.get_history()), 2, "get_history() 返回副本：清空副本不影响对象内部（防御性拷贝）")

    b = MyAgent("另一个", FakeLLM())
    c.eq(len(b.get_history()), 0, "每个实例的历史互相独立（_history 建在 __init__ 里，不是类属性）")
