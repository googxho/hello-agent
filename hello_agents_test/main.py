#!/usr/bin/env python3
"""hello_agents_test —— 框架演示 + 回归自检

三种用法：

    python main.py --selftest   # 离线自检：不需要 API Key，逐项打 ✓/✗（推荐先跑这个）
    python main.py --live       # 真实调用：需要仓库根目录的 .env 里有 API Key
    python main.py --config     # 只打印配置来源，不调用模型

设计说明：自检全部离线完成（用 fake_llm.FakeLLM 顶替真客户端），
所以它可以随时复跑、不花一分钱 —— 这正是它能当「回归网」的前提。
"""
import argparse
import os
import pathlib
import sys
import time

DEMO_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent          # 里面放着 hello_agents/ 和 .env
ENV_FILE = REPO_ROOT / ".env"

sys.path.insert(0, str(REPO_ROOT))   # 让 import hello_agents 命中本地源码
sys.path.insert(0, str(DEMO_DIR))    # 让 import tests_core / fake_llm 生效


# ══════════════════════════════════════════════════════════════
# 环境变量加载
# ══════════════════════════════════════════════════════════════
def sanitize_proxy_env() -> list[str]:
    """清掉本机那两个会把 httpx 弄崩的代理环境变量，返回「删了哪几个」。

    实测（本机环境）：
      1) all_proxy=socks5://127.0.0.1:7890
         → httpx 会去找 socksio 包，没装就抛 ImportError: Using SOCKS proxy, ...
      2) no_proxy / NO_PROXY 里含 [::1]
         → httpx 解析时抛 InvalidURL: Invalid port: ':1]'

    为什么敢删：实测直连 https://api.deepseek.com 可用（HTTP 200，约 2.3 秒）。
    ⚠ 如果你的网络【必须】走代理才能出网，请不要用这个函数，改成安装 socks 支持：
         pip install "httpx[socks]"
    """
    removed = []
    for key in ("all_proxy", "ALL_PROXY"):
        if os.environ.get(key, "").lower().startswith("socks"):
            os.environ.pop(key, None)
            removed.append(key)
    for key in ("no_proxy", "NO_PROXY"):
        if "[::1]" in os.environ.get(key, ""):
            os.environ.pop(key, None)
            removed.append(key)
    return removed


def load_env() -> list[str]:
    """把仓库根的 .env 灌进进程环境，返回「本次真正设置了哪些键」。

    为什么需要这一步：`core/llm.py` 用的是 os.getenv，读的是【进程环境变量】，
    而不是 .env 文件本身 —— 全项目只有 core/database_config.py 调了 load_dotenv()。
    所以直接 import 就构造 HelloAgentsLLM，会读不到任何配置。
    """
    if not ENV_FILE.exists():
        return []
    applied = []
    for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:      # 真实环境变量优先，不覆盖
            os.environ[key] = value.strip()
            applied.append(key)
    return applied


# ══════════════════════════════════════════════════════════════
# --selftest
# ══════════════════════════════════════════════════════════════
def cmd_selftest() -> int:
    print("=" * 68)
    print("hello_agents 离线自检（不需要 API Key，可反复跑）")
    print("=" * 68)
    print("被测源码:", REPO_ROOT / "hello_agents")

    from checks import Check
    import tests_core
    import tests_tools
    import tests_agents

    c = Check("hello_agents_test")
    for title, module in (("第 1 组 · core 三件套", tests_core),
                          ("第 2 组 · 工具系统", tests_tools),
                          ("第 3 组 · 六个 Agent 范式", tests_agents)):
        print(f"\n\n{'#' * 68}\n# {title}\n{'#' * 68}")
        module.run(c)

    return c.summary()


# ══════════════════════════════════════════════════════════════
# --config
# ══════════════════════════════════════════════════════════════
def cmd_config() -> int:
    print("=" * 68)
    print("配置来源检查")
    print("=" * 68)
    print(f"  demo 目录      : {DEMO_DIR}")
    print(f"  仓库根目录     : {REPO_ROOT}")
    print(f"  源码包位置     : {REPO_ROOT / 'hello_agents'}")
    print(f"  .env 文件      : {ENV_FILE}  ({'存在' if ENV_FILE.exists() else '不存在'})")

    applied = load_env()
    print(f"  从 .env 载入   : {applied if applied else '（无，或已由真实环境变量提供）'}")

    key = os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    base = os.environ.get("LLM_BASE_URL")
    model = os.environ.get("LLM_MODEL_ID") or os.environ.get("LLM_MODEL")
    print(f"  LLM_BASE_URL   : {base or '（未设置）'}")
    print(f"  LLM_API_KEY    : {'已设置（' + key[:6] + '****）' if key else '（未设置）'}")
    print(f"  模型名（.env）  : {model or '（未设置）'}")

    # 关键对比：.env 里写的模型名 vs 框架【实际会用的】模型名
    try:
        from hello_agents import HelloAgentsLLM
        eff = HelloAgentsLLM()
        print(f"  实际生效的模型 : {eff.model}   （provider={eff.provider}）")
        if model and eff.model != model:
            print()
            print(f"  ⚠⚠ .env 里配置的是 {model!r}，但框架实际用的是 {eff.model!r} —— 配置没生效！")
            print("      原因：core/llm.py 读的是环境变量 LLM_MODEL_ID，而 .env 里写的是 LLM_MODEL。")
            print("      于是 self.model 一路是 None，最后靠 _get_default_model() 从 base_url 猜出来的。")
            print("      修法（二选一）：把 .env 的键改名成 LLM_MODEL_ID，或在构造时显式传 model=...")
    except Exception as e:
        print(f"  实际生效的模型 : （无法构造客户端：{type(e).__name__}: {e}）")

    return 0


# ══════════════════════════════════════════════════════════════
# --live
# ══════════════════════════════════════════════════════════════
def cmd_live() -> int:
    load_env()
    from hello_agents import HelloAgentsLLM, SimpleAgent, ToolRegistry
    from hello_agents.tools.base import Tool, ToolParameter, tool_action

    print("=" * 68)
    print("真实调用演示（会消耗极少量 API 额度）")
    print("=" * 68)

    llm = HelloAgentsLLM()
    print(f"  provider = {llm.provider}   model = {llm.model}")
    print()

    # ── 1. 最小闭环：纯对话 ────────────────────────────────
    print("─" * 68)
    print("【1】最小闭环：只给一个模型，不给工具")
    print("─" * 68)
    agent = SimpleAgent(name="闲聊助手", llm=llm, system_prompt="用一句话回答，不要展开。")
    t0 = time.time()
    answer = agent.run("用一句话解释什么是 Agent。")
    print(f"\n  ← 回答：{answer}")
    print(f"  ← 耗时 {time.time() - t0:.2f}s，历史里有 {len(agent.get_history())} 条消息")
    print()

    # ── 2. 工具调用：走 @tool_action 正道 ──────────────────
    print("─" * 68)
    print("【2】工具调用：自定义工具 + 让模型自己决定什么时候用")
    print("─" * 68)

    class TextTool(Tool):
        def __init__(self):
            super().__init__(name="text", description="文本工具箱", expandable=True)

        def run(self, parameters):
            return "请调用展开后的子工具"

        def get_parameters(self):
            return []

        @tool_action("text_char_count", "统计一段文本的字符数（含空格和标点）")
        def _char_count(self, input: str) -> str:
            # ⚠ 参数名必须叫 input：注册表固定传 {"input": ...}
            print(f"      🛠  工具被调用，收到 {input!r}")
            return f"{len(input)} 个字符"

    registry = ToolRegistry()
    registry.register_tool(TextTool(), auto_expand=True)
    print(f"  展开后的工具：{registry.list_tools()}")
    print(f"  模型看到的说明：{registry.get_tools_description()}")
    print("  ⚠ 注意上面这行：它【只】有「工具名: 描述」，不含参数名。")
    print("     而 SimpleAgent 的提示词又要求「参数名必须完全匹配」—— 模型只能猜。")
    print("     所以这里在 system_prompt 里把调用格式钉死，绕开这个坑。")

    tool_agent = SimpleAgent(
        name="统计员",
        llm=llm,
        tool_registry=registry,
        system_prompt=(
            "你是一个严谨的助手。需要使用工具时，严格按下面格式输出：\n"
            "[TOOL_CALL:工具名:参数]\n"
            "参数直接写裸文本，【不要】写成 key=value 形式。"
        ),
    )
    question = "请用 text_char_count 工具统计这句话有多少字符：HelloAgents 是一个 Agent 框架"
    print(f"\n  问：{question}")
    t0 = time.time()
    tool_answer = tool_agent.run(question)
    print(f"\n  ← 回答：{tool_answer}")
    print(f"  ← 耗时 {time.time() - t0:.2f}s")
    print()

    # ── 3. 流式：观察「第一片等多久」 ──────────────────────
    print("─" * 68)
    print("【3】流式调用：观察首片延迟 vs 总耗时（理解阻塞 I/O）")
    print("─" * 68)
    t0 = time.time()
    first_at = None
    chunks = 0
    for _piece in llm.think([{"role": "user", "content": "从 1 数到 10，用中文顿号分隔，不要别的字"}]):
        chunks += 1
        if first_at is None:
            first_at = time.time() - t0
    total = time.time() - t0
    print(f"\n  ← 收到 {chunks} 片；首片等了 {first_at:.2f}s，总耗时 {total:.2f}s")
    print(f"  ← 首片延迟就是模型「思考」的时间 —— 这段时间你的线程阻塞在 socket.recv 上，CPU 几乎不耗")

    return 0


# ══════════════════════════════════════════════════════════════
def main() -> int:
    parser = argparse.ArgumentParser(
        description="hello_agents 演示与自检",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--selftest", action="store_true", help="离线自检（不需要 API Key）")
    group.add_argument("--live", action="store_true", help="真实调用演示（需要 .env）")
    group.add_argument("--config", action="store_true", help="只打印配置来源")
    args = parser.parse_args()

    # 统一的网络环境预处理（只在真的删了东西时才说话 —— 沉默会让人以为一切正常）
    removed = sanitize_proxy_env()
    if removed:
        print(f"[环境] 已临时清理会干扰 httpx 的代理变量：{removed}")
        print("[环境] 原因见 main.py 里 sanitize_proxy_env() 的注释；直连实测可用。\n")

    if args.selftest:
        return cmd_selftest()
    if args.config:
        return cmd_config()
    return cmd_live()


if __name__ == "__main__":
    raise SystemExit(main())
