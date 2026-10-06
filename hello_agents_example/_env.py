"""example 共用的准备代码 —— 只有两件事。

1. 让 `import hello_agents` 找到仓库里的源码（而不是 myenv 里装的旧版）
2. 把仓库根的 .env 读进环境变量 —— 框架用 os.getenv 读配置，它自己不读 .env 文件

用法：每个 example 开头两行
    import _env; _env.setup()
"""
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent      # 仓库根（里面有 hello_agents/ 和 .env）


def setup():
    sys.path.insert(0, str(ROOT))

    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())

    # 本机设了 socks5 代理，httpx 没装 socksio 会直接报错；直连可用，所以清掉
    for key in ("all_proxy", "ALL_PROXY", "no_proxy", "NO_PROXY"):
        value = os.environ.get(key, "").lower()
        if value.startswith("socks") or "[::1]" in value:
            os.environ.pop(key, None)
