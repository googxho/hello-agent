# ===== 文件导读 =====
# [职责] HelloAgents 的统一 LLM 客户端：把各家「兼容 OpenAI 接口」的服务，收敛成 think() / invoke() 两个入口。
# [位置] 被 core/agent.py 和各 agents/*.py 持有（每个 Agent 内部都存着一个 HelloAgentsLLM）；本文件只依赖 .exceptions。
# [阅读顺序] ① 模块常量 SUPPORTED_PROVIDERS → ② __init__（配置优先级全在这里）
#            → ③ 三个查找表 _auto_detect_provider / _resolve_credentials / _get_default_model（可快速跳读）
#            → ④ 文件末尾 think / invoke / stream_invoke（真正的调用入口，重点看生成器）。
# ===================
"""HelloAgents统一LLM接口 - 基于OpenAI原生API"""

# [语法] 三类导入按来源分开：os 是标准库；OpenAI 是第三方包；下面的 .exceptions 是包内相对导入。
import os
from typing import Literal, Optional, Iterator
from openai import OpenAI

# [语法] 相对导入：开头的那个点号表示「当前所在这个包」（hello_agents.core），
#        所以它指向 hello_agents/core/exceptions.py，而不是 site-packages 里可能同名的包。
from .exceptions import HelloAgentsException

# 支持的LLM提供商
# [语法] 类型别名：把 Literal[...] 这个类型赋给一个名字，方便下面给 provider 参数复用。
# [概念] Literal["openai", ...] 表示「只能取列出的这几个字符串之一」；
#        但它只在 mypy / pyright 这类类型检查器里生效 —— 运行时既不校验也不报错，
#        写成 "openaii" 照样跑得过去，所以别拿它当参数校验用。
SUPPORTED_PROVIDERS = Literal[
    "openai",
    "deepseek",
    "qwen",
    "modelscope",
    "kimi",
    "zhipu",
    "ollama",
    "vllm",
    "local",
    "auto",
    "custom",
]

# [作用] LLM 客户端：负责「读配置 → 建底层 OpenAI 客户端 → 提供流式 / 非流式调用」。
# [易错] 它不是抽象基类，可以直接实例化；但构造时就要求 api_key 和 base_url 同时齐全，
#        否则 __init__ 末尾那句 all(...) 检查会直接抛 HelloAgentsException。
class HelloAgentsLLM:
    """
    为HelloAgents定制的LLM客户端。
    它用于调用任何兼容OpenAI接口的服务，并默认使用流式响应。

    设计理念：
    - 参数优先，环境变量兜底
    - 流式响应为默认，提供更好的用户体验
    - 支持多种LLM提供商
    - 统一的调用接口
    """

    # [作用] 解析配置并建好底层客户端；优先级是「传入参数 > 环境变量」，没给的值才去环境里找。
    # [参数] model / api_key / base_url / provider 都允许传 None（等于「去环境变量找」）；
    #        **kwargs 收集所有没在上面声明过的关键字参数，在函数体里就是一个 dict（见下面 self.kwargs）。
    # [易错] **kwargs 必须放在参数列表最后；它的作用是「以后加参数不用改签名」，不是本文件在用。
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[SUPPORTED_PROVIDERS] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs
    ):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        支持自动检测provider或使用统一的LLM_*环境变量配置。

        Args:
            model: 模型名称，如果未提供则从环境变量LLM_MODEL_ID读取
            api_key: API密钥，如果未提供则从环境变量读取
            base_url: 服务地址，如果未提供则从环境变量LLM_BASE_URL读取
            provider: LLM提供商，如果未提供则自动检测
            temperature: 温度参数
            max_tokens: 最大token数
            timeout: 超时时间，从环境变量LLM_TIMEOUT读取，默认60秒
        """
        # 优先使用传入参数，如果未提供，则从环境变量加载
        # [语法] or 短路求值：左边为真就返回左边，否则返回右边 —— 这是 Python 里最常见的「默认值」写法。
        # [易错] or 判的是真 / 假，不是「是否为 None」：model="" 这种空字符串也会被当成没传，从而被环境变量顶掉。
        self.model = model or os.getenv("LLM_MODEL_ID")
        self.temperature = temperature
        self.max_tokens = max_tokens
        # [易错] os.getenv 读到的永远是字符串或 None，所以 "60" 必须 int() 转一下；
        #        第二个参数是「环境变量不存在时的默认值」——查不到就用 60 秒。
        self.timeout = timeout or int(os.getenv("LLM_TIMEOUT", "60"))
        # [作用] 把多余的关键字参数原样存下来。注意：本文件里从头到尾没读过它，
        #        这是留给子类或后续扩展的口子（不是 bug，但也别指望它现在起作用）。
        self.kwargs = kwargs

        # 自动检测provider或使用指定的provider
        # [语法] 条件表达式（三元写法）：A if 条件 else B；这里等于「传了 provider 就转小写，没传就是 None」。
        requested_provider = (provider or "").lower() if provider else None
        # [概念] 名字以下划线开头（_auto_detect_provider）是「内部方法」约定：语言不拦你调用，
        #        但它等于告诉使用者「这是实现细节，别依赖」。类外面才会用双下划线开头。
        self.provider = provider or self._auto_detect_provider(api_key, base_url)

        if requested_provider == "custom":
            self.provider = "custom"
            self.api_key = api_key or os.getenv("LLM_API_KEY")
            self.base_url = base_url or os.getenv("LLM_BASE_URL")
        else:
            # 根据provider确定API密钥和base_url
            self.api_key, self.base_url = self._resolve_credentials(api_key, base_url)

        # 验证必要参数
        if not self.model:
            self.model = self._get_default_model()
        # [易错] all([...]) 要求列表里每一项都为真：api_key 和 base_url 只要有一个是 None 或 ""，这里就抛异常。
        #        这叫「快速失败」——在本地立刻报「配置没配好」，比发请求后收到 401 更容易定位。
        if not all([self.api_key, self.base_url]):
            raise HelloAgentsException("API密钥和服务地址必须被提供或在.env文件中定义。")

        # 创建OpenAI客户端
        self._client = self._create_client()

    # [作用] 没显式指定 provider 时，猜用户想用哪家：先看各家的专属环境变量，再看 api_key 的格式，最后看 base_url。
    # [参数] api_key / base_url 是用户刚传进来的原始值，可能为 None（None 时会再去环境变量里找一遍）。
    # [返回] 提供商标识字符串；三条线索都没命中时返回 "auto"。
    def _auto_detect_provider(self, api_key: Optional[str], base_url: Optional[str]) -> str:
        """
        自动检测LLM提供商

        检测逻辑：
        1. 优先检查特定提供商的环境变量
        2. 根据API密钥格式判断
        3. 根据base_url判断
        4. 默认返回通用配置
        """
        # 1. 检查特定提供商的环境变量
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("DEEPSEEK_API_KEY"):
            return "deepseek"
        if os.getenv("DASHSCOPE_API_KEY"):
            return "qwen"
        if os.getenv("MODELSCOPE_API_KEY"):
            return "modelscope"
        if os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY"):
            return "kimi"
        if os.getenv("ZHIPU_API_KEY") or os.getenv("GLM_API_KEY"):
            return "zhipu"
        if os.getenv("OLLAMA_API_KEY") or os.getenv("OLLAMA_HOST"):
            return "ollama"
        if os.getenv("VLLM_API_KEY") or os.getenv("VLLM_HOST"):
            return "vllm"

        # 2. 根据API密钥格式判断
        actual_api_key = api_key or os.getenv("LLM_API_KEY")
        if actual_api_key:
            actual_key_lower = actual_api_key.lower()
            if actual_api_key.startswith("ms-"):
                return "modelscope"
            elif actual_key_lower == "ollama":
                return "ollama"
            elif actual_key_lower == "vllm":
                return "vllm"
            elif actual_key_lower == "local":
                return "local"
            elif actual_api_key.startswith("sk-") and len(actual_api_key) > 50:
                # 可能是OpenAI、DeepSeek或Kimi，需要进一步判断
                pass
            elif actual_api_key.endswith(".") or "." in actual_api_key[-20:]:
                # 智谱AI的API密钥格式通常包含点号
                return "zhipu"

        # 3. 根据base_url判断
        actual_base_url = base_url or os.getenv("LLM_BASE_URL")
        if actual_base_url:
            base_url_lower = actual_base_url.lower()
            if "api.openai.com" in base_url_lower:
                return "openai"
            elif "api.deepseek.com" in base_url_lower:
                return "deepseek"
            elif "dashscope.aliyuncs.com" in base_url_lower:
                return "qwen"
            elif "api-inference.modelscope.cn" in base_url_lower:
                return "modelscope"
            elif "api.moonshot.cn" in base_url_lower:
                return "kimi"
            elif "open.bigmodel.cn" in base_url_lower:
                return "zhipu"
            elif "localhost" in base_url_lower or "127.0.0.1" in base_url_lower:
                # 本地部署检测 - 优先检查特定服务
                if ":11434" in base_url_lower or "ollama" in base_url_lower:
                    return "ollama"
                elif ":8000" in base_url_lower and "vllm" in base_url_lower:
                    return "vllm"
                elif ":8080" in base_url_lower or ":7860" in base_url_lower:
                    return "local"
                else:
                    # 根据API密钥进一步判断
                    if actual_api_key and actual_api_key.lower() == "ollama":
                        return "ollama"
                    elif actual_api_key and actual_api_key.lower() == "vllm":
                        return "vllm"
                    else:
                        return "local"
            elif any(port in base_url_lower for port in [":8080", ":7860", ":5000"]):
                # 常见的本地部署端口
                return "local"

        # 4. 默认返回auto，使用通用配置
        return "auto"

    # [作用] 按 provider 决定「去哪个环境变量拿 key、连哪个地址」，本质是一张大的 if / elif 查找表。
    # [返回] 元组 (api_key, base_url)；__init__ 里写成 `a, b = self._resolve_credentials(...)`，
    #        一次把元组拆成两个变量，这叫元组解包（unpacking）。
    def _resolve_credentials(self, api_key: Optional[str], base_url: Optional[str]) -> tuple[str, str]:
        """根据provider解析API密钥和base_url"""
        if self.provider == "openai":
            resolved_api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1"
            return resolved_api_key, resolved_base_url

        elif self.provider == "deepseek":
            resolved_api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api.deepseek.com"
            return resolved_api_key, resolved_base_url

        elif self.provider == "qwen":
            resolved_api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            return resolved_api_key, resolved_base_url

        elif self.provider == "modelscope":
            resolved_api_key = api_key or os.getenv("MODELSCOPE_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api-inference.modelscope.cn/v1/"
            return resolved_api_key, resolved_base_url

        elif self.provider == "kimi":
            resolved_api_key = api_key or os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api.moonshot.cn/v1"
            return resolved_api_key, resolved_base_url

        elif self.provider == "zhipu":
            resolved_api_key = api_key or os.getenv("ZHIPU_API_KEY") or os.getenv("GLM_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://open.bigmodel.cn/api/paas/v4"
            return resolved_api_key, resolved_base_url

        elif self.provider == "ollama":
            resolved_api_key = api_key or os.getenv("OLLAMA_API_KEY") or os.getenv("LLM_API_KEY") or "ollama"
            resolved_base_url = base_url or os.getenv("OLLAMA_HOST") or os.getenv("LLM_BASE_URL") or "http://localhost:11434/v1"
            return resolved_api_key, resolved_base_url

        elif self.provider == "vllm":
            resolved_api_key = api_key or os.getenv("VLLM_API_KEY") or os.getenv("LLM_API_KEY") or "vllm"
            resolved_base_url = base_url or os.getenv("VLLM_HOST") or os.getenv("LLM_BASE_URL") or "http://localhost:8000/v1"
            return resolved_api_key, resolved_base_url

        elif self.provider == "local":
            resolved_api_key = api_key or os.getenv("LLM_API_KEY") or "local"
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "http://localhost:8000/v1"
            return resolved_api_key, resolved_base_url

        elif self.provider == "custom":
            resolved_api_key = api_key or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL")
            return resolved_api_key, resolved_base_url

        else:
            # auto或其他情况：使用通用配置，支持任何OpenAI兼容的服务
            resolved_api_key = api_key or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL")
            return resolved_api_key, resolved_base_url

    # [作用] 真正 new 出一个 OpenAI 客户端实例。
    # [语法] 返回类型标注可以直接写类名 OpenAI —— 类型注解里写类名，意思就是「返回这个类的实例」。
    def _create_client(self) -> OpenAI:
        """创建OpenAI客户端"""
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )
    
    # [作用] 用户和配置都没给出模型名时的兜底默认值，仍是一张查表；
    #        auto 分支还会反过来从 base_url 里的关键词猜模型，属于「猜中即用」的尽力而为。
    def _get_default_model(self) -> str:
        """获取默认模型"""
        if self.provider == "openai":
            return "gpt-3.5-turbo"
        elif self.provider == "deepseek":
            return "deepseek-chat"
        elif self.provider == "qwen":
            return "qwen-plus"
        elif self.provider == "modelscope":
            return "Qwen/Qwen2.5-72B-Instruct"
        elif self.provider == "kimi":
            return "moonshot-v1-8k"
        elif self.provider == "zhipu":
            return "glm-4"
        elif self.provider == "ollama":
            return "llama3.2"  # Ollama常用模型
        elif self.provider == "vllm":
            return "meta-llama/Llama-2-7b-chat-hf"  # vLLM常用模型
        elif self.provider == "local":
            return "local-model"  # 本地模型占位符
        elif self.provider == "custom":
            return self.model or "gpt-3.5-turbo"
        else:
            # auto或其他情况：根据base_url智能推断默认模型
            base_url = os.getenv("LLM_BASE_URL", "")
            base_url_lower = base_url.lower()
            if "modelscope" in base_url_lower:
                return "Qwen/Qwen2.5-72B-Instruct"
            elif "deepseek" in base_url_lower:
                return "deepseek-chat"
            elif "dashscope" in base_url_lower:
                return "qwen-plus"
            elif "moonshot" in base_url_lower:
                return "moonshot-v1-8k"
            elif "bigmodel" in base_url_lower:
                return "glm-4"
            elif "ollama" in base_url_lower or ":11434" in base_url_lower:
                return "llama3.2"
            elif ":8000" in base_url_lower or "vllm" in base_url_lower:
                return "meta-llama/Llama-2-7b-chat-hf"
            elif "localhost" in base_url_lower or "127.0.0.1" in base_url_lower:
                return "local-model"
            else:
                return "gpt-3.5-turbo"

    # [作用] 流式调用模型：一边收、一边打印、一边把片段交出去，是 Agent 内部真正走的入口。
    # [参数] messages 是 OpenAI 格式的对话列表；temperature 传 None 表示沿用构造时的默认温度。
    # [返回] 标注为 Iterator[str]，但准确说它是个「生成器函数」：
    #        调用 think() 不会执行函数体，只返回一个生成器对象；等 for 循环或 next() 时才一段段往下跑。
    # [副作用] 它会直接 print 到终端（模型名、响应正文、错误），不是纯计算。
    # [易错] 判断一个函数是不是生成器，只看函数体里有没有 yield —— 一旦有，
    #        连里面的 print 和 API 请求都会被推迟到「开始迭代」时才发生。
    # [实测] g = llm.think(msgs) 这一行耗时 0.0000 秒，终端一片安静，网络请求根本没发；
    #        直到第 1 次 next(g) / 进入 for，才打印「🧠 正在调用…」并发起请求。
    # [易错] 由此推出一个坑：想用 try/except 包住调用错误，必须包「迭代」那一侧；
    #        `try: g = llm.think(...)` 这种写法什么都拦不住，异常会在你 for 的时候才炸出来。
    def think(self, messages: list[dict[str, str]], temperature: Optional[float] = None) -> Iterator[str]:
        """
        调用大语言模型进行思考，并返回流式响应。
        这是主要的调用方法，默认使用流式响应以获得更好的用户体验。

        Args:
            messages: 消息列表
            temperature: 温度参数，如果未提供则使用初始化时的值

        Yields:
            str: 流式响应的文本片段
        """
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            # [作用] 发起真正的 HTTP 请求：stream=True 表示「服务端一条条推给我」，所以下面才能 for 着拿片段。
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )

            # 处理流式响应
            print("✅ 大语言模型响应成功:")
            # [概念] 惰性求值：response 不是一次性拿完的完整结果，每转一圈才向服务端要下一片。
            # [概念] 底层协议是 SSE（Server-Sent Events）：一个「不关闭的 HTTP 响应」，
            #        服务端反复往下写 `data: {一行 JSON}\n\n`，客户端边收边解析。
            # [实测] 服务端推来的原始行长这样（content-type: text/event-stream）：
            #        data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}],...}
            #        data: {"choices":[{"index":0,"delta":{"content":"1"},"finish_reason":null}],...}
            #        data: {"choices":[{"index":0,"delta":{"content":""},"finish_reason":"stop"}],...,"usage":{...}}
            #        data: [DONE]
            # [概念] 切片粒度由服务端决定（实测每片 1~3 个字符，不等于一个字也不等于一个词）；
            #        末尾的 [DONE] 是纯文本哨兵、不是 JSON，SDK 见到它就停掉迭代器，不会当成 chunk 交给你。
            # [概念] 别看它叫「流式」就以为是回调 / 事件驱动 —— 这段代码里没有任何「数据到了就触发我」的逻辑。
            #        真相是同步阻塞：for 每转一圈都主动去「要」下一片（pull，不是 push）；
            #        要不到就把当前线程挂起在内核的 socket.recv() 上，直到数据到达由操作系统唤醒它。
            # [实测] 一次 21 片的回复：墙上耗时 658ms，其中进程 CPU 只有 12.1ms（1.8%）；
            #        每圈循环体约 0.2~0.3ms，剩下的时间全是在 recv() 里睡着。CPU 占比这么低，
            #        正是「它在等，不是在转」的量化证据（若真在空转轮询，CPU 会接近墙上时间）。
            # [语法] `for 变量 in 对象:` 的三步规则（for 语句本身的语言规定）：
            #        ① 先执行 iter(对象) 拿到「迭代器」；② 反复调 next() 把返回值绑定给变量；
            #        ③ next() 抛 StopIteration 时循环结束 —— 这个异常被 for 自动吞掉，不会冒出来。
            #        它只认 __iter__ / __next__ 两个魔术方法，不认对象是什么类型：
            #        response 的类型是 openai.Stream，既不是 list 也不是生成器，照样能 for，靠的就是这套协议。
            # [语法] 同一行里的 `chunk.choices[0].delta.content` 是三级属性访问；
            #        choices 是列表，[0] 是下标取值（因为一次可以要多个候选答案，这里固定取第 0 个）。
            # [易错] 迭代器是一次性的：同一个生成器对象第二次 for 会一片都拿不到
            #        （不报错，就是空的 —— 因为它已经停在对末尾了，不是没数据）。
            for chunk in response:
                # [易错] 一个 chunk 的 delta.content 有三种形态（实测）：
                #        ① 首个 chunk 只有 role（content 是空串）——它不是正文，只是「开始回话了」的信号；
                #        ② 中间的 chunk 才带正文，1~3 个字符；③ 末个 chunk content 为空，finish_reason 变成 "stop" 或 "length"。
                #        所以这里必须 `or ""` 兜底：deepseek 给的是空串 ""，有些兼容实现直接给 None，两种都被它挡下。
                #        紧跟着的 `if content:` 再把空串滤掉，保证 yield 出去的每一片都真有内容。
                content = chunk.choices[0].delta.content or ""
                if content:
                    print(content, end="", flush=True)
                    # [语法] yield：把 content 交出去，然后「暂停」在这一行 —— 局部变量和运行进度全都保留，
                    #        下次迭代从暂停处继续。这就是它和 return 的根本区别：return 只能返回一次，yield 可以返回很多次。
                    yield content
            print()  # 在流式输出结束后换行

        # [易错] except Exception 会兜住所有普通异常；这里重新抛出自定义的 HelloAgentsException，
        #        好处是上层只需要认一种异常类型。原始异常会挂在 __context__ 上（隐式链），
        #        想显式控制链条得写 `raise ... from e` —— 这里没写，所以报错信息里是「处理上述异常时又发生了…」。
        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            raise HelloAgentsException(f"LLM调用失败: {str(e)}")

    # [作用] 非流式调用：一次拿完整回复，返回字符串。
    # [易错] 注意它和 think 的返回类型不一样：invoke 返回 str，think 返回生成器；
    #        本方法里也没有 print 和 yield。把 invoke 的结果拿去 for 循环会逐字符遍历，不是逐片段。
    def invoke(self, messages: list[dict[str, str]], **kwargs) -> str:
        """
        非流式调用LLM，返回完整响应。
        适用于不需要流式输出的场景。
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                # [语法] 字典推导式 + ** 解包：先剔除已经单独传过的 temperature / max_tokens，
                #        再把剩下的 kwargs 展开成 create() 的关键字参数。
                **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens']}
            )
            return response.choices[0].message.content
        except Exception as e:
            raise HelloAgentsException(f"LLM调用失败: {str(e)}")

    # [作用] think 的别名，纯粹为了向后兼容（老代码里这个名字叫 stream_invoke）。
    # [语法] yield from：把迭代「委托」给另一个生成器，逐条转发它的产出，
    #        等价于手写 `for x in gen: yield x`，但更短。这里每次调用都新建一个 think 生成器再转发。
    def stream_invoke(self, messages: list[dict[str, str]], **kwargs) -> Iterator[str]:
        """
        流式调用LLM的别名方法，与think方法功能相同。
        保持向后兼容性。
        """
        temperature = kwargs.get('temperature')
        yield from self.think(messages, temperature)
