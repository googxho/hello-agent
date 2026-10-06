# ===== 文件导读 =====
# [职责] 定义所有 Agent 的抽象基类 Agent：规定「每个 Agent 必须能 run，且自带历史记录」。
# [位置] 被 agents/ 下 5 个子类继承（SimpleAgent / FunctionCallAgent / ReActAgent / ReflectionAgent /
#        PlanAndSolveAgent；ToolAwareSimpleAgent 间接继承）；它持有 HelloAgentsLLM 和 Config。
# [阅读顺序] ① 为什么继承 ABC → ② __init__ 的实例属性 → ③ @abstractmethod 的 run
#            → ④ 历史记录三件套 → ⑤ __str__ / __repr__。
# ===================
"""Agent基类"""

# [概念] abc = abstract base class（抽象基类）模块。ABC 是本文件要继承的基类，
#        @abstractmethod 用来标记「子类必须实现」的方法。
# [机制] 这两个零件有明确分工：ABC 负责提供「执法者」元类 ABCMeta，@abstractmethod 只负责贴标签。
#        （元类 = 造类的类。普通类的 type(SomeClass) 是 type；实测 type(Agent) 是 abc.ABCMeta。）
from abc import ABC, abstractmethod
from typing import Optional
# [语法] 相对导入：`.message` 开头的点表示「当前所在这个包」（hello_agents.core）。
#        这里三行分别导入消息、模型客户端、配置 —— 也就是 Agent 依赖的三个零件。
from .message import Message
from .llm import HelloAgentsLLM
from .config import Config

# [作用] 所有 Agent 的公共父类：统一持有 name / llm / system_prompt / config / 历史记录。
# [概念] 继承：子类自动获得父类的方法和属性，只需补上自己特有的部分。
# [为什么] 用 ABC 而不是普通类：它把「必须实现 run」从口头约定变成强制约束 —— 漏了就实例化不了。
# [机制] 强制力是怎么来的（不是魔法，是一个集合）：
#        ① 建类时，ABCMeta 扫描这个类（含继承来的），把所有「还带着抽象标签、且没有具体实现」的
#           方法名记进一本账 —— 实测 Agent.__abstractmethods__ == frozenset({'run'})；
#        ② 每次要造实例时，object.__new__ 先看这本账：非空就直接抛 TypeError，连 __init__ 都不进。
# [机制] 实测铁证：手动把账本清空（Agent.__abstractmethods__ = frozenset()）后，
#        实例立刻就能造出来 —— 说明全部强制力就是这一个集合，没有任何隐藏机制。
class Agent(ABC):
    """Agent基类"""
    
    # [参数] name 是标识；llm 是上一轮讲的模型客户端；system_prompt 和 config 可选（默认 None）。
    # [易错] Optional[X] = None 是「调用时可以省略」的标准写法。config 传 None 时会走下面的
    #        `config or Config()` 兜底新建一份，所以这里允许不传；但 llm 没有默认值，是必填的。
    # [语法] **kwargs 让子类能接收额外参数。本类自己不用它，只是把口子留着方便扩展。
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None
    ):
        # [作用] 把参数绑定成实例属性 —— 每个实例各有一份，互不干扰。self 指「当前这个实例」。
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        # [语法] `config or Config()`：传了就用传进来的，传 None（或其他假值）就新建一个默认配置 —— or 短路求值。
        self.config = config or Config()
        # [语法] 这是「变量注解」：`self._history: list[Message] = []` 里的注解只写给人和类型检查器看，
        #        运行时不做任何校验，右边依然是普通空列表。
        # [概念] 单下划线开头 = 「内部属性」约定，提示外部别直接改它，走 add_message / get_history。
        # [易错] 关键点：这个 [] 写在 __init__ 里，所以每个实例都会新建一个独立列表。若改写成
        #        类属性 `_history = []` 或默认参数 `def __init__(self, history=[])`，所有实例会共享同一个列表
        #        —— 这是 Python 最著名的坑（可变默认值）。
        self._history: list[Message] = []
    
    # [作用] 声明「所有子类都必须实现 run」。
    # [语法] 装饰器 @abstractmethod 必须配合 ABC 基类使用，单独写没有强制力。
    # [机制] 这个装饰器只做一件事：给函数贴上 __isabstractmethod__ = True 标签（实测 Agent.run 上有它）。
    #        它自己毫无执法能力 —— 实测在【不继承 ABC】的普通类里写 @abstractmethod，类照样能实例化，
    #        因为压根没有 ABCMeta 来读这个标签。
    # [实测] Agent("裸的", llm=None) 抛 TypeError: Can't instantiate abstract class Agent without an
    #        implementation for abstract method 'run'；子类忘了实现 run，同样实例化不了。
    #        子类补上 run 之后，账本变成空集（实测 SimpleAgent.__abstractmethods__ == frozenset()），于是能实例化。
    # [参数] input_text 是用户输入；**kwargs 让子类能加自己的可选参数（如 max_tool_iterations）。
    # [返回] str。注意子类可能另有 stream_run() 返回生成器 —— 同族方法的返回类型并不一致。
    # [易错] 子类重写 run 时签名要兼容：参数名和顺序最好保持一致，否则调用方按位置传参会错位。
    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """运行Agent"""
        # [语法] pass 是空语句，用来占位 —— 函数体不允许什么都不写。
        #        这里的真正实现由各子类提供，父类只规定「有这个方法」。
        pass
    
    # [作用] 追加一条消息到历史（把列表 append 封装成方法，外部不用碰 _history）。
    # [概念] 这就是「封装」：把内部数据结构藏起来，只暴露有意义的操作。
    def add_message(self, message: Message):
        """添加消息到历史记录"""
        self._history.append(message)
    
    # [语法] list.clear() 是【原地】清空：在原来那个列表对象上删元素。
    #        和 `self._history = []` 不同 —— 后者是换一个新列表，外部若还握着旧引用就看不到清空效果。
    def clear_history(self):
        """清空历史记录"""
        self._history.clear()
    
    # [作用] 返回历史记录的副本。
    # [易错] 结尾的 .copy() 是防御性拷贝：如果直接 `return self._history`，调用方一句 .clear()
    #        就能把对象内部状态清空（外面拿到了内部容器的引用）。
    # [实测] 拿到副本后 clear() 它，本对象内部仍有消息 —— 副本挡住了外部修改。
    def get_history(self) -> list[Message]:
        """获取历史记录"""
        return self._history.copy()
    
    # [语法] __str__ / __repr__ 都是魔术方法：print(agent) 走 __str__，交互式环境里直接敲 agent 走 __repr__。
    # [概念] 这里让 __repr__ 直接返回 __str__，避免两个输出不一致（常见约定）。
    # [易错] 这句 f-string 读了 self.llm.provider —— Python 运行时不做类型检查（鸭子类型），
    #        所以传任何「有 provider 属性的对象」都能通过，实测传个假对象 FakeLLM() 也能跑起来。
    def __str__(self) -> str:
        return f"Agent(name={self.name}, provider={self.llm.provider})"
    
    # [作用] 让调试时的输出和 str() 保持一致。
    def __repr__(self) -> str:
        return self.__str__()