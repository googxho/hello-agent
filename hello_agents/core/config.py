# ===== 文件导读 =====
# [职责] 框架的统一配置对象 Config，给所有 Agent 一份「默认参数」（模型、温度、日志、历史上限）。
# [位置] 被 core/agent.py 持有（self.config = config or Config()）；只依赖标准库 os 和 pydantic。
# [阅读顺序] ① 字段声明（Pydantic 字段默认值）→ ② from_env 类方法（装饰器 + 环境变量）→ ③ to_dict。
# [提醒] 实测全仓库没有任何地方调用过 Config.from_env() —— 它目前是个「备用入口」，别以为它在生效。
# ===================
"""配置管理"""

# [语法] os 提供 os.getenv 读环境变量（永远返回字符串或 None）；typing 提供三种注解包装。
import os
from typing import Optional, Dict, Any
from pydantic import BaseModel

# [作用] 框架级默认配置。继承 BaseModel 后自动获得类型校验和序列化能力。
# [概念] 在 Pydantic 模型里，带注解的类属性 = 字段；后面 `= 值` 就是这个字段的默认值。
# [易错] 别和普通类混淆：普通类里写 `temperature: float = 0.7` 是【类属性】，所有实例共享同一份；
#        Pydantic 会把它变成每个实例自己的字段副本，所以改一个实例不影响另一个。
class Config(BaseModel):
    """HelloAgents配置类"""
    
    # LLM配置
    # [作用] 每个字段都是「可被外部覆盖的默认值」：Config(temperature=0.2) 只改这一项，其余保持默认。
    default_model: str = "gpt-3.5-turbo"
    default_provider: str = "openai"
    temperature: float = 0.7
    # [易错] Optional[int] = None 用来表达「可以没有上限」。注意 Optional[X] 的意思不是「可选参数」，
    #        而是「类型是 X 或 None」—— 这是最常见的误读。
    max_tokens: Optional[int] = None
    
    # 系统配置
    debug: bool = False
    log_level: str = "INFO"
    
    # 其他配置
    max_history_length: int = 100
    
    # [语法] @classmethod 是装饰器：被它装饰后，第一个参数自动接收「类本身」而不是实例，
    #        所以习惯上把 self 改名叫 cls。
    # [概念] 装饰器 = 「接收一个函数、返回另一个（通常被增强过的）函数」的函数；
    #        @ 只是语法糖，等价于写 from_env = classmethod(from_env)。
    # [机制] classmethod 的底牌是「描述符」：存在类字典里的原始形态就是一个 classmethod 对象（实测
    #        type(Config.__dict__['from_env']) 是 classmethod）。读属性时由它的 __get__ 决定绑定谁 ——
    #        classmethod 绑定到【类】，普通方法绑定到【实例】，staticmethod 谁都不绑。这就是同一份函数
    #        能有三种调用形态的底层原因。
    # [为什么] from_env 不需要先造出实例就能调用：Config.from_env()。
    # [语法] 返回类型写成字符串 "Config" 叫前向引用（forward reference）：此刻类还没定义完，
    #        在 3.11 及更早的 Python 里不加引号会直接 NameError。本机是 Python 3.14（PEP 649 让注解变懒），
    #        实测不加引号也能跑 —— 但加引号仍是跨版本更稳的写法。
    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量创建配置"""
        # [作用] 从环境变量造一份配置。字段本身都有默认值，这里只挑几个允许被环境覆盖的。
        # [易错] os.getenv 拿到的永远是字符串，所以 0.7 要 float() 转、MAX_TOKENS 要 int() 转；
        #        忘了转的话 temperature 会是字符串 "0.7"，等到参与运算时才炸。
        return cls(
            # [易错] 布尔值必须自己判："false".lower() == "true"。
            #        因为非空字符串 "false" 在 Python 里是【真值】，写成 bool(os.getenv("DEBUG", "false")) 永远是 True。
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            # [语法] 条件表达式：A if 条件 else B。这里是「有 MAX_TOKENS 才 int() 转，否则给 None」，
            #        避免 int(None) 直接抛 TypeError。
            # [易错] os.getenv("MAX_TOKENS") 被调用了两次；能跑，但先存进变量更省。另外写成 MAX_TOKENS= 这种
            #        空字符串时会被当成「没设置」，结果给 None。
            max_tokens=int(os.getenv("MAX_TOKENS")) if os.getenv("MAX_TOKENS") else None,
        )
    
    # [作用] 转成普通 dict，方便打印 / 存盘 / 传参。
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        # [易错] self.dict() 是 Pydantic v1 的老 API。本环境装的是 pydantic 2.13.4，实测会发
        #        PydanticDeprecatedSince20 警告：The `dict` method is deprecated; use `model_dump` instead。
        #        现在还能跑（v2 保留兼容），v3 会删掉 —— 属于「知道就行、别急着改」的那类问题。
        return self.dict()
