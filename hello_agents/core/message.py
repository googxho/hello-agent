# ===== 文件导读 =====
# [职责] 定义框架内统一的消息格式 Message，把「一条对话」标准化成 role + content + 时间戳 + 元数据。
# [位置] 被 core/agent.py 的历史记录（list[Message]）和各 agents/*.py 使用；本文件只依赖标准库 + pydantic。
# [阅读顺序] ① 顶部 MessageRole 类型别名 → ② 类字段声明（Pydantic 的「字段」是什么）
#            → ③ 被覆盖的 __init__（为什么要覆盖）→ ④ to_dict / __str__ 两个出口方法。
# ===================
"""消息系统"""

# [语法] 类型注解的四种常见包装：Optional[X] 表示「X 或 None」；Dict[K, V] / list[X] 表示容器；
#        Any 表示「什么都行」；Literal[...] 表示「只能取列出的几个值之一」。
from typing import Optional, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel

# [语法] 类型别名：给 Literal[...] 起个名字 MessageRole，下面直接当类型用。
# [易错] 同样是 Literal，写在哪里效果完全不同：写在函数注解里（如 llm.py 的 provider）
#        只是给类型检查器看的，运行时不校验；但写在 Pydantic 的模型字段上，就【真的会在运行时校验】。
#        实测：Message("hi", "boss") 直接抛 pydantic ValidationError（"boss" 不在 Literal 里）。
MessageRole = Literal["user", "assistant", "system", "tool"]

# [作用] 一条对话消息的数据结构。继承 pydantic 的 BaseModel，白拿三样东西：
#        ① 构造时按注解做类型校验 ② 自动生成 __repr__ 等样板方法 ③ 序列化能力（.dict() / .model_dump()）。
# [概念] 继承：Message 自动拥有 BaseModel 的全部行为，只需声明自己的字段。
#        不过本文件的 to_dict 是手工拼 dict，并没用到那套序列化能力。
# [易错] 下面那种 `字段名: 类型` 的写法，在 Pydantic 模型里表示「声明一个字段」；
#        但在【普通类】里同样写一行，Python 只把它记进 __annotations__，既不建属性也不校验 —— 别混淆。
class Message(BaseModel):
    """消息类"""
    
    # [作用] 四个字段：正文、角色、时间戳、附加信息。Pydantic 会把它们变成实例属性，并在赋值时校验类型。
    content: str
    role: MessageRole
    # [易错] 注解说是 datetime，默认值却给了 None —— 类型不一致（mypy 会报）。
    #        又因为 Pydantic 默认不校验「默认值」本身，真走到默认分支时这个字段会是 None。
    #        不过本文件的 __init__ 总会把 timestamp 传进来，所以这行默认值实际没被用上。
    timestamp: datetime = None
    # [作用] metadata 是给调用方留的扩展口袋（比如挂工具调用记录）；它不参与和模型对话，
    #        下面的 to_dict 也不会把它带出去。
    metadata: Optional[Dict[str, Any]] = None
    
    # [作用] 覆盖父类的 __init__，额外做两件事：允许位置参数 (content, role)，并给 timestamp / metadata 填默认值。
    # [为什么] Pydantic 的 BaseModel.__init__ 只接受【关键字】参数，实测写 Plain("hi", "user") 会抛
    #        TypeError: BaseModel.__init__() takes 1 positional argument but 3 were given。
    #        而全项目 23 处都在用位置写法 Message(text, "user")，所以必须在这里开这个口子。
    # [语法] **kwargs 收集所有没在参数表里出现的关键字参数，在函数体里就是一个 dict。
    # [易错] 下面 kwargs.get('timestamp', datetime.now()) 里的 datetime.now() 是「每次调用都重新求值」的表达式，
    #        这没问题；但若把它写成函数默认值 def __init__(self, timestamp=datetime.now())，
    #        就变成「模块导入时求值一次」，所有实例会共用同一个时间戳 —— 这是经典陷阱（默认参数只求值一次）。
    def __init__(self, content: str, role: MessageRole, **kwargs):
        # [语法] super().__init__(...) 调用父类（BaseModel）的初始化 —— 上面那三个字段的类型校验就发生在这里。
        super().__init__(
            content=content,
            role=role,
            timestamp=kwargs.get('timestamp', datetime.now()),
            metadata=kwargs.get('metadata', {})
        )
    
    # [作用] 转成 OpenAI API 需要的格式。
    # [实测] Message("你好", "user").to_dict() → {'role': 'user', 'content': '你好'}；
    #        注意 timestamp 和 metadata 都【没】被带出去 —— 只留 role 和 content，因为接口只认这两个键。
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（OpenAI API格式）"""
        return {
            "role": self.role,
            "content": self.content
        }
    
    # [语法] __str__ 是魔术方法（dunder，双下划线前后各两条）：它定义了 str(obj) 和 print(obj) 的行为。
    # [概念] 魔术方法 = Python 在特定语法下自动替你调用的方法：你写 print(m)，解释器去找 m.__str__()。
    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"
