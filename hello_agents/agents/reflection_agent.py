# ===== 文件导读 =====
# [职责] Reflection（反思）范式的 Agent：先生成初稿 → 让模型自己批评自己 → 按批评改稿 → 重复若干轮。
# [位置] 继承 core/agent.py 的 Agent 并实现抽象方法 run；依赖 core 三件套，
#        外加一个本文件私有的 Memory 小类（它跟 memory/ 包、core/message.py 的 Message 都无关）。
# [阅读顺序] ① DEFAULT_PROMPTS（三段提示词模板）→ ② Memory（私有的轨迹记录器）
#            → ③ ReflectionAgent.__init__（模板怎么被替换）→ ④ run（初稿 → 反思 → 判断停 → 优化）
#            → ⑤ _get_llm_response（一段字符串怎么变成 messages）。
# ===================
"""Reflection Agent实现 - 自我反思与迭代优化的智能体"""

# [语法] typing 里的注解工具：Optional[X] 读作「X 或 None」，List/Dict/Any 用来标注容器与任意类型。
# [易错] 这些注解运行时不做任何校验（见 core/agent.py 的说明）；写成别的类型也照样能跑。
from typing import Optional, List, Dict, Any
# [语法] `..core.agent` 里的两个点表示「上一层包」：本文件在 hello_agents.agents，上一层是 hello_agents，
#        于是它指向 hello_agents/core/agent.py —— 下面 ReflectionAgent 要继承的父类就在那里。
from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message

# [概念] 这三个模板就是 Reflection 范式的骨架：initial 出初稿、reflect 做自我批评、refine 按批评改稿。
# [为什么] 把它们放在模块级常量里（而不是写死在 run 内）是为了「可替换」：构造函数能整套换掉。
# 默认提示词模板
# [概念] 它是个普通 dict：键是阶段名，值是带 {task} / {content} 这类占位符的模板字符串；
#        运行时靠 str.format() 把真实内容填进去 —— 模板和值就是这样拼起来的。
DEFAULT_PROMPTS = {
    "initial": """
请根据以下要求完成任务：

任务: {task}

请提供一个完整、准确的回答。
""",
    "reflect": """
请仔细审查以下回答，并找出可能的问题或改进空间：

# 原始任务:
{task}

# 当前回答:
{content}

请分析这个回答的质量，指出不足之处，并提出具体的改进建议。
如果回答已经很好，请回答"无需改进"。
""",
    "refine": """
请根据反馈意见改进你的回答：

# 原始任务:
{task}

# 上一轮回答:
{last_attempt}

# 反馈意见:
{feedback}

请提供一个改进后的回答。
"""
}

# [易错] 上面这些三引号里的换行、缩进、中文标点全都是提示词正文，会原样发给模型 ——
#        它不是「代码排版」，改动时连一个空行都算改内容。
# [作用] 本文件私有的轨迹记录器：按时间顺序存「执行结果」和「反思反馈」两类记录。
# [概念] 它不是 core/message.py 里的 Message，也不是 memory/ 包的一部分 —— 只是这个小文件内的工具类。
# [为什么] 只存两类字符串、不检索、不持久化：Reflection 只需要「最近一次尝试」和「全部轨迹」，够用就不加。
class Memory:
    """
    简单的短期记忆模块，用于存储智能体的行动与反思轨迹。
    """
    # [作用] 唯一的实例属性：一个按时间顺序存的列表，元素是 {"type": ..., "content": ...} 这样的字典。
    # [易错] 它在 __init__ 里新建，所以每个 Memory 各有一份；若写成类属性，所有实例会共享同一个列表。
    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    # [作用] 追加一条记录，并打印一行提示。
    # [参数] record_type 目前只有两个取值：'execution' 和 'reflection'（下面 get_trajectory 按它分支）。
    # [副作用] 会 print —— 这个类既管数据又管输出，职责其实是混的。
    def add_record(self, record_type: str, content: str):
        """向记忆中添加一条新记录"""
        self.records.append({"type": record_type, "content": content})
        print(f"📝 记忆已更新，新增一条 '{record_type}' 记录。")

    # [作用] 把全部记录拼成一段多行文本，本意是塞进提示词让模型看到完整轨迹。
    # [概念] for + if/elif 按 record['type'] 分发内容格式，是「用条件判断代替多态」的最朴素写法。
    # [易错] 在本仓库里搜过一遍：没有任何地方调用它（run 只用了 get_last_execution）——
    #        也就是说「把完整轨迹喂给模型」这个设计目前并未生效，是一段死代码。
    def get_trajectory(self) -> str:
        """将所有记忆记录格式化为一个连贯的字符串文本"""
        trajectory = ""
        for record in self.records:
            if record['type'] == 'execution':
                trajectory += f"--- 上一轮尝试 (代码) ---\n{record['content']}\n\n"
            elif record['type'] == 'reflection':
                trajectory += f"--- 评审员反馈 ---\n{record['content']}\n\n"
        return trajectory.strip()

    # [作用] 取最近一次执行结果 —— 反思要拿它当批评对象，所以它是每轮迭代的输入。
    # [机制] reversed(self.records) 让 for 从列表尾部往前走，第一个命中的就是最新那条 execution；
    #        不认识的 type 会被自动跳过而不报错。找不到时返回空串 ""，不是 None。
    def get_last_execution(self) -> str:
        """获取最近一次的执行结果"""
        for record in reversed(self.records):
            if record['type'] == 'execution':
                return record['content']
        return ""

# [作用] 反思式 Agent：把「生成 → 自评 → 改写」跑 max_iterations 轮，返回最后一版结果。
# [概念] 组合优于继承：它继承的是 Agent（拿到 name / llm / _history），
#        而 Memory 是「持有」的（self.memory = Memory()）—— 功能靠「有一个」拼进来，不靠「是一个」承下来。
# [机制] 父类 Agent 用 @abstractmethod 记了账，本类必须实现 run，否则实例化直接抛 TypeError。
class ReflectionAgent(Agent):
    """
    Reflection Agent - 自我反思与迭代优化的智能体

    这个Agent能够：
    1. 执行初始任务
    2. 对结果进行自我反思
    3. 根据反思结果进行优化
    4. 迭代改进直到满意

    特别适合代码生成、文档写作、分析报告等需要迭代优化的任务。

    支持多种专业领域的提示词模板，用户可以自定义或使用内置模板。
    """

    # [参数] custom_prompts 的键要和 DEFAULT_PROMPTS 对齐（initial / reflect / refine），
    #        否则下面 self.prompts["initial"] 这种取值会直接 KeyError。
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_iterations: int = 3,
        custom_prompts: Optional[Dict[str, str]] = None
    ):
        """
        初始化ReflectionAgent

        Args:
            name: Agent名称
            llm: LLM实例
            system_prompt: 系统提示词
            config: 配置对象
            max_iterations: 最大迭代次数
            custom_prompts: 自定义提示词模板 {"initial": "", "reflect": "", "refine": ""}
        """
        # [机制] 必须显式调父类构造：self.name / self.llm / self.config / self._history 都建在它里面，
        #        漏了这句，子类自己的属性还在，但 self.add_message 与 print 里的 self.name 会立刻 AttributeError。
        super().__init__(name, llm, system_prompt, config)
        self.max_iterations = max_iterations
        # [概念] 这就是「持有」的具体形态：构造时 new 一个私有 Memory 挂到实例上，而不是继承它。
        self.memory = Memory()

        # [易错] ⭐ 这里是「整体替换」而不是「逐键覆盖」：传了 custom_prompts 就整本换掉，
        #        没写到的键不会回退到默认值，于是 self.prompts["refine"] 会 KeyError。
        # [为什么] 想只改一个模板，得自己合并：{**DEFAULT_PROMPTS, **custom_prompts}；本文件没这么做（只报告，不改）。
        # 设置提示词模板：用户自定义优先，否则使用默认模板
        self.prompts = custom_prompts if custom_prompts else DEFAULT_PROMPTS
    
    # [作用] 主流程：初稿 → for 循环（反思 → 判断是否停 → 优化）→ 返回记忆里最后一条 execution。
    # [参数] input_text 是任务描述；**kwargs 一路透传给 llm.invoke（可借此覆盖 temperature 等）。
    # [返回] str，最终那一版结果；注意反思反馈不会作为返回值，它只留在 memory 里。
    # [副作用] 打印大量进度信息；把「任务 + 最终结果」写进父类的 _history。
    def run(self, input_text: str, **kwargs) -> str:
        """
        运行Reflection Agent

        Args:
            input_text: 任务描述
            **kwargs: 其他参数

        Returns:
            最终优化后的结果
        """
        print(f"\n🤖 {self.name} 开始处理任务: {input_text}")

        # [易错] 每次 run 都新建一个 Memory，把上一轮的轨迹整包丢掉（__init__ 里已经建过一次）。
        #        好处是两次任务不串味；代价是「同一个 Agent 连跑两个任务」也留不下上一轮的轨迹。
        # 重置记忆
        self.memory = Memory()

        # [概念] 阶段①「先做出来」：Reflection 的前提是先有一个可批评的对象，没有初稿就无从反思。
        # [易错] 初稿同样是一次真实 API 调用：总调用次数是 1 + 每轮 2 次，max_iterations=3 时最多 7 次。
        # 1. 初始执行
        print("\n--- 正在进行初始尝试 ---")
        # [语法] str.format(task=...) 把模板里的 {task} 换成真实任务文本；
        #        模板里出现、而这里没传的键会抛 KeyError（反过来多传则无害）。
        initial_prompt = self.prompts["initial"].format(task=input_text)
        # [概念] 所有模型调用都走本文件末尾的 _get_llm_response —— 它是本文件唯一的出口。
        initial_result = self._get_llm_response(initial_prompt, **kwargs)
        self.memory.add_record("execution", initial_result)

        # [概念] 阶段②③：一个固定次数的 for，把「批评」和「改稿」串成一轮，中途可以 break 提前收工。
        # [易错] 循环变量 i 只用于打印轮次；真正的「第几轮」状态其实存在 memory.records 里。
        # 2. 迭代循环：反思与优化
        for i in range(self.max_iterations):
            print(f"\n--- 第 {i+1}/{self.max_iterations} 轮迭代 ---")

            # [概念] 反思 = 把「原始任务 + 当前回答」一起发给模型，让它当评审员 ——
            #        它批评的其实是自己上一轮的输出，这就是 Reflection 的全部秘密。
            # [为什么] 这招有效，是因为「挑毛病」比「一次写对」容易：把难任务拆成一个更简单的子任务。
            # a. 反思
            print("\n-> 正在进行反思...")
            # [机制] 取记忆里最后一条 execution：第 1 轮是初稿，之后是上一轮的改稿 ——
            #        批评对象始终是「上一版」，而不是越滚越长的全部历史。
            last_result = self.memory.get_last_execution()
            # [易错] 这里显式传了 task / content 两个键，模板里必须正好用得上；
            #        用户自定义模板若漏了 {content}，不报错，但模型就看不到要批评的回答了。
            reflect_prompt = self.prompts["reflect"].format(
                task=input_text,
                content=last_result
            )
            feedback = self._get_llm_response(reflect_prompt, **kwargs)
            self.memory.add_record("reflection", feedback)

            # [易错] ⭐ 停止条件靠在反馈里做字符串匹配：硬编码了「无需改进」和 "no need for improvement" 两种说法。
            #        模型只要换个说法（「已经很完善」「无明显问题」），就匹配不上，循环会一直跑到 max_iterations。
            # [为什么] 模板里那句「请回答"无需改进"」只是请求、不是协议，模型没有任何机制必须照做 ——
            #        对比 function_call_agent 用结构化字段判断，这里的刹车天生不可靠。
            # b. 检查是否需要停止
            # [机制] feedback.lower() 对英文那半边有用；中文没有大小写，lower() 会原样返回，不改变匹配结果。
            if "无需改进" in feedback or "no need for improvement" in feedback.lower():
                print("\n✅ 反思认为结果已无需改进，任务完成。")
                break

            # [概念] 优化 = 把「任务 + 上一版回答 + 反馈」再发回去，让它按批评重写一版。
            # [易错] 反馈是自然语言，模型可能理解偏、甚至改坏；本文件没有任何「质量是否真的提升」的检测，
            #        唯一的裁判是下一轮它自己的反思 —— 这就是「自我反思」的天花板。
            # c. 优化
            print("\n-> 正在进行优化...")
            refine_prompt = self.prompts["refine"].format(
                task=input_text,
                last_attempt=last_result,
                feedback=feedback
            )
            refined_result = self._get_llm_response(refine_prompt, **kwargs)
            self.memory.add_record("execution", refined_result)

        # [易错] 「最终结果」就是记忆里最后一条 execution：因 break 而结束时它是上一轮那版，
        #        max_iterations 跑满时它是最后一次改稿 —— 两种都合理，但语义并不相同。
        final_result = self.memory.get_last_execution()
        print(f"\n--- 任务完成 ---\n最终结果:\n{final_result}")

        # [概念] 只把「任务 + 最终结果」写进父类的对话历史：中间初稿和每轮反馈都留在私有的 memory 里。
        #        两个容器各管一段 —— memory 管本轮迭代轨迹，_history 管跨轮对话。
        # 保存到历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_result, "assistant"))

        return final_result
    
    # [作用] 本文件唯一的模型出口：把一段纯字符串 prompt 包装成 OpenAI 要求的消息列表，再走非流式调用。
    # [参数] prompt 是已经 format 好的完整提示词；**kwargs 原样透传给 llm.invoke（例如 temperature）。
    # [返回] str —— 模型回空时兜底成 ""（invoke 可能返回 None，这里挡一道）。
    # [副作用] 无：不打印、不存历史，状态变更全由调用方 run() 负责。
    def _get_llm_response(self, prompt: str, **kwargs) -> str:
        """调用LLM并获取完整响应"""
        # [概念] 模型只认 messages 这种结构（role + content 的列表），不认裸字符串 ——
        #        这就是「包装」这一步存在的理由。这里固定用 "user" 角色，没用到 system。
        messages = [{"role": "user", "content": prompt}]
        return self.llm.invoke(messages, **kwargs) or ""
