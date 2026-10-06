# ===== 文件导读 =====
# [职责] Plan-and-Solve 范式：先用 Planner 把问题拆成步骤列表，再用 Executor 逐步执行，PlanAndSolveAgent 只做编排。
# [位置] 继承 core/agent.py 的 Agent（必须实现抽象方法 run），只依赖 core 四件套，不碰 tools/。
#        本文件是「一个文件里三个类」的样板：两个普通零件类 + 一个只做组装的 Agent。
# [阅读顺序] ① Planner（问题 → List[str]）→ ② Executor（List[str] → 最终答案）
#            → ③ PlanAndSolveAgent.__init__（把两个零件装到自己身上）
#            → ④ run（先调 plan、再调 execute，自己不写解析逻辑）。
# [一句话] 本文件的本质：**把「规划」和「执行」拆成两个可单独替换的零件，
#          Agent 只负责按顺序调用它们** —— 对比 SimpleAgent 把全部逻辑塞进 run 的写法。
# ===================
"""Plan and Solve Agent实现 - 分解规划与逐步执行的智能体"""

# [概念] 概念②（文本协议解析）的零件：ast.literal_eval ——「把模型吐出来的文本还原成 Python 对象」的工具，
#        它到底怎么做到「安全求值」的，在下面 plan() 里展开。
import ast
from typing import Optional, List, Dict
# [概念] 概念①（组合 vs 继承）：注意下面 Planner / Executor 两个类**都没有**继承这个 Agent，
#        整份文件里只有末尾的 PlanAndSolveAgent 继承了它 —— 另外两个是「零件」，不是「Agent 的变种」。
from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message

# [概念] 概念②（文本协议）的「约定」那一半：提示词里明确要求模型输出 ```python 包裹的 Python 列表。
#        ⚠ 下面这段三引号模板是**提示词**不是注释，一个字都不能动 —— 它就是协议的书面条款。
# 默认规划器提示词模板
DEFAULT_PLANNER_PROMPT = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

# [概念] 第二份提示词：它带 {question} / {plan} / {history} / {current_step} 四个占位符，
#        会在 Executor.execute 里的 .format(...) 那句被逐个填满。
# 默认执行器提示词模板
DEFAULT_EXECUTOR_PROMPT = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:
"""

# [作用] 规划器：只干一件事 —— 把一句自然语言问题，变成一串可执行的步骤。
# [概念] 概念①：它是**普通类**，父类是 object，不是 Agent。所以它没有 run()、没有历史记录、
#        也不知道 PlanAndSolveAgent 的存在 —— 这正是组合的好处：零件不需要认识组装它的人。
class Planner:
    """规划器 - 负责将复杂问题分解为简单步骤"""

    # [参数] llm_client 就是 core/llm.py 那个 HelloAgentsLLM；prompt_template 传 None 表示「用默认模板」。
    # [易错] 命名不一致：零件类这个形参叫 llm_client，而装配它的 Agent __init__ 里叫 llm ——
    #        同一个东西两个名字，读代码时要自己把这根线接上。
    def __init__(self, llm_client: HelloAgentsLLM, prompt_template: Optional[str] = None):
        self.llm_client = llm_client
        # [语法] 概念③（默认值兜底）：三元表达式 `A if 条件 else B`，条件为真取 A，否则取 B。
        # [易错] 判的是真/假而不是「是否为 None」：prompt_template="" 这种空字符串也会被当成「没传」，
        #        静默用回默认模板（实测 {"planner": ""} 确实走默认）。
        self.prompt_template = prompt_template if prompt_template else DEFAULT_PLANNER_PROMPT

    # [作用] 把问题交给模型，再把模型回复里的计划文本解析成 Python 列表。
    # [参数] question 是用户问题；**kwargs 是给 LLM 的调用参数（temperature 之类），原样转发。
    # [返回] List[str]。返回空列表 [] 表示「解析失败」—— 这是**用空值表达失败**的约定，
    #        它不抛异常，所以调用方必须自己检查（见 run 里的卫语句 if not plan）。
    def plan(self, question: str, **kwargs) -> List[str]:
        """
        生成执行计划

        Args:
            question: 要解决的问题
            **kwargs: LLM调用参数

        Returns:
            步骤列表
        """
        # [语法] str.format()：把模板里的 {question} 换成实参。占位符写错会报错 ——
        #        多传一个不存在的键会 KeyError，模板里有没被填的占位符则原样留在字符串里（不报错）。
        prompt = self.prompt_template.format(question=question)
        # [概念] 只有一条 user 消息，没有 system —— 角色设定全靠提示词正文里那段话。
        # [为什么] messages 是「本次调用」的局部变量，刻意不带历史：规划是一次性动作，不需要记住上一轮。
        messages = [{"role": "user", "content": prompt}]

        print("--- 正在生成计划 ---")
        # [易错] 末尾 `or ""` 是防御：invoke 返回 None 时，下面的 .split() 会直接 AttributeError；
        #        兜成空串后，失败路径就统一收敛成「split 越界 → 返回 []」这一条。
        response_text = self.llm_client.invoke(messages, **kwargs) or ""
        print(f"✅ 计划已生成:\n{response_text}")

        try:
            # 提取Python代码块中的列表
            # [概念] 概念②（文本协议）的核心现场：模型只能返回**字符串**，返回不了 Python 对象，
            #        所以「它说了什么」全靠 约定格式 + 字符串切割 来还原 —— 这就是文本协议的代价。
            # [机制] split("```python")[1] 取「第一个代码块之后」，再 split("```")[0] 取「下一个 ``` 之前」，
            #        两头一夹，剩下的正好是列表字面量（str.split 返回列表，[1] 就是取第二段，找不到就 IndexError）。
            # [实测] 只有标准格式能过：'```python\n["查资料","写摘要"]\n```' → ['查资料', '写摘要']；
            #        而「只写 1. 2. 编号」「只写 ``` 没写 python」→ 都返回 []；「回复里有两个 ```python 块」→ 静默只取第一个。
            plan_str = response_text.split("```python")[1].split("```")[0].strip()
            # [概念] 概念②的下一半：把「长得像列表的字符串」变成**真的列表**。
            # [机制] 为什么不用 eval()：literal_eval 先把字符串 parse 成 AST（语法树），再逐个节点检查类型，
            #        只放行字面量（数字/字符串/列表/元组/字典/布尔/None），遇到函数调用这类节点直接抛 ValueError。
            # [实测] literal_eval("__import__('os').system('echo x')") → ValueError: malformed node or string；
            #        同一个串交给 eval()，终端真的把 x 打了出来 —— 即模型只要输出恶意代码就会被执行。
            plan = ast.literal_eval(plan_str)
            # [易错] 类型闸门：literal_eval 还可能返回 dict / tuple / str，只有 list 才放行，其余一律变成 []。
            # [实测] 围栏内写 '("a","b")'（元组）或 '"查资料"'（字符串）→ literal_eval 拿到的是元组/字符串，
            #        过不了类型闸门，都返回 []。注意它**不过滤空字符串**：围栏内写 '["", "b"]' 原样返回
            #        ['', 'b'] —— 空步骤会被当成真步骤喂给模型（见 Executor 那一侧）。
            return plan if isinstance(plan, list) else []
        # [语法] 一个 except 用元组接多种异常，对应上面三个可能的故障点：
        #        IndexError（没有 ```python 围栏，split(...)[1] 越界）、SyntaxError（列表括号/引号不配对）、
        #        ValueError（literal_eval 认不出的表达式，比如 1+1 或函数调用）。
        # [易错] 兜底策略是「打印 + 返回 []」而不往上抛：调用方只能靠「空列表」这一个信号判断失败。
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"❌ 解析计划时出错: {e}")
            print(f"原始响应: {response_text}")
            return []
        # [概念] 第二层兜底：把前面三种之外的异常也一起吞掉，保证 plan() 永远返回列表、永不向外抛。
        # [易错] except Exception 的代价：真正的 bug（比如变量名写错导致的 NameError）也会被降级成
        #        一句「未知错误」+ 空计划，排查时容易迷路。
        except Exception as e:
            print(f"❌ 解析计划时发生未知错误: {e}")
            return []

# [作用] 执行器：拿着 Planner 给的步骤列表，一步一步地问模型。
# [概念] 概念①：同样是**普通类**（父类 object），和 Planner 之间没有任何继承或调用关系 ——
#        两个零件互不认识，只通过「List[str] 计划」这一种数据格式对接。
class Executor:
    """执行器 - 负责按计划逐步执行"""

    def __init__(self, llm_client: HelloAgentsLLM, prompt_template: Optional[str] = None):
        self.llm_client = llm_client
        self.prompt_template = prompt_template if prompt_template else DEFAULT_EXECUTOR_PROMPT

    # [作用] 按顺序跑完计划里的每一步，把**最后一步**的结果当作最终答案返回。
    # [参数] question 原问题（每一步都要重发一遍）；plan 是 Planner 给的 List[str]；**kwargs 转发给 LLM。
    # [返回] str。plan 为空列表时它一次 LLM 都不调，直接返回空字符串 ""。
    def execute(self, question: str, plan: List[str], **kwargs) -> str:
        """
        按计划执行任务

        Args:
            question: 原始问题
            plan: 执行计划
            **kwargs: LLM调用参数

        Returns:
            最终答案
        """
        # [概念] 两个累加器：history 攒「已经做过什么」，final_answer 存「最后一步的答案」。
        # [易错] final_answer 初值是 ""，所以「计划为空」时 execute 会安静地返回空串而不是报错 ——
        #        好在 PlanAndSolveAgent.run 里的卫语句 if not plan 挡住了这种情况。
        history = ""
        final_answer = ""

        print("\n--- 正在执行计划 ---")
        # [语法] enumerate(可迭代对象, start)：边遍历边给出「序号, 元素」两个值。
        #        第二个参数 1 表示序号从 1 开始（默认是 0），所以下面打印的是「步骤 1/3」而不是「步骤 0/3」。
        for i, step in enumerate(plan, 1):
            print(f"\n-> 正在执行步骤 {i}/{len(plan)}: {step}")
            # [概念] 概念⑤（插值即协议）：四个占位符在这里被填满，模型看到的就是一段完整的自然语言。
            # [易错] `plan=plan` 传进去的是**列表对象**，str.format 会对它调 str() → 渲染成 Python 的 repr。
            # [实测] "{plan}".format(plan=["A","B"]) 得到 "['A', 'B']"（带方括号和引号）——
            #        模型是「忍着看」这种带引号的文本的，这正是文本协议将就的地方。
            prompt = self.prompt_template.format(
                question=question,
                plan=plan,
                # [语法] 又是三元 + 假值判断：第一步时 history 还是空串，会被换成 "无"，
                #        免得提示词里出现一段空的「历史步骤与结果:」（空段落容易让模型以为自己漏看了内容）。
                history=history if history else "无",
                current_step=step
            )
            messages = [{"role": "user", "content": prompt}]

            response_text = self.llm_client.invoke(messages, **kwargs) or ""

            # [概念] 历史是**只增不减**的字符串，每步都在后面追加「步骤 + 结果」。
            # [为什么] 不能省：模型自己没有记忆，第 3 步要理解上下文，就必须把前两步的结果再发一遍。
            # [易错] 代价是 token 随步数近似平方增长 —— 3 步要发 1+2+3=6 份内容，10 步就是 55 份。
            #        本文件没有任何截断或压缩机制，步数一多就会又慢又贵。
            history += f"步骤 {i}: {step}\n结果: {response_text}\n\n"
            # [易错] 每循环一次就覆盖一次：循环跑完时 final_answer 里只剩**最后一步**的答案，
            #        中间步骤的结果只活在 history 那个字符串里，没有被结构化保存下来。
            final_answer = response_text
            print(f"✅ 步骤 {i} 已完成，结果: {final_answer}")

        return final_answer

# [作用] 编排者：先调 planner.plan()，再调 executor.execute()，自己不写一行解析逻辑。
# [概念] 概念①的对照面：这个类**才是** Agent 的子类，但它没有把规划和执行的逻辑塞进 run，
#        而是组装两个现成的零件 ——「继承」负责复用框架能力，「组合」负责拼装业务能力，两者分工不同。
class PlanAndSolveAgent(Agent):
    """
    Plan and Solve Agent - 分解规划与逐步执行的智能体
    
    这个Agent能够：
    1. 将复杂问题分解为简单步骤
    2. 按照计划逐步执行
    3. 维护执行历史和上下文
    4. 得出最终答案
    
    特别适合多步骤推理、数学问题、复杂分析等任务。
    """
    
    # [作用] 构造三步走：① 交给父类初始化 ② 决定用哪套提示词 ③ 造出两个零件挂在自己身上。
    # [参数] 前四个和父类 Agent 完全一致；custom_prompts 是本类独有的可选参数。
    # [语法] `Optional[X] = None` 要拆成两半理解：Optional 只声明「类型允许是 None」，
    #        真正让调用方可以省略不传的是 `= None` 这个默认值 —— 只写注解不写默认值，照样是必填参数。
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        custom_prompts: Optional[Dict[str, str]] = None
    ):
        """
        初始化PlanAndSolveAgent

        Args:
            name: Agent名称
            llm: LLM实例
            system_prompt: 系统提示词
            config: 配置对象
            custom_prompts: 自定义提示词模板 {"planner": "", "executor": ""}
        """
        # [机制] 必须先调父类：self.name / self.llm / self.system_prompt / self.config 都是父类建的，
        #        不调这一句，下面构造 self.planner 时用的 self.llm 就会 AttributeError（self.llm 来自父类）。
        super().__init__(name, llm, system_prompt, config)

        # [概念] 概念③（默认值兜底）在这里的形态：用户给了就用用户的，没给就往下交给零件类再兜一次。
        # [概念] 「层层兜底」是配置类代码的常见结构 —— 每一层只判断自己能不能定，定不了就把 None 传下去。
        # 设置提示词模板：用户自定义优先，否则使用默认模板
        # [易错] 判的是真假而不是判 None：custom_prompts=None 和 custom_prompts={} **走同一条路**（都进 else）——
        #        传个空字典进来，效果和完全不传一模一样。
        if custom_prompts:
            # [语法] dict.get(key) 取不到时返回 None（而不是像 d[key] 那样抛 KeyError），这是刻意选的：
            #        拿不到就交给下一层兜底。
            # [易错] 两个键是**各自独立**兜底的：只传 {"planner": "..."} 时 executor_prompt 是 None，
            #        于是规划用你的、执行用默认（实测如此）—— 混搭是被允许的。
            planner_prompt = custom_prompts.get("planner")
            executor_prompt = custom_prompts.get("executor")
        else:
            planner_prompt = None
            executor_prompt = None

        # [概念] 概念①（组合）的落地点：Agent **持有**一个 Planner 实例，而不是继承它。
        # [机制] 组合的机制就是「属性存引用」：self.planner 指向一个独立对象，它可以被单独构造、单独测试，
        #        也可以整只换掉 —— 鸭子类型：只要它有 plan() 方法，Agent 根本不在乎它是什么类。
        # [实测] 传进去的 self.llm 是**同一个对象**：planner.llm_client is agent.llm → True（共享，不是复制）。
        self.planner = Planner(self.llm, planner_prompt)
        # [概念] 同理：Executor 也只接收「一个 llm + 一套提示词」，照样不认识 Agent 是谁。
        #        三个类的关系是「编排者持有两个零件」，而不是「零件是编排者的变种」。
        self.executor = Executor(self.llm, executor_prompt)
    
    # [作用] 唯一的编排点，只有十几行：拿计划 → 有就执行、没有就早退 → 存历史。
    # [返回] str。对比 SimpleAgent.run（几十行的工具调用循环），这里几乎不含业务逻辑。
    # [副作用] 无论走哪条分支，都会往 self._history 里写两条消息（user + assistant）。
    def run(self, input_text: str, **kwargs) -> str:
        """
        运行Plan and Solve Agent
        
        Args:
            input_text: 要解决的问题
            **kwargs: 其他参数
            
        Returns:
            最终答案
        """
        print(f"\n🤖 {self.name} 开始处理问题: {input_text}")
        
        # [机制] `**kwargs` 原样透传：run 收到的 temperature 等参数穿过 Agent 直达 Planner.plan，
        #        最后进到 llm.invoke —— 中间两跳都不需要知道具体有哪些参数。
        # [概念] 这一行也是组合的收益：Agent 完全不关心计划是怎么解析出来的，
        #        想换一种规划策略（比如先检索再规划），只要换掉 self.planner 这个零件，run 一个字都不用改。
        # 1. 生成计划
        plan = self.planner.plan(input_text, **kwargs)
        # [概念] 概念④（卫语句 guard clause）：先在最前面把「不正常的情况」挡掉并直接返回，
        #        剩下的主干代码就不用再套一层缩进 —— 这是替代深层 if/else 的常用手法。
        # [易错] `not plan` 判的是真假，空列表 [] 为假，所以「解析失败」和「模型真的给了个空计划」
        #        在这里被合并成同一种情况处理（对调用方来说确实没区别，都是「没有计划」）。
        if not plan:
            final_answer = "无法生成有效的行动计划，任务终止。"
            print(f"\n--- 任务终止 ---\n{final_answer}")
            
            # [概念] 早退也必须**收尾**：历史记录要么两条都写、要么两条都不写，
            #        否则下一轮对话会看到「有问无答」的残缺记录。
            # [易错] 报告点：这两行和末尾主干分支里的那两行**完全重复** —— 两条分支各写了一遍，
            #        以后想给历史加个字段（比如时间戳）就得记得改两处，漏一处就会出现行为不一致。
            # 保存到历史记录
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(final_answer, "assistant"))
            
            return final_answer
        
        # 2. 执行计划
        # [概念] 概念④的另一半：主干路径。`plan` 这个列表被原样交给 Executor ——
        #        它是两个零件之间**唯一的数据协议**（List[str]），也正是组合能成立的原因。
        final_answer = self.executor.execute(input_text, plan, **kwargs)
        print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")
        
        # [易错] 和上面早退分支那两行一字不差（重复代码）。
        # [概念] 注意存进去的是 Message(input_text, "user")：Message 的字段顺序是 (content, role)，
        #        这里存的是**原始问题**，而不是发给模型的完整提示词 —— 历史记的是对话，不是提示词。
        # 保存到历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))
        
        return final_answer
