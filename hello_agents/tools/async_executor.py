# ===== 文件导读 =====
# [职责] 异步工具执行器：把「同步阻塞的工具」丢进线程池，用 async / await 的写法一次跑多个。
# [位置] tools/ 包里的旁支 —— 全仓库没有任何 Agent 调用它，只在 tools/__init__.py 的导出列表里出现。
#        真正在跑工具的是 registry.execute_tool()，那是同步的。
# [阅读顺序] ① 类头部 + __init__（线程池从哪来）
#            → ② execute_tool_async（本文件里唯一正确的核心）
#            → ③ execute_tools_parallel（重点：它为什么不是并行的）
#            → ④ 四个便捷函数（重点：它们为什么全都用不了）
#            → ⑤ 文件末尾的 __main__ 演示。
# [本文件新概念] ① 协程 ≠ 已执行　② 事件循环与 run_in_executor　③ async with 与同步 with 的协议差别
#                ④ asyncio.run 的适用边界　⑤ 线程池只救 I/O，不救 CPU。
# [实测] 文件末尾自带的演示跑不起来：python -m hello_agents.tools.async_executor
#        在执行到第一处 await 时就抛 TypeError（原因见 run_parallel_tools）。
# ===================
"""异步工具执行器 - HelloAgents异步工具执行支持"""

# [语法] 三个导入各代表一类：asyncio 是标准库的异步框架；concurrent.futures 是标准库的
#        「线程池 / 进程池」抽象；.registry 是本包内的相对导入（指向同目录的 registry.py）。
# [概念] asyncio 和线程池本来是两套调度系统，本文件把它们**拼**在一起用，理由见 execute_tool_async。
import asyncio
import concurrent.futures
from typing import Dict, Any, List
from .registry import ToolRegistry


# [作用] 把「同步的工具调用」包装成「可以 await 的调用」，并提供一个一次跑一批的入口。
# [为什么] Agent 的工具大多是同步阻塞的（读写文件、发 HTTP、查数据库）。
#        协程不能直接等待一个同步函数 —— 直接调用会把整个事件循环卡住。线程池就是那座桥。
# [实测] 但全仓库搜不到任何一处调用它，只有 tools/__init__.py 的导出列表提到它：
#        这是一段「写好了但没接上」的能力。
class AsyncToolExecutor:
    """异步工具执行器"""

    # [作用] 存下注册表，并立刻建一个线程池。
    # [参数] max_workers 是线程池的上限，默认 4 —— 意味着同一时刻最多 4 个工具在真跑，
    #        第 5 个得排队。这个数字是「并发度」，不是「任务数」。
    # [副作用] 构造即建池。线程池不会被垃圾回收自动关掉，
    #        所以不用时必须调 close()（或者用 with 语句），否则工作线程会一直挂到进程退出。
    def __init__(self, registry: ToolRegistry, max_workers: int = 4):
        self.registry = registry
        # [概念] ThreadPoolExecutor = 线程池：它维护一批工作线程，你把任务丢进去，它替你排队执行。
        # [易错] 它和 asyncio 的事件循环是**两套**调度系统 —— 一个管线程，一个管协程。
        #        本文件的做法是「让事件循环去等线程池」，而不是让协程自己干活。
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    # [概念] ★ 但在用线程池之前，得先知道它能买来什么、买不来什么。
    #        CPython 有 GIL（全局解释器锁）：同一时刻只有一个线程在执行 Python 字节码。
    #        所以线程池的收益完全取决于「任务的时间花在哪」：
    #        · 花在等待（网络、磁盘、数据库）→ 线程等待时会释放 GIL → 真能并行
    #        · 花在计算（循环、解析、加密）→ 全都得抢同一把锁 → 只是轮流跑，甚至更慢
    # [实测] 4 个任务，线程池 max_workers=4：
    #        I/O 型（各 sleep 0.3s）：串行 1.21s → 4 线程 0.30s，加速 4.00 倍
    #        CPU 型（各算 300 万次平方和）：串行 1.29s → 4 线程 1.30s，加速 1.00 倍
    # [为什么] 工具系统里的工具大多在等外部服务，所以用线程池是对路的；
    #        真要压榨 CPU 得换 ProcessPoolExecutor（多进程，各自一把 GIL）。
    # [作用] 异步执行单个工具：把同步的 registry.execute_tool 丢进线程池，然后 await 它。
    # [概念] ★ 本文件最重要的一个认知：async def 定义的是「协程函数」。
    #        调用它**不会执行任何代码**，只会返回一个协程对象（coroutine object）。
    #        真正开始跑，要等到有人 await 它，或把它交给 asyncio.gather / create_task。
    # [实测] 这一点和生成器同源：之前实测过 g = llm.think(...) 那一瞬间什么都没发生；
    #        因为生成器和协程都是「可以被挂起、之后再恢复」的函数，只是恢复它们的人不同 ——
    #        生成器靠 for / next()，协程靠事件循环。
    async def execute_tool_async(self, tool_name: str, input_data: str) -> str:
        """异步执行单个工具"""
        # [机制] 拿到「当前正在跑的这个事件循环」—— 后面要靠它把活派给线程池。
        # [易错] asyncio.get_event_loop() 是旧写法：在没有运行中的循环时会直接抛
        #        RuntimeError: There is no current event loop in thread 'MainThread'（实测）。
        #        现在的推荐写法是 asyncio.get_running_loop()，语义更准，也不会顺手给你造一个假循环。
        # [实测] 在协程内部调用它（也就是本文件的场景）在 Python 3.14 上不会告警，是安全的。
        loop = asyncio.get_event_loop()
        
        # [概念] 闭包：这个内层函数用到了外层的 tool_name 和 input_data。
        # [为什么] 要包这一层：run_in_executor 只接受「零参数的可调用对象」，
        #        所以得先把它变成不带参数的 _execute，参数靠闭包带进去。
        def _execute():
            return self.registry.execute_tool(tool_name, input_data)
        
        try:
            # [机制] 这一行就是整座桥：run_in_executor(线程池, 函数) 把 _execute 排进线程池，
            #        立刻返回一个 Future；await 表示「我在这里让出控制权，等它跑完再回来」。
            # [概念] await 的含义不是「开始跑」，而是「挂起当前协程，把控制权还给事件循环」。
            #        正因为挂起了，await 期间别的协程才能继续做事 —— 这才是并发的来源。
            result = await loop.run_in_executor(self.executor, _execute)
            return result
        # [易错] 这个 except 几乎接不到工具自己的报错：registry.execute_tool 内部已经自己 try 过一遍，
        #        出错也只会返回一句「错误：...」字符串，根本不会抛。
        #        所以这里真正能兜住的，是注册表本身出问题（比如传进来的根本不是 ToolRegistry）。
        # [易错] 更麻烦的是它把异常转成了普通字符串：对应的任务会显示 status="success"，
        #        但实际上失败了 —— 下面 for 循环里的 except 分支因此也基本是死代码。
        except Exception as e:
            return f"❌ 工具 '{tool_name}' 异步执行失败: {e}"

    # [作用] 「并行执行多个工具」—— 方法名、docstring、下面的日志，三处都在这么说。
    # [实测] 它其实**不是并行的**。4 个各睡 0.3 秒的任务，实测耗时 1.21 秒（≈ 串行），
    #        改成 asyncio.gather 之后才是 0.30 秒。原因看下面第 51 行和第 56 行的注释。
    # [为什么] 这是本文件最值得读的一段：它把「造协程」和「跑协程」的时机搞混了。
    async def execute_tools_parallel(self, tasks: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        并行执行多个工具
        
        Args:
            tasks: 任务列表，每个任务包含 tool_name 和 input_data
            
        Returns:
            执行结果列表，包含任务信息和结果
        """
        # [易错] 这条日志在撒谎：此刻一个工具都还没开始跑，它只是在往下建协程对象。
        #        按本项目「输出必须自证清白」的标准，这里应当说清「已排队 N 个，尚未开始」。
        print(f"🚀 开始并行执行 {len(tasks)} 个工具任务")
        
        # 创建异步任务
        # [概念] ★ 关键：下面往这个列表里放的，是「协程对象」，不是「正在跑的任务」。
        #        协程对象 = 一份待办说明书；它躺在列表里的时候，一行代码都没执行过。
        #        要让它真的跑起来，得 await 它，或者交给 asyncio.gather / loop.create_task。
        async_tasks = []
        for i, task in enumerate(tasks):
            tool_name = task.get("tool_name")
            input_data = task.get("input_data", "")
            
            if not tool_name:
                continue
                
            print(f"📝 创建任务 {i+1}: {tool_name}")
            # [机制] 这一行只负责「造」协程，不负责「跑」—— 返回的协程对象被塞进列表就完事了。
            # [易错] 想要真并行，正确写法是 asyncio.gather(*[self.execute_tool_async(...) for ...])，
            #        或者先 loop.create_task(...) 让它们立刻开始跑。
            async_task = self.execute_tool_async(tool_name, input_data)
            async_tasks.append((i, task, async_task))
        
        # 等待所有任务完成
        results = []
        # [机制] ★ 并行就是在这里变成串行的：for 循环一个接一个 await，
        #        第 2 个协程必须等第 1 个彻底跑完才会被唤醒 —— 排队了。
        # [概念] 判断一段异步代码到底有没有并发的铁律：**看 await 是不是在循环里逐个等**。
        #        真正的并发是一句 gather 同时等一批。
        # [实测] 4 × 0.3 秒：这段循环 1.21 秒，换成 gather 是 0.30 秒。
        for i, task, async_task in async_tasks:
            try:
                # [机制] await 会挂起当前协程，直到这个工具在线程池里跑完。
                #        因为是逐个 await，线程池虽然开着 4 个线程，每次却只派进去 1 个活。
                result = await async_task
                results.append({
                    "task_id": i,
                    "tool_name": task["tool_name"],
                    "input_data": task["input_data"],
                    "result": result,
                    "status": "success"
                })
                print(f"✅ 任务 {i+1} 完成: {task['tool_name']}")
            except Exception as e:
                results.append({
                    "task_id": i,
                    "tool_name": task["tool_name"],
                    "input_data": task["input_data"],
                    "result": str(e),
                    "status": "error"
                })
                print(f"❌ 任务 {i+1} 失败: {task['tool_name']} - {e}")
        
        # [概念] 这里统计成功的口径是 status 字段，而 status 是按「有没有抛异常」判定的。
        #        结合上面那个会吞异常的 except，这句「成功 N/N」的可信度要打折。
        print(f"🎉 并行执行完成，成功: {sum(1 for r in results if r['status'] == 'success')}/{len(results)}")
        return results

    # [作用] 批量跑同一个工具：把输入列表转成任务列表，然后**原样**走上面那个串行实现。
    # [易错] 所以它继承了同样的「名为并行、实为串行」的问题。
    async def execute_tools_batch(self, tool_name: str, input_list: List[str]) -> List[Dict[str, Any]]:
        """
        批量执行同一个工具
        
        Args:
            tool_name: 工具名称
            input_list: 输入数据列表
            
        Returns:
            执行结果列表
        """
        tasks = [
            {"tool_name": tool_name, "input_data": input_data}
            for input_data in input_list
        ]
        return await self.execute_tools_parallel(tasks)

    # [作用] 关掉线程池。
    # [机制] shutdown(wait=True) = 不再接新任务，并且等已经派进去的任务跑完才返回；
    #        传 wait=False 则是立刻返回、不等收尾。
    def close(self):
        """关闭执行器"""
        self.executor.shutdown(wait=True)
        print("🔒 异步工具执行器已关闭")

    # [概念] 上下文管理器协议：写了 __enter__ / __exit__，对象就能用在 with 语句里 ——
    #        进入时调 __enter__，退出时自动调 __exit__（这里就是 close()）。
    # [易错] ★ 但 async with 用的是**另外两个**方法名：__aenter__ / __aexit__
    #        （多一个 a，async 的缩写）。本类只写了同步的那一对，所以 async with 用不了 ——
    #        这正是下面四个便捷函数全部报废的原因。
    #        实测报错原文：'AsyncToolExecutor' object does not support the asynchronous context
    #        manager protocol (missed __aexit__ method) but it supports the context manager protocol.
    def __enter__(self):
        return self

    # [参数] exc_type / exc_val / exc_tb 是 Python 传给 __exit__ 的三个异常信息。
    #        这里没用上（只是调 close），所以即便 with 块里出了错，也不会被这里吞掉。
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# 便捷函数
# [作用] 便捷函数：建执行器 → 跑 → 自动关闭，三步压缩成一步。
# [实测] 它**用不了**：第一次调用就抛 TypeError（原因见上面 __enter__ 的注释）。
#        连带 run_batch_tool、run_parallel_tools_sync、run_batch_tool_sync 一起废掉。
# [易错] 这类「导出了、能 import、一调就炸」的函数最误导人：
#        from hello_agents.tools import run_parallel_tools 完全成功，问题要到运行时才暴露。
async def run_parallel_tools(registry: ToolRegistry, tasks: List[Dict[str, str]], max_workers: int = 4) -> List[Dict[str, Any]]:
    """
    便捷函数：并行执行多个工具
    
    Args:
        registry: 工具注册表
        tasks: 任务列表
        max_workers: 最大工作线程数
        
    Returns:
        执行结果列表
    """
    # [实测] 就是这一行报的错：async with 找的是 __aenter__ / __aexit__，
    #        而本类只提供了同步的 __enter__ / __exit__。
    # [修法] 两条路 —— 给类补上 __aenter__ / __aexit__，或者这里直接换成普通 with。
    #        后者更省事，因为退出时要做的事（关线程池）根本不是异步操作。
    async with AsyncToolExecutor(registry, max_workers) as executor:
        return await executor.execute_tools_parallel(tasks)


# [作用] 和上面同源，只是先转成任务列表再跑 —— 因此也同样是坏的。
async def run_batch_tool(registry: ToolRegistry, tool_name: str, input_list: List[str], max_workers: int = 4) -> List[Dict[str, Any]]:
    """
    便捷函数：批量执行同一个工具
    
    Args:
        registry: 工具注册表
        tool_name: 工具名称
        input_list: 输入数据列表
        max_workers: 最大工作线程数
        
    Returns:
        执行结果列表
    """
    async with AsyncToolExecutor(registry, max_workers) as executor:
        return await executor.execute_tools_batch(tool_name, input_list)


# 同步包装函数（为了兼容性）
# [作用] 同步包装：让不会写 async 的调用方也能用。
# [机制] asyncio.run(协程) 会新建一个事件循环、跑完、再关掉它 —— 一句话走完全程。
# [易错] ★ 它的硬限制：当前线程里已经有一个在跑的循环时会直接抛
#        RuntimeError: asyncio.run() cannot be called from a running event loop（实测）。
#        所以在 Jupyter 里、或别的协程里调这个函数必然失败。
#        那种环境要改用 await run_parallel_tools(...)，或者上 nest_asyncio 之类的补丁。
def run_parallel_tools_sync(registry: ToolRegistry, tasks: List[Dict[str, str]], max_workers: int = 4) -> List[Dict[str, Any]]:
    """同步版本的并行工具执行"""
    return asyncio.run(run_parallel_tools(registry, tasks, max_workers))


# [作用] 同上，同步版的批量执行。
# [概念] 另外注意它和 run_batch_tool 的重名风险：一个带 _sync 后缀一个不带，
#        只看名字很容易调错那个会抛 RuntimeError 的版本。
def run_batch_tool_sync(registry: ToolRegistry, tool_name: str, input_list: List[str], max_workers: int = 4) -> List[Dict[str, Any]]:
    """同步版本的批量工具执行"""
    return asyncio.run(run_batch_tool(registry, tool_name, input_list, max_workers))


# 示例函数
# [作用] 作者自己给自己写的演示。
# [易错] 这个 registry 是新建的空表，一行工具都没注册（作者的注释写着「这里假设已经注册了工具」），
#        所以就算它能跑起来，4 个任务也只会各返回一句「未找到名为 'my_calculator' 的工具」。
# [实测] 而实际上它连这一步都到不了：第一处 await run_parallel_tools 就抛 TypeError。
async def demo_parallel_execution():
    """演示并行执行的示例"""
    # [易错] 这里又 import 了一次 ToolRegistry，而文件顶部已经导过了 —— 多余的重复导入。
    from .registry import ToolRegistry
    
    # 创建注册表（这里假设已经注册了工具）
    registry = ToolRegistry()
    
    # 定义并行任务
    tasks = [
        {"tool_name": "my_calculator", "input_data": "2 + 2"},
        {"tool_name": "my_calculator", "input_data": "3 * 4"},
        {"tool_name": "my_calculator", "input_data": "sqrt(16)"},
        {"tool_name": "my_calculator", "input_data": "10 / 2"},
    ]
    
    # 并行执行
    results = await run_parallel_tools(registry, tasks)
    
    # 显示结果
    print("\n📊 并行执行结果:")
    for result in results:
        status_icon = "✅" if result["status"] == "success" else "❌"
        print(f"{status_icon} {result['tool_name']}({result['input_data']}) = {result['result']}")
    
    return results


# [语法] __name__ 是模块的内置变量：直接 python xxx.py 运行时它是 "__main__"，
#        被别的模块 import 时它是模块名。所以这句的意思是「只在直接运行时才执行下面的演示」。
# [实测] python -m hello_agents.tools.async_executor → 直接 TypeError。
if __name__ == "__main__":
    # 运行演示
    asyncio.run(demo_parallel_execution())
