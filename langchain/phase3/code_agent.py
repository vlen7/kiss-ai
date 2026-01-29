"""
第三阶段综合项目：代码执行 Agent

功能：
1. Python 代码执行
2. 代码分析和解释
3. 多工具协作（计算+搜索+代码）
4. 安全的代码沙箱
5. 交互式编程助手
"""

import os
import sys
import io
import ast
import traceback
from typing import List, Dict, Any, Optional
from contextlib import redirect_stdout, redirect_stderr
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from pydantic import BaseModel, Field

load_dotenv()


def get_model() -> ChatOpenAI:
    """获取 LLM 模型"""
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
    )


# ==================== 代码执行工具 ====================

class CodeInput(BaseModel):
    """代码执行参数"""
    code: str = Field(description="要执行的 Python 代码")


# 安全的内置函数白名单
SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bin": bin,
    "bool": bool,
    "chr": chr,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "hash": hash,
    "hex": hex,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
}

# 允许导入的模块白名单
ALLOWED_MODULES = {
    "math",
    "random",
    "datetime",
    "json",
    "re",
    "collections",
    "itertools",
    "functools",
    "statistics",
}


def safe_import(name: str, *args, **kwargs):
    """安全的导入函数"""
    if name in ALLOWED_MODULES:
        return __import__(name, *args, **kwargs)
    raise ImportError(f"不允许导入模块：{name}")


@tool(args_schema=CodeInput)
def execute_python(code: str) -> str:
    """
    在安全沙箱中执行 Python 代码。

    支持的功能：
    - 基础数据类型和运算
    - 数学运算（math 模块）
    - 随机数生成（random 模块）
    - 日期时间处理（datetime 模块）
    - JSON 处理（json 模块）
    - 正则表达式（re 模块）

    Args:
        code: 要执行的 Python 代码

    Returns:
        代码执行结果或错误信息
    """
    # 捕获输出
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    # 准备安全的执行环境
    safe_globals = {
        "__builtins__": SAFE_BUILTINS,
        "__import__": safe_import,
    }

    # 预导入允许的模块
    for module_name in ALLOWED_MODULES:
        try:
            safe_globals[module_name] = __import__(module_name)
        except ImportError:
            pass

    safe_locals = {}

    try:
        # 语法检查
        ast.parse(code)

        # 执行代码
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, safe_globals, safe_locals)

        # 获取输出
        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()

        result_parts = []

        if stdout_output:
            result_parts.append(f"输出：\n{stdout_output}")

        if stderr_output:
            result_parts.append(f"错误输出：\n{stderr_output}")

        # 检查是否有返回值（最后一个表达式）
        if not result_parts:
            # 尝试获取最后一个表达式的值
            try:
                tree = ast.parse(code)
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    last_expr = compile(ast.Expression(tree.body[-1].value), "<string>", "eval")
                    result = eval(last_expr, safe_globals, safe_locals)
                    if result is not None:
                        result_parts.append(f"结果：{result}")
            except Exception:
                pass

        if not result_parts:
            result_parts.append("代码执行成功（无输出）")

        return "\n".join(result_parts)

    except SyntaxError as e:
        return f"语法错误：{e}"
    except Exception as e:
        return f"执行错误：{type(e).__name__}: {e}"


@tool
def analyze_code(code: str) -> str:
    """
    分析 Python 代码，提供代码解释和潜在问题。

    Args:
        code: 要分析的 Python 代码

    Returns:
        代码分析结果
    """
    analysis = []

    try:
        tree = ast.parse(code)

        # 统计代码结构
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)

        analysis.append("代码结构分析：")
        analysis.append(f"  - 函数定义：{functions if functions else '无'}")
        analysis.append(f"  - 类定义：{classes if classes else '无'}")
        analysis.append(f"  - 导入模块：{imports if imports else '无'}")
        analysis.append(f"  - 代码行数：{len(code.splitlines())}")

        # 简单的代码质量检查
        issues = []
        if len(code.splitlines()) > 50:
            issues.append("代码较长，考虑拆分为多个函数")
        if any(len(line) > 100 for line in code.splitlines()):
            issues.append("存在过长的行，建议换行")

        if issues:
            analysis.append("\n潜在问题：")
            for issue in issues:
                analysis.append(f"  - {issue}")
        else:
            analysis.append("\n代码质量：良好")

    except SyntaxError as e:
        analysis.append(f"语法错误：{e}")

    return "\n".join(analysis)


@tool
def explain_code(code: str) -> str:
    """
    解释代码的功能和工作原理。

    Args:
        code: 要解释的代码

    Returns:
        代码解释
    """
    model = get_model()

    prompt = f"""请解释以下 Python 代码的功能和工作原理：

```python
{code}
```

请包含：
1. 代码的整体功能
2. 关键步骤的解释
3. 使用的主要技术或算法
"""

    response = model.invoke(prompt)
    return response.content


# ==================== 辅助工具 ====================

@tool
def calculate_expression(expression: str) -> str:
    """
    计算数学表达式。

    Args:
        expression: 数学表达式，如 "2**10" 或 "sum(range(100))"

    Returns:
        计算结果
    """
    import math

    safe_dict = {
        **SAFE_BUILTINS,
        "math": math,
        "pi": math.pi,
        "e": math.e,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "factorial": math.factorial,
    }

    try:
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"


@tool
def generate_code(task_description: str) -> str:
    """
    根据任务描述生成 Python 代码。

    Args:
        task_description: 任务描述

    Returns:
        生成的代码
    """
    model = get_model()

    prompt = f"""请根据以下任务描述生成 Python 代码：

任务：{task_description}

要求：
1. 代码要简洁、高效
2. 添加必要的注释
3. 只输出代码，不要其他解释
"""

    response = model.invoke(prompt)

    # 提取代码块
    content = response.content
    if "```python" in content:
        code = content.split("```python")[1].split("```")[0].strip()
    elif "```" in content:
        code = content.split("```")[1].split("```")[0].strip()
    else:
        code = content.strip()

    return code


# ==================== 代码 Agent ====================

def create_code_agent() -> AgentExecutor:
    """创建代码执行 Agent"""
    model = get_model()

    tools = [
        execute_python,
        analyze_code,
        explain_code,
        calculate_expression,
        generate_code,
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的 Python 编程助手。你可以：

1. **执行代码**：使用 execute_python 工具运行 Python 代码
2. **分析代码**：使用 analyze_code 工具分析代码结构和质量
3. **解释代码**：使用 explain_code 工具解释代码功能
4. **计算表达式**：使用 calculate_expression 工具计算数学表达式
5. **生成代码**：使用 generate_code 工具根据描述生成代码

工作流程：
- 如果用户要求执行代码，直接使用 execute_python
- 如果用户问代码是什么意思，使用 explain_code
- 如果用户要求写代码，先用 generate_code 生成，然后可以用 execute_python 测试
- 复杂任务可以组合使用多个工具

注意事项：
- 代码在安全沙箱中执行，只能使用基础模块
- 不能访问文件系统或网络
- 执行有超时限制"""),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(model, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
    )


class CodeAssistant:
    """代码助手类"""

    def __init__(self):
        self.agent = create_code_agent()
        self.chat_history: List = []

    def chat(self, user_input: str) -> str:
        """对话"""
        result = self.agent.invoke({
            "input": user_input,
            "chat_history": self.chat_history
        })

        self.chat_history.append(HumanMessage(content=user_input))
        self.chat_history.append(AIMessage(content=result["output"]))

        return result["output"]

    def clear_history(self):
        """清空历史"""
        self.chat_history = []


# ==================== 示例演示 ====================

def basic_execution_example():
    """基础代码执行示例"""
    print("【1. 基础代码执行】")

    # 简单计算
    code1 = "print(2 ** 10)"
    print(f"\n代码：{code1}")
    print(f"结果：{execute_python.invoke({'code': code1})}")

    # 使用模块
    code2 = """
import math
radius = 5
area = math.pi * radius ** 2
print(f"半径为 {radius} 的圆面积是 {area:.2f}")
"""
    print(f"\n代码：{code2}")
    print(f"结果：{execute_python.invoke({'code': code2})}")

    # 定义函数
    code3 = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

for i in range(10):
    print(fibonacci(i), end=' ')
"""
    print(f"\n代码：{code3}")
    print(f"结果：{execute_python.invoke({'code': code3})}")
    print()


def code_analysis_example():
    """代码分析示例"""
    print("【2. 代码分析】")

    code = """
import math
from collections import defaultdict

class Calculator:
    def __init__(self):
        self.history = []

    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

    def multiply(self, a, b):
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result

calc = Calculator()
print(calc.add(5, 3))
print(calc.multiply(4, 7))
"""

    print(f"代码：\n{code}")
    print(f"\n分析结果：\n{analyze_code.invoke({'code': code})}")
    print()


def agent_example():
    """Agent 示例"""
    print("【3. 代码 Agent】")

    agent = create_code_agent()

    # 测试执行代码
    questions = [
        "帮我计算 1 到 100 的所有偶数之和",
        "写一个函数判断一个数是否是质数，然后找出 100 以内的所有质数",
        "解释一下什么是列表推导式，并给出例子",
    ]

    for question in questions:
        print(f"\n问题：{question}")
        print("-" * 40)
        result = agent.invoke({"input": question})
        print(f"\n回答：{result['output']}")

    print()


def interactive_code_assistant():
    """交互式代码助手"""
    print("\n" + "=" * 50)
    print("Python 代码助手")
    print("=" * 50)
    print("""
我可以帮你：
- 执行 Python 代码
- 解释代码功能
- 生成代码
- 分析代码质量

命令：
- 直接输入问题或代码
- 'run: <代码>' 直接执行代码
- 'clear' 清空对话历史
- 'quit' 退出
""")

    assistant = CodeAssistant()

    while True:
        user_input = input("\n你：").strip()

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("再见！")
            break

        if user_input.lower() == "clear":
            assistant.clear_history()
            print("对话历史已清空。")
            continue

        # 直接执行代码
        if user_input.lower().startswith("run:"):
            code = user_input[4:].strip()
            print(f"\n执行结果：\n{execute_python.invoke({'code': code})}")
            continue

        try:
            response = assistant.chat(user_input)
            print(f"\n助手：{response}")
        except Exception as e:
            print(f"\n出错了：{e}")


# ==================== 演示所有功能 ====================

def demo_all():
    """演示所有功能"""
    print("=" * 60)
    print("代码执行 Agent - 功能演示")
    print("=" * 60)
    print()

    basic_execution_example()
    code_analysis_example()
    agent_example()

    print("=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "-i":
        interactive_code_assistant()
    else:
        demo_all()
