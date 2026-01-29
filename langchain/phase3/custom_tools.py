"""
第三阶段：自定义工具开发

学习要点：
1. @tool 装饰器定义工具
2. StructuredTool 创建结构化工具
3. 工具参数验证（Pydantic）
4. 异步工具
5. 工具错误处理
"""

import os
import math
import httpx
from datetime import datetime
from typing import Optional, List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, StructuredTool
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()


def get_model() -> ChatOpenAI:
    """获取 LLM 模型"""
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
    )


# ==================== 基础工具定义 ====================

@tool
def get_current_time() -> str:
    """获取当前时间，返回格式化的时间字符串"""
    now = datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M:%S")


@tool
def calculate(expression: str) -> str:
    """
    计算数学表达式的结果。

    Args:
        expression: 数学表达式，如 "2 + 3 * 4" 或 "sqrt(16)"

    Returns:
        计算结果的字符串表示
    """
    # 安全的数学函数
    safe_dict = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "pi": math.pi,
        "e": math.e,
    }

    try:
        # 只允许安全的数学运算
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算错误：{str(e)}"


@tool
def string_length(text: str) -> int:
    """
    计算字符串的长度。

    Args:
        text: 要计算长度的字符串

    Returns:
        字符串的字符数
    """
    return len(text)


def basic_tools_example():
    """基础工具示例"""
    print("【1. 基础工具定义与调用】")

    # 查看工具信息
    print(f"工具名称：{get_current_time.name}")
    print(f"工具描述：{get_current_time.description}")
    print()

    # 直接调用工具
    print("直接调用工具：")
    print(f"  当前时间：{get_current_time.invoke({})}")
    print(f"  计算 2+3*4：{calculate.invoke({'expression': '2+3*4'})}")
    print(f"  字符串长度：{string_length.invoke({'text': 'Hello, LangChain!'})}")
    print()


# ==================== 带参数验证的工具 ====================

class SearchInput(BaseModel):
    """搜索输入参数"""
    query: str = Field(description="搜索关键词")
    max_results: int = Field(default=5, description="最大结果数量", ge=1, le=20)


@tool(args_schema=SearchInput)
def search_knowledge(query: str, max_results: int = 5) -> str:
    """
    在知识库中搜索相关信息。

    Args:
        query: 搜索关键词
        max_results: 返回的最大结果数量

    Returns:
        搜索结果
    """
    # 模拟搜索结果
    mock_results = [
        f"结果 {i+1}: 关于 '{query}' 的相关信息..." for i in range(max_results)
    ]
    return "\n".join(mock_results)


class WeatherInput(BaseModel):
    """天气查询参数"""
    city: str = Field(description="城市名称")
    unit: str = Field(default="celsius", description="温度单位：celsius 或 fahrenheit")


@tool(args_schema=WeatherInput)
def get_weather(city: str, unit: str = "celsius") -> str:
    """
    获取指定城市的天气信息。

    Args:
        city: 城市名称
        unit: 温度单位

    Returns:
        天气信息
    """
    # 模拟天气数据
    mock_weather = {
        "北京": {"temp": 22, "condition": "晴"},
        "上海": {"temp": 25, "condition": "多云"},
        "广州": {"temp": 28, "condition": "阴"},
        "深圳": {"temp": 27, "condition": "晴"},
    }

    weather = mock_weather.get(city, {"temp": 20, "condition": "未知"})
    temp = weather["temp"]

    if unit == "fahrenheit":
        temp = temp * 9 / 5 + 32
        unit_str = "°F"
    else:
        unit_str = "°C"

    return f"{city}天气：{weather['condition']}，温度 {temp}{unit_str}"


def validated_tools_example():
    """参数验证工具示例"""
    print("【2. 带参数验证的工具】")

    # 查看参数模式
    print(f"搜索工具参数：{search_knowledge.args}")
    print()

    # 调用工具
    print("调用工具：")
    print(f"  搜索：{search_knowledge.invoke({'query': 'LangChain', 'max_results': 3})}")
    print(f"  天气：{get_weather.invoke({'city': '北京'})}")
    print(f"  天气（华氏）：{get_weather.invoke({'city': '上海', 'unit': 'fahrenheit'})}")
    print()


# ==================== StructuredTool 创建工具 ====================

def multiply(a: int, b: int) -> int:
    """两数相乘"""
    return a * b


multiply_tool = StructuredTool.from_function(
    func=multiply,
    name="multiply",
    description="计算两个整数的乘积",
)


class DivideInput(BaseModel):
    """除法输入"""
    dividend: float = Field(description="被除数")
    divisor: float = Field(description="除数，不能为0")


def divide(dividend: float, divisor: float) -> str:
    """除法运算"""
    if divisor == 0:
        return "错误：除数不能为0"
    return f"{dividend} / {divisor} = {dividend / divisor}"


divide_tool = StructuredTool.from_function(
    func=divide,
    name="divide",
    description="计算两个数的商",
    args_schema=DivideInput,
)


def structured_tool_example():
    """StructuredTool 示例"""
    print("【3. StructuredTool 创建工具】")

    print(f"乘法工具：{multiply_tool.name} - {multiply_tool.description}")
    print(f"除法工具：{divide_tool.name} - {divide_tool.description}")
    print()

    print("调用工具：")
    print(f"  7 * 8 = {multiply_tool.invoke({'a': 7, 'b': 8})}")
    print(f"  {divide_tool.invoke({'dividend': 100, 'divisor': 4})}")
    print(f"  {divide_tool.invoke({'dividend': 10, 'divisor': 0})}")
    print()


# ==================== 异步工具 ====================

@tool
async def async_fetch_url(url: str) -> str:
    """
    异步获取网页内容。

    Args:
        url: 要获取的网页URL

    Returns:
        网页内容摘要
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            content = response.text[:500]
            return f"状态码：{response.status_code}\n内容预览：{content}..."
    except Exception as e:
        return f"获取失败：{str(e)}"


async def async_tool_example():
    """异步工具示例"""
    print("【4. 异步工具】")
    print("异步工具适用于 I/O 密集型操作，如网络请求")
    print()

    # 异步调用
    result = await async_fetch_url.ainvoke({"url": "https://httpbin.org/get"})
    print(f"异步获取结果：{result[:200]}...")
    print()


# ==================== 工具错误处理 ====================

@tool(handle_tool_error=True)
def risky_tool(value: int) -> str:
    """
    一个可能抛出异常的工具。

    Args:
        value: 输入值，必须为正数

    Returns:
        处理结果
    """
    if value < 0:
        raise ValueError("输入值必须为正数")
    if value == 0:
        raise ZeroDivisionError("不能处理零值")
    return f"成功处理值：{value}"


@tool(handle_tool_error="发生了一个错误，请检查输入")
def custom_error_tool(text: str) -> str:
    """
    带自定义错误消息的工具。

    Args:
        text: 输入文本

    Returns:
        处理结果
    """
    if not text:
        raise ValueError("文本不能为空")
    return f"处理文本：{text}"


def error_handling_example():
    """错误处理示例"""
    print("【5. 工具错误处理】")

    print("正常调用：")
    print(f"  {risky_tool.invoke({'value': 10})}")

    print("\n错误调用（handle_tool_error=True）：")
    print(f"  {risky_tool.invoke({'value': -1})}")

    print("\n自定义错误消息：")
    print(f"  {custom_error_tool.invoke({'text': ''})}")
    print()


# ==================== 工具与 LLM 绑定 ====================

def tool_binding_example():
    """工具与 LLM 绑定示例"""
    print("【6. 工具与 LLM 绑定】")

    model = get_model()

    # 定义工具列表
    tools = [get_current_time, calculate, get_weather]

    # 绑定工具到模型
    model_with_tools = model.bind_tools(tools)

    # 调用模型
    messages = [
        {"role": "user", "content": "现在几点了？北京天气怎么样？"}
    ]

    response = model_with_tools.invoke(messages)

    print(f"模型响应：{response.content}")
    print(f"工具调用：{response.tool_calls}")
    print()


# ==================== 完整工具调用流程 ====================

def complete_tool_flow():
    """完整的工具调用流程"""
    print("【7. 完整工具调用流程】")

    model = get_model()
    tools = [calculate, get_current_time, get_weather]
    model_with_tools = model.bind_tools(tools)

    # 用户问题
    user_question = "请帮我计算 sqrt(144) + 25，然后告诉我北京的天气"
    print(f"用户问题：{user_question}")

    # 第一次调用：获取工具调用请求
    messages = [{"role": "user", "content": user_question}]
    response = model_with_tools.invoke(messages)

    print(f"\n模型响应：")
    print(f"  内容：{response.content}")
    print(f"  工具调用数量：{len(response.tool_calls)}")

    # 执行工具调用
    tool_map = {t.name: t for t in tools}
    tool_results = []

    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        print(f"\n执行工具：{tool_name}({tool_args})")

        if tool_name in tool_map:
            result = tool_map[tool_name].invoke(tool_args)
            tool_results.append({
                "tool_call_id": tool_call["id"],
                "name": tool_name,
                "result": result
            })
            print(f"  结果：{result}")

    # 将工具结果返回给模型
    messages.append(response)
    for tr in tool_results:
        messages.append({
            "role": "tool",
            "tool_call_id": tr["tool_call_id"],
            "name": tr["name"],
            "content": str(tr["result"])
        })

    # 第二次调用：生成最终回答
    final_response = model_with_tools.invoke(messages)
    print(f"\n最终回答：{final_response.content}")
    print()


# ==================== 演示所有功能 ====================

def demo_all():
    """演示所有工具功能"""
    print("=" * 60)
    print("自定义工具开发 - 功能演示")
    print("=" * 60)
    print()

    basic_tools_example()
    validated_tools_example()
    structured_tool_example()
    error_handling_example()
    tool_binding_example()
    complete_tool_flow()

    print("=" * 60)
    print("演示完成！")
    print("=" * 60)


async def demo_async():
    """演示异步功能"""
    await async_tool_example()


if __name__ == "__main__":
    import sys
    import asyncio

    if len(sys.argv) > 1 and sys.argv[1] == "--async":
        asyncio.run(demo_async())
    else:
        demo_all()
