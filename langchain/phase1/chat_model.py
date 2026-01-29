"""
第一阶段：Chat Model 基础对话示例

学习要点：
1. ChatOpenAI 的基本使用
2. 消息类型：HumanMessage, AIMessage, SystemMessage
3. invoke() 方法调用
4. 流式输出 stream()
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()


def basic_chat():
    """基础对话示例"""
    model = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
    )

    # 方式1：直接传入字符串
    response = model.invoke("你好，请介绍一下你自己")
    print("【基础调用】")
    print(response.content)
    print()


def chat_with_messages():
    """使用消息对象进行对话"""
    model = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
    )

    # 方式2：使用消息对象列表
    messages = [
        SystemMessage(content="你是一个专业的Python编程助手，回答要简洁明了。"),
        HumanMessage(content="什么是装饰器？"),
    ]

    response = model.invoke(messages)
    print("【带系统提示的对话】")
    print(response.content)
    print()


def chat_with_history():
    """带历史记录的多轮对话"""
    model = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
    )

    # 模拟多轮对话历史
    messages = [
        SystemMessage(content="你是一个友好的助手。"),
        HumanMessage(content="我叫小明"),
        AIMessage(content="你好小明！很高兴认识你。有什么我可以帮助你的吗？"),
        HumanMessage(content="我叫什么名字？"),
    ]

    response = model.invoke(messages)
    print("【多轮对话（记忆上下文）】")
    print(response.content)
    print()


def streaming_chat():
    """流式输出示例"""
    model = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        streaming=True,
    )

    print("【流式输出】")
    for chunk in model.stream("用一句话解释什么是人工智能"):
        print(chunk.content, end="", flush=True)
    print("\n")


def chat_with_parameters():
    """自定义模型参数"""
    model = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0.7,  # 控制创造性，0-2，越高越随机
        max_tokens=100,   # 最大输出 token 数
    )

    response = model.invoke("写一首关于编程的俳句")
    print("【自定义参数（temperature=0.7）】")
    print(response.content)
    print()


if __name__ == "__main__":
    print("=" * 50)
    print("Chat Model 基础示例")
    print("=" * 50)
    print()

    basic_chat()
    chat_with_messages()
    chat_with_history()
    streaming_chat()
    chat_with_parameters()
