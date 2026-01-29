"""
第一阶段：Prompt Template 提示词模板示例

学习要点：
1. PromptTemplate - 基础字符串模板
2. ChatPromptTemplate - 对话模板
3. MessagesPlaceholder - 动态消息占位符
4. 模板变量和格式化
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()


def get_model():
    """获取模型实例"""
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
    )


def basic_prompt_template():
    """基础 PromptTemplate 示例"""
    # 创建模板
    template = PromptTemplate(
        input_variables=["topic"],
        template="请用简单的语言解释什么是{topic}",
    )

    # 格式化模板
    prompt = template.format(topic="机器学习")
    print("【基础模板】")
    print(f"生成的提示词: {prompt}")

    # 调用模型
    model = get_model()
    response = model.invoke(prompt)
    print(f"回答: {response.content}")
    print()


def prompt_template_from_string():
    """从字符串创建模板"""
    template = PromptTemplate.from_template(
        "你是一个{role}专家，请回答以下问题：{question}"
    )

    prompt = template.format(role="Python", question="如何读取JSON文件？")
    print("【从字符串创建模板】")
    print(f"生成的提示词: {prompt}")

    model = get_model()
    response = model.invoke(prompt)
    print(f"回答: {response.content}")
    print()


def chat_prompt_template():
    """ChatPromptTemplate 对话模板示例"""
    # 创建对话模板
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "你是一个{specialty}领域的专家助手。回答要简洁专业。"),
        ("human", "{question}"),
    ])

    # 格式化为消息列表
    messages = chat_template.format_messages(
        specialty="数据库",
        question="什么是索引？"
    )

    print("【对话模板】")
    print(f"生成的消息: {messages}")

    model = get_model()
    response = model.invoke(messages)
    print(f"回答: {response.content}")
    print()


def chat_template_with_classes():
    """使用类创建对话模板"""
    chat_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "你是一个乐于助人的AI助手，名字叫{name}。"
        ),
        HumanMessagePromptTemplate.from_template(
            "{user_input}"
        ),
    ])

    messages = chat_template.format_messages(
        name="小智",
        user_input="你好，介绍一下你自己"
    )

    print("【使用类创建模板】")
    model = get_model()
    response = model.invoke(messages)
    print(f"回答: {response.content}")
    print()


def template_with_history():
    """带历史记录占位符的模板"""
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "你是一个友好的助手。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    # 模拟历史消息
    history = [
        HumanMessage(content="我喜欢Python"),
        AIMessage(content="Python是一门很棒的语言！它简洁易读，适合各种应用场景。"),
    ]

    messages = chat_template.format_messages(
        history=history,
        input="根据我的喜好，推荐我学习什么框架？"
    )

    print("【带历史记录的模板】")
    model = get_model()
    response = model.invoke(messages)
    print(f"回答: {response.content}")
    print()


def few_shot_template():
    """Few-shot 提示模板（示例学习）"""
    template = PromptTemplate.from_template("""
请将以下文本的情感分类为：积极、消极、中性

示例：
文本：这个产品太棒了！
情感：积极

文本：服务态度很差，不会再来了
情感：消极

文本：今天天气一般
情感：中性

现在请分类：
文本：{text}
情感：""")

    prompt = template.format(text="这家餐厅的菜品味道不错，下次还会来")

    print("【Few-shot 模板】")
    model = get_model()
    response = model.invoke(prompt)
    print(f"分类结果: {response.content}")
    print()


if __name__ == "__main__":
    print("=" * 50)
    print("Prompt Template 提示词模板示例")
    print("=" * 50)
    print()

    basic_prompt_template()
    prompt_template_from_string()
    chat_prompt_template()
    chat_template_with_classes()
    template_with_history()
    few_shot_template()
