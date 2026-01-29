"""
第三阶段：联网搜索 Agent

学习要点：
1. ReAct Agent 模式
2. 搜索工具集成
3. Agent 执行器
4. 思考-行动-观察循环
5. Agent 记忆和上下文
"""

import os
from typing import List, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
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


# ==================== 搜索相关工具 ====================

class SearchQuery(BaseModel):
    """搜索查询参数"""
    query: str = Field(description="搜索关键词")
    num_results: int = Field(default=3, description="返回结果数量")


@tool(args_schema=SearchQuery)
def web_search(query: str, num_results: int = 3) -> str:
    """
    搜索网络获取相关信息。适用于查询实时信息、新闻、百科知识等。

    Args:
        query: 搜索关键词
        num_results: 返回结果数量

    Returns:
        搜索结果摘要
    """
    # 模拟搜索结果（实际项目中可接入 Google/Bing/DuckDuckGo API）
    mock_results = {
        "langchain": [
            "LangChain 是一个用于开发 LLM 应用的框架，支持 Python 和 JavaScript。",
            "LangChain 提供了 Chains、Agents、Memory 等核心组件。",
            "LangChain 官网：https://langchain.com",
        ],
        "python": [
            "Python 是一种解释型、面向对象的编程语言。",
            "Python 3.12 是最新的稳定版本，带来了多项性能改进。",
            "Python 广泛应用于 Web 开发、数据科学、AI 等领域。",
        ],
        "ai": [
            "人工智能(AI)是模拟人类智能的技术。",
            "GPT-4、Claude 是当前最先进的大语言模型。",
            "AI 正在改变各行各业的工作方式。",
        ],
    }

    # 简单的关键词匹配
    results = []
    query_lower = query.lower()
    for key, values in mock_results.items():
        if key in query_lower:
            results.extend(values[:num_results])

    if not results:
        results = [f"关于 '{query}' 的搜索结果：这是一个模拟的搜索结果。"]

    return "\n".join(f"[{i+1}] {r}" for i, r in enumerate(results[:num_results]))


@tool
def get_webpage_content(url: str) -> str:
    """
    获取网页内容。用于深入了解某个链接的详细信息。

    Args:
        url: 网页URL

    Returns:
        网页内容摘要
    """
    # 模拟网页内容获取
    mock_pages = {
        "langchain.com": "LangChain 官方网站，提供文档、教程和 API 参考。",
        "python.org": "Python 官方网站，提供下载、文档和社区资源。",
        "github.com": "GitHub 是全球最大的代码托管平台。",
    }

    for domain, content in mock_pages.items():
        if domain in url:
            return f"网页内容：{content}"

    return f"获取 {url} 的内容：这是模拟的网页内容。"


@tool
def get_current_news(topic: str) -> str:
    """
    获取某个主题的最新新闻。

    Args:
        topic: 新闻主题

    Returns:
        相关新闻列表
    """
    # 模拟新闻
    mock_news = {
        "科技": [
            "OpenAI 发布 GPT-5，性能大幅提升",
            "苹果发布新一代 M4 芯片",
            "特斯拉 FSD 获得中国批准",
        ],
        "ai": [
            "Anthropic 发布 Claude 3.5",
            "Google DeepMind 在 AGI 研究取得突破",
            "AI 监管法案在欧盟通过",
        ],
    }

    for key, news_list in mock_news.items():
        if key.lower() in topic.lower():
            return "\n".join(f"- {news}" for news in news_list)

    return f"关于 {topic} 的最新新闻：暂无相关新闻。"


# ==================== 创建 ReAct Agent ====================

def create_search_agent() -> AgentExecutor:
    """创建搜索 Agent"""
    model = get_model()

    # 定义工具
    tools = [web_search, get_webpage_content, get_current_news]

    # 定义 Agent 提示词
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个智能搜索助手，可以使用以下工具来帮助用户获取信息：

1. web_search: 搜索网络获取相关信息
2. get_webpage_content: 获取特定网页的内容
3. get_current_news: 获取某个主题的最新新闻

请根据用户的问题，合理使用这些工具来获取信息，然后给出完整、准确的回答。

注意：
- 如果需要多次搜索，请分步进行
- 综合多个来源的信息给出回答
- 如果信息不足，诚实告知用户"""),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 创建 Agent
    agent = create_tool_calling_agent(model, tools, prompt)

    # 创建执行器
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # 显示详细执行过程
        max_iterations=5,  # 最大迭代次数
        handle_parsing_errors=True,
    )

    return executor


def basic_agent_example():
    """基础 Agent 示例"""
    print("【1. 基础搜索 Agent】")

    agent = create_search_agent()

    # 简单问题
    question = "什么是 LangChain？"
    print(f"\n问题：{question}")
    print("-" * 40)

    result = agent.invoke({"input": question})
    print(f"\n回答：{result['output']}")
    print()


def multi_step_agent_example():
    """多步骤 Agent 示例"""
    print("【2. 多步骤搜索】")

    agent = create_search_agent()

    # 复杂问题，需要多次搜索
    question = "请搜索 Python 和 AI 的最新动态，并总结它们之间的关系"
    print(f"\n问题：{question}")
    print("-" * 40)

    result = agent.invoke({"input": question})
    print(f"\n回答：{result['output']}")
    print()


# ==================== 带记忆的 Agent ====================

class AgentWithMemory:
    """带记忆的搜索 Agent"""

    def __init__(self):
        self.agent = create_search_agent()
        self.chat_history: List = []

    def chat(self, user_input: str) -> str:
        """带历史记录的对话"""
        result = self.agent.invoke({
            "input": user_input,
            "chat_history": self.chat_history
        })

        # 更新历史记录
        self.chat_history.append(HumanMessage(content=user_input))
        self.chat_history.append(AIMessage(content=result["output"]))

        return result["output"]

    def clear_history(self):
        """清空历史记录"""
        self.chat_history = []


def agent_with_memory_example():
    """带记忆的 Agent 示例"""
    print("【3. 带记忆的 Agent】")

    agent = AgentWithMemory()

    # 多轮对话
    conversations = [
        "搜索一下 LangChain 的信息",
        "它有哪些核心组件？",
        "能详细说说 Agents 吗？",
    ]

    for question in conversations:
        print(f"\n用户：{question}")
        print("-" * 40)
        response = agent.chat(question)
        print(f"\n助手：{response}")

    print()


# ==================== 自定义 Agent 行为 ====================

def create_cautious_agent() -> AgentExecutor:
    """创建谨慎的 Agent（会确认信息来源）"""
    model = get_model()
    tools = [web_search, get_webpage_content, get_current_news]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个谨慎的搜索助手。在回答问题时：

1. 始终标注信息来源
2. 对不确定的信息明确说明
3. 如果搜索结果不足，建议用户提供更多信息
4. 避免给出未经验证的结论

使用工具时要有策略：
- 先用 web_search 获取概览
- 如果需要详细信息，使用 get_webpage_content
- 如果涉及时事，使用 get_current_news"""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(model, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
    )


def cautious_agent_example():
    """谨慎 Agent 示例"""
    print("【4. 谨慎的搜索 Agent】")

    agent = create_cautious_agent()

    question = "AI 技术的最新进展是什么？请提供可靠的信息来源"
    print(f"\n问题：{question}")
    print("-" * 40)

    result = agent.invoke({"input": question})
    print(f"\n回答：{result['output']}")
    print()


# ==================== Agent 错误处理 ====================

@tool
def unreliable_tool(query: str) -> str:
    """
    一个不可靠的工具（模拟可能失败的外部服务）。

    Args:
        query: 查询内容

    Returns:
        查询结果
    """
    import random
    if random.random() < 0.5:
        raise Exception("服务暂时不可用")
    return f"查询 '{query}' 的结果：成功获取数据"


def create_robust_agent() -> AgentExecutor:
    """创建健壮的 Agent（能处理工具失败）"""
    model = get_model()
    tools = [web_search, unreliable_tool]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个健壮的搜索助手。

当工具调用失败时：
1. 尝试使用其他工具获取信息
2. 如果所有方法都失败，诚实告知用户
3. 不要因为一个工具失败就放弃回答

始终保持礼貌和专业。"""),
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


def robust_agent_example():
    """健壮 Agent 示例"""
    print("【5. 健壮的 Agent（错误处理）】")

    agent = create_robust_agent()

    question = "请使用 unreliable_tool 查询 'test'"
    print(f"\n问题：{question}")
    print("-" * 40)

    result = agent.invoke({"input": question})
    print(f"\n回答：{result['output']}")
    print()


# ==================== 交互式 Agent ====================

def interactive_agent():
    """交互式搜索助手"""
    print("\n" + "=" * 50)
    print("智能搜索助手")
    print("=" * 50)
    print("\n我可以帮你搜索网络信息、获取新闻等。")
    print("输入 'quit' 退出，输入 'clear' 清空对话历史。\n")

    agent = AgentWithMemory()

    while True:
        user_input = input("你：").strip()

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("再见！")
            break

        if user_input.lower() == "clear":
            agent.clear_history()
            print("对话历史已清空。\n")
            continue

        print()
        try:
            response = agent.chat(user_input)
            print(f"\n助手：{response}\n")
        except Exception as e:
            print(f"\n出错了：{e}\n")


# ==================== 演示所有功能 ====================

def demo_all():
    """演示所有 Agent 功能"""
    print("=" * 60)
    print("联网搜索 Agent - 功能演示")
    print("=" * 60)
    print()

    basic_agent_example()
    multi_step_agent_example()
    agent_with_memory_example()
    cautious_agent_example()
    robust_agent_example()

    print("=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "-i":
        interactive_agent()
    else:
        demo_all()
