import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()


@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"The weather in {city} is sunny, 22°C!"


if __name__ == '__main__':
    # 初始化模型
    model = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
    )

    # 定义工具列表
    tools = [get_weather]

    # 创建 agent
    agent = create_agent(model, tools)

    # 运行 agent
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is the weather in San Francisco?"}]}
    )

    # 打印结果
    for message in result["messages"]:
        print(f"{message.type}: {message.content}")
