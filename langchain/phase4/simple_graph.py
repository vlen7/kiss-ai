"""
第四阶段：LangGraph 基础状态图

学习要点：
1. StateGraph - 状态图定义
2. TypedDict - 状态类型定义
3. Nodes - 节点函数
4. Edges - 边的连接
5. 编译和运行图
"""

import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()


def get_model() -> ChatOpenAI:
    """获取 LLM 模型"""
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
    )


# ==================== 1. 最简单的状态图 ====================

class SimpleState(TypedDict):
    """简单状态：只包含一个消息"""
    message: str


def greet_node(state: SimpleState) -> SimpleState:
    """问候节点：添加问候语"""
    return {"message": f"你好！{state['message']}"}


def process_node(state: SimpleState) -> SimpleState:
    """处理节点：处理消息"""
    return {"message": f"{state['message']} - 已处理"}


def simple_graph_example():
    """最简单的状态图示例"""
    print("【1. 最简单的状态图】")

    # 创建状态图
    graph = StateGraph(SimpleState)

    # 添加节点
    graph.add_node("greet", greet_node)
    graph.add_node("process", process_node)

    # 添加边
    graph.add_edge(START, "greet")
    graph.add_edge("greet", "process")
    graph.add_edge("process", END)

    # 编译图
    app = graph.compile()

    # 运行图
    result = app.invoke({"message": "欢迎使用 LangGraph"})
    print(f"结果：{result['message']}")
    print()


# ==================== 2. 带消息列表的状态图 ====================

class ChatState(TypedDict):
    """聊天状态：使用 add_messages 自动合并消息"""
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState) -> ChatState:
    """聊天节点：调用 LLM 生成回复"""
    model = get_model()
    response = model.invoke(state["messages"])
    return {"messages": [response]}


def chat_graph_example():
    """带消息列表的状态图示例"""
    print("【2. 聊天状态图】")

    # 创建状态图
    graph = StateGraph(ChatState)

    # 添加聊天节点
    graph.add_node("chat", chat_node)

    # 添加边
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)

    # 编译图
    app = graph.compile()

    # 运行图
    result = app.invoke({
        "messages": [HumanMessage(content="用一句话介绍 LangGraph")]
    })

    print(f"用户：用一句话介绍 LangGraph")
    print(f"助手：{result['messages'][-1].content}")
    print()


# ==================== 3. 多节点流水线 ====================

class PipelineState(TypedDict):
    """流水线状态"""
    input_text: str
    translated: str
    summarized: str
    final_output: str


def translate_node(state: PipelineState) -> PipelineState:
    """翻译节点：将文本翻译成英文"""
    model = get_model()
    messages = [
        SystemMessage(content="你是一个翻译专家，将输入的中文翻译成英文。只输出翻译结果，不要解释。"),
        HumanMessage(content=state["input_text"])
    ]
    response = model.invoke(messages)
    return {"translated": response.content}


def summarize_node(state: PipelineState) -> PipelineState:
    """摘要节点：对翻译结果进行摘要"""
    model = get_model()
    messages = [
        SystemMessage(content="你是一个摘要专家，用一句话总结以下内容。"),
        HumanMessage(content=state["translated"])
    ]
    response = model.invoke(messages)
    return {"summarized": response.content}


def format_node(state: PipelineState) -> PipelineState:
    """格式化节点：整理最终输出"""
    final = f"""
原文：{state['input_text']}
翻译：{state['translated']}
摘要：{state['summarized']}
""".strip()
    return {"final_output": final}


def pipeline_graph_example():
    """多节点流水线示例"""
    print("【3. 多节点流水线】")

    # 创建状态图
    graph = StateGraph(PipelineState)

    # 添加节点
    graph.add_node("translate", translate_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("format", format_node)

    # 添加边（顺序执行）
    graph.add_edge(START, "translate")
    graph.add_edge("translate", "summarize")
    graph.add_edge("summarize", "format")
    graph.add_edge("format", END)

    # 编译图
    app = graph.compile()

    # 运行图
    input_text = "人工智能正在改变我们的生活方式，从智能家居到自动驾驶，AI 技术无处不在。"
    result = app.invoke({
        "input_text": input_text,
        "translated": "",
        "summarized": "",
        "final_output": ""
    })

    print(result["final_output"])
    print()


# ==================== 4. 带工具的状态图 ====================

class ToolState(TypedDict):
    """工具状态"""
    messages: Annotated[list[BaseMessage], add_messages]
    tool_called: bool


def should_use_tool(state: ToolState) -> ToolState:
    """判断是否需要使用工具"""
    last_message = state["messages"][-1].content if state["messages"] else ""

    # 简单判断：如果包含"计算"就需要工具
    needs_tool = "计算" in last_message or "几点" in last_message
    return {"tool_called": needs_tool}


def tool_node(state: ToolState) -> ToolState:
    """工具节点：模拟工具调用"""
    last_message = state["messages"][-1].content

    if "计算" in last_message:
        result = "计算结果：42"
    elif "几点" in last_message:
        from datetime import datetime
        result = f"当前时间：{datetime.now().strftime('%H:%M:%S')}"
    else:
        result = "工具调用完成"

    return {"messages": [AIMessage(content=result)]}


def llm_node(state: ToolState) -> ToolState:
    """LLM 节点：直接使用 LLM 回答"""
    model = get_model()
    response = model.invoke(state["messages"])
    return {"messages": [response]}


def tool_graph_example():
    """带工具的状态图示例"""
    print("【4. 带工具判断的状态图】")

    # 创建状态图
    graph = StateGraph(ToolState)

    # 添加节点
    graph.add_node("check", should_use_tool)
    graph.add_node("tool", tool_node)
    graph.add_node("llm", llm_node)

    # 添加边
    graph.add_edge(START, "check")

    # 条件边：根据是否需要工具决定下一步
    def route_after_check(state: ToolState) -> str:
        if state.get("tool_called"):
            return "tool"
        return "llm"

    graph.add_conditional_edges("check", route_after_check, ["tool", "llm"])
    graph.add_edge("tool", END)
    graph.add_edge("llm", END)

    # 编译图
    app = graph.compile()

    # 测试不同输入
    test_inputs = [
        "现在几点了？",
        "你好，介绍一下你自己",
    ]

    for input_text in test_inputs:
        result = app.invoke({
            "messages": [HumanMessage(content=input_text)],
            "tool_called": False
        })
        print(f"输入：{input_text}")
        print(f"输出：{result['messages'][-1].content}")
        print()


# ==================== 5. 循环状态图 ====================

class LoopState(TypedDict):
    """循环状态"""
    counter: int
    max_count: int
    results: list[str]


def increment_node(state: LoopState) -> LoopState:
    """递增节点：增加计数器"""
    new_counter = state["counter"] + 1
    new_results = state["results"] + [f"第 {new_counter} 次迭代"]
    return {"counter": new_counter, "results": new_results}


def should_continue(state: LoopState) -> str:
    """判断是否继续循环"""
    if state["counter"] < state["max_count"]:
        return "continue"
    return "end"


def loop_graph_example():
    """循环状态图示例"""
    print("【5. 循环状态图】")

    # 创建状态图
    graph = StateGraph(LoopState)

    # 添加节点
    graph.add_node("increment", increment_node)

    # 添加边
    graph.add_edge(START, "increment")

    # 条件边：决定是继续循环还是结束
    graph.add_conditional_edges(
        "increment",
        should_continue,
        {
            "continue": "increment",  # 继续循环
            "end": END  # 结束
        }
    )

    # 编译图
    app = graph.compile()

    # 运行图
    result = app.invoke({
        "counter": 0,
        "max_count": 5,
        "results": []
    })

    print(f"最终计数：{result['counter']}")
    print(f"迭代记录：{result['results']}")
    print()


# ==================== 6. 可视化图结构 ====================

def visualize_graph_example():
    """图结构可视化示例"""
    print("【6. 图结构可视化】")

    # 创建一个示例图
    graph = StateGraph(SimpleState)
    graph.add_node("start_node", greet_node)
    graph.add_node("process_node", process_node)
    graph.add_edge(START, "start_node")
    graph.add_edge("start_node", "process_node")
    graph.add_edge("process_node", END)

    app = graph.compile()

    # 打印图结构（ASCII）
    print("图结构：")
    try:
        print(app.get_graph().draw_ascii())
    except Exception:
        # 如果 ASCII 绘制不可用，打印节点和边
        print("  START -> start_node -> process_node -> END")

    print()


# ==================== 演示所有功能 ====================

def demo_all():
    """演示所有状态图功能"""
    print("=" * 60)
    print("LangGraph 基础状态图 - 功能演示")
    print("=" * 60)
    print()

    simple_graph_example()
    chat_graph_example()
    pipeline_graph_example()
    tool_graph_example()
    loop_graph_example()
    visualize_graph_example()

    print("=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    demo_all()
