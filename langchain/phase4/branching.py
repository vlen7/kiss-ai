"""
第四阶段：LangGraph 条件分支

学习要点：
1. add_conditional_edges - 条件边
2. 路由函数 - 决定下一个节点
3. 多分支流程 - 复杂的条件逻辑
4. 并行分支 - 同时执行多个节点
5. 子图 - 嵌套图结构
"""

import os
from typing import TypedDict, Annotated, Literal
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


# ==================== 1. 基础条件分支 ====================

class IntentState(TypedDict):
    """意图识别状态"""
    user_input: str
    intent: str
    response: str


def classify_intent(state: IntentState) -> IntentState:
    """分类用户意图"""
    model = get_model()
    messages = [
        SystemMessage(content="""分析用户输入的意图，只返回以下类别之一：
- question: 用户在提问
- greeting: 用户在打招呼
- complaint: 用户在抱怨
- other: 其他情况

只返回类别名称，不要其他内容。"""),
        HumanMessage(content=state["user_input"])
    ]
    response = model.invoke(messages)
    intent = response.content.strip().lower()

    # 确保返回有效的意图
    valid_intents = ["question", "greeting", "complaint", "other"]
    if intent not in valid_intents:
        intent = "other"

    return {"intent": intent}


def handle_question(state: IntentState) -> IntentState:
    """处理问题类意图"""
    model = get_model()
    messages = [
        SystemMessage(content="你是一个知识丰富的助手，请回答用户的问题。"),
        HumanMessage(content=state["user_input"])
    ]
    response = model.invoke(messages)
    return {"response": response.content}


def handle_greeting(state: IntentState) -> IntentState:
    """处理问候类意图"""
    return {"response": "你好！很高兴见到你，有什么我可以帮助你的吗？"}


def handle_complaint(state: IntentState) -> IntentState:
    """处理抱怨类意图"""
    return {"response": "非常抱歉给您带来了不好的体验，请告诉我具体问题，我会尽力帮您解决。"}


def handle_other(state: IntentState) -> IntentState:
    """处理其他意图"""
    model = get_model()
    messages = [
        SystemMessage(content="你是一个友好的助手，请适当回应用户。"),
        HumanMessage(content=state["user_input"])
    ]
    response = model.invoke(messages)
    return {"response": response.content}


def route_by_intent(state: IntentState) -> str:
    """根据意图路由到不同处理节点"""
    intent_map = {
        "question": "handle_question",
        "greeting": "handle_greeting",
        "complaint": "handle_complaint",
        "other": "handle_other"
    }
    return intent_map.get(state["intent"], "handle_other")


def basic_branching_example():
    """基础条件分支示例"""
    print("【1. 基础条件分支 - 意图路由】")

    # 创建状态图
    graph = StateGraph(IntentState)

    # 添加节点
    graph.add_node("classify", classify_intent)
    graph.add_node("handle_question", handle_question)
    graph.add_node("handle_greeting", handle_greeting)
    graph.add_node("handle_complaint", handle_complaint)
    graph.add_node("handle_other", handle_other)

    # 添加边
    graph.add_edge(START, "classify")

    # 条件边：根据意图路由
    graph.add_conditional_edges(
        "classify",
        route_by_intent,
        {
            "handle_question": "handle_question",
            "handle_greeting": "handle_greeting",
            "handle_complaint": "handle_complaint",
            "handle_other": "handle_other"
        }
    )

    # 所有处理节点都指向结束
    graph.add_edge("handle_question", END)
    graph.add_edge("handle_greeting", END)
    graph.add_edge("handle_complaint", END)
    graph.add_edge("handle_other", END)

    # 编译图
    app = graph.compile()

    # 测试不同输入
    test_inputs = [
        "你好！",
        "Python 的装饰器是什么？",
        "你们的服务太差了！",
        "今天天气真不错",
    ]

    for user_input in test_inputs:
        result = app.invoke({
            "user_input": user_input,
            "intent": "",
            "response": ""
        })
        print(f"输入：{user_input}")
        print(f"意图：{result['intent']}")
        print(f"回复：{result['response'][:100]}...")
        print()


# ==================== 2. 多级条件分支 ====================

class OrderState(TypedDict):
    """订单处理状态"""
    order_type: str  # new, modify, cancel, query
    priority: str    # high, normal, low
    order_id: str
    result: str


def check_order_type(state: OrderState) -> str:
    """检查订单类型"""
    return state["order_type"]


def check_priority(state: OrderState) -> str:
    """检查优先级"""
    return state["priority"]


def process_new_order(state: OrderState) -> OrderState:
    """处理新订单"""
    return {"result": f"新订单已创建，订单号：{state['order_id']}"}


def process_modify_order(state: OrderState) -> OrderState:
    """处理订单修改"""
    return {"result": f"订单 {state['order_id']} 已修改"}


def process_cancel_order(state: OrderState) -> OrderState:
    """处理订单取消"""
    return {"result": f"订单 {state['order_id']} 已取消"}


def process_query_order(state: OrderState) -> OrderState:
    """处理订单查询"""
    return {"result": f"订单 {state['order_id']} 状态：处理中"}


def priority_handler(state: OrderState) -> OrderState:
    """优先级处理"""
    priority_msg = {
        "high": "【紧急】",
        "normal": "",
        "low": "【低优先级】"
    }
    prefix = priority_msg.get(state["priority"], "")
    return {"result": f"{prefix}{state['result']}"}


def multi_level_branching_example():
    """多级条件分支示例"""
    print("【2. 多级条件分支 - 订单处理】")

    # 创建状态图
    graph = StateGraph(OrderState)

    # 添加节点
    graph.add_node("new", process_new_order)
    graph.add_node("modify", process_modify_order)
    graph.add_node("cancel", process_cancel_order)
    graph.add_node("query", process_query_order)
    graph.add_node("priority", priority_handler)

    # 第一级分支：订单类型
    graph.add_conditional_edges(
        START,
        check_order_type,
        {
            "new": "new",
            "modify": "modify",
            "cancel": "cancel",
            "query": "query"
        }
    )

    # 所有订单处理后进入优先级处理
    graph.add_edge("new", "priority")
    graph.add_edge("modify", "priority")
    graph.add_edge("cancel", "priority")
    graph.add_edge("query", "priority")
    graph.add_edge("priority", END)

    # 编译图
    app = graph.compile()

    # 测试
    test_orders = [
        {"order_type": "new", "priority": "high", "order_id": "ORD001"},
        {"order_type": "cancel", "priority": "normal", "order_id": "ORD002"},
        {"order_type": "query", "priority": "low", "order_id": "ORD003"},
    ]

    for order in test_orders:
        result = app.invoke({**order, "result": ""})
        print(f"订单：{order}")
        print(f"结果：{result['result']}")
        print()


# ==================== 3. 带重试的条件分支 ====================

class RetryState(TypedDict):
    """重试状态"""
    task: str
    attempt: int
    max_attempts: int
    success: bool
    result: str


def execute_task(state: RetryState) -> RetryState:
    """执行任务（模拟可能失败的操作）"""
    import random

    attempt = state["attempt"] + 1
    # 模拟：70% 的概率成功
    success = random.random() > 0.3

    if success:
        result = f"任务 '{state['task']}' 在第 {attempt} 次尝试时成功"
    else:
        result = f"任务 '{state['task']}' 第 {attempt} 次尝试失败"

    return {"attempt": attempt, "success": success, "result": result}


def should_retry(state: RetryState) -> str:
    """判断是否需要重试"""
    if state["success"]:
        return "success"
    if state["attempt"] >= state["max_attempts"]:
        return "failed"
    return "retry"


def success_handler(state: RetryState) -> RetryState:
    """成功处理"""
    return {"result": f"成功：{state['result']}"}


def failed_handler(state: RetryState) -> RetryState:
    """失败处理"""
    return {"result": f"最终失败：已尝试 {state['attempt']} 次"}


def retry_branching_example():
    """带重试的条件分支示例"""
    print("【3. 带重试的条件分支】")

    # 创建状态图
    graph = StateGraph(RetryState)

    # 添加节点
    graph.add_node("execute", execute_task)
    graph.add_node("success", success_handler)
    graph.add_node("failed", failed_handler)

    # 添加边
    graph.add_edge(START, "execute")

    # 条件边：成功/重试/失败
    graph.add_conditional_edges(
        "execute",
        should_retry,
        {
            "success": "success",
            "retry": "execute",  # 循环重试
            "failed": "failed"
        }
    )

    graph.add_edge("success", END)
    graph.add_edge("failed", END)

    # 编译图
    app = graph.compile()

    # 测试多次
    print("运行 3 次测试：")
    for i in range(3):
        result = app.invoke({
            "task": f"任务{i+1}",
            "attempt": 0,
            "max_attempts": 3,
            "success": False,
            "result": ""
        })
        print(f"测试 {i+1}：{result['result']}")
    print()


# ==================== 4. 基于内容的动态路由 ====================

class ContentState(TypedDict):
    """内容处理状态"""
    content: str
    content_type: str
    processed: str


def detect_content_type(state: ContentState) -> ContentState:
    """检测内容类型"""
    content = state["content"].lower()

    if any(ext in content for ext in [".py", "def ", "class ", "import "]):
        content_type = "code"
    elif any(ext in content for ext in [".json", "{", "}"]):
        content_type = "json"
    elif content.startswith("http"):
        content_type = "url"
    else:
        content_type = "text"

    return {"content_type": content_type}


def route_by_content(state: ContentState) -> str:
    """根据内容类型路由"""
    return f"process_{state['content_type']}"


def process_code(state: ContentState) -> ContentState:
    """处理代码"""
    return {"processed": f"代码分析：检测到 Python 代码\n{state['content'][:100]}..."}


def process_json(state: ContentState) -> ContentState:
    """处理 JSON"""
    return {"processed": f"JSON 解析：\n{state['content'][:100]}..."}


def process_url(state: ContentState) -> ContentState:
    """处理 URL"""
    return {"processed": f"URL 检测：{state['content']}"}


def process_text(state: ContentState) -> ContentState:
    """处理普通文本"""
    word_count = len(state["content"].split())
    return {"processed": f"文本分析：共 {word_count} 个词"}


def content_routing_example():
    """基于内容的动态路由示例"""
    print("【4. 基于内容的动态路由】")

    # 创建状态图
    graph = StateGraph(ContentState)

    # 添加节点
    graph.add_node("detect", detect_content_type)
    graph.add_node("process_code", process_code)
    graph.add_node("process_json", process_json)
    graph.add_node("process_url", process_url)
    graph.add_node("process_text", process_text)

    # 添加边
    graph.add_edge(START, "detect")

    # 动态路由
    graph.add_conditional_edges(
        "detect",
        route_by_content,
        {
            "process_code": "process_code",
            "process_json": "process_json",
            "process_url": "process_url",
            "process_text": "process_text"
        }
    )

    graph.add_edge("process_code", END)
    graph.add_edge("process_json", END)
    graph.add_edge("process_url", END)
    graph.add_edge("process_text", END)

    # 编译图
    app = graph.compile()

    # 测试不同内容
    test_contents = [
        "def hello():\n    print('Hello')",
        '{"name": "test", "value": 123}',
        "https://langchain.com",
        "这是一段普通的中文文本",
    ]

    for content in test_contents:
        result = app.invoke({
            "content": content,
            "content_type": "",
            "processed": ""
        })
        print(f"内容：{content[:30]}...")
        print(f"类型：{result['content_type']}")
        print(f"处理结果：{result['processed']}")
        print()


# ==================== 5. 带评分的路由决策 ====================

class ScoreState(TypedDict):
    """评分状态"""
    text: str
    sentiment_score: float
    quality_score: float
    route: str
    result: str


def analyze_sentiment(state: ScoreState) -> ScoreState:
    """分析情感（模拟）"""
    text = state["text"].lower()
    positive_words = ["好", "棒", "喜欢", "感谢", "满意", "优秀"]
    negative_words = ["差", "糟", "讨厌", "失望", "问题", "bug"]

    pos_count = sum(1 for w in positive_words if w in text)
    neg_count = sum(1 for w in negative_words if w in text)

    if pos_count + neg_count == 0:
        score = 0.5
    else:
        score = pos_count / (pos_count + neg_count)

    return {"sentiment_score": score}


def analyze_quality(state: ScoreState) -> ScoreState:
    """分析质量（模拟）"""
    text = state["text"]
    # 简单的质量评分：基于长度和标点
    length_score = min(len(text) / 100, 1.0)
    punct_score = 0.5 if any(p in text for p in "。！？,.!?") else 0.2
    score = (length_score + punct_score) / 2
    return {"quality_score": score}


def decide_route(state: ScoreState) -> ScoreState:
    """决定路由"""
    sentiment = state["sentiment_score"]
    quality = state["quality_score"]

    if sentiment > 0.7 and quality > 0.5:
        route = "positive"
    elif sentiment < 0.3:
        route = "negative"
    else:
        route = "neutral"

    return {"route": route}


def route_by_score(state: ScoreState) -> str:
    """根据评分路由"""
    return state["route"]


def positive_response(state: ScoreState) -> ScoreState:
    """正面回复"""
    return {"result": "感谢您的好评！我们会继续努力！"}


def negative_response(state: ScoreState) -> ScoreState:
    """负面回复"""
    return {"result": "非常抱歉给您带来不好的体验，我们会立即改进。"}


def neutral_response(state: ScoreState) -> ScoreState:
    """中性回复"""
    return {"result": "感谢您的反馈，我们会认真考虑您的意见。"}


def score_routing_example():
    """带评分的路由决策示例"""
    print("【5. 带评分的路由决策】")

    # 创建状态图
    graph = StateGraph(ScoreState)

    # 添加节点
    graph.add_node("sentiment", analyze_sentiment)
    graph.add_node("quality", analyze_quality)
    graph.add_node("decide", decide_route)
    graph.add_node("positive", positive_response)
    graph.add_node("negative", negative_response)
    graph.add_node("neutral", neutral_response)

    # 添加边（顺序分析）
    graph.add_edge(START, "sentiment")
    graph.add_edge("sentiment", "quality")
    graph.add_edge("quality", "decide")

    # 条件路由
    graph.add_conditional_edges(
        "decide",
        route_by_score,
        {
            "positive": "positive",
            "negative": "negative",
            "neutral": "neutral"
        }
    )

    graph.add_edge("positive", END)
    graph.add_edge("negative", END)
    graph.add_edge("neutral", END)

    # 编译图
    app = graph.compile()

    # 测试
    test_texts = [
        "你们的产品非常棒！我很喜欢，感谢你们的努力！",
        "这个产品太差了，有很多 bug，让人失望。",
        "产品还行吧，一般般。",
    ]

    for text in test_texts:
        result = app.invoke({
            "text": text,
            "sentiment_score": 0.0,
            "quality_score": 0.0,
            "route": "",
            "result": ""
        })
        print(f"输入：{text}")
        print(f"情感分：{result['sentiment_score']:.2f}")
        print(f"质量分：{result['quality_score']:.2f}")
        print(f"路由：{result['route']}")
        print(f"回复：{result['result']}")
        print()


# ==================== 演示所有功能 ====================

def demo_all():
    """演示所有条件分支功能"""
    print("=" * 60)
    print("LangGraph 条件分支 - 功能演示")
    print("=" * 60)
    print()

    basic_branching_example()
    multi_level_branching_example()
    retry_branching_example()
    content_routing_example()
    score_routing_example()

    print("=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    demo_all()
