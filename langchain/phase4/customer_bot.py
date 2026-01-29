"""
第四阶段：LangGraph 客服机器人

学习要点：
1. 完整的多步骤工作流
2. Human-in-the-loop（人机协作）
3. 状态持久化（Checkpointing）
4. 中断和恢复
5. 实际业务场景应用
"""

import os
import uuid
from typing import TypedDict, Annotated, Literal, Optional
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


def get_model() -> ChatOpenAI:
    """获取 LLM 模型"""
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
    )


# ==================== 状态定义 ====================

class CustomerState(TypedDict):
    """客服机器人状态"""
    # 对话消息
    messages: Annotated[list[BaseMessage], add_messages]
    # 用户信息
    user_id: str
    user_name: str
    # 会话信息
    session_id: str
    # 意图分类
    intent: str
    # 子意图/具体问题类型
    sub_intent: str
    # 收集的信息
    collected_info: dict
    # 是否需要人工介入
    needs_human: bool
    # 人工介入原因
    human_reason: str
    # 当前步骤
    current_step: str
    # 工单 ID（如果创建了工单）
    ticket_id: str
    # 最终结果
    resolution: str


# ==================== 模拟数据库 ====================

class MockDatabase:
    """模拟数据库"""

    # 用户数据
    users = {
        "U001": {"name": "张三", "level": "VIP", "orders": ["ORD001", "ORD002"]},
        "U002": {"name": "李四", "level": "普通", "orders": ["ORD003"]},
    }

    # 订单数据
    orders = {
        "ORD001": {"status": "已发货", "product": "手机", "amount": 5999},
        "ORD002": {"status": "待发货", "product": "耳机", "amount": 299},
        "ORD003": {"status": "已完成", "product": "充电器", "amount": 99},
    }

    # 常见问题
    faq = {
        "退货": "您可以在收到商品后 7 天内申请退货，请保持商品完好。",
        "换货": "换货需要在 15 天内申请，请联系客服提供订单号。",
        "运费": "满 99 元免运费，不满 99 元收取 10 元运费。",
        "发票": "我们提供电子发票，下单时选择开票选项即可。",
    }

    # 工单
    tickets = {}

    @classmethod
    def get_user(cls, user_id: str) -> Optional[dict]:
        """获取用户信息"""
        return cls.users.get(user_id)

    @classmethod
    def get_order(cls, order_id: str) -> Optional[dict]:
        """获取订单信息"""
        return cls.orders.get(order_id)

    @classmethod
    def get_faq(cls, keyword: str) -> Optional[str]:
        """获取 FAQ 答案"""
        for key, answer in cls.faq.items():
            if key in keyword:
                return answer
        return None

    @classmethod
    def create_ticket(cls, user_id: str, issue: str, priority: str) -> str:
        """创建工单"""
        ticket_id = f"TKT{len(cls.tickets) + 1:04d}"
        cls.tickets[ticket_id] = {
            "user_id": user_id,
            "issue": issue,
            "priority": priority,
            "status": "待处理",
            "created_at": datetime.now().isoformat()
        }
        return ticket_id


# ==================== 节点函数 ====================

def welcome_node(state: CustomerState) -> CustomerState:
    """欢迎节点：初始化会话"""
    user = MockDatabase.get_user(state["user_id"])
    user_name = user["name"] if user else "尊敬的用户"

    welcome_msg = f"""你好 {user_name}！我是智能客服小助手。
我可以帮您处理以下问题：
1. 订单查询
2. 退换货咨询
3. 常见问题解答
4. 投诉建议

请问有什么可以帮您的？"""

    return {
        "messages": [AIMessage(content=welcome_msg)],
        "user_name": user_name,
        "current_step": "classify"
    }


def classify_intent_node(state: CustomerState) -> CustomerState:
    """意图分类节点"""
    model = get_model()

    # 获取最后一条用户消息
    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    if not last_user_msg:
        return {"intent": "unknown", "current_step": "clarify"}

    messages = [
        SystemMessage(content="""你是一个意图分类器。根据用户输入，判断用户的意图。
只返回以下类别之一：
- order_query: 订单查询（查订单、物流、发货等）
- return_refund: 退换货（退货、换货、退款等）
- faq: 常见问题（运费、发票、售后政策等）
- complaint: 投诉建议
- chat: 闲聊
- unknown: 无法判断

只返回类别名称，不要其他内容。"""),
        HumanMessage(content=last_user_msg)
    ]

    response = model.invoke(messages)
    intent = response.content.strip().lower()

    # 验证意图
    valid_intents = ["order_query", "return_refund", "faq", "complaint", "chat", "unknown"]
    if intent not in valid_intents:
        intent = "unknown"

    return {"intent": intent, "current_step": "route"}


def route_intent(state: CustomerState) -> str:
    """根据意图路由"""
    intent_routes = {
        "order_query": "order_handler",
        "return_refund": "return_handler",
        "faq": "faq_handler",
        "complaint": "complaint_handler",
        "chat": "chat_handler",
        "unknown": "clarify_handler"
    }
    return intent_routes.get(state["intent"], "clarify_handler")


def order_handler_node(state: CustomerState) -> CustomerState:
    """订单查询处理"""
    model = get_model()
    user = MockDatabase.get_user(state["user_id"])

    if not user:
        return {
            "messages": [AIMessage(content="抱歉，无法找到您的用户信息，请提供您的订单号。")],
            "current_step": "collect_order_id"
        }

    # 获取用户订单
    order_ids = user.get("orders", [])
    orders_info = []
    for oid in order_ids:
        order = MockDatabase.get_order(oid)
        if order:
            orders_info.append(f"- {oid}: {order['product']}，状态：{order['status']}，金额：¥{order['amount']}")

    if orders_info:
        order_list = "\n".join(orders_info)
        response = f"""您的订单信息如下：
{order_list}

请问您想查询哪个订单的详细信息？或者有什么其他问题？"""
    else:
        response = "您目前没有订单记录。请问还有什么可以帮您的吗？"

    return {
        "messages": [AIMessage(content=response)],
        "current_step": "order_detail",
        "resolution": "订单查询已处理"
    }


def return_handler_node(state: CustomerState) -> CustomerState:
    """退换货处理"""
    response = """关于退换货，以下是我们的政策：

【退货】
- 收到商品 7 天内可申请无理由退货
- 商品需保持原包装完好
- 退款将在 3-5 个工作日内到账

【换货】
- 收到商品 15 天内可申请换货
- 如因质量问题换货，运费由我们承担

请问您是想退货还是换货？可以告诉我您的订单号吗？"""

    return {
        "messages": [AIMessage(content=response)],
        "current_step": "collect_return_info",
        "collected_info": {"type": "return_refund"}
    }


def faq_handler_node(state: CustomerState) -> CustomerState:
    """FAQ 处理"""
    # 获取最后一条用户消息
    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    # 查找 FAQ
    answer = MockDatabase.get_faq(last_user_msg)

    if answer:
        response = f"关于您的问题，答案如下：\n\n{answer}\n\n请问还有其他问题吗？"
    else:
        response = """以下是一些常见问题：
1. 运费政策
2. 发票开具
3. 退货流程
4. 换货流程

请问您想了解哪方面的信息？"""

    return {
        "messages": [AIMessage(content=response)],
        "current_step": "faq_followup",
        "resolution": "FAQ 已回答"
    }


def complaint_handler_node(state: CustomerState) -> CustomerState:
    """投诉处理 - 需要人工介入"""
    # 获取投诉内容
    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    # 创建工单
    ticket_id = MockDatabase.create_ticket(
        user_id=state["user_id"],
        issue=last_user_msg,
        priority="高"
    )

    response = f"""非常抱歉给您带来了不好的体验！

我已经为您创建了投诉工单（工单号：{ticket_id}），我们的专员会在 2 小时内与您联系。

同时，我会将此问题升级给人工客服，请稍等。"""

    return {
        "messages": [AIMessage(content=response)],
        "needs_human": True,
        "human_reason": "用户投诉需要人工处理",
        "ticket_id": ticket_id,
        "current_step": "human_handoff"
    }


def chat_handler_node(state: CustomerState) -> CustomerState:
    """闲聊处理"""
    model = get_model()

    messages = [
        SystemMessage(content="""你是一个友好的客服机器人。用户在和你闲聊。
请友好地回应，但也适当引导用户回到业务问题上。
回复要简短、友好。"""),
    ] + state["messages"]

    response = model.invoke(messages)

    return {
        "messages": [AIMessage(content=response.content)],
        "current_step": "classify"
    }


def clarify_handler_node(state: CustomerState) -> CustomerState:
    """澄清处理"""
    response = """抱歉，我没有完全理解您的意思。请问您是想：

1. 查询订单 - 回复"订单"
2. 退换货咨询 - 回复"退货"或"换货"
3. 了解常见问题 - 回复"问题"
4. 提出投诉或建议 - 回复"投诉"

或者您可以直接描述您的问题，我会尽力帮您解决。"""

    return {
        "messages": [AIMessage(content=response)],
        "current_step": "classify"
    }


def check_human_needed(state: CustomerState) -> str:
    """检查是否需要人工介入"""
    if state.get("needs_human"):
        return "human_handoff"
    return "continue"


def human_handoff_node(state: CustomerState) -> CustomerState:
    """人工交接节点"""
    response = f"""正在为您转接人工客服...

转接原因：{state.get('human_reason', '用户请求')}
工单号：{state.get('ticket_id', '无')}

人工客服将在 30 秒内接入，请稍候。

【模拟】人工客服已接入：您好，我是人工客服小王，已了解您的问题，正在为您处理。"""

    return {
        "messages": [AIMessage(content=response)],
        "current_step": "human_handling",
        "resolution": "已转接人工客服"
    }


def end_conversation_node(state: CustomerState) -> CustomerState:
    """结束对话"""
    response = f"""感谢您的咨询！

本次服务摘要：
- 处理结果：{state.get('resolution', '已处理')}
- 工单号：{state.get('ticket_id', '无')}

如果还有其他问题，随时可以找我。祝您生活愉快！"""

    return {
        "messages": [AIMessage(content=response)],
        "current_step": "ended"
    }


# ==================== 构建完整的客服机器人图 ====================

def create_customer_bot():
    """创建客服机器人图"""
    # 创建状态图
    graph = StateGraph(CustomerState)

    # 添加节点
    graph.add_node("welcome", welcome_node)
    graph.add_node("classify", classify_intent_node)
    graph.add_node("order_handler", order_handler_node)
    graph.add_node("return_handler", return_handler_node)
    graph.add_node("faq_handler", faq_handler_node)
    graph.add_node("complaint_handler", complaint_handler_node)
    graph.add_node("chat_handler", chat_handler_node)
    graph.add_node("clarify_handler", clarify_handler_node)
    graph.add_node("human_handoff", human_handoff_node)
    graph.add_node("end", end_conversation_node)

    # 添加边
    graph.add_edge(START, "welcome")
    graph.add_edge("welcome", "classify")

    # 意图路由
    graph.add_conditional_edges(
        "classify",
        route_intent,
        {
            "order_handler": "order_handler",
            "return_handler": "return_handler",
            "faq_handler": "faq_handler",
            "complaint_handler": "complaint_handler",
            "chat_handler": "chat_handler",
            "clarify_handler": "clarify_handler"
        }
    )

    # 投诉处理后检查是否需要人工
    graph.add_conditional_edges(
        "complaint_handler",
        check_human_needed,
        {
            "human_handoff": "human_handoff",
            "continue": "end"
        }
    )

    # 其他处理完成后结束
    graph.add_edge("order_handler", "end")
    graph.add_edge("return_handler", "end")
    graph.add_edge("faq_handler", "end")
    graph.add_edge("chat_handler", "end")
    graph.add_edge("clarify_handler", "end")
    graph.add_edge("human_handoff", "end")
    graph.add_edge("end", END)

    # 使用内存检查点（状态持久化）
    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)

    return app


# ==================== 示例运行 ====================

def single_turn_example():
    """单轮对话示例"""
    print("【1. 单轮对话示例】")

    app = create_customer_bot()

    # 初始状态
    initial_state = {
        "messages": [],
        "user_id": "U001",
        "user_name": "",
        "session_id": str(uuid.uuid4()),
        "intent": "",
        "sub_intent": "",
        "collected_info": {},
        "needs_human": False,
        "human_reason": "",
        "current_step": "",
        "ticket_id": "",
        "resolution": ""
    }

    # 配置（用于状态持久化）
    config = {"configurable": {"thread_id": "demo-1"}}

    # 第一次调用：欢迎
    result = app.invoke(initial_state, config)
    print("机器人：", result["messages"][-1].content)
    print()


def order_query_example():
    """订单查询示例"""
    print("【2. 订单查询示例】")

    app = create_customer_bot()
    config = {"configurable": {"thread_id": "demo-2"}}

    # 初始状态
    state = {
        "messages": [],
        "user_id": "U001",
        "user_name": "",
        "session_id": str(uuid.uuid4()),
        "intent": "",
        "sub_intent": "",
        "collected_info": {},
        "needs_human": False,
        "human_reason": "",
        "current_step": "",
        "ticket_id": "",
        "resolution": ""
    }

    # 第一次：欢迎
    result = app.invoke(state, config)
    print("机器人：", result["messages"][-1].content[:100], "...")
    print()

    # 用户输入
    state["messages"] = result["messages"] + [HumanMessage(content="我想查一下我的订单")]

    # 第二次：处理订单查询
    result = app.invoke(state, config)
    print("用户：我想查一下我的订单")
    print("机器人：", result["messages"][-1].content)
    print()


def complaint_example():
    """投诉处理示例"""
    print("【3. 投诉处理示例（转人工）】")

    app = create_customer_bot()
    config = {"configurable": {"thread_id": "demo-3"}}

    # 初始状态
    state = {
        "messages": [],
        "user_id": "U002",
        "user_name": "",
        "session_id": str(uuid.uuid4()),
        "intent": "",
        "sub_intent": "",
        "collected_info": {},
        "needs_human": False,
        "human_reason": "",
        "current_step": "",
        "ticket_id": "",
        "resolution": ""
    }

    # 欢迎
    result = app.invoke(state, config)
    print("机器人：", result["messages"][-1].content[:50], "...")
    print()

    # 用户投诉
    state["messages"] = result["messages"] + [HumanMessage(content="我要投诉！你们的产品质量太差了！")]

    result = app.invoke(state, config)
    print("用户：我要投诉！你们的产品质量太差了！")
    print("机器人：", result["messages"][-1].content)
    print()


def faq_example():
    """FAQ 示例"""
    print("【4. FAQ 查询示例】")

    app = create_customer_bot()
    config = {"configurable": {"thread_id": "demo-4"}}

    state = {
        "messages": [],
        "user_id": "U001",
        "user_name": "",
        "session_id": str(uuid.uuid4()),
        "intent": "",
        "sub_intent": "",
        "collected_info": {},
        "needs_human": False,
        "human_reason": "",
        "current_step": "",
        "ticket_id": "",
        "resolution": ""
    }

    # 欢迎
    result = app.invoke(state, config)

    # 询问运费
    state["messages"] = result["messages"] + [HumanMessage(content="请问运费是多少？")]

    result = app.invoke(state, config)
    print("用户：请问运费是多少？")
    print("机器人：", result["messages"][-1].content)
    print()


# ==================== 交互式客服 ====================

def interactive_customer_bot():
    """交互式客服机器人"""
    print("\n" + "=" * 50)
    print("智能客服系统")
    print("=" * 50)
    print("\n输入 'quit' 退出，输入 'new' 开始新会话\n")

    app = create_customer_bot()
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    # 初始状态
    state = {
        "messages": [],
        "user_id": "U001",  # 模拟登录用户
        "user_name": "",
        "session_id": session_id,
        "intent": "",
        "sub_intent": "",
        "collected_info": {},
        "needs_human": False,
        "human_reason": "",
        "current_step": "",
        "ticket_id": "",
        "resolution": ""
    }

    # 首次欢迎
    result = app.invoke(state, config)
    print(f"客服：{result['messages'][-1].content}\n")

    while True:
        user_input = input("您：").strip()

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("感谢使用，再见！")
            break

        if user_input.lower() == "new":
            session_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": session_id}}
            state["messages"] = []
            state["session_id"] = session_id
            result = app.invoke(state, config)
            print(f"\n客服：{result['messages'][-1].content}\n")
            continue

        # 添加用户消息
        state["messages"] = result["messages"] + [HumanMessage(content=user_input)]

        try:
            result = app.invoke(state, config)
            # 获取最新的 AI 回复
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage):
                    print(f"\n客服：{msg.content}\n")
                    break
        except Exception as e:
            print(f"\n系统错误：{e}\n")


# ==================== 演示所有功能 ====================

def demo_all():
    """演示所有客服机器人功能"""
    print("=" * 60)
    print("LangGraph 客服机器人 - 功能演示")
    print("=" * 60)
    print()

    single_turn_example()
    order_query_example()
    complaint_example()
    faq_example()

    print("=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "-i":
        interactive_customer_bot()
    else:
        demo_all()
