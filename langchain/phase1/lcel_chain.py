"""
第一阶段：LCEL (LangChain Expression Language) 链式调用示例

学习要点：
1. | 管道操作符 - 链接组件
2. RunnablePassthrough - 传递输入
3. RunnableLambda - 自定义函数
4. RunnableParallel - 并行执行
5. 链的组合与复用
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel,
)

load_dotenv()


def get_model():
    """获取模型实例"""
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
    )


def basic_chain():
    """基础链：Prompt | Model | OutputParser"""
    prompt = ChatPromptTemplate.from_template(
        "用一句话解释什么是{topic}"
    )
    model = get_model()
    parser = StrOutputParser()

    # 使用 | 操作符连接组件
    chain = prompt | model | parser

    result = chain.invoke({"topic": "区块链"})

    print("【基础链 (Prompt | Model | Parser)】")
    print(f"结果: {result}")
    print()


def chain_with_passthrough():
    """使用 RunnablePassthrough 传递原始输入"""
    prompt = ChatPromptTemplate.from_template("""
原始问题：{question}

请详细回答上述问题。
""")
    model = get_model()
    parser = StrOutputParser()

    # RunnablePassthrough 直接传递输入
    chain = (
        {"question": RunnablePassthrough()}
        | prompt
        | model
        | parser
    )

    result = chain.invoke("什么是微服务架构？")

    print("【RunnablePassthrough 传递输入】")
    print(f"结果: {result[:100]}...")
    print()


def chain_with_lambda():
    """使用 RunnableLambda 自定义处理函数"""

    # 自定义预处理函数
    def preprocess(text: str) -> str:
        return text.strip().lower()

    # 自定义后处理函数
    def postprocess(text: str) -> dict:
        return {
            "answer": text,
            "length": len(text),
            "word_count": len(text.split())
        }

    prompt = ChatPromptTemplate.from_template("简要解释：{input}")
    model = get_model()
    parser = StrOutputParser()

    chain = (
        RunnableLambda(preprocess)
        | {"input": RunnablePassthrough()}
        | prompt
        | model
        | parser
        | RunnableLambda(postprocess)
    )

    result = chain.invoke("  PYTHON 装饰器  ")

    print("【RunnableLambda 自定义函数】")
    print(f"答案: {result['answer'][:80]}...")
    print(f"字符数: {result['length']}")
    print(f"词数: {result['word_count']}")
    print()


def parallel_chain():
    """使用 RunnableParallel 并行执行多个链"""
    model = get_model()
    parser = StrOutputParser()

    # 定义多个提示模板
    joke_prompt = ChatPromptTemplate.from_template(
        "讲一个关于{topic}的笑话，要简短"
    )
    fact_prompt = ChatPromptTemplate.from_template(
        "说一个关于{topic}的有趣事实"
    )
    poem_prompt = ChatPromptTemplate.from_template(
        "写一首关于{topic}的两行小诗"
    )

    # 创建并行链
    parallel_chain = RunnableParallel(
        joke=joke_prompt | model | parser,
        fact=fact_prompt | model | parser,
        poem=poem_prompt | model | parser,
    )

    result = parallel_chain.invoke({"topic": "程序员"})

    print("【RunnableParallel 并行执行】")
    print(f"笑话: {result['joke']}")
    print(f"事实: {result['fact']}")
    print(f"小诗: {result['poem']}")
    print()


def chained_chains():
    """链的嵌套与组合"""
    model = get_model()
    parser = StrOutputParser()

    # 第一个链：生成概念解释
    explain_prompt = ChatPromptTemplate.from_template(
        "用一句话解释{concept}"
    )
    explain_chain = explain_prompt | model | parser

    # 第二个链：基于解释生成例子
    example_prompt = ChatPromptTemplate.from_template(
        "基于这个解释：'{explanation}'，给出一个简单的代码示例"
    )
    example_chain = example_prompt | model | parser

    # 组合链
    combined_chain = (
        {"explanation": explain_chain, "concept": RunnablePassthrough()}
        | example_prompt
        | model
        | parser
    )

    result = combined_chain.invoke({"concept": "递归"})

    print("【链的组合】")
    print(f"结果: {result}")
    print()


def streaming_chain():
    """链的流式输出"""
    prompt = ChatPromptTemplate.from_template(
        "写一段关于{topic}的简短介绍（约100字）"
    )
    model = get_model()
    parser = StrOutputParser()

    chain = prompt | model | parser

    print("【流式输出】")
    print("回答: ", end="")
    for chunk in chain.stream({"topic": "人工智能"}):
        print(chunk, end="", flush=True)
    print("\n")


def batch_processing():
    """批量处理"""
    prompt = ChatPromptTemplate.from_template(
        "用一个词形容{item}"
    )
    model = get_model()
    parser = StrOutputParser()

    chain = prompt | model | parser

    # 批量调用
    items = [
        {"item": "大海"},
        {"item": "森林"},
        {"item": "沙漠"},
    ]

    results = chain.batch(items)

    print("【批量处理】")
    for item, result in zip(items, results):
        print(f"{item['item']}: {result}")
    print()


if __name__ == "__main__":
    print("=" * 50)
    print("LCEL 链式调用示例")
    print("=" * 50)
    print()

    basic_chain()
    chain_with_passthrough()
    chain_with_lambda()
    parallel_chain()
    chained_chains()
    streaming_chain()
    batch_processing()