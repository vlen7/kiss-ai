"""
第一阶段：Output Parser 输出解析示例

学习要点：
1. StrOutputParser - 字符串输出解析
2. JsonOutputParser - JSON 格式解析
3. PydanticOutputParser - 结构化数据解析
4. CommaSeparatedListOutputParser - 列表解析
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    CommaSeparatedListOutputParser,
    PydanticOutputParser
)
from pydantic import BaseModel, Field
from typing import List

load_dotenv()


def get_model():
    """获取模型实例"""
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
    )


def str_output_parser():
    """字符串输出解析器 - 最基础的解析器"""
    model = get_model()
    parser = StrOutputParser()

    # 链式调用：模型 | 解析器
    chain = model | parser

    result = chain.invoke("用一句话介绍Python")

    print("【StrOutputParser】")
    print(f"类型: {type(result)}")
    print(f"结果: {result}")
    print()


def json_output_parser():
    """JSON 输出解析器"""
    model = get_model()
    parser = JsonOutputParser()

    prompt = ChatPromptTemplate.from_template("""
请分析以下编程语言，并以JSON格式返回信息。
JSON格式要求：{{"name": "语言名", "year": 创建年份, "creator": "创建者", "use_cases": ["用途1", "用途2"]}}

编程语言：{language}

请只返回JSON，不要有其他文字：
""")

    chain = prompt | model | parser

    result = chain.invoke({"language": "Python"})

    print("【JsonOutputParser】")
    print(f"类型: {type(result)}")
    print(f"结果: {result}")
    print(f"访问字段: name={result.get('name')}, year={result.get('year')}")
    print()


def comma_separated_parser():
    """逗号分隔列表解析器"""
    model = get_model()
    parser = CommaSeparatedListOutputParser()

    # 获取格式说明
    format_instructions = parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_template("""
列出5种常用的Web开发框架。
{format_instructions}
""")

    chain = prompt | model | parser

    result = chain.invoke({"format_instructions": format_instructions})

    print("【CommaSeparatedListOutputParser】")
    print(f"类型: {type(result)}")
    print(f"结果: {result}")
    print(f"第一个元素: {result[0] if result else 'N/A'}")
    print()


# 定义 Pydantic 模型
class Book(BaseModel):
    """书籍信息模型"""
    title: str = Field(description="书籍标题")
    author: str = Field(description="作者名字")
    year: int = Field(description="出版年份")
    genre: str = Field(description="书籍类型")
    summary: str = Field(description="简短摘要，不超过50字")


def pydantic_output_parser():
    """Pydantic 结构化输出解析器"""

    model = get_model()
    parser = PydanticOutputParser(pydantic_object=Book)

    # 获取格式说明
    format_instructions = parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_template("""
请为以下书名生成详细的书籍信息。

书名：{book_name}

{format_instructions}
""")

    chain = prompt | model | parser

    result = chain.invoke({
        "book_name": "三体",
        "format_instructions": format_instructions
    })

    print("【PydanticOutputParser】")
    print(f"类型: {type(result)}")
    print(f"标题: {result.title}")
    print(f"作者: {result.author}")
    print(f"年份: {result.year}")
    print(f"类型: {result.genre}")
    print(f"摘要: {result.summary}")
    print()


class MovieReview(BaseModel):
    """电影评论分析模型"""
    movie_name: str = Field(description="电影名称")
    sentiment: str = Field(description="情感倾向：积极/消极/中性")
    score: int = Field(description="评分1-10")
    keywords: List[str] = Field(description="关键词列表")


def structured_output_example():
    """使用 with_structured_output 方法（推荐方式）"""
    model = get_model()

    # 直接绑定结构化输出
    structured_model = model.with_structured_output(MovieReview)

    result = structured_model.invoke("""
分析以下电影评论：
"《盗梦空间》是诺兰的神作！剧情烧脑但不失趣味，视觉效果震撼，
莱昂纳多的演技无可挑剔。强烈推荐！"
""")

    print("【with_structured_output（推荐）】")
    print(f"电影: {result.movie_name}")
    print(f"情感: {result.sentiment}")
    print(f"评分: {result.score}")
    print(f"关键词: {result.keywords}")
    print()


if __name__ == "__main__":
    print("=" * 50)
    print("Output Parser 输出解析示例")
    print("=" * 50)
    print()

    # str_output_parser()
    # json_output_parser()
    # comma_separated_parser()
    # pydantic_output_parser()
    structured_output_example()
