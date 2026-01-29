"""
第一阶段综合项目：多语言翻译助手

功能：
1. 支持多种语言互译
2. 自动检测源语言
3. 翻译质量评估
4. 批量翻译
5. 交互式翻译对话
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
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


# ==================== 基础翻译功能 ====================

def simple_translate(text: str, target_language: str) -> str:
    """简单翻译功能"""
    prompt = ChatPromptTemplate.from_template(
        "将以下文本翻译成{target_language}，只返回翻译结果：\n\n{text}"
    )
    model = get_model()
    parser = StrOutputParser()

    chain = prompt | model | parser
    result = chain.invoke({"text": text, "target_language": target_language})
    return result


# ==================== 语言检测 ====================

class LanguageDetection(BaseModel):
    """语言检测结果"""
    detected_language: str = Field(description="检测到的语言名称")
    confidence: str = Field(description="置信度：高/中/低")


def detect_language(text: str) -> LanguageDetection:
    """检测文本语言"""
    model = get_model()
    structured_model = model.with_structured_output(LanguageDetection)

    prompt = ChatPromptTemplate.from_template(
        "检测以下文本的语言：\n\n{text}"
    )

    chain = prompt | structured_model
    result = chain.invoke({"text": text})
    return result


# ==================== 高级翻译（带语言检测）====================

class TranslationResult(BaseModel):
    """翻译结果模型"""
    source_language: str = Field(description="源语言")
    target_language: str = Field(description="目标语言")
    original_text: str = Field(description="原文")
    translated_text: str = Field(description="译文")
    notes: str = Field(description="翻译备注，如有特殊处理说明")


def advanced_translate(text: str, target_language: str) -> TranslationResult:
    """高级翻译，自动检测语言并返回结构化结果"""
    model = get_model()
    structured_model = model.with_structured_output(TranslationResult)

    prompt = ChatPromptTemplate.from_template("""
请完成以下翻译任务：

原文：{text}
目标语言：{target_language}

要求：
1. 自动识别源语言
2. 翻译要准确自然
3. 如有特殊术语或文化相关表达，在备注中说明
""")

    chain = prompt | structured_model
    result = chain.invoke({"text": text, "target_language": target_language})
    return result


# ==================== 多语言并行翻译 ====================

def multi_language_translate(text: str, languages: List[str]) -> dict:
    """同时翻译成多种语言"""
    model = get_model()
    parser = StrOutputParser()

    # 为每种语言创建翻译链
    chains = {}
    for lang in languages:
        prompt = ChatPromptTemplate.from_template(
            f"将以下文本翻译成{lang}，只返回翻译结果：\n\n{{text}}"
        )
        chains[lang] = prompt | model | parser

    # 创建并行链
    parallel_chain = RunnableParallel(**chains)

    result = parallel_chain.invoke({"text": text})
    return result


# ==================== 翻译质量评估 ====================

class QualityAssessment(BaseModel):
    """翻译质量评估"""
    accuracy_score: int = Field(description="准确性评分 1-10")
    fluency_score: int = Field(description="流畅性评分 1-10")
    overall_score: int = Field(description="综合评分 1-10")
    issues: List[str] = Field(description="发现的问题列表")
    suggestions: List[str] = Field(description="改进建议")


def assess_translation(original: str, translated: str, target_language: str) -> QualityAssessment:
    """评估翻译质量"""
    model = get_model()
    structured_model = model.with_structured_output(QualityAssessment)

    prompt = ChatPromptTemplate.from_template("""
请评估以下翻译的质量：

原文：{original}
译文（{target_language}）：{translated}

请从准确性、流畅性等方面进行评估。
""")

    chain = prompt | structured_model
    result = chain.invoke({
        "original": original,
        "translated": translated,
        "target_language": target_language
    })
    return result


# ==================== 翻译+评估一体化 ====================

def translate_and_assess(text: str, target_language: str):
    """翻译并评估质量"""
    # 先翻译
    translation = advanced_translate(text, target_language)

    # 再评估
    assessment = assess_translation(
        text,
        translation.translated_text,
        target_language
    )

    return {
        "translation": translation,
        "assessment": assessment
    }


# ==================== 交互式翻译对话 ====================

def interactive_translator():
    """交互式翻译助手"""
    print("\n" + "=" * 50)
    print("欢迎使用多语言翻译助手！")
    print("=" * 50)
    print("\n支持的语言：中文、英语、日语、韩语、法语、德语、西班牙语等")
    print("输入 'quit' 退出\n")

    while True:
        text = input("请输入要翻译的文本：").strip()
        if text.lower() == 'quit':
            print("感谢使用，再见！")
            break
        if not text:
            continue

        target = input("请输入目标语言：").strip()
        if not target:
            target = "英语"

        print("\n翻译中...")
        try:
            result = advanced_translate(text, target)
            print(f"\n【翻译结果】")
            print(f"源语言：{result.source_language}")
            print(f"目标语言：{result.target_language}")
            print(f"原文：{result.original_text}")
            print(f"译文：{result.translated_text}")
            if result.notes:
                print(f"备注：{result.notes}")
        except Exception as e:
            print(f"翻译出错：{e}")

        print()


# ==================== 演示所有功能 ====================

def demo_all_features():
    """演示所有翻译功能"""

    print("=" * 60)
    print("多语言翻译助手 - 功能演示")
    print("=" * 60)

    # 1. 简单翻译
    print("\n【1. 简单翻译】")
    text = "人工智能正在改变世界"
    result = simple_translate(text, "英语")
    print(f"原文：{text}")
    print(f"译文：{result}")

    # 2. 语言检测
    print("\n【2. 语言检测】")
    test_texts = [
        "Hello, how are you?",
        "こんにちは",
        "Bonjour le monde",
    ]
    for t in test_texts:
        detection = detect_language(t)
        print(f"'{t}' -> {detection.detected_language} (置信度: {detection.confidence})")

    # 3. 高级翻译
    print("\n【3. 高级翻译（结构化输出）】")
    text = "The early bird catches the worm"
    result = advanced_translate(text, "中文")
    print(f"源语言：{result.source_language}")
    print(f"原文：{result.original_text}")
    print(f"译文：{result.translated_text}")
    print(f"备注：{result.notes}")

    # 4. 多语言并行翻译
    print("\n【4. 多语言并行翻译】")
    text = "科技让生活更美好"
    languages = ["英语", "日语", "法语"]
    results = multi_language_translate(text, languages)
    print(f"原文：{text}")
    for lang, translation in results.items():
        print(f"{lang}：{translation}")

    # 5. 翻译质量评估
    print("\n【5. 翻译质量评估】")
    original = "Knowledge is power"
    translated = "知识就是力量"
    assessment = assess_translation(original, translated, "中文")
    print(f"原文：{original}")
    print(f"译文：{translated}")
    print(f"准确性：{assessment.accuracy_score}/10")
    print(f"流畅性：{assessment.fluency_score}/10")
    print(f"综合分：{assessment.overall_score}/10")
    if assessment.issues:
        print(f"问题：{assessment.issues}")
    if assessment.suggestions:
        print(f"建议：{assessment.suggestions}")

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "-i":
        # 交互模式
        interactive_translator()
    else:
        # 演示模式
        demo_all_features()
