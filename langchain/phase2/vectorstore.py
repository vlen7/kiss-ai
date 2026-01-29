"""
第二阶段：Embeddings & Vector Stores 向量存储示例

学习要点：
1. Embeddings - 文本向量化（OpenAI Embeddings）
2. FAISS - Facebook 向量搜索库
3. Chroma - 轻量级向量数据库
4. Retrievers - 检索器的使用
5. 相似度搜索与 MMR 搜索
"""

import os
from dotenv import load_dotenv
from typing import List

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def get_embeddings() -> OpenAIEmbeddings:
    """获取 Embeddings 模型"""
    return OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
    )


# ==================== Embeddings 基础 ====================

def embeddings_basic_example():
    """Embeddings 基础示例"""
    print("【1. Embeddings 基础使用】")

    embeddings = get_embeddings()

    # 单个文本向量化
    text = "LangChain 是一个强大的 LLM 应用开发框架"
    vector = embeddings.embed_query(text)

    print(f"文本：{text}")
    print(f"向量维度：{len(vector)}")
    print(f"向量前5个值：{vector[:5]}")
    print()


def embeddings_batch_example():
    """批量向量化示例"""
    print("【2. 批量文本向量化】")

    embeddings = get_embeddings()

    texts = [
        "Python 是一种编程语言",
        "机器学习是人工智能的一个分支",
        "LangChain 用于构建 LLM 应用",
    ]

    vectors = embeddings.embed_documents(texts)

    print(f"文本数量：{len(texts)}")
    print(f"向量数量：{len(vectors)}")
    print(f"每个向量维度：{len(vectors[0])}")
    print()


def embeddings_similarity_example():
    """向量相似度计算示例"""
    print("【3. 向量相似度计算】")

    embeddings = get_embeddings()

    # 准备文本
    query = "如何学习编程"
    docs = [
        "Python 编程入门教程",
        "今天天气很好",
        "学习 JavaScript 的最佳方法",
        "人工智能的发展历史",
    ]

    # 计算向量
    query_vector = embeddings.embed_query(query)
    doc_vectors = embeddings.embed_documents(docs)

    # 计算余弦相似度
    import numpy as np

    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    print(f"查询：{query}")
    print("相似度排名：")

    similarities = []
    for i, doc in enumerate(docs):
        sim = cosine_similarity(query_vector, doc_vectors[i])
        similarities.append((doc, sim))

    # 按相似度排序
    similarities.sort(key=lambda x: x[1], reverse=True)
    for doc, sim in similarities:
        print(f"  {sim:.4f} - {doc}")
    print()


# ==================== FAISS 向量存储 ====================

def create_sample_documents() -> List[Document]:
    """创建示例文档"""
    texts = [
        "LangChain 是一个用于开发由语言模型驱动的应用程序的框架。它提供了统一的接口来连接各种 LLM。",
        "RAG 即检索增强生成，通过检索外部知识来增强 LLM 的回答能力，减少幻觉问题。",
        "向量数据库用于存储和检索高维向量，常用于语义搜索和相似度匹配场景。",
        "FAISS 是 Facebook 开发的向量搜索库，支持高效的相似度搜索和聚类。",
        "Chroma 是一个轻量级的向量数据库，专为 AI 应用设计，易于使用和部署。",
        "Embeddings 将文本转换为高维向量，使得语义相似的文本在向量空间中距离更近。",
        "Python 是一种简单易学的编程语言，广泛用于数据科学和人工智能领域。",
        "机器学习是人工智能的一个分支，通过数据训练模型来完成特定任务。",
    ]

    return [Document(page_content=text, metadata={"source": f"doc_{i}"}) for i, text in enumerate(texts)]


def faiss_basic_example():
    """FAISS 基础示例"""
    print("【4. FAISS 向量存储】")

    embeddings = get_embeddings()
    docs = create_sample_documents()

    # 创建 FAISS 索引
    vectorstore = FAISS.from_documents(docs, embeddings)

    print(f"创建了包含 {len(docs)} 个文档的 FAISS 索引")

    # 相似度搜索
    query = "什么是 RAG？"
    results = vectorstore.similarity_search(query, k=3)

    print(f"\n查询：{query}")
    print("搜索结果：")
    for i, doc in enumerate(results):
        print(f"  [{i+1}] {doc.page_content[:60]}...")
    print()

    return vectorstore


def faiss_with_scores_example():
    """FAISS 带分数的搜索"""
    print("【5. FAISS 相似度搜索（带分数）】")

    embeddings = get_embeddings()
    docs = create_sample_documents()
    vectorstore = FAISS.from_documents(docs, embeddings)

    query = "向量数据库有哪些？"
    results = vectorstore.similarity_search_with_score(query, k=4)

    print(f"查询：{query}")
    print("搜索结果（分数越低越相似）：")
    for doc, score in results:
        print(f"  [{score:.4f}] {doc.page_content[:50]}...")
    print()


def faiss_persistence_example():
    """FAISS 持久化示例"""
    print("【6. FAISS 索引持久化】")

    embeddings = get_embeddings()
    docs = create_sample_documents()

    # 创建并保存索引
    vectorstore = FAISS.from_documents(docs, embeddings)
    save_path = "/tmp/faiss_index"
    vectorstore.save_local(save_path)
    print(f"索引已保存到：{save_path}")

    # 加载索引
    loaded_vectorstore = FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("索引已加载")

    # 验证搜索功能
    results = loaded_vectorstore.similarity_search("LangChain", k=2)
    print(f"搜索验证：找到 {len(results)} 个结果")
    print()


# ==================== Chroma 向量存储 ====================

def chroma_basic_example():
    """Chroma 基础示例"""
    print("【7. Chroma 向量数据库】")

    embeddings = get_embeddings()
    docs = create_sample_documents()

    # 创建 Chroma 索引（内存模式）
    vectorstore = Chroma.from_documents(
        docs,
        embeddings,
        collection_name="langchain_demo"
    )

    print(f"创建了 Chroma 集合，包含 {len(docs)} 个文档")

    # 搜索
    query = "如何构建 AI 应用？"
    results = vectorstore.similarity_search(query, k=3)

    print(f"\n查询：{query}")
    print("搜索结果：")
    for i, doc in enumerate(results):
        print(f"  [{i+1}] {doc.page_content[:60]}...")
    print()


def chroma_persistence_example():
    """Chroma 持久化示例"""
    print("【8. Chroma 持久化存储】")

    embeddings = get_embeddings()
    docs = create_sample_documents()

    # 持久化到磁盘
    persist_directory = "/tmp/chroma_db"

    vectorstore = Chroma.from_documents(
        docs,
        embeddings,
        collection_name="persistent_demo",
        persist_directory=persist_directory
    )
    print(f"Chroma 数据库已保存到：{persist_directory}")

    # 重新加载
    loaded_vectorstore = Chroma(
        collection_name="persistent_demo",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    print("数据库已重新加载")

    # 验证
    results = loaded_vectorstore.similarity_search("Python", k=2)
    print(f"搜索验证：找到 {len(results)} 个结果")
    print()


# ==================== MMR 搜索（最大边际相关性）====================

def mmr_search_example():
    """MMR 搜索示例（增加结果多样性）"""
    print("【9. MMR 搜索（Maximum Marginal Relevance）】")

    embeddings = get_embeddings()

    # 准备有重复内容的文档
    docs = [
        Document(page_content="Python 是一种流行的编程语言，用于 Web 开发"),
        Document(page_content="Python 是一种流行的编程语言，用于数据科学"),
        Document(page_content="Python 是一种流行的编程语言，用于机器学习"),
        Document(page_content="JavaScript 是 Web 前端的主要语言"),
        Document(page_content="Java 广泛用于企业级应用开发"),
        Document(page_content="Go 语言以高性能著称"),
    ]

    vectorstore = FAISS.from_documents(docs, embeddings)
    query = "编程语言"

    # 普通相似度搜索
    print(f"查询：{query}")
    print("\n普通相似度搜索（可能有重复）：")
    results = vectorstore.similarity_search(query, k=4)
    for i, doc in enumerate(results):
        print(f"  [{i+1}] {doc.page_content}")

    # MMR 搜索（增加多样性）
    print("\nMMR 搜索（增加多样性）：")
    mmr_results = vectorstore.max_marginal_relevance_search(
        query,
        k=4,
        fetch_k=6,       # 先获取6个候选
        lambda_mult=0.5  # 多样性参数，0-1，越小越多样
    )
    for i, doc in enumerate(mmr_results):
        print(f"  [{i+1}] {doc.page_content}")
    print()


# ==================== Retriever 检索器 ====================

def retriever_example():
    """Retriever 检索器示例"""
    print("【10. Retriever 检索器】")

    embeddings = get_embeddings()
    docs = create_sample_documents()
    vectorstore = FAISS.from_documents(docs, embeddings)

    # 创建检索器
    retriever = vectorstore.as_retriever(
        search_type="similarity",  # 或 "mmr"
        search_kwargs={"k": 3}
    )

    # 使用检索器
    query = "什么是向量数据库？"
    results = retriever.invoke(query)

    print(f"查询：{query}")
    print(f"检索到 {len(results)} 个文档：")
    for i, doc in enumerate(results):
        print(f"  [{i+1}] {doc.page_content[:60]}...")
    print()


def retriever_with_filter_example():
    """带过滤的检索器"""
    print("【11. 带元数据过滤的检索器】")

    embeddings = get_embeddings()

    # 带元数据的文档
    docs = [
        Document(page_content="Python 基础教程", metadata={"category": "programming", "level": "beginner"}),
        Document(page_content="Python 高级特性", metadata={"category": "programming", "level": "advanced"}),
        Document(page_content="机器学习入门", metadata={"category": "ai", "level": "beginner"}),
        Document(page_content="深度学习进阶", metadata={"category": "ai", "level": "advanced"}),
    ]

    vectorstore = Chroma.from_documents(docs, embeddings)

    # 带过滤的检索
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 2,
            "filter": {"category": "programming"}
        }
    )

    results = retriever.invoke("学习教程")

    print("过滤条件：category = 'programming'")
    print("检索结果：")
    for doc in results:
        print(f"  - {doc.page_content} | {doc.metadata}")
    print()


# ==================== 添加和删除文档 ====================

def add_delete_documents_example():
    """动态添加和删除文档"""
    print("【12. 动态添加和删除文档】")

    embeddings = get_embeddings()
    docs = [
        Document(page_content="初始文档1", metadata={"id": "1"}),
        Document(page_content="初始文档2", metadata={"id": "2"}),
    ]

    vectorstore = FAISS.from_documents(docs, embeddings)
    print(f"初始文档数量：2")

    # 添加新文档
    new_docs = [
        Document(page_content="新增文档3", metadata={"id": "3"}),
        Document(page_content="新增文档4", metadata={"id": "4"}),
    ]
    vectorstore.add_documents(new_docs)
    print(f"添加 2 个文档后：可搜索到更多结果")

    # 搜索验证
    results = vectorstore.similarity_search("文档", k=10)
    print(f"搜索 '文档' 找到 {len(results)} 个结果")
    print()


# ==================== 完整流程示例 ====================

def complete_workflow_example():
    """完整的向量存储工作流"""
    print("【13. 完整工作流：加载 → 分块 → 向量化 → 存储 → 检索】")

    # 1. 准备长文档
    long_text = """
# LangChain 完整指南

## 简介
LangChain 是一个强大的框架，用于开发由大型语言模型 (LLM) 驱动的应用程序。
它提供了一套工具和抽象，使得构建复杂的 AI 应用变得简单。

## 核心概念

### Models
LangChain 支持多种语言模型，包括 OpenAI、Anthropic、Hugging Face 等。
通过统一的接口，你可以轻松切换不同的模型。

### Prompts
提示词工程是 LLM 应用的关键。LangChain 提供了强大的提示词模板系统，
支持变量替换、示例选择等高级功能。

### Chains
链允许你将多个组件连接在一起，创建复杂的处理流程。
最简单的链是 LLMChain，将提示词和模型连接起来。

### RAG
检索增强生成是 LangChain 的重要应用场景。
通过结合向量存储和语言模型，可以构建强大的知识问答系统。

## 最佳实践
1. 始终使用环境变量存储 API 密钥
2. 对长文档进行适当的分块
3. 选择合适的向量数据库
4. 实施错误处理和重试机制
"""

    # 2. 分块
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30,
    )
    docs = splitter.create_documents([long_text])
    print(f"文档分块：{len(docs)} 块")

    # 3. 向量化并存储
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    print(f"向量存储创建完成")

    # 4. 创建检索器
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 5. 检索测试
    queries = [
        "LangChain 是什么？",
        "如何使用提示词？",
        "RAG 有什么用？",
    ]

    print("\n检索测试：")
    for query in queries:
        results = retriever.invoke(query)
        print(f"\n问：{query}")
        print(f"答：{results[0].page_content[:100]}...")

    print()


# ==================== 演示所有功能 ====================

def demo_all():
    """演示所有向量存储功能"""
    print("=" * 60)
    print("Embeddings & Vector Stores 功能演示")
    print("=" * 60)
    print()

    embeddings_basic_example()
    embeddings_batch_example()
    embeddings_similarity_example()
    faiss_basic_example()
    faiss_with_scores_example()
    faiss_persistence_example()
    chroma_basic_example()
    chroma_persistence_example()
    mmr_search_example()
    retriever_example()
    retriever_with_filter_example()
    add_delete_documents_example()
    complete_workflow_example()

    print("=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    demo_all()
