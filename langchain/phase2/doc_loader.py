"""
第二阶段：Document Loaders 文档加载示例

学习要点：
1. TextLoader - 加载纯文本文件
2. PyPDFLoader - 加载 PDF 文件
3. WebBaseLoader - 加载网页内容
4. DirectoryLoader - 批量加载目录下的文件
5. Text Splitters - 文档分块策略
"""

import os
from dotenv import load_dotenv
from typing import List

# Document Loaders
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    WebBaseLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)

# Text Splitters
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

from langchain_core.documents import Document

load_dotenv()


# ==================== 文本文件加载 ====================

def load_text_file(file_path: str) -> List[Document]:
    """加载纯文本文件"""
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    return documents


def load_text_example():
    """文本加载示例"""
    print("【1. 文本文件加载】")

    # 创建示例文本文件
    sample_text = """
LangChain 是一个用于构建 LLM 应用的框架。

它提供了以下核心功能：
1. Models - 各种 LLM 的统一接口
2. Prompts - 提示词模板管理
3. Chains - 链式调用组合
4. Agents - 智能代理系统
5. Memory - 对话记忆管理

LangChain 让开发者能够快速构建强大的 AI 应用。
"""

    # 写入示例文件
    sample_file = "/tmp/sample_doc.txt"
    with open(sample_file, "w", encoding="utf-8") as f:
        f.write(sample_text)

    # 加载文件
    docs = load_text_file(sample_file)

    print(f"加载了 {len(docs)} 个文档")
    print(f"文档内容预览：{docs[0].page_content[:100]}...")
    print(f"元数据：{docs[0].metadata}")
    print()


# ==================== PDF 文件加载 ====================

def load_pdf_file(file_path: str) -> List[Document]:
    """加载 PDF 文件（每页作为一个文档）"""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents


def load_pdf_example():
    """PDF 加载示例（需要实际 PDF 文件）"""
    print("【2. PDF 文件加载】")
    print("示例代码：")
    print("""
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader("document.pdf")
    pages = loader.load()

    # 每一页是一个 Document 对象
    for i, page in enumerate(pages):
        print(f"第 {i+1} 页: {page.page_content[:100]}...")
    """)
    print()


# ==================== 网页加载 ====================

def load_webpage(url: str) -> List[Document]:
    """加载网页内容"""
    loader = WebBaseLoader(url)
    documents = loader.load()
    return documents


def load_webpage_example():
    """网页加载示例"""
    print("【3. 网页内容加载】")
    print("示例代码：")
    print("""
    from langchain_community.document_loaders import WebBaseLoader

    loader = WebBaseLoader("https://python.langchain.com/docs/")
    docs = loader.load()

    print(f"网页标题: {docs[0].metadata.get('title', 'N/A')}")
    print(f"内容预览: {docs[0].page_content[:200]}...")
    """)
    print()


# ==================== 目录批量加载 ====================

def load_directory(dir_path: str, glob_pattern: str = "**/*.txt") -> List[Document]:
    """批量加载目录下的文件"""
    loader = DirectoryLoader(
        dir_path,
        glob=glob_pattern,
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    return documents


def load_directory_example():
    """目录加载示例"""
    print("【4. 目录批量加载】")

    # 创建示例目录和文件
    import os
    sample_dir = "/tmp/sample_docs"
    os.makedirs(sample_dir, exist_ok=True)

    # 创建多个示例文件
    files = {
        "intro.txt": "LangChain 简介：这是一个强大的 LLM 应用开发框架。",
        "features.txt": "核心特性：Models, Prompts, Chains, Agents, Memory。",
        "usage.txt": "使用场景：聊天机器人、文档问答、代码助手等。",
    }

    for filename, content in files.items():
        with open(os.path.join(sample_dir, filename), "w", encoding="utf-8") as f:
            f.write(content)

    # 加载目录
    docs = load_directory(sample_dir, "*.txt")

    print(f"加载了 {len(docs)} 个文档")
    for doc in docs:
        print(f"  - {doc.metadata['source']}: {doc.page_content[:50]}...")
    print()


# ==================== 文档分块 - Character Splitter ====================

def split_by_character(text: str, chunk_size: int = 200, chunk_overlap: int = 20) -> List[Document]:
    """按字符数分块"""
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.create_documents([text])
    return chunks


def character_splitter_example():
    """字符分块示例"""
    print("【5. Character Text Splitter（字符分块）】")

    long_text = """
LangChain 是一个用于开发由语言模型驱动的应用程序的框架。

主要特点：
1. 数据感知：将语言模型连接到其他数据源
2. 具有代理性：允许语言模型与其环境交互

核心模块包括：
- Models：各种 LLM 和 Chat Model 的标准接口
- Prompts：提示词模板、示例选择器
- Indexes：文档加载器、向量存储、检索器
- Chains：将多个组件组合成复杂的管道
- Agents：让 LLM 决定采取什么行动
- Memory：在多次调用之间保持状态

LangChain 使开发者能够构建强大的端到端应用，包括聊天机器人、问答系统、摘要生成等。
"""

    chunks = split_by_character(long_text, chunk_size=150, chunk_overlap=20)

    print(f"原文长度：{len(long_text)} 字符")
    print(f"分块数量：{len(chunks)} 块")
    print()
    for i, chunk in enumerate(chunks):
        print(f"【块 {i+1}】({len(chunk.page_content)} 字符)")
        print(chunk.page_content[:80] + "..." if len(chunk.page_content) > 80 else chunk.page_content)
        print()


# ==================== 文档分块 - Recursive Splitter ====================

def split_recursive(text: str, chunk_size: int = 200, chunk_overlap: int = 20) -> List[Document]:
    """递归分块（推荐方式）"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    chunks = splitter.create_documents([text])
    return chunks


def recursive_splitter_example():
    """递归分块示例"""
    print("【6. Recursive Character Splitter（递归分块 - 推荐）】")

    long_text = """
什么是 RAG？

RAG（Retrieval Augmented Generation）即检索增强生成，是一种结合检索和生成的技术。

工作原理：
1. 检索阶段：根据用户查询，从知识库中检索相关文档
2. 增强阶段：将检索到的文档作为上下文传递给 LLM
3. 生成阶段：LLM 基于上下文生成回答

优势：
- 减少幻觉：基于真实文档回答
- 知识更新：无需重新训练模型
- 可追溯：可以引用信息来源
- 成本低：比微调模型便宜

应用场景：企业知识库问答、文档助手、客服机器人等。
"""

    chunks = split_recursive(long_text, chunk_size=150, chunk_overlap=30)

    print(f"原文长度：{len(long_text)} 字符")
    print(f"分块数量：{len(chunks)} 块")
    print()

    # 展示分块边界更自然
    for i, chunk in enumerate(chunks):
        print(f"【块 {i+1}】({len(chunk.page_content)} 字符)")
        content = chunk.page_content.strip()
        print(content[:100] + "..." if len(content) > 100 else content)
        print()


# ==================== 文档分块 - Token Splitter ====================

def split_by_tokens(text: str, chunk_size: int = 100, chunk_overlap: int = 10) -> List[Document]:
    """按 Token 数分块"""
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.create_documents([text])
    return chunks


def token_splitter_example():
    """Token 分块示例"""
    print("【7. Token Text Splitter（按 Token 分块）】")
    print("按 Token 数量而非字符数分块，更适合控制 LLM 输入长度")
    print()
    print("示例代码：")
    print("""
    from langchain_text_splitters import TokenTextSplitter

    splitter = TokenTextSplitter(
        chunk_size=100,      # 每块最多 100 tokens
        chunk_overlap=10,    # 重叠 10 tokens
    )

    chunks = splitter.create_documents([long_text])
    """)
    print()


# ==================== 加载文档并分块的完整流程 ====================

def load_and_split_example():
    """完整的加载和分块流程"""
    print("【8. 完整流程：加载 + 分块】")

    # 1. 创建示例文档
    sample_text = """
# LangChain RAG 教程

## 什么是 RAG

RAG 是检索增强生成的缩写，它通过检索外部知识来增强 LLM 的回答能力。

## 核心组件

### Document Loaders
文档加载器用于从各种来源加载文档，包括：
- 文本文件 (TextLoader)
- PDF 文件 (PyPDFLoader)
- 网页 (WebBaseLoader)
- 数据库 (SQLLoader)

### Text Splitters
文本分块器将长文档分割成小块：
- CharacterTextSplitter: 按字符分块
- RecursiveCharacterTextSplitter: 递归分块
- TokenTextSplitter: 按 Token 分块

### Vector Stores
向量存储用于存储和检索向量化的文档：
- FAISS: Facebook 的向量搜索库
- Chroma: 轻量级向量数据库
- Pinecone: 云原生向量数据库

## 构建 RAG 应用

完整的 RAG 应用包含以下步骤：
1. 加载文档
2. 分块处理
3. 向量化存储
4. 检索相关文档
5. 生成回答
"""

    sample_file = "/tmp/langchain_rag_tutorial.txt"
    with open(sample_file, "w", encoding="utf-8") as f:
        f.write(sample_text)

    # 2. 加载文档
    loader = TextLoader(sample_file, encoding="utf-8")
    documents = loader.load()
    print(f"加载了 {len(documents)} 个文档，总字符数：{len(documents[0].page_content)}")

    # 3. 分块处理
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    print(f"分块后得到 {len(chunks)} 个文档块")
    print()

    # 4. 展示分块结果
    for i, chunk in enumerate(chunks[:3]):  # 只展示前3块
        print(f"【块 {i+1}】")
        print(f"内容：{chunk.page_content[:100]}...")
        print(f"元数据：{chunk.metadata}")
        print()

    print(f"...省略剩余 {len(chunks)-3} 块")
    print()

    return chunks


# ==================== 演示所有功能 ====================

def demo_all():
    """演示所有文档加载功能"""
    print("=" * 60)
    print("Document Loaders & Text Splitters 功能演示")
    print("=" * 60)
    print()

    load_text_example()
    load_pdf_example()
    load_webpage_example()
    load_directory_example()
    character_splitter_example()
    recursive_splitter_example()
    token_splitter_example()
    load_and_split_example()

    print("=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    demo_all()
