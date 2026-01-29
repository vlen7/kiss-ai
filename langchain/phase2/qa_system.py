"""
第二阶段综合项目：本地知识库问答系统

功能：
1. 支持多种文档格式（TXT、PDF、Markdown）
2. 自动文档分块和向量化
3. 基于 RAG 的智能问答
4. 引用来源追溯
5. 交互式问答对话
6. 知识库管理（添加、删除文档）
"""

import os
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from pydantic import BaseModel, Field

load_dotenv()


def get_model() -> ChatOpenAI:
    """获取 LLM 模型"""
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
    )


def get_embeddings() -> OpenAIEmbeddings:
    """获取 Embeddings 模型"""
    return OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
    )


# ==================== 数据模型 ====================

@dataclass
class QAResult:
    """问答结果"""
    question: str
    answer: str
    sources: List[str]
    confidence: str


class StructuredAnswer(BaseModel):
    """结构化回答"""
    answer: str = Field(description="问题的回答")
    confidence: str = Field(description="回答的置信度：高/中/低")
    key_points: List[str] = Field(description="回答的要点列表")
    sources_used: List[int] = Field(description="使用的来源编号列表（从1开始）")


# ==================== 知识库类 ====================

class KnowledgeBase:
    """知识库管理类"""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        self.embeddings = get_embeddings()
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore: Optional[FAISS] = None
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )

    def _load_file(self, file_path: str) -> List[Document]:
        """根据文件类型加载文档"""
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".md":
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            raise ValueError(f"不支持的文件类型：{ext}")

        return loader.load()

    def add_documents(self, file_paths: List[str]) -> int:
        """添加文档到知识库"""
        all_docs = []

        for file_path in file_paths:
            try:
                docs = self._load_file(file_path)
                # 添加来源信息
                for doc in docs:
                    doc.metadata["source"] = os.path.basename(file_path)
                all_docs.extend(docs)
                print(f"✓ 加载文件：{file_path}")
            except Exception as e:
                print(f"✗ 加载失败：{file_path} - {e}")

        if not all_docs:
            return 0

        # 分块
        chunks = self.splitter.split_documents(all_docs)

        # 添加到向量存储
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vectorstore.add_documents(chunks)

        # 持久化
        if self.persist_directory:
            self.vectorstore.save_local(self.persist_directory)

        return len(chunks)

    def add_texts(self, texts: List[str], source: str = "manual_input") -> int:
        """直接添加文本到知识库"""
        docs = [
            Document(page_content=text, metadata={"source": source})
            for text in texts
        ]

        chunks = self.splitter.split_documents(docs)

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vectorstore.add_documents(chunks)

        if self.persist_directory:
            self.vectorstore.save_local(self.persist_directory)

        return len(chunks)

    def load(self) -> bool:
        """从持久化目录加载知识库"""
        if not self.persist_directory or not os.path.exists(self.persist_directory):
            return False

        try:
            self.vectorstore = FAISS.load_local(
                self.persist_directory,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            return True
        except Exception as e:
            print(f"加载知识库失败：{e}")
            return False

    def search(self, query: str, k: int = 4) -> List[Document]:
        """搜索相关文档"""
        if self.vectorstore is None:
            return []
        return self.vectorstore.similarity_search(query, k=k)

    def search_with_scores(self, query: str, k: int = 4) -> List[tuple]:
        """搜索并返回相似度分数"""
        if self.vectorstore is None:
            return []
        return self.vectorstore.similarity_search_with_score(query, k=k)

    def get_retriever(self, k: int = 4):
        """获取检索器"""
        if self.vectorstore is None:
            raise ValueError("知识库为空，请先添加文档")
        return self.vectorstore.as_retriever(search_kwargs={"k": k})


# ==================== RAG 问答链 ====================

class RAGQASystem:
    """RAG 问答系统"""

    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        self.model = get_model()
        self.qa_prompt = ChatPromptTemplate.from_template("""
你是一个知识库问答助手。请根据以下提供的上下文信息回答用户的问题。

要求：
1. 只基于提供的上下文回答，不要编造信息
2. 如果上下文中没有相关信息，请诚实说明
3. 回答要简洁、准确、有条理
4. 如果有多个来源，可以综合多个来源的信息

上下文信息：
{context}

用户问题：{question}

回答：
""")

    def _format_docs(self, docs: List[Document]) -> str:
        """格式化文档列表"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "未知来源")
            formatted.append(f"[来源{i}: {source}]\n{doc.page_content}")
        return "\n\n".join(formatted)

    def ask(self, question: str, k: int = 4) -> QAResult:
        """简单问答"""
        # 检索相关文档
        docs = self.kb.search(question, k=k)

        if not docs:
            return QAResult(
                question=question,
                answer="知识库中没有找到相关信息。",
                sources=[],
                confidence="低"
            )

        # 构建上下文
        context = self._format_docs(docs)

        # 生成回答
        chain = self.qa_prompt | self.model | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})

        # 提取来源
        sources = list(set(doc.metadata.get("source", "未知") for doc in docs))

        return QAResult(
            question=question,
            answer=answer,
            sources=sources,
            confidence="高" if len(docs) >= 2 else "中"
        )

    def ask_with_chain(self, question: str, k: int = 4) -> str:
        """使用 LCEL 链式调用"""
        retriever = self.kb.get_retriever(k=k)

        # 构建 RAG 链
        rag_chain = (
            {"context": retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.qa_prompt
            | self.model
            | StrOutputParser()
        )

        return rag_chain.invoke(question)

    def ask_structured(self, question: str, k: int = 4) -> StructuredAnswer:
        """结构化问答，返回详细信息"""
        docs = self.kb.search(question, k=k)

        if not docs:
            return StructuredAnswer(
                answer="知识库中没有找到相关信息。",
                confidence="低",
                key_points=[],
                sources_used=[]
            )

        context = self._format_docs(docs)

        structured_prompt = ChatPromptTemplate.from_template("""
你是一个知识库问答助手。请根据以下上下文信息回答用户的问题。

上下文信息：
{context}

用户问题：{question}

请提供结构化的回答，包括：
1. 详细的回答内容
2. 回答的置信度（高/中/低）
3. 关键要点列表
4. 使用了哪些来源（编号列表）
""")

        structured_model = self.model.with_structured_output(StructuredAnswer)
        chain = structured_prompt | structured_model

        return chain.invoke({"context": context, "question": question})

    def ask_with_sources(self, question: str, k: int = 4) -> Dict[str, Any]:
        """问答并返回引用来源"""
        docs_with_scores = self.kb.search_with_scores(question, k=k)

        if not docs_with_scores:
            return {
                "question": question,
                "answer": "知识库中没有找到相关信息。",
                "sources": []
            }

        # 格式化文档
        docs = [doc for doc, _ in docs_with_scores]
        context = self._format_docs(docs)

        # 生成回答
        chain = self.qa_prompt | self.model | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})

        # 整理来源信息
        sources = []
        for doc, score in docs_with_scores:
            sources.append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "source": doc.metadata.get("source", "未知"),
                "similarity": round(1 - score, 4)  # 转换为相似度
            })

        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }


# ==================== 交互式问答 ====================

def interactive_qa(qa_system: RAGQASystem):
    """交互式问答对话"""
    print("\n" + "=" * 50)
    print("知识库问答系统")
    print("=" * 50)
    print("\n命令：")
    print("  直接输入问题进行问答")
    print("  输入 'sources' 查看带来源的详细回答")
    print("  输入 'quit' 退出")
    print()

    while True:
        user_input = input("问：").strip()

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("感谢使用，再见！")
            break

        if user_input.lower() == "sources":
            question = input("请输入问题：").strip()
            if question:
                print("\n查询中...")
                result = qa_system.ask_with_sources(question)
                print(f"\n答：{result['answer']}")
                print("\n引用来源：")
                for i, source in enumerate(result['sources'], 1):
                    print(f"  [{i}] {source['source']} (相似度: {source['similarity']:.2%})")
                    print(f"      {source['content'][:100]}...")
            print()
            continue

        # 普通问答
        print("\n查询中...")
        result = qa_system.ask(user_input)

        print(f"\n答：{result.answer}")
        if result.sources:
            print(f"\n来源：{', '.join(result.sources)}")
        print(f"置信度：{result.confidence}")
        print()


# ==================== 演示功能 ====================

def create_demo_knowledge_base() -> KnowledgeBase:
    """创建演示用知识库"""
    kb = KnowledgeBase(
        persist_directory="/tmp/demo_knowledge_base",
        chunk_size=300,
        chunk_overlap=30
    )

    # 添加示例文档
    sample_docs = [
        """
# LangChain 框架介绍

LangChain 是一个用于开发由大型语言模型 (LLM) 驱动的应用程序的框架。

## 核心特点

1. **模块化设计**：LangChain 采用模块化架构，各组件可以独立使用或组合使用。

2. **统一接口**：提供统一的接口来连接不同的 LLM 提供商，如 OpenAI、Anthropic 等。

3. **链式组合**：通过 Chain 机制，可以将多个组件串联起来构建复杂的处理流程。

4. **内置工具**：提供丰富的内置工具，包括文档加载器、文本分割器、向量存储等。

## 主要组件

- **Models**：语言模型的封装
- **Prompts**：提示词模板系统
- **Chains**：组件链接
- **Agents**：智能代理
- **Memory**：对话记忆
""",
        """
# RAG 检索增强生成

RAG（Retrieval Augmented Generation）是一种结合检索和生成的技术。

## 工作原理

1. **索引阶段**：
   - 将文档分割成小块
   - 使用 Embedding 模型将文本块向量化
   - 存储到向量数据库中

2. **检索阶段**：
   - 将用户问题向量化
   - 在向量数据库中搜索相似文档
   - 返回最相关的文档块

3. **生成阶段**：
   - 将检索到的文档作为上下文
   - 结合用户问题生成提示词
   - 调用 LLM 生成最终回答

## 优势

- 减少幻觉：基于真实文档回答
- 知识实时更新：无需重新训练模型
- 可追溯：可以引用信息来源
- 成本效益：比微调模型更经济
""",
        """
# 向量数据库介绍

向量数据库是专门用于存储和检索高维向量的数据库系统。

## 常用向量数据库

### FAISS
- Facebook AI 开发的向量搜索库
- 支持 CPU 和 GPU 加速
- 适合本地部署和快速原型开发

### Chroma
- 轻量级向量数据库
- 专为 AI 应用设计
- 支持持久化存储
- 易于安装和使用

### Pinecone
- 云原生向量数据库
- 完全托管服务
- 支持大规模数据
- 适合生产环境

### Milvus
- 开源向量数据库
- 支持多种索引类型
- 高性能和可扩展性
- 适合企业级应用

## 选择建议

- 开发测试：FAISS 或 Chroma
- 小规模生产：Chroma
- 大规模生产：Pinecone 或 Milvus
"""
    ]

    # 添加文本到知识库
    total_chunks = 0
    for i, doc in enumerate(sample_docs):
        chunks = kb.add_texts([doc], source=f"sample_doc_{i+1}.md")
        total_chunks += chunks

    print(f"知识库创建完成，共 {total_chunks} 个文档块")
    return kb


def demo_basic_qa():
    """基础问答演示"""
    print("=" * 60)
    print("知识库问答系统 - 功能演示")
    print("=" * 60)
    print()

    # 创建知识库
    print("【1. 创建知识库】")
    kb = create_demo_knowledge_base()
    print()

    # 创建问答系统
    qa = RAGQASystem(kb)

    # 基础问答
    print("【2. 基础问答】")
    questions = [
        "什么是 LangChain？",
        "RAG 的工作原理是什么？",
        "有哪些常用的向量数据库？",
    ]

    for question in questions:
        result = qa.ask(question)
        print(f"\n问：{question}")
        print(f"答：{result.answer}")
        print(f"来源：{', '.join(result.sources)}")
        print("-" * 40)

    print()


def demo_structured_qa():
    """结构化问答演示"""
    print("【3. 结构化问答】")

    kb = KnowledgeBase()
    kb.add_texts([
        "Python 是一种高级编程语言，以简洁易读著称。",
        "Python 广泛用于 Web 开发、数据科学、人工智能等领域。",
        "Python 的主要特点包括：动态类型、自动内存管理、丰富的库生态。",
    ], source="python_intro.txt")

    qa = RAGQASystem(kb)

    question = "Python 语言有什么特点和应用场景？"
    result = qa.ask_structured(question)

    print(f"\n问：{question}")
    print(f"\n答：{result.answer}")
    print(f"\n置信度：{result.confidence}")
    print(f"\n关键要点：")
    for point in result.key_points:
        print(f"  • {point}")
    print(f"\n使用来源：{result.sources_used}")
    print()


def demo_sources_qa():
    """带来源追溯的问答演示"""
    print("【4. 带来源追溯的问答】")

    kb = create_demo_knowledge_base()
    qa = RAGQASystem(kb)

    question = "如何选择合适的向量数据库？"
    result = qa.ask_with_sources(question)

    print(f"\n问：{question}")
    print(f"\n答：{result['answer']}")
    print(f"\n引用来源（共 {len(result['sources'])} 个）：")
    for i, source in enumerate(result['sources'], 1):
        print(f"\n  [{i}] {source['source']}")
        print(f"      相似度：{source['similarity']:.2%}")
        print(f"      内容：{source['content'][:100]}...")
    print()


def demo_all():
    """运行所有演示"""
    demo_basic_qa()
    demo_structured_qa()
    demo_sources_qa()

    print("=" * 60)
    print("演示完成！")
    print("=" * 60)


# ==================== 主程序 ====================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "-i":
        # 交互模式
        print("正在初始化知识库...")
        kb = create_demo_knowledge_base()
        qa = RAGQASystem(kb)
        interactive_qa(qa)
    else:
        # 演示模式
        demo_all()
