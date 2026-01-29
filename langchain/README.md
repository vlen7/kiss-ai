⏺ LangChain 学习计划

  第一阶段：基础概念

  学习内容：
  1. Models - LLM 和 Chat Model 的使用
  2. Prompts - 提示词模板和管理
  3. Output Parsers - 结构化输出解析
  4. LCEL (LangChain Expression Language) - 链式调用语法

  文档：
  - https://python.langchain.com/docs/concepts/
  - https://python.langchain.com/docs/how_to/#prompt-templates
  - https://python.langchain.com/docs/concepts/lcel/

  实践项目： 构建一个多语言翻译助手
  langchain/
  ├── 01_basics/
  │   ├── chat_model.py      # 基础对话
  │   ├── prompt_template.py # 提示词模板
  │   └── translator.py      # 翻译助手

  ---
  第二阶段：RAG (检索增强生成)

  学习内容：
  1. Document Loaders - 加载 PDF、网页、文本等
  2. Text Splitters - 文档分块策略
  3. Embeddings - 文本向量化
  4. Vector Stores - 向量数据库（FAISS、Chroma）
  5. Retrievers - 检索策略

  文档：
  - https://python.langchain.com/docs/tutorials/rag/
  - https://python.langchain.com/docs/how_to/#document-loaders
  - https://python.langchain.com/docs/how_to/#text-splitters

  实践项目： 构建本地知识库问答系统
  langchain/
  ├── 02_rag/
  │   ├── doc_loader.py      # 文档加载
  │   ├── vectorstore.py     # 向量存储
  │   └── qa_system.py       # 知识库问答

  ---
  第三阶段：Agents 和 Tools

  学习内容：
  1. Tools - 自定义工具开发
  2. ReAct Agent - 推理+行动模式
  3. Tool Calling - 函数调用
  4. Multi-tool Agent - 多工具协作

  文档：
  - https://python.langchain.com/docs/concepts/agents/
  - https://python.langchain.com/docs/how_to/#tools
  - https://python.langchain.com/docs/tutorials/agents/

  实践项目： 构建智能助手（搜索+计算+代码执行）
  langchain/
  ├── 03_agents/
  │   ├── custom_tools.py    # 自定义工具
  │   ├── web_agent.py       # 联网搜索 Agent
  │   └── code_agent.py      # 代码执行 Agent

  ---
  第四阶段：LangGraph (复杂工作流)

  学习内容：
  1. StateGraph - 状态图定义
  2. Nodes & Edges - 节点和边
  3. Conditional Routing - 条件路由
  4. Human-in-the-loop - 人机协作
  5. Persistence - 状态持久化

  文档：
  - https://langchain-ai.github.io/langgraph/
  - https://langchain-ai.github.io/langgraph/tutorials/

  实践项目： 构建多步骤工作流（客服机器人）
  langchain/
  ├── 04_langgraph/
  │   ├── simple_graph.py    # 基础状态图
  │   ├── branching.py       # 条件分支
  │   └── customer_bot.py    # 客服机器人

  ---
  第五阶段：Memory (对话记忆)

  学习内容：
  1. Buffer Memory - 简单缓冲记忆
  2. Summary Memory - 摘要记忆
  3. Conversation History - 对话历史管理
  4. Persistent Memory - 持久化存储

  文档：
  - https://python.langchain.com/docs/how_to/#memory
  - https://langchain-ai.github.io/langgraph/concepts/memory/

  实践项目： 构建有记忆的聊天机器人
  langchain/
  ├── 05_memory/
  │   ├── buffer_memory.py   # 缓冲记忆
  │   ├── summary_memory.py  # 摘要记忆
  │   └── chatbot.py         # 完整聊天机器人

  ---
  第六阶段：生产部署

  学习内容：
  1. LangSmith - 调试、监控、追踪
  2. LangServe - API 部署
  3. Streaming - 流式输出
  4. Error Handling - 错误处理和重试

  文档：
  - https://docs.smith.langchain.com/
  - https://python.langchain.com/docs/langserve/

  实践项目： 部署 RAG API 服务
  langchain/
  ├── 06_production/
  │   ├── langsmith_demo.py  # 追踪和监控
  │   ├── api_server.py      # FastAPI 服务
  │   └── streaming.py       # 流式输出

  ---
  推荐学习顺序
  ┌──────┬─────────────────┬──────────┐
  │ 阶段 │      主题       │   难度   │
  ├──────┼─────────────────┼──────────┤
  │ 1    │ 基础概念 + LCEL │ ⭐       │
  ├──────┼─────────────────┼──────────┤
  │ 2    │ RAG 知识库      │ ⭐⭐     │
  ├──────┼─────────────────┼──────────┤
  │ 3    │ Agents          │ ⭐⭐⭐   │
  ├──────┼─────────────────┼──────────┤
  │ 4    │ LangGraph       │ ⭐⭐⭐⭐ │
  ├──────┼─────────────────┼──────────┤
  │ 5    │ Memory          │ ⭐⭐     │
  ├──────┼─────────────────┼──────────┤
  │ 6    │ 生产部署        │ ⭐⭐⭐   │
  └──────┴─────────────────┴──────────┘
  ---
  综合实战项目

  最终项目：AI 文档助手
  - RAG + Agent + Memory + LangGraph 综合应用
  - 功能：上传文档、智能问答、联网搜索、对话记忆、工作流编排