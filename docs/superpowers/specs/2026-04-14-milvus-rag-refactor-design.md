# Milvus RAG 知识库重构设计文档

**日期**: 2026-04-14
**作者**: Claude Code
**状态**: 设计阶段 v2
**目标**: 简化知识库查询为直接 Milvus 向量检索，使用 LangChain RAG 标准流程，学习 vLLM 部署

**变更历史**:
- v1 (2026-04-14): 初始设计
- v2 (2026-04-14): 添加 vLLM Embedding 部署、AppContext 全局单例、任务解耦

---

## 1. 背景与目标

### 1.1 当前状态

项目已有完整的知识库架构：
- `KnowledgeBaseService`: 封装 Milvus 向量存储
- `create_kb_multi_tool_workflow`: 复杂的多工具工作流
- 支持多种检索源: PostgreSQL pgvector (优先) + Milvus (兜底) + 外部搜索
- 包含 guardrails、router、多路由决策等复杂逻辑

**问题**:
- 配置被注释掉，未启用
- 流程过于复杂，维护成本高
- PostgreSQL pgvector 和 Milvus 混合使用，增加复杂度

### 1.2 重构目标

**简化为单一 Milvus 向量检索流程**:
- 移除复杂的多工具工作流
- 移除 PostgreSQL pgvector 检索
- 使用 LangChain 标准 RAG 流程
- **使用 vLLM 部署 BGE-M3 Embedding** (学习目标)
- **在 AppContext 中注册全局 Retriever 单例**
- 每个模块都有完整的单元测试
- **任务解耦，MVP 优先**

---

## 2. 整体架构

### 2.1 架构图

```
┌─────────────────────────────────────────────────────────┐
│                   基础设施层                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  vLLM Embed  │  │   Milvus     │  │  vLLM LLM    │ │
│  │  (BGE-M3)    │  │  Vector DB   │  │  (Qwen)      │ │
│  │  Port: 8001  │  │  Port: 19530 │  │  Port: 8000  │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                   应用层 (AppContext)                    │
│  ┌────────────────────────────────────────────────────┐ │
│  │  GlobalRetriever (单例)                            │ │
│  │  - EmbeddingClient (vLLM)                          │ │
│  │  - RerankerModel (SentenceTransformers)           │ │
│  │  - MilvusClient                                    │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                   业务层 (LangGraph)                     │
│  用户查询 → Router → create_kb_query 节点               │
│                     ↓                                    │
│              RAG Chain                                  │
│                     ↓                                    │
│  返回 AIMessage(content, sources)                       │
└─────────────────────────────────────────────────────────┘
```

### 2.2 核心组件

| 组件 | 技术选型 | 部署方式 | 用途 |
|------|---------|---------|------|
| **Embedding** | BAAI/bge-m3 | **vLLM Server** | 查询和文档向量化 (1024维) |
| **向量存储** | Milvus 2.6+ | Docker Compose | 存储和检索向量 |
| **重排序** | BAAI/bge-reranker-v2-m3 | SentenceTransformers (本地) | 提高检索准确度 |
| **全局检索器** | GlobalRetriever | AppContext 单例 | 统一检索接口 |
| **RAG 框架** | LangChain | 代码模块 | 标准化 RAG 流程 |
| **生成模型** | Qwen LLM | vLLM Server (已有) | 生成最终回答 |

### 2.3 全局单例设计

**AppContext 中的 GlobalRetriever**:

```python
# muscleking/app/core/context.py

class GlobalContext:
    """全局上下文，管理共享资源"""

    def __init__(self):
        self._retriever: Optional[GlobalRetriever] = None

    @property
    def retriever(self) -> GlobalRetriever:
        """获取全局检索器单例"""
        if self._retriever is None:
            self._retriever = GlobalRetriever()
        return self._retriever

# 全局访问
ctx = get_global_context()
retriever = ctx.retriever  # 自动初始化并复用
```

### 2.3 检索策略

```
召回阶段: Milvus 向量检索 → top-20 候选文档
精排阶段: BGE-Reranker 重排序 → top-5 最终结果
```

---

## 3. 详细设计

### 3.1 vLLM Embedding 部署 (学习重点) 🎯

#### 3.1.1 部署架构

**vLLM Embedding Server** (独立容器):

```yaml
# docker-compose.yml
services:
  vllm-embedding:
    image: vllm/vllm:v0.6.4
    container_name: vllm-embedding
    ports:
      - "8001:8000"  # 映射到 8001 端口
    environment:
      - MODEL_NAME=BAAI/bge-m3
      - MAX_MODEL_LEN=8192
      - GPU_MEMORY_UTILIZATION=0.9
    command: >
      --model BAAI/bge-m3
      --port 8000
      --embedding-mode True
      --max-model-len 8192
      --gpu-memory-utilization 0.9
      --dtype auto
      --trust-remote-code
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    restart: unless-stopped
```

#### 3.1.2 启动脚本

```bash
#!/bin/bash
# scripts/start_vllm_embedding.sh

echo "🚀 Starting vLLM Embedding Server (BGE-M3)..."

# 检查 GPU
nvidia-smi

# 启动 vLLM Embedding Server
docker run -d \
  --name vllm-embedding \
  --gpus all \
  -p 8001:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm:v0.6.4 \
  --model BAAI/bge-m3 \
  --embedding-mode True \
  --max-model-len 8192 \
  --trust-remote-code

echo "⏳ Waiting for vLLM to start..."
sleep 30

# 健康检查
curl -s http://localhost:8001/health || echo "❌ Health check failed"

echo "✅ vLLM Embedding Server started on port 8001"
```

#### 3.1.3 Embedding Client

```python
# muscleking/app/rag/embeddings.py

from openai import OpenAI
from typing import List
import numpy as np

class VLLMEmbeddingClient:
    """vLLM Embedding Client"""

    def __init__(self, base_url: str = "http://localhost:8001/v1"):
        self.client = OpenAI(
            base_url=base_url,
            api_key="dummy"  # vLLM 不验证
        )

    def embed_query(self, text: str) -> List[float]:
        """对单个查询进行 embedding"""
        response = self.client.embeddings.create(
            model="BAAI/bge-m3",
            input=text
        )
        return response.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量 embedding 文档"""
        response = self.client.embeddings.create(
            model="BAAI/bge-m3",
            input=texts
        )
        return [item.embedding for item in response.data]

    @property
    def dimension(self) -> int:
        """返回 embedding 维度"""
        return 1024  # BGE-M3 的维度
```

#### 3.1.4 验证脚本

```python
# scripts/test_vllm_embedding.py

import requests
import json

def test_vllm_embedding():
    """测试 vLLM Embedding 服务"""

    # 健康检查
    response = requests.get("http://localhost:8001/health")
    assert response.status_code == 200
    print("✅ Health check passed")

    # Embedding 测试
    payload = {
        "model": "BAAI/bge-m3",
        "input": ["如何锻炼胸肌？", "卧推的正确姿势"]
    }

    response = requests.post(
        "http://localhost:8001/v1/embeddings",
        json=payload
    )

    assert response.status_code == 200
    data = response.json()

    assert "data" in data
    assert len(data["data"]) == 2
    assert len(data["data"][0]["embedding"]) == 1024

    print(f"✅ Embedding test passed")
    print(f"   - Dimension: {len(data['data'][0]['embedding'])}")
    print(f"   - Sample values: {data['data'][0]['embedding'][:5]}")

if __name__ == "__main__":
    test_vllm_embedding()
```

#### 3.1.5 性能测试

```python
# scripts/benchmark_vllm_embedding.py

import time
import requests
import numpy as np

def benchmark_embedding():
    """测试 vLLM Embedding 性能"""

    client_url = "http://localhost:8001/v1/embeddings"

    # 测试不同 batch size
    batch_sizes = [1, 8, 16, 32, 64]
    texts = ["测试文本" * 20] * 64  # 固定文本

    results = []

    for batch_size in batch_sizes:
        batch_texts = texts[:batch_size]

        start = time.time()
        response = requests.post(
            client_url,
            json={
                "model": "BAAI/bge-m3",
                "input": batch_texts
            }
        )
        elapsed = time.time() - start

        throughput = batch_size / elapsed
        results.append({
            "batch_size": batch_size,
            "elapsed": elapsed,
            "throughput": throughput
        })

        print(f"Batch {batch_size:2d}: {elapsed:.3f}s → {throughput:.1f} docs/s")

    # 找到最佳 batch size
    best = max(results, key=lambda x: x["throughput"])
    print(f"\n🏆 最佳配置: batch_size={best['batch_size']}, throughput={best['throughput']:.1f} docs/s")

if __name__ == "__main__":
    benchmark_embedding()
```

#### 3.1.6 故障排查

```python
# scripts/debug_vllm_embedding.py

def debug_vllm_embedding():
    """调试 vLLM Embedding 服务"""

    import subprocess

    print("🔍 检查 vLLM 容器状态...")
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", "name=vllm-embedding"],
        capture_output=True,
        text=True
    )
    print(result.stdout)

    print("\n🔍 检查容器日志...")
    result = subprocess.run(
        ["docker", "logs", "--tail", "50", "vllm-embedding"],
        capture_output=True,
        text=True
    )
    print(result.stdout)

    print("\n🔍 检查 GPU 使用情况...")
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print(result.stdout)

    print("\n🔍 测试 API 连接...")
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        print(f"✅ API 响应: {response.status_code}")
    except Exception as e:
        print(f"❌ API 连接失败: {e}")

if __name__ == "__main__":
    debug_vllm_embedding()
```

### 3.2 Milvus 部署

**部署方式**: Docker Compose (Standalone 模式)

**组件**:
- `milvus-standalone`: 核心向量数据库
- `etcd`: 元数据存储 (Milvus 依赖)
- `minio`: 对象存储 (Milvus 依赖)

**配置** (`docker-compose.yml`):
```yaml
services:
  milvus-standalone:
    image: milvusdb/milvus:v2.6.0
    ports:
      - "19530:19530"  # gRPC
      - "9091:9091"    # Metrics
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    depends_on:
      - etcd
      - minio

  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    # ... etcd 配置

  minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    # ... minio 配置
```

**连接配置** (`settings.py`):
```python
MILVUS_HOST: str = "localhost"
MILVUS_PORT: int = 19530
MILVUS_COLLECTION: str = "fitness_knowledge"
```

### 3.2 Embedding 模型

**模型**: BAAI/bge-m3

**特性**:
- 支持 8192 token 长度 (适合长文档)
- 中英文混合效果好
- 1024 维向量，精度高

**配置**:
```python
EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
EMBEDDING_DIMENSION: int = 1024

# LangChain Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 32}
)
```

### 3.3 Reranker 配置

**模型**: BAAI/bge-reranker-v2-m3

**配置**:
```python
RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
ENABLE_RERANK: bool = True
RERANK_MAX_CANDIDATES: int = 20  # 召回候选数
RERANK_TOP_N: int = 5            # 最终返回数
RERANK_SCORE_THRESHOLD: float = 0.3  # 重排序分数阈值
```

### 3.4 代码结构

#### 目录结构

```
muscleking/
├── app/
│   ├── agents/
│   │   ├── lg_builder.py          # ✏️ 修改: 简化 create_kb_query
│   │   └── kb_workflow.py         # 🔒 保留: 可能被其他地方引用
│   ├── services/
│   │   ├── knowledge_base_service.py  # ✏️ 优化: 使用 LangChain VectorStore
│   │   └── vector_store.py        # ✅ 保持: Milvus 封装
│   ├── rag/                       # 🆕 新建: LangChain RAG 模块
│   │   ├── __init__.py
│   │   ├── retriever.py           # Milvus Retriever with Reranker
│   │   ├── chain.py               # RAG Chain (检索+生成)
│   │   └── prompts.py             # RAG Prompt 模板
│   ├── config/
│   │   └── settings.py            # ✏️ 启用配置
│   └── scripts/                   # 🆕 新建: 脚本目录
│       ├── deploy_milvus.py       # 部署 Milvus
│       ├── import_documents.py    # 批量导入文档
│       └── test_kb_query.py       # 测试查询功能
└── tests/                         # 🆕 新建: 测试目录
    ├── rag/
    │   ├── test_retriever.py      # 测试 retriever
    │   ├── test_chain.py          # 测试 RAG chain
    │   └── test_embeddings.py     # 测试 embedding
    ├── services/
    │   ├── test_vector_store.py   # 测试 Milvus 连接
    │   └── test_kb_service.py     # 测试 KB service
    ├── agents/
    │   └── test_kb_query.py       # 测试 kb_query 节点
    └── integration/
        └── test_e2e_rag.py        # 端到端测试
```

#### 核心代码修改

**1. `lg_builder.py` - 简化 `create_kb_query`**

```python
async def create_kb_query(
    state: AgentState, *, config: RunnableConfig
) -> Dict[str, List[BaseMessage]]:
    """简化的知识库查询: 直接使用 LangChain Milvus RAG

    流程:
    1. 提取用户查询
    2. 创建 RAG Chain
    3. 执行检索和生成
    4. 返回 AIMessage with sources
    """
    from muscleking.app.rag.chain import create_rag_chain

    # 提取查询
    last_message = state.messages[-1].content if state.messages else ""
    if not last_message.strip():
        return {
            "messages": [
                AIMessage(content="请告诉我具体的问题，我才能帮您查询知识库。")
            ]
        }

    # 提取配置参数
    config_opts = _extract_configurable(config)
    top_k = config_opts.get("kb_top_k", 5)

    # 创建 RAG Chain
    rag_chain = create_rag_chain(top_k=top_k)

    # 执行查询
    try:
        result = await rag_chain.ainvoke({"query": last_message})

        # 构建返回消息
        ai_message = AIMessage(content=result["answer"])
        ai_message.additional_kwargs["sources"] = result.get("sources", [])

        return {"messages": [ai_message]}

    except Exception as exc:
        logger.error(f"RAG query failed: {exc}")
        return {
            "messages": [
                AIMessage(
                    content="抱歉，知识库查询遇到问题，请稍后重试。"
                )
            ]
        }
```

**2. `rag/retriever.py` - 新建 Milvus Retriever**

```python
"""LangChain Milvus Retriever with Reranker"""

from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List

def create_milvus_retriever(
    top_k: int = 20,
    collection_name: str = "fitness_knowledge",
) -> BaseRetriever:
    """创建 Milvus Retriever with Reranker

    Args:
        top_k: 召回候选数量 (用于 reranker)
        collection_name: Milvus 集合名称

    Returns:
        LangChain Retriever
    """
    # 初始化 Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        encode_kwargs={"normalize_embeddings": True}
    )

    # 连接 Milvus
    vectorstore = Milvus(
        embedding_function=embeddings,
        collection_name=collection_name,
        connection_args={"host": "localhost", "port": "19530"},
    )

    # 基础 Retriever (召回 top-k)
    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )

    # 包装 Reranker
    return RerankerRetriever(
        base_retriever=base_retriever,
        top_n=5,  # 最终返回 top-5
        reranker_model="BAAI/bge-reranker-v2-m3"
    )


class RerankerRetriever(BaseRetriever):
    """带 Reranker 的 Retriever"""

    def __init__(
        self,
        base_retriever: BaseRetriever,
        top_n: int = 5,
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
    ):
        super().__init__()
        self.base_retriever = base_retriever
        self.top_n = top_n
        self.reranker = CrossEncoder(reranker_model)

    def _get_relevant_documents(
        self, query: str, **kwargs
    ) -> List[Document]:
        # 召回阶段
        candidates = self.base_retriever.get_relevant_documents(query)

        if not candidates:
            return []

        # Reranker 精排
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = self.reranker.predict(pairs)

        # 按分数排序并返回 top-n
        scored_docs = list(zip(candidates, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, score in scored_docs[:self.top_n]]
```

**3. `rag/chain.py` - 新建 RAG Chain**

```python
"""LangChain RAG Chain"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from muscleking.app.rag.retriever import create_milvus_retriever
from muscleking.app.core.context import get_global_context

# RAG Prompt 模板
RAG_TEMPLATE = """你是健身知识科普助手，请根据以下检索结果回答用户问题。

检索结果：
{context}

用户问题：{query}

回答要求：
1. 仅基于检索结果回答，不要编造信息
2. 保持专业、客观、通俗易懂
3. 如果检索结果不足，明确说明
4. 回答结尾列出参考来源
"""

def create_rag_chain(top_k: int = 5):
    """创建 RAG Chain

    Args:
        top_k: 检索返回的文档数量

    Returns:
        RAG Chain
    """
    ctx = get_global_context()
    llm = ctx.llm

    # 创建 Retriever
    retriever = create_milvus_retriever(top_k=top_k)

    # 构建 Prompt
    prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

    # 构建 Chain
    chain = (
        {"context": retriever, "query": RunnablePassthrough()}
        | prompt
        | llm
        | (lambda msg: {
            "answer": msg.content,
            "sources": _extract_sources(retriever)
        })
    )

    return chain


def _extract_sources(retriever) -> List[str]:
    """从 Retriever 结果中提取来源"""
    # 实现来源提取逻辑
    return []
```

**4. `settings.py` - 启用配置**

```python
# 取消注释并更新以下配置

# Milvus 配置
MILVUS_HOST: str = "localhost"
MILVUS_PORT: int = 19530
MILVUS_COLLECTION: str = "fitness_knowledge"
MILVUS_INDEX_TYPE: str = "IVF_FLAT"
MILVUS_METRIC_TYPE: str = "IP"

# Embedding 配置
EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
EMBEDDING_DIMENSION: int = 1024

# Reranker 配置
ENABLE_RERANK: bool = True
RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
RERANK_MAX_CANDIDATES: int = 20
RERANK_TOP_N: int = 5
RERANK_SCORE_THRESHOLD: float = 0.3

# 检索配置
KB_TOP_K: int = 5
KB_SIMILARITY_THRESHOLD: float = 0.2
KB_CHUNK_SIZE: int = 512
KB_CHUNK_OVERLAP: int = 80
```

---

## 4. 数据流与错误处理

### 4.1 完整数据流

```
用户输入查询
    ↓
LangGraph Router 分析意图
    ↓ (route: kb-query)
create_kb_query 节点
    ↓
创建 RAG Chain
    ↓
┌─────────────────────────────┐
│ 检索阶段                    │
│ 1. Embed Query (BGE-M3)     │
│ 2. Milvus 搜索 (top-20)     │
│ 3. Reranker 重排 (top-5)    │
│ 4. 构建 Context             │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│ 生成阶段                    │
│ 1. 构建 Prompt (C+Q)        │
│ 2. LLM 生成回答             │
│ 3. 提取 Sources             │
└─────────────────────────────┘
    ↓
返回 AIMessage(content, sources)
```

### 4.2 错误处理策略

| 错误类型 | 检测方式 | 处理策略 | 用户返回 |
|---------|---------|---------|---------|
| **Milvus 连接失败** | ConnectionError | 降级到纯 LLM | "知识库暂时不可用，但我可以基于通用知识回答..." |
| **Embedding 超时** | TimeoutError | 重试 1 次，失败则跳过 | 同上 |
| **无相关文档** | 检索结果为空 | 跳过检索阶段 | "抱歉，知识库中没有找到相关内容..." |
| **LLM 生成失败** | LLMException | 返回原始检索结果 | "我找到了以下相关资料，但无法生成总结..." |
| **Reranker 失败** | ModelError | 退回向量检索结果 | 正常返回 (不影响主流程) |

### 4.3 降级策略

```python
# 降级层级 1: Reranker 失败 → 使用向量检索结果
if reranker_fails:
    results = vector_search_results

# 降级层级 2: Milvus 不可用 → 使用纯 LLM
if milvus_unavailable:
    answer = llm.invoke(f"基于通用知识回答: {query}")
    return {"answer": answer, "sources": []}

# 降级层级 3: LLM 失败 → 返回原始文档
if llm_fails:
    return {"answer": format_docs(results), "sources": sources}
```

---

## 5. 测试策略

### 5.1 测试目录结构

```
tests/
├── rag/                          # RAG 模块测试
│   ├── test_retriever.py         # 测试 Milvus Retriever
│   ├── test_chain.py             # 测试 RAG Chain
│   └── test_embeddings.py        # 测试 Embedding 模型
├── services/                     # 服务层测试
│   ├── test_vector_store.py      # 测试 Milvus 连接
│   └── test_kb_service.py        # 测试 KnowledgeBaseService
├── agents/                       # Agent 节点测试
│   └── test_kb_query.py          # 测试 create_kb_query
└── integration/                  # 集成测试
    └── test_e2e_rag.py           # 端到端 RAG 测试
```

### 5.2 核心测试用例

#### test_retriever.py

```python
import pytest
from muscleking.app.rag.retriever import create_milvus_retriever

def test_milvus_retriever_creation():
    """测试 Milvus Retriever 创建"""
    retriever = create_milvus_retriever()
    assert retriever is not None
    assert retriever.top_n == 5

@pytest.mark.asyncio
async def test_retriever_returns_documents():
    """测试检索返回文档"""
    retriever = create_milvus_retriever()
    docs = await retriever.ainvoke("如何锻炼胸肌？")

    assert len(docs) <= 5
    assert all(doc.page_content for doc in docs)
    assert all(doc.metadata for doc in docs)

def test_reranker_reorders_results():
    """测试 Reranker 重排序"""
    retriever = create_milvus_retriever()
    mock_docs = [
        Document(page_content="低相关内容", metadata={}),
        Document(page_content="高相关内容：胸肌锻炼方法", metadata={}),
        Document(page_content="中相关内容", metadata={}),
    ]

    # Mock reranker 返回分数
    with mock_reranker_scores([0.2, 0.9, 0.5]):
        results = retriever._get_relevant_documents("如何锻炼胸肌？", docs=mock_docs)

    assert results[0].page_content == "高相关内容：胸肌锻炼方法"
    assert len(results) == 3

def test_empty_query_handling():
    """测试空查询处理"""
    retriever = create_milvus_retriever()
    docs = retriever.invoke("")
    assert docs == []
```

#### test_chain.py

```python
import pytest
from muscleking.app.rag.chain import create_rag_chain

@pytest.mark.asyncio
async def test_rag_chain_generates_answer():
    """测试 RAG Chain 生成回答"""
    chain = create_rag_chain(top_k=5)
    result = await chain.ainvoke("如何锻炼胸肌？")

    assert "answer" in result
    assert "sources" in result
    assert len(result["answer"]) > 0
    assert isinstance(result["sources"], list)

@pytest.mark.asyncio
async def test_rag_chain_with_no_results():
    """测试无检索结果时的行为"""
    chain = create_rag_chain()
    result = await chain.ainvoke("量子力学是什么？")  # 假设知识库无此内容

    assert "知识库中没有找到" in result["answer"] or result["answer"]

@pytest.mark.asyncio
async def test_rag_chain_context_injection():
    """测试 Context 正确注入"""
    chain = create_rag_chain()

    # Mock retriever 返回特定文档
    with mock_retriever_results([Document(page_content="胸肌锻炼：卧推、飞鸟...")]):
        result = await chain.ainvoke("如何锻炼胸肌？")

    assert "卧推" in result["answer"] or "胸肌" in result["answer"]
```

#### test_kb_query.py

```python
import pytest
from muscleking.app.agents.lg_builder import create_kb_query
from muscleking.app.agents.models.model_lg_state import AgentState
from langchain_core.messages import HumanMessage

@pytest.mark.asyncio
async def test_kb_query_node():
    """测试 create_kb_query 节点"""
    state = AgentState(
        messages=[HumanMessage(content="如何锻炼胸肌？")],
        router=None
    )

    result = await create_kb_query(state, config={})

    assert "messages" in result
    assert len(result["messages"]) == 1
    assert result["messages"][0].content  # 有回答内容
    assert "sources" in result["messages"][0].additional_kwargs

@pytest.mark.asyncio
async def test_kb_query_with_empty_message():
    """测试空消息处理"""
    state = AgentState(messages=[], router=None)
    result = await create_kb_query(state, config={})

    assert "请告诉我具体的问题" in result["messages"][0].content

@pytest.mark.asyncio
async def test_kb_query_error_handling():
    """测试错误处理"""
    state = AgentState(messages=[HumanMessage(content="测试查询")], router=None)

    # Mock Milvus 连接失败
    with mock_milvus_connection_error():
        result = await create_kb_query(state, config={})

    assert "知识库查询遇到问题" in result["messages"][0].content
```

### 5.3 测试工具

**依赖** (`pyproject.toml`):
```toml
[project.optional-dependencies]
test = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-mock>=3.12.0",
    "pytest-cov>=4.1.0",
    "httpx>=0.27.0",  # 用于 mocking LLM
]
```

**运行测试**:
```bash
# 运行所有测试
pytest

# 运行特定模块测试
pytest tests/rag/

# 生成覆盖率报告
pytest --cov=muscleking/app/rag --cov-report=html
```

### 5.4 测试覆盖率目标

| 模块 | 目标覆盖率 |
|------|-----------|
| `rag/retriever.py` | ≥ 85% |
| `rag/chain.py` | ≥ 80% |
| `agents/lg_builder.py` (kb_query部分) | ≥ 75% |
| `services/knowledge_base_service.py` | ≥ 70% |

---

## 6. 实施步骤

### 阶段 1: 环境准备 (1-2小时)
- [ ] 部署 Milvus (Docker Compose)
- [ ] 验证 Milvus 连接
- [ ] 安装 BGE-M3 和 Reranker 模型
- [ ] 更新 settings.py 配置

### 阶段 2: 核心代码实现 (3-4小时)
- [ ] 创建 `rag/` 模块
- [ ] 实现 `rag/retriever.py`
- [ ] 实现 `rag/chain.py`
- [ ] 实现 `rag/prompts.py`
- [ ] 简化 `lg_builder.py` 的 `create_kb_query`
- [ ] 更新 `knowledge_base_service.py`

### 阶段 3: 单元测试 (2-3小时)
- [ ] 编写 `test_retriever.py`
- [ ] 编写 `test_chain.py`
- [ ] 编写 `test_kb_query.py`
- [ ] 编写 `test_kb_service.py`
- [ ] 运行测试并修复问题

### 阶段 4: 数据导入 (1-2小时)
- [ ] 实现 `scripts/import_documents.py`
- [ ] 导入健身知识文档
- [ ] 验证向量数据

### 阶段 5: 集成测试与验证 (1-2小时)
- [ ] 端到端测试
- [ ] 性能测试
- [ ] 优化和调整

**总预计时间**: 8-13 小时

---

## 7. 风险与依赖

### 7.1 风险

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| Milvus 部署失败 | 阻塞开发 | 提供 Docker Compose 一键部署脚本 |
| 模型下载慢 | 延迟启动 | 使用镜像站或提前下载 |
| 向量检索效果不佳 | 用户体验差 | 可调整 chunk_size、overlap、top_k 参数 |
| Reranker 性能问题 | 响应慢 | 可禁用 Reranker 或降低召回数量 |

### 7.2 依赖

**外部依赖**:
- Milvus 2.6+ (Docker)
- BGE-M3 模型 (~2.3GB)
- BGE-Reranker 模型 (~1.1GB)

**Python 依赖**:
- `pymilvus>=2.6.11`
- `sentence-transformers>=5.3.0`
- `langchain>=1.2.15`
- `langchain-community>=0.4.1`

---

## 8. 后续优化方向

1. **混合检索**: 结合稀疏检索 (BM25) + 密集检索 (向量)
2. **查询重写**: 使用 LLM 优化用户查询
3. **缓存机制**: 缓存常见问题的回答
4. **增量更新**: 支持文档增量导入和更新
5. **多轮对话**: 结合对话历史的检索

---

## 附录

### A. 配置参数参考

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `MILVUS_HOST` | localhost | Milvus 服务器地址 |
| `MILVUS_PORT` | 19530 | Milvus 端口 |
| `MILVUS_COLLECTION` | fitness_knowledge | 集合名称 |
| `EMBEDDING_MODEL_NAME` | BAAI/bge-m3 | Embedding 模型 |
| `EMBEDDING_DIMENSION` | 1024 | 向量维度 |
| `RERANK_TOP_N` | 5 | 返回结果数 |
| `RERANK_MAX_CANDIDATES` | 20 | 召回候选数 |
| `KB_CHUNK_SIZE` | 512 | 文档分块大小 |
| `KB_CHUNK_OVERLAP` | 80 | 分块重叠 |
| `KB_TOP_K` | 5 | 检索返回数 |
| `KB_SIMILARITY_THRESHOLD` | 0.2 | 相似度阈值 |

### B. RAG Prompt 模板

```
你是健身知识科普助手，请根据以下检索结果回答用户问题。

检索结果：
{context}

用户问题：{query}

回答要求：
1. 仅基于检索结果回答，不要编造信息
2. 保持专业、客观、通俗易懂
3. 如果检索结果不足，明确说明
4. 回答结尾列出参考来源
```

### C. 参考资料

- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [Milvus Documentation](https://milvus.io/docs)
- [BGE-M3 Model Card](https://huggingface.co/BAAI/bge-m3)
- [BGE-Reranker Model Card](https://huggingface.co/BAAI/bge-reranker-v2-m3)
