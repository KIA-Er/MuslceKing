"""全局检索器单例"""

import threading
import torch
from typing import List, Dict, Any, Optional
from pymilvus import MilvusClient
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from loguru import logger

from muscleking.app.rag.embeddings import VLLMEmbeddingClient
from muscleking.app.config import settings


class GlobalRetriever:
    """
    全局检索器单例

    职责：
    - 管理 vLLM Embedding Client
    - 管理 Reranker 模型 (本地)
    - 管理 Milvus 连接
    - 提供统一的检索接口
    """

    _instance: Optional['GlobalRetriever'] = None
    _lock = threading.Lock()

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """初始化全局检索器（只执行一次）"""
        if self._initialized:
            return

        logger.info("🔄 初始化 GlobalRetriever...")

        # 初始化 vLLM Embedding Client
        self.embedding_client = VLLMEmbeddingClient(
            base_url=settings.VLLM_EMBEDDING_BASE_URL
        )

        # 初始化 Reranker (本地模型)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"📦 加载 Reranker 模型到 {device}...")
        self.reranker = CrossEncoder(
            "BAAI/bge-reranker-v2-m3",
            device=device
        )

        # 初始化 Milvus Client
        logger.info("🔗 连接 Milvus...")
        self.milvus_client = MilvusClient(
            uri="http://localhost:19530"
        )

        self._initialized = True
        logger.info("✅ GlobalRetriever 初始化完成")

    async def aretrieve(
        self,
        query: str,
        top_k: int = 20,
        rerank_top_n: int = 5,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        异步检索接口

        Args:
            query: 用户查询
            top_k: 召回候选数量
            rerank_top_n: Rerank 后返回数量
            collection_name: Milvus 集合名称

        Returns:
            {
                "results": List[Document],  # 最终结果
                "recall": List[Document],  # 召回结果
                "rerank_scores": List[float],  # 重排序分数
            }
        """
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.retrieve(query, top_k, rerank_top_n, collection_name)
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        rerank_top_n: int = 5,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        同步检索接口

        Args:
            query: 用户查询
            top_k: 召回候选数量
            rerank_top_n: Rerank 后返回数量
            collection_name: Milvus 集合名称

        Returns:
            {
                "results": List[Document],  # 最终结果
                "recall": List[Document],  # 召回结果
                "rerank_scores": List[float],  # 重排序分数
            }
        """
        collection_name = collection_name or "fitness_knowledge"

        # Step 1: Embed Query (使用 vLLM)
        logger.info(f"🔤 Embedding query: {query[:50]}...")
        query_embedding = self.embedding_client.embed_query(query)

        # Step 2: Milvus 向量检索 (召回 top-k)
        logger.info(f"🔍 Searching Milvus (top_k={top_k})...")
        try:
            search_results = self.milvus_client.search(
                collection_name=collection_name,
                data=[query_embedding],
                limit=top_k,
            )
        except Exception as e:
            logger.error(f"Milvus search failed: {e}")
            return {
                "results": [],
                "recall": [],
                "rerank_scores": [],
            }

        # 转换为 Document 对象
        documents = self._parse_milvus_results(search_results[0])
        logger.info(f"✅ Recalled {len(documents)} documents")

        if not documents:
            return {
                "results": [],
                "recall": [],
                "rerank_scores": [],
            }

        # Step 3: Reranker 精排
        logger.info(f"🔄 Reranking {len(documents)} documents...")
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.reranker.predict(pairs)

        # 排序并返回 top-n
        scored_docs = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )[:rerank_top_n]

        final_results = [doc for doc, score in scored_docs]
        rerank_scores = [float(score) for doc, score in scored_docs]
        logger.info(f"✅ Reranked to {len(final_results)} documents")

        return {
            "results": final_results,
            "recall": documents,
            "rerank_scores": rerank_scores,
        }

    def _parse_milvus_results(self, results: List[Dict]) -> List[Document]:
        """
        解析 Milvus 检索结果为 Document 对象

        Args:
            results: Milvus 搜索结果

        Returns:
            Document 对象列表
        """
        documents = []
        for result in results:
            # Milvus 返回格式: {"id": ..., "distance": ..., "entity": {...}}
            entity = result.get("entity", {})
            doc = Document(
                page_content=entity.get("content", ""),
                metadata={
                    "id": result.get("id"),
                    "score": float(result.get("distance", 0.0)),
                    "source": entity.get("metadata", {}).get("source", ""),
                    "title": entity.get("metadata", {}).get("title", ""),
                }
            )
            documents.append(doc)
        return documents
