"""全局检索器"""

from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from loguru import logger

from muscleking.app.retrieval.embedding_model.qwen_embedding import QwenEmbedding
from muscleking.app.retrieval.reranker.qwen_reranker import QwenReranker
from muscleking.app.models.rerank import RerankResult
from muscleking.app.storage import MilvusStorage


class Retriever:
    """
    全局检索器

    职责：
    - 管理 Embedding 模型
    - 管理 Reranker 模型
    - 管理 Milvus 连接
    - 提供统一的检索接口
    """

    def __init__(
        self,
        embedding_model: QwenEmbedding,
        vector_storage: MilvusStorage,
        reranker: Optional[QwenReranker] = None,
    ):
        self.embedding_model = embedding_model
        self.vector_storage = vector_storage
        self.reranker = reranker
        self.collection_name = self.vector_storage.collection_name

        logger.info("✅ Retriever 初始化完成")

    async def aretrieve(
        self,
        query: str,
        top_k: int = 20,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        异步检索接口

        Args:
            query: 用户查询
            top_k: 召回候选数量
            collection_name: Milvus 集合名称

        Returns:
            {
                "results": List[Document],
                "recall_count": int,
                "rerank_result": Optional[RerankResult],
            }
        """
        collection = collection_name or self.collection_name

        # Step 1: Embed Query
        logger.info(f"Embedding query: {query[:50]}...")
        query_embedding = await self.embedding_model.embed(query)

        # Step 2: Milvus 向量检索
        logger.info(f"Searching Milvus (top_k={top_k})...")
        try:
            search_results = self.vector_storage.search(
                collection_name=collection,
                data=[query_embedding],
                limit=top_k,
                output_fields=["id", "content", "source", "title"],
            )
        except Exception as e:
            logger.error(f"Milvus search failed: {e}")
            return {
                "results": [],
                "recall_count": 0,
                "rerank_result": None,
            }

        documents = self._parse_milvus_results(search_results[0] if search_results else [])
        logger.info(f"Recalled {len(documents)} documents")

        if not documents:
            return {
                "results": [],
                "recall_count": 0,
                "rerank_result": None,
            }

        # Step 3: Reranker 精排（可选）
        rerank_result: Optional[RerankResult] = None
        if self.reranker is not None:
            logger.info(f"Reranking {len(documents)} documents...")
            rerank_docs = [{"content": doc.page_content, "source": doc.metadata.get("source", "")} for doc in documents]
            rerank_result = await self.reranker.rerank(query, rerank_docs)

            # 用 rerank 结果覆盖返回的 documents
            final_docs = [
                Document(
                    page_content=rd.content,
                    metadata={
                        "source": rd.source,
                        "rerank_score": rd.rerank_score,
                    },
                )
                for rd in rerank_result.documents
            ]
        else:
            final_docs = documents

        return {
            "results": final_docs,
            "recall_count": len(documents),
            "rerank_result": rerank_result,
        }

    def _parse_milvus_results(self, results: List[Dict]) -> List[Document]:
        """解析 Milvus 检索结果为 Document 对象"""
        documents = []
        for result in results:
            entity = result.get("entity", {})
            doc = Document(
                page_content=entity.get("content", ""),
                metadata={
                    "id": result.get("id"),
                    "score": float(result.get("distance", 0.0)),
                    "source": entity.get("source", ""),
                    "title": entity.get("title", ""),
                },
            )
            documents.append(doc)
        return documents