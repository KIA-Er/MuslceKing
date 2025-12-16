"""知识库服务."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from uuid import uuid4

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger
from langchain.embeddings import HuggingFaceEmbeddings

from muscleking.config import settings
from .embeddings import OpenAICompatibleEmbeddings
import torch
from .vector_store import VectorStore
from .reranker import Reranker

# 注意，原本的recipe_id对应现在的base_id，代表文档的唯一ID
class KnowledgeBaseService:
    """知识库服务类"""

    def __init__(
        self,
        *,
        vector_store: Optional[VectorStore] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> None:
        
        self.chunk_size = chunk_size or settings.KB_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.KB_CHUNK_OVERLAP

        # 分块用递归字符分块
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", " "],
        )

        # 载入嵌入模型
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embeddings = HuggingFaceEmbeddings(
            model_name= settings.EMBEDDING_MODEL_NAME,
            model_kwargs={
                "device": device,
                "trust_remote_code": True,
            },
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 32,
            },
        )
        
        # 载入向量存储
        self.vector_store = vector_store or VectorStore(
            collection_name=settings.MILVUS_COLLECTION,
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT,
            dimension=settings.EMBEDDING_DIMENSION,
            index_type=settings.MILVUS_INDEX_TYPE,
            metric_type=settings.MILVUS_METRIC_TYPE,
        )
        
        # 载入重排序模型
        self.reranker = Reranker()

        logger.info(
            "KnowledgeBaseService 初始化完成 (chunk_size=%s, chunk_overlap=%s)",
            self.chunk_size,
            self.chunk_overlap,
        )

    async def add_document(
        self,
        *,
        doc_id: Optional[str],
        title: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """添加单个文档到知识库中
        Args:
            doc_id: 文档唯一ID
            title: 文档标题
            content: 文档内容
            metadata: 文档元数据
        Returns:
            是否添加成功
        """
        meta = metadata.copy() if metadata else {}
        meta.setdefault("title", title)
        meta.setdefault("name", title)
        meta.setdefault("base_id", doc_id if doc_id else uuid4().hex)
        result = await self.ingest_text(content, metadata=meta)
        return result.get("add_count", 0) > 0

    async def ingest_text(
        self,
        text: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """将文本内容分块并存储到知识库中
        Args:
            text: 原始文本内容
            metadata: 该文本的元数据
        Returns:
            一个包含 添加数量、分块ID列表 的字典
        """

        if not text or not text.strip():
            return {"add_count": 0, "ids": []}

        # 对文本进行分块(List[Document])
        documents = await asyncio.to_thread(self._split_into_documents, text, metadata or {})
        if not documents:
            return {"add_count": 0, "ids": []}

        embeddings = await asyncio.to_thread(
            self.embeddings.embed_documents,
            [doc.page_content for doc in documents],
        )

        result = await asyncio.to_thread(self._store_documents, documents, embeddings)
        return result

    # async def add_recipe(self, recipe_id: str, recipe_data: Dict[str, Any]) -> bool:
    #     document = self._format_recipe_document(recipe_data)
    #     metadata = {
    #         "recipe_id": recipe_id,
    #         "name": recipe_data.get("name", ""),
    #         "category": recipe_data.get("category", ""),
    #         "difficulty": recipe_data.get("difficulty", ""),
    #     }
    #     result = await self.ingest_text(document, metadata=metadata)
    #     return result.get("add_count", 0) > 0

    # async def add_recipes_batch(self, recipes: List[Dict[str, Any]]) -> Dict[str, int]:
    #     success_count = 0
    #     error_count = 0
    #     for recipe in recipes:
    #         recipe_id = recipe.get("id") or recipe.get("recipe_id") or str(uuid4())
    #         if await self.add_recipe(recipe_id, recipe):
    #             success_count += 1
    #         else:
    #             error_count += 1
    #     return {"success": success_count, "error": error_count, "total": len(recipes)}

    async def search(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        filter_expr: Optional[str] = None,
        filter_by_similarity: bool = True,
    ) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []

        top_k = top_k or settings.KB_TOP_K
        similarity_threshold = (
            similarity_threshold if similarity_threshold is not None else settings.KB_SIMILARITY_THRESHOLD
        )

        # 如果启用 reranker，先召回更多候选文档
        recall_k = top_k
        if self.reranker.enabled:
            recall_k = settings.RERANK_MAX_CANDIDATES  # 召回更多文档用于重排

        embedding = await asyncio.to_thread(self.embeddings.embed_query, query)
        results = await asyncio.to_thread(
            self.vector_store.search,
            embedding,
            recall_k,  # 使用更大的召回数量
            filter_expr,
        )

        candidates = results
        if filter_by_similarity and similarity_threshold is not None:
            candidates = [r for r in candidates if r.get("score", 0.0) >= similarity_threshold]

        # 使用 reranker 精排
        if candidates and self.reranker.enabled:
            candidates = await self.reranker.rerank(query, candidates, top_k)

        if self.reranker.enabled:
            candidates = [
                r
                for r in candidates
                if r.get("rerank_score", 0.0) >= settings.KB_RERANK_SCORE_THRESHOLD
            ]
        elif not filter_by_similarity and similarity_threshold is not None:
            candidates = [r for r in candidates if r.get("score", 0.0) >= similarity_threshold]

        return candidates[:top_k]

    async def delete_recipe(self, recipe_id: str) -> bool:
        return await asyncio.to_thread(self.vector_store.delete_documents, [recipe_id])

    async def get_stats(self) -> Dict[str, Any]:
        def _stats() -> Dict[str, Any]:
            stats = self.vector_store.get_collection_stats()
            stats.update(
                {
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "embedding_model": settings.EMBEDDING_MODEL,
                }
            )
            return stats

        return await asyncio.to_thread(_stats)

    async def clear(self) -> bool:
        return await asyncio.to_thread(self.vector_store.clear_collection)

    async def close(self) -> None:
        await asyncio.to_thread(self.vector_store.close)

    def _split_into_documents(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """将文本分块为多个chunk(每个都是一个Document对象)
        Args:
            text: 原始文本内容
            metadata: 该文本的元数据
        Returns: 
            一个Document对象列表
        """
        base_document = Document(page_content=text, metadata=metadata)
        # 每个分块都会继承原始文档的元数据，这保证了分块后的文档仍然可以追溯到原始文档的信息
        chunks = self.splitter.split_documents([base_document])
        return chunks or [base_document]

    def _store_documents(
        self,
        documents: List[Document],
        embeddings: List[List[float]],
    ) -> Dict[str, Any]:
        """将文档分块及其嵌入存储到向量数据库中
        Args:
            documents: 文档分块列表
            embeddings: 对应的嵌入向量列表
        Returns:
            一个包含 添加数量、分块ID列表 和 是否存储成功状态 的字典
        """
        if not documents or not embeddings:
            return {"add_count": 0, "ids": [], "stored": False}

        ids: List[str] = []
        contents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for index, doc in enumerate(documents):
            metadata = dict(doc.metadata or {})
            # 得到父文档（分块之前）的ID
            base_id = metadata.get("base_id") or metadata.get("id") or metadata.get("source") or uuid4().hex
            # 当前分块的ID, 格式为: 父文档ID_分块索引
            chunk_id = f"{base_id}_{index}"
            metadata.setdefault("base_id", base_id)
            metadata.setdefault("chunk_id", chunk_id)
            metadata.setdefault("name", metadata.get("name") or metadata.get("title") or "")

            ids.append(chunk_id)
            contents.append(doc.page_content)
            metadatas.append(metadata)

        # 向量数据库存储四个列表： 
        # ids（分块ID列表）, embeddings（嵌入向量列表）, documents（分块内容列表）, metadatas（分块元数据列表）
        # metadata中至少包含了base_id、chunk_id和name等信息，方便后续查询和管理
        success = self.vector_store.add_documents(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas,
        )

        return {"add_count": len(ids) if success else 0, "ids": ids, "stored": success}

    def _format_recipe_document(self, recipe: Dict[str, Any]) -> str:
        parts: List[str] = []
        name = recipe.get("name")
        if name:
            parts.append(f"菜名：{name}")

        category = recipe.get("category")
        if category:
            parts.append(f"分类：{category}")

        difficulty = recipe.get("difficulty")
        if difficulty:
            parts.append(f"难度：{difficulty}")

        time_cost = recipe.get("time") or recipe.get("cook_time")
        if time_cost:
            parts.append(f"耗时：{time_cost}")

        ingredients = recipe.get("ingredients") or recipe.get("ingredient_list")
        if ingredients:
            if isinstance(ingredients, list):
                formatted = "、".join(str(item) for item in ingredients)
            else:
                formatted = str(ingredients)
            parts.append(f"食材：{formatted}")

        steps = recipe.get("steps")
        if steps:
            if isinstance(steps, list):
                step_lines = [f"步骤{idx + 1}：{step}" for idx, step in enumerate(steps)]
                parts.extend(step_lines)
            else:
                parts.append(f"步骤：{steps}")

        tips = recipe.get("tips")
        if tips:
            parts.append(f"小贴士：{tips}")

        nutrition = recipe.get("nutrition")
        if nutrition:
            if isinstance(nutrition, dict):
                nutritions = [f"{k}: {v}" for k, v in nutrition.items()]
                parts.append("营养：" + "、".join(nutritions))
            else:
                parts.append(f"营养：{nutrition}")

        return "\n".join(parts)
