"""知识库服务."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from uuid import uuid4

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from langchain_community.embeddings import HuggingFaceEmbeddings

from muscleking.app.config import settings
import torch
from .vector_store import VectorStore
from sentence_transformers import CrossEncoder


# 注意，exercise_id对应父文档的id，chunk_id(即ids)对应子文档的id
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
            model_name=settings.EMBEDDING_MODEL_NAME,
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
        self.reranker = CrossEncoder(
            settings.RERANK_MODEL,
            device=device,
        )

        self.enable_rerank = settings.ENABLE_RERANK if settings.ENABLE_RERANK else False

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
        """
        添加单个文档到知识库中(适用于文本类型的文档数据, 如Markdown、Word、txt等)
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
        meta.setdefault("exercise_id", doc_id if doc_id else uuid4().hex)
        result = await self.ingest_text(content, metadata=meta)
        return result.get("add_count", 0) > 0

    async def add_exercise(
        self, exercise_id: str, exercise_data: Dict[str, Any]
    ) -> bool:
        """
        添加单个锻炼动作到知识库中(适用于json格式的锻炼动作数据)
        Args:
            exercise_id: 锻炼动作唯一ID
            exercise_data: 锻炼动作数据(json格式)
        Returns:
            是否添加成功
        """
        content = self._format_exercise_content(exercise_data)
        metadata = {
            "exercise_id": exercise_id,
            "name": exercise_data.get("name", ""),
            "equipments": exercise_data.get("equipments", ""),
            "body_parts": exercise_data.get("bodyParts", ""),
        }
        result = await self.ingest_text(content, metadata=metadata)
        return result.get("add_count", 0) > 0

    async def add_exercises_batch(
        self, exercises: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """批量添加锻炼动作到知识库中(适用于json格式的锻炼动作数据)"""
        success_count = 0
        error_count = 0
        for exercise in exercises:
            exercise_id = (
                exercise.get("exerciseId") or exercise.get("id") or str(uuid4())
            )
            if await self.add_exercise(exercise_id, exercise):
                success_count += 1
            else:
                error_count += 1
        return {"success": success_count, "error": error_count, "total": len(exercises)}

    async def ingest_text(
        self,
        content: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """将文本内容分块并存储到知识库中
        Args:
            content: 原始文本内容
            metadata: 该文本的元数据
        Returns:
            一个包含 添加数量、分块ID列表 的字典
        """

        if not content or not content.strip():
            return {"add_count": 0, "ids": []}

        # 对文本进行分块(List[Document])
        documents = await asyncio.to_thread(
            self._split_into_documents, content, metadata or {}
        )
        if not documents:
            return {"add_count": 0, "ids": []}

        embeddings = await asyncio.to_thread(
            self.embeddings.embed_documents,
            [doc.page_content for doc in documents],
        )

        result = await asyncio.to_thread(self._store_documents, documents, embeddings)
        return result

    async def rerank_candidates(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        *,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """对文档列表进行重排序
        Args:
            query: 查询内容
            candidates: 待排序的文档列表
            top_k: 排序返回的文档数量
        Returns:
            一个包含文档元数据和重排序分数的列表
        """
        if not candidates:
            return []

        top_k = top_k or len(candidates)

        scores = await asyncio.to_thread(
            self.reranker.predict,
            [(query, candidate["content"]) for candidate in candidates],
        )

        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = score

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k]

    async def search(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        filter_expr: Optional[str] = None,
        filter_by_similarity: bool = True,
    ) -> List[Dict[str, Any]]:
        """根据查询内容，从知识库中检索出最相关的文档
        Args:
            query: 查询内容
            top_k: 检索返回的文档数量
            similarity_threshold: 文档与查询内容的相似度阈值
            filter_expr: 用于过滤文档的表达式, 如filter_expr = "category == '家常菜'"
            filter_by_similarity: 是否根据相似度进行过滤
        Returns:
            一个包含文档元数据和相似度分数的列表
        """
        if not query or not query.strip():
            return []

        top_k = top_k or settings.KB_TOP_K
        similarity_threshold = (
            similarity_threshold
            if similarity_threshold
            else settings.KB_SIMILARITY_THRESHOLD
        )

        # 如果启用 reranker，先召回更多候选文档
        recall_k = top_k
        if self.enable_rerank:
            recall_k = settings.RERANK_MAX_CANDIDATES  # 召回更多文档用于重排
        # 对用户查询进行向量嵌入
        query_embedding = await asyncio.to_thread(self.embeddings.embed_query, query)
        results = await asyncio.to_thread(
            self.vector_store.search,
            query_embedding,
            recall_k,  # 最大召回数量, 如果后续要用reranker, recall_k需要大于top_k
            filter_expr,  # 用于过滤文档的表达式
        )

        # 过滤出符合相似度阈值的文档
        candidates = results
        if filter_by_similarity and similarity_threshold is not None:
            candidates = [
                r for r in candidates if r.get("score", 0.0) >= similarity_threshold
            ]

        # 使用 reranker 精排
        if candidates and self.enable_rerank:
            candidates = await self.rerank_candidates(query, candidates, top_k=top_k)
        if self.enable_rerank:
            candidates = [
                r
                for r in candidates
                if r.get("rerank_score", 0.0) >= settings.KB_RERANK_SCORE_THRESHOLD
            ]
        elif not filter_by_similarity and similarity_threshold is not None:
            candidates = [
                r for r in candidates if r.get("score", 0.0) >= similarity_threshold
            ]

        return candidates[
            :top_k
        ]  # 这里的top_k和精排前的top_k是一样的，后续看看要不要修改

    async def delete_exercise(self, exercise_id: str) -> bool:
        """删除知识库中的指定锻炼动作"""
        return await asyncio.to_thread(
            self.vector_store.delete_documents, [exercise_id]
        )

    async def get_stats(self) -> Dict[str, Any]:
        """
        获取向量库的统计信息
        """

        def _stats() -> Dict[str, Any]:
            stats = self.vector_store.get_collection_stats()
            stats.update(
                {
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "embedding_model": settings.EMBEDDING_MODEL_NAME,
                }
            )
            return stats

        return await asyncio.to_thread(_stats)

    async def clear(self) -> bool:
        """清空知识库, 删除所有数据"""
        return await asyncio.to_thread(self.vector_store.clear_collection)

    async def close(self) -> None:
        """关闭数据库连接"""
        await asyncio.to_thread(self.vector_store.close)

    def _split_into_documents(
        self, content: str, metadata: Dict[str, Any]
    ) -> List[Document]:
        """将文本分块为多个chunk(每个都是一个Document对象)
        Args:
            content: 原始文本内容
            metadata: 该文本的元数据
        Returns:
            一个Document对象列表
        """
        base_document = Document(page_content=content, metadata=metadata)
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
            documents: 分块后的文档列表
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
            exercise_id = (
                metadata.get("exercise_id")
                or metadata.get("id")
                or metadata.get("source")
                or uuid4().hex
            )
            # 当前分块的ID, 格式为: 父文档ID_分块索引
            chunk_id = f"{exercise_id}_{index}"
            metadata.setdefault("exercise_id", exercise_id)
            metadata.setdefault("chunk_id", chunk_id)
            metadata.setdefault(
                "name", metadata.get("name") or metadata.get("title") or ""
            )

            ids.append(chunk_id)
            contents.append(doc.page_content)
            metadatas.append(metadata)

        # 向量数据库存储四个列表：
        # ids（分块ID列表）, embeddings（嵌入向量列表）, contents（分块内容列表）, metadatas（分块元数据列表）
        # metadata中至少包含了base_id、chunk_id和name等信息，方便后续查询和管理
        success = self.vector_store.add_documents(
            ids=ids,
            embeddings=embeddings,
            contents=contents,
            metadatas=metadatas,
        )

        return {"add_count": len(ids) if success else 0, "ids": ids, "stored": success}

    def _normalize_str_list(value) -> List[str]:
        """把字符串或字符串列表统一转换为字符串列表"""
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [value]
        return []

    def _format_exercise_content(self, exercise_data: Dict[str, Any]) -> str:
        """
        格式化字典格式的动作内容为字符串(动作的长文本具体描述,即content)
        Args:
            exercise_data: 包含动作信息的字典
        Returns:
            格式化后的字符串
        """
        parts: List[str] = []

        name = self._normalize_str_list(exercise_data.get("name"))
        if name:
            parts.append(f"动作名称：{', '.join(name)}")

        equipments = self._normalize_str_list(exercise_data.get("equipments"))
        if equipments:
            parts.append(f"动作器械：{', '.join(equipments)}")

        bodyParts = self._normalize_str_list(exercise_data.get("bodyParts"))
        if bodyParts:
            parts.append(f"主要锻炼部位：{', '.join(bodyParts)}")

        targetMuscles = self._normalize_str_list(exercise_data.get("targetMuscles"))
        if targetMuscles:
            parts.append(f"目标肌肉：{', '.join(targetMuscles)}")

        secondaryMuscles = self._normalize_str_list(
            exercise_data.get("secondaryMuscles")
        )
        if secondaryMuscles:
            parts.append(f"辅助肌肉：{', '.join(secondaryMuscles)}")

        exerciseType = self._normalize_str_list(exercise_data.get("exerciseType"))
        if exerciseType:
            parts.append(f"动作类型：{', '.join(exerciseType)}")

        gender = self._normalize_str_list(exercise_data.get("gender"))
        if gender:
            parts.append(f"适用性别：{', '.join(gender)}")

        overview = self._normalize_str_list(exercise_data.get("overview"))
        if overview:
            parts.append(f"动作概述：{', '.join(overview)}")

        instructions = self._normalize_str_list(exercise_data.get("instructions"))
        if instructions:
            parts.append(f"动作指导：{', '.join(instructions)}")

        return "\n".join(parts)
