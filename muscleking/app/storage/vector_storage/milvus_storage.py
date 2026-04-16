"""
Milvus 向量存储层

封装 pymilvus MilvusClient，提供向量数据的增删改查和检索能力。
"""

from typing import Any, Dict, List, Optional

from pymilvus import DataType, Function, FunctionType, MilvusClient
from loguru import logger


class MilvusStorage:
    """
    Milvus 向量存储

    职责：
    - 管理 Milvus 连接
    - Collection 的创建和初始化
    - 向量数据的增删改查
    - 向量相似度检索
    """

    def __init__(
        self,
        milvus_client: MilvusClient,
        collection_name: str = "fitness_knowledge",
        embedding_dimension: int = 1024,
    ) -> None:
        self._client: MilvusClient = milvus_client
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension

    # ── 连接管理 ──────────────────────────────────────────

    def close(self) -> None:
        """关闭 Milvus 连接"""
        pass

    @property
    def client(self) -> MilvusClient:
        """获取底层 MilvusClient 实例"""
        return self._client

    # ── Collection 管理 ───────────────────────────────────

    def create_collection(
        self,
        collection_name: Optional[str] = None,
        dimension: Optional[int] = None,
        drop_existing: bool = False,
    ) -> None:
        """
        创建 Collection

        Schema:
            child_chunk_id   VARCHAR   (PK)        子文档 ID
            parent_chunk_id  VARCHAR               父文档 ID
            chunk_index      INT32                 子块排序序号
            content          VARCHAR               文本内容 (enable_analyzer, BM25 输入)
            dense_vector     FLOAT_VECTOR          语义向量
            sparse_vector    SPARSE_FLOAT_VECTOR   BM25 稀疏向量 (由 Function 自动生成)
            file_name        VARCHAR               原始文件名
            timestamp        INT64                 入库时间戳

        Args:
            collection_name: 集合名称，默认使用初始化时的名称
            dimension: 向量维度，默认使用初始化时的维度
            drop_existing: 是否删除已存在的同名集合
        """
        name = collection_name or self.collection_name
        dim = dimension or self.embedding_dimension

        if drop_existing and self._client.has_collection(name):
            logger.info(f"删除已存在的集合: {name}")
            self._client.drop_collection(name)

        if self._client.has_collection(name):
            logger.info(f"集合已存在，跳过创建: {name}")
            return

        # ── Schema ──
        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=True)

        schema.add_field("child_chunk_id", DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field("parent_chunk_id", DataType.VARCHAR, max_length=128)
        schema.add_field("chunk_index", DataType.INT32)
        schema.add_field("content", DataType.VARCHAR, max_length=8192, enable_analyzer=True)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("file_name", DataType.VARCHAR, max_length=512)
        schema.add_field("timestamp", DataType.INT64)

        # ── BM25 Function: content -> sparse_vector ──
        bm25_fn = Function(
            name="bm25_fn",
            input_field_names=["content"],
            output_field_names="sparse_vector",
            function_type=FunctionType.BM25,
        )
        schema.add_function(bm25_fn)

        # ── Index ──
        index_params = self._client.prepare_index_params()

        # dense 向量索引 (HNSW + COSINE)
        index_params.add_index(
            field_name="dense_vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 256},
        )

        # sparse 向量索引 (BM25)
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )

        # ── 创建 ──
        self._client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=index_params,
        )
        logger.info(f"集合创建成功: {name}")

    def has_collection(self, collection_name: Optional[str] = None) -> bool:
        """
        检查 Collection 是否存在

        Args:
            collection_name: 集合名称，默认使用默认集合名

        Returns:
            是否存在
        """
        pass

    def describe_collection(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        获取 Collection 详细信息

        Args:
            collection_name: 集合名称

        Returns:
            集合元信息（含 num_entities 等）
        """
        pass

    def drop_collection(self, collection_name: Optional[str] = None) -> None:
        """
        删除 Collection

        Args:
            collection_name: 集合名称
        """
        pass

    # ── 数据写入 ──────────────────────────────────────────

    def insert(
        self,
        data: List[Dict[str, Any]],
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        插入文档向量

        Args:
            data: 文档列表，每条包含 id、vector、content、source 等字段
            collection_name: 集合名称

        Returns:
            插入结果（含 insert_count 等）
        """
        pass

    def upsert(
        self,
        data: List[Dict[str, Any]],
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        插入或更新文档向量

        Args:
            data: 文档列表
            collection_name: 集合名称

        Returns:
            upsert 结果
        """
        pass

    # ── 数据查询 ──────────────────────────────────────────

    def get(
        self,
        ids: List[str],
        collection_name: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        根据 ID 获取文档

        Args:
            ids: 文档 ID 列表
            collection_name: 集合名称
            output_fields: 需要返回的字段

        Returns:
            文档列表
        """
        pass

    def search(
        self,
        data: List[List[float]],
        limit: int = 10,
        output_fields: Optional[List[str]] = None,
        collection_name: Optional[str] = None,
        filter_expr: Optional[str] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        向量相似度检索

        Args:
            data: 查询向量列表
            limit: 返回数量上限
            output_fields: 需要返回的字段
            collection_name: 集合名称
            filter_expr: 过滤表达式

        Returns:
            检索结果（二维列表，外层对应每条查询向量）
        """
        pass

    def query(
        self,
        filter_expr: str,
        output_fields: Optional[List[str]] = None,
        collection_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        标量条件查询

        Args:
            filter_expr: 过滤表达式，如 'source == "xxx"'
            output_fields: 需要返回的字段
            collection_name: 集合名称
            limit: 返回数量上限

        Returns:
            匹配的文档列表
        """
        pass

    # ── 数据删除 ──────────────────────────────────────────

    def delete(
        self,
        ids: List[str],
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        根据 ID 删除文档

        Args:
            ids: 要删除的文档 ID 列表
            collection_name: 集合名称

        Returns:
            删除结果
        """
        pass

    def delete_by_filter(
        self,
        filter_expr: str,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        根据过滤条件删除文档

        Args:
            filter_expr: 过滤表达式
            collection_name: 集合名称

        Returns:
            删除结果
        """
        pass

    # ── 统计信息 ──────────────────────────────────────────

    def count(self, collection_name: Optional[str] = None) -> int:
        """
        获取文档总数

        Args:
            collection_name: 集合名称

        Returns:
            文档数量
        """
        pass
milvus_s = MilvusStorage(MilvusClient(uri="http://localhost:19530"), collection_name="fitness_knowledge")
milvus_s.create_collection()