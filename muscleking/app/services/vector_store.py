"""
Milvus向量数据库封装
使用Milvus作为向量存储引擎
"""

from typing import List, Dict, Any, Optional
from loguru import logger
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)


class VectorStore:
    """
    Milvus向量数据库封装类
    """

    def __init__(
        self,
        collection_name: str = "exercises",
        host: str = "localhost",
        port: int = 19530,
        dimension: int = 1024,
        index_type: str = "IVF_FLAT",
        metric_type: str = "IP",  # 向量内积（已归一化），越大越相似
    ):
        """
        初始化Milvus向量数据库连接和集合

        collection_name: 集合名称
        host: Milvus服务器地址
        port: Milvus服务器端口
        dimension: 向量维度
        index_type: 索引类型，默认IVF_FLAT
        metric_type: 距离度量类型，默认IP
        """
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.dimension = dimension
        self.index_type = index_type
        self.metric_type = metric_type

        self.collection = None
        self._initialize()

    def _initialize(self):
        """
        初始化Milvus向量数据库连接和集合
        """
        try:
            # 连接Milvus服务器
            connections.connect(
                alias="default",  # 该连接的别名，默认是"default"
                host=self.host,
                port=self.port,
            )
            logger.info(f"Connected to Milvus server at {self.host}:{self.port}")

            # 检查集合是否存在
            if utility.has_collection(self.collection_name, using="default"):
                self.collection = Collection(self.collection_name, using="default")
                logger.info(f"Loaded existing collection: {self.collection_name}")
            else:
                # 创建集合
                self._create_collection()
                logger.info(f" Created new collection: {self.collection_name}...")

            # 把集合加载到内存中，因为Milvus需要将集合加载到内存后才能进行搜索操作
            self.collection.load()

        except Exception as e:
            logger.error(f"Failed to connect to Milvus server: {e}")
            raise

    def _create_collection(self):
        """
        创建集合
        先定义字段，再创建集合schema，最后创建集合
        创建集合之后需要创建索引
        """
        # 定义字段
        fields = [
            FieldSchema(
                name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=256
            ),
            FieldSchema(
                name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension
            ),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="exercise_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=256),
        ]

        # 创建集合schema
        schema = CollectionSchema(fields=fields, description="Exercises collection")

        # 创建集合
        self.collection = Collection(
            name=self.collection_name,
            using="default",  # 该集合使用的连接别名，默认是"default"
            schema=schema,
        )

        # 创建索引
        index_params = {
            "index_type": self.index_type,
            "metric_type": self.metric_type,
            "params": {"nlist": 128} if self.index_type == "IVF_FLAT" else {},
        }

        self.collection.create_index(field_name="embedding", index_params=index_params)

        logger.info(f"Created index with type: {self.index_type}")

    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        contents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """
        添加文档到向量数据库
        Args:
            ids: chunk_id列表, 格式为{exercise_id}_{index}
            embeddings: chunk向量列表
            contents: chunk内容列表
            metadatas: chunk元数据列表
        Returns:
            是否添加成功
        """
        # 将文档转换为向量
        try:
            if not metadatas:
                # 如果没有提供元数据，默认创建空字典, 每个文档对应一个空字典
                metadatas = [{}] * len(ids)

            # 准备插入数据
            entities = [
                ids,  # 即chunk_id，格式为exercise_id_index
                embeddings,
                contents,
                [
                    metadata.get("exercise_id", "") for metadata in metadatas
                ],  # exercise_id即父文档id
                [metadata.get("name", "") for metadata in metadatas],
            ]
            # 批量插入数据
            self.collection.insert(entities)
            self.collection.flush()

            logger.info(
                f"Successfully added {len(ids)} documents to collection {self.collection_name}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to add documents to Milvus: {e}")
            return False

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_expr: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        搜索向量数据库中最相似的文档
        Args:
            query_embedding: 查询向量
            top_k: 返回的文档数量
            filter_expr: 可选的过滤表达式，例如 "exercise_id == '123'"
        Returns:
            搜索结果列表
        """
        try:
            # 搜索参数
            search_params = {
                "metric_type": self.metric_type,
                "params": {"nprobe": 10},
            }
            # 执行搜索
            # results的类型是List[List[Hit]]
            # 外层列表对应每个查询向量，内层列表对应每个向量的搜索结果
            # Hit是Entity对象，通过 hit.entity.get("字段名") 获取
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=["id", "content", "exercise_id", "name"],
            )
            # 格式化搜索结果
            formatted_results = []
            for hits in results:  # 得到单个查询向量的所有搜索结果
                for hit in hits:  # 得到单个查询向量的单个搜索结果
                    search_result = {
                        "id": hit.entity.get("id"),
                        "content": hit.entity.get("content"),
                        "score": float(hit.score),
                        "metadata": {
                            "exercise_id": hit.entity.get("exercise_id"),
                            "name": hit.entity.get("name"),
                        },
                    }
                    formatted_results.append(search_result)
            logger.info(
                f"Successfully searched {len(formatted_results)} documents in collection {self.collection_name}"
            )
            return formatted_results

        except Exception as e:
            logger.error(f"Failed to search in Milvus: {e}")
            return []

    def delete_documents(self, exercise_ids: List[str]) -> bool:
        """
        删除文档

        Args:
            exercise_ids: 文档ID列表

        Returns:
            是否成功
        """
        try:
            expr = f"exercise_id in {exercise_ids}"
            self.collection.delete(expr)
            self.collection.flush()

            logger.info(f"Deleted {len(exercise_ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            return {
                "collection_name": self.collection_name,
                "chunk_count": self.collection.num_entities,
                "dimension": self.dimension,
                "index_type": self.index_type,
                "metric_type": self.metric_type,
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}

    def clear_collection(self) -> bool:
        """清空集合的所有数据"""
        try:
            self.collection.drop()
            self._create_collection()
            self.collection.load()

            logger.info(f"Cleared collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False

    def close(self):
        """关闭连接"""
        try:
            connections.disconnect("default")
            logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.error(f"Failed to disconnect: {e}")
