"""
Milvus向量数据库封装
使用新版 MilvusClient API
"""

from typing import List, Dict, Any, Optional
from loguru import logger
from pymilvus import MilvusClient


class VectorStore:
    """
    Milvus向量数据库封装类（使用新版 MilvusClient API）

    提供完整的 Milvus 操作接口：
    - 集合管理（创建、删除、列出、检查存在）
    - 数据管理（插入、删除、更新）
    - 向量搜索
    - 索引管理
    - 内存管理
    """

    def __init__(
        self,
        collection_name: str = "exercises",
        uri: str = "http://localhost:19530",
        dimension: int = 1024,
        index_type: str = "IVF_FLAT",
        metric_type: str = "IP",
    ):
        """
        初始化Milvus向量数据库连接和集合

        Args:
            collection_name: 集合名称
            uri: Milvus连接字符串 (例如: http://localhost:19530)
            dimension: 向量维度
            index_type: 索引类型，默认IVF_FLAT
            metric_type: 距离度量类型，默认IP (内积)
        """
        self.collection_name = collection_name
        self.uri = uri
        self.dimension = dimension
        self.index_type = index_type
        self.metric_type = metric_type

        self.client = MilvusClient(uri=uri)
        self._initialize()

    def _initialize(self):
        """
        初始化Milvus向量数据库连接和集合
        """
        try:
            logger.info(f"Connected to Milvus at {self.uri}")

            # 检查集合是否存在
            if self.has_collection():
                logger.info(f"Collection {self.collection_name} already exists")
            else:
                # 创建集合
                self._create_collection()
                logger.info(f"Created new collection: {self.collection_name}")

            # 加载集合到内存
            self.load_collection()

        except Exception as e:
            logger.error(f"Failed to initialize Milvus: {e}")
            raise

    def _create_collection(self):
        """
        创建集合和索引
        """
        try:
            # 定义 schema
            schema = MilvusClient.create_schema(
                auto_id=True,
                enable_dynamic_field=True,
                fields=[
                    {"field_name": "id", "datatype": "VARCHAR", "is_primary": True, "max_length": 256},
                    {"field_name": "embedding", "datatype": "FLOAT_VECTOR", "dim": self.dimension},
                    {"field_name": "content", "datatype": "VARCHAR", "max_length": 65535},
                    {"field_name": "source", "datatype": "VARCHAR", "max_length": 512},
                    {"field_name": "title", "datatype": "VARCHAR", "max_length": 256},
                    {"field_name": "exercise_id", "datatype": "VARCHAR", "max_length": 256},
                    {"field_name": "name", "datatype": "VARCHAR", "max_length": 256},
                ],
                description="Fitness knowledge collection",
            )

            # 创建集合
            self.client.create_collection(self.collection_name, schema=schema)

            # 创建索引
            index_config = {
                "index_type": self.index_type,
                "metric_type": self.metric_type,
                "params": {"nlist": 128} if self.index_type == "IVF_FLAT" else {},
            }

            self.client.create_index(
                collection_name=self.collection_name,
                field_name="embedding",
                index_config=index_config
            )

            logger.info(f"Created collection {self.collection_name} with {self.index_type} index")

        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise

    def has_collection(self, collection_name: Optional[str] = None) -> bool:
        """
        检查集合是否存在

        Args:
            collection_name: 集合名称，默认使用初始化时的集合名

        Returns:
            集合是否存在
        """
        try:
            name = collection_name or self.collection_name
            return self.client.has_collection(name)
        except Exception as e:
            logger.error(f"Failed to check collection existence: {e}")
            return False

    def list_collections(self) -> List[str]:
        """
        列出所有集合

        Returns:
            集合名称列表
        """
        try:
            collections = self.client.list_collections()
            # 处理返回格式
            result = []
            for coll in collections:
                if isinstance(coll, dict):
                    result.append(coll.get('collection_name', coll.get('name', '')))
                elif isinstance(coll, str):
                    result.append(coll)
            return result
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []

    def drop_collection(self, collection_name: Optional[str] = None) -> bool:
        """
        删除集合

        Args:
            collection_name: 集合名称，默认为初始化时的集合

        Returns:
            是否删除成功
        """
        try:
            name = collection_name or self.collection_name
            self.client.drop_collection(name)
            logger.info(f"Collection {name} dropped successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to drop collection: {e}")
            return False

    def insert_documents(
        self,
        data: List[Dict[str, Any]],
    ) -> bool:
        """
        插入文档数据

        Args:
            data: 文档数据列表，每项包含：
                - embedding: List[float] - 向量
                - content: str - 内容
                - 其他元数据字段（source, title, exercise_id, name等）

        Returns:
            是否插入成功
        """
        try:
            # 确保集合存在
            if not self.has_collection():
                self._create_collection()

            # 插入数据
            self.client.insert(
                collection_name=self.collection_name,
                data=data
            )

            logger.info(f"Inserted {len(data)} documents into {self.collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to insert documents: {e}")
            return False

    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        contents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """
        添加文档（向后兼容方法）

        Args:
            ids: 文档ID列表
            embeddings: 向量列表
            contents: 内容列表
            metadatas: 元数据列表

        Returns:
            是否添加成功
        """
        try:
            if not metadatas:
                metadatas = [{}] * len(ids)

            # 转换为新版 API 格式
            data = []
            for i, (id_val, emb, content) in enumerate(zip(ids, embeddings, contents)):
                metadata = metadatas[i] if i < len(metadatas) else {}
                item = {
                    "id": id_val,
                    "embedding": emb,
                    "content": content,
                    "exercise_id": metadata.get("exercise_id", ""),
                    "name": metadata.get("name", ""),
                    "source": metadata.get("source", ""),
                    "title": metadata.get("title", ""),
                }
                data.append(item)

            return self.insert_documents(data)

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_expr: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        向量搜索

        Args:
            query_embedding: 查询向量
            top_k: 返回的文档数量
            filter_expr: 可选的过滤表达式，例如 "source == 'wikipedia'"

        Returns:
            搜索结果列表
        """
        try:
            # 使用新版 API 搜索
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=top_k,
                filter=filter_expr or "",
                output_fields=["id", "content", "exercise_id", "name", "source", "title"],
            )

            # 格式化搜索结果
            formatted_results = []
            for result in results[0]:  # 第一个查询向量的结果
                formatted_results.append({
                    "id": result.get("id"),
                    "content": result.get("entity", {}).get("content", ""),
                    "score": float(result.get("distance", 0.0)),
                    "metadata": result.get("entity", {}),
                })

            logger.info(f"Successfully searched {len(formatted_results)} documents")
            return formatted_results

        except Exception as e:
            logger.error(f"Failed to search: {e}")
            return []

    def delete_by_ids(self, ids: List[str]) -> bool:
        """
        根据ID删除文档

        Args:
            ids: 文档ID列表

        Returns:
            是否删除成功
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                ids=ids
            )
            logger.info(f"Deleted {len(ids)} documents by IDs")
            return True
        except Exception as e:
            logger.error(f"Failed to delete by IDs: {e}")
            return False

    def delete_by_filter(self, filter_expression: str) -> bool:
        """
        根据过滤表达式删除文档

        Args:
            filter_expression: 过滤表达式，例如 "source == 'wikipedia'"

        Returns:
            是否删除成功
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                filter=filter_expression
            )
            logger.info(f"Deleted documents matching: {filter_expression}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete by filter: {e}")
            return False

    def delete_documents(self, exercise_ids: List[str]) -> bool:
        """
        删除文档（向后兼容方法）

        Args:
            exercise_ids: exercise_id列表

        Returns:
            是否成功
        """
        try:
            # 构建过滤表达式
            expr = " or ".join([f'exercise_id == "{eid}"' for eid in exercise_ids])
            return self.delete_by_filter(expr)
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息

        Returns:
            统计信息字典
        """
        try:
            info = self.client.describe_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "chunk_count": info.get("num_entities", 0),
                "dimension": self.dimension,
                "index_type": self.index_type,
                "metric_type": self.metric_type,
                "description": info.get("description", ""),
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}

    def get_collection_info(self) -> Dict[str, Any]:
        """
        获取集合详细信息

        Returns:
            集合信息字典
        """
        return self.get_collection_stats()

    def count_entities(self) -> int:
        """
        统计集合中的实体数量

        Returns:
            实体数量
        """
        try:
            info = self.client.describe_collection(self.collection_name)
            return info.get("num_entities", 0)
        except Exception as e:
            logger.error(f"Failed to count entities: {e}")
            return 0

    def clear_collection(self) -> bool:
        """
        清空集合的所有数据

        Returns:
            是否成功
        """
        try:
            # 删除并重新创建集合
            self.client.drop_collection(self.collection_name)
            self._create_collection()
            self.load_collection()

            logger.info(f"Cleared collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False

    def load_collection(self) -> bool:
        """
        加载集合到内存（用于搜索）

        Returns:
            是否加载成功
        """
        try:
            self.client.load_collection(self.collection_name)
            logger.info(f"Collection {self.collection_name} loaded to memory")
            return True
        except Exception as e:
            logger.error(f"Failed to load collection: {e}")
            return False

    def release_collection(self) -> bool:
        """
        释放集合内存

        Returns:
            是否释放成功
        """
        try:
            self.client.release_collection(self.collection_name)
            logger.info(f"Collection {self.collection_name} released from memory")
            return True
        except Exception as e:
            logger.error(f"Failed to release collection: {e}")
            return False

    def create_index(
        self,
        field_name: str = "embedding",
        index_type: Optional[str] = None,
        metric_type: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        创建索引

        Args:
            field_name: 字段名
            index_type: 索引类型（默认使用初始化时的类型）
            metric_type: 度量类型（默认使用初始化时的类型）
            params: 索引参数

        Returns:
            是否创建成功
        """
        try:
            idx_type = index_type or self.index_type
            m_type = metric_type or self.metric_type

            # 默认索引参数
            if params is None:
                params = {"nlist": 128} if idx_type == "IVF_FLAT" else {}

            index_config = {
                "index_type": idx_type,
                "metric_type": m_type,
                "params": params,
            }

            self.client.create_index(
                collection_name=self.collection_name,
                field_name=field_name,
                index_config=index_config
            )

            logger.info(f"Index created on field {field_name} with type {idx_type}")
            return True

        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return False

    def list_indexes(self) -> List[Dict[str, Any]]:
        """
        列出集合的所有索引

        Returns:
            索引信息列表
        """
        try:
            indexes = self.client.list_indexes(collection_name=self.collection_name)
            return indexes
        except Exception as e:
            logger.error(f"Failed to list indexes: {e}")
            return []

    def drop_index(self, field_name: str) -> bool:
        """
        删除索引

        Args:
            field_name: 字段名（索引名）

        Returns:
            是否删除成功
        """
        try:
            self.client.drop_index(
                collection_name=self.collection_name,
                index_name=field_name
            )
            logger.info(f"Index on field {field_name} dropped")
            return True
        except Exception as e:
            logger.error(f"Failed to drop index: {e}")
            return False

    def close(self):
        """
        关闭连接
        """
        try:
            self.client.close()
            logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.error(f"Failed to close connection: {e}")


# 使用示例
def example_usage():
    """
    Milvus VectorStore 使用示例
    """
    # 初始化
    store = VectorStore(
        collection_name="fitness_knowledge",
        uri="http://localhost:19530",
        dimension=1024,
    )

    # 检查集合是否存在
    if not store.has_collection():
        print("集合不存在，将自动创建")

    # 插入文档
    documents = [
        {
            "embedding": [0.1] * 1024,
            "content": "卧推是锻炼胸肌的最佳动作",
            "source": "fitness_guide",
            "title": "卧推教学",
            "exercise_id": "chest_001",
            "name": "卧推",
        },
        {
            "embedding": [0.2] * 1024,
            "content": "深蹲主要锻炼腿部肌肉",
            "source": "fitness_guide",
            "title": "深蹲教学",
            "exercise_id": "leg_001",
            "name": "深蹲",
        },
    ]
    store.insert_documents(documents)

    # 搜索
    query_vector = [0.1] * 1024
    results = store.search(query_vector, top_k=5)
    for result in results:
        print(f"Score: {result['score']:.3f}")
        print(f"Content: {result['content'][:50]}...")
        print(f"Source: {result['metadata'].get('source', 'N/A')}")
        print("---")

    # 统计
    count = store.count_entities()
    print(f"Total documents: {count}")

    # 获取集合信息
    info = store.get_collection_info()
    print(f"Collection info: {info}")

    # 列出所有集合
    collections = store.list_collections()
    print(f"All collections: {collections}")

    # 关闭连接
    store.close()
