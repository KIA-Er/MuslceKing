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
    MilvusClient,
)


class VectorStore:
    """
    Milvus向量数据库封装类

    提供完整的 Milvus 操作接口：
    - 集合管理（创建、删除、列出、检查存在）
    - 数据管理（插入、删除、更新）
    - 向量搜索
    - 索引管理
    - 统计信息
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

        Args:
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
        self.uri = f"http://{host}:{port}"

        self.collection = None
        self.client = None  # 新版 MilvusClient
        self._initialize()

    def _initialize(self):
        """
        初始化Milvus向量数据库连接和集合
        同时支持旧版和新版 API
        """
        try:
            # 初始化新版 MilvusClient
            self.client = MilvusClient(uri=self.uri)

            # 检查集合是否存在（使用新版 API）
            if self.has_collection():
                logger.info(f"Collection {self.collection_name} already exists")
                # 仍然加载到旧版 Collection 对象（保持兼容性）
                if utility.has_collection(self.collection_name, using="default"):
                    self.collection = Collection(self.collection_name, using="default")
            else:
                # 创建集合
                self._create_collection()
                logger.info(f"Created new collection: {self.collection_name}")

            # 把集合加载到内存中
            self.load_collection()

        except Exception as e:
            logger.error(f"Failed to connect to Milvus server: {e}")
            raise

    def _create_collection(self):
        """
        创建集合
        使用新版 MilvusClient API 创建集合
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

            # 使用新版 API 创建集合
            self.client.create_collection(self.collection_name, schema=schema)

            # 创建索引
            index_params = {
                "index_type": self.index_type,
                "metric_type": self.metric_type,
                "params": {"nlist": 128} if self.index_type == "IVF_FLAT" else {},
            }

            self.client.create_index(
                collection_name=self.collection_name,
                field_name="embedding",
                index_config=index_params
            )

            logger.info(f"Created collection {self.collection_name} with index")

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
            return [coll.get('collection_name', coll.get('name', '')) for coll in collections]
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
        try:
            if not metadatas:
                metadatas = [{}] * len(ids)

            # 准备插入数据（使用新版 API 格式）
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

            # 使用新版 API 插入
            self.client.insert(
                collection_name=self.collection_name,
                data=data
            )

            logger.info(f"Successfully added {len(ids)} documents to collection {self.collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to add documents to Milvus: {e}")
            return False

    def insert_documents(self, data: List[Dict[str, Any]]) -> bool:
        """
        直接插入文档数据（新版 API 风格）

        Args:
            data: 文档数据列表，每项包含：
                - embedding: List[float] - 向量
                - content: str - 内容
                - 其他元数据字段

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
            # 使用新版 API 搜索
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=top_k,
                filter=filter_expr,
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
            logger.error(f"Failed to search in Milvus: {e}")
            return []

    def delete_documents(self, exercise_ids: List[str]) -> bool:
        """
        删除文档（保持向后兼容）

        Args:
            exercise_ids: 文档ID列表（exercise_id）

        Returns:
            是否成功
        """
        try:
            # 构建过滤表达式
            expr = " or ".join([f'exercise_id == "{eid}"' for eid in exercise_ids])
            self.client.delete(
                collection_name=self.collection_name,
                filter=expr
            )
            logger.info(f"Deleted {len(exercise_ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False

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

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息
        """
        try:
            # 使用新版 API 获取信息
            info = self.client.describe_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "chunk_count": info.get("num_entities", 0),
                "dimension": self.dimension,
                "index_type": self.index_type,
                "metric_type": self.metric_type,
                "description": info.get("description", ""),
                "fields": info.get("fields", []),
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}

    def get_collection_info(self) -> Dict[str, Any]:
        """
        获取集合详细信息（别名方法）

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
        """
        try:
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
            # 使用新版 API
            self.client.load_collection(self.collection_name)

            # 如果有旧版 Collection 对象，也加载
            if self.collection:
                self.collection.load()

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
            # 使用新版 API
            self.client.release_collection(self.collection_name)

            # 如果有旧版 Collection 对象，也释放
            if self.collection:
                self.collection.release()

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

            logger.info(f"Index created on field {field_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return False

    def list_indexes(self) -> List[str]:
        """
        列出集合的所有索引

        Returns:
            索引名称列表
        """
        try:
            indexes = self.client.list_indexes(collection_name=self.collection_name)
            return [idx.get("index_name", idx.get("field_name", "")) for idx in indexes]
        except Exception as e:
            logger.error(f"Failed to list indexes: {e}")
            return []

    def drop_index(self, index_name: str) -> bool:
        """
        删除索引

        Args:
            index_name: 索引名称或字段名

        Returns:
            是否删除成功
        """
        try:
            self.client.drop_index(
                collection_name=self.collection_name,
                index_name=index_name
            )
            logger.info(f"Index {index_name} dropped")
            return True
        except Exception as e:
            logger.error(f"Failed to drop index: {e}")
            return False

    def close(self):
        """
        关闭连接
        """
        try:
            # 关闭新版连接
            if self.client:
                self.client.close()

            # 关闭旧版连接
            connections.disconnect("default")

            logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.error(f"Failed to disconnect: {e}")
