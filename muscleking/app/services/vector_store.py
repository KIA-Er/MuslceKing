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


class MilvusService:
    """
    Milvus 服务管理类 - 使用新版 MilvusClient API

    提供更简洁的 Milvus 操作接口，支持：
    - 集合管理（创建、删除、列出）
    - 数据管理（插入、删除、更新）
    - 向量搜索
    - 索引管理
    """

    def __init__(
        self,
        collection_name: str = "fitness_knowledge",
        uri: str = "http://localhost:19530",
        dimension: int = 1024,
    ):
        """
        初始化 Milvus 服务

        Args:
            collection_name: 集合名称
            uri: Milvus 连接字符串
            dimension: 向量维度
        """
        self.collection_name = collection_name
        self.uri = uri
        self.dimension = dimension
        self.client = MilvusClient(uri=uri)
        logger.info(f"MilvusService initialized with collection: {collection_name}")

    def create_collection(
        self,
        dimension: int = 1024,
        description: str = "Fitness knowledge collection",
    ) -> bool:
        """
        创建新集合

        Args:
            dimension: 向量维度
            description: 集合描述

        Returns:
            是否创建成功
        """
        try:
            # 检查集合是否已存在
            if self.client.has_collection(self.collection_name):
                logger.warning(f"Collection {self.collection_name} already exists")
                return True

            # 定义 schema
            schema = MilvusClient.create_schema(
                auto_id=True,
                enable_dynamic_field=True,
                fields=[
                    {"field_name": "id", "datatype": "VARCHAR", "is_primary": True, "max_length": 256},
                    {"field_name": "embedding", "datatype": "FLOAT_VECTOR", "dim": dimension},
                    {"field_name": "content", "datatype": "VARCHAR", "max_length": 65535},
                    {"field_name": "source", "datatype": "VARCHAR", "max_length": 512},
                    {"field_name": "title", "datatype": "VARCHAR", "max_length": 256},
                ],
                description=description,
            )

            # 创建集合
            self.client.create_collection(self.collection_name, schema=schema)
            logger.info(f"Collection {self.collection_name} created successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False

    def list_collections(self) -> List[str]:
        """
        列出所有集合

        Returns:
            集合名称列表
        """
        try:
            collections = self.client.list_collections()
            return [coll['collection_name'] for coll in collections]
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

    def has_collection(self, collection_name: Optional[str] = None) -> bool:
        """
        检查集合是否存在

        Args:
            collection_name: 集合名称，默认为初始化时的集合

        Returns:
            集合是否存在
        """
        try:
            name = collection_name or self.collection_name
            return self.client.has_collection(name)
        except Exception as e:
            logger.error(f"Failed to check collection existence: {e}")
            return False

    def insert_documents(
        self,
        data: List[Dict[str, Any]],
    ) -> bool:
        """
        插入文档到集合

        Args:
            data: 文档数据列表，每项包含：
                - embedding: List[float] - 向量
                - content: str - 内容
                - source: str - 来源
                - title: str - 标题
                - 其他元数据字段

        Returns:
            是否插入成功
        """
        try:
            # 确保集合存在
            if not self.has_collection():
                self.create_collection(dimension=self.dimension)

            # 插入数据
            insert_result = self.client.insert(
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
        filter_expression: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        向量搜索

        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量
            filter_expression: 过滤表达式，例如 "source == 'wikipedia'"
            output_fields: 返回的字段列表

        Returns:
            搜索结果列表，每项包含：
                - id: 文档ID
                - distance: 相似度距离
                - entity: 文档实体（包含所有字段）
        """
        try:
            # 确保集合存在
            if not self.has_collection():
                logger.warning(f"Collection {self.collection_name} does not exist")
                return []

            # 默认返回字段
            if output_fields is None:
                output_fields = ["content", "source", "title"]

            # 执行搜索
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=top_k,
                output_fields=output_fields,
                filter=filter_expression,
            )

            # 格式化结果
            formatted_results = []
            for result in results[0]:  # 第一个查询向量的结果
                formatted_results.append({
                    "id": result.get("id"),
                    "distance": result.get("distance"),
                    "entity": result.get("entity", {}),
                })

            logger.info(f"Found {len(formatted_results)} results")
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
            logger.info(f"Deleted {len(ids)} documents")
            return True

        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
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

    def get_collection_info(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        获取集合信息

        Args:
            collection_name: 集合名称，默认为初始化时的集合

        Returns:
            集合信息字典
        """
        try:
            name = collection_name or self.collection_name

            # 使用 describe_collection 获取详细信息
            info = self.client.describe_collection(name)

            return {
                "collection_name": name,
                "description": info.get("description", ""),
                "num_entities": info.get("num_entities", 0),
                "fields": info.get("fields", []),
            }

        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}

    def create_index(
        self,
        field_name: str = "embedding",
        index_type: str = "IVF_FLAT",
        metric_type: str = "IP",
        params: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        创建索引

        Args:
            field_name: 字段名
            index_type: 索引类型 (IVF_FLAT, HNSW, FLAT 等)
            metric_type: 度量类型 (IP, L2, COSINE)
            params: 索引参数

        Returns:
            是否创建成功
        """
        try:
            # 默认索引参数
            if params is None:
                params = {"nlist": 128} if index_type == "IVF_FLAT" else {}

            index_config = {
                "index_type": index_type,
                "metric_type": metric_type,
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
            return [idx.get("index_name", "") for idx in indexes]
        except Exception as e:
            logger.error(f"Failed to list indexes: {e}")
            return []

    def drop_index(self, index_name: str) -> bool:
        """
        删除索引

        Args:
            index_name: 索引名称

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

    def close(self):
        """关闭 Milvus 连接"""
        try:
            self.client.close()
            logger.info("Milvus connection closed")
        except Exception as e:
            logger.error(f"Failed to close connection: {e}")


# 使用示例
def example_usage():
    """
    Milvus 服务使用示例
    """
    # 初始化服务
    service = MilvusService(
        collection_name="fitness_knowledge",
        uri="http://localhost:19530",
        dimension=1024,
    )

    # 创建集合
    service.create_collection(dimension=1024)

    # 插入文档
    documents = [
        {
            "embedding": [0.1] * 1024,
            "content": "卧推是锻炼胸肌的最佳动作",
            "source": "fitness_guide",
            "title": "卧推教学",
        },
        {
            "embedding": [0.2] * 1024,
            "content": "深蹲主要锻炼腿部肌肉",
            "source": "fitness_guide",
            "title": "深蹲教学",
        },
    ]
    service.insert_documents(documents)

    # 搜索
    query_vector = [0.1] * 1024
    results = service.search(query_vector, top_k=5)
    for result in results:
        print(f"Score: {result['distance']}, Content: {result['entity']['content'][:50]}...")

    # 获取集合信息
    info = service.get_collection_info()
    print(f"Collection info: {info}")

    # 统计文档数量
    count = service.count_entities()
    print(f"Total documents: {count}")

    # 关闭连接
    service.close()
