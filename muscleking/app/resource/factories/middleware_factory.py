"""
工厂模式创建各种中间件和资源
包括 Milvus Client、PostgreSQL Checkpointer 等
"""

from pymilvus import MilvusClient
from loguru import logger

from muscleking.app.config import Settings
from muscleking.app.storage.vector_storage.milvus_storage import MilvusStorage

logger = logger.bind(module="中间件工厂")

class MiddlewareFactory:
    """中间件和资源工厂类"""

    @staticmethod
    def create_milvus_client(settings: Settings) -> MilvusClient:
        """创建 Milvus 客户端"""
        client = MilvusClient(uri=settings.MILVUS_CONNECTION_STRING)
        logger.info("✅ Milvus 客户端连接成功")
        return client
    
    @staticmethod
    def create_milvus_storage(milvus_client: MilvusClient, settings: Settings) -> MilvusStorage:
        """创建 Milvus 存储实例"""
        milvus_storage = MilvusStorage(
            milvus_client,
            collection_name=settings.MILVUS_COLLECTION,
            embedding_dimension=settings.EMBEDDING_DIMENSION,
        )
        logger.info("✅ Milvus 存储对象实例化成功")
        return milvus_storage