from langchain.chat_models import BaseChatModel, init_chat_model
from loguru import logger

from muscleking.app.config import Settings
from muscleking.app.retrieval.embedding_model.qwen_embedding import QwenEmbedding
from muscleking.app.retrieval.reranker.qwen_reranker import QwenReranker
from muscleking.app.retrieval.retriever import Retriever

from muscleking.app.storage.vector_storage.milvus_storage import MilvusStorage

logger = logger.bind(module="模型后端工厂")

class ModelBackendFactory:

    @staticmethod
    def create_llm(settings: Settings) -> BaseChatModel:
        llm = init_chat_model(
            model=settings.LLM_MODEL,
            model_provider=settings.LLM_PROVIDER,
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
        )
        
        logger.info("✅ llm 对象实例化成功")
        return llm

    @staticmethod
    def create_embedding_model(settings: Settings) -> QwenEmbedding:
        qwen_embedding = QwenEmbedding(
            model_name=settings.EMBEDDING_MODEL,
        )

        logger.info("✅ Milvus 存储对象实例化成功")
        return qwen_embedding

    @staticmethod
    def create_reranker(settings: Settings) -> QwenReranker:
        qwen_reranker = QwenReranker(
            model_name=settings.RERANK_MODEL,
            top_k=settings.RERANK_TOP_N,
            threshold=settings.RERANK_SCORE_THRESHOLD,
        )

        logger.info("✅ Milvus 存储对象实例化成功")
        return qwen_reranker

    @staticmethod
    def create_retriever(
        vector_storeage: MilvusStorage,
        embedding_model: QwenEmbedding,
        reranker: QwenReranker | None = None,
    ) -> Retriever:
        retriever = Retriever(
            embedding_model=embedding_model,
            vector_storage=vector_storeage,
            reranker=reranker,
        )

        logger.info("✅ Milvus 存储对象实例化成功")
        return retriever