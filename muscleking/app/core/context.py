from muscleking.app.config import Settings
from muscleking.app.resource.factories.middleware_factory import MiddlewareFactory
from muscleking.app.resource.factories.model_factory import ModelBackendFactory
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from muscleking.app.retrieval.embedding_model.qwen_embedding import QwenEmbedding
from muscleking.app.retrieval.reranker.qwen_reranker import QwenReranker
from muscleking.app.retrieval.retriever import Retriever


class AppContext:
    def __init__(self, settings: Settings) -> None:
        self.llm = ModelBackendFactory.create_llm(settings)

        # Milvus 客户端
        self.milvus_client = MiddlewareFactory.create_milvus_client(settings)
        self.milvus_storage = MiddlewareFactory.create_milvus_storage(self.milvus_client, settings)

        # 检索器（内部持有 embedding_model + reranker）
        self.embedding_model: QwenEmbedding = ModelBackendFactory.create_embedding_model(settings)
        self.reranker: QwenReranker | None = ModelBackendFactory.create_reranker(settings) if settings.ENABLE_RERANK else None
        self.retriever: Retriever = ModelBackendFactory.create_retriever(
            self.milvus_storage,
            self.embedding_model,
            self.reranker,
        )

        self.checkpointer: AsyncPostgresSaver | None = None


    async def close(self):
        if self.milvus_client:
            self.milvus_client.close()


_global_app_context: AppContext | None = None


def init_global_context(settings: Settings):
    global _global_app_context
    if _global_app_context is not None:
        raise ValueError("Global context is already initialized")
    _global_app_context = AppContext(settings)


def get_global_context() -> AppContext:
    global _global_app_context
    if not _global_app_context:
        raise ValueError("Global context is not initialized")
    return _global_app_context


async def close_global_context():
    global _global_app_context
    if _global_app_context:
        await _global_app_context.close()
        _global_app_context = None