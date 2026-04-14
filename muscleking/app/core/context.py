from typing import Optional
from muscleking.app.config import Settings
from muscleking.app.resource.factories.middleware_factory import MiddlewareFactory
from muscleking.app.resource.factories.model_factory import ModelBackendFactory
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from muscleking.app.rag.global_retriever import GlobalRetriever


class AppContext:
    def __init__(self, settings: Settings) -> None:
        self.llm = ModelBackendFactory.create_llm(settings)
        self.embedding_model = ModelBackendFactory.create_embedding_model(settings)
        self.reranker = ModelBackendFactory.create_reranker(settings)

        self.checkpointer:AsyncPostgresSaver | None = None
        self._retriever: Optional[GlobalRetriever] = None

    @property
    def retriever(self) -> GlobalRetriever:
        """获取全局检索器单例"""
        if self._retriever is None:
            self._retriever = GlobalRetriever()
        return self._retriever

    def close(self,):
        pass

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
