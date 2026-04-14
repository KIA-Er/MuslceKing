"""测试 GlobalRetriever"""

import pytest
from muscleking.app.rag.global_retriever import GlobalRetriever


def test_retriever_singleton():
    """测试单例模式"""
    retriever1 = GlobalRetriever()
    retriever2 = GlobalRetriever()
    assert retriever1 is retriever2


@pytest.mark.skipif(
    not _milvus_available(),
    reason="Milvus not available"
)
def test_retriever_initialized():
    """测试初始化状态"""
    retriever = GlobalRetriever()
    assert retriever._initialized is True
    assert retriever.embedding_client is not None
    assert retriever.reranker is not None
    assert retriever.milvus_client is not None


@pytest.mark.skipif(
    not _milvus_available(),
    reason="Milvus not available"
)
def test_retrieve_returns_documents():
    """测试检索返回文档"""
    retriever = GlobalRetriever()
    result = retriever.retrieve(
        query="如何锻炼胸肌？",
        top_k=5,
        rerank_top_n=3,
    )

    assert "results" in result
    assert "recall" in result
    assert "rerank_scores" in result
    assert isinstance(result["results"], list)


@pytest.mark.asyncio
@pytest.mark.skipif(
    not _milvus_available(),
    reason="Milvus not available"
)
async def test_aretrieve():
    """测试异步检索"""
    retriever = GlobalRetriever()
    result = await retriever.aretrieve(
        query="如何锻炼胸肌？",
        top_k=5,
        rerank_top_n=3,
    )

    assert "results" in result
    assert isinstance(result["results"], list)


def _milvus_available() -> bool:
    """检查 Milvus 是否可用"""
    try:
        from pymilvus import MilvusClient
        client = MilvusClient(uri="http://localhost:19530")
        # 尝试列出集合
        client.list_collections()
        return True
    except:
        return False
