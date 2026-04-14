"""测试 VLLMEmbeddingClient"""

import pytest
from muscleking.app.rag.embeddings import VLLMEmbeddingClient


@pytest.fixture
def embedding_client():
    """创建 EmbeddingClient fixture"""
    return VLLMEmbeddingClient()


def test_embedding_client_init(embedding_client):
    """测试客户端初始化"""
    assert embedding_client is not None
    assert embedding_client.dimension == 1024


@pytest.mark.skipif(
    not _vllm_available(),
    reason="vLLM service not available"
)
def test_embed_query(embedding_client):
    """测试单个文本 embedding"""
    text = "如何锻炼胸肌？"
    result = embedding_client.embed_query(text)

    assert isinstance(result, list)
    assert len(result) == 1024
    assert all(isinstance(x, float) for x in result)


@pytest.mark.skipif(
    not _vllm_available(),
    reason="vLLM service not available"
)
def test_embed_documents(embedding_client):
    """测试批量 embedding"""
    texts = ["如何锻炼胸肌？", "卧推的正确姿势"]
    results = embedding_client.embed_documents(texts)

    assert isinstance(results, list)
    assert len(results) == 2
    assert all(len(r) == 1024 for r in results)


@pytest.mark.skipif(
    not _vllm_available(),
    reason="vLLM service not available"
)
def test_health_check(embedding_client):
    """测试健康检查"""
    result = embedding_client.health_check()
    assert isinstance(result, bool)


def _vllm_available() -> bool:
    """检查 vLLM 服务是否可用"""
    import requests
    try:
        response = requests.get("http://localhost:50001/health", timeout=2)
        return response.status_code == 200
    except:
        return False
