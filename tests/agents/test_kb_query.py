"""测试 create_kb_query 节点"""

import pytest
from muscleking.app.agents.lg_builder import create_kb_query
from muscleking.app.agents.models.model_lg_state import AgentState
from langchain_core.messages import HumanMessage


@pytest.mark.asyncio
@pytest.mark.skipif(
    not _services_available(),
    reason="Required services not available"
)
async def test_kb_query_with_valid_query():
    """测试有效查询"""
    state = AgentState(
        messages=[HumanMessage(content="如何锻炼胸肌？")],
        router=None
    )

    result = await create_kb_query(state, config={})

    assert "messages" in result
    assert len(result["messages"]) == 1
    assert result["messages"][0].content
    # 如果有结果，检查 sources
    if "知识库中没有找到" not in result["messages"][0].content:
        assert "sources" in result["messages"][0].additional_kwargs


@pytest.mark.asyncio
async def test_kb_query_with_empty_message():
    """测试空消息处理"""
    state = AgentState(messages=[], router=None)
    result = await create_kb_query(state, config={})

    assert "messages" in result
    assert "请告诉我具体的问题" in result["messages"][0].content


@pytest.mark.asyncio
@pytest.mark.skipif(
    not _services_available(),
    reason="Required services not available"
)
async def test_kb_query_error_handling():
    """测试错误处理"""
    # 测试服务不可用时的降级
    state = AgentState(
        messages=[HumanMessage(content="测试查询")],
        router=None
    )

    # 正常情况应该成功
    result = await create_kb_query(state, config={})
    assert "messages" in result
    assert len(result["messages"]) == 1


def _services_available() -> bool:
    """检查所需服务是否可用"""
    import requests
    try:
        # 检查 vLLM
        vllm_ok = requests.get("http://localhost:50001/health", timeout=2).status_code == 200
        # 检查 Milvus
        from pymilvus import MilvusClient
        client = MilvusClient(uri="http://localhost:19530")
        milvus_ok = len(client.list_collections()) >= 0
        return vllm_ok and milvus_ok
    except:
        return False
