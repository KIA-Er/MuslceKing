#!/usr/bin/env python3
"""端到端 RAG 测试"""

import asyncio
import sys
from muscleking.app.core.context import get_global_context
from muscleking.app.agents.lg_builder import create_kb_query
from muscleking.app.agents.models.model_lg_state import AgentState
from langchain_core.messages import HumanMessage


async def test_global_retriever():
    """测试 GlobalRetriever"""
    print("🧪 Testing GlobalRetriever...")
    ctx = get_global_context()
    retriever = ctx.retriever

    result = await retriever.aretrieve(
        query="如何锻炼胸肌？",
        top_k=20,
        rerank_top_n=5,
    )

    print(f"✅ Retriever returned {len(result['results'])} documents")
    for i, doc in enumerate(result['results'][:3]):
        print(f"   {i+1}. {doc.page_content[:50]}...")
    return True


async def test_kb_query_node():
    """测试 create_kb_query 节点"""
    print("\n🧪 Testing create_kb_query node...")
    state = AgentState(
        messages=[HumanMessage(content="如何锻炼胸肌？")],
        router=None
    )

    result = await create_kb_query(state, config={})

    print(f"✅ Generated response: {result['messages'][0].content[:100]}...")
    if "sources" in result["messages"][0].additional_kwargs:
        print(f"   Sources: {result['messages'][0].additional_kwargs['sources']}")
    return True


async def main():
    """运行所有 E2E 测试"""
    print("🚀 RAG End-to-End Tests")
    print("=" * 50)

    try:
        await test_global_retriever()
        await test_kb_query_node()

        print("\n" + "=" * 50)
        print("✅ All E2E tests passed!")
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
