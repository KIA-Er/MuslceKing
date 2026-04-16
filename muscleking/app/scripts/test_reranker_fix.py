"""
测试修复后的 Qwen3-Reranker 模型
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from muscleking.app.retrieval.reranker.qwen_reranker import QwenReranker


def test_reranker():
    """
    测试修复后的 Qwen3-Reranker 模型
    """
    import time

    print("=" * 60)
    print("测试修复后的 Qwen3-Reranker-0.6B 模型")
    print("=" * 60)

    # 初始化模型
    print("\n1. 初始化模型...")
    start_time = time.time()

    reranker = QwenReranker(
        model_name="Qwen/Qwen3-Reranker-0.6B",
    )

    init_time = time.time() - start_time
    print(f"初始化耗时: {init_time:.2f}秒")

    # 准备测试数据
    print("\n2. 准备测试数据...")
    query = "如何锻炼胸肌"
    documents = [
        {
            "content": "卧推是锻炼胸肌的最佳动作，主要包括杠铃卧推和哑铃卧推两种方式，可以全面刺激胸大肌。",
            "source": "fitness_guide",
        },
        {
            "content": "深蹲是锻炼腿部肌肉的最佳动作，主要针对股四头肌、臀大肌等肌群。",
            "source": "leg_training",
        },
        {
            "content": "俯卧撑是锻炼胸肌的经典动作，不需要任何器械，随时随地进行。",
            "source": "home_workout",
        },
    ]

    print(f"查询: {query}")
    print(f"文档数: {len(documents)}")

    # 测试重排序
    print("\n3. 执行重排序...")
    start_time = time.time()
    reranked_docs = reranker.rerank(query, documents, top_k=3)
    rerank_time = time.time() - start_time

    print(f"重排序耗时: {rerank_time:.4f}秒")
    print(f"返回 top-3 结果:")

    for i, doc in enumerate(reranked_docs, 1):
        score = doc.get("rerank_score", 0.0)
        content = doc.get("content", "")[:50]
        source = doc.get("source", "unknown")
        print(f"  {i}. [{score:.4f}] {content}... (来源: {source})")

    # 测试分数计算
    print("\n4. 测试分数计算...")
    doc_texts = [doc["content"] for doc in documents]
    scores = reranker.compute_scores(query, doc_texts)

    print("所有文档的相关性分数:")
    for i, (doc, score) in enumerate(zip(documents, scores)):
        content = doc["content"][:30]
        print(f"  {i+1}. [{score:.4f}] {content}...")

    print("\n" + "=" * 60)
    print("✅ 测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_reranker()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()