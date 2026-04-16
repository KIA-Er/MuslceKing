"""
测试 Qwen3-Embedding 和 Qwen3-Reranker 模型
首次运行时会自动下载模型到本地缓存
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from muscleking.app.retrieval.embedding_model.qwen_embedding import QwenEmbedding
from muscleking.app.retrieval.reranker.qwen_reranker import QwenReranker


def test_embedding():
    """测试嵌入模型"""
    print("=" * 60)
    print("测试 Qwen3-Embedding-0.6B")
    print("=" * 60)

    # 初始化模型
    print("\n正在初始化嵌入模型（首次运行会下载模型）...")
    embedding_model = QwenEmbedding()

    # 测试编码
    test_texts = [
        "卧推是锻炼胸肌的最佳动作",
        "深蹲主要锻炼腿部肌肉",
        "硬拉是背部和腿部的重要训练动作"
    ]

    print(f"\n编码 {len(test_texts)} 个文本...")
    embeddings = embedding_model.encode(test_texts)

    print(f"✅ 编码成功!")
    print(f"嵌入维度: {len(embeddings[0])}")

    # 显示模型信息
    print(f"\n模型信息:")
    info = embedding_model.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    return embedding_model


def test_reranker():
    """测试重排序模型"""
    print("\n" + "=" * 60)
    print("测试 Qwen3-Reranker-0.6B")
    print("=" * 60)

    # 初始化模型
    print("\n正在初始化重排序模型（首次运行会下载模型）...")
    reranker = QwenReranker()

    # 测试重排序
    query = "如何锻炼胸肌"
    documents = [
        {
            "content": "卧推是锻炼胸肌的最佳动作，主要包括杠铃卧推和哑铃卧推。",
            "source": "fitness_guide",
        },
        {
            "content": "深蹲是锻炼腿部肌肉的最佳动作，主要针对股四头肌。",
            "source": "leg_training",
        },
        {
            "content": "俯卧撑是锻炼胸肌的经典动作，不需要任何器械。",
            "source": "home_workout",
        },
    ]

    print(f"\n查询: {query}")
    print(f"文档数: {len(documents)}")
    print(f"\n执行重排序...")

    reranked_docs = reranker.rerank(query, documents, top_k=2)

    print(f"✅ 重排序成功!")
    print(f"\nTop-2 结果:")
    for i, doc in enumerate(reranked_docs, 1):
        score = doc.get("rerank_score", 0.0)
        content = doc.get("content", "")[:40]
        print(f"  {i}. [{score:.4f}] {content}...")

    # 显示模型信息
    print(f"\n模型信息:")
    info = reranker.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    return reranker


def test_integration():
    """测试嵌入和重排序的联合使用"""
    print("\n" + "=" * 60)
    print("测试联合使用：嵌入 + 重排序")
    print("=" * 60)

    # 初始化模型
    print("\n初始化模型...")
    embedding_model = QwenEmbedding()
    reranker = QwenReranker()

    # 模拟检索场景
    query = "如何增肌"
    candidate_docs = [
        "肌肥大训练需要每组8-12次的重复次数",
        "有氧运动可以帮助减脂",
        "蛋白质摄入对肌肉生长至关重要",
        "睡眠对肌肉恢复非常重要",
    ]

    print(f"\n查询: {query}")
    print(f"候选文档数: {len(candidate_docs)}")

    # 1. 使用嵌入模型进行向量检索
    print(f"\n步骤1: 计算嵌入向量...")
    query_embedding = embedding_model.encode_single(query)
    doc_embeddings = embedding_model.encode(candidate_docs)

    # 简单的余弦相似度计算
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    query_vec = np.array(query_embedding).reshape(1, -1)
    doc_vecs = np.array(doc_embeddings)

    similarities = cosine_similarity(query_vec, doc_vecs)[0]

    print(f"嵌入相似度:")
    for i, (doc, sim) in enumerate(zip(candidate_docs, similarities)):
        print(f"  {i+1}. [{sim:.4f}] {doc[:30]}...")

    # 2. 使用重排序模型进行精细排序
    print(f"\n步骤2: 使用重排序模型...")
    doc_dicts = [{"content": doc} for doc in candidate_docs]
    reranked_docs = reranker.rerank(query, doc_dicts)

    print(f"重排序结果:")
    for i, doc in enumerate(reranked_docs, 1):
        score = doc.get("rerank_score", 0.0)
        content = doc.get("content", "")[:40]
        print(f"  {i}. [{score:.4f}] {content}...")

    print("\n✅ 联合测试完成!")


if __name__ == "__main__":
    print("🚀 开始测试 Qwen3 模型")
    print("首次运行时，模型会自动下载到本地缓存，请耐心等待...\n")

    try:
        # 测试嵌入模型
        embedding_model = test_embedding()

        # 测试重排序模型
        reranker = test_reranker()

        # 测试联合使用
        test_integration()

        print("\n" + "=" * 60)
        print("🎉 所有测试完成!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()