"""
知识库入库流程完整测试
测试文档处理、向量生成和向量存储的完整流程
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from muscleking.app.services.knowledge.ingestion_service import KnowledgeIngestionService
from muscleking.app.services.knowledge.document_processor import DocumentProcessor
from muscleking.app.services.knowledge.vector_schema import VectorSchema, VectorDocumentBuilder
from loguru import logger


def test_document_processing():
    """测试文档处理功能"""
    print("\n" + "=" * 60)
    print("测试1: 文档处理")
    print("=" * 60)

    # 初始化文档处理器
    processor = DocumentProcessor(
        chunk_size=500,
        chunk_overlap=100,
    )

    # 测试文件路径
    test_file = "/home/wenkai/project/MuslceKing/docs/knowledge_base/book_data_extract/MinerU_力量训练原理与实践__20251201151605.md"

    if not Path(test_file).exists():
        logger.error(f"测试文件不存在: {test_file}")
        return None

    # 处理文档
    chunks = processor.process_markdown_file(test_file)

    print(f"\n✅ 文档处理完成!")
    print(f"  文档块数: {len(chunks)}")

    # 显示前3个块的信息
    print(f"\n前3个文档块:")
    for i, chunk in enumerate(chunks[:3]):
        content = chunk["content"][:100]
        metadata = chunk["metadata"]
        print(f"\n  块 {i+1}:")
        print(f"    内容: {content}...")
        print(f"    章节: {metadata.get('chapter_title', 'N/A')}")
        print(f"    长度: {len(chunk['content'])} 字符")

    return chunks


def test_vector_schema():
    """测试向量库模式"""
    print("\n" + "=" * 60)
    print("测试2: 向量库模式")
    print("=" * 60)

    # 显示模式信息
    print(f"\n向量维度: {VectorSchema.EMBEDDING_DIMENSION}")

    print(f"\n推荐的集合名称:")
    for name, collection in VectorSchema.COLLECTION_NAMES.items():
        print(f"  {name}: {collection}")

    # 示例文档构建
    print(f"\n示例文档构建:")
    sample_chunk = {
        "content": "卧推是锻炼胸肌的最佳动作，主要包括杠铃卧推和哑铃卧推。这项运动可以有效刺激胸大肌、三角肌前束和肱三头肌。",
        "metadata": {
            "book_title": "力量训练原理与实践",
            "chapter_title": "上肢训练",
            "source_file": "/path/to/book.md",
        }
    }

    # 模拟嵌入向量
    import random
    fake_embedding = [random.random() for _ in range(VectorSchema.EMBEDDING_DIMENSION)]

    document = VectorDocumentBuilder.build_document(
        sample_chunk,
        fake_embedding,
        "doc_001"
    )

    print(f"  文档ID: {document['id']}")
    print(f"  类别: {document['category']}")
    print(f"  难度: {document['difficulty_level']}")
    print(f"  质量分数: {document['quality_score']:.2f}")
    print(f"  关键词: {document['keywords']}")
    print(f"  标签: {document['tags']}")

    print(f"\n✅ 向量库模式测试完成!")
    return True


def test_embedding_generation():
    """测试嵌入生成"""
    print("\n" + "=" * 60)
    print("测试3: 嵌入向量生成")
    print("=" * 60)

    try:
        from muscleking.app.retrieval.embedding_model.qwen_embedding import QwenEmbedding

        # 初始化嵌入模型
        print("\n初始化嵌入模型...")
        embedding_model = QwenEmbedding()

        # 测试文本
        test_texts = [
            "卧推是锻炼胸肌的最佳动作",
            "深蹲主要锻炼腿部肌肉",
            "硬拉是背部和腿部的重要训练动作"
        ]

        print(f"生成 {len(test_texts)} 个文本的嵌入向量...")
        embeddings = embedding_model.encode(test_texts)

        print(f"\n✅ 嵌入生成完成!")
        print(f"  文本数: {len(test_texts)}")
        print(f"  嵌入维度: {len(embeddings[0])}")

        # 显示前几个值
        print(f"\n第一个嵌入向量的前5个值:")
        print(f"  {embeddings[0][:5]}")

        return embeddings

    except Exception as e:
        logger.error(f"嵌入生成测试失败: {e}")
        return None


def test_full_ingestion():
    """测试完整的入库流程"""
    print("\n" + "=" * 60)
    print("测试4: 完整入库流程")
    print("=" * 60)

    try:
        # 初始化入库服务
        print("\n初始化入库服务...")
        ingestion_service = KnowledgeIngestionService(
            milvus_uri="http://localhost:19530",
            collection_name="fitness_knowledge_test",  # 使用测试集合
        )

        # 测试文件路径
        test_file = "/home/wenkai/project/MuslceKing/docs/knowledge_base/book_data_extract/MinerU_力量训练原理与实践__20251201151605.md"

        if not Path(test_file).exists():
            logger.error(f"测试文件不存在: {test_file}")
            return False

        print(f"\n开始处理文件: {Path(test_file).name}")
        print("这可能需要几分钟时间...")

        # 处理文件（只处理前10个块以加快测试速度）
        result = ingestion_service.ingest_file(
            file_path=test_file,
            use_semantic_chunking=True,
            skip_existing=False,  # 测试时不跳过已存在的文档
        )

        print(f"\n✅ 入库完成!")
        print(f"  成功: {result['success']}")
        print(f"  总块数: {result.get('total_chunks', 0)}")
        print(f"  新文档: {result.get('new_documents', 0)}")
        print(f"  跳过: {result.get('skipped_documents', 0)}")
        print(f"  耗时: {result.get('processing_time', 0):.2f}秒")

        # 获取统计信息
        stats = ingestion_service.get_collection_stats()
        print(f"\n向量库统计:")
        print(f"  集合名称: {stats['collection_name']}")
        print(f"  总文档数: {stats['total_documents']}")
        print(f"  嵌入维度: {stats['embedding_dimension']}")

        return True

    except Exception as e:
        logger.error(f"入库测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_ingestion():
    """测试批量入库"""
    print("\n" + "=" * 60)
    print("测试5: 批量入库（可选）")
    print("=" * 60)

    try:
        # 初始化入库服务
        print("\n初始化入库服务...")
        ingestion_service = KnowledgeIngestionService(
            milvus_uri="http://localhost:19530",
            collection_name="fitness_knowledge_batch",
        )

        # 测试目录
        test_directory = "/home/wenkai/project/MuslceKing/docs/knowledge_base/book_data_extract"

        if not Path(test_directory).exists():
            logger.error(f"测试目录不存在: {test_directory}")
            return False

        print(f"\n开始批量处理目录: {test_directory}")
        print("这将处理目录下的所有 .md 文件，可能需要较长时间...")

        # 批量处理（只处理前3个文件）
        batch_result = ingestion_service.ingest_directory(
            directory_path=test_directory,
            file_pattern="*.md",
            use_semantic_chunking=True,
            skip_existing=False,
        )

        print(f"\n✅ 批量入库完成!")
        print(f"  成功: {batch_result['success']}")
        print(f"  总文件数: {batch_result['total_files']}")
        print(f"  成功文件: {batch_result['successful_files']}")
        print(f"  失败文件: {batch_result['failed_files']}")
        print(f"  新文档: {batch_result['total_new_documents']}")
        print(f"  跳过文档: {batch_result['total_skipped_documents']}")
        print(f"  总耗时: {batch_result['processing_time']:.2f}秒")

        return True

    except Exception as e:
        logger.error(f"批量入库测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🚀 知识库入库流程完整测试")
    print("=" * 60)

    # 测试1: 文档处理
    chunks = test_document_processing()
    if not chunks:
        logger.warning("文档处理测试失败，跳过后续测试")
        return

    # 测试2: 向量库模式
    test_vector_schema()

    # 测试3: 嵌入生成
    embeddings = test_embedding_generation()
    if not embeddings:
        logger.warning("嵌入生成测试失败，跳过后续测试")
        return

    # 测试4: 完整入库流程
    print("\n" + "=" * 60)
    print("⚠️  即将开始完整入库测试")
    print("这将连接到 Milvus 并实际插入数据")
    print("确保 Milvus 已启动 (docker-compose up -d)")
    print("=" * 60)

    response = input("\n是否继续完整入库测试？(y/n): ")
    if response.lower() == 'y':
        test_full_ingestion()
    else:
        print("跳过完整入库测试")

    # 测试5: 批量入库（可选）
    print("\n" + "=" * 60)
    print("⚠️  批量入库测试（可选）")
    print("这将处理所有书籍文件，可能需要很长时间")
    print("=" * 60)

    response = input("\n是否运行批量入库测试？(y/n): ")
    if response.lower() == 'y':
        test_batch_ingestion()
    else:
        print("跳过批量入库测试")

    print("\n" + "=" * 60)
    print("🎉 测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()