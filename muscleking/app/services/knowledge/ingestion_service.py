"""
知识库入库服务
整合文档处理、向量生成和向量存储的完整流程
"""

import time
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
from loguru import logger

from pymilvus import MilvusClient

from muscleking.app.retrieval.embedding_model.qwen_embedding import QwenEmbedding
from muscleking.app.services.knowledge.document_processor import DocumentProcessor
from muscleking.app.services.knowledge.vector_schema import (
    VectorSchema,
    VectorDocumentBuilder,
)


class KnowledgeIngestionService:
    """
    知识库入库服务

    功能：
    1. 文档处理和分块
    2. 向量生成
    3. 向量库存储
    4. 增量更新
    """

    def __init__(
        self,
        milvus_uri: str = "http://localhost:19530",
        collection_name: str = "fitness_knowledge",
        embedding_model: Optional[QwenEmbedding] = None,
        batch_size: int = 32,
    ):
        """
        初始化入库服务

        Args:
            milvus_uri: Milvus连接URI
            collection_name: 集合名称
            embedding_model: 嵌入模型
            batch_size: 批处理大小
        """
        self.milvus_uri = milvus_uri
        self.collection_name = collection_name
        self.batch_size = batch_size

        # 初始化Milvus客户端
        self.milvus_client = MilvusClient(uri=milvus_uri)

        # 初始化嵌入模型
        self.embedding_model = embedding_model or QwenEmbedding()

        # 初始化文档处理器
        self.document_processor = DocumentProcessor(
            embedding_model=self.embedding_model,
        )

        # 初始化向量库
        self._initialize_collection()

        logger.info("知识库入库服务初始化完成")
        logger.info(f"Milvus URI: {milvus_uri}")
        logger.info(f"集合名称: {collection_name}")
        logger.info(f"批处理大小: {batch_size}")

    def _initialize_collection(self):
        """
        初始化向量库集合
        """
        try:
            if not self.milvus_client.has_collection(self.collection_name):
                logger.info(f"创建新的向量库集合: {self.collection_name}")
                VectorSchema.create_collection(
                    self.milvus_client,
                    self.collection_name,
                    drop_existing=False,
                )
            else:
                logger.info(f"向量库集合已存在: {self.collection_name}")

        except Exception as e:
            logger.error(f"向量库初始化失败: {e}")
            raise

    def _generate_document_id(self, chunk: Dict[str, Any]) -> str:
        """
        生成文档唯一ID

        Args:
            chunk: 文档块

        Returns:
            文档ID
        """
        # 使用内容的哈希值作为ID
        content = chunk["content"]
        metadata = chunk.get("metadata", {})

        # 组合关键信息
        key_fields = [
            metadata.get("source_file", ""),
            metadata.get("chapter_title", ""),
            metadata.get("chunk_id", ""),
            content[:100],  # 内容的前100个字符
        ]

        key_string = "|".join(key_fields)
        hash_value = hashlib.md5(key_string.encode()).hexdigest()

        return f"doc_{hash_value}"

    def _batch_generate_embeddings(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[List[float]]:
        """
        批量生成嵌入向量

        Args:
            chunks: 文档块列表

        Returns:
            嵌入向量列表
        """
        try:
            # 提取文本内容
            texts = [chunk["content"] for chunk in chunks]

            # 批量生成嵌入
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress=True,
            )

            return embeddings

        except Exception as e:
            logger.error(f"嵌入生成失败: {e}")
            raise

    def _batch_insert_documents(
        self,
        documents: List[Dict[str, Any]],
    ) -> bool:
        """
        批量插入文档到向量库

        Args:
            documents: 文档列表

        Returns:
            是否插入成功
        """
        try:
            # 批量插入
            self.milvus_client.insert(
                collection_name=self.collection_name,
                data=documents,
            )

            logger.info(f"成功插入 {len(documents)} 个文档到向量库")
            return True

        except Exception as e:
            logger.error(f"文档插入失败: {e}")
            return False

    def ingest_file(
        self,
        file_path: str,
        use_semantic_chunking: bool = True,
        skip_existing: bool = True,
    ) -> Dict[str, Any]:
        """
        处理单个文件并入库

        Args:
            file_path: 文件路径
            use_semantic_chunking: 是否使用语义分块
            skip_existing: 是否跳过已存在的文档

        Returns:
            处理结果统计
        """
        start_time = time.time()

        logger.info(f"开始处理文件: {file_path}")

        try:
            # 步骤1: 文档处理
            chunks = self.document_processor.process_markdown_file(
                file_path,
                use_semantic_chunking=use_semantic_chunking,
            )

            if not chunks:
                logger.warning(f"文件处理结果为空: {file_path}")
                return {
                    "success": False,
                    "file_path": file_path,
                    "error": "处理结果为空",
                }

            logger.info(f"文档分块完成: {len(chunks)} 块")

            # 步骤2: 批量生成嵌入向量
            logger.info("开始生成嵌入向量...")
            embeddings = self._batch_generate_embeddings(chunks)
            logger.info(f"嵌入向量生成完成: {len(embeddings)} 个")

            # 步骤3: 构建向量文档
            logger.info("构建向量文档...")
            documents = []
            skipped_count = 0

            for chunk, embedding in zip(chunks, embeddings):
                # 生成文档ID
                doc_id = self._generate_document_id(chunk)

                # 检查是否已存在
                if skip_existing:
                    existing_docs = self.milvus_client.get(
                        collection_name=self.collection_name,
                        ids=[doc_id],
                    )
                    if existing_docs:
                        skipped_count += 1
                        continue

                # 构建文档
                document = VectorDocumentBuilder.build_document(
                    chunk,
                    embedding,
                    doc_id,
                )
                documents.append(document)

            if skipped_count > 0:
                logger.info(f"跳过已存在的文档: {skipped_count} 个")

            if not documents:
                logger.warning(f"没有新文档需要插入: {file_path}")
                return {
                    "success": True,
                    "file_path": file_path,
                    "total_chunks": len(chunks),
                    "new_documents": 0,
                    "skipped_documents": skipped_count,
                    "processing_time": time.time() - start_time,
                }

            # 步骤4: 批量插入向量库
            logger.info(f"插入 {len(documents)} 个文档到向量库...")
            success = self._batch_insert_documents(documents)

            processing_time = time.time() - start_time

            if success:
                logger.success(f"✅ 文件处理完成: {file_path}")
                logger.info(f"  总块数: {len(chunks)}")
                logger.info(f"  新文档: {len(documents)}")
                logger.info(f"  跳过: {skipped_count}")
                logger.info(f"  耗时: {processing_time:.2f}秒")

                return {
                    "success": True,
                    "file_path": file_path,
                    "total_chunks": len(chunks),
                    "new_documents": len(documents),
                    "skipped_documents": skipped_count,
                    "processing_time": processing_time,
                }
            else:
                return {
                    "success": False,
                    "file_path": file_path,
                    "error": "向量库插入失败",
                }

        except Exception as e:
            logger.error(f"文件处理失败: {file_path} - {e}")
            return {
                "success": False,
                "file_path": file_path,
                "error": str(e),
            }

    def ingest_directory(
        self,
        directory_path: str,
        file_pattern: str = "*.md",
        use_semantic_chunking: bool = True,
        skip_existing: bool = True,
    ) -> Dict[str, Any]:
        """
        批量处理目录下的文件

        Args:
            directory_path: 目录路径
            file_pattern: 文件匹配模式
            use_semantic_chunking: 是否使用语义分块
            skip_existing: 是否跳过已存在的文档

        Returns:
            批量处理结果统计
        """
        start_time = time.time()

        logger.info(f"开始批量处理目录: {directory_path}")

        # 查找匹配的文件
        directory = Path(directory_path)
        files = list(directory.glob(file_pattern))

        if not files:
            logger.warning(f"没有找到匹配的文件: {directory_path}/{file_pattern}")
            return {
                "success": False,
                "directory_path": directory_path,
                "error": "没有找到匹配的文件",
            }

        logger.info(f"找到 {len(files)} 个文件")

        # 处理每个文件
        results = []
        total_new_docs = 0
        total_skipped = 0
        failed_files = []

        for i, file_path in enumerate(files, 1):
            logger.info(f"\n处理文件 {i}/{len(files)}: {file_path.name}")

            result = self.ingest_file(
                str(file_path),
                use_semantic_chunking=use_semantic_chunking,
                skip_existing=skip_existing,
            )

            results.append(result)

            if result["success"]:
                total_new_docs += result.get("new_documents", 0)
                total_skipped += result.get("skipped_documents", 0)
            else:
                failed_files.append(file_path.name)

        processing_time = time.time() - start_time

        # 生成统计报告
        logger.info("\n" + "=" * 60)
        logger.info("批量处理完成!")
        logger.info(f"总文件数: {len(files)}")
        logger.info(f"成功: {len(files) - len(failed_files)}")
        logger.info(f"失败: {len(failed_files)}")
        logger.info(f"新文档: {total_new_docs}")
        logger.info(f"跳过文档: {total_skipped}")
        logger.info(f"总耗时: {processing_time:.2f}秒")
        logger.info("=" * 60)

        return {
            "success": len(failed_files) == 0,
            "directory_path": directory_path,
            "total_files": len(files),
            "successful_files": len(files) - len(failed_files),
            "failed_files": failed_files,
            "total_new_documents": total_new_docs,
            "total_skipped_documents": total_skipped,
            "processing_time": processing_time,
            "details": results,
        }

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取向量库统计信息

        Returns:
            统计信息
        """
        try:
            # 获取集合信息
            info = self.milvus_client.describe_collection(self.collection_name)

            # 统计文档数量
            num_entities = info.get("num_entities", 0)

            return {
                "collection_name": self.collection_name,
                "total_documents": num_entities,
                "embedding_dimension": VectorSchema.EMBEDDING_DIMENSION,
                "collection_info": info,
            }

        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}


# 使用示例
def example_usage():
    """
    知识库入库服务使用示例
    """
    print("=" * 60)
    print("知识库入库服务示例")
    print("=" * 60)

    # 初始化入库服务
    print("\n1. 初始化入库服务...")
    ingestion_service = KnowledgeIngestionService(
        milvus_uri="http://localhost:19530",
        collection_name="fitness_knowledge",
    )

    # 处理单个文件
    print("\n2. 处理单个文件...")
    file_path = "/home/wenkai/project/MuslceKing/docs/knowledge_base/book_data_extract/MinerU_力量训练原理与实践__20251201151605.md"

    result = ingestion_service.ingest_file(
        file_path=file_path,
        use_semantic_chunking=True,
        skip_existing=True,
    )

    print(f"\n处理结果:")
    print(f"  成功: {result['success']}")
    print(f"  新文档: {result.get('new_documents', 0)}")
    print(f"  耗时: {result.get('processing_time', 0):.2f}秒")

    # 获取统计信息
    print("\n3. 获取向量库统计信息...")
    stats = ingestion_service.get_collection_stats()
    print(f"总文档数: {stats.get('total_documents', 0)}")
    print(f"嵌入维度: {stats.get('embedding_dimension', 0)}")

    # 批量处理示例
    print("\n4. 批量处理目录...")
    directory_path = "/home/wenkai/project/MuslceKing/docs/knowledge_base/book_data_extract"

    # 取消注释以运行批量处理
    # batch_result = ingestion_service.ingest_directory(
    #     directory_path=directory_path,
    #     file_pattern="*.md",
    # )
    # print(f"\n批量处理结果:")
    # print(f"  总文件数: {batch_result['total_files']}")
    # print(f"  新文档: {batch_result['total_new_documents']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    example_usage()