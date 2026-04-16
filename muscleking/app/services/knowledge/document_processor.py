"""
知识库文档处理服务
结合 LangChain Markdown 分词器和语义分块进行文本切分
"""

import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from loguru import logger

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.schema.document import Document as LangchainDocument

from muscleking.app.retrieval.embedding_model.qwen_embedding import QwenEmbedding


class DocumentProcessor:
    """
    文档处理器，专门处理OCR后的健身书籍数据

    功能：
    1. 基于 Markdown 结构的智能分块
    2. 语义感知的文本切分
    3. 元数据提取和增强
    """

    def __init__(
        self,
        embedding_model: Optional[QwenEmbedding] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        semantic_chunk_size: int = 3,
    ):
        """
        初始化文档处理器

        Args:
            embedding_model: 嵌入模型，用于语义分块
            chunk_size: 基础分块大小（字符数）
            chunk_overlap: 分块重叠大小
            semantic_chunk_size: 语义分块时考虑的段落数量
        """
        self.embedding_model = embedding_model or QwenEmbedding()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.semantic_chunk_size = semantic_chunk_size

        # 初始化 Markdown 分割器
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "book_title"),
                ("##", "chapter_title"),
                ("###", "section_title"),
                ("####", "subsection_title"),
            ]
        )

        # 初始化递归文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
        )

        logger.info("文档处理器初始化完成")
        logger.info(f"基础分块大小: {chunk_size}, 重叠: {chunk_overlap}")
        logger.info(f"语义分块段落数: {semantic_chunk_size}")

    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        从文件路径和内容中提取元数据

        Args:
            file_path: 文件路径

        Returns:
            元数据字典
        """
        path = Path(file_path)

        # 从文件名提取信息
        filename = path.stem
        metadata = {
            "source_file": str(path),
            "filename": filename,
            "file_type": path.suffix,
        }

        # 尝试解析文件名获取书籍信息
        # 例如: "MinerU_力量训练原理与实践__20251201151605.md"
        if "MinerU_" in filename:
            parts = filename.replace("MinerU_", "").split("__")
            if len(parts) >= 1:
                metadata["book_title"] = parts[0]
            if len(parts) >= 2:
                metadata["extract_date"] = parts[1]

        return metadata

    def clean_content(self, content: str) -> str:
        """
        清理文档内容

        Args:
            content: 原始内容

        Returns:
            清理后的内容
        """
        # 移除图片链接
        content = re.sub(r'!\[.*?\]\(.*?\)', '', content)

        # 移除HTML表格（保留纯文本描述）
        content = re.sub(r'<table>.*?</table>', '[表格数据]', content, flags=re.DOTALL)

        # 清理多余的空行
        content = re.sub(r'\n{3,}', '\n\n', content)

        # 移除页码等干扰信息
        content = re.sub(r'\（\d+\）', '', content)

        return content.strip()

    def semantic_chunk_merge(
        self,
        chunks: List[Dict[str, Any]],
        max_size: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        基于语义相似度合并小块

        Args:
            chunks: 文档块列表
            max_size: 合并后的最大大小

        Returns:
            合并后的文档块列表
        """
        if len(chunks) <= 1:
            return chunks

        merged_chunks = []
        current_chunk = chunks[0].copy()
        current_text = current_chunk.get("content", "")

        for i in range(1, len(chunks)):
            next_chunk = chunks[i]
            next_text = next_chunk.get("content", "")

            # 检查合并后的大小
            potential_size = len(current_text) + len(next_text)

            if potential_size <= max_size:
                # 合并
                current_text += "\n\n" + next_text
                current_chunk["content"] = current_text

                # 更新元数据
                if "section_title" in next_chunk:
                    current_chunk["section_title"] = next_chunk["section_title"]
                current_chunk["chunk_count"] = current_chunk.get("chunk_count", 1) + 1
            else:
                # 完成当前块，开始新块
                merged_chunks.append(current_chunk)
                current_chunk = next_chunk.copy()
                current_text = current_chunk.get("content", "")

        # 添加最后一个块
        if current_chunk:
            merged_chunks.append(current_chunk)

        logger.info(f"语义合并: {len(chunks)} -> {len(merged_chunks)} 块")
        return merged_chunks

    def process_markdown_file(
        self,
        file_path: str,
        use_semantic_chunking: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        处理 Markdown 文件

        Args:
            file_path: 文件路径
            use_semantic_chunking: 是否使用语义分块

        Returns:
            处理后的文档块列表
        """
        try:
            # 读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            logger.info(f"处理文件: {file_path}")
            logger.info(f"原始大小: {len(content)} 字符")

            # 清理内容
            cleaned_content = self.clean_content(content)

            # 提取基础元数据
            base_metadata = self.extract_metadata(file_path)

            # 步骤1: 基于 Markdown 结构的分块
            logger.info("步骤1: Markdown 结构分块...")
            md_docs = self.markdown_splitter.split_text(cleaned_content)

            logger.info(f"Markdown 分块结果: {len(md_docs)} 块")

            # 步骤2: 进一步细粒度分块
            logger.info("步骤2: 细粒度文本分块...")
            all_chunks = []

            for i, md_doc in enumerate(md_docs):
                # 对每个 Markdown 块进行递归分割
                sub_chunks = self.text_splitter.split_documents([md_doc])

                for j, chunk in enumerate(sub_chunks):
                    chunk_data = {
                        "content": chunk.page_content,
                        "metadata": {
                            **base_metadata,
                            **chunk.metadata,
                            "chunk_id": f"{i}_{j}",
                            "original_chunk_index": i,
                        },
                    }
                    all_chunks.append(chunk_data)

            logger.info(f"文本分块结果: {len(all_chunks)} 块")

            # 步骤3: 语义合并（可选）
            if use_semantic_chunking:
                logger.info("步骤3: 语义合并...")
                all_chunks = self.semantic_chunk_merge(all_chunks)

            # 步骤4: 添加嵌入向量（可选，用于进一步分析）
            # 这里暂时不计算嵌入，等到入库时再计算

            logger.info(f"✅ 文档处理完成: {len(all_chunks)} 块")
            return all_chunks

        except Exception as e:
            logger.error(f"文档处理失败: {e}")
            raise

    def process_multiple_files(
        self,
        file_paths: List[str],
        use_semantic_chunking: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        批量处理多个文件

        Args:
            file_paths: 文件路径列表
            use_semantic_chunking: 是否使用语义分块

        Returns:
            所有文档块列表
        """
        all_chunks = []

        for file_path in file_paths:
            try:
                chunks = self.process_markdown_file(
                    file_path,
                    use_semantic_chunking=use_semantic_chunking,
                )
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"处理文件 {file_path} 失败: {e}")
                continue

        logger.info(f"批量处理完成: 总计 {len(all_chunks)} 块")
        return all_chunks

    def create_vector_documents(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        创建向量库文档格式

        Args:
            chunks: 文档块列表

        Returns:
            向量库文档格式
        """
        vector_docs = []

        for chunk in chunks:
            vector_doc = {
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                # 嵌入向量将在入库时计算
            }
            vector_docs.append(vector_doc)

        return vector_docs


# 使用示例
def example_usage():
    """
    文档处理器使用示例
    """
    import time

    print("=" * 60)
    print("文档处理器示例")
    print("=" * 60)

    # 初始化处理器
    print("\n1. 初始化文档处理器...")
    processor = DocumentProcessor(
        chunk_size=500,
        chunk_overlap=100,
    )

    # 处理单个文件
    print("\n2. 处理Markdown文件...")
    file_path = "/home/wenkai/project/MuslceKing/docs/knowledge_base/book_data_extract/MinerU_力量训练原理与实践__20251201151605.md"

    start_time = time.time()
    chunks = processor.process_markdown_file(file_path)
    process_time = time.time() - start_time

    print(f"\n处理完成!")
    print(f"文档块数: {len(chunks)}")
    print(f"处理耗时: {process_time:.2f}秒")

    # 显示前几个块的信息
    print(f"\n前3个文档块示例:")
    for i, chunk in enumerate(chunks[:3]):
        content = chunk["content"][:100]
        metadata = chunk["metadata"]
        print(f"\n块 {i+1}:")
        print(f"  内容: {content}...")
        print(f"  章节: {metadata.get('chapter_title', 'N/A')}")
        print(f"  长度: {len(chunk['content'])} 字符")

    # 创建向量库文档
    print(f"\n3. 创建向量库文档...")
    vector_docs = processor.create_vector_documents(chunks)
    print(f"向量库文档数: {len(vector_docs)}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    example_usage()