"""
Qwen3-Embedding-0.6B 模型初始化和使用
使用 sentence-transformers 库
"""

from typing import List, Optional
from sentence_transformers import SentenceTransformer
from loguru import logger
import torch


class QwenEmbedding:
    """Qwen3-Embedding-0.6B 嵌入模型"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: Optional[str] = None,
        cache_folder: Optional[str] = None,
        local_files_only: bool = True,  # 默认只使用本地文件
    ):
        """
        初始化 Qwen3-Embedding-0.6B 模型

        Args:
            model_name: 模型名称
            device: 运行设备 ("cuda", "cpu", None表示自动检测)
            cache_folder: 模型缓存目录，None表示使用默认缓存
            local_files_only: 是否只使用本地文件，避免网络访问
        """
        self.model_name = model_name
        self.device = device or self._get_device()
        self.cache_folder = cache_folder

        logger.info("正在初始化 Qwen3-Embedding-0.6B 模型...")
        logger.info(f"设备: {self.device}")

        try:
            self.model = SentenceTransformer(
                model_name_or_path=model_name,
                device=self.device,
                cache_folder=cache_folder,
                trust_remote_code=True,
                local_files_only=local_files_only,  # 避免访问 HuggingFace
            )

            self.embedding_dimension = self.model.get_sentence_embedding_dimension()

            logger.success("✅ Qwen3-Embedding-0.6B 模型加载成功!")
            logger.info(f"嵌入维度: {self.embedding_dimension}")
            logger.info(f"最大序列长度: {self.model.max_seq_length}")

        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            raise

    def _get_device(self) -> str:
        """自动检测最佳运行设备"""
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    async def embed_list(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize_embeddings: bool = True,
    ) -> List[List[float]]:
        """
        将文本编码为向量嵌入

        Args:
            texts: 待编码的文本列表
            batch_size: 批处理大小
            show_progress: 是否显示进度条
            normalize_embeddings: 是否归一化嵌入向量

        Returns:
            嵌入向量列表
        """
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=True,
            )

            return embeddings.tolist()

        except Exception as e:
            logger.error(f"编码失败: {e}")
            raise

    async def embed(
        self,
        text: str,
        normalize_embedding: bool = True,
    ) -> List[float]:
        """
        编码单个文本

        Args:
            text: 待编码的文本
            normalize_embedding: 是否归一化嵌入向量

        Returns:
            嵌入向量
        """
        try:
            embedding = self.model.encode(
                text,
                normalize_embeddings=normalize_embedding,
                convert_to_numpy=True,
            )
            return embedding.tolist()

        except Exception as e:
            logger.error(f"单文本编码失败: {e}")
            raise

    def get_dimension(self) -> Optional[int]:
        """获取嵌入向量维度"""
        return self.embedding_dimension

    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "device": self.device,
            "max_seq_length": self.model.max_seq_length,
            "cache_folder": self.cache_folder,
        }
