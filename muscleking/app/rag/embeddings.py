"""vLLM Embedding Client"""

from openai import OpenAI
from typing import List
import logging

logger = logging.getLogger(__name__)


class VLLMEmbeddingClient:
    """
    vLLM Embedding Client

    用于调用 vLLM 部署的 BGE-M3 Embedding 模型
    """

    def __init__(self, base_url: str = "http://localhost:50001/v1"):
        """
        初始化 vLLM Embedding Client

        Args:
            base_url: vLLM Embedding 服务地址
        """
        self.client = OpenAI(
            base_url=base_url,
            api_key="dummy"  # vLLM 不验证 api_key
        )
        self.base_url = base_url
        logger.info(f"VLLMEmbeddingClient initialized with {base_url}")

    def embed_query(self, text: str) -> List[float]:
        """
        对单个查询进行 embedding

        Args:
            text: 输入文本

        Returns:
            1024 维的 embedding 向量
        """
        try:
            response = self.client.embeddings.create(
                model="BAAI/bge-m3",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量 embedding 文档

        Args:
            texts: 输入文本列表

        Returns:
            每个文本的 embedding 向量列表
        """
        try:
            response = self.client.embeddings.create(
                model="BAAI/bge-m3",
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            raise

    @property
    def dimension(self) -> int:
        """
        返回 embedding 维度

        Returns:
            BGE-M3 的向量维度 (1024)
        """
        return 1024

    def health_check(self) -> bool:
        """
        检查 vLLM 服务是否健康

        Returns:
            服务是否可用
        """
        import requests
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
