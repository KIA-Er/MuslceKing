"""
Qwen3-Reranker-0.6B 模型初始化和使用
使用 sentence-transformers 的 CrossEncoder 进行重排序
"""

from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder
from loguru import logger
import torch

from muscleking.app.models.rerank import RerankDocument, RerankResult


class QwenReranker:
    """Qwen3-Reranker-0.6B 重排序模型"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        device: Optional[str] = None,
        cache_folder: Optional[str] = None,
        top_k: Optional[int] = 5,
        threshold: float = 0.0,
        local_files_only: bool = True,
    ):
        """
        初始化 Qwen3-Reranker-0.6B 模型

        Args:
            model_name: 模型名称
            device: 运行设备 ("cuda", "cpu", None表示自动检测)
            cache_folder: 模型缓存目录
            top_k: 返回的top-k文档数
            threshold: 重排序分数阈值
            local_files_only: 是否只使用本地文件
        """
        self.model_name = model_name
        self.device = device or self._get_device()
        self.cache_folder = cache_folder
        self.top_k = top_k
        self.threshold = threshold

        logger.info("正在初始化 Qwen3-Reranker-0.6B 模型...")
        logger.info(f"设备: {self.device}")

        try:
            self.model = CrossEncoder(
                model_name_or_path=model_name,
                device=self.device,
                cache_folder=cache_folder,
                trust_remote_code=True,
                local_files_only=local_files_only,
            )

            # 设置 padding token
            try:
                tokenizer = self.model.tokenizer
                if tokenizer.pad_token is None:
                    if tokenizer.eos_token is not None:
                        tokenizer.pad_token = tokenizer.eos_token
                    else:
                        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                        self.model.model.resize_token_embeddings(len(tokenizer))
                    logger.info(f"已设置 padding token: {tokenizer.pad_token}")
            except Exception as pad_error:
                logger.warning(f"设置 padding token 失败: {pad_error}")

            logger.success("✅ Qwen3-Reranker-0.6B 模型加载成功!")

        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            raise

    def _get_device(self) -> str:
        """自动检测最佳运行设备"""
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _predict(self, pairs: list, batch_size: int = 32) -> list:
        """统一的预测逻辑，处理 padding token 兼容性问题"""
        try:
            scores = self.model.predict(
                pairs,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return scores.tolist()
        except ValueError as ve:
            if "padding token" not in str(ve):
                raise
            logger.warning("批量处理失败，切换到逐个处理模式")
            scores = []
            for pair in pairs:
                try:
                    score = self.model.predict(
                        [pair], batch_size=1, show_progress_bar=False, convert_to_numpy=True,
                    )
                    scores.append(float(score[0]))
                except Exception:
                    scores.append(0.0)
            return scores

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        batch_size: int = 32,
    ) -> RerankResult:
        """
        对文档进行重排序

        Args:
            query: 查询文本
            documents: 文档列表
            batch_size: 批处理大小

        Returns:
            RerankResult 重排序结果
        """
        total = len(documents)
        if not documents:
            return RerankResult(query=query, documents=[], total=0, returned=0)

        try:
            doc_texts = [
                doc.get("content", "") if isinstance(doc, dict) else str(doc)
                for doc in documents
            ]

            pairs = [[query, text] for text in doc_texts]
            scores = self._predict(pairs, batch_size)

            scored_docs: list[RerankDocument] = []
            for doc, score in zip(documents, scores):
                if score < self.threshold:
                    continue
                if isinstance(doc, dict):
                    scored_docs.append(RerankDocument(
                        content=doc.get("content", ""),
                        rerank_score=score,
                        source=doc.get("source"),
                        metadata={k: v for k, v in doc.items() if k not in ("content", "source")},
                    ))
                else:
                    scored_docs.append(RerankDocument(
                        content=str(doc),
                        rerank_score=score,
                    ))

            scored_docs.sort(key=lambda x: x.rerank_score, reverse=True)

            if self.top_k is not None:
                scored_docs = scored_docs[:self.top_k]

            return RerankResult(
                query=query,
                documents=scored_docs,
                total=total,
                returned=len(scored_docs),
            )

        except Exception as e:
            logger.error(f"重排序失败: {e}")
            return RerankResult(query=query, documents=[], total=total, returned=0)

    async def compute_scores(
        self,
        query: str,
        documents: List[str],
        batch_size: int = 32,
    ) -> List[float]:
        """计算查询与文档的相关性分数"""
        try:
            pairs = [[query, doc] for doc in documents]
            return self._predict(pairs, batch_size)
        except Exception as e:
            logger.error(f"分数计算失败: {e}")
            raise

    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "device": self.device,
        }