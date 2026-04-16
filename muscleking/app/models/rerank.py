"""
Rerank 相关数据模型
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Any


class RerankDocument(BaseModel):
    """重排序后的单个文档"""

    content: str = Field(description="文档内容")
    rerank_score: float = Field(description="重排序相关性分数")
    source: Optional[str] = Field(default=None, description="来源标识")
    metadata: Optional[dict[str, Any]] = Field(default=None, description="额外元数据")


class RerankResult(BaseModel):
    """重排序结果"""

    query: str = Field(description="查询文本")
    documents: List[RerankDocument] = Field(description="重排序后的文档列表")
    total: int = Field(description="输入文档总数")
    returned: int = Field(description="返回文档数")
