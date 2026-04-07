"""
聊天信息数据模型定义
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ChatRequest(BaseModel):
    """聊天请求模型"""

    message: str = Field(..., description="聊天请求内容")
    session_id: Optional[str] = Field(None, description="会话ID")
    user_id: Optional[str] = Field("default_user", description="用户ID")
    stream: bool = Field(False, description="是否使用流失传输")
    image_path: Optional[str] = Field(None, description="图片路径")
    file_path: Optional[str] = Field(None, description="文件路径")


class ChatResponse(BaseModel):
    """聊天响应模型"""

    message: str = Field(..., description="聊天回复内容")
    session_id: Optional[str] = Field(None, description="会话ID")
    message_id: Optional[str] = Field(None, description="消息ID")
    route: Optional[str] = Field(None, description="路由信息")
    route_logic: Optional[str] = Field(None, description="路由逻辑说明")
    sources: Optional[List[Dict[str, Any]]] = Field(
        None, description="相关的源文档列表"
    )
    metadata: Optional[Dict[str, Any]] = Field(None, description="附加的元数据信息")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间戳")


class SessionInfo(BaseModel):
    """会话信息模型"""

    session_id: str = Field(..., description="会话ID")
    user_id: str = Field(..., description="用户ID")
    title: str = Field(..., description="会话标题")
    created_at: datetime = Field(..., description="创建时间")
    message_count: int = Field(..., description="消息数量")
