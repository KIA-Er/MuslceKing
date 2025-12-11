"""
具有智能体聊天功能的API
"""
from fastapi import APIRouter, HTTPException
from muscleking.app.models.model_chat import ChatRequest, ChatResponse
import uuid
from typing import Optional
from sqlalchemy.orm import Session
from muscleking.app.services.service_chat import get_or_create_session, save_message, process_agent_query   

router = APIRouter()

@router.post("/",response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db : Session = None  # 假设有一个数据库会话依赖
) -> ChatResponse:
    """
    具有Agent智能体聊天功能的API端点
    """
    # 获取或创建会话
    session_id = get_or_create_session(db, request.session_id, request.user_id)

    # 存储用户消息
    await save_message(db, session_id, request.message, is_user=True)

    # 进行智能体查询
    result = await process_agent_query(
        request.message,
        session_id,
        request.image_path,
        request.file_path,
    )

    # 存储系统回复消息
    message_id = await save_message(
        db,
        session_id,
        result["message"],
        is_user=False,
        route=result["route"],
        metadata=result.get("metadata")
    )
    
    return ChatResponse(
        message=result["message"],
        session_id=session_id,
        message_id=message_id or str(uuid.uuid4()),
        route=result["route"],
        route_logic=result["route_logic"],
        sources=result.get("sources"),
        metadata=result.get("metadata")
    )