"""
聊天服务模块
"""
from typing import Optional, Dict, Any
import uuid
from sqlalchemy.orm import Session
from muscleking.app.agents.models.model_chat import SessionInfo
from muscleking.app.agents.lg_builder import graph
from loguru import logger
from fastapi import Depends
from muscleking.app.persistence.crud.base import chat_session


async def get_or_create_session(db: Session, session_id: Optional[str], user_id: str) -> str:
    """获取或创建聊天会话"""
    if session_id:
        # 查询数据库是否存在该会话
        session = chat_session.get(db, session_id=session_id)
        if session:
            return session_id
        else:
            # 如果会话不存在，创建新会话
            new_session_id = str(uuid.uuid4())
            chat_session.create(db, obj_in={
                "id": new_session_id,
                "user_id": user_id,
                "title": "新会话"
            })
            return new_session_id
    else:
        # 创建新会话
        new_session_id = str(uuid.uuid4())
        chat_session.create(db, obj_in={
            "id": new_session_id,
            "user_id": user_id,
            "title": "新会话"
        })
        return new_session_id

async def save_message(db: Session, session_id: str, message: str, is_user: bool,
                       route: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
    """保存聊天消息到数据库并返回message_id"""
    message_id = str(uuid.uuid4())
    
    # 获取当前会话的消息数量作为order_index
    from muscleking.app.persistence.db.models.chat_message import ChatMessage
    max_order = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).count()
    
    # 创建消息对象并保存到数据库
    chat_message = ChatMessage(
        id=message_id,
        session_id=session_id,
        content=message,
        is_user=is_user,
        route=route,
        message_metadata=metadata,
        order_index=max_order + 1
    )
    db.add(chat_message)
    db.commit()
    db.refresh(chat_message)
    
    return message_id

async def process_agent_query(message: str, session_id: str,
                            image_path: Optional[str] = None,
                            file_path: Optional[str] = None,) -> Dict[str, Any]:
    """进行智能体查询并返回结果"""
    config = {
        "configurable": {
            "thread_id": session_id,
            "image_path": image_path,
            "file_path": file_path,
        }
    }
    input_state = {
        "messages": [{"type":"human", "content": message}]
    }
    try:
        # 调用智能体查询得到智能体回复
        result = await graph.ainvoke(input_state, config)

        # 提取回复信息
        response_text = result["messages"][-1].content if result.get("messages") else ""

        # 提取路由信息
        router_info = result.get("router", {})
        route = router_info.get("type")
        route_logic = router_info.get("logic")

        # 提取相关源文档信息
        sources_raw = result.get("sources", [])
        sources = []
        if sources_raw:
            # 如果是字符串列表，转换为字典列表
            if isinstance(sources_raw[0], str):
                for src in sources_raw:
                    sources.append({"document_id": src, "source": src})
            else:
                sources = sources_raw

        return {
            "message": response_text,
            "route": route,
            "route_logic": route_logic,
            "sources": sources,
            "metadata": {
                "session_id": session_id,
                "agent_state": result
            }
        }
    except Exception as e:
        logger.error(f"Agent查询处理失败: {e}", exc_info=True)
        return {
            "message": "抱歉，处理您的请求时出现错误。",
            "route": "error",
            "route_logic": f"Error: {str(e)}",
            "sources": [],
            "metadata":  {"error": str(e)}
        }
