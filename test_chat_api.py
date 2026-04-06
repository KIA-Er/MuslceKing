"""
简化的FastAPI聊天API测试应用
不依赖agent系统，专门测试数据库会话和消息功能
"""
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
import uuid
from datetime import datetime

from muscleking.app.persistence.core.database import get_db, init_db
from muscleking.app.persistence.db.models.chat_session import ChatSession
from muscleking.app.persistence.db.models.chat_message import ChatMessage
from muscleking.app.config.settings import settings

# 请求和响应模型
class ChatRequest(BaseModel):
    """聊天请求模型"""
    message: str = Field(..., description="聊天请求内容")
    session_id: Optional[str] = Field(None, description="会话ID")
    user_id: Optional[str] = Field("default_user", description="用户ID")

class ChatResponse(BaseModel):
    """聊天响应模型"""
    message: str = Field(..., description="聊天回复内容")
    session_id: Optional[str] = Field(None, description="会话ID")
    message_id: Optional[str] = Field(None, description="消息ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间戳")

class SessionInfo(BaseModel):
    """会话信息模型"""
    session_id: str = Field(..., description="会话ID")
    user_id: str = Field(..., description="用户ID")
    title: str = Field(..., description="会话标题")
    created_at: datetime = Field(..., description="创建时间")
    message_count: int = Field(..., description="消息数量")

# 创建FastAPI应用
app = FastAPI(
    title="MuscleKing Chat API Test",
    version="0.1.0",
    description="简化的聊天API测试接口",
    docs_url="/docs",
    redoc_url="/redoc",
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


from muscleking.app.persistence.crud.base import chat_session,chat_message


def get_or_create_session(db: Session, session_id: Optional[str], user_id: str) -> str:
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



def save_message(db: Session, session_id: str, message: str, is_user: bool) -> str:
    """保存聊天消息到数据库并返回message_id"""
    message_id = str(uuid.uuid4())
    
    # 获取当前会话的消息数量作为order_index
    max_order = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).count()
    
    # 创建消息对象并保存到数据库
    chat_message = ChatMessage(
        id=message_id,
        session_id=session_id,
        content=message,
        is_user=is_user,
        order_index=max_order + 1
    )
    db.add(chat_message)
    db.commit()
    db.refresh(chat_message)
    
    return message_id

# API路由
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    """聊天接口 - POST请求"""
    try:
        # 获取或创建会话
        session_id = get_or_create_session(db, request.session_id, request.user_id)

        # 存储用户消息
        user_message_id = save_message(db, session_id, request.message, is_user=True)

        # 生成简单的回复（不使用agent系统）
        response_message = f"收到您的消息: '{request.message}'。会话ID: {session_id}"
        
        # 存储系统回复消息
        assistant_message_id = save_message(db, session_id, response_message, is_user=False)
        
        return ChatResponse(
            message=response_message,
            session_id=session_id,
            message_id=assistant_message_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"聊天处理失败: {str(e)}")

@app.get("/chat/{session_id}", response_model=List[Dict[str, Any]])
async def get_chat_messages(session_id: str, db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    """获取会话消息 - GET请求"""
    try:
        # 检查会话是否存在
        session = chat_session.get(db, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")
        
        # 获取会话消息
        messages = db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).order_by(ChatMessage.order_index).all()
        
        result = []
        for msg in messages:
            result.append({
                "message_id": msg.id,
                "content": msg.content,
                "is_user": msg.is_user,
                "route": msg.route,
                "route_logic": msg.route_logic,
                "created_at": msg.created_at,
                "order_index": msg.order_index
            })
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取消息失败: {str(e)}")

@app.get("/sessions", response_model=List[SessionInfo])
async def get_sessions(db: Session = Depends(get_db)) -> List[SessionInfo]:
    """获取用户会话列表"""
    try:
        sessions = db.query(ChatSession).order_by(ChatSession.created_at.desc()).all()
        
        result = []
        for session in sessions:
            # 获取每个会话的消息数量
            message_count = db.query(ChatMessage).filter(
                ChatMessage.session_id == session.id
            ).count()
            
            result.append(SessionInfo(
                session_id=session.id,
                user_id=session.user_id,
                title=session.title,
                created_at=session.created_at,
                message_count=message_count
            ))
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取会话列表失败: {str(e)}")

@app.post("/sessions", response_model=SessionInfo)
async def create_session(user_id: str = "default_user", db: Session = Depends(get_db)) -> SessionInfo:
    """创建新会话"""
    try:
        session_id = get_or_create_session(db, None, user_id)
        
        # 获取创建的会话信息
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        
        return SessionInfo(
            session_id=session.id,
            user_id=session.user_id,
            title=session.title,
            created_at=session.created_at,
            message_count=0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建会话失败: {str(e)}")

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, db: Session = Depends(get_db)) -> Dict[str, str]:
    """删除会话"""
    try:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")
        
        # 删除会话（级联删除消息）
        db.delete(session)
        db.commit()
        
        return {"message": f"会话 {session_id} 已删除"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除会话失败: {str(e)}")

@app.get("/")
async def root() -> dict:
    """根路径"""
    return {
        "name": "MuscleKing Chat API Test",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "POST /chat": "发送聊天消息",
            "GET /chat/{session_id}": "获取会话消息",
            "GET /sessions": "获取会话列表",
            "POST /sessions": "创建新会话",
            "DELETE /sessions/{session_id}": "删除会话"
        }
    }

@app.get("/health")
async def health_check() -> dict:
    """健康检查"""
    return {"status": "healthy", "version": "0.1.0"}

@app.on_event("startup")
async def startup_event() -> None:
    """应用启动事件"""
    print(f"MuscleKing Chat API Test v0.1.0")
    print(f"Debug mode: {settings.DEBUG}")
    print(f"API docs available at http://{settings.HOST}:{settings.PORT}/docs")
    
    # 初始化数据库表
    init_db(create_all=True)
    print("数据库表初始化完成")

@app.on_event("shutdown")
async def shutdown_event() -> None:
    """应用关闭事件"""
    print("Shutting down MuscleKing Chat API Test")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "test_chat_api:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )