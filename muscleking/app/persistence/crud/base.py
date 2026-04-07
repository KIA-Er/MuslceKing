"""
对通用数据库的基础的增删改查
"""

from typing import Any, Generic, Optional, Type, TypeVar

from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from sqlalchemy.orm import Session

from muscleking.app.persistence.core.database import Base
from muscleking.app.persistence.db.models.chat_session import ChatSession
from muscleking.app.persistence.db.models.chat_message import ChatMessage


ModelType = TypeVar("ModelType", bound=Base)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """
    Generic CRUD operations for database models.

    Provides standard create, read, update, delete operations that can be
    inherited by specific model CRUD classes.
    """

    def __init__(self, model: Type[ModelType]):
        """初始化一个SQLAlchemy模型类，用于定义ORM模型"""
        self.model = model

    def get(
        self,
        db: Session,
        session_id: Any,
    ) -> Optional[ModelType]:
        """通过特定的ID获取一个聊天会话记录"""
        return db.query(self.model).filter(self.model.id == session_id).first()

    def create(self, db: Session, obj_in: CreateSchemaType) -> ModelType:
        """创建一个聊天会话记录到表中"""
        obj_in_data = jsonable_encoder(obj_in)
        db_obj = self.model(**obj_in_data)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj


# 创建模型实例
chat_session = CRUDBase(ChatSession)
chat_message = CRUDBase(ChatMessage)
