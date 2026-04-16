"""
Chat message models for persistent message storage.
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    String,
    Text,
    Integer,
    ForeignKey,
    JSON,
    func,
)
from sqlalchemy.orm import relationship

from muscleking.app.storage.core.database import Base


class ChatMessage(Base):
    """
    Chat message model for persistent message storage.
    """

    __tablename__ = "chat_messages"

    # Primary key - using UUID string for message ID
    id = Column(String(255), primary_key=True, index=True, comment="Message UUID")

    # Foreign key to ChatSession
    session_id = Column(
        String(255),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Session UUID this message belongs to",
    )

    # Message content
    content = Column(Text, nullable=False, comment="Message content")

    # Message type
    is_user = Column(
        Boolean,
        nullable=False,
        default=True,
        comment="True if user message, False if assistant message",
    )

    # Message routing information
    route = Column(
        String(100), nullable=True, comment="Route type for agent processing"
    )

    route_logic = Column(Text, nullable=True, comment="Route logic explanation")

    # Message metadata
    message_metadata = Column(
        "metadata", JSON, nullable=True, comment="Additional metadata as JSON"
    )

    # Message order in conversation
    order_index = Column(
        Integer, nullable=False, comment="Order of message in conversation"
    )

    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Message creation timestamp",
    )

    # Relationships
    session = relationship("ChatSession", back_populates="messages")

    def __repr__(self) -> str:
        content_preview = self.content[:30] if self.content else ""
        return f"<ChatMessage id={self.id} session_id={self.session_id} is_user={self.is_user} content={content_preview!r}>"
