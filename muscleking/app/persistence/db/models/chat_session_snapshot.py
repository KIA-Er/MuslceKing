"""
Chat session snapshot models for session state backup.
"""

from __future__ import annotations

from sqlalchemy import Column, DateTime, String, ForeignKey, JSON, func
from sqlalchemy.orm import relationship

from muscleking.app.persistence.core.database import Base


class ChatSessionSnapshot(Base):
    """
    Chat session snapshot model for session state backup.
    """

    __tablename__ = "chat_session_snapshots"

    # Primary key
    id = Column(String(255), primary_key=True, index=True, comment="Snapshot UUID")

    # Foreign key to ChatSession
    session_id = Column(
        String(255),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Session UUID this snapshot belongs to",
    )

    # Snapshot data
    state_data = Column(JSON, nullable=False, comment="Session state data as JSON")

    # Snapshot metadata
    title = Column(String(500), nullable=True, comment="Snapshot title or description")

    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Snapshot creation timestamp",
    )

    # Relationships
    session = relationship("ChatSession", back_populates="snapshots")

    def __repr__(self) -> str:
        return f"<ChatSessionSnapshot id={self.id} session_id={self.session_id} created_at={self.created_at}>"
