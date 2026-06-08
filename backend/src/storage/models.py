"""SQLAlchemy ORM models for database storage."""

import enum
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class SessionStatusEnum(enum.Enum):
    """Session status enumeration."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SessionModel(Base):
    """Session database model.
    
    Represents a processing session (e.g., processing one tray).
    """
    __tablename__ = "sessions"

    id = Column(String(36), primary_key=True)
    start_time = Column(DateTime, default=datetime.now, nullable=False)
    end_time = Column(DateTime, nullable=True)
    total_frames = Column(Integer, default=0)
    total_detections = Column(Integer, default=0)
    status = Column(
        Enum(SessionStatusEnum),
        default=SessionStatusEnum.RUNNING,
        nullable=False
    )
    
    # Relationship to detections
    detections = relationship(
        "DetectionModel",
        back_populates="session",
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return (
            f"<Session(id={self.id}, status={self.status.value}, "
            f"detections={self.total_detections})>"
        )


class DetectionModel(Base):
    """Detection database model.
    
    Represents a single detection result with bounding box
    and classification information.
    """
    __tablename__ = "detections"

    id = Column(String(36), primary_key=True)
    image_path = Column(String(512), nullable=False, index=True)
    
    # Bounding box coordinates
    bbox_x1 = Column(Float, nullable=False)
    bbox_y1 = Column(Float, nullable=False)
    bbox_x2 = Column(Float, nullable=False)
    bbox_y2 = Column(Float, nullable=False)
    
    # Classification
    object_type = Column(String(64), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    
    # Timestamps
    created_at = Column(
        DateTime,
        default=datetime.now,
        nullable=False,
        index=True
    )
    
    # Foreign key to session
    session_id = Column(
        String(36),
        ForeignKey("sessions.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    
    # Relationship to session
    session = relationship("SessionModel", back_populates="detections")

    def __repr__(self):
        return (
            f"<Detection(id={self.id}, type={self.object_type}, "
            f"confidence={self.confidence:.2f})>"
        )

    @property
    def bbox_width(self) -> float:
        """Bounding box width."""
        return self.bbox_x2 - self.bbox_x1

    @property
    def bbox_height(self) -> float:
        """Bounding box height."""
        return self.bbox_y2 - self.bbox_y1

    @property
    def bbox_area(self) -> float:
        """Bounding box area."""
        return self.bbox_width * self.bbox_height


