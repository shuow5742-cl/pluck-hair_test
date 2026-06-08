"""SQLite implementation for database operations.

Lightweight database backend for development and testing.
Uses the same SQLAlchemy models as PostgreSQL.
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from sqlalchemy import create_engine, func, text
from sqlalchemy.orm import Session, sessionmaker

from .interfaces import Database, DetectionRecord, SessionRecord
from .models import (
    Base,
    DetectionModel,
    SessionModel,
    SessionStatusEnum,
)

logger = logging.getLogger(__name__)


class SQLiteDatabase(Database):
    """SQLite implementation for database operations.
    
    Provides the same interface as PostgresDatabase but uses
    SQLite for storage. Ideal for development and testing.
    
    Example:
        >>> db = SQLiteDatabase("./data/pluck.db")
        >>> record = DetectionRecord(...)
        >>> db.save_detection(record)
        
        # In-memory database for testing
        >>> db = SQLiteDatabase(":memory:")
    """

    def __init__(
        self,
        db_path: str = "./data/pluck.db",
        echo: bool = False,
    ):
        """Initialize SQLite database connection.
        
        Args:
            db_path: Path to SQLite database file.
                Use ":memory:" for in-memory database.
            echo: Echo SQL statements (for debugging).
        """
        # Create parent directory if needed
        if db_path != ":memory:":
            db_dir = Path(db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            # Use forward slashes for cross-platform compatibility
            db_path_normalized = Path(db_path).as_posix()
        else:
            db_path_normalized = db_path
        
        # SQLite connection string
        connection_string = f"sqlite:///{db_path_normalized}"
        
        self.engine = create_engine(
            connection_string,
            echo=echo,
            connect_args={"check_same_thread": False},  # Allow multi-threaded access
        )
        
        # Create tables if not exist
        Base.metadata.create_all(self.engine)
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
        )
        
        logger.info(f"SQLite database initialized: {db_path}")

    def _get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    # Detection operations

    def save_detection(self, record: DetectionRecord) -> str:
        """Save a single detection record."""
        session = self._get_session()
        try:
            detection_id = record.id or str(uuid.uuid4())
            
            db_record = DetectionModel(
                id=detection_id,
                image_path=record.image_path,
                bbox_x1=record.bbox_x1,
                bbox_y1=record.bbox_y1,
                bbox_x2=record.bbox_x2,
                bbox_y2=record.bbox_y2,
                object_type=record.object_type,
                confidence=record.confidence,
                created_at=record.created_at or datetime.now(),
                session_id=record.session_id,
            )
            
            session.add(db_record)
            session.commit()
            
            logger.debug(f"Saved detection: {detection_id}")
            return detection_id
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save detection: {e}")
            raise
        finally:
            session.close()

    def save_detections_batch(self, records: List[DetectionRecord]) -> List[str]:
        """Save multiple detection records in a batch."""
        if not records:
            return []
        
        session = self._get_session()
        try:
            ids = []
            db_records = []
            
            for record in records:
                detection_id = record.id or str(uuid.uuid4())
                ids.append(detection_id)
                
                db_record = DetectionModel(
                    id=detection_id,
                    image_path=record.image_path,
                    bbox_x1=record.bbox_x1,
                    bbox_y1=record.bbox_y1,
                    bbox_x2=record.bbox_x2,
                    bbox_y2=record.bbox_y2,
                    object_type=record.object_type,
                    confidence=record.confidence,
                    created_at=record.created_at or datetime.now(),
                    session_id=record.session_id,
                )
                db_records.append(db_record)

            session.add_all(db_records)
            session.commit()

            return ids
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save detections batch: {e}")
            raise
        finally:
            session.close()

    def get_detection(self, detection_id: str) -> Optional[DetectionRecord]:
        """Get detection by ID."""
        session = self._get_session()
        try:
            result = session.query(DetectionModel).filter(
                DetectionModel.id == detection_id
            ).first()
            
            if result is None:
                return None
            
            return self._to_detection_record(result)
            
        finally:
            session.close()

    def query_detections(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        object_type: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[DetectionRecord]:
        """Query detections with filters."""
        session = self._get_session()
        try:
            query = session.query(DetectionModel)
            
            # Apply filters
            if start_time is not None:
                query = query.filter(DetectionModel.created_at >= start_time)
            if end_time is not None:
                query = query.filter(DetectionModel.created_at <= end_time)
            if object_type is not None:
                query = query.filter(DetectionModel.object_type == object_type)
            if session_id is not None:
                query = query.filter(DetectionModel.session_id == session_id)
            
            # Order and paginate
            results = query.order_by(
                DetectionModel.created_at.desc()
            ).offset(offset).limit(limit).all()
            
            return [self._to_detection_record(r) for r in results]
            
        finally:
            session.close()

    def count_detections(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        object_type: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """Count detections matching filters."""
        session = self._get_session()
        try:
            query = session.query(func.count(DetectionModel.id))
            
            if start_time is not None:
                query = query.filter(DetectionModel.created_at >= start_time)
            if end_time is not None:
                query = query.filter(DetectionModel.created_at <= end_time)
            if object_type is not None:
                query = query.filter(DetectionModel.object_type == object_type)
            if session_id is not None:
                query = query.filter(DetectionModel.session_id == session_id)
            
            return query.scalar() or 0

        finally:
            session.close()

    def delete_detection(self, detection_id: str) -> bool:
        """Delete a detection record."""
        session = self._get_session()
        try:
            result = session.query(DetectionModel).filter(
                DetectionModel.id == detection_id
            ).delete()
            session.commit()
            return bool(result)
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to delete detection: {e}")
            raise
        finally:
            session.close()

    def ping(self) -> None:
        """Health check probe."""
        session = self._get_session()
        try:
            session.execute(text("SELECT 1"))
        finally:
            session.close()

    # Session operations

    def create_session(self, session_record: SessionRecord) -> str:
        """Create a new session record."""
        session = self._get_session()
        try:
            session_id = session_record.id or str(uuid.uuid4())
            
            db_record = SessionModel(
                id=session_id,
                start_time=session_record.start_time or datetime.now(),
                end_time=session_record.end_time,
                total_frames=session_record.total_frames,
                total_detections=session_record.total_detections,
                status=SessionStatusEnum(session_record.status),
            )
            
            session.add(db_record)
            session.commit()
            
            logger.info(f"Created session: {session_id}")
            return session_id
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to create session: {e}")
            raise
        finally:
            session.close()

    def update_session(self, session_record: SessionRecord) -> bool:
        """Update an existing session record."""
        if session_record.id is None:
            raise ValueError("Session ID is required for update")
        
        session = self._get_session()
        try:
            db_record = session.query(SessionModel).filter(
                SessionModel.id == session_record.id
            ).first()
            
            if db_record is None:
                return False
            
            # Update fields
            if session_record.end_time is not None:
                db_record.end_time = session_record.end_time
            db_record.total_frames = session_record.total_frames
            db_record.total_detections = session_record.total_detections
            db_record.status = SessionStatusEnum(session_record.status)

            session.commit()

            return True
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update session: {e}")
            raise
        finally:
            session.close()

    def get_session(self, session_id: str) -> Optional[SessionRecord]:
        """Get session by ID."""
        session = self._get_session()
        try:
            result = session.query(SessionModel).filter(
                SessionModel.id == session_id
            ).first()
            
            if result is None:
                return None
            
            return self._to_session_record(result)
            
        finally:
            session.close()

    # Conversion helpers

    def _to_detection_record(self, model: DetectionModel) -> DetectionRecord:
        """Convert ORM model to DetectionRecord."""
        return DetectionRecord(
            id=model.id,
            image_path=model.image_path,
            bbox_x1=model.bbox_x1,
            bbox_y1=model.bbox_y1,
            bbox_x2=model.bbox_x2,
            bbox_y2=model.bbox_y2,
            object_type=model.object_type,
            confidence=model.confidence,
            created_at=model.created_at,
            session_id=model.session_id,
        )

    def _to_session_record(self, model: SessionModel) -> SessionRecord:
        """Convert ORM model to SessionRecord."""
        return SessionRecord(
            id=model.id,
            start_time=model.start_time,
            end_time=model.end_time,
            total_frames=model.total_frames,
            total_detections=model.total_detections,
            status=model.status.value,
        )


