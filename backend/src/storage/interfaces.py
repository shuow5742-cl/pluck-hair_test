"""Abstract interfaces for storage modules."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np


@dataclass
class DetectionRecord:
    """Detection record for database storage.
    
    Represents a single detection stored in the database,
    with references to the source image.
    
    Attributes:
        id: Unique identifier (UUID).
        image_path: Path to image in object storage.
        bbox_x1, bbox_y1, bbox_x2, bbox_y2: Bounding box coordinates.
        object_type: Type of detected object.
        confidence: Detection confidence score.
        created_at: Timestamp when record was created.
        session_id: Optional session identifier.
    """
    id: Optional[str] = None
    image_path: str = ""
    bbox_x1: float = 0.0
    bbox_y1: float = 0.0
    bbox_x2: float = 0.0
    bbox_y2: float = 0.0
    object_type: str = "unknown"
    confidence: float = 0.0
    created_at: Optional[datetime] = None
    session_id: Optional[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class SessionRecord:
    """Session record for database storage.
    
    Represents a processing session (e.g., one tray being processed).
    
    Attributes:
        id: Unique identifier (UUID).
        start_time: Session start timestamp.
        end_time: Session end timestamp.
        total_frames: Number of frames processed.
        total_detections: Number of detections found.
        status: Session status (running, completed, failed).
    """
    id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_frames: int = 0
    total_detections: int = 0
    status: str = "running"

    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now()


class ImageStorage(ABC):
    """Abstract interface for image storage.
    
    Implementations should handle binary image data storage,
    supporting save, load, and delete operations.
    """

    @abstractmethod
    def save(self, image: np.ndarray, path: str) -> str:
        """Save image to storage.
        
        Args:
            image: Image as numpy array (BGR format).
            path: Storage path/key for the image.
            
        Returns:
            Full storage path (may include bucket/prefix).
        """
        pass

    @abstractmethod
    def save_bytes(self, data: bytes, path: str, content_type: str = "image/jpeg") -> str:
        """Save raw bytes to storage.
        
        Args:
            data: Raw binary data.
            path: Storage path/key.
            content_type: MIME type of the data.
            
        Returns:
            Full storage path.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> np.ndarray:
        """Load image from storage.
        
        Args:
            path: Storage path/key.
            
        Returns:
            Image as numpy array.
            
        Raises:
            FileNotFoundError: If image not found.
        """
        pass

    @abstractmethod
    def load_bytes(self, path: str) -> bytes:
        """Load raw bytes from storage.
        
        Args:
            path: Storage path/key.
            
        Returns:
            Raw binary data.
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> bool:
        """Delete image from storage.
        
        Args:
            path: Storage path/key.
            
        Returns:
            True if deleted, False if not found.
        """
        pass

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if image exists in storage.
        
        Args:
            path: Storage path/key.
            
        Returns:
            True if exists.
        """
        pass

    @abstractmethod
    def list_objects(self, prefix: str = "") -> List[str]:
        """List objects with given prefix.
        
        Args:
            prefix: Path prefix to filter.
            
        Returns:
            List of object paths.
        """
        pass


class Database(ABC):
    """Abstract interface for database operations.
    
    Implementations should handle structured data storage
    for detection records and sessions.
    """

    @abstractmethod
    def save_detection(self, record: DetectionRecord) -> str:
        """Save detection record.
        
        Args:
            record: Detection record to save.
            
        Returns:
            Generated or existing record ID.
        """
        pass

    @abstractmethod
    def save_detections_batch(self, records: List[DetectionRecord]) -> List[str]:
        """Save multiple detection records in a batch.
        
        Args:
            records: List of detection records.
            
        Returns:
            List of record IDs.
        """
        pass

    @abstractmethod
    def get_detection(self, detection_id: str) -> Optional[DetectionRecord]:
        """Get detection by ID.
        
        Args:
            detection_id: Record ID.
            
        Returns:
            Detection record or None if not found.
        """
        pass

    @abstractmethod
    def query_detections(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        object_type: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[DetectionRecord]:
        """Query detections with filters.
        
        Args:
            start_time: Filter by created_at >= start_time.
            end_time: Filter by created_at <= end_time.
            object_type: Filter by object type.
            session_id: Filter by session.
            limit: Maximum number of results.
            offset: Number of results to skip.
            
        Returns:
            List of matching detection records.
        """
        pass

    @abstractmethod
    def count_detections(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        object_type: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """Count detections matching filters.
        
        Args:
            Same as query_detections.
            
        Returns:
            Number of matching records.
        """
        pass

    @abstractmethod
    def create_session(self, session: SessionRecord) -> str:
        """Create a new session record.
        
        Args:
            session: Session record.
            
        Returns:
            Session ID.
        """
        pass

    @abstractmethod
    def update_session(self, session: SessionRecord) -> bool:
        """Update an existing session record.
        
        Args:
            session: Session record with updated fields.
            
        Returns:
            True if updated successfully.
        """
        pass

    @abstractmethod
    def get_session(self, session_id: str) -> Optional[SessionRecord]:
        """Get session by ID.

        Args:
            session_id: Session ID.

        Returns:
            Session record or None.
        """
        pass

    @abstractmethod
    def delete_detection(self, detection_id: str) -> bool:
        """Delete a detection record by ID.

        Args:
            detection_id: Record ID to remove.

        Returns:
            True if a record was deleted, False if not found.
        """
        pass

    @abstractmethod
    def ping(self) -> None:
        """Lightweight connectivity probe for health checks."""
        pass


