"""Detection service for business logic and data access.

This service encapsulates all detection-related business logic,
providing a clean interface for API routes.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from src.storage.interfaces import Database, DetectionRecord, ImageStorage

logger = logging.getLogger(__name__)


@dataclass
class DetectionDTO:
    """Detection data transfer object for API responses."""
    id: str
    image_path: str
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    object_type: str
    confidence: float
    created_at: datetime
    session_id: Optional[str] = None

    @classmethod
    def from_record(cls, record: DetectionRecord) -> "DetectionDTO":
        """Convert from database record."""
        return cls(
            id=record.id or "",
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


@dataclass
class DetectionListResult:
    """Result of list_detections operation."""
    items: List[DetectionDTO]
    total: int
    offset: int
    limit: int


@dataclass
class DetectionStats:
    """Detection statistics."""
    total_detections: int
    by_type: Dict[str, int]
    start_time: Optional[datetime]
    end_time: Optional[datetime]


class DetectionService:
    """Service for detection-related operations.
    
    Encapsulates database queries and business logic for detections.
    Routes should use this service instead of accessing the database directly.
    
    Example:
        service = DetectionService(database, image_storage)
        result = service.list_detections(limit=10)
        for detection in result.items:
            print(detection.object_type)
    """
    
    def __init__(
        self,
        database: Database,
        image_storage: Optional[ImageStorage] = None,
    ):
        """Initialize detection service.
        
        Args:
            database: Database instance for queries.
            image_storage: Optional image storage for URL generation.
        """
        self.database = database
        self.image_storage = image_storage
    
    def list_detections(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        object_type: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> DetectionListResult:
        """List detections with optional filters.
        
        Args:
            start_time: Filter by created_at >= start_time.
            end_time: Filter by created_at <= end_time.
            object_type: Filter by object type (hair, black_spot, etc.).
            session_id: Filter by session ID.
            limit: Maximum number of results.
            offset: Number of results to skip.
            
        Returns:
            DetectionListResult with items and pagination info.
        """
        # Get total count for pagination
        total = self.database.count_detections(
            start_time=start_time,
            end_time=end_time,
            object_type=object_type,
            session_id=session_id,
        )
        
        # Get filtered records
        records = self.database.query_detections(
            start_time=start_time,
            end_time=end_time,
            object_type=object_type,
            session_id=session_id,
            limit=limit,
            offset=offset,
        )
        
        # Convert to DTOs
        items = [DetectionDTO.from_record(r) for r in records]
        
        return DetectionListResult(
            items=items,
            total=total,
            offset=offset,
            limit=limit,
        )
    
    def get_detection(self, detection_id: str) -> Optional[DetectionDTO]:
        """Get a specific detection by ID.
        
        Args:
            detection_id: Detection ID.
            
        Returns:
            DetectionDTO or None if not found.
        """
        record = self.database.get_detection(detection_id)
        if record is None:
            return None
        return DetectionDTO.from_record(record)
    
    def get_stats(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> DetectionStats:
        """Get detection statistics for a time range.
        
        Args:
            start_time: Start of time range.
            end_time: End of time range.
            
        Returns:
            DetectionStats with counts by type.
        """
        # Get total
        total = self.database.count_detections(
            start_time=start_time,
            end_time=end_time,
        )
        
        # Get counts by type
        object_types = ["hair", "black_spot", "yellow_spot", "unknown"]
        by_type = {}
        
        for obj_type in object_types:
            count = self.database.count_detections(
                start_time=start_time,
                end_time=end_time,
                object_type=obj_type,
            )
            by_type[obj_type] = count
        
        return DetectionStats(
            total_detections=total,
            by_type=by_type,
            start_time=start_time,
            end_time=end_time,
        )
    
    def delete_detection(self, detection_id: str) -> bool:
        """Delete a detection by ID.
        
        Args:
            detection_id: Detection ID.
            
        Returns:
            True if deleted, False if not found.
        """
        return self.database.delete_detection(detection_id)