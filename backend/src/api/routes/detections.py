"""Detection-related API endpoints.

These endpoints provide access to detection records stored in the database.
All logic is delegated to DetectionService.

NOTE: bbox data is NOT exposed in API responses because:
- Bounding boxes are already drawn on stored annotated images
- Real-time video streaming (MJPEG) has boxes drawn on frames
- This keeps API responses lightweight
- Raw bbox data remains in database for future analysis if needed
"""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from src.api.dependencies import get_database, get_optional_image_storage
from src.services.detection_service import DetectionService
from src.storage.interfaces import Database, ImageStorage

router = APIRouter()


# ============================================================
# Response Models (DTO - Data Transfer Objects)
# ============================================================

class DetectionResponse(BaseModel):
    """Single detection in response.
    
    NOTE: bbox is intentionally excluded - boxes are drawn on images.
    """
    id: str
    image_path: str
    object_type: str
    confidence: float
    created_at: datetime
    session_id: Optional[str] = None


class DetectionListResponse(BaseModel):
    """List of detections response."""
    total: int
    offset: int
    limit: int
    items: List[DetectionResponse]


class DetectionStatsResponse(BaseModel):
    """Detection statistics response."""
    total_detections: int
    by_type: dict
    time_range: dict


# ============================================================
# Dependency: Detection Service
# ============================================================

def get_detection_service(
    database: Database = Depends(get_database),
    image_storage: Optional[ImageStorage] = Depends(get_optional_image_storage),
) -> DetectionService:
    """Get detection service instance."""
    return DetectionService(database=database, image_storage=image_storage)


# ============================================================
# Endpoints
# ============================================================

@router.get("/detections", response_model=DetectionListResponse)
async def list_detections(
    start_time: Optional[datetime] = Query(None, description="Filter by start time"),
    end_time: Optional[datetime] = Query(None, description="Filter by end time"),
    object_type: Optional[str] = Query(None, description="Filter by object type"),
    session_id: Optional[str] = Query(None, description="Filter by session"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum items to return"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
    service: DetectionService = Depends(get_detection_service),
):
    """List detection records with optional filters.
    
    Returns paginated list of detections with filtering options.
    """
    result = await run_in_threadpool(
        service.list_detections,
        start_time,
        end_time,
        object_type,
        session_id,
        limit,
        offset,
    )
    
    # Convert DTOs to response models (bbox excluded - drawn on images)
    items = [
        DetectionResponse(
            id=d.id,
            image_path=d.image_path,
            object_type=d.object_type,
            confidence=d.confidence,
            created_at=d.created_at,
            session_id=d.session_id,
        )
        for d in result.items
    ]
    
    return DetectionListResponse(
        total=result.total,
        offset=result.offset,
        limit=result.limit,
        items=items,
    )


@router.get("/detections/stats", response_model=DetectionStatsResponse)
async def get_detection_stats(
    start_time: Optional[datetime] = Query(None, description="Filter by start time"),
    end_time: Optional[datetime] = Query(None, description="Filter by end time"),
    service: DetectionService = Depends(get_detection_service),
):
    """Get detection statistics for a time range.
    
    Returns counts of detections grouped by object type.
    """
    stats = await run_in_threadpool(service.get_stats, start_time, end_time)
    
    return DetectionStatsResponse(
        total_detections=stats.total_detections,
        by_type=stats.by_type,
        time_range={
            "start": start_time.isoformat() if start_time else None,
            "end": end_time.isoformat() if end_time else None,
        },
    )


@router.get("/detections/{detection_id}", response_model=DetectionResponse)
async def get_detection(
    detection_id: str,
    service: DetectionService = Depends(get_detection_service),
):
    """Get a specific detection by ID.
    
    Returns detection details (bbox excluded - drawn on images).
    """
    detection = await run_in_threadpool(service.get_detection, detection_id)
    
    if detection is None:
        raise HTTPException(
            status_code=404,
            detail=f"Detection not found: {detection_id}"
        )
    
    return DetectionResponse(
        id=detection.id,
        image_path=detection.image_path,
        object_type=detection.object_type,
        confidence=detection.confidence,
        created_at=detection.created_at,
        session_id=detection.session_id,
    )
