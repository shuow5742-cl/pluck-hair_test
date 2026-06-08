"""Image-related API endpoints.

NOTE: These endpoints are placeholders and will be implemented
when the API requirements are finalized with the frontend team.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

router = APIRouter()


# ============================================================
# Response Models
# ============================================================

class ImageInfoResponse(BaseModel):
    """Image information response."""
    path: str
    size_bytes: int
    width: int
    height: int
    content_type: str
    created_at: str


# ============================================================
# Endpoints (Placeholders)
# ============================================================

@router.get("/images/{path:path}")
async def get_image(
    path: str,
    thumbnail: bool = Query(False, description="Return thumbnail instead"),
    width: Optional[int] = Query(None, description="Resize width"),
    height: Optional[int] = Query(None, description="Resize height"),
):
    """Get an image from storage.
    
    Args:
        path: Image path in storage.
        thumbnail: If True, return a smaller thumbnail.
        width: Optional resize width.
        height: Optional resize height.
    
    TODO: Implement actual image retrieval from MinIO.
    """
    raise HTTPException(
        status_code=501,
        detail="Not implemented yet"
    )


@router.get("/images/{path:path}/info", response_model=ImageInfoResponse)
async def get_image_info(path: str):
    """Get image metadata without downloading the image.
    
    TODO: Implement actual metadata lookup.
    """
    raise HTTPException(
        status_code=501,
        detail="Not implemented yet"
    )


@router.get("/images/{path:path}/url")
async def get_image_url(
    path: str,
    expires_hours: int = Query(24, ge=1, le=168, description="URL expiration hours"),
):
    """Get a presigned URL for an image.
    
    Useful for direct browser access without going through the API.
    
    TODO: Implement actual presigned URL generation.
    """
    raise HTTPException(
        status_code=501,
        detail="Not implemented yet"
    )


