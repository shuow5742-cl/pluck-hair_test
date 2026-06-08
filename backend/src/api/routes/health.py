"""Health check endpoints.

These endpoints follow Kubernetes health check conventions:
- /health: Basic health (always returns healthy if service is up)
- /live: Liveness probe (service is running)
- /ready: Readiness probe (dependencies are available)
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel

from src.api.dependencies import (
    AppState,
    check_database_health,
    check_storage_health,
    get_app_state,
    get_optional_database,
    get_optional_image_storage,
)
from src.storage.interfaces import Database, ImageStorage

router = APIRouter()
logger = logging.getLogger(__name__)


# ============================================================
# Response Models
# ============================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    version: str


class ReadinessResponse(BaseModel):
    """Readiness check response."""
    ready: bool
    database: str
    storage: str
    details: Optional[dict] = None


class LivenessResponse(BaseModel):
    """Liveness check response."""
    alive: bool
    timestamp: datetime


# ============================================================
# Endpoints
# ============================================================

@router.get("/health", response_model=HealthResponse)
async def health_check(state: AppState = Depends(get_app_state)):
    """Basic health check endpoint.
    
    Returns the service status and current timestamp.
    Used by load balancers and monitoring systems.
    This is a lightweight check that doesn't verify dependencies.
    """
    version = state.config.app.version if state.config else "0.1.0"
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version=version,
    )


@router.get("/live", response_model=LivenessResponse)
async def liveness_check():
    """Liveness probe endpoint.
    
    Simple check that the service is running.
    Used by Kubernetes liveness probes.
    """
    return LivenessResponse(
        alive=True,
        timestamp=datetime.now(),
    )


@router.get("/ready", response_model=ReadinessResponse)
async def readiness_check(
    response: Response,
    database: Optional[Database] = Depends(get_optional_database),
    storage: Optional[ImageStorage] = Depends(get_optional_image_storage),
):
    """Readiness check endpoint.
    
    Verifies that all dependencies (database, storage) are available.
    Used by Kubernetes readiness probes and health monitoring.
    
    Returns HTTP 200 if ready, HTTP 503 if not ready.
    """
    # Check database
    db_ok, db_status = check_database_health(database)
    
    # Check storage
    storage_ok, storage_status = check_storage_health(storage)
    
    # Overall readiness
    is_ready = db_ok and storage_ok
    
    if not is_ready:
        response.status_code = 503  # Service Unavailable
        logger.warning(
            f"Readiness check failed: database={db_status}, storage={storage_status}"
        )
    
    return ReadinessResponse(
        ready=is_ready,
        database=db_status,
        storage=storage_status,
        details={
            "database_ok": db_ok,
            "storage_ok": storage_ok,
        }
    )
