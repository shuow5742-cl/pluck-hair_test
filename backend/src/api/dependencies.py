"""FastAPI dependency injection for API routes.

This module provides dependency injection using FastAPI's app.state pattern.
Dependencies are created once at startup and injected via create_app().

Usage:
    # In main.py
    config = AppConfig.from_yaml("config/settings.yaml")
    storage, database = create_storage(config)
    app = create_app(config, database=database, image_storage=storage)
    
    # In routes
    @router.get("/detections")
    async def list_detections(db: Database = Depends(get_database)):
        return db.query_detections()
"""

import logging
from typing import Optional

from fastapi import Depends, HTTPException, Request

from src.config import AppConfig
from src.storage.interfaces import Database, ImageStorage

logger = logging.getLogger(__name__)


# ============================================================
# State Container
# ============================================================

class AppState:
    """Application state container.
    
    Holds all injected dependencies. Attached to app.state during startup.
    This ensures a single source of truth for all components.
    """
    
    def __init__(
        self,
        config: Optional[AppConfig] = None,
        database: Optional[Database] = None,
        image_storage: Optional[ImageStorage] = None,
    ):
        self.config = config
        self.database = database
        self.image_storage = image_storage
    
    @property
    def is_configured(self) -> bool:
        """Check if essential dependencies are configured."""
        return (
            self.config is not None
            and self.database is not None
            and self.image_storage is not None
        )


# ============================================================
# Dependency Getters
# ============================================================

def get_app_state(request: Request) -> AppState:
    """Get application state from request.
    
    Args:
        request: FastAPI request object.
        
    Returns:
        AppState instance.
        
    Raises:
        HTTPException: If app state not initialized.
    """
    state: Optional[AppState] = getattr(request.app.state, "app_state", None)
    if state is None:
        raise HTTPException(
            status_code=503,
            detail="Application not properly initialized"
        )
    return state


def get_config(state: AppState = Depends(get_app_state)) -> AppConfig:
    """Get application configuration.
    
    Args:
        state: Application state (injected).
        
    Returns:
        AppConfig instance.
        
    Raises:
        HTTPException: If config not available.
    """
    if state.config is None:
        raise HTTPException(
            status_code=503,
            detail="Configuration not available"
        )
    return state.config


def get_database(state: AppState = Depends(get_app_state)) -> Database:
    """Get database instance.
    
    Args:
        state: Application state (injected).
        
    Returns:
        Database instance.
        
    Raises:
        HTTPException: If database not available.
    """
    if state.database is None:
        raise HTTPException(
            status_code=503,
            detail="Database not available"
        )
    return state.database


def get_image_storage(state: AppState = Depends(get_app_state)) -> ImageStorage:
    """Get image storage instance.
    
    Args:
        state: Application state (injected).
        
    Returns:
        ImageStorage instance.
        
    Raises:
        HTTPException: If storage not available.
    """
    if state.image_storage is None:
        raise HTTPException(
            status_code=503,
            detail="Image storage not available"
        )
    return state.image_storage


def get_optional_database(state: AppState = Depends(get_app_state)) -> Optional[Database]:
    """Get database instance (optional, no error if missing).
    
    Use this for health checks where missing dependencies should be reported,
    not raise exceptions.
    """
    return state.database


def get_optional_image_storage(state: AppState = Depends(get_app_state)) -> Optional[ImageStorage]:
    """Get image storage instance (optional, no error if missing).
    
    Use this for health checks where missing dependencies should be reported,
    not raise exceptions.
    """
    return state.image_storage


# ============================================================
# Health Check Utilities
# ============================================================

def check_database_health(database: Optional[Database]) -> tuple[bool, str]:
    """Check database connectivity with a lightweight probe.

    Args:
        database: Database instance or None.

    Returns:
        Tuple of (is_healthy, status_message)
    """
    if database is None:
        return False, "not_configured"

    try:
        database.ping()
        return True, "ok"
    except Exception as e:
        logger.warning(f"Database health check failed: {e}")
        return False, f"error: {str(e)[:50]}"


def check_storage_health(storage: Optional[ImageStorage]) -> tuple[bool, str]:
    """Check image storage connectivity using an existence probe.

    Args:
        storage: ImageStorage instance or None.

    Returns:
        Tuple of (is_healthy, status_message)
    """
    if storage is None:
        return False, "not_configured"

    try:
        # Use a sentinel exists call instead of listing to avoid heavy scans
        storage.exists("__health_check__")
        return True, "ok"
    except Exception as e:
        logger.warning(f"Storage health check failed: {e}")
        return False, f"error: {str(e)[:50]}"
