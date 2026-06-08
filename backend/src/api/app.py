"""FastAPI application for Pluck Backend.

This module provides the application factory with proper dependency injection.
All dependencies (config, database, storage) are injected at creation time.
"""

import asyncio
import logging
from contextlib import suppress
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import AppConfig
from src.storage.interfaces import Database, ImageStorage

from .dependencies import AppState
from .routes import control_panel
from .routes import detections, health, images
from .routes import events as events_ws
from .routes import jsonrpc
from .routes import video
from .websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)


def create_app(
    config: Optional[AppConfig] = None,
    database: Optional[Database] = None,
    image_storage: Optional[ImageStorage] = None,
    title: str = "Pluck Backend API",
    version: str = "0.1.0",
) -> FastAPI:
    """Create and configure FastAPI application with dependency injection.
    
    Args:
        config: Application configuration (injected).
        database: Database instance (injected).
        image_storage: Image storage instance (injected).
        title: API title.
        version: API version (overridden by config if provided).
        
    Returns:
        Configured FastAPI application with injected dependencies.
    """
    # Use version from config if available
    if config is not None:
        version = config.app.version
    
    app = FastAPI(
        title=title,
        description="REST API for Bird's Nest Inspection System",
        version=version,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )
    
    # Configure CORS
    cors_config = config.api.cors if config else None
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config.allow_origins if cors_config else ["*"],
        allow_credentials=cors_config.allow_credentials if cors_config else True,
        allow_methods=cors_config.allow_methods if cors_config else ["*"],
        allow_headers=cors_config.allow_headers if cors_config else ["*"],
    )
    
    # Inject dependencies via app.state
    app.state.app_state = AppState(
        config=config,
        database=database,
        image_storage=image_storage,
    )
    app.state.ws_manager = WebSocketManager()
    app.state.redis_consumer = None
    app.state.redis_task = None
    app.state.command_publisher = None
    
    # Include routers
    app.include_router(health.router, prefix="/api", tags=["Health"])
    app.include_router(detections.router, prefix="/api", tags=["Detections"])
    app.include_router(images.router, prefix="/api", tags=["Images"])
    app.include_router(video.router, prefix="/api", tags=["Video"])
    app.include_router(events_ws.router, prefix="/api", tags=["Events"])
    app.include_router(control_panel.router, prefix="/api", tags=["TestConsole"])
    app.include_router(jsonrpc.router, tags=["JSON-RPC"])

    # Start background Redis consumer for event streaming
    if config and getattr(config, "redis", None) and config.redis.enabled:
        from src.events.redis_streams import RedisStreamConsumer

        async def start_consumer():
            consumer = RedisStreamConsumer(
                url=config.redis.url,
                stream=config.redis.stream,
                group=config.redis.consumer_group,
                consumer_name=config.redis.consumer_name,
                block_ms=config.redis.block_ms,
                count=config.redis.read_count,
            )
            try:
                await consumer.ensure_group()
                app.state.redis_consumer = consumer
                app.state.redis_task = asyncio.create_task(
                    consumer.consume(app.state.ws_manager.broadcast_json)
                )
                logger.info(
                    "Redis consumer started for stream '%s' as %s",
                    config.redis.stream,
                    config.redis.consumer_name,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to start Redis consumer: %s", exc)

        async def stop_consumer():
            consumer = getattr(app.state, "redis_consumer", None)
            task = getattr(app.state, "redis_task", None)

            if consumer:
                await consumer.stop()
            if task:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            if consumer:
                await consumer.close()

        app.add_event_handler("startup", start_consumer)
        app.add_event_handler("shutdown", stop_consumer)

        # Start RedisControlClient for comm signals (Redis mode only)
        if getattr(config, "comm", None) and config.comm.comm_signal.type == "redis":
            from src.comm.redis_control_client import RedisControlClient

            async def start_comm_client():
                try:
                    client = RedisControlClient(
                        url=config.redis.url,
                        command_key=config.comm.comm_signal.redis.command_key,
                        response_timeout=config.comm.comm_signal.redis.response_timeout,
                    )
                    await client.connect()
                    app.state.command_publisher = client
                    logger.info(
                        "Comm signal client started (redis key=%s)",
                        config.comm.comm_signal.redis.command_key,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to start comm signal client: %s", exc)

            async def stop_comm_client():
                client = getattr(app.state, "command_publisher", None)
                if client:
                    await client.close()

            app.add_event_handler("startup", start_comm_client)
            app.add_event_handler("shutdown", stop_comm_client)

    return app


# Default app instance (for uvicorn direct run without main.py)
# Note: This will have no dependencies injected - use main.py for production
app = create_app()
