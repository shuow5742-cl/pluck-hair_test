"""Simple WebSocket connection manager."""

import asyncio
import logging
from typing import Set

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Tracks connected WebSocket clients and supports broadcast."""

    def __init__(self):
        self._connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)

    async def broadcast_json(self, message: dict) -> None:
        """Broadcast JSON message to all clients."""
        async with self._lock:
            targets = list(self._connections)

        for ws in targets:
            try:
                await ws.send_json(message)
            except Exception as exc:  # noqa: BLE001
                logger.warning("WebSocket send failed, dropping client: %s", exc)
                await self.disconnect(ws)
