"""Async Redis client for comm-signal publishing (API side)."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, Optional

from redis import asyncio as aioredis

logger = logging.getLogger(__name__)


class RedisControlClient:
    """Send comm signals via Redis list and await response.

    Pure I/O: does not encode business semantics. The caller provides payload.
    """

    def __init__(
        self,
        url: str,
        command_key: str = "pluck:control",
        response_prefix: str = "pluck:response:",
        response_timeout: float = 5.0,
    ):
        self.command_key = command_key
        self.response_prefix = response_prefix
        self.response_timeout = response_timeout
        self._client: Optional[aioredis.Redis] = None
        self._url = url

    async def connect(self) -> None:
        if self._client is None:
            self._client = aioredis.Redis.from_url(self._url, decode_responses=True)

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    async def send(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send payload and wait for response (blocking with timeout)."""
        if self._client is None:
            await self.connect()

        request_id = payload.get("request_id") or str(uuid.uuid4())
        response_key = f"{self.response_prefix}{request_id}"
        command = {"request_id": request_id, **payload}

        try:
            await self._client.rpush(self.command_key, json.dumps(command))
            result = await self._client.blpop(response_key, timeout=self.response_timeout)
            if result is None:
                return {"type": "error", "message": "Comm signal timeout"}
            _, response_json = result
            return json.loads(response_json)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Comm signal send failed: %s", exc)
            return {"type": "error", "message": str(exc)}
        finally:
            try:
                if self._client:
                    await self._client.delete(response_key)
            except Exception:
                pass

__all__ = ["RedisControlClient"]
