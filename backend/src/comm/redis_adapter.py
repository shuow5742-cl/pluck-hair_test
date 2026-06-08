"""Redis-based comm signal adapter (development/debug)."""

from __future__ import annotations

import json
import logging
from typing import Optional, Dict, Any

import redis

from autoweaver.comm import CommSignalBase

logger = logging.getLogger(__name__)


class RedisAdapter(CommSignalBase):
    """Redis List + response key implementation for comm signals."""

    def __init__(
        self,
        redis_url: str,
        command_key: str = "pluck:control",
        response_key_prefix: str = "pluck:response:",
        response_ttl_sec: int = 60,
    ) -> None:
        self.command_key = command_key
        self.response_key_prefix = response_key_prefix
        self.response_ttl_sec = response_ttl_sec
        self._client = redis.Redis.from_url(redis_url, decode_responses=True)

    def receive(self) -> Optional[Dict[str, Any]]:
        """Receive a command from Redis queue (non-blocking)."""
        try:
            raw = self._client.lpop(self.command_key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to receive comm signal: %s", exc)
            return None

    def send(self, message: Dict[str, Any]) -> None:
        """Send a response back to the requester."""
        request_id = message.get("request_id")
        if not request_id:
            if message.get("type") == "pick_result":
                return
            logger.warning("Comm response missing request_id: %s", message)
            return
        response_key = f"{self.response_key_prefix}{request_id}"
        try:
            response = {k: v for k, v in message.items() if k != "request_id"}
            self._client.rpush(response_key, json.dumps(response))
            self._client.expire(response_key, self.response_ttl_sec)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to send comm response: %s", exc)

    def close(self) -> None:
        try:
            self._client.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to close Redis adapter: %s", exc)
