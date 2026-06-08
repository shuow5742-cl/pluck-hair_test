"""Redis Streams helpers for inter-process messaging."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Awaitable, Callable, Optional

import redis
from redis import asyncio as aioredis
from redis.exceptions import ResponseError

logger = logging.getLogger(__name__)

MessageHandler = Callable[[dict], Awaitable[None] | None]


class RedisStreamPublisher:
    """Lightweight Redis Streams publisher for detection events."""

    def __init__(
        self,
        url: str,
        stream: str,
        maxlen: Optional[int] = None,
    ):
        self.stream = stream
        self.maxlen = maxlen
        self._client = redis.Redis.from_url(url, decode_responses=True)

    def publish(self, payload: dict) -> None:
        """Publish a payload to the stream."""
        try:
            self._client.xadd(
                self.stream,
                {"payload": json.dumps(payload)},
                maxlen=self.maxlen,
                approximate=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to publish event to Redis Streams: %s", exc)

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass


class RedisStreamConsumer:
    """Async Redis Streams consumer with consumer group support."""

    def __init__(
        self,
        url: str,
        stream: str,
        group: str,
        consumer_name: str,
        *,
        block_ms: int = 5000,
        count: int = 10,
    ):
        self.stream = stream
        self.group = group
        self.consumer_name = consumer_name
        self.block_ms = block_ms
        self.count = count
        self._client = aioredis.Redis.from_url(url, decode_responses=True)
        self._stopped = asyncio.Event()

    async def ensure_group(self) -> None:
        """Create the consumer group if it does not exist."""
        try:
            await self._client.xgroup_create(
                name=self.stream,
                groupname=self.group,
                id="$",
                mkstream=True,
            )
            logger.info("Created Redis consumer group '%s' on stream '%s'", self.group, self.stream)
        except ResponseError as exc:
            # BUSYGROUP means it already exists
            if "BUSYGROUP" not in str(exc):
                raise

    async def consume(self, handler: MessageHandler) -> None:
        """Continuously consume messages and pass to handler."""
        await self.ensure_group()

        while not self._stopped.is_set():
            try:
                messages = await self._client.xreadgroup(
                    groupname=self.group,
                    consumername=self.consumer_name,
                    streams={self.stream: ">"},
                    count=self.count,
                    block=self.block_ms,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Redis consumer read failed: %s", exc)
                await asyncio.sleep(1)
                continue

            if not messages:
                continue

            for stream_name, entries in messages:
                if stream_name != self.stream:
                    continue
                for entry_id, fields in entries:
                    payload_text = fields.get("payload")
                    try:
                        payload = json.loads(payload_text) if payload_text else {}
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Invalid payload in stream %s: %s", self.stream, exc)
                        await self._ack(entry_id)
                        continue

                    try:
                        result = handler(payload)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Handler failed for stream entry %s: %s", entry_id, exc)
                    finally:
                        await self._ack(entry_id)

    async def _ack(self, entry_id: str) -> None:
        try:
            await self._client.xack(self.stream, self.group, entry_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to ACK Redis entry %s: %s", entry_id, exc)

    async def stop(self) -> None:
        self._stopped.set()

    async def close(self) -> None:
        try:
            await self._client.close()
        except Exception:
            pass
