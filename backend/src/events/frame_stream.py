"""Redis Streams helpers for MJPEG frame transport.

Frames are stored as raw JPEG bytes in Redis (no base64 encoding).
"""

from __future__ import annotations

import logging

import redis

logger = logging.getLogger(__name__)


class FrameStreamPublisher:
    """Publishes JPEG frames to a Redis Stream.
    
    Frames are stored as raw bytes (not base64 encoded) for efficiency.
    The reader must use decode_responses=False to receive bytes.
    """

    def __init__(self, url: str, stream: str, maxlen: int = 50):
        self.stream = stream
        self.maxlen = maxlen
        # Use decode_responses=False to store raw bytes
        self._client = redis.Redis.from_url(url, decode_responses=False)

    def publish(self, frame_bytes: bytes, frame_id: str, timestamp: str) -> None:
        """Publish a JPEG frame to the stream.
        
        Args:
            frame_bytes: Raw JPEG bytes (from cv2.imencode).
            frame_id: Unique frame identifier.
            timestamp: ISO format timestamp string.
        """
        try:
            # Store raw bytes directly, no base64 encoding needed
            self._client.xadd(
                self.stream,
                {
                    b"frame": frame_bytes,
                    b"frame_id": frame_id.encode(),
                    b"timestamp": timestamp.encode(),
                },
                maxlen=self.maxlen,
                approximate=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to publish frame to Redis Streams: %s", exc)

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass
