"""Video streaming (MJPEG) endpoint.

Reads annotated frames from Redis Streams and serves them as MJPEG stream.
Frames are stored as raw JPEG bytes in Redis (no base64 encoding).
"""

import asyncio
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from redis import asyncio as aioredis

logger = logging.getLogger(__name__)

router = APIRouter()

BOUNDARY = b"--frame"


def _build_mjpeg_part(frame_bytes: bytes) -> bytes:
    """Build a single MJPEG multipart frame."""
    headers = (
        BOUNDARY
        + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
        + str(len(frame_bytes)).encode()
        + b"\r\n\r\n"
    )
    return headers + frame_bytes + b"\r\n"


@router.get("/stream/video")
async def stream_video(request: Request) -> StreamingResponse:
    """Stream latest annotated frames as MJPEG.
    
    Reads frames from Redis Streams (raw JPEG bytes) and outputs
    as multipart/x-mixed-replace for browser <img> consumption.
    """
    app_state = getattr(request.app.state, "app_state", None)
    if app_state is None or app_state.config is None:
        raise HTTPException(status_code=503, detail="Application not initialized")

    cfg = app_state.config
    if not getattr(cfg, "video_stream", None) or not cfg.video_stream.enabled:
        raise HTTPException(status_code=503, detail="Video stream not enabled")
    if not getattr(cfg, "redis", None) or not cfg.redis.enabled:
        raise HTTPException(status_code=503, detail="Redis not enabled")

    stream_key = cfg.video_stream.stream
    block_ms = cfg.video_stream.block_ms

    async def frame_generator() -> AsyncGenerator[bytes, None]:
        # Use decode_responses=False to get raw bytes from Redis
        client = aioredis.Redis.from_url(cfg.redis.url, decode_responses=False)
        last_id = b"0-0"

        # Prime with latest frame if available
        try:
            latest = await client.xrevrange(stream_key, count=1)
            if latest:
                last_id = latest[0][0]
                frame_bytes = latest[0][1].get(b"frame")
                if frame_bytes:
                    yield _build_mjpeg_part(frame_bytes)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to prime video stream: %s", exc)

        try:
            while True:
                try:
                    results = await client.xread(
                        {stream_key: last_id},
                        block=block_ms,
                        count=1,
                    )
                except asyncio.CancelledError:
                    break
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Video stream read failed: %s", exc)
                    await asyncio.sleep(1)
                    continue

                if not results:
                    continue

                for _, entries in results:
                    for entry_id, fields in entries:
                        last_id = entry_id
                        # frame is raw JPEG bytes, no decoding needed
                        frame_bytes = fields.get(b"frame")
                        if not frame_bytes:
                            continue
                        yield _build_mjpeg_part(frame_bytes)
        finally:
            try:
                await client.close()
            except Exception:
                pass

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
