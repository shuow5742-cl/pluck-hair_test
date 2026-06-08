"""Routes for the pluck-hair_test console (NEW).

Adds, on top of the original API:

* ``GET /api/test/stream/video`` — MJPEG straight from the in-process
  :class:`FrameBus` (no Redis). This is the left-half live view with the camera
  feed + detection + tweezer overlay already baked in by the frame loop.
* ``GET /api/test/state`` — latest live telemetry (fps, tweezer tip/state,
  tip→pick distance in mm, detection count). The frontend polls this.
* ``/api/epson/*`` and ``/api/io/*`` and ``/api/system`` — the right-half Epson
  manual control and device-IO control, delegated to ``EpsonIoController``.

The Epson controller is attached at ``app.state.epson_io`` by the test
entrypoint; routes 503 cleanly if it is absent (e.g. plain ``--mode api``).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.comm.inproc_bus import get_frame_bus, get_live_state_bus

logger = logging.getLogger(__name__)

router = APIRouter()

BOUNDARY = b"--frame"


def _get_controller(request: Request):
    ctl = getattr(request.app.state, "epson_io", None)
    if ctl is None:
        raise HTTPException(status_code=503, detail="Epson/IO controller not configured")
    return ctl


# ----------------------------------------------------------------------------
# Left half: video + live state
# ----------------------------------------------------------------------------
def _mjpeg_part(frame_bytes: bytes) -> bytes:
    return (
        BOUNDARY
        + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
        + str(len(frame_bytes)).encode()
        + b"\r\n\r\n"
        + frame_bytes
        + b"\r\n"
    )


@router.get("/test/stream/video")
async def test_stream_video() -> StreamingResponse:
    """Stream the annotated live frames as MJPEG from the in-process FrameBus."""
    bus = get_frame_bus()

    async def gen():
        last_seq = -1
        # Prime immediately with whatever frame exists.
        frame, seq = bus.latest()
        if frame is not None:
            last_seq = seq
            yield _mjpeg_part(frame)
        while True:
            frame, seq = await asyncio.to_thread(bus.wait_for, last_seq, 4.0)
            if seq == last_seq or frame is None:
                # keep the connection warm even if the loop is briefly idle
                await asyncio.sleep(0.05)
                continue
            last_seq = seq
            yield _mjpeg_part(frame)

    return StreamingResponse(
        gen(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/test/state")
async def test_state() -> Dict[str, Any]:
    return get_live_state_bus().snapshot()


# ----------------------------------------------------------------------------
# Right half: Epson manual control
# ----------------------------------------------------------------------------
class JogRequest(BaseModel):
    axis: str
    direction: str               # "+" | "-"
    step_mode: str = "short"


class MoveRequest(BaseModel):
    target: Optional[Dict[str, float]] = None
    point: Optional[str] = None
    command: str = "Move"        # "Move" | "Go"


class IoRequest(BaseModel):
    device_id: str
    action: str                  # extend|retract|open|close|start|stop|lock|unlock


class SystemRequest(BaseModel):
    command: str                 # auto|init|start|stop|pause|reset|manual_pick_ok


@router.get("/epson/describe")
async def epson_describe(request: Request) -> Dict[str, Any]:
    return _get_controller(request).describe()


@router.get("/epson/pose")
async def epson_pose(request: Request) -> Dict[str, Any]:
    ctl = _get_controller(request)
    return {"pose": ctl.get_pose(), "units": ctl.units}


@router.get("/epson/points")
async def epson_points(request: Request) -> Dict[str, Any]:
    return {"points": _get_controller(request).list_points()}


@router.post("/epson/jog")
async def epson_jog(request: Request, body: JogRequest) -> Dict[str, Any]:
    return _get_controller(request).jog(body.axis, body.direction, body.step_mode)


@router.post("/epson/move")
async def epson_move(request: Request, body: MoveRequest) -> Dict[str, Any]:
    ctl = _get_controller(request)
    if body.point:
        return ctl.goto_point(body.point, body.command)
    if body.target:
        return ctl.move(body.target, body.command)
    raise HTTPException(status_code=400, detail="provide either 'point' or 'target'")


# ----------------------------------------------------------------------------
# Right half: device IO + system control
# ----------------------------------------------------------------------------
@router.get("/io/states")
async def io_states(request: Request) -> Dict[str, Any]:
    return {"devices": _get_controller(request).get_io_states()}


@router.post("/io/set")
async def io_set(request: Request, body: IoRequest) -> Dict[str, Any]:
    return _get_controller(request).set_io(body.device_id, body.action)


@router.post("/system")
async def system_command(request: Request, body: SystemRequest) -> Dict[str, Any]:
    return _get_controller(request).system_command(body.command)


@router.get("/system/state")
async def system_state(request: Request) -> Dict[str, Any]:
    return _get_controller(request).get_system_state()
