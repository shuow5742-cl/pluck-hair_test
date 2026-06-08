"""In-process frame + live-state bus for the single-process test app.

The production backend is *dual-process* (vision loop ``--mode run`` and the
FastAPI server ``--mode api``) and bridges them through Redis Streams. The
``pluck-hair_test`` console instead runs everything in ONE process (``--mode
test``): the vision :class:`WorkflowEngine` in a background thread and uvicorn in
the main thread. They share state through these tiny thread-safe singletons —
no Redis required, so the operator can just launch the script and open the page.

* :class:`FrameBus` holds the latest annotated JPEG frame (camera + detection +
  tweezer overlay). The MJPEG route waits on its condition variable and streams
  each new frame.
* :class:`LiveStateBus` holds the latest live telemetry dict (fps, tweezer tip /
  state / distance-to-pick, detection counts). The frontend polls it.

Both are module-level singletons so the frame-loop side task (which is
constructed deep inside the workflow wiring) and the API routes can reach the
same instance without threading a reference through every constructor.
"""

from __future__ import annotations

import threading
import time
from typing import Optional


class FrameBus:
    """Latest-JPEG-frame holder with a wait primitive for MJPEG streaming."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._frame: Optional[bytes] = None
        self._seq: int = 0
        self._frame_id: str = ""
        self._updated_at: float = 0.0

    def publish(self, frame_bytes: bytes, *, frame_id: str = "", timestamp: str = "") -> None:
        """Publisher interface compatible with ``FrameStreamer``.

        ``FrameStreamer`` calls ``publisher.publish(jpeg_bytes, frame_id=...,
        timestamp=...)`` — we match that signature so the frame loop can use this
        bus as a drop-in ``frame_publisher``.
        """
        with self._cond:
            self._frame = frame_bytes
            self._seq += 1
            self._frame_id = frame_id
            self._updated_at = time.time()
            self._cond.notify_all()

    def latest(self) -> tuple[Optional[bytes], int]:
        with self._lock:
            return self._frame, self._seq

    def wait_for(self, last_seq: int, timeout: float = 5.0) -> tuple[Optional[bytes], int]:
        """Block until a frame newer than ``last_seq`` arrives (or timeout)."""
        with self._cond:
            if self._seq <= last_seq:
                self._cond.wait(timeout=timeout)
            return self._frame, self._seq

    @property
    def age_seconds(self) -> Optional[float]:
        with self._lock:
            if not self._updated_at:
                return None
            return time.time() - self._updated_at


class LiveStateBus:
    """Latest live-telemetry holder (simple last-write-wins dict)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict = {
            "fps": 0.0,
            "frame": 0,
            "detection_count": 0,
            "tweezer": {"found": False},
            "tip_to_pick_mm": None,
            "ar_mode": "LIVE",
            "updated_at": 0.0,
        }

    def update(self, patch: dict) -> None:
        with self._lock:
            self._state.update(patch)
            self._state["updated_at"] = time.time()

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._state)


# Module-level singletons -----------------------------------------------------
_FRAME_BUS: Optional[FrameBus] = None
_LIVE_STATE_BUS: Optional[LiveStateBus] = None


def get_frame_bus() -> FrameBus:
    global _FRAME_BUS
    if _FRAME_BUS is None:
        _FRAME_BUS = FrameBus()
    return _FRAME_BUS


def get_live_state_bus() -> LiveStateBus:
    global _LIVE_STATE_BUS
    if _LIVE_STATE_BUS is None:
        _LIVE_STATE_BUS = LiveStateBus()
    return _LIVE_STATE_BUS
