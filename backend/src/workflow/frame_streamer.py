"""Frame streaming helper for WorkflowEngine (helper, not a Task)."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np


class FrameStreamer:
    """Handle fps throttling, encoding, and publishing frames."""

    def __init__(self, publisher, config) -> None:
        self.publisher = publisher
        self.config = config
        self._last_push_time = 0.0
        self._logger = logging.getLogger(__name__)

    def publish(
        self,
        frame: Optional[np.ndarray],
        *,
        frame_id: str,
        timestamp: datetime,
    ) -> None:
        if frame is None or not self.publisher or not self.config or not self.config.enabled:
            return

        now = time.time()
        min_interval = 1.0 / self.config.fps_limit if self.config.fps_limit > 0 else 0
        if min_interval and (now - self._last_push_time) < min_interval:
            return

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, int(self.config.jpeg_quality)]
        success, buffer = cv2.imencode(".jpg", frame, encode_params)
        if not success:
            self._logger.warning("Failed to encode frame for streaming (id=%s)", frame_id)
            return

        try:
            self.publisher.publish(
                buffer.tobytes(),
                frame_id=frame_id,
                timestamp=timestamp.isoformat() + "Z",
            )
            self._last_push_time = now
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("Failed to publish frame: %s", exc)
