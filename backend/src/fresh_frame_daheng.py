"""Daheng camera wrapper that flushes stale stream frames before capture."""

from __future__ import annotations

import logging

import numpy as np
from autoweaver.camera import DahengCamera

logger = logging.getLogger(__name__)


class FreshFrameDahengCamera(DahengCamera):
    """Capture the next fresh frame instead of an older buffered frame.

    The base DahengCamera reads the next frame from the continuous stream.
    When the rest of the pipeline is slower than camera FPS, that can return a
    stale frame that was buffered while the robot was moving. Flushing the
    stream queue first makes each `capture()` behave like "capture now".
    """

    def capture(self) -> np.ndarray:
        if not self._is_opened:
            raise RuntimeError("Camera not opened")

        gx = self._gx
        data_stream = self._cam.data_stream[0]
        try:
            data_stream.flush_queue()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to flush Daheng stream queue before capture: %s", exc)

        raw_image = data_stream.get_image()
        if raw_image is None:
            raise RuntimeError("Failed to capture image")

        if raw_image.get_status() != gx.GxFrameStatusList.SUCCESS:
            raise RuntimeError("Frame capture failed: incomplete frame")

        rgb_image = raw_image.convert(
            "RGB",
            channel_order=gx.DxRGBChannelOrder.ORDER_BGR,
        )
        if rgb_image is None:
            raise RuntimeError("Failed to convert image to BGR")

        image = rgb_image.get_numpy_array()
        if image is None:
            raise RuntimeError("Failed to get numpy array")

        return image
