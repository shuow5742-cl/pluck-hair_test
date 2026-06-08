"""Frame rendering helpers for WorkflowEngine (helper, not a Task)."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from src.utils.visualization import draw_detections, draw_stable_targets


class FrameRenderer:
    """Render preview, stream, and annotated frames."""

    def __init__(
        self,
        color_map: dict,
        max_display_width: int = 1280,
    ) -> None:
        self.color_map = color_map
        self.max_display_width = max_display_width

    def build_preview(self, image: np.ndarray, result) -> np.ndarray:
        preview = self._draw_for_display(image, result)
        h, w = preview.shape[:2]
        if w > self.max_display_width:
            scale = self.max_display_width / w
            preview = cv2.resize(preview, (int(w * scale), int(h * scale)))
        return preview

    def build_stream_frame(self, image: np.ndarray, result) -> np.ndarray:
        return self._draw_for_display(image, result)

    def build_annotated(
        self,
        image: np.ndarray,
        result,
        *,
        enabled: bool,
    ) -> Optional[np.ndarray]:
        if not enabled or not result.detections:
            return None
        return draw_detections(image, result.detections, self.color_map)

    def _draw_for_display(self, image: np.ndarray, result) -> np.ndarray:
        if getattr(result, "tracked_targets", None):
            return draw_stable_targets(
                image, result.tracked_targets, self.color_map,
            )
        if result.stable_targets:
            return draw_stable_targets(
                image, result.stable_targets, self.color_map,
            )
        if result.detections:
            return draw_detections(image, result.detections, self.color_map)
        return image.copy()
