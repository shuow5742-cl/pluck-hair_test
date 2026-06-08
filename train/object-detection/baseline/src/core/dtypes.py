from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DetectionResult:
    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    image_path: Optional[str] = None

    def __post_init__(self) -> None:
        num = len(self.boxes)
        if len(self.scores) != num or len(self.labels) != num:
            raise ValueError("boxes, scores, labels length mismatch")
