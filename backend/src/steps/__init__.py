"""Hub-side pipeline steps that plug into autoweaver's VisionPipeline.

Each step registers itself with autoweaver via ``register_step`` at import
time so configs can reference them by name (e.g. ``type: yolo_seg``).
"""

from .crop_single_square import CropSingleSquareStep
from .abstain_near_metal import AbstainNearMetalStep
from .yolo_seg import YOLOSegStep

__all__ = ["CropSingleSquareStep", "YOLOSegStep", "AbstainNearMetalStep"]
