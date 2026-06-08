"""StabilizedDetectionTask bounded context."""

from .task import StabilizedDetectionTask

# PickTask is a legacy alias
PickTask = StabilizedDetectionTask

__all__ = ["StabilizedDetectionTask", "PickTask"]
