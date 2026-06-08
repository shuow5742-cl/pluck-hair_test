"""Task-level statistics tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class TaskStats:
    """Lightweight stats container for a task."""

    total_frames: int = 0
    total_detections: int = 0
    confirmed_detections: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def record(self, result: Any) -> None:
        """Record a single iteration."""
        if self.start_time is None:
            self.start_time = datetime.utcnow()
        self.total_frames += 1
        self.total_detections += len(getattr(result, "detections", []))
        self.confirmed_detections += result.metadata.get("confirmed_count", 0)
        self.metadata.update(result.metadata)

    def summary(self) -> Dict[str, Any]:
        """Return a summary dictionary."""
        duration = None
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()

        return {
            "total_frames": self.total_frames,
            "total_detections": self.total_detections,
            "confirmed_detections": self.confirmed_detections,
            "duration_seconds": duration,
            "metadata": self.metadata,
        }

    def reset(self) -> None:
        """Reset stats for a new run."""
        self.total_frames = 0
        self.total_detections = 0
        self.confirmed_detections = 0
        self.start_time = None
        self.end_time = None
        self.metadata = {}
