"""Event payload builders for WorkflowEngine (helper, not a Task)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.storage.interfaces import DetectionRecord


@dataclass(frozen=True)
class EventContext:
    """Context needed to build event payloads."""

    session_id: Optional[str]
    frame: int
    total_detections: int


class EventBuilder:
    """Build event payloads for external publishers."""

    @staticmethod
    def _enum_value(value: object) -> object:
        return value.value if hasattr(value, "value") else value

    def build_live_detection_event(
        self,
        *,
        detections: list[object],
        tracked_targets: list[object],
        timestamp: datetime,
        context: EventContext,
        event_metadata: Optional[dict] = None,
    ) -> dict:
        by_type: dict[str, int] = {}
        for detection in detections:
            object_type = str(getattr(detection, "object_type", "unknown"))
            by_type[object_type] = by_type.get(object_type, 0) + 1

        targets: list[dict] = []
        for target in tracked_targets:
            targets.append(
                {
                    "track_id": getattr(target, "track_id", None),
                    "cluster_id": getattr(target, "cluster_id", None),
                    "state": self._enum_value(getattr(target, "state", None)),
                    "object_type": getattr(target, "object_type", None),
                    "confidence": getattr(target, "confidence", None),
                    "x_px": getattr(target, "x", None),
                    "y_px": getattr(target, "y", None),
                    "width_px": getattr(target, "width", None),
                    "height_px": getattr(target, "height", None),
                    "world_x_mm": getattr(target, "world_x", None),
                    "world_y_mm": getattr(target, "world_y", None),
                    "u_deg": getattr(target, "u", None),
                }
            )

        return {
            "type": "detection",
            "session_id": context.session_id,
            "frame": context.frame,
            "timestamp": timestamp.isoformat() + "Z",
            "detection_count": len(detections),
            "by_type": by_type,
            "total_detections": context.total_detections,
            "region_picked": int((event_metadata or {}).get("region_picked") or 0),
            "current_pending": int((event_metadata or {}).get("current_pending") or 0),
            "cluster_count": int((event_metadata or {}).get("cluster_count") or 0),
            "stable_count": int((event_metadata or {}).get("stable_count") or 0),
            "phase": (event_metadata or {}).get("phase"),
            "fps": (event_metadata or {}).get("fps"),
            "targets": targets,
        }

    def build_detection_event(
        self,
        *,
        records: list[DetectionRecord],
        image_path: str,
        annotated_path: Optional[str],
        timestamp: datetime,
        event_metadata: Optional[dict],
        context: EventContext,
    ) -> dict:
        by_type: dict[str, int] = {}
        for record in records:
            by_type[record.object_type] = by_type.get(record.object_type, 0) + 1

        return {
            "type": "detection",
            "session_id": context.session_id,
            "frame": context.frame,
            "timestamp": timestamp.isoformat() + "Z",
            "image_path": image_path,
            "annotated_path": annotated_path,
            "detection_count": len(records),
            "by_type": by_type,
            "total_detections": context.total_detections,
            "region_picked": int((event_metadata or {}).get("region_picked") or 0),
            "current_pending": int((event_metadata or {}).get("current_pending") or 0),
        }

    def build_pick_result_event(
        self,
        *,
        pick_result: object,
        context: EventContext,
    ) -> dict:
        return {
            "type": "pick_result",
            "success": pick_result.success,
            "target_id": pick_result.target_id,
            "message": pick_result.message,
            "session_id": context.session_id,
            "frame": context.frame,
        }
