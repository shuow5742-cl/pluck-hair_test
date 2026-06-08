"""PixelToWorldTask — reactive pixel→world coordinate transform.

Listens on TASK:ITERATION (broadcast by SegPickTask), reads the live flange
pose from a PoseSource registered by name, applies the eye-in-hand 3-DOF
extrinsic transform to each pick's pixel coordinates, and re-broadcasts
TASK:WORLD_PICKS for downstream consumers (e.g. CommunicationTask sending
the result to the PLC).

Architectural choice (route B): keep pipeline as pure image processing and
do coordinate work as a SideTask. Lets us swap pose sources (mock / PLC /
replay) without touching the segmentation pipeline.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from autoweaver.reactive.event_bus import EventBus

from src.core.coordinate_transform import (
    CoordinateTransformer,
    ExtrinsicCalibration,
)
from src.pose_sources import FlangePose, PoseSource, get_pose_source
from src.types import SegDetection

logger = logging.getLogger(__name__)


class PixelToWorldTask:
    """SideTask: pixel picks → world picks (mm), via PoseSource lookup.

    Wired by name, not by object — instantiation reads ``pose_source_name``
    out of config and resolves to a live PoseSource through the registry.
    This is the same dependency-injection pattern that storage / publisher
    setup uses in main.py.

    The autoweaver SideTask protocol is satisfied structurally (name +
    attach + close); no need to inherit anything.
    """

    def __init__(
        self,
        calibration: ExtrinsicCalibration,
        pose_source_name: str = "default",
        *,
        name: str = "pixel_to_world",
    ) -> None:
        self._name = name
        self._transformer = CoordinateTransformer(calibration)
        self._pose_source_name = pose_source_name
        # Resolve lazily — the registry may not be populated at construction
        # time if config wiring happens before main.py's startup hooks.
        self._pose_source: Optional[PoseSource] = None
        self._event_bus: Optional[EventBus] = None
        self._unsubscribers: list[Callable[[], None]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def pose_source(self) -> PoseSource:
        if self._pose_source is None:
            self._pose_source = get_pose_source(self._pose_source_name)
        return self._pose_source

    # ---- SideTask protocol ----

    def attach(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._unsubscribers.append(
            event_bus.subscribe("TASK:ITERATION", self._on_iteration)
        )

    def close(self) -> None:
        for unsub in self._unsubscribers:
            try:
                unsub()
            except Exception:  # noqa: BLE001
                pass
        self._unsubscribers.clear()
        self._event_bus = None

    # ---- Event handlers ----

    def _on_iteration(self, _event: str, data: dict) -> None:
        payload = data.get("payload", {})
        detections = payload.get("detections") or []
        seg_dets = [d for d in detections if isinstance(d, SegDetection)]

        # Always broadcast — an empty world_picks list is a meaningful signal
        # (e.g. the PLC orchestrator treats it as "no hairs at this photo
        # position, skip"). Silently swallowing 0-detection frames would
        # leave that downstream consumer hanging on a stale buffer.

        # Pose at the moment we process this frame. With the current setup
        # (synchronous pipeline → broadcast → here), this is the closest we
        # can come to "the pose when the frame was captured" until the PLC
        # supplies timestamped pose history we can interpolate against.
        pose = self._read_pose()

        # Per-frame audit line so the field test can untangle which stage of
        # the pixel→world→Epson chain a Δ(target,actual) attribution lands
        # in. Drops one line per frame even with zero detections so we can
        # see "pose was X when the camera fired but no hair detected".
        cal = self._transformer._cal  # frozen dataclass; safe to reach in
        if pose is not None:
            logger.info(
                "p2w pose flange_xy=(%.4f, %.4f) extrinsic dx=%.4f dy=%.4f "
                "mm_per_pixel=%s axis=(x:%s, y:%s) cx=%.2f cy=%.2f "
                "n_dets=%d seg_frame=%s",
                pose.x, pose.y, cal.dx, cal.dy, cal.mm_per_pixel,
                cal.flange_x_from, cal.flange_y_from, cal.cx, cal.cy,
                len(seg_dets),
                (payload.get("metadata") or {}).get("seg_frame_id"),
            )
        else:
            logger.warning(
                "p2w pose UNAVAILABLE (degraded) — n_dets=%d seg_frame=%s",
                len(seg_dets),
                (payload.get("metadata") or {}).get("seg_frame_id"),
            )

        world_picks: list[dict[str, Any]] = []
        for d in seg_dets:
            world_xy = _pixel_to_world(d.pick_point_xy, pose, self._transformer)
            d.world_xy = world_xy  # mutate detection in place
            if world_xy is not None and d.pick_point_xy is not None and pose is not None:
                # Reproduce the pieces the formula uses so the line is
                # self-explanatory without needing to re-derive anything.
                px = float(d.pick_point_xy[0])
                py = float(d.pick_point_xy[1])
                dpx_mm = (px - cal.cx) * cal.mm_per_pixel
                dpy_mm = (py - cal.cy) * cal.mm_per_pixel
                logger.info(
                    "  p2w det=%s px=(%.1f, %.1f) Δpx=(%.1f, %.1f) "
                    "Δmm=(%.4f, %.4f) → world_xy=(%.4f, %.4f) "
                    "[= flange + extrinsic + pixel_offset]",
                    d.detection_id, px, py, px - cal.cx, py - cal.cy,
                    dpx_mm, dpy_mm, world_xy[0], world_xy[1],
                )
            if world_xy is not None:
                # flange_target_xy = "what flange XY would put the optical
                # center on this pick pixel". Computed as world_xy − (dx, dy)
                # to undo the camera-extrinsic offset baked into pixel_to_world.
                # Downstream consumers (plc_orchestrator) feed this into the
                # nova5_to_epson grid lookup, whose anchors are keyed on
                # flange position. world_xy alone is in the optical-center
                # frame and would mismatch the grid by exactly (dx, dy).
                flange_target_xy = [
                    float(world_xy[0]) - cal.dx,
                    float(world_xy[1]) - cal.dy,
                ]
                world_picks.append({
                    "detection_id": d.detection_id,
                    "object_type": d.object_type,
                    "confidence": d.confidence,
                    "shape_class": d.shape_class,
                    "pick_point_xy_px": d.pick_point_xy,
                    "bbox_center_xy_px": list(d.bbox.center),
                    "bbox_width_px": float(d.bbox.width),
                    "bbox_height_px": float(d.bbox.height),
                    "preferred_epson_tool": getattr(d, "preferred_epson_tool", None),
                    "pick_angle_deg": d.pick_angle_deg,
                    "pick_method": d.pick_method,
                    "world_xy_mm": world_xy,
                    "flange_target_xy_mm": flange_target_xy,
                    "flange_pose_mm": [pose.x, pose.y, pose.z] if pose else None,
                })

        out_payload = {
            "source": self._name,
            "payload": {
                "world_picks": world_picks,
                "flange_pose_mm": [pose.x, pose.y, pose.z] if pose else None,
                "seg_frame_id": (payload.get("metadata") or {}).get("seg_frame_id"),
                "degraded": pose is None,
                "metadata": payload.get("metadata") or {},
            },
        }
        if self._event_bus is not None:
            self._event_bus.publish("TASK:WORLD_PICKS", out_payload)

    def _read_pose(self) -> Optional[FlangePose]:
        try:
            return self.pose_source.read()
        except Exception as exc:  # noqa: BLE001
            logger.warning("PoseSource %s failed: %s", self._pose_source_name, exc)
            return None


def _pixel_to_world(
    pixel_xy: Optional[list[float]],
    pose: Optional[FlangePose],
    transformer: CoordinateTransformer,
) -> Optional[list[float]]:
    if pixel_xy is None or pose is None:
        return None
    wp = transformer.pixel_to_world(
        float(pixel_xy[0]), float(pixel_xy[1]),
        arm_x=pose.x, arm_y=pose.y,
    )
    return [wp.x, wp.y]
