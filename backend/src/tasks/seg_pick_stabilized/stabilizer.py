"""SegDetection stabilizer wrapper for multi-frame hit gating."""

from __future__ import annotations

import copy
from typing import Iterable

from src.tasks.stabilized_detection.stabilizer import (
    ClusterState,
    Stabilizer,
    StabilizerConfig,
    TargetCluster,
)
from src.types import SegDetection


class SegDetectionStabilizer:
    """Gate SegDetections through the existing multi-frame cluster stabilizer."""

    def __init__(self, config: StabilizerConfig | None = None) -> None:
        self._base = Stabilizer(config=config)

    @property
    def config(self) -> StabilizerConfig:
        return self._base.config

    def reset(self) -> None:
        self._base.reset()

    def get_cluster_count(self) -> int:
        return self._base.get_cluster_count()

    def update(self, detections: list[SegDetection]) -> list[SegDetection]:
        # Reuse the existing clustering / window / hysteresis implementation,
        # then project each stable cluster back to the representative segmented
        # detection from the current temporal window. The representative frame
        # is chosen by maximal mean IoU against the other hits in the cluster,
        # which is more robust than "always take the last frame" when the
        # detector jitters or briefly deforms the bbox near pick time.
        self._base.update(detections)

        stable_detections: list[SegDetection] = []
        for cluster in self._base._clusters:
            if cluster.state != ClusterState.STABLE or not cluster.history:
                continue
            representative = self._select_representative_detection(cluster)
            if not isinstance(representative, SegDetection):
                continue
            stable_detections.append(copy.deepcopy(representative))
        return stable_detections

    def _select_representative_detection(
        self,
        cluster: TargetCluster,
    ) -> SegDetection | None:
        entries = [
            entry for entry in cluster.history
            if isinstance(entry.detection, SegDetection)
        ]
        if not entries:
            return None
        if len(entries) == 1:
            return entries[0].detection

        best_entry = max(
            entries,
            key=lambda entry: (
                _mean_iou(entry.detection, (e.detection for e in entries if e is not entry)),
                float(entry.detection.confidence),
                int(entry.frame_id),
            ),
        )
        return best_entry.detection


def _mean_iou(det: SegDetection, others: Iterable[SegDetection]) -> float:
    scores = [_bbox_iou(det, other) for other in others]
    if not scores:
        return 1.0
    return sum(scores) / float(len(scores))


def _bbox_iou(a: SegDetection, b: SegDetection) -> float:
    ax1, ay1, ax2, ay2 = float(a.bbox.x1), float(a.bbox.y1), float(a.bbox.x2), float(a.bbox.y2)
    bx1, by1, bx2, by2 = float(b.bbox.x1), float(b.bbox.y1), float(b.bbox.x2), float(b.bbox.y2)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union
