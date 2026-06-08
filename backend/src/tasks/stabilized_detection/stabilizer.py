"""Stabilizer for multi-frame target clustering and stabilization."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
import itertools
import math
import statistics

from autoweaver.pipeline import Detection

type Point = tuple[float, float]


class ClusterState(StrEnum):
    """Cluster lifecycle state."""

    TENTATIVE = "tentative"
    STABLE = "stable"


@dataclass(slots=True, frozen=True)
class ClusterDetection:
    """Detection stored with frame context."""

    frame_id: int
    detection: Detection
    center: Point


@dataclass(slots=True)
class TargetCluster:
    """Cluster of detections belonging to the same target."""

    cluster_id: str
    object_type: str
    center: Point
    history: deque[ClusterDetection] = field(default_factory=deque)
    last_seen_frame: int = 0
    state: ClusterState = ClusterState.TENTATIVE
    created_frame: int = 0

    def prune_history(self, current_frame: int, window_size: int) -> None:
        """Drop history entries outside the sliding window."""
        min_frame = current_frame - window_size
        while self.history and self.history[0].frame_id <= min_frame:
            self.history.popleft()

    def occurrence_ratio(self, window_size: int, current_frame: int) -> float:
        """Ratio of frames where this cluster appears within the window."""
        if window_size <= 0:
            return 0.0
        effective_window = min(current_frame - self.created_frame + 1, window_size)
        if effective_window <= 0:
            return 0.0
        return len(self.history) / float(effective_window)


@dataclass(slots=True)
class StableTarget:
    """Stable target derived from a cluster.

    All coordinates and sizes are in pixels.
    World coordinates (mm) are optionally attached after conversion.
    """

    x: float  # center x (pixels)
    y: float  # center y (pixels)
    width: float  # bbox width (pixels)
    height: float  # bbox height (pixels)
    confidence: float  # 0-1, weighted average
    occurrence_ratio: float  # 0-1, frames appeared / window size
    object_type: str
    cluster_id: str
    world_x: float | None = None  # world X (mm), set by TargetConverter
    world_y: float | None = None  # world Y (mm), set by TargetConverter
    u: float | None = None  # grasp-axis yaw in image coordinates (deg)


@dataclass(slots=True)
class StabilizerConfig:
    """Configuration for Stabilizer."""

    window_size: int = 10
    min_occurrence_ratio: float = 0.6
    # Hysteresis: exit threshold lower than entry to prevent flickering
    # Set to None to use same threshold as min_occurrence_ratio (no hysteresis)
    stable_exit_ratio: float | None = 0.3
    # Minimum frames a cluster must exist before it can become stable
    # Prevents noise from quickly becoming stable in early frames
    min_frames_to_stable: int = 6
    distance_threshold_px: float = 30.0
    jump_threshold_px: float = 60.0
    missing_frames_to_delete: int = 4
    reset_on_jump: bool = True
    bbox_aggregation: str = "iqr_median"  # iqr_median | median


class Stabilizer:
    """Multi-frame stabilizer using center-distance association."""

    def __init__(self, config: StabilizerConfig | None = None) -> None:
        self.config = config or StabilizerConfig()
        self._clusters: list[TargetCluster] = []
        self._frame_id = 0
        self._cluster_counter = itertools.count(1)

    def reset(self) -> None:
        """Clear all clusters and counters."""
        self._clusters = []
        self._frame_id = 0
        self._cluster_counter = itertools.count(1)

    def get_cluster_count(self) -> int:
        """Return current number of clusters."""
        return len(self._clusters)

    def update(self, detections: list[Detection]) -> list[StableTarget]:
        """Update clusters with detections of the current frame."""
        self._frame_id += 1
        matched_cluster_ids: set[str] = set()

        for detection in sorted(detections, key=lambda d: d.confidence, reverse=True):
            center = detection.bbox.center
            best_cluster, best_distance = self._find_best_cluster(
                detection, center, matched_cluster_ids
            )
            if best_cluster is None:
                new_cluster = self._create_cluster(detection, center)
                self._clusters.append(new_cluster)
                matched_cluster_ids.add(new_cluster.cluster_id)
                continue

            self._update_cluster(best_cluster, detection, center, best_distance)
            matched_cluster_ids.add(best_cluster.cluster_id)

        self._cleanup_clusters()
        self._update_cluster_states()
        return self._collect_stable_targets()

    def _find_best_cluster(
        self,
        detection: Detection,
        center: Point,
        matched_cluster_ids: set[str],
    ) -> tuple[TargetCluster | None, float]:
        """Find the best matching cluster for a detection. Using Greedy Nearest Neighbor Matching method. Skipping when the distance exceeds threshold."""
        best_cluster: TargetCluster | None = None
        best_distance = float("inf")
        for cluster in self._clusters:
            if cluster.object_type != detection.object_type:
                continue
            if cluster.cluster_id in matched_cluster_ids:
                continue
            distance = self._distance(center, cluster.center)
            if distance > self.config.distance_threshold_px:
                continue
            if distance < best_distance:
                best_distance = distance
                best_cluster = cluster
        return best_cluster, best_distance

    def _create_cluster(self, detection: Detection, center: Point) -> TargetCluster:
        cluster_id = f"cluster_{next(self._cluster_counter)}"
        history = deque([ClusterDetection(self._frame_id, detection, center)])
        return TargetCluster(
            cluster_id=cluster_id,
            object_type=detection.object_type,
            center=center,
            history=history,
            last_seen_frame=self._frame_id,
            state=ClusterState.TENTATIVE,
            created_frame=self._frame_id,
        )

    def _update_cluster(
        self,
        cluster: TargetCluster,
        detection: Detection,
        center: Point,
        distance: float,
    ) -> None:
        if distance > self.config.jump_threshold_px and self.config.reset_on_jump:
            cluster.history.clear()
        cluster.center = center
        cluster.history.append(ClusterDetection(self._frame_id, detection, center))
        cluster.last_seen_frame = self._frame_id

    def _cleanup_clusters(self) -> None:
        """Prune history and drop clusters missing for too many frames."""
        keep_clusters = []
        for cluster in self._clusters:
            cluster.prune_history(self._frame_id, self.config.window_size)
            if self._frame_id - cluster.last_seen_frame > self.config.missing_frames_to_delete:
                continue
            keep_clusters.append(cluster)
        self._clusters = keep_clusters

    def _update_cluster_states(self) -> None:
        # Hysteresis: use different thresholds for entering vs exiting stable
        entry_threshold = self.config.min_occurrence_ratio
        exit_threshold = (
            self.config.stable_exit_ratio
            if self.config.stable_exit_ratio is not None
            else self.config.min_occurrence_ratio
        )

        for cluster in self._clusters:
            ratio = cluster.occurrence_ratio(self.config.window_size, self._frame_id)
            cluster_age = self._frame_id - cluster.created_frame + 1

            if cluster.state == ClusterState.TENTATIVE:
                # Must meet both: ratio threshold AND minimum age
                if ratio >= entry_threshold and cluster_age >= self.config.min_frames_to_stable:
                    cluster.state = ClusterState.STABLE
            elif cluster.state == ClusterState.STABLE and ratio < exit_threshold:
                cluster.state = ClusterState.TENTATIVE

    def _collect_stable_targets(self) -> list[StableTarget]:
        stable_targets: list[StableTarget] = []
        for cluster in self._clusters:
            if cluster.state != ClusterState.STABLE:
                continue
            center = self._aggregate_center(cluster)
            width, height = self._aggregate_bbox_size(cluster)
            confidence = self._aggregate_confidence(cluster)
            ratio = cluster.occurrence_ratio(self.config.window_size, self._frame_id)
            stable_targets.append(
                StableTarget(
                    x=center[0],
                    y=center[1],
                    width=width,
                    height=height,
                    confidence=confidence,
                    occurrence_ratio=ratio,
                    object_type=cluster.object_type,
                    cluster_id=cluster.cluster_id,
                )
            )
        return stable_targets

    def _aggregate_center(self, cluster: TargetCluster) -> Point:
        """Return median center from the cluster's recent history."""
        centers = [entry.center for entry in cluster.history]
        if not centers:
            return cluster.center
        xs = [center[0] for center in centers]
        ys = [center[1] for center in centers]
        return (statistics.median(xs), statistics.median(ys))

    def _aggregate_confidence(self, cluster: TargetCluster) -> float:
        """Return weighted average confidence from recent history."""
        confidences = [entry.detection.confidence for entry in cluster.history]
        if not confidences:
            return 0.0
        weights = list(range(1, len(confidences) + 1))
        weighted = sum(c * w for c, w in zip(confidences, weights))
        return weighted / float(sum(weights))

    def _aggregate_bbox_size(self, cluster: TargetCluster) -> tuple[float, float]:
        """Return aggregated bbox width and height using configured method."""
        widths = [entry.detection.bbox.width for entry in cluster.history]
        heights = [entry.detection.bbox.height for entry in cluster.history]
        
        if not widths:
            # Fallback: use latest detection if history is somehow empty
            return (0.0, 0.0)
        
        if self.config.bbox_aggregation == "iqr_median":
            width = self._aggregate_iqr_median(widths)
            height = self._aggregate_iqr_median(heights)
        else:
            # Default: simple median
            width = statistics.median(widths)
            height = statistics.median(heights)
        
        return (width, height)

    @staticmethod
    def _aggregate_iqr_median(values: list[float]) -> float:
        """IQR outlier filtering + median, with fallback for small samples.
        
        Uses standard 1.5×IQR rule (Tukey, 1977) for outlier detection.
        Falls back to simple median when sample is too small or all filtered.
        """
        if len(values) <= 3:
            return statistics.median(values)
        
        # Calculate IQR bounds
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        q1 = sorted_vals[q1_idx]
        q3 = sorted_vals[q3_idx]
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Filter outliers
        filtered = [v for v in values if lower_bound <= v <= upper_bound]
        
        # Fallback if too few remain
        if len(filtered) < 2:
            return statistics.median(values)
        
        return statistics.median(filtered)

    @staticmethod
    def _distance(a: Point, b: Point) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])
