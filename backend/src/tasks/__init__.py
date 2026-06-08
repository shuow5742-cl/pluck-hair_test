"""Task layer package (business tasks + internal algorithms)."""

from autoweaver.tasks import (
    AlwaysFalseCondition,
    DoneCondition,
    SideTask,
    Task,
    TaskBase,
)
from .detection import DetectionTask
from .factory import create_task
from .seg_pick import SegPickTask
from .seg_pick_stabilized import SegPickStabilizedTask
from .stabilized_detection import PickTask, StabilizedDetectionTask
from .stabilized_detection.pick_process import (
    Phase,
    PickProcess,
    PickProcessConfig,
    TargetState,
    TrackedTarget,
    TrackStats,
)
from .stabilized_detection.stabilizer import (
    ClusterState,
    StableTarget,
    Stabilizer,
    StabilizerConfig,
    TargetCluster,
)
from .stats import TaskStats

__all__ = [
    "TaskBase",
    "SideTask",
    "Task",
    "DoneCondition",
    "AlwaysFalseCondition",
    "TaskStats",
    "DetectionTask",
    "StabilizedDetectionTask",
    "SegPickTask",
    "SegPickStabilizedTask",
    "create_task",
    "PickTask",
    # Stabilizer (technical domain)
    "ClusterState",
    "StableTarget",
    "Stabilizer",
    "StabilizerConfig",
    "TargetCluster",
    # PickProcess (business domain)
    "Phase",
    "PickProcess",
    "PickProcessConfig",
    "TargetState",
    "TrackedTarget",
    "TrackStats",
]
