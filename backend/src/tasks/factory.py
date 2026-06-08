"""Task factory for scheduler selection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from autoweaver.tasks import Task
from src.config import TaskConfig
from src.core.target_converter import TargetConverter
from autoweaver.pipeline import VisionPipeline
from .detection import DetectionTask
from .seg_pick import SegPickTask
from .seg_pick_stabilized import SegPickStabilizedTask
from .stabilized_detection import StabilizedDetectionTask
from .stabilized_detection.pick_process import PickProcessConfig
from .stabilized_detection.stabilizer import StabilizerConfig

logger = logging.getLogger(__name__)

_TASK_ALIASES = {
    "stabilized": "stabilized_detection",
}


def _try_load_target_converter(config: TaskConfig) -> Optional[TargetConverter]:
    """Load TargetConverter from calibration YAML files if configured.

    Raises if calibration paths are configured but loading fails — this is
    a hard requirement, not optional.
    """
    cal = config.calibration or {}
    extrinsic = cal.get("extrinsic_path")
    intrinsic = cal.get("intrinsic_path")
    if not extrinsic or not intrinsic:
        return None
    if not Path(extrinsic).exists():
        raise FileNotFoundError(
            f"Extrinsic calibration not found: {extrinsic}. "
            "Calibration is configured but the file is missing."
        )
    if not Path(intrinsic).exists():
        raise FileNotFoundError(
            f"Intrinsic calibration not found: {intrinsic}. "
            "Calibration is configured but the file is missing."
        )
    converter = TargetConverter.from_yaml(extrinsic, intrinsic)
    logger.info("Coordinate transform enabled (extrinsic=%s)", extrinsic)
    return converter


def create_task(
    pipeline: VisionPipeline,
    task_config: Optional[TaskConfig] = None,
) -> Task:
    config = task_config or TaskConfig()
    task_type = (config.type or "").strip().lower() or "stabilized_detection"
    task_type = _TASK_ALIASES.get(task_type, task_type)
    name = (config.name or "").strip() or None

    if config.stabilizer and task_type not in {"stabilized_detection", "seg_pick_stabilized"}:
        logger.warning(
            "Ignoring scheduler.task.stabilizer for task type '%s'",
            task_type,
        )

    if config.pick_process and task_type != "stabilized_detection":
        logger.warning(
            "Ignoring scheduler.task.pick_process for task type '%s'",
            task_type,
        )

    if task_type == "detection":
        kwargs = {"pipeline": pipeline}
        if name:
            kwargs["name"] = name
        return DetectionTask(**kwargs)

    if task_type == "seg_pick":
        kwargs = {"pipeline": pipeline}
        if name:
            kwargs["name"] = name
        return SegPickTask(**kwargs)

    if task_type == "seg_pick_stabilized":
        stabilizer_config = None
        if config.stabilizer:
            if not isinstance(config.stabilizer, dict):
                raise TypeError("scheduler.task.stabilizer must be a mapping")
            stabilizer_config = StabilizerConfig(**config.stabilizer)

        kwargs = {
            "pipeline": pipeline,
            "stabilizer_config": stabilizer_config,
        }
        if name:
            kwargs["name"] = name
        return SegPickStabilizedTask(**kwargs)

    if task_type == "stabilized_detection":
        stabilizer_config = None
        if config.stabilizer:
            if not isinstance(config.stabilizer, dict):
                raise TypeError("scheduler.task.stabilizer must be a mapping")
            stabilizer_config = StabilizerConfig(**config.stabilizer)

        pick_process_config = None
        if config.pick_process:
            if not isinstance(config.pick_process, dict):
                raise TypeError("scheduler.task.pick_process must be a mapping")
            pick_process_config = PickProcessConfig(**config.pick_process)

        target_converter = _try_load_target_converter(config)

        kwargs = {
            "pipeline": pipeline,
            "stabilizer_config": stabilizer_config,
            "pick_process_config": pick_process_config,
            "target_converter": target_converter,
        }
        if name:
            kwargs["name"] = name
        return StabilizedDetectionTask(**kwargs)

    raise ValueError(
        f"Unknown scheduler.task.type '{config.type}'. "
        "Supported: detection, stabilized_detection, seg_pick, seg_pick_stabilized."
    )
