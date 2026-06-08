from __future__ import annotations

from typing import Dict, List

import src.detectors  # noqa: F401
import src.runners  # noqa: F401
from src.core.config import load_config
from src.core.interfaces import TrainingCallback
from src.core.registry import DETECTORS, RUNNERS
from src.engine.callbacks.experiment_logger import ExperimentLogger


class Trainer:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = load_config(config_path)
        self.detector = self._build_detector(self.config["detector"])

    def _build_detector(self, detector_cfg: Dict) -> object:
        detector_cls = DETECTORS.get(detector_cfg["type"])
        cfg = dict(detector_cfg.get("config", {}))
        if detector_cfg.get("runner") and "runner" not in cfg:
            cfg["runner"] = detector_cfg["runner"]
        return detector_cls(cfg)

    def _build_callbacks(self) -> List[TrainingCallback]:
        return [ExperimentLogger(self.config["experiment"])]

    def train(self) -> Dict:
        profile = getattr(self.detector, "training_profile", None)
        runner_name = (
            profile.runner
            if profile
            else self.config["detector"].get("runner")
            or self.config["training"].get("runner")
        )
        if not runner_name:
            raise ValueError("Runner not specified.")

        runner_cls = RUNNERS.get(runner_name)
        runner_options = profile.options if profile else {}
        runner = runner_cls(**runner_options)
        callbacks = self._build_callbacks()
        metrics = runner.run(self.detector, self.config, callbacks)
        return metrics
