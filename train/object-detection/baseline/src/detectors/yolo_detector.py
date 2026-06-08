from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np
from ultralytics import YOLO

from src.core.dtypes import DetectionResult
from src.core.interfaces import DetectorInterface
from src.core.registry import DETECTORS


@DETECTORS.register("yolo")
class YoloDetector(DetectorInterface):
    """Ultralytics YOLO 包装器。"""

    default_runner = "yolo"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = self.config.get("model", "yolov8n.pt")
        self.device = self.config.get("device", "auto")

    def build(self) -> None:
        if self.model is None:
            self.model = YOLO(self.model_name)

    def load(self, path: str) -> None:
        self.model = YOLO(path)

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Model not built yet.")
        self.model.save(path)

    def train(self, **kwargs) -> Any:
        self.build()
        train_args = self._prepare_train_args(kwargs)
        return self.model.train(**train_args)

    def predict(self, source: Any, **kwargs) -> Iterable[DetectionResult]:
        self.build()
        predictions = self.model.predict(source=source, **kwargs)
        return [self._to_detection_result(res) for res in predictions]

    def _prepare_train_args(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        train_args = dict(self.config.get("train_kwargs", {}))
        train_args.update(overrides)
        data_yaml = train_args.pop("data_yaml", None) or self.config.get("data_yaml")
        if not data_yaml:
            raise ValueError("data_yaml must be specified for YOLO training.")
        train_args["data"] = data_yaml
        if "device" not in train_args and self.device:
            train_args["device"] = self.device
        return train_args

    @staticmethod
    def _to_detection_result(result: Any) -> DetectionResult:
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else np.empty((0, 4))
        scores = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.empty((0,))
        labels = (
            result.boxes.cls.cpu().numpy().astype(int)
            if result.boxes.cls is not None
            else np.empty((0,), dtype=int)
        )
        image_path = getattr(result, "path", None)
        return DetectionResult(boxes=boxes, scores=scores, labels=labels, image_path=image_path)
