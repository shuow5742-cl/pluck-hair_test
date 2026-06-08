from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional


@dataclass
class TrainingProfile:
    runner: str
    options: Dict[str, Any] = field(default_factory=dict)


class DetectorInterface(ABC):
    default_runner: Optional[str] = None

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model = None
        self.training_profile: Optional[TrainingProfile] = self._build_training_profile()

    def _build_training_profile(self) -> Optional[TrainingProfile]:
        runner = self.config.get("runner") or self.default_runner
        profile_cfg = self.config.get("training_profile", {})
        if isinstance(profile_cfg, TrainingProfile):
            return profile_cfg
        if runner:
            options = dict(profile_cfg) if isinstance(profile_cfg, dict) else {}
            return TrainingProfile(runner=runner, options=options)
        return None

    def get_runner_name(self) -> Optional[str]:
        if self.training_profile:
            return self.training_profile.runner
        return None

    @abstractmethod
    def build(self) -> None:
        ...

    @abstractmethod
    def train(self, **kwargs) -> Any:
        ...

    @abstractmethod
    def predict(self, source: Any, **kwargs) -> Iterable[Any]:
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        ...


class TrainingCallback:
    """轻量训练回调接口。"""

    def on_train_begin(self, config: Dict[str, Any]) -> None:  # pragma: no cover - hook
        ...

    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]) -> None:  # pragma: no cover - hook
        ...

    def on_train_end(self, metrics: Dict[str, Any]) -> None:  # pragma: no cover - hook
        ...

    def on_framework_output(self, path: str) -> None:  # pragma: no cover - hook
        ...
