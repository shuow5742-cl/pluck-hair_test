from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

from src.core.interfaces import DetectorInterface, TrainingCallback


class Runner(ABC):
    name: str

    def __init__(self, **options):
        self.options = options

    @abstractmethod
    def run(
        self,
        detector: DetectorInterface,
        exp_config: Dict,
        callbacks: List[TrainingCallback],
    ) -> Dict:
        ...
