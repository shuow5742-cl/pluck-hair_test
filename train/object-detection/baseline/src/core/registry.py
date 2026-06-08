from __future__ import annotations

from typing import Any, Callable, Dict, Optional, TypeVar


T = TypeVar("T")


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._modules: Dict[str, Callable[..., Any]] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}

    def register(self, name: Optional[str] = None) -> Callable[[T], T]:
        def decorator(obj: T) -> T:
            key = name or getattr(obj, "__name__", None)
            if not key:
                raise ValueError("Registered object must have a name")
            if key in self._modules or key in self._factories:
                raise ValueError(f"{key} already registered in {self.name}")
            self._modules[key] = obj
            return obj

        return decorator

    def register_lazy(self, name: str, factory: Callable[[], Any]) -> None:
        if name in self._modules or name in self._factories:
            raise ValueError(f"{name} already registered in {self.name}")
        self._factories[name] = factory

    def get(self, name: str) -> Callable[..., Any]:
        if name in self._modules:
            return self._modules[name]
        if name in self._factories:
            module = self._factories.pop(name)()
            self._modules[name] = module
            return module
        raise KeyError(f"{name} not found in registry '{self.name}'")

    def available(self) -> Dict[str, Callable[..., Any]]:
        return {**self._modules, **self._factories}


DETECTORS = Registry("detectors")
RUNNERS = Registry("runners")
POSTPROCESSORS = Registry("postprocessors")
