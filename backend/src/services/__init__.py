"""Service layer for business logic.

Services encapsulate business logic and data access,
keeping routes thin and focused on HTTP concerns.
"""

from .detection_service import DetectionService

__all__ = ["DetectionService"]


