"""Storage handling for scheduler.

Encapsulates sync/async save logic, retries, and event publishing.
Keeps WorkflowEngine thin by handling image/detection persistence here.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional

import numpy as np

from autoweaver.pipeline import Detection
from src.storage.interfaces import Database, DetectionRecord, ImageStorage

logger = logging.getLogger(__name__)

# Callback type for publishing detection events after storage succeeds
EventCallback = Callable[[List[DetectionRecord], str, Optional[str], datetime, Optional[dict]], None]


@dataclass
class StorageSaverConfig:
    """Configuration for StorageSaver."""

    save_images: bool = True
    save_annotated: bool = True
    async_storage: bool = True
    storage_workers: int = 4
    max_pending_saves: int = 100
    storage_retry_count: int = 3


class StorageSaver:
    """Handle persistence of images and detection records."""

    def __init__(
        self,
        image_storage: ImageStorage,
        database: Database,
        config: StorageSaverConfig,
        event_callback: Optional[EventCallback] = None,
    ):
        self.image_storage = image_storage
        self.database = database
        self.config = config
        self.event_callback = event_callback

        self._executor: Optional[ThreadPoolExecutor] = None
        self._pending_futures: List[Future] = []
        self._lock = threading.Lock()
        self._errors = 0

    @property
    def storage_errors(self) -> int:
        """Number of storage failures."""
        return self._errors

    def start(self):
        """Initialize async executor if needed."""
        if self.config.async_storage:
            self._executor = ThreadPoolExecutor(
                max_workers=self.config.storage_workers,
                thread_name_prefix="storage",
            )
            logger.info("StorageSaver async mode enabled with %s workers", self.config.storage_workers)

    def save(
        self,
        image: np.ndarray,
        image_path: str,
        detections: List[Detection],
        timestamp: datetime,
        annotated_image: Optional[np.ndarray],
        annotated_path: Optional[str],
        session_id: Optional[str],
        event_metadata: Optional[dict] = None,
    ):
        """Persist image/detections, optionally async with retries."""
        if self.config.async_storage and self._executor:
            self._save_async(
                image,
                image_path,
                detections,
                timestamp,
                annotated_image,
                annotated_path,
                session_id,
                event_metadata,
            )
        else:
            self._save_sync(
                image,
                image_path,
                detections,
                timestamp,
                annotated_image,
                annotated_path,
                session_id,
                event_metadata,
            )

    def _save_sync(
        self,
        image: np.ndarray,
        image_path: str,
        detections: List[Detection],
        timestamp: datetime,
        annotated_image: Optional[np.ndarray],
        annotated_path: Optional[str],
        session_id: Optional[str],
        event_metadata: Optional[dict],
    ):
        """Save results synchronously (blocking)."""
        full_path = image_path
        if self.config.save_images:
            full_path = self.image_storage.save(image, image_path)

        if detections:
            records = [
                self._to_detection_record(det, full_path, timestamp, session_id)
                for det in detections
            ]
            self.database.save_detections_batch(records)
            if self.event_callback:
                self.event_callback(records, full_path, annotated_path, timestamp, event_metadata)

        if (
            self.config.save_images
            and self.config.save_annotated
            and annotated_image is not None
            and annotated_path
        ):
            self.image_storage.save(annotated_image, annotated_path)

    def _save_async(
        self,
        image: np.ndarray,
        image_path: str,
        detections: List[Detection],
        timestamp: datetime,
        annotated_image: Optional[np.ndarray],
        annotated_path: Optional[str],
        session_id: Optional[str],
        event_metadata: Optional[dict],
    ):
        """Save results asynchronously (non-blocking)."""
        with self._lock:
            pending_count = len([f for f in self._pending_futures if not f.done()])

        if pending_count >= self.config.max_pending_saves:
            logger.warning(
                "Storage queue full (%s pending), falling back to sync save",
                pending_count,
            )
            self._save_sync(
                image,
                image_path,
                detections,
                timestamp,
                annotated_image,
                annotated_path,
                session_id,
                event_metadata,
            )
            return

        future = self._executor.submit(
            self._save_with_retry,
            image.copy(),
            image_path,
            detections,
            timestamp,
            annotated_image.copy() if annotated_image is not None else None,
            annotated_path,
            session_id,
            dict(event_metadata) if event_metadata else None,
        )

        with self._lock:
            self._pending_futures.append(future)

    def _save_with_retry(
        self,
        image: np.ndarray,
        image_path: str,
        detections: List[Detection],
        timestamp: datetime,
        annotated_image: Optional[np.ndarray],
        annotated_path: Optional[str],
        session_id: Optional[str],
        event_metadata: Optional[dict],
    ):
        """Save with retry logic for resilience."""
        last_error = None

        for attempt in range(self.config.storage_retry_count):
            try:
                self._save_sync(
                    image,
                    image_path,
                    detections,
                    timestamp,
                    annotated_image,
                    annotated_path,
                    session_id,
                    event_metadata,
                )
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "Storage attempt %s/%s failed: %s",
                    attempt + 1,
                    self.config.storage_retry_count,
                    exc,
                )
                if attempt < self.config.storage_retry_count - 1:
                    time.sleep(0.5 * (attempt + 1))

        with self._lock:
            self._errors += 1
        logger.error(
            "Storage failed after %s attempts: %s",
            self.config.storage_retry_count,
            last_error,
        )

    def cleanup_futures(self):
        """Clean up completed futures and check for errors."""
        with self._lock:
            still_pending: List[Future] = []
            for future in self._pending_futures:
                if future.done():
                    try:
                        future.result()
                    except Exception as exc:  # noqa: BLE001
                        logger.error("Async storage task failed: %s", exc)
                else:
                    still_pending.append(future)
            self._pending_futures = still_pending

    def shutdown(self):
        """Shutdown executor and wait for pending tasks."""
        if self._executor:
            pending_count = len([f for f in self._pending_futures if not f.done()])
            if pending_count > 0:
                logger.info("Waiting for %s pending storage operations...", pending_count)

            for future in self._pending_futures:
                try:
                    future.result(timeout=30)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Pending storage task failed: %s", exc)

            self._executor.shutdown(wait=True)
            self._executor = None
            logger.info("StorageSaver executor shut down")

    @staticmethod
    def _to_detection_record(
        detection: Detection,
        image_path: str,
        timestamp: datetime,
        session_id: Optional[str],
    ) -> DetectionRecord:
        """Convert Detection to DetectionRecord."""
        obj_type = (
            detection.object_type
            if isinstance(detection.object_type, str)
            else detection.object_type.value
        )
        return DetectionRecord(
            id=str(uuid.uuid4()),
            image_path=image_path,
            bbox_x1=detection.bbox.x1,
            bbox_y1=detection.bbox.y1,
            bbox_x2=detection.bbox.x2,
            bbox_y2=detection.bbox.y2,
            object_type=obj_type,
            confidence=detection.confidence,
            created_at=timestamp,
            session_id=session_id,
        )
