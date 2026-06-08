"""Frame loop side-task — drives the camera capture + task execution cycle."""

from __future__ import annotations

import http.client
import logging
import math
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
from urllib.parse import urlsplit

import cv2

from autoweaver.camera import CameraBase
from autoweaver.reactive import EventBus
from autoweaver.tasks import Task

from src.config import HttpImageUploadConfig, VideoStreamConfig
from src.workflow.event_builder import EventBuilder, EventContext
from src.workflow.frame_renderer import FrameRenderer
from src.workflow.frame_streamer import FrameStreamer

logger = logging.getLogger(__name__)


@dataclass
class FrameLoopConfig:
    """Configuration for FrameLoopSideTask."""
    loop_delay_ms: int = 100
    max_errors: int = 10
    show_preview: bool = True
    # pluck-hair_test: optional classical-CV tweezer-tip overlay. When set, the
    # tip cross + tip→pick distance are drawn on every streamed live frame.
    tweezer_config: Any = None  # Optional[TweezerConfig]
    http_image_upload: HttpImageUploadConfig = field(
        default_factory=HttpImageUploadConfig
    )
    annotation_color_map: dict = field(default_factory=lambda: {
        "hair": (255, 0, 0),
        "black_spot": (0, 0, 255),
        "yellow_spot": (0, 255, 255),
        "unknown": (255, 0, 0),
    })


@dataclass
class _TaskFrameResult:
    """Normalized view of one task iteration output."""
    detections: list = field(default_factory=list)
    stable_targets: list = field(default_factory=list)
    tracked_targets: list = field(default_factory=list)
    is_done: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class _UploadJob:
    """One HTTP upload request for the upper-machine display."""

    image_bytes: bytes
    photo_key: str
    timestamp: str
    robot: str
    position: str
    filename: str
    content_type: str


class _HttpImageUploader:
    """Best-effort async multipart uploader for one-shot recognition images."""

    def __init__(self, config: HttpImageUploadConfig) -> None:
        self._config = config
        self._queue: queue.Queue[_UploadJob] = queue.Queue(
            maxsize=max(1, int(config.queue_size))
        )
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._url = urlsplit(config.url)

    def start(self) -> None:
        if not self._config.enabled:
            logger.info("HTTP image upload disabled by config")
            return
        if self._thread is not None:
            logger.info("HTTP image upload worker already running")
            return
        if self._url.scheme not in {"http", "https"} or not self._url.netloc:
            logger.warning(
                "HTTP image upload disabled: invalid url %r", self._config.url
            )
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="http-image-upload",
        )
        self._thread.start()
        logger.info(
            "HTTP image upload enabled: url=%s robot=%s queue_size=%d timeout=(%.1fs, %.1fs) retry=%d",
            self._config.url,
            self._config.robot,
            max(1, int(self._config.queue_size)),
            float(self._config.connect_timeout_s),
            float(self._config.upload_timeout_s),
            int(self._config.retry_count),
        )

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
            logger.info("HTTP image upload worker stopped")

    def enqueue(self, job: _UploadJob) -> None:
        if not self._config.enabled:
            logger.warning(
                "Skip HTTP image upload enqueue because feature is disabled: photo_key=%s",
                job.photo_key,
            )
            return
        if self._thread is None:
            logger.warning(
                "Skip HTTP image upload enqueue because worker is not running: photo_key=%s url=%s",
                job.photo_key,
                self._config.url,
            )
            return
        try:
            self._queue.put_nowait(job)
            logger.info(
                "HTTP image upload enqueued: photo_key=%s position=%s bytes=%d pending=%d",
                job.photo_key,
                job.position,
                len(job.image_bytes),
                self._queue.qsize(),
            )
        except queue.Full:
            logger.warning(
                "HTTP image upload queue full, dropping frame for photo_key=%s",
                job.photo_key,
            )

    def _run(self) -> None:
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._upload_with_retry(job)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "HTTP image upload failed for photo_key=%s: %s",
                    job.photo_key, exc,
                )
            finally:
                self._queue.task_done()

    def _upload_with_retry(self, job: _UploadJob) -> None:
        attempts = max(1, int(self._config.retry_count) + 1)
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                logger.info(
                    "HTTP image upload sending: attempt=%d/%d photo_key=%s position=%s url=%s",
                    attempt,
                    attempts,
                    job.photo_key,
                    job.position,
                    self._config.url,
                )
                self._post_multipart(job)
                logger.info(
                    "HTTP image upload succeeded: photo_key=%s position=%s",
                    job.photo_key,
                    job.position,
                )
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= attempts:
                    break
                logger.warning(
                    "HTTP image upload retry %d/%d for photo_key=%s after error: %s",
                    attempt,
                    attempts - 1,
                    job.photo_key,
                    exc,
                )
        if last_error is not None:
            raise last_error

    def _post_multipart(self, job: _UploadJob) -> None:
        boundary = f"pluck-{uuid.uuid4().hex}"
        body = _build_multipart_body(boundary, job)
        path = self._url.path or "/"
        if self._url.query:
            path = f"{path}?{self._url.query}"

        connection_cls = (
            http.client.HTTPSConnection
            if self._url.scheme == "https"
            else http.client.HTTPConnection
        )
        conn = connection_cls(
            self._url.hostname,
            self._url.port,
            timeout=float(self._config.connect_timeout_s),
        )
        try:
            conn.connect()
            if conn.sock is not None:
                conn.sock.settimeout(float(self._config.upload_timeout_s))
            conn.request(
                "POST",
                path,
                body=body,
                headers={
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                    "Content-Length": str(len(body)),
                },
            )
            response = conn.getresponse()
            response_body = response.read().decode("utf-8", errors="replace")
            if response.status < 200 or response.status >= 300:
                raise RuntimeError(
                    f"status={response.status}, body={response_body[:200]}"
                )
            logger.info(
                "HTTP image upload response: photo_key=%s status=%d body=%s",
                job.photo_key,
                response.status,
                response_body[:200],
            )
        finally:
            conn.close()


class FrameLoopSideTask:
    """SideTask that runs a camera capture + task execution loop.

    Implements the SideTask protocol: attach(event_bus) / close().
    Internally runs a thread that captures frames, runs the current
    task, renders preview, and streams frames.
    """

    name = "frame_loop"

    def __init__(
        self,
        camera: CameraBase,
        task_map: Dict[str, Task],
        config: Optional[FrameLoopConfig] = None,
        event_publisher=None,
        frame_publisher=None,
        video_stream_config=None,
    ) -> None:
        self._camera = camera
        self._task_map = task_map
        self._config = config or FrameLoopConfig()
        self._event_publisher = event_publisher
        self._event_bus: Optional[EventBus] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._current_task: Optional[Task] = None
        self._handoff: dict = {}
        self._pending_task_input: Optional[Any] = None
        self._resume_gate = threading.Event()
        self._resume_gate.set()
        self._is_paused = False
        self._frame_count = 0
        self._total_detections = 0
        self._error_count = 0
        self._last_frame_at: Optional[float] = None
        # Diagnostic snapshot: one annotated frame per RESUME (= per nova5
        # photo position). Set on RESUME, cleared after the first
        # post-RESUME _process_frame call writes the file. To disable
        # the feature, drop the `_snapshot_*` block in `_process_frame`
        # and the helper import — no protocol or config changes needed.
        self._snapshot_pending = False
        self._snapshot_label: Optional[str] = None
        self._capture_session_dir = _create_capture_session_dir()
        # Latest flange pose published by PixelToWorldTask alongside the
        # current frame's world picks. Snapshot reads it for the banner
        # overlay so a still frame is self-contained for audit.
        self._latest_flange_pose_mm: Optional[tuple[float, float]] = None
        self._latest_epson_target_overlay: Optional[dict[str, Any]] = None
        self._preview_tracking_radius_px: float = 30.0

        # AR-style preview state machine (active only when show_preview=True):
        #   LIVE     — camera passthrough (initial / between photo positions)
        #   FROZEN   — annotated still frame, shown for one tick after the
        #              first pipeline run at a new photo position
        #   OVERLAY  — live camera with frozen annotations re-pasted every
        #              frame, so the operator can watch the pick arm enter
        #              the scene against the original pick targets.
        # Triggers (event-driven, no extra config):
        #   RESUME (photo arrived) → arm pipeline; the next _process_frame
        #                            tick runs yolo_seg ONCE, caches the
        #                            result, and enters FROZEN. Subsequent
        #                            ticks at this photo position skip the
        #                            pipeline (GPU idle) and render OVERLAY.
        #   PAUSE  (nova5 leaving) → drop frozen data, return to LIVE.
        # PAUSE no longer fully blocks the loop — it suppresses pipeline
        # execution but keeps capture+preview running so OVERLAY can render.
        self._ar_state: str = "LIVE"
        self._ar_frozen_image = None
        self._ar_frozen_dets: list = []
        self._ar_frozen_preview_only_dets: list = []
        self._ar_frozen_flange: Optional[tuple[float, float]] = None
        # AR overlay decorations baked from the same pipeline run as the
        # frozen detections, so the preview can show "what crop_single_square
        # locked onto" + "where the abstain band is" without re-reading
        # configs at render time.
        self._ar_frozen_cell_box: Optional[tuple[int, int, int, int]] = None
        self._ar_frozen_safety_margin_px: Optional[float] = None
        # Number of pipeline runs still allowed at the current photo
        # position. Default tasks use 1; seg_pick_stabilized overrides to
        # the configured multi-frame window so the same parked pose can be
        # evaluated across N consecutive captures before we lock a batch.
        self._pipeline_runs_remaining: int = 0

        self._event_builder = EventBuilder()
        self._frame_renderer = FrameRenderer(self._config.annotation_color_map)
        self._frame_streamer = FrameStreamer(frame_publisher, video_stream_config)
        self._http_image_uploader = _HttpImageUploader(self._config.http_image_upload)
        logger.info("Per-photo crop images will be saved to %s", self._capture_session_dir)

        # pluck-hair_test: tweezer-tip detector (classical CV) + live-state bus.
        # Both are no-ops in the production run/api modes (tweezer_config=None);
        # the live-state bus is a cheap process-global singleton either way.
        self._tweezer = None
        tcfg = self._config.tweezer_config
        if tcfg is not None and getattr(tcfg, "enabled", True):
            from src.core.tweezer_detector import TweezerDetector
            self._tweezer = TweezerDetector(tcfg)
            logger.info("Tweezer overlay enabled (entry_side=%s)", tcfg.entry_side)
        from src.comm.inproc_bus import get_live_state_bus
        self._live_state = get_live_state_bus()
        self._latest_tweezer = None
        self._latest_tip_to_pick_mm: Optional[float] = None

    def attach(self, event_bus: EventBus) -> None:
        """Inject EventBus, subscribe to events, open camera, start loop thread."""
        self._event_bus = event_bus
        event_bus.subscribe("STATE:CHANGED", self._on_state_changed)
        event_bus.subscribe("TASK:ITERATION", self._on_task_result)
        event_bus.subscribe("TASK:DONE", self._on_task_result)
        event_bus.subscribe("TASK:PICK_RESULT", self._on_task_result)
        event_bus.subscribe("TASK:PICK_RESULT", self._on_pick_result)
        # TASK:WORLD_PICKS is the only event carrying flange pose for the
        # current frame. We cache it for the snapshot helper, not for
        # control flow.
        event_bus.subscribe("TASK:WORLD_PICKS", self._on_world_picks)
        event_bus.subscribe("PLC_ORCH:EPSON_TARGET_SENT", self._on_epson_target_sent)
        event_bus.subscribe("PLC_ORCH:PREVIEW_CLEAR", self._on_preview_clear)
        event_bus.subscribe("FRAME_LOOP:PAUSE", self._on_pause_requested)
        event_bus.subscribe("FRAME_LOOP:RESUME", self._on_resume_requested)

        # Open camera
        if not self._camera.open():
            raise RuntimeError("Failed to open camera")

        # Resolve initial task from current state (engine already set up state machine)
        # We need to listen to STATE:CHANGED to track current task
        # The initial task is set by the first STATE:CHANGED or we peek at state machine
        # For now, we don't directly access state machine — we listen to events

        self._http_image_uploader.start()
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="frame-loop"
        )
        self._thread.start()

    def close(self) -> None:
        """Stop loop, close camera, clean up."""
        self._running = False
        self._resume_gate.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        # Close OpenCV windows
        if self._config.show_preview:
            cv2.destroyAllWindows()

        self._http_image_uploader.close()
        try:
            self._camera.close()
        except Exception as e:
            logger.warning("Error closing camera: %s", e)

        self._event_bus = None

    def _loop(self) -> None:
        """Main frame processing loop.

        PAUSE no longer blocks (used to gate on _resume_gate.wait()) —
        instead, _process_frame branches: when paused, it captures a
        frame and renders the AR preview without invoking the pipeline.
        That keeps OVERLAY mode responsive while nova5 is in transit,
        but still satisfies the original PAUSE contract ("don't run
        pipeline / publish events on stale frames").
        """
        while self._running:
            loop_start = time.time()

            try:
                self._process_frame()
                self._error_count = 0
            except Exception as e:
                self._error_count += 1
                logger.error(
                    "Error processing frame (attempt %d): %s",
                    self._error_count, e, exc_info=True
                )
                if self._error_count >= self._config.max_errors:
                    logger.critical(
                        "Max consecutive errors (%d) reached, stopping frame loop",
                        self._config.max_errors,
                    )
                    self._running = False
                    break

            elapsed_ms = (time.time() - loop_start) * 1000
            remaining_ms = self._config.loop_delay_ms - elapsed_ms
            if remaining_ms > 0:
                time.sleep(remaining_ms / 1000.0)

    def _process_frame(self) -> None:
        """Process a single frame.

        Two non-pipeline branches keep GPU idle when nothing changed:
        - paused (nova5 in transit): capture+preview only, no pipeline,
          no event publish, no snapshot.
        - resumed but pipeline already ran for this photo position:
          capture+preview only, OVERLAY render against frozen dets.
        Pipeline runs while _pipeline_runs_remaining > 0.
        """
        if self._current_task is None:
            return

        if self._is_paused:
            try:
                paused_image = self._camera.capture()
            except Exception:  # noqa: BLE001
                return
            self._render_ar_preview(paused_image, dets=None)
            return

        if self._pipeline_runs_remaining <= 0:
            # Resumed, pipeline already fired once for this photo position.
            # Keep the preview animating (OVERLAY) but do not re-run yolo_seg.
            try:
                idle_image = self._camera.capture()
            except Exception:  # noqa: BLE001
                return
            self._render_ar_preview(idle_image, dets=None)
            return

        region_start = time.perf_counter()
        timestamp = datetime.now()
        now_monotonic = time.monotonic()
        capture_start = time.perf_counter()
        capture_batch = self._capture_pipeline_batch(self._pipeline_runs_remaining)
        capture_end = time.perf_counter()
        if not capture_batch:
            return
        capture_batch_ms = round((capture_end - capture_start) * 1000.0, 2)
        task_name = self._current_task.name
        result = self._build_task_frame_result()
        image = capture_batch[-1]
        task_run_rows: list[dict[str, float | int]] = []
        for captured_image in capture_batch:
            self._handoff.pop("TASK:ITERATION", None)
            self._handoff.pop("TASK:DONE", None)
            self._frame_count += 1
            task_input = (
                captured_image
                if self._pending_task_input is None
                else self._pending_task_input
            )
            self._pending_task_input = None
            run_start = time.perf_counter()
            self._current_task.run(task_input)
            run_end = time.perf_counter()
            # Consume one run budget before downstream work so an exception
            # cannot accidentally duplicate a cycle frame on the next tick.
            self._pipeline_runs_remaining = max(0, self._pipeline_runs_remaining - 1)
            result = self._build_task_frame_result()
            image = captured_image
            task_run_rows.append(
                {
                    "frame": len(task_run_rows) + 1,
                    "task_run_ms": round((run_end - run_start) * 1000.0, 2),
                }
            )
        region_total_ms = round((time.perf_counter() - region_start) * 1000.0, 2)
        result.metadata["capture_batch_ms"] = capture_batch_ms
        result.metadata["frame_task_run_rows"] = task_run_rows
        result.metadata["frame_task_run_total_ms"] = round(
            sum(float(row["task_run_ms"]) for row in task_run_rows),
            2,
        )
        result.metadata["region_total_ms"] = region_total_ms
        fps = 0.0
        if self._last_frame_at is not None:
            delta = now_monotonic - self._last_frame_at
            if delta > 0:
                fps = 1.0 / delta
        self._last_frame_at = now_monotonic

        self._total_detections += len(result.detections)

        self._log_region_timing_summary(task_name, result)

        if self._event_publisher:
            context = EventContext(
                session_id=None,
                frame=self._frame_count,
                total_detections=self._total_detections,
            )
            event_metadata = dict(result.metadata)
            event_metadata["fps"] = round(fps, 2)
            payload = self._event_builder.build_live_detection_event(
                detections=result.detections,
                tracked_targets=result.tracked_targets,
                timestamp=timestamp,
                context=context,
                event_metadata=event_metadata,
            )
            try:
                self._event_publisher.publish(payload)
            except Exception as exc:
                logger.warning("Failed to publish detection event: %s", exc)

        # Preview + stream — AR state machine (LIVE / FROZEN / OVERLAY). On the
        # first post-RESUME pipeline run, _render_ar_preview takes the
        # current frame + dets as the new "frozen" snapshot and shows it
        # one tick as FROZEN. Subsequent paused ticks render OVERLAY.
        # _render_ar_preview now also draws the tweezer overlay and publishes
        # the annotated frame to the web stream itself (every tick), so there is
        # no separate build_stream_frame/publish here anymore.
        self._update_live_stats(result, fps)
        self._render_ar_preview(
            image, dets=result.detections, metadata=result.metadata,
            timestamp=timestamp,
        )

        # Handle task completion
        if result.is_done:
            logger.info(
                "Task '%s' reported completion after %s frames",
                task_name, self._frame_count,
            )

        # Diagnostic snapshot — annotated PNG + JSON sidecar to
        # data/machine_result/, one per nova5 RESUME (= per photo
        # position). Failure swallowed inside the helper; this block
        # can be removed wholesale to retire the feature.
        if self._snapshot_pending and self._is_snapshot_result_ready(result):
            snapshot_detections = _merge_preview_detections(
                result.detections,
                _preview_only_detections_from_metadata(result.metadata),
            )
            logger.info(
                "Upper-machine upload hook triggered: snapshot_label=%s detections=%d frame=%d",
                self._snapshot_label,
                len(snapshot_detections),
                self._frame_count,
            )
            self._save_photo_position_image(image, result.metadata)
            self._upload_upper_machine_image(
                image=image,
                detections=result.detections,
                metadata=result.metadata,
                timestamp=timestamp,
            )
            try:
                from tools.machine_result_snapshot import try_save_machine_result
                try_save_machine_result(
                    frame_bgr=image,
                    detections=snapshot_detections,
                    output_dir="data/machine_result",
                    photo_key=self._snapshot_label,
                    seg_frame_id=str(result.metadata.get("seg_frame_id") or self._frame_count),
                    flange_pose_mm=self._latest_flange_pose_mm,
                    cell_box_xyxy=_cell_box_from_metadata(result.metadata),
                    safety_margin_px=_safety_margin_from_metadata(result.metadata),
                )
            except Exception as exc:  # noqa: BLE001 — diagnostic only
                logger.warning("machine_result hook failed: %s", exc)
            finally:
                self._snapshot_pending = False
                self._snapshot_label = None

    def _save_photo_position_image(self, image, metadata: dict) -> None:
        """Save exactly one crop-or-original image for the current photo position."""
        photo_key = self._snapshot_label or f"frame_{self._frame_count:06d}"
        output_path = self._capture_session_dir / f"photo_{photo_key}.png"
        if output_path.exists():
            return

        save_image = image
        crop_meta = metadata.get("crop_single_square") if isinstance(metadata, dict) else None
        if isinstance(crop_meta, dict) and crop_meta.get("source") != "geometric_fallback":
            box = crop_meta.get("box_xyxy_in_original")
            cropped = _crop_from_original_box(image, box)
            if cropped is not None:
                save_image = cropped

        try:
            if not cv2.imwrite(str(output_path), save_image):
                logger.warning("Failed to save per-photo image: %s", output_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Per-photo image save failed (%s): %s", output_path, exc)

    def _is_snapshot_result_ready(self, result: _TaskFrameResult) -> bool:
        """Upload only the final recognition image for the current photo position.

        For seg_pick_stabilized this means waiting until the last frame of the
        configured multi-frame window. For single-frame tasks, metadata usually
        arrives immediately and this returns True on that first frame.
        """
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        if "resume_cycle_final" in metadata:
            return bool(metadata.get("resume_cycle_final"))
        if metadata:
            return True
        return False

    def _capture_pipeline_batch(self, count: int) -> list:
        """Capture the full resume window up front, before image processing."""
        batch_size = max(1, int(count))
        return [self._camera.capture() for _ in range(batch_size)]

    def _log_region_timing_summary(
        self,
        task_name: str,
        result: _TaskFrameResult,
    ) -> None:
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        if not metadata:
            return
        photo_key = self._snapshot_label or "unknown"
        logger.info(
            "%s timing summary photo=%s capture_batch=%.2fms yolo_total=%.2fms postprocess_total=%.2fms pick_total=%.2fms pick_metal_detect=%.2fms pick_dark_line=%.2fms pick_density=%.2fms pick_straight=%.2fms pick_curved=%.2fms pick_distance=%.2fms pipeline_total=%.2fms task_run_total=%.2fms region_total=%.2fms dets=%d rows=%s task_rows=%s",
            task_name,
            photo_key,
            float(metadata.get("capture_batch_ms", 0.0) or 0.0),
            float(metadata.get("cycle_yolo_total_ms", 0.0) or 0.0),
            float(metadata.get("cycle_postprocess_total_ms", 0.0) or 0.0),
            float(metadata.get("cycle_pick_total_ms", 0.0) or 0.0),
            float(metadata.get("cycle_pick_metal_detect_total_ms", 0.0) or 0.0),
            float(metadata.get("cycle_pick_dark_line_total_ms", 0.0) or 0.0),
            float(metadata.get("cycle_pick_density_total_ms", 0.0) or 0.0),
            float(metadata.get("cycle_pick_straight_total_ms", 0.0) or 0.0),
            float(metadata.get("cycle_pick_curved_total_ms", 0.0) or 0.0),
            float(metadata.get("cycle_pick_distance_total_ms", 0.0) or 0.0),
            float(metadata.get("cycle_pipeline_total_ms", 0.0) or 0.0),
            float(metadata.get("frame_task_run_total_ms", 0.0) or 0.0),
            float(metadata.get("region_total_ms", 0.0) or 0.0),
            len(result.detections),
            metadata.get("cycle_timing_rows", []),
            metadata.get("frame_task_run_rows", []),
        )

    def _upload_upper_machine_image(
        self,
        *,
        image,
        detections: Sequence[Any],
        metadata: Optional[dict],
        timestamp: datetime,
    ) -> None:
        position = self._snapshot_label or f"frame_{self._frame_count:06d}"
        upload_key = f"{timestamp:%Y%m%d_%H%M%S}_{position}"
        logger.info(
            "Preparing upper-machine image upload: photo_key=%s position=%s detections=%d",
            upload_key,
            position,
            len(detections),
        )
        upload_image = _build_upper_machine_image(
            image=image,
            detections=_merge_preview_detections(
                detections,
                _preview_only_detections_from_metadata(metadata),
            ),
            flange_pose_mm=self._latest_flange_pose_mm,
            plc_overlay=self._latest_epson_target_overlay,
            tracking_radius_px=self._preview_tracking_radius_px,
        )
        ok, encoded = cv2.imencode(
            ".jpg",
            upload_image,
            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
        )
        if not ok:
            logger.warning(
                "Failed to encode upper-machine image for photo_key=%s", upload_key
            )
            return
        image_bytes = encoded.tobytes()
        logger.info(
            "Upper-machine image encoded: photo_key=%s bytes=%d shape=%s",
            upload_key,
            len(image_bytes),
            tuple(upload_image.shape),
        )
        self._http_image_uploader.enqueue(
            _UploadJob(
                image_bytes=image_bytes,
                photo_key=upload_key,
                timestamp=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                robot=self._config.http_image_upload.robot,
                position=position,
                filename=f"{upload_key}.jpg",
                content_type="image/jpeg",
            )
        )

    def _on_state_changed(self, event: str, data: dict) -> None:
        """Track current task based on state changes."""
        payload = data.get("payload", {})
        new_state = payload.get("new_state")
        self._current_task = self._task_map.get(new_state)
        # Handle handoff
        if self._current_task is not None:
            if self._handoff and getattr(self._current_task, "accepts_handoff", False):
                self._pending_task_input = dict(self._handoff)
            else:
                self._pending_task_input = None
        else:
            self._pending_task_input = None
            if new_state == "idle":
                self._handoff.clear()

    def _on_task_result(self, event: str, data: dict) -> None:
        self._handoff[event] = data

    def _on_world_picks(self, _event: str, data: dict) -> None:
        """Cache flange pose published alongside this frame's world picks.

        Snapshot uses it for the top-left banner. Payload pose is
        ``[x, y, z]``; we keep just the XY pair the helper expects.
        """
        payload = data.get("payload") or {}
        pose = payload.get("flange_pose_mm")
        if pose and len(pose) >= 2:
            self._latest_flange_pose_mm = (float(pose[0]), float(pose[1]))
        else:
            self._latest_flange_pose_mm = None

    def _on_pick_result(self, event: str, data: dict) -> None:
        """Forward pick result to external publisher."""
        pick_result = data.get("payload", {}).get("pick_result")
        if pick_result is None or not self._event_publisher:
            return
        context = EventContext(
            session_id=None,
            frame=self._frame_count,
            total_detections=self._total_detections,
        )
        payload = self._event_builder.build_pick_result_event(
            pick_result=pick_result,
            context=context,
        )
        try:
            self._event_publisher.publish(payload)
        except Exception as exc:
            logger.warning("Failed to publish pick result: %s", exc)

    def _on_epson_target_sent(self, _event: str, data: dict) -> None:
        payload = data.get("payload") or {}
        coord = payload.get("epson_coord") or {}
        if not {"x", "y", "z", "u"} <= set(coord.keys()):
            return
        self._latest_epson_target_overlay = {
            "photo_key": payload.get("photo_key"),
            "detection_id": payload.get("detection_id"),
            "attempt": int(payload.get("attempt", 0) or 0),
            "x": float(coord["x"]),
            "y": float(coord["y"]),
            "z": float(coord["z"]),
            "u": float(coord["u"]),
            "bbox_center_xy_px": payload.get("bbox_center_xy_px"),
            "tracking_radius_px": float(payload.get("tracking_radius_px", 0.0) or 0.0),
        }
        if self._latest_epson_target_overlay["tracking_radius_px"] > 0:
            self._preview_tracking_radius_px = self._latest_epson_target_overlay["tracking_radius_px"]

    def _on_preview_clear(self, _event: str, _data: dict) -> None:
        self._latest_epson_target_overlay = None

    def _on_pause_requested(self, _event: str, data: dict) -> None:
        payload = data.get("payload", {}) if isinstance(data, dict) else {}
        reason = payload.get("reason", "unspecified")
        if self._is_paused:
            return
        self._is_paused = True
        self._resume_gate.clear()
        # nova5 is leaving the photo position — frozen pick targets are
        # no longer relevant for that location. Reset preview to LIVE.
        self._ar_state = "LIVE"
        self._ar_frozen_image = None
        self._ar_frozen_dets = []
        self._ar_frozen_preview_only_dets = []
        self._ar_frozen_flange = None
        self._ar_frozen_cell_box = None
        self._ar_frozen_safety_margin_px = None
        self._pipeline_runs_remaining = 0
        logger.info("Frame loop paused: %s", reason)

    def _on_resume_requested(self, _event: str, data: dict) -> None:
        payload = data.get("payload", {}) if isinstance(data, dict) else {}
        reason = payload.get("reason", "unspecified")
        # Arm the diagnostic snapshot for the next _process_frame call.
        # Reason text is something like "nova5 parked at 1-1" — we slice
        # off the photo key for the filename.
        self._snapshot_pending = True
        self._snapshot_label = _extract_photo_key(reason)
        requested_runs = payload.get("pipeline_runs")
        if requested_runs is None:
            requested_runs = self._default_pipeline_runs_for_current_task()
        try:
            pipeline_runs = max(1, int(requested_runs))
        except Exception:  # noqa: BLE001
            pipeline_runs = 1
        self._pipeline_runs_remaining = pipeline_runs
        on_resume = getattr(self._current_task, "on_resume", None)
        if callable(on_resume):
            try:
                on_resume(pipeline_runs)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Task on_resume hook failed: %s", exc)
        if not self._is_paused:
            self._resume_gate.set()
            logger.info("Frame loop re-armed: %s (pipeline_runs=%d)", reason, pipeline_runs)
            return
        self._is_paused = False
        self._resume_gate.set()
        logger.info("Frame loop resumed: %s (pipeline_runs=%d)", reason, pipeline_runs)

    def _default_pipeline_runs_for_current_task(self) -> int:
        task = self._current_task
        if task is None:
            return 1
        raw = getattr(task, "frames_per_resume", 1)
        try:
            return max(1, int(raw))
        except Exception:  # noqa: BLE001
            return 1

    def _build_task_frame_result(self) -> _TaskFrameResult:
        iteration_event = self._handoff.get("TASK:ITERATION", {})
        payload = iteration_event.get("payload", {}) if isinstance(iteration_event, dict) else {}
        if not isinstance(payload, dict):
            payload = {}
        return _TaskFrameResult(
            detections=list(payload.get("detections") or []),
            stable_targets=list(payload.get("stable_targets") or []),
            tracked_targets=list(payload.get("tracked_targets") or []),
            is_done="TASK:DONE" in self._handoff,
            metadata=dict(payload.get("metadata") or {}),
        )

    def _update_live_stats(self, result, fps: float) -> None:
        """Push fps / frame / detection count to the in-process live-state bus."""
        try:
            self._live_state.update({
                "fps": round(float(fps), 2),
                "frame": self._frame_count,
                "detection_count": len(result.detections),
            })
        except Exception:  # noqa: BLE001
            pass

    def _render_ar_preview(
        self, image, *, dets, metadata: Optional[dict] = None, timestamp=None,
    ):
        """Render the AR preview window and advance the FROZEN/OVERLAY state.

        Modes:
        - LIVE     — clean camera passthrough (no annotations).
        - FROZEN   — the post-RESUME pipeline frame, dets baked in,
                     held for one tick.
        - OVERLAY  — live camera with the frozen dets re-rendered
                     each frame (operator watches the pick arm enter
                     against the original target).

        ``dets`` is None when this tick is paused (no pipeline run),
        non-None when the pipeline just produced fresh detections.
        Receiving non-None dets always advances LIVE → FROZEN.

        ``metadata`` is the pipeline's metadata dict from the same run as
        ``dets``. We pull crop_single_square's matched cell and
        abstain_near_metal's safety margin so the preview can draw both
        shapes — purely visual, never feeds back into pick coordinates.

        pluck-hair_test: this method now ALSO (a) draws the live tweezer-tip
        cross + tip→pick distance, (b) publishes the annotated full-res frame to
        the web stream every tick (so the left-half live view tracks the tweezer
        in real time, not just at photo positions), and (c) pushes live state to
        the in-process state bus. It returns the annotated display frame.
        """
        # Lazy import keeps frame_loop independent of the diagnostic helper
        # at import time — useful for headless deployments that strip tools/.
        try:
            from tools.machine_result_snapshot import render_annotated
        except Exception:  # noqa: BLE001
            render_annotated = None  # type: ignore[assignment]

        # Pipeline just ran → take this frame as the new FROZEN snapshot.
        if dets is not None:
            self._ar_frozen_image = image.copy()
            self._ar_frozen_dets = list(dets)
            self._ar_frozen_preview_only_dets = _preview_only_detections_from_metadata(metadata)
            self._ar_frozen_flange = self._latest_flange_pose_mm
            self._ar_frozen_cell_box = _cell_box_from_metadata(metadata)
            self._ar_frozen_safety_margin_px = _safety_margin_from_metadata(metadata)
            self._ar_state = "FROZEN"

        # Decide what to display this tick.
        frozen_display_dets = _merge_preview_detections(
            self._ar_frozen_dets,
            self._ar_frozen_preview_only_dets,
        )
        if self._ar_state == "FROZEN" and self._ar_frozen_image is not None:
            display = self._ar_frozen_image.copy()
            if render_annotated is not None and frozen_display_dets:
                display = render_annotated(
                    display,
                    frozen_display_dets,
                    flange_pose_mm=None,
                    cell_box_xyxy=self._ar_frozen_cell_box,
                    safety_margin_px=self._ar_frozen_safety_margin_px,
                )
            mode_label = "FROZEN"
            # Auto-advance: next tick we go to OVERLAY (or LIVE if dets empty).
            if frozen_display_dets:
                self._ar_state = "OVERLAY"
            else:
                self._ar_state = "LIVE"
                self._ar_frozen_image = None
                self._ar_frozen_dets = []
                self._ar_frozen_preview_only_dets = []
                self._ar_frozen_flange = None
                self._ar_frozen_cell_box = None
                self._ar_frozen_safety_margin_px = None
        elif self._ar_state == "OVERLAY" and frozen_display_dets:
            if render_annotated is not None:
                display = render_annotated(
                    image,
                    frozen_display_dets,
                    flange_pose_mm=None,
                    cell_box_xyxy=self._ar_frozen_cell_box,
                    safety_margin_px=self._ar_frozen_safety_margin_px,
                )
            else:
                display = image.copy()
            mode_label = "OVERLAY"
        else:
            display = image.copy()
            mode_label = "LIVE"

        # HUD banner: top-left, black-stroke yellow text. Survives any
        # background — useful when the field-of-view goes mostly black or
        # mostly white.
        ar_lines = [
            f"mode: {mode_label}",
            f"frozen dets: {len(frozen_display_dets)}",
        ]
        y = 36
        for line in ar_lines:
            cv2.putText(display, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(display, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2, cv2.LINE_AA)
            y += 32

        _draw_plc_overlay(display, self._latest_epson_target_overlay)

        # --- pluck-hair_test: tweezer-tip overlay + distance to nearest pick ---
        # Pick points come from whatever detections are currently displayed
        # (frozen at the photo position), so the tip→pick distance is measured
        # against the real predicted pick targets the operator sees.
        pick_points: list = []
        for d in frozen_display_dets:
            pp = getattr(d, "pick_point_xy", None)
            if pp and len(pp) >= 2:
                pick_points.append((float(pp[0]), float(pp[1])))

        tweezer_dict: dict = {"found": False}
        if self._tweezer is not None:
            try:
                # Detect on the CLEAN camera frame, draw on the annotated display.
                tw = self._tweezer.detect(image)
                _, dist_mm = self._tweezer.draw_overlay(display, tw, pick_points)
                self._latest_tweezer = tw
                self._latest_tip_to_pick_mm = dist_mm
                tweezer_dict = tw.as_dict()
                if dist_mm is not None:
                    txt = f"tip->pick: {dist_mm:.2f} mm"
                    cv2.putText(display, txt, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 0, 0), 5, cv2.LINE_AA)
                    cv2.putText(display, txt, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 0, 255), 2, cv2.LINE_AA)
            except Exception as exc:  # noqa: BLE001 — overlay must never kill the loop
                logger.debug("tweezer overlay failed: %s", exc)

        try:
            self._live_state.update({
                "ar_mode": mode_label,
                "detection_count": len(frozen_display_dets),
                "tweezer": tweezer_dict,
                "tip_to_pick_mm": self._latest_tip_to_pick_mm,
            })
        except Exception:  # noqa: BLE001
            pass

        # Publish the annotated full-res frame to the web stream EVERY tick.
        if timestamp is None:
            timestamp = datetime.now()
        self._frame_streamer.publish(
            display, frame_id=f"{self._frame_count:06d}", timestamp=timestamp,
        )

        # Local OpenCV preview window (optional) — downscaled copy only.
        if self._config.show_preview:
            h, w = display.shape[:2]
            max_w = 1280
            shown = display
            if w > max_w:
                scale = max_w / w
                shown = cv2.resize(
                    display, (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            cv2.imshow("Detection Preview", shown)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("User pressed 'q', stopping...")
                self._running = False

        return display


def _extract_photo_key(reason: str) -> Optional[str]:
    """Slice a photo key like '1-1' out of a RESUME reason string.

    Today plc_orchestrator emits ``"nova5 parked at <key>"`` (see
    ``PlcOrchestratorTask._on_photo_arrived``). We tolerate other
    phrasings by just returning the last whitespace-delimited token —
    callers use it as a filename hint only, so a slightly off label
    never breaks anything.
    """
    if not reason:
        return None
    tail = reason.rsplit(None, 1)[-1]
    return tail or None


def _build_upper_machine_image(
    *,
    image,
    detections: Sequence[Any],
    flange_pose_mm: Optional[tuple[float, float]],
    plc_overlay: Optional[dict[str, Any]] = None,
    tracking_radius_px: float = 30.0,
):
    """Build the still image shown to the upper machine after one recognition."""
    try:
        from tools.machine_result_snapshot import render_annotated
    except Exception:  # noqa: BLE001
        display = image.copy()
    else:
        display = render_annotated(
            image,
            detections,
            flange_pose_mm=flange_pose_mm,
        )

    if plc_overlay:
        _draw_plc_overlay(display, plc_overlay)
    else:
        _draw_detection_reference_overlay(
            display,
            detections,
            tracking_radius_px=tracking_radius_px,
        )
    h, w = display.shape[:2]
    max_w = 1280
    if w > max_w:
        scale = max_w / w
        display = cv2.resize(
            display,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return display


def _draw_detection_reference_overlay(
    image,
    detections: Sequence[Any],
    *,
    tracking_radius_px: float,
) -> None:
    det = _first_detection_with_pick(detections)
    if det is None:
        return

    reference_color = (225, 245, 245)
    pick_color = (150, 40, 170)
    _draw_u_reference_line(image, 30.0, reference_color, "30")
    _draw_u_reference_line(image, 110.0, reference_color, "110")

    angle = getattr(det, "pick_angle_deg", None)
    if angle is not None:
        _draw_u_reference_line(
            image,
            float(angle),
            pick_color,
            f"U={float(angle):.1f}",
            thickness=3,
            length_scale=0.5,
        )


def _first_detection_with_pick(detections: Sequence[Any]) -> Optional[Any]:
    for det in detections:
        if getattr(det, "pick_point_xy", None):
            return det
    return detections[0] if detections else None


def _preview_only_detections_from_metadata(metadata: Optional[dict]) -> list[Any]:
    if not isinstance(metadata, dict):
        return []
    raw = metadata.get("preview_only_detections")
    if not isinstance(raw, list):
        return []
    return list(raw)


def _merge_preview_detections(
    detections: Sequence[Any],
    preview_only_detections: Sequence[Any],
) -> list[Any]:
    if not preview_only_detections:
        return list(detections)
    return list(detections) + list(preview_only_detections)


def _draw_plc_overlay(
    display,
    overlay: Optional[dict[str, Any]],
) -> None:
    if not overlay:
        return

    h, w = display.shape[:2]
    reference_color = (225, 245, 245)
    pick_color = (150, 40, 170)
    _draw_u_reference_line(display, 30.0, reference_color, "30")
    _draw_u_reference_line(display, 110.0, reference_color, "110")
    _draw_u_reference_line(
        display,
        float(overlay["u"]),
        pick_color,
        f"U={float(overlay['u']):.1f}",
        thickness=3,
        length_scale=0.5,
    )

    hud_lines = [
        f"PLC X={float(overlay['x']):.3f}",
        f"PLC Y={float(overlay['y']):.3f}",
        f"PLC Z={float(overlay['z']):.3f}",
        f"PLC U={float(overlay['u']):.1f}",
        f"send attempt={int(overlay.get('attempt', 0) or 0)}",
    ]
    x = max(15, w - 320)
    y = 36
    for line in hud_lines:
        cv2.putText(
            display,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 32


def _draw_u_reference_line(
    image,
    u_deg: float,
    color: tuple[int, int, int],
    label: str,
    *,
    thickness: int = 2,
    length_scale: float = 1.0,
) -> None:
    h, w = image.shape[:2]
    theta = math.radians(_pick_u_to_preview_axis_deg(u_deg))
    dx = math.cos(theta)
    dy = -math.sin(theta)
    cx = w / 2.0
    cy = h / 2.0
    radius = max(w, h) * max(0.0, float(length_scale))
    pt2 = (
        int(round(cx + dx * radius)),
        int(round(cy + dy * radius)),
    )
    pt1 = (
        int(round(cx)),
        int(round(cy)),
    )
    cv2.line(image, pt1, pt2, color, thickness, cv2.LINE_AA)
    label_radius = min(radius * 0.78, 300.0)
    label_x = int(round(cx + dx * label_radius))
    label_y = int(round(cy + dy * label_radius))
    cv2.putText(
        image,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA,
    )


def _pick_u_to_preview_axis_deg(u_deg: float) -> float:
    angle_deg = float(u_deg) - 50.0
    while angle_deg >= 90.0:
        angle_deg -= 180.0
    while angle_deg < -90.0:
        angle_deg += 180.0
    return angle_deg


def _build_multipart_body(boundary: str, job: _UploadJob) -> bytes:
    """Serialize one upload into multipart/form-data."""
    crlf = b"\r\n"
    body = bytearray()
    fields = {
        "photo_key": job.photo_key,
        "timestamp": job.timestamp,
        "robot": job.robot,
        "position": job.position,
    }
    for key, value in fields.items():
        body.extend(f"--{boundary}".encode("utf-8"))
        body.extend(crlf)
        body.extend(
            f'Content-Disposition: form-data; name="{key}"'.encode("utf-8")
        )
        body.extend(crlf)
        body.extend(crlf)
        body.extend(str(value).encode("utf-8"))
        body.extend(crlf)

    body.extend(f"--{boundary}".encode("utf-8"))
    body.extend(crlf)
    body.extend(
        (
            f'Content-Disposition: form-data; name="file"; '
            f'filename="{job.filename}"'
        ).encode("utf-8")
    )
    body.extend(crlf)
    body.extend(f"Content-Type: {job.content_type}".encode("utf-8"))
    body.extend(crlf)
    body.extend(crlf)
    body.extend(job.image_bytes)
    body.extend(crlf)
    body.extend(f"--{boundary}--".encode("utf-8"))
    body.extend(crlf)
    return bytes(body)


def _create_capture_session_dir() -> Path:
    """Create the startup-scoped folder for per-photo crop images."""
    project_root = Path(__file__).resolve().parents[3]
    session_dir = project_root / datetime.now().strftime("%m.%d.%H.%M")
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def _crop_from_original_box(image, box: object):
    """Crop the original frame using crop_single_square's original-frame box."""
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
    except Exception:  # noqa: BLE001
        return None

    h, w = image.shape[:2]
    x1 = max(0, min(w, x1))
    y1 = max(0, min(h, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2].copy()


def _cell_box_from_metadata(
    metadata: Optional[dict],
) -> Optional[tuple[int, int, int, int]]:
    """Pull crop_single_square's matched cell rect (in original-frame px).

    Returns None when the crop step abstained, fell back, or metadata is
    missing — preview just won't draw the cell rectangle in those cases.
    """
    if not isinstance(metadata, dict):
        return None
    crop_meta = metadata.get("crop_single_square")
    if not isinstance(crop_meta, dict):
        return None
    if crop_meta.get("source") == "geometric_fallback":
        return None
    box = crop_meta.get("box_xyxy_in_original")
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    try:
        return tuple(int(round(float(v))) for v in box[:4])  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def _safety_margin_from_metadata(metadata: Optional[dict]) -> Optional[float]:
    """Pull abstain_near_metal's safety_margin_px (the 1.5 mm inset width)."""
    if not isinstance(metadata, dict):
        return None
    abstain_meta = metadata.get("abstain_near_metal")
    if not isinstance(abstain_meta, dict):
        return None
    raw = abstain_meta.get("safety_margin_px")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None
