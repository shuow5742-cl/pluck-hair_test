#!/usr/bin/env python3
"""Main entry point for Pluck Backend.

Usage:
    # Run detection loop
    python main.py --config config/settings.yaml --mode run

    # Run API server only
    python main.py --config config/settings.yaml --mode api

    # Development mode with mock camera
    python main.py --config config/settings.dev.yaml --mode run
"""

import argparse
from dataclasses import asdict
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import AppConfig


def setup_logging(level: str = "INFO"):
    """Setup logging: stderr stream + rotating file handler.

    File logs land in ``logs/backend.log`` (rotating at 20 MB, keeping
    the last 10 files). New runs append to the active file rather than
    overwrite, so cross-restart diagnostics survive (e.g. comparing
    "PLC says it received X" against the PC-side TX log from an earlier
    run). Old runs are split by the natural rotation boundary; if you
    need to bookmark a specific run, look for the startup banner
    ``"Starting Pluck Backend in run mode"``.
    """
    from logging.handlers import RotatingFileHandler

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    resolved_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(resolved_level)
    # Clear any handlers basicConfig or a previous call may have installed
    # — we own the configuration here.
    for h in list(root.handlers):
        root.removeHandler(h)

    stream = logging.StreamHandler()
    stream.setLevel(resolved_level)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    file_handler = RotatingFileHandler(
        log_dir / "backend.log",
        maxBytes=20 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8",
    )
    file_handler.setLevel(resolved_level)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


def create_camera(config: AppConfig):
    """Create camera instance from configuration.

    Args:
        config: Application configuration.

    Returns:
        Camera instance.
    """
    camera_config = config.camera

    if camera_config.type == "daheng":
        from autoweaver.camera import CameraConfig as BaseCameraConfig
        from src.fresh_frame_daheng import FreshFrameDahengCamera

        base_config = BaseCameraConfig(
            device_index=camera_config.device_index,
            device_sn=camera_config.device_sn,
            exposure_auto=camera_config.exposure_auto,
            gain_auto=camera_config.gain_auto,
            exposure_time=camera_config.exposure_time,
            gain=camera_config.gain,
            white_balance_mode=camera_config.white_balance_mode,
        )
        return FreshFrameDahengCamera(base_config)

    elif camera_config.type == "mock":
        from autoweaver.camera import CameraConfig as BaseCameraConfig
        from autoweaver.camera import MockCamera

        base_config = BaseCameraConfig()
        return MockCamera(
            base_config,
            mode=camera_config.mode,
            image_dir=camera_config.image_dir,
            width=camera_config.width,
            height=camera_config.height,
        )

    else:
        raise ValueError(f"Unknown camera type: {camera_config.type}")


def create_pipeline(config: AppConfig):
    """Create vision pipeline from configuration.

    Args:
        config: Application configuration.

    Returns:
        VisionPipeline instance.
    """
    from autoweaver.pipeline import VisionPipeline

    # Register hub-side steps (e.g. yolo_seg) before from_config resolves types.
    import src.steps  # noqa: F401  (import registers step types as a side effect)

    return VisionPipeline.from_config(
        {"pipeline": {"steps": [asdict(step) for step in config.vision.steps]}}
    )


def create_storage(config: AppConfig):
    """Create storage instances from configuration.

    Args:
        config: Application configuration.

    Returns:
        Tuple of (image_storage, database).
    """
    storage_config = config.storage

    # Image storage
    images_config = storage_config.images
    if images_config.type == "minio":
        from src.storage.minio_storage import MinIOStorage

        image_storage = MinIOStorage(
            endpoint=images_config.endpoint,
            access_key=images_config.access_key,
            secret_key=images_config.secret_key,
            bucket=images_config.bucket,
            secure=images_config.secure,
        )
    elif images_config.type == "local":
        from src.storage.local_storage import LocalStorage

        image_storage = LocalStorage(base_path=images_config.path)
    else:
        raise ValueError(f"Unknown storage type: {images_config.type}")

    # Database
    db_config = storage_config.database
    if db_config.type == "postgres":
        from src.storage.postgres_db import PostgresDatabase

        database = PostgresDatabase(
            connection_string=db_config.connection_string,
            echo=db_config.echo,
            pool_size=db_config.pool_size,
        )
    elif db_config.type == "sqlite":
        from src.storage.sqlite_db import SQLiteDatabase

        database = SQLiteDatabase(
            db_path=db_config.path,
            echo=db_config.echo,
        )
    else:
        raise ValueError(
            f"Unknown database type: {db_config.type}. Supported: postgres, sqlite"
        )

    return image_storage, database


def build_vision_engine(config: AppConfig, frame_publisher_override=None):
    """Build (but do not start) the vision WorkflowEngine.

    Extracted from ``run_detection_loop`` so the single-process test console
    (``--mode test``) can inject an in-process frame publisher and run the
    engine in a background thread. ``frame_publisher_override``, when given,
    replaces the Redis frame publisher — the test app passes the in-process
    ``FrameBus`` so MJPEG works without Redis.
    """
    logger = logging.getLogger(__name__)
    logger.info("Initializing detection system...")

    # Create components with unified config
    camera = create_camera(config)
    pipeline = create_pipeline(config)

    # Optional Redis publisher for real-time events
    event_publisher = None
    if getattr(config, "redis", None) and config.redis.enabled:
        try:
            from src.events.redis_streams import RedisStreamPublisher

            event_publisher = RedisStreamPublisher(
                url=config.redis.url,
                stream=config.redis.stream,
                maxlen=config.redis.maxlen,
            )
            logger.info("Redis Streams publisher enabled for stream %s", config.redis.stream)
        except Exception as e:
            logger.warning(f"Failed to initialize Redis publisher: {e}")

    # Optional Redis publisher for MJPEG frames
    frame_publisher = None
    if (
        getattr(config, "redis", None)
        and config.redis.enabled
        and getattr(config, "video_stream", None)
        and config.video_stream.enabled
    ):
        try:
            from src.events.frame_stream import FrameStreamPublisher

            frame_publisher = FrameStreamPublisher(
                url=config.redis.url,
                stream=config.video_stream.stream,
                maxlen=config.video_stream.maxlen,
            )
            logger.info(
                "Frame publisher enabled for stream %s",
                config.video_stream.stream,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize frame publisher: {e}")

    # Test console: use the in-process FrameBus instead of Redis for streaming.
    if frame_publisher_override is not None:
        frame_publisher = frame_publisher_override
        logger.info("Frame publisher overridden with in-process FrameBus")

    from src.workflow import WorkflowEngine, load_workflow_from_yaml
    from src.tasks import create_task
    from src.tasks.frame_loop import FrameLoopSideTask, FrameLoopConfig
    from src.config import TaskConfig

    # Register pose sources before any task that may resolve them.
    # MockPoseSource keeps the pre-PLC behavior (flange at origin); the
    # real PLC adapter will replace this entry once communication lands.
    from src.pose_sources import register_pose_source, MockPoseSource
    register_pose_source("default", MockPoseSource(x=0.0, y=0.0, z=0.0))

    scheduler_config = config.scheduler

    # Load workflow definition (mandatory)
    definition = load_workflow_from_yaml(config.workflow.path)
    state_machine = definition.state_machine

    # Build state -> Task instance mapping
    # Merge workflow task type with scheduler's task config (calibration, stabilizer, etc.)
    base_task_cfg = scheduler_config.task
    resolved_tasks = {}
    for state_name, task_type in definition.task_map.items():
        task_cfg = TaskConfig(
            type=task_type,
            stabilizer=base_task_cfg.stabilizer,
            pick_process=base_task_cfg.pick_process,
            calibration=base_task_cfg.calibration,
        )
        resolved_tasks[state_name] = create_task(pipeline, task_cfg)

    # FrameLoopSideTask is the FIRST side task so its FRAME_LOOP:PAUSE/RESUME
    # subscriptions are wired before plc_orchestrator.attach() publishes the
    # initial pause. Previously it was appended last and the startup pause
    # event was lost — camera ran free, only the _accepting_batch second
    # gate prevented bogus Epson coords. Order now:
    #   frame_loop → workflow side_tasks (pixel_to_world, plc_orchestrator)
    # pluck-hair_test: build the tweezer detector config from settings.
    tweezer_config = None
    if getattr(config, "tweezer", None) and config.tweezer.enabled:
        from src.core.tweezer_detector import TweezerConfig
        tweezer_config = TweezerConfig.from_dict(config.tweezer.params)
        tweezer_config.enabled = True
        logger.info("Tweezer detection enabled for frame loop")

    frame_loop_config = FrameLoopConfig(
        loop_delay_ms=scheduler_config.loop_delay_ms,
        max_errors=scheduler_config.max_errors,
        show_preview=scheduler_config.show_preview,
        http_image_upload=config.http_image_upload,
        tweezer_config=tweezer_config,
    )
    side_tasks = [FrameLoopSideTask(
        camera=camera,
        task_map=resolved_tasks,
        config=frame_loop_config,
        event_publisher=event_publisher,
        frame_publisher=frame_publisher,
        video_stream_config=config.video_stream if getattr(config, "video_stream", None) else None,
    )]
    for st_type in definition.side_task_types:
        if st_type == "communication" and getattr(config, "comm", None):
            try:
                from src.comm import ModbusAdapter, RedisAdapter
                from src.tasks.communication import CommunicationTask

                comm_signal = None
                comm_type = config.comm.comm_signal.type
                if comm_type == "redis":
                    if not (getattr(config, "redis", None) and config.redis.enabled):
                        logger.warning("Redis comm signal requested but redis.enabled=false")
                    else:
                        comm_signal = RedisAdapter(
                            redis_url=config.redis.url,
                            command_key=config.comm.comm_signal.redis.command_key,
                        )
                        logger.info(
                            "Comm signal adapter enabled (redis, key=%s)",
                            config.comm.comm_signal.redis.command_key,
                        )
                elif comm_type == "modbus":
                    comm_signal = ModbusAdapter(
                        host=config.comm.comm_signal.modbus.host,
                        port=config.comm.comm_signal.modbus.port,
                        unit_id=config.comm.comm_signal.modbus.unit_id,
                        timeout=config.comm.comm_signal.modbus.timeout,
                    )
                else:
                    logger.warning("Unknown comm signal type: %s", comm_type)

                if comm_signal is not None:
                    side_tasks.append(
                        CommunicationTask(comm_signal=comm_signal)
                    )
            except Exception as e:
                logger.warning(f"Failed to initialize comm signal: {e}")

        if st_type == "pixel_to_world":
            try:
                from src.core.coordinate_transform import ExtrinsicCalibration
                from src.tasks.pixel_to_world import PixelToWorldTask

                cal_cfg = base_task_cfg.calibration or {}
                extrinsic_path = cal_cfg.get("extrinsic_path")
                intrinsic_path = cal_cfg.get("intrinsic_path")
                if not extrinsic_path or not intrinsic_path:
                    raise ValueError(
                        "pixel_to_world side task requires "
                        "scheduler.task.calibration.{extrinsic_path,intrinsic_path}"
                    )
                calibration = ExtrinsicCalibration.load(extrinsic_path, intrinsic_path)
                pose_source_name = cal_cfg.get("pose_source", "default")
                side_tasks.append(
                    PixelToWorldTask(calibration, pose_source_name=pose_source_name)
                )
                logger.info(
                    "PixelToWorldTask attached (pose_source=%s, mm_per_pixel=%s)",
                    pose_source_name,
                    calibration.mm_per_pixel,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize pixel_to_world: {e}")

        if st_type == "plc_orchestrator":
            try:
                from src.tasks.plc_orchestrator import PlcOrchestratorTask

                plc_cfg = config.plc_orchestrator
                if not plc_cfg.enabled:
                    logger.info(
                        "plc_orchestrator side task requested but plc_orchestrator.enabled=false"
                    )
                else:
                    side_tasks.append(PlcOrchestratorTask(plc_cfg))
                    logger.info(
                        "PlcOrchestratorTask attached (host=%s:%s, points=%s)",
                        plc_cfg.host, plc_cfg.port, plc_cfg.points_path,
                    )
            except Exception as e:
                logger.warning(f"Failed to initialize plc_orchestrator: {e}")

    engine = WorkflowEngine(
        state_machine=state_machine,
        task_map=resolved_tasks,
        side_tasks=side_tasks,
    )
    return engine


def run_detection_loop(config: AppConfig):
    """Run the main detection loop (blocking)."""
    logger = logging.getLogger(__name__)
    engine = build_vision_engine(config)
    logger.info("Starting detection loop...")
    engine.loop()


def run_test_app(config: AppConfig):
    """Single-process test console: vision engine + FastAPI in one process.

    The production deployment splits vision (``--mode run``) and API
    (``--mode api``) across two processes bridged by Redis. The pluck-hair_test
    console instead runs both here so the operator can launch one script and get
    the split UI (left = live camera + tweezer overlay, right = Epson/IO
    control). The vision engine runs in a daemon thread and streams annotated
    frames through the in-process FrameBus; uvicorn serves the API + MJPEG on the
    main thread.
    """
    import threading
    import uvicorn

    from src.api.app import create_app
    from src.comm.inproc_bus import get_frame_bus

    logger = logging.getLogger(__name__)

    # Storage backs the original REST/health routes.
    image_storage, database = create_storage(config)

    app = create_app(
        config=config,
        database=database,
        image_storage=image_storage,
        title="Pluck Test Console",
    )

    # Attach the Epson manual-control + IO controller (right-half console).
    app.state.epson_io = None
    if getattr(config, "epson_io", None) and config.epson_io.enabled:
        try:
            from src.comm.epson_io_controller import load_epson_io_controller

            app.state.epson_io = load_epson_io_controller(config.epson_io.path)
            logger.info(
                "Epson/IO controller loaded (%s, backend=%s)",
                config.epson_io.path,
                app.state.epson_io.backend_kind,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load Epson/IO controller: %s", exc)

    # Build + start the vision engine in a background thread, streaming to the
    # in-process FrameBus (no Redis needed).
    frame_bus = get_frame_bus()
    engine = build_vision_engine(config, frame_publisher_override=frame_bus)

    engine_thread = threading.Thread(
        target=engine.loop, daemon=True, name="vision-engine"
    )
    engine_thread.start()
    logger.info("Vision engine started in background thread")

    try:
        uvicorn.run(app, host=config.api.host, port=config.api.port)
    finally:
        logger.info("Shutting down vision engine...")
        try:
            engine.stop()
        except Exception:  # noqa: BLE001
            pass


def run_api_server(config: AppConfig):
    """Run the API server with properly injected dependencies.

    Args:
        config: Application configuration.
    """
    import uvicorn
    from src.api.app import create_app

    logger = logging.getLogger(__name__)

    # Create storage instances (same ones used by both API and health checks)
    image_storage, database = create_storage(config)
    logger.info(
        f"Storage initialized: images={config.storage.images.type}, db={config.storage.database.type}"
    )

    # Create app with injected dependencies
    app = create_app(
        config=config,
        database=database,
        image_storage=image_storage,
        title="Pluck Backend API",
    )

    logger.info("Starting API server with injected dependencies")

    uvicorn.run(
        app,
        host=config.api.host,
        port=config.api.port,
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Pluck Backend - Bird's Nest Inspection System"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["run", "api", "test"],
        default="run",
        help=(
            "Run mode: 'run' detection loop, 'api' API server, "
            "'test' single-process test console (vision + API + MJPEG)"
        ),
    )
    args = parser.parse_args()

    # Load configuration using unified AppConfig
    try:
        config = AppConfig.from_yaml(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Setup logging
    setup_logging(config.app.log_level)

    logger = logging.getLogger(__name__)
    logger.info(f"Starting Pluck Backend in {args.mode} mode")
    logger.info(f"Configuration: {args.config}")

    # Run selected mode
    try:
        if args.mode == "run":
            run_detection_loop(config)
        elif args.mode == "api":
            run_api_server(config)
        elif args.mode == "test":
            run_test_app(config)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
