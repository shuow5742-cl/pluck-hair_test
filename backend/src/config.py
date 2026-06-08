"""Configuration loading and management.

This module provides unified configuration management with:
- Type-safe configuration classes using dataclasses
- Environment variable substitution
- Single source of truth for all components
"""

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


# ============================================================
# Configuration Data Classes
# ============================================================

@dataclass
class CameraConfig:
    """Camera configuration."""
    type: str = "daheng"
    device_index: int = 1
    device_sn: Optional[str] = None
    exposure_auto: bool = False
    gain_auto: bool = False
    exposure_time: Optional[float] = None
    gain: Optional[float] = None
    white_balance_mode: str = "once"  # auto, once, manual, off
    white_balance_red: Optional[float] = None
    white_balance_green: Optional[float] = None
    white_balance_blue: Optional[float] = None
    gamma_enable: bool = False
    gamma_value: Optional[float] = None
    # For mock camera
    mode: str = "random"
    image_dir: Optional[str] = None
    width: int = 640
    height: int = 480


@dataclass
class PipelineStepConfig:
    """Single pipeline step configuration."""
    name: str = ""
    type: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisionConfig:
    """Vision pipeline configuration."""
    steps: List[PipelineStepConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "VisionConfig":
        """Create VisionConfig from dictionary."""
        pipeline_data = data.get("pipeline", {})
        steps_data = pipeline_data.get("steps", [])
        steps = [
            PipelineStepConfig(
                name=s.get("name", ""),
                type=s.get("type", ""),
                params=s.get("params", {}),
            )
            for s in steps_data
        ]
        return cls(steps=steps)


@dataclass
class ImageStorageConfig:
    """Image storage configuration."""
    type: str = "minio"
    # MinIO settings
    endpoint: str = "localhost:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"
    bucket: str = "pluck-images"
    secure: bool = False
    # Local storage settings
    path: str = "./data/images"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    type: str = "postgres"
    connection_string: str = "postgresql://pluck:pluck123@localhost:5432/pluck"
    pool_size: int = 5
    echo: bool = False
    # SQLite settings
    path: str = "./data/pluck.db"


@dataclass
class StorageConfig:
    """Combined storage configuration."""
    images: ImageStorageConfig = field(default_factory=ImageStorageConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)

    @classmethod
    def from_dict(cls, data: dict) -> "StorageConfig":
        """Create StorageConfig from dictionary."""
        images_data = data.get("images", {})
        db_data = data.get("database", {})
        return cls(
            images=ImageStorageConfig(**images_data),
            database=DatabaseConfig(**db_data),
        )


@dataclass
class TaskConfig:
    """Task selection/configuration for the scheduler."""

    type: str = "stabilized_detection"
    name: Optional[str] = None
    stabilizer: Dict[str, Any] = field(default_factory=dict)
    pick_process: Dict[str, Any] = field(default_factory=dict)
    calibration: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Any) -> "TaskConfig":
        if not data:
            return cls()
        if isinstance(data, str):
            return cls(type=data)
        if not isinstance(data, dict):
            raise TypeError("scheduler.task must be a mapping or string")
        stabilizer_data = data.get("stabilizer") or {}
        if stabilizer_data is None:
            stabilizer_data = {}
        pick_process_data = data.get("pick_process") or {}
        calibration_data = data.get("calibration") or {}
        return cls(
            type=data.get("type", cls.type),
            name=data.get("name"),
            stabilizer=stabilizer_data,
            pick_process=pick_process_data,
            calibration=calibration_data,
        )


@dataclass
class SchedulerConfig:
    """Scheduler/Workflow configuration."""
    loop_delay_ms: int = 100
    max_errors: int = 10
    save_images: bool = True
    save_annotated: bool = True
    show_preview: bool = True  # Show real-time OpenCV preview window
    async_storage: bool = True
    storage_workers: int = 4
    max_pending_saves: int = 100
    storage_retry_count: int = 3
    task: TaskConfig = field(default_factory=TaskConfig)


@dataclass
class WorkflowRuntimeConfig:
    """Workflow runtime configuration."""

    enabled: bool = False
    path: str = "config/workflow.yaml"


@dataclass
class RedisConfig:
    """Redis settings for inter-process messaging."""
    url: str = "redis://localhost:6379/0"
    stream: str = "pluck:events"
    consumer_group: str = "api"
    consumer_name: str = "api-1"
    maxlen: int = 10000
    read_count: int = 10
    block_ms: int = 5000  # Max time (ms) to block XREAD when waiting for detection events
    enabled: bool = True  # whether to enable Redis Streams


@dataclass
class VideoStreamConfig:
    """Video streaming (MJPEG) settings."""
    enabled: bool = True
    stream: str = "pluck:frames"
    maxlen: int = 50
    fps_limit: float = 15.0
    jpeg_quality: int = 80
    block_ms: int = 2000  # Max time (ms) to block XREAD when waiting for new video frames


@dataclass
class HttpImageUploadConfig:
    """HTTP push settings for one-shot recognition images."""

    enabled: bool = False
    url: str = ""
    robot: str = "Nova5"
    connect_timeout_s: float = 2.0
    upload_timeout_s: float = 5.0
    retry_count: int = 1
    queue_size: int = 32


@dataclass
@dataclass
class CommSignalRedisConfig:
    """Redis settings for comm signal channel."""

    command_key: str = "pluck:control"
    response_timeout: float = 5.0


@dataclass
class CommSignalModbusConfig:
    """Modbus settings for comm signal channel."""

    host: str = "127.0.0.1"
    port: int = 502
    unit_id: int = 1
    timeout: float = 1.0


@dataclass
class CommSignalConfig:
    """Comm signal channel configuration."""

    type: str = "redis"  # redis | modbus
    redis: CommSignalRedisConfig = field(default_factory=CommSignalRedisConfig)
    modbus: CommSignalModbusConfig = field(default_factory=CommSignalModbusConfig)


@dataclass
class CommConfig:
    """Communication configuration."""

    comm_signal: CommSignalConfig = field(default_factory=CommSignalConfig)

    @classmethod
    def from_dict(cls, data: dict) -> "CommConfig":
        data = data or {}
        comm_signal_data = data.get("comm_signal") or {}
        redis_data = comm_signal_data.get("redis") or {}
        modbus_data = comm_signal_data.get("modbus") or {}
        comm_signal = CommSignalConfig(
            type=comm_signal_data.get("type", CommSignalConfig.type),
            redis=CommSignalRedisConfig(**redis_data),
            modbus=CommSignalModbusConfig(**modbus_data),
        )
        return cls(comm_signal=comm_signal)


@dataclass
class PlcRegisterMap:
    """Modbus protocol register addresses (PLC's view, base_addr-relative).

    Defaults match the validated standalone test GUI
    `plc_modbus_auto_test_gui_v2_strict.py`. PLC engineer owns these —
    override per deployment if the PLC program moves a block.
    """

    base_addr: int = 40001

    # PLC → PC: request flags + function codes (PLC writes, PC reads)
    plc_device_state: int = 41063
    plc_nova2_send_flag: int = 41064
    plc_nova2_func: int = 41065
    plc_nova5_send_flag: int = 41066
    plc_nova5_func: int = 41067
    plc_epson_send_flag: int = 41068
    plc_epson_func: int = 41069

    # PLC → PC: echo of PC-written target coords (for closed-loop verify)
    plc_recv_nova2_x: int = 41071
    plc_recv_nova2_y: int = 41073
    plc_recv_nova2_z: int = 41075
    plc_recv_nova2_u: int = 41077
    plc_recv_nova2_v: int = 41079
    plc_recv_nova2_w: int = 41081
    plc_recv_nova5_x: int = 41083
    plc_recv_nova5_y: int = 41085
    plc_recv_nova5_z: int = 41087
    plc_recv_nova5_u: int = 41089
    plc_recv_nova5_v: int = 41091
    plc_recv_nova5_w: int = 41093
    plc_recv_epson_x: int = 41095
    plc_recv_epson_y: int = 41097
    plc_recv_epson_z: int = 41099
    plc_recv_epson_u: int = 41101

    # PLC → PC: real-time arm positions (telemetry, read-only)
    plc_rt_nova2_x: int = 41103
    plc_rt_nova2_y: int = 41105
    plc_rt_nova2_z: int = 41107
    plc_rt_nova2_u: int = 41109
    plc_rt_nova2_v: int = 41111
    plc_rt_nova2_w: int = 41113
    plc_rt_nova5_x: int = 41115
    plc_rt_nova5_y: int = 41117
    plc_rt_nova5_z: int = 41119
    plc_rt_nova5_u: int = 41121
    plc_rt_nova5_v: int = 41123
    plc_rt_nova5_w: int = 41125
    plc_rt_epson_x: int = 41127
    plc_rt_epson_y: int = 41129
    plc_rt_epson_z: int = 41131
    plc_rt_epson_u: int = 41133
    plc_rt_epson_tool: int = 41135

    # PC → PLC: heartbeat + ack flags + function codes
    pc_heartbeat: int = 41163
    pc_nova2_send_flag: int = 41164
    pc_nova2_func: int = 41165
    pc_nova5_send_flag: int = 41166
    pc_nova5_func: int = 41167
    pc_epson_send_flag: int = 41168
    pc_epson_func: int = 41169

    # PC → PLC: target coord write area
    pc_nova2_x: int = 41171
    pc_nova2_y: int = 41173
    pc_nova2_z: int = 41175
    pc_nova2_u: int = 41177
    pc_nova2_v: int = 41179
    pc_nova2_w: int = 41181
    pc_nova5_x: int = 41183
    pc_nova5_y: int = 41185
    pc_nova5_z: int = 41187
    pc_nova5_u: int = 41189
    pc_nova5_v: int = 41191
    pc_nova5_w: int = 41193
    pc_epson_x: int = 41195
    pc_epson_y: int = 41197
    pc_epson_z: int = 41199
    pc_epson_u: int = 41201


@dataclass
class Nova5ToEpsonMappingConfig:
    """Nearest-grid XY compensation from Nova5 coordinates to Epson coordinates."""

    enabled: bool = False
    path: str = "config/nova5_to_epson_grid.yaml"


@dataclass
class PlcOrchestratorConfig:
    """Settings for the PLC-driven 3-arm orchestrator side task."""

    enabled: bool = False
    host: str = "192.168.1.88"
    port: int = 502
    unit_id: int = 1
    poll_ms: int = 250
    heartbeat_ms: int = 1000
    word_order: str = "LOW_WORD_FIRST"   # LOW_WORD_FIRST | HIGH_WORD_FIRST
    code_mode: str = "U16_AS_FLOAT"      # U16_AS_FLOAT | REAL32
    compare_epsilon: float = 0.001
    echo_wait_ms: int = 3000
    connect_timeout_s: float = 1.5
    points_path: str = "config/plc_points.yaml"
    start_press_row: int = 1         # 1-based X-direction row in the 10x10 nova2 press matrix
    start_press_index: int = 1       # 1-based sequence within the active sub-matrix
    auto_start: bool = True              # begin loop on attach (vs. wait for API trigger)
    epson_world_pick_timeout_ms: int = 2000  # wait for vision picks after photo-arrived
    # Final per-axis adjustments applied at the very end of _resolve_epson_coord:
    # - epson_offset_x_mm / epson_offset_y_mm: additive bias on top of grid output
    #   (mechanical/optical fine-tuning learned in the field).
    # - epson_z_mm: legacy/common absolute Z override. None → fall through to
    #   plc_points epson_ls6_fallback.z.
    # - epson_tweezer_* / epson_suction_*: tool-specific Z controls. When set,
    #   they override the legacy/common values for the active tool.
    epson_offset_x_mm: float = 0.0
    epson_offset_y_mm: float = 0.0
    epson_z_mm: Optional[float] = None
    epson_tweezer_z_mm: Optional[float] = None
    epson_suction_z_mm: Optional[float] = None
    # Additional Z offsets for retries of the SAME target.
    # Tweezers: retry attempts increase Z (lift).
    # Suction: retry attempts decrease Z (descend).
    epson_z_retry2_offset_mm: float = 0.0
    epson_z_retry3_offset_mm: float = 0.0
    epson_tweezer_z_retry2_offset_mm: float = 0.0
    epson_tweezer_z_retry3_offset_mm: float = 0.0
    epson_suction_z_retry2_offset_mm: float = 0.0
    epson_suction_z_retry3_offset_mm: float = 0.0
    # When PLC reports Epson current tool = 2 (suction head), apply these
    # additive XYZ offsets before sending the target coord.
    epson_tool2_offset_x_mm: float = 0.0
    epson_tool2_offset_y_mm: float = 0.0
    epson_tool2_offset_z_mm: float = 0.0
    # Only use algorithm-derived U inside the arm's reachable rotation window.
    # Outside this range, plc_orchestrator falls back to the fixed YAML U.
    epson_u_min_deg: float = 30.0
    epson_u_max_deg: float = 110.0
    # Post-pick confirmation thresholds used to decide whether the object
    # is still present after Epson reports one pick attempt complete.
    pick_confirm_match_distance_px: float = 30.0
    pick_confirm_match_size_ratio: float = 0.3
    nova5_to_epson_mapping: Nova5ToEpsonMappingConfig = field(
        default_factory=Nova5ToEpsonMappingConfig
    )
    registers: PlcRegisterMap = field(default_factory=PlcRegisterMap)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "PlcOrchestratorConfig":
        data = data or {}
        registers_data = data.get("registers") or {}
        mapping_data = data.get("nova5_to_epson_mapping") or {}
        return cls(
            enabled=bool(data.get("enabled", cls.enabled)),
            host=str(data.get("host", cls.host)),
            port=int(data.get("port", cls.port)),
            unit_id=int(data.get("unit_id", cls.unit_id)),
            poll_ms=int(data.get("poll_ms", cls.poll_ms)),
            heartbeat_ms=int(data.get("heartbeat_ms", cls.heartbeat_ms)),
            word_order=str(data.get("word_order", cls.word_order)),
            code_mode=str(data.get("code_mode", cls.code_mode)),
            compare_epsilon=float(data.get("compare_epsilon", cls.compare_epsilon)),
            echo_wait_ms=int(data.get("echo_wait_ms", cls.echo_wait_ms)),
            connect_timeout_s=float(data.get("connect_timeout_s", cls.connect_timeout_s)),
            points_path=str(data.get("points_path", cls.points_path)),
            start_press_row=int(data.get("start_press_row", cls.start_press_row)),
            start_press_index=int(data.get("start_press_index", cls.start_press_index)),
            auto_start=bool(data.get("auto_start", cls.auto_start)),
            epson_world_pick_timeout_ms=int(
                data.get("epson_world_pick_timeout_ms", cls.epson_world_pick_timeout_ms)
            ),
            epson_offset_x_mm=float(data.get("epson_offset_x_mm", cls.epson_offset_x_mm)),
            epson_offset_y_mm=float(data.get("epson_offset_y_mm", cls.epson_offset_y_mm)),
            epson_z_mm=(
                float(data["epson_z_mm"])
                if data.get("epson_z_mm") is not None
                else cls.epson_z_mm
            ),
            epson_tweezer_z_mm=(
                float(data["epson_tweezer_z_mm"])
                if data.get("epson_tweezer_z_mm") is not None
                else (
                    float(data["epson_z_mm"])
                    if data.get("epson_z_mm") is not None
                    else cls.epson_tweezer_z_mm
                )
            ),
            epson_suction_z_mm=(
                float(data["epson_suction_z_mm"])
                if data.get("epson_suction_z_mm") is not None
                else cls.epson_suction_z_mm
            ),
            epson_z_retry2_offset_mm=float(
                data.get(
                    "epson_z_retry2_offset_mm",
                    cls.epson_z_retry2_offset_mm,
                )
            ),
            epson_z_retry3_offset_mm=float(
                data.get(
                    "epson_z_retry3_offset_mm",
                    cls.epson_z_retry3_offset_mm,
                )
            ),
            epson_tweezer_z_retry2_offset_mm=float(
                data.get(
                    "epson_tweezer_z_retry2_offset_mm",
                    data.get(
                        "epson_z_retry2_offset_mm",
                        cls.epson_tweezer_z_retry2_offset_mm,
                    ),
                )
            ),
            epson_tweezer_z_retry3_offset_mm=float(
                data.get(
                    "epson_tweezer_z_retry3_offset_mm",
                    data.get(
                        "epson_z_retry3_offset_mm",
                        cls.epson_tweezer_z_retry3_offset_mm,
                    ),
                )
            ),
            epson_suction_z_retry2_offset_mm=float(
                data.get(
                    "epson_suction_z_retry2_offset_mm",
                    cls.epson_suction_z_retry2_offset_mm,
                )
            ),
            epson_suction_z_retry3_offset_mm=float(
                data.get(
                    "epson_suction_z_retry3_offset_mm",
                    cls.epson_suction_z_retry3_offset_mm,
                )
            ),
            epson_tool2_offset_x_mm=float(
                data.get(
                    "epson_tool2_offset_x_mm",
                    cls.epson_tool2_offset_x_mm,
                )
            ),
            epson_tool2_offset_y_mm=float(
                data.get(
                    "epson_tool2_offset_y_mm",
                    cls.epson_tool2_offset_y_mm,
                )
            ),
            epson_tool2_offset_z_mm=float(
                data.get(
                    "epson_tool2_offset_z_mm",
                    cls.epson_tool2_offset_z_mm,
                )
            ),
            epson_u_min_deg=float(data.get("epson_u_min_deg", cls.epson_u_min_deg)),
            epson_u_max_deg=float(data.get("epson_u_max_deg", cls.epson_u_max_deg)),
            pick_confirm_match_distance_px=float(
                data.get(
                    "pick_confirm_match_distance_px",
                    cls.pick_confirm_match_distance_px,
                )
            ),
            pick_confirm_match_size_ratio=float(
                data.get(
                    "pick_confirm_match_size_ratio",
                    cls.pick_confirm_match_size_ratio,
                )
            ),
            nova5_to_epson_mapping=Nova5ToEpsonMappingConfig(**mapping_data),
            registers=PlcRegisterMap(**registers_data) if registers_data else PlcRegisterMap(),
        )


@dataclass
class CorsConfig:
    """CORS configuration."""
    allow_origins: List[str] = field(default_factory=lambda: ["*"])
    allow_credentials: bool = True
    allow_methods: List[str] = field(default_factory=lambda: ["*"])
    allow_headers: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class ApiConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    cors: CorsConfig = field(default_factory=CorsConfig)

    @classmethod
    def from_dict(cls, data: dict) -> "ApiConfig":
        """Create ApiConfig from dictionary."""
        cors_data = data.get("cors", {})
        return cls(
            host=data.get("host", "0.0.0.0"),
            port=data.get("port", 8000),
            cors=CorsConfig(**cors_data) if cors_data else CorsConfig(),
        )


@dataclass
class TweezerSettings:
    """Tweezer-tip detection settings (pluck-hair_test addition).

    ``params`` is passed straight through to ``TweezerConfig.from_dict`` so the
    full classical-CV parameter set can be field-tuned from YAML without this
    dataclass having to mirror every knob.
    """
    enabled: bool = False
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Any) -> "TweezerSettings":
        if not data:
            return cls()
        if not isinstance(data, dict):
            raise TypeError("tweezer must be a mapping")
        params = dict(data)
        enabled = bool(params.pop("enabled", True))
        return cls(enabled=enabled, params=params)


@dataclass
class EpsonIoSettings:
    """Pointer to the Epson manual-control + IO communication map (test app)."""
    enabled: bool = False
    path: str = "config/epson_io.yaml"


@dataclass
class AppSettings:
    """Application-level settings."""
    name: str = "pluck-backend"
    version: str = "0.1.0"
    log_level: str = "INFO"


@dataclass
class AppConfig:
    """Main application configuration container.
    
    This is the single source of truth for all configuration.
    Create once at startup and inject into all components.
    
    Example:
        >>> config = AppConfig.from_yaml("config/settings.yaml")
        >>> app = create_app(config)
        >>> storage = create_storage(config)
    """
    app: AppSettings = field(default_factory=AppSettings)
    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    workflow: WorkflowRuntimeConfig = field(default_factory=WorkflowRuntimeConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    video_stream: VideoStreamConfig = field(default_factory=VideoStreamConfig)
    http_image_upload: HttpImageUploadConfig = field(default_factory=HttpImageUploadConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    comm: CommConfig = field(default_factory=CommConfig)
    plc_orchestrator: PlcOrchestratorConfig = field(default_factory=PlcOrchestratorConfig)
    tweezer: TweezerSettings = field(default_factory=TweezerSettings)
    epson_io: EpsonIoSettings = field(default_factory=EpsonIoSettings)

    # Path to the config file (for reference/logging)
    _config_path: Optional[str] = field(default=None, repr=False)

    @classmethod
    def from_dict(cls, data: dict, config_path: Optional[str] = None) -> "AppConfig":
        """Create AppConfig from dictionary.
        
        Args:
            data: Configuration dictionary.
            config_path: Optional path for logging purposes.
            
        Returns:
            Populated AppConfig instance.
        """
        app_data = data.get("app", {})
        camera_data = data.get("camera", {})
        vision_data = data.get("vision", {})
        storage_data = data.get("storage", {})
        scheduler_data = data.get("scheduler", {}) or {}
        workflow_data = data.get("workflow", {}) or {}
        api_data = data.get("api", {})
        video_stream_data = data.get("video_stream", {})
        http_image_upload_data = data.get("http_image_upload", {}) or {}
        redis_data = data.get("redis", {})
        comm_data = data.get("communication") or data.get("comm") or {}
        plc_orchestrator_data = data.get("plc_orchestrator") or {}
        tweezer_data = data.get("tweezer") or {}
        epson_io_data = data.get("epson_io") or {}

        task_config = TaskConfig.from_dict(scheduler_data.get("task"))
        scheduler_data = {**scheduler_data, "task": task_config}

        return cls(
            app=AppSettings(**app_data),
            camera=CameraConfig(**camera_data),
            vision=VisionConfig.from_dict(vision_data),
            storage=StorageConfig.from_dict(storage_data),
            scheduler=SchedulerConfig(**scheduler_data),
            workflow=WorkflowRuntimeConfig(**workflow_data),
            api=ApiConfig.from_dict(api_data),
            video_stream=VideoStreamConfig(**video_stream_data),
            http_image_upload=HttpImageUploadConfig(**http_image_upload_data),
            redis=RedisConfig(**redis_data),
            comm=CommConfig.from_dict(comm_data),
            plc_orchestrator=PlcOrchestratorConfig.from_dict(plc_orchestrator_data),
            tweezer=TweezerSettings.from_dict(tweezer_data),
            epson_io=EpsonIoSettings(**epson_io_data),
            _config_path=config_path,
        )

    @classmethod
    def from_yaml(cls, config_path: str) -> "AppConfig":
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file.
            
        Returns:
            Populated AppConfig instance.
            
        Raises:
            FileNotFoundError: If config file not found.
            yaml.YAMLError: If YAML parsing fails.
        """
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(path) as f:
            content = f.read()
        
        # Substitute environment variables
        content = _substitute_env_vars(content)
        
        data = yaml.safe_load(content) or {}
        
        logger.info(f"Loaded configuration from {config_path}")
        return cls.from_dict(data, config_path=str(path.absolute()))

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (for legacy compatibility)."""
        return {
            "app": {
                "name": self.app.name,
                "version": self.app.version,
                "log_level": self.app.log_level,
            },
            "camera": {
                "type": self.camera.type,
                "device_index": self.camera.device_index,
            "exposure_auto": self.camera.exposure_auto,
            "gain_auto": self.camera.gain_auto,
            "exposure_time": self.camera.exposure_time,
            "gain": self.camera.gain,
            "white_balance_mode": self.camera.white_balance_mode,
            "white_balance_red": self.camera.white_balance_red,
            "white_balance_green": self.camera.white_balance_green,
            "white_balance_blue": self.camera.white_balance_blue,
            "gamma_enable": self.camera.gamma_enable,
            "gamma_value": self.camera.gamma_value,
            "mode": self.camera.mode,
            "image_dir": self.camera.image_dir,
            "width": self.camera.width,
            "height": self.camera.height,
        },
            "vision": {
                "pipeline": {
                    "steps": [
                        {"name": s.name, "type": s.type, "params": s.params}
                        for s in self.vision.steps
                    ]
                }
            },
            "storage": {
                "images": {
                    "type": self.storage.images.type,
                    "endpoint": self.storage.images.endpoint,
                    "access_key": self.storage.images.access_key,
                    "secret_key": self.storage.images.secret_key,
                    "bucket": self.storage.images.bucket,
                    "secure": self.storage.images.secure,
                    "path": self.storage.images.path,
                },
                "database": {
                    "type": self.storage.database.type,
                    "connection_string": self.storage.database.connection_string,
                    "pool_size": self.storage.database.pool_size,
                    "echo": self.storage.database.echo,
                    "path": self.storage.database.path,
                },
            },
            "scheduler": {
                "loop_delay_ms": self.scheduler.loop_delay_ms,
                "max_errors": self.scheduler.max_errors,
                "save_images": self.scheduler.save_images,
                "save_annotated": self.scheduler.save_annotated,
                "show_preview": self.scheduler.show_preview,
                "async_storage": self.scheduler.async_storage,
                "storage_workers": self.scheduler.storage_workers,
                "max_pending_saves": self.scheduler.max_pending_saves,
                "storage_retry_count": self.scheduler.storage_retry_count,
                "task": {
                    "type": self.scheduler.task.type,
                    "name": self.scheduler.task.name,
                    "stabilizer": self.scheduler.task.stabilizer,
                    "pick_process": self.scheduler.task.pick_process,
                },
            },
            "workflow": {
                "enabled": self.workflow.enabled,
                "path": self.workflow.path,
            },
            "api": {
                "host": self.api.host,
                "port": self.api.port,
                "cors": {
                    "allow_origins": self.api.cors.allow_origins,
                    "allow_credentials": self.api.cors.allow_credentials,
                    "allow_methods": self.api.cors.allow_methods,
                    "allow_headers": self.api.cors.allow_headers,
                },
            },
            "video_stream": {
                "enabled": self.video_stream.enabled,
                "stream": self.video_stream.stream,
                "maxlen": self.video_stream.maxlen,
                "fps_limit": self.video_stream.fps_limit,
                "jpeg_quality": self.video_stream.jpeg_quality,
                "block_ms": self.video_stream.block_ms,
            },
            "http_image_upload": {
                "enabled": self.http_image_upload.enabled,
                "url": self.http_image_upload.url,
                "robot": self.http_image_upload.robot,
                "connect_timeout_s": self.http_image_upload.connect_timeout_s,
                "upload_timeout_s": self.http_image_upload.upload_timeout_s,
                "retry_count": self.http_image_upload.retry_count,
                "queue_size": self.http_image_upload.queue_size,
            },
            "redis": {
                "url": self.redis.url,
                "stream": self.redis.stream,
                "consumer_group": self.redis.consumer_group,
                "consumer_name": self.redis.consumer_name,
                "maxlen": self.redis.maxlen,
                "read_count": self.redis.read_count,
                "block_ms": self.redis.block_ms,
                "enabled": self.redis.enabled,
            },
            "communication": {
                "comm_signal": {
                    "type": self.comm.comm_signal.type,
                    "redis": {
                        "command_key": self.comm.comm_signal.redis.command_key,
                        "response_timeout": self.comm.comm_signal.redis.response_timeout,
                    },
                    "modbus": {
                        "host": self.comm.comm_signal.modbus.host,
                        "port": self.comm.comm_signal.modbus.port,
                        "unit_id": self.comm.comm_signal.modbus.unit_id,
                        "timeout": self.comm.comm_signal.modbus.timeout,
                    },
                },
            },
        }


# ============================================================
# Helper Functions
# ============================================================

def _substitute_env_vars(content: str) -> str:
    """Substitute ${VAR_NAME} patterns with environment variable values.
    
    Args:
        content: String content with potential ${VAR} patterns.
        
    Returns:
        String with environment variables substituted.
    """
    pattern = r"\$\{([^}]+)\}"
    
    def replacer(match):
        expression = match.group(1)
        default_value = None
        if ":-" in expression:
            var_name, default_value = expression.split(":-", 1)
        else:
            var_name = expression

        value = os.environ.get(var_name)
        if value not in (None, ""):
            return value
        if default_value is not None:
            return default_value

        logger.warning(f"Environment variable not set: {var_name}")
        return ""
    
    return re.sub(pattern, replacer, content)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file (legacy function).
    
    DEPRECATED: Use AppConfig.from_yaml() instead for type-safe config.
    
    Args:
        config_path: Path to YAML configuration file.
        
    Returns:
        Configuration dictionary.
        
    Raises:
        FileNotFoundError: If config file not found.
        yaml.YAMLError: If YAML parsing fails.
    """
    config = AppConfig.from_yaml(config_path)
    return config.to_dict()


def merge_configs(base: Dict, override: Dict) -> Dict:
    """Deep merge two configuration dictionaries.
    
    Override values take precedence over base values.
    
    Args:
        base: Base configuration.
        override: Override configuration.
        
    Returns:
        Merged configuration.
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def get_config_value(config: Dict, key_path: str, default: Any = None) -> Any:
    """Get a nested configuration value using dot notation.
    
    Args:
        config: Configuration dictionary.
        key_path: Dot-separated path (e.g., "camera.device_index").
        default: Default value if key not found.
        
    Returns:
        Configuration value or default.
    
    Example:
        >>> config = {"camera": {"type": "daheng"}}
        >>> get_config_value(config, "camera.type")
        'daheng'
    """
    keys = key_path.split(".")
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value
