# 存储模块 (storage/)

> 相关文档：[架构总览](./00-overview.md) | [任务系统](./03-task-system.md)

## 概述

存储模块负责图像和检测结果的持久化，支持 MinIO/本地存储图像，PostgreSQL/SQLite 存储结构化数据。

**设计决策**：MinIO 存储图像，PostgreSQL 存储结构化数据。

```
┌─────────────────┐         ┌─────────────────┐
│   PostgreSQL    │         │     MinIO       │
│                 │  引用    │                 │
│  • 检测记录     │────────►│  • 原始图像     │
│  • 运行日志     │         │  • 结果图像     │
│  + image_path   │         │  bucket: pluck/ │
└─────────────────┘         └─────────────────┘
```

---

## 图像存储接口

```python
┌─────────────────────────────────┐
│         ImageStorage            │  ◄── 抽象基类
├─────────────────────────────────┤
│ + save(image, path) -> str      │  # 返回完整存储路径
│ + load(path) -> np.ndarray      │
│ + delete(path) -> bool          │
│ + exists(path) -> bool          │
└────────────────┬────────────────┘
                 │ 实现
        ┌────────┴────────┐
        ▼                 ▼
   MinIOStorage      LocalStorage
```

### MinIOStorage

使用 MinIO（S3 兼容）存储图像。

**配置示例**：

```yaml
storage:
  images:
    type: minio
    endpoint: "localhost:9000"
    access_key: "${MINIO_ACCESS_KEY}"
    secret_key: "${MINIO_SECRET_KEY}"
    bucket: "pluck-images"
    secure: false  # 开发环境用 HTTP
```

### LocalStorage

使用本地文件系统存储（开发/测试环境）。

**配置示例**：

```yaml
storage:
  images:
    type: local
    base_dir: "/tmp/pluck_images"
```

---

## 数据库接口

```python
┌─────────────────────────────────────────┐
│              Database                    │  ◄── 抽象基类
├─────────────────────────────────────────┤
│ + save_detection(record) -> str          │
│ + save_detections_batch(records) -> [str]│
│ + get_detection(id) -> DetectionRecord   │
│ + query_detections(filters) -> [records] │
│ + create_session(session) -> None        │
│ + update_session(session) -> None        │
└─────────────────┬───────────────────────┘
                  │ 实现
         ┌────────┴────────┐
         ▼                 ▼
  PostgresDatabase    SQLiteDatabase
```

---

## 数据模型

**DetectionRecord**：存储单个检测结果，包含帧信息、bbox、类别、置信度、所属簇 ID 等。

**SessionRecord**：存储运行会话信息，包含开始/结束时间、帧数、检测数、状态等。

ORM 使用 SQLAlchemy 定义，详见：`src/storage/models.py`

---

## StorageSaver（异步存储）

TaskManager 使用 StorageSaver 进行异步存储，避免阻塞主循环。

```python
┌───────────────────────────────────────────────────────────────────────┐
│                         StorageSaver                                  │
│                                                                       │
│   异步存储处理器，避免阻塞 Vision 主循环                              │
├───────────────────────────────────────────────────────────────────────┤
│ + save(image, result: TaskIterationResult) -> None                   │
│ + flush() -> None              # 等待所有任务完成                     │
│ + get_stats() -> Dict          # 获取存储统计                        │
└───────────────────────────────────────────────────────────────────────┘
```

### 工作原理

```python
class StorageSaver:
    def __init__(self, image_storage, database, max_workers=4):
        self.image_storage = image_storage
        self.database = database
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def save(self, image, result):
        """异步保存（非阻塞）"""
        future = self.executor.submit(self._save_sync, image, result)
        return future

    def _save_sync(self, image, result):
        """同步保存逻辑（在后台线程执行）"""
        # 1. 保存图像
        image_path = self.image_storage.save(
            image, f"{session_id}/frame_{frame_id:06d}.jpg"
        )

        # 2. 保存检测记录
        records = [
            DetectionRecord(
                id=str(uuid.uuid4()),
                session_id=session_id,
                frame_id=frame_id,
                image_path=image_path,
                bbox_x1=det.bbox.x1,
                bbox_y1=det.bbox.y1,
                bbox_x2=det.bbox.x2,
                bbox_y2=det.bbox.y2,
                object_type=det.object_type,
                confidence=det.confidence,
                is_stable=False,
            )
            for det in result.detections
        ]
        self.database.save_detections_batch(records)

        # 3. 保存稳定目标记录（标记 is_stable=True）
        stable_records = [...]
        self.database.save_detections_batch(stable_records)
```

### 使用示例

```python
# TaskManager 中
def _process_frame(self):
    image = self.camera.capture()
    result = self.task.run_iteration(image)

    # 异步存储（不阻塞）
    self.storage_saver.save(image, result)

    # 继续处理下一帧
```

---

## 配置示例

### 完整配置

```yaml
storage:
  # 图像存储
  images:
    type: minio                      # minio | local
    endpoint: "localhost:9000"
    access_key: "${MINIO_ACCESS_KEY}"
    secret_key: "${MINIO_SECRET_KEY}"
    bucket: "pluck-images"
    secure: false

  # 数据库
  database:
    type: postgres                   # postgres | sqlite
    connection_string: "${DATABASE_URL}"
    # 或 SQLite
    # connection_string: "sqlite:///data/pluck.db"

  # 存储策略
  saver:
    max_workers: 4                   # 异步存储线程数
    save_original: true              # 是否保存原始图像
    save_annotated: true             # 是否保存标注图像
    compression_quality: 85          # JPEG 压缩质量
```

### 开发环境（轻量）

```yaml
storage:
  images:
    type: local
    base_dir: "/tmp/pluck_images"

  database:
    type: sqlite
    connection_string: "sqlite:///data/pluck_dev.db"
```

### 生产环境

```yaml
storage:
  images:
    type: minio
    endpoint: "minio.prod.local:9000"
    access_key: "${MINIO_ACCESS_KEY}"
    secret_key: "${MINIO_SECRET_KEY}"
    bucket: "pluck-images"
    secure: true

  database:
    type: postgres
    connection_string: "postgresql://user:pass@postgres.prod.local:5432/pluck"
```

---

## 查询接口

支持按会话、时间范围、对象类型、是否稳定等维度查询检测记录。

详见：`src/storage/database.py`

---

## 数据迁移

使用 Alembic 管理数据库迁移。

初始化：`alembic upgrade head`

---

## 性能优化

- **批量插入**：使用 `save_detections_batch()` 而非逐条插入
- **索引优化**：在 `session_id`、`timestamp`、`is_stable` 等字段建立索引
- **异步写入**：使用 StorageSaver 避免阻塞主循环

---

## Docker Compose（基础设施）

```yaml
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: pluck
      POSTGRES_PASSWORD: pluck123
      POSTGRES_DB: pluck
    ports: ["5432:5432"]
    volumes:
      - postgres_data:/var/lib/postgresql/data

  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    ports: ["9000:9000", "9001:9001"]
    volumes:
      - minio_data:/data

volumes:
  postgres_data:
  minio_data:
```

启动：

```bash
docker-compose up -d
```

---

## 参考

- MinIO: https://min.io/docs/minio/linux/index.html
- SQLAlchemy: https://docs.sqlalchemy.org/
- Alembic: https://alembic.sqlalchemy.org/
