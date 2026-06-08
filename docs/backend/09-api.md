# API 模块 (api/)

> 相关文档：[架构总览](./00-overview.md) | [存储模块](./08-storage.md)

## 概述

API 模块提供 REST API 和 WebSocket 接口，用于前端交互和系统监控。

**技术栈**：FastAPI + Uvicorn

---

## 接口列表

### 健康检查

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/health` | GET | 健康检查（快速响应） |
| `/api/ready` | GET | 就绪检查（检查依赖服务） |

#### GET /api/health

```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### GET /api/ready

```json
{
  "status": "ready",
  "services": {
    "camera": "ok",
    "database": "ok",
    "storage": "ok",
    "ethercat": "unavailable"
  }
}
```

---

### 检测结果查询

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/detections` | GET | 查询检测结果 |
| `/api/detections/{id}` | GET | 获取单条检测 |

#### GET /api/detections

查询参数：

| 参数 | 类型 | 说明 |
|------|------|------|
| `session_id` | `str` | 会话 ID |
| `start_time` | `datetime` | 开始时间 |
| `end_time` | `datetime` | 结束时间 |
| `object_type` | `str` | 对象类型 |
| `min_confidence` | `float` | 最小置信度 |
| `is_stable` | `bool` | 只查询稳定目标 |
| `limit` | `int` | 返回数量（默认 100） |
| `offset` | `int` | 偏移量 |

响应：

```json
{
  "total": 1234,
  "items": [
    {
      "id": "det_abc123",
      "session_id": "session_001",
      "frame_id": 42,
      "timestamp": "2024-01-01T12:00:00Z",
      "bbox": {"x1": 100, "y1": 200, "x2": 150, "y2": 250},
      "object_type": "debris",
      "confidence": 0.95,
      "is_stable": true
    }
  ]
}
```

---

### 图像获取

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/images/{path}` | GET | 获取图像 |

#### GET /api/images/{path}

返回图像文件（JPEG）。

```bash
curl http://localhost:8000/api/images/session_1/frame_0001.jpg -o image.jpg
```

---

### 会话管理

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/sessions` | GET | 查询会话列表 |
| `/api/sessions/{id}` | GET | 获取单个会话 |
| `/api/sessions/{id}/stats` | GET | 获取会话统计 |

#### GET /api/sessions/{id}/stats

```json
{
  "session_id": "session_001",
  "total_frames": 1234,
  "total_detections": 567,
  "duration_seconds": 123.4,
  "avg_fps": 10.0,
  "stable_targets_count": 45
}
```

---

### 系统控制（预留）

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/start` | POST | 启动检测 |
| `/api/stop` | POST | 停止检测 |
| `/api/reset` | POST | 重置系统 |

#### POST /api/reset

```json
{
  "status": "success",
  "message": "System reset successfully"
}
```

---

### Track 管理（v2.3）

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/pick/request-target` | POST | 请求下一个待挑目标（模拟 GIVE_ME_TARGET） |
| `/api/pick/done` | POST | 标记挑取完成（模拟 PICK_DONE） |
| `/api/track/stats` | GET | 获取计数统计 |

#### POST /api/pick/request-target

响应：

```json
{
  "track_id": 1,
  "position": {"x": 100.5, "y": 200.3},
  "confidence": 0.95,
  "category": "initial"
}
```

#### POST /api/pick/done

响应：

```json
{
  "picked_count": 3,
  "message": "3 targets marked as picked"
}
```

#### GET /api/track/stats

```json
{
  "total_picked": 156,
  "current_pending": 12,
  "initial_count": 10,
  "ghost_count": 2
}
```

---

## WebSocket

### 实时状态推送

连接：`ws://localhost:8000/ws`

#### 消息格式

```json
{
  "type": "status_update",
  "timestamp": "2024-01-01T12:00:00Z",
  "data": {
    "frame_id": 1234,
    "detection_count": 5,
    "stable_count": 3,
    "cluster_count": 8,
    "fps": 10.2
  }
}
```

---

## MJPEG 视频流（预留）

### 实时视频流

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/video/stream` | GET | MJPEG 视频流 |

使用 HTML `<img>` 标签直接显示：`<img src="http://localhost:8000/api/video/stream" />`

---

## 配置示例

```yaml
api:
  host: "0.0.0.0"
  port: 8000
  reload: false               # 开发环境用 true
  workers: 1                  # 生产环境用多个 worker
  cors_origins:               # 允许的跨域来源
    - "http://localhost:3000"
    - "http://frontend.local"
```

---

## FastAPI 应用结构

应用使用 FastAPI 框架，按功能分模块路由（health, detections, sessions, track）。

CORS 中间件支持跨域访问。

详见：`src/api/app.py`

---

## 启动命令

### 开发环境

```bash
uvicorn src.api.app:app --reload --port 8000
```

### 生产环境

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 认证与授权（未来）

计划使用 JWT 认证，支持 operator（操作）、viewer（只读）、admin（管理员）三种角色。

---

## 参考

- FastAPI: https://fastapi.tiangolo.com/
- Uvicorn: https://www.uvicorn.org/
- WebSocket: https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API
