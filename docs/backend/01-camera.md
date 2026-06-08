# 相机模块 (core/camera/)

> 相关文档：[架构总览](./00-overview.md) | [视觉处理](./02-vision-pipeline.md)

## 概述

相机模块提供统一的相机抽象接口，支持大恒工业相机和 Mock 相机（测试用）。

## 接口定义

```python
┌─────────────────────────────────┐
│         CameraBase              │  ◄── 抽象基类
├─────────────────────────────────┤
│ + open() -> bool                │
│ + close() -> None               │
│ + capture() -> np.ndarray       │  # BGR, shape (H, W, C)
│ + is_opened() -> bool           │
│ + get_frame_size() -> (W, H)    │
└────────────────┬────────────────┘
                 │ 实现
        ┌────────┴────────┐
        ▼                 ▼
  DahengCamera        MockCamera
```

### CameraBase 抽象类

所有相机实现必须继承 `CameraBase` 并实现以下方法：

| 方法 | 返回值 | 说明 |
|------|--------|------|
| `open()` | `bool` | 打开相机，返回是否成功 |
| `close()` | `None` | 关闭相机，释放资源 |
| `capture()` | `np.ndarray` | 采集一帧图像（BGR 格式） |
| `is_opened()` | `bool` | 检查相机是否已打开 |
| `get_frame_size()` | `Tuple[int, int]` | 获取图像尺寸 (宽, 高) |

### 图像格式约定

- **颜色空间**：BGR（OpenCV 标准）
- **数据类型**：`np.ndarray`，dtype=`uint8`
- **形状**：`(H, W, 3)` 或 `(H, W)` (灰度)
- **坐标系**：左上角为原点 `(0, 0)`

## 实现类

### 1. DahengCamera（大恒工业相机）

支持大恒 USB3.1 工业相机，使用 `gxipy` SDK。

**关键特性**：
- 支持设置曝光时间、增益
- 支持自动/手动曝光模式
- 连续采集模式

**示例配置**：

```yaml
camera:
  type: daheng
  device_index: 1              # 设备索引（多相机时使用）
  exposure_auto: false         # 是否自动曝光
  exposure_time: 10000         # 曝光时间（微秒）
  gain: 10.0                   # 增益值
```

### 2. MockCamera（测试用）

用于单元测试和无相机环境的开发。

**关键特性**：
- 读取本地图像文件或目录
- 循环播放图像序列
- 可模拟帧率

**示例配置**：

```yaml
camera:
  type: mock
  image_source: "tests/data/test_images/*.jpg"  # 图像路径或 glob 模式
  fps: 10                                        # 模拟帧率
```

**使用场景**：
- 单元测试 Pipeline
- 离线数据回放
- CI/CD 环境（无硬件）

## 配置加载

相机配置通过 `config/settings.yaml` 加载，使用工厂函数根据 `type` 字段创建对应的相机实例。

详见：`src/core/camera/__init__.py:create_camera()`

## 错误处理

### 常见错误

| 错误类型 | 原因 | 解决方案 |
|---------|------|---------|
| `CameraOpenError` | 设备未连接或被占用 | 检查 USB 连接，关闭其他占用相机的程序 |
| `CameraCaptureError` | 采集超时或硬件故障 | 检查曝光时间设置，重启相机 |
| `SDKNotFoundError` | gxipy SDK 未安装 | 安装大恒相机 SDK |

### 重试机制

相机模块内置重试机制（可配置）：

```yaml
camera:
  retry:
    max_attempts: 3           # 最大重试次数
    interval_ms: 1000         # 重试间隔（毫秒）
```

## 与其他模块的集成

### 与 TaskManager 集成

TaskManager 持有 Camera 实例，每帧调用 `capture()` 获取图像并传给 Task 处理。

详见：[任务系统 - TaskManager](./03-task-system.md#taskmanager)

### 与 Pipeline 集成

Pipeline 接收 Camera 输出的 `np.ndarray` 图像（BGR 格式）作为输入。

详见：[视觉处理 - Pipeline](./02-vision-pipeline.md)

## 性能考虑

### 采集频率

- **硬件限制**：大恒相机支持最高 30fps（取决于曝光时间）
- **系统瓶颈**：通常在 GPU 推理，而非相机采集
- **建议配置**：10-15fps 足够（机械臂动作更慢）

### 内存管理

- `capture()` 返回的图像是**新分配的内存**
- TaskManager 需及时处理图像，避免内存堆积
- 异步存储模块会复制图像，原图可释放

## 未来扩展

### 支持更多相机类型

扩展新相机类型：
1. 继承 `CameraBase`
2. 实现 5 个抽象方法
3. 在工厂函数中注册

### 多相机支持（未来）

```yaml
camera:
  type: multi
  cameras:
    - { type: daheng, device_index: 1 }
    - { type: daheng, device_index: 2 }
  sync_mode: hardware  # 硬件触发同步
```
