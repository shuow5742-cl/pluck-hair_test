# 视觉处理模块 (core/vision/)

> 相关文档：[架构总览](./00-overview.md) | [相机模块](./01-camera.md) | [任务系统](./03-task-system.md)

## 概述

Vision 模块提供**无状态的单帧图像处理**能力，采用可配置的 Pipeline 架构。

**核心特性**：
- ✅ **无状态**：纯函数式，每帧独立处理
- ✅ **可配置**：通过 YAML 组合不同步骤
- ✅ **可复用**：可被 ROS2 节点复用
- ✅ **可测试**：易于单元测试

## 数据类型定义

### BoundingBox

| 字段 | 类型 | 说明 |
|------|------|------|
| `x1, y1` | `float` | 左上角坐标 |
| `x2, y2` | `float` | 右下角坐标 |
| `center` | 属性 | 中心点坐标 |
| `width, height` | 属性 | 宽度和高度 |

### Detection

| 字段 | 类型 | 说明 |
|------|------|------|
| `bbox` | `BoundingBox` | 边界框 |
| `object_type` | `str` | 类别（如 "debris"） |
| `confidence` | `float` | 置信度 [0, 1] |
| `detection_id` | `str` | 唯一标识（UUID） |

**使用场景**：
- Pipeline 的输出结果
- Stabilizer 的输入数据（详见 [Stabilizer](./04-stabilizer.md)）

### PipelineContext

用于 Pipeline 步骤间传递数据的上下文对象，包含原始/处理后图像、检测结果和元数据。

### PipelineResult

Pipeline 最终输出，包含检测结果列表、处理时间和调试元数据。

---

## Pipeline 接口

```python
┌───────────────────────────────────────────────────────────────────────┐
│                         VisionPipeline                                │
│                                                                       │
│   特性：无状态，纯函数式，可配置的步骤组合                             │
├───────────────────────────────────────────────────────────────────────┤
│ + add_step(step: ProcessStep) -> VisionPipeline                       │
│ + run(image: np.ndarray) -> PipelineResult                            │
│ + from_config(config: dict) -> VisionPipeline                         │
│ + clear() -> None                                                     │
└───────────────────────────────────────────────────────────────────────┘
```

### 核心方法

| 方法 | 说明 |
|------|------|
| `add_step(step)` | 添加处理步骤（链式调用） |
| `run(image)` | 执行 Pipeline，返回检测结果 |
| `from_config(config)` | 从配置文件创建 Pipeline |
| `clear()` | 清空所有步骤 |

---

## ProcessStep 抽象

```python
┌───────────────────────────────────┐
│           ProcessStep             │  ◄── 抽象基类
├───────────────────────────────────┤
│ + name: str                       │
│ + process(ctx) -> ctx             │
└───────────────────────────────────┘
```

所有步骤必须实现：
- `name`：步骤名称（用于日志）
- `process(ctx)`：处理逻辑，接收并返回 `PipelineContext`

**设计原则**：
- **输入输出一致**：都是 `PipelineContext`
- **不修改原图**：需要时复制图像
- **异常处理**：抛出明确的异常，由 Pipeline 捕获

---

## 内置处理步骤

### 1. TileStep（图像切片）

将大图切分成小块（tiles），用于处理高分辨率图像。

**参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `tile_size` | `int` | 640 | 切片尺寸（正方形） |
| `overlap` | `float` | 0.2 | 重叠比例 [0, 1) |

**工作原理**：

```
原图 (1920x1080) → 切片 (640x640, overlap=0.2)

┌─────────────────────────────┐
│  ┌───┐                       │
│  │ 1 │──┐                    │
│  └───┘  │  ┌───┐             │
│         └──│ 2 │──┐          │
│            └───┘  │  ┌───┐   │
│                   └──│ 3 │   │
│                      └───┘   │
└─────────────────────────────┘
```

**配置示例**：

```yaml
- name: tile
  type: tile
  params:
    tile_size: 640
    overlap: 0.2
```

### 2. YOLODetectStep（YOLO 检测）

使用 Ultralytics YOLO 模型进行目标检测。

**参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | `str` | - | 模型路径（.pt 或 .engine） |
| `conf` | `float` | 0.25 | 置信度阈值 |
| `device` | `str` | "cuda" | 设备（cuda/cpu） |
| `batch_size` | `int` | 1 | 批量推理大小 |

**配置示例**：

```yaml
- name: detect
  type: yolo
  params:
    model: assets/best.engine  # TensorRT 加速
    conf: 0.25
    device: cuda
    batch_size: 16             # 多 tile 批量推理
```

### 3. MergeTilesStep（合并切片结果）

将多个 tile 的检测结果合并，去除重复检测。

**参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `iou_threshold` | `float` | 0.5 | IoU 阈值，超过则认为重复 |

**工作原理**：

```
Tile 1: [det_a, det_b]    原图坐标: (10,10), (30,30)
Tile 2: [det_c, det_d]               (25,25), (50,50)

IoU(det_b, det_c) = 0.6 > threshold → 保留置信度更高的
```

**算法**：
1. 将所有 tile 的检测转换回原图坐标
2. 按置信度降序排序
3. 使用 NMS（非极大值抑制）去重

**配置示例**：

```yaml
- name: merge
  type: merge_tiles
  params:
    iou_threshold: 0.5
```

### 4. NMSStep（非极大值抑制）

对同一图像的检测结果去重（不同于 MergeTiles，这是单图 NMS）。

**参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `iou_threshold` | `float` | 0.5 | IoU 阈值 |

**使用场景**：
- 不使用 tile 时的去重
- MergeTiles 后的二次过滤

**配置示例**：

```yaml
- name: nms
  type: nms
  params:
    iou_threshold: 0.45
```

### 5. FilterStep（结果过滤）

根据条件过滤检测结果。

**参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `min_confidence` | `float` | 最小置信度 |
| `classes` | `List[str]` | 保留的类别列表 |
| `min_area` | `float` | 最小面积（像素²） |

**配置示例**：

```yaml
- name: filter
  type: filter
  params:
    min_confidence: 0.3
    classes: ["debris"]        # 只保留 debris 类别
    min_area: 100              # 过滤太小的框
```

### 6. SortStep（结果排序）

对检测结果排序（用于优先级挑取）。

**参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `by` | `str` | "confidence" | 排序字段（confidence/area/position） |
| `ascending` | `bool` | False | 是否升序 |

**配置示例**：

```yaml
- name: sort
  type: sort
  params:
    by: confidence
    ascending: false  # 置信度从高到低
```

---

## 配置约定

### 完整示例

```yaml
vision:
  pipeline:
    steps:
      # Step 1: 切片
      - name: tile
        type: tile
        params:
          tile_size: 640
          overlap: 0.2

      # Step 2: YOLO 检测
      - name: detect
        type: yolo
        params:
          model: assets/best.engine
          conf: 0.25
          device: cuda
          batch_size: 16

      # Step 3: 合并切片
      - name: merge
        type: merge_tiles
        params:
          iou_threshold: 0.5

      # Step 4: 过滤
      - name: filter
        type: filter
        params:
          min_confidence: 0.3
          classes: ["debris"]

      # Step 5: 排序
      - name: sort
        type: sort
        params:
          by: confidence
          ascending: false
```

---

## 与其他模块的集成

### 与 Task 集成

Task 通过构造函数注入 Pipeline 实例，每帧调用 `pipeline.run(image)` 获取检测结果，再传给 Stabilizer 进行跨帧稳定。

详见：[任务系统 - Task](./03-task-system.md)

### 输出到 Stabilizer

Pipeline 输出的 `Detection` 列表直接传给 Stabilizer 的 `update()` 方法。

详见：[Stabilizer](./04-stabilizer.md)

---

## 性能优化

### 1. TensorRT 加速

将 PyTorch 模型转换为 TensorRT：

```bash
# 导出 ONNX
yolo export model=best.pt format=onnx

# 转换为 TensorRT
trtexec --onnx=best.onnx --saveEngine=best.engine --fp16
```

配置中使用 `.engine` 文件：

```yaml
params:
  model: assets/best.engine  # 自动识别 TensorRT
```

### 2. Batch 推理

多个 tile 批量推理：

```yaml
params:
  batch_size: 16  # 16 个 tile 一批
```

**注意**：需要显存足够（约 4GB for batch=16）

### 3. 半精度推理

```yaml
params:
  model: assets/best.engine
  half: true  # FP16
```
