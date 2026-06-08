# Backend 架构总览

> 本文档提供 Backend 模块的整体架构视图。详细设计请查看各子模块文档。

## 快速导航

- **[相机模块](./01-camera.md)** - 相机抽象与实现
- **[视觉处理](./02-vision-pipeline.md)** - Pipeline 和检测步骤
- **[任务系统](./03-task-system.md)** - Task 抽象和 TaskManager
- **[稳定器](./04-stabilizer.md)** - 跨帧稳定与聚类
- **[Track ID 管理](./05-track-id-management.md)** - 目标计数与生命周期
- **[坐标转换](./06-coordinate-transform.md)** - 像素坐标转物理坐标
- **[通信架构](./07-communication.md)** - PLC/前端/跨进程通信
- **[存储模块](./08-storage.md)** - 图像和数据库存储
- **[API 模块](./09-api.md)** - REST API 和 WebSocket
- **[术语表](./10-glossary.md)** - 关键术语定义

---

## 1. 背景与需求

### 1.1 项目背景

燕窝挑毛系统使用 3 台协作机械臂完成泡发燕窝表面的异物检测和夹取。当前阶段 Backend 作为视觉主控，负责 **视觉检测、坐标定位、现场通信（协议待定）**。

```
当前阶段：Backend 视觉主控 + 现场通信（协议待定）
──────────────────────────────────────────

┌─────────────────────────────────────────────────────────────┐
│                        Backend                               │
│                                                              │
│  • 相机采集           ────────►  坐标/状态  ────────►  现场控制链路         │
│  • YOLO 检测                     (毫米坐标)                   + 机械臂    │
│  • 多帧稳定                                                  │
│  • 坐标转换                                                  │
│  • 数据存储                                                  │
│  • 前端 API                                                  │
└─────────────────────────────────────────────────────────────┘

后期演进：
──────────

┌─────────────────┐          ┌─────────────────┐
│    pluck_ws     │ ◄── 主控  │    Backend      │ ◄── 数据服务
│   (ROS2节点)    │          │                 │
│ • 复用 core/    │ ────────► │ • 存储/API      │
│ • 机械臂控制    │  调用     │                 │
└─────────────────┘          └─────────────────┘
```

### 1.2 功能需求

| 模块 | 需求 | 优先级 |
|------|------|--------|
| **相机采集** | 支持大恒工业相机 (USB3.1, gxipy SDK) | P0 |
| | 支持配置曝光、增益等参数 | P1 |
| **异物检测** | 检测燕窝表面异物 (debris) | P0 |
| | 输出位置、类别、置信度 | P0 |
| | 支持 Pipeline 多步骤处理 | P0 |
| **坐标稳定** | 多帧融合，输出稳定坐标 | P0 |
| | 像素坐标 → 物理坐标转换 | P0 |
| **现场通信** | 与控制端双向通信（协议待定） | P0 |
| | 接收定位请求，返回稳定坐标/状态码 | P0 |
| **数据存储** | 存储原始图像 (MinIO / 本地) | P1 |
| | 存储检测结果 (PostgreSQL / SQLite) | P1 |
| **REST API** | 检测结果查询、图像获取、状态监控 | P1 |
| **前端通信** | WebSocket 状态推送、MJPEG 视频流 | P1 |

### 1.3 非功能需求

| 类别 | 需求 | 指标 |
|------|------|------|
| 稳定性 | 长期运行、异常自恢复 | 7×24 小时 |
| 性能 | 单帧处理延迟 | < 500ms |
| 可扩展 | core/ 模块可被 ROS2 复用 | - |
| 可维护 | 配置外置、日志完整 | - |

---

## 2. 架构设计决策

### 2.1 为什么 Backend 独立于 pluck_ws？

**决策**：Backend 作为独立项目，与 ROS2 工作空间分离。

| 考虑因素 | 分离的好处 |
|----------|------------|
| 关注点分离 | 数据管理 vs 机器人控制，职责清晰 |
| 独立开发测试 | 不需要 ROS 环境即可开发和测试 |
| 部署灵活性 | 可独立部署，或与 ROS 节点运行在不同机器 |

### 2.2 为什么用 Python？

**决策**：当前阶段全部使用 Python 开发。

| 考虑因素 | Python 的优势 |
|----------|---------------|
| 开发效率 | 快速迭代，调试方便 |
| 生态支持 | Ultralytics YOLO、SQLAlchemy、minio-py 原生支持 |
| 性能足够 | 系统瓶颈在机械臂动作（秒级），Python 性能可接受 |

**后期优化路径**：如需更高推理性能，使用 TensorRT + pybind11 封装。

### 2.3 为什么选择 MinIO + PostgreSQL？

**决策**：MinIO 存储图像，PostgreSQL 存储结构化数据。

```
┌─────────────────┐         ┌─────────────────┐
│   PostgreSQL    │         │     MinIO       │
│                 │  引用    │                 │
│  • 检测记录     │────────►│  • 原始图像     │
│  • 运行日志     │         │  • 结果图像     │
│  + image_path   │         │  bucket: pluck/ │
└─────────────────┘         └─────────────────┘
```

**理由**：
- MinIO：S3 兼容、工具丰富、易扩展
- PostgreSQL：支持复杂查询、功能强大
- 开发环境可用 SQLite + 本地文件替代

详见：[存储模块](./08-storage.md)

### 2.4 为什么 Task 主导业务逻辑？

**决策 v2.1**：Task 层持有所有有状态组件（Stabilizer、PickProcess），主导检测→稳定→输出的完整流程。

**设计原则**：
- **Pipeline 无状态**：纯函数式单帧处理，可复用可测试
- **Task 有状态**：持有 Stabilizer 等跨帧状态，负责业务逻辑
- **单向数据流**：Task 计算 → 推送结果 → 前端/PLC 消费
- **职责清晰**：状态管理集中在 Task，通信层纯粹做 I/O

详见：[任务系统](./03-task-system.md)

### 2.5 通信协议的抽象优先

**决策**：去掉 EtherCAT 方案，通信层仅保留抽象接口，具体协议（如 PLC 使用何种现场总线）待定时再新增对应 Bridge，不在架构层写死。

**理由**：

| 方面 | 抽象优先 | 写死具体协议 |
|------|----------|--------------|
| **演进** | 协议确定后只新增实现，不改 TaskManager | 协议变更需改业务层 |
| **测试** | 可用 MockBridge 覆盖 | 受限于具体硬件/协议 |
| **复杂度** | 配置与代码最小化 | 易引入无关字段和耦合 |

详见：[通信架构](./07-communication.md)

---

## 3. 系统架构

### 3.1 架构总览（v2.1）

**核心原则**：Task 主导业务逻辑，单向数据流，前端实时显示稳定目标。

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Python Backend (单进程)                               │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                   Vision Thread (主循环)                                  │ │
│  │                                                                          │ │
│  │  TaskManager (调度器):                                                   │ │
│  │    loop:                                                                 │ │
│  │      image = camera.capture()                                           │ │
│  │      result = task.run_iteration(image)  ◄── Task 主导                  │ │
│  │          ├─> pipeline.run(image) → detections (无状态)                  │ │
│  │          └─> stabilizer.update(detections) → stable_targets (有状态)    │ │
│  │                                                                          │ │
│  │      # 前端显示（减少抖动）                                               │ │
│  │      annotated = draw_boxes(image, result.stable_targets)               │ │
│  │      frame_publisher.publish(annotated)  → MJPEG 流                     │ │
│  │      websocket.send(stable_targets_stats) → 实时统计                     │ │
│  │                                                                          │ │
│  │      # 存储（原始检测 + 稳定目标）                                        │ │
│  │      storage_saver.save(image, detections, stable_targets)              │ │
│  │                                                                          │ │
│  │  职责：相机采集、Task 调度、前端推送、数据存储                            │ │
│  │  资源：GPU + CPU                                                         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │            现场通信（待集成）                                            │ │
│  │                                                                          │ │
│  │  外部控制触发定位 → Comm Bridge → 读取 stable_targets → 写回响应         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 模块职责说明

| 模块 | 职责 | 状态 | 详细文档 |
|------|------|------|---------|
| **Pipeline** | 单帧检测（tile → detect → merge） | 无状态 | [02-vision-pipeline.md](./02-vision-pipeline.md) |
| **Stabilizer** | 跨帧聚合（关联 → 簇管理 → 输出稳定目标） | 有状态 | [04-stabilizer.md](./04-stabilizer.md) |
| **PickProcess** | Track ID 管理（计数、生命周期） | 有状态 | [05-track-id-management.md](./05-track-id-management.md) |
| **Task** | 组合 Pipeline + Stabilizer，每帧输出 | 有状态 | [03-task-system.md](./03-task-system.md) |
| **TaskManager** | 调度循环、前端推送、存储 | 无状态 | [03-task-system.md](./03-task-system.md#taskmanager) |
| **通信** | PLC/前端/跨进程通信 | - | [07-communication.md](./07-communication.md) |

### 3.3 核心数据流

```
【主循环流程】

TaskManager.loop (Vision Thread):
    │
    ├─> camera.capture() → image
    │
    ├─> task.run_iteration(image)
    │     ├─> pipeline.run(image) → detections (原始，每帧波动)
    │     └─> stabilizer.update(detections) → stable_targets (稳定，减少抖动)
    │
    ├─> draw_boxes(image, stable_targets) → annotated_image
    │
    ├─> frame_publisher.publish(annotated_image) → 前端 MJPEG 流
    │
    ├─> websocket.send({
    │       detection_count: len(detections),
    │       stable_count: len(stable_targets),
    │       cluster_count: stabilizer.get_cluster_count()
    │   }) → 前端实时统计
    │
    └─> storage_saver.save(image, detections, stable_targets) → 存储
```

### 3.4 目录结构

```
backend/
├── src/
│   ├── core/                       # 核心算法（可被 ROS2 复用）
│   │   ├── camera/                 # 相机抽象和实现
│   │   └── vision/                 # 视觉处理
│   │
│   ├── tasks/                      # Task 层（业务任务 + 业务状态）
│   │   ├── pick_task.py            # 燕窝挑毛 Task（Pipeline + Stabilizer + PickProcess）
│   │   ├── stabilizer.py           # 多帧稳定
│   │   └── pick_process.py         # Track ID 管理
│   │
│   ├── scheduler/                  # 调度模块（过渡期）
│   │   └── task_manager.py         # TaskManager 主循环
│   ├── workflow/                   # Workflow 层（编排/引擎，占位演进）
│   │
│   ├── storage/                    # 存储模块
│   ├── api/                        # REST API（前端通信）
│   ├── comm/                       # 通信模块（外部通信/Redis/WebSocket）
│   └── config.py                   # 配置加载
│
├── config/                         # 配置文件
├── assets/                         # 模型和资源文件
├── tests/                          # 测试代码
└── main.py                         # 主程序入口
```

### 3.5 开发运行（dev）

使用 `pyproject.toml` 的脚本入口（只面向 dev）：

```
dev-run   # 等价于 python main.py --mode run --config ./config/settings.dev.yaml
dev-api   # 等价于 python main.py --mode api --config ./config/settings.dev.yaml
```

---

## 4. 演进路线

### 当前阶段（v2.3）
- ✅ Pipeline 单帧检测
- ✅ Stabilizer 跨帧稳定
- ✅ Track ID 管理（初始化判定、Ghost 处理）
- ⏳ 前端显示与控制信号集成

### 下一步（外部通信集成）
- 与控制端确认协议/载荷格式
- 按协议实现对应 Bridge，验证单请求-响应
- 联调超时/看门狗策略

### 未来扩展
- Grip 阶段状态管理（locked/ACK）
- 多任务 Workflow 编排
- TensorRT 推理加速

详见：[配置部署](./10-config-deployment.md#演进路线)

---

## 架构要点

- **Task 主导**：持有所有有状态组件，完整控制业务流程
- **单向数据流**：Task → stable_targets → 前端/存储/通信链路
- **Pipeline 无状态**：纯函数式，可复用可测试
- **双重稳定性保护**：min_frames_to_stable + 双阈值滞后
- **Track ID 分级**：Initial（高可信）/ Ghost（低可信）
- **控制信号驱动**：机械臂触发检测，避免误计数
- **DDD 限界上下文**：Stabilizer（技术域）与 PickProcess（业务域）职责分离

详见：[术语表](./11-glossary.md)
