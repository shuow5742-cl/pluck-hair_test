# 任务系统 (tasks/)

> 相关文档：[架构总览](./00-overview.md) | [视觉处理](./02-vision-pipeline.md) | [Stabilizer](./04-stabilizer.md) | [Workflow](./03a-workflow.md)

## 概述

Task 模块是**有状态的业务逻辑层**，负责消费 Pipeline 产出或其他 Task 的结果，完成跨帧累积、业务决策等有状态操作。

**核心原则**：
- **Pipeline 无状态**：纯函数式单帧处理，可复用可测试
- **Task 有状态**：持有跨帧状态，负责业务逻辑
- **事件驱动产出**：Task 通过 EventBus broadcast 结果，不通过返回值
- **上层调动下层**：Workflow 编排 Task，Task 调用 Pipeline，各层自由组合

---

## 三层可组合架构

系统采用三层可组合架构，每层职责清晰、可独立演进：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        三层可组合架构                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  【Workflow 层】流程编排                                                 │
│                                                                          │
│         ┌──────┐                                                        │
│         │Task A│                                                        │
│         └───┬──┘                                                        │
│             │                                                           │
│             ▼                                                           │
│         ┌──────┐      ┌──────┐                                         │
│         │Task B│─────▶│Task C│  (顺序/分支/并行)                       │
│         └──────┘      └──────┘                                         │
│             │                                                           │
│             ▼                                                           │
│         ┌──────┐                                                        │
│         │Task D│                                                        │
│         └──────┘                                                        │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  【Task 层】业务逻辑封装                                                │
│                                                                          │
│    ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐              │
│    │Task A│   │Task B│   │Task C│   │Task D│   │Task E│  ...          │
│    └──────┘   └──────┘   └──────┘   └──────┘   └──────┘              │
│                                                                          │
│    每个 Task 独立、可组合：                                              │
│    • DetectionTask = Pipeline + 检测逻辑                                │
│    • StabilizationTask = Stabilizer + 稳定逻辑                          │
│    • PickTask = PickProcess + 挑取逻辑                                  │
│    • AlarmTask = 阈值判定 + 报警逻辑                                    │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  【Pipeline/基础设施层】技术能力积木                                     │
│                                                                          │
│    Pipeline (YAML 配置驱动):                                            │
│    ┌─────┐  ┌─────┐  ┌─────┐  ┌──────┐  ┌──────┐                     │
│    │ Tile│─▶│ YOLO│─▶│Merge│─▶│Filter│─▶│ Sort │                     │
│    └─────┘  └─────┘  └─────┘  └──────┘  └──────┘                     │
│                                                                          │
│    基础设施:                                                             │
│    ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐                    │
│    │Camera  │  │Storage │  │Database│  │Control │                    │
│    │        │  │        │  │        │  │Signal  │                    │
│    └────────┘  └────────┘  └────────┘  └────────┘                    │
│                                                                          │
│    能力库: 可复用的技术积木，通过配置自由组合                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 设计要点

| 层级 | 职责 | 特点 |
|------|------|------|
| **Workflow 层** | 流程编排 | 组合多个 Task，支持顺序/分支/并行 |
| **Task 层** | 业务逻辑封装 | 独立、可组合、有状态 |
| **Pipeline/基础设施层** | 技术能力积木 | 无状态、可复用、配置驱动 |

### 层间边界

| 边界 | 接口 | 原则 |
|------|------|------|
| Workflow → Task | `run(data)` / `attach` / `reset` / `close` | Workflow 只管调度，不关心 Task 内部 |
| Task → Pipeline | `pipeline.run(image) → detections` | Task 自由调用 Pipeline，Pipeline 不知道 Task 的存在 |
| Task ↔ Task | EventBus broadcast / subscribe | 事件驱动，不直接引用 |

### Task 可组合性

Task 是独立的业务单元，可以自由组合：

```
示例 1：燕窝挑毛 Workflow（task 链）
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Detection    │──▶│ Stabilization│──▶│ Pick         │
│ Task         │   │ Task         │   │ Task         │
└──────────────┘   └──────────────┘   └──────────────┘
     broadcast          broadcast          broadcast
     detections      stable_targets     pick_result
                  ┌──────────────┐
                  │Communication │  (SideTask, 始终存活)
                  │ Task         │
                  └──────────────┘

示例 2：简单检测 Workflow
┌──────────────┐   ┌──────────────┐
│ Detection    │──▶│ Storage      │
│ Task         │   │ Task         │
└──────────────┘   └──────────────┘

示例 3：缺陷检测 + 报警
┌──────────────┐   ┌──────────────┐
│ Detection    │──▶│ Alarm        │
│ Task         │   │ Task         │
└──────────────┘   └──────────────┘
```

---

## TaskBase 抽象

### 设计思想

Pipeline 层有 `ProcessStep` 作为明确的抽象基类，开发者继承它就知道该实现什么。Task 层同样需要一个 `TaskBase`，提供两个构建块和一个 Engine 接口，引导开发者组织业务逻辑：

```
┌───────────────────────────────────────────────────────────────────┐
│                          TaskBase                                  │
│                                                                    │
│   构建块（子类按需使用）：                                          │
│                                                                    │
│   subscribe()    订阅 EventBus 事件                                │
│   broadcast()    通过 EventBus 发布结果                            │
│                                                                    │
│   对外接口（Engine 调用）：                                         │
│                                                                    │
│   run(data)      Engine 每帧调用，执行业务逻辑或触发内部流程         │
│   attach()       注入 EventBus，触发 subscribe                     │
│   reset()        重置有状态组件                                     │
│   close()        清理资源，取消订阅                                 │
│                                                                    │
└───────────────────────────────────────────────────────────────────┘
```

### 关键设计决策

**run(data) 而非 run(image)**：

`run` 的参数是泛化的 `data`，不绑定为 image。这使得 Task 可以串联：
- 有 Pipeline 的 Task：data 是 image，Task 调用 `pipeline(image)` 处理
- 纯消费型 Task：data 是上游 Task 的结果（通过 Engine 文件柜传递）

**无返回值**：

`run` 不返回任何东西。所有 Task 产出通过 `broadcast` 发到 EventBus。Engine 不消费 Task 结果，只做调度。这保持了限界上下文的干净——Engine 是调度器，不是数据消费者。

**基类给能力，不给流程**：

TaskBase 提供 `subscribe()` 和 `broadcast()` 两个构建块，以及 `run(data)` 这个 Engine 调用入口，但不规定内部流程。这保证了灵活性——不同类型的 Task 可以把业务逻辑放在 `run` 中，也可以放在事件回调中。

### 使用示例

```python
# 有 Pipeline 的 Task
class DetectionTask(TaskBase):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def run(self, data):
        result = self.pipeline(data)          # data 是 image
        self.broadcast("TASK:DETECTION_DONE", {
            "detections": result.detections
        })

# 纯消费型 Task（消费其他 Task 的结果）
class AlarmTask(TaskBase):
    def subscribe(self):
        self._event_bus.subscribe("TASK:DETECTION_DONE", self._on_detection)

    def _on_detection(self, event, data):
        detections = data["payload"]["detections"]
        if len(detections) > self.threshold:
            self.broadcast("TASK:ALARM", {"count": len(detections)})

    def run(self, data):
        pass  # 不需要 Engine 每帧驱动，纯事件驱动

# 组合型 Task（Pipeline + 业务逻辑）
class StabilizationTask(TaskBase):
    def __init__(self, stabilizer):
        self.stabilizer = stabilizer

    def subscribe(self):
        self._event_bus.subscribe("TASK:DETECTION_DONE", self._on_detections)

    def _on_detections(self, event, data):
        detections = data["payload"]["detections"]
        stable_targets = self.stabilizer.update(detections)
        self.broadcast("TASK:STABILIZATION_DONE", {
            "stable_targets": stable_targets
        })

    def run(self, data):
        pass  # 事件驱动
```

---

## Task 间数据流转

### 并发场景（SideTask ↔ MainTask）

SideTask 和 MainTask 同时存活，EventBus 实时分发，无时序问题：

```
CommunicationTask ──broadcast──→ EventBus ──subscribe──→ MainTask
```

### 接力场景（MainTask A → MainTask B）

MainTask 是串行的，同一时刻只有一个。Task A broadcast 时 Task B 可能尚未 attach，事件会丢失。

**解决方案：Engine 文件柜（Handoff）**

Engine 作为调度器，在 Task 切换时暂存上一个 Task 的产出，传递给下一个 Task：

```
Task A 运行中
    → broadcast 结果
    → Engine 订阅并暂存到文件柜
    → broadcast TASK:DONE

Engine 切换 Task
    → close Task A
    → attach Task B
    → run(文件柜中的数据) 传递给 Task B

Task B
    → run(data) 中 data 包含 Task A 的产出
    → 按需使用或忽略
```

**领域解读**：文件柜是 Engine 的自然职责延伸——Engine 已经管理 Task 的生命周期切换（close A → attach B），在切换时顺手传递数据是内聚的。

---

## Task 分类

### MainTask vs SideTask

| 维度 | MainTask | SideTask |
|------|----------|----------|
| 生命周期 | 随状态机切换（close → attach → reset） | 始终存活，贯穿整个 Engine 会话 |
| 驱动方式 | Engine 每帧调用 `run(data)` | 内部自治（事件驱动/轮询/线程） |
| 数量 | 同一时刻一个 | 任意多个并行 |
| 与 Engine 的接口 | `attach` / `run` / `reset` / `close` | 仅 `attach` / `close` |
| 典型用途 | 检测、稳定化、业务决策 | 通信（PLC）、监控、日志 |

### SideTask Protocol

SideTask 对 Engine 只暴露 attach/close，内部执行模型自治（DDD 边界）：

```python
class SideTask(Protocol):
    @property
    def name(self) -> str: ...
    def attach(self, event_bus: EventBus) -> None: ...
    def close(self) -> None: ...
```

详见：[Workflow 架构](./03a-workflow.md#5-并行模型maintask--sidetask)

---

## DoneCondition（完成条件）

Task 完成时通过 broadcast `TASK:DONE` 事件通知 StateMachine，触发状态转换。

```
┌───────────────────────────────────────────────────────────────────────┐
│                        DoneCondition                                  │
│                                                                       │
│   任务完成条件判定接口                                                 │
├───────────────────────────────────────────────────────────────────────┤
│ + check(stats, ...) -> bool      # 检查是否满足完成条件               │
│ + reset() -> None                # 重置条件状态                       │
└───────────────────────────────────────────────────────────────────────┘
```

### 内置实现

| 条件 | 说明 | 参数 |
|------|------|------|
| `ConsecutiveEmptyFrames` | 连续 N 帧无检测则完成 | `n` |
| `MaxIterations` | 达到最大迭代次数 | `max_iter` |
| `Timeout` | 超时 | `seconds` |
| `Composite` | 组合条件（任一满足即完成） | `conditions` |

**配置示例**：

```yaml
task:
  done_condition:
    consecutive_empty: 3     # 连续 3 帧无检测
    max_iterations: 1000     # 或达到 1000 帧
    # timeout_seconds: 60    # 或超时 60 秒
```

---

## TaskStats（统计）

```
┌───────────────────────────────────────────────────────────────────────┐
│                          TaskStats                                    │
│                                                                       │
│   任务运行统计                                                         │
├───────────────────────────────────────────────────────────────────────┤
│ + total_frames: int                # 总帧数                           │
│ + total_detections: int            # 总检测数                         │
│ + start_time: datetime             # 开始时间                         │
│ + end_time: datetime               # 结束时间                         │
├───────────────────────────────────────────────────────────────────────┤
│ + record(...) -> None              # 记录一次迭代结果                  │
│ + summary() -> Dict                # 获取统计摘要                     │
│ + reset() -> None                  # 重置统计                         │
└───────────────────────────────────────────────────────────────────────┘
```

---

## WorkflowEngine（调度器）

> 历史名称：TaskManager → WorkflowEngine

```
┌───────────────────────────────────────────────────────────────────────┐
│                       WorkflowEngine                                  │
│                                                                       │
│   主循环调度器                                                         │
│   职责：相机采集、调用 Task、Task 切换、文件柜管理                     │
├───────────────────────────────────────────────────────────────────────┤
│ - camera: CameraBase                                                  │
│ - state_machine: StateMachine      # 状态机                          │
│ - task_map: Dict[str, Task]        # state → task 映射               │
│ - side_tasks: List[SideTask]       # 常驻 SideTask                   │
│ - _handoff: Dict                   # 文件柜（Task 间数据传递）        │
├───────────────────────────────────────────────────────────────────────┤
│ + start() -> None                                                     │
│ + stop() -> None                                                      │
│ + is_running: bool                                                    │
└───────────────────────────────────────────────────────────────────────┘
```

### 主循环流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                    主循环 _process_frame()                            │
│                                                                      │
│   1. camera.capture() ──────────────────► image                     │
│                                              │                       │
│   2. task.run(data) ◄────────────────────────┘                      │
│      data = image 或 文件柜内容                                      │
│      Task 在 run() 或事件回调中执行业务逻辑                          │
│                                                                      │
│   3. Task broadcast 结果到 EventBus                                  │
│      → 前端推送（订阅者）                                             │
│      → 存储（订阅者）                                                 │
│      → Engine 文件柜暂存（用于 Task 切换传递）                        │
│      → StateMachine（TASK:DONE 触发状态转换）                        │
│                                                                      │
│   横切关注点（存储、前端推送等）通过 EventBus 订阅消费                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 与其他模块的集成

### 与 Camera 集成

WorkflowEngine 持有 Camera 实例，每帧调用 `capture()` 获取图像，作为 `data` 传给当前 Task。

详见：[相机模块](./01-camera.md)

### 与 Pipeline 集成

Pipeline 通过构造函数注入给需要它的 Task。Task 在 `run` 内部自由调用 `pipeline.run(image)`。Pipeline 由外部创建并注入，避免 GPU 模型重复加载。

详见：[视觉处理](./02-vision-pipeline.md)

### 与 Stabilizer 集成

需要跨帧稳定的 Task 内部持有 Stabilizer 实例，在 `run` 或事件回调中调用 `stabilizer.update(detections)` 进行跨帧稳定。

详见：[Stabilizer](./04-stabilizer.md)

### 与 EventBus 集成

Task 通过 `attach(event_bus)` 获得 EventBus 引用。`subscribe` 订阅事件，`broadcast` 发布结果。所有 Task 间通信和横切关注点（存储、前端推送）都通过 EventBus。

详见：[Workflow 架构](./03a-workflow.md)

### 与 Storage 集成

StorageSaver 作为 EventBus 的订阅者，监听 Task 的 broadcast 事件，异步存储图像和检测结果。

详见：[存储模块](./08-storage.md)

---

## 会话管理

### Session 概念

一次完整的运行周期称为一个 **Session**，从 `WorkflowEngine.start()` 开始到任务完成或手动停止结束。

Session 包含运行的元数据（ID、时间、帧数、状态等），用于前端查询历史记录、数据分析和调试。

---

## Workflow/Task 分层设计

### 业务场景：多区域扫描

一个燕窝盘通常分为多个区域（如 20 个区域），系统按顺序扫描每个区域。每个区域作为独立的 Task，全盘由 Workflow 协调。

**关键特性**：
- **全盘累计**：`total_picked` 跨区域累加（Workflow 层维护）
- **区域隔离**：每个区域是独立 Task，换区域时销毁旧 Task、创建新 Task
- **状态隔离**：区域 A 的状态不会污染区域 B

### 架构分层

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Workflow 层（盘子级，长生命周期）                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  职责：                                                                  │
│    - 管理多区域扫描流程                                                   │
│    - 累加全盘计数（total_picked）                                         │
│    - 创建和销毁 Task                                                     │
│                                                                          │
│  生命周期：整个盘子完成前一直存在                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ 创建/销毁
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Task 层（区域级，短生命周期）                                             │
├─────────────────────────────────────────────────────────────────────────┤
│  职责：                                                                  │
│    - 管理单区域的检测和业务决策                                           │
│    - 通过 broadcast 发布结果                                             │
│                                                                          │
│  生命周期：随区域创建/销毁                                                │
│  作用域：不知道"盘子"概念，只关心当前区域                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 为什么要分层？

| 问题 | 不分层的后果 | 分层的好处 |
|------|------------|-----------|
| **生命周期冲突** | 全盘计数和区域计数生命周期不同，难以重置 | 各层管理各自生命周期，职责清晰 |
| **状态污染** | 区域 A 的状态残留影响区域 B | 每个区域是全新 Task，状态完全隔离 |
| **可测试性** | Task 依赖全局状态，难以单元测试 | Task 可独立测试，不依赖 Workflow |
| **可复用性** | Task 耦合多区域逻辑，无法复用 | Task 可被不同 Workflow 复用 |

---

## 实现状态

### 已完成

- [x] Task Protocol 定义（`_protocol.py`）
- [x] SideTask Protocol 定义
- [x] TaskEventHelper 组合件
- [x] DetectionTask 实现
- [x] StabilizedDetectionTask 实现（当前为单体，待拆分）
- [x] CommunicationTask 事件驱动迁移（SideTask）
- [x] DoneCondition 接口和内置实现
- [x] TaskStats 统计
- [x] Pipeline 抽离到 autoweaver 框架

### 待完成

- [ ] **TaskBase 抽象基类**：提供 `subscribe()` / `broadcast()` 构建块，以及 `run(data)` / `attach()` / `reset()` / `close()` 接口
- [ ] **run(data) 接口**：替换 `run_iteration(image) → TaskIterationResult`，无返回值
- [ ] **TaskIterationResult 移除**：所有产出走 EventBus broadcast
- [ ] **StabilizedDetectionTask 拆分**：拆为 DetectionTask → StabilizationTask → PickTask 链
- [ ] **Engine 文件柜**：实现 Task 切换时的数据传递机制
- [ ] **Task 注册开放**：`register_task()` 模式，类似 Pipeline 的 `register_step()`
