# Workflow 架构设计

> 相关文档：[架构总览](./00-overview.md) | [任务系统](./03-task-system.md) | [通信架构](./07-communication.md)

---

## 1. 概述

Workflow 层是 **Task 编排层**，负责定义 Task 之间的流转关系、触发条件和数据传递。

**核心思想**：
- **Task 自治**：每个 Task 独立运行，不直接调用其他 Task
- **事件驱动**：Task 通过 EventBus 发布/订阅事件进行通信
- **状态约束**：StateMachine 定义合法的状态流转路径

**技术选型**：
- 状态机：内置轻量 `StateMachine`（当前实现）
- 发布订阅：内置轻量 `EventBus`（当前实现）
- 配置：YAML → dict
- 演进策略：先用轻量实现，未来如需 hierarchical states/guards/callbacks 再按需引入框架（如 transitions、blinker）

---

## 2. 架构总图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         WorkflowEngine                               │
│                                                                      │
│  ┌─────────────────┐              ┌─────────────────────────────┐   │
│  │  StateMachine   │◄────────────►│         EventBus            │   │
│  │  (lightweight)  │  订阅/发布    │        (lightweight)        │   │
│  │  当前状态: X    │              │  publish(event, **payload)  │   │
│  └─────────────────┘              └──────────────┬──────────────┘   │
│                                                  │                   │
│                                   ┌──────────────┼──────────────┐   │
│                                   │              │              │   │
│                                   ▼              ▼              ▼   │
│                               Task A         Task B    ...    Task N│
│                             (订阅/发布)     (订阅/发布)      (订阅/发布)│
└─────────────────────────────────────────────────────────────────────┘
```
**说明**：架构不限制 Task 数量，业务可根据需要注册任意多个 Task。

---

## 3. 核心组件

### 3.1 职责划分

| 组件 | 职责 | 不做什么 |
|------|------|----------|
| **StateMachine** | 定义状态、监听事件、决定状态流转 | 不执行业务逻辑，不直接调用 Task |
| **EventBus** | 广播/订阅中心，解耦 Task 之间的通信，携带数据 | 不持有状态，不做业务判断 |
| **Task** | 执行具体业务逻辑，发布事件，订阅感兴趣的事件 | 不直接调用其他 Task |
| **Engine** | 启动/停止，资源管理，主循环驱动 | 不编排具体业务逻辑 |

### 3.2 StateMachine

StateMachine 是 EventBus 上的一个特殊订阅者：
- 订阅所有可能触发状态转移的事件
- 根据「当前状态 + 事件」决定是否转移
- 转移后发布 `state_changed` 事件

```
StateMachine 行为：

    收到事件
        │
        ▼
    判断：当前状态 + 事件 → 是否触发转移？
        │
        ├─ 否 → 忽略
        │
        └─ 是 → 更新状态 → 发布 state_changed
```

### 3.3 EventBus

EventBus 是纯粹的消息通道：
- `publish(event_name, payload)` — 发布事件，携带数据（dict）
- `subscribe(event_name, callback)` — 订阅事件，回调签名 `callback(event_name, payload)`

所有组件（Task、StateMachine）都通过 EventBus 通信，互不直接引用。

**分发策略（本期约定）**：
- **同步分发**：`publish()` 直接在当前调用栈执行所有订阅回调
- 目的：简单、可预测、便于调试
- 后续如需异步再扩展（不影响现阶段设计）

**Payload 传递原则**：
- **优先传数据**：事件发生时的快照，明确、可追溯、只读
- **保留函数式能力**：特殊场景可传 getter 函数获取最新值

```
# 常规：传数据快照
publish("TASK:DETECTION_DONE", {"detections": [...], "stats": {...}})

# 特殊：需要"最新值"时，传只读 getter
publish("TASK:READY", {"get_pending_count": lambda: len(task.pending_targets)})
```

**设计理由**：
- 传数据：解耦、可追溯、无副作用
- 传函数：灵活、按需取值，但只暴露必要能力
- 不传整个对象：避免耦合、副作用、生命周期问题

### 3.4 Task

Task 是自治的业务单元：
- 订阅感兴趣的事件（如 `state_changed`）
- 执行自己的业务逻辑
- 产出结果后发布事件

```
Task 自治循环：

    ┌─────────────────────────────────────┐
    │                                     │
    ▼                                     │
 [等待/监听]                              │
    │                                     │
    │ 收到感兴趣的事件                     │
    ▼                                     │
 [执行自己的逻辑]                          │
    │                                     │
    │ 产出结果                            │
    ▼                                     │
 [发布新的事件] ──────────────────────────┘
```

---

## 4. Task 抽象

### 4.1 Task 接口

所有 Task 实现统一 Protocol（鸭子类型），框架不关心具体业务逻辑：

```
Task (Protocol，非 ABC 继承)
  │
  ├── name: str               # 任务名称
  ├── attach(event_bus)        # 注入 EventBus，订阅感兴趣的事件
  ├── run_iteration(image)     # 执行业务逻辑（Engine 驱动调用）
  ├── reset()                  # 重置状态
  └── close()                  # 清理资源，取消订阅
```

> **关于 `TaskIterationResult`**：当前版本暂时保留该返回类型（包含 detections、stable_targets、tracked_targets 等字段），
> 作为 Task 与 Engine 之间的过渡契约。未来 Workflow 进一步解耦后，将重构为更通用的结构或完全通过事件传递数据，届时删除此类型。

### 4.2 运行模式

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| **持续型** | 每帧/每周期都 run，直到条件触发离开 | 视觉处理、数据采集、持续监听 |
| **一次型** | run 一次就完成，自动触发离开 | 保存文件、发送消息、触发动作 |

框架同时支持两种模式，由具体 Task 实现决定。

### 4.3 Task 分类（示例）

以下是常见的 Task 分类，业务可根据需要扩展：

| 类别 | 职责 | 运行模式 |
|------|------|----------|
| **ProcessTask** | 处理/计算（视觉、算法） | 通常持续型 |
| **CommTask** | 对外通信（PLC、消息队列） | 持续型或一次型 |
| **SinkTask** | 输出（存储、推流、事件发布） | 通常一次型 |

**注意**：这只是分类建议，不是框架强制约束。业务可定义任意类型的 Task。

---

## 5. 并行模型：MainTask + SideTask

### 5.1 MainTask vs SideTask

| 维度 | MainTask | SideTask |
|------|----------|----------|
| 生命周期 | 随状态机切换（close → attach → reset） | 始终存活，贯穿整个 Engine 会话 |
| 驱动方式 | Engine 每帧调用 `run_iteration(image)` | 内部自治（事件驱动/轮询/线程，Engine 不关心） |
| 数量 | 同一时刻一个 | 任意多个并行 |
| 产出 | `TaskIterationResult`（detections 等） | 无固定产出格式 |
| 与 Engine 的接口 | `attach` / `run_iteration` / `reset` / `close` | 仅 `attach` / `close` |

### 5.2 SideTask 与 EventBus 的交互

SideTask 通过 EventBus 与 Workflow 交互，遵循 DDD 边界：

- **订阅事件**：`STATE:CHANGED`（感知状态变化）、`TASK:*`（接收 MainTask 产出的数据）、`SYS:*`（感知系统事件）
- **发布事件**：可以发布事件回 EventBus，甚至触发状态转换（如外部指令 → `SYS:RESET`）
- **不直接引用** Engine 或 StateMachine，所有通信走 EventBus

### 5.3 SideTask 内部执行模型

对 Engine 只暴露 `attach(event_bus)` / `close()`，内部执行方式是 SideTask 自己的事：
- **纯事件驱动**：在 `attach()` 中订阅事件，事件来了才干活
- **轮询型**：内部起线程或定时器轮询外部资源（如 PLC 消息队列）
- **混合型**：以上两者结合

Engine 不关心 SideTask 内部怎么运行，只负责生命周期管理（attach / close）。

### 5.4 SideTask Protocol

```python
class SideTask(Protocol):
    @property
    def name(self) -> str: ...
    def attach(self, event_bus: EventBus) -> None: ...
    def close(self) -> None: ...
```

### 5.5 模型示意

```
┌─────────────────────────────────────────────────────────────────┐
│                       WorkflowEngine                             │
│                                                                  │
│   MainTask (状态机编排，同一时刻一个)                             │
│      │                                                           │
│      ▼                                                           │
│   [State A] ──→ [State B] ──→ [State C] ──→ ...                 │
│       │             │             │                              │
│       ▼             ▼             ▼                              │
│    Task X        Task Y        Task Z        (由配置决定)        │
│                                                                  │
│   SideTasks (响应式，按需运行，数量不限)                          │
│      ├── SideTask 1                                              │
│      ├── SideTask 2                                              │
│      └── ...                                                     │
└─────────────────────────────────────────────────────────────────┘
```

- **MainTask**：由状态机编排，同一时刻只有一个 MainTask 在运行（`Task` Protocol）
- **SideTasks**：自治运行，Engine 只管 attach/close，数量不限（`SideTask` Protocol）

### 5.6 时间轴示意

```
时间轴 ──────────────────────────────────────────────────────→

MainTask (状态机切换):
  Task X:         ████████████
  Task Y:                     ████████
  Task Z:                             ████████████████

SideTasks (响应式):
  SideTask 1:          ███      ███           ███       (间歇响应)
  SideTask 2:            █        █             █       (事件触发)
  SideTask 3:     ████████████████████████████████████  (持续运行)
```

---

## 6. 事件命名规范

### 6.1 命名风格

**格式**：`类别:事件名`，使用冒号 `:` 分隔类别，下划线 `_` 分隔单词，全大写。

| 类别 | 前缀 | 示例 |
|------|------|------|
| 状态事件 | `STATE:` | `STATE:CHANGED`, `STATE:ENTERED`, `STATE:EXITED` |
| 业务事件 | `TASK:` | `TASK:DETECTION_DONE`, `TASK:PICK_DONE` |
| 系统事件 | `SYS:` | `SYS:STARTED`, `SYS:STOPPED`, `SYS:ERROR` |

### 6.2 Payload 约定

事件名由 `publish(event_name, payload)` 的第一个参数提供；payload 为 `dict`，约定最小公共字段如下：

| 字段 | 类型 | 说明 |
|------|------|------|
| `timestamp` | `float` / `str` | 事件发生时间（秒或 ISO） |
| `source` | `str` | 事件来源（task / engine / comm） |
| `payload` | `dict` | 业务数据（可选） |

```
# 状态事件
publish("STATE:CHANGED", {
  "timestamp": 1700000000.0,
  "source": "engine",
  "payload": {"old_state": "idle", "new_state": "detecting"}
})

# 业务事件
publish("TASK:DETECTION_DONE", {
  "timestamp": 1700000000.0,
  "source": "task",
  "payload": {"detections": [...], "frame_id": 123}
})

# 系统事件
publish("SYS:ERROR", {
  "timestamp": 1700000000.0,
  "source": "engine",
  "payload": {"message": "Camera disconnected"}
})

# 无 payload
publish("SYS:STARTED", {})
```

---

## 7. 事件流转

### 7.1 交互时序

```
Task A               EventBus              StateMachine
  │                      │                       │
  │  publish(event_x)    │                       │
  │─────────────────────►│                       │
  │                      │   dispatch            │
  │                      │──────────────────────►│
  │                      │                       │
  │                      │   判断: current_state + event_x
  │                      │         → 是否触发状态转移？
  │                      │                       │
  │                      │   publish(state_changed)
  │                      │◄──────────────────────│
  │                      │                       │
  │   dispatch           │                       │
  │◄─────────────────────│                       │
  │                      │                       │
Task B, C, ... 收到 state_changed，各自决定是否行动
```

### 7.3 状态机触发方式（本期约定）

- **直接触发**：StateMachine 作为 EventBus 的订阅者，收到事件后直接 `trigger(event)`
- 省去中间转发层，减少耦合与复杂度

### 7.2 状态流转示例（通用）

```
1. Engine 启动
   State = initial_state (由配置定义)

2. 触发事件 event_start
   State → state_a
   广播 state_changed(state_a)

3. Task A 收到广播，开始处理
   ...处理中...
   产出结果，发布 event_x

4. StateMachine 收到 event_x
   State → state_b
   广播 state_changed(state_b)

5. Task B 收到广播，开始处理
   ...

6. 其他 Task 也可能订阅了 event_x
   各自独立响应

7. 循环继续...
```

---

## 8. 配置示例

### 8.1 状态机配置 (workflow.yaml)

```yaml
workflow:
  name: my_workflow
  initial: idle

  # 状态列表（业务自定义，数量不限）
  states:
    - idle
    - state_a
    - state_b
    - state_c
    - error

  # 状态转移规则（业务自定义，数量不限）
  transitions:
    - trigger: start
      source: idle
      dest: state_a

    - trigger: event_x
      source: state_a
      dest: state_b

    - trigger: event_y
      source: state_b
      dest: state_c

    - trigger: done
      source: state_c
      dest: idle

    - trigger: error
      source: "*"          # 任意状态
      dest: error

    - trigger: reset
      source: "*"
      dest: idle
```

### 8.2 Task 注册配置

```yaml
tasks:
  # 主 Task：与状态绑定，由状态机编排
  main:
    state_a: TaskA
    state_b: TaskB
    state_c: TaskC

  # 附属 Task：响应式运行，数量不限
  side:
    - SideTask1
    - SideTask2
    - SideTask3
```

**说明**：
- 状态名、事件名、Task 名均由业务定义
- 框架只负责加载配置、驱动状态机、分发事件

---

## 9. 与现有架构的关系

### 9.1 演进路径

```
当前 (TaskManager)
        │
        │ Phase 1: 重命名，行为不变
        ▼
WorkflowEngine (单 Task)
        │
        │ Phase 2: 引入 EventBus
        ▼
WorkflowEngine + EventBus
        │
        │ Phase 3: 引入 StateMachine
        ▼
WorkflowEngine + EventBus + StateMachine
        │
        │ Phase 4: Task 拆分
        ▼
完整 Workflow 架构
```

### 9.2 与 03-task-system.md 的关系

| 文档 | 关注点 |
|------|--------|
| 03-task-system.md | Task 层：单个 Task 的接口、实现、生命周期 |
| 03a-workflow.md (本文档) | Workflow 层：Task 之间的编排、事件驱动、状态流转 |

---

## 10. 实现状态与待办

### 10.1 已完成

- [x] **Phase 1**: TaskManager → WorkflowEngine 重命名，单 Task 模式可用
- [x] **Phase 2**: EventBus 基础设施（同步 pub/sub，通配符 `*`，payload 规范）
- [x] **Phase 3**: StateMachine 基础设施（transitions、通配符 source、YAML loader、attach EventBus）
- [x] **辅助模块提取**: EventBuilder、FrameRenderer、FrameStreamer
- [x] **P1: Task 事件感知 + Bounded Context 重组**
  - 用 Protocol 替代 ABC 继承（`_protocol.py`），定义 `attach` / `run_iteration` / `reset` / `close` 契约
  - 提供 `TaskEventHelper` 组合件（`_event_helper.py`），封装 EventBus 交互
  - Task 自行发布 `TASK:ITERATION`、`TASK:DONE`、`TASK:PICK_RESULT` 事件，Engine 不再代发
  - 每个具体 Task 重组为独立文件夹（`detection/`、`stabilized_detection/`）
  - Engine 调用 `task.attach(event_bus)` 和 `task.close()` 管理生命周期
  - Engine 通过订阅 `TASK:PICK_RESULT` 事件转发 pick result，替代每帧轮询
  - **已知妥协**：
    - `TaskEventHelper` 目前是过渡抽象——envelope 格式、订阅管理等逻辑集中在 helper 而非由各 Task 自行决定。当前 Task 只 publish 不 subscribe，helper 的订阅管理能力空转。待后续 Task 真正需要订阅事件做自治决策时，helper 逻辑应下沉到具体 Task，helper 本身可能变薄或消失
    - Engine 仍承担 pick result 桥接职责（订阅 `TASK:PICK_RESULT` 再转发给 `event_publisher`），因为 `event_publisher` 尚未接入 EventBus。待 P2/P3 阶段 `event_publisher` 直接订阅 EventBus 后可移除此桥接
- [x] **P2: 状态机编排 Task（核心）**
  - YAML `workflow.tasks` 字段定义 state → task_type 映射
  - `WorkflowDefinition` dataclass 封装 state_machine + task_map，`load_workflow_from_yaml()` 返回完整定义
  - `main.py` 遍历 task_map，用 `create_task()` 工厂为每个 task_type 创建实例，传入 Engine
  - Engine 接收 `state_machine`（必选）和 `task_map`（必选）参数，订阅 `STATE:CHANGED` 事件驱动 task 切换
  - `_on_state_changed` 处理器：close 旧 task → attach + reset 新 task
  - `_process_frame` 守卫：当前状态无 task 时跳过帧处理
  - `result.is_done` 时由 TASK:DONE 事件驱动 StateMachine 状态转换，StateMachine 再通过 STATE:CHANGED 驱动 task 切换
  - `_cleanup` 关闭 task_map 中所有 task 实例
  - **设计决策**：workflow 是强制性的，不存在无 workflow 的场景。最简退化形态是单状态 workflow（如仅 detection），而非无 workflow。因此不保留向后兼容路径，`state_machine` 和 `task_map` 均为必选参数
- [x] **P3: SideTask 并行模型**
  - 定义 `SideTask` Protocol（`_protocol.py`）：仅 `name` / `attach(event_bus)` / `close()`，内部执行模型自治
  - YAML `workflow.side_tasks` 字段定义 side task 类型列表
  - `WorkflowDefinition` 新增 `side_task_types` 字段，`load_workflow_from_yaml()` 解析
  - Engine 接收 `side_tasks: Sequence[SideTask]` 参数，初始化时 attach 所有 SideTask，cleanup 时 close
  - 移除 Engine 中 `communication_task` 特殊参数和 `_process_comm_signals()` 方法
  - `CommunicationTask` 改造为 SideTask：内部轮询线程替代 Engine 每轮调用
  - `main.py` 从 `definition.side_task_types` 构建 side_tasks 列表传入 Engine
  - **设计决策**：SideTask 对 Engine 只暴露 attach/close，内部执行模型是 SideTask 自己的事（DDD 边界）
  - **已知妥协**：CommunicationTask 内部仍持有 task 引用（务实妥协），P4 迁移为纯事件驱动
- [x] **P4: CommunicationTask 事件驱动迁移**
  - 去掉 `self.task` 直接引用，CommunicationTask 改为纯事件驱动 SideTask
  - 文件搬迁：`src/scheduler/communication_task.py` → `src/tasks/communication/` 子包
  - 入站事件：poll loop 收到 PLC 消息 → publish `COMM:REQUEST_TARGET` / `COMM:PICK_DONE` / `COMM:RESET`
  - 出站事件：subscribe `TASK:PICK_RESULT` / `COMM:TARGET_RESPONSE` → 通过 comm_signal 发给 PLC
  - `request_target` 使用两阶段同步事件：CommTask publish → MainTask handler → publish response → CommTask handler 暂存 → poll thread 读取
  - StabilizedDetectionTask 新增三个 COMM 事件 handler（`_on_comm_request_target` / `_on_comm_pick_done` / `_on_comm_reset`）
  - `main.py` 构造 CommunicationTask 不再传入 task 引用
  - 修复旧代码 `target.category.value` 的 `AttributeError`（`TrackedTarget` 无 `category` 字段）
  - **已知妥协**：
    - 线程安全：poll 线程通过 EventBus 同步调用 MainTask handler，与 P3 之前风险相同，不新增。后续可通过加锁或队列解决
    - Task 生命周期：Engine 切换 task 时 close 取消所有订阅（含 COMM 事件），新 task attach 后重新订阅。不支持 pick 的 task 对 COMM:REQUEST_TARGET 无响应，CommTask 返回 error 给 PLC——正确行为

### 10.2 待完成

（暂无）

#### P5: Task 注册配置化（增强）

P2 已实现基础的 YAML `tasks` 配置段（state → task_type 映射）和 `create_task()` 工厂。
P3 已实现 `side_tasks` 配置段。

剩余：
- [ ] Task 工厂支持更丰富的参数注入

#### P6: 错误处理策略

- [ ] Task 失败时的状态流转（进入 error 状态？重试？）
- [ ] Engine 级别的异常捕获与恢复策略

#### P7: DAG 节点运行时（远期）

`nodes.py` 当前为占位符（`NotImplementedError`），属于未来演进方向。

- [ ] DAG 节点定义与执行
- [ ] 节点间数据传递

---

## 11. 参考

- 当前实现：`backend/src/workflow/state_machine.py`
- 当前实现：`backend/src/workflow/event_bus.py`
- 可选框架（按需引入）：[transitions - Python State Machine](https://github.com/pytransitions/transitions)
- 可选框架（按需引入）：[blinker - Python Signal Library](https://github.com/pallets-eco/blinker)
