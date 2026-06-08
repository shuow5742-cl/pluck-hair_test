# Track ID 管理模块

> 相关文档：[架构总览](./00-overview.md) | [Stabilizer](./04-stabilizer.md) | [任务系统](./03-task-system.md)

## 概述

Track ID 用于：
1. **前端计数显示**：累计已挑多少，当前视野还剩多少
2. **机械臂协调**：发送目标坐标时附带 ID，机械臂反馈后根据 ID 确认结果
3. **复检验证**：通过位置匹配判断目标是否被挑走

## 业务场景

一个燕窝盘通常分为多个区域（如 20 个区域），系统按顺序扫描每个区域：

```
盘子 (Workflow 层)
  ├─ 区域 1 → Task 1 → 挑完 10 个
  ├─ 区域 2 → Task 2 → 挑完 8 个
  ├─ ...
  └─ 区域 20 → Task 20 → 挑完 5 个

全盘累计：total_picked = 10 + 8 + ... + 5
```

**关键特性**：
- **全盘累计**：`total_picked` 跨区域累加（Workflow 层维护）
- **区域隔离**：每个区域是独立 Task，换区域时重置 PickProcess
- **生命周期不同**：全盘计数活到整个盘子完成，区域计数随 Task 销毁

**详见**：[任务系统 - Workflow/Task 分层设计](./03-task-system.md#workflowtask-分层设计)

---

## 关键设计决策

| 决策 | 说明 |
|------|------|
| **Workflow/Task 分层** | Workflow 管理全盘，Task 管理单区域，生命周期隔离 |
| **位置匹配确认** | 用位置匹配判断目标是否消失，而非依赖 cluster_id |
| **track_id 独立管理** | 自增整数，不复用 |
| **控制信号驱动** | 计数由 PICK_DONE 触发，而非视觉自动检测 |
| **Phase 显式化流程** | 用状态机表达业务流程 |

---

## 核心问题：为什么不能纯视觉驱动计数？

纯视觉驱动的计数存在致命缺陷：

```
图像静止 → 幽灵框闪烁 → 消失就+1 → 计数越来越多
                                    ↑
                              这很荒谬
```

**解决方案**：控制信号驱动 + 位置匹配确认

---

## 位置匹配确认机制

### 设计原理

挑取确认的本质是"那个位置的毛没了"，而不是"那个 cluster_id 没了"。

使用位置匹配代替 cluster_id 追踪：
1. 下发目标时记录位置 (x, y, width, height)
2. 机械臂返回后，在新检测结果中按位置找是否还有接近的框
3. 找不到 → 挑走；找到 → 未挑走

### 匹配规则

位置匹配采用两个约束条件：

1. **距离约束**：计算目标中心点与检测框中心点的欧氏距离，距离必须小于 `match_distance_threshold`（默认 30 像素）

2. **尺寸约束**：计算宽度和高度的差异比例，差异比例必须小于 `match_size_ratio_threshold`（默认 30%）

匹配时遍历所有检测框，返回满足上述条件且距离最近的框。如果没有满足条件的框，则认为目标已消失。

### 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `match_distance_threshold` | 30.0 | 位置匹配距离阈值（像素） |
| `match_size_ratio_threshold` | 0.3 | 尺寸差异容忍度（30%） |

---

## Phase 状态机

Phase 是流程级别的概念，显式表达当前处于什么业务阶段。

```python
class Phase(StrEnum):
    INIT = "init"                 # 初始化阶段，Stabilizer 稳定化中
    READY = "ready"               # 准备好，等待机械臂请求目标
    AWAITING_PICK = "awaiting_pick"  # 已下发目标，等待机械臂完成
    CONFIRMING = "confirming"     # 收到 PICK_DONE，复检确认中
    DONE = "done"                 # 全部完成
```

### 状态转换图

```
INIT ──(init_stable_threshold 帧后)──► READY
                                        │
                            get_next_target()
                                        ▼
                                   AWAITING_PICK
                                        │
                              on_pick_done(target_id)
                                        ▼
                                   CONFIRMING
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                       │
          位置匹配找不到目标                    confirm_window_frames 帧后
          (region_picked += 1)                    目标仍在（挑取失败）
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        ▼
                                      READY ◄──(还有目标)
                                        │
                                   (无目标)
                                        ▼
                                      DONE
```

---

## 消失判定机制

### CONFIRMING 阶段逻辑

1. 收到 `on_pick_done(target_id)` 后进入 CONFIRMING
2. 每帧用位置匹配检查目标是否还在
3. 找不到匹配 → 目标消失 → 计数 +1
4. 窗口期结束仍能匹配到 → 挑取失败

### 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `confirm_window_frames` | 10 | 确认窗口帧数 |

---

## 数据结构

### TrackedTarget

```python
@dataclass(slots=True)
class TrackedTarget:
    track_id: int           # 业务标识（自增分配）
    x: float                # 中心 x（像素）
    y: float                # 中心 y（像素）
    width: float            # bbox 宽度（像素）
    height: float           # bbox 高度（像素）
    confidence: float
    object_type: str
    state: TargetState      # PENDING | PICKED
    pick_attempts: int = 0
```

### TargetState

```python
class TargetState(StrEnum):
    PENDING = "pending"    # 待挑
    PICKED = "picked"      # 已挑走
```

---

## PickProcess 接口

```python
┌───────────────────────────────────────────────────────────────────────┐
│                         PickProcess                                   │
│                                                                       │
│   职责：管理 Track ID 的生命周期和业务状态                              │
├───────────────────────────────────────────────────────────────────────┤
│  属性：                                                               │
│  - phase: Phase                          # 当前业务流程阶段            │
│  - _targets: Dict[int, TrackedTarget]    # 所有目标                   │
│                                                                       │
│  配置参数：                                                           │
│  - init_stable_threshold: int      # 初始化观察期帧数                  │
│  - confirm_window_frames: int      # CONFIRMING 阶段等待帧数          │
│  - match_distance_threshold: float # 位置匹配距离阈值                  │
│  - match_size_ratio_threshold: float # 尺寸差异容忍度                  │
│                                                                       │
│  方法：                                                               │
│  + update(stable_targets: List[StableTarget]) -> None                │
│  + get_next_target() -> Optional[TrackedTarget]   # READY → AWAITING │
│  + on_pick_done(target_id: int) -> None           # → CONFIRMING     │
│  + reset() -> None                                # 重置所有状态       │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 统计信息

```python
@dataclass(slots=True)
class TrackStats:
    region_picked: int       # 本区域已挑数量
    current_pending: int     # 当前待挑数量（PENDING 状态）
```

**职责分层**：
- `region_picked`：Task 层维护，单区域计数
- `total_picked`：Workflow 层累加所有区域的 `region_picked`

---

## 信号定义

### GIVE_ME_TARGET

请求下一个待挑目标。

**响应**：
```json
{
  "track_id": 1,
  "x": 100.5,
  "y": 200.3,
  "width": 50.0,
  "height": 30.0,
  "confidence": 0.95,
  "object_type": "hair"
}
```

**行为**：
- Phase: READY → AWAITING_PICK
- 返回第一个 PENDING 状态的目标

### PICK_DONE

机械臂挑完一次，触发复检。

**请求参数**：
```json
{
  "target_id": 5
}
```

**行为**：
- Phase: AWAITING_PICK → CONFIRMING
- 后续帧内用位置匹配检查目标是否消失
- 消失 → 计数 +1；窗口期结束仍在 → 挑取失败

### RESET

清空所有状态，重新初始化。

---

## 初始化窗口机制

**目的**：等待 Stabilizer 稳定后再创建目标

**流程**：

1. **第1帧**：Stabilizer 首次推送的 `stable_targets` → 创建所有 TrackedTarget
2. **窗口期（第2-N帧）**：等待，不创建新目标
3. **窗口结束**：Phase: INIT → READY

**配置**：
- `init_stable_threshold`：初始化窗口帧数（默认 10）

### 未来：外部控制链路

GIVE_ME_TARGET 和 PICK_DONE 将通过外部控制链路下发/上报（协议待定，统一通过 CommSignalBase 抽象适配）。

详见：[通信架构](./07-communication.md)

---

## 配置示例

```yaml
task:
  track:
    # 初始化窗口
    init_stable_threshold: 10      # 初始化观察期帧数

    # CONFIRMING 阶段
    confirm_window_frames: 10      # 挑取确认窗口帧数

    # 位置匹配参数
    match_distance_threshold: 30.0  # 位置匹配距离阈值（像素）
    match_size_ratio_threshold: 0.3 # 尺寸差异容忍度（30%）
```

---

## 唯一性与生命周期

```
Track ID 设计：
  - 由 PickProcess 独立管理（自增整数，不复用）
  - 作用域：单次工作周期内唯一（从开始挑毛到完成这块燕窝）
  - 重置时机：收到 RESET 信号时（换燕窝 / 手动重置）
```

---

## 参考

- 状态机设计：有限状态自动机（FSA）
- 计数逻辑：工业视觉中的目标追踪与计数
