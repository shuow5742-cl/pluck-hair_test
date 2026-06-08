# Stabilizer（稳定器）模块

> 相关文档：[架构总览](./00-overview.md) | [视觉处理](./02-vision-pipeline.md) | [任务系统](./03-task-system.md) | [Track ID 管理](./05-track-id-management.md)

## 概述

Stabilizer 负责将**多帧检测结果融合成稳定的目标列表**，通过跨帧关联和统计聚合，输出不抖动的坐标和 bbox。

**设计背景**：

Track 功能分为两个阶段：
1. **Stabilize（本节）**：跨帧聚类 + 坐标稳定，输出稳定目标
2. **Grip（后续）**：抓取状态管理（locked/ACK），避免重复下发

当前只实现 Stabilize 阶段。核心思路是把"track"定义为**跨帧聚类与稳定化**，而不是传统的运动跟踪（如 ByteTrack/SORT）。因为燕窝挑毛场景目标基本静止或仅有轻微移动，运动预测模型（卡尔曼滤波）反而会引入不必要的漂移。

---

## 架构位置

Stabilizer 是**有状态**组件，放在 **Task 层**而非 Pipeline 层：

```python
┌─────────────────────────────────────────────────────────────────────┐
│                   StabilizedDetectionTask (v2.1)                     │
│                                                                      │
│   持有：                                                             │
│   • pipeline: VisionPipeline    # 注入，无状态                       │
│   • stabilizer: Stabilizer      # 有状态，跨帧                       │
│   • stats: TaskStats                                                │
│                                                                      │
│   run_iteration(image):                                             │
│       detections = pipeline.run(image)          # Step 1: 单帧检测   │
│       stable_targets = stabilizer.update(detections)  # Step 2: 跨帧稳定 │
│       return TaskIterationResult(detections, stable_targets, ...)   │
└─────────────────────────────────────────────────────────────────────┘
```

**设计理由**：
- Pipeline 保持无状态、单帧处理的纯函数特性
- Stabilizer 需要维护跨帧的簇状态，由 Task 持有和管理

---

## 数据类型

### Detection（输入）

来自 Pipeline 的输出，每帧波动。详见：[视觉处理](./02-vision-pipeline.md#detection)

### StableTarget（输出）

**关键字段**：
- `x, y`：通过 median 聚合，抗抖
- `width, height`：通过 IQR + median 聚合，过滤离群值
- `cluster_id`：簇的唯一标识（技术域），由 Stabilizer 分配
- `occurrence_ratio`：窗口内出现的帧数比例（用于判断稳定性）

**下游使用**：
- **前端显示**：绘制稳定的红框（不抖动）
- **PickProcess**：将 `cluster_id` 映射为 `track_id`（业务标识）
- **CoordinateTransformer**：转换为物理坐标（毫米）

详见：[Track ID 管理](./05-track-id-management.md)

### TargetCluster（内部数据结构）

```python
┌───────────────────────────────────────────────────────────────────────┐
│                        TargetCluster                                  │
│                                                                       │
│   表示跨帧关联到一起的同一个目标                                       │
├───────────────────────────────────────────────────────────────────────┤
│  - cluster_id: str                  # 唯一标识                        │
│  - center: Tuple[float, float]      # 当前簇中心 (x, y)               │
│  - history: deque[Detection]        # 历史检测（最近 N 帧）            │
│  - last_seen_frame: int             # 最后一次被检测到的帧号           │
│  - state: ClusterState              # tentative | stable              │
│  - created_frame: int               # 簇创建时的帧号                   │
│  - object_type: str                 # 类别                            │
└───────────────────────────────────────────────────────────────────────┘
```

### ClusterState（簇状态）

```python
class ClusterState(Enum):
    TENTATIVE = "tentative"    # 新建簇，尚未满足稳定条件
    STABLE = "stable"          # 满足稳定条件，可输出
```

---

## Stabilizer 接口

```python
┌───────────────────────────────────────────────────────────────────────┐
│                         Stabilizer                                    │
│                                                                       │
│   职责：把多帧检测结果融合成稳定的目标列表                             │
│   特性：有状态，增量式更新，双重稳定性保护                            │
├───────────────────────────────────────────────────────────────────────┤
│  属性：                                                               │
│  - clusters: List[TargetCluster]    # 当前维护的目标簇                │
│  - frame_id: int                    # 当前帧计数                      │
│  - config: StabilizerConfig         # 配置参数                        │
│                                                                       │
│  方法：                                                               │
│  + update(detections: List[Detection]) -> List[StableTarget]         │
│  + reset() -> None                  # 区域/任务切换时清空状态          │
│  + get_cluster_count() -> int       # 当前簇数量（调试用）            │
└───────────────────────────────────────────────────────────────────────┘
```

### 配置参数（v2.2 增强）

```python
@dataclass
class StabilizerConfig:
    # 历史窗口
    window_size: int = 10                    # 簇历史长度上限

    # 关联参数
    distance_threshold_px: float = 30        # 关联距离阈值（像素）

    # 稳定判定（v2.2 双重保护）
    min_occurrence_ratio: float = 0.6        # 进入阈值（TENTATIVE → STABLE）
    stable_exit_ratio: float = 0.3           # 退出阈值（STABLE → TENTATIVE）
    min_frames_to_stable: int = 6            # 升级 STABLE 所需最小帧数

    # 簇删除
    missing_frames_to_delete: int = 4        # 连续缺失多少帧后删除簇

    # 微动处理
    jump_threshold_px: float = 60            # 突变阈值（像素）
    reset_on_jump: bool = True               # 突变时是否重置簇历史

    # bbox 聚合策略（v2.1）
    bbox_aggregation: str = "iqr_median"     # iqr_median | median | ema
```

**关键语义**：

| 概念 | 具体定义 |
|------|---------|
| `window_size` | 簇历史长度上限（每个簇独立维护），非全局滑窗 |
| `occurrence_ratio` | 出现帧数 / min(当前帧 - 创建帧 + 1, window_size) |
| `cluster_age` | 当前帧 - 创建帧 + 1，簇存在的帧数 |
| `min_frames_to_stable` | 簇升级为 STABLE 需满足的最小 cluster_age |
| `stable_exit_ratio` | STABLE → TENTATIVE 的降级阈值，低于 min_occurrence_ratio 形成滞后 |
| `missing_frames_to_delete` | 连续缺失超过此值的帧数后，簇被删除 |

---

## 核心算法

### update() 流程

```python
update(detections) 每帧调用：
│
├── 1. 跨帧关联（Associate）
│       对每个 detection，找最近的 cluster（贪心）
│       ├── 距离 < distance_threshold → 关联到该 cluster
│       └── 否则 → 新建 cluster（状态 = tentative）
│
├── 2. 更新簇中心（Update Center）
│       对每个被关联的 cluster：
│       ├── 计算新旧中心距离 delta
│       ├── if delta > jump_threshold:
│       │       直接重置中心 = 新检测位置（处理突变）
│       └── else:
│               中心 = 新检测位置（不做 EMA，避免滞后）
│
├── 3. 清理过期簇（Cleanup）
│       删除 last_seen_frame 超出窗口的 cluster
│
├── 4. 状态转换（State Transition）【v2.2 增强】
│       采用双阈值滞后 + 最小帧数要求，防止幽灵框和频闪
│       对每个 cluster：
│       ├── 计算 occurrence_ratio = 出现帧数 / 有效窗口大小
│       ├── 计算 cluster_age = 当前帧 - 创建帧 + 1
│       ├── TENTATIVE → STABLE：
│       │       必须同时满足：
│       │       ├── occurrence_ratio >= min_occurrence_ratio（进入阈值）
│       │       └── cluster_age >= min_frames_to_stable（最小帧数）
│       └── STABLE → TENTATIVE：
│               occurrence_ratio < stable_exit_ratio（退出阈值，更低）
│
└── 5. 输出稳定目标（Output）
        返回所有 state == stable 的 cluster，坐标用 median/trimmed_mean
```

---

## 状态转换的双重保护机制（v2.2）

为了确保稳定框的可靠性，状态转换采用两个互补的机制：

| 机制 | 解决的问题 | 原理 |
|------|-----------|------|
| **min_frames_to_stable** | 防止幽灵框（新簇立刻输出） | 簇必须存在至少 N 帧才能升级为 STABLE，过滤掉短暂的噪声检测 |
| **双阈值滞后** | 防止频闪（状态反复切换） | 进入阈值高、退出阈值低，在滞后区间内波动不触发状态切换 |

### 状态转换示意图

```
occurrence_ratio
      ▲
  1.0 │
      │
  0.6 │─────────────────────────── min_occurrence_ratio（进入阈值）
      │          ┌──────────────────────────────────────────
      │          │              STABLE 区域
      │          │   (ratio 在此区间内不会降级，形成滞后)
  0.3 │──────────┴─────────────── stable_exit_ratio（退出阈值）
      │
      │              TENTATIVE 区域
  0.0 └─────────────────────────────────────────────────────► 时间
      │←─ min_frames_to_stable ─→│
           (新簇需等待的最小帧数)
```

**为什么需要双重保护**：

1. **单独 min_frames_to_stable**：解决新簇"一闪而过"的问题，但无法防止已稳定簇因 ratio 波动而频闪
2. **单独双阈值滞后**：解决频闪问题，但无法阻止新簇在第一帧就因 ratio=100% 而升级
3. **两者结合**：既防止幽灵框（新簇需等待），又防止频闪（稳定后不轻易降级）

---

## 关联策略

### 距离度量：仅中心点距离

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 关联度量 | **仅中心点距离** | IoU 对细长目标敏感，bbox 轻微角度变化会导致 IoU 剧烈波动 |
| 匹配算法 | **贪心最近邻** | 毛发目标稀疏，贪心足够；复杂场景可切换匈牙利 |
| 类别约束 | 同类别才能关联 | 避免不同类型目标混淆 |
| 处理顺序 | 按 confidence 降序 | 高置信度检测优先匹配，避免误检干扰 |

### 伪代码

```python
def associate(detections, clusters):
    # 1. 按置信度降序排序
    detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

    matched = []
    unmatched = []

    for det in detections:
        # 2. 找最近的同类别簇
        best_cluster = None
        min_distance = float('inf')

        for cluster in clusters:
            if cluster.object_type != det.object_type:
                continue

            dist = euclidean_distance(det.bbox.center, cluster.center)
            if dist < min_distance and dist < distance_threshold:
                min_distance = dist
                best_cluster = cluster

        # 3. 关联或新建
        if best_cluster:
            matched.append((det, best_cluster))
        else:
            unmatched.append(det)

    return matched, unmatched
```

---

## 微动与突变处理

场景：目标可能因水流或机械臂动作有 mm 级移动。

| 情况 | 处理方式 | 理由 |
|------|----------|------|
| 正常帧间抖动 | 直接更新中心为新位置 | 抖动在阈值内，下一帧检测会自然收敛 |
| 突变（delta > jump_threshold） | **直接重置**中心为新位置 | 避免 EMA 滞后导致输出坐标追不上实际位置 |

**不使用 EMA 的原因**：
- EMA 有固有滞后，目标移动后中心需要多帧才能收敛
- 静止场景下，每帧直接取检测位置更新中心，输出时用 median 聚合即可抗抖
- 突变时 EMA 会持续输出"追赶中"的错误坐标

---

## 稳定性判定与输出

### 输出条件（同时满足）

1. `occurrence_ratio >= min_occurrence_ratio`：窗口内出现比例足够
2. `state == stable`：已通过状态转换
3. `cluster_age >= min_frames_to_stable`：存在帧数足够

### 聚合方法（关键）

```python
输出 StableTarget:
  # 中心坐标：median（对异常值鲁棒）
  x, y = median(cluster.history 中所有检测的中心)

  # bbox 尺寸：IQR 过滤离群值 + median（v2.1 新增）
  widths = [detection.bbox.width for detection in cluster.history]
  heights = [detection.bbox.height for detection in cluster.history]

  # Step 1: IQR 方法过滤离群值（部分遮挡、误检）
  filtered_widths = filter_outliers_iqr(widths)
  filtered_heights = filter_outliers_iqr(heights)

  # Step 2: 对过滤后的值取 median
  width = median(filtered_widths)
  height = median(filtered_heights)

  # 其他字段
  confidence = weighted_mean(最近帧权重更大)
  occurrence_ratio = len(recent_detections) / window_size
```

**bbox 聚合理由**：
- bbox 尺寸变化比中心点大（边缘判定敏感）
- 单用 median 可能跳变，IQR 先过滤异常再 median 更稳定
- 未来可切换为 Winsorized Mean 或 EMA（可配置）

### IQR 过滤定义

```python
filter_outliers_iqr(values):
  Q1 = 第 25 百分位
  Q3 = 第 75 百分位
  IQR = Q3 - Q1
  lower = Q1 - 1.5 × IQR
  upper = Q3 + 1.5 × IQR
  返回 [v for v in values if lower ≤ v ≤ upper]
```

标准箱线图离群值定义（Tukey, 1977）。

**降级策略**：
- 历史样本 ≤ 3 个：直接用原始 median（IQR 需要至少 4 个样本）
- 过滤后 < 2 个：全被判定为离群值，回退原始 median（保证输出稳定）

---

## 参数建议

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `window_size` | 8-12 | 滑动窗口大小，影响稳定性判定 |
| `min_occurrence_ratio` | 0.5-0.7 | 进入阈值：TENTATIVE → STABLE 需要的最小出现比例 |
| `stable_exit_ratio` | 0.2-0.4 | 退出阈值：STABLE → TENTATIVE 的降级阈值（建议为进入阈值的一半） |
| `min_frames_to_stable` | 4-8 | 簇必须存在的最小帧数才能升级为 STABLE，防止幽灵框 |
| `distance_threshold_px` | 20-40 | 关联距离阈值（像素），需根据实际检测抖动调整 |
| `jump_threshold_px` | 50-80 | 突变阈值（像素），超过此值重置簇中心 |
| `missing_frames_to_delete` | 3-5 | 连续缺失多少帧后删除簇 |

### 参数调优方法

1. 采集静止目标的多帧检测数据
2. 统计 bbox 中心的帧间方差 σ
3. `distance_threshold_px ≈ 3σ`（覆盖 99% 正常抖动）
4. `jump_threshold_px ≈ 目标预期最大位移 / 帧间隔`
5. `min_frames_to_stable`：根据帧率调整，10fps 时设为 6 约等于 600ms 延迟
6. `stable_exit_ratio`：设为 `min_occurrence_ratio` 的 40%-60%，形成足够的滞后区间

---

## 配置示例

```yaml
task:
  name: "stabilized_detection"

  stabilizer:
    # 历史窗口
    window_size: 10                    # 簇历史窗口大小（帧数）

    # 关联参数
    distance_threshold_px: 30          # 关联距离阈值（像素）

    # 稳定判定（v2.2 增强：双阈值滞后 + 最小帧数）
    min_occurrence_ratio: 0.6          # 进入阈值：至少出现 60% 的帧才能升级为 STABLE
    stable_exit_ratio: 0.3             # 退出阈值：低于 30% 才降级为 TENTATIVE（防止频闪）
    min_frames_to_stable: 6            # 最小帧数：簇必须存在 6 帧以上才能升级（防止幽灵框）
    missing_frames_to_delete: 4        # 连续缺失 4 帧后删除簇

    # 微动处理
    jump_threshold_px: 60              # 突变阈值（像素），超过则重置簇中心
    reset_on_jump: true                # 突变时重置簇历史

    # bbox 聚合策略（v2.1 新增）
    bbox_aggregation: "iqr_median"     # iqr_median | median | winsorized_mean | ema
```

---

## 参考

- IQR 离群值检测：Tukey, J.W. (1977). Exploratory Data Analysis
- 跨帧关联：SORT/DeepSORT 算法
- 状态机设计：Hysteresis (滞后) 模式
