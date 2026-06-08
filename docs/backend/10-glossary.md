# 术语表

> 相关文档：[架构总览](./00-overview.md)

## 核心概念

| 术语 | 说明 |
|------|------|
| **Pipeline** | 无状态的多步骤单帧图像处理流程，由多个 ProcessStep 组合而成 |
| **Task** | 有状态的任务单元，持有 Stabilizer/Stats，主导业务逻辑 |
| **StabilizedDetectionTask** | Task 的核心实现，集成 Pipeline + Stabilizer |
| **TaskManager** | 主循环调度器，调用 Task 并处理前端推送、存储 |
| **Workflow** | 多任务编排引擎（未来实现），按顺序执行多个 Task |

## 数据类型

| 术语 | 说明 |
|------|------|
| **Detection** | 单帧检测结果（bbox, 类别, 置信度） |
| **StableTarget** | 稳定化后的目标输出（像素坐标 + bbox 尺寸） |
| **TargetCluster** | 跨帧关联的目标簇，包含历史检测和状态 |
| **ClusterState** | 簇状态：tentative（待定）/ stable（稳定） |
| **TrackedTarget** | 带 Track ID 的业务目标，包含生命周期状态 |
| **PhysicalTarget** | 物理坐标目标（毫米） |

## Stabilizer 相关

| 术语 | 说明 |
|------|------|
| **Stabilizer** | 稳定器，负责跨帧关联和 bbox 聚合 |
| **双阈值滞后** | 进入阈值高、退出阈值低，防止状态在阈值边界频繁切换 |
| **min_frames_to_stable** | 簇升级为 STABLE 所需的最小存在帧数，防止幽灵框 |
| **幽灵框** | 噪声检测导致的短暂出现的框，新簇快速变 STABLE 又消失 |
| **频闪** | 已稳定的簇因 ratio 在阈值边界波动导致反复切换状态 |
| **bbox 聚合** | IQR 过滤离群值 + median，用于稳定 bbox 尺寸输出 |
| **occurrence_ratio** | 出现帧数 / min(当前帧 - 创建帧 + 1, window_size) |
| **cluster_age** | 当前帧 - 创建帧 + 1，簇存在的帧数 |
| **missing_frames_to_delete** | 连续缺失超过此值的帧数后，簇被删除 |

## Track ID 管理

| 术语 | 说明 |
|------|------|
| **Track ID** | 目标的唯一标识，用于计数、机械臂协调和复检验证 |
| **Initial Track** | 初始化阶段稳定的目标（高可信度），系统启动时识别的目标 |
| **Ghost Track** | 初始化后新出现的目标（低可信度），可能是误检或之前被遮挡的目标 |
| **PickProcess** | Track ID 生命周期管理器，维护 initial/ghost 两类目标的状态 |
| **控制信号驱动** | 机械臂主导、视觉服务的架构模式，避免纯视觉持续检测导致的误计数 |
| **TrackCategory** | INITIAL（高可信）/ GHOST（低可信） |
| **TargetState** | PENDING（待挑）/ PICKED（已挑走）/ LOST（丢失） |

## 领域驱动设计（DDD）

| 术语 | 说明 |
|------|------|
| **限界上下文** | DDD 概念：Stabilizer（技术域）和 PickProcess（业务域）各自独立，通过映射关联 |
| **cluster_id** | Stabilizer 的技术标识，生命周期由技术规则（missing_frames_to_delete）决定 |
| **track_id** | PickProcess 的业务标识，生命周期由业务规则（PICK_DONE 确认）决定 |
| **防腐层** | DDD 模式：PickProcess 通过 _cluster_to_track 映射隔离 Stabilizer 的实现细节 |

## 其他核心概念

| 术语 | 说明 |
|------|------|
| **Grip** | 抓取状态管理（未来实现，包含 locked/ACK） |
| **控制端/PLC** | 现场控制器，通过通信链路请求坐标/状态 |
| **Session** | 一次完整的运行周期（从启动到停止） |
| **DoneCondition** | 任务完成条件判定接口（如连续 N 帧无检测） |
| **TaskStats** | 任务运行统计（总帧数、总检测数等） |

## 坐标系统

| 术语 | 说明 |
|------|------|
| **像素坐标** | 图像坐标系，原点在左上角，X 向右，Y 向下 |
| **物理坐标** | 托盘坐标系，原点在托盘左下角，X 向右，Y 向上 |
| **CoordinateTransformer** | 像素坐标 → 物理坐标转换器 |
| **pixel_to_mm** | 每像素对应的毫米数（相机标定得出） |

## 存储

| 术语 | 说明 |
|------|------|
| **MinIO** | S3 兼容的对象存储，用于存储图像 |
| **ImageStorage** | 图像存储抽象接口（MinIO / Local） |
| **Database** | 数据库抽象接口（PostgreSQL / SQLite） |
| **StorageSaver** | 异步存储处理器，避免阻塞 Vision 主循环 |
| **DetectionRecord** | 检测记录数据模型（ORM） |
| **SessionRecord** | 会话记录数据模型（ORM） |

## 通信

| 术语 | 说明 |
|------|------|
| **CommInterface** | 通信抽象接口（协议无关） |
| **CommSignalBase** | 控制信号通道的抽象接口（Redis/Modbus 互斥实现） |
| **CommunicationTask** | 解析控制信号并执行业务逻辑的任务组件 |
| **Comm Bridge** | 现场通信桥的占位实现（协议确定后落地） |
| **控制映射** | 控制端与 Backend 交互所需的字段/地址映射（待定义） |
| **GIVE_ME_TARGET** | 请求下一个待挑目标的控制信号 |
| **PICK_DONE** | 机械臂挑完一次的反馈信号 |
| **RESET** | 清空所有状态，重新初始化的控制信号 |

## 视觉处理

| 术语 | 说明 |
|------|------|
| **ProcessStep** | Pipeline 中的单个处理步骤抽象类 |
| **TileStep** | 图像切片步骤 |
| **YOLODetectStep** | YOLO 检测步骤 |
| **MergeTilesStep** | 合并切片结果步骤 |
| **NMSStep** | 非极大值抑制步骤 |
| **FilterStep** | 结果过滤步骤 |
| **SortStep** | 结果排序步骤 |
| **BoundingBox** | 边界框（x1, y1, x2, y2） |
| **PipelineContext** | Pipeline 步骤间传递的上下文 |
| **PipelineResult** | Pipeline 最终输出结果 |

## 相机

| 术语 | 说明 |
|------|------|
| **CameraBase** | 相机抽象基类 |
| **DahengCamera** | 大恒工业相机实现 |
| **MockCamera** | 测试用 Mock 相机 |
| **gxipy** | 大恒相机 Python SDK |

## 统计与监控

| 术语 | 说明 |
|------|------|
| **IQR** | 四分位距（Interquartile Range），用于离群值检测 |
| **median** | 中位数，对异常值鲁棒的统计量 |
| **EMA** | 指数移动平均（Exponential Moving Average） |
| **Winsorized Mean** | 缩尾均值，一种鲁棒的平均值计算方法 |
| **occurrence_ratio** | 出现比例，簇在窗口内出现的帧数比例 |

## 架构模式

| 术语 | 说明 |
|------|------|
| **单向数据流** | Task → stable_targets → 前端/存储/通信链路，数据单向流动 |
| **依赖注入** | Pipeline 通过构造函数注入给 Task，避免重复创建 |
| **工厂模式** | 根据配置创建不同类型的组件（如 create_camera） |
| **策略模式** | 可配置的算法实现（如 bbox_aggregation 策略） |
| **状态机** | ClusterState、TargetState 的状态转换逻辑 |
| **防腐层** | DDD 模式，通过映射表隔离不同上下文 |

---

## 架构要点总结

- **Task 主导**：持有所有有状态组件，完整控制业务流程
- **单向数据流**：Task → stable_targets → 前端/存储/通信链路
- **Pipeline 无状态**：纯函数式，可复用可测试
- **双重稳定性保护**：min_frames_to_stable 防止幽灵框 + 双阈值滞后防止频闪
- **Track ID 分级**：Initial（高可信）/ Ghost（低可信）两类，利用场景先验简化问题
- **控制信号驱动**：机械臂触发检测，而非视觉持续检测，避免幽灵框闪烁导致误计数
- **计数单位是框**：不纠结"一根毛几个框"，框消失即计数，务实简洁
- **DDD 限界上下文**：Stabilizer（技术域）与 PickProcess（业务域）职责分离，通过防腐层映射
- **track_id 独立管理**：由 PickProcess 自增分配，不直接复用 cluster_id，避免领域污染

---

## 参考资料

- **领域驱动设计**：Eric Evans, "Domain-Driven Design"
- **IQR 离群值检测**：Tukey, J.W. (1977). "Exploratory Data Analysis"
- **目标跟踪**：SORT/DeepSORT 算法
- **状态机设计**：Hysteresis (滞后) 模式
- **现场通信协议**：待与控制端确认（保持抽象，不在此文档写死）
- **YOLO 检测**：Ultralytics YOLOv8
- **S3 协议**：MinIO 文档
