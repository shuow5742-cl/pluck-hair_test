# TODO / 演进计划

> 相关文档：[架构总览](./00-overview.md) | [任务系统](./03-task-system.md) | [Track ID 管理](./05-track-id-management.md) | [通信架构](./07-communication.md) | [API 模块](./09-api.md)

---

## 当前任务（本次）：WorkflowEngine 职责瘦身（第二点）

目标：让 WorkflowEngine 只负责调度/生命周期管理，把事件构造、存储/推流/预览等横切逻辑拆出去，降低耦合。

### 范围与原则
- **保留在 Engine**：主循环、会话管理、错误处理、任务切换/停止
- **拆出**：事件构造、帧渲染/编码、存储保存、对外发布
- **避免**：Engine 直接访问 Task 内部状态（如 PickProcess）

### 待办清单
- [ ] 引入 `EventBuilder`（或 `ResultAdapter`），由 `TaskIterationResult` 生成事件 payload
- [ ] Engine 只调用 `event_sink.publish()`，不拼具体 payload
- [ ] 抽出 `FrameBuilder`（渲染 + 编码 + fps 控制）
- [ ] 抽出 `StoragePipeline`（或继续使用 `StorageSaver`，但让 Engine 调度更干净）
- [ ] Task 接口补充 `get_last_pick_result()` / `get_business_events()`，避免 Engine 直接访问内部对象
- [ ] CommunicationTask 迁移到 workflow 层（或 workflow/nodes），形成可插拔节点

---

## 后续任务（第一点）：Workflow 结构设计（DAG/Node）

目标：让 Workflow 支持顺序/分支/并行的编排，并成为统一的流程执行框架。

### 待办清单
- [ ] 定义 `WorkflowNode` 接口（run/inputs/outputs/status）
- [ ] 设计 `WorkflowGraph` / `DAG` 数据结构（节点 + 边 + 条件）
- [ ] 设计 `WorkflowRuntime`（调度策略：顺序 / 分支 / 并行）
- [ ] 设计 Node 类型：TaskNode / CommNode / StorageNode / EventNode
- [ ] 约定节点间的数据契约（Result / Event / Context）
- [ ] YAML/配置加载（workflow loader）

---

## 后续任务（第三点）：Task 拆分与重组

目标：把业务能力拆成更小的 Task，支持灵活组合与复用。

### 待办清单
- [ ] 抽出 `DetectionTask` / `StabilizerTask` / `PickTask` / `CommTask`
- [ ] Task 只暴露业务结果，不暴露内部状态（PickProcess 封装）
- [ ] 定义 Task 输出契约（Result / Events）
- [ ] 为 Task 增加最小可测单元（mock pipeline / mock stabilizer）
- [ ] 对接 WorkflowNode（TaskNode 封装 Task 执行）

---

## 待办：Track ID 计数改进

### 问题背景

当前 Track ID 计数不准确，只是"偶尔成功"。

**根本原因**：
1. CONFIRMING 阶段只有一帧窗口，太短，无法可靠检测消失
2. `on_pick_done()` 不知道挑的是哪个 target，无法精准验证
3. Ghost 处理逻辑与 Stabilizer 存在 DDD 污染

### 改进方案

1. **`on_pick_done(target_id)` 带参数**：前端指定被挑的 target_id
2. **延长 CONFIRMING 窗口**：`confirm_window_frames`（默认 10 帧，可配置）
3. **精准计数**：只有指定 target 消失才 `region_picked += 1`
4. **挑取结果通知**：成功/失败通过 WebSocket 推送给前端
5. **临时禁用 Ghost**：`ghost_confirm_frames = 999999`

### 实施步骤

- [ ] 修改 `PickProcessConfig`：添加 `confirm_window_frames`
- [ ] 修改 `PickProcess`：
  - 新增 `_machine_picked_id`、`_confirm_frames_remaining` 属性
  - 修改 `on_pick_done(target_id)` 方法
  - 修改 `_handle_disappeared_targets()` 逻辑
  - 修改 `_finish_confirming()` 逻辑
  - 新增 `PickResult` 和 `get_last_pick_result()` 方法
- [ ] 修改 API：`/api/pick/done` 接收 `target_id` 参数
- [ ] 修改 WorkflowEngine：检查 `get_last_pick_result()` 并推送 WebSocket
- [ ] 更新配置文件：`ghost_confirm_frames = 999999`
- [ ] 测试验证

### 后续任务

- Stabilizer DDD 重构：解决 Ghost 处理的职责混乱问题

---

## 备注：可视化（Preview / Stream）当前画的是什么

目前画面上的框不是“Stabilizer 的 stable_targets 视图”，而是“PickProcess 的 tracked_targets 视图”。

现状逻辑（用于 debug/理解 ghost）：
1. `Stabilizer.update(detections)` 只输出 `stable_targets`（cluster.state == STABLE）
2. `PickProcess.update(stable_targets)` 用稳定目标驱动业务轨迹（Initial/Ghost、PENDING/PICKED、CONFIRMING 等）
3. `WorkflowEngine` 预览/推流优先画 `result.tracked_targets`（来自 `PickProcess.get_all_targets()`，包含 Initial + Ghost、PENDING + PICKED）
4. 只有在 `tracked_targets` 为空时，才 fallback 画原始 `detections`（warmup / 无目标）
5. 保存到磁盘的 annotated 图目前画的是 `detections`（不是 tracked_targets）

提示：
- 为了在画面上更容易分辨业务状态，`draw_stable_targets()` 的 label 已扩展为包含 `category/state`（例如 `id=12 cid=cluster_3 cat=ghost state=pending conf=0.87`）。
- 已在 `WorkflowConfig` 增加显示过滤开关（仅影响画面/推流，不影响业务逻辑）：
  - `display_only_pending=True`：只画 PENDING（隐藏 PICKED）
  - `display_hide_ghost=True`：隐藏 ghost
  建议先用这个组合跑一遍，确认幽灵框是否主要来自 Ghost/PICKED 的展示策略，而不是 Stabilizer 本身。

### 观察：幽灵框反复出现/消失会导致 track_id 快速自增

当目标/簇因为短暂稳定又很快消失时，会发生：
- Stabilizer 端 cluster 可能被清理（TTL / 缺失阈值），再次出现会生成新 cluster_id
- PickProcess 侧会把“新 cluster_id”当作新目标，分配新的 `track_id`
- 结果：画面里 track_id 不停增长，且同一物体看起来“换号”很频繁

TODO（后续可选方向）：
- 分离“显示用目标集合”与“业务用目标集合”：显示层过滤 ghost/PICKED/长时间未见；业务层保持更严格的确认
- 稳定 track_id：在 PickProcess 层引入“重关联”策略（基于位置/IoU/距离，把新 cluster 绑定回旧 track），或调大 cluster 清理阈值以减少重建

## 背景

当前后端采用“两进程”形态：
- **检测进程**：`WorkflowEngine` 驱动视觉主循环（采集 → 推理 → 稳定 → 存储/推送）
- **API 进程**：FastAPI 提供 REST/WebSocket/MJPEG

进程间通信目前使用 Redis 作为薄薄的一层中间件（事件推送 + MJPEG 帧 + command 请求/响应）。

## 现状问题（提醒）

目前 command 通信相关逻辑部分落在 `WorkflowEngine` 内（例如：轮询 Redis 命令、分发 action、回写 response）。

这能快速跑通，但在架构层面会带来耦合：
- `WorkflowEngine` 作为调度器/主循环，不应关心 Redis key、序列化协议等基础设施细节
- 若在通信层写死协议，会把更多 I/O 细节继续挤进 `WorkflowEngine`
- 违反“领域逻辑 / 通信适配层”分离的目标（Task/PickProcess 应专注业务状态机）

另外，“事件推送”目前也有类似风险：对外 payload（给前端计数器/看板用）与领域状态/统计混在一起，
并且通过 `TaskIterationResult.metadata` 这种“万能 dict”进行透传（例如 `region_picked/current_pending` 这类字段）。
短期方便，但会让领域含义不显式、消息契约不清晰，后续演进（换通道/换消费者/补充读模型）成本更高。

## TODO：通信抽象（设计已完成 / 实现落地）

> 详见：[通信架构](./07-communication.md) 第 3、4 节

**设计已完成**：
- ✅ 定义 `CommSignalBase` 接口（`receive()` / `send()` / `close()`）
- ✅ 设计 `CommunicationTask`（通信业务逻辑封装）
- ✅ 确定双通道架构（通信信号/事件推送）
- ✅ 确定通信信号通道 Redis/Modbus 互斥策略

**实现落地**：
- ✅ 实现 `RedisAdapter`（`backend/src/comm/`）
- ✅ 实现 `ModbusAdapter`（pymodbus）
- ✅ 实现 `CommunicationTask`
- ✅ 集成到 WorkflowEngine
- [ ] 定义 PLC 寄存器映射配置

**说明**：
- TaskManager 已迁移为 WorkflowEngine，调度逻辑在 `workflow/engine.py`。
- 本次仅保证通信链路落地，Task 体系的进一步拆分放在下一阶段处理。

## 完成标准（验收点）

- ✅ `WorkflowEngine` 不再引用 Redis key、Redis client、或任何通信协议细节（设计层面）
- ✅ 新增现场通信时，仅新增/替换适配器，不修改 `WorkflowEngine` 核心流程（设计层面）
- ✅ 前端 API 与现场通信共用同一套业务用例（request_target / pick_done / reset）（设计层面）
- [ ] 实际集成验证

---

## 备注：为什么要优先做 Engine 瘦身

现状问题（已确认）：
- 事件构造过于具体（payload 在 Engine 内拼装）
- Engine 直接访问 Task 内部状态（PickProcess），封装被破坏
- command_channel 与 RedisAdapter 职责重叠，语义污染基础设施层

这也是本次“第二点”的直接动机。
