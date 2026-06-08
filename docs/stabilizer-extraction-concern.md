# Stabilizer 抽离关切

## 背景
- Stabilizer 是跨帧累积的稳定化机制，负责将单帧 detections 聚合为 stable targets。
- 目前与 `StabilizedDetectionTask` 紧耦合。
- 新的 task 领域模型将 `StabilizedDetectionTask` 拆分为 task 链：`DetectionTask → StabilizationTask → PickTask`。
- Stabilizer 不只服务挑毛，任何需要跨帧稳定的工业视觉场景都可能需要。

## 开放问题
当 task 层抽离到 AutoWeaver（v0.2）时，Stabilizer 应该以什么形态进入框架？

### 方案 A：框架提供 `StabilizationTask`（独立 `ProcessTask`）
- 框架内置，开箱即用。
- 用户通过配置使用，不需要自己实现。
- 约束：固定了稳定化的接口和行为。

### 方案 B：框架提供 `Stabilizer` 工具类
- 作为工具类/组件提供。
- 用户在自己的 `task.process()` 里调用。
- 更灵活，但用户需要自己组合。

### 方案 C：两者都提供
- `StabilizationTask` 作为开箱即用的默认实现。
- `Stabilizer` 工具类作为底层能力，供高级用户自定义。

## 决策时机
- 不在当前阶段决定。
- 等挑毛项目完成 task 层重构 + 第二个项目（多区域检测）跑通后，再做决策。
- 届时对比两个项目的稳定化需求差异，指导选择。

## 相关文档
- `docs/backend/03-task-system.md`
- `docs/backend/03a-workflow.md`
