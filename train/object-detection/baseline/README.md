# 统一检测基线

基于多框架（当前实现 YOLO）的一体化训练入口。详细的架构说明参见 `docs/ARCHITECTURE.md`。

## 快速开始

1. 创建并激活虚拟环境（推荐 uv）：
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -e ".[yolo]"
   ```
2. 运行最小 YOLO 烟雾实验：
   ```bash
   python scripts/train.py --config configs/experiments/exp_yolo_dota8.yaml
   ````
   该配置指向 `dataset/dota8` 迷你数据集，适合验证流程。

### Orchestrator（隔离模式）

Phase 1 开始引入多框架 orchestrator。以 YOLO 子工程为例：

```bash
uv run python orchestrator/scripts/train.py \
  --config configs/experiments/exp_orchestrator_yolo_dota8.yaml
```

该命令会在 `experiments/yolov11_dota8_iso/` 下生成 orchestrator/框架日志与 `metrics.json`，并通过 `uv run --project frameworks/yolo` 触发独立的 YOLO 训练环境。

## 目录速览

- `configs/`：实验与训练配置（YAML），`configs/experiments/` 中包含示例。
- `src/core/`：配置加载、注册器、公共接口、`DetectionResult`。
- `src/detectors/`：各框架封装（目前已接入 YOLO）。
- `src/runners/`：对应框架的训练调度器（`yolo_runner`）。
- `src/engine/`：`Trainer`、回调及日志管理。
- `docs/ARCHITECTURE.md`：详细设计文档。

## 下一步

- 扩展 MMDet/MMRotate 训练流程与 Runner。
- 增加推理/评估脚本、SAHI/集成等推理插件。
- 结合更完整的数据集与实验管理工具（TensorBoard/WandB）。 
