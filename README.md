# pluck-hair_test — 测试调试台

基于 `pluck-hair` 复制改造的**独立测试程序**，与原程序作用于同一套硬件（同一台大恒相机、同一套 PLC/Epson/IO），但二者互相独立、不要同时运行。

界面一分为二：

```
┌───────────────────────────────┬──────────────────────────────┐
│  左半：实时画面                │  右半（上）：Epson 手动控制     │
│  · 相机实时视频                │  · X/Y/Z/U 点动 + 当前坐标      │
│  · 异物识别（辅助线同原程序）   │  · 目标坐标输入 / 示教点选择     │
│  · 镊子尖实时识别（醒目小叉）   │  · Move(直线)/Go(点到点) 执行    │
│  · 镊子尖↔预测夹取点 实际距离   ├──────────────────────────────┤
│    (mm)，用于判断是否到位       │  右半（下）：系统 + 设备 IO 控制 │
│                               │  · 自动/初始化/启动/停止/暂停/复位│
│                               │  · 气缸控制 / 其他控制 + 反馈状态 │
└───────────────────────────────┴──────────────────────────────┘
```

## 与原程序的差异（新增/改动）

| 项 | 说明 | 文件 |
|----|------|------|
| 镊子尖识别 | 传统 CV（无需训练数据）。闭合→单尖；张开→两尖连线上的偏移点。实时显示醒目小叉（洋红 `MARKER_CROSS`，大小与异物预测夹取点一致） | `backend/src/core/tweezer_detector.py` |
| 镊子↔预测点距离 | 用遥心镜头 `mm_per_pixel` 把像素距离换算成**实际 mm**，在画面与 `/api/test/state` 输出 | 同上 + `frame_loop.py` |
| 单进程测试模式 | `--mode test`：视觉引擎（后台线程）+ FastAPI + MJPEG 同进程，用进程内 FrameBus 替代 Redis | `backend/main.py`, `backend/src/comm/inproc_bus.py` |
| Epson 手动控制 + 设备 IO | 右半界面后端，`mock`/`modbus` 两种实现，地址全在 YAML | `backend/src/comm/epson_io_controller.py` |
| 通信占位配置 | **右半的通信条件/内容均为暂定编造值**，待现场确定后修改 | `backend/config/epson_io.yaml` |
| 前端拆分界面 | 左视频右控制 | `frontend/src/pages/TestConsolePage.tsx` 等 |

左半的检测/稳定/辅助线/裁切流程与 `settings.live.seg_stable53` **完全一致**，只是额外叠加了镊子识别。

## 运行

### 后端（同一进程跑视觉 + API + 视频流）

```bash
cd backend
uv run python main.py --config config/settings.test.yaml --mode test
```

- 监听 `0.0.0.0:8000`。需要大恒相机（`device_sn: FCM26010005`）与 GPU 模型可用。
- `plc_orchestrator` 默认开启（指向 192.168.1.88），现场无 PLC 时该侧任务会被跳过，左半仍可显示实时相机 + 镊子识别（异物识别由 PLC 拍照流程触发，与原程序一致）。

### 前端

```bash
cd frontend
deno task dev            # 默认连 http://localhost:8000/api
# 或指定后端： VITE_API_BASE=http://<backend-ip>:8000/api deno task dev
```

打开浏览器即为分屏界面。`/legacy` 仍保留原 HomePage。

## 镊子识别调参

本机标定图显示：**镊子（不锈钢）比亮灰底板更暗**，故默认用 `seg_method: dark`（`gray < dark_threshold`）。已在两张样张（闭合/张开）上验证：闭合→尖部单点；张开→两尖连线上的偏移点（`open_pick_ratio=0.35`，从上尖起算）。

全部参数在 `config/settings.test.yaml` 的 `tweezer:` 段，关键项：

- `seg_method`: `dark` | `edge` | `bright`
- `dark_threshold`: 金属判定阈值（底板更暗时调高）
- `entry_side`: 镊子进入方向（样张为 `right`，可设 `auto` 自动推断）
- `open_notch_depth_px` / `open_two_blob_min_ratio`: 张开/闭合判定灵敏度
- `open_pick_ratio` / `open_ratio_from`: 张开时偏移点位置
- `mm_per_pixel`: 像素→毫米（遥心镜头默认 0.009857）
- `debug_dump_dir`: 设置后会落盘 mask/overlay PNG，便于现场调阈值

> 样张是带十字标识的 HMI 截图（含全幅干扰线），真实相机帧无这些叠加线，识别更干净。现场首次部署建议开 `debug_dump_dir` 微调 `dark_threshold` 与形态学核。

## 右半通信配置（占位，待现场修改）

`config/epson_io.yaml` 里**所有寄存器/线圈地址都是编造的占位值**，目的是让整套界面今天就能用 `backend: mock` 跑通。现场确定真实协议后：

1. 改 `epson_io.backend: modbus`，填 `modbus.host` 等；
2. 按实际 Modbus 映射改各地址（Epson 位姿读写、点动、IO 线圈、反馈、系统按钮）；
3. `%MXx.y` 标签只是对照 SENKE HMI 截图，用于追溯。

`mock` 后端会模拟位姿点动、运动到点、IO 通断与反馈，无硬件即可联调界面。

## 新增 API（在原 API 之上）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/api/test/stream/video` | 进程内 MJPEG（相机+异物+镊子叠加，无需 Redis） |
| GET  | `/api/test/state` | 实时遥测：fps、镊子状态/尖坐标、尖↔预测点 mm |
| GET  | `/api/epson/describe` `/epson/pose` `/epson/points` | Epson 元数据 / 当前位姿 / 示教点 |
| POST | `/api/epson/jog` `/epson/move` | 点动 / 运动到目标(或示教点) Move\|Go |
| GET  | `/api/io/states` ; POST `/api/io/set` | IO 状态 / 通断 |
| POST | `/api/system` | 自动/初始化/启动/停止/暂停/复位/人工挑毛OK |
