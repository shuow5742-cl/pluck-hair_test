---
title: Frame-loop freeze + JSON-RPC detect_once index
summary: 记录临时的 frame-loop 冻结补丁以及未来将 request_target/pick_done 流映射到 JSON-RPC detect_once 的步骤，便于后续拆分
---

# 临时冻结补丁说明

1. 这是一个临时补丁，核心目的是在最小范围内让 `frame-loop` 在派发目标之后暂停，在 `pick_done/reset` 之后恢复，避免镊子进入视野时继续推进视觉计算。
2. 补丁明确了 `frame-loop` 线程在 `backend/src/tasks/frame_loop.py` 中以 `EventBus` 订阅 `FRAME_LOOP:PAUSE`/`FRAME_LOOP:RESUME`，暂停时调用 `threading.Event.wait()`，恢复时放行。该事件只影响主循环本身，不会重置 `Stabilizer`/`PickProcess` 数据，所以状态保持在内存中。
3. 补丁的触发语义：
   - `StabilizedDetectionTask` 在 `COMM:REQUEST_TARGET` 响应后立刻广播 `FRAME_LOOP:PAUSE` 并附上 `track_id`，表示目标已发给机械臂，后续帧应暂停。
   - 通信层收到 `COMM:PICK_DONE` 或 `COMM:RESET` 时会广播 `FRAME_LOOP:RESUME`（`pick_done` 带上 `track_id`），解冻主循环供确认帧使用。
   - 关闭/重置时也放开 gate，防止线程卡住。

# 当前状态流（request_target / pick_done）

1. 现在仓库里仍然是 Modbus/前端调用 `request_target` 拿目标、`pick_done` 报告完成的双事件流程。
2. 相关文件：
   - `backend/src/tasks/stabilized_detection/task.py`：处理 `COMM:REQUEST_TARGET`、`COMM:PICK_DONE`、`COMM:RESET`，并在 `COMM:TARGET_RESPONSE`/`COMM:PICK_DONE` 处插入 pause/resume 事件。【该文件需保留】
   - `backend/src/tasks/frame_loop.py`：实现 pause/resume gate，订阅 `FRAME_LOOP:*` 事件。【该文件需保留】
   - `backend/src/tasks/communication/task.py`：Modbus/WebSocket 轮询线程，向 `EventBus` 发表 `COMM:*` 事件。【未来需要调整】
   - `backend/src/tasks/stabilized_detection/pick_process.py`：当前仍是 `READY -> AWAITING_PICK -> CONFIRMING -> READY/DONE` 的旧业务状态机，还没有 `retry_1/retry_2/abandon` 这套 JSON-RPC 语义。
3. 当前补丁只解决“冻结 frame-loop，避免镊子污染画面”，没有改变 `PickProcess` 的核心业务语义。也就是说，`retry_1`、`retry_2`、第三次失败后跳过目标，这些仍然是后续 JSON-RPC 接入时要补上的业务逻辑。

# 面向未来的 JSON-RPC detect_once 接入方向

1. 目标：把 `request_target`/`pick_done` 两步折成一条 `detect_once`，每次调用既是“给目标又兼做恢复信号”。
2. 预期映射（暂不写代码）：
   - 第一次 `detect_once`：当前没有在途目标，选出目标、广播 `FRAME_LOOP:PAUSE`，返回 `state=new_target` + `x/y/u`。  
   - 接下来每次 `detect_once`：默认表示上一次机械臂动作完成，先广播 `FRAME_LOOP:RESUME`，让 `frame-loop` 确认目标是否消失，然后：
     * 若原目标消失 → 立即准备下一个目标（`new_target` 或 `no_target`）。
     * 若仍在 → 封装 `retry_1`/`retry_2` 并重发坐标，同时再次 pause。
     * 第三次还在 → 直接 `abandon` 该目标，切到下一个。
3. 所需修改点索引：
   - `backend/src/tasks/communication/task.py`：改 `handle_message` 逻辑，新增 `detect_once` handler，内部仍可复用 `StabilizedDetectionTask` 的事件。完成后删除或保留 `request_target`/`pick_done` 命令需由团队决定。
   - `backend/src/tasks/stabilized_detection/task.py`：保持 pause/resume 事件，但需要一条新的 `FRAME_LOOP:RESUME` 触发路径，来源从 `pick_done` 变成“下一次 `detect_once` 请求到来”。`u` 的计算和 `state=new_target/retry_n/no_target` 的封装也会在这一层或其下游完成。
   - `backend/src/tasks/stabilized_detection/pick_process.py`：需要补上 retry 计数、第三次失败后 abandon、切下一个目标等业务状态。
   - `backend/src/api/routes/events.py` 及前端：如果前端仍要走 WebSocket 调试命令，需要决定是继续复用旧的 `request_target/pick_done/reset`，还是一起切到新的 `detect_once` 语义。
   - `tmp/comm/pluck_hair_ModbusTCP/JSON-RPC 对接说明.md`：协议真值文档在这里，接入时要以该文档为准核对 `x/y/u/state` 返回格式。

# 拆补丁重点

 - 在移除 `request_target`/`pick_done` 事件时，先确保 `FRAME_LOOP:PAUSE/RESUME` 仍然由新的控件驱动，不要直接删掉 `frame-loop` 的 gate。
 - 把状态机（`PickProcess`）的确认逻辑依然留在 `run()` 周期内，只是触发点从 `COMM:PICK_DONE` 变成“下一次 `detect_once` 到来后的恢复确认阶段”。
 - 确保 `StabilizedDetectionTask` 在新流程中仍会在目标选中后立即 pause，如果后续 `detect_once` 判断为复抓，它也能再次 pause。
 - 如果未来彻底移除这个补丁，必须先有新的“视觉暂停/恢复”机制替代它，否则镊子进画面时会再次污染 `Stabilizer` 和业务状态。

# 时序/状态参考

1. 当前（request_target/pick_done）时序：
   - `request_target` → `PickProcess.get_next_target()` → `COMM:TARGET_RESPONSE` → `FRAME_LOOP:PAUSE` → 机械臂运动。
   - 机械臂完成后调用 `pick_done(track_id)` → `COMM:PICK_DONE` → `PickProcess.on_pick_done`（`CONFIRMING`）→ `FRAME_LOOP:RESUME` → `PickProcess` 观察目标消失 → 下一个目标或 `DONE`。
2. 未来（detect_once）时序：
   - `detect_once`(no pending) → 选目标 → pause → 返回 `new_target`。
   - `detect_once`(pending) → resume → run 1-2 帧确认 → 目标消失➡切新目标；目标仍在➡`retry_n` + pause；三次失败➡`abandon` + next target。
3. 每次 `FRAME_LOOP:PAUSE` 都必须伴随一次与之对应的 `FRAME_LOOP:RESUME`，任何跳过会导致主循环永久挂起；审查代码时请保留这对事件。

补丁拆除时建议按这个清单逐项核对并更新文档，确保未来团队能快速定位并恢复或替换这段临时逻辑。
