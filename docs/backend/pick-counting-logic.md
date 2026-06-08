# Pick Counting Logic - 待重构

## 当前问题

pick_done 计数逻辑与 Stabilizer 的滞后机制存在时序冲突。

## 组件关系

```
Stabilizer (技术域)
    ↓ stable_targets (cluster_id)
PickProcess (业务域)
    ↓ tracked_targets (track_id)
可视化显示
```

## Stabilizer 滞后机制

- `window_size: 10` - 滑动窗口
- `stable_exit_ratio: 0.3` - 目标需要在窗口中出现率低于 30% 才认为消失
- 实际效果：目标被挑走后，需要 7-8 帧才能从 Stabilizer 中消失

## 原有计数逻辑问题

1. `pick_done` 触发 → phase=CONFIRMING → 等待 10 帧
2. 如果 10 帧内目标从 Stabilizer 消失 → 计数成功
3. 如果 10 帧内目标仍在 Stabilizer 中 → 失败 → phase=READY
4. 之后目标从 Stabilizer 消失 → 因为 phase=READY，目标被**删除**而不是计数
5. 再次 `pick_done` → 目标已不存在 → 无法计数

## 临时修复 (2026-01-14)

在 `on_pick_done` 中增加判断：如果 target 已经不存在于 PickProcess，直接计数。

```python
if target is None:
    # Target already gone = already picked, count it directly
    self._region_picked += 1
    self._last_pick_result = PickResult(success=True, ...)
    return
```

理由：目标从 PickProcess 消失的唯一原因是它从 Stabilizer 中消失了，说明确实被挑走了。

## 待重构方向

1. 解耦 Stabilizer 和 PickProcess 的消失判断逻辑
2. 考虑增加 `confirm_window_frames` 或减少 Stabilizer 的 `stable_exit_ratio`
3. 或者改用其他机制判断目标是否被挑走（如检测区域变化）
