# 坐标转换模块

> 相关文档：[架构总览](./00-overview.md) | [Stabilizer](./04-stabilizer.md)

## 概述

CoordinateTransformer 负责将像素坐标转换为物理坐标（毫米），是连接视觉检测和机械臂控制的桥梁。

**设计决策**：坐标转换在 Backend 做，下游控制端收到的就是毫米坐标（协议待定）。

**理由**：
- 相机标定参数是视觉系统的责任
- Backend 可以做可视化验证
- 主站逻辑保持简单

---

## 接口定义

```python
┌───────────────────────────────────────────────────────────────────────┐
│                    CoordinateTransformer                              │
│                                                                       │
│   职责：像素坐标 → 物理坐标（毫米）                                    │
├───────────────────────────────────────────────────────────────────────┤
│  配置参数：                                                           │
│  - pixel_to_mm: float          # mm/pixel，相机标定得出               │
│  - offset_x: float             # 原点偏移 X（mm）                     │
│  - offset_y: float             # 原点偏移 Y（mm）                     │
│  - rotation: float             # 旋转角度（弧度）                     │
│                                                                       │
│  方法：                                                               │
│  + to_physical(targets: List[StableTarget]) -> List[PhysicalTarget]  │
│  + pixel_to_tray(px, py, img_height) -> (x_mm, y_mm)                 │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 数据类型

**输入**：`StableTarget`（像素坐标），来自 Stabilizer。详见：[Stabilizer](./04-stabilizer.md#stabletarget输出)

**输出**：`PhysicalTarget`（毫米坐标），发送给下游控制端（协议待定）。

---

## 坐标系定义

### 图像坐标系

```
(0,0) ──────────► X
  │
  │    图像
  │
  ▼
  Y
```

- 原点：左上角
- X 轴：向右
- Y 轴：向下

### 托盘坐标系（物理）

```
      Y
      ▲
      │
      │    托盘
      │
(0,0) ──────────► X
```

- 原点：托盘左下角（或用户定义）
- X 轴：向右
- Y 轴：向上

---

## 转换公式

**基础转换步骤**：
1. Y 轴翻转（图像坐标 Y 向下，托盘坐标 Y 向上）
2. 像素 → 毫米（乘以 `pixel_to_mm`）
3. 加上原点偏移
4. 可选：旋转变换

---

## 相机标定

### 标定方法

1. **棋盘格标定**（推荐）
   - 打印标准棋盘格（如 10x10mm 方格）
   - 拍摄多角度图像
   - 使用 OpenCV 标定工具

```bash
python tools/calibrate/camera_calibration.py --chessboard 9x6 --square-size 10
```

2. **简易标定**
   - 放置已知尺寸的物体（如 100mm x 100mm 方块）
   - 测量图像中的像素尺寸
   - `pixel_to_mm = 100mm / pixel_width`

### 标定输出

```yaml
motion:
  transform:
    pixel_to_mm: 0.08           # 相机标定得出
    offset_x: 0.0               # 托盘原点偏移
    offset_y: 0.0
    rotation: 0.0               # 弧度
```

---

## 配置示例

```yaml
motion:
  transform:
    pixel_to_mm: 0.08           # 每像素对应 0.08mm
    offset_x: -50.0             # 托盘原点相对图像原点的 X 偏移
    offset_y: -30.0             # Y 偏移
    rotation: 0.017             # 约 1 度旋转（弧度）
```

---

## 与其他模块集成

**输入来源**：从 Stabilizer 获取稳定目标（像素坐标）

**输出使用**：转换为物理坐标后发送给控制端

详见：[通信架构](./07-communication.md)

---

## 精度验证

### 验证方法

1. 放置已知位置的标记物
2. 运行检测 + 转换
3. 对比实际物理坐标与转换结果

工具：`tools/calibrate/verify_transform.py`

### 误差来源

| 误差源 | 影响 | 解决方案 |
|--------|------|---------|
| 镜头畸变 | 边缘坐标偏移 | 使用畸变校正 |
| 标定参数 | 整体缩放偏差 | 重新标定 |
| 检测抖动 | 随机噪声 | Stabilizer 已处理 |
| 机械振动 | 位置偏移 | 硬件优化 |

---

## 未来扩展

- **多相机融合**：将多个相机的目标转换到统一坐标系
- **畸变校正**：使用 OpenCV 畸变参数进行去畸变处理

---

## 参考

- OpenCV 相机标定：https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
- 坐标变换：齐次坐标与变换矩阵
- 工具：[calibrate/README.md](../../backend/tools/calibrate/README.md)
