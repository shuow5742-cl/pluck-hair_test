# 外参标定与坐标转换

> 相关文档：[坐标转换模块](./06-coordinate-transform.md) | [标定工具](../../backend/tools/calibrate/README.md)

## 背景

双臂协作场景：视觉臂（Eye-in-Hand）负责定位燕窝表面异物的世界坐标，执行臂负责挑毛。视觉臂只需输出毛的世界坐标系下的绝对物理位置，不关心执行臂的控制。

---

## 坐标系定义

```
像素坐标系 (u, v)
    │
    │  内参 + 深度 → 反投影
    ▼
相机坐标系 (Xc, Yc, Zc)
    │
    │  T_cam_to_ee（手眼标定）
    ▼
末端执行器坐标系 (Xee, Yee, Zee)
    │
    │  T_ee_to_base（机械臂 TCP 位姿，实时读取）
    ▼
视觉臂基座坐标系 (Xbase, Ybase, Zbase)
    │
    │  T_base_to_world（基座到世界坐标系偏移，标定一次）
    ▼
世界坐标系 (Xw, Yw, Zw)
```

---

## 完整转换公式

```
P_world = T_base_to_world × T_ee_to_base × T_cam_to_ee × P_camera
```

各项含义：

| 符号 | 含义 | 来源 |
|------|------|------|
| P_camera | 相机坐标系下的 3D 点 | 内参反投影 + 深度 |
| T_cam_to_ee | 相机→末端的刚体变换 | 手眼标定（已有） |
| T_ee_to_base | 末端→基座的变换 | 拍照时刻从机械臂读取 TCP 位姿 |
| T_base_to_world | 基座→世界坐标系的变换 | 基座偏移标定（一次性） |
| P_world | 世界坐标系下的物理位置 | 最终输出 |

---

## 第一步：像素→相机坐标系（内参反投影）

已知内参矩阵：

```
K = | fx  0  cx |
    | 0  fy  cy |
    | 0   0   1 |
```

给定像素坐标 (u, v) 和深度 Z：

```
Xc = (u - cx) * Z / fx
Yc = (v - cy) * Z / fy
Zc = Z
```

### 深度 Z 的获取

单目相机没有直接深度信息，需要根据场景确定：

**方案 A：固定工作距离（推荐，适合燕窝平面场景）**

如果燕窝放在固定平台上，视觉臂每次移动到固定高度拍照，则 Z 可以通过以下方式确定：

- Z = 已知的相机到工作平面的距离（通过臂的位姿和工作台高度计算）
- 适用条件：燕窝表面近似平面，高度变化远小于工作距离

**方案 B：多位姿三角化**

视觉臂在不同位置拍摄同一目标点，利用臂的位姿差异做三角化求解深度。精度更高但速度慢。

**方案 C：辅助测距**

加装线激光或点激光，成本低（几十元），可直接测量表面高度。

---

## 第二步：相机坐标系→末端坐标系（手眼标定）

这一步使用已有的手眼标定结果 `T_cam_to_ee`。

标定工具位于 `backend/tools/calibrate/eye_in_hand.py`，输出格式：

```yaml
T_cam_to_ee:
  position: {x, y, z}          # mm
  orientation: {rx, ry, rz}    # degrees
  rotation_matrix: [[...]]     # 3x3
  translation: [x, y, z]       # mm
```

构造 4x4 齐次变换矩阵：

```
T_cam_to_ee = | R  t |
              | 0  1 |
```

其中 R 是 3x3 旋转矩阵，t 是 3x1 平移向量。

转换：

```
P_ee = T_cam_to_ee × P_camera_homogeneous
```

---

## 第三步：末端坐标系→基座坐标系（TCP 位姿）

这一步不需要自己算正运动学。机械臂控制器在任意时刻都能直接返回末端相对于基座的位姿。

**获取方式**（取决于臂的品牌）：

| 品牌 | 接口 | 返回格式 |
|------|------|---------|
| UR | `get_actual_tcp_pose()` | [x, y, z, rx, ry, rz] (mm + 轴角) |
| FANUC | CURPOS 寄存器 | [x, y, z, w, p, r] (mm + 欧拉角) |
| ABB | `CRobT()` | Pos + Orient |
| 越疆 Dobot | `GetPose()` | [x, y, z, r] |
| 其他 | 查对应 SDK | 通常都有类似接口 |

**关键**：在拍照的同一时刻记录 TCP 位姿，确保时间同步。

将返回的位姿转换为 4x4 齐次变换矩阵 `T_ee_to_base`：

```python
import numpy as np
from scipy.spatial.transform import Rotation

def tcp_pose_to_matrix(position, orientation, angle_format='euler_degrees'):
    """
    将机械臂 TCP 位姿转换为 4x4 齐次变换矩阵。

    Args:
        position: [x, y, z] 单位 mm
        orientation: [rx, ry, rz] 旋转表示
        angle_format: 'euler_degrees' | 'rotvec' (UR 用轴角)
    """
    T = np.eye(4)
    if angle_format == 'rotvec':
        R = Rotation.from_rotvec(orientation).as_matrix()
    else:
        R = Rotation.from_euler('xyz', orientation, degrees=True).as_matrix()
    T[:3, :3] = R
    T[:3, 3] = position
    return T
```

---

## 第四步：基座坐标系→世界坐标系（基座偏移标定）

视觉臂的基座安装位置和世界坐标系原点之间存在固定偏移，需要标定一次 `T_base_to_world`。

### 标定方法

#### 情况 1：仅平移（基座坐标轴与世界坐标轴平行）

如果视觉臂基座的坐标轴方向和世界坐标系一致，只是原点位置不同：

1. 在世界坐标系中选一个已知点 P_world（比如工作台上的固定标记）
2. 让视觉臂末端精确触碰该点
3. 读取臂报告的基座坐标系下的位置 P_base
4. 偏移量 = P_world - P_base

```yaml
T_base_to_world:
  translation: [dx, dy, dz]    # mm
  rotation: identity            # 无旋转
```

#### 情况 2：平移 + 旋转（基座安装有角度）

如果基座坐标轴和世界坐标系之间有旋转关系：

1. 选取至少 3 个不共线的已知世界坐标点
2. 让臂末端依次触碰，记录每个点的基座坐标
3. 用刚体变换求解（SVD 方法）：

```python
def calibrate_base_to_world(points_base, points_world):
    """
    通过对应点对求解 T_base_to_world。

    Args:
        points_base: Nx3 数组，基座坐标系下的点
        points_world: Nx3 数组，世界坐标系下的对应点
    Returns:
        T: 4x4 齐次变换矩阵
    """
    centroid_base = np.mean(points_base, axis=0)
    centroid_world = np.mean(points_world, axis=0)

    base_centered = points_base - centroid_base
    world_centered = points_world - centroid_world

    H = base_centered.T @ world_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # 确保是正交旋转（行列式为 +1）
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_world - R @ centroid_base

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
```

---

## 完整转换代码

```python
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


class ExtrinsicTransformer:
    """像素坐标 → 世界坐标的完整转换。"""

    def __init__(self, intrinsic_path, hand_eye_path, base_to_world_path):
        self.K, self.dist = self._load_intrinsic(intrinsic_path)
        self.T_cam_to_ee = self._load_hand_eye(hand_eye_path)
        self.T_base_to_world = self._load_base_to_world(base_to_world_path)

    def pixel_to_world(self, u, v, z_depth, tcp_pose, angle_format='euler_degrees'):
        """
        将像素坐标转换为世界坐标。

        Args:
            u, v: 像素坐标
            z_depth: 深度值 (mm)
            tcp_pose: (position, orientation) 拍照时刻的臂 TCP 位姿
            angle_format: TCP 姿态的角度格式

        Returns:
            (x, y, z) 世界坐标 (mm)
        """
        # 1. 像素 → 相机坐标系
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        P_cam = np.array([
            (u - cx) * z_depth / fx,
            (v - cy) * z_depth / fy,
            z_depth,
            1.0
        ])

        # 2. 相机 → 末端
        P_ee = self.T_cam_to_ee @ P_cam

        # 3. 末端 → 基座
        position, orientation = tcp_pose
        T_ee_to_base = tcp_pose_to_matrix(position, orientation, angle_format)
        P_base = T_ee_to_base @ P_ee

        # 4. 基座 → 世界
        P_world = self.T_base_to_world @ P_base

        return P_world[:3]

    def _load_intrinsic(self, path):
        with open(path) as f:
            data = yaml.safe_load(f)
        cm = data['camera_matrix']
        K = np.array([
            [cm['fx'], 0, cm['cx']],
            [0, cm['fy'], cm['cy']],
            [0, 0, 1]
        ])
        d = data['distortion']
        dist = np.array([d['k1'], d['k2'], d['p1'], d['p2'], d['k3']])
        return K, dist

    def _load_hand_eye(self, path):
        with open(path) as f:
            data = yaml.safe_load(f)
        T = np.eye(4)
        T[:3, :3] = np.array(data['T_cam_to_ee']['rotation_matrix'])
        T[:3, 3] = np.array(data['T_cam_to_ee']['translation'])
        return T

    def _load_base_to_world(self, path):
        with open(path) as f:
            data = yaml.safe_load(f)
        T = np.eye(4)
        if 'rotation_matrix' in data:
            T[:3, :3] = np.array(data['rotation_matrix'])
        T[:3, 3] = np.array(data['translation'])
        return T
```

---

## 精度分析

目标精度：10 μm（0.01 mm）

### 误差来源与预估

| 误差源 | 典型量级 | 备注 |
|--------|---------|------|
| 内参标定（重投影误差） | 0.1~0.5 px → 约 0.008~0.04 mm | 取决于标定质量 |
| 手眼标定 | 0.1~0.5 mm | OpenCV 方法的典型精度 |
| 机械臂重复定位精度 | ±0.02~0.05 mm | 取决于臂的型号 |
| 深度估计误差 | 取决于方案 | 固定高度方案误差最小 |
| 基座偏移标定 | 0.05~0.2 mm | 取决于标定点的精度 |

### 提升精度的建议

1. **手眼标定**：采集 15~20 组以上位姿，覆盖不同角度，使用多种方法对比选最优
2. **基座标定**：使用高精度标定块或量具确定世界坐标系参考点
3. **深度**：如果用固定工作距离方案，确保平台平整度在 0.01 mm 以内
4. **去畸变**：在反投影之前先对像素坐标做去畸变处理（`cv2.undistortPoints`）
5. **温度**：工业相机和机械臂在温度变化时会有热漂移，建议在稳定温度环境下标定和工作

---

## 标定流程总结

```
1. 内参标定（已完成）
   └─ 输出：camera_intrinsic.yaml

2. 手眼标定（已完成）
   └─ 工具：backend/tools/calibrate/eye_in_hand.py
   └─ 输出：T_cam_to_ee（hand_eye_result.yaml）

3. 基座偏移标定（待实现）
   └─ 选取世界坐标系参考点
   └─ 让臂末端触碰参考点，记录基座坐标
   └─ 计算 T_base_to_world
   └─ 输出：base_to_world.yaml

4. 运行时转换
   └─ 拍照 + 同步读取 TCP 位姿
   └─ 调用 ExtrinsicTransformer.pixel_to_world()
   └─ 输出世界坐标给执行臂
```

---

## 参考

- OpenCV 手眼标定：https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b
- 刚体变换求解（SVD）：Arun, K.S., Huang, T.S., Blostein, S.D. (1987)
- 标定工具：[calibrate/README.md](../../backend/tools/calibrate/README.md)
