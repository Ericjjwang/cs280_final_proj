# Product Context

## 项目目标

验证核心假设：**对透明/折射物体拍照时，LoFTR 在折射渲染（refractive）与哑光渲染（matte）之间的 Sampson 距离之比 > 2×**，而背景区域的比值应 < 1.5×（sanity check）。实验面向 4 分钟 oral presentation。

## 方法概述

1. 用 Blender 渲染同一场景的两路图像：`rgb/`（折射）和 `matte/`（哑光替代）
2. 以 `rgb/cond.png` 和 `matte/cond.png` 分别输入 Zero123++ v1.2，生成各自的 6 张新视角
3. 用 LoFTR 对 cond↔view 做半密集匹配，计算 Sampson epipolar 距离
4. 按 Blender mask 将匹配点分为 object 和 background 两区域
5. 计算每视角的 object/background Sampson 中位数，以及 refractive/matte 的比值
6. 跨视角、跨场景聚合，给出 PASS/FAIL 判定

## 技术栈

| 组件 | 细节 |
|------|------|
| 新视角合成 | Zero123++ v1.2 (`sudo-ai/zero123plus-v1.2`) |
| 特征匹配 | LoFTR outdoor weights (kornia), conf ≥ 0.5 |
| 相机约定 | OpenCV world-to-cam; Y-up; az CW from above |
| 匹配分辨率 | 320×320 (IMG_SIZE) |
| Blender 渲染分辨率 | 512×512 |
| 相机半径 | 1.5 m |
| FOV | 30° |
| 判定阈值 | object_ratio > 2.0 且 background_ratio < 1.5 |
| 包管理 | uv, Python 3.10, CUDA 12.1, RTX 4060 8GB |

## Zero123++ 视角约定

| view | azimuth | elevation |
|------|---------|-----------|
| cond | 0° | 0° |
| view_0 | 30° | +20° |
| view_1 | 90° | −10° |
| view_2 | 150° | +20° |
| view_3 | 210° | −10° |
| view_4 | 270° | +20° |
| view_5 | 330° | −10° |

## 场景数据规范（Blender 输出）

```
scene_root/
  rgb/    cond.png  view_0.png … view_5.png   512×512
  matte/  cond.png  view_0.png … view_5.png   512×512
  mask/   cond.png  view_0.png … view_5.png   512×512, 严格二值 {0,255}
  depth/  cond.exr  view_0.exr … view_5.exr   512×512, float32, ~1.5 m
  poses.json
```

## 关键设计决策

- **缓存键区分 rgb/matte**：`{model_id}__r{radius}__{sha256[:16]}__{label}`，label ∈ {"rgb","matte"}，避免碰撞
- **K 矩阵始终用 320×320**：即使 pose 来自 poses.json（512×512），匹配时统一用 `get_intrinsics(320, 30°)`
- **Blender mask 缩放**：512→320 用 nearest-neighbor 避免引入非二值像素
- **cond pose**：az=0°, el=0°（早期版本曾用 el=10°，已修正）
