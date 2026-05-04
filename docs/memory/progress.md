# Progress

_最后更新：2026-05-03_

## 已完成

### Task A — 环境与协作准备
- [x] `pyproject.toml` 添加 `rembg` 依赖
- [x] 创建 `.gitignore`（排除 `outputs/`, `__pycache__/`, `.venv/`, `.cache/`）

### Task B — 数据 IO 与场景验证
- [x] `src/io_utils.py`：4 个函数
  - `load_blender_alpha_mask` — 512×512 bool mask，阈值 127
  - `load_blender_depth_exr` — float32，背景为 inf（需 OPENCV_IO_ENABLE_OPENEXR=1）
  - `load_poses_json` / `save_poses_json`
- [x] `scripts/verify_scene_data.py`：14 项检查，输出 PASS/FAIL + `verification_report.json`
- [x] `scripts/_make_fake_scene.py`：生成全通过的测试场景 `outputs/_fake_scene/`

### Task C — 双路分析数据结构
- [x] `src/pipeline.py` 新增模块级私有辅助函数：
  - `_make_cache_key`, `_resize_mask_to`, `_az_el_from_pose`
  - `_generate_or_load_views`, `_run_pair`
- [x] `DualPathResult` dataclass（含 `to_json(include_arrays=False)` / `to_csv_rows()`）
- [x] `analyze_dual_path()` 函数

### Task D — 实验脚本
- [x] `scripts/run_dual_path_experiment.py`：单场景 CLI
  - 输出：`results.json`, `results_full.json`（含 arrays）, `summary.csv`, `scene_summary.txt`
- [x] `scripts/aggregate_scenes.py`：多场景聚合
  - 默认输出：`all_scenes_summary.csv`, figure_2, figure_3, `report.md`
  - `--extra-figures` 才生成 boxplot

### Task E — 演示图表（4 分钟 oral）
- [x] `scripts/make_ratio_scatter.py`
  - Figure 2：object_ratio（log x）vs background_ratio（linear y）散点图
  - PASS 象限绿色阴影，参考线 x=2.0/y=1.5，场景着色，视角编号标注
  - 可 import：`from scripts.make_ratio_scatter import make_ratio_scatter`
- [x] `scripts/make_per_view_pattern.py`
  - Figure 3：V0–V5 各视角 object_ratio，所有场景叠加
  - 偶数视角（el=+20°）蓝底，奇数（el=−10°）黄底；加粗黑色均值线
  - 可 import：`from scripts.make_per_view_pattern import make_per_view_pattern`
- [x] `scripts/make_figure_1.py`
  - Figure 1：2×2 panel（refractive/matte 行 × cond/view 列）
  - 右列叠加 Sampson 热图（RdYlGn_r）+ 白色 mask 边界轮廓
  - 单场景模式：`--scene-dir --view-idx --output`
  - 批量模式：`--auto-select --results-dir --output-dir`（按 summary.csv 选最高 object_ratio 视角）

## 当前状态

- fake scene 全部验证通过（14/14 PASS）
- fake scene 实验已跑（3 个视角，全部 N/A — 纯色图无匹配，符合预期）
- 三张 figure 布局已目视确认，layout 正确
- `report.md` 包含 Figure recommendations 章节
- **等待真实 Blender 场景数据**

## 待办

- [ ] Blender 渲染真实透明物体场景（按 `verify_scene_data.py` 规范）
- [ ] 在真实场景上运行 `run_dual_path_experiment.py`（需 GPU，~10 min/场景）
- [ ] 运行 `aggregate_scenes.py` 聚合多场景
- [ ] 用 `make_figure_1.py --auto-select` 生成最终演示 figure_1
- [ ] 检验核心假设：object_ratio > 2.0 且 background_ratio < 1.5

## 主要文件索引

```
src/
  io_utils.py          — Blender 数据 IO
  pipeline.py          — PairResult, DualPathResult, analyze_dual_path, _run_pair 等
  camera.py            — 相机约定常数，get_intrinsics, get_zero123plus_poses
  generation.py        — Zero123PlusPipeline
  matching.py          — LoFTRMatcher
  epipolar.py          — compute_F, sampson_distance, sampson_stats
  region_analysis.py   — segment_object_background, split_matches_by_region
  viz.py               — plot_sampson_heatmap, plot_4panel, save_figure

scripts/
  verify_scene_data.py        — 14 项场景验证
  _make_fake_scene.py         — 生成测试场景
  run_dual_path_experiment.py — 单场景实验 CLI
  aggregate_scenes.py         — 多场景聚合 + 默认 figure_2/3
  make_ratio_scatter.py       — Figure 2（可 import）
  make_per_view_pattern.py    — Figure 3（可 import）
  make_figure_1.py            — Figure 1 2×2 panel（可 import）

outputs/
  _fake_scene/         — 测试场景（Blender 格式，纯色）
  _cache/              — Zero123++ 生成缓存
  dual_path/
    _fake_scene/       — fake scene 实验结果
    figure_2_ratio_scatter.png
    figure_3_per_view_pattern.png
    report.md
```

## 常用命令

```bash
# 验证场景
uv run python scripts/verify_scene_data.py data/my_scene/

# 单场景实验（GPU，~10 min）
uv run python scripts/run_dual_path_experiment.py \
  --scene-dir data/my_scene \
  --output-dir outputs/dual_path/my_scene \
  --cache-dir outputs/_cache

# 多场景聚合 + figure_2/3
uv run python scripts/aggregate_scenes.py \
  --results-dir outputs/dual_path

# Figure 1（批量，选最佳视角，需 GPU）
uv run python scripts/make_figure_1.py \
  --auto-select \
  --results-dir outputs/dual_path \
  --output-dir outputs/dual_path \
  --cache-dir outputs/_cache
```
