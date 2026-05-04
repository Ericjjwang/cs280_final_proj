# Dual-Path Experiment Report

Generated: 2026-05-03

## Per-Scene Summary

| Scene | Views | Obj Ref px | Obj Mat px | Obj Ratio | BG Ref px | BG Mat px | BG Ratio | Verdict |
|-------|-------|-----------|-----------|----------|----------|----------|----------|---------|
| _fake_scene | 3 | N/A | N/A | N/A | N/A  | N/A  | N/A | ✗ FAIL |
| scene_brass_goblet_v2 | 6 | 24.70 | 6.41 | 2.50× | 25.53  | 17.51  | 0.96× | ✓ PASS |
| scene_glass_suzanne | 6 | 11.87 | 11.03 | 0.69× | 25.84  | 24.92  | 0.74× | ✗ FAIL |
| scene_v2_test | 6 | N/A | 6.62 | N/A | 24.66  | 30.85  | 1.58× | ✗ FAIL |

## Aggregate

- Total scenes: 4
- PASS: 1 / 4 (25%)
- Object ratio — mean: 1.60×,  range: 0.69× – 2.50×
- Background ratio — mean: 1.09×

## Thresholds

- Object Sampson ratio > **2.0×** (refractive / matte)
- Background Sampson ratio < **1.5×** (sanity bound)
- Views with fewer than 20 LoFTR matches are excluded from ratio computation.

## Figure recommendations for presentation

### Figure 1 — best (scene, view) candidate per scene

- **scene_brass_goblet_v2** — view `1` (object ratio 6.27×, background ratio 2.62×)
- **scene_glass_suzanne** — view `5` (object ratio 0.99×, background ratio 1.28×)

Generate with:
```
uv run python scripts/make_figure_1.py \
    --auto-select \
    --results-dir outputs/dual_path \
    --output-dir outputs/dual_path \
    --cache-dir outputs/_cache
```

### Figure 2 — ratio scatter

`figure_2_ratio_scatter.png` — object vs background ratio, all (scene, view) pairs.

### Figure 3 — per-view pattern

`figure_3_per_view_pattern.png` — per-view object ratio overlaid for all scenes, with elevation shading.
