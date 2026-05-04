# Dual-Path Experiment Report

Generated: 2026-05-03

## Per-Scene Summary

| Scene | Views | Obj Ref px | Obj Mat px | Obj Ratio | BG Ref px | BG Mat px | BG Ratio | Verdict |
|-------|-------|-----------|-----------|----------|----------|----------|----------|---------|
| _fake_scene | 3 | N/A | N/A | N/A | N/A  | N/A  | N/A | ✗ FAIL |

## Aggregate

- Total scenes: 1
- PASS: 0 / 1 (0%)
- Object ratio — mean: N/A,  range: N/A
- Background ratio — mean: N/A

## Thresholds

- Object Sampson ratio > **2.0×** (refractive / matte)
- Background Sampson ratio < **1.5×** (sanity bound)
- Views with fewer than 20 LoFTR matches are excluded from ratio computation.

## Figure recommendations for presentation

### Figure 1 — best (scene, view) candidate per scene

*(no scenes with valid ratios)*

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
