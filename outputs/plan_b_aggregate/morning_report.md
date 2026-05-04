# Plan B Morning Report

Generated from: outputs/plan_b_aggregate

---

## 1. Verify Status

- **scene_brass_goblet_v2**: PASS
- **scene_glass_suzanne**: PASS

## 2. Dual-Path object_sampson_ratio

### scene_brass_goblet_v2
- Per-view ratios: ['2.12', '6.27', '5.99', '2.50', '1.85']
- Median: 2.50x  |  Mean: 3.75x
- Background Sampson mean: 7.47 px

### scene_glass_suzanne
- Per-view ratios: ['0.40', '0.99']
- Median: 0.69x  |  Mean: 0.69x
- Background Sampson mean: 21.22 px

## 3. Key Figures

- `outputs/plan_b_aggregate/figure_2_ratio_scatter.png` — ✓ exists
- `outputs/plan_b_aggregate/figure_3_per_view_pattern.png` — ✓ exists
- `outputs/plan_b_aggregate/all_scenes_summary.csv` — ✓ exists

## 4. Anomalies

```
[15:37:37] aggregate_scenes failed
[15:37:37] morning report generation failed
```

## 5. Verdict

Scene **scene_brass_goblet_v2**: object_ratio mean=3.75x (median=2.50x, range 1.85–6.27x). 显著高于背景 (object ratio >> 1). BG Sampson mean = 7.47 px.
Scene **scene_glass_suzanne**: object_ratio mean=0.69x (median=0.69x, range 0.40–0.99x). 反向! (object ratio < 1 — glass may match better than matte). BG Sampson mean = 21.22 px.

**Background Sampson** across both scenes: mean = 13.72 px (≥ 5 px — high)

**Recommended main case**: `scene_brass_goblet_v2` (highest object_ratio mean = 3.75x). Use for figure_1 and presentation examples.
