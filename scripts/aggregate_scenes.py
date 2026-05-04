"""
Aggregate dual-path experiment results across multiple scenes.

Reads  :  {results_dir}/*/summary.csv
           {results_dir}/*/results.json  (for verdicts / scene_dir)

Default outputs
---------------
  all_scenes_summary.csv
  figure_2_ratio_scatter.png    (object vs background ratio, all (scene,view) pairs)
  figure_3_per_view_pattern.png (per-view object ratio, all scenes overlaid)
  report.md

Optional (--extra-figures)
--------------------------
  boxplot_object_sampson.png
  boxplot_background_sampson.png

Usage
-----
    uv run python scripts/aggregate_scenes.py \\
        --results-dir outputs/dual_path \\
        [--output-dir outputs/dual_path] \\
        [--extra-figures] \\
        [--object-ratio-threshold 2.0] \\
        [--bg-ratio-threshold 1.5]
"""

from __future__ import annotations
import argparse
import csv
import json
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Figure-function imports (same directory; sys.path extended by uv run)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_ratio_scatter import make_ratio_scatter
from make_per_view_pattern import make_per_view_pattern


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_scene_data(results_dir: Path) -> dict[str, list[dict]]:
    """Returns {scene_id: [row_dict, …]} from each scene's summary.csv."""
    scene_data: dict[str, list[dict]] = {}
    for csv_path in sorted(results_dir.glob("*/summary.csv")):
        scene_id = csv_path.parent.name
        rows: list[dict] = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                parsed: dict = {}
                for k, v in row.items():
                    if v == "" or v.lower() in ("none", "null", "n/a"):
                        parsed[k] = None
                    else:
                        try:
                            parsed[k] = float(v)
                        except ValueError:
                            parsed[k] = v
                rows.append(parsed)
        if rows:
            scene_data[scene_id] = rows
    return scene_data


def _load_verdicts(results_dir: Path, scene_ids: list[str]) -> dict[str, str]:
    verdicts: dict[str, str] = {}
    for sid in scene_ids:
        j = results_dir / sid / "results.json"
        if j.exists():
            with open(j) as f:
                data = json.load(f)
            verdicts[sid] = data.get("scene_aggregate", {}).get("verdict", "?")
    return verdicts


def _load_thresholds(results_dir: Path) -> tuple[float, float]:
    for j in sorted(results_dir.glob("*/results.json")):
        with open(j) as f:
            data = json.load(f)
        thr = data.get("thresholds", {})
        return thr.get("object_ratio_threshold", 2.0), thr.get("background_ratio_threshold", 1.5)
    return 2.0, 1.5


# ── Statistics ────────────────────────────────────────────────────────────────

def _nanmedian(values: list) -> Optional[float]:
    valid = [x for x in values if x is not None and np.isfinite(float(x))]
    return float(np.median(valid)) if valid else None


def _scene_agg(rows: list[dict]) -> dict:
    return {
        "obj_ref":   _nanmedian([r.get("object_sampson_refractive_median")     for r in rows]),
        "obj_mat":   _nanmedian([r.get("object_sampson_matte_median")          for r in rows]),
        "obj_ratio": _nanmedian([r.get("object_sampson_ratio")                 for r in rows]),
        "bg_ref":    _nanmedian([r.get("background_sampson_refractive_median") for r in rows]),
        "bg_mat":    _nanmedian([r.get("background_sampson_matte_median")      for r in rows]),
        "bg_ratio":  _nanmedian([r.get("background_sampson_ratio")             for r in rows]),
        "n_views":   len(rows),
    }


# ── Writers ───────────────────────────────────────────────────────────────────

def _write_all_csv(path: Path, scene_data: dict[str, list[dict]]) -> None:
    all_rows = []
    for sid, rows in scene_data.items():
        for row in rows:
            if "scene_id" not in row or row.get("scene_id") != sid:
                all_rows.append({"scene_id": sid, **row})
            else:
                all_rows.append(row)
    if not all_rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"  Saved {path.name}")


def _boxplot_sampson(
    path: Path,
    scene_data: dict[str, list[dict]],
    col_ref: str,
    col_mat: str,
    title: str,
) -> None:
    scene_ids = list(scene_data.keys())
    n = len(scene_ids)
    if n == 0:
        return

    ref_data = [
        [r[col_ref] for r in rows if r.get(col_ref) is not None]
        for rows in scene_data.values()
    ]
    mat_data = [
        [r[col_mat] for r in rows if r.get(col_mat) is not None]
        for rows in scene_data.values()
    ]

    ref_pos = np.arange(1, n * 3, 3, dtype=float)
    mat_pos = ref_pos + 1.0

    fig, ax = plt.subplots(figsize=(max(6, n * 2.8), 5))

    def _draw_boxes(data_list: list, positions: np.ndarray, color: str) -> None:
        safe = [d if d else [np.nan] for d in data_list]
        bp = ax.boxplot(
            safe, positions=positions, widths=0.7, patch_artist=True,
            medianprops=dict(color="black", lw=2),
            whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2),
            flierprops=dict(marker="o", markersize=4, alpha=0.5, markeredgewidth=0.5),
            showfliers=True,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

    _draw_boxes(ref_data, ref_pos, "#e74c3c")
    _draw_boxes(mat_data, mat_pos, "#3498db")

    ax.set_xticks(ref_pos + 0.5)
    ax.set_xticklabels(scene_ids, rotation=40, ha="right", fontsize=9)
    ax.set_xlim(0, n * 3)
    ax.set_ylabel("Sampson distance (px)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.35, linestyle="--")
    ax.legend(
        handles=[
            mpatches.Patch(color="#e74c3c", alpha=0.75, label="refractive"),
            mpatches.Patch(color="#3498db", alpha=0.75, label="matte"),
        ],
        loc="upper right", fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"  Saved {path.name}")


def _best_view_per_scene(scene_data: dict[str, list[dict]]) -> dict[str, dict]:
    """Return {scene_id: {view_idx, obj_ratio}} for the highest object_ratio view."""
    bests: dict[str, dict] = {}
    for sid, rows in scene_data.items():
        best = max(
            (r for r in rows if r.get("object_sampson_ratio") is not None),
            key=lambda r: r["object_sampson_ratio"],
            default=None,
        )
        if best is not None:
            bests[sid] = {
                "view_idx":  int(best["view_idx"]),
                "obj_ratio": best["object_sampson_ratio"],
                "bg_ratio":  best.get("background_sampson_ratio"),
            }
    return bests


def _write_report(
    path: Path,
    scene_data: dict[str, list[dict]],
    verdicts: dict[str, str],
    obj_thresh: float,
    bg_thresh: float,
    best_views: dict[str, dict],
) -> None:
    def _f(v: Optional[float], unit: str = "") -> str:
        return f"{v:.2f}{unit}" if v is not None else "N/A"

    n_total = len(scene_data)
    n_pass  = sum(1 for v in verdicts.values() if v == "PASS")

    # Ratio stats across all scenes
    all_obj = [agg["obj_ratio"]
               for agg in (_scene_agg(rows) for rows in scene_data.values())
               if agg["obj_ratio"] is not None]
    all_bg  = [agg["bg_ratio"]
               for agg in (_scene_agg(rows) for rows in scene_data.values())
               if agg["bg_ratio"] is not None]
    obj_mean_str = _f(float(np.mean(all_obj)) if all_obj else None, "×")
    obj_range_str = (f"{min(all_obj):.2f}× – {max(all_obj):.2f}×" if all_obj else "N/A")
    bg_mean_str   = _f(float(np.mean(all_bg))  if all_bg  else None, "×")

    table_rows: list[str] = []
    for sid, rows in scene_data.items():
        agg  = _scene_agg(rows)
        vt   = verdicts.get(sid, "?")
        icon = "✓" if vt == "PASS" else ("✗" if vt == "FAIL" else "?")
        table_rows.append(
            f"| {sid} | {agg['n_views']} "
            f"| {_f(agg['obj_ref'])} | {_f(agg['obj_mat'])} | {_f(agg['obj_ratio'], '×')} "
            f"| {_f(agg['bg_ref'])}  | {_f(agg['bg_mat'])}  | {_f(agg['bg_ratio'], '×')} "
            f"| {icon} {vt} |"
        )

    header = (
        "| Scene | Views "
        "| Obj Ref px | Obj Mat px | Obj Ratio "
        "| BG Ref px | BG Mat px | BG Ratio "
        "| Verdict |"
    )
    sep = (
        "|-------|-------"
        "|-----------|-----------|----------"
        "|----------|----------|----------"
        "|---------|"
    )

    # Figure-1 recommendations
    fig1_recs: list[str] = []
    for sid, bv in best_views.items():
        vi  = bv["view_idx"]
        obj = _f(bv["obj_ratio"], "×")
        bg  = _f(bv["bg_ratio"],  "×")
        fig1_recs.append(
            f"- **{sid}** — view `{vi}` "
            f"(object ratio {obj}, background ratio {bg})"
        )
    if not fig1_recs:
        fig1_recs = ["*(no scenes with valid ratios)*"]

    lines = [
        "# Dual-Path Experiment Report",
        "",
        f"Generated: {date.today()}",
        "",
        "## Per-Scene Summary",
        "",
        header, sep,
        *table_rows,
        "",
        "## Aggregate",
        "",
        f"- Total scenes: {n_total}",
        f"- PASS: {n_pass} / {n_total}"
        + (f" ({100 * n_pass // n_total}%)" if n_total else ""),
        f"- Object ratio — mean: {obj_mean_str},  range: {obj_range_str}",
        f"- Background ratio — mean: {bg_mean_str}",
        "",
        "## Thresholds",
        "",
        f"- Object Sampson ratio > **{obj_thresh}×** (refractive / matte)",
        f"- Background Sampson ratio < **{bg_thresh}×** (sanity bound)",
        "- Views with fewer than 20 LoFTR matches are excluded from ratio computation.",
        "",
        "## Figure recommendations for presentation",
        "",
        "### Figure 1 — best (scene, view) candidate per scene",
        "",
        *fig1_recs,
        "",
        "Generate with:",
        "```",
        "uv run python scripts/make_figure_1.py \\",
        "    --auto-select \\",
        "    --results-dir outputs/dual_path \\",
        "    --output-dir outputs/dual_path \\",
        "    --cache-dir outputs/_cache",
        "```",
        "",
        "### Figure 2 — ratio scatter",
        "",
        "`figure_2_ratio_scatter.png` — object vs background ratio, "
        "all (scene, view) pairs.",
        "",
        "### Figure 3 — per-view pattern",
        "",
        "`figure_3_per_view_pattern.png` — per-view object ratio overlaid "
        "for all scenes, with elevation shading.",
    ]
    path.write_text("\n".join(lines) + "\n")
    print(f"  Saved {path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate dual-path results across multiple scenes.")
    p.add_argument("--results-dir", required=True,
                   help="Directory containing per-scene output sub-folders.")
    p.add_argument("--output-dir", default=None,
                   help="Where to write aggregate outputs (default: --results-dir).")
    p.add_argument("--object-ratio-threshold", type=float, default=None)
    p.add_argument("--bg-ratio-threshold",     type=float, default=None)
    p.add_argument("--extra-figures", action="store_true",
                   help="Also write boxplot_object_sampson.png and "
                        "boxplot_background_sampson.png (not written by default).")
    return p.parse_args()


def main() -> None:
    args        = parse_args()
    results_dir = Path(args.results_dir)
    out_dir     = Path(args.output_dir) if args.output_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nScanning {results_dir} for summary.csv files …")
    scene_data = _load_scene_data(results_dir)
    if not scene_data:
        print("No summary.csv files found. Run run_dual_path_experiment.py first.")
        return

    print(f"Found {len(scene_data)} scene(s): {list(scene_data.keys())}")

    stored_obj, stored_bg = _load_thresholds(results_dir)
    obj_thr = args.object_ratio_threshold if args.object_ratio_threshold is not None else stored_obj
    bg_thr  = args.bg_ratio_threshold     if args.bg_ratio_threshold     is not None else stored_bg

    verdicts   = _load_verdicts(results_dir, list(scene_data.keys()))
    best_views = _best_view_per_scene(scene_data)

    print("\nWriting outputs:")
    _write_all_csv(out_dir / "all_scenes_summary.csv", scene_data)

    # Default figures
    fig2 = make_ratio_scatter(scene_data, obj_ref=obj_thr, bg_ref=bg_thr)
    p2   = out_dir / "figure_2_ratio_scatter.png"
    fig2.savefig(str(p2), dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved figure_2_ratio_scatter.png")

    fig3 = make_per_view_pattern(scene_data)
    p3   = out_dir / "figure_3_per_view_pattern.png"
    fig3.savefig(str(p3), dpi=200, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved figure_3_per_view_pattern.png")

    _write_report(out_dir / "report.md", scene_data, verdicts,
                  obj_thr, bg_thr, best_views)

    if args.extra_figures:
        print("  Writing extra figures (boxplots) …")
        _boxplot_sampson(
            out_dir / "boxplot_object_sampson.png", scene_data,
            col_ref="object_sampson_refractive_median",
            col_mat="object_sampson_matte_median",
            title="Object-Region Sampson Distance: Refractive vs Matte",
        )
        _boxplot_sampson(
            out_dir / "boxplot_background_sampson.png", scene_data,
            col_ref="background_sampson_refractive_median",
            col_mat="background_sampson_matte_median",
            title="Background-Region Sampson Distance: Refractive vs Matte",
        )

    n_pass = sum(1 for v in verdicts.values() if v == "PASS")
    print(f"\nDone.  {n_pass}/{len(scene_data)} scenes PASS.  Outputs → {out_dir}/")


if __name__ == "__main__":
    main()
