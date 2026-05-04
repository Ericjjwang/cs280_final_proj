"""
Figure 2: Object vs Background Sampson ratio scatter.

Each point = one (scene, view) pair; x = object Sampson ratio (log scale),
y = background Sampson ratio (linear).  Dashed reference lines at the
default thresholds (obj > 2.0×, bg < 1.5×).  The lower-right quadrant is
shaded as the "PASS" region.

Importable function
-------------------
    from scripts.make_ratio_scatter import make_ratio_scatter
    fig = make_ratio_scatter(scene_data)       # scene_data: {sid: [row, …]}

Standalone
----------
    uv run python scripts/make_ratio_scatter.py \\
        --results-dir outputs/dual_path \\
        [--output outputs/dual_path/figure_2_ratio_scatter.png] \\
        [--obj-ref 2.0] [--bg-ref 1.5]
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AZIMUTHS   = [30, 90, 150, 210, 270, 330]
ELEVATIONS = [20, -10, 20, -10, 20, -10]

_OBJ_REF = 2.0
_BG_REF  = 1.5


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_scene_data(results_dir: Path) -> dict[str, list[dict]]:
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


# ── Figure ────────────────────────────────────────────────────────────────────

def make_ratio_scatter(
    scene_data: dict[str, list[dict]],
    obj_ref: float = _OBJ_REF,
    bg_ref:  float = _BG_REF,
) -> plt.Figure:
    """
    Scatter plot of object_ratio (x, log) vs background_ratio (y, linear).

    Args:
        scene_data: {scene_id: [row_dict, …]} typically loaded from summary.csv files.
        obj_ref:    Vertical reference line position (object ratio threshold).
        bg_ref:     Horizontal reference line position (background ratio threshold).

    Returns:
        matplotlib Figure (16:9 aspect, dpi-agnostic — caller decides dpi at save time).
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    scene_ids = list(scene_data.keys())
    cmap      = plt.colormaps.get_cmap("tab10")
    colors    = {sid: cmap(i % 10) for i, sid in enumerate(scene_ids)}

    all_obj: list[float] = []
    all_bg:  list[float] = []

    for sid, rows in scene_data.items():
        color   = colors[sid]
        labeled = False
        for row in rows:
            x  = row.get("object_sampson_ratio")
            y  = row.get("background_sampson_ratio")
            vi = row.get("view_idx")
            if x is None or y is None:
                continue
            all_obj.append(x)
            all_bg.append(y)
            ax.scatter(x, y, color=color, s=90, alpha=0.85, zorder=3,
                       label=(sid if not labeled else None))
            labeled = True
            if vi is not None:
                ax.annotate(f"V{int(vi)}", (x, y),
                            fontsize=7, alpha=0.65,
                            xytext=(4, 4), textcoords="offset points")

    # Axis limits
    x_lo  = 0.5
    x_hi  = max(20.0, max(all_obj) * 1.15) if all_obj else 20.0
    y_hi  = max(5.0,  max(all_bg)  * 1.15) if all_bg  else 5.0

    # PASS-region shading (right of obj_ref, below bg_ref)
    bg_frac = bg_ref / y_hi
    ax.axvspan(obj_ref, x_hi, ymin=0, ymax=bg_frac,
               color="#2ecc71", alpha=0.09, zorder=0)
    mid_x = np.exp((np.log(obj_ref) + np.log(x_hi)) / 2)  # geometric mid on log axis
    mid_y = bg_ref / 2
    ax.text(mid_x, mid_y,
            "Object error\n>> Background error",
            ha="center", va="center", fontsize=9,
            color="#27ae60", alpha=0.75, style="italic")

    # Reference lines
    ax.axvline(obj_ref, color="gray", linestyle="--", lw=1.3, alpha=0.7,
               label=f"obj ref = {obj_ref}×")
    ax.axhline(bg_ref,  color="gray", linestyle=":",  lw=1.3, alpha=0.7,
               label=f"bg ref = {bg_ref}×")

    ax.set_xscale("log")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0.0,  y_hi)
    ax.set_xlabel("Object Sampson ratio  (refractive / matte)  [log scale]", fontsize=12)
    ax.set_ylabel("Background Sampson ratio  (refractive / matte)", fontsize=12)
    ax.set_title("Dual-Path Sampson Ratio: Object vs Background",
                 fontsize=14, fontweight="bold")
    ax.grid(True, which="both", alpha=0.22, linestyle="--")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels,
              loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=9, borderaxespad=0, frameon=True,
              title="Scene / threshold")
    fig.tight_layout(rect=[0, 0, 0.81, 1])
    return fig


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Figure 2: ratio scatter (object vs background Sampson ratio).")
    p.add_argument("--results-dir", required=True,
                   help="Directory containing per-scene output sub-folders with summary.csv.")
    p.add_argument("--output", default=None,
                   help="Output PNG path (default: <results-dir>/figure_2_ratio_scatter.png).")
    p.add_argument("--obj-ref", type=float, default=_OBJ_REF,
                   help=f"Object ratio reference line (default {_OBJ_REF}).")
    p.add_argument("--bg-ref",  type=float, default=_BG_REF,
                   help=f"Background ratio reference line (default {_BG_REF}).")
    return p.parse_args()


def main() -> None:
    args        = _parse_args()
    results_dir = Path(args.results_dir)
    out_path    = (Path(args.output) if args.output
                   else results_dir / "figure_2_ratio_scatter.png")

    scene_data = _load_scene_data(results_dir)
    if not scene_data:
        print("No summary.csv files found in", results_dir)
        return

    print(f"Building ratio scatter for {len(scene_data)} scene(s) …")
    fig = make_ratio_scatter(scene_data, obj_ref=args.obj_ref, bg_ref=args.bg_ref)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
