"""
Figure 3: per-view object Sampson ratio pattern.

All scenes are overlaid on one figure (one line per scene + bold black mean).
Even views (0, 2, 4 — el=+20°) and odd views (1, 3, 5 — el=-10°) receive
different background shading to highlight the elevation alternation.

Importable function
-------------------
    from scripts.make_per_view_pattern import make_per_view_pattern
    fig = make_per_view_pattern(scene_data)

Standalone
----------
    uv run python scripts/make_per_view_pattern.py \\
        --results-dir outputs/dual_path \\
        [--output outputs/dual_path/figure_3_per_view_pattern.png]
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

AZIMUTHS   = [30, 90, 150, 210, 270, 330]
ELEVATIONS = [20, -10, 20, -10, 20, -10]
N_VIEWS    = 6

_EL_POS_COLOR = "#dce9f5"   # even views: el=+20°
_EL_NEG_COLOR = "#fdf3dc"   # odd views:  el=-10°


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

def make_per_view_pattern(scene_data: dict[str, list[dict]]) -> plt.Figure:
    """
    Per-view object Sampson ratio pattern with all scenes overlaid.

    Args:
        scene_data: {scene_id: [row_dict, …]} from summary.csv files.

    Returns:
        matplotlib Figure (12×6 inches, dpi set at save time).
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Alternating elevation shading
    for vi in range(N_VIEWS):
        color = _EL_POS_COLOR if vi % 2 == 0 else _EL_NEG_COLOR
        ax.axvspan(vi - 0.5, vi + 0.5, color=color, alpha=0.65, zorder=0)

    scene_ids = list(scene_data.keys())
    cmap      = plt.colormaps.get_cmap("tab10")
    colors    = {sid: cmap(i % 10) for i, sid in enumerate(scene_ids)}

    # Per-scene lines
    all_scene_ratios: list[list[float | None]] = []
    for sid, rows in scene_data.items():
        ratio_by_view: dict[int, float] = {}
        for row in rows:
            vi  = row.get("view_idx")
            rat = row.get("object_sampson_ratio")
            if vi is not None and rat is not None:
                ratio_by_view[int(vi)] = rat

        ys = [ratio_by_view.get(vi) for vi in range(N_VIEWS)]
        all_scene_ratios.append(ys)

        xs_ok = [vi for vi in range(N_VIEWS) if ys[vi] is not None]
        ys_ok = [ys[vi] for vi in xs_ok]
        if xs_ok:
            ax.plot(xs_ok, ys_ok, marker="o", color=colors[sid],
                    lw=1.6, ms=6, alpha=0.78, label=sid, zorder=2)

    # Mean line across all scenes
    mean_ratios: list[float | None] = []
    for vi in range(N_VIEWS):
        vals = [sr[vi] for sr in all_scene_ratios if sr[vi] is not None]
        mean_ratios.append(float(np.mean(vals)) if vals else None)

    xs_m = [vi for vi in range(N_VIEWS) if mean_ratios[vi] is not None]
    ys_m = [mean_ratios[vi] for vi in xs_m]
    if xs_m:
        ax.plot(xs_m, ys_m, marker="D", color="black", lw=2.5, ms=7,
                zorder=4, label="Mean")

    # Baseline
    ax.axhline(1.0, color="gray", linestyle="--", lw=1.0, alpha=0.55, label="ratio = 1.0")

    # X-axis: view labels with az/el annotation
    def _el_str(el: int) -> str:
        return f"+{el}°" if el > 0 else f"{el}°"

    xlabels = [
        f"V{vi}\naz={AZIMUTHS[vi]}°\nel={_el_str(ELEVATIONS[vi])}"
        for vi in range(N_VIEWS)
    ]
    ax.set_xticks(range(N_VIEWS))
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_xlim(-0.5, N_VIEWS - 0.5)

    ax.set_ylabel("Object Sampson ratio  (refractive / matte)", fontsize=11)
    ax.set_title("Per-View Object Sampson Ratio  (all scenes overlaid)",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.30, linestyle=":")

    # Legend — scene lines + elevation shading patches
    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Patch(color=_EL_POS_COLOR, alpha=0.85, label="el = +20°"),
        Patch(color=_EL_NEG_COLOR, alpha=0.85, label="el = −10°"),
    ]
    labels  += ["el = +20°", "el = −10°"]
    ax.legend(handles, labels, loc="upper right", fontsize=9,
              ncol=2, framealpha=0.9)

    fig.tight_layout()
    return fig


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Figure 3: per-view object Sampson ratio pattern.")
    p.add_argument("--results-dir", required=True,
                   help="Directory containing per-scene output sub-folders with summary.csv.")
    p.add_argument("--output", default=None,
                   help="Output PNG (default: <results-dir>/figure_3_per_view_pattern.png).")
    return p.parse_args()


def main() -> None:
    args        = _parse_args()
    results_dir = Path(args.results_dir)
    out_path    = (Path(args.output) if args.output
                   else results_dir / "figure_3_per_view_pattern.png")

    scene_data = _load_scene_data(results_dir)
    if not scene_data:
        print("No summary.csv files found in", results_dir)
        return

    print(f"Building per-view pattern for {len(scene_data)} scene(s) …")
    fig = make_per_view_pattern(scene_data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
