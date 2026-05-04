"""
Figure 1: Dual-path failure case — 2×2 panel.

Layout:
    [ Refractive cond image  |  Refractive view + Sampson heatmap ]
    [ Matte cond image       |  Matte view + Sampson heatmap      ]

Sampson heatmap coloring: green (low error) → red (high error) on keypoints
matched in the view frame.  Object boundary drawn as a white contour.

Usage — single scene/view
--------------------------
    uv run python scripts/make_figure_1.py \\
        --scene-dir data/scene_0001 \\
        --view-idx 2 \\
        --output outputs/dual_path/figure_1.png \\
        [--cache-dir outputs/_cache] \\
        [--device cuda]

Usage — batch (auto-select best view per scene)
------------------------------------------------
    uv run python scripts/make_figure_1.py \\
        --auto-select \\
        --results-dir outputs/dual_path \\
        --output-dir outputs/dual_path \\
        [--cache-dir outputs/_cache] \\
        [--device cuda]
"""

from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.camera import AZIMUTHS_DEG, ELEVATIONS_DEG, IMG_SIZE
from src.generation import Zero123PlusPipeline
from src.io_utils import load_blender_alpha_mask
from src.matching import LoFTRMatcher
from src.pipeline import (
    analyze_dual_path,
    DualPathResult,
    _make_cache_key,
    _resize_mask_to,
)

SAMPSON_VMAX = 20.0


# ── Image helpers ─────────────────────────────────────────────────────────────

def _open_rgb(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if img.size != (IMG_SIZE, IMG_SIZE):
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    return img


def _load_view_from_cache(
    cond_path: Path,
    model_id:  str,
    cache_dir: Path,
    radius:    float,
    label:     str,
    view_idx:  int,
) -> Optional[Image.Image]:
    """Return the cached Zero123++ view PNG, or None if not found."""
    key  = _make_cache_key(cond_path, radius, model_id, label)
    p    = cache_dir / key / f"view_{view_idx}.png"
    return Image.open(p).convert("RGB") if p.exists() else None


# ── Figure builder ────────────────────────────────────────────────────────────

def make_figure_1(
    result:         DualPathResult,
    rgb_cond_img:   Image.Image,
    matte_cond_img: Image.Image,
    rgb_view_img:   Image.Image,
    matte_view_img: Image.Image,
    mask_cond:      np.ndarray,
    mask_view:      np.ndarray,
    vmax:           float = SAMPSON_VMAX,
) -> plt.Figure:
    """
    Build the 2×2 dual-path failure-case figure.

    Args:
        result:          DualPathResult carrying Sampson data for both paths.
        rgb_cond_img:    Refractive (RGB) condition image.
        matte_cond_img:  Matte condition image.
        rgb_view_img:    Refractive novel-view image (Zero123++ output).
        matte_view_img:  Matte novel-view image.
        mask_cond:       Object mask for condition frame (uint8, 0/1, matching res).
        mask_view:       Object mask for view frame     (uint8, 0/1, matching res).
        vmax:            Sampson colormap upper bound in pixels.

    Returns:
        matplotlib Figure (12×7 inches).
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    vi  = result.view_idx
    az  = result.azimuth
    el  = result.elevation

    def _el_str(e: float) -> str:
        return f"+{e:.0f}°" if e > 0 else f"{e:.0f}°"

    view_label = f"V{vi}  az={az:.0f}°  el={_el_str(el)}"

    # ── Column 0: condition images ─────────────────────────────────────────────
    for row_idx, (img, title) in enumerate([
        (rgb_cond_img,   "Refractive — condition"),
        (matte_cond_img, "Matte — condition"),
    ]):
        ax = axes[row_idx, 0]
        ax.imshow(img)
        if mask_cond is not None:
            ax.contour(mask_cond, levels=[0.5], colors="white", linewidths=1.2)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

    # ── Column 1: novel-view images with Sampson heatmap ─────────────────────
    for row_idx, (view_img, pair, path_label) in enumerate([
        (rgb_view_img,   result.refractive, "Refractive"),
        (matte_view_img, result.matte,      "Matte"),
    ]):
        ax  = axes[row_idx, 1]
        ax.imshow(view_img)

        if mask_view is not None:
            ax.contour(mask_view, levels=[0.5], colors="white", linewidths=1.2)

        kpts = pair.kpts_b
        sdst = pair.sampson_all
        if len(kpts) > 0 and len(sdst) > 0:
            sc = ax.scatter(
                kpts[:, 0], kpts[:, 1],
                c=np.clip(sdst, 0, vmax),
                cmap="RdYlGn_r", s=10, alpha=0.82,
                vmin=0, vmax=vmax,
            )
            plt.colorbar(sc, ax=ax, label="Sampson dist (px)", fraction=0.046, pad=0.04)
            med = float(np.median(sdst))
            n   = len(kpts)
        else:
            med = float("nan")
            n   = 0

        title = (f"{path_label} — {view_label}\n"
                 f"median={med:.1f} px  N={n}")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # ── Figure-level annotation ────────────────────────────────────────────────
    obj_r = result.object_sampson_ratio
    bg_r  = result.background_sampson_ratio
    obj_str = f"{obj_r:.2f}×" if obj_r is not None else "N/A"
    bg_str  = f"{bg_r:.2f}×"  if bg_r  is not None else "N/A"
    fig.suptitle(
        f"Scene: {result.scene_id}  |  {view_label}\n"
        f"Object ratio: {obj_str}  |  Background ratio: {bg_str}",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    return fig


# ── Scene-level runner ────────────────────────────────────────────────────────

def _run_one(
    scene_dir: Path,
    view_idx:  int,
    out_path:  Path,
    generator: Zero123PlusPipeline,
    matcher:   LoFTRMatcher,
    cache_dir: Optional[Path],
    radius:    float = 1.5,
) -> None:
    """Analyse one (scene, view) pair and save figure_1."""
    result = analyze_dual_path(
        scene_dir        = scene_dir,
        view_idx         = view_idx,
        generator        = generator,
        matcher          = matcher,
        cache_dir        = cache_dir,
        radius           = radius,
        use_blender_pose = True,
    )

    # Load condition images (Blender quality, 512×512 → resized to match res)
    rgb_cond   = _open_rgb(scene_dir / "rgb"   / "cond.png")
    matte_cond = _open_rgb(scene_dir / "matte" / "cond.png")

    # Load novel-view images from cache (warmed by analyze_dual_path above)
    rgb_view = matte_view = None
    if cache_dir is not None:
        rgb_view   = _load_view_from_cache(
            scene_dir / "rgb"   / "cond.png", generator.model_id,
            cache_dir, radius, "rgb", view_idx)
        matte_view = _load_view_from_cache(
            scene_dir / "matte" / "cond.png", generator.model_id,
            cache_dir, radius, "matte", view_idx)

    # Fallback: use the PairResult to get a blank placeholder
    def _blank() -> Image.Image:
        return Image.new("RGB", (IMG_SIZE, IMG_SIZE), (80, 80, 80))

    if rgb_view   is None: rgb_view   = _blank()
    if matte_view is None: matte_view = _blank()

    # Masks at matching resolution
    mask_cond = _resize_mask_to(
        load_blender_alpha_mask(scene_dir / "mask" / "cond.png"), IMG_SIZE)
    mask_view = _resize_mask_to(
        load_blender_alpha_mask(scene_dir / "mask" / f"view_{view_idx}.png"), IMG_SIZE)

    fig = make_figure_1(result, rgb_cond, matte_cond, rgb_view, matte_view,
                        mask_cond, mask_view)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Auto-select helpers ───────────────────────────────────────────────────────

def _best_view(summary_csv: Path) -> Optional[int]:
    """Return the view_idx with the highest object_sampson_ratio, or None."""
    best_vi  = None
    best_rat = -1.0
    with open(summary_csv) as f:
        for row in csv.DictReader(f):
            v = row.get("view_idx", "")
            r = row.get("object_sampson_ratio", "")
            if v == "" or r in ("", "None", "N/A"):
                continue
            try:
                rat = float(r)
                vi  = int(float(v))
            except ValueError:
                continue
            if rat > best_rat:
                best_rat, best_vi = rat, vi
    return best_vi


def _scene_dir_from_results(results_json: Path) -> Optional[Path]:
    """Read scene_dir from results.json."""
    if not results_json.exists():
        return None
    with open(results_json) as f:
        data = json.load(f)
    sd = data.get("scene_dir")
    return Path(sd) if sd else None


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Figure 1: dual-path failure-case 2×2 panel.")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--scene-dir", default=None,
                      help="Blender scene directory (single-scene mode).")
    mode.add_argument("--auto-select", action="store_true",
                      help="Scan results-dir and pick best view per scene.")
    p.add_argument("--view-idx", type=int, default=None,
                   help="View index for single-scene mode.")
    p.add_argument("--output", default=None,
                   help="Output PNG path (single-scene mode).")
    p.add_argument("--results-dir", default=None,
                   help="Experiment output directory (auto-select mode).")
    p.add_argument("--output-dir", default=None,
                   help="Where to write figures in auto-select mode.")
    p.add_argument("--cache-dir", default=None,
                   help="Shared view cache root.")
    p.add_argument("--radius", type=float, default=1.5)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args      = _parse_args()
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    matcher   = LoFTRMatcher(weights="outdoor", device=args.device)
    generator = Zero123PlusPipeline(device=args.device)

    with generator:
        if args.auto_select:
            if not args.results_dir:
                print("--results-dir is required with --auto-select")
                sys.exit(1)
            results_dir = Path(args.results_dir)
            out_dir     = Path(args.output_dir) if args.output_dir else results_dir
            out_dir.mkdir(parents=True, exist_ok=True)

            for csv_path in sorted(results_dir.glob("*/summary.csv")):
                scene_id  = csv_path.parent.name
                view_idx  = _best_view(csv_path)
                scene_dir = _scene_dir_from_results(csv_path.parent / "results.json")
                if view_idx is None or scene_dir is None or not scene_dir.exists():
                    print(f"[skip] {scene_id}: no valid view or scene_dir not found")
                    continue
                print(f"\n[{scene_id}] best view_idx={view_idx}, scene_dir={scene_dir}")
                out_path = out_dir / f"{scene_id}_figure_1.png"
                _run_one(scene_dir, view_idx, out_path,
                         generator, matcher, cache_dir, args.radius)

        else:
            if args.scene_dir is None:
                print("--scene-dir is required in single-scene mode")
                sys.exit(1)
            if args.view_idx is None:
                print("--view-idx is required in single-scene mode")
                sys.exit(1)
            scene_dir = Path(args.scene_dir)
            out_path  = (Path(args.output) if args.output
                         else scene_dir.parent / "figure_1.png")
            _run_one(scene_dir, args.view_idx, out_path,
                     generator, matcher, cache_dir, args.radius)


if __name__ == "__main__":
    main()
