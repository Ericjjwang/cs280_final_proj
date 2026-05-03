"""
Run the full transparent-object analysis pipeline on a single condition image.

Usage:
    uv run python scripts/run_single.py \\
        --cond outputs/convention_test/cond.png \\
        --output_dir outputs/single_run/blue_chair \\
        --radius 1.5 \\
        --cache_dir outputs/_cache/views
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation import Zero123PlusPipeline
from src.matching import LoFTRMatcher
from src.pipeline import TransparentObjectAnalyzer, AnalyzeResult
import src.viz as viz


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cond",       required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--radius",     type=float, default=1.5)
    p.add_argument("--steps",      type=int,   default=36)
    p.add_argument("--device",     default="cuda")
    p.add_argument("--seg",        default="auto", choices=["auto", "threshold"])
    p.add_argument("--cache_dir",   default=None)
    p.add_argument("--force_regen", action="store_true")
    p.add_argument("--mask_cond",   default=None,
                   help="Blender alpha mask for the condition image (PNG).")
    p.add_argument("--mask_views",  nargs=6, default=None, metavar="MASK",
                   help="Blender alpha masks for the 6 novel views, in view order.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir) if args.cache_dir else None

    matcher   = LoFTRMatcher(weights="outdoor", device=args.device)
    generator = Zero123PlusPipeline(device=args.device, num_inference_steps=args.steps)

    with generator:
        analyzer = TransparentObjectAnalyzer(
            generator=generator,
            matcher=matcher,
            radius=args.radius,
            seg_method=args.seg,
        )
        mask_cond_path  = Path(args.mask_cond)  if args.mask_cond  else None
        mask_view_paths = [Path(p) for p in args.mask_views] if args.mask_views else None
        result = analyzer.analyze(
            args.cond,
            radius=args.radius,
            cache_dir=cache,
            force_regenerate=args.force_regen,
            mask_cond_path=mask_cond_path,
            mask_view_paths=mask_view_paths,
        )

    # ── Save images ───────────────────────────────────────────────────────────
    result.cond_image.save(out / "cond.png")
    Image.fromarray(result.object_mask_cond * 255).save(out / "mask_cond.png")
    for i, (v, m) in enumerate(zip(result.novel_views, result.object_masks_views)):
        v.save(out / f"view_{i}.png")
        Image.fromarray(m * 255).save(out / f"mask_view_{i}.png")

    # ── Per-view heatmaps ─────────────────────────────────────────────────────
    for pair in result.pairs:
        i = pair.view_idx
        if pair.skipped:
            print(f"  view {i}: skipped (< 20 matches)")
            continue
        fig = viz.plot_sampson_heatmap(
            result.novel_views[i],
            pair.kpts_b,
            pair.sampson_all,
            mask=result.object_masks_views[i],
            title=f"View {i}  az={pair.azimuth}°  el={pair.elevation}°",
        )
        viz.save_figure(fig, str(out / f"heatmap_view{i}.png"))

    # ── Region bar chart ──────────────────────────────────────────────────────
    fig = viz.plot_region_bar([p.stats() for p in result.pairs])
    viz.save_figure(fig, str(out / "region_analysis.png"))

    # ── 4-panel for view 1 (closest-to-front, usually best matches) ───────────
    best = next((p for p in result.pairs if not p.skipped), None)
    if best:
        i   = best.view_idx
        fig = viz.plot_4panel(
            cond_img        = result.cond_image,
            view_img        = result.novel_views[i],
            kpts_b          = best.kpts_b,
            sampson_dist    = best.sampson_all,
            pair_stats_list = [p.stats() for p in result.pairs],
            mask_cond       = result.object_mask_cond,
            mask_view       = result.object_masks_views[i],
            view_label      = f"V{i} az={best.azimuth}°",
        )
        viz.save_figure(fig, str(out / "4panel.png"))

    # ── Stats JSON ────────────────────────────────────────────────────────────
    result.to_json(out / "stats.json", include_arrays=False)
    result.to_json(out / "stats_full.json", include_arrays=True)

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"{'V':>2} {'az':>5} {'el':>4} | {'N':>4} | {'med_all':>8} | {'obj':>5} | {'bg':>5}")
    print("-" * 55)
    for p in result.pairs:
        s = p.stats()
        if p.skipped:
            print(f"{p.view_idx:>2} {p.azimuth:>5.0f}° {p.elevation:>3.0f}° | SKIP")
        else:
            med  = f"{s['median_sampson_all']:.2f}" if s['median_sampson_all'] is not None else "N/A"
            mobj = f"{s['median_sampson_object']:.2f}" if s['median_sampson_object'] is not None else "N/A"
            mbg  = f"{s['median_sampson_background']:.2f}" if s['median_sampson_background'] is not None else "N/A"
            print(f"{p.view_idx:>2} {p.azimuth:>5.0f}° {p.elevation:>3.0f}° | "
                  f"{p.n_matches:>4} | {med:>8} | {mobj:>5} | {mbg:>5}")
    print(f"{'='*55}")
    print(f"\nOutputs: {out}")


if __name__ == "__main__":
    main()
