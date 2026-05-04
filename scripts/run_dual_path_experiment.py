"""
Run the dual-path (refractive vs matte) Sampson experiment for one Blender scene.

Usage
-----
    uv run python scripts/run_dual_path_experiment.py \\
        --scene-dir data/scene_0001 \\
        --output-dir outputs/dual_path/scene_0001 \\
        [--views 0,1,2,3,4,5] \\
        [--cache-dir outputs/_cache] \\
        [--radius 1.5] [--steps 36] [--device cuda] \\
        [--no-blender-pose] \\
        [--object-ratio-threshold 2.0] \\
        [--bg-ratio-threshold 1.5] \\
        [--skip-verify]
"""

from __future__ import annotations
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation import Zero123PlusPipeline
from src.matching import LoFTRMatcher
from src.pipeline import analyze_dual_path, DualPathResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _nanmedian(values: list) -> Optional[float]:
    valid = [x for x in values if x is not None and np.isfinite(float(x))]
    return float(np.median(valid)) if valid else None


def _safe_ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0.0:
        return None
    return a / b


def _fmt(v: Optional[float], unit: str = " px", suffix: str = "") -> str:
    if v is None:
        return "N/A"
    return f"{v:.2f}{unit}{suffix}"


def _verdict(
    obj_ratio:  Optional[float],
    bg_ratio:   Optional[float],
    obj_thresh: float,
    bg_thresh:  float,
) -> tuple[str, str]:
    reasons: list[str] = []
    if obj_ratio is None:
        reasons.append("object_ratio=N/A (no object matches)")
    elif obj_ratio <= obj_thresh:
        reasons.append(f"object_ratio={obj_ratio:.2f}x ≤ {obj_thresh}x")
    if bg_ratio is None:
        reasons.append("background_ratio=N/A (no background matches)")
    elif bg_ratio >= bg_thresh:
        reasons.append(f"background_ratio={bg_ratio:.2f}x ≥ {bg_thresh}x")
    return ("PASS", "") if not reasons else ("FAIL", "; ".join(reasons))


def _aggregate(results: list[DualPathResult]) -> dict:
    """Pool per-view Sampson medians → scene-level statistics."""
    obj_ref = _nanmedian([r.object_sampson_refractive_median     for r in results])
    obj_mat = _nanmedian([r.object_sampson_matte_median          for r in results])
    bg_ref  = _nanmedian([r.background_sampson_refractive_median for r in results])
    bg_mat  = _nanmedian([r.background_sampson_matte_median      for r in results])
    return {
        "object_sampson_refractive_median":      obj_ref,
        "object_sampson_matte_median":           obj_mat,
        "object_sampson_ratio":                  _safe_ratio(obj_ref, obj_mat),
        "background_sampson_refractive_median":  bg_ref,
        "background_sampson_matte_median":       bg_mat,
        "background_sampson_ratio":              _safe_ratio(bg_ref, bg_mat),
    }


def _run_verify(scene_dir: Path, script_dir: Path) -> bool:
    r = subprocess.run(
        [sys.executable, str(script_dir / "verify_scene_data.py"), str(scene_dir)],
        cwd=str(script_dir.parent),
    )
    return r.returncode == 0


# ── Writers ───────────────────────────────────────────────────────────────────

def _write_results_json(
    path: Path,
    scene_dir: Path,
    views: list[int],
    results: list[DualPathResult],
    agg: dict,
    verdict: str,
    reason: str,
    obj_thresh: float,
    bg_thresh: float,
    include_arrays: bool = False,
) -> None:
    data = {
        "scene_id":   results[0].scene_id if results else scene_dir.name,
        "scene_dir":  str(scene_dir.resolve()),
        "views":      views,
        "thresholds": {
            "object_ratio_threshold":     obj_thresh,
            "background_ratio_threshold": bg_thresh,
        },
        "scene_aggregate": {**agg, "verdict": verdict, "reason": reason},
        "per_view": [r.to_json(include_arrays=include_arrays) for r in results],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _write_summary_csv(path: Path, results: list[DualPathResult]) -> None:
    rows = [row for r in results for row in r.to_csv_rows()]
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_scene_txt(
    path: Path,
    scene_id: str,
    agg: dict,
    verdict: str,
    reason: str,
    obj_thresh: float,
    bg_thresh: float,
) -> None:
    obj_ref = agg["object_sampson_refractive_median"]
    obj_mat = agg["object_sampson_matte_median"]
    obj_rat = agg["object_sampson_ratio"]
    bg_ref  = agg["background_sampson_refractive_median"]
    bg_mat  = agg["background_sampson_matte_median"]
    bg_rat  = agg["background_sampson_ratio"]

    lines = [
        f"Scene {scene_id}:",
        f"  Object Sampson median:     "
        f"refractive={_fmt(obj_ref)}, matte={_fmt(obj_mat)}, ratio={_fmt(obj_rat, '', 'x')}",
        f"  Background Sampson median: "
        f"refractive={_fmt(bg_ref)}, matte={_fmt(bg_mat)}, ratio={_fmt(bg_rat, '', 'x')}",
        f"  Verdict: [{verdict}] core claim "
        f"(object_ratio > {obj_thresh} and background_ratio < {bg_thresh})",
    ]
    if reason:
        lines.append(f"  Reason:  {reason}")
    path.write_text("\n".join(lines) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dual-path Sampson experiment for one Blender scene.")
    p.add_argument("--scene-dir",    required=True,
                   help="Root of a verified Blender scene directory.")
    p.add_argument("--output-dir",   required=True,
                   help="Where to write results.json, summary.csv, scene_summary.txt.")
    p.add_argument("--views",        default="0,1,2,3,4,5",
                   help="Comma-separated view indices (default: 0,1,2,3,4,5).")
    p.add_argument("--cache-dir",    default=None,
                   help="Shared view cache root.")
    p.add_argument("--radius",       type=float, default=1.5)
    p.add_argument("--steps",        type=int,   default=36)
    p.add_argument("--device",       default="cuda")
    p.add_argument("--no-blender-pose", action="store_true",
                   help="Use Zero123++ nominal poses instead of poses.json.")
    p.add_argument("--object-ratio-threshold", type=float, default=2.0,
                   help="Minimum refractive/matte ratio on object region to PASS (default 2.0).")
    p.add_argument("--bg-ratio-threshold",     type=float, default=1.5,
                   help="Maximum refractive/matte ratio on background to PASS (default 1.5).")
    p.add_argument("--skip-verify",  action="store_true",
                   help="Skip verify_scene_data.py (for debugging).")
    return p.parse_args()


def main() -> None:
    args      = parse_args()
    scene     = Path(args.scene_dir)
    out_dir   = Path(args.output_dir)
    script_d  = Path(__file__).resolve().parent
    cache     = Path(args.cache_dir) if args.cache_dir else None
    views     = [int(v.strip()) for v in args.views.split(",")]
    obj_thr   = args.object_ratio_threshold
    bg_thr    = args.bg_ratio_threshold

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: verify scene ──────────────────────────────────────────────────
    if not args.skip_verify:
        print(f"\n{'='*60}")
        print(f"[verify] {scene}")
        print(f"{'='*60}")
        if not _run_verify(scene, script_d):
            print("\n[ABORT] Verification failed — fix the scene data first.")
            sys.exit(1)
        print("[verify] OK\n")

    # ── Step 2: per-view dual-path analysis ───────────────────────────────────
    matcher   = LoFTRMatcher(weights="outdoor", device=args.device)
    generator = Zero123PlusPipeline(device=args.device, num_inference_steps=args.steps)

    results: list[DualPathResult] = []
    with generator:
        for view_idx in views:
            print(f"\n{'='*60}")
            print(f"[view {view_idx}]")
            print(f"{'='*60}")
            r = analyze_dual_path(
                scene_dir        = scene,
                view_idx         = view_idx,
                generator        = generator,
                matcher          = matcher,
                cache_dir        = cache,
                radius           = args.radius,
                use_blender_pose = not args.no_blender_pose,
            )
            results.append(r)
            print(f"  object  : ref={_fmt(r.object_sampson_refractive_median)}"
                  f"  mat={_fmt(r.object_sampson_matte_median)}"
                  f"  ratio={_fmt(r.object_sampson_ratio, '', 'x')}")
            print(f"  bg      : ref={_fmt(r.background_sampson_refractive_median)}"
                  f"  mat={_fmt(r.background_sampson_matte_median)}"
                  f"  ratio={_fmt(r.background_sampson_ratio, '', 'x')}")

    # ── Step 3: aggregate ─────────────────────────────────────────────────────
    agg             = _aggregate(results)
    verdict, reason = _verdict(
        agg["object_sampson_ratio"], agg["background_sampson_ratio"],
        obj_thr, bg_thr,
    )
    scene_id = results[0].scene_id if results else scene.name

    _write_results_json(
        out_dir / "results.json", scene, views, results,
        agg, verdict, reason, obj_thr, bg_thr,
    )
    # Full JSON includes raw Sampson arrays (for figure generation)
    _write_results_json(
        out_dir / "results_full.json", scene, views, results,
        agg, verdict, reason, obj_thr, bg_thr,
        include_arrays=True,
    )
    _write_summary_csv(out_dir / "summary.csv", results)
    _write_scene_txt(
        out_dir / "scene_summary.txt", scene_id,
        agg, verdict, reason, obj_thr, bg_thr,
    )

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Scene {scene_id}:")
    print(f"  Object    : ref={_fmt(agg['object_sampson_refractive_median'])}"
          f"  mat={_fmt(agg['object_sampson_matte_median'])}"
          f"  ratio={_fmt(agg['object_sampson_ratio'], '', 'x')}")
    print(f"  Background: ref={_fmt(agg['background_sampson_refractive_median'])}"
          f"  mat={_fmt(agg['background_sampson_matte_median'])}"
          f"  ratio={_fmt(agg['background_sampson_ratio'], '', 'x')}")
    print(f"  Verdict: [{verdict}]" + (f"  — {reason}" if reason else ""))
    print(f"{'='*60}")
    print(f"\nOutputs → {out_dir}/")
    for f in ["results.json", "summary.csv", "scene_summary.txt"]:
        print(f"  {f}")


if __name__ == "__main__":
    main()
