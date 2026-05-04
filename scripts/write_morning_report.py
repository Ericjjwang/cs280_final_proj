"""
Generate outputs/plan_b_aggregate/morning_report.md from experiment results.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load_results(exp_dir: Path) -> dict | None:
    for name in ["results.json", "scene_results.json"]:
        p = exp_dir / name
        if p.exists():
            with open(p) as f:
                return json.load(f)
    # Walk for any json
    for p in exp_dir.glob("*.json"):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            continue
    return None


def _load_verify_report(scene_dir: Path) -> dict | None:
    p = scene_dir / "verification_report.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _verify_status(report: dict | None) -> str:
    if report is None:
        return "NOT RUN"
    results = report.get("results", [])
    n_fail  = sum(1 for r in results if not r.get("passed", True))
    return "PASS" if n_fail == 0 else f"FAIL ({n_fail} checks failed)"


def _extract_view_ratios(results_data: dict | None) -> list[float]:
    """Pull per-view object_sampson_ratio from results.json."""
    if results_data is None or not isinstance(results_data, dict):
        return []
    ratios = []
    for item in results_data.get("per_view", []):
        if not isinstance(item, dict):
            continue
        summary = item.get("summary", {})
        v = summary.get("object_sampson_ratio")
        if v is not None:
            ratios.append(float(v))
    # Fallback: scene-level aggregate
    if not ratios:
        agg = results_data.get("scene_aggregate", {})
        v = agg.get("object_sampson_ratio")
        if v is not None:
            ratios.append(float(v))
    return ratios


def _extract_bg_sampson(results_data: dict | None) -> list[float]:
    if results_data is None or not isinstance(results_data, dict):
        return []
    vals = []
    for item in results_data.get("per_view", []):
        if not isinstance(item, dict):
            continue
        summary = item.get("summary", {})
        for key in ("background_sampson_ratio", "background_sampson_refractive_median"):
            v = summary.get(key)
            if v is not None:
                vals.append(float(v))
                break
    return vals


def _verdict(ratios: list[float], bg_vals: list[float], scene_name: str) -> str:
    if not ratios:
        return f"Scene {scene_name}: no data"
    med   = float(np.median(ratios))
    lo    = float(min(ratios))
    hi    = float(max(ratios))
    mean  = float(np.mean(ratios))
    bg_m  = float(np.mean(bg_vals)) if bg_vals else float('nan')

    if mean > 2.5:
        qual = "显著高于背景 (object ratio >> 1)"
    elif mean > 1.5:
        qual = "中等偏高 (object ratio > 1)"
    elif mean > 0.8:
        qual = "不显著 (object ratio ≈ 1)"
    else:
        qual = "反向! (object ratio < 1 — glass may match better than matte)"

    bg_comment = (f"BG Sampson mean = {bg_m:.2f} px" if not np.isnan(bg_m)
                  else "BG Sampson: not available")
    return (f"Scene **{scene_name}**: object_ratio mean={mean:.2f}x "
            f"(median={med:.2f}x, range {lo:.2f}–{hi:.2f}x). {qual}. {bg_comment}.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir",  required=True)
    p.add_argument("--scene-dirs",  nargs="+", required=True)
    p.add_argument("--anom-log",    default=None)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_info = {}
    for sd in args.scene_dirs:
        scene_path = Path(sd)
        name = scene_path.name
        exp_dir = Path("outputs/dual_path") / name
        verify  = _load_verify_report(scene_path)
        results = _load_results(exp_dir)
        ratios  = _extract_view_ratios(results)
        bg_vals = _extract_bg_sampson(results)
        scene_info[name] = {
            "scene_path": scene_path,
            "exp_dir":    exp_dir,
            "verify":     verify,
            "results":    results,
            "ratios":     ratios,
            "bg_vals":    bg_vals,
        }

    lines = ["# Plan B Morning Report\n"]
    lines.append(f"Generated from: {out_dir}\n")
    lines.append("---\n")

    # 1. Verify status
    lines.append("## 1. Verify Status\n")
    for name, info in scene_info.items():
        status = _verify_status(info["verify"])
        lines.append(f"- **{name}**: {status}")
    lines.append("")

    # 2. Dual-path ratios
    lines.append("## 2. Dual-Path object_sampson_ratio\n")
    for name, info in scene_info.items():
        ratios = info["ratios"]
        if ratios:
            med  = float(np.median(ratios))
            mean = float(np.mean(ratios))
            lines.append(f"### {name}")
            lines.append(f"- Per-view ratios: {[f'{r:.2f}' for r in ratios]}")
            lines.append(f"- Median: {med:.2f}x  |  Mean: {mean:.2f}x")
            bg = info["bg_vals"]
            if bg:
                lines.append(f"- Background Sampson mean: {np.mean(bg):.2f} px")
            lines.append("")
        else:
            lines.append(f"### {name}\n- No data (experiment may have failed)\n")

    # 3. Figure paths
    lines.append("## 3. Key Figures\n")
    for fn in ["figure_2_ratio_scatter.png",
               "figure_3_per_view_pattern.png",
               "all_scenes_summary.csv"]:
        fp = out_dir / fn
        status = "✓ exists" if fp.exists() else "✗ missing"
        lines.append(f"- `{fp}` — {status}")
    lines.append("")

    # 4. Anomalies
    lines.append("## 4. Anomalies\n")
    if args.anom_log and Path(args.anom_log).exists():
        content = Path(args.anom_log).read_text().strip()
        if content:
            lines.append("```")
            lines.append(content)
            lines.append("```")
        else:
            lines.append("None.")
    else:
        lines.append("No anomaly log found.")
    lines.append("")

    # 5. Verdict
    lines.append("## 5. Verdict\n")
    all_bg = []
    for info in scene_info.values():
        all_bg.extend(info["bg_vals"])

    for name, info in scene_info.items():
        lines.append(_verdict(info["ratios"], info["bg_vals"], name))
    lines.append("")

    if all_bg:
        lines.append(f"**Background Sampson** across both scenes: "
                     f"mean = {np.mean(all_bg):.2f} px "
                     f"({'< 5 px — good ✓' if np.mean(all_bg) < 5 else '≥ 5 px — high'})")
        lines.append("")

    # Recommend best scene
    scored = [(name, float(np.mean(info["ratios"])) if info["ratios"] else 0)
              for name, info in scene_info.items()]
    best = max(scored, key=lambda x: x[1], default=(None, 0))
    if best[0]:
        lines.append(f"**Recommended main case**: `{best[0]}` "
                     f"(highest object_ratio mean = {best[1]:.2f}x). "
                     "Use for figure_1 and presentation examples.")
    lines.append("")

    report_path = out_dir / "morning_report.md"
    report_path.write_text("\n".join(lines))
    print(f"Morning report → {report_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
