"""
Verify a Blender-rendered scene directory against the project spec.

Checks
------
 1  Directory structure present (rgb/, matte/, mask/, depth/, poses.json)
 2  Each folder contains cond + view_0..5 (7 files)
 3  All images 512×512
 4  mask/ values strictly {0, 255} — no antialiasing
 5  depth/ p10–p90 in [0.5, 3.0] m for each view
 6  poses.json view_0..5 az/el within ±2° of Zero123++ convention
 7  poses.json cond az=0°, el=0°, within ±2°
 8  All cameras at radius 1.5 m ±0.05
 9  rgb/ and matte/ contain the same set of view files (shared poses)
10  [bonus] K matrices identical across all 7 views
11  [bonus] R matrices are valid rotations (det≈1, R Rᵀ≈I)

Outputs
-------
  • Coloured PASS/FAIL to stdout
  • verification_report.json written to the scene root
  • sys.exit(1) on any failure

Usage
-----
    uv run python scripts/verify_scene_data.py outputs/my_scene/
"""

from __future__ import annotations
import sys
import json
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.io_utils import load_blender_depth_exr, load_poses_json

# ── Constants ─────────────────────────────────────────────────────────────────
VIEWS      = ["cond"] + [f"view_{i}" for i in range(6)]
COND_AZ    = 0.0
COND_EL    = 10.0
VIEW_AZ    = [30.0, 90.0, 150.0, 210.0, 270.0, 330.0]
VIEW_EL    = [20.0, -10.0, 20.0, -10.0, 20.0, -10.0]
ANGLE_TOL  = 2.0
RADIUS_EXP = 1.5
RADIUS_TOL = 0.05
DEPTH_LO   = 0.5
DEPTH_HI   = 6.0
IMG_SIZE   = 512
IMG_EXTS   = {".png", ".jpg", ".jpeg"}

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED   = "\033[91m"
BOLD  = "\033[1m"
RESET = "\033[0m"


# ── Reporting ─────────────────────────────────────────────────────────────────

def _report(ok: bool, label: str, detail: str = "") -> dict:
    color  = GREEN if ok else RED
    mark   = "PASS" if ok else "FAIL"
    suffix = f"  {detail}" if detail else ""
    print(f"  {color}{BOLD}[{mark}]{RESET} {label}{suffix}")
    return {"check": label, "passed": ok, "detail": detail}


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _to_spherical(R: np.ndarray, t: np.ndarray):
    """World-to-cam (R, t) → (az_deg ∈ [0,360), el_deg, radius)."""
    pos    = -R.T @ t
    radius = float(np.linalg.norm(pos))
    el_deg = float(np.degrees(np.arcsin(np.clip(pos[1] / radius, -1.0, 1.0))))
    az_deg = float(np.degrees(np.arctan2(pos[0], pos[2])) % 360)
    return az_deg, el_deg, radius


def _az_diff(a: float, b: float) -> float:
    """Minimum absolute azimuth difference, handling 0°/360° wrap."""
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d


# ── File helpers ──────────────────────────────────────────────────────────────

def _find_img(folder: Path, stem: str) -> "Path | None":
    for ext in IMG_EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


# ── Checks ────────────────────────────────────────────────────────────────────

def chk_structure(scene: Path) -> list[dict]:
    missing = [d for d in ["rgb", "matte", "mask", "depth"] if not (scene / d).is_dir()]
    if not (scene / "poses.json").is_file():
        missing.append("poses.json")
    ok = len(missing) == 0
    return [_report(ok, "File structure complete",
                    f"missing: {missing}" if missing else "")]


def chk_counts(scene: Path) -> list[dict]:
    results = []
    for folder, exts in [("rgb",   IMG_EXTS), ("matte", IMG_EXTS),
                          ("mask",  IMG_EXTS), ("depth", {".exr"})]:
        d       = scene / folder
        missing = [v for v in VIEWS if not any((d / f"{v}{e}").exists() for e in exts)]
        ok      = len(missing) == 0
        results.append(_report(
            ok, f"File count {folder}/",
            f"missing: {missing}" if missing else "7 files present",
        ))
    return results


def chk_resolution(scene: Path) -> list[dict]:
    bad = []
    for folder in ["rgb", "matte", "mask"]:
        for stem in VIEWS:
            p = _find_img(scene / folder, stem)
            if p is None:
                continue
            with Image.open(p) as img:
                if img.size != (IMG_SIZE, IMG_SIZE):
                    bad.append(f"{folder}/{stem}: {img.size}")
    ok = len(bad) == 0
    return [_report(ok, f"Resolution {IMG_SIZE}×{IMG_SIZE}",
                    "; ".join(bad) if bad else "all images OK")]


def chk_mask_binary(scene: Path) -> list[dict]:
    bad = []
    for stem in VIEWS:
        p = _find_img(scene / "mask", stem)
        if p is None:
            continue
        arr   = np.array(Image.open(p).convert("L"))
        extra = sorted(set(arr.ravel().tolist()) - {0, 255})
        if extra:
            bad.append(f"mask/{stem}: non-binary values {extra[:8]}")
    ok = len(bad) == 0
    return [_report(ok, "Mask strictly binary (0/255)",
                    "; ".join(bad) if bad else "all 7 masks OK")]


def chk_depth_range(scene: Path) -> list[dict]:
    bad = []
    for stem in VIEWS:
        p = scene / "depth" / f"{stem}.exr"
        if not p.exists():
            continue
        depth = load_blender_depth_exr(p)
        valid = depth[np.isfinite(depth)]
        if len(valid) == 0:
            bad.append(f"{stem}: no finite pixels"); continue
        p10, p90 = float(np.percentile(valid, 10)), float(np.percentile(valid, 90))
        if not (DEPTH_LO <= p10 and p90 <= DEPTH_HI):
            bad.append(f"{stem}: p10={p10:.3f} p90={p90:.3f} "
                       f"(want [{DEPTH_LO}, {DEPTH_HI}] m)")
    ok = len(bad) == 0
    return [_report(ok, f"Depth p10–p90 within [{DEPTH_LO}, {DEPTH_HI}] m",
                    "; ".join(bad) if bad else "all 7 EXRs OK")]


def chk_view_poses(poses: dict) -> list[dict]:
    bad = []
    for i in range(6):
        key = f"view_{i}"
        if key not in poses:
            bad.append(f"{key}: missing"); continue
        az, el, _ = _to_spherical(poses[key]["R"], poses[key]["t"])
        az_err    = _az_diff(az, VIEW_AZ[i])
        el_err    = abs(el - VIEW_EL[i])
        if az_err > ANGLE_TOL or el_err > ANGLE_TOL:
            bad.append(f"{key}: got az={az:.1f}° el={el:.1f}°, "
                       f"want az={VIEW_AZ[i]:.0f}° el={VIEW_EL[i]:.0f}°")
    ok = len(bad) == 0
    return [_report(ok, "View az/el match Zero123++ convention (±2°)",
                    "; ".join(bad) if bad else "all 6 views OK")]


def chk_cond_pose(poses: dict) -> list[dict]:
    if "cond" not in poses:
        return [_report(False, "Cond pose (az=0°, el=0°)", "missing 'cond' key")]
    az, el, _ = _to_spherical(poses["cond"]["R"], poses["cond"]["t"])
    az_err    = _az_diff(az, COND_AZ)
    el_err    = abs(el - COND_EL)
    ok        = az_err <= ANGLE_TOL and el_err <= ANGLE_TOL
    detail    = (f"az={az:.2f}° el={el:.2f}°"
                 + (f"  az_err={az_err:.2f}° el_err={el_err:.2f}°" if not ok else ""))
    return [_report(ok, "Cond pose az=0° el=0° (±2°)", detail)]


def chk_radius(poses: dict) -> list[dict]:
    bad = []
    for key, p in poses.items():
        _, _, r = _to_spherical(p["R"], p["t"])
        if abs(r - RADIUS_EXP) > RADIUS_TOL:
            bad.append(f"{key}: r={r:.4f} m")
    ok = len(bad) == 0
    return [_report(ok, f"Camera radius {RADIUS_EXP} m ±{RADIUS_TOL} m",
                    "; ".join(bad) if bad else "all 7 cameras OK")]


def chk_pose_consistency(scene: Path) -> list[dict]:
    def _stems(folder: str) -> set:
        d = scene / folder
        return {f.stem for f in d.iterdir() if f.suffix.lower() in IMG_EXTS}
    rgb_s = _stems("rgb")
    mat_s = _stems("matte")
    ok     = rgb_s == mat_s
    detail = ("" if ok else
              f"rgb-only={sorted(rgb_s - mat_s)}  matte-only={sorted(mat_s - rgb_s)}")
    return [_report(ok, "rgb/ and matte/ share same view set (pose consistency)", detail)]


def chk_K_consistent(poses: dict) -> list[dict]:
    ref = next(iter(poses.values()))["K"]
    bad = [k for k, p in poses.items() if not np.allclose(p["K"], ref, atol=1e-4)]
    ok  = len(bad) == 0
    return [_report(ok, "K matrices identical across all views",
                    f"differ: {bad}" if bad else "")]


def chk_R_valid(poses: dict) -> list[dict]:
    bad = []
    for key, p in poses.items():
        R   = p["R"]
        det = float(np.linalg.det(R))
        I   = R @ R.T
        if abs(det - 1.0) > 1e-5 or not np.allclose(I, np.eye(3), atol=1e-5):
            bad.append(f"{key}: det={det:.6f}")
    ok = len(bad) == 0
    return [_report(ok, "R matrices valid rotations (det≈1, R Rᵀ≈I)",
                    "; ".join(bad) if bad else "")]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Verify a Blender-rendered scene against the project spec.")
    parser.add_argument("scene_dir", help="Root of the rendered scene directory")
    args  = parser.parse_args()
    scene = Path(args.scene_dir)

    print(f"\n{BOLD}Verifying scene:{RESET} {scene.resolve()}\n")
    results: list[dict] = []

    # Check 1 — structure (abort early if dirs missing)
    r = chk_structure(scene)
    results += r
    if not r[0]["passed"]:
        return _finish(scene, results)

    print()
    # Checks 2–5: per-file
    results += chk_counts(scene)
    print()
    results += chk_resolution(scene)
    results += chk_mask_binary(scene)
    results += chk_depth_range(scene)

    # Load poses (abort if unparseable)
    print()
    try:
        poses = load_poses_json(scene / "poses.json")
    except Exception as e:
        results.append(_report(False, "poses.json parseable", str(e)))
        return _finish(scene, results)

    # Checks 6–9: geometry
    results += chk_view_poses(poses)
    results += chk_cond_pose(poses)
    results += chk_radius(poses)
    results += chk_pose_consistency(scene)

    # Bonus checks 10–11
    print()
    results += chk_K_consistent(poses)
    results += chk_R_valid(poses)

    _finish(scene, results)


def _finish(scene: Path, results: list[dict]):
    n_pass = sum(r["passed"] for r in results)
    n_fail = sum(not r["passed"] for r in results)

    print(f"\n{'='*54}")
    color = GREEN if n_fail == 0 else RED
    print(f"  {color}{BOLD}{n_pass} PASS  {n_fail} FAIL{RESET}")
    print(f"{'='*54}")

    report_path = scene / "verification_report.json"
    with open(report_path, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"  Report → {report_path}\n")

    sys.exit(1 if n_fail > 0 else 0)


if __name__ == "__main__":
    main()
