"""
Generate a spec-compliant fake scene for testing verify_scene_data.py.

Default output: outputs/_fake_scene/

  rgb/    cond.png  view_0.png … view_5.png   512×512 solid-colour RGB
  matte/  cond.png  view_0.png … view_5.png   512×512 solid-colour RGB
  mask/   cond.png  view_0.png … view_5.png   512×512 filled white circle on black
  depth/  cond.exr  view_0.exr … view_5.exr   512×512 constant 1.5 m float32
  poses.json                                   exact Zero123++ poses, cond el=0°

Usage
-----
    uv run python scripts/_make_fake_scene.py [--out outputs/_fake_scene]
    uv run python scripts/verify_scene_data.py outputs/_fake_scene
"""

from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Enable EXR before cv2 is first imported.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.io_utils import save_poses_json

# ── Spec constants ────────────────────────────────────────────────────────────
VIEWS    = ["cond"] + [f"view_{i}" for i in range(6)]
SIZE     = 512
RADIUS   = 1.5
FOV_DEG  = 30.0

COND_AZ  = 0.0
COND_EL  = 0.0
VIEW_AZ  = [30.0,  90.0, 150.0, 210.0, 270.0, 330.0]
VIEW_EL  = [20.0, -10.0,  20.0, -10.0,  20.0, -10.0]


# ── Geometry (mirrors src/camera.py private helpers) ──────────────────────────

def _sph_to_xyz(az_deg: float, el_deg: float, r: float = RADIUS) -> np.ndarray:
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    return np.array([
        r * np.cos(el) * np.sin(az),
        r * np.sin(el),
        r * np.cos(el) * np.cos(az),
    ], dtype=np.float64)


def _look_at(eye: np.ndarray, target: np.ndarray = np.zeros(3)):
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    up = np.array([0., 1., 0.])
    if abs(np.dot(fwd, up)) > 0.999:
        up = np.array([0., 0., -1.])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    t = -R @ eye
    return R, t


def _K(size: int = SIZE, fov_deg: float = FOV_DEG) -> np.ndarray:
    f = (size / 2) / np.tan(np.deg2rad(fov_deg / 2))
    c = size / 2.0
    return np.array([[f, 0, c], [0, f, c], [0, 0, 1.0]], dtype=np.float64)


# ── Image factories ───────────────────────────────────────────────────────────

def _solid(color: tuple) -> Image.Image:
    return Image.new("RGB", (SIZE, SIZE), color)


def _circle_mask() -> Image.Image:
    img  = Image.new("L", (SIZE, SIZE), 0)
    draw = ImageDraw.Draw(img)
    cx, cy, r = SIZE // 2, SIZE // 2, SIZE // 3
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    return img


def _write_exr(path: Path, value: float = RADIUS):
    depth = np.full((SIZE, SIZE), value, dtype=np.float32)
    ok    = cv2.imwrite(str(path), depth)
    if not ok:
        raise RuntimeError(
            f"cv2.imwrite failed writing {path}. "
            "Ensure OPENCV_IO_ENABLE_OPENEXR=1 is exported."
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/_fake_scene",
                        help="Output directory (default: outputs/_fake_scene)")
    args  = parser.parse_args()
    scene = Path(args.out)

    for sub in ["rgb", "matte", "mask", "depth"]:
        (scene / sub).mkdir(parents=True, exist_ok=True)

    # Distinct solid colours so rgb/ and matte/ are visually different
    rgb_colors   = [(180, 200, 230)] + [(140 + i * 10, 160, 200) for i in range(6)]
    matte_colors = [(200, 180, 160)] + [(180, 140 + i * 8,  130) for i in range(6)]

    mask_img = _circle_mask()   # same mask for all views (fake data)

    for idx, stem in enumerate(VIEWS):
        _solid(rgb_colors[idx]).save(scene / "rgb"   / f"{stem}.png")
        _solid(matte_colors[idx]).save(scene / "matte" / f"{stem}.png")
        mask_img.save(scene / "mask" / f"{stem}.png")
        _write_exr(scene / "depth" / f"{stem}.exr")

    # Poses: compute from spherical coordinates
    K     = _K()
    poses = {}
    R, t  = _look_at(_sph_to_xyz(COND_AZ, COND_EL))
    poses["cond"] = {"R": R, "t": t, "K": K}
    for i, (az, el) in enumerate(zip(VIEW_AZ, VIEW_EL)):
        R, t = _look_at(_sph_to_xyz(az, el))
        poses[f"view_{i}"] = {"R": R, "t": t, "K": K}

    save_poses_json(scene / "poses.json", poses)

    print(f"Fake scene → {scene.resolve()}")
    print(f"Run verify: uv run python scripts/verify_scene_data.py {scene}")


if __name__ == "__main__":
    main()
