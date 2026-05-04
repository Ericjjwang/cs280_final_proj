"""
Zero123++ v1.2 camera geometry.

Conventions (fixed throughout the project):
  World    : Y-up, right-handed.  +Z = front, +X = right.
  Camera   : OpenCV. +X = right, +Y = down, +Z = into scene (depth).
  Azimuth  : CW from above (+Z → +X → -Z → -X). 0° = front (+Z).
  Elevation: from equatorial plane. positive = above (camera looks down).
  Radius   : distance from camera centre to world origin.

Six fixed output poses of Zero123++ v1.2 (arXiv:2310.15110, Table 1):
  Index  Azimuth  Elevation   Grid position (640×960, 2-col × 3-row)
  0      30°      +20°        top-left
  1      90°      -10°        top-right
  2      150°     +20°        mid-left
  3      210°     -10°        mid-right
  4      270°     +20°        bot-left
  5      330°     -10°        bot-right

Condition image is implicitly at az=0°, el=10°.
Output FOV is unified to 30° in v1.2.
"""

from __future__ import annotations
import numpy as np

AZIMUTHS_DEG:   list[int] = [30,  90, 150, 210, 270, 330]
ELEVATIONS_DEG: list[int] = [20, -10,  20, -10,  20, -10]

IMG_SIZE: int   = 320
FOV_DEG:  float = 30.0


# ── Public API ────────────────────────────────────────────────────────────────

def get_intrinsics(img_size: int = IMG_SIZE, fov_deg: float = FOV_DEG) -> np.ndarray:
    """Pinhole camera intrinsic matrix K for a square image (zero skew, centred PP)."""
    focal = (img_size / 2) / np.tan(np.deg2rad(fov_deg / 2))
    cx = cy = img_size / 2.0
    return np.array([[focal, 0,     cx],
                     [0,     focal, cy],
                     [0,     0,     1.0]], dtype=np.float64)


def get_condition_pose(radius: float = 1.5, el_deg: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Extrinsic (R, t) of the condition camera: az=0°, el=10°."""
    eye = _spherical_to_xyz(0.0, el_deg, radius)
    return _look_at_opencv(eye, np.zeros(3))


def get_zero123plus_poses(radius: float = 1.5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Extrinsics of the 6 fixed output views of Zero123++ v1.2 (grid reading order)."""
    return [
        _look_at_opencv(_spherical_to_xyz(az, el, radius), np.zeros(3))
        for az, el in zip(AZIMUTHS_DEG, ELEVATIONS_DEG)
    ]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _spherical_to_xyz(az_deg: float, el_deg: float, radius: float) -> np.ndarray:
    """Spherical (az CW-from-above, el above-equator) → Cartesian (Y-up, +Z front)."""
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    return np.array([
        radius * np.cos(el) * np.sin(az),
        radius * np.sin(el),
        radius * np.cos(el) * np.cos(az),
    ], dtype=np.float64)


def _look_at_opencv(eye: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV look-at: camera at `eye`, looking at `target`. Y-up world."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    world_up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(forward, world_up)) > 0.999:   # gimbal-lock guard
        world_up = np.array([0.0, 0.0, -1.0])

    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)

    R = np.stack([right, down, forward], axis=0)
    t = -R @ eye
    return R, t


if __name__ == "__main__":
    K = get_intrinsics()
    print(f"K (fov={FOV_DEG}°, size={IMG_SIZE}):\n{K.round(2)}\n")
    for i, (az, el, (R, t)) in enumerate(
        zip(AZIMUTHS_DEG, ELEVATIONS_DEG, get_zero123plus_poses())
    ):
        print(f"View {i} az={az:>3}° el={el:>3}°  cam={(-R.T @ t).round(3)}")
