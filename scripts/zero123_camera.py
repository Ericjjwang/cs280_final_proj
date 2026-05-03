import numpy as np

# Zero123++ v1.2 fixed output poses (from arXiv:2310.15110, Table 1)
# Azimuths: relative to input view (input = 0°), CW from above in Y-up world
# Elevations: absolute, positive = above equator
# Grid layout: 3-col × 2-row, row-major
#   [view0 az=30  el=+20] [view1 az=90  el=+20] [view2 az=150 el=+20]
#   [view3 az=210 el=-10] [view4 az=270 el=-10] [view5 az=330 el=-10]
AZIMUTHS_DEG   = [30,  90, 150, 210, 270, 330]
ELEVATIONS_DEG = [20, -10,  20, -10,  20, -10]


def _look_at_opencv(eye: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute OpenCV camera extrinsics (R, t) for a camera at `eye` looking at `target`.

    OpenCV convention: x=right, y=down, z=into-scene (depth).
    World convention: Y-up, right-handed.

    R is 3×3 (world→camera rotation), t is (3,) translation.
    World point X_w maps to camera point: X_c = R @ X_w + t
    """
    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    world_up = np.array([0.0, 1.0, 0.0])
    # Gimbal-lock guard: if camera is directly above/below target
    if abs(np.dot(forward, world_up)) > 0.999:
        world_up = np.array([0.0, 0.0, -1.0])

    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)   # already unit, forward⊥right

    R = np.stack([right, down, forward], axis=0)  # rows = camera axes in world
    t = -R @ eye
    return R, t


def _spherical_to_xyz(az_deg: float, el_deg: float, radius: float) -> np.ndarray:
    """
    Spherical → Cartesian. Y-up world, az=0 → +Z front.
    az increases CW from above (standard 3D rendering convention).
    el: 0 = equator, positive = above.
    """
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    x = radius * np.cos(el) * np.sin(az)
    y = radius * np.sin(el)
    z = radius * np.cos(el) * np.cos(az)
    return np.array([x, y, z], dtype=np.float64)


def get_zero123plus_poses(radius: float = 1.5) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Returns the 6 fixed output camera extrinsics of Zero123++ v1.2.

    Assumptions:
    - Object at world origin
    - Input (condition) image at az=0°, el=0° (front, level)
    - Camera intrinsics not included; FOV is 30° for all views in v1.2

    Args:
        radius: distance from camera to object centre (default 1.5)

    Returns:
        List of 6 (R, t) tuples in row-major grid order.
        R: (3, 3) float64, world-to-camera rotation (OpenCV)
        t: (3,)  float64, translation (OpenCV)
    """
    poses = []
    for az, el in zip(AZIMUTHS_DEG, ELEVATIONS_DEG):
        eye = _spherical_to_xyz(az, el, radius)
        R, t = _look_at_opencv(eye, target=np.zeros(3))
        poses.append((R, t))
    return poses


def get_condition_pose(radius: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    """Input image camera pose (az=0, el=0, front view)."""
    eye = _spherical_to_xyz(0.0, 0.0, radius)
    return _look_at_opencv(eye, target=np.zeros(3))


if __name__ == "__main__":
    poses = get_zero123plus_poses(radius=1.5)
    print("Zero123++ v1.2 — 6 output camera extrinsics (OpenCV, radius=1.5)\n")
    for i, (az, el, (R, t)) in enumerate(
        zip(AZIMUTHS_DEG, ELEVATIONS_DEG, poses)
    ):
        cam_pos = -R.T @ t   # camera centre in world = R^T @ (-t)
        print(f"View {i}  az={az:>3}°  el={el:>3}°")
        print(f"  camera position (world): {cam_pos.round(4)}")
        print(f"  t: {t.round(4)}")
        print(f"  R:\n{R.round(4)}\n")
