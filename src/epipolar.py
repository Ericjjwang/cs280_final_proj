"""
Epipolar geometry utilities — pure numpy, no side effects.
OpenCV convention throughout: x=right, y=down, z=into-scene.
"""

from __future__ import annotations
import numpy as np


def compute_F(
    K_a: np.ndarray,
    R_a: np.ndarray,
    t_a: np.ndarray,
    K_b: np.ndarray,
    R_b: np.ndarray,
    t_b: np.ndarray,
) -> np.ndarray:
    """
    Fundamental matrix F such that  x_b^T F x_a = 0.

    World-to-camera: P_cam = R @ P_world + t

        R_ba = R_b @ R_a.T
        t_ba = t_b - R_ba @ t_a
        E    = [t_ba]_x @ R_ba
        F    = K_b^{-T} @ E @ K_a^{-1}
    """
    R_ba = R_b @ R_a.T
    t_ba = t_b - R_ba @ t_a
    E    = _skew(t_ba) @ R_ba
    F    = np.linalg.inv(K_b).T @ E @ np.linalg.inv(K_a)
    return F


def sampson_distance(
    F: np.ndarray,
    pts_a: np.ndarray,
    pts_b: np.ndarray,
) -> np.ndarray:
    """
    Sampson (first-order geometric) distance in ~pixels.

        d = sqrt( (x_b^T F x_a)^2
                  / ( (Fx_a)_1^2 + (Fx_a)_2^2 + (F^T x_b)_1^2 + (F^T x_b)_2^2 ) )

    Args:
        F:     (3, 3) fundamental matrix.
        pts_a: (N, 2) pixel coords in image A.
        pts_b: (N, 2) pixel coords in image B.

    Returns:
        (N,) float64 Sampson distances in ~pixels.
    """
    N = len(pts_a)
    ha = np.c_[pts_a, np.ones(N)]          # (N, 3)
    hb = np.c_[pts_b, np.ones(N)]

    Fha  = (F   @ ha.T).T                  # (N, 3)
    FThb = (F.T @ hb.T).T

    num  = np.einsum("ij,ij->i", hb, Fha) ** 2
    den  = Fha[:, 0]**2 + Fha[:, 1]**2 + FThb[:, 0]**2 + FThb[:, 1]**2
    return np.sqrt(np.abs(num) / (den + 1e-10))


def sampson_stats(distances: np.ndarray) -> dict:
    """Descriptive statistics for a set of Sampson distances."""
    if len(distances) == 0:
        return dict(n=0, mean=None, median=None, std=None, pct90=None)
    return dict(
        n      = int(len(distances)),
        mean   = float(distances.mean()),
        median = float(np.median(distances)),
        std    = float(distances.std()),
        pct90  = float(np.percentile(distances, 90)),
    )


def _skew(v: np.ndarray) -> np.ndarray:
    """3-vector → 3×3 skew-symmetric matrix."""
    return np.array([[ 0,    -v[2],  v[1]],
                     [ v[2],  0,    -v[0]],
                     [-v[1],  v[0],  0   ]], dtype=np.float64)
