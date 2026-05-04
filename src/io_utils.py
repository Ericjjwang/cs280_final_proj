"""I/O utilities for Blender-rendered scene data (masks, depth EXR, poses)."""

from __future__ import annotations
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

# Must be set before cv2 is first imported; re-enables the EXR codec that
# opencv-python disables by default (see opencv/opencv#21326).
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2


def load_blender_alpha_mask(path: "str | Path") -> np.ndarray:
    """
    Load a Blender-exported alpha/matte mask PNG.

    Args:
        path: Greyscale or RGBA PNG from Blender (white=object, black=background).

    Returns:
        (H, W) bool — True = object pixel.
    """
    arr = np.array(Image.open(Path(path)).convert("L"))
    return arr > 127


def load_blender_depth_exr(path: "str | Path") -> np.ndarray:
    """
    Load a Blender Cycles depth-pass EXR (single-channel float32, metres).

    Background / sky pixels that Blender encodes as very large values
    (>= 1e10) are normalised to np.inf.

    Args:
        path: Float32 EXR exported from Blender (Z or Depth pass).

    Returns:
        (H, W) float32.  Invalid / background pixels are inf or nan.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(
            f"Cannot read EXR: {path}\n"
            "Ensure OPENCV_IO_ENABLE_OPENEXR=1 is set before importing cv2."
        )
    if img.ndim == 3:
        img = img[:, :, 0]
    img = img.astype(np.float32)
    img[img >= 1e10] = np.inf
    return img


def load_poses_json(path: "str | Path") -> dict:
    """
    Load poses.json and return numpy arrays.

    Expected JSON schema::

        {
          "cond":   {"R": [[...3x3...]], "t": [...3...], "K": [[...3x3...]]},
          "view_0": { ... },
          ...
          "view_5": { ... }
        }

    R and t follow the World-to-Camera OpenCV convention.

    Returns:
        dict mapping each view name to {"R": (3,3), "t": (3,), "K": (3,3)}
        float64 numpy arrays.
    """
    with open(path) as f:
        data = json.load(f)
    return {
        key: {
            "R": np.array(val["R"], dtype=np.float64),
            "t": np.array(val["t"], dtype=np.float64),
            "K": np.array(val["K"], dtype=np.float64),
        }
        for key, val in data.items()
    }


def save_poses_json(path: "str | Path", poses_dict: dict) -> None:
    """
    Write poses_dict to JSON (numpy arrays → nested lists).

    Inverse of load_poses_json; intended for test fixtures.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        key: {
            "R": np.asarray(val["R"]).tolist(),
            "t": np.asarray(val["t"]).tolist(),
            "K": np.asarray(val["K"]).tolist(),
        }
        for key, val in poses_dict.items()
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
