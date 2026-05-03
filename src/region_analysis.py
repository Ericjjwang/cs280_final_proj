"""
Object/background segmentation and region-aware match splitting.

Segmentation backends:
  "auto"       – rembg (U2-Net); best quality. Requires: pip install rembg
  "threshold"  – HSV+Otsu threshold; no extra deps; works for plain backgrounds.
  "depth_clean"– foreground = non-zero pixels in a depth/clean-mask image.
                 Caller must pass `depth_mask` kwarg (HxW uint8).
"""

from __future__ import annotations
import numpy as np
import cv2
from PIL import Image


# ── Segmentation ─────────────────────────────────────────────────────────────

def segment_object_background(
    image: np.ndarray | Image.Image,
    method: str = "auto",
    **kwargs,
) -> np.ndarray:
    """
    Binary foreground mask for a single image.

    Args:
        image:  HxWx3 uint8 RGB ndarray or PIL Image.
        method: "auto" | "threshold" | "depth_clean".
        **kwargs:
            depth_mask (ndarray): required for method="depth_clean".

    Returns:
        mask: (H, W) uint8.  1 = object, 0 = background.
    """
    arr = _to_array(image)
    if method == "auto":
        return _segment_rembg(arr)
    elif method == "threshold":
        return _segment_threshold(arr)
    elif method == "depth_clean":
        dm = kwargs.get("depth_mask")
        if dm is None:
            raise ValueError("depth_clean requires `depth_mask` kwarg")
        return (dm > 0).astype(np.uint8)
    else:
        raise ValueError(f"Unknown segmentation method: {method!r}")


def _segment_rembg(image: np.ndarray) -> np.ndarray:
    """Use rembg (U2-Net) to remove background; return binary foreground mask."""
    from rembg import remove
    pil_in = Image.fromarray(image)
    pil_out = remove(pil_in)          # RGBA output
    alpha = np.array(pil_out)[:, :, 3]
    return (alpha > 10).astype(np.uint8)


def _segment_threshold(image: np.ndarray) -> np.ndarray:
    """
    HSV-based background segmentation for plain light backgrounds.

    Converts to HSV, applies Otsu on the V channel, then morphological cleanup.
    Objects are assumed darker/more colourful than the background.
    """
    hsv   = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v     = hsv[:, :, 2]
    _, th = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,  kernel)
    return (th > 0).astype(np.uint8)


# ── Match splitting ───────────────────────────────────────────────────────────

def sample_mask(mask: np.ndarray, kpts: np.ndarray) -> np.ndarray:
    """
    Sample binary mask at keypoint locations (nearest neighbour).

    Args:
        mask: (H, W) uint8 binary.
        kpts: (N, 2) float (x, y) pixel coordinates.

    Returns:
        (N,) bool array — True where mask == 1.
    """
    h, w = mask.shape
    xs = np.clip(np.round(kpts[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(kpts[:, 1]).astype(int), 0, h - 1)
    return mask[ys, xs].astype(bool)


def split_matches_by_region(
    kpts_a: np.ndarray,
    kpts_b: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> dict:
    """
    Classify matches into object / background / mixed using BOTH image masks.

    A match is "object"     if both endpoints lie on the foreground (mask==1).
    A match is "background" if both endpoints lie on the background (mask==0).
    A match is "mixed"      if the two endpoints disagree.

    Args:
        kpts_a: (N, 2) float – cond-side pixel coordinates (x, y).
        kpts_b: (N, 2) float – view-side pixel coordinates (x, y).
        mask_a: (H, W) uint8 – foreground mask for cond image.
        mask_b: (H, W) uint8 – foreground mask for view image.

    Returns:
        dict with keys:
            "object_idx":     (M,) int64 – indices where both on foreground.
            "background_idx": (K,) int64 – indices where both on background.
            "mixed_idx":      (L,) int64 – indices where masks disagree.
            M + K + L == N.
    """
    in_obj_a = sample_mask(mask_a, kpts_a)
    in_obj_b = sample_mask(mask_b, kpts_b)

    return {
        "object_idx":     np.where( in_obj_a &  in_obj_b)[0],
        "background_idx": np.where(~in_obj_a & ~in_obj_b)[0],
        "mixed_idx":      np.where( in_obj_a ^  in_obj_b)[0],
    }


# ── Blender mask loader ───────────────────────────────────────────────────────

def load_blender_mask(mask_path: "str | Path") -> np.ndarray:
    """Read a Blender-exported alpha mask PNG.

    Blender renders object alpha as a greyscale PNG where white (255) = object
    and black (0) = background.  Threshold at 127 to get a binary mask.

    Args:
        mask_path: path to a greyscale or RGBA PNG exported from Blender.

    Returns:
        (H, W) uint8 binary mask: 1 = object, 0 = background.
    """
    from pathlib import Path as _Path
    arr = np.array(Image.open(_Path(mask_path)).convert("L"))
    return (arr > 127).astype(np.uint8)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _to_array(img: np.ndarray | Image.Image) -> np.ndarray:
    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB"), dtype=np.uint8)
    return img
