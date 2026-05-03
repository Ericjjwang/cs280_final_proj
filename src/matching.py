"""
Feature matching via LoFTR (kornia).

LoFTR requires:
  - Grayscale float32 tensors in [0, 1], shape (1, 1, H, W).
  - H and W both multiples of 8.

Padding strategy: zero-pad on the right/bottom only → original-coords are unchanged,
so no rescaling is needed. We filter out keypoints that fall in the padded region.
"""

from __future__ import annotations
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import kornia.feature as KF


class LoFTRMatcher:
    """Semi-dense feature matcher wrapping kornia LoFTR."""

    def __init__(self, weights: str = "outdoor", device: str = "cuda") -> None:
        self.device  = device
        self.weights = weights
        self._model  = KF.LoFTR(pretrained=weights).eval().to(device)

    # ── Public ────────────────────────────────────────────────────────────────

    def match(
        self,
        img_a: np.ndarray | Image.Image,
        img_b: np.ndarray | Image.Image,
        min_conf: float = 0.5,
    ) -> dict:
        """
        Match keypoints between two images.

        Args:
            img_a:    HxWx3 uint8 RGB ndarray or PIL Image.
            img_b:    HxWx3 uint8 RGB ndarray or PIL Image.
            min_conf: confidence threshold.

        Returns:
            dict:
                "kpts_a": (N, 2) float32 – pixel (x, y) in ORIGINAL img_a coords.
                "kpts_b": (N, 2) float32 – pixel (x, y) in ORIGINAL img_b coords.
                "conf":   (N,)  float32.
        """
        arr_a = self._to_array(img_a)
        arr_b = self._to_array(img_b)

        h_a, w_a = arr_a.shape[:2]
        h_b, w_b = arr_b.shape[:2]
        assert h_a >= 224 and w_a >= 224, f"img_a too small: {arr_a.shape}"
        assert h_b >= 224 and w_b >= 224, f"img_b too small: {arr_b.shape}"

        ta = self._to_loftr_tensor(arr_a)   # (1,1,H',W') padded
        tb = self._to_loftr_tensor(arr_b)

        with torch.no_grad():
            out = self._model({"image0": ta, "image1": tb})

        kpts_a = out["keypoints0"].cpu().numpy()   # (N, 2) in padded coords
        kpts_b = out["keypoints1"].cpu().numpy()
        conf   = out["confidence"].cpu().numpy()

        # Confidence filter
        mask = conf >= min_conf
        kpts_a, kpts_b, conf = kpts_a[mask], kpts_b[mask], conf[mask]

        # Discard keypoints that landed in the padded region
        valid = (
            (kpts_a[:, 0] < w_a) & (kpts_a[:, 1] < h_a) &
            (kpts_b[:, 0] < w_b) & (kpts_b[:, 1] < h_b)
        )
        return {
            "kpts_a": kpts_a[valid].astype(np.float32),
            "kpts_b": kpts_b[valid].astype(np.float32),
            "conf":   conf[valid].astype(np.float32),
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _to_array(img: np.ndarray | Image.Image) -> np.ndarray:
        """Ensure HxWx3 uint8 RGB ndarray."""
        if isinstance(img, Image.Image):
            return np.array(img.convert("RGB"), dtype=np.uint8)
        return img

    def _to_loftr_tensor(self, img: np.ndarray) -> torch.Tensor:
        """HxWx3 RGB uint8 (or HxW gray) → (1,1,H',W') float32 on device, padded."""
        if img.ndim == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif img.ndim == 2:
            gray = img
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")

        t = torch.from_numpy(gray).float() / 255.0          # (H, W)
        t = t.unsqueeze(0).unsqueeze(0).to(self.device)     # (1, 1, H, W)
        return self._pad_to_multiple(t)

    @staticmethod
    def _pad_to_multiple(t: torch.Tensor, multiple: int = 8) -> torch.Tensor:
        """Zero-pad (1,1,H,W) on right/bottom so H and W are multiples of `multiple`."""
        _, _, h, w = t.shape
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        if pad_h or pad_w:
            t = F.pad(t, (0, pad_w, 0, pad_h))   # (left, right, top, bottom)
        return t
