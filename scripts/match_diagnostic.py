"""
Generate match_diagnostic.png: side-by-side cond/view_0 with mask overlays
and match lines coloured by region (blue=object, yellow=bg, orange=mixed).
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.region_analysis import split_matches_by_region

SMOKE = Path("outputs/single_run/smoke_test")


def load_mask(path: Path) -> np.ndarray:
    """Load saved mask PNG → (H,W) uint8 binary."""
    arr = np.array(Image.open(path).convert("L"))
    return (arr > 127).astype(np.uint8)


def overlay_mask(ax, mask: np.ndarray, color=(1, 0, 0), alpha=0.35):
    """Draw semi-transparent coloured overlay where mask==1."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[mask == 1, :3] = color
    rgba[mask == 1,  3] = alpha
    ax.imshow(rgba)


def main():
    # ── Load assets ──────────────────────────────────────────────────────────
    with open(SMOKE / "stats_full.json") as f:
        data = json.load(f)

    pair0 = data["pairs"][0]          # view 0, az=30°
    kpts_a = np.array(pair0["kpts_a"], dtype=np.float32)   # (50, 2)
    kpts_b = np.array(pair0["kpts_b"], dtype=np.float32)

    cond_img  = Image.open(SMOKE / "cond.png").convert("RGB")
    view0_img = Image.open(SMOKE / "view_0.png").convert("RGB")
    mask_cond  = load_mask(SMOKE / "mask_cond.png")
    mask_view0 = load_mask(SMOKE / "mask_view_0.png")

    # ── Re-derive region indices ──────────────────────────────────────────────
    split = split_matches_by_region(kpts_a, kpts_b, mask_cond, mask_view0)
    obj_idx = split["object_idx"]
    bg_idx  = split["background_idx"]
    mx_idx  = split["mixed_idx"]

    print(f"object={len(obj_idx)}  bg={len(bg_idx)}  mixed={len(mx_idx)}")

    # ── Figure: two images side by side ──────────────────────────────────────
    W = cond_img.width    # 320
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(wspace=0.02)

    for ax, img, mask, title in [
        (axes[0], cond_img,  mask_cond,  "Cond (kpts_a)"),
        (axes[1], view0_img, mask_view0, "View 0  az=30°  (kpts_b)"),
    ]:
        ax.imshow(img)
        overlay_mask(ax, mask)
        ax.contour(mask, levels=[0.5], colors="white", linewidths=0.8)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    # ── Draw match lines using ConnectionPatch ───────────────────────────────
    STYLE = {
        "obj": dict(color="#3a86ff", lw=0.9, alpha=0.85),   # blue
        "bg":  dict(color="#ffd166", lw=0.9, alpha=0.85),   # yellow
        "mx":  dict(color="#ef8c00", lw=0.9, alpha=0.85),   # orange
    }

    def draw_matches(indices, style):
        for i in indices:
            con = matplotlib.patches.ConnectionPatch(
                xyA=tuple(kpts_a[i]), coordsA=axes[0].transData,
                xyB=tuple(kpts_b[i]), coordsB=axes[1].transData,
                axesA=axes[0], axesB=axes[1],
                **style,
            )
            fig.add_artist(con)

    draw_matches(bg_idx,  STYLE["bg"])   # draw bg first (under object)
    draw_matches(mx_idx,  STYLE["mx"])
    draw_matches(obj_idx, STYLE["obj"])

    # ── Scatter keypoints on each panel ──────────────────────────────────────
    for idx, c in [(obj_idx, "#3a86ff"), (bg_idx, "#ffd166"), (mx_idx, "#ef8c00")]:
        if len(idx):
            axes[0].scatter(kpts_a[idx, 0], kpts_a[idx, 1],
                            s=18, c=c, edgecolors="k", linewidths=0.4, zorder=5)
            axes[1].scatter(kpts_b[idx, 0], kpts_b[idx, 1],
                            s=18, c=c, edgecolors="k", linewidths=0.4, zorder=5)

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color="#3a86ff", label=f"object ({len(obj_idx)})"),
        mpatches.Patch(color="#ffd166", label=f"background ({len(bg_idx)})"),
        mpatches.Patch(color="#ef8c00", label=f"mixed ({len(mx_idx)})"),
        mpatches.Patch(color=(1,0,0,0.4), label="mask overlay"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    out = SMOKE / "match_diagnostic.png"
    fig.savefig(str(out), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
