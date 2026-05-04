"""
Run Zero123++ on brass_goblet_v2 cond images (refractive and matte) and save all
outputs alongside Blender GT for direct visual comparison.

No cache. Forces fresh inference every run.

Output: outputs/zero123_inspection/brass_goblet_v2/
  refractive_cond.png         — copy of rgb/cond.png
  refractive_view_0..5.png    — Zero123++ output (320×320)
  matte_cond.png              — copy of matte/cond.png
  matte_view_0..5.png         — Zero123++ output (320×320)
  gt_refractive_view_0..5.png — Blender GT from rgb/
  gt_matte_view_0..5.png      — Blender GT from matte/
"""

from __future__ import annotations
import shutil
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.generation import Zero123PlusPipeline

SCENE_DIR = ROOT / "data" / "scene_brass_goblet_v2"
OUT_DIR   = ROOT / "outputs" / "zero123_inspection" / "brass_goblet_v2"

COND_REFR  = SCENE_DIR / "rgb"   / "cond.png"
COND_MATTE = SCENE_DIR / "matte" / "cond.png"


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Copy GT Blender views ──────────────────────────────────────────────────
    print("Copying Blender GT views …")
    for i in range(6):
        shutil.copy(SCENE_DIR / "rgb"   / f"view_{i}.png", OUT_DIR / f"gt_refractive_view_{i}.png")
        shutil.copy(SCENE_DIR / "matte" / f"view_{i}.png", OUT_DIR / f"gt_matte_view_{i}.png")

    # ── Load cond images ───────────────────────────────────────────────────────
    cond_refr  = Image.open(COND_REFR).convert("RGB")
    cond_matte = Image.open(COND_MATTE).convert("RGB")

    # ── Run Zero123++ ──────────────────────────────────────────────────────────
    print("Loading Zero123++ pipeline …")
    with Zero123PlusPipeline(num_inference_steps=36) as gen:
        print("Generating refractive views …")
        refr_views = gen.generate(cond_refr)
        print("Generating matte views …")
        matte_views = gen.generate(cond_matte)

    # ── Save outputs ───────────────────────────────────────────────────────────
    print("Saving outputs …")
    shutil.copy(COND_REFR,  OUT_DIR / "refractive_cond.png")
    shutil.copy(COND_MATTE, OUT_DIR / "matte_cond.png")

    for i, img in enumerate(refr_views):
        img.save(OUT_DIR / f"refractive_view_{i}.png")

    for i, img in enumerate(matte_views):
        img.save(OUT_DIR / f"matte_view_{i}.png")

    print(f"\nDone. Output: {OUT_DIR}")


if __name__ == "__main__":
    run()
