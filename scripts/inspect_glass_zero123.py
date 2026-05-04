"""
Run Zero123++ on the glass goblet cond image and save the 6 output views.
No cache. Forces fresh inference.

Output: outputs/zero123_inspection/glass_goblet/
  glass_cond.png        — copy of input cond image
  glass_view_0..5.png   — Zero123++ output (320×320)
"""

from __future__ import annotations
import shutil
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.generation import Zero123PlusPipeline

COND_PATH = ROOT / "data" / "scene_glass_goblet_cond_only" / "cond.png"
OUT_DIR   = ROOT / "outputs" / "zero123_inspection" / "glass_goblet"


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cond = Image.open(COND_PATH).convert("RGB")

    print("Loading Zero123++ pipeline …")
    with Zero123PlusPipeline(num_inference_steps=36) as gen:
        print("Generating 6 views …")
        views = gen.generate(cond)

    shutil.copy(COND_PATH, OUT_DIR / "glass_cond.png")
    for i, img in enumerate(views):
        img.save(OUT_DIR / f"glass_view_{i}.png")

    print(f"\nDone. Output: {OUT_DIR}")


if __name__ == "__main__":
    run()
