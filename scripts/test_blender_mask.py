"""
Test load_blender_mask() consistency.

Strategy (no real Blender data needed):
  1. Load the rembg masks already saved by the smoke test.
  2. Save them as PNG files → these are the "fake Blender masks".
  3. Run the pipeline twice on the same condition image using the cache:
       run A: rembg segmentation (default)
       run B: load_blender_mask() from the PNGs saved in step 2
  4. Assert that mask arrays are bit-identical and that pair stats match exactly.
"""

from __future__ import annotations
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation import Zero123PlusPipeline
from src.matching import LoFTRMatcher
from src.pipeline import TransparentObjectAnalyzer
from src.region_analysis import load_blender_mask

SMOKE     = Path("outputs/single_run/smoke_test")
COND_PATH = Path("outputs/convention_test/cond.png")
CACHE_DIR = Path("outputs/_cache/views")


def main() -> None:
    # ── Step 1: load the rembg masks that were saved during the smoke test ──
    def load_mask(p: Path) -> np.ndarray:
        return (np.array(Image.open(p).convert("L")) > 127).astype(np.uint8)

    rembg_mask_cond  = load_mask(SMOKE / "mask_cond.png")
    rembg_mask_views = [load_mask(SMOKE / f"mask_view_{i}.png") for i in range(6)]

    # ── Step 2: save them as "fake Blender masks" PNGs in a temp dir ─────────
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        fake_mask_cond  = tmp / "blender_cond.png"
        fake_mask_views = [tmp / f"blender_view_{i}.png" for i in range(6)]

        Image.fromarray(rembg_mask_cond * 255).save(fake_mask_cond)
        for i, m in enumerate(rembg_mask_views):
            Image.fromarray(m * 255).save(fake_mask_views[i])

        # ── Step 3a: verify load_blender_mask round-trips correctly ─────────
        loaded_cond = load_blender_mask(fake_mask_cond)
        assert np.array_equal(rembg_mask_cond, loaded_cond), \
            "mask_cond mismatch after round-trip!"
        for i in range(6):
            loaded_v = load_blender_mask(fake_mask_views[i])
            assert np.array_equal(rembg_mask_views[i], loaded_v), \
                f"mask_view_{i} mismatch after round-trip!"
        print("[PASS] load_blender_mask: all 7 masks round-trip bit-identical")

        # ── Step 3b: run pipeline twice and compare pair stats ───────────────
        matcher   = LoFTRMatcher(weights="outdoor", device="cuda")
        generator = Zero123PlusPipeline(device="cuda", num_inference_steps=36)

        with generator:
            analyzer = TransparentObjectAnalyzer(
                generator=generator,
                matcher=matcher,
                seg_method="auto",
            )

            print("\n--- Run A: rembg segmentation ---")
            result_rembg = analyzer.analyze(
                COND_PATH,
                radius=1.5,
                cache_dir=CACHE_DIR,
            )

            print("\n--- Run B: fake Blender masks ---")
            result_blender = analyzer.analyze(
                COND_PATH,
                radius=1.5,
                cache_dir=CACHE_DIR,
                mask_cond_path=fake_mask_cond,
                mask_view_paths=fake_mask_views,
            )

        # ── Step 4: compare stats ─────────────────────────────────────────────
        print("\n{'='*60}")
        print(f"{'V':>2} | {'metric':<28} | {'rembg':>10} | {'blender':>10} | match")
        print("-" * 65)

        all_match = True
        for pa, pb in zip(result_rembg.pairs, result_blender.pairs):
            sa, sb = pa.stats(), pb.stats()
            for key in ("n_object_matches", "n_background_matches",
                        "mean_sampson_object", "mean_sampson_background",
                        "mean_sampson_all"):
                va, vb = sa[key], sb[key]
                if va is None and vb is None:
                    ok = True
                elif va is None or vb is None:
                    ok = False
                else:
                    ok = abs(va - vb) < 1e-6
                mark = "✓" if ok else "✗ MISMATCH"
                if not ok:
                    all_match = False
                va_s = f"{va:.4f}" if isinstance(va, float) else str(va)
                vb_s = f"{vb:.4f}" if isinstance(vb, float) else str(vb)
                print(f"{pa.view_idx:>2} | {key:<28} | {va_s:>10} | {vb_s:>10} | {mark}")

        print("=" * 65)
        if all_match:
            print("[PASS] All stats identical — load_blender_mask is a perfect drop-in.")
        else:
            print("[FAIL] Stats diverged — check the comparison above.")
            sys.exit(1)


if __name__ == "__main__":
    main()
