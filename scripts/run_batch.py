"""
Batch transparent-object analysis over a directory of condition images.

Usage:
    uv run python scripts/run_batch.py \\
        --input_dir data/test_objects/ \\
        --output_dir outputs/batch_run/ \\
        --radius 1.5

Input convention:
    <input_dir>/
      object_a.png   (or .jpg / .jpeg / .webp)
      object_b.png
      ...

Output structure:
    <output_dir>/
      object_a/      (same layout as run_single.py output)
      object_b/
      ...
      summary.csv    (one row per image × 6 views)
      summary.json   (same data as nested dict)

summary.csv columns:
    image_name, view_idx, az_deg, el_deg,
    n_matches, n_object, n_background,
    median_sampson_all, median_sampson_object, median_sampson_background,
    mean_sampson_all, mean_sampson_object, mean_sampson_background
"""

from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation import Zero123PlusPipeline
from src.matching import LoFTRMatcher
from src.pipeline import TransparentObjectAnalyzer


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch transparent-object analysis")
    p.add_argument("--input_dir",  required=True,          help="Directory of condition images")
    p.add_argument("--output_dir", required=True,          help="Root directory for outputs")
    p.add_argument("--radius",     type=float, default=1.5)
    p.add_argument("--steps",      type=int,   default=36)
    p.add_argument("--device",     default="cuda")
    p.add_argument("--seg",        default="auto",
                   choices=["auto", "threshold"])
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip images whose output directory already has stats.json")
    return p.parse_args()


def result_to_csv_rows(image_name: str, result: dict) -> list[dict]:
    """
    Flatten one image's result into a list of CSV-writable row dicts.

    Args:
        image_name: stem of the source image file (used as identifier).
        result:     full dict from TransparentObjectAnalyzer.analyze().

    Returns:
        List of 6 dicts (one per view), each with the columns defined in the
        module docstring.  Skipped pairs have NaN for numerical columns.
    """
    raise NotImplementedError


def main() -> None:
    args   = parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
    if not images:
        print(f"No supported images found in {in_dir}")
        return
    print(f"Found {len(images)} images.")

    # ── Load models once; reuse across all images ─────────────────────────────
    matcher = LoFTRMatcher(weights="outdoor", device=args.device)
    generator = Zero123PlusPipeline(device=args.device, num_inference_steps=args.steps)

    all_rows: list[dict] = []
    all_results: dict[str, dict] = {}

    with generator:
        analyzer = TransparentObjectAnalyzer(
            generator=generator,
            matcher=matcher,
            radius=args.radius,
            seg_method=args.seg,
        )

        for img_path in images:
            name = img_path.stem
            img_out = out_dir / name

            if args.skip_existing and (img_out / "stats.json").exists():
                print(f"  skip {name} (already done)")
                continue

            print(f"\n── {name} ──────────────────────────────")
            img_out.mkdir(parents=True, exist_ok=True)

            try:
                result = analyzer.analyze(str(img_path), radius=args.radius)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

            rows = result_to_csv_rows(name, result)
            all_rows.extend(rows)
            all_results[name] = {
                r["view_idx"]: r["stats"] for r in result["pairs"]
            }

            # Save per-image stats.json (mirrors run_single.py)
            # (heavy outputs like images omitted from batch run for speed)
            with open(img_out / "stats.json", "w") as f:
                json.dump({"pairs": [
                    {"view_idx": p["view_idx"], "stats": p["stats"]}
                    for p in result["pairs"]
                ]}, f, indent=2)

    # ── Write summary CSV & JSON ──────────────────────────────────────────────
    if all_rows:
        csv_path = out_dir / "summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nSaved: {csv_path}")

        json_path = out_dir / "summary.json"
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved: {json_path}")

    print(f"\nBatch complete.  {len(images)} images processed.")


if __name__ == "__main__":
    main()
