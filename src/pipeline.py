"""
Main analysis pipeline integrating all sub-modules.
"""

from __future__ import annotations
import gc
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from src.camera import (
    get_intrinsics, get_condition_pose, get_zero123plus_poses,
    AZIMUTHS_DEG, ELEVATIONS_DEG, IMG_SIZE, FOV_DEG,
)
from src.epipolar import compute_F, sampson_distance, sampson_stats
from src.generation import Zero123PlusPipeline
from src.matching import LoFTRMatcher
from src.region_analysis import segment_object_background, split_matches_by_region, load_blender_mask


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class PairResult:
    view_idx:   int
    azimuth:    float
    elevation:  float
    skipped:    bool

    n_matches:            int
    n_object_matches:     int
    n_background_matches: int
    n_mixed_matches:      int

    sampson_all:        np.ndarray = field(repr=False)
    sampson_object:     np.ndarray = field(repr=False)
    sampson_background: np.ndarray = field(repr=False)
    sampson_mixed:      np.ndarray = field(repr=False)

    kpts_a: np.ndarray = field(repr=False)
    kpts_b: np.ndarray = field(repr=False)
    conf:   np.ndarray = field(repr=False)
    F:      np.ndarray = field(repr=False)

    def stats(self) -> dict:
        """Return scalar statistics (JSON-serialisable)."""
        def _mean(a):   return float(a.mean())   if len(a) > 0 else None
        def _median(a): return float(np.median(a)) if len(a) > 0 else None

        return {
            "view_idx":               self.view_idx,
            "azimuth":                self.azimuth,
            "elevation":              self.elevation,
            "skipped":                self.skipped,
            "n_matches":              self.n_matches,
            "n_object_matches":       self.n_object_matches,
            "n_background_matches":   self.n_background_matches,
            "n_mixed_matches":        self.n_mixed_matches,
            "mean_sampson_all":       _mean(self.sampson_all),
            "median_sampson_all":     _median(self.sampson_all),
            "mean_sampson_object":    _mean(self.sampson_object),
            "median_sampson_object":  _median(self.sampson_object),
            "mean_sampson_background":   _mean(self.sampson_background),
            "median_sampson_background": _median(self.sampson_background),
            "mean_sampson_mixed":        _mean(self.sampson_mixed),
        }

    def to_dict(self, include_arrays: bool = False) -> dict:
        d = self.stats()
        if include_arrays:
            for k in ("sampson_all", "sampson_object", "sampson_background",
                      "sampson_mixed", "kpts_a", "kpts_b", "conf"):
                d[k] = getattr(self, k).tolist()
            d["F"] = self.F.tolist()
        return d


@dataclass
class AnalyzeResult:
    cond_image_path: str
    radius:          float
    model_id:        str
    pairs:           list[PairResult]

    # heavy fields — not serialised by default
    cond_image:          Image.Image = field(repr=False, default=None)
    novel_views:         list        = field(repr=False, default_factory=list)
    object_mask_cond:    np.ndarray  = field(repr=False, default=None)
    object_masks_views:  list        = field(repr=False, default_factory=list)
    K:                   np.ndarray  = field(repr=False, default=None)

    def to_json(self, path: str | Path, include_arrays: bool = False) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "cond_image_path": str(self.cond_image_path),
            "radius":          self.radius,
            "model_id":        self.model_id,
            "pairs":           [p.to_dict(include_arrays) for p in self.pairs],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def to_csv_rows(self, scene_id: Optional[str] = None) -> list[dict]:
        rows = []
        for p in self.pairs:
            row = p.stats()
            if scene_id is not None:
                row = {"scene_id": scene_id, **row}
            rows.append(row)
        return rows


# ── Analyzer ──────────────────────────────────────────────────────────────────

class TransparentObjectAnalyzer:
    """
    End-to-end: cond image → novel views → per-view Sampson / region stats.
    """

    def __init__(
        self,
        generator:      Zero123PlusPipeline,
        matcher:        LoFTRMatcher,
        radius:         float = 1.5,
        fov_deg:        float = FOV_DEG,
        seg_method:     str   = "auto",
        min_matches:    int   = 20,
        loftr_min_conf: float = 0.5,
    ) -> None:
        self.generator      = generator
        self.matcher        = matcher
        self.radius         = radius
        self.fov_deg        = fov_deg
        self.seg_method     = seg_method
        self.min_matches    = min_matches
        self.loftr_min_conf = loftr_min_conf

    def analyze(
        self,
        cond_image_path: str | Path,
        radius:            Optional[float]      = None,
        cache_dir:         Optional[Path]       = None,
        force_regenerate:  bool                 = False,
        mask_cond_path:    Optional[Path]       = None,
        mask_view_paths:   Optional[list[Path]] = None,
    ) -> AnalyzeResult:
        """
        Full pipeline for one condition image.

        Args:
            cond_image_path: path to condition image.
            radius:          override instance radius.
            cache_dir:       if set, cache generated views keyed by (image hash, model, radius).
            force_regenerate:ignore cache if True.
            mask_cond_path:  Blender alpha mask for the condition image (optional).
                             If provided, skips rembg segmentation for the cond image.
            mask_view_paths: Blender alpha masks for all 6 novel views (optional, len==6).
                             If provided, skips rembg segmentation for all views.

        Returns:
            AnalyzeResult dataclass.
        """
        r = radius if radius is not None else self.radius
        cond_path = Path(cond_image_path)
        if not cond_path.exists():
            raise FileNotFoundError(cond_path)

        cond_img   = self._load_cond(cond_path)
        novel_views = self._get_novel_views(cond_path, cond_img, r, cache_dir, force_regenerate)

        K         = get_intrinsics(IMG_SIZE, self.fov_deg)
        R_c, t_c  = get_condition_pose(r)
        view_poses = get_zero123plus_poses(r)

        # Segmentation — use Blender masks if provided, else fall back to rembg
        if mask_cond_path is not None:
            print(f"[seg] condition image — loading Blender mask from {mask_cond_path}")
            mask_cond = load_blender_mask(mask_cond_path)
        else:
            print("[seg] condition image ...")
            mask_cond = segment_object_background(cond_img, method=self.seg_method)

        masks_views = []
        for i, v in enumerate(novel_views):
            if mask_view_paths is not None:
                print(f"[seg] view {i} — loading Blender mask from {mask_view_paths[i]}")
                masks_views.append(load_blender_mask(mask_view_paths[i]))
            else:
                print(f"[seg] view {i} ...")
                masks_views.append(segment_object_background(v, method=self.seg_method))

        # Per-pair analysis
        pairs = []
        for i, (view_img, (R_v, t_v), mask_v) in enumerate(
            zip(novel_views, view_poses, masks_views)
        ):
            pair = self._build_pair(
                view_idx=i,
                cond_img=cond_img, view_img=view_img,
                mask_cond=mask_cond, mask_view=mask_v,
                K=K, R_c=R_c, t_c=t_c, R_v=R_v, t_v=t_v,
            )
            pairs.append(pair)
            s = pair.stats()
            flag = "SKIP" if pair.skipped else (
                f"all={s['median_sampson_all']:.1f}px  "
                f"obj={s['median_sampson_object']}  bg={s['median_sampson_background']}"
            )
            print(f"  [pair {i}] az={AZIMUTHS_DEG[i]:>3}°  N={pair.n_matches:>3}  {flag}")

        return AnalyzeResult(
            cond_image_path  = str(cond_path),
            radius           = r,
            model_id         = self.generator.model_id,
            pairs            = pairs,
            cond_image       = cond_img,
            novel_views      = novel_views,
            object_mask_cond = mask_cond,
            object_masks_views = masks_views,
            K                = K,
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_cond(self, path: Path) -> Image.Image:
        img = Image.open(path).convert("RGB")
        if img.size != (IMG_SIZE, IMG_SIZE):
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        return img

    def _get_novel_views(
        self,
        cond_path: Path,
        cond_img:  Image.Image,
        radius:    float,
        cache_dir: Optional[Path],
        force:     bool,
    ) -> list[Image.Image]:
        if cache_dir is not None:
            key     = self._cache_key(cond_path, radius)
            sub     = Path(cache_dir) / key
            cached  = [sub / f"view_{i}.png" for i in range(6)]
            if not force and all(p.exists() for p in cached):
                print(f"[cache hit] {sub}")
                return [Image.open(p).convert("RGB") for p in cached]
            sub.mkdir(parents=True, exist_ok=True)

        print("[generate] running Zero123++ ...")
        views = self.generator.generate(cond_img)

        if cache_dir is not None:
            for i, v in enumerate(views):
                v.save(cached[i])
            print(f"[cache save] {sub}")
        return views

    def _cache_key(self, cond_path: Path, radius: float) -> str:
        h = hashlib.sha256(cond_path.read_bytes()).hexdigest()[:16]
        mid = self.generator.model_id.replace("/", "_")
        return f"{mid}__r{radius}__{h}"

    def _build_pair(
        self,
        view_idx: int,
        cond_img: Image.Image, view_img: Image.Image,
        mask_cond: np.ndarray, mask_view: np.ndarray,
        K: np.ndarray,
        R_c: np.ndarray, t_c: np.ndarray,
        R_v: np.ndarray, t_v: np.ndarray,
    ) -> PairResult:
        az  = float(AZIMUTHS_DEG[view_idx])
        el  = float(ELEVATIONS_DEG[view_idx])
        empty = np.array([], dtype=np.float32)

        # Matching
        m     = self.matcher.match(cond_img, view_img, min_conf=self.loftr_min_conf)
        kpts_a, kpts_b, conf = m["kpts_a"], m["kpts_b"], m["conf"]
        n     = len(kpts_a)

        if n < self.min_matches:
            return PairResult(
                view_idx=view_idx, azimuth=az, elevation=el, skipped=True,
                n_matches=n, n_object_matches=0, n_background_matches=0, n_mixed_matches=0,
                sampson_all=empty, sampson_object=empty,
                sampson_background=empty, sampson_mixed=empty,
                kpts_a=kpts_a, kpts_b=kpts_b, conf=conf,
                F=np.zeros((3, 3)),
            )

        # Epipolar
        F    = compute_F(K, R_c, t_c, K, R_v, t_v)
        sd   = sampson_distance(F, kpts_a, kpts_b)

        # Region split
        split = split_matches_by_region(kpts_a, kpts_b, mask_cond, mask_view)
        obj_idx = split["object_idx"]
        bg_idx  = split["background_idx"]
        mx_idx  = split["mixed_idx"]

        return PairResult(
            view_idx=view_idx, azimuth=az, elevation=el, skipped=False,
            n_matches=n,
            n_object_matches=len(obj_idx),
            n_background_matches=len(bg_idx),
            n_mixed_matches=len(mx_idx),
            sampson_all=sd,
            sampson_object=sd[obj_idx],
            sampson_background=sd[bg_idx],
            sampson_mixed=sd[mx_idx],
            kpts_a=kpts_a, kpts_b=kpts_b, conf=conf,
            F=F,
        )
