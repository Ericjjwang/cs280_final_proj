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
from src.io_utils import load_blender_alpha_mask, load_poses_json


# ── Module-level private helpers ──────────────────────────────────────────────

def _make_cache_key(
    cond_path: Path, radius: float, model_id: str, label: str = ""
) -> str:
    h   = hashlib.sha256(cond_path.read_bytes()).hexdigest()[:16]
    mid = model_id.replace("/", "_")
    tag = f"__{label}" if label else ""
    return f"{mid}__r{radius}__{h}{tag}"


def _resize_mask_to(mask: np.ndarray, size: int) -> np.ndarray:
    """Resize binary mask (bool or uint8) to (size × size); returns uint8."""
    pil = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    return (np.array(pil.resize((size, size), Image.NEAREST)) > 127).astype(np.uint8)


def _az_el_from_pose(R: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    """Extract (azimuth_deg, elevation_deg) from World-to-Camera (R, t)."""
    pos    = -R.T @ t
    radius = float(np.linalg.norm(pos))
    el     = float(np.degrees(np.arcsin(np.clip(pos[1] / radius, -1.0, 1.0))))
    az     = float(np.degrees(np.arctan2(pos[0], pos[2])) % 360)
    return az, el


def _generate_or_load_views(
    cond_path: Path,
    cond_img:  Image.Image,
    generator: Zero123PlusPipeline,
    cache_dir: Optional[Path],
    radius:    float,
    label:     str,
) -> list:
    """Generate 6 novel views with a per-label cache key, or load from cache."""
    if cache_dir is not None:
        key    = _make_cache_key(cond_path, radius, generator.model_id, label)
        sub    = Path(cache_dir) / key
        cached = [sub / f"view_{i}.png" for i in range(6)]
        if all(p.exists() for p in cached):
            print(f"[cache hit] {sub}")
            return [Image.open(p).convert("RGB") for p in cached]
        sub.mkdir(parents=True, exist_ok=True)

    print(f"[generate] Zero123++ for {label!r} ...")
    views = generator.generate(cond_img)

    if cache_dir is not None:
        for i, v in enumerate(views):
            v.save(cached[i])
        print(f"[cache save] {sub}")
    return views


def _run_pair(
    view_idx:       int,
    az:             float,
    el:             float,
    cond_img:       Image.Image,
    view_img:       Image.Image,
    mask_cond:      np.ndarray,
    mask_view:      np.ndarray,
    K:              np.ndarray,
    R_c:            np.ndarray,
    t_c:            np.ndarray,
    R_v:            np.ndarray,
    t_v:            np.ndarray,
    matcher:        LoFTRMatcher,
    loftr_min_conf: float,
    min_matches:    int,
) -> PairResult:
    """Match → epipolar → region split → PairResult (shared by both class and dual-path)."""
    empty = np.array([], dtype=np.float32)
    m     = matcher.match(cond_img, view_img, min_conf=loftr_min_conf)
    kpts_a, kpts_b, conf = m["kpts_a"], m["kpts_b"], m["conf"]
    n = len(kpts_a)

    if n < min_matches:
        return PairResult(
            view_idx=view_idx, azimuth=az, elevation=el, skipped=True,
            n_matches=n, n_object_matches=0, n_background_matches=0, n_mixed_matches=0,
            sampson_all=empty, sampson_object=empty,
            sampson_background=empty, sampson_mixed=empty,
            kpts_a=kpts_a, kpts_b=kpts_b, conf=conf,
            F=np.zeros((3, 3)),
        )

    F   = compute_F(K, R_c, t_c, K, R_v, t_v)
    sd  = sampson_distance(F, kpts_a, kpts_b)

    split   = split_matches_by_region(kpts_a, kpts_b, mask_cond, mask_view)
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


@dataclass
class DualPathResult:
    """
    Side-by-side Sampson / region stats for the refractive and matte
    rendering paths of the same scene and view index.
    """
    scene_id:  str
    view_idx:  int
    azimuth:   float    # camera azimuth  (degrees, from pose)
    elevation: float    # camera elevation (degrees, from pose)

    refractive: PairResult = field(repr=False)
    matte:      PairResult = field(repr=False)

    # Object-region medians and ratio (core claim: refractive > matte on object)
    object_sampson_refractive_median: Optional[float]
    object_sampson_matte_median:      Optional[float]
    object_sampson_ratio:             Optional[float]   # refractive / matte

    # Background-region medians and ratio (sanity-check: should be ~1)
    background_sampson_refractive_median: Optional[float]
    background_sampson_matte_median:      Optional[float]
    background_sampson_ratio:             Optional[float]

    # Match counts per region × path
    n_object_matches_refractive:     int
    n_object_matches_matte:          int
    n_background_matches_refractive: int
    n_background_matches_matte:      int

    def to_json(self, include_arrays: bool = False) -> dict:
        return {
            "scene_id":  self.scene_id,
            "view_idx":  self.view_idx,
            "azimuth":   self.azimuth,
            "elevation": self.elevation,
            "refractive": self.refractive.to_dict(include_arrays),
            "matte":      self.matte.to_dict(include_arrays),
            "summary": {
                "object_sampson_refractive_median":      self.object_sampson_refractive_median,
                "object_sampson_matte_median":           self.object_sampson_matte_median,
                "object_sampson_ratio":                  self.object_sampson_ratio,
                "background_sampson_refractive_median":  self.background_sampson_refractive_median,
                "background_sampson_matte_median":       self.background_sampson_matte_median,
                "background_sampson_ratio":              self.background_sampson_ratio,
                "n_object_matches_refractive":           self.n_object_matches_refractive,
                "n_object_matches_matte":                self.n_object_matches_matte,
                "n_background_matches_refractive":       self.n_background_matches_refractive,
                "n_background_matches_matte":            self.n_background_matches_matte,
            },
        }

    def to_csv_rows(self) -> list[dict]:
        """One CSV row with all summary scalars for this (scene, view) pair."""
        return [{
            "scene_id":   self.scene_id,
            "view_idx":   self.view_idx,
            "azimuth":    self.azimuth,
            "elevation":  self.elevation,
            "n_matches_refractive":            self.refractive.n_matches,
            "n_matches_matte":                 self.matte.n_matches,
            "n_object_matches_refractive":     self.n_object_matches_refractive,
            "n_object_matches_matte":          self.n_object_matches_matte,
            "n_background_matches_refractive": self.n_background_matches_refractive,
            "n_background_matches_matte":      self.n_background_matches_matte,
            "object_sampson_refractive_median":      self.object_sampson_refractive_median,
            "object_sampson_matte_median":           self.object_sampson_matte_median,
            "object_sampson_ratio":                  self.object_sampson_ratio,
            "background_sampson_refractive_median":  self.background_sampson_refractive_median,
            "background_sampson_matte_median":       self.background_sampson_matte_median,
            "background_sampson_ratio":              self.background_sampson_ratio,
        }]


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
        return _make_cache_key(cond_path, radius, self.generator.model_id)

    def _build_pair(
        self,
        view_idx: int,
        cond_img: Image.Image, view_img: Image.Image,
        mask_cond: np.ndarray, mask_view: np.ndarray,
        K: np.ndarray,
        R_c: np.ndarray, t_c: np.ndarray,
        R_v: np.ndarray, t_v: np.ndarray,
    ) -> PairResult:
        return _run_pair(
            view_idx=view_idx,
            az=float(AZIMUTHS_DEG[view_idx]),
            el=float(ELEVATIONS_DEG[view_idx]),
            cond_img=cond_img, view_img=view_img,
            mask_cond=mask_cond, mask_view=mask_view,
            K=K, R_c=R_c, t_c=t_c, R_v=R_v, t_v=t_v,
            matcher=self.matcher,
            loftr_min_conf=self.loftr_min_conf,
            min_matches=self.min_matches,
        )


# ── Dual-path analysis ────────────────────────────────────────────────────────

def analyze_dual_path(
    scene_dir:        Path,
    view_idx:         int,
    generator:        Zero123PlusPipeline,
    matcher:          LoFTRMatcher,
    cache_dir:        Optional[Path]  = None,
    radius:           float           = 1.5,
    loftr_min_conf:   float           = 0.5,
    min_matches:      int             = 20,
    use_blender_pose: bool            = True,
) -> DualPathResult:
    """
    Compare refractive vs matte rendering for a single view of one scene.

    Both paths share the same Blender object/background masks and the same
    camera geometry.  Each path runs its own Zero123++ inference, cached
    independently under the ``rgb`` and ``matte`` labels so the two cache
    entries never collide even when the condition images are similar.

    Args:
        scene_dir:        Root of a verified Blender scene directory
                          (must contain rgb/, matte/, mask/, poses.json).
        view_idx:         Which novel view to analyse (0–5).
        generator:        Active Zero123PlusPipeline (model loaded, GPU ready).
        matcher:          LoFTRMatcher instance.
        cache_dir:        Shared cache root; sub-folders are keyed per image + label.
        radius:           Camera-to-origin distance assumed by Zero123++ (metres).
        loftr_min_conf:   LoFTR confidence threshold for valid matches.
        min_matches:      Minimum matches required; pairs below this are skipped.
        use_blender_pose: True  → R/t from poses.json (recommended).
                          False → Zero123++ nominal poses (sanity-check mode).

    Returns:
        DualPathResult with full per-region Sampson statistics for both paths.
    """
    scene_dir = Path(scene_dir)
    scene_id  = scene_dir.name

    rgb_cond_path   = scene_dir / "rgb"   / "cond.png"
    matte_cond_path = scene_dir / "matte" / "cond.png"

    # ── Load & resize cond images to 320×320 (LoFTR / Zero123++ resolution) ──
    def _load(p: Path) -> Image.Image:
        img = Image.open(p).convert("RGB")
        if img.size != (IMG_SIZE, IMG_SIZE):
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        return img

    rgb_cond_img   = _load(rgb_cond_path)
    matte_cond_img = _load(matte_cond_path)

    # ── Novel views — each path gets its own cache sub-folder ────────────────
    rgb_views   = _generate_or_load_views(
        rgb_cond_path,   rgb_cond_img,   generator, cache_dir, radius, label="rgb")
    matte_views = _generate_or_load_views(
        matte_cond_path, matte_cond_img, generator, cache_dir, radius, label="matte")

    rgb_view   = rgb_views[view_idx]
    matte_view = matte_views[view_idx]

    # ── Camera geometry ───────────────────────────────────────────────────────
    # K is always for the 320×320 matching resolution regardless of source.
    K = get_intrinsics(IMG_SIZE, FOV_DEG)

    if use_blender_pose:
        poses    = load_poses_json(scene_dir / "poses.json")
        R_c, t_c = poses["cond"]["R"],              poses["cond"]["t"]
        R_v, t_v = poses[f"view_{view_idx}"]["R"],  poses[f"view_{view_idx}"]["t"]
        az, el   = _az_el_from_pose(R_v, t_v)
    else:
        R_c, t_c = get_condition_pose(radius)
        R_v, t_v = get_zero123plus_poses(radius)[view_idx]
        az       = float(AZIMUTHS_DEG[view_idx])
        el       = float(ELEVATIONS_DEG[view_idx])

    # ── Blender masks — resize Blender resolution → 320×320 ──────────────────
    mask_cond = _resize_mask_to(
        load_blender_alpha_mask(scene_dir / "mask" / "cond.png"), IMG_SIZE)
    mask_view = _resize_mask_to(
        load_blender_alpha_mask(scene_dir / "mask" / f"view_{view_idx}.png"), IMG_SIZE)

    # ── Build PairResult for each path (same pose / masks, different images) ──
    pair_kw = dict(
        view_idx=view_idx, az=az, el=el,
        mask_cond=mask_cond, mask_view=mask_view,
        K=K, R_c=R_c, t_c=t_c, R_v=R_v, t_v=t_v,
        matcher=matcher, loftr_min_conf=loftr_min_conf, min_matches=min_matches,
    )
    print(f"\n--- Refractive (view {view_idx}, az={az:.0f}°, el={el:.0f}°) ---")
    ref_pair = _run_pair(cond_img=rgb_cond_img,   view_img=rgb_view,   **pair_kw)
    print(f"--- Matte ---")
    mat_pair = _run_pair(cond_img=matte_cond_img, view_img=matte_view, **pair_kw)

    # ── Summary statistics ────────────────────────────────────────────────────
    def _med(arr: np.ndarray) -> Optional[float]:
        return float(np.median(arr)) if len(arr) > 0 else None

    def _ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None or b == 0.0:
            return None
        return a / b

    obj_r = _med(ref_pair.sampson_object)
    obj_m = _med(mat_pair.sampson_object)
    bg_r  = _med(ref_pair.sampson_background)
    bg_m  = _med(mat_pair.sampson_background)

    return DualPathResult(
        scene_id   = scene_id,
        view_idx   = view_idx,
        azimuth    = az,
        elevation  = el,
        refractive = ref_pair,
        matte      = mat_pair,
        object_sampson_refractive_median     = obj_r,
        object_sampson_matte_median          = obj_m,
        object_sampson_ratio                 = _ratio(obj_r, obj_m),
        background_sampson_refractive_median = bg_r,
        background_sampson_matte_median      = bg_m,
        background_sampson_ratio             = _ratio(bg_r, bg_m),
        n_object_matches_refractive          = ref_pair.n_object_matches,
        n_object_matches_matte               = mat_pair.n_object_matches,
        n_background_matches_refractive      = ref_pair.n_background_matches,
        n_background_matches_matte           = mat_pair.n_background_matches,
    )
