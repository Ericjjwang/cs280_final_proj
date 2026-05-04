"""
Render a dual-path Blender scene (refractive + matte) for the transparent-object
epipolar experiment.

Usage
-----
    blender --background --python scripts/render_blender_scene.py -- \\
        --asset model/brass_goblets_4k.blend \\
        --object-name brass_goblet_01 \\
        --output-dir data/scene_brass_goblet \\
        [--radius 1.5] [--img-size 512] [--samples 256]

    The '--' separates Blender arguments from script arguments.

Output layout
-------------
    <output-dir>/
      rgb/    cond.png  view_0..5.png   512×512 RGBA, refractive material
      matte/  cond.png  view_0..5.png   512×512 RGBA, gray matte material
      mask/   cond.png  view_0..5.png   512×512 BW, binary {0,255}
      depth/  cond.exr  view_0..5.exr   float32, metres, background = 1e10→inf
      poses.json                         W2C OpenCV convention, K for img-size
"""

from __future__ import annotations
import sys
import os
import json
import argparse
import shutil
import subprocess
import math
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector, Matrix

# ── Project root on sys.path (for src.camera) ─────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
# Add src/ directly — avoids triggering src/__init__.py which imports PIL/torch
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from camera import get_condition_pose, get_zero123plus_poses, get_intrinsics

# Flips OpenCV camera frame (+Y down, +Z fwd) ↔ Blender frame (+Y up, -Z fwd)
_FLIP = np.diag([1.0, -1.0, -1.0])

VIEW_NAMES = ["cond"] + [f"view_{i}" for i in range(6)]


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser(description="Blender dual-path scene renderer.")
    p.add_argument("--asset",        required=True, help="Path to .blend file.")
    p.add_argument("--object-name",  required=True, help="Mesh object name inside .blend.")
    p.add_argument("--output-dir",   required=True, help="Root of the output scene directory.")
    p.add_argument("--radius",       type=float, default=1.5, help="Camera orbit radius (m).")
    p.add_argument("--img-size",     type=int,   default=512, help="Square render resolution.")
    p.add_argument("--samples",      type=int,   default=256, help="Cycles sample count.")
    return p.parse_args(argv)


# ── Step 1 — Scene init ───────────────────────────────────────────────────────

def _setup_scene(img_size: int, samples: int) -> bpy.types.Scene:
    _log("1/9", "Initialising empty scene …")
    bpy.ops.wm.read_factory_settings(use_empty=True)

    scene = bpy.context.scene
    scene.render.engine              = 'CYCLES'
    scene.cycles.samples             = samples
    scene.cycles.seed                = 42
    scene.render.resolution_x        = img_size
    scene.render.resolution_y        = img_size
    scene.render.resolution_percentage = 100
    scene.render.film_transparent    = True
    scene.render.use_file_extension  = False   # we supply full path with extension
    scene.render.use_compositing     = True
    scene.render.use_sequencer       = False

    scene.render.image_settings.file_format  = 'PNG'
    scene.render.image_settings.color_mode   = 'RGBA'
    scene.render.image_settings.color_depth  = '16'

    scene.unit_settings.system       = 'METRIC'
    scene.unit_settings.scale_length = 1.0

    # Pure black environment (no HDRI / sky)
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background") or \
         world.node_tree.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value    = (0.0, 0.0, 0.0, 1.0)
    bg.inputs["Strength"].default_value = 0.0

    # Bounces tuned for metallic/reflective objects (not glass, so 8 is enough)
    scene.cycles.max_bounces          = 8
    scene.cycles.transmission_bounces = 8
    scene.cycles.glossy_bounces       = 8
    scene.cycles.diffuse_bounces      = 4

    return scene


def _enable_gpu(scene: bpy.types.Scene) -> None:
    """Activate OPTIX → CUDA → HIP in priority order; log result."""
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        for device_type in ('OPTIX', 'CUDA', 'HIP', 'METAL'):
            try:
                prefs.compute_device_type = device_type
                prefs.refresh_devices()
                gpu_devs = [d for d in prefs.devices if d.type != 'CPU']
                if gpu_devs:
                    for d in prefs.devices:
                        d.use = (d.type != 'CPU')
                    scene.cycles.device = 'GPU'
                    names = [d.name for d in gpu_devs if d.use]
                    _log("1/9", f"  GPU ({device_type}) activated: {names}")
                    return
            except Exception:
                continue
        _log("1/9", "  No GPU found — rendering on CPU")
    except Exception as e:
        _log("1/9", f"  GPU activation skipped: {e}")


# ── Step 2 — Asset inspection & append ───────────────────────────────────────

def _list_blend_objects(blend_path: Path) -> list[str]:
    with bpy.data.libraries.load(str(blend_path)) as (data_from, _):
        return list(data_from.objects)


def _append_goblet(blend_path: Path, object_name: str) -> bpy.types.Object:
    available = _list_blend_objects(blend_path)
    _log("2/9", f"Available objects in .blend: {available}")

    if object_name not in available:
        raise ValueError(
            f"Object '{object_name}' not found in {blend_path}.\n"
            f"Available objects: {available}"
        )

    bpy.ops.wm.append(
        filepath=str(blend_path) + "/Object/" + object_name,
        directory=str(blend_path) + "/Object/",
        filename=object_name,
        link=False,
    )

    obj = bpy.data.objects.get(object_name)
    if obj is None:
        raise RuntimeError(
            f"bpy.ops.wm.append reported success but '{object_name}' "
            "is not in bpy.data.objects."
        )

    mat_names = [s.material.name if s.material else "None"
                 for s in obj.material_slots]
    _log("2/9", f"  Appended '{object_name}'  material_slots={mat_names}")
    return obj


# ── Step 3 — Normalise ────────────────────────────────────────────────────────

def _normalize_goblet(obj: bpy.types.Object) -> None:
    # World-space bounding-box corners
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    zs = [c[2] for c in corners]
    max_dim = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    if max_dim < 1e-9:
        raise RuntimeError("Goblet bounding box is degenerate (zero size).")
    _log("3/9", f"bbox max_dim = {max_dim:.4f} m  →  scaling to 1.0 m")

    # Apply uniform scale
    scale = 1.0 / max_dim
    obj.scale = (scale, scale, scale)
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # Recompute centre and translate to origin
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    zs = [c[2] for c in corners]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    cz = (min(zs) + max(zs)) / 2.0
    obj.location = (-cx, -cy, -cz)
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

    new_dim = max(
        max(c[0] for c in [obj.matrix_world @ Vector(v) for v in obj.bound_box]) -
        min(c[0] for c in [obj.matrix_world @ Vector(v) for v in obj.bound_box]),
        max(c[1] for c in [obj.matrix_world @ Vector(v) for v in obj.bound_box]) -
        min(c[1] for c in [obj.matrix_world @ Vector(v) for v in obj.bound_box]),
        max(c[2] for c in [obj.matrix_world @ Vector(v) for v in obj.bound_box]) -
        min(c[2] for c in [obj.matrix_world @ Vector(v) for v in obj.bound_box]),
    )
    _log("3/9", f"  Done.  Verified new max_dim = {new_dim:.4f} m ✓")


# ── Step 4 — Lighting ─────────────────────────────────────────────────────────

def _setup_lighting() -> None:
    _log("4/9", "Adding sun + area lights …")
    # Sun: hard directional light for specular highlights on brass
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 5))
    sun = bpy.context.active_object
    sun.data.energy = 3.0
    sun.data.angle  = 0.1                    # radians (narrow = crisp)
    sun.rotation_euler = (
        math.radians(45), 0, math.radians(30)
    )

    # Area: soft fill so underside isn't completely black
    bpy.ops.object.light_add(type='AREA', location=(2.0, -2.0, 3.0))
    area = bpy.context.active_object
    area.data.energy = 200.0
    area.data.size   = 2.0
    direction = Vector((0, 0, 0)) - Vector((2.0, -2.0, 3.0))
    area.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()


# ── Step 5 — Camera & sanity check ───────────────────────────────────────────

def _opencv_to_blender_matrix(R_cv: np.ndarray, t_cv: np.ndarray) -> Matrix:
    """
    Convert OpenCV W2C (R, t) → Blender 4×4 C2W matrix_world.

    OpenCV camera: +X right, +Y down, +Z forward.
    Blender camera: +X right, +Y up,   -Z forward.
    Flip = diag(1, -1, -1) converts between them.
    """
    R_bl_w2c = _FLIP @ R_cv
    t_bl_w2c = _FLIP @ t_cv
    R_c2w    = R_bl_w2c.T          # R is orthogonal → inv = T
    t_c2w    = -R_c2w @ t_bl_w2c  # camera position in world
    M        = np.eye(4)
    M[:3, :3] = R_c2w
    M[:3,  3] = t_c2w
    return Matrix(M.tolist())


def _make_camera(fov_deg: float) -> bpy.types.Object:
    cam_data = bpy.data.cameras.new("RenderCamera")
    cam_data.lens_unit  = 'FOV'
    cam_data.angle      = math.radians(fov_deg)   # horizontal FOV (square sensor → same vertical)
    cam_data.clip_start = 0.01
    cam_data.clip_end   = 1e10    # background Z → 1e10; load_blender_depth_exr converts to inf

    cam_obj = bpy.data.objects.new("RenderCamera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    return cam_obj


def _set_camera_pose(cam_obj: bpy.types.Object,
                     R_cv: np.ndarray, t_cv: np.ndarray) -> None:
    cam_obj.matrix_world = _opencv_to_blender_matrix(R_cv, t_cv)
    bpy.context.view_layer.update()


def _sanity_check_all_poses(
    all_poses: dict[str, tuple[np.ndarray, np.ndarray]],
    K: np.ndarray,
    img_size: int,
) -> None:
    _log("5/9", "Sanity check — project world origin to image plane …")
    cx, cy = K[0, 2], K[1, 2]
    tol = 10.0
    for name, (R, t) in all_poses.items():
        p_cam = R @ np.zeros(3) + t   # world origin in camera frame = t
        if p_cam[2] <= 0:
            raise AssertionError(
                f"[{name}] World origin is behind camera (z_cam={p_cam[2]:.4f}). "
                "Check pose convention."
            )
        p_img = K @ p_cam
        u = p_img[0] / p_img[2]
        v = p_img[1] / p_img[2]
        if abs(u - cx) > tol or abs(v - cy) > tol:
            raise AssertionError(
                f"[{name}] Origin projects to ({u:.1f}, {v:.1f}), "
                f"expected ({cx:.0f}±{tol:.0f}, {cy:.0f}±{tol:.0f}). "
                "Camera matrix conversion is wrong — aborting before any render."
            )
        _log("5/9", f"  {name:>8}: origin → ({u:.1f}, {v:.1f})  ✓")


# ── Step 6 — Compositor (Blender 5.x) ────────────────────────────────────────

def _build_compositor(tmp_dir: Path):
    """
    Build compositing_node_group for Blender 5.x.

    Blender 5.x removed scene.node_tree; the compositor lives in
    scene.compositing_node_group.  CompositorNodeOutputFile only supports
    OPEN_EXR_MULTILAYER, which OpenCV cannot read directly — so depth is
    captured as a temp multilayer EXR and converted post-render via
    _parse_blender_multilayer_exr() + _save_depth_exr().

    Topology
    --------
    Render Layers ──► Depth ──► FileOutput (OPEN_EXR_MULTILAYER, tmp/)
    """
    _log("6/9", "Building compositor node tree (Blender 5.x) …")
    scene = bpy.context.scene
    scene.render.use_compositing = True

    # Clear any existing group
    old_ng = scene.compositing_node_group
    if old_ng:
        scene.compositing_node_group = None
        bpy.data.node_groups.remove(old_ng)

    cng = bpy.data.node_groups.new("Compositor", 'CompositorNodeTree')
    scene.compositing_node_group = cng

    bpy.context.view_layer.use_pass_z = True

    rl = cng.nodes.new('CompositorNodeRLayers')

    fo_depth = cng.nodes.new('CompositorNodeOutputFile')
    fo_depth.name = "DepthOutput"
    fo_depth.format.file_format = 'OPEN_EXR_MULTILAYER'
    fo_depth.format.color_depth = '32'
    fo_depth.file_output_items.new(socket_type='FLOAT', name='Z')
    cng.links.new(rl.outputs['Depth'], fo_depth.inputs['Z'])

    return fo_depth


# ── Depth / mask post-processing ──────────────────────────────────────────────

def _parse_blender_multilayer_exr(path: Path) -> np.ndarray:
    """
    Parse the uncompressed single-channel float32 multilayer EXR that
    Blender 5.x compositor writes.  Returns a (H, W) float32 array in
    Blender scanline order (y=0 is the bottom of the render).
    """
    import struct as _struct
    with open(path, 'rb') as f:
        data = f.read()

    assert _struct.unpack_from('<I', data, 0)[0] == 0x01312F76, \
        f"Not an EXR file: {path}"

    pos = 8
    width = height = None
    channels: dict = {}

    while pos < len(data):
        name_end = data.index(b'\x00', pos)
        name = data[pos:name_end].decode('ascii', errors='replace')
        pos = name_end + 1
        if name == '':
            break
        type_end = data.index(b'\x00', pos)
        pos = type_end + 1
        size = _struct.unpack_from('<i', data, pos)[0]; pos += 4
        value = data[pos:pos + size]; pos += size

        if name == 'dataWindow':
            x1, y1, x2, y2 = _struct.unpack_from('<4i', value)
            width = x2 - x1 + 1
            height = y2 - y1 + 1
        elif name == 'channels':
            ci = 0
            while ci < len(value) - 1 and value[ci:ci+1] != b'\x00':
                ch_end = value.index(b'\x00', ci)
                ch_name = value[ci:ch_end].decode('ascii', errors='replace')
                ci = ch_end + 1
                ptype = _struct.unpack_from('<i', value, ci)[0]
                ci += 16
                channels[ch_name] = ptype

    offsets = _struct.unpack_from(f'<{height}Q', data, pos)
    ptype = list(channels.values())[0]
    bytes_per = 4 if ptype == 2 else 2  # 2=FLOAT32, 1=HALF

    result = np.zeros((height, width), dtype=np.float32)
    for _, offset in enumerate(offsets):
        sl_y    = _struct.unpack_from('<i', data, offset)[0]
        sl_size = _struct.unpack_from('<i', data, offset + 4)[0]
        sl_data = data[offset + 8: offset + 8 + sl_size]
        vals = np.frombuffer(sl_data[:width * bytes_per], dtype=np.float32).copy()
        result[sl_y, :] = vals

    return result


def _save_depth_exr(depth: np.ndarray, path: Path) -> None:
    """
    Write float32 depth (H×W, Blender bottom-up scanline order) as a
    standard OPEN_EXR that cv2.IMREAD_UNCHANGED can read.
    depth[sl_y] = bottom-most row first (Blender convention).
    """
    H, W = depth.shape
    img = bpy.data.images.new("_depth_tmp", width=W, height=H, float_buffer=True)
    img.colorspace_settings.name = 'Non-Color'
    pix = np.zeros((H, W, 4), dtype=np.float32)
    pix[:, :, 0] = depth   # R = depth (bottom-up — matches bpy pixel order)
    pix[:, :, 1] = depth   # G, B for cv2 multi-channel read (io_utils takes ch[0])
    pix[:, :, 2] = depth
    pix[:, :, 3] = 1.0
    img.pixels = pix.flatten().tolist()
    img.filepath_raw = str(path)
    img.file_format = 'OPEN_EXR'
    img.save()
    bpy.data.images.remove(img)


def _save_mask_png(rgba_path: Path, mask_path: Path) -> None:
    """
    Load the RGBA PNG rendered by Blender, threshold alpha at 0.5,
    and save a binary {0,255} grayscale PNG for load_mask().
    """
    img = bpy.data.images.load(str(rgba_path))
    W, H = img.size[0], img.size[1]
    pix = np.array(img.pixels[:], dtype=np.float32).reshape(H, W, 4)
    bpy.data.images.remove(img)

    alpha = pix[:, :, 3]
    mask_f = (alpha > 0.5).astype(np.float32)  # already in bpy bottom-up order

    mask_img = bpy.data.images.new("_mask_tmp", width=W, height=H, float_buffer=True)
    mask_img.colorspace_settings.name = 'Non-Color'
    pix_out = np.zeros((H, W, 4), dtype=np.float32)
    pix_out[:, :, 0] = mask_f
    pix_out[:, :, 1] = mask_f
    pix_out[:, :, 2] = mask_f
    pix_out[:, :, 3] = 1.0
    mask_img.pixels = pix_out.flatten().tolist()
    mask_img.filepath_raw = str(mask_path)
    mask_img.file_format = 'PNG'
    mask_img.save()
    bpy.data.images.remove(mask_img)


# ── Materials ─────────────────────────────────────────────────────────────────

def _make_matte_material() -> bpy.types.Material:
    mat = bpy.data.materials.new(name="_MattGray_Temp")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1.0)
    bsdf.inputs["Roughness"].default_value  = 1.0
    bsdf.inputs["Metallic"].default_value   = 0.0
    # Zero transmission — handle Blender 3.x ("Transmission") and 4.x ("Transmission Weight")
    for key in ("Transmission Weight", "Transmission"):
        if key in bsdf.inputs:
            bsdf.inputs[key].default_value = 0.0
            break
    return mat


# ── Render loop ───────────────────────────────────────────────────────────────

def _render_matte_path(
    cam_obj:    bpy.types.Object,
    all_poses:  dict[str, tuple[np.ndarray, np.ndarray]],
    matte_dir:  Path,
    depth_dir:  Path,
    mask_dir:   Path,
    fo_depth,
    tmp_dir:    Path,
) -> None:
    """
    Render all 7 views with matte material.
    For each view:
      - write_still=True → matte_dir/{name}.png  (RGBA)
      - compositor FileOutput → tmp_dir/depth_{name}  (multilayer EXR)
    Then post-process: parse depth EXR → standard EXR; extract alpha → mask PNG.
    """
    scene = bpy.context.scene
    matte_dir.mkdir(parents=True, exist_ok=True)

    for name in VIEW_NAMES:
        R, t = all_poses[name]
        _set_camera_pose(cam_obj, R, t)

        tmp_exr = tmp_dir / f"depth_{name}"
        fo_depth.directory = str(tmp_dir) + '/'
        fo_depth.file_name  = f"depth_{name}"

        scene.render.filepath = str(matte_dir / f"{name}.png")
        bpy.ops.render.render(write_still=True)

        # ── Depth EXR ──
        if tmp_exr.exists():
            depth = _parse_blender_multilayer_exr(tmp_exr)
            _save_depth_exr(depth, depth_dir / f"{name}.exr")
            tmp_exr.unlink()
        else:
            print(f"  WARNING: depth temp EXR not found: {tmp_exr}", file=sys.stderr)

        # ── Binary mask ──
        _save_mask_png(matte_dir / f"{name}.png", mask_dir / f"{name}.png")


def _render_refractive_path(
    cam_obj:   bpy.types.Object,
    all_poses: dict[str, tuple[np.ndarray, np.ndarray]],
    rgb_dir:   Path,
    fo_depth,
) -> None:
    """Render all 7 views with original (refractive) material → rgb_dir/{name}.png."""
    scene = bpy.context.scene
    rgb_dir.mkdir(parents=True, exist_ok=True)

    # Mute depth output so compositor doesn't write stale EXR files
    fo_depth.mute = True

    for name in VIEW_NAMES:
        R, t = all_poses[name]
        _set_camera_pose(cam_obj, R, t)
        scene.render.filepath = str(rgb_dir / f"{name}.png")
        bpy.ops.render.render(write_still=True)

    fo_depth.mute = False


# ── poses.json ────────────────────────────────────────────────────────────────

def _save_poses(path: Path,
                all_poses: dict[str, tuple[np.ndarray, np.ndarray]],
                K: np.ndarray) -> None:
    out = {
        name: {"R": R.tolist(), "t": t.tolist(), "K": K.tolist()}
        for name, (R, t) in all_poses.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


# ── Logging ───────────────────────────────────────────────────────────────────

def _log(step: str, msg: str) -> None:
    print(f"[{step}] {msg}", file=sys.stderr, flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args       = _parse_args()
    blend_path = Path(args.asset).resolve()
    output_dir = Path(args.output_dir).resolve()
    img_size   = args.img_size

    if not blend_path.exists():
        raise FileNotFoundError(f"Asset not found: {blend_path}")

    for sub in ("rgb", "matte", "mask", "depth"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    # 1. Init
    scene = _setup_scene(img_size, args.samples)
    _enable_gpu(scene)

    # 2. Load
    goblet = _append_goblet(blend_path, args.object_name)

    # 3. Normalise
    _normalize_goblet(goblet)

    # 4. Lights
    _setup_lighting()

    # 5. Poses & sanity check
    _log("5/9", "Computing Zero123++ camera poses …")
    K         = get_intrinsics(img_size=img_size, fov_deg=30.0)
    R_c, t_c  = get_condition_pose(radius=args.radius)
    view_poses = get_zero123plus_poses(radius=args.radius)
    all_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "cond": (R_c, t_c),
        **{f"view_{i}": (R, t) for i, (R, t) in enumerate(view_poses)},
    }
    cam_obj = _make_camera(fov_deg=30.0)
    _sanity_check_all_poses(all_poses, K, img_size)

    # 6. Compositor
    depth_dir = output_dir / "depth"
    mask_dir  = output_dir / "mask"
    tmp_dir   = output_dir / ".depth_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fo_depth = _build_compositor(tmp_dir)

    # 7. Matte path: matte/ + depth/ + mask/
    _log("7/9", "Rendering matte path (7 views → matte + depth + mask) …")
    original_mats = [slot.material for slot in goblet.material_slots]
    matte_mat     = _make_matte_material()
    for slot in goblet.material_slots:
        slot.material = matte_mat

    _render_matte_path(cam_obj, all_poses,
                       matte_dir=output_dir / "matte",
                       depth_dir=depth_dir,
                       mask_dir=mask_dir,
                       fo_depth=fo_depth,
                       tmp_dir=tmp_dir)

    for i, slot in enumerate(goblet.material_slots):
        slot.material = original_mats[i]
    bpy.data.materials.remove(matte_mat)
    if tmp_dir.exists():
        shutil.rmtree(str(tmp_dir), ignore_errors=True)

    # 8. Refractive path: original material → rgb/
    _log("8/9", "Rendering refractive path (7 views → rgb/) …")
    _render_refractive_path(cam_obj, all_poses,
                            rgb_dir=output_dir / "rgb",
                            fo_depth=fo_depth)

    # poses.json
    _save_poses(output_dir / "poses.json", all_poses, K)
    _log("8/9", "  poses.json written ✓")

    # 9. Verify (run with the project's uv environment, not Blender's Python)
    _log("9/9", "Running verify_scene_data.py …")
    verify_script = _PROJECT_ROOT / "scripts" / "verify_scene_data.py"
    # uv resolves to the project venv from _PROJECT_ROOT
    result = subprocess.run(
        ["uv", "run", "python3", str(verify_script), str(output_dir)],
        capture_output=True, text=True,
        cwd=str(_PROJECT_ROOT),
    )
    print(result.stdout, file=sys.stderr)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        _log("9/9", "FAILED — rendered files kept for inspection.")
        sys.exit(1)
    else:
        _log("9/9", "PASS ✓  All 14 checks passed.")


if __name__ == "__main__":
    main()
