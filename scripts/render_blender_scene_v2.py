"""
Render Plan-B dual-path scene: main object on a wood-textured ground plane.

Scenes
------
  brass_goblet  – append brass_goblet_01 from model/brass_goblets_4k.blend,
                  rotate -90° around X so the body faces the camera
  glass_suzanne – bpy primitive monkey with Glass BSDF (IOR=1.5)

Output layout (same spec as v1, verified by verify_scene_data.py)
------------------------------------------------------------------
  <output-dir>/
    rgb/    cond.png  view_0..5.png   512×512 RGBA, original material
    matte/  cond.png  view_0..5.png   512×512 RGBA, gray matte material
    mask/   cond.png  view_0..5.png   512×512 BW binary {0,255} — MAIN OBJECT ONLY
    depth/  cond.exr  view_0..5.exr   float32 metres, bg=1e10 → inf
    poses.json                         W2C OpenCV convention

Usage
-----
    blender --background --python scripts/render_blender_scene_v2.py -- \\
        --scene-name brass_goblet \\
        --output-dir data/scene_brass_goblet_v2 \\
        --samples 64
"""

from __future__ import annotations
import sys
import os
import json
import argparse
import shutil
import subprocess
import math
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector, Matrix

_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from camera import get_condition_pose, get_zero123plus_poses, get_intrinsics

_FLIP      = np.diag([1.0, -1.0, -1.0])
VIEW_NAMES = ["cond"] + [f"view_{i}" for i in range(6)]

BLEND_PATH   = _PROJECT_ROOT / "model" / "brass_goblets_4k.blend"
WOOD_TEX     = _PROJECT_ROOT / "model" / "wood-2045379_640.jpg"
GROUND_EXTRA = 0.02  # ground plane sits this far below object bottom (m)

# Per-scene target bounding-box max dimension (metres).
# Goblet is rotated -90° around X so cup faces +Y; cond camera at (0,0,1.5)
# sees the full side profile. Side views show ~0.6 m fill.
TARGET_DIM_BY_SCENE = {
    "brass_goblet":  0.45,
    "glass_suzanne": 0.45,
}


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser(description="Blender Plan-B scene renderer v2.")
    p.add_argument("--scene-name",  required=True,
                   choices=["brass_goblet", "glass_suzanne"],
                   help="Which scene to render.")
    p.add_argument("--output-dir",  required=True)
    p.add_argument("--radius",      type=float, default=1.5)
    p.add_argument("--img-size",    type=int,   default=512)
    p.add_argument("--samples",     type=int,   default=64)
    return p.parse_args(argv)


# ── Logging ───────────────────────────────────────────────────────────────────

def _log(step: str, msg: str) -> None:
    print(f"[{step}] {msg}", file=sys.stderr, flush=True)


# ── Step 1 — Scene init ───────────────────────────────────────────────────────

def _setup_scene(img_size: int, samples: int) -> bpy.types.Scene:
    _log("1", "Initialising scene …")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    scene.render.engine              = 'CYCLES'
    scene.cycles.samples             = samples
    scene.cycles.seed                = 42
    scene.render.resolution_x        = img_size
    scene.render.resolution_y        = img_size
    scene.render.resolution_percentage = 100
    scene.render.film_transparent    = True
    scene.render.use_file_extension  = False
    scene.render.use_compositing     = True
    scene.render.use_sequencer       = False

    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode  = 'RGBA'
    scene.render.image_settings.color_depth = '16'

    scene.unit_settings.system       = 'METRIC'
    scene.unit_settings.scale_length = 1.0

    # Bounces for glass / metal
    scene.cycles.max_bounces          = 12
    scene.cycles.transmission_bounces = 12
    scene.cycles.glossy_bounces       = 8
    scene.cycles.diffuse_bounces      = 4

    # Gray world background (helps transparent objects show refraction)
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = (world.node_tree.nodes.get("Background") or
          world.node_tree.nodes.new("ShaderNodeBackground"))
    bg.inputs["Color"].default_value    = (0.2, 0.2, 0.2, 1.0)
    bg.inputs["Strength"].default_value = 1.0

    return scene


def _enable_gpu(scene: bpy.types.Scene) -> None:
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        for dtype in ('OPTIX', 'CUDA', 'HIP', 'METAL'):
            try:
                prefs.compute_device_type = dtype
                prefs.refresh_devices()
                gpu_devs = [d for d in prefs.devices if d.type != 'CPU']
                if gpu_devs:
                    for d in prefs.devices:
                        d.use = (d.type != 'CPU')
                    scene.cycles.device = 'GPU'
                    _log("1", f"  GPU ({dtype}) activated: "
                         f"{[d.name for d in gpu_devs if d.use]}")
                    return
            except Exception:
                continue
        _log("1", "  No GPU found — CPU render")
    except Exception as e:
        _log("1", f"  GPU skip: {e}")


# ── Step 2 — Main object ──────────────────────────────────────────────────────

def _add_brass_goblet() -> bpy.types.Object:
    """Append brass_goblet_01, rotate -90° around X so body faces camera."""
    if not BLEND_PATH.exists():
        raise FileNotFoundError(f"Blend file not found: {BLEND_PATH}")

    with bpy.data.libraries.load(str(BLEND_PATH)) as (data_from, _):
        available = list(data_from.objects)
    _log("2", f"Objects in .blend: {available}")

    obj_name = "brass_goblet_01"
    if obj_name not in available:
        raise ValueError(f"'{obj_name}' not in {BLEND_PATH}: {available}")

    bpy.ops.wm.append(
        filepath=str(BLEND_PATH) + "/Object/" + obj_name,
        directory=str(BLEND_PATH) + "/Object/",
        filename=obj_name,
        link=False,
    )
    obj = bpy.data.objects.get(obj_name)
    if obj is None:
        raise RuntimeError(f"Append succeeded but '{obj_name}' missing from scene")

    mats = [s.material.name if s.material else "None" for s in obj.material_slots]
    _log("2", f"  Appended '{obj_name}'  materials={mats}")

    # Rotate -90° around X so the cup opening (originally +Z) faces Blender +Y.
    # In camera.py's Y-up world, +Y is up; the cond camera at (0,0,1.5) then
    # sees the full side profile instead of looking straight down into the cup.
    obj.rotation_euler = (-(math.pi / 2), 0.0, 0.0)
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    _log("2", "  Applied -90° X rotation (cup now faces Blender +Y)")
    return obj


def _add_glass_suzanne() -> bpy.types.Object:
    """Create a Suzanne monkey with Glass BSDF."""
    bpy.ops.mesh.primitive_monkey_add(size=1.0, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = "GlassSuzanne"

    mat = bpy.data.materials.new("GlassBSDF")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    glass = nodes.new('ShaderNodeBsdfGlass')
    glass.inputs['IOR'].default_value       = 1.5
    glass.inputs['Roughness'].default_value = 0.0
    glass.inputs['Color'].default_value     = (1.0, 1.0, 1.0, 1.0)

    out = nodes.new('ShaderNodeOutputMaterial')
    links.new(glass.outputs['BSDF'], out.inputs['Surface'])

    obj.data.materials.clear()
    obj.data.materials.append(mat)
    _log("2", "  Created GlassSuzanne with Glass BSDF (IOR=1.5)")
    return obj


def _normalize_object(obj: bpy.types.Object, target_dim: float) -> tuple[float, float]:
    """Scale + centre obj so max bounding box dimension = target_dim. Returns (min_z, min_y)."""
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [c[0] for c in corners]; ys = [c[1] for c in corners]; zs = [c[2] for c in corners]
    max_dim = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))
    if max_dim < 1e-9:
        raise RuntimeError("Object has zero bounding box.")

    scale = target_dim / max_dim
    obj.scale = (scale, scale, scale)
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [c[0] for c in corners]; ys = [c[1] for c in corners]; zs = [c[2] for c in corners]
    cx = (min(xs)+max(xs))/2; cy = (min(ys)+max(ys))/2; cz = (min(zs)+max(zs))/2
    obj.location = (-cx, -cy, -cz)
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    new_min_z = min(c[2] for c in corners)
    new_min_y = min(c[1] for c in corners)
    new_max_dim = max(
        max(c[0] for c in corners)-min(c[0] for c in corners),
        max(c[1] for c in corners)-min(c[1] for c in corners),
        max(c[2] for c in corners)-min(c[2] for c in corners),
    )
    _log("2", f"  Normalised: max_dim={new_max_dim:.4f} m, min_z={new_min_z:.4f} m, min_y={new_min_y:.4f} m")
    return new_min_z, new_min_y


# ── Step 3 — Ground plane ─────────────────────────────────────────────────────

def _add_ground_plane(z: float) -> bpy.types.Object:
    """4×4 m horizontal plane at z with wood texture (or gray fallback)."""
    bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0, 0, z))
    plane = bpy.context.active_object
    plane.name = "GroundPlane"

    mat = bpy.data.materials.new("GroundMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    bsdf = nodes.get("Principled BSDF") or nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Roughness'].default_value  = 0.8
    bsdf.inputs['Metallic'].default_value   = 0.0

    if WOOD_TEX.exists():
        # Add UV mapping with tiling
        tex_coord = nodes.new('ShaderNodeTexCoord')
        mapping   = nodes.new('ShaderNodeMapping')
        mapping.inputs['Scale'].default_value = (4.0, 4.0, 4.0)
        tex_img   = nodes.new('ShaderNodeTexImage')
        img = bpy.data.images.load(str(WOOD_TEX))
        tex_img.image = img
        links.new(tex_coord.outputs['UV'],    mapping.inputs['Vector'])
        links.new(mapping.outputs['Vector'],  tex_img.inputs['Vector'])
        links.new(tex_img.outputs['Color'],   bsdf.inputs['Base Color'])
        _log("3", f"  Ground plane at z={z:.3f} with wood texture")
    else:
        bsdf.inputs['Base Color'].default_value = (0.4, 0.3, 0.2, 1.0)
        _log("3", f"  Ground plane at z={z:.3f} (no texture, using brown fallback)")

    plane.data.materials.clear()
    plane.data.materials.append(mat)
    return plane


def _add_vertical_ground_plane(y: float) -> bpy.types.Object:
    """8×8 m plane in XZ (normal=+Y) at y=y. Used when goblet is rotated so base faces -Y."""
    # Default plane is in XY (normal=+Z). Rx(-90°) maps normal +Z → +Y, giving XZ plane.
    bpy.ops.mesh.primitive_plane_add(
        size=8.0,
        location=(0.0, y, 0.0),
        rotation=(-math.pi / 2, 0.0, 0.0),
    )
    plane = bpy.context.active_object
    plane.name = "GroundPlane"

    mat = bpy.data.materials.new("GroundMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    bsdf = nodes.get("Principled BSDF") or nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Roughness'].default_value = 0.8
    bsdf.inputs['Metallic'].default_value  = 0.0

    if WOOD_TEX.exists():
        tex_coord = nodes.new('ShaderNodeTexCoord')
        mapping   = nodes.new('ShaderNodeMapping')
        mapping.inputs['Scale'].default_value = (4.0, 4.0, 4.0)
        tex_img   = nodes.new('ShaderNodeTexImage')
        img = bpy.data.images.load(str(WOOD_TEX))
        tex_img.image = img
        links.new(tex_coord.outputs['UV'],   mapping.inputs['Vector'])
        links.new(mapping.outputs['Vector'], tex_img.inputs['Vector'])
        links.new(tex_img.outputs['Color'],  bsdf.inputs['Base Color'])
        _log("3", f"  Vertical ground plane at y={y:.3f} with wood texture")
    else:
        bsdf.inputs['Base Color'].default_value = (0.4, 0.3, 0.2, 1.0)
        _log("3", f"  Vertical ground plane at y={y:.3f} (brown fallback)")

    plane.data.materials.clear()
    plane.data.materials.append(mat)
    return plane


# ── Step 4 — Lighting ─────────────────────────────────────────────────────────

def _setup_lighting() -> None:
    _log("4", "Adding sun + area lights …")
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 5))
    sun = bpy.context.active_object
    sun.data.energy = 2.5
    sun.data.angle  = 0.05
    sun.rotation_euler = (math.radians(45), 0, math.radians(30))

    bpy.ops.object.light_add(type='AREA', location=(2.0, -2.0, 3.0))
    area = bpy.context.active_object
    area.data.energy = 150.0
    area.data.size   = 2.0
    direction = Vector((0, 0, 0)) - Vector((2.0, -2.0, 3.0))
    area.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()


# ── Step 5 — Camera & sanity check ───────────────────────────────────────────

def _opencv_to_blender_matrix(R_cv: np.ndarray, t_cv: np.ndarray) -> Matrix:
    R_bl  = _FLIP @ R_cv
    t_bl  = _FLIP @ t_cv
    Rc2w  = R_bl.T
    tc2w  = -Rc2w @ t_bl
    M     = np.eye(4); M[:3, :3] = Rc2w; M[:3, 3] = tc2w
    return Matrix(M.tolist())


def _make_camera(fov_deg: float) -> bpy.types.Object:
    cam_d = bpy.data.cameras.new("RenderCamera")
    cam_d.lens_unit  = 'FOV'
    cam_d.angle      = math.radians(fov_deg)
    cam_d.clip_start = 0.01
    cam_d.clip_end   = 1e10
    cam_o = bpy.data.objects.new("RenderCamera", cam_d)
    bpy.context.scene.collection.objects.link(cam_o)
    bpy.context.scene.camera = cam_o
    return cam_o


def _set_camera_pose(cam: bpy.types.Object, R: np.ndarray, t: np.ndarray) -> None:
    cam.matrix_world = _opencv_to_blender_matrix(R, t)
    bpy.context.view_layer.update()


def _project_point(P_world: np.ndarray, R: np.ndarray, t: np.ndarray,
                   K: np.ndarray) -> tuple[float, float] | None:
    """Project a world point through camera (R,t) and K.  Returns (u,v) or None if behind."""
    p_cam = R @ P_world + t
    if p_cam[2] <= 0:
        return None
    p_img = K @ p_cam
    return float(p_img[0] / p_img[2]), float(p_img[1] / p_img[2])


def _sanity_check_poses(all_poses: dict, K: np.ndarray, img_size: int,
                        scene_name: str = "brass_goblet",
                        ground_z: float = -0.32,
                        ground_y: float = -0.32) -> None:
    _log("5", "Sanity check — project world origin to image plane …")
    cx, cy = K[0, 2], K[1, 2]
    for name, (R, t) in all_poses.items():
        p_cam = R @ np.zeros(3) + t
        if p_cam[2] <= 0:
            raise AssertionError(f"[{name}] Origin behind camera.")
        p_img = K @ p_cam
        u, v  = p_img[0] / p_img[2], p_img[1] / p_img[2]
        if abs(u - cx) > 10 or abs(v - cy) > 10:
            raise AssertionError(
                f"[{name}] Origin projects to ({u:.1f},{v:.1f}), "
                f"expected ({cx:.0f}±10, {cy:.0f}±10)."
            )
        _log("5", f"  {name:>8}: origin → ({u:.1f}, {v:.1f})  ✓")

    # ── Cond-camera sanity check (brass_goblet only) ──────────────────────────
    # After -90° X rotation, cup faces +Y. In camera.py Y-up world, +Y is up,
    # so cup projects to the upper half of the cond image, base to lower half.
    if scene_name == "brass_goblet":
        R_cond, t_cond = all_poses["cond"]
        cup_top  = np.array([0.0,  0.3, 0.0])
        cup_base = np.array([0.0, -0.3, 0.0])

        uv_top  = _project_point(cup_top,  R_cond, t_cond, K)
        uv_base = _project_point(cup_base, R_cond, t_cond, K)

        if uv_top is None or uv_base is None:
            raise AssertionError("[cond] Goblet rim or base is behind the cond camera!")

        u_top,  v_top  = uv_top
        u_base, v_base = uv_base
        half = img_size / 2.0

        if v_top >= half:
            raise AssertionError(
                f"[cond] Cup rim projects to v={v_top:.0f} (≥ {half:.0f}) — "
                f"goblet still appears top-down; check that -90° X rotation was applied."
            )
        if v_base <= half:
            raise AssertionError(
                f"[cond] Base projects to v={v_base:.0f} (≤ {half:.0f}) — "
                f"goblet base should be in lower half of image."
            )
        px_height = abs(v_base - v_top)
        if px_height < 100:
            raise AssertionError(
                f"[cond] Goblet image height {px_height:.0f}px < 100px — "
                f"can't see full goblet height."
            )
        _log("5", f"  cond: rim→v={v_top:.0f}  base→v={v_base:.0f}  height={px_height:.0f}px  ✓")

    # ── view_2 geometry check ─────────────────────────────────────────────────
    # World coords: numeric XYZ same in both camera.py Y-up and Blender world
    # (object centred at origin; only the axis labelled "up" differs).
    R2, t2 = all_poses["view_2"]
    if scene_name == "brass_goblet":
        goblet_top = np.array([0.0, 0.3, 0.0])   # cup rim in +Y after rotation
        ground_far = np.array([2.0, ground_y, 0.0])  # far point on vertical XZ plane
        top_label  = "goblet_top(0,0.3,0)"
        far_label  = f"ground_far(2,{ground_y:.2f},0)"
    else:
        goblet_top = np.array([0.0, 0.0, 0.3])   # top in +Z (original orientation)
        ground_far = np.array([0.0, 2.0, ground_z])
        top_label  = "goblet_top(0,0,0.3)"
        far_label  = f"ground_far(0,2,{ground_z:.2f})"

    uv = _project_point(goblet_top, R2, t2, K)
    if uv is None:
        raise AssertionError("[view_2] goblet_top is behind camera — geometry wrong!")
    u, v = uv
    if not (0 <= u <= img_size and 0 <= v <= img_size):
        raise AssertionError(
            f"[view_2] goblet_top projects out of image ({u:.0f},{v:.0f}) "
            f"— goblet will not be visible from view_2!")
    _log("5", f"  view_2 {top_label:35s} → ({u:.0f},{v:.0f})  ✓")

    uv2 = _project_point(ground_far, R2, t2, K)
    if uv2 is None:
        _log("5", f"  view_2 ground_far: behind camera (far side clipped — OK)")
    else:
        u2, v2 = uv2
        status = "✓" if (0 <= u2 <= img_size and 0 <= v2 <= img_size) else "outside (far side)"
        _log("5", f"  view_2 {far_label:35s} → ({u2:.0f},{v2:.0f})  {status}")


# ── Step 6 — Compositor (Blender 5.x) ────────────────────────────────────────

def _build_compositor(tmp_dir: Path) -> bpy.types.Node:
    """
    Build compositing_node_group for depth EXR capture.

    Only depth is captured via the compositor (multilayer EXR → parsed → standard EXR).
    Mask is extracted post-render from a ground-hidden alpha render.
    """
    _log("6", "Building compositor (Blender 5.x) …")
    scene = bpy.context.scene
    scene.render.use_compositing = True

    old_ng = scene.compositing_node_group
    if old_ng:
        scene.compositing_node_group = None
        bpy.data.node_groups.remove(old_ng)

    cng = bpy.data.node_groups.new("Compositor", 'CompositorNodeTree')
    scene.compositing_node_group = cng
    bpy.context.view_layer.use_pass_z = True

    rl = cng.nodes.new('CompositorNodeRLayers')
    fo  = cng.nodes.new('CompositorNodeOutputFile')
    fo.name = "DepthOutput"
    fo.format.file_format = 'OPEN_EXR_MULTILAYER'
    fo.format.color_depth = '32'
    fo.file_output_items.new(socket_type='FLOAT', name='Z')
    cng.links.new(rl.outputs['Depth'], fo.inputs['Z'])

    return fo


# ── EXR / mask helpers ────────────────────────────────────────────────────────

def _parse_blender_multilayer_exr(path: Path) -> np.ndarray:
    """Parse uncompressed single-channel float32 multilayer EXR → (H,W) float32."""
    import struct as _s
    with open(path, 'rb') as f:
        data = f.read()
    assert _s.unpack_from('<I', data, 0)[0] == 0x01312F76, "Not EXR"
    pos = 8; width = height = None; channels: dict = {}
    while pos < len(data):
        ne = data.index(b'\x00', pos); name = data[pos:ne].decode('ascii', errors='replace')
        pos = ne + 1
        if name == '': break
        te = data.index(b'\x00', pos); pos = te + 1
        size = _s.unpack_from('<i', data, pos)[0]; pos += 4
        val  = data[pos:pos+size]; pos += size
        if name == 'dataWindow':
            x1,y1,x2,y2 = _s.unpack_from('<4i', val)
            width = x2-x1+1; height = y2-y1+1
        elif name == 'channels':
            ci = 0
            while ci < len(val)-1 and val[ci:ci+1] != b'\x00':
                ce = val.index(b'\x00', ci)
                ch = val[ci:ce].decode('ascii', errors='replace'); ci = ce+1
                pt = _s.unpack_from('<i', val, ci)[0]; ci += 16
                channels[ch] = pt
    offsets = _s.unpack_from(f'<{height}Q', data, pos)
    pt  = list(channels.values())[0]; bpv = 4 if pt == 2 else 2
    res = np.zeros((height, width), dtype=np.float32)
    for _, off in enumerate(offsets):
        sy = _s.unpack_from('<i', data, off)[0]
        sz = _s.unpack_from('<i', data, off+4)[0]
        sd = data[off+8: off+8+sz]
        res[sy, :] = np.frombuffer(sd[:width*bpv], dtype=np.float32).copy()
    return res


def _save_depth_exr(depth: np.ndarray, path: Path) -> None:
    """Write float32 depth (Blender bottom-up order) as standard OPEN_EXR readable by cv2."""
    H, W = depth.shape
    img  = bpy.data.images.new("_depth_tmp", width=W, height=H, float_buffer=True)
    img.colorspace_settings.name = 'Non-Color'
    pix  = np.zeros((H, W, 4), dtype=np.float32)
    pix[:, :, 0] = depth; pix[:, :, 1] = depth; pix[:, :, 2] = depth; pix[:, :, 3] = 1.0
    img.pixels = pix.flatten().tolist()
    img.filepath_raw = str(path)
    img.file_format  = 'OPEN_EXR'
    img.save()
    bpy.data.images.remove(img)


def _save_mask_from_alpha(rgba_path: Path, mask_path: Path) -> None:
    """Load RGBA PNG, threshold alpha at 0.5 → binary BW PNG."""
    img  = bpy.data.images.load(str(rgba_path))
    W, H = img.size[0], img.size[1]
    pix  = np.array(img.pixels[:], dtype=np.float32).reshape(H, W, 4)
    bpy.data.images.remove(img)
    alpha   = pix[:, :, 3]
    mask_f  = (alpha > 0.5).astype(np.float32)
    mi = bpy.data.images.new("_mask_tmp", width=W, height=H, float_buffer=True)
    mi.colorspace_settings.name = 'Non-Color'
    po = np.zeros((H, W, 4), dtype=np.float32)
    po[:, :, 0] = mask_f; po[:, :, 1] = mask_f; po[:, :, 2] = mask_f; po[:, :, 3] = 1.0
    mi.pixels = po.flatten().tolist()
    mi.filepath_raw = str(mask_path)
    mi.file_format  = 'PNG'
    mi.save()
    bpy.data.images.remove(mi)


# ── Materials ─────────────────────────────────────────────────────────────────

def _make_matte_material() -> bpy.types.Material:
    # Procedural noise gives LoFTR enough texture to find keypoints on the matte pass.
    matte_mat = bpy.data.materials.new("Matte_Gray_Textured")
    matte_mat.use_nodes = True
    nodes = matte_mat.node_tree.nodes
    links = matte_mat.node_tree.links

    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 1.0
    bsdf.inputs["Metallic"].default_value  = 0.0
    for key in ("Transmission Weight", "Transmission"):
        if key in bsdf.inputs:
            bsdf.inputs[key].default_value = 0.0
            break

    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value  = 50.0
    noise.inputs["Detail"].default_value = 4.0

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.42, 0.42, 0.42, 1.0)
    ramp.color_ramp.elements[1].color = (0.58, 0.58, 0.58, 1.0)

    tex_coord = nodes.new("ShaderNodeTexCoord")
    links.new(tex_coord.outputs["Object"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"],        ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"],       bsdf.inputs["Base Color"])

    return matte_mat


# ── Render loops ──────────────────────────────────────────────────────────────

def _render_matte_path(
    cam: bpy.types.Object,
    all_poses: dict,
    matte_dir: Path,
    depth_dir: Path,
    mask_dir: Path,
    ground: bpy.types.Object,
    fo_depth,
    tmp_dir: Path,
) -> None:
    """
    For each view:
      1. Render object+ground → matte PNG + compositor depth EXR
      2. Hide ground, render object-only → extract alpha → mask PNG
    """
    scene = bpy.context.scene
    matte_dir.mkdir(parents=True, exist_ok=True)

    for name in VIEW_NAMES:
        R, t = all_poses[name]
        _set_camera_pose(cam, R, t)
        t0 = time.time()

        # ── Render 1: matte + depth ──────────────────────────────────────────
        tmp_exr = tmp_dir / f"depth_{name}"
        fo_depth.directory = str(tmp_dir) + '/'
        fo_depth.file_name  = f"depth_{name}"
        fo_depth.mute       = False

        scene.render.filepath = str(matte_dir / f"{name}.png")
        bpy.ops.render.render(write_still=True)

        if tmp_exr.exists():
            depth = _parse_blender_multilayer_exr(tmp_exr)
            _save_depth_exr(depth, depth_dir / f"{name}.exr")
            tmp_exr.unlink()
        else:
            _log("7", f"  WARNING: depth EXR missing for {name}")

        # ── Render 2: mask (object only, ground hidden) ──────────────────────
        fo_depth.mute          = True
        ground.hide_render     = True
        tmp_mask               = tmp_dir / f"mask_{name}.png"
        scene.render.filepath  = str(tmp_mask)
        bpy.ops.render.render(write_still=True)
        ground.hide_render     = False
        fo_depth.mute          = False

        _save_mask_from_alpha(tmp_mask, mask_dir / f"{name}.png")
        if tmp_mask.exists():
            tmp_mask.unlink()

        _log("7", f"  {name}: {time.time()-t0:.1f}s")


def _render_refractive_path(
    cam: bpy.types.Object,
    all_poses: dict,
    rgb_dir: Path,
    fo_depth,
) -> None:
    scene = bpy.context.scene
    rgb_dir.mkdir(parents=True, exist_ok=True)
    fo_depth.mute = True

    for name in VIEW_NAMES:
        R, t = all_poses[name]
        _set_camera_pose(cam, R, t)
        t0 = time.time()
        scene.render.filepath = str(rgb_dir / f"{name}.png")
        bpy.ops.render.render(write_still=True)
        _log("8", f"  {name}: {time.time()-t0:.1f}s")

    fo_depth.mute = False


# ── poses.json ────────────────────────────────────────────────────────────────

def _save_poses(path: Path, all_poses: dict, K: np.ndarray) -> None:
    out = {name: {"R": R.tolist(), "t": t.tolist(), "K": K.tolist()}
           for name, (R, t) in all_poses.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


# ── Sanity check: object ratio ────────────────────────────────────────────────

def _check_object_ratio(mask_path: Path, lo: float = 0.15, hi: float = 0.85,
                        view_name: str = "") -> float:
    """Load mask PNG (via bpy), return object pixel fraction."""
    img = bpy.data.images.load(str(mask_path))
    W, H = img.size[0], img.size[1]
    pix  = np.array(img.pixels[:], dtype=np.float32).reshape(H, W, 4)
    bpy.data.images.remove(img)
    ratio = float((pix[:, :, 0] > 0.5).sum()) / (H * W)
    tag   = f"[{view_name}]" if view_name else ""
    _log("7", f"  {tag} mask coverage: {ratio:.1%}")
    if ratio < 0.15:
        _log("7", f"  WARNING {tag}: coverage {ratio:.1%} < 15% — object barely visible!")
    elif not (lo <= ratio <= hi):
        _log("7", f"  WARNING {tag}: coverage {ratio:.1%} outside [{lo:.0%},{hi:.0%}]")
    return ratio


# ── Post-render pipeline ──────────────────────────────────────────────────────

def _run_verify(output_dir: Path) -> bool:
    _log("9", "Running verify_scene_data.py …")
    result = subprocess.run(
        ["uv", "run", "python3",
         str(_PROJECT_ROOT / "scripts" / "verify_scene_data.py"),
         str(output_dir)],
        capture_output=True, text=True,
        cwd=str(_PROJECT_ROOT),
    )
    print(result.stdout, file=sys.stderr)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        _log("9", "VERIFY FAILED")
        return False
    _log("9", "VERIFY PASS ✓")
    return True


def _run_dual_path(output_dir: Path) -> bool:
    exp_out = _PROJECT_ROOT / "outputs" / "dual_path" / output_dir.name
    _log("9", f"Running dual-path experiment → {exp_out} …")
    result = subprocess.run(
        ["uv", "run", "python3",
         str(_PROJECT_ROOT / "scripts" / "run_dual_path_experiment.py"),
         "--scene-dir",  str(output_dir),
         "--views",      "0,1,2,3,4,5",
         "--output-dir", str(exp_out)],
        capture_output=True, text=True,
        cwd=str(_PROJECT_ROOT),
        env={**os.environ, "OPENCV_IO_ENABLE_OPENEXR": "1"},
    )
    print(result.stdout, file=sys.stderr)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        _log("9", "Dual-path experiment FAILED (results kept)")
        return False
    _log("9", "Dual-path experiment DONE ✓")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args       = _parse_args()
    output_dir = Path(args.output_dir).resolve()

    for sub in ("rgb", "matte", "mask", "depth"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / ".depth_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Scene
    scene = _setup_scene(args.img_size, args.samples)
    _enable_gpu(scene)

    # 2. Main object
    _log("2", f"Loading scene: {args.scene_name}")
    if args.scene_name == "brass_goblet":
        main_obj = _add_brass_goblet()
    else:
        main_obj = _add_glass_suzanne()

    target_dim = TARGET_DIM_BY_SCENE[args.scene_name]
    min_z, min_y = _normalize_object(main_obj, target_dim)

    # 3. Ground plane
    # brass_goblet: cup faces +Y after -90° X rotation → base is at min_y → vertical XZ plane
    # glass_suzanne: normal horizontal plane at min_z
    if args.scene_name == "brass_goblet":
        # Hardcode y=-0.50: el=-10° cameras sit at Blender y≈-0.261; using min_y-GROUND_EXTRA
        # (≈-0.20) puts those cameras behind the plane, completely blocking view_1/3/5.
        # y=-0.50 keeps every camera on the front (+Y) side of the plane.
        ground_y = -0.50
        ground_z = -0.32  # unused but keep for sanity check signature
        ground = _add_vertical_ground_plane(y=ground_y)
    else:
        ground_z = min_z - GROUND_EXTRA
        ground_y = -0.32  # unused
        ground = _add_ground_plane(z=ground_z)

    # 4. Lighting
    _setup_lighting()

    # 5. Poses
    _log("5", "Computing Zero123++ camera poses …")
    K          = get_intrinsics(img_size=args.img_size, fov_deg=30.0)
    R_c, t_c   = get_condition_pose(radius=args.radius)
    view_poses = get_zero123plus_poses(radius=args.radius)
    all_poses: dict = {
        "cond": (R_c, t_c),
        **{f"view_{i}": (R, t) for i, (R, t) in enumerate(view_poses)},
    }
    cam = _make_camera(fov_deg=30.0)
    _sanity_check_poses(all_poses, K, args.img_size,
                        scene_name=args.scene_name,
                        ground_z=ground_z,
                        ground_y=ground_y)

    # 6. Compositor
    fo_depth = _build_compositor(tmp_dir)

    # 7. Matte path
    _log("7", f"Rendering matte path (7 views × 2 renders) …")
    original_mats = [s.material for s in main_obj.material_slots]
    matte_mat     = _make_matte_material()
    for s in main_obj.material_slots:
        s.material = matte_mat

    _render_matte_path(cam, all_poses,
                       matte_dir=output_dir / "matte",
                       depth_dir=output_dir / "depth",
                       mask_dir=output_dir / "mask",
                       ground=ground,
                       fo_depth=fo_depth,
                       tmp_dir=tmp_dir)

    for i, s in enumerate(main_obj.material_slots):
        s.material = original_mats[i]
    bpy.data.materials.remove(matte_mat)
    shutil.rmtree(str(tmp_dir), ignore_errors=True)

    # Sanity: per-view mask coverage
    _log("7", "Per-view mask coverage:")
    for vname in VIEW_NAMES:
        mask_p = output_dir / "mask" / f"{vname}.png"
        if mask_p.exists():
            ratio = _check_object_ratio(mask_p, lo=0.15, hi=0.85,
                                        view_name=vname)

    # 8. Refractive path
    _log("8", "Rendering refractive path (7 views) …")
    _render_refractive_path(cam, all_poses,
                            rgb_dir=output_dir / "rgb",
                            fo_depth=fo_depth)

    # Poses
    _save_poses(output_dir / "poses.json", all_poses, K)
    _log("8", "  poses.json written ✓")

    # 9. Verify + experiment
    verified = _run_verify(output_dir)
    if verified:
        _run_dual_path(output_dir)
    else:
        _log("9", "Skipping dual-path (verify failed)")

    _log("DONE", f"Scene '{args.scene_name}' complete → {output_dir}")


if __name__ == "__main__":
    main()
