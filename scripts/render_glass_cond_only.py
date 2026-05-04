"""
Render a single cond-pose image of the brass goblet with a Glass BSDF material.

Reuses all scene setup from render_blender_scene_v2.py (goblet append, -90° X
rotation, normalize to 0.45 m, vertical ground plane at y=-0.50, lighting).
Only changes: Glass material instead of brass, single cond render, no
matte/depth/mask/poses outputs.

Usage:
    blender --background --python scripts/render_glass_cond_only.py -- \\
        --output-dir data/scene_glass_goblet_cond_only
"""

from __future__ import annotations
import sys
import argparse
import math
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector, Matrix

_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from camera import get_condition_pose, get_intrinsics

_FLIP      = np.diag([1.0, -1.0, -1.0])
BLEND_PATH = _PROJECT_ROOT / "model" / "brass_goblets_4k.blend"
WOOD_TEX   = _PROJECT_ROOT / "model" / "wood-2045379_640.jpg"

TARGET_DIM = 0.45
GROUND_Y   = -0.50
RADIUS     = 1.5
IMG_SIZE   = 512
SAMPLES    = 64


def _log(msg: str) -> None:
    print(f"[glass_cond] {msg}", file=sys.stderr, flush=True)


def _parse_args() -> argparse.Namespace:
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--roughness", type=float, default=0.0,
                   help="Glass roughness (0.0 = perfect; try 0.05 if invisible)")
    p.add_argument("--samples", type=int, default=SAMPLES)
    return p.parse_args(argv)


# ── Scene init ────────────────────────────────────────────────────────────────

def _setup_scene(samples: int) -> bpy.types.Scene:
    _log("Setting up scene …")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    scene.render.engine                    = 'CYCLES'
    scene.cycles.samples                   = samples
    scene.cycles.seed                      = 42
    scene.render.resolution_x              = IMG_SIZE
    scene.render.resolution_y              = IMG_SIZE
    scene.render.resolution_percentage     = 100
    scene.render.film_transparent          = True
    scene.render.use_file_extension        = False
    scene.render.use_compositing           = False
    scene.render.use_sequencer             = False
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode  = 'RGBA'
    scene.render.image_settings.color_depth = '16'
    scene.unit_settings.system             = 'METRIC'
    scene.unit_settings.scale_length       = 1.0

    scene.cycles.max_bounces          = 12
    scene.cycles.transmission_bounces = 12
    scene.cycles.glossy_bounces       = 8
    scene.cycles.diffuse_bounces      = 4

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
                    _log(f"GPU ({dtype}): {[d.name for d in gpu_devs if d.use]}")
                    return
            except Exception:
                continue
        _log("No GPU — CPU render")
    except Exception as e:
        _log(f"GPU skip: {e}")


# ── Object ────────────────────────────────────────────────────────────────────

def _append_goblet() -> bpy.types.Object:
    if not BLEND_PATH.exists():
        raise FileNotFoundError(f"Blend file not found: {BLEND_PATH}")

    with bpy.data.libraries.load(str(BLEND_PATH)) as (data_from, _):
        available = list(data_from.objects)
    _log(f"Objects in .blend: {available}")

    obj_name = "brass_goblet_01"
    if obj_name not in available:
        raise ValueError(f"'{obj_name}' not in blend: {available}")

    bpy.ops.wm.append(
        filepath=str(BLEND_PATH) + "/Object/" + obj_name,
        directory=str(BLEND_PATH) + "/Object/",
        filename=obj_name,
        link=False,
    )
    obj = bpy.data.objects.get(obj_name)
    if obj is None:
        raise RuntimeError(f"Append OK but '{obj_name}' missing from scene")

    # -90° X rotation: cup opening (originally +Z) → Blender +Y
    obj.rotation_euler = (-(math.pi / 2), 0.0, 0.0)
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    _log("Applied -90° X rotation (cup faces +Y)")
    return obj


def _normalize(obj: bpy.types.Object, target_dim: float) -> None:
    corners  = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [c[0] for c in corners]; ys = [c[1] for c in corners]; zs = [c[2] for c in corners]
    max_dim  = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))
    if max_dim < 1e-9:
        raise RuntimeError("Object has zero bounding box.")
    obj.scale = (target_dim / max_dim,) * 3
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [c[0] for c in corners]; ys = [c[1] for c in corners]; zs = [c[2] for c in corners]
    cx = (min(xs)+max(xs))/2; cy = (min(ys)+max(ys))/2; cz = (min(zs)+max(zs))/2
    obj.location = (-cx, -cy, -cz)
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

    corners  = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    final_dim = max(
        max(c[0] for c in corners)-min(c[0] for c in corners),
        max(c[1] for c in corners)-min(c[1] for c in corners),
        max(c[2] for c in corners)-min(c[2] for c in corners),
    )
    _log(f"Normalised: max_dim={final_dim:.4f} m")


def _apply_glass_material(obj: bpy.types.Object, roughness: float) -> None:
    mat = bpy.data.materials.new("GlassBSDF")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    glass = nodes.new('ShaderNodeBsdfGlass')
    glass.inputs['IOR'].default_value       = 1.5
    glass.inputs['Roughness'].default_value = roughness
    glass.inputs['Color'].default_value     = (1.0, 1.0, 1.0, 1.0)

    out = nodes.new('ShaderNodeOutputMaterial')
    links.new(glass.outputs['BSDF'], out.inputs['Surface'])

    obj.data.materials.clear()
    obj.data.materials.append(mat)
    _log(f"Applied Glass BSDF (IOR=1.5, roughness={roughness})")


# ── Ground plane ──────────────────────────────────────────────────────────────

def _add_vertical_ground(y: float) -> None:
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
    bsdf  = nodes.get("Principled BSDF") or nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Roughness'].default_value = 0.8
    bsdf.inputs['Metallic'].default_value  = 0.0

    if WOOD_TEX.exists():
        tex_coord = nodes.new('ShaderNodeTexCoord')
        mapping   = nodes.new('ShaderNodeMapping')
        mapping.inputs['Scale'].default_value = (4.0, 4.0, 4.0)
        tex_img   = nodes.new('ShaderNodeTexImage')
        tex_img.image = bpy.data.images.load(str(WOOD_TEX))
        links.new(tex_coord.outputs['UV'],   mapping.inputs['Vector'])
        links.new(mapping.outputs['Vector'], tex_img.inputs['Vector'])
        links.new(tex_img.outputs['Color'],  bsdf.inputs['Base Color'])
        _log(f"Ground plane at y={y:.2f} with wood texture")
    else:
        bsdf.inputs['Base Color'].default_value = (0.4, 0.3, 0.2, 1.0)
        _log(f"Ground plane at y={y:.2f} (brown fallback)")

    plane.data.materials.clear()
    plane.data.materials.append(mat)


# ── Lighting ──────────────────────────────────────────────────────────────────

def _setup_lighting() -> None:
    _log("Adding sun + area lights …")
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 5))
    sun = bpy.context.active_object
    sun.data.energy = 2.5
    sun.data.angle  = 0.05
    sun.rotation_euler = (math.radians(45), 0, math.radians(30))

    bpy.ops.object.light_add(type='AREA', location=(2.0, -2.0, 3.0))
    area = bpy.context.active_object
    area.data.energy = 150.0
    area.data.size   = 2.0
    from mathutils import Vector as V
    direction = V((0, 0, 0)) - V((2.0, -2.0, 3.0))
    area.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()


# ── Camera ────────────────────────────────────────────────────────────────────

def _opencv_to_blender(R_cv: np.ndarray, t_cv: np.ndarray) -> Matrix:
    R_bl = _FLIP @ R_cv
    t_bl = _FLIP @ t_cv
    Rc2w = R_bl.T
    tc2w = -Rc2w @ t_bl
    M    = np.eye(4); M[:3, :3] = Rc2w; M[:3, 3] = tc2w
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    out  = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    scene = _setup_scene(args.samples)
    _enable_gpu(scene)

    obj = _append_goblet()
    _normalize(obj, TARGET_DIM)
    _apply_glass_material(obj, args.roughness)

    _add_vertical_ground(GROUND_Y)
    _setup_lighting()

    K    = get_intrinsics(img_size=IMG_SIZE, fov_deg=30.0)
    R, t = get_condition_pose(radius=RADIUS)
    cam  = _make_camera(fov_deg=30.0)
    cam.matrix_world = _opencv_to_blender(R, t)
    bpy.context.view_layer.update()

    out_path = str(out / "cond.png")
    scene.render.filepath = out_path
    _log(f"Rendering cond → {out_path} …")
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    _log(f"Done in {time.time()-t0:.1f}s → {out_path}")


if __name__ == "__main__":
    main()
