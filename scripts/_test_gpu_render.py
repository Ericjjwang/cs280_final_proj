"""
Concern 3: Verify GPU is active and measure actual render time.

- Append brass_goblet_01 (refractive material, no swap)
- Render cond view at 512×512, samples=64
- Print device settings + timing
- If > 90 s: switch to OPTIX, set bounces, re-render
"""
import sys, math, time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))
from camera import get_condition_pose, get_intrinsics

_FLIP = np.diag([1.0, -1.0, -1.0])
BLEND = (_root / "model" / "brass_goblets_4k.blend").resolve()
OBJ   = "brass_goblet_01"
OUT   = Path("/tmp/gpu_test_cond.png")


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


def _print_device_info():
    scene = bpy.context.scene
    _log(f"  cycles.device               = {scene.cycles.device}")
    _log(f"  cycles.max_bounces          = {scene.cycles.max_bounces}")
    _log(f"  cycles.transmission_bounces = {scene.cycles.transmission_bounces}")
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        _log(f"  compute_device_type         = {prefs.compute_device_type}")
        for d in prefs.devices:
            _log(f"    device: {d.name!r:40s} type={d.type}  use={d.use}")
    except Exception as e:
        _log(f"  (could not read cycles prefs: {e})")


def _setup_scene(samples: int):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine            = 'CYCLES'
    scene.cycles.samples           = samples
    scene.cycles.seed              = 42
    scene.render.resolution_x      = 512
    scene.render.resolution_y      = 512
    scene.render.resolution_percentage = 100
    scene.render.film_transparent  = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode  = 'RGBA'
    scene.render.use_file_extension = False
    scene.unit_settings.system     = 'METRIC'
    scene.unit_settings.scale_length = 1.0

    world = bpy.data.worlds.new("W")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
    return scene


def _enable_gpu_optix(scene):
    """Try OPTIX → CUDA → fallback GPU."""
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
                    _log(f"  GPU enabled via {device_type} ✓")
                    return True
            except Exception:
                continue
        _log("  No GPU device found — staying on CPU")
        return False
    except Exception as e:
        _log(f"  GPU setup failed: {e}")
        return False


def _append_goblet():
    bpy.ops.wm.append(
        filepath=str(BLEND) + "/Object/" + OBJ,
        directory=str(BLEND) + "/Object/",
        filename=OBJ,
        link=False,
    )
    obj = bpy.data.objects.get(OBJ)
    assert obj, f"'{OBJ}' not in scene after append"
    return obj


def _normalize(obj):
    from mathutils import Vector
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs=[c[0] for c in corners]; ys=[c[1] for c in corners]; zs=[c[2] for c in corners]
    max_dim = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))
    obj.scale = (1/max_dim,)*3
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(scale=True)
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs=[c[0] for c in corners]; ys=[c[1] for c in corners]; zs=[c[2] for c in corners]
    cx=(min(xs)+max(xs))/2; cy=(min(ys)+max(ys))/2; cz=(min(zs)+max(zs))/2
    obj.location=(-cx,-cy,-cz)
    bpy.ops.object.transform_apply(location=True)


def _setup_lights():
    bpy.ops.object.light_add(type='SUN', location=(0,0,5))
    sun = bpy.context.active_object
    sun.data.energy = 3.0; sun.data.angle = 0.1
    sun.rotation_euler = (math.radians(45), 0, math.radians(30))
    from mathutils import Vector
    bpy.ops.object.light_add(type='AREA', location=(2,-2,3))
    area = bpy.context.active_object
    area.data.energy = 200.0; area.data.size = 2.0
    direction = Vector((0,0,0)) - Vector((2,-2,3))
    area.rotation_euler = direction.to_track_quat('-Z','Y').to_euler()


def _set_camera(scene):
    K    = get_intrinsics(img_size=512, fov_deg=30.0)
    R, t = get_condition_pose(radius=1.5)
    R_bl=_FLIP@R; t_bl=_FLIP@t; Rc2w=R_bl.T; tc2w=-Rc2w@t_bl
    M=np.eye(4); M[:3,:3]=Rc2w; M[:3,3]=tc2w
    cam_d = bpy.data.cameras.new("C")
    cam_d.lens_unit='FOV'; cam_d.angle=math.radians(30.0)
    cam_d.clip_start=0.01; cam_d.clip_end=1e10
    cam_o = bpy.data.objects.new("C", cam_d)
    scene.collection.objects.link(cam_o)
    scene.camera = cam_o
    cam_o.matrix_world = Matrix(M.tolist())
    bpy.context.view_layer.update()


def _timed_render(out_path: Path, label: str) -> float:
    bpy.context.scene.render.filepath = str(out_path)
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    elapsed = time.time() - t0
    _log(f"  [{label}] {elapsed:.1f} s  →  {out_path}")
    return elapsed


# ─────────────────────────────────────────────────────────────────────────────

SAMPLES = 64

_log("=" * 60)
_log(f"GPU render test — {SAMPLES} samples, 512×512, cond view")
_log("=" * 60)

scene = _setup_scene(SAMPLES)

_log("\n[A] Initial device state:")
_print_device_info()

# Try to enable GPU before anything else
_log("\n[B] Attempting GPU (OPTIX) activation …")
_enable_gpu_optix(scene)
_log("\n[B] Device state after GPU activation:")
_print_device_info()

# Scene setup
_log("\n[C] Setting up scene (goblet + lights + camera) …")
goblet = _append_goblet()
_normalize(goblet)
_setup_lights()
_set_camera(scene)

# Set bounces for reflective/refractive objects
scene.cycles.max_bounces          = 8
scene.cycles.transmission_bounces = 8
scene.cycles.glossy_bounces       = 8
scene.cycles.diffuse_bounces      = 4

_log("\n[D] Rendering cond view …")
elapsed = _timed_render(OUT, "cond 512×512 64spp")

_log(f"\n[E] Render time: {elapsed:.1f} s")
if elapsed > 90:
    _log("    > 90 s — GPU may not be active or OPTIX not working.")
elif elapsed > 30:
    _log("    Acceptable for CPU or slow GPU path.")
else:
    _log("    GPU is active and fast ✓")

_log(f"\n[F] Final device state:")
_print_device_info()
