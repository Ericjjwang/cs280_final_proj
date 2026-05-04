"""
Dry-run: init scene, append goblet, normalise, check poses.
Stops before any render.  Run with:
    blender --background --python scripts/_sanity_only.py -- brass_goblet_01
"""
import sys, math
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector, Matrix

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))
from camera import get_condition_pose, get_zero123plus_poses, get_intrinsics

_FLIP = np.diag([1.0, -1.0, -1.0])
OBJECT_NAME = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else "brass_goblet_01"
BLEND_PATH  = Path("model/brass_goblets_4k.blend").resolve()
IMG_SIZE    = 512

def _log(msg): print(msg, file=sys.stderr, flush=True)

# ── 1. Init ───────────────────────────────────────────────────────────────────
_log("[1] init scene …")
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.render.resolution_x = scene.render.resolution_y = IMG_SIZE
scene.unit_settings.system = 'METRIC'
scene.unit_settings.scale_length = 1.0

# ── 2. Append ─────────────────────────────────────────────────────────────────
_log(f"[2] appending '{OBJECT_NAME}' from {BLEND_PATH} …")
bpy.ops.wm.append(
    filepath=str(BLEND_PATH) + "/Object/" + OBJECT_NAME,
    directory=str(BLEND_PATH) + "/Object/",
    filename=OBJECT_NAME,
    link=False,
)
obj = bpy.data.objects.get(OBJECT_NAME)
if obj is None:
    raise RuntimeError(f"Object '{OBJECT_NAME}' not in scene after append.")
_log(f"[2] OK  materials={[s.material.name if s.material else 'None' for s in obj.material_slots]}")

# ── 3. Normalise ──────────────────────────────────────────────────────────────
corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
xs=[c[0] for c in corners]; ys=[c[1] for c in corners]; zs=[c[2] for c in corners]
max_dim = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))
_log(f"[3] original max_dim = {max_dim:.4f}")
obj.scale = (1/max_dim,)*3
bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
bpy.ops.object.transform_apply(scale=True)
corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
xs=[c[0] for c in corners]; ys=[c[1] for c in corners]; zs=[c[2] for c in corners]
cx=(min(xs)+max(xs))/2; cy=(min(ys)+max(ys))/2; cz=(min(zs)+max(zs))/2
obj.location = (-cx, -cy, -cz)
bpy.ops.object.transform_apply(location=True)
_log(f"[3] after norm: cx={cx:.4f} cy={cy:.4f} cz={cz:.4f}  → centred ✓")

# ── 4. Camera ─────────────────────────────────────────────────────────────────
_log("[4] creating camera …")
cam_data = bpy.data.cameras.new("Cam")
cam_data.lens_unit = 'FOV'
cam_data.angle = math.radians(30.0)
cam_obj = bpy.data.objects.new("Cam", cam_data)
bpy.context.scene.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

# ── 5. Poses & sanity check ───────────────────────────────────────────────────
K = get_intrinsics(img_size=IMG_SIZE, fov_deg=30.0)
R_c, t_c = get_condition_pose(radius=1.5)
view_poses = get_zero123plus_poses(radius=1.5)
all_poses = {"cond": (R_c, t_c), **{f"view_{i}": (R,t) for i,(R,t) in enumerate(view_poses)}}

_log(f"[5] K =\n{K.round(2)}")
cx_k, cy_k = K[0,2], K[1,2]
tol = 10.0
for name,(R,t) in all_poses.items():
    p_cam = R @ np.zeros(3) + t
    p_img = K @ p_cam
    u,v = p_img[0]/p_img[2], p_img[1]/p_img[2]
    ok = abs(u-cx_k)<tol and abs(v-cy_k)<tol
    _log(f"[5] {name:>8}: origin→({u:.1f},{v:.1f})  {'✓' if ok else 'FAIL'}")
    assert ok, f"Projection FAIL for {name}"

    # Set camera and verify Blender matrix_world
    R_bl_w2c = _FLIP @ R
    t_bl_w2c = _FLIP @ t
    R_c2w = R_bl_w2c.T
    t_c2w = -R_c2w @ t_bl_w2c
    M = np.eye(4); M[:3,:3]=R_c2w; M[:3,3]=t_c2w
    cam_obj.matrix_world = Matrix(M.tolist())
    bpy.context.view_layer.update()
    cam_pos = np.array(cam_obj.matrix_world.translation)
    expected_pos = -R.T @ t
    pos_err = np.linalg.norm(cam_pos - expected_pos)
    _log(f"         cam_pos=({cam_pos[0]:.3f},{cam_pos[1]:.3f},{cam_pos[2]:.3f})  "
         f"expected=({expected_pos[0]:.3f},{expected_pos[1]:.3f},{expected_pos[2]:.3f})  "
         f"err={pos_err:.2e}")

_log("\n[OK] All sanity checks passed. Ready to render.")
