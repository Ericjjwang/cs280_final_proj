"""
Concern 1: Prove Blender's render pipeline places world-origin at image centre.

- Red emission sphere at (0,0,0), cond camera, 64×64, samples=4
- Uses 'Raw' view transform to disable tone-mapping so pure red stays red
- Checks centroid within ±2 px of image centre
"""
import sys, math, time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))
from camera import get_condition_pose, get_intrinsics

_FLIP = np.diag([1.0, -1.0, -1.0])
IMG   = 64
OUT   = Path("/tmp/origin_test.png")


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


# ── Init ──────────────────────────────────────────────────────────────────────
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine            = 'CYCLES'
scene.cycles.samples           = 4
scene.cycles.seed              = 42
scene.render.resolution_x      = IMG
scene.render.resolution_y      = IMG
scene.render.resolution_percentage = 100
scene.render.film_transparent  = True
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode  = 'RGBA'
scene.render.image_settings.color_depth = '8'
scene.render.use_file_extension = False

# Disable tone-mapping so emission R=1.0 → PNG R=255 exactly
scene.view_settings.view_transform = 'Raw'

scene.unit_settings.system = 'METRIC'

# Black world, no sky
world = bpy.data.worlds.new("W")
scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0

# ── Red emission sphere at origin ─────────────────────────────────────────────
_log("[1] Creating red emission sphere (r=0.05 m) at (0,0,0) …")
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.05, location=(0, 0, 0), segments=32, ring_count=16)
sphere = bpy.context.active_object

mat = bpy.data.materials.new("RedEmit")
mat.use_nodes = True
mat.node_tree.nodes.clear()
emit = mat.node_tree.nodes.new("ShaderNodeEmission")
out  = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
emit.inputs["Color"].default_value    = (1.0, 0.0, 0.0, 1.0)
emit.inputs["Strength"].default_value = 1.0    # ≤1 avoids clipping before Raw → PNG
mat.node_tree.links.new(emit.outputs["Emission"], out.inputs["Surface"])
sphere.data.materials.append(mat)
_log("[1] Sphere + red emission material created ✓")

# ── Camera at cond pose ───────────────────────────────────────────────────────
_log("[2] Setting cond camera (az=0°, el=0°, r=1.5 m) …")
K    = get_intrinsics(img_size=IMG, fov_deg=30.0)
R, t = get_condition_pose(radius=1.5)

R_bl = _FLIP @ R; t_bl = _FLIP @ t
Rc2w = R_bl.T;    tc2w = -Rc2w @ t_bl
M = np.eye(4); M[:3,:3] = Rc2w; M[:3,3] = tc2w

cam_d = bpy.data.cameras.new("C")
cam_d.lens_unit  = 'FOV'
cam_d.angle      = math.radians(30.0)
cam_d.clip_start = 0.01
cam_d.clip_end   = 100.0
cam_o = bpy.data.objects.new("C", cam_d)
bpy.context.scene.collection.objects.link(cam_o)
bpy.context.scene.camera = cam_o
cam_o.matrix_world = Matrix(M.tolist())
bpy.context.view_layer.update()

cam_pos = np.array(cam_o.matrix_world.translation)
_log(f"[2] Camera world pos = {cam_pos.round(3)}  (expected [0, 0, 1.5])")

# ── Render ────────────────────────────────────────────────────────────────────
_log(f"[3] Rendering {IMG}×{IMG} …")
scene.render.filepath = str(OUT)
t0 = time.time()
bpy.ops.render.render(write_still=True)
_log(f"[3] Done in {time.time()-t0:.1f}s")

# ── Analyse ───────────────────────────────────────────────────────────────────
_log("[4] Loading rendered image …")
img_data = bpy.data.images.load(str(OUT))
w, h = img_data.size
pixels = np.array(img_data.pixels[:], dtype=np.float32).reshape(h, w, 4)
pixels = pixels[::-1]   # Blender bottom-up → top-down

r_ch = pixels[:, :, 0]
g_ch = pixels[:, :, 1]
b_ch = pixels[:, :, 2]

_log(f"    Image size: {w}×{h}")
_log(f"    R  max={r_ch.max():.3f}  mean={r_ch.mean():.4f}")
_log(f"    G  max={g_ch.max():.3f}  mean={g_ch.mean():.4f}")
_log(f"    B  max={b_ch.max():.3f}  mean={b_ch.mean():.4f}")

# Red pixels: R dominant, low G and B
red_mask = (r_ch > 0.5) & (g_ch < 0.3) & (b_ch < 0.3)
n_red = int(red_mask.sum())
_log(f"    Red pixels (R>0.5, G<0.3, B<0.3): {n_red}")

# Show a few actual pixel values near centre for debugging
mid = IMG // 2
patch = pixels[mid-3:mid+3, mid-3:mid+3, :3]
_log(f"    Centre 6×6 patch max per channel: R={patch[:,:,0].max():.3f} "
     f"G={patch[:,:,1].max():.3f} B={patch[:,:,2].max():.3f}")

if n_red == 0:
    _log("    Full channel maxima per row at mid-column:")
    for row in range(IMG):
        if r_ch[row, mid] > 0.1:
            _log(f"      row={row} RGBA=({r_ch[row,mid]:.3f},{g_ch[row,mid]:.3f},"
                 f"{b_ch[row,mid]:.3f},{pixels[row,mid,3]:.3f})")
    raise AssertionError(
        "No red pixels found. Check emission material / view_transform='Raw'."
    )

ys, xs = np.where(red_mask)
cx = float(xs.mean())
cy = float(ys.mean())
centre = IMG / 2.0   # 32.0
_log(f"    Centroid: ({cx:.2f}, {cy:.2f})")
_log(f"    Expected: ({centre:.1f}, {centre:.1f})")
_log(f"    Error: Δx={abs(cx-centre):.2f} px  Δy={abs(cy-centre):.2f} px")

tol = 2.0
if abs(cx - centre) > tol or abs(cy - centre) > tol:
    raise AssertionError(
        f"Centroid ({cx:.2f}, {cy:.2f}) > {tol} px from image centre. "
        "Camera-to-Blender matrix conversion is wrong."
    )

_log(f"\n[PASS] Blender renders world-origin at image centre ±{tol:.0f} px ✓")
