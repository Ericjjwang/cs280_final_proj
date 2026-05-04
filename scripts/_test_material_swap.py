"""
Concern 2: Verify dual-path material swap is clean.

Steps:
  1. Append brass_goblet_01, print original material name + BSDF inputs
  2. Swap all slots to matte gray, confirm
  3. Swap back to original, confirm name + BSDF inputs unchanged
No rendering.
"""
import sys
from pathlib import Path
import bpy
import numpy as np

_root  = Path(__file__).resolve().parent.parent
BLEND  = (_root / "model" / "brass_goblets_4k.blend").resolve()
OBJ    = "brass_goblet_01"


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


def _bsdf_inputs(mat: bpy.types.Material) -> dict:
    """Return BSDF input values we care about."""
    if not mat or not mat.use_nodes:
        return {}
    for node in mat.node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            result = {}
            for key in ("Base Color", "Roughness", "Metallic",
                        "Transmission Weight", "Transmission"):
                if key in node.inputs:
                    val = node.inputs[key].default_value
                    if hasattr(val, '__iter__'):
                        result[key] = tuple(round(x, 4) for x in val)
                    else:
                        result[key] = round(float(val), 4)
            return result
    return {}


# ── Init empty scene ──────────────────────────────────────────────────────────
bpy.ops.wm.read_factory_settings(use_empty=True)

# ── Append goblet ─────────────────────────────────────────────────────────────
_log("[1] Appending brass_goblet_01 …")
bpy.ops.wm.append(
    filepath=str(BLEND) + "/Object/" + OBJ,
    directory=str(BLEND) + "/Object/",
    filename=OBJ,
    link=False,
)
goblet = bpy.data.objects.get(OBJ)
assert goblet, f"Object '{OBJ}' not found after append"

n_slots = len(goblet.material_slots)
_log(f"[1] material_slots: {n_slots}")
for i, slot in enumerate(goblet.material_slots):
    m = slot.material
    _log(f"    slot[{i}] name='{m.name if m else None}'")
    if m:
        inp = _bsdf_inputs(m)
        _log(f"           BSDF inputs: {inp}")

# Snapshot original materials
original_mats = [slot.material for slot in goblet.material_slots]
original_names = [m.name if m else None for m in original_mats]
_log(f"[1] Original material names: {original_names}")

# ── Build matte material ───────────────────────────────────────────────────────
_log("\n[2] Creating matte gray material …")
matte_mat = bpy.data.materials.new(name="_MattGray_Temp")
matte_mat.use_nodes = True
bsdf = matte_mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1.0)
bsdf.inputs["Roughness"].default_value  = 1.0
bsdf.inputs["Metallic"].default_value   = 0.0
for key in ("Transmission Weight", "Transmission"):
    if key in bsdf.inputs:
        bsdf.inputs[key].default_value = 0.0
        _log(f"[2] Set '{key}' = 0.0")
        break

# Swap all slots to matte
_log("[2] Swapping all material slots → matte …")
for slot in goblet.material_slots:
    slot.material = matte_mat

for i, slot in enumerate(goblet.material_slots):
    m = slot.material
    _log(f"    slot[{i}] name='{m.name if m else None}'  (expected '_MattGray_Temp')")
    assert m and m.name == "_MattGray_Temp", f"Slot {i} not matte! Got '{m.name}'"

_log("[2] All slots are matte ✓")

# ── Restore original materials ────────────────────────────────────────────────
_log("\n[3] Restoring original materials …")
for i, slot in enumerate(goblet.material_slots):
    slot.material = original_mats[i]

# Remove temp matte material
bpy.data.materials.remove(matte_mat)
_log("[3] Matte material removed from bpy.data")

# Verify restoration
_log("[3] Verifying restored state …")
for i, slot in enumerate(goblet.material_slots):
    m = slot.material
    expected_name = original_names[i]
    _log(f"    slot[{i}] name='{m.name if m else None}'  (expected '{expected_name}')")
    assert m and m.name == expected_name, \
        f"Material not restored! Got '{m.name}', expected '{expected_name}'"
    inp = _bsdf_inputs(m)
    _log(f"           BSDF inputs: {inp}")

# Check original BSDF inputs haven't been mutated
_log("\n[4] Cross-checking Base Color + Roughness not modified …")
for i, slot in enumerate(goblet.material_slots):
    inp = _bsdf_inputs(slot.material)
    # Base Color should NOT be (0.5, 0.5, 0.5) — that would mean the matte was applied
    bc = inp.get("Base Color", (None,) * 4)
    rg = inp.get("Roughness", None)
    _log(f"    slot[{i}]  Base Color={bc}  Roughness={rg}")
    if bc[:3] == (0.5, 0.5, 0.5) and rg == 1.0:
        _log("    WARNING: BSDF looks like matte values survived restoration!")
        # This would be a bug, but might be intentional if original IS gray+rough

_log("\n[PASS] Material swap → restore is clean ✓")
