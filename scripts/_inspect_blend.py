"""Inspect a .blend file and print available objects/materials to stderr."""
import sys
from pathlib import Path
import bpy

if "--" in sys.argv:
    argv = sys.argv[sys.argv.index("--") + 1:]
else:
    argv = ["model/brass_goblets_4k.blend"]

blend = Path(argv[0]).resolve()
print(f"Inspecting: {blend}", file=sys.stderr)

with bpy.data.libraries.load(str(blend)) as (data_from, _):
    objects   = list(data_from.objects)
    meshes    = list(data_from.meshes)
    materials = list(data_from.materials)

print(f"  Objects:   {objects}",   file=sys.stderr)
print(f"  Meshes:    {meshes}",    file=sys.stderr)
print(f"  Materials: {materials}", file=sys.stderr)
