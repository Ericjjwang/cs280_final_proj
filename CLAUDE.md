# CLAUDE.md

This file gives Claude Code persistent context for this project. Read it before doing anything.

## Project: Geometry-Consistent Generative Data Augmentation

We are building a Sim2Real data augmentation pipeline for robotic perception of non-Lambertian materials (transparent glass, reflective metals). The output is a multi-view dataset of geometry-consistent photorealistic RGB images that can train downstream 6D pose estimators.

**Presentation deadline: 2026-05-04.** Today is 2026-04-30. We have 4 days. Optimize for working end-to-end > polished components.

## Team & Scope

- **Tian (me, this repo):** data generation side — Blender rendering, dataset structure, F-matrix precomputation, LoFTR-based epipolar evaluation.
- **Yujie (separate repo, Modal compute):** model side — multi-view diffusion with Epipolar Cross-View Attention, ControlNet conditioning on clean depth.
- **Interface between us:** `DATA_INTERFACE_SPEC.md` in this repo. Treat it as a contract. Do not change the on-disk format without updating that file and flagging it to me explicitly.

## My Existing Workflow (IMPORTANT — read before suggesting anything)

I am NOT starting from scratch. My current state:

- I have **Blender 4.x already installed and working**.
- I have a **`.blend` file with the scene already built by hand in the GUI**: assets imported, materials assigned, lighting set up, ground plane positioned. This took real effort and I'm not throwing it away.
- I have **a working Python script that exports camera poses and depth maps** from this `.blend` file. It's rough but functional.

**My direction of travel:** progressively scriptify the rest. Goal is eventually `blender --background --python render_all.py` runs the entire pipeline headlessly with no GUI, but I want to get there in stages, not in one rewrite.

**What this means for you (Claude Code):**
- **Do not propose rebuilding the scene from scratch in Python.** My `.blend` file is the source of truth for assets, materials, and lighting.
- **The pattern is: open the existing `.blend`, then script the variable parts** (camera placement, rendering, export). Use `bpy.ops.wm.open_mainfile(filepath=...)` at the top of every script.
- **Cameras are the first thing to scriptify**, because we need 8 of them at known poses and that's painful in the GUI. The existing manual cameras can be deleted/replaced by the script after loading the .blend.
- **Materials and lighting stay manual for now.** If we have time on Day 2/3, we can scriptify material swapping (glass ↔ matte gray for the dual-path render). Until then, dual-path means saving two copies of the .blend with different materials.
- **Reuse my existing export script as a starting point.** Ask me to paste it before writing a new one. Don't write a new exporter unless mine is broken.

## Conventions (non-negotiable)

- **Camera convention is OpenCV everywhere on disk.** Camera frame: +X right, +Y down, +Z forward. Image origin top-left, +u right, +v down.
- **Extrinsics stored as world-to-camera** `[R | t]`. `x_cam = R @ x_world + t`. If you need cam-to-world internally, invert it explicitly — never store cam-to-world to disk under the same name.
- **Blender's camera frame is +Y up, +Z backward.** Conversion: `R_cv = diag(1,-1,-1) @ R_blender`, `t_cv = diag(1,-1,-1) @ t_blender`. Apply this once, at write-time, in the rendering script. Downstream code never sees Blender convention.
- **Translations in meters.** Depth on disk in **millimeters** (uint16). Loader exposes meters as float32.
- **Image resolution: 512x512.** Square. Match diffusion U-Net input. Don't change without asking.
- **Default views per scene: N=8.**

## Repository Layout

```
.
├── CLAUDE.md                       # this file
├── DATA_INTERFACE_SPEC.md          # contract with Yujie's side
├── README.md
├── pyproject.toml
├── blend_files/
│   └── scene_base.blend            # my existing GUI-built scene (source of truth)
├── scripts/
│   ├── existing_export.py          # MY existing script — read it before replacing it
│   ├── render_scene.py             # main render entrypoint (loads .blend, places cameras, renders)
│   ├── precompute_F.py             # F-matrix precomputation (Day 2)
│   └── eval_loftr_epipolar.py      # LoFTR + epipolar metric (Day 3)
├── src/
│   ├── dataset.py                  # MultiViewScene loader (see spec §8)
│   ├── camera_utils.py             # K, R, t helpers; Blender<->OpenCV conversion
│   └── geometry.py                 # F-matrix, epipolar distance
├── data/
│   └── output/                     # rendered scenes go here, one folder per scene
└── tests/
    └── test_camera_conventions.py
```

## Tech Stack

- Python 3.10+ (system, for `src/` and `tests/`)
- **Blender 4.x already installed** (its bundled Python runs `scripts/render_scene.py`)
- numpy, opencv-python, Pillow (system Python only)
- torch + kornia for LoFTR eval (Day 3 only)

## The Two-Python Trap

Blender ships its own Python interpreter, separate from your system Python. **Never try to import `cv2` or `PIL` inside a Blender script** — they're not installed there and shouldn't be. The clean pattern:

- **Inside Blender scripts (`scripts/render_scene.py`):** only `bpy`, `mathutils`, and stdlib (`json`, `math`, `os`, `pathlib`).
- **Inside system Python (`src/`, `tests/`):** numpy, cv2, PIL, etc.
- **Bridge between them:** JSON files on disk. Blender writes `poses.json`; system Python reads it.

If you find yourself wanting to `pip install` something into Blender's Python, stop and ask first.

## Working Style for Claude Code

- **Verify before generating volume.** Render 1 scene end-to-end and inspect outputs (open a depth PNG, project a 3D point and check it lands on the right pixel) BEFORE rendering 50 scenes.
- **Convention sanity checks are not optional.** After any change to camera handling, run `tests/test_camera_conventions.py`. If it doesn't exist yet, write it.
- **Read my existing code before replacing it.** I have `scripts/existing_export.py`. Ask me to paste it. Diff against it. Don't reinvent from scratch.
- **No silent fallbacks.** If a Blender API call might fail, raise loudly. We don't have time to debug a dataset that "looks fine" but has 3 broken scenes.
- **Don't add features that aren't in the spec.** Scope creep kills 4-day timelines.

## Known Pitfalls

- `bpy.context.scene.render` settings persist across script runs in the same session. Reset them explicitly at script start.
- Blender's depth output (Z-pass / Mist / View Layer's Depth output) needs the compositor or `bpy.context.view_layer.use_pass_z = True` enabled. The exact path depends on Cycles vs EEVEE.
- Cycles transparent materials produce nonsense in the depth pass — that's the "depth paradox" we're solving with the matte-gray pass.
- LoFTR expects grayscale, normalized to [0,1], with H and W divisible by 8.

## Out of Scope

- 6D pose estimation training (preliminary numbers only at the end, if time).
- HDRI environments. Flat gray is fine for v1.
- Anything Yujie owns: diffusion, attention layers, ControlNet integration.
