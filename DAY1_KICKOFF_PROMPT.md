# Day 1 Kickoff Prompt for Claude Code

You already have Blender installed and a `.blend` file with your scene built by hand. The workflow today is **augment what you have**, not rebuild from zero. Drop `CLAUDE.md` and `DATA_INTERFACE_SPEC.md` in the repo root, copy your `.blend` file to `blend_files/scene_base.blend`, copy your existing export script to `scripts/existing_export.py`, then run `claude` and paste the prompt below.

---

## The prompt

> Read `CLAUDE.md` and `DATA_INTERFACE_SPEC.md` first, in full. Then read `scripts/existing_export.py` — that's my current working code, not something to throw away. Tell me in 2-3 sentences:
> 1. What you understood about the project goal.
> 2. What my existing export script currently does.
> 3. What's missing or wrong in it relative to the spec.
>
> Wait for my reply before writing any code.
>
> ---
>
> Today is Day 1 of a 4-day project. The goal for today is **upgrade my existing pipeline so that one .blend file produces a fully spec-compliant scene folder with 8 views**, all driven by a single command. We will scale to multiple scenes tomorrow.
>
> The big shift from my current workflow:
> - **Before:** I built one camera in GUI, ran a script to export its pose + depth.
> - **After today:** Script loads my .blend, deletes any existing cameras, programmatically places 8 cameras on a ring, renders RGB + clean depth from each, writes spec-compliant `poses.json` and `meta.json`.
>
> Materials and lighting stay as I built them in the GUI. Don't touch them.
>
> Build the following, in order. Stop and ask before each step transition.
>
> **Step 1 — Repo skeleton + dependencies.**
> Create the directory layout from `CLAUDE.md`. Make `pyproject.toml` listing `numpy`, `opencv-python`, `Pillow` for the system venv. Add `.gitignore` for `data/output/`, `__pycache__/`, `.venv/`, `notes/`, `*.blend1` (Blender backup files). Note that we install nothing into Blender's bundled Python — Blender scripts use only `bpy`, `mathutils`, and stdlib. See "The Two-Python Trap" in CLAUDE.md.
>
> **Step 2 — `src/camera_utils.py` + tests (system Python).**
> Implement:
> - `blender_to_opencv(R_bl, t_bl) -> (R_cv, t_cv)` using `diag(1, -1, -1)`.
> - `opencv_to_blender(R_cv, t_cv) -> (R_bl, t_bl)` — the inverse, needed because we'll compute camera placement in OpenCV convention then hand it back to Blender's API.
> - `make_intrinsics(fx, fy, cx, cy) -> K`.
> - `project_points(K, R, t, X_world) -> (u, v, z_cam)`. Returns pixel coords AND camera-space Z.
> - `look_at(eye, target, up=(0, 0, 1)) -> (R_world2cam_opencv, t_world2cam_opencv)` for placing cameras on a ring.
> - `ring_camera_poses(n=8, radius=1.0, target=(0,0,0), height_range=(0.2, 0.6)) -> list of (R, t) pairs in OpenCV convention`. Vary heights via a sine wave so views aren't coplanar (coplanar cameras give degenerate fundamental matrices).
>
> Write `tests/test_camera_conventions.py`:
> 1. Camera at (0, 0, -1) looking at origin → world origin projects to image center.
> 2. World point at +X → projects to u > cx (right of center) when camera looks down +Z.
> 3. World point at +Z (above origin in world frame, since world +Z is up) → with a camera at +X looking at origin, the point should project to v < cy (above image center). **This is the +Y-down trap test — most common OpenCV bug.**
> 4. `blender_to_opencv` and `opencv_to_blender` round-trip to identity.
> 5. `ring_camera_poses(n=8)` returns 8 distinct poses, all with `look_at` direction pointing at the target.
>
> Run `pytest tests/`. All must pass before Step 3.
>
> **Step 3 — Modify or rewrite `scripts/render_scene.py`.**
> First, look at `scripts/existing_export.py` and decide: extend it, or replace it? Whichever is faster. Tell me which and why before you start coding.
>
> The new script must:
> - Be invoked as `<blender-path> --background --python scripts/render_scene.py -- --blend blend_files/scene_base.blend --scene-id scene_0001 --output-dir data/output`.
> - Open my .blend file via `bpy.ops.wm.open_mainfile(filepath=args.blend)`.
> - **Delete all existing cameras** in the scene (I may have left one or more from manual setup). Do NOT delete meshes, materials, or lights — those are mine and I built them by hand.
> - Compute 8 camera poses in OpenCV convention (call into a Python helper that mirrors the logic of `ring_camera_poses` from `src/camera_utils.py`, but reimplemented inline because Blender's Python can't import from `src/`).
> - For each pose, convert OpenCV → Blender convention and create a Blender camera object with that pose. Set the camera's `lens` and sensor size such that intrinsics match `fx=fy=512, cx=cy=256` for a 512x512 render.
> - Set render settings: Cycles, 64 samples (low for Day 1; bump to 256 on Day 2), 512x512 resolution, sRGB output.
> - For each camera, render the RGB pass and save to `rgb/view_XX.png`.
> - For the clean-depth pass: see Step 3b below.
> - Convert each pose back to OpenCV convention and write `poses.json` per the spec. Include intrinsics, world-to-camera R and t, image paths.
> - Write `meta.json` with `augmentation_group: "G1"`, `spec_version: "0.1"`, my object info, and the render settings used.
> - Skip `fundamental.json` — that's Day 2.
>
> **Step 3b — The depth question (ask me before coding).**
> Dual-Path Rendering needs a "matte gray" depth pass with all transparent materials swapped for diffuse. There are three ways to handle this and I need to pick:
>
> Option A: **Two .blend files** — `scene_base.blend` (real materials) and `scene_base_matte.blend` (all materials replaced with gray diffuse). Script renders RGB from one, depth from the other. Simplest, no material scripting needed today.
>
> Option B: **Material swap in script** — script saves the existing material assignments, swaps everything to a gray diffuse for the depth pass, restores afterward. Cleaner long-term but more code today.
>
> Option C: **Skip depth_clean today, only render RGB and depth from real materials.** Defer Dual-Path to Day 2 entirely.
>
> Ask me which. Don't pick one yourself. Each has different tradeoffs depending on how complex my material setup is.
>
> **Step 4 (stretch) — `scripts/validate_scene.py`.**
> Read `data/output/scene_0001/poses.json`, project the 3D point at world origin through every camera using saved intrinsics + extrinsics. Print (u, v) for each view; all should be within ~5 pixels of (256, 256) since cameras look at origin. Print `✅` or `❌ CONVENTION BUG SUSPECTED` accordingly.
>
> **End of day:** write `notes/day1_status.md` with: what got built, reprojection numbers from validate_scene.py (paste actual output), open issues, one-sentence "ready to scale tomorrow?" verdict.
>
> ---
>
> **Important reminders:**
> - When in doubt about format, consult `DATA_INTERFACE_SPEC.md`.
> - Two Pythons exist (system + Blender's). They don't share installed packages. Blender script writes JSON, system Python reads it.
> - I haven't yet locked the spec's open questions (§9). Treat 8 views and 512x512 as defaults, but parameterize them — don't hardcode.
> - If you want to add ControlNet preprocessing, depth visualization, fancy material handling, scope creep features: stop. Ask first.
> - I built the .blend by hand. **Do not regenerate the scene programmatically.** Load it, augment cameras, render, that's it.
>
> Begin with Step 1, but actually start by reading those two markdown files and `existing_export.py` and giving me the 2-3 sentence summary I asked for at the top.

---

## Why this prompt is shaped this way

- **"Read existing_export.py and tell me what's missing"** — this is the new opening move. Forces Claude Code to engage with your prior work instead of bulldozing it. If it skips this step or claims it'll "rewrite cleaner from scratch", interrupt and make it actually read.
- **Step 3b is a question, not an instruction** — your manual material setup might be 1 material or 20 materials. Option A (two .blend files) is dead simple if you have few materials; Option B is worth the code if you have many. Don't let Claude Code guess.
- **No "Step 0 install"** — you already have Blender working. Saved a step.
- **Sine-wave heights** — coplanar cameras (all at same height) make F matrices weird later. Spreading heights costs nothing and avoids a Day 3 surprise.
- **Material-swap deferred to Day 2 if needed** — Option C exists in Step 3b for a reason. If your scene has complex material setups (procedural shaders, node trees), don't try to scriptify material swap on Day 1. Render only RGB today, do dual-path tomorrow when you have a clean baseline working.

## When to interrupt Claude Code

- **After it summarizes existing_export.py:** if its summary is wrong (says it does something it doesn't), stop and correct. The rest of the day depends on it understanding your existing code.
- **At Step 3b:** make this an actual conversation, not a default. The right answer depends on facts only you know about your `.blend`.
- **After Step 3 produces scene_0001:** open `rgb/view_00.png` and `rgb/view_04.png` (front and back of ring). You should see your asset from opposite angles. If they look identical, the camera ring is broken. If your asset is upside down, you have a +Y sign error.
- **After Step 4:** reprojection numbers within ~2 pixels of (256, 256) means convention is solid. Anything farther off → debug now, not later.

## If you want to share existing_export.py with me

Before pasting the prompt to Claude Code, you can paste your `existing_export.py` to me here and I'll review it for spec-compliance gaps and convention bugs. That way Claude Code starts from cleaner ground, and we catch any "your existing script is silently doing the wrong thing with extrinsics" issues before they propagate.
