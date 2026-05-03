"""
Epipolar geometry validation of Zero123++ v1.2 azimuth convention.

Tests two conventions:
  current : azimuths = [30, 90, 150, 210, 270, 330]
  negated : azimuths = [-30, -90, -150, -210, -270, -330]

Smaller median Sampson distance → correct convention.
"""

import os, sys, csv, time, requests
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zero123_camera import (
    get_zero123plus_poses, get_condition_pose,
    AZIMUTHS_DEG, ELEVATIONS_DEG,
    _spherical_to_xyz, _look_at_opencv,
)

# ── Config ────────────────────────────────────────────────────────────────────
RADIUS   = 1.5
OUT      = "outputs/convention_test"
IMG_SIZE = 320
FOV_DEG  = 30.0
focal    = (IMG_SIZE / 2) / np.tan(np.deg2rad(FOV_DEG / 2))   # ≈ 597.3
K = np.array([[focal, 0, IMG_SIZE/2],
              [0, focal, IMG_SIZE/2],
              [0, 0,     1.0        ]], dtype=np.float64)

os.makedirs(OUT, exist_ok=True)
print(f"K: focal={focal:.1f}  cx=cy={IMG_SIZE/2}")

# ── Step 1: Textured test image ───────────────────────────────────────────────
cond_path = f"{OUT}/cond.png"
URLS = [
    "https://raw.githubusercontent.com/SUDO-AI-3D/zero123plus/main/assets/9.png",
    "https://raw.githubusercontent.com/SUDO-AI-3D/zero123plus/main/assets/6.png",
    "https://raw.githubusercontent.com/SUDO-AI-3D/zero123plus/main/assets/1.png",
    "https://raw.githubusercontent.com/SUDO-AI-3D/zero123plus/main/assets/5.png",
    "https://raw.githubusercontent.com/SUDO-AI-3D/zero123plus/main/assets/7.png",
]
if not os.path.exists(cond_path):
    for url in URLS:
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200 and len(r.content) > 5_000:
                img = Image.open(BytesIO(r.content)).convert("RGB")
                img.save(cond_path)
                print(f"Downloaded cond: {url}  size={img.size}")
                break
        except Exception as e:
            print(f"  skip {url}: {e}")
    else:
        raise RuntimeError(
            "All URLs failed. Manually place a textured image at:\n"
            f"  {cond_path}"
        )
else:
    print(f"Using existing: {cond_path}")

cond_pil = Image.open(cond_path).convert("RGB")
if cond_pil.size != (IMG_SIZE, IMG_SIZE):
    cond_pil = cond_pil.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    cond_pil.save(cond_path)

# ── Step 2: Zero123++ inference (cached) ──────────────────────────────────────
view_paths = [f"{OUT}/view_{i}.png" for i in range(6)]
if not all(os.path.exists(p) for p in view_paths):
    print("\nRunning Zero123++ ...")
    from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
    pipe = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2",
        custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=torch.float16,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    pipe.to("cuda"); pipe.enable_attention_slicing()

    t0 = time.time()
    grid = pipe(cond_pil, num_inference_steps=36).images[0]   # 640×960
    print(f"  inference: {time.time()-t0:.1f}s")

    for row in range(3):                   # 3 rows, 2 cols → 2-col × 3-row
        for col in range(2):
            i = row * 2 + col
            tile = grid.crop((col*IMG_SIZE, row*IMG_SIZE,
                              (col+1)*IMG_SIZE, (row+1)*IMG_SIZE))
            tile.save(view_paths[i])
    del pipe; torch.cuda.empty_cache()
    print(f"  saved 6 views to {OUT}/")
else:
    print("Using cached views.")

views = [Image.open(p).convert("RGB") for p in view_paths]

# ── Step 3: LoFTR matching ────────────────────────────────────────────────────
import kornia.feature as KF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nLoading LoFTR (outdoor) on {device} ...")
matcher = KF.LoFTR(pretrained="outdoor").eval().to(device)

def to_gray_t(pil_img):
    arr = np.array(pil_img.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, None].to(device)

def match_pair(a: Image.Image, b: Image.Image, min_conf=0.5):
    with torch.no_grad():
        out = matcher({"image0": to_gray_t(a), "image1": to_gray_t(b)})
    pts_a = out["keypoints0"].cpu().numpy()
    pts_b = out["keypoints1"].cpu().numpy()
    conf  = out["confidence"].cpu().numpy()
    mask  = conf > min_conf
    return pts_a[mask], pts_b[mask]

# ── Steps 4-6: F matrix + Sampson distance ────────────────────────────────────
def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[ 0,    -v[2],  v[1]],
                     [ v[2],  0,    -v[0]],
                     [-v[1],  v[0],  0   ]], dtype=np.float64)

def compute_F(R_a, t_a, R_b, t_b, K_a, K_b):
    """F = K_b^{-T} [t_ba]x R_ba K_a^{-1}"""
    R_ba = R_b @ R_a.T
    t_ba = t_b - R_ba @ t_a
    E    = skew(t_ba) @ R_ba
    return np.linalg.inv(K_b).T @ E @ np.linalg.inv(K_a)

def sampson(F, pa, pb):
    """Sampson distance in ~pixels for N point pairs."""
    N   = len(pa)
    ha  = np.c_[pa, np.ones(N)]
    hb  = np.c_[pb, np.ones(N)]
    Fha  = (F   @ ha.T).T   # (N,3)
    FThb = (F.T @ hb.T).T   # (N,3)
    num  = np.einsum("ij,ij->i", hb, Fha) ** 2
    den  = Fha[:,0]**2 + Fha[:,1]**2 + FThb[:,0]**2 + FThb[:,1]**2
    return np.sqrt(np.abs(num) / (den + 1e-10))

# Condition pose: az=0, el=0 (front, level)
R_c, t_c = get_condition_pose(RADIUS)

# Current convention poses
poses_cur = get_zero123plus_poses(RADIUS)

# Negated azimuth poses
poses_neg = [
    _look_at_opencv(_spherical_to_xyz(-az, el, RADIUS), np.zeros(3))
    for az, el in zip(AZIMUTHS_DEG, ELEVATIONS_DEG)
]

# ── Main loop ─────────────────────────────────────────────────────────────────
header = f"{'V':>2} {'az':>4} {'el':>4} | {'N':>5} | {'cur_med':>9} {'cur_mean':>9} | {'neg_med':>9} {'neg_mean':>9}"
print("\n" + header)
print("-" * len(header))

csv_rows = []
heatmap_data = None   # capture view_1

for i in range(6):
    az, el = AZIMUTHS_DEG[i], ELEVATIONS_DEG[i]
    pa, pb = match_pair(cond_pil, views[i])
    n = len(pa)

    if n < 20:
        print(f"{i:>2} {az:>4} {el:>4} | {n:>5} | {'SKIP (< 20 matches)'}")
        csv_rows.append(dict(view=i, az=az, el=el, matches=n,
                             sampson_cur_med="", sampson_cur_mean="",
                             sampson_neg_med="", sampson_neg_mean=""))
        continue

    F_cur = compute_F(R_c, t_c, *poses_cur[i], K, K)
    F_neg = compute_F(R_c, t_c, *poses_neg[i], K, K)
    sd_cur = sampson(F_cur, pa, pb)
    sd_neg = sampson(F_neg, pa, pb)

    mc, uc = float(np.median(sd_cur)), float(np.mean(sd_cur))
    mn, un = float(np.median(sd_neg)), float(np.mean(sd_neg))
    flag_c = " ←" if mc < mn else "   "
    flag_n = " ←" if mn < mc else "   "

    print(f"{i:>2} {az:>4} {el:>4} | {n:>5} | "
          f"{mc:>9.2f}{un:>9.2f}{flag_c}| "
          f"{mn:>9.2f}{un:>9.2f}{flag_n}")

    csv_rows.append(dict(view=i, az=az, el=el, matches=n,
                         sampson_cur_med=f"{mc:.2f}",
                         sampson_cur_mean=f"{uc:.2f}",
                         sampson_neg_med=f"{mn:.2f}",
                         sampson_neg_mean=f"{un:.2f}"))
    if i == 1:
        heatmap_data = dict(pts_b=pb, sd_cur=sd_cur, sd_neg=sd_neg, view_img=views[i])

print("-" * len(header))

valid = [r for r in csv_rows if r["sampson_cur_med"] != ""]
if valid:
    avg_c = np.mean([float(r["sampson_cur_med"]) for r in valid])
    avg_n = np.mean([float(r["sampson_neg_med"]) for r in valid])
    print(f"\nMean-of-medians →  current={avg_c:.2f}px   negated={avg_n:.2f}px")
    winner = "CURRENT (az CW)" if avg_c < avg_n else "NEGATED (az CCW)"
    print(f"Verdict: {winner} convention is correct.\n")

# ── Step 7: CSV ───────────────────────────────────────────────────────────────
csv_path = f"{OUT}/sampson_comparison.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
    w.writeheader(); w.writerows(csv_rows)
print(f"Saved: {csv_path}")

# ── Step 8: Heatmap ───────────────────────────────────────────────────────────
if heatmap_data is not None:
    d = heatmap_data
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    titles = [
        f"Current conv. (az=90°)\nmedian={np.median(d['sd_cur']):.1f} px",
        f"Negated az (az=-90°)\nmedian={np.median(d['sd_neg']):.1f} px",
    ]
    for ax, sd, title in zip(axes, [d["sd_cur"], d["sd_neg"]], titles):
        ax.imshow(d["view_img"])
        sc = ax.scatter(d["pts_b"][:,0], d["pts_b"][:,1],
                        c=np.clip(sd, 0, 50), cmap="RdYlGn_r",
                        s=12, alpha=0.85, vmin=0, vmax=50)
        plt.colorbar(sc, ax=ax, label="Sampson dist (px)")
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    plt.suptitle("Epipolar validation: cond → view_1  (green=good, red=bad)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    hp = f"{OUT}/heatmap_comparison.png"
    plt.savefig(hp, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {hp}")
