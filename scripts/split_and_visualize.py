import os
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

# ── 1. Split test.png into 6 views ──────────────────────────────────────────
os.makedirs("outputs/views_split", exist_ok=True)

grid = Image.open("outputs/test.png")  # 640x960, 2-col × 3-row
W, H = grid.size          # 640, 960
cols, rows = 2, 3
tw, th = W // cols, H // rows  # tile: 320×320

# Reading order: top-left, top-right, middle-left, middle-right, bottom-left, bottom-right
tiles = []
for row in range(rows):
    for col in range(cols):
        x0, y0 = col * tw, row * th
        tile = grid.crop((x0, y0, x0 + tw, y0 + th))
        path = f"outputs/views_split/view_{len(tiles)}.png"
        tile.save(path)
        tiles.append(tile)
        print(f"Saved {path}  ({x0},{y0})→({x0+tw},{y0+th})")

# ── 2. Download condition image ──────────────────────────────────────────────
cond_url = "https://d.skis.ltd/nrp/sample-data/0_cond.png"
cond = Image.open(BytesIO(requests.get(cond_url).content)).convert("RGB")
cond = cond.resize((tw, th), Image.LANCZOS)
cond.save("outputs/views_split/view_cond.png")
print(f"Saved outputs/views_split/view_cond.png  ({cond.size})")

# ── 3. Compose 2×4 overview with labels ─────────────────────────────────────
# Zero123++ v1.2 fixed poses (from zero123_camera.py)
labels = [
    "cond\n(input)",
    "view_0\naz=30°, el=+20°",
    "view_1\naz=90°, el=-10°",
    "view_2\naz=150°, el=+20°",
    "view_3\naz=210°, el=-10°",
    "view_4\naz=270°, el=+20°",
    "view_5\naz=330°, el=-10°",
]
images = [cond] + tiles   # 7 images

NCOLS = 4
NROWS = 2
PAD   = 8     # px padding around each tile
LABEL_H = 52  # px height reserved for text below each tile

cell_w = tw + 2 * PAD
cell_h = th + 2 * PAD + LABEL_H
canvas_w = NCOLS * cell_w
canvas_h = NROWS * cell_h

canvas = Image.new("RGB", (canvas_w, canvas_h), color=(240, 240, 240))
draw = ImageDraw.Draw(canvas)

# Try to load a small font; fall back to default if not available
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
except Exception:
    font = ImageFont.load_default()

for idx, (img, label) in enumerate(zip(images, labels)):
    row = idx // NCOLS
    col = idx % NCOLS
    x0 = col * cell_w + PAD
    y0 = row * cell_h + PAD

    # Paste image
    canvas.paste(img, (x0, y0))

    # Draw label (multi-line, centred below image)
    lines = label.split("\n")
    line_h = 22
    text_y = y0 + th + 6
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        text_x = x0 + (tw - text_w) // 2
        draw.text((text_x, text_y), line, fill=(30, 30, 30), font=font)
        text_y += line_h

# Leave cell (1, 3) empty — grey placeholder
empty_col, empty_row = 3, 1
ex0 = empty_col * cell_w + PAD
ey0 = empty_row * cell_h + PAD
draw.rectangle([ex0, ey0, ex0 + tw, ey0 + th], fill=(200, 200, 200))

canvas.save("outputs/convention_check.png")
print(f"\nSaved outputs/convention_check.png  ({canvas.size})")
