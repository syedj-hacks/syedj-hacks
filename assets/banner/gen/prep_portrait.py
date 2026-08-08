"""
Portrait dithering pipeline (Master Prompt, Phase 1).
Source: assets/Kali-Linux-Logo.png (stand-in subject; no personal photo supplied).
Output: gen/portrait_dots.npy  -- structured dot data used by the SVG assembler.

Grid: 300x340. 1-bit Floyd-Steinberg dither, serpentine order (manual, not PIL's
built-in ditherer, which is raster-only and doesn't alternate direction).
"""
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from scipy import ndimage
import json

SRC = "../../../assets/banner/portrait.png"
GRID_W, GRID_H = 300, 340
OUT_NPY = "portrait_dots.npy"
OUT_PREVIEW = "preview_portrait.png"
OUT_MASK_PREVIEW = "preview_mask.png"

im = Image.open(SRC).convert("RGBA")
# flatten onto white (source already white bg, but be explicit)
bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
im = Image.alpha_composite(bg, im).convert("L")

# Source is a square 1032x1032 logo with the subject weighted to the left
# (matches "head + shoulders, not tight crop" framing rule: don't zoom into
# just the head/eye of the dragon, keep the whole figure).
w, h = im.size
scale = GRID_H / h
im = im.resize((int(w * scale), GRID_H), Image.LANCZOS)
# crop to grid width, left-aligned (subject sits left, right side is blank margin)
im = im.crop((0, 0, GRID_W, GRID_H))

# contrast 1.3x + autocontrast(cutoff=1) + unsharp mask, per spec
im = ImageOps.autocontrast(im, cutoff=1)
im = ImageEnhance.Contrast(im).enhance(1.3)
im = im.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
im.save(OUT_PREVIEW.replace(".png", "_gray.png"))

arr = np.asarray(im, dtype=np.float64).copy()

# ---- foreground mask (for dark-mode error-diffusion clearing) ----
# distance from white; threshold; binary closing; fill holes; keep largest component
dist_from_white = 255.0 - arr
mask = dist_from_white > 18  # threshold on colour distance
mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)), iterations=2)
mask = ndimage.binary_fill_holes(mask)
labeled, n = ndimage.label(mask)
if n > 0:
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    largest = np.argmax(sizes) + 1
    mask = labeled == largest
Image.fromarray((mask * 255).astype(np.uint8)).save(OUT_MASK_PREVIEW)

# ---- manual serpentine Floyd-Steinberg dither ----
dots_light = []  # (x, y) dark pixels -> dot in light mode (all dark pixels)
dark_bitmap = np.zeros((GRID_H, GRID_W), dtype=bool)

for y in range(GRID_H):
    left_to_right = (y % 2 == 0)
    xs = range(GRID_W) if left_to_right else range(GRID_W - 1, -1, -1)
    for x in xs:
        old = arr[y, x]
        new = 0.0 if old < 128 else 255.0
        err = old - new
        arr[y, x] = new
        if new == 0.0:
            dark_bitmap[y, x] = True
        # serpentine error diffusion neighbour offsets flip with direction
        step = 1 if left_to_right else -1
        if 0 <= x + step < GRID_W:
            arr[y, x + step] += err * 7 / 16
        if y + 1 < GRID_H:
            if 0 <= x - step < GRID_W:
                arr[y + 1, x - step] += err * 3 / 16
            arr[y + 1, x] += err * 5 / 16
            if 0 <= x + step < GRID_W:
                arr[y + 1, x + step] += err * 1 / 16

# light-mode dots: every dark pixel from the dither (background is already
# white so it contributes nothing -- no masking needed)
ys, xs = np.nonzero(dark_bitmap)
light_dots = np.stack([xs, ys], axis=1)

# dark-mode dots: hard-clear any error-diffusion bleed outside the segmented
# subject silhouette (mask dilated by 1px so edge dots on the boundary survive)
mask_dilated = ndimage.binary_dilation(mask, iterations=1)
dark_keep = mask_dilated[ys, xs]
dark_dots = np.stack([xs[dark_keep], ys[dark_keep]], axis=1)

print(f"grid {GRID_W}x{GRID_H} = {GRID_W*GRID_H} px")
print(f"light-mode dots: {len(light_dots)}  ({100*len(light_dots)/(GRID_W*GRID_H):.1f}% ink)")
print(f"dark-mode dots:  {len(dark_dots)}  ({100*len(dark_dots)/(GRID_W*GRID_H):.1f}% ink)")
print(f"cleared as bleed: {len(light_dots) - len(dark_dots)}")

# preview renders (black dots on white / white dots on near-black)
prev_light = Image.new("L", (GRID_W, GRID_H), 255)
pl = np.asarray(prev_light).copy()
pl[ys, xs] = 0
Image.fromarray(pl).resize((GRID_W * 3, GRID_H * 3), Image.NEAREST).save(OUT_PREVIEW)

prev_dark = np.full((GRID_H, GRID_W), 10, dtype=np.uint8)
prev_dark[ys[dark_keep], xs[dark_keep]] = 235
Image.fromarray(prev_dark).resize((GRID_W * 3, GRID_H * 3), Image.NEAREST).save(
    OUT_PREVIEW.replace(".png", "_dark.png")
)

np.save(
    OUT_NPY,
    {
        "grid_w": GRID_W,
        "grid_h": GRID_H,
        "light_dots": light_dots,
        "dark_dots": dark_dots,
        "mask": mask,
    },
    allow_pickle=True,
)
print("saved", OUT_NPY)
