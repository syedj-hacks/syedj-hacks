"""
Build the 3 morph-target glyphs in the same 300x340 dot-grid space as the
portrait. No brand references were supplied, so these are constructed as
precise geometric glyphs (terminal prompt / shield-lock / crosshair) rather
than traced logos -- generic symbols, not hand-drawn brand marks.

Each glyph is rasterised, then Poisson-disc-ish sampled down to exactly
TRAVELLER_N points so every logo contributes the same dot count to the
travellers layer (needed for 1:1 optimal-transport matching).
"""
import numpy as np
from PIL import Image, ImageDraw
import json

GRID_W, GRID_H = 300, 340
TRAVELLER_N = 900
RNG = np.random.default_rng(7)


def sample_mask(mask, n):
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], axis=1).astype(float)
    if len(pts) == 0:
        raise ValueError("empty glyph mask")
    if len(pts) >= n:
        idx = RNG.choice(len(pts), size=n, replace=False)
        chosen = pts[idx]
    else:
        idx = RNG.choice(len(pts), size=n, replace=True)
        chosen = pts[idx]
    # jitter sub-pixel so exact duplicates (when upsampling) don't overlap
    chosen += RNG.normal(0, 0.35, chosen.shape)
    return chosen


def rasterize(draw_fn):
    im = Image.new("L", (GRID_W, GRID_H), 0)
    d = ImageDraw.Draw(im)
    draw_fn(d)
    return np.asarray(im) > 127


def glyph_terminal(d):
    cx, cy = GRID_W * 0.5, GRID_H * 0.5
    s = 92  # stroke half-length scale
    lw = 20
    # '>' chevron
    d.line([(cx - s * 1.3, cy - s * 0.9), (cx - s * 0.55, cy), (cx - s * 1.3, cy + s * 0.9)],
           fill=255, width=lw, joint="curve")
    # '_' underscore
    d.line([(cx - s * 0.35, cy + s * 0.95), (cx + s * 1.15, cy + s * 0.95)], fill=255, width=lw)
    # end caps as circles so the stroke reads clean at low res
    for pt in [(cx - s * 1.3, cy - s * 0.9), (cx - s * 0.55, cy), (cx - s * 1.3, cy + s * 0.9),
               (cx - s * 0.35, cy + s * 0.95), (cx + s * 1.15, cy + s * 0.95)]:
        r = lw / 2
        d.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], fill=255)


def glyph_shield(d):
    cx, cy = GRID_W * 0.5, GRID_H * 0.5
    w, h = 150, 190
    top = cy - h / 2
    pts = [
        (cx, top),
        (cx + w / 2, top + h * 0.16),
        (cx + w / 2, top + h * 0.55),
        (cx, top + h),
        (cx - w / 2, top + h * 0.55),
        (cx - w / 2, top + h * 0.16),
    ]
    d.polygon(pts, outline=255, width=16)
    d.line(pts + [pts[0]], fill=255, width=16, joint="curve")
    # keyhole cutout: draw then erase circle+triangle
    kr = 22
    d.ellipse([cx - kr, cy - kr - 10, cx + kr, cy + kr - 10], fill=255)
    d.polygon([(cx - kr * 0.55, cy + 4), (cx + kr * 0.55, cy + 4), (cx, cy + kr * 1.8)], fill=255)
    # punch the hole out (draw background colour on top)
    hr = 10
    d.ellipse([cx - hr, cy - hr - 10, cx + hr, cy + hr - 10], fill=0)
    d.polygon([(cx - hr * 0.5, cy + 2), (cx + hr * 0.5, cy + 2), (cx, cy + hr * 1.6)], fill=0)


def glyph_crosshair(d):
    cx, cy = GRID_W * 0.5, GRID_H * 0.5
    r_out, r_in = 95, 55
    lw = 14
    d.ellipse([cx - r_out, cy - r_out, cx + r_out, cy + r_out], outline=255, width=lw)
    d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=255)
    for ang in (0, 90, 180, 270):
        rad = np.deg2rad(ang)
        x0 = cx + np.cos(rad) * r_in
        y0 = cy + np.sin(rad) * r_in
        x1 = cx + np.cos(rad) * (r_out + 34)
        y1 = cy + np.sin(rad) * (r_out + 34)
        d.line([(x0, y0), (x1, y1)], fill=255, width=lw)


GLYPHS = {
    "terminal": glyph_terminal,
    "shield": glyph_shield,
    "crosshair": glyph_crosshair,
}

out = {}
for name, fn in GLYPHS.items():
    mask = rasterize(fn)
    pts = sample_mask(mask, TRAVELLER_N)
    out[name] = pts
    Image.fromarray((mask * 255).astype(np.uint8)).resize(
        (GRID_W * 2, GRID_H * 2), Image.NEAREST
    ).save(f"preview_glyph_{name}.png")
    print(name, "ink px:", int(mask.sum()), "sampled:", len(pts))

np.save("logo_dots.npy", out, allow_pickle=True)
print("saved logo_dots.npy")
