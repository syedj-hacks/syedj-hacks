# -*- coding: utf-8 -*-
"""
Writes assets/banner/dark.svg and assets/banner/light.svg from the data
produced by prep_portrait.py / build_logos.py / assemble_banner.py.
"""
import numpy as np
from assemble_banner import (
    THEMES, GRID_W, GRID_H, CANVAS_W, CANVAS_H, TITLEBAR_H,
    FRAME_X, FRAME_Y, FRAME_W, FRAME_H, DOT_UNIT,
    PORTRAIT_OX, PORTRAIT_OY, DOT_SIZE, INFO_X, INFO_W,
    LOOP_DUR, T_PORTRAIT_END, T_TRANS1_END, T_LOGO1_END, T_TRANS2_END,
    T_LOGO2_END, T_TRANS3_END, T_LOGO3_END, T_WRAP, INTRO_DUR,
    ROWS_GROUP1, ROWS_GROUP2, ROWS_GROUP3,
    N_BANDS, N_INTRO, portrait, logos,
)

_data = np.load("assembled.npy", allow_pickle=True).item()
band_members = _data["band_members"]
band_drift = _data["band_drift"]
coord_to_band = _data["coord_to_band"]
coord_to_intro = _data["coord_to_intro"]
band_centroid_arr = _data["band_centroid_arr"]
logo1_pts = _data["logo1_pts"]
logo2_pts = _data["logo2_pts"]
logo3_pts = _data["logo3_pts"]


def dots_by_group(dots, lookup, n_groups, centroid_arr=None):
    """Bucket a theme's dot array by band/intro-group id, looking up via the
    coordinate dict built from dark_dots. A handful of light-mode-only stray
    pixels (error-diffusion bleed kept in light mode) fall back to nearest
    centroid (bands) or a coordinate-hashed group (intro)."""
    buckets = {g: [] for g in range(n_groups)}
    for x, y in dots.astype(int):
        key = (int(x), int(y))
        g = lookup.get(key)
        if g is None:
            if centroid_arr is not None:
                g = int(np.argmin(np.linalg.norm(centroid_arr - np.array([x, y]), axis=1)))
            else:
                g = (x * 131 + y * 137) % n_groups
        buckets[g].append((x, y))
    return {g: np.array(v) if v else np.empty((0, 2)) for g, v in buckets.items()}

FONT = "'Fira Code','Cascadia Code',ui-monospace,monospace"
KEYTIMES9 = [0, T_PORTRAIT_END, T_TRANS1_END, T_LOGO1_END, T_TRANS2_END,
             T_LOGO2_END, T_TRANS3_END, T_LOGO3_END, T_WRAP]
KEYTIMES9_N = [round(t / LOOP_DUR, 5) for t in KEYTIMES9]
KEYTIMES5 = [0, T_PORTRAIT_END, T_TRANS1_END, T_LOGO3_END, T_WRAP]
KEYTIMES5_N = [round(t / LOOP_DUR, 5) for t in KEYTIMES5]


def fnum(v):
    return f"{v:.2f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)


def runs_path(coords):
    """Merge same-row adjacent dots into horizontal run rectangles; compact
    'd' path string in local (grid-unit) coordinates."""
    if len(coords) == 0:
        return ""
    by_row = {}
    for x, y in coords:
        by_row.setdefault(int(y), []).append(int(x))
    parts = []
    pad = (1 - DOT_SIZE) / 2
    for y, xs in by_row.items():
        xs.sort()
        run_start = xs[0]
        prev = xs[0]
        for x in xs[1:] + [None]:
            if x is not None and x == prev + 1:
                prev = x
                continue
            w = (prev - run_start) + DOT_SIZE
            parts.append(f"M{run_start+pad:.2f} {y+pad:.2f}h{w:.2f}v{DOT_SIZE:.2f}h{-w:.2f}Z")
            if x is not None:
                run_start = x
                prev = x
    return "".join(parts)


def leader_row(x, y, label, value, theme, font_size=14):
    mono_w = font_size * 0.6
    label_w = len(label) * mono_w
    value_w = min(len(value) * mono_w, INFO_W * 0.62)
    dots_start = x + label_w + 8
    dots_end = x + INFO_W - value_w - 8
    n_dots = max(3, int((dots_end - dots_start) / (font_size * 0.42)))
    out = []
    out.append(
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{font_size}" '
        f'fill="{theme["muted"]}">{label}</text>'
    )
    if dots_end > dots_start:
        out.append(
            f'<text x="{dots_start:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{font_size}" '
            f'fill="{theme["border"]}" textLength="{dots_end-dots_start:.1f}" lengthAdjust="spacing">'
            f'{"." * n_dots}</text>'
        )
    out.append(
        f'<text x="{x+INFO_W:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{font_size}" '
        f'fill="{theme["bright"]}" text-anchor="end" textLength="{value_w:.1f}" '
        f'lengthAdjust="spacingAndGlyphs">{value}</text>'
    )
    return "\n".join(out)


def build_intro_layer(theme):
    dots = portrait[theme["dots_key"]]
    buckets = dots_by_group(dots, coord_to_intro, N_INTRO)
    groups = []
    for g in range(N_INTRO):
        pts = buckets[g]
        d = runs_path(pts)
        if not d:
            continue
        begin = round((g / max(1, N_INTRO - 1)) * 1.8, 3)
        groups.append(
            f'<g opacity="0" fill="{theme["dot_portrait"]}">'
            f'<animate attributeName="opacity" begin="{begin}s" dur="0.5s" '
            f'from="0" to="1" fill="freeze" calcMode="spline" keySplines="0.2 0 0.2 1"/>'
            f'<path d="{d}"/></g>'
        )
    body = "".join(groups)
    return (
        f'<g id="introLayer" transform="translate({PORTRAIT_OX:.2f},{PORTRAIT_OY:.2f}) '
        f'scale({DOT_UNIT:.4f})">{body}'
        f'<animate attributeName="opacity" begin="{T_PORTRAIT_END}s" dur="0.2s" '
        f'values="1;0" fill="freeze"/></g>'
    )


def build_main_layer(theme):
    dots = portrait[theme["dots_key"]]
    buckets = dots_by_group(dots, coord_to_band, N_BANDS, centroid_arr=band_centroid_arr)
    groups = []
    for b in range(N_BANDS):
        pts = buckets[b]
        if len(pts) == 0:
            continue
        d = runs_path(pts)
        dx, dy = band_drift[b]
        tvals = f"0,0;0,0;0,0;{dx/DOT_UNIT:.3f},{dy/DOT_UNIT:.3f};0,0"
        # translate deltas are in grid units already (band_drift computed in
        # portrait-grid space); dividing by DOT_UNIT undoes the outer scale()
        ovals = "1;1;0.16;0.16;1"
        groups.append(
            f'<g fill="{theme["dot_portrait"]}">'
            f'<animateTransform attributeName="transform" type="translate" '
            f'values="{tvals}" keyTimes="{";".join(str(k) for k in KEYTIMES5_N)}" '
            f'begin="{INTRO_DUR}s" dur="{LOOP_DUR}s" repeatCount="indefinite"/>'
            f'<animate attributeName="opacity" values="{ovals}" '
            f'keyTimes="{";".join(str(k) for k in KEYTIMES5_N)}" '
            f'begin="{INTRO_DUR}s" dur="{LOOP_DUR}s" repeatCount="indefinite"/>'
            f'<path d="{d}"/></g>'
        )
    body = "".join(groups)
    return (
        f'<g id="mainLayer" opacity="0" transform="translate({PORTRAIT_OX:.2f},{PORTRAIT_OY:.2f}) '
        f'scale({DOT_UNIT:.4f})">'
        f'<animate attributeName="opacity" begin="{T_PORTRAIT_END}s" dur="0.2s" '
        f'values="0;1" fill="freeze"/>{body}</g>'
    )


def build_traveler_layer(theme):
    kt = ";".join(str(k) for k in KEYTIMES9_N)
    op_vals = "0;0;1;1;1;1;1;1;0"
    dots = []
    size = DOT_SIZE * 1.15  # travellers read as slightly thicker per spec
    for i in range(len(logo1_pts)):
        p1x, p1y = logo1_pts[i]
        p2x, p2y = logo2_pts[i]
        p3x, p3y = logo3_pts[i]
        pos1 = f"{p1x:.2f},{p1y:.2f}"
        pos2 = f"{p2x:.2f},{p2y:.2f}"
        pos3 = f"{p3x:.2f},{p3y:.2f}"
        tvals = ";".join([pos1, pos1, pos1, pos1, pos2, pos2, pos3, pos3, pos3])
        pad = (1 - size) / 2
        dots.append(
            f'<g opacity="0" fill="{theme["dot_traveler"]}">'
            f'<animateTransform attributeName="transform" type="translate" '
            f'values="{tvals}" keyTimes="{kt}" '
            f'begin="{INTRO_DUR}s" dur="{LOOP_DUR}s" repeatCount="indefinite"/>'
            f'<animate attributeName="opacity" values="{op_vals}" keyTimes="{kt}" '
            f'begin="{INTRO_DUR}s" dur="{LOOP_DUR}s" repeatCount="indefinite"/>'
            f'<rect x="{-pad:.2f}" y="{-pad:.2f}" width="{size:.2f}" height="{size:.2f}"/></g>'
        )
    body = "".join(dots)
    return (
        f'<g id="travelerLayer" transform="translate({PORTRAIT_OX:.2f},{PORTRAIT_OY:.2f}) '
        f'scale({DOT_UNIT:.4f})" shape-rendering="crispEdges">{body}</g>'
    )


def build_info_panel(theme):
    y = TITLEBAR_H + 100
    row_h = 23
    out = []
    out.append(
        f'<text x="{INFO_X}" y="{TITLEBAR_H+38}" font-family="{FONT}" font-size="13" '
        f'letter-spacing="2" fill="{theme["chrome"]}">SYSTEM.INFO</text>'
    )
    # LIVE badge
    lx, ly = INFO_X + INFO_W - 150, TITLEBAR_H + 30
    out.append(
        f'<circle cx="{lx}" cy="{ly-4}" r="4" fill="{theme["live"]}">'
        f'<animate attributeName="opacity" values="1;0.25;1" dur="1.2s" repeatCount="indefinite"/>'
        f'</circle>'
        f'<text x="{lx+10}" y="{ly}" font-family="{FONT}" font-size="12" letter-spacing="2" '
        f'fill="{theme["live"]}">LIVE</text>'
    )
    # handle pill
    pill_w, pill_h = 150, 24
    px, py = INFO_X + INFO_W - pill_w, TITLEBAR_H + 42
    out.append(
        f'<rect x="{px}" y="{py}" width="{pill_w}" height="{pill_h}" rx="{pill_h/2}" '
        f'fill="none" stroke="{theme["chrome"]}" stroke-width="1"/>'
        f'<text x="{px+pill_w/2}" y="{py+16}" font-family="{FONT}" font-size="13" '
        f'text-anchor="middle" fill="{theme["chrome"]}">@syedj-hacks</text>'
    )
    yy = y
    for label, value in ROWS_GROUP1:
        out.append(leader_row(INFO_X, yy, label, value, theme))
        yy += row_h
    yy += 10
    for label, value in ROWS_GROUP2:
        out.append(leader_row(INFO_X, yy, label, value, theme))
        yy += row_h
    yy += 10
    for label, value in ROWS_GROUP3:
        out.append(leader_row(INFO_X, yy, label, value, theme))
        yy += row_h
    return "\n".join(out)


def build_svg(theme_name):
    theme = THEMES[theme_name]
    clip_id = f"visualClip-{theme_name}"
    parts = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" height="{CANVAS_H}" '
        f'viewBox="0 0 {CANVAS_W} {CANVAS_H}" role="img" '
        f'aria-label="Syed Jafar terminal profile banner">'
    )
    parts.append(f'<rect width="{CANVAS_W}" height="{CANVAS_H}" fill="{theme["bg"]}"/>')
    # terminal window
    parts.append(
        f'<rect x="6" y="6" width="{CANVAS_W-12}" height="{CANVAS_H-12}" rx="10" '
        f'fill="{theme["panel"]}" stroke="{theme["border"]}"/>'
    )
    parts.append(
        f'<rect x="6" y="6" width="{CANVAS_W-12}" height="{TITLEBAR_H}" rx="10" '
        f'fill="{theme["panel"]}" stroke="{theme["border"]}"/>'
    )
    for i, c in enumerate(["#FF5F56", "#FFBD2E", "#27C93F"]):
        parts.append(f'<circle cx="{28+i*18}" cy="{6+TITLEBAR_H/2}" r="5.5" fill="{c}"/>')
    parts.append(
        f'<text x="{CANVAS_W/2}" y="{6+TITLEBAR_H/2+4}" text-anchor="middle" '
        f'font-family="{FONT}" font-size="12" fill="{theme["muted"]}">profile.sh --live</text>'
    )
    # portrait frame
    parts.append(
        f'<text x="{FRAME_X}" y="{TITLEBAR_H+34}" font-family="{FONT}" font-size="13" '
        f'letter-spacing="2" fill="{theme["chrome"]}">VISUAL.MAP</text>'
    )
    parts.append(
        f'<rect x="{FRAME_X}" y="{FRAME_Y}" width="{FRAME_W}" height="{FRAME_H}" rx="6" '
        f'fill="{theme["bg"]}" stroke="{theme["border"]}"/>'
    )
    parts.append(
        f'<clipPath id="{clip_id}"><rect x="{FRAME_X+2}" y="{FRAME_Y+2}" '
        f'width="{FRAME_W-4}" height="{FRAME_H-4}" rx="5"/></clipPath>'
    )
    parts.append(f'<g clip-path="url(#{clip_id})" shape-rendering="crispEdges">')
    parts.append(build_intro_layer(theme))
    parts.append(build_main_layer(theme))
    parts.append(build_traveler_layer(theme))
    parts.append("</g>")
    # divider
    parts.append(
        f'<line x1="{INFO_X-20}" y1="{TITLEBAR_H+30}" x2="{INFO_X-20}" y2="{CANVAS_H-30}" '
        f'stroke="{theme["border"]}"/>'
    )
    parts.append(build_info_panel(theme))
    parts.append("</svg>")
    return "".join(parts)


for name in ("dark", "light"):
    svg = build_svg(name)
    out_path = f"../{name}.svg"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(svg)
    import os
    size_kb = os.path.getsize(out_path) / 1024
    print(f"{out_path}: {size_kb:.1f} KB")
