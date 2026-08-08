"""
Assembles dark.svg / light.svg from the dot data produced by prep_portrait.py
and build_logos.py. Implements: 94-band drift grouping (noise-decorrelated to
avoid the grid trap), 60-group interleaved intro shimmer, optimal-transport
traveller morph between the 3 logo glyphs, and the SYSTEM.INFO text panel.

Keep this script + the .npy files -- they're the source of truth, not the SVG.
"""
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment
import json

RNG = np.random.default_rng(42)

portrait = np.load("portrait_dots.npy", allow_pickle=True).item()
logos = np.load("logo_dots.npy", allow_pickle=True).item()
GRID_W, GRID_H = portrait["grid_w"], portrait["grid_h"]
LOGO_ORDER = ["terminal", "shield", "crosshair"]

# ---------------------------------------------------------------- geometry --
CANVAS_W, CANVAS_H = 1180, 610
TITLEBAR_H = 34
FRAME_X, FRAME_Y = 34, TITLEBAR_H + 46
FRAME_W = int(CANVAS_W * 0.38) - FRAME_X - 10   # ~410
FRAME_H = CANVAS_H - FRAME_Y - 34
DOT_UNIT = min(FRAME_W / GRID_W, FRAME_H / GRID_H)
PORTRAIT_PX_W = GRID_W * DOT_UNIT
PORTRAIT_PX_H = GRID_H * DOT_UNIT
PORTRAIT_OX = FRAME_X + (FRAME_W - PORTRAIT_PX_W) / 2
PORTRAIT_OY = FRAME_Y + (FRAME_H - PORTRAIT_PX_H) / 2
DOT_SIZE = 0.8  # fraction of grid cell -- "shorter dots" moire finding

INFO_X = int(CANVAS_W * 0.42)
INFO_W = CANVAS_W - INFO_X - 40

LOOP_DUR = 14.2
T_PORTRAIT_END = 3.00
T_TRANS1_END = 4.30
T_LOGO1_END = 6.30
T_TRANS2_END = 7.60
T_LOGO2_END = 9.60
T_TRANS3_END = 10.90
T_LOGO3_END = 12.90
T_WRAP = 14.20
INTRO_DUR = 3.2

THEMES = {
    "dark": dict(
        name="dark",
        bg="#080B10",
        panel="#0F141C",
        border="#1E2836",
        chrome="#2F9BF5",
        chrome_dim="#1B5B92",
        muted="#77879B",
        bright="#DDE5EF",
        dot_portrait="#B9AEFA",   # violet -- distinct hue from the blue chrome
        dot_traveler="#FF5F56",   # reuse existing red already in palette
        live="#FF5F56",
        dots_key="dark_dots",
    ),
    "light": dict(
        name="light",
        bg="#F4F6F9",
        panel="#FFFFFF",
        border="#D7DEE6",
        chrome="#1B6FD1",
        chrome_dim="#B7D3F2",
        muted="#5B6B7D",
        bright="#141B24",
        dot_portrait="#6B4FE0",
        dot_traveler="#D8402F",
        live="#D8402F",
        dots_key="light_dots",
    ),
}

ROWS_GROUP1 = [
    ("Subject", "Syed Jafar"),
    ("Role", "Junior Penetration Tester"),
    ("Origin", "Karachi, PK"),
    ("Education", "BS CS - UBIT Karachi"),
    ("Status", "OSCP-track // job-seeking"),
    ("ToolChain", "Kali . Burp . VS Code"),
]
ROWS_GROUP2 = [
    ("Core.Lang", "Python . Bash . C++"),
    ("Core.Recon", "Nmap . Gobuster . Recon-ng"),
    ("Core.Exploit", "Metasploit . Hydra . JtR"),
    ("Core.WebSec", "Burp Suite . OWASP ZAP"),
    ("Core.Infra", "Kali . Windows . PowerShell"),
]
ROWS_GROUP3 = [
    ("Grid.Mail", "syedjafar.sec@gmail.com"),
    ("Grid.LinkedIn", "/in/syed-jaffar-gdet"),
    ("Grid.GitHub", "/syedj-hacks"),
    ("Grid.TryHackMe", "top 9% . 82-day streak"),
]

# ------------------------------------------------------------- 94 drift bands
N_BANDS = 94
dark_dots = portrait["dark_dots"].astype(float)
noisy = dark_dots + RNG.normal(0, 4.0, dark_dots.shape)
centroids, band_id = kmeans2(noisy, N_BANDS, minit="++", seed=3)

logo1_centroid = logos[LOGO_ORDER[0]].mean(axis=0)


def straight_boundary_metric(coords, labels, grid_w, grid_h):
    """Herfindahl-style concentration of band-boundary edges by column/row.
    Low = boundaries organically spread across many columns/rows (good).
    High = boundaries concentrated in a few columns/rows (a straight grid line)."""
    lut = {}
    for (x, y), b in zip(coords.astype(int), labels):
        lut[(x, y)] = b
    v_cols = {}
    h_rows = {}
    for (x, y), b in lut.items():
        rb = lut.get((x + 1, y))
        if rb is not None and rb != b:
            v_cols[x] = v_cols.get(x, 0) + 1
        db = lut.get((x, y + 1))
        if db is not None and db != b:
            h_rows[y] = h_rows.get(y, 0) + 1

    def herfindahl(counts):
        tot = sum(counts.values())
        if tot == 0:
            return 0.0
        return sum((c / tot) ** 2 for c in counts.values())

    return (herfindahl(v_cols) + herfindahl(h_rows)) / 2


straight_with_noise = straight_boundary_metric(dark_dots, band_id, GRID_W, GRID_H)
_, band_id_nonoise = kmeans2(dark_dots, N_BANDS, minit="++", seed=3)
straight_without_noise = straight_boundary_metric(dark_dots, band_id_nonoise, GRID_W, GRID_H)
print(f"[metric] straight-boundary WITH noise:    {straight_with_noise:.4f}")
print(f"[metric] straight-boundary WITHOUT noise: {straight_without_noise:.4f}  (ablation, expect higher)")

band_members = {b: dark_dots[band_id == b] for b in range(N_BANDS)}
band_centroid = {b: pts.mean(axis=0) for b, pts in band_members.items() if len(pts)}
band_drift = {b: 0.42 * (logo1_centroid - c) for b, c in band_centroid.items()}
coord_to_band = {(int(x), int(y)): int(b) for (x, y), b in zip(dark_dots, band_id)}
band_centroid_arr = np.array([band_centroid[b] for b in range(N_BANDS)])

# ------------------------------------------------------------- 60 intro groups
N_INTRO = 60
order = RNG.permutation(len(dark_dots))
intro_group = np.empty(len(dark_dots), dtype=int)
intro_group[order] = np.arange(len(dark_dots)) % N_INTRO
coord_to_intro = {(int(x), int(y)): int(g) for (x, y), g in zip(dark_dots, intro_group)}


def evenness_metric(coords, groups, n_groups):
    overall_std = coords.std(axis=0)
    overall_extent = np.linalg.norm(overall_std)
    devs = []
    for g in range(n_groups):
        pts = coords[groups == g]
        if len(pts) < 2:
            continue
        d = np.linalg.norm(pts.mean(axis=0) - coords.mean(axis=0))
        devs.append(d)
    return float(np.mean(devs) / overall_extent)


evenness = evenness_metric(dark_dots, intro_group, N_INTRO)
# ablation: spatial-region grouping (row-band strips) instead of interleaved
row_band_groups = (dark_dots[:, 1] // (GRID_H / N_INTRO)).astype(int).clip(0, N_INTRO - 1)
evenness_bad = evenness_metric(dark_dots, row_band_groups, N_INTRO)
print(f"[metric] intro evenness interleaved: {evenness:.4f}")
print(f"[metric] intro evenness spatial-region (ablation): {evenness_bad:.4f}  (expect higher/patchy)")

# ------------------------------------------------------- traveller assignment
TRAVELER_N = 900


def match(a, b):
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    row, col = linear_sum_assignment(cost)
    return b[col]


logo1_pts = logos[LOGO_ORDER[0]]
logo2_raw = logos[LOGO_ORDER[1]]
logo3_raw = logos[LOGO_ORDER[2]]
logo2_pts = match(logo1_pts, logo2_raw)
logo3_pts = match(logo2_pts, logo3_raw)
total_hop1 = np.linalg.norm(logo2_pts - logo1_pts, axis=1).sum()
total_hop2 = np.linalg.norm(logo3_pts - logo2_pts, axis=1).sum()
print(f"[metric] traveller optimal-transport total path length: hop1={total_hop1:.0f}px hop2={total_hop2:.0f}px (grid units)")

np.save(
    "assembled.npy",
    dict(
        band_members=band_members,
        band_drift=band_drift,
        intro_group=intro_group,
        coord_to_band=coord_to_band,
        coord_to_intro=coord_to_intro,
        band_centroid_arr=band_centroid_arr,
        logo1_pts=logo1_pts,
        logo2_pts=logo2_pts,
        logo3_pts=logo3_pts,
    ),
    allow_pickle=True,
)
print("core data assembled -- see build_svg.py for the markup writer")
