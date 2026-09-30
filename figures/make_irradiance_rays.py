#!/usr/bin/env python
"""make_irradiance_rays.py -- the hemisphere of the direct irradiance, on the kernel's set.

    python make_irradiance_rays.py --out ../../PresentationImages

Writes `irradiance_rays.png`: the texel, its normal, the fixed set of directions traced
above it, an occluder that blocks some of them, and the environment map beside it with the
pixel one escaping ray reads.  The two lines of the legend are the only words.

It replaces `Doc/images/diagrams/irradiance_hemisphere.png` for the presentation in one
respect that is not cosmetic: the rays are the ones the kernel traces.  `kernel_directions`
is the loop of `__raygen__renderIrradiance` transcribed, same `z = 1 - (i+0.5)/N`, same
golden turn, same tangent frame, azimuth reduced to one turn before the trigonometry as the
kernel does it.  A drawing of a golden-angle set and a drawing of a scatter look alike to a
room, which is a problem for a pipeline whose whole claim is that no direction is ever drawn
at random: at least the set drawn here is the set.

The pixel the highlighted ray reads is COMPUTED, with the mapping of `sampleEnvmap` from
`deviceProgramsIrradiance.cu` (u from the azimuth, v from asin of the elevation, Z up), not
placed by eye.  Move `AIM` and the mark follows.

The occluder projects to the upper left and the highlighted ray to the upper right, on
purpose: the arrow that leaves the ray has to reach the environment map without crossing
the blocked cluster.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _paths  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle      # noqa: E402

from make_geometry_diagrams import (                                   # noqa: E402
    C_GEOM, C_EDGE, C_CAM, C_DIRECT, C_INDIR,
)

C_INK = C_CAM
C_MUTED = C_EDGE

# -- Canvas, in axes units (10 to the inch) -----------------------------------
# The frame is measured off what is drawn and not written down: the rays reach their full
# radius sideways at grazing incidence and the sphere sticks out where it likes, so a
# rectangle kept by hand is wrong as soon as the view or the occluder moves.
MARGIN = 3.0
DPI = 200

DOME = (44.0, 40.0)                  # centre of the hemisphere drawing
DOME_R = 24.0                        # length of a unit direction, in axes units
QUAD = 1.18                          # half-side of the surface square, in ray lengths
ENVMAP = (80.0, 62.0, 136.0, 90.0)   # x0, y0, x1, y1 of the environment map
LEGEND_Y = (100.0, 94.0)

# -- The geometry the kernel traces -------------------------------------------
GOLDEN_TURN = 0.5 * (np.sqrt(5.0) - 1.0)   # IRR_GOLDEN_TURN, the golden angle as a turn
N_RAYS = 118                  # directions drawn; the kernel traces 512**2 = 262 144
OCC_C = np.array([-0.75, 0.20, 0.55])      # occluder centre, in the texel's frame
OCC_R = 0.32
AIM = np.array([0.70, 0.35, 0.55])         # the direction the highlighted ray follows

ELEV, AZIM = 24.0, -58.0      # the view

SKYBOX = "Doc/images/lighting/skybox_studio.png"


def kernel_directions(n: int) -> np.ndarray:
    """The directions of `__raygen__renderIrradiance`, in the texel's tangent frame."""
    i = np.arange(n, dtype=np.float64)
    z = 1.0 - (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = 2.0 * np.pi * np.mod(i * GOLDEN_TURN, 1.0)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=-1)


def occluded_t(d: np.ndarray, c: np.ndarray, rad: float) -> np.ndarray:
    """Distance to the occluding sphere along each direction, inf where it is missed."""
    b = d @ c
    disc = b * b - (c @ c - rad * rad)
    hit = (disc > 0.0) & (b > 0.0)
    t = np.full(len(d), np.inf)
    t[hit] = b[hit] - np.sqrt(disc[hit])
    return t


def envmap_uv(d: np.ndarray) -> tuple[float, float]:
    """Where a direction reads the environment map: `sampleEnvmap`, verbatim."""
    u = 0.5 - np.arctan2(d[1], d[0]) / (2.0 * np.pi)
    u -= np.floor(u)
    v = 0.5 - np.arcsin(np.clip(d[2], -1.0, 1.0)) / np.pi
    return float(u), float(v)


def projector(elev: float, azim: float):
    ce, se = np.cos(np.radians(elev)), np.sin(np.radians(elev))
    ca, sa = np.cos(np.radians(azim)), np.sin(np.radians(azim))
    right = np.array([-sa, ca, 0.0])
    up = np.array([-ca * se, -sa * se, ce])

    def p(v):
        v = np.asarray(v, dtype=float)
        return np.stack([v @ right, v @ up], axis=-1)

    return p


def draw_dome(ax, proj):
    """The texel, its normal, and the fixed set of rays leaving it."""
    cx, cy = DOME

    def to_ax(p3):
        q = proj(p3) * DOME_R
        return np.stack([q[..., 0] + cx, q[..., 1] + cy], axis=-1)

    quad = np.array([[-QUAD, -QUAD, 0], [QUAD, -QUAD, 0], [QUAD, QUAD, 0], [-QUAD, QUAD, 0]])
    ax.fill(*to_ax(quad).T, facecolor=C_GEOM, edgecolor=C_MUTED, lw=1.0, alpha=0.45, zorder=1)

    d = kernel_directions(N_RAYS)
    t = occluded_t(d, OCC_C, OCC_R)
    blocked = np.isfinite(t)

    occ = to_ax(OCC_C)
    ax.add_patch(Circle((occ[0], occ[1]), OCC_R * DOME_R, facecolor=C_GEOM,
                        edgecolor=C_MUTED, lw=1.0, alpha=0.85, zorder=2))

    origin = to_ax(np.zeros(3))
    for e in to_ax(d[~blocked]):
        ax.plot([origin[0], e[0]], [origin[1], e[1]], color=C_DIRECT,
                lw=1.2, alpha=0.85, solid_capstyle="round", zorder=3)
    ends_hit = to_ax(d[blocked] * t[blocked, None])
    for e in ends_hit:
        ax.plot([origin[0], e[0]], [origin[1], e[1]], color=C_INDIR,
                lw=1.4, alpha=0.95, solid_capstyle="round", zorder=4)
    ax.scatter(ends_hit[:, 0], ends_hit[:, 1], s=14, color=C_INDIR, zorder=5, linewidths=0)

    nrm = to_ax(np.array([0.0, 0.0, 0.62]))
    ax.add_patch(FancyArrowPatch((origin[0], origin[1]), (nrm[0], nrm[1]),
                                 arrowstyle="-|>", mutation_scale=17, color=C_INK,
                                 lw=2.2, shrinkA=0, shrinkB=0, zorder=6))
    ax.text(nrm[0] - 1.7, nrm[1] + 0.4, "n", color=C_INK, fontsize=18,
            fontweight="bold", ha="right", va="bottom", zorder=7)
    ax.scatter([origin[0]], [origin[1]], s=52, color=C_INK, zorder=7, linewidths=0)

    k = int(np.argmax(d @ (AIM / np.linalg.norm(AIM))))
    if blocked[k]:
        raise SystemExit("the highlighted direction is occluded; move AIM or OCC_C")
    tip = to_ax(d[k] * 1.32)
    ax.plot([origin[0], tip[0]], [origin[1], tip[1]], color=C_DIRECT, lw=3.4,
            solid_capstyle="round", zorder=8)
    ax.scatter([tip[0]], [tip[1]], s=72, color=C_DIRECT, zorder=9, linewidths=0)

    pts = np.concatenate([to_ax(quad), to_ax(d), ends_hit, nrm[None, :], tip[None, :],
                          occ + np.array([[-1, -1], [1, 1]]) * OCC_R * DOME_R])
    bbox = (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())
    return d[k], tip, int(blocked.sum()), bbox


def draw_envmap(ax, img, d_hit, tip):
    """The environment map, and the pixel the highlighted direction reads."""
    x0, y0, x1, y1 = ENVMAP
    ax.imshow(img, extent=(x0, x1, y0, y1), aspect="auto", zorder=3,
              interpolation="antialiased")
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none",
                           edgecolor=C_INK, lw=1.6, zorder=4))

    u, v = envmap_uv(d_hit)
    px, py = x0 + u * (x1 - x0), y1 - v * (y1 - y0)
    ax.add_patch(Circle((px, py), 2.0, facecolor="none", edgecolor=C_DIRECT,
                        lw=2.8, zorder=5))
    ax.add_patch(FancyArrowPatch((tip[0], tip[1]), (px, py), arrowstyle="-|>",
                                 mutation_scale=22, color=C_DIRECT, lw=2.6,
                                 connectionstyle="arc3,rad=-0.22", linestyle=(0, (5, 3)),
                                 shrinkA=6, shrinkB=6, zorder=6))
    return u, v


def build(root: Path, out: Path) -> None:
    img = np.asarray(plt.imread(root / SKYBOX), dtype=np.float32)[..., :3]
    step = max(1, int(np.ceil(max(img.shape[:2]) / 1600)))
    img = img[::step, ::step]

    fig = plt.figure(figsize=(20.0, 14.0))          # scratch canvas, cropped below
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 140)

    proj = projector(ELEV, AZIM)
    d_hit, tip, n_blocked, bbox = draw_dome(ax, proj)
    u, v = draw_envmap(ax, img, d_hit, tip)

    for y, col, label in ((LEGEND_Y[0], C_DIRECT,
                           "unoccluded: direct irradiance from the environment map"),
                          (LEGEND_Y[1], C_INDIR,
                           "blocked: indirect, radiance queried from the NeRF")):
        ax.plot([5.0, 12.0], [y, y], color=col, lw=3.4, solid_capstyle="round")
        ax.text(14.5, y, label, fontsize=17, color=C_INK, ha="left", va="center")

    # The legend sets the top; a text height in axes units is fontsize/72 * 10.
    top = LEGEND_Y[0] + 17.0 / 72.0 * 10.0 * 0.8
    x0 = min(bbox[0], ENVMAP[0], 5.0) - MARGIN
    x1 = max(bbox[2], ENVMAP[2]) + MARGIN
    y0 = min(bbox[1], ENVMAP[1]) - MARGIN
    y1 = max(top, ENVMAP[3]) + MARGIN
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    fig.set_size_inches((x1 - x0) / 10.0, (y1 - y0) / 10.0)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"  frame {x1 - x0:.1f} x {y1 - y0:.1f} units, ratio {(x1 - x0) / (y1 - y0):.3f}")
    print(f"  {N_RAYS} directions drawn, {n_blocked} blocked by the occluder")
    print(f"  highlighted ray reads (u, v) = ({u:.4f}, {v:.4f})")
    print(f"  + {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="../../PresentationImages")
    ap.add_argument("--assets", default=None,
                    help="folder the asset paths are relative to (default: the repo root)")
    a = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]
    root = Path(a.assets) if a.assets else repo
    build(root, Path(a.out).resolve() / "irradiance_rays.png")


if __name__ == "__main__":
    main()
