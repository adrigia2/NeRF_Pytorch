#!/usr/bin/env python
"""make_full_model_slide.py -- the full model, its two terms, and what is still unknown.

    python make_full_model_slide.py --out ../../PresentationImages

Writes `full_model.png`: the model set as an equation whose colours do the explaining, and
under it the family of cones the roughness selects from.

Three things the typed version of this slide could not do.

  THE TWO TERMS ARE COLOURED, NOT NAMED.  Amber is the diffuse half and blue the specular
  half, the same two colours `make_pbr_model_diagram.py` uses for them in the thesis, so the
  sentence "the colour we see is the combination of two elements" is the first thing the eye
  gets and the words under the equation only confirm it.

  THE UNKNOWNS ARE MARKED ON THE SYMBOLS.  `m`, `a` and `r` carry a highlight, and the
  closing question of the slide ("retrieve the missing parameters") then points at something
  that is actually on the screen.  The highlight is a mark and not a colour, so it stacks on
  top of the term colour instead of competing with it.

  THE MAPPING FROM ROUGHNESS TO APERTURE IS DRAWN.  "0 roughness is 0 degrees and 1
  roughness is 180" is a sentence about a shape, and a row of cones says it without being
  read.  The apertures shown are round numbers, but the count printed beside them is the
  real one: `spec_cone_apertures_deg` of the operational configuration, 14 candidates
  refined where the lobe is narrow.

A note on the symbol.  This slide writes `m` for the metallic and the diffuse weight as
`1 - m`.  The thesis and `pbr_solver.py` write the DIFFUSE weight as `x`, with
`metallic = 1 - x`; the model is the same one either way, but a slide that said "x =
metallic" would contradict the document it presents, which is why `x` does not appear here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _paths  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Wedge  # noqa: E402

from make_geometry_diagrams import C_EDGE, C_CAM                       # noqa: E402

C_INK = C_CAM
C_MUTED = C_EDGE
C_DIFF = "#c98a20"             # the diffuse term, the amber of the thesis PBR diagram
C_SPEC = "#2171b5"             # the specular term, the blue of the cone ramp
C_MARK = "#ffe27a"             # highlight behind a symbol still to be recovered

# The grid the bake actually evaluates, from RenderConfig in the operational run.
APERTURES = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0,
             60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0]
SHOWN = [0.0, 45.0, 90.0, 135.0, 180.0]     # round numbers, one per quarter of roughness

# -- Layout, in axes units (10 to the inch) -----------------------------------
MARGIN = 3.0
DPI = 200

EQ = (6.0, 74.0)               # left edge and baseline of the equation
EQ_SIZE = 30.0
KEY_Y = (62.0, 56.0, 50.0)     # the three lines under it
GLYPH_Y = 30.0                 # centre of the row of cones
GLYPH_R = 9.0                  # radius of a cone glyph
GLYPH_X = (14.0, 38.0, 62.0, 86.0, 110.0)
VIEW_DEG = 50.0                # angle of the view direction from the normal


def pieces():
    """The equation, in the order it is written, as (text, colour, marked)."""
    return (("C = ", C_INK, False),
            ("(1 - ", C_DIFF, False), ("m", C_DIFF, True), (")", C_DIFF, False),
            (" · ", C_DIFF, False),
            ("a", C_DIFF, True), ("/π · E", C_DIFF, False),
            ("  +  ", C_INK, False),
            ("m", C_SPEC, True), (" · L(", C_SPEC, False),
            ("r", C_SPEC, True), (")", C_SPEC, False))


def draw_equation(fig, ax):
    """Place the pieces left to right, measuring each one, and mark the unknowns."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    x = EQ[0]
    for txt, col, marked in pieces():
        t = ax.text(x, EQ[1], txt, fontsize=EQ_SIZE, color=col, ha="left",
                    va="baseline", zorder=5)
        bb = t.get_window_extent(renderer=rend)
        (x0, y0), (x1, y1) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
        if marked:
            ax.add_patch(FancyBboxPatch((x0 - 0.5, y0 - 0.8), (x1 - x0) + 1.0,
                                        (y1 - y0) + 1.6,
                                        boxstyle="round,pad=0,rounding_size=1.2",
                                        facecolor=C_MARK, edgecolor="none", zorder=4))
        x = x1
    return x


def draw_keys(ax):
    ax.text(EQ[0], KEY_Y[0],
            "diffuse: the colour of the object under the irradiance we just computed",
            fontsize=17, color=C_DIFF, ha="left", va="center")
    ax.text(EQ[0], KEY_Y[1],
            "specular: the radiance that comes back from the reflected cone",
            fontsize=17, color=C_SPEC, ha="left", va="center")
    ax.add_patch(FancyBboxPatch((EQ[0], KEY_Y[2] - 2.0), 5.0, 4.0,
                                boxstyle="round,pad=0,rounding_size=1.0",
                                facecolor=C_MARK, edgecolor="none", zorder=3))
    ax.text(EQ[0] + 7.0, KEY_Y[2],
            "still to be recovered:  m = metallic,   a = albedo,   r = roughness",
            fontsize=17, color=C_INK, ha="left", va="center")


def cone_glyph(ax, cx, aperture):
    """The texel, its normal, a view direction, its reflection, and the cone around it."""
    base = np.array([cx, GLYPH_Y - GLYPH_R * 0.62])
    ax.plot([cx - GLYPH_R * 1.12, cx + GLYPH_R * 1.12], [base[1], base[1]],
            color=C_MUTED, lw=1.4, zorder=2)

    v = np.radians(90.0 + VIEW_DEG)          # incoming, upper left
    r_ang = 90.0 - VIEW_DEG                  # reflected, upper right
    ax.add_patch(FancyArrowPatch(base + GLYPH_R * np.array([np.cos(v), np.sin(v)]), base,
                                 arrowstyle="-|>", mutation_scale=12, color=C_MUTED,
                                 lw=1.5, shrinkA=0, shrinkB=0, zorder=3))

    if aperture > 0.0:
        half = aperture / 2.0
        t1, t2 = max(0.0, r_ang - half), min(180.0, r_ang + half)
        ax.add_patch(Wedge(base, GLYPH_R, t1, t2, facecolor=C_SPEC, alpha=0.22,
                           edgecolor=C_SPEC, lw=1.0, zorder=3))
    d = np.radians(r_ang)
    ax.add_patch(FancyArrowPatch(base, base + GLYPH_R * np.array([np.cos(d), np.sin(d)]),
                                 arrowstyle="-|>", mutation_scale=13, color=C_SPEC,
                                 lw=2.4, shrinkA=0, shrinkB=0, zorder=5))
    ax.plot([cx, cx], [base[1], base[1] + GLYPH_R * 0.86], color=C_INK, lw=1.6, zorder=4)
    ax.scatter([cx], [base[1]], s=26, color=C_INK, zorder=6, linewidths=0)

    ax.text(cx, base[1] - 4.0, f"r = {aperture / 180.0:g}", fontsize=16, color=C_INK,
            ha="center", va="top")
    ax.text(cx, base[1] - 9.5, f"{aperture:g}°", fontsize=15, color=C_SPEC,
            ha="center", va="top")


def build(out: Path) -> None:
    fig = plt.figure(figsize=(20.0, 14.0))       # scratch canvas, cropped below
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 140)

    eq_right = draw_equation(fig, ax)
    draw_keys(ax)
    for cx, ap in zip(GLYPH_X, SHOWN):
        cone_glyph(ax, cx, ap)

    # Under the row and not beside it: beside, the note alone made the figure a third
    # wider than the drawing in it, which on a 4:3 page costs height everywhere else.
    ax.text(EQ[0], 5.0,
            f"the bake evaluates {len(APERTURES)} of these, refined where the lobe is narrow",
            fontsize=14, color=C_MUTED, ha="left", va="center")

    x0 = EQ[0] - MARGIN
    x1 = max(eq_right, GLYPH_X[-1] + GLYPH_R * 1.12) + MARGIN
    y0 = 5.0 - 2.0 - MARGIN
    y1 = EQ[1] + EQ_SIZE / 72.0 * 10.0 + MARGIN
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    fig.set_size_inches((x1 - x0) / 10.0, (y1 - y0) / 10.0)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"  apertures {APERTURES}")
    print(f"  frame {x1 - x0:.1f} x {y1 - y0:.1f} units, ratio {(x1 - x0) / (y1 - y0):.3f}")
    print(f"  + {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="../../PresentationImages")
    a = ap.parse_args()
    build(Path(a.out).resolve() / "full_model.png")


if __name__ == "__main__":
    main()
