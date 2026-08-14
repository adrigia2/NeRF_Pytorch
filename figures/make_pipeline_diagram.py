#!/usr/bin/env python
"""make_pipeline_diagram.py -- block diagram of the pipeline (figure 3.1).

    python make_pipeline_diagram.py --out ../Doc/images/diagrams

Writes `pipeline_overview.png`.

The diagram is DATA, not drawing: it is edited by changing `NODES` and `EDGES` below, and
the layout follows on its own.  Each node declares the column and row it sits in, its
width in columns, and which group it belongs to (the group only decides the colour).  The
arrows are pairs of identifiers: there are no hand-written coordinates anywhere, so moving
a block does not require rearranging the arrows.

The columns are equally spaced and so are the rows; if a block turns out too narrow for
its text, the thing to change is `COL_W` or the text, not the position.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({"font.size": 13})

DPI = 190

# ── Appearance ───────────────────────────────────────────────────────────────
# The measurements are in INCHES and the figure is sized so that one data unit is exactly
# one inch.  That makes it possible to reason about a box's width in typographic points:
# with arbitrary units the text overflows as soon as figsize changes, which is exactly
# what used to happen.
COL_W, COL_GAP = 2.05, 0.62      # column width and gap between columns
ROW_H, ROW_GAP = 0.88, 0.34      # row height and gap between rows
FONT = 11                        # at 11pt a line of ~20 characters fits in COL_W
MARGIN = 0.22

STYLE = {
    "input":  dict(fc="#eef2f6", ec="#8a99a8", tc="#25303a"),
    "nerf":   dict(fc="#e7dcf7", ec="#8452c9", tc="#3d2a63"),
    "optix":  dict(fc="#d9e8f7", ec="#2f7ec4", tc="#123a5c"),
    "fit":    dict(fc="#fdeacd", ec="#d9932a", tc="#5c3d0c"),
    "output": dict(fc="#dcf0dc", ec="#2ca02c", tc="#12470f"),
}

# id, column, row, width in columns, group, text
NODES = [
    ("img",   0, 2.0, 1, "input",  "Multi-view images\nwith known poses"),
    ("mesh",  0, 1.0, 1, "input",  "Scene mesh\n(UV-unwrapped)"),
    ("nmap",  0, 0.0, 1, "input",  "Object-space\nnormal map"),

    ("nerf",  1, 2.0, 1, "nerf",   "NeRF training\n(Step 2)"),
    ("optix", 1, 0.5, 1, "optix",  "OptiX passes\n(Steps 1 and 3)"),

    ("rad",   2, 2.0, 1, "nerf",   "Radiance queries\n(indirect, sky)"),
    ("geo",   2, 1.0, 1, "optix",  "Geometry, visibility\nand color"),
    ("light", 2, 0.0, 1, "optix",  "Irradiance and\nspecular cones"),

    ("fit",   3, 1.0, 1, "fit",    "PBR fit\n(Step 4)"),

    ("base",  4, 2.0, 1, "output", "Base color"),
    ("met",   4, 1.0, 1, "output", "Metallic"),
    ("rough", 4, 0.0, 1, "output", "Roughness"),
]

EDGES = [
    ("img", "nerf"), ("mesh", "optix"), ("nmap", "optix"), ("img", "optix"),
    ("nerf", "rad"), ("optix", "geo"), ("optix", "light"), ("rad", "light"),
    ("rad", "fit"), ("geo", "fit"), ("light", "fit"),
    ("fit", "base"), ("fit", "met"), ("fit", "rough"),
]


def box_rect(col, row, span):
    x = col * (COL_W + COL_GAP)
    y = row * (ROW_H + ROW_GAP)
    return x, y, COL_W * span + COL_GAP * (span - 1), ROW_H


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="../Doc/images/diagrams")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rects = {}
    for nid, col, row, span, group, text in NODES:
        rects[nid] = (*box_rect(col, row, span), group, text)

    xs = [r[0] for r in rects.values()] + [r[0] + r[2] for r in rects.values()]
    ys = [r[1] for r in rects.values()] + [r[1] + r[3] for r in rects.values()]
    w, h = max(xs) - min(xs), max(ys) - min(ys)

    # One data unit = one inch: figsize and limits are consistent by construction
    fig, ax = plt.subplots(figsize=(w + 2 * MARGIN, h + 2 * MARGIN))

    for a, b in EDGES:
        xa, ya, wa, ha, _, _ = rects[a]
        xb, yb, wb, hb, _, _ = rects[b]
        if abs(xa - xb) < 1e-9:
            # Same column: the arc goes vertical, otherwise it leaves on the right and
            # comes back on the left, going around the block.
            up = yb > ya
            p0 = (xa + wa / 2, ya + (ha if up else 0.0))
            p1 = (xb + wb / 2, yb + (0.0 if up else hb))
            # Pronounced curvature: a straight vertical connector would pass behind the
            # intervening block and read as two separate arcs.
            rad = 0.55
        else:
            p0, p1, rad = (xa + wa, ya + ha / 2), (xb, yb + hb / 2), 0.06
        ax.add_patch(FancyArrowPatch(
            p0, p1, arrowstyle="-|>,head_width=0.18,head_length=0.36",
            connectionstyle=f"arc3,rad={rad}", color="#7c8894", lw=1.3,
            shrinkA=2, shrinkB=3, zorder=1))

    for nid, (x, y, bw, bh, group, text) in rects.items():
        s = STYLE[group]
        ax.add_patch(FancyBboxPatch(
            (x, y), bw, bh, boxstyle="round,pad=0.0,rounding_size=0.12",
            facecolor=s["fc"], edgecolor=s["ec"], linewidth=1.6, zorder=2))
        ax.text(x + bw / 2, y + bh / 2, text, ha="center", va="center",
                color=s["tc"], fontsize=FONT, zorder=3, linespacing=1.35)

    ax.set_xlim(min(xs) - MARGIN, max(xs) + MARGIN)
    ax.set_ylim(min(ys) - MARGIN, max(ys) + MARGIN)
    ax.set_aspect("equal")
    ax.axis("off")
    path = out / "pipeline_overview.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}  ({len(NODES)} blocks, {len(EDGES)} arcs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
