#!/usr/bin/env python
"""make_optix_pipeline_diagram.py -- schematic of the OptiX programmable pipeline.

    python make_optix_pipeline_diagram.py --out ../Doc/images/diagrams

Writes `optix_pipeline.png`, the figure of Section 2.2.5 (`fig:optix-pipeline`).

The caption was written before the picture, and it names five things the drawing has to
contain: the ray generation program that spawns the rays, the traversal through the GAS,
the closest-hit program invoked on intersection, the miss program invoked on escape, and
the SBT binding geometry to programs and to per-primitive data.  The layout below is
organised around exactly those five and nothing else.

TWO STRUCTURAL DECISIONS, both of them about what the picture is claiming:

  * The SBT is drawn OFF the ray's path, as a table on its own with arrows reaching into
    the two shading blocks.  It is the only one of the five that is not a stage the ray
    passes through: it is the lookup that decides WHICH program runs and with what data.
    Putting it in the chain would make it read as a sixth stage between traversal and
    shading, which is precisely the misunderstanding the caption is trying to prevent.

  * The payload goes BACK to the ray generation program, as a return arrow, and the
    output buffer hangs off raygen and not off the two shading blocks.  That is how OptiX
    actually works, and it is what makes the raygen program the place where a pass's
    result is assembled -- which is what every generator in Chapter 3 does.

The program names are the real ones of this project (`depthMap/cuda/devicePrograms.cu`)
and the three record types are those of `depthMap/src/OptixActor.cpp`, so a reader can go
from this figure to the code without an intermediate translation step.

The diagram is DATA: it is edited by changing NODES, EDGES and SBT_LINKS, and the layout
follows.  Same construction as make_pipeline_diagram.py, from which the geometry is
borrowed: one data unit is one inch, so a box width can be reasoned about in points.
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
# Measurements in INCHES, and the figure sized so one data unit is one inch: that is what
# makes it possible to check that a label fits its box before rendering.
#
# SIZED FOR THE PAGE.  The figure is included at \linewidth, about 5.7 in, so a label
# lands on paper at its point size times 5.7 / (figure width in inches).  At the 1.95 in
# columns this started from the figure came out 10.4 in wide and 10.5 pt text arrived at
# 5.2 pt, illegible in print while looking fine on screen.
#
# Here the usual fix, shrinking the figure and leaving the fonts alone, does NOT work: one
# data unit is one inch and the boxes are measured in data units while the text is in
# points, so a narrower column holds the same string in less space and the label overflows.
# The only thing that buys width back is shorter strings, which is why the labels below are
# as terse as they are.  Any text added here has to be paid for in column width.
# The arithmetic that decides everything here: a label lands on paper at roughly
#     684 / (number of columns x longest string in characters)
# points, independent of the font size chosen, because a bigger font needs a wider column
# and the two cancel.  At five columns and eighteen characters that ceiling is 7.6 pt, and
# the figure measured 5.2 pt.  Only two things move it: fewer columns and shorter strings,
# which is why `optixLaunch` is an annotation rather than a block of its own and why the
# labels below are terse.  Any text added here is paid for by every other label.
# COL_W is set by the longest label, which is `__closesthit__`: the monospace math face is
# noticeably wider per character than the sans one, so the column has to be sized on it
# and not on the longer-looking prose labels.
COL_W, COL_GAP = 1.66, 0.40
ROW_H, ROW_GAP = 0.74, 0.26
FONT = 10.0
MONO = 9.0                       # program names, in a monospace face
MARGIN = 0.20
ELBOW = 0.52                     # how far past the blocks the payload route runs

# Groups only decide colour.  "host" is what the CPU does, "device" the programs the user
# writes, "fixed" the stage OptiX and the RT cores own and the user cannot program, and
# "table" the SBT, which is data rather than code.
STYLE = {
    "host":   dict(fc="#eef2f6", ec="#8a99a8", tc="#25303a"),
    "device": dict(fc="#d9e8f7", ec="#2f7ec4", tc="#123a5c"),
    "fixed":  dict(fc="#e4e9ee", ec="#5f6c79", tc="#25303a"),
    "table":  dict(fc="#fdeacd", ec="#d9932a", tc="#5c3d0c"),
    "out":    dict(fc="#dcf0dc", ec="#2ca02c", tc="#12470f"),
}

# id, column, row, width in columns, group, text
#
# The SBT sits in the LAST column, beside the two shading blocks it binds, and not under
# the chain: its links are then short and horizontal, and the whole strip below the
# diagram stays free for the payload route back to raygen.  Every arrangement that put
# the table under the chain had those two crossing each other.
NODES = [
    ("raygen", 0, 1.0, 1, "device", "Ray generation\n$\\tt \\_\\_raygen\\_\\_$"),
    ("trav",   1, 1.0, 1, "fixed",  "$\\tt GAS$ traversal\n(BVH, RT cores)"),
    ("hit",    2, 2.15, 1, "device", "Closest hit\n$\\tt \\_\\_closesthit\\_\\_$"),
    ("miss",   2, -0.15, 1, "device", "Miss\n$\\tt \\_\\_miss\\_\\_$"),
    ("buf",    0, 2.15, 1, "out",    "Output buffers"),
    # Wider than one column: the table carries the longest strings in the figure, and it
    # is the only block that can afford the width, sitting at the end of a row.
    ("sbt",    3, 0.24, 1.15, "table", ""),        # drawn by hand, see draw_sbt
]

# `optixLaunch` is an annotation and not a block: as a sixth box it cost a whole column,
# and a column is worth about two points of label size on the printed page.  It is the
# host call that starts the chain, not a stage the ray passes through, so an entry arrow
# carries it perfectly well.
# Two lines: on one it became the longest string in the figure and set the scale for
# everything else, which is exactly the cost this block was removed to avoid.
LAUNCH = "$\\tt optixLaunch$\n2D thread grid"
LAUNCH_W = 1.15                  # how much room the annotation needs, in data units

# The ray's path.  The returns are a separate list because they are drawn differently:
# the payload coming home is not another stage of the chain.
EDGES = [
    ("raygen", "trav"),
    ("trav", "hit"),
    ("trav", "miss"),
]
# (source, above?) -- the hit payload is routed over the diagram and the miss one under,
# so the two never share a segment and neither crosses the traversal block.
RETURNS = [("hit", True), ("miss", False)]

# Two lines, because on one this was the single longest string in the figure and it set
# the width of the whole table by itself.
SBT_TITLE = "Shader binding\ntable ($\\tt SBT$)"
# The SBT table: (row label, what the record carries), top to bottom.
SBT_ROWS = [
    (r"$\tt raygen$ record", "launch entry point"),
    (r"$\tt miss$ record", "program for\nescaping rays"),
    (r"$\tt hitgroup$ record", "program, plus the\nper-primitive data:\nvertices, indices,\nmaterial"),
]
# Which SBT row points at which block.
SBT_LINKS = [(1, "miss"), (2, "hit")]

C_ARROW = "#7c8894"
C_RET = "#8452c9"                # the payload, in the violet the other figures use
C_SBT = "#d9932a"


def box_rect(col, row, span):
    x = col * (COL_W + COL_GAP)
    y = row * (ROW_H + ROW_GAP)
    return x, y, COL_W * span + COL_GAP * (span - 1), ROW_H


def arrow(ax, p0, p1, *, color, rad, lw=1.4, ls="-", zorder=1):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>,head_width=0.17,head_length=0.34",
        connectionstyle=f"arc3,rad={rad}", color=color, lw=lw, linestyle=ls,
        shrinkA=2, shrinkB=3, zorder=zorder))


def elbow(ax, p0, p1, y_via, *, color, lw=1.3, ls=(0, (5, 3)), zorder=1):
    """Route p0 -> (p0.x, y_via) -> (p1.x, y_via) -> p1, arrowhead on the last leg.

    A right-angled route and not an arc: an arc from the shading blocks back to raygen
    has to bow so far to clear the traversal block that it stops reading as a return and
    starts reading as another branch.  The corners are what say "this comes back".
    """
    ax.plot([p0[0], p0[0], p1[0]], [p0[1], y_via, y_via],
            color=color, lw=lw, ls=ls, zorder=zorder,
            solid_capstyle="round", dash_capstyle="round")
    ax.add_patch(FancyArrowPatch(
        (p1[0], y_via), p1, arrowstyle="-|>,head_width=0.17,head_length=0.34",
        connectionstyle="arc3,rad=0.0", color=color, lw=lw, linestyle=ls,
        shrinkA=0, shrinkB=3, zorder=zorder))


SBT_TITLE_H = 0.62               # two lines, see SBT_TITLE
SBT_NAME_H = 0.26                # the record name
SBT_LINE_H = 0.21                # one line of the description


def sbt_row_heights() -> list:
    """One height per row, from how many lines its description takes.  A single height
    for all of them made the four-line hitgroup row overflow into the one above."""
    return [SBT_NAME_H + SBT_LINE_H * (what.count("\n") + 1) + 0.14
            for _, what in SBT_ROWS]


def sbt_height() -> float:
    return SBT_TITLE_H + sum(sbt_row_heights())


def draw_sbt(ax, rect) -> list:
    """The SBT as a three-row table.  Returns the y centre of each row, so the links can
    leave from the row that owns them instead of from the block as a whole: which record
    binds which program is the entire content of the SBT, and an arrow from the middle of
    the table would say nothing.

    The rows stack the record name over what the record carries instead of putting them
    side by side: in one column's width the two ran into each other.
    """
    x, y, w, _ = rect
    h = sbt_height()
    s = STYLE["table"]
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.10",
        facecolor=s["fc"], edgecolor=s["ec"], linewidth=1.6, zorder=2))
    ax.text(x + w / 2, y + h - SBT_TITLE_H / 2, SBT_TITLE, ha="center", va="center",
            color=s["tc"], fontsize=FONT - 0.5, zorder=3, linespacing=1.25)

    centres = []
    top = y + h - SBT_TITLE_H
    for (name, what), rh in zip(SBT_ROWS, sbt_row_heights()):
        centres.append(top - rh / 2)
        ax.plot([x + 0.08, x + w - 0.08], [top] * 2,
                color=s["ec"], lw=0.7, alpha=0.55, zorder=3)
        ax.text(x + w / 2, top - SBT_NAME_H / 2 - 0.04, name, ha="center", va="center",
                color=s["tc"], fontsize=MONO, zorder=3)
        ax.text(x + w / 2, top - SBT_NAME_H - (rh - SBT_NAME_H) / 2 + 0.02, what,
                ha="center", va="center", color="#7a6440", fontsize=MONO - 1.0,
                zorder=3, linespacing=1.2)
        top -= rh
    return centres


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="../Doc/images/diagrams")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rects = {nid: (*box_rect(col, row, span), group, text)
             for nid, col, row, span, group, text in NODES}

    sbt_rect = rects["sbt"][:4]
    xs = [r[0] for r in rects.values()] + [r[0] + r[2] for r in rects.values()]
    ys = [r[1] for r in rects.values()] + [r[1] + r[3] for r in rects.values()]
    # The SBT block is taller than one row, so its top has to be accounted for by hand.
    ys.append(sbt_rect[1] + sbt_height())
    # The entry annotation sits left of every block and has to enter the extent BEFORE
    # figsize is computed from it: added afterwards it only widened the limits, and with
    # an equal aspect that shrinks the axes instead, so a data unit stopped being an inch
    # and every label overflowed its box.
    xs.append(rects["raygen"][0] - COL_GAP * 1.5 - LAUNCH_W)
    # The two payload routes run outside the blocks, so they set the extent too.
    y_top = max(ys) + ELBOW
    y_bot = min(ys) - ELBOW
    ys += [y_top, y_bot]
    w, h = max(xs) - min(xs), max(ys) - min(ys)

    fig, ax = plt.subplots(figsize=(w + 2 * MARGIN, h + 2 * MARGIN))

    for a, b in EDGES:
        xa, ya, wa, ha, _, _ = rects[a]
        xb, yb, wb, hb, _, _ = rects[b]
        # A branch that also changes row leaves from the right edge and curves: leaving
        # from the corner would cross the block it starts from.
        arrow(ax, (xa + wa, ya + ha / 2), (xb, yb + hb / 2),
              color=C_ARROW, rad=0.0 if abs(ya - yb) < 1e-9 else 0.14)

    # The payload home, dashed and in its own colour so it does not read as another stage
    # of the chain.  One route over the diagram and one under: they then share no segment
    # with each other, with the chain, or with the SBT links.
    xr, yr, wr, hr, _, _ = rects["raygen"]
    # The two routes come down onto DIFFERENT points of raygen: landing on the same one
    # put their vertical legs on a single line running the full height of the figure,
    # which read as a box drawn around the diagram rather than as two returns.
    for (src, above), frac in zip(RETURNS, (0.34, 0.66)):
        xa, ya, wa, ha, _, _ = rects[src]
        elbow(ax, (xa + wa / 2, ya + (ha if above else 0.0)),
              (xr + wr * frac, yr + (hr if above else 0.0)),
              y_top if above else y_bot, color=C_RET)
    ax.text((xr + wr * 0.34 + rects["hit"][0]) / 2, y_top + 0.10, "payload",
            ha="center", va="bottom", color=C_RET, fontsize=MONO)

    # raygen sits in column 0 now, and the buffers directly above it, so the arrow is a
    # short vertical instead of the arc it needed when a launch block held that column.
    xb, yb, wb, hb, _, _ = rects["buf"]
    arrow(ax, (xr + wr / 2, yr + hr), (xb + wb / 2, yb), color=C_RET, lw=1.3, rad=0.0)

    # The host call that starts everything, as an entry arrow into raygen.
    x_entry = xr - COL_GAP * 1.5
    arrow(ax, (x_entry, yr + hr / 2), (xr, yr + hr / 2), color=C_ARROW, rad=0.0)
    ax.text(x_entry - 0.08, yr + hr / 2, LAUNCH, ha="right", va="center",
            color=STYLE["host"]["tc"], fontsize=MONO)

    # The SBT links: from the row that holds the record to the block it binds.  Straight
    # and horizontal, which is the point of putting the table in this column.
    centres = draw_sbt(ax, sbt_rect)
    sx, _, sw, _ = sbt_rect
    # The two links cross once, and they have to: in the table the miss record comes
    # before the hitgroup one, which is the order the SBT is actually built in
    # (OptixActor.cpp), while on the diagram the hit branch is the upper one.  Rather
    # than reorder either to avoid it, they are bowed apart so the crossing reads as a
    # crossing and not as a junction.
    for (k, target), rad in zip(SBT_LINKS, (0.30, -0.30)):
        xt, yt, wt, ht, _, _ = rects[target]
        arrow(ax, (sx, centres[k]), (xt + wt, yt + ht / 2),
              color=C_SBT, rad=rad, lw=1.3, ls=(0, (4, 2.5)), zorder=4)

    for nid, (x, y, bw, bh, group, text) in rects.items():
        if nid == "sbt":
            continue
        s = STYLE[group]
        ax.add_patch(FancyBboxPatch(
            (x, y), bw, bh, boxstyle="round,pad=0.0,rounding_size=0.10",
            facecolor=s["fc"], edgecolor=s["ec"], linewidth=1.6, zorder=2))
        ax.text(x + bw / 2, y + bh / 2, text, ha="center", va="center",
                color=s["tc"], fontsize=FONT, zorder=3, linespacing=1.4)

    # The two outcomes named on the branches, which is the one thing the boxes alone do
    # not say: what decides whether a ray ends up in one or the other.
    xt, yt, wt, ht, _, _ = rects["trav"]
    ax.text(xt + wt + COL_GAP / 2, yt + ht + 0.46, "hit", ha="center", va="center",
            color="#123a5c", fontsize=MONO, zorder=4,
            bbox=dict(fc="white", ec="none", pad=1.0))
    ax.text(xt + wt + COL_GAP / 2, yt - 0.46, "no hit", ha="center", va="center",
            color="#123a5c", fontsize=MONO, zorder=4,
            bbox=dict(fc="white", ec="none", pad=1.0))

    ax.set_xlim(min(xs) - MARGIN, max(xs) + MARGIN)
    ax.set_ylim(min(ys) - MARGIN, max(ys) + MARGIN)
    ax.set_aspect("equal")
    ax.axis("off")
    path = out / "optix_pipeline.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # Every element the caption names has to be on the drawing.  Cheap to check and it is
    # the only failure mode that matters here: a diagram that quietly drops one of them.
    drawn = " ".join(t for *_, t in rects.values()) + " " + SBT_TITLE + " " + \
            " ".join(n + w for n, w in SBT_ROWS)
    for needed in ("raygen", "GAS", "closesthit", "miss", "SBT", "hitgroup"):
        assert needed in drawn, f"the caption names {needed!r} and the figure does not"
    print(f"  + {path}  ({len(rects)} blocks, {len(EDGES)} arcs, "
          f"{len(RETURNS) + 1} payload arcs, {len(SBT_LINKS)} SBT links)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
