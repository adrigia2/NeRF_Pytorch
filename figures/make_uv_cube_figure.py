#!/usr/bin/env python
"""make_uv_cube_figure.py -- the cube and its UV atlas, side by side.

    python make_uv_cube_figure.py --out ../../PresentationImages

Writes one PNG:

  uv_cube_atlas.png   left, the cube in object space with the position of each of its
                      eight vertices; right, its UV atlas with the six faces coloured
                      and the (u, v) of every corner of the unwrap.

The figure exists to make one sentence of the thesis visible: the pipeline needs a
one-to-one mapping between valid texels and surface patches.  A cube is the smallest
object that shows what that costs, because it forces the seam duplication into the
open.  Six faces of four corners are twenty-four face corners, but the cube only has
eight vertices, and the cross unwrap glues the shared ones back together and leaves
fourteen distinct points in UV space.  Two of the eight vertices lie on the cut and
therefore appear three times, at u = 0 and again at u = 1.  The script prints that
table; it is not written into the drawing by hand.

**The unwrap is folded, not transcribed.**  Only the face loops and the order of the
four side faces in the band are given.  Which vertex lands on which corner of which
cell is then derived, by taking the edge a cell shares with its neighbour and walking
around the face loop.  Four checks run before anything is drawn:

  1. every face loop is really outward-CCW (its cross product is the axis in its name);
  2. the four vertices a cell receives are the four vertices of its face;
  3. two cells that touch a UV point claim the same vertex there -- which is what makes
     the layout a fold rather than six loose quads;
  4. the corners of a cell, taken CCW in UV, are a cyclic *rotation* of the outward-CCW
     face loop, never a reversal.

Check 4 is the one with teeth.  Walking the band the other way around the cube, e.g.
BAND = ("-X", "+Y", "+X", "-Y"), draws a picture that looks identical and is wrong:
every island comes out mirrored, which no real unwrapper emits and which would flip a
normal map.  The check refuses to draw it.

Two conventions worth knowing.  The cube is [-1, 1]^3 and not the unit cube on purpose:
positions inside [0, 1] would read as if they were already the UVs.  And the v ticks
print 0.33 and 0.67 while the geometry uses exact thirds -- only the label is rounded.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _paths  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
import matplotlib.patheffects as pe                                    # noqa: E402

# The orthographic projector and the painter-ordered Scene are the ones every geometric
# diagram of the thesis is drawn with; see the docstring of make_geometry_diagrams for
# why they exist instead of mplot3d.  Importing them keeps the two from diverging.
from make_geometry_diagrams import Projector, Scene, C_EDGE            # noqa: E402

# 240 dpi and a wide canvas, not the 190 of the thesis diagrams: this one is for a slide,
# projected at 1920 px or more.  Same reasoning as make_presentation_magnitude.py.
DPI = 240
# 16:9, the aspect of the slide.  The height is what sets the scale of the atlas, since
# its axes are equal-aspect: two corners of the cross are a quarter of the atlas apart,
# and a "(0.00, 0.67)" label is nearly that wide, so a short canvas makes neighbouring
# corner labels collide however they are offset.
FIGSIZE = (15.0, 8.4)

C_INK = "#222222"
C_HIDDEN = "#aab4be"    # edges of the cube facing away from the observer
C_FRAME = "#9aa7b4"     # the empty part of the UV square

# -- The cube ---------------------------------------------------------------
LO, HI = -1.0, 1.0
VERTS = np.array([
    (LO, LO, LO),   # V0
    (HI, LO, LO),   # V1
    (HI, HI, LO),   # V2
    (LO, HI, LO),   # V3
    (LO, LO, HI),   # V4
    (HI, LO, HI),   # V5
    (HI, HI, HI),   # V6
    (LO, HI, HI),   # V7
], dtype=float)

# Outward-CCW loops.  Check 1 verifies the claim rather than trusting it.
FACES = {
    "-Z": (0, 3, 2, 1),
    "+Z": (4, 5, 6, 7),
    "-Y": (0, 1, 5, 4),
    "+X": (1, 2, 6, 5),
    "+Y": (2, 3, 7, 6),
    "-X": (3, 0, 4, 7),
}
AXIS = {"+X": (1, 0, 0), "-X": (-1, 0, 0), "+Y": (0, 1, 0),
        "-Y": (0, -1, 0), "+Z": (0, 0, 1), "-Z": (0, 0, -1)}

# One hue per axis, saturated for the positive face and pale for the negative one, so the
# axis is readable at a glance.  The hues are the house ones: amber, blue, violet.
FACE_COLOR = {"+X": "#f2b45c", "-X": "#f7d9a8",
              "+Y": "#7fb6e2", "-Y": "#bcd9ef",
              "+Z": "#a586d8", "-Z": "#d3c2ea"}

# The four side faces, left to right in the band.  This order walks around the cube
# counter-clockwise seen from +Z; the other sense mirrors every island (see check 4).
BAND = ("-Y", "+X", "+Y", "-X")
BAND_ROW = 1            # the band occupies v in [1/3, 2/3]
CAP_COL = 1             # +Z folds up from this column of the band, -Z folds down
NCOL, NROW = 4, 3

VIEW = (22.0, -58.0)    # elevation, azimuth: shows +X, -Y and +Z


# -- Folding the cross ------------------------------------------------------

def _shared_edge(fa: str, fb: str) -> list:
    e = [v for v in FACES[fa] if v in FACES[fb]]
    if len(e) != 2:
        raise ValueError(f"faces {fa} and {fb} share {len(e)} vertices, not an edge")
    return e


def _split_vertical(edge: list) -> tuple:
    """(bottom, top) of an edge of the cube parallel to z."""
    lo = [v for v in edge if VERTS[v][2] == LO]
    hi = [v for v in edge if VERTS[v][2] == HI]
    if len(lo) != 1 or len(hi) != 1:
        raise ValueError(f"edge {edge} is not vertical, the band cannot be built on it")
    return lo[0], hi[0]


def _neighbour(face: str, v: int, exclude: int) -> int:
    """The vertex adjacent to `v` in the loop of `face` that is not `exclude`."""
    loop = FACES[face]
    i = loop.index(v)
    cand = [c for c in (loop[(i - 1) % 4], loop[(i + 1) % 4]) if c != exclude]
    if len(cand) != 1:
        raise ValueError(f"{v} and {exclude} are not adjacent in face {face}")
    return cand[0]


def cell_uv(col: int, row: int) -> list:
    """The four corners of a cell, CCW from the bottom-left one."""
    u0, u1 = col / NCOL, (col + 1) / NCOL
    v0, v1 = row / NROW, (row + 1) / NROW
    return [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]


def build_unwrap() -> dict:
    """face -> {col, row, uv, verts}, uv and verts both CCW from the bottom-left corner.

    Nothing here is a table of coordinates: the band is walked, each cell takes the edge
    it shares with the previous and the next face, and the two caps are folded over the
    top and bottom edges of the column CAP_COL.
    """
    cells = {}

    for c, face in enumerate(BAND):
        bl, tl = _split_vertical(_shared_edge(face, BAND[(c - 1) % NCOL]))
        br, tr = _split_vertical(_shared_edge(face, BAND[(c + 1) % NCOL]))
        cells[face] = dict(col=c, row=BAND_ROW, uv=cell_uv(c, BAND_ROW),
                           verts=[bl, br, tr, tl])

    bl_b, br_b, tr_b, tl_b = cells[BAND[CAP_COL]]["verts"]

    # +Z folds up over the top edge of the hinge cell, -Z folds down over its bottom edge.
    cells["+Z"] = dict(col=CAP_COL, row=BAND_ROW + 1, uv=cell_uv(CAP_COL, BAND_ROW + 1),
                       verts=[tl_b, tr_b,
                              _neighbour("+Z", tr_b, tl_b), _neighbour("+Z", tl_b, tr_b)])
    cells["-Z"] = dict(col=CAP_COL, row=BAND_ROW - 1, uv=cell_uv(CAP_COL, BAND_ROW - 1),
                       verts=[_neighbour("-Z", bl_b, br_b), _neighbour("-Z", br_b, bl_b),
                              br_b, bl_b])
    return cells


def _is_rotation(a: list, b: tuple) -> bool:
    return any(tuple(b[i:] + b[:i]) == tuple(a) for i in range(len(b)))


def check(cells: dict) -> dict:
    """The four guarantees of the docstring.  Returns the corner -> vertex map."""
    for name, loop in FACES.items():                                    # 1
        p = VERTS[list(loop)]
        n = np.cross(p[1] - p[0], p[2] - p[1])
        n = n / np.linalg.norm(n)
        if not np.allclose(n, AXIS[name], atol=1e-9):
            raise ValueError(f"face {name} is not outward-CCW: its normal is {n}")

    corner = {}
    for name, cell in cells.items():
        if set(cell["verts"]) != set(FACES[name]):                      # 2
            raise ValueError(f"cell {name} got vertices {cell['verts']}, "
                             f"the face has {FACES[name]}")
        if not _is_rotation(cell["verts"], FACES[name]):                # 4
            raise ValueError(
                f"island {name} is mirrored: its corners CCW in UV are {cell['verts']}, "
                f"which is not a rotation of the outward-CCW loop {FACES[name]}. "
                f"BAND walks around the cube the wrong way.")
        for uv, v in zip(cell["uv"], cell["verts"]):                    # 3
            key = (round(uv[0], 9), round(uv[1], 9))
            if corner.setdefault(key, v) != v:
                raise ValueError(f"UV corner {key} is claimed by V{corner[key]} and "
                                 f"by V{v}: the layout is not a fold")
    return corner


# -- Left panel: the cube ---------------------------------------------------

def _fmt_pos(p) -> str:
    def f(x: float) -> str:
        return f"{x:g}" if abs(x - round(x)) < 1e-9 else f"{x:.2f}"
    return "(" + ", ".join(f(x) for x in p) + ")"


def _spread(ang, min_gap: float):
    """Push angles apart until no two are closer than `min_gap`, keeping their order.

    Two pairs of cube corners project only 15 degrees apart on the ring, which is less
    than a two-line label needs.  Relaxing beats a table of hand-picked offsets: it
    survives a change of VIEW.
    """
    ang = np.asarray(ang, float).copy()
    n = len(ang)
    if n * min_gap >= 2 * np.pi:
        raise ValueError("min_gap too large: the labels cannot fit on the ring")
    order = np.argsort(ang)
    a = ang[order]
    for _ in range(400):
        moved = False
        for i in range(n):
            j = (i + 1) % n
            gap = (a[j] - a[i]) % (2 * np.pi)
            if gap < min_gap - 1e-12:
                push = 0.5 * (min_gap - gap)
                a[i] -= push
                a[j] += push
                moved = True
        if not moved:
            break
    ang[order] = a
    return ang


def _gap(ang) -> float:
    """The middle of the widest angular gap between consecutive labels."""
    a = np.sort(np.asarray(ang, float) % (2 * np.pi))
    gaps = (np.roll(a, -1) - a) % (2 * np.pi)
    i = int(np.argmax(gaps))
    return a[i] + 0.5 * gaps[i]


def draw_cube(ax) -> Scene:
    sc = Scene(ax, Projector(*VIEW))
    view = sc.proj.v

    visible = {n: float(np.dot(AXIS[n], view)) > 0.0 for n in FACES}

    # Painter: farthest first, so the three back faces end up under the three front ones.
    for name in sorted(FACES, key=lambda n: sc.proj.depth(VERTS[list(FACES[n])])):
        sc.poly(VERTS[list(FACES[name])], FACE_COLOR[name], alpha=1.0,
                edge=C_EDGE, lw=1.2, z=2)

    # The three edges whose both faces look away are redrawn dashed, so the far vertex
    # has an anchor and all eight labels point at something.
    edges = {}
    for name, loop in FACES.items():
        for i in range(4):
            edges.setdefault(frozenset((loop[i], loop[(i + 1) % 4])), []).append(name)
    for e, owners in edges.items():
        if not any(visible[o] for o in owners):
            a, b = tuple(e)
            sc.line(VERTS[a], VERTS[b], C_HIDDEN, lw=1.1, ls=(0, (3, 3)), z=3)

    # The face name is pushed away from the centre of the drawing, into the outer half
    # of its own face: the centroid of a face seen at an angle can project on top of the
    # far vertex, which is exactly where the hidden-vertex marker sits.
    mid = sc.proj(np.zeros(3))
    for name in FACES:
        if visible[name]:
            c = VERTS[list(FACES[name])].mean(axis=0)
            d = sc.proj(c) - mid
            sc.text(c, name, C_INK, dxy=tuple(0.32 * d / max(np.linalg.norm(d), 1e-9)),
                    size=17, weight="bold", z=5)

    # The eight labels go on a ring around the cube rather than next to their vertex:
    # V3 and V5 project inside the silhouette from any general view, so a short local
    # offset puts their text on top of the geometry.  The ring keeps the direction (each
    # label still sits on the side its vertex is on) and drops the distance problem; a
    # leader line is drawn for the ones that had to travel.
    centre = sc.proj(VERTS.mean(axis=0))
    off = np.array([sc.proj(p) for p in VERTS]) - centre
    ring = np.linalg.norm(off, axis=1).max() + 0.55
    ang = _spread(np.arctan2(off[:, 1], off[:, 0]), np.radians(26.0))

    for k, p in enumerate(VERTS):
        hidden = not any(visible[n] for n in FACES if k in FACES[n])
        sc.dot(p, "white" if hidden else C_INK, s=52, z=8,
               edge=C_HIDDEN if hidden else "white", lw=1.4)
        d = np.array([np.cos(ang[k]), np.sin(ang[k])])
        anchor = centre + ring * d
        if np.linalg.norm(anchor - sc.proj(p)) > 0.60:
            seg = np.array([sc.proj(p), anchor - 0.14 * d])
            ax.plot(seg[:, 0], seg[:, 1], color="#b3bcc5", lw=0.9, zorder=6)
        sc.text(p, f"V{k}\n{_fmt_pos(p)}", C_INK, dxy=tuple(anchor - sc.proj(p)), size=12,
                ha="left" if d[0] > 0.20 else "right" if d[0] < -0.20 else "center",
                va="bottom" if d[1] > 0.20 else "top" if d[1] < -0.20 else "center")

    # The triad is drawn in the projected plane, in the widest angular gap of the ring,
    # so it cannot land on a label.  Its arrows are the projected world axes, foreshorten-
    # ing included, so their relative lengths are the ones the cube is drawn with.
    o = centre + (ring - 0.15) * np.array([np.cos(_gap(ang)), np.sin(_gap(ang))])
    for ax_name, e in (("x", (1, 0, 0)), ("y", (0, 1, 0)), ("z", (0, 0, 1))):
        d = sc.proj(np.array(e, float)) - sc.proj(np.zeros(3))
        ax.annotate("", xy=o + d * 0.75, xytext=o, zorder=7,
                    arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.5",
                                    color="#8894a1", lw=1.4, shrinkA=0, shrinkB=0))
        t = o + d * 0.92
        ax.text(t[0], t[1], ax_name, color="#5f6c79", fontsize=13, ha="center",
                va="center", zorder=7)
        sc._add(np.array([t]))

    sc.finish(pad=0.05)
    ax.set_title(r"Object space: vertex position $\mathbf{p}=(x,y,z)$",
                 fontsize=15, pad=12)
    return sc


# -- Right panel: the atlas -------------------------------------------------

def draw_atlas(ax, cells: dict, corner: dict) -> None:
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="#f4f6f8", edgecolor=C_FRAME,
                               lw=1.2, ls=(0, (4, 3)), zorder=0))

    for name, cell in cells.items():
        uv = np.array(cell["uv"])
        ax.fill(uv[:, 0], uv[:, 1], color=FACE_COLOR[name], edgecolor=C_EDGE,
                lw=1.2, zorder=1)
        ax.text(uv[:, 0].mean(), uv[:, 1].mean(), name, color=C_INK, fontsize=17,
                fontweight="bold", ha="center", va="center", zorder=4)

    # Every label is pushed straight up or straight down, away from the mid-line of the
    # band, and stays centred on its corner.  Diagonal offsets look tidier one corner at
    # a time and do not work here: consecutive corners are a quarter of the atlas apart,
    # which is barely wider than the text, so any horizontal shift makes two neighbours
    # collide.  Centred labels sit on four clean rows instead.
    mid_v = (BAND_ROW + 0.5) / NROW
    for (u, v), k in sorted(corner.items()):
        up = v > mid_v
        ax.scatter([u], [v], s=46, color=C_INK, edgecolors="white", lw=1.2, zorder=6)
        ax.annotate(f"V{k}\n({u:.2f}, {v:.2f})", (u, v), textcoords="offset points",
                    xytext=(0.0, 15.0 if up else -15.0), fontsize=9.5, color=C_INK,
                    zorder=7, ha="center", va="bottom" if up else "top",
                    path_effects=[pe.withStroke(linewidth=2.8, foreground="white")])

    ax.set_xlim(-0.13, 1.13)
    ax.set_ylim(-0.22, 1.22)
    ax.set_aspect("equal")
    ax.set_xticks([i / NCOL for i in range(NCOL + 1)])
    ax.set_yticks([i / NROW for i in range(NROW + 1)])
    ax.set_yticklabels([f"{i / NROW:.2f}" for i in range(NROW + 1)])
    ax.tick_params(labelsize=11, colors="#5f6c79")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("$u$", fontsize=14)
    ax.set_ylabel("$v$", fontsize=14, rotation=0, labelpad=12)
    ax.set_title("UV atlas: texture coordinates $(u,v)$", fontsize=15, pad=12)


# -- Main -------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="directory the PNG is written to")
    ap.add_argument("--name", default="uv_cube_atlas")
    ap.add_argument("--dpi", type=int, default=DPI)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cells = build_unwrap()
    corner = check(cells)

    mult = {k: 0 for k in range(len(VERTS))}
    for v in corner.values():
        mult[v] += 1
    n_face_corners = 4 * len(FACES)
    print(f"  band {' '.join(BAND)}, +Z and -Z folded on column {CAP_COL}: "
          f"all {len(FACES)} islands non-mirrored")
    print(f"  {len(VERTS)} vertices -> {n_face_corners} face corners "
          f"-> {len(corner)} distinct UV corners")
    print("  UV corners per vertex: "
          + "  ".join(f"V{k}:{n}" for k, n in sorted(mult.items())))
    if sum(mult.values()) != len(corner):
        raise ValueError("the multiplicities do not add up to the corners")

    fig, (axl, axr) = plt.subplots(1, 2, figsize=FIGSIZE)
    draw_cube(axl)
    draw_atlas(axr, cells, corner)

    top = max(mult.values())
    seams = sorted(k for k, n in mult.items() if n == top)
    fig.text(0.5, 0.035,
             f"{len(VERTS)} vertex positions  $\\rightarrow$  {len(FACES)}$\\times$4 = "
             f"{n_face_corners} face corners  $\\rightarrow$  {len(corner)} distinct UV "
             "corners.    " + ", ".join(f"V{k}" for k in seams)
             + f" lie on the cut and appear {top} times each.",
             ha="center", fontsize=12.5, color="#5f6c79")

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.17, top=0.91, wspace=0.02)
    path = out / f"{args.name}.png"
    fig.savefig(path, dpi=args.dpi, facecolor="white", bbox_inches="tight",
                pad_inches=0.28)
    plt.close(fig)
    print(f"  + {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
