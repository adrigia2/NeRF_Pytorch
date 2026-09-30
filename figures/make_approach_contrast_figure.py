#!/usr/bin/env python
"""make_approach_contrast_figure.py -- joint optimisation against a closed form, no words.

    python make_approach_contrast_figure.py --out ../../PresentationImages

Writes three PNGs: `approach_contrast.png`, the two panels side by side on a 4:3 page, and
`approach_contrast_joint.png` and `approach_contrast_closed_form.png`, each panel on its
own canvas at `PANEL_DPI` for a slide that shows them one at a time.  The single panels are
not crops of the pair, they are drawn again at the same scale and a higher density, so a
card is the same size in all three files and the two panels can be laid side by side
without anything being resized to match.

The figure is for the related-work slide, the one that says the neural
methods estimate geometry, normals, lighting and materials TOGETHER, that different
combinations of them explain the same photograph, and that our reconstruction is instead a
single pass from quantities that are already known.

The slide is spoken, so the figure carries no text at all: every claim it makes is made by
the SHAPE of the drawing, and the two panels share one visual vocabulary so that the only
difference the eye can find is the one the sentence is about.

  LEFT   the four quantities inside one enclosure, joined by double-headed arrows: every
         one of them is a function of every other, and the enclosure has to be solved as
         one thing.  One arrow leaves it, towards the result.

  RIGHT  its own enclosure holding the colour texture derived from what is given and the
         environment beside it, three arrows into it and one out of the bottom, at the
         three material maps below.  Every arrow points down, none is double-headed.

Each enclosure holds one method's WORK, and only that: what is given to a method stays
above it and what it arrives at stays below, on both sides, so arrows cross the boundary
rather than starting inside it.  The dark slate of the given cards is neither method's
colour.  That is what makes the two boxes comparable, and the contrast is what is drawn
inside them: four things that are each a function of the others, against two that are made
and handed on.

A return arrow used to carry the render back into the red enclosure, and with it the figure
said "iterative" on its own.  It is gone at the author's request, and what is left says
that everything in there has to be solved together, not how many times.  The rest of that
sentence is now the speaker's to finish.

Both panels read top to bottom, because they go side by side on one 4:3 slide and only
upright do they fill it; see the note on the layout constants.

Two encodings run across both panels and are never mixed:

  dashed red frame   the quantity is unknown and is being estimated;
  solid frame        the quantity is known, either given or already solved;
  an empty frame     a placeholder, filled by hand after the file is written.

The input views were drawn as a stack of offset cards, to say that there are many of them.
The mark is gone: it was the only place where a card meant a set rather than a quantity,
and one picture standing for many is a thing the person filling the frame can say better
with the picture they choose than the frame can say around it.

A third mark, an empty dashed frame behind each unknown standing for the other values that
would have explained the photograph just as well, was drawn and removed: at slide distance
it reads as a misprint rather than as an alternative, and the ambiguity has a figure of its
own in `illposed_albedo_light_bare.png`, which argues it with the numbers.

No card in the figure is a result.  The material cards carry the ORIGINAL maps, the ones
the scene was authored with, and the same three appear on both sides on purpose: base
colour, roughness and metallic are what the joint methods estimate and what the closed form
solves for, so the panels differ in topology, not in ingredients.  Putting the pipeline's own fit in the last box
would turn a drawing about the shape of a process into a claim about its output, and
putting a noisy albedo in the left panel would put a claim about someone else's output
into a thesis that has not measured it.

Five cards are left empty for the author to fill: the geometry and the input views, in both
panels, and the result the left panel arrives at.  That result and the views it starts from
are the same photograph, which is why the two are emptied together and why the identity
matters: what comes out of the optimisation matches the observation whatever the four cards
inside happen to contain.  `illposed_albedo_light.png` argues that quantitatively, with the
products worked out.

A card is drawn at the shape of what it holds: 16:9 for the placeholders, which take
photographs, and for the envmap; 1:1 for every texture atlas, which is square on disk and
is now shown whole, cropped and padded by nothing.  What all the cards share is their AREA
rather than their side, so that mixing two shapes keeps them the same visual weight instead
of making the wide ones 78% wider than the square ones and the grid ragged.  Every image is
read-only and its path is relative to the repository root, `Doc/images` for all of them but
the colour texture, which only exists in `PresentationImages`.  The only one reshaped is the
envmap, 2:1 on disk and cropped to 16:9.
"""
from __future__ import annotations

import argparse
from collections import namedtuple
from pathlib import Path

import numpy as np

import _paths  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from make_geometry_diagrams import (                                       # noqa: E402
    DPI, C_EDGE, C_HILITE, C_CAM, C_ACCEPT,
)

# -- Palette ------------------------------------------------------------------
# One accent per panel, and the accent is the panel: everything the method does is drawn in
# it, so the eye separates the two before reading either.  Dark slate is neither, and marks
# what is given to a method rather than done by it.
C_UNKNOWN = C_HILITE          # frames and arrows of the joint optimisation
C_CHAIN   = C_ACCEPT          # the forward pass of the closed form
C_FIELD   = "#f7f1f1"         # the enclosure of the optimisation
C_FIELD_OK = "#f0f6f0"        # the enclosure of the closed form

# -- Layout, in axes units (10 to the inch) -----------------------------------
# Both panels flow TOP TO BOTTOM, and that is a consequence of the page rather than a
# taste: the two go side by side on one 4:3 slide with nothing else on it, so each gets
# half its width and all its height, about 0.65 wide for 1 tall.  Laid out left to right
# a panel is about 1.5 wide for 1 tall, and two of those on that page are limited by the
# width until a card is barely a hundred pixels across.  Turned upright they are limited
# by neither: `joint_extent()` and `closed_form_extent()` come out 0.661 and 0.643, and
# side by side with a small gap that is 1.343 against the 1.333 of the page.
#
# The contrast survives the turn intact, because it was never about direction: one panel
# closes on itself and the other does not.
PAGE_AR = 4.0 / 3.0           # the slide the two panels are meant to share

# Two aspect ratios, each the shape of what the card actually holds: 16:9 for the
# placeholders (they take photographs) and for the envmap, 1:1 for every texture atlas,
# which is square on disk and is now shown whole, cropped and padded by nothing.
#
# What is constant across the figure is the AREA, not the side.  Sizing mixed shapes by a
# common height makes a 16:9 card 78% wider than a square one and the grid goes ragged;
# sizing by a common area keeps them the same visual weight, which is what "the same kind
# of card" has to mean once they are no longer the same shape.  The area is the one of the
# 14.4 x 10.8 card the layout was tuned at, so nothing moved by much.
AR_WIDE = 16.0 / 9.0
AR_SQ = 1.0
CARD_AREA = 14.4 * 10.8

Card = namedtuple("Card", "c ar key")     # centre, aspect ratio, asset (None = empty)
Group = namedtuple("Group", "c w h")      # several cards standing for one quantity

# Left panel: in at the top, the four unknowns, out at the bottom, back up the side.
#
# The four sit in a DIAMOND and not in a grid, because six arrows between four things means
# two of them join opposite corners, and a diamond is the arrangement that leaves its own
# middle empty for them to cross in.  The three material maps are one corner: they are one
# estimated quantity with three channels, and drawn as three separate nodes the coupling
# would need fifteen arrows and would say nothing except that the figure is busy.
L_IN = Card((30.0, 95.0), AR_WIDE, None)                   # the input views
L_TOP = Card((30.0, 76.0), AR_WIDE, None)                  # geometry
L_LEFT = Card((13.0, 52.0), AR_SQ, "normals")
L_RIGHT = Card((47.0, 52.0), AR_WIDE, "lighting")
L_MAPS = (Card((14.0, 26.0), AR_SQ, "base_color"),
          Card((30.0, 26.0), AR_SQ, "metallic"),
          Card((46.0, 26.0), AR_SQ, "roughness"))
L_OUT = Card((30.0, 3.0), AR_WIDE, None)                   # the render the loop produces

# Right panel: what is given, what is derived from it, what comes out.  The skybox sits
# beside the colour texture and not in the row above it: the two are what the solve reads,
# and the environment is not something the colour texture is made of.
# Laid out in the same coordinates as the left panel and then pushed right by R_DX, so the
# pair does not draw one on top of the other; each panel is framed on its own content, so
# the offset is invisible in the two single files.
R_DX = 72.0
R_IN = (Card((R_DX + 10.31, 92.98), AR_WIDE, None),        # views
        Card((R_DX + 34.93, 92.98), AR_WIDE, None),        # geometry
        Card((R_DX + 57.48, 94.54), AR_SQ, "normals"))     # bottom-aligned with the two
R_MID = (Card((R_DX + 21.56, 52.00), AR_SQ, "color_texture"),
         Card((R_DX + 42.11, 50.44), AR_WIDE, "lighting"))  # bottom-aligned with it
R_OUT = (Card((R_DX + 14.41, 7.66), AR_SQ, "base_color"),
         Card((R_DX + 32.87, 7.66), AR_SQ, "roughness"),
         Card((R_DX + 51.34, 7.66), AR_SQ, "metallic"))

BOX_PAD   = 3.0               # gap between the four unknowns and the enclosure
GROUP_PAD = 2.0               # gap between cards grouped as one quantity and their box
MARGIN    = 3.0               # white kept around a panel drawn on its own

# The panels are also written one per file, each on its own canvas rather than cropped out
# of the pair, at a higher density so that either one can fill a slide by itself.  The
# scale is deliberately the same in all three (10 units to the inch), so a card is the same
# size wherever it appears and the two panels can be laid side by side without matching.
PANEL_DPI = 300

# -- The images, all of them read-only out of Doc/images ----------------------
# The material cards are the ORIGINAL maps, the ones the scene was authored and baked with
# in Blender, not the pipeline's fit of them.  Every card in this figure is an icon of a
# quantity and none of them is a result: the drawing is about the shape of the process, and
# the moment a real output sits in the last box the room starts reading the diagram as a
# results slide.  `Doc/images/results/interior/` holds what the pipeline actually recovers
# and belongs on the slide that claims it.
#
# The geometry and the input views, in both panels, and the render the joint loop produces
# are drawn EMPTY, as placeholders to be filled by hand.
_ORIG = "Doc/images/table-and-other-specular-png/BakedMaterial_"
ASSETS = {
    "normals":       ("Doc/images/ium/ium_normal.png", AR_SQ),
    "lighting":      ("Doc/images/lighting/skybox_studio.png", AR_WIDE),
    "color_texture": ("PresentationImages/color_texture_reinhard_gamma22.png", AR_SQ),
    "base_color":    (_ORIG + "base_color.png", AR_SQ),
    "roughness":     (_ORIG + "roughness.png", AR_SQ),
    "metallic":      (_ORIG + "metallic.png", AR_SQ),
}

MAXPX = 1400                  # a card is about 430 px at PANEL_DPI; 4096 helps nobody


def size(ar: float) -> "tuple[float, float]":
    """The (width, height) of a card of aspect `ar`, at the area every card shares."""
    return float(np.sqrt(CARD_AREA * ar)), float(np.sqrt(CARD_AREA / ar))


def to_ar(img: np.ndarray, ar: float) -> np.ndarray:
    """Bring an image to the target aspect ratio, cropping the wide and padding the tall.

    Padding whatever is narrower than the target, rather than cropping it, is what makes
    the operation safe on an atlas, whose islands run to the edges and whose background is
    already the black the padding would add.  With the atlases drawn square it does not
    come up any more: the only image reshaped here is the envmap, 2:1 on disk and cropped
    to 16:9, which loses a little of its sides and nothing else.
    """
    img = img[..., :3]
    h, w = img.shape[:2]
    nw = int(round(h * ar))
    if w >= nw:
        x0 = (w - nw) // 2
        return img[:, x0:x0 + nw]
    out = np.zeros((h, nw, 3), dtype=img.dtype)
    left = (nw - w) // 2
    out[:, left:left + w] = img
    return out


def load(root: Path, rel: str, ar: float) -> np.ndarray:
    """Read one card image as (H, W, 3), whatever the file happens to hold.

    The original roughness and metallic are single-channel PNGs, so the channel count is
    settled here rather than in `to_ar`: slicing `[..., :3]` on a 2-D array takes three
    COLUMNS, which is a silent and very confusing way to fail.
    """
    img = np.asarray(plt.imread(root / rel), dtype=np.float32)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    step = max(1, int(np.ceil(max(img.shape[:2]) / MAXPX)))
    return to_ar(img[::step, ::step], ar)


# -- Drawing primitives -------------------------------------------------------
def card(ax, spec, img, edge, lw=1.8, dashed=False, z=5):
    """One picture in a frame, at the shape the card declares.

    `img=None` leaves the frame empty, on white: a placeholder for a picture that goes in
    by hand afterwards.  White and not a grey, so that an empty card stays legible against
    the tinted field of the enclosure.

    The frame and the picture are drawn at the SAME aspect, and that is checked rather than
    assumed: `imshow` with an explicit extent stretches whatever it is given to fill it, so
    an atlas that reached a 16:9 frame would be silently squashed and still look plausible.
    """
    w, h = dims(spec)
    x0, y0 = spec.c[0] - w / 2.0, spec.c[1] - h / 2.0
    if img is None:
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor="white",
                               edgecolor="none", zorder=z + 1))
    else:
        got = img.shape[1] / img.shape[0]
        if abs(got - spec.ar) > 0.01:
            raise ValueError(f"{spec.key}: image is {got:.3f}, frame is {spec.ar:.3f}")
        ax.imshow(img, extent=(x0, x0 + w, y0, y0 + h), aspect="auto",
                  zorder=z + 1, interpolation="antialiased")
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor="none", edgecolor=edge, lw=lw,
                           zorder=z + 2, linestyle=(0, (4.0, 2.6)) if dashed else "-"))


def arrow(ax, p0, p1, color, lw=2.4, rad=0.0, head=17.0, alpha=1.0, double=False, z=4):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="<|-|>" if double else "-|>",
                                 mutation_scale=head, color=color, lw=lw, alpha=alpha,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0.0, shrinkB=0.0, zorder=z))


def dims(spec):
    """(width, height) of a card or of a group of them."""
    return size(spec.ar) if hasattr(spec, "ar") else (spec.w, spec.h)


def group_of(specs, pad=GROUP_PAD):
    """The several cards of one quantity, as a single node the arrows can point at."""
    x0, y0, x1, y1 = box_of(specs, pad)
    return Group((0.5 * (x0 + x1), 0.5 * (y0 + y1)), x1 - x0, y1 - y0)


def right_of(spec):
    return (spec.c[0] + dims(spec)[0] / 2.0, spec.c[1])


def left_of(spec):
    return (spec.c[0] - dims(spec)[0] / 2.0, spec.c[1])


def below(spec):
    return (spec.c[0], spec.c[1] - dims(spec)[1] / 2.0)


def above(spec):
    return (spec.c[0], spec.c[1] + dims(spec)[1] / 2.0)


def box_of(specs, pad=0.0):
    """(x0, y0, x1, y1) around a group of cards.

    The enclosure and the panel extents are measured off the cards rather than written
    down beside them: with two aspect ratios in play a hand-kept rectangle stops being
    right the moment one card changes shape.
    """
    xs, ys = [], []
    for s in specs:
        w, h = dims(s)
        xs += [s.c[0] - w / 2.0 - pad, s.c[0] + w / 2.0 + pad]
        ys += [s.c[1] - h / 2.0 - pad, s.c[1] + h / 2.0 + pad]
    return min(xs), min(ys), max(xs), max(ys)


def edge_point(spec, towards, pad=0.6):
    """Where the segment from one card centre to another leaves the first card.

    Drawn centre to centre, an arrow between two neighbouring cards is entirely hidden
    under them and its heads with it, which for the six arrows that carry the coupling is
    the whole content of the drawing.
    """
    w, h = dims(spec)
    dx, dy = towards.c[0] - spec.c[0], towards.c[1] - spec.c[1]
    tx = (w / 2.0 + pad) / abs(dx) if dx else np.inf
    ty = (h / 2.0 + pad) / abs(dy) if dy else np.inf
    t = min(tx, ty)
    return (spec.c[0] + dx * t, spec.c[1] + dy * t)


def l_nodes():
    """The four coupled unknowns; the material maps count as one."""
    return (L_TOP, L_LEFT, L_RIGHT, group_of(L_MAPS))


def l_box():
    """The enclosure: everything being estimated, and nothing else."""
    return box_of(l_nodes(), BOX_PAD)


def joint_extent():
    """Bounding box of the left panel: the enclosure, and the cards above and below it."""
    bx0, by0, bx1, by1 = l_box()
    x0, y0, x1, y1 = box_of((L_IN, L_OUT))
    return ((min(x0, bx0) - MARGIN, max(x1, bx1) + MARGIN),
            (min(y0, by0) - MARGIN, max(y1, by1) + MARGIN))


def r_box():
    """The enclosure of the closed form: the WORK, with what goes in and what comes out
    left outside it.

    The same rule as the other panel, and the reason both boxes are readable: a box holds
    one method's work, the data given to it stays above, the result it arrives at stays
    below.  Arrows cross the boundary rather than starting inside it, so what the two boxes
    contain is comparable and neither is confused with its own inputs.
    """
    return box_of(R_MID, BOX_PAD)


def closed_form_extent():
    """Bounding box of the right panel: the given above, the enclosure, the maps below."""
    bx0, by0, bx1, by1 = r_box()
    ix0, iy0, ix1, iy1 = box_of(R_IN)
    ox0, oy0, ox1, oy1 = box_of(R_OUT, GROUP_PAD)
    return ((min(ix0, bx0, ox0) - MARGIN, max(ix1, bx1, ox1) + MARGIN),
            (min(iy0, by0, oy0) - MARGIN, max(iy1, by1, oy1) + MARGIN))


def pair_extent(jx, jy, cx, cy):
    """The two panels together, padded out to the aspect of the page they are meant for.

    The pair is a preview of the slide, so it is written at the slide's shape: what it puts
    on screen is what the two panels do to a 4:3 page, white included.  Padding rather than
    cropping, and on one axis only, so the panels inside it stay exactly the figures the
    other two files hold.
    """
    x0, x1 = min(jx[0], cx[0]), max(jx[1], cx[1])
    y0, y1 = min(jy[0], cy[0]), max(jy[1], cy[1])
    w, h = x1 - x0, y1 - y0
    if w / h < PAGE_AR:
        pad = (PAGE_AR * h - w) / 2.0
        x0, x1 = x0 - pad, x1 + pad
    else:
        pad = (w / PAGE_AR - h) / 2.0
        y0, y1 = y0 - pad, y1 + pad
    return (x0, x1), (y0, y1)


def draw_joint(ax, img):
    """One enclosure, everything unknown inside it, and a way back."""
    x0, y0, x1, y1 = l_box()
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=3.2",
                                facecolor=C_FIELD, edgecolor=C_UNKNOWN,
                                lw=1.6, alpha=0.95, zorder=1))

    # every unknown is a function of every other one: six double-headed arrows
    nodes = l_nodes()
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, b = nodes[i], nodes[j]
            arrow(ax, edge_point(a, b), edge_point(b, a), C_UNKNOWN,
                  lw=2.0, head=11.0, alpha=0.85, double=True, z=2)

    for spec in (L_TOP, L_LEFT, L_RIGHT) + L_MAPS:
        card(ax, spec, img.get(spec.key), C_UNKNOWN, dashed=True)

    # The three maps are bound into one node by a solid line, the same grouping device the
    # outputs of the other panel use.  Solid, and no contradiction with the dashed frames
    # inside it: the dashes say the quantity is unknown, the box says how many cards it
    # takes to show it.
    mx0, my0, mx1, my1 = box_of(L_MAPS, GROUP_PAD)
    ax.add_patch(FancyBboxPatch((mx0, my0), mx1 - mx0, my1 - my0,
                                boxstyle="round,pad=0,rounding_size=2.2",
                                facecolor="none", edgecolor=C_UNKNOWN,
                                lw=1.4, alpha=0.55, zorder=3))

    card(ax, L_IN, None, C_CAM)
    arrow(ax, below(L_IN), (L_IN.c[0], y1 + 1.3), C_CAM, lw=2.6)

    card(ax, L_OUT, None, C_CAM)
    arrow(ax, (L_OUT.c[0], y0 - 1.3), above(L_OUT), C_UNKNOWN, lw=2.6)


def draw_closed_form(ax, img):
    """What is given, what is derived, what comes out.  One direction throughout."""
    # The enclosure is the counterpart of the other panel's, and it does cost something: not
    # drawing one used to be part of the argument.  What carried that argument was never the
    # absence of a box though, it was the return arrow, and side by side on a slide two
    # bounded blocks read as two comparable methods where one loose column read as an
    # appendix to the other.
    bx0, by0, bx1, by1 = r_box()
    ax.add_patch(FancyBboxPatch((bx0, by0), bx1 - bx0, by1 - by0,
                                boxstyle="round,pad=0,rounding_size=3.2",
                                facecolor=C_FIELD_OK, edgecolor=C_CHAIN,
                                lw=1.6, alpha=0.95, zorder=1))

    # The three given cards point at the BOX, each landing on its own stretch of the top
    # edge.  Aimed at a card inside it they would be saying which of the two the mesh is an
    # input to, a claim about the inside of the method and not what this panel is about;
    # aimed at one single point they would overlap into one blob.
    bmid = 0.5 * (bx0 + bx1)
    for k, spec in enumerate(R_IN):
        card(ax, spec, img.get(spec.key), C_CAM)
        arrow(ax, below(spec), (bmid + (k - 1) * 8.0, by1), C_CAM, lw=2.2, alpha=0.85)

    for spec in R_MID:
        card(ax, spec, img[spec.key], C_CAM)

    # One arrow out of the bottom of the box, at the maps it arrives at.  One and not three:
    # the closed form is a single pass and returns them together.
    gx0, gy0, gx1, gy1 = box_of(R_OUT, GROUP_PAD)
    arrow(ax, (bmid, by0), (0.5 * (gx0 + gx1), gy1 + 1.0), C_CHAIN, lw=4.2, head=26.0)

    for spec in R_OUT:
        card(ax, spec, img[spec.key], C_CHAIN, lw=2.4)
    ax.add_patch(FancyBboxPatch((gx0, gy0), gx1 - gx0, gy1 - gy0,
                                boxstyle="round,pad=0,rounding_size=2.6",
                                facecolor="none", edgecolor=C_CHAIN,
                                lw=1.8, alpha=0.60, zorder=3))


def render(out: Path, xlim, ylim, dpi: float, draw) -> None:
    (xa, xb), (ya, yb) = xlim, ylim
    fig = plt.figure(figsize=((xb - xa) / 10.0, (yb - ya) / 10.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(xa, xb)
    ax.set_ylim(ya, yb)
    ax.set_aspect("equal")
    ax.axis("off")
    draw(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"  + {out}  ({int(round((xb - xa) / 10.0 * dpi))}x"
          f"{int(round((yb - ya) / 10.0 * dpi))})")


def build(root: Path, out_dir: Path, dpi: float, panel_dpi: float) -> None:
    img = {k: load(root, rel, ar) for k, (rel, ar) in ASSETS.items()}

    jx, jy = joint_extent()
    cx, cy = closed_form_extent()
    div = 0.5 * (jx[1] + cx[0])

    def pair(ax):
        draw_joint(ax, img)
        draw_closed_form(ax, img)
        ax.plot([div, div], [min(jy[0], cy[0]) + 2.0, max(jy[1], cy[1]) - 2.0],
                color=C_EDGE, lw=1.0, alpha=0.30, zorder=0)

    render(out_dir / "approach_contrast.png", *pair_extent(jx, jy, cx, cy), dpi, pair)

    # The two panels alone, each framed on its own content.  A shared vertical extent was
    # tried first, so that side by side they would line up row by row; it left the left
    # panel with a band of white at the top as wide as the box around the outputs of the
    # right one, which is a visible defect in the file that gets projected alone against an
    # alignment nobody checks.  The scale is common, which is what keeps a card the same
    # size in both.
    render(out_dir / "approach_contrast_joint.png", jx, jy, panel_dpi,
           lambda ax: draw_joint(ax, img))
    render(out_dir / "approach_contrast_closed_form.png", cx, cy, panel_dpi,
           lambda ax: draw_closed_form(ax, img))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="../../PresentationImages",
                    help="folder the PNGs are written to")
    ap.add_argument("--assets", default=None,
                    help="folder the asset paths are relative to (default: the repo root)")
    ap.add_argument("--dpi", type=float, default=DPI,
                    help="density of the figure holding both panels")
    ap.add_argument("--panel-dpi", type=float, default=PANEL_DPI,
                    help="density of the two single-panel figures")
    a = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    root = Path(a.assets) if a.assets else repo
    missing = [rel for rel, _ in ASSETS.values() if not (root / rel).is_file()]
    if missing:
        raise SystemExit(f"missing under {root}: {', '.join(sorted(set(missing)))}")

    build(root, Path(a.out).resolve(), a.dpi, a.panel_dpi)


if __name__ == "__main__":
    main()
