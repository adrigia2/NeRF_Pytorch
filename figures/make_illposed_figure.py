#!/usr/bin/env python
"""make_illposed_figure.py -- why inverse rendering is ill-posed, in one picture.

    python make_illposed_figure.py --out ../../PresentationImages
    python make_illposed_figure.py --out ../../PresentationImages --bare
    python make_illposed_figure.py --out ../../PresentationImages --reflection --bare

Writes one PNG per invocation:

  illposed_albedo_light.png        top, the same photograph twice, with an equals sign
                                   between; bottom, the two different scenes that both
                                   produce it, arrows pointing up into the photograph
                                   each one explains.  Two hypotheses, one observation.

  illposed_albedo_light_bare.png   the same drawing with every word removed (--bare),
                                   for a slide that talks about the ambiguity in
                                   general without committing to a reflectance model.

  illposed_reflection*.png         --reflection: a third hypothesis, in which the tint
                                   belongs to the light rather than to the surface,
                                   because the surface is grey and what reaches it has
                                   bounced off a coloured neighbour.

The figure exists to make one sentence of the presentation visible: the same photograph
has more than one explanation.  Under the Lambertian model the observed colour is
C = a*E/pi per channel, so the camera never sees the albedo and the illumination, only
their product.  Two scenes with the same product are not merely similar, they are the
same pixels.

The two ambiguities are not the same ambiguity.  The first pair differ in *level*: a
bright surface in dim light against a dark surface in bright light, both neutral.  The
third differs in *chromaticity*: a neutral surface whose warm cast is borrowed from a
neighbour it happens to sit next to.  Level is what the direct irradiance bake resolves;
chromaticity from a neighbour is interreflection, which is why the pipeline queries the
NeRF along the occluded directions instead of assuming the environment is all there is.

**The identity is computed, not drawn.**  Every photograph card is filled from the
numbers in the scene list, and `check()` refuses to draw unless all of them agree on the
radiance and on the 8-bit colour it quantises to.  Editing an albedo or an illumination
into a set whose products differ stops the script instead of producing a figure that
asserts an equality the numbers no longer support.

One display convention.  C = 0.15 linear would read as almost black on a slide, so every
patch is sRGB-encoded before being drawn, the same transform a viewer applies to an HDR
render; the albedo swatches go through it too, so the whole figure is in one display
space.  The numbers printed next to the patches are the linear ones.  The light glyphs
are not patches: a sun encodes its irradiance in size, ray count and saturation, and a
neighbour is drawn at its own chromaticity normalised to full brightness, since what
matters about it is its hue, not how far down the exposure it sits.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _paths  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle       # noqa: E402

# The palette every diagram of the thesis is drawn with; C_DIRECT is already the
# colour of the direct (sky) illumination there, which is what the suns carry here.
from make_geometry_diagrams import (                                   # noqa: E402
    DPI, C_DIRECT, C_EDGE, C_HILITE, C_CAM,
)

# -- The hypotheses.  Only the product a*E is observable ---------------------
# `a` and `E` are per channel; `light` says which glyph stands for E.
SCENES_LEVEL = [
    dict(a=(0.80, 0.80, 0.80), E=(0.25, 0.25, 0.25), light="sun",
         caption="a bright surface in dim light"),
    dict(a=(0.10, 0.10, 0.10), E=(2.00, 2.00, 2.00), light="sun",
         caption="a dark surface in bright light"),
]

SCENES_REFLECTION = [
    dict(a=(0.96, 0.6600, 0.4200), E=(0.5, 0.5, 0.5), light="sun",
         caption="a bright warm surface\nin dim neutral light"),
    dict(a=(0.12, 0.0825, 0.0525), E=(4.0, 4.0, 4.0), light="sun",
         caption="a dark warm surface\nin bright neutral light"),
    dict(a=(0.30, 0.3000, 0.3000), E=(1.6, 1.1, 0.7), light="bounce",
         caption="a grey surface lit by\na coloured neighbour"),
]

C_INK   = C_CAM          # body text
C_MUTED = C_EDGE         # secondary text and frames
C_PAPER = "#f4f6f8"      # the inside of a photograph card

# Layout, in axes units (10 to the inch).  The two-scene figure keeps the hand-tuned
# constants it was approved with; wider sets are spaced from them.
PHOTO_W, PHOTO_H = 34.0, 22.0
SCENE_W, SCENE_H = 40.0, 23.0
CX_PAIR = (32.0, 88.0)   # column centres, two scenes
XLIM_PAIR = 120.0
COL_STEP = 50.0          # column spacing when there are more than two
PHOTO_Y = 41.0           # bottom edge of the photograph cards
SCENE_Y = 6.0            # bottom edge of the scene cards, labelled layout

# Sun sizing.  SUN_REACH is how far the longest ray of the brightest sun gets from its
# centre; scene_box() sizes the bare card from it, so the two cannot drift apart.
SUN_R0, SUN_R1, SUN_GAP, SUN_RAY = 1.7, 1.3, 1.0, 2.2
SUN_REACH = SUN_R0 + SUN_R1 + SUN_GAP + SUN_RAY
SWATCH = 9.0             # side of the albedo swatch and of the neighbour block
BOUNCE_REACH = SWATCH / 2.0


def srgb(v):
    """Linear radiance to displayed colour, the standard sRGB transfer curve."""
    x = np.clip(np.broadcast_to(np.asarray(v, dtype=float), (3,)), 0.0, 1.0)
    y = np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1.0 / 2.4) - 0.055)
    return tuple(float(c) for c in y)


def quantise(rgb):
    """The 8-bit colour a patch lands on, which is what 'identical' has to mean."""
    return tuple(int(round(255.0 * c)) for c in rgb)


def observed(scene):
    """The Lambertian colour the camera records: C_c = a_c * E_c / pi."""
    a = np.asarray(scene["a"], dtype=float)
    E = np.asarray(scene["E"], dtype=float)
    return a * E / np.pi


def luminance(v):
    return float(np.dot(np.asarray(v, dtype=float), (0.2126, 0.7152, 0.0722)))


def is_neutral(scene):
    """True when one number says everything about this scene's albedo and light."""
    return len(set(scene["a"])) == 1 and len(set(scene["E"])) == 1


def check(scenes):
    """Refuse to draw unless every scene really does produce the same photograph."""
    obs = [observed(s) for s in scenes]
    px = [quantise(srgb(o)) for o in obs]
    for s, o, p in zip(scenes, obs, px):
        print("  a={:<22} E={:<18} ->  C={}  ->  sRGB{}".format(
            np.array2string(np.asarray(s["a"]), precision=4),
            np.array2string(np.asarray(s["E"]), precision=2),
            np.array2string(o, precision=6), p))
    spread = float(np.max(np.abs(np.asarray(obs) - obs[0])))
    if spread > 1e-12 * float(np.max(obs)):
        raise ValueError("the scenes do not produce the same radiance, "
                         "largest difference {:.3e}".format(spread))
    if len(set(px)) != 1:
        raise ValueError("the scenes quantise to different pixels: {}".format(px))
    print("  identical: {} scenes -> one photograph, sRGB{}".format(len(scenes), px[0]))
    return obs[0]


def columns(n):
    """Column centres and the x extent that holds them."""
    if n == 2:
        return list(CX_PAIR), XLIM_PAIR
    first = PHOTO_W / 2.0 + 8.0
    cx = [first + k * COL_STEP for k in range(n)]
    return cx, cx[-1] + first


def card(ax, cx, y0, w, h, facecolor, lw=1.6, edgecolor=None, z=1):
    """A rounded panel, the one container shape the figure uses."""
    p = FancyBboxPatch((cx - w / 2.0, y0), w, h,
                       boxstyle="round,pad=0,rounding_size=1.4",
                       facecolor=facecolor, edgecolor=edgecolor or C_MUTED,
                       linewidth=lw, zorder=z)
    ax.add_patch(p)
    return p


def draw_photograph(ax, cx, obs, labels=True, numbers=True):
    """One photograph: a flat object patch on a neutral ground, inside a frame."""
    card(ax, cx, PHOTO_Y, PHOTO_W, PHOTO_H, C_PAPER)
    # the ground line, enough to read the panel as a scene rather than as a swatch
    ax.plot([cx - PHOTO_W / 2 + 3.0, cx + PHOTO_W / 2 - 3.0], [PHOTO_Y + 6.2] * 2,
            color="#dde3e8", lw=2.0, zorder=2, solid_capstyle="round")
    # the object, filled with the observed radiance
    ax.add_patch(FancyBboxPatch((cx - 7.0, PHOTO_Y + 6.2), 14.0, 10.0,
                                boxstyle="round,pad=0,rounding_size=1.0",
                                facecolor=srgb(obs), edgecolor="none", zorder=3))
    if labels and numbers:
        ax.text(cx, PHOTO_Y - 2.6, "pixel value  {:.3f}".format(float(obs[0])),
                ha="center", va="center", fontsize=12.5, color=C_MUTED)


def draw_sun(ax, cx, cy, E, E_ref):
    """A sun whose size, ray length and saturation all read off E.

    Normalised against the brightest scene rather than against 1, so the contrast
    survives whatever absolute units the numbers in the scene list happen to be in.
    """
    t = float(np.clip(luminance(E) / E_ref, 0.0, 1.0)) ** 0.5   # perceptual spacing
    colour = tuple(np.array(matplotlib.colors.to_rgb(C_DIRECT)) * t + (1.0 - t))
    r = SUN_R0 + SUN_R1 * t
    n_rays = 8 + int(round(6 * t))
    inner, outer = r + 0.9, r + SUN_GAP + SUN_RAY * t
    for k in range(n_rays):
        a = 2.0 * np.pi * k / n_rays
        ax.plot([cx + inner * np.cos(a), cx + outer * np.cos(a)],
                [cy + inner * np.sin(a), cy + outer * np.sin(a)],
                color=colour, lw=2.2, zorder=4, solid_capstyle="round")
    ax.add_patch(Circle((cx, cy), r, facecolor=colour, edgecolor="none", zorder=5))


def draw_neighbour(ax, cx, cy, E):
    """A neighbouring object sending its own colour towards the surface.

    Drawn at the chromaticity of E normalised to full brightness: what the glyph has
    to carry is the hue it lends to the surface, not where it sits on the exposure.
    The arrows point left, towards the swatch, so the card still reads left to right
    as surface times light even though the light is now an object.
    """
    E = np.asarray(E, dtype=float)
    hue = E / max(float(E.max()), 1e-12)
    colour = srgb(hue)
    # The arrows are thin strokes on white, where the block's own tint washes out;
    # darkening the same hue keeps them legible once the slide is projected.
    arrow_colour = srgb(0.35 * hue)
    half = SWATCH / 2.0
    ax.add_patch(FancyBboxPatch((cx - half, cy - half), SWATCH, SWATCH,
                                boxstyle="round,pad=0,rounding_size=1.0",
                                facecolor=colour, edgecolor=C_MUTED,
                                linewidth=1.2, zorder=4))
    for dy in (-2.6, 0.0, 2.6):
        ax.annotate("", xy=(cx - half - 4.2, cy + dy), xytext=(cx - half - 0.6, cy + dy),
                    arrowprops=dict(arrowstyle="-|>", color=arrow_colour, lw=2.4,
                                    shrinkA=0, shrinkB=0, mutation_scale=13),
                    zorder=4)


def scene_box(labels, n_caption_lines=1, numbers=True):
    """Geometry of a scene card: (y0, height, centre of the row of symbols).

    Three sets, because the caption and the numbers are the only things that need the
    head- and footroom, and they are dropped independently.  The card closes around
    whichever of them survives: leaving the full box would put a void under the glyphs
    that reads as a missing element rather than as margin.
    """
    if labels and numbers:
        h = SCENE_H + 3.2 * (n_caption_lines - 1)
        return SCENE_Y, h, SCENE_Y + 11.0
    if labels:
        h = 2.8 + 2.0 * SUN_REACH + 6.0 + 3.2 * (n_caption_lines - 1)
        return SCENE_Y, h, SCENE_Y + 2.8 + SUN_REACH
    h = 2.0 * (SUN_REACH + 2.6)
    y0 = PHOTO_Y - 9.0 - h
    return y0, h, y0 + h / 2.0


def draw_scene(ax, cx, scene, E_ref, labels=True, numbers=True, n_caption_lines=1,
               wide=True):
    """One explanation: an albedo swatch times a light.

    `wide` buys the horizontal room a caption needs, and the room a neighbour needs for
    the arrows it sends towards the swatch.  A card holding two suns and no words does
    without it and closes in on the glyphs.
    """
    bounce = scene["light"] == "bounce"
    y0, h, cy = scene_box(labels, n_caption_lines, numbers)
    w = SCENE_W if wide else SCENE_W - 8.0
    card(ax, cx, y0, w, h, "white", lw=1.4)
    if labels:
        ax.text(cx, y0 + h - 3.4 - 1.6 * (n_caption_lines - 1), scene["caption"],
                ha="center", va="center", fontsize=12.5, color=C_INK,
                linespacing=1.45)

    # The sun reaches further from its centre than the swatch does, so a pair placed
    # symmetrically about cx sits visibly off-centre in the card; shift it by half the
    # difference and the margins match on both sides.  A neighbour is the same size as
    # the swatch, so that set needs no shift.
    dx = 10.0 if wide else 8.0
    reach = BOUNCE_REACH if bounce else SUN_REACH
    cx = cx + (SWATCH / 2.0 - reach) / 2.0
    sx, ex = cx - dx, cx + dx
    ax.add_patch(Rectangle((sx - SWATCH / 2.0, cy - SWATCH / 2.0), SWATCH, SWATCH,
                           facecolor=srgb(scene["a"]), edgecolor=C_MUTED,
                           linewidth=1.2, zorder=4))
    ax.text(cx, cy, r"$\times$", ha="center", va="center", fontsize=21, color=C_MUTED)
    if bounce:
        draw_neighbour(ax, ex, cy, scene["E"])
    else:
        draw_sun(ax, ex, cy, scene["E"], E_ref)

    if labels and numbers:
        ax.text(sx, y0 + 2.4, r"$\rho = {:.1f}$".format(scene["a"][0]),
                ha="center", va="center", fontsize=14, color=C_INK)
        ax.text(ex, y0 + 2.4, r"$E = {:.2f}$".format(scene["E"][0]),
                ha="center", va="center", fontsize=14, color=C_INK)
    return y0 + h


def draw(scenes, obs, title=None, labels=True):
    # The numeric annotations only mean anything when a single number says everything
    # about a scene; a coloured albedo or a coloured light would need three.
    numbers = all(is_neutral(s) for s in scenes)
    n_caption_lines = max(s["caption"].count("\n") + 1 for s in scenes)
    cx_all, xlim = columns(len(scenes))
    box = scene_box(labels, n_caption_lines, numbers)
    scene_top = box[0] + box[1]

    fig, ax = plt.subplots(figsize=(xlim / 10.0, 6.9))
    ax.set_xlim(0, xlim)
    # `bbox_inches="tight"` crops to the axes, not to the artists inside it, so the
    # band the labels would have occupied has to come off the limits themselves.
    ax.set_ylim(0.0 if labels else scene_box(labels)[0] - 3.0,
                69.0 if labels else PHOTO_Y + PHOTO_H + 3.0)
    ax.set_aspect("equal")
    ax.axis("off")

    if labels:
        ax.text(sum(cx_all) / len(cx_all), 66.5, "what the camera records",
                ha="center", va="center", fontsize=13.5, color=C_MUTED)
    for cx in cx_all:
        draw_photograph(ax, cx, obs, labels, numbers)
    for left, right in zip(cx_all, cx_all[1:]):
        ax.text(0.5 * (left + right), PHOTO_Y + PHOTO_H / 2.0, "=",
                ha="center", va="center", fontsize=34, color=C_HILITE)
        if labels:
            ax.text(0.5 * (left + right), 0.5 * (scene_top + box[0]), "or",
                    ha="center", va="center", fontsize=14, color=C_MUTED,
                    style="italic")

    sun_lum = [luminance(s["E"]) for s in scenes if s["light"] == "sun"]
    E_ref = max(sun_lum) if sun_lum else 1.0
    wide = labels or any(s["light"] == "bounce" for s in scenes)
    for cx, scene in zip(cx_all, scenes):
        top = draw_scene(ax, cx, scene, E_ref, labels, numbers, n_caption_lines, wide)
        ax.annotate("", xy=(cx, PHOTO_Y - (5.6 if labels and numbers else 1.8)),
                    xytext=(cx, top + 1.4),
                    arrowprops=dict(arrowstyle="-|>", color=C_MUTED, lw=1.8,
                                    shrinkA=0, shrinkB=0, mutation_scale=17))

    if labels and numbers:
        prod = " = ".join(r"${:.1f} \times {:.2f}$".format(s["a"][0], s["E"][0])
                          for s in scenes)
        ax.text(sum(cx_all) / len(cx_all), 2.2,
                r"the camera only ever sees the product $\rho E$:   " + prod,
                ha="center", va="center", fontsize=14.5, color=C_INK)
    if title:
        ax.text(sum(cx_all) / len(cx_all), 68.6, title, ha="center", va="top",
                fontsize=17, color=C_INK)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    default_out = Path(__file__).resolve().parents[2] / "PresentationImages"
    ap.add_argument("--out", default=str(default_out),
                    help="directory the PNG is written to")
    ap.add_argument("--name", default=None,
                    help="output stem (default: illposed_albedo_light, or "
                         "illposed_reflection under --reflection, with _bare "
                         "appended under --bare)")
    ap.add_argument("--dpi", type=int, default=DPI)
    ap.add_argument("--title", default=None,
                    help="optional heading drawn inside the figure "
                         "(off by default: the slide carries its own)")
    ap.add_argument("--bare", action="store_true",
                    help="drop every word, keeping only the shapes, the = and the "
                         "times: for a slide that talks about the ambiguity in "
                         "general, without committing to a reflectance model")
    ap.add_argument("--reflection", action="store_true",
                    help="add a third hypothesis, in which the tint is borrowed from "
                         "a coloured neighbour rather than owned by the surface")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    scenes = SCENES_REFLECTION if args.reflection else SCENES_LEVEL
    obs = check(scenes)

    fig = draw(scenes, obs, args.title, labels=not args.bare)
    stem = args.name or ("illposed_reflection" if args.reflection
                         else "illposed_albedo_light")
    if args.name is None and args.bare:
        stem += "_bare"
    path = out / "{}.png".format(stem)
    fig.savefig(path, dpi=args.dpi, facecolor="white", bbox_inches="tight",
                pad_inches=0.28)
    plt.close(fig)
    print("  + {}".format(path.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
