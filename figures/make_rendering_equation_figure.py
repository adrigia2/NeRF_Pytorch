#!/usr/bin/env python
"""make_rendering_equation_figure.py -- every term of the rendering equation, drawn.

    python make_rendering_equation_figure.py --out ../../PresentationImages

Writes one PNG:

  rendering_equation.png   the equation of background.tex:21 printed across the top with
                           each term in its own colour, and below it a cross-section of a
                           surface point in which the element each term names is drawn in
                           that same colour.

The figure exists so that the equation can be read as a picture rather than as a formula.
The colour is the only link between the two halves, which is why there is no legend: an
eye that lands on the amber $L_i$ has already found the amber arrows arriving at the point.

**A cross-section, not a hemisphere in perspective.**  The other geometry diagrams of the
thesis draw the hemisphere in assonometry, which is right when the subject is a set of
directions, and wrong here: the angle theta between an incident direction and the normal is
what the cosine factor is about, and in assonometry it is foreshortened into something the
reader has to take on trust.  In section it is simply the angle it is.

**The lobe is evaluated, not drawn.**  It comes from `cook_torrance` in
make_brdf_lobes_figure, the same Cook-Torrance used for the BRDF figure of the thesis, at
the incident direction actually drawn; `check()` verifies its peak really sits at the
mirror direction and that it vanishes at the horizon, so the picture cannot drift away from
the physics it illustrates.  The radius is compressed by `compress` (r = f^(1/4)) because a
GGX peak runs some three orders of magnitude above rho/pi and would otherwise be a spike.

L_e is drawn greyed out.  Nothing in this pipeline emits light, so its value is zero
everywhere; dropping it from the picture, though, would leave a term of the printed
equation with nothing to point at, which is worse than showing a term that happens to
vanish.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _paths  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.patches import Circle, Polygon                         # noqa: E402

# The palette every diagram of the thesis is drawn with.
from make_geometry_diagrams import (                                   # noqa: E402
    DPI, C_EDGE, C_RAY, C_HILITE, C_ACCEPT, C_DIRECT, C_INDIR, C_GEOM, C_CAM,
)
# The BRDF itself, and the two conventions the polar drawing depends on.
from make_brdf_lobes_figure import (                                   # noqa: E402
    cook_torrance, ggx_d, lambert, luminance, hemisphere_dirs, compress, polar_xy,
)

# -- One colour per term, used in the equation and in the drawing alike ------
C_OUT   = C_RAY        # L_o, omega_o: what leaves the point
C_EMIT  = "#9aa7b4"    # L_e: greyed out, it is zero here
C_HEMI  = C_ACCEPT     # Omega+ and d omega_i: the domain of the integral
C_BRDF  = C_INDIR      # f_r: the lobe
C_IN    = C_DIRECT     # L_i: what arrives
C_COS   = C_HILITE     # (omega_i . n): the highlighted direction and its angle
C_INK   = C_CAM        # everything that is not a term of the equation

# -- The configuration the drawing and the checks share ---------------------
THETA_I_DEG = 55.0                      # incident direction, left of the normal
THETA_O_DEG = 38.0                      # the direction we are solving for
FAN_DEG = (-79.0, -68.0, -41.0, -27.0)  # the rest of the incoming fan
DOMEGA_DEG = 4.6                        # half-width of the d omega_i wedge
ROUGHNESS = 0.30
F0 = np.array([0.05, 0.05, 0.05])
RHO = np.array([0.40, 0.42, 0.47])
W_DIFF, W_SPEC = 0.55, 0.45
N_DIR = 2001            # as in make_brdf_lobes_figure: the sharp lobe needs the density

# The free-standing red note on the cosine.  Off: the red arrow, the angle arc and the
# colour it shares with the term in the equation already say it, and on a slide the
# sentence was the only prose left in the drawing.  Set True to bring it back.
COS_NOTE = False

R_ARC = 1.0             # radius of the hemisphere arc
R_LOBE = 0.46           # radius the lobe peak is scaled to

# Type sizes.  What decides whether a label is readable on a projected slide is not its
# size in points but its size against the drawing, and the drawing is sized in inches by
# LAYOUT below: a smaller figure at the same point size is a bigger label.  So the two
# tables belong together, and neither should be touched without the other.
FS = dict(equation=27, symbol=21, named=20, small=19, note=18, caption=13.5, gloss=12.5)

# The gloss row is nearly as wide as the equation, so the variant that carries it needs a
# wider page; the one that does not can close in on the drawing, which is what makes the
# labels of that variant read about 40 % larger.
LAYOUT = {
    True:  dict(figsize=(13.2, 7.0), top=0.875, bottom=0.115),
    False: dict(figsize=(10.2, 5.4), top=0.885, bottom=0.050),
}


def lobe():
    """The BRDF as a function of the outgoing direction, for the incident one drawn.

    Returns (theta, radius, specular luminance).  Diffuse and specular are summed because
    the equation has one f_r, not two; the split belongs to the BRDF figure, not here.
    """
    theta = np.linspace(-np.pi / 2.0, np.pi / 2.0, N_DIR)
    wo = hemisphere_dirs(theta)
    n = np.array([0.0, 0.0, 1.0])
    ti = np.radians(THETA_I_DEG)
    wi = np.array([-np.sin(ti), 0.0, np.cos(ti)])          # left of the normal

    f_d = W_DIFF * lambert(RHO, theta.shape)
    f_s = W_SPEC * cook_torrance(wi, wo, n, ROUGHNESS, F0)
    lum = luminance(f_d + f_s)
    r = compress(lum)

    # The microfacet distribution on its own, for the check: D peaks exactly at the
    # mirror direction, the whole f_s does not.
    h = wi + wo
    h = h / np.maximum(np.linalg.norm(h, axis=-1, keepdims=True), 1e-12)
    ndf = ggx_d(np.clip(h @ n, 0.0, 1.0), ROUGHNESS)
    return theta, R_LOBE * r / r.max(), luminance(f_s), ndf


def check(theta, spec, ndf):
    """Refuse to draw unless the lobe really is the BRDF it claims to be.

    The peak of the *whole* specular term does not sit at the mirror direction: the
    1/(4 (n.wi)(n.wo)) denominator and the Fresnel factor both grow towards grazing and
    drag it outwards, by about a degree at this roughness.  That is real, and a check
    demanding the peak land exactly on the mirror direction would be demanding a wrong
    picture.  So the exact test is made on the microfacet distribution, which does peak
    there by construction, and the displacement of the full lobe is measured and
    reported rather than forbidden.
    """
    n = np.array([0.0, 0.0, 1.0])
    drawn = np.radians(np.array(FAN_DEG + (-THETA_I_DEG,)))
    cos_i = hemisphere_dirs(drawn) @ n
    print("  incoming directions: {} drawn, min (omega_i . n) = {:.4f}"
          .format(len(drawn), float(cos_i.min())))
    if cos_i.min() <= 0.0:
        raise ValueError("an incoming direction is not in the upper hemisphere")

    horizon = float(max(spec[0], spec[-1]))
    print("  specular lobe at the horizon: {:.3e}".format(horizon))
    if horizon != 0.0:
        raise ValueError("the specular lobe does not vanish at theta = +/- 90 deg")

    step = np.degrees(theta[1] - theta[0])
    ndf_peak = np.degrees(theta[int(np.argmax(ndf))])
    print("  microfacet distribution peaks at {:+.3f} deg, mirror direction {:+.3f} deg,"
          " error {:.4f} deg (one sample = {:.4f})"
          .format(ndf_peak, THETA_I_DEG, abs(ndf_peak - THETA_I_DEG), step))
    if abs(ndf_peak - THETA_I_DEG) > step:
        raise ValueError("the microfacet distribution does not peak at the mirror "
                         "direction: the lobe is not the BRDF it is labelled as")

    peak = np.degrees(theta[int(np.argmax(spec))])
    print("  whole specular term peaks at {:+.3f} deg, dragged {:+.3f} deg towards "
          "grazing by the geometric and Fresnel factors".format(peak, peak - THETA_I_DEG))
    if not 0.0 <= peak - THETA_I_DEG <= 5.0:
        raise ValueError("the specular peak is displaced by {:+.3f} deg, which is not "
                         "the small outward shift the model predicts"
                         .format(peak - THETA_I_DEG))


def place_chunks(fig, y, chunks, fontsize, pad_em=0.24):
    """Lay coloured text chunks left to right on one centred row.

    matplotlib has no coloured-run text, so the row is tiled by hand: draw every chunk,
    measure it against the renderer, then move each one to where the widths say it goes.
    """
    texts = [fig.text(0.0, y, s, color=c, fontsize=fontsize, ha="left", va="center")
             for s, c in chunks]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    widths = [t.get_window_extent(renderer=renderer).width for t in texts]
    pad = pad_em * fontsize * fig.dpi / 72.0
    total = sum(widths) + pad * (len(texts) - 1)
    page = fig.get_figwidth() * fig.dpi
    x = 0.5 * page - 0.5 * total
    for t, w in zip(texts, widths):
        t.set_x(x / page)
        x += w + pad
    return texts


def arc(ax, a0, a1, r, color, lw=2.2, ls="-", z=3, n=200):
    """A circular arc, angles measured from the normal as everywhere else here."""
    t = np.linspace(np.radians(a0), np.radians(a1), n)
    ax.plot(*polar_xy(t, np.full_like(t, r)), color=color, lw=lw, ls=ls,
            zorder=z, solid_capstyle="round")


def at(a_deg, r):
    """Polar to page coordinates, angle from the normal, positive towards +x."""
    x, y = polar_xy(np.radians(a_deg), r)
    return float(x), float(y)


def draw_eye(ax, a_deg, r, color):
    """A small eye at the far end of omega_o, looking back down the direction."""
    cx, cy = at(a_deg, r)
    u = np.array(at(a_deg, 1.0))
    u = u / np.linalg.norm(u)
    v = np.array([-u[1], u[0]])
    s = np.linspace(-1.0, 1.0, 60)
    for sign in (1.0, -1.0):
        pts = np.array([cx, cy]) + np.outer(s * 0.135, v) \
            + np.outer(sign * 0.070 * (1.0 - s ** 2), u)
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=1.8, zorder=8,
                solid_capstyle="round")
    ax.add_patch(Circle((cx, cy), 0.032, facecolor=color, edgecolor="none", zorder=9))


def draw(theta, radius, title=None, gloss=True):
    lay = LAYOUT[bool(gloss)]
    fig, ax = plt.subplots(figsize=lay["figsize"])
    fig.subplots_adjust(left=0.02, right=0.98, bottom=lay["bottom"], top=lay["top"])
    ax.set_aspect("equal")
    ax.axis("off")

    # -- the surface -------------------------------------------------------
    ax.fill_between([-1.55, 1.55], -0.20, 0.0, facecolor=C_GEOM, alpha=0.45,
                    edgecolor="none", zorder=0)
    ax.plot([-1.55, 1.55], [0.0, 0.0], color=C_EDGE, lw=2.0, zorder=2)

    # -- Omega+, the domain of the integral --------------------------------
    arc(ax, -90.0, 90.0, R_ARC, C_HEMI, lw=2.2, z=3)
    ax.text(*at(14.0, R_ARC + 0.11), r"$\Omega^+$", color=C_HEMI, fontsize=FS["symbol"],
            ha="center", va="center", zorder=12)

    # -- L_e, present in the equation and zero in this pipeline ------------
    for a in np.linspace(-72.0, 72.0, 9):
        p0, p1 = at(a, 0.055), at(a, 0.120)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=C_EMIT, lw=1.8, zorder=7,
                solid_capstyle="round")
    # Both the emission and the BRDF live at x, so this label cannot sit next to what it
    # names without landing inside the lobe; it is led out to the empty ground instead.
    ax.annotate(r"$L_e$", xy=(0.105, 0.055), xytext=(0.40, -0.135), color=C_EMIT,
                fontsize=FS["named"], ha="left", va="center", zorder=12,
                arrowprops=dict(arrowstyle="-", color=C_EMIT, lw=1.1,
                                shrinkA=3, shrinkB=3))

    # -- the BRDF lobe, evaluated ------------------------------------------
    lx, ly = polar_xy(theta, radius)
    ax.fill(np.concatenate([[0.0], lx, [0.0]]), np.concatenate([[0.0], ly, [0.0]]),
            facecolor=C_BRDF, alpha=0.18, edgecolor="none", zorder=5)
    ax.plot(lx, ly, color=C_BRDF, lw=2.0, zorder=6)
    ax.text(*at(53.0, 0.66), r"$f_r$", color=C_BRDF, fontsize=FS["symbol"], ha="left",
            va="center", zorder=12)

    # -- L_i, the incoming fan ---------------------------------------------
    for a in FAN_DEG:
        ax.annotate("", xy=at(a, 0.205), xytext=at(a, R_ARC - 0.02),
                    arrowprops=dict(arrowstyle="-|>", color=C_IN, lw=2.0,
                                    shrinkA=0, shrinkB=0, mutation_scale=15), zorder=7)
    ax.text(*at(-42.0, R_ARC + 0.42), r"$L_i(x,\,\omega_i)$", color=C_IN, fontsize=FS["named"],
            ha="center", va="center", zorder=12)

    # -- the highlighted direction and its cosine --------------------------
    ax.annotate("", xy=at(-THETA_I_DEG, 0.205), xytext=at(-THETA_I_DEG, R_ARC - 0.02),
                arrowprops=dict(arrowstyle="-|>", color=C_COS, lw=2.6,
                                shrinkA=0, shrinkB=0, mutation_scale=17), zorder=8)
    ax.text(*at(-THETA_I_DEG, R_ARC + 0.22), r"$\omega_i$", color=C_COS, fontsize=FS["symbol"],
            ha="center", va="center", zorder=12)
    arc(ax, -THETA_I_DEG, 0.0, 0.52, C_COS, lw=1.8, z=7)
    ax.text(*at(-33.0, 0.42), r"$\theta$", color=C_COS, fontsize=FS["small"],
            ha="center", va="center", zorder=12)
    if COS_NOTE:
        # The incoming fan starts at x = -0.98, so this block has to stay clear of it.
        ax.text(-1.80, 0.32, r"$(\omega_i\cdot\mathbf{n}) = \cos\theta$", color=C_COS,
                fontsize=FS["note"], ha="left", va="center", zorder=12)
        ax.text(-1.80, 0.205, "light arriving at a slant\nspreads over more surface",
                color=C_COS, fontsize=FS["caption"], ha="left", va="top", zorder=12,
                linespacing=1.35)

    # -- d omega_i, the differential wedge on the arc ----------------------
    t = np.linspace(np.radians(-THETA_I_DEG - DOMEGA_DEG),
                    np.radians(-THETA_I_DEG + DOMEGA_DEG), 40)
    inner = np.stack(polar_xy(t, np.full_like(t, R_ARC - 0.045)), axis=-1)
    outer = np.stack(polar_xy(t, np.full_like(t, R_ARC + 0.045)), axis=-1)
    ax.add_patch(Polygon(np.vstack([inner, outer[::-1]]), closed=True,
                         facecolor=C_HEMI, edgecolor=C_HEMI, lw=1.2, alpha=0.85,
                         zorder=9))
    ax.annotate(r"$\mathrm{d}\omega_i$",
                xy=at(-THETA_I_DEG - DOMEGA_DEG, R_ARC + 0.045),
                xytext=at(-THETA_I_DEG - 15.0, R_ARC + 0.34),
                color=C_HEMI, fontsize=FS["small"], ha="center", va="center", zorder=12,
                arrowprops=dict(arrowstyle="-", color=C_HEMI, lw=1.2,
                                shrinkA=3, shrinkB=2))

    # -- the normal --------------------------------------------------------
    ax.annotate("", xy=(0.0, 1.24), xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle="-|>", color=C_INK, lw=2.0,
                                shrinkA=0, shrinkB=0, mutation_scale=16), zorder=10)
    ax.text(0.055, 1.26, r"$\mathbf{n}$", color=C_INK, fontsize=FS["symbol"], ha="left",
            va="center", zorder=12)

    # -- L_o, what leaves the point towards the camera ---------------------
    ax.annotate("", xy=at(THETA_O_DEG, 1.30), xytext=at(THETA_O_DEG, 0.145),
                arrowprops=dict(arrowstyle="-|>", color=C_OUT, lw=2.6,
                                shrinkA=0, shrinkB=0, mutation_scale=17), zorder=10)
    ax.text(*at(THETA_O_DEG - 6.0, 0.80), r"$\omega_o$", color=C_OUT, fontsize=FS["symbol"],
            ha="right", va="center", zorder=12)
    draw_eye(ax, THETA_O_DEG, 1.46, C_OUT)
    ax.text(*at(THETA_O_DEG + 13.0, 1.58), r"$L_o(x,\,\omega_o)$", color=C_OUT,
            fontsize=FS["named"], ha="center", va="center", zorder=12)

    # -- the point itself --------------------------------------------------
    ax.add_patch(Circle((0.0, 0.0), 0.030, facecolor=C_INK, edgecolor="white",
                        lw=1.0, zorder=11))
    ax.text(-0.055, -0.105, r"$x$", color=C_INK, fontsize=FS["named"], ha="right", va="center",
            zorder=12)

    ax.set_xlim(-1.62, 1.62)
    ax.set_ylim(-0.30, 1.36)

    # -- the equation, one colour per term ---------------------------------
    place_chunks(fig, 0.945, [
        (r"$L_o(x,\,\omega_o)$", C_OUT),
        (r"$=$", C_INK),
        (r"$L_e(x,\,\omega_o)$", C_EMIT),
        (r"$+$", C_INK),
        (r"$\int_{\Omega^+}$", C_HEMI),
        (r"$f_r(x,\,\omega_i,\,\omega_o)$", C_BRDF),
        (r"$L_i(x,\,\omega_i)$", C_IN),
        (r"$(\omega_i\cdot\mathbf{n})$", C_COS),
        (r"$\mathrm{d}\omega_i$", C_HEMI),
    ], fontsize=FS["equation"])

    if not gloss:
        return fig

    place_chunks(fig, 0.048, [
        (r"$L_o$", C_OUT), ("what leaves x towards the camera", C_INK), ("|", C_GEOM),
        (r"$L_e$", C_EMIT), ("what x emits, zero here", C_INK), ("|", C_GEOM),
        (r"$\Omega^+$", C_HEMI), ("every direction above the surface", C_INK),
        ("|", C_GEOM),
        (r"$f_r$", C_BRDF), ("how much of it is reflected", C_INK), ("|", C_GEOM),
        (r"$L_i$", C_IN), ("what arrives", C_INK), ("|", C_GEOM),
        (r"$(\omega_i\cdot\mathbf{n})$", C_COS), ("how slanted it arrives", C_INK),
    ], fontsize=FS["gloss"])

    if title:
        fig.text(0.5, 0.995, title, ha="center", va="top", fontsize=FS["symbol"], color=C_INK)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    default_out = Path(__file__).resolve().parents[2] / "PresentationImages"
    ap.add_argument("--out", default=str(default_out),
                    help="directory the PNG is written to")
    ap.add_argument("--name", default=None,
                    help="output stem (default: rendering_equation, or "
                         "rendering_equation_plain under --no-gloss)")
    ap.add_argument("--dpi", type=int, default=DPI)
    ap.add_argument("--title", default=None,
                    help="optional heading drawn inside the figure "
                         "(off by default: the slide carries its own)")
    ap.add_argument("--no-gloss", dest="gloss", action="store_false",
                    help="drop the row of plain-English glosses under the drawing; the "
                         "page closes in on the picture, so every label reads larger")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    theta, radius, spec, ndf = lobe()
    check(theta, spec, ndf)

    fig = draw(theta, radius, args.title, gloss=args.gloss)
    stem = args.name or ("rendering_equation" if args.gloss
                         else "rendering_equation_plain")
    path = out / "{}.png".format(stem)
    fig.savefig(path, dpi=args.dpi, facecolor="white", bbox_inches="tight",
                pad_inches=0.24)
    plt.close(fig)
    print("  + {}".format(path.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
