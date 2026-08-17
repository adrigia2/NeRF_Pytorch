#!/usr/bin/env python
"""make_brdf_lobes_figure.py -- the three BRDF lobes combined into one material.

    python make_brdf_lobes_figure.py --out ../Doc/images/brdf-explenation

Writes `brdf_combined.png`, the figure that closes Section 2.1.2.

The figure right above it in the thesis (`fig:brdf_lobes`, three CC0 panels) shows the
three lobe shapes SEPARATELY, one per panel.  This one shows what the text says next: that
the balance between them, and their relative spectral content, is what makes a material
look matte, glossy or metallic.  So it is a single picture, and the three lobes are
STACKED: the diffuse hemisphere at the base, the glossy lobe on top of it, the mirror
needle on top of that, so the outer boundary is f_r itself.  The stacking is the whole
argument -- nothing has to be written to say "combination", it is what the reader sees.

The shapes are computed with the formulas the thesis has already written down, not drawn
by hand: GGX D (background.tex:145), Schlick F (:157), Smith G = G1(wi)G1(wo) (:165),
assembled into the Cook-Torrance BRDF of eq:cook-torrance (:130), plus the Lambertian
rho/pi of the diffuse item.  `report()` then checks the drawing against the physics.

TWO THINGS THE CAPTION HAS TO SAY, because the figure alone would suggest otherwise:

  1. Glossy and mirror are the SAME microfacet term at two roughnesses, not two
     independent physical terms -- in 2.1.2 both are called f_s.  The superposition drawn
     here is a didactic composition, not a three-term model the thesis uses anywhere.

  2. THE RADIAL AXIS IS COMPRESSED, r = f_r ** RADIAL_POWER, and the same compression is
     applied to every band.  This is not a cosmetic choice, it is forced: a GGX peak goes
     as 1/roughness^4, so at the two roughnesses drawn here the sharp lobe stands three
     orders of magnitude above rho/pi.  On a linear radius the diffuse disc is thinner
     than the line that draws it, and the figure ends up asserting that a material is its
     specular lobe alone, which is the opposite of what the section says.  Compressing
     keeps the ORDER and the nesting of the three bands, which is what has to be read
     here, and gives up the proportions, which are not readable on this scale anyway.
     The radial ticks are labelled in BRDF units so the compression stays visible.

The three weights are a convex mixture, summing to 1, so each is the share of the
reflection its component carries and the "balance between the lobes" of the section text
is literally what they are.  They are not free: with all three at 1 the material reflects
more than it receives, which `report` catches as a directional albedo of 1.28.

The mirror lobe stays a thin needle whatever weight it is given, because its width is set
by the roughness and not by the weight.  That is correct and it is the point: a mirror
concentrates its energy, it does not spread it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({"font.size": 13})

DPI = 190

# ── The material being decomposed ────────────────────────────────────────────
# One material, one incident direction.  The lobes are its three components, and the
# weights are how much of the reflection each one carries.
THETA_I = 55.0                 # incident direction, degrees from the normal
BASE_COLOR = np.array([0.62, 0.45, 0.31])     # rho, and F0 of the metal component

# Roughness of the two specular components.  The gap has to be wide enough that the two
# read as different lobes and not as one lobe with a bump: at 0.35 the glossy one spans
# tens of degrees, at 0.15 the mirror one a couple.  Not sharper than 0.15, even though a
# real mirror is: the peak goes as 1/r^4, and past that even the compressed radius cannot
# hold the needle and the disc in one frame.
ROUGH_GLOSSY = 0.35
ROUGH_MIRROR = 0.15

# Weights on the three components.  They are a CONVEX MIXTURE: they sum to 1 and each one
# is the share of the reflection that component carries, which is exactly the "balance
# between the lobes" the section talks about.  Not free parameters, then: with all three
# at 1 the material reflects more light than it receives (directional albedo 1.28, caught
# by `report`), because each component is already a complete BRDF on its own.
#
# The glossy share is high for a dielectric coat, whose F0 of 4% makes its lobe
# intrinsically dim next to a conductor's: at a smaller share its band came out a sliver
# between the diffuse disc and the mirror needle, and the figure showed two components
# where it claims three.  `report` measures that band and fails if it thins again.
W_DIFFUSE = 0.40
W_GLOSSY = 0.50
W_MIRROR = 0.10

# Compression of the radial axis, r = f_r ** RADIAL_POWER.  See the module docstring: it
# is forced by the 1/r^4 of the GGX peak, not chosen for looks.  1/4 puts the sharp lobe
# about seven times the diffuse disc, which shows both.
RADIAL_POWER = 0.25

# F0 of the two specular components.  Achromatic for the dielectric glossy lobe, tinted
# with the base colour for the metallic mirror one: that contrast IS the "relative
# spectral content" of the sentence this figure illustrates, and it is why the two lobes
# are drawn in two different colours rather than two shades of one.
F0_GLOSSY = np.array([0.04, 0.04, 0.04])      # dielectric, ~4% at normal incidence
F0_MIRROR = BASE_COLOR                        # conductor: F0 is the base colour

# ── Appearance ───────────────────────────────────────────────────────────────
C_INK = "#222222"
C_DIFF = "#e8a33d"     # amber, the same the geometry diagrams use for the diffuse term
C_GLOSS = "#5b9bd5"    # the generic-ray blue of make_geometry_diagrams
C_MIRR = "#8452c9"     # the indirect/violet, kept for the sharpest component
C_INCID = "#d62728"

N_DIR = 2001           # directions the lobes are evaluated at, over 180 degrees.
                       # The sharp lobe is a couple of degrees wide, so at 721 it was
                       # carried by three samples and came out as a polygon.
SPHERE_PX = 260        # side of the shaded sphere, in pixels
LIGHT_DIR = np.array([-0.55, 0.45, 0.70])     # key light of the shaded sphere


def ggx_d(cos_h: np.ndarray, roughness: float) -> np.ndarray:
    """GGX NDF, the formula of background.tex:145: D = a^2 / (pi (c^2(a^2-1)+1)^2).

    The thesis writes alpha = roughness^2, so the argument here is the roughness and the
    squaring happens inside.  Same function as `ggx_d` in make_pbr_model_diagram, which
    takes alpha directly.
    """
    a2 = (roughness ** 2) ** 2
    return a2 / (np.pi * (cos_h ** 2 * (a2 - 1.0) + 1.0) ** 2)


def schlick_f(cos_d: np.ndarray, f0: np.ndarray) -> np.ndarray:
    """Schlick's approximation, background.tex:157.  cos_d is wo . h."""
    return f0 + (1.0 - f0) * (1.0 - cos_d)[..., None] ** 5


def smith_g1(cos_x: np.ndarray, roughness: float) -> np.ndarray:
    """One factor of the Smith geometry term, background.tex:165.

    Schlick-GGX with the direct-lighting remap k = alpha/2, which is the pairing that
    goes with the D above; G = G1(wi) G1(wo) is assembled by the caller.
    """
    k = (roughness ** 2) / 2.0
    return cos_x / (cos_x * (1.0 - k) + k)


def cook_torrance(wi: np.ndarray, wo: np.ndarray, n: np.ndarray,
                  roughness: float, f0: np.ndarray) -> np.ndarray:
    """f_s of eq:cook-torrance, background.tex:130.  Returns (..., 3).

    Zero wherever either direction is below the surface: the BRDF is only defined on the
    upper hemisphere, and without the clamp the denominator changes sign and the lobe
    grows a mirror image underneath the surface.
    """
    h = wi + wo
    h = h / np.maximum(np.linalg.norm(h, axis=-1, keepdims=True), 1e-12)
    ndl = np.sum(n * wi, axis=-1)
    ndv = np.sum(n * wo, axis=-1)
    ndh = np.clip(np.sum(n * h, axis=-1), 0.0, 1.0)
    vdh = np.clip(np.sum(wo * h, axis=-1), 0.0, 1.0)

    ok = (ndl > 1e-6) & (ndv > 1e-6)
    ndl_s, ndv_s = np.where(ok, ndl, 1.0), np.where(ok, ndv, 1.0)

    d = ggx_d(ndh, roughness)
    g = smith_g1(ndl_s, roughness) * smith_g1(ndv_s, roughness)
    f = schlick_f(vdh, f0)
    return np.where(ok[..., None], f * (d * g / (4.0 * ndl_s * ndv_s))[..., None], 0.0)


def lambert(rho: np.ndarray, shape: tuple) -> np.ndarray:
    """f_d = rho/pi, background.tex:72.  Constant over the hemisphere by definition."""
    return np.broadcast_to(rho / np.pi, shape + (3,)).copy()


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.2126, 0.7152, 0.0722])


def hemisphere_dirs(theta: np.ndarray) -> np.ndarray:
    """(N, 3) outgoing directions in the plane of incidence, indexed by the signed angle
    from the normal.  The plane of incidence is the x-z plane, the normal is +z."""
    return np.stack([np.sin(theta), np.zeros_like(theta), np.cos(theta)], axis=-1)


def build_case() -> dict:
    """The three components evaluated over the outgoing hemisphere."""
    n = np.array([0.0, 0.0, 1.0])
    ti = np.radians(THETA_I)
    # wi points AWAY from the surface, towards where the light comes from, so that the
    # half vector wi + wo is the usual one.  It sits at -x so that the mirror direction,
    # and with it the two specular lobes, land at +x where there is room for them.
    wi = np.array([-np.sin(ti), 0.0, np.cos(ti)])
    r = 2.0 * np.dot(n, wi) * n - wi              # mirror direction

    theta = np.linspace(-np.pi / 2, np.pi / 2, N_DIR)
    wo = hemisphere_dirs(theta)

    comps = [
        dict(key="diffuse", label=r"diffuse  $f_d = \rho/\pi$", color=C_DIFF,
             weight=W_DIFFUSE, f=lambert(BASE_COLOR, theta.shape)),
        dict(key="glossy", label=rf"glossy  $f_s$, $r = {ROUGH_GLOSSY:.2f}$",
             color=C_GLOSS, weight=W_GLOSSY,
             f=cook_torrance(wi, wo, n, ROUGH_GLOSSY, F0_GLOSSY)),
        dict(key="mirror", label=rf"mirror  $f_s$, $r = {ROUGH_MIRROR:.2f}$",
             color=C_MIRR, weight=W_MIRROR,
             f=cook_torrance(wi, wo, n, ROUGH_MIRROR, F0_MIRROR)),
    ]
    for c in comps:
        c["lum"] = c["weight"] * luminance(c["f"])

    total = sum(c["weight"] * c["f"] for c in comps)
    return dict(n=n, wi=wi, r=r, theta=theta, wo=wo, comps=comps, total=total)


def brdf_total(wi: np.ndarray, wo: np.ndarray, n: np.ndarray) -> np.ndarray:
    """The weighted sum, for any pair of directions.  Used by the checks and by the
    shaded sphere, so that both are looking at the same material as the polar plot."""
    return (W_DIFFUSE * lambert(BASE_COLOR, wo.shape[:-1])
            + W_GLOSSY * cook_torrance(wi, wo, n, ROUGH_GLOSSY, F0_GLOSSY)
            + W_MIRROR * cook_torrance(wi, wo, n, ROUGH_MIRROR, F0_MIRROR))


# ────────────────────────────────────────────────────────── the polar panel
def compress(f: np.ndarray | float) -> np.ndarray:
    """BRDF value -> drawn radius.  See RADIAL_POWER."""
    return np.asarray(f) ** RADIAL_POWER


def polar_xy(theta: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Angle from the normal and radius -> plot coordinates, normal along +y."""
    return r * np.sin(theta), r * np.cos(theta)


def panel_lobes(ax, case: dict) -> float:
    """The three lobes stacked, drawn in Cartesian coordinates.  Returns the radius.

    Cartesian and not a `projection="polar"` axis: a polar axis restricted to
    [-90, 90] still reserves a SQUARE box and puts the half disc in its upper half,
    leaving an empty band underneath as tall as the drawing.  That band is margin
    internal to the axis, so no tight-bbox setting removes it, and every attempt to hide
    it by oversizing the axis moves the legend somewhere else instead.  Converting by
    hand costs the grid arcs below and buys limits that are exactly the content.
    """
    theta = case["theta"]
    # Stacking, not overlaying: each band is drawn between the running total before it
    # and after it, so the outer boundary of the last one is f_r.  Overlaying would hide
    # the small components behind the large ones and lose the sum.
    #
    # The compression is applied to the CUMULATIVE sums, not to each component on its
    # own: that is what keeps the bands nested and the outer boundary equal to the
    # compressed total.  Compressing each component and then stacking would draw a
    # boundary that is not a function of f_r at all.
    cum = np.zeros_like(theta)
    rmax = float(compress(sum(c["lum"] for c in case["comps"]).max()) * 1.06)

    # ── grid: arcs at round BRDF values, rays every 30 degrees ──
    arc = np.linspace(-np.pi / 2, np.pi / 2, 400)
    ticks = [t for t in (0.1, 1.0, 10.0, 100.0) if compress(t) < rmax * 0.97]
    for t in ticks:
        ax.plot(*polar_xy(arc, compress(t)), color="0.88", lw=0.6, zorder=1)
    for a in np.radians([-60, -30, 0, 30, 60]):
        ax.plot(*polar_xy(np.array([a, a]), np.array([0.0, rmax])),
                color="0.88", lw=0.6, zorder=1)
    ax.plot(*polar_xy(arc, rmax), color="0.45", lw=1.0, zorder=1)
    ax.plot([-rmax, rmax], [0, 0], color="0.45", lw=1.0, zorder=1)

    # The rings are unevenly spaced on purpose, and that uneven spacing is what tells the
    # reader the axis is not linear; the labels give it a unit.  They go in the
    # upper-left wedge, the only part of the disc that stays empty: both specular lobes
    # sit right of the normal and the diffuse disc holds the centre.
    for t in ticks:
        x, y = polar_xy(np.array(np.radians(-73.0)), compress(t))
        ax.text(float(x), float(y), f"{t:g}", fontsize=10, color="0.45",
                ha="center", va="center", zorder=6,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8))
    for a, lab in ((-90, r"$90^\circ$"), (-30, r"$30^\circ$"), (30, r"$30^\circ$"),
                   (90, r"$90^\circ$")):
        x, y = polar_xy(np.array(np.radians(a)), rmax * 1.05)
        ax.text(float(x), float(y), lab, fontsize=10.5, color="0.4",
                ha="center", va="center", zorder=6)

    # ── the three bands ──
    lower = compress(cum)
    for c in case["comps"]:
        cum = cum + c["lum"]
        upper = compress(cum)
        xo, yo = polar_xy(theta, upper)
        xi, yi = polar_xy(theta, lower)
        ax.fill(np.concatenate([xo, xi[::-1]]), np.concatenate([yo, yi[::-1]]),
                color=c["color"], alpha=0.55, linewidth=0, zorder=2)
        ax.plot(xo, yo, color=c["color"], lw=1.7, zorder=3)
        lower = upper

    # The normal and the two special directions, drawn to the rim so that the lobes can
    # be read against them.  The mirror direction is what both specular lobes centre on.
    ax.plot([0, 0], [0, rmax], color=C_INK, lw=1.6, zorder=4)
    ax.text(0.0, rmax * 1.05, r"$\mathbf{n}$", ha="center", va="bottom",
            fontsize=15, color=C_INK, zorder=6)

    ti = np.radians(THETA_I)
    tip = polar_xy(np.array(-ti), np.array(0.0))
    tail = polar_xy(np.array(-ti), np.array(rmax * 0.99))
    ax.annotate("", xy=(float(tip[0]), float(tip[1])),
                xytext=(float(tail[0]), float(tail[1])),
                arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.5",
                                color=C_INCID, lw=2.0), zorder=5)
    lx, ly = polar_xy(np.array(-ti), np.array(rmax * 1.06))
    ax.text(float(lx), float(ly), r"$\omega_i$", ha="right", va="bottom",
            fontsize=15, color=C_INCID, zorder=6)
    ax.plot(*polar_xy(np.array([ti, ti]), np.array([0.0, rmax])),
            color=C_INK, lw=1.2, ls=(0, (5, 3)), zorder=4)
    mx, my = polar_xy(np.array(ti), np.array(rmax * 1.06))
    ax.text(float(mx), float(my), "mirror\ndirection", ha="left", va="bottom",
            fontsize=11, color="0.35", linespacing=1.2, zorder=6)

    ax.set_aspect("equal")
    ax.set_xlim(-rmax * 1.34, rmax * 1.34)
    ax.set_ylim(-rmax * 0.06, rmax * 1.30)
    ax.axis("off")
    return rmax


# ────────────────────────────────────────────────── the shaded sphere panel
def shaded_sphere(case: dict) -> np.ndarray:
    """(P, P, 4) RGBA: a sphere lit by one directional light through the SAME BRDF.

    It is what turns the diagram into an adjective.  A single light and no environment:
    the point is the shape of the response, and an environment would add its own colour
    to it.  The view direction is fixed at the camera, the surface normal varies over the
    sphere, so every pixel is a different (wi, wo) pair of the same material.
    """
    p = SPHERE_PX
    y, x = np.mgrid[0:p, 0:p]
    u = (x - (p - 1) / 2) / ((p - 1) / 2)
    v = -(y - (p - 1) / 2) / ((p - 1) / 2)
    rr = u ** 2 + v ** 2
    inside = rr <= 1.0
    w = np.sqrt(np.maximum(0.0, 1.0 - rr))
    nrm = np.stack([u, v, w], axis=-1)                     # sphere normal, +z at camera

    wo = np.broadcast_to(np.array([0.0, 0.0, 1.0]), nrm.shape)
    wi = np.broadcast_to(LIGHT_DIR / np.linalg.norm(LIGHT_DIR), nrm.shape)

    f = brdf_total(wi, wo, nrm)
    ndl = np.clip(np.sum(nrm * wi, axis=-1), 0.0, 1.0)
    rgb = f * ndl[..., None] * np.pi                        # unit irradiance, times pi

    rgba = np.zeros((p, p, 4), dtype=np.float32)
    rgba[..., :3] = np.clip(rgb, 0.0, 1.0) ** (1.0 / 2.2)
    rgba[..., 3] = inside.astype(np.float32)
    return rgba


def trim_white(path: Path, pad: int = 12) -> None:
    """Trim the uniform white border.  `bbox_inches="tight"` trims on the axes box, and a
    polar axis restricted to a half disc leaves most of its box empty."""
    from PIL import Image, ImageChops

    im = Image.open(path).convert("RGB")
    box = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).getbbox()
    if box is None:
        return
    left, top, right, bottom = box
    im.crop((max(left - pad, 0), max(top - pad, 0),
             min(right + pad, im.width), min(bottom + pad, im.height))).save(path)


def figure(case: dict, out: Path) -> None:
    fig = plt.figure(figsize=(9.4, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.6, 1.0], wspace=0.0,
                          left=0.01, right=0.99, top=0.98, bottom=0.19)

    panel_lobes(fig.add_subplot(gs[0]), case)

    # Sphere deliberately smaller than the lobe panel: it is the gloss on the argument,
    # not the argument.  At equal sizes it took the eye first.
    axs = fig.add_subplot(gs[1])
    axs.imshow(shaded_sphere(case), interpolation="bilinear")
    axs.axis("off")
    axs.set_title("the same BRDF,\non a sphere", fontsize=11.5, color="0.4", pad=8,
                  linespacing=1.25)

    # The legend carries the whole reading of the figure, so it goes under the lobe
    # panel and not inside it, where at these lobe sizes it would sit on the mirror one.
    handles = [Patch(facecolor=c["color"], edgecolor=c["color"], alpha=0.7,
                     label=c["label"]) for c in case["comps"]]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.36, 0.005),
               ncol=3, frameon=False, fontsize=12, handlelength=1.4,
               columnspacing=1.6)
    fig.text(0.36, 0.105, r"stacked, so the outer boundary is $f_r = f_d + f_s$",
             ha="center", fontsize=11.5, color="0.4")

    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    trim_white(out)
    print(f"  + {out}")


def report(case: dict) -> None:
    """Checks that the drawing is telling the truth about the physics."""
    n, wi = case["n"], case["wi"]
    peaks = {c["key"]: float(c["lum"].max()) for c in case["comps"]}
    widths = {}
    for c in case["comps"][1:]:
        # Full width at half maximum of the lobe, in degrees: the number that says the
        # mirror one is sharper, which the eye reads off the picture.
        half = c["lum"] >= 0.5 * c["lum"].max()
        widths[c["key"]] = float(np.degrees(np.ptp(case["theta"][half])))

    print(f"\n  incident at {THETA_I:.0f} deg, base colour "
          + ", ".join(f"{v:.2f}" for v in BASE_COLOR))
    for c in case["comps"]:
        w = widths.get(c["key"])
        print(f"  {c['key']:8s} weight {c['weight']:.3f}   peak {peaks[c['key']]:10.4f}"
              + (f"   FWHM {w:5.2f} deg" if w is not None
                 else "   (constant over the hemisphere)"))
    # What the reader actually sees is not a peak but a BAND: the gap between the drawn
    # boundary before a component is added and after it, at its thickest.  A component
    # can have a healthy peak and still be a sliver on the page, which is exactly what
    # happened to the glossy one, so this and not the peak is what the assert measures.
    rmax = float(compress(sum(c["lum"] for c in case["comps"]).max()))
    cum = np.zeros_like(case["theta"])
    bands = {}
    for c in case["comps"]:
        lower, cum = compress(cum), cum + c["lum"]
        bands[c["key"]] = float(np.max(compress(cum) - lower)) / rmax
    print(f"  drawn (r = f ** {RADIAL_POWER}, plot radius {rmax:.2f}): "
          + ", ".join(f"{k} band {100.0 * v:.0f}%" for k, v in bands.items()))

    # Directional albedo: integral of f_r cos(theta_o) over the hemisphere, by a
    # Fibonacci set uniform in solid angle.  Under 1 is energy conservation, the second
    # of the two properties background.tex:59-65 requires.
    s = 200_000
    i = np.arange(s)
    cz = 1.0 - (i + 0.5) / s
    sz = np.sqrt(np.maximum(0.0, 1.0 - cz ** 2))
    az = i * np.pi * (3.0 - np.sqrt(5.0))
    wo = np.stack([sz * np.cos(az), sz * np.sin(az), cz], axis=-1)
    rho_dir = (brdf_total(wi, wo, n) * cz[:, None]).sum(axis=0) * (2.0 * np.pi / s)
    print(f"  directional albedo = " + ", ".join(f"{v:.4f}" for v in rho_dir)
          + f"   (max {rho_dir.max():.4f}, must stay under 1)")

    # Reciprocity: f_r(wi, wo) == f_r(wo, wi).  It holds by construction for both terms,
    # so a failure here means the Cook-Torrance assembly has a typo in it.
    probe = wo[:2000]
    a = brdf_total(np.broadcast_to(wi, probe.shape), probe, n)
    b = np.stack([brdf_total(probe[k], wi, n) for k in range(probe.shape[0])])
    err = float(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-12))
    print(f"  reciprocity: relative gap {err:.2e}")

    total_w = W_DIFFUSE + W_GLOSSY + W_MIRROR
    assert abs(total_w - 1.0) < 1e-9, \
        f"the weights are not a mixture: they sum to {total_w:.3f} and not to 1, so " \
        "they are no longer the share of the reflection each lobe carries"
    assert rho_dir.max() < 1.0, \
        f"the weighted material reflects more than it receives ({rho_dir.max():.3f}): " \
        "lower a weight"
    assert err < 1e-6, "f_r is not reciprocal: the Cook-Torrance assembly is wrong"
    # "Broad against sharp" is a claim the figure makes, so it gets a margin rather than a
    # bare inequality: at roughness 0.28 the glossy lobe was only 3.8 times the mirror's
    # width and the two read as one lobe drawn twice.
    assert widths["glossy"] > 4.0 * widths["mirror"], \
        f"the glossy lobe ({widths['glossy']:.1f} deg) is not clearly broader than the " \
        f"mirror one ({widths['mirror']:.1f} deg): raise ROUGH_GLOSSY"
    assert peaks["mirror"] > peaks["glossy"], \
        "the mirror lobe does not stand above the glossy one: raise W_MIRROR"
    # Legibility, measured on the DRAWN band and not on the BRDF value: after the
    # compression a band under a tenth of the plot radius is a line on the page, and the
    # figure would be claiming a decomposition into three the reader can only see as two.
    # This is the assert that fires if RADIAL_POWER, a roughness or a weight is retuned
    # too far.
    for k, v in bands.items():
        assert v > 0.10, \
            f"the {k} band is {100 * v:.1f}% of the plot radius at its thickest and " \
            f"would not be visible: raise its weight, or RADIAL_POWER"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    case = build_case()
    figure(case, out / "brdf_combined.png")
    report(case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
