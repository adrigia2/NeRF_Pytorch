#!/usr/bin/env python
"""make_pbr_model_diagram.py -- figures for the PBR model equation.

    python make_pbr_model_diagram.py --out ../Doc/images/pbr-model
    python make_pbr_model_diagram.py --out ../figure_review/pbr-model --figures kernel

Writes two PNGs:

  pbr_model.png       one texel, its normal, the hemisphere the diffuse irradiance
                      arrives from, and two cameras with their own reflected ray
                      and the cone of aperture Theta around it.

  tophat_vs_ggx.png   the SAME rays seen along the reflected direction, weighted in
                      the two ways: flat inside the cone (top-hat) and with the GGX
                      NDF.  It is not a figure about the shape of the kernel but
                      about its MEASURABILITY: with the top-hat the weights are 1
                      and known before tracing, so the estimate is the mean of the
                      rays; with the GGX every weight depends on alpha, i.e. on the
                      width the fit has yet to recover.

The figure is deliberately silent on the numbers: the split of the colour into the two
terms is done by the thesis text, here only the geometry that makes the two terms
different is drawn.  The numbers do stay in the script, printed at the end of a run,
because they are the check that the drawing is telling the truth: the sun falls inside
camera 1's cone and outside camera 2's, so L_1 and L_2 must come out different, and
that is exactly what the fit works on.

The environment is an analytic sky (a gradient plus a warm sun), the irradiance is its
cosine-weighted integral over the hemisphere and L_j is the pure solid-angle mean over
the cone, closed by the SAME functions the bake uses (`ring_weights_mean` and
`_cones_from_rings_np` of images_generator), so the printed numbers cannot diverge from
the pipeline's mathematics.  An independent check by direct sampling of the cone is
printed next to it.

The dome's dots are the incident radiance after a shared exposure and gamma, i.e. as it
would look on screen; the exposure is chosen on the cone means and on the irradiance, so
that the sun saturates to white and the rest of the sky stays readable.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d

import _paths  # noqa: F401

from images_generator import (_cones_from_rings_np, ring_weights_mean,
                              spec_cone_shared_ring_samples)

# ── Parameters of the didactic case ──────────────────────────────────────────
# One aperture only, not the grid of candidates: that is the subject of
# fig:cone-geometry.  Here Theta has already been chosen by the fit and is the same for
# both cameras, because in the model it is a property of the texel and not of the view.
THETA = 40.0                      # TOTAL cone aperture, in degrees
X_DIFFUSE = 0.6                   # weight of the diffuse term; metallic = 1 - x
ALBEDO = np.array([0.62, 0.45, 0.31])     # a, diffuse reflectance in [0,1]

# The cameras sit low over the horizon for two reasons: their cones open out like a fan
# instead of overlapping around the normal, and the cap at the top stays free for the
# normal and the hemisphere labels.  The constraint to respect is
# theta + Theta/2 <= 90 degrees, otherwise the cone cuts the horizon and the mean is no
# longer that of a whole cone (`report` checks it).
# The azimuths are separated by more than Theta even between one camera and the other's
# cone (R_j sits at the azimuth opposite v_j): at 170 and 20 degrees the camera 1 glyph
# ended up inside the border of camera 2's cone.
CAMS = (                          # (theta from the normal, azimuth) in degrees
    # Red and teal rather than red and blue: the dome dots are pale blue (it is the
    # colour of the sky), and a blue camera was confused with them exactly inside its
    # own cone, where the two populations overlap.
    dict(theta=55.0, phi=160.0, name="camera 1", color="#c0392b"),
    dict(theta=45.0, phi=30.0,  name="camera 2", color="#12776b"),
)

S_INT = 200_000                   # directions for the integrals (E and L_j)
S_DOME = 520                      # directions drawn as the dome
DOME_S = 22                       # area of the dome dot at theta = 0
CONE_DOTS = 34                    # directions drawn inside each cone
CONE_S = 24                       # area, the same for all: inside the cone there is no weight

# Analytic sky: horizon-to-zenith gradient plus a Gaussian sun.  The sun is placed near
# camera 1's reflected ray and far from camera 2's: that is what makes L_1 and L_2
# different, which is the point of the figure.
# The sky is kept low in saturation so that the diffuse term, which is the albedo
# multiplied by this light, stays recognisable as the material's colour.
SKY_HORIZON = np.array([0.78, 0.80, 0.84])
SKY_ZENITH = np.array([0.24, 0.36, 0.60])
SUN_RGB = np.array([1.00, 0.80, 0.52]) * 2.2
SUN_SIGMA = 16.0                  # degrees
SUN_OFFSET = 8.0                  # offset of the sun from R_1, in degrees

# ── Parameters of the kernel figure (top-hat against GGX lobe) ───────────────
ALPHA_GGX = 0.30                  # GGX roughness of the drawn lobe (alpha = r^2)
S_KERNEL = 520                    # directions traced, THE SAME in both panels
R_TILT = 40.0                     # tilt of R from the normal, in degrees
KERNEL_CONTAIN = 0.90             # fraction of the lobe's weight the cone must hold
KERNEL_S = 78.0                   # area of the max-weight dot, shared scale

C_INK = "#222222"
C_DIFF = "#e8a33d"                # amber: everything belonging to the diffuse term
C_KERNEL = "#2171b5"              # blue of the cone ramp, the same in both panels
C_OUT = "#b8b8b8"                 # rays the top-hat does not even trace

# White halo for the 3D labels.  Needed because in 3D axes zorder does not decide the
# draw order: matplotlib reorders the artists by depth, and a label behind a cone's fill
# ends up under it regardless.
HALO = dict(fc="white", ec="none", alpha=0.75, pad=1.0)

# On the page the figure is shrunk: the sizes are chosen so that the text stays readable
# AFTER the reduction, not on screen.
plt.rcParams.update({"font.size": 13})


def onb(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal basis around `a`."""
    t = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(t, a)
    u /= np.linalg.norm(u)
    return u, np.cross(a, u)


def fibonacci_hemisphere(s: int) -> np.ndarray:
    """(S, 3) directions above z, uniform in solid angle.

    Same construction as the shared kernel: equispaced cos(theta) and azimuth on the
    golden angle.
    """
    i = np.arange(s)
    cos_t = 1.0 - (i + 0.5) / s
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t ** 2))
    phi = i * np.pi * (3.0 - np.sqrt(5.0))
    return np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=-1)


def cone_directions(axis: np.ndarray, half_deg: float, n: int) -> np.ndarray:
    """(n, 3) directions uniform in solid angle INSIDE the cone around `axis`.

    Uniform in solid angle means equispaced cos(angle from the axis), not the angle:
    it is the same condition that makes the mean over the cone a plain mean of the
    samples, with no weights.
    """
    t, b = onb(axis)
    i = np.arange(n)
    cz = 1.0 - (i + 0.5) / n * (1.0 - np.cos(np.radians(half_deg)))
    sz = np.sqrt(np.maximum(0.0, 1.0 - cz ** 2))
    az = i * np.pi * (3.0 - np.sqrt(5.0))
    return (cz[:, None] * axis
            + sz[:, None] * (np.cos(az)[:, None] * t + np.sin(az)[:, None] * b))


def circle_on_sphere(axis: np.ndarray, half_deg: float, n: int = 240) -> np.ndarray:
    """Directions at `half_deg` degrees from `axis`: the edge of a cone."""
    t, b = onb(axis)
    a = np.radians(half_deg)
    ang = np.linspace(0, 2 * np.pi, n)
    return (np.cos(a) * axis[None, :]
            + np.sin(a) * (np.cos(ang)[:, None] * t + np.sin(ang)[:, None] * b))


def label3d(ax, p, text: str, **kw) -> None:
    """Label anchored to the 3D point `p` but drawn as 2D text.

    In 3D axes zorder does not decide the draw order: matplotlib reorders the 3D artists
    by depth, so a label ends up under a cone's fill every time a piece of the cone is
    closer than the anchor, even when the anchor sits outside the solid.  Projecting by
    hand and drawing a 2D Text restores a deterministic order.  It has to be called after
    limits, box_aspect and view_init are fixed, since those are what `get_proj()` reads.
    """
    x, y, _ = proj3d.proj_transform(*p, ax.get_proj())
    # A high zorder and not 20: 3D artists get a zorder computed from the depth, which
    # with many collections exceeds small values.
    ax.text2D(x, y, text, transform=ax.transData, zorder=200, **kw)


def sph(theta_deg: float, phi_deg: float) -> np.ndarray:
    t, p = np.radians(theta_deg), np.radians(phi_deg)
    return np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])


def sky(d: np.ndarray, sun_dir: np.ndarray) -> np.ndarray:
    """(..., 3) -> (..., 3): radiance of the environment along `d`."""
    t = np.clip(d[..., 2], 0.0, 1.0)[..., None] ** 0.6
    base = SKY_HORIZON * (1.0 - t) + SKY_ZENITH * t
    ang = np.degrees(np.arccos(np.clip(d @ sun_dir, -1.0, 1.0)))
    glow = np.exp(-(ang / SUN_SIGMA) ** 2)[..., None]
    return base + SUN_RGB * glow


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.2126, 0.7152, 0.0722])


def build_case() -> dict:
    """Geometry, integrals and the three terms of the equation."""
    n = np.array([0.0, 0.0, 1.0])
    dirs = fibonacci_hemisphere(S_INT)
    half = THETA / 2.0

    cams = []
    for spec in CAMS:
        v = sph(spec["theta"], spec["phi"])
        r = 2.0 * np.dot(n, v) * n - v          # Reflected ray equation
        cams.append(dict(spec, v=v, r=r))

    # The sun sits near R_1, offset by SUN_OFFSET towards the normal.
    r1 = cams[0]["r"]
    u = n - np.dot(n, r1) * r1
    u /= np.linalg.norm(u)
    a = np.radians(SUN_OFFSET)
    sun_dir = np.cos(a) * r1 + np.sin(a) * u

    rad = sky(dirs, sun_dir)                    # radiance over the whole hemisphere

    # E = cosine-weighted integral: every direction of the set is worth 2*pi/S sr.
    e_irr = (rad * dirs[:, 2:3]).sum(axis=0) * (2.0 * np.pi / S_INT)

    # L_j: pure solid-angle mean over the cone, closed by the bake's own code.
    # Aperture grid with a single ring, [0, Theta]: level 1 is the cone.
    apertures = [0.0, THETA]
    cos_b = np.cos(np.radians(np.asarray(apertures)) * 0.5)
    w = ring_weights_mean(cos_b, 1,
                          np.asarray(spec_cone_shared_ring_samples(apertures, S_INT)))
    for c in cams:
        ang = np.degrees(np.arccos(np.clip(dirs @ c["r"], -1.0, 1.0)))
        m = ang <= half
        ring_sum = np.zeros((1, 2, 3))
        ring_valid = np.zeros((1, 2))
        ring_sum[0, 1] = rad[m].sum(axis=0)
        ring_valid[0, 1] = m.sum()
        ring_sum[0, 0] = sky(c["r"], sun_dir)   # mirror level: a single ray
        ring_valid[0, 0] = 1.0
        c["L"] = _cones_from_rings_np(ring_sum, ring_valid, w)[0, 1]
        c["n_rays"] = int(m.sum())
        # Does the cone stay entirely above the horizon? If not the mean is truncated and
        # the comparison with the direct sampling would make no sense.
        c["unclipped"] = np.degrees(np.arccos(c["r"][2])) + half <= 90.0
        # Independent check: mean uniform in solid angle inside the cone, sampled
        # directly instead of by filtering the hemisphere.  With the samples uniform in
        # solid angle, the mean is the plain one.
        c["L_check"] = sky(cone_directions(c["r"], half, 20_000), sun_dir).mean(axis=0)

    diffuse = ALBEDO * X_DIFFUSE / np.pi * e_irr
    for c in cams:
        c["spec"] = (1.0 - X_DIFFUSE) * c["L"]
        c["C"] = diffuse + c["spec"]

    return dict(n=n, cams=cams, sun_dir=sun_dir, E=e_irr, diffuse=diffuse, half=half)


def make_tonemap(case: dict):
    """Exposure shared by every radiance swatch, plus gamma."""
    peak = max(float(np.max(case["E"] / np.pi)),
               max(float(np.max(c["C"])) for c in case["cams"]),
               max(float(np.max(c["L"])) for c in case["cams"]))
    scale = 0.95 / peak

    def tm(rgb):
        return np.clip(np.asarray(rgb) * scale, 0.0, 1.0) ** (1.0 / 2.2)

    return tm


# ────────────────────────────────────────────────────────── panel (a), in 3D
def panel_geometry(ax, case: dict, tm) -> None:
    n = case["n"]

    # The framing is fixed BEFORE drawing, because `label3d` projects by hand and reads
    # the projection matrix: changing it afterwards would leave the labels where they
    # were.  The box ratio MUST follow the extent of the data, otherwise the vertical
    # scale differs from the horizontal one and the angles can no longer be read: the
    # cone circles look rotated and n looks longer than R even though both are unit
    # vectors.  In a diagram about angles that is a bug, not a framing choice.
    #
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(0, 1.1)
    ax.set_box_aspect((2.1, 2.1, 1.1))
    ax.set_axis_off()
    ax.view_init(elev=26, azim=-72)

    # The surface square sits inside the equator: any larger and it would cover the
    # grazing directions and make them look as if they were below the surface.
    g = np.linspace(-0.62, 0.62, 2)
    gx, gy = np.meshgrid(g, g)
    ax.plot_surface(gx, gy, np.zeros_like(gx), color="0.82", alpha=0.6,
                    edgecolor="0.5", linewidth=0.7, zorder=0)
    a = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(a), np.sin(a), np.zeros_like(a), color="0.6", lw=0.9, zorder=1)

    # The dome: the hemisphere directions coloured with the environment the radiance
    # comes from.  It is the same construction E is integrated with, and it makes visible
    # why the two cones collect different colours.
    #
    # The SIZE of the dot is the weight with which that direction enters the irradiance,
    # i.e. cos(theta): area exactly proportional, with no minimum.  The dots that fade
    # out against the horizon are not a rendering defect, they are grazing directions
    # that carry almost nothing; the hemisphere's silhouette is held by the equator
    # circle anyway.
    dome = fibonacci_hemisphere(S_DOME)
    ax.scatter(dome[:, 0] * 1.02, dome[:, 1] * 1.02, dome[:, 2] * 1.02,
               s=DOME_S * dome[:, 2], c=tm(sky(dome, case["sun_dir"])),
               depthshade=False, alpha=0.9, linewidths=0, zorder=2)
    # The sun drawn separately: among the dome dots it would be a detail of a few
    # samples, and instead it is the reason L_1 and L_2 do not coincide.  Larger radius
    # than the cone samples that fall on top of it: it is the 3D depth ordering that
    # decides who covers whom.
    ax.scatter(*(case["sun_dir"] * 1.07)[:, None], s=300, marker="o",
               color=tm(sky(case["sun_dir"], case["sun_dir"])),
               edgecolor=C_DIFF, linewidth=1.6, depthshade=False, zorder=3)

    # Incoming arrows scattered over the dome: the irradiance collects the whole
    # hemisphere.  Those that would fall inside a cone are skipped, it is already full there.
    for d in fibonacci_hemisphere(13):
        if d[2] < 0.25 or any(d @ c["r"] > np.cos(np.radians(case["half"] + 14))
                              for c in case["cams"]):
            continue
        ax.quiver(*(d * 0.90), *(-d * 0.26), color=C_DIFF, lw=1.4,
                  arrow_length_ratio=0.40, alpha=0.95, zorder=3)
    # The two captions live in axes coordinates and not in the scene: inside the dome
    # every free 3D position lands on n or on one of the R's.
    ax.text2D(0.12, 0.755, r"$E$:  the whole hemisphere", color=C_DIFF,
              fontsize=12.5, transform=ax.transAxes)
    ax.text2D(0.20, 0.155, "incident radiance", color="0.45", fontsize=11.5,
              transform=ax.transAxes)

    ax.quiver(0, 0, 0, *n, color=C_INK, lw=2.0, arrow_length_ratio=0.12, zorder=6)
    label3d(ax, n * 1.04 + np.array([0.02, 0.10, 0.02]), r"$\mathbf{n}$",
            color=C_INK, fontsize=15, bbox=HALO)
    ax.scatter([0], [0], [0], color=C_INK, s=22, zorder=7)
    label3d(ax, (0.20, -0.12, 0.02), "texel", color=C_INK, fontsize=12, bbox=HALO)

    for k, c in enumerate(case["cams"], start=1):
        v, r, col = c["v"], c["r"], c["color"]
        # The view is dashed and thin, the reflected ray solid and thick: it is the
        # second that carries the cone, and the pair reads as a reflection.
        ax.plot([0, v[0]], [0, v[1]], [0, v[2]], color=col, lw=1.5, ls=(0, (4, 2)),
                zorder=6)
        ax.quiver(0, 0, 0, *r, color=col, lw=2.2, arrow_length_ratio=0.12, zorder=6)
        ax.scatter([v[0] * 1.13], [v[1] * 1.13], [v[2] * 1.13], marker="s", s=85,
                   color=col, depthshade=False, zorder=8)
        label3d(ax, v * 1.13 + np.array([0.0, 0.0, 0.24]), c["name"], color=col,
                fontsize=12, ha="center", va="bottom", bbox=HALO)
        label3d(ax, v * 0.62 + np.array([0.0, 0.0, -0.13]), rf"$\mathbf{{v}}_{k}$",
                color=col, fontsize=13, ha="center", bbox=HALO)
        # R's label beyond the cone's edge, on the axis: any closer it would end up under
        # the edge circle, and to the side it would land on the zenith.
        label3d(ax, r * 1.30, rf"$\mathbf{{R}}_{k}$", color=col, fontsize=15,
                ha="center", va="center", bbox=HALO)

        # The cone samples, all of the SAME size, above the dome ones which instead
        # shrink: it is the difference between the two terms of the model shown at a
        # single glance.  L_j is a pure solid-angle mean, with no cos(theta) weight, so
        # inside the cone every direction counts the same.
        #
        #
        # Solid tint in the camera's colour instead of the sampled radiance: coloured
        # like the environment they were confused with the dome dots exactly where the
        # two populations overlap, and in the cone facing the sun they saturated to
        # white.  What radiance they see is still told by the dome dots underneath,
        # which remain.
        cd = cone_directions(r, case["half"], CONE_DOTS)
        ax.scatter(cd[:, 0] * 1.03, cd[:, 1] * 1.03, cd[:, 2] * 1.03, s=CONE_S,
                   color=col, edgecolor="white", linewidth=0.5,
                   depthshade=False, zorder=5)

        # Lateral surface of the cone, from the texel to the edge on the unit sphere.
        # Low alpha: the sun and the dome dots have to stay visible through the cone,
        # otherwise the reason for the two colours disappears.
        rim = circle_on_sphere(r, case["half"], 90)
        t = np.linspace(0.0, 1.0, 2)[:, None, None]
        cone = t * rim[None, :, :]
        ax.plot_surface(cone[..., 0], cone[..., 1], cone[..., 2], color=col,
                        alpha=0.15, linewidth=0, shade=False, zorder=4)
        ax.plot(rim[:, 0], rim[:, 1], rim[:, 2], color=col, lw=1.6, zorder=5)

    # The Theta/2 arc on one camera only: doubling it would only double the labels.
    # Opened on the side opposite the normal, where there is nothing else.
    c0 = case["cams"][0]
    u = n - np.dot(n, c0["r"]) * c0["r"]
    u = -u / np.linalg.norm(u)
    arc = np.array([np.cos(np.radians(g)) * c0["r"] + np.sin(np.radians(g)) * u
                    for g in np.linspace(0, case["half"], 60)])
    ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color=C_INK, lw=1.1, zorder=6)
    label3d(ax, arc[len(arc) // 2] * 1.16, r"$\Theta/2$", color=C_INK, fontsize=14,
            va="center", bbox=HALO)


# ───────────────────────── figure 2: top-hat kernel against GGX lobe
def ggx_d(cos_h: np.ndarray, alpha: float) -> np.ndarray:
    """GGX \\gls{ndf}, the formula from background.tex: D = a^2 / (pi (c^2(a^2-1)+1)^2)."""
    a2 = alpha * alpha
    return a2 / (np.pi * (cos_h ** 2 * (a2 - 1.0) + 1.0) ** 2)


def build_kernel_case() -> dict:
    """The two kernels on the SAME set of directions, both with integral 1.

    The prefiltering kernel depends on the angle from the reflected direction alone only
    under the assumption n = v = R, the one that makes it possible to prefilter the
    environment once instead of per view.  Under that assumption the half vector of a
    sample at gamma from R sits at gamma/2, and Karis' weight (D times the cosine, its
    N.L) becomes

        w(gamma) = D(cos(gamma/2); alpha) * cos(gamma)

    which vanishes at 90 degrees: under the assumption, beyond 90 degrees from R one is
    below the surface.  The cosine is not a cosmetic detail: without it the tail weighs
    enough to make the equivalent cone wider than the hemisphere.

    The top-hat's aperture is not chosen: it is the one that holds KERNEL_CONTAIN of the
    lobe's weight.  Putting the two panels side by side with a width decided by hand
    would be a rigged comparison.
    """
    n = np.array([0.0, 0.0, 1.0])
    r = sph(R_TILT, 0.0)
    dirs = fibonacci_hemisphere(S_KERNEL)
    gamma = np.degrees(np.arccos(np.clip(dirs @ r, -1.0, 1.0)))
    dw = 2.0 * np.pi / S_KERNEL          # solid angle per sample, uniform

    w_ggx = (ggx_d(np.cos(np.radians(gamma * 0.5)), ALPHA_GGX)
             * np.maximum(np.cos(np.radians(gamma)), 0.0))
    w_ggx /= w_ggx.sum() * dw            # integral 1 over the sampled domain

    order = np.argsort(gamma)
    cum = np.cumsum(w_ggx[order]) * dw
    half = float(np.interp(KERNEL_CONTAIN, cum, gamma[order]))
    omega = 2.0 * np.pi * (1.0 - np.cos(np.radians(half)))
    inside = gamma <= half
    w_top = np.where(inside, 1.0 / omega, 0.0)

    t, b = onb(r)
    return dict(n=n, r=r, dirs=dirs, gamma=gamma, dw=dw, azim=np.arctan2(dirs @ b,
                dirs @ t), w_ggx=w_ggx, w_top=w_top, inside=inside, half=half,
                omega=omega, t=t, b=b)


def panel_kernel(ax, case: dict, weights: np.ndarray, *, scale: float,
                 title: str, edge_solid: bool, note: str,
                 note_at: tuple[float, float] = (139.0, 76.0)) -> None:
    """View along R: radius = angle from R, dot area = the ray's weight."""
    ax.scatter(case["azim"], case["gamma"], s=weights * scale, color=C_KERNEL,
               edgecolor="white", linewidth=0.3, zorder=4)
    if edge_solid:
        # The zero-weight rays the top-hat does not trace at all: it is half of what
        # makes it measurable, and they must be shown as excluded, not as absent.
        out = ~case["inside"]
        ax.scatter(case["azim"][out], case["gamma"][out], s=7, color=C_OUT,
                   alpha=0.75, zorder=3)
    # The cone's edge stays blue in the lobe panel too, where it is only a reference:
    # dashed grey would be confused with the horizon.
    ang = np.linspace(0, 2 * np.pi, 200)
    ax.plot(ang, np.full(200, case["half"]), color=C_KERNEL,
            lw=1.8 if edge_solid else 1.2, ls="-" if edge_solid else (0, (5, 3)),
            zorder=5)
    ax.text(np.radians(note_at[0]), note_at[1], note, color="0.4", fontsize=11,
            ha="center", va="center", zorder=6, bbox=HALO)

    # The horizon seen along R, the same construction as the polar panel of
    # make_cone_diagram: a direction at gamma from R sinks below the surface where its
    # vertical component vanishes, i.e. tan(g) = -R_z / k_z.
    aa = np.linspace(0, 2 * np.pi, 361)
    kz = np.cos(aa) * case["t"][2] + np.sin(aa) * case["b"][2]
    ax.plot(aa, np.degrees(np.arctan2(-case["r"][2], kz)) % 180.0, color="0.45",
            lw=1.1, ls="--", zorder=2)

    ax.set_rmax(90)
    ax.set_rticks([30, 60, 90])
    ax.set_rlabel_position(112)
    ax.set_yticklabels(["30$^\\circ$", "60$^\\circ$", "90$^\\circ$"], fontsize=10.5,
                       color="0.35")
    ax.set_xticklabels([])
    ax.grid(color="0.85", lw=0.6)
    ax.set_title(title, fontsize=13, color=C_INK, pad=12)


def figure_kernel(case: dict, out: Path) -> None:
    scale = KERNEL_S / case["w_ggx"].max()      # SHARED scale between the panels:
    fig = plt.figure(figsize=(9.6, 5.9))        # it is the only way to compare them
    gs = fig.add_gridspec(1, 2, wspace=0.06, left=0.01, right=0.99,
                          top=0.90, bottom=0.30)

    tail = case["w_ggx"][~case["inside"]].sum() * case["dw"]
    panel_kernel(fig.add_subplot(gs[0], projection="polar"), case, case["w_top"],
                 scale=scale, edge_solid=True, note="outside:\nnever traced",
                 title=f"Top-hat cone, aperture $\\Theta = {2 * case['half']:.0f}"
                       f"^\\circ$")
    panel_kernel(fig.add_subplot(gs[1], projection="polar"), case, case["w_ggx"],
                 scale=scale, edge_solid=False, note_at=(148.0, 82.0),
                 note=f"no cut, only fade:\n{tail * 100:.0f}% is past the edge",
                 title=f"GGX lobe, roughness $\\alpha = {ALPHA_GGX:.2f}$")

    # The estimators under their respective panels.  The grey line is the point of the
    # figure: not the shape of the kernel, but what one has to know to evaluate it.
    for x, expr, note in (
            (0.25, r"$L(\Theta) = \dfrac{1}{N}\,\sum_s L_s$",
             "every weight is 1, and known\nbefore a single ray is traced"),
            (0.75, r"$L = \dfrac{\sum_s w_s L_s}{\sum_s w_s}$,"
                   r"$\quad w_s = D(\mathbf{h}_s;\,\alpha)\,\cos\gamma_s$",
             "every weight is set by $\\alpha$,\nthe width the fit has to recover")):
        fig.text(x, 0.21, expr, ha="center", va="center", fontsize=15, color=C_INK)
        fig.text(x, 0.06, note, ha="center", va="center", fontsize=11.5,
                 color="0.35", linespacing=1.25)

    fig.text(0.5, 0.965, "The same rays, weighted in two ways", ha="center",
             fontsize=14, color=C_INK)
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)
    trim_white(out)
    print(f"  + {out}")


def report_kernel(case: dict) -> None:
    """Checks on the two kernels, plus the numbers that end up in the caption."""
    i_top = float(case["w_top"].sum() * case["dw"])
    i_ggx = float(case["w_ggx"].sum() * case["dw"])
    tail = float(case["w_ggx"][~case["inside"]].sum() * case["dw"])
    print(f"\n  alpha = {ALPHA_GGX:.2f}   ->   Theta = {2 * case['half']:.1f} deg "
          f"(half-aperture {case['half']:.1f}, Omega {case['omega']:.3f} sr)")
    print(f"  kernel integrals: top-hat {i_top:.4f}, GGX {i_ggx:.4f}")
    print(f"  GGX weight past the cone edge: {tail * 100:.1f} %")
    print(f"  GGX peak / top-hat level: "
          f"{case['w_ggx'].max() / (1.0 / case['omega']):.2f}x")
    print("  Theta from the same rule for other roughnesses: "
          + ", ".join(f"alpha {a:.2f} -> {_theta_for_alpha(a):.0f} deg"
                      for a in (0.1, 0.2, 0.5)))
    assert abs(i_top - 1.0) < 0.05 and abs(i_ggx - 1.0) < 1e-6, \
        "the two kernels are not normalised the same way: the areas do not compare"
    assert tail > 0.02, "the tail past the cone is invisible: the figure says nothing"
    assert case["w_ggx"].max() * case["omega"] > 1.5, \
        "the lobe is not more peaked than the top-hat: revisit alpha or the width rule"


def _theta_for_alpha(alpha: float) -> float:
    """The aperture the same rule would give at another roughness."""
    g = np.degrees(np.arccos(np.clip(fibonacci_hemisphere(S_KERNEL)
                                     @ sph(R_TILT, 0.0), -1.0, 1.0)))
    w = (ggx_d(np.cos(np.radians(g * 0.5)), alpha)
         * np.maximum(np.cos(np.radians(g)), 0.0))
    o = np.argsort(g)
    cum = np.cumsum(w[o]) / w.sum()
    return 2.0 * float(np.interp(KERNEL_CONTAIN, cum, g[o]))


def trim_white(path: Path, pad: int = 14) -> None:
    """Trim the uniform white border of the PNG.

    `bbox_inches="tight"` trims on the axes BOX, not on the content, and a 3D axis only
    partly fills it: the projected box juts out with its corners beyond the dome, which
    is inscribed in it.  Two white bands are left above and below which on the page would
    be margin paid for by the inch, and which no matplotlib parameter removes without
    shrinking the drawing too.
    """
    from PIL import Image, ImageChops

    im = Image.open(path).convert("RGB")
    box = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))
                                ).getbbox()
    if box is None:                       # all-white image: nothing to do
        return
    left, top, right, bottom = box
    im.crop((max(left - pad, 0), max(top - pad, 0),
             min(right + pad, im.width), min(bottom + pad, im.height))).save(path)


def figure(case: dict, out: Path) -> None:
    tm = make_tonemap(case)
    fig = plt.figure(figsize=(7.0, 5.0))

    # Axes taller than the figure itself: the 3D box is wide and low (box_aspect
    # 2.1 x 2.1 x 1.1) and matplotlib fits it entirely inside the frame, leaving two
    # empty bands above and below the content.  They are margin internal to the axis, so
    # `bbox_inches="tight"` does not see them and they would stay inside the figure on
    # the page; oversizing the axis eats them.
    ax = fig.add_axes([0.0, -0.07, 1.0, 1.10], projection="3d")
    panel_geometry(ax, case, tm)

    # Title as figure text and not axis text: 3D axes leave an empty band above the
    # content and a title attached to the axis would turn it into margin inside the page.
    #
    # The title has to be kept low, close to the content: above the dome the 3D axis has
    # a tall empty band (the corners of the projected box) and a title at the top of the
    # figure would stay detached from the drawing even after the trim.
    fig.text(0.5, 0.84, "One hemisphere for the diffuse term, one cone per camera",
             ha="center", fontsize=14, color=C_INK)
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)
    trim_white(out)
    print(f"  + {out}")


def report(case: dict) -> None:
    """Checks that the figure is showing what it says it shows."""
    e = case["E"]
    print(f"\n  E            = ({e[0]:.4f}, {e[1]:.4f}, {e[2]:.4f})   "
          f"lum {luminance(e):.4f}")
    print(f"  (a x/pi) E   = " + ", ".join(f"{v:.4f}" for v in case["diffuse"])
          + f"   lum {luminance(case['diffuse']):.4f}")
    for k, c in enumerate(case["cams"], start=1):
        err = float(np.max(np.abs(c["L"] - c["L_check"])) / np.max(c["L_check"]))
        print(f"  L_{k}({THETA:.0f} deg) = " + ", ".join(f"{v:.4f}" for v in c["L"])
              + f"   lum {luminance(c['L']):.4f}   ({c['n_rays']} rays, "
                f"cone {'whole' if c['unclipped'] else 'TRUNCATED'}, "
                f"gap from direct sampling {err:.2e})")
        assert c["unclipped"], "the cone leaves the horizon: truncated mean"
        assert err < 1e-2, "the bake's closing does not match the sampled cone"
    c1, c2 = (c["C"] for c in case["cams"])
    print(f"  C_1          = " + ", ".join(f"{v:.4f}" for v in c1)
          + f"   lum {luminance(c1):.4f}")
    print(f"  C_2          = " + ", ".join(f"{v:.4f}" for v in c2)
          + f"   lum {luminance(c2):.4f}")
    ratio = luminance(c1) / luminance(c2)
    print(f"  C_1 / C_2 (lum) = {ratio:.2f}   "
          f"(the whole gap comes from the specular term)")
    assert ratio > 1.3, "the two cameras record almost the same colour: " \
                        "the figure loses its point, move the cameras or the sun"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--figures", default="model,kernel",
                    help="which figures to write, comma-separated: model, kernel")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    want = {f.strip() for f in args.figures.split(",") if f.strip()}
    if not want <= {"model", "kernel"}:
        ap.error(f"unknown figures: {sorted(want - {'model', 'kernel'})}")

    if "model" in want:
        case = build_case()
        figure(case, out / "pbr_model.png")
        report(case)
    if "kernel" in want:
        kcase = build_kernel_case()
        figure_kernel(kcase, out / "tophat_vs_ggx.png")
        report_kernel(kcase)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
