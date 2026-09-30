#!/usr/bin/env python
"""make_disney_brdf_figure.py -- how albedo, metallic and roughness make a colour.

    python make_disney_brdf_figure.py --out ../../PresentationImages
    python make_disney_brdf_figure.py --out /tmp --sphere 110 --samples 48   # draft

Writes `disney_brdf.png`, one picture in three bands:

  the model      the Disney parametrisation of sec:pbr-material-maps, written once, with
                 each of the three parameters coloured.  The same colour then names the
                 parameter underneath, so a symbol and its knob need no legend to link.

  the two terms  the same material with m = 0 and with m = 1, each split into
                 diffuse + specular = final.  The diffuse sphere of the conductor row is
                 BLACK, which is the whole content of the metallic parameter: it does not
                 dim the diffuse lobe, it removes it, and moves the base colour into F0.

  the grid       roughness across, metallic down, one albedo throughout.  Reading a row
                 gives the roughness knob, reading a column the metallic one, and the
                 corners give the four extremes: matte paint, gloss, rough metal, mirror.

WHAT IS ACTUALLY BEING COMPUTED.  Every sphere is a render, not a drawing.  The reflected
radiance is

    L_o(n) = (1 - m) * rho * E(n) / pi  +  int f_s(l, v) L(l) (n.l) dl

with f_s the Cook-Torrance BRDF of eq:cook-torrance -- GGX D, Schlick F, Smith G -- taken
from make_brdf_lobes_figure so that this figure and the thesis's BRDF figure cannot end up
using two different formulas.  The Disney parametrisation enters exactly where the thesis
says it does: alpha = r^2, F0 = 0.04 (1 - m) + m rho, and the diffuse lobe is weighted by
(1 - m).

The specular integral is evaluated by NDF importance sampling on a DETERMINISTIC golden
lattice: u1 = (i + 1/2) / N and u2 = i phi, shifted per pixel on both axes by two
irrational multiples of the pixel index.  No direction is ever drawn at random, here as in
the rest of the pipeline; the lattice is a quadrature rule, and the GGX inverse CDF places
its points inside the lobe, which is why a 0.14-degree lobe at r = 0.05 is resolved by a
couple of hundred samples.  The azimuth is accumulated in float64 and reduced mod 1 before
the trigonometry, for the reason recorded in the irradiance pass: in float32 the product
i * phi walks into the range where one ULP is already a fraction of a degree.

Schlick's F is linear in F0, F = F0 (1 - (1-c)^5) + (1-c)^5, so the estimator is
accumulated ONCE per roughness into the two channels that multiply F0 and 1.  Every
metallic value is then a linear combination of the same two integrals: the twenty-one
spheres of the figure cost six integrations, one per roughness, and the m = 0 and m = 1
spheres of a column are guaranteed to be the same integral seen through two F0.

The environment is analytic: a sky and ground gradient plus two caps of finite angular
radius, an 11-degree warm key and a 34-degree cool fill.  Analytic and not an HDR file so
the script stands on its own, and caps of finite size rather than point lights because a
point light has no solid angle: a mirror would reflect nothing and r = 0 would come out
black, which is the opposite of what the figure has to show.  The diffuse term integrates
the gradient on a Fibonacci set and the two caps as L * Omega * (n.d), the small-cap form
-- an approximation worth a few percent on the wide fill, and none of the figure's claims
rest on it.

DISPLAY.  One exposure for every sphere in the figure, then gamma 1/2.2, so the panels are
comparable across bands.  The sum is exact in radiance and only there: gamma is not linear,
so the two term panels do not add to the third in the encoded pixels.  `report()` checks
what the figure asserts -- that the split of F reproduces the shared `schlick_f`, that the
diffuse term of a conductor is identically zero, that no sphere reflects more than the
brightest direction of the environment, and that the peak specular radiance falls as the
roughness grows -- and refuses to draw when one of them fails.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.offsetbox import (AnchoredOffsetbox, HPacker,     # noqa: E402
                                  TextArea)

import _paths  # noqa: F401,E402

from make_brdf_lobes_figure import schlick_f, smith_g1            # noqa: E402
from make_geometry_diagrams import DPI, C_CAM, C_EDGE             # noqa: E402

# -- the material -----------------------------------------------------------
# The base colour of the thesis's own BRDF figure, so the two show one material.
ALBEDO = np.array([0.62, 0.45, 0.31])
F0_DIELECTRIC = 0.04          # background.tex:184, the ~4% of a dielectric

# The grid.  Five roughnesses spanning mirror to matte, three metallic values: the two
# ends, which are the physical ones, and the midpoint, which is what a texel holds when
# the bake blends across a rust boundary.
GRID_ROUGH = [0.05, 0.15, 0.30, 0.55, 0.85]
GRID_METAL = [0.0, 0.5, 1.0]

# The decomposition band: one roughness, the two ends of the metallic axis.
SPLIT_ROUGH = 0.12
SPLIT_METAL = [0.0, 1.0]

# -- the environment --------------------------------------------------------
# Camera space: +z towards the viewer, +y up.  A sky and ground gradient, plus two caps.
# The ambient is kept near neutral on purpose: a blue sky would tint every sphere, and
# the figure needs the tint of the m = 1 row to be the base colour and nothing else.
# The ground is a studio floor and not a night field.  A conductor reflects whatever is
# along its mirror direction, and over the lower half of a sphere that is the ground: at
# 0.09 the whole m = 1 row came out muddy brown, which says more about the floor than
# about the material.
SKY_ZENITH = np.array([0.30, 0.36, 0.48])
SKY_HORIZON = np.array([0.56, 0.58, 0.62])
GROUND = np.array([0.22, 0.205, 0.19])

# A softbox rather than a pinpoint sun.  The specular integral is sampled from the NDF,
# so a source much smaller than the lobe is hit by a handful of the lattice's points and
# its variance lands in the image as grain -- which is exactly what the rough conductors
# showed at 4.5 degrees.  Eleven degrees at a fourth of the radiance carries about the
# same power, still reads as a compact highlight at r = 0.05, and is what a material
# chart is lit with anyway.
KEY_DIR = np.array([-0.42, 0.72, 0.55])       # warm, above and to the left
KEY_DEG = 11.0
KEY_L = np.array([13.5, 12.2, 10.1])
FILL_DIR = np.array([0.80, 0.10, 0.58])       # wide, cool, to the right
FILL_DEG = 34.0
FILL_L = np.array([0.80, 0.92, 1.20])

N_DIFFUSE = 4096              # Fibonacci directions for the gradient's irradiance
GAMMA = 2.2
EXPOSURE_PCTL = 99.6          # the percentile of the brightest sphere the exposure sets

# -- appearance -------------------------------------------------------------
C_INK = C_CAM
C_MUTED = C_EDGE
C_ALB = "#bf6f1c"             # albedo
C_MET = "#3b7cbd"             # metallic
C_ROU = "#7a45bd"             # roughness

PHI = (np.sqrt(5.0) - 1.0) / 2.0      # the golden fraction, for the lattice


# ---------------------------------------------------------------- environment
def smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def cap(d: np.ndarray, direction: np.ndarray, deg: float, radiance: np.ndarray):
    """A disc of finite angular radius, with a softened edge.

    The edge is smoothed over the outer quarter of the cap and not left hard: at
    r = 0.05 the lobe is narrower than the cap, so a hard edge would land in the image
    as a jagged boundary carrying the lattice's own pattern.
    """
    c = d @ (direction / np.linalg.norm(direction))
    c_out, c_in = np.cos(np.radians(deg)), np.cos(np.radians(deg * 0.75))
    w = smoothstep((c - c_out) / (c_in - c_out))
    return w[..., None] * radiance


def env(d: np.ndarray) -> np.ndarray:
    """Radiance arriving from direction d (unit, (..., 3)).  Analytic, so no texture
    lookup and no resolution to declare."""
    y = d[..., 1]
    up = np.clip(y, 0.0, 1.0)[..., None] ** 0.65
    down = np.clip(-y, 0.0, 1.0)[..., None] ** 0.5
    base = np.where(y[..., None] >= 0.0,
                    SKY_HORIZON + (SKY_ZENITH - SKY_HORIZON) * up,
                    SKY_HORIZON * 0.42 + (GROUND - SKY_HORIZON * 0.42) * down)
    return base + cap(d, KEY_DIR, KEY_DEG, KEY_L) + cap(d, FILL_DIR, FILL_DEG, FILL_L)


def fibonacci_sphere(n: int) -> np.ndarray:
    """n directions of equal solid angle over the whole sphere, golden-angle azimuth.

    The azimuth runs in float64 and is reduced mod 1 before the trigonometry, the fix the
    irradiance pass carries: in float32 i * phi reaches the range where one ULP is worth a
    fraction of a degree.
    """
    i = np.arange(n, dtype=np.float64)
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = 2.0 * np.pi * np.mod(i * PHI, 1.0)
    return np.stack([r * np.cos(phi), z, r * np.sin(phi)], axis=-1)


def irradiance(nrm: np.ndarray) -> np.ndarray:
    """E(n) = int L (n.l) dl, the gradient by quadrature and the caps analytically."""
    dirs = fibonacci_sphere(N_DIFFUSE)
    y = dirs[..., 1]
    up = np.clip(y, 0.0, 1.0)[..., None] ** 0.65
    down = np.clip(-y, 0.0, 1.0)[..., None] ** 0.5
    grad = np.where(y[..., None] >= 0.0,
                    SKY_HORIZON + (SKY_ZENITH - SKY_HORIZON) * up,
                    SKY_HORIZON * 0.42 + (GROUND - SKY_HORIZON * 0.42) * down)
    dw = 4.0 * np.pi / N_DIFFUSE

    out = np.zeros(nrm.shape[:-1] + (3,))
    for a in range(0, nrm.shape[0], 4096):                 # the cosine matrix is large
        blk = nrm[a:a + 4096]
        cosw = np.maximum(blk @ dirs.T, 0.0)
        out[a:a + 4096] = (cosw @ grad) * dw

    for direction, deg, radiance in ((KEY_DIR, KEY_DEG, KEY_L),
                                     (FILL_DIR, FILL_DEG, FILL_L)):
        u = direction / np.linalg.norm(direction)
        omega = 2.0 * np.pi * (1.0 - np.cos(np.radians(deg)))
        out += np.maximum(nrm @ u, 0.0)[..., None] * radiance * omega
    return out


# ------------------------------------------------------------------- the BRDF
def onb(n: np.ndarray):
    """A tangent frame around n, branchless (Duff et al.), so no pixel is special."""
    s = np.where(n[..., 2] >= 0.0, 1.0, -1.0)
    a = -1.0 / (s + n[..., 2])
    b = n[..., 0] * n[..., 1] * a
    t = np.stack([1.0 + s * n[..., 0] ** 2 * a, s * b, -s * n[..., 0]], axis=-1)
    bt = np.stack([b, s + n[..., 1] ** 2 * a, -n[..., 1]], axis=-1)
    return t, bt


def specular_terms(nrm: np.ndarray, roughness: float, n_samples: int):
    """The specular integral, split into the two channels Schlick's F is linear in.

    Returns (A, B) with (..., 3) each, so that for any F0

        int f_s L (n.l) dl  =  F0 * A + B

    exactly, one integration serving every metallic value of a column.  The estimator is
    the standard NDF one: sampling h from D(h) (n.h) gives the weight
    L F G (v.h) / ((n.v)(n.h)), with F already factored out into its two pieces.
    """
    alpha = max(roughness ** 2, 1e-4)
    v = np.array([0.0, 0.0, 1.0])
    t, bt = onb(nrm)
    ndv = np.clip(nrm @ v, 1e-4, 1.0)

    # The lattice is shifted per pixel on BOTH axes (a Cranley-Patterson rotation, which
    # a uniform point set survives unchanged).  The same lattice everywhere leaves its own
    # residual in every pixel identically, and that residual then reads as structure in
    # the image -- the rough conductors came out with a wood grain on them.  The two
    # shifts come from the pixel index through two irrational multipliers, so they are
    # reproducible and correlate with nothing.
    i = np.arange(n_samples, dtype=np.float64)
    idx = np.arange(nrm.shape[0], dtype=np.float64)[:, None]
    rot1 = np.mod(idx * 0.7548776662466927 + 0.5, 1.0)         # the plastic constant
    rot2 = np.mod(idx * 0.5698402909980532 + 0.5, 1.0)         # its square
    u1 = np.mod((i[None, :] + 0.5) / n_samples + rot1, 1.0)
    u2 = np.mod(i[None, :] * PHI + rot2, 1.0)

    cos_h = np.sqrt((1.0 - u1) / (1.0 + (alpha ** 2 - 1.0) * u1))
    sin_h = np.sqrt(np.maximum(0.0, 1.0 - cos_h ** 2))
    phi = 2.0 * np.pi * u2

    A = np.zeros((nrm.shape[0], 3))
    B = np.zeros((nrm.shape[0], 3))
    step = max(1, int(4.0e6 / n_samples))
    for a0 in range(0, nrm.shape[0], step):
        sl = slice(a0, a0 + step)
        hl_x = (sin_h[sl] * np.cos(phi[sl]))[..., None] * t[sl, None, :]
        hl_y = (sin_h[sl] * np.sin(phi[sl]))[..., None] * bt[sl, None, :]
        hl_z = cos_h[sl][..., None] * nrm[sl, None, :]
        h = hl_x + hl_y + hl_z

        vdh = h[..., 2]                                  # v is +z, so v.h is h_z
        l = 2.0 * vdh[..., None] * h - v
        ndl = np.sum(l * nrm[sl, None, :], axis=-1)
        ndh = np.sum(h * nrm[sl, None, :], axis=-1)

        ok = (ndl > 1e-6) & (vdh > 1e-6) & (ndh > 1e-6)
        ndl_s = np.where(ok, ndl, 1.0)
        g = smith_g1(ndl_s, roughness) * smith_g1(ndv[sl, None], roughness)
        w = np.where(ok, g * vdh / (ndv[sl, None] * np.maximum(ndh, 1e-6)), 0.0)

        rad = env(l) * w[..., None]
        fb = (1.0 - np.clip(vdh, 0.0, 1.0)) ** 5          # the (1-c)^5 of Schlick
        A[sl] = np.mean(rad * (1.0 - fb)[..., None], axis=1)
        B[sl] = np.mean(rad * fb[..., None], axis=1)
    return A, B


# ------------------------------------------------------------------- spheres
def sphere_normals(px: int, ss: int):
    """Normals and coverage of an orthographic sphere, supersampled ss times."""
    p = px * ss
    y, x = np.mgrid[0:p, 0:p]
    u = (x + 0.5 - p / 2) / (p / 2) / 0.98
    w = -(y + 0.5 - p / 2) / (p / 2) / 0.98
    rr = u ** 2 + w ** 2
    inside = rr <= 1.0
    z = np.sqrt(np.maximum(0.0, 1.0 - rr))
    nrm = np.stack([u, w, z], axis=-1)[inside]
    return nrm, inside


def downsample(flat: np.ndarray, inside: np.ndarray, px: int, ss: int):
    """Scatter the shaded pixels back, then box-average the supersampling away."""
    p = px * ss
    img = np.zeros((p, p, 3))
    img[inside] = flat
    cov = inside.astype(np.float64)
    img = img.reshape(px, ss, px, ss, 3).mean(axis=(1, 3))
    cov = cov.reshape(px, ss, px, ss).mean(axis=(1, 3))
    return img, cov


def rgba(img: np.ndarray, cov: np.ndarray, scale: float) -> np.ndarray:
    out = np.zeros(img.shape[:2] + (4,), dtype=np.float32)
    out[..., :3] = np.clip(img * scale, 0.0, 1.0) ** (1.0 / GAMMA)
    out[..., 3] = cov
    return out


def f0_of(metallic: float) -> np.ndarray:
    """background.tex:214 -- 0.04 for a dielectric, the base colour for a conductor."""
    return F0_DIELECTRIC * (1.0 - metallic) + metallic * ALBEDO


def terms(E: np.ndarray, ab: tuple, metallic: float):
    """The two halves of one sphere, in radiance."""
    A, B = ab
    diffuse = (1.0 - metallic) * ALBEDO * E / np.pi
    specular = f0_of(metallic) * A + B
    return diffuse, specular


# -------------------------------------------------------------------- checks
def env_peak() -> float:
    """The brightest radiance the environment holds.  Measured at the cap centres and not
    only on the Fibonacci set: a 5.5-degree cap is 0.09% of the sphere, and a quadrature
    set that happens to miss it would understate the peak and make the energy check
    vacuous."""
    dirs = [fibonacci_sphere(4096),
            (KEY_DIR / np.linalg.norm(KEY_DIR))[None, :],
            (FILL_DIR / np.linalg.norm(FILL_DIR))[None, :]]
    return max(float(env(d).max()) for d in dirs)


def report(E, ab_by_r, verbose: bool = True) -> None:
    """Everything the figure asserts, measured on the arrays it is drawn from."""
    c = np.linspace(0.0, 1.0, 17)
    f0 = np.array([0.04, 0.2, 0.9])
    mine = f0 * (1.0 - (1.0 - c[:, None]) ** 5) + (1.0 - c[:, None]) ** 5
    err = np.abs(mine - schlick_f(c, f0)).max()
    if err > 1e-9:
        raise SystemExit(f"the split of Schlick's F does not reproduce schlick_f "
                         f"(max {err:.2e}): the two-channel accumulation is not the "
                         "same Fresnel the thesis figure uses")

    d1, _ = terms(E, ab_by_r[GRID_ROUGH[0]], 1.0)
    if np.abs(d1).max() != 0.0:
        raise SystemExit("the diffuse term of a conductor is not identically zero: "
                         "the (1 - m) weight is not wired to the diffuse lobe")

    peaks = []
    for r in GRID_ROUGH:
        A, B = ab_by_r[r]
        rho_s = (np.ones(3) * A + B).max()          # F0 = 1: the directional albedo
        peaks.append(float((f0_of(1.0) * A + B).max()))
        if rho_s > 1.02 * env_peak():
            raise SystemExit(f"the specular estimator at r = {r} reflects more than the "
                             "brightest direction of the environment")
    if any(b >= a for a, b in zip(peaks, peaks[1:])):
        raise SystemExit(f"the peak specular radiance does not fall with roughness "
                         f"({[round(p, 2) for p in peaks]}): the roughness knob is not "
                         "reaching the GGX lobe")
    if verbose:
        print("  peak specular radiance, conductor, per roughness: "
              + ", ".join(f"r={r:.2f}: {p:8.2f}" for r, p in zip(GRID_ROUGH, peaks)))


# -------------------------------------------------------------------- drawing
def equation_band(ax) -> None:
    """The model, with the three parameters coloured, and what each one does."""
    def row(fragments, size, y):
        boxes = [TextArea(s, textprops=dict(color=c, fontsize=size))
                 for s, c in fragments]
        box = AnchoredOffsetbox(loc="center", child=HPacker(children=boxes, sep=0,
                                                            pad=0, align="center"),
                                bbox_to_anchor=(0.5, y), bbox_transform=ax.transAxes,
                                frameon=False, pad=0.0, borderpad=0.0)
        ax.add_artist(box)

    row([(r"$f_r(\omega_i,\omega_o)\;=\;(1-$", C_INK), (r"$m$", C_MET),
         (r"$)\,$", C_INK), (r"$\rho$", C_ALB), (r"$/\pi$", C_INK),
         (r"$\;+\;$", C_INK),
         (r"$\frac{D(\alpha)\;F(F_0)\;G}"
          r"{4\,(\mathbf{n}\cdot\omega_i)(\mathbf{n}\cdot\omega_o)}$", C_INK)],
        23, 0.79)
    row([(r"$\alpha=$", C_INK), (r"$r^{2}$", C_ROU),
         (r"$\qquad F_0=0.04\,(1-$", C_INK), (r"$m$", C_MET), (r"$)+$", C_INK),
         (r"$m$", C_MET), (r"$\,$", C_INK), (r"$\rho$", C_ALB)],
        15, 0.45)

    roles = [(0.170, C_ALB, r"$\rho$   albedo",
              "the colour: what a dielectric\nscatters, what a conductor tints"),
             (0.500, C_MET, r"$m$   metallic",
              "removes the diffuse lobe\nand moves $\\rho$ into $F_0$"),
             (0.830, C_ROU, r"$r$   roughness",
              "widens the specular lobe,\nfrom mirror to matte")]
    for x, colour, head, body in roles:
        ax.text(x, 0.235, head, ha="center", va="center", fontsize=16, color=colour,
                transform=ax.transAxes)
        ax.text(x, 0.045, body, ha="center", va="center", fontsize=11.5,
                color=C_MUTED, transform=ax.transAxes, linespacing=1.45)


def put_sphere(fig, rect, img) -> None:
    ax = fig.add_axes(rect)
    ax.imshow(img, interpolation="bilinear")
    ax.axis("off")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="destination folder")
    ap.add_argument("--name", default="disney_brdf", help="output stem, without .png")
    ap.add_argument("--sphere", type=int, default=230, help="sphere side, in pixels")
    ap.add_argument("--ss", type=int, default=2, help="supersampling factor")
    ap.add_argument("--samples", type=int, default=224,
                    help="lattice points of the specular quadrature")
    ap.add_argument("--dpi", type=int, default=DPI)
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    nrm, inside = sphere_normals(a.sphere, a.ss)
    print(f"sphere {a.sphere} px x{a.ss}, {nrm.shape[0]} shaded samples, "
          f"{a.samples} lattice points")

    E = irradiance(nrm)
    print(f"  irradiance E(n) in [{E.min():.3f}, {E.max():.3f}]")

    ab_by_r = {}
    for r in sorted(set(GRID_ROUGH + [SPLIT_ROUGH])):
        ab_by_r[r] = specular_terms(nrm, r, a.samples)
        print(f"  specular integral at r = {r:.2f} done")

    report(E, ab_by_r)

    # -- one exposure for every sphere in the figure -------------------------
    finals = []
    for r in GRID_ROUGH:
        for m in GRID_METAL:
            d, s = terms(E, ab_by_r[r], m)
            finals.append(d + s)
    scale = 1.0 / np.percentile(np.concatenate(finals), EXPOSURE_PCTL)
    clipped = float(np.mean(np.concatenate(finals) * scale > 1.0))
    print(f"  exposure 1/p{EXPOSURE_PCTL} = {scale:.4f}, "
          f"{100 * clipped:.2f}% of the shaded samples clip")

    def shade(r, m, part):
        d, s = terms(E, ab_by_r[r], m)
        flat = {"diffuse": d, "specular": s, "final": d + s}[part]
        return rgba(*downsample(flat, inside, a.sphere, a.ss), scale)

    # -- layout, in inches ---------------------------------------------------
    m_side, m_top, m_bot = 0.45, 0.30, 0.62
    band_h = 1.85                       # the equation band
    gap_band = 0.34
    sph = 1.42                          # sphere side in the decomposition
    op_w = 0.44                         # room for the + and = signs
    split_w = 3 * sph + 2 * op_w
    split_lab = 0.86                    # the row label to the left of the spheres
    split_head = 0.30                   # the column captions under the spheres
    split_block_w = split_lab + split_w
    split_block_h = 2 * sph + 0.24 + split_head + 0.42

    gsph = 1.22                         # sphere side in the grid
    grid_lab_l, grid_lab_t = 0.86, 0.72     # room for the axis name AND the tick row
    grid_block_w = grid_lab_l + len(GRID_ROUGH) * gsph
    grid_block_h = grid_lab_t + len(GRID_METAL) * gsph + 0.42

    gap_blocks = 0.62
    body_h = max(split_block_h, grid_block_h)
    fig_w = m_side * 2 + split_block_w + gap_blocks + grid_block_w
    fig_h = m_top + band_h + gap_band + body_h + m_bot
    fig = plt.figure(figsize=(fig_w, fig_h))

    ax_eq = fig.add_axes([m_side / fig_w, (m_bot + body_h + gap_band) / fig_h,
                          1.0 - 2 * m_side / fig_w, band_h / fig_h])
    ax_eq.axis("off")
    equation_band(ax_eq)

    y_rule = (m_bot + body_h + gap_band * 0.5) / fig_h      # separates model from renders
    fig.add_artist(plt.Line2D([m_side / fig_w, 1.0 - m_side / fig_w], [y_rule, y_rule],
                              color=C_MUTED, lw=0.8, alpha=0.35,
                              transform=fig.transFigure))

    # -- the decomposition ---------------------------------------------------
    x_split = m_side
    y_top = m_bot + body_h                       # top of the body band
    # The two blocks are not the same height and are top-aligned: their titles have to
    # sit on one line, and a gap under the shorter one reads better than a heading that
    # floats above its own content.
    fig.text((x_split + split_block_w / 2) / fig_w, (y_top - 0.02) / fig_h,
             f"one material, its two terms   ($r={SPLIT_ROUGH:g}$)",
             ha="center", va="top", fontsize=13.5, color=C_INK)
    row_y = y_top - 0.42
    for k, m in enumerate(SPLIT_METAL):
        yb = row_y - (k + 1) * sph - k * 0.24
        fig.text((x_split + split_lab - 0.16) / fig_w, (yb + sph / 2) / fig_h,
                 f"$m={m:.0f}$\n" + ("dielectric" if m == 0 else "conductor"),
                 ha="right", va="center", fontsize=13, color=C_INK, linespacing=1.5)
        for j, part in enumerate(("diffuse", "specular", "final")):
            xb = x_split + split_lab + j * (sph + op_w)
            put_sphere(fig, [xb / fig_w, yb / fig_h, sph / fig_w, sph / fig_h],
                       shade(SPLIT_ROUGH, m, part))
            if j < 2:
                fig.text((xb + sph + op_w / 2) / fig_w, (yb + sph / 2) / fig_h,
                         "+" if j == 0 else "=", ha="center", va="center",
                         fontsize=22, color=C_MUTED)
    y_cap = row_y - 2 * sph - 0.24 - 0.20
    for j, lab in enumerate((r"$(1-m)\,\rho\,E/\pi$", "specular", "= final colour")):
        xb = x_split + split_lab + j * (sph + op_w)
        fig.text((xb + sph / 2) / fig_w, y_cap / fig_h, lab, ha="center", va="top",
                 fontsize=12, color=C_MUTED)
    # The middle sphere of the first row is nearly black, and a reader is entitled to
    # wonder whether something failed.  Nothing did: 4% is what a dielectric reflects.
    fig.text((x_split + 0.34) / fig_w, (y_cap - 0.42) / fig_h,
             "a dielectric returns about 4% through the specular lobe, so\n"
             "its colour is the diffuse term; a conductor has no diffuse\n"
             "term at all and is coloured by $F_0 = \\rho$ alone",
             ha="left", va="top", fontsize=11.5, color=C_MUTED, linespacing=1.6)

    # -- the grid ------------------------------------------------------------
    x_grid = m_side + split_block_w + gap_blocks
    fig.text((x_grid + grid_block_w / 2) / fig_w, (y_top - 0.02) / fig_h,
             "the two knobs, on one albedo", ha="center", va="top",
             fontsize=13.5, color=C_INK)
    g_top = y_top - 0.42
    fig.text((x_grid + grid_lab_l + len(GRID_ROUGH) * gsph / 2) / fig_w,
             (g_top - 0.04) / fig_h, r"roughness $r$", ha="center", va="top",
             fontsize=13, color=C_ROU)
    for j, r in enumerate(GRID_ROUGH):
        xb = x_grid + grid_lab_l + j * gsph
        fig.text((xb + gsph / 2) / fig_w, (g_top - 0.32) / fig_h, f"{r:.2f}",
                 ha="center", va="top", fontsize=12.5, color=C_ROU)
    for i, m in enumerate(GRID_METAL):
        yb = g_top - grid_lab_t - (i + 1) * gsph
        fig.text((x_grid + grid_lab_l - 0.14) / fig_w, (yb + gsph / 2) / fig_h,
                 f"{m:.1f}", ha="right", va="center", fontsize=12.5, color=C_MET)
        for j, r in enumerate(GRID_ROUGH):
            xb = x_grid + grid_lab_l + j * gsph
            put_sphere(fig, [xb / fig_w, yb / fig_h, gsph / fig_w, gsph / fig_h],
                       shade(r, m, "final"))
    fig.text((x_grid + 0.18) / fig_w,
             (g_top - grid_lab_t - len(GRID_METAL) * gsph / 2) / fig_h,
             r"metallic $m$", ha="center", va="center", rotation=90,
             fontsize=13, color=C_MET)

    fig.text(0.5, 0.14 / fig_h,
             "every sphere is rendered with the BRDF above, under one analytic "
             "environment and one exposure, then gamma 1/2.2\n"
             "the two terms add in radiance, not in the encoded pixels",
             ha="center", va="bottom", fontsize=10.5, color=C_MUTED, linespacing=1.5)

    path = out / f"{a.name}.png"
    fig.savefig(path, dpi=a.dpi, facecolor="white")
    plt.close(fig)
    print(f"  + {path}  ({int(fig_w * a.dpi)}x{int(fig_h * a.dpi)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
