#!/usr/bin/env python
"""make_cone_diagram.py -- figures and worked example for the cone equation.

    python make_cone_diagram.py --out ../Doc/images/cone

Writes three PNGs and prints the rows of the worked example's LaTeX table:

  cone_geometry.png  texel, normal, camera, reflected ray and the candidate cones
  cone_rings.png     the shared Fibonacci set, coloured by ring, in 3D and in the
                     view along R where the binning can be counted by eye
  cone_weights.png   what a ray is worth: the same region in an equal-area
                     projection, on the left the real shared set with its Voronoi
                     cells, on the right a flat budget of N rays per ring

The weights and the cone closing are NOT rewritten here: they come from
images_generator, the same ones the bake uses.  If they ever diverge, the thesis figure
diverges too and it gets noticed.

The case is didactic but not fake: Fibonacci directions like the shared kernel's
(equispaced cos(theta), azimuth on the golden angle) and an analytic radiance
L(d) = 0.5 + 0.5*d_z, the same 'gradient' envmap test_hemivis_shared.py checks the bake
with.  On that envmap the mean over the cone of half-aperture b around R, when the cone
lies entirely above the horizon, has the closed form

    L = 0.5 + 0.5 * R_z * (1 + cos b) / 2

so the table can show the estimate next to the exact value instead of asking the reader
to take it on trust.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Wedge
from scipy.spatial import SphericalVoronoi

import _paths  # noqa: F401

from images_generator import (_cones_from_rings_np, ring_weights_mean,
                              spec_cone_ring_samples,
                              spec_cone_shared_ring_samples)

# Didactic case: few rays, countable by eye, and an aperture grid reduced to four rings
# instead of the thirteen of the operational configuration.
APERTURES = [0.0, 30.0, 60.0, 90.0, 140.0]   # TOTAL apertures, in degrees
S = 96                                        # shared directions over the hemisphere
THETA_V = 30.0                                # camera tilt from the normal
PHI_V = 200.0                                 # its azimuth, chosen only for the framing

# Flat budget of the weights figure: an arbitrary number of rays per ring, the same for
# all.  It is the didactic exaggeration of the aimed allocation, where the floor on the
# first rings still leaves about a factor of twenty between the densest and sparsest ray.
AIMED_PER_RING = 10

# Single-hue sequential ramp: the rings are ordered, so the colour has to grow with the
# index.  No rainbow.  Grey for the rays outside every cone.
RING_COLORS = plt.get_cmap("Blues")(np.linspace(0.45, 0.92, len(APERTURES) - 1))
C_OUT = "#b8b8b8"
C_INK = "#222222"
C_MIRROR = "#c0392b"

# On the page the figure is shrunk by about a third: the sizes here are chosen so that
# the text stays readable AFTER that reduction, not on screen.
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
    golden angle.  The kernel's per-texel rotation is not needed here, there is only one
    texel.
    """
    i = np.arange(s)
    cos_t = 1.0 - (i + 0.5) / s
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t ** 2))
    phi = i * np.pi * (3.0 - np.sqrt(5.0))
    return np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=-1)


def radiance(d: np.ndarray) -> np.ndarray:
    """The tests' 'gradient' envmap: linear in the vertical component."""
    return 0.5 + 0.5 * d[..., 2]


def circle_on_sphere(axis: np.ndarray, half_deg: float, n: int = 240) -> np.ndarray:
    """Directions at `half_deg` degrees from `axis`: the edge of a cone."""
    t, b = onb(axis)
    a = np.radians(half_deg)
    ang = np.linspace(0, 2 * np.pi, n)
    return (np.cos(a) * axis[None, :]
            + np.sin(a) * (np.cos(ang)[:, None] * t + np.sin(ang)[:, None] * b))


def _hemisphere_voronoi(dirs: np.ndarray) -> tuple[list, np.ndarray]:
    """Spherical Voronoi cells of the shared set, closed exactly on the horizon.

    The patch of sky a ray stands for is the region of directions closer to it than to any
    other ray: that, and not a sector drawn by hand, is what says how much a sample is
    worth.

    The set lives only above n, and a Voronoi on half a sphere would leave the border
    cells open down below the surface.  Adding the mirrored points below the horizon
    closes them at z = 0 without approximating anything: for a direction u with u_z > 0
    and a pair q, q' = (q_x, q_y, -q_z), u.q > u.q' always holds, so no upper cell
    crosses the equator.  The sum of the first S areas comes back to exactly 2*pi.

    The areas are NOT all equal: half are within 0.6% of 2*pi/S, but at the edge of the
    lattice (the points a tenth of a degree from the equator) the cell is cut by the
    surface and is about half.  That is lattice geometry, not bake geometry: the cone
    equation uses the nominal weight 2*pi/S, and with a constant W_i that factor cancels,
    so the estimate stays the mean of the samples.  That is why the figure colours the
    cells by the nominal weight and draws their true shape.
    """
    mirrored = dirs * np.array([1.0, 1.0, -1.0])
    sv = SphericalVoronoi(np.concatenate([dirs, mirrored], axis=0),
                          radius=1.0, center=np.zeros(3))
    sv.sort_vertices_of_regions()
    cells = [sv.vertices[r] for r in sv.regions[:len(dirs)]]
    return cells, sv.calculate_areas()[:len(dirs)]


def build_case() -> dict:
    """Geometry, binning and every quantity of the equation."""
    n = np.array([0.0, 0.0, 1.0])
    tv, pv = np.radians(THETA_V), np.radians(PHI_V)
    v = np.array([np.sin(tv) * np.cos(pv), np.sin(tv) * np.sin(pv), np.cos(tv)])
    r = 2.0 * np.dot(n, v) * n - v          # Reflected ray equation

    dirs = fibonacci_hemisphere(S)
    ang = np.degrees(np.arccos(np.clip(dirs @ r, -1.0, 1.0)))   # angle from R
    half = np.array(APERTURES) / 2.0
    ring = np.digitize(ang, half[1:], right=True)               # 0..K-2 inside, K-1 outside
    ring[ang > half[-1]] = len(half) - 1                        # outside the widest cone

    k = len(APERTURES)
    ring_sum = np.zeros((1, k, 3))
    ring_valid = np.zeros((1, k))
    lum = radiance(dirs)
    for i in range(k - 1):
        m = ring == i
        ring_sum[0, i + 1] = lum[m].sum()
        ring_valid[0, i + 1] = m.sum()
    ring_sum[0, 0] = radiance(r)            # mirror level: a single ray
    ring_valid[0, 0] = 1.0

    cos_b = np.cos(np.radians(np.asarray(APERTURES)) * 0.5)
    omega = 2.0 * np.pi * (cos_b[:-1] - cos_b[1:])
    n_nom = np.asarray(spec_cone_shared_ring_samples(APERTURES, S))
    w = ring_weights_mean(cos_b, k - 1, n_nom)
    cones = _cones_from_rings_np(ring_sum, ring_valid, w)[0, :, 0]

    # The two budget allocations compared in fig_weights.  The aimed one goes through the
    # production function, so if the allocation ever changes the figure changes with it.
    n_aimed = np.asarray(spec_cone_ring_samples(APERTURES, AIMED_PER_RING, alloc="uniform"),
                         dtype=np.float64)
    # The drawn weights are divided by hand and do NOT go through ring_weights_mean: with
    # a uniform ring_samples that one skips the division on purpose, because a constant
    # factor cancels between numerator and denominator of the cone equation.  Exact for
    # the cone's value, wrong by a factor N for a figure whose subject is precisely how
    # much a single patch is worth.
    w_draw_aimed = omega / n_aimed
    w_shared = 2.0 * np.pi / S          # = Omega_i/N_i with N_i = S*Omega_i/2pi, for every i

    cells, cell_area = _hemisphere_voronoi(dirs)

    # Closed form, valid only where the cone does not touch the horizon
    exact = 0.5 + 0.5 * r[2] * (1.0 + cos_b) / 2.0
    unclipped = np.degrees(np.arccos(r[2])) + half <= 90.0

    return dict(n=n, v=v, r=r, dirs=dirs, ang=ang, ring=ring, lum=lum,
                omega=omega, n_nom=n_nom, w=w, ring_sum=ring_sum[0, :, 0],
                ring_valid=ring_valid[0], cones=cones, exact=exact,
                unclipped=unclipped, half=half, cos_b=cos_b,
                n_aimed=n_aimed, w_draw_aimed=w_draw_aimed, w_shared=w_shared,
                cells=cells, cell_area=cell_area)


def _frame(ax, case: dict, arrows: bool = True) -> None:
    """Surface plane, horizon circle, normal, camera, reflected ray."""
    # The surface square sits inside the equator: any larger and it would cover the
    # grazing directions and make them look as if they were below the surface, which is
    g = np.linspace(-0.62, 0.62, 2)
    gx, gy = np.meshgrid(g, g)
    ax.plot_surface(gx, gy, np.zeros_like(gx), color="0.9", alpha=0.45,
                    edgecolor="0.65", linewidth=0.6, zorder=0)
    # the hemisphere's equator: without it the 3D reads as a flat drawing
    a = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(a), np.sin(a), np.zeros_like(a), color="0.6", lw=0.9, zorder=1)
    if arrows:
        # The labels are shifted to the side: above the tip there are also the aperture
        # ones, which sit on top of their respective circles.
        for vec, lab, col, off in (
                (case["n"], r"$\mathbf{n}$", C_INK, (-0.12, 0.12, 0.06)),
                (case["v"], r"$\mathbf{v}$", C_INK, (-0.10, 0.10, 0.06)),
                (case["r"], r"$\mathbf{R}$", C_MIRROR, (0.10, -0.10, 0.06))):
            ax.quiver(0, 0, 0, *vec, color=col, lw=2.0, arrow_length_ratio=0.12)
            ax.text(*(vec * 1.06 + np.array(off)), lab, color=col, fontsize=15)
    ax.scatter([0], [0], [0], color=C_INK, s=22, zorder=5)
    # The box ratio MUST follow the extent of the data, otherwise the vertical scale
    # differs from the horizontal one and the angles can no longer be read: the cone
    # circles, which lie on planes perpendicular to R, look wrongly rotated and n looks
    # longer than R even though both are unit vectors.  In a diagram whose content is
    # the angles that is a bug, not a framing choice.
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(0, 1.1)
    ax.set_box_aspect((2.1, 2.1, 1.1))
    ax.set_axis_off()
    ax.view_init(elev=32, azim=-72)


def fig_geometry(case: dict, out: Path) -> None:
    fig = plt.figure(figsize=(6.2, 4.6))
    ax = fig.add_subplot(111, projection="3d")
    _frame(ax, case)

    ax.text(0.26, -0.06, 0.0, "texel", color=C_INK, fontsize=12)
    # Arc between R and the edge of the 60-degree cone, drawn on the unit sphere so that
    # its ends fall exactly on R and on the circle: that is where one sees the aperture is
    # TOTAL and the edge is at half.  It opens on the side opposite the normal, where
    # there is nothing else to read.
    hd = case["half"][2]
    u = case["n"] - np.dot(case["n"], case["r"]) * case["r"]
    u = -u / np.linalg.norm(u)
    arc = np.array([np.cos(np.radians(g)) * case["r"] + np.sin(np.radians(g)) * u
                    for g in np.linspace(0, hd, 60)])
    ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color=C_INK, lw=1.1)
    ax.text(*(arc[len(arc) // 2] * 1.06), r"$\Theta/2$", color=C_INK, fontsize=14)

    for i, hd in enumerate(case["half"][1:]):
        c = circle_on_sphere(case["r"], hd)
        keep = c[:, 2] >= 0                      # below the horizon there are no directions
        cc = c.copy()
        cc[~keep] = np.nan
        ax.plot(cc[:, 0], cc[:, 1], cc[:, 2], color=RING_COLORS[i], lw=2.0)
        # Label on the side of the circle, in the direction perpendicular to the plane
        # containing n and R: it is the only area where the circles overlap neither each
        # other nor the arrows, and the different radii separate the labels by themselves.
        _, side = onb(case["r"])
        p = np.cos(np.radians(hd)) * case["r"] + np.sin(np.radians(hd)) * side
        ax.text(*(p * 1.08), f"{APERTURES[i + 1]:.0f}$^\\circ$",
                color=C_INK, fontsize=12, ha="center")

    # generatrices of the widest cone, to make the solid read instead of the border
    for t in np.linspace(0, 2 * np.pi, 13)[:-1]:
        e = circle_on_sphere(case["r"], case["half"][-1], 240)
        p = e[int(t / (2 * np.pi) * 239)]
        if p[2] < 0:
            continue
        ax.plot([0, p[0]], [0, p[1]], [0, p[2]], color=RING_COLORS[-1],
                lw=0.6, alpha=0.35)

    # Title as figure text and not axis text: 3D axes leave an empty band above the
    # content, and a title attached to the axis would turn it into margin inside the
    # page.
    fig.text(0.5, 0.9, "Candidate cones share the reflected ray as their axis",
             ha="center", fontsize=14, color=C_INK)
    ax.set_position([0.0, -0.06, 1.0, 1.06])
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"  + {out}")


def fig_rings(case: dict, out: Path) -> None:
    fig = plt.figure(figsize=(10.0, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.05), wspace=0.02)

    ax = fig.add_subplot(gs[0], projection="3d")
    _frame(ax, case)
    d, ring = case["dirs"], case["ring"]
    for i in range(len(APERTURES) - 1):
        m = ring == i
        ax.scatter(d[m, 0], d[m, 1], d[m, 2], s=30, color=RING_COLORS[i],
                   depthshade=False, edgecolor="white", linewidth=0.4, zorder=4)
    m = ring == len(APERTURES) - 1
    ax.scatter(d[m, 0], d[m, 1], d[m, 2], s=14, color=C_OUT, depthshade=False,
               alpha=0.6, zorder=3)
    for i, hd in enumerate(case["half"][1:]):
        c = circle_on_sphere(case["r"], hd)
        c[c[:, 2] < 0] = np.nan
        ax.plot(c[:, 0], c[:, 1], c[:, 2], color=RING_COLORS[i], lw=1.2, alpha=0.85)
    ax.set_title(f"The {S} shared directions, binned by angle to $\\mathbf{{R}}$",
                 fontsize=13, color=C_INK, pad=-8)

    # View along R: radius = angle from R, azimuth around R.  This is where the points
    # per ring get counted.
    axp = fig.add_subplot(gs[1], projection="polar")
    t, b = onb(case["r"])
    az = np.arctan2(d @ b, d @ t)
    for i in range(len(APERTURES) - 1):
        m = ring == i
        axp.scatter(az[m], case["ang"][m], s=30, color=RING_COLORS[i],
                    edgecolor="white", linewidth=0.4, zorder=4,
                    label=f"ring {i + 1}: {APERTURES[i]:.0f}$^\\circ$ to "
                          f"{APERTURES[i + 1]:.0f}$^\\circ$  ($n_{i + 1}$ = "
                          f"{int(case['ring_valid'][i + 1])})")
    m = ring == len(APERTURES) - 1
    axp.scatter(az[m], case["ang"][m], s=18, color=C_OUT, zorder=3,
                label="outside every candidate")
    axp.scatter([0], [0], marker="+", s=110, color=C_MIRROR, zorder=6,
                linewidth=2.0, label="mirror ray $\\mathbf{R}$")

    # The horizon seen from here: where a direction at that angle from R sinks below the
    # surface.  It is the reason the outer rings never fill up.
    # Maximum angle from the R axis that stays above the horizon, for each azimuth: the
    # direction is cos(g)*R + sin(g)*(cos(az)*T + sin(az)*B) and it sinks where its
    # vertical component vanishes, i.e. tan(g) = -R_z / k_z.  Where k_z >= 0 the solution
    # falls beyond 90 degrees and off the plot, which is correct: on that side the cone
    # never meets the surface.
    aa = np.linspace(0, 2 * np.pi, 361)
    kz = np.cos(aa) * t[2] + np.sin(aa) * b[2]
    horiz = np.degrees(np.arctan2(-case["r"][2], kz)) % 180.0
    axp.plot(aa, horiz, color="0.45", lw=1.2, ls="--", zorder=2,
             label="surface horizon")

    for hd in case["half"][1:]:
        axp.plot(np.linspace(0, 2 * np.pi, 200), np.full(200, hd),
                 color="0.55", lw=0.8, zorder=1)
    axp.set_rmax(90)
    axp.set_rticks(list(case["half"][1:]))
    axp.set_rlabel_position(112)     # away from the area where the points crowd
    axp.set_yticklabels([f"{h:.0f}$^\\circ$" for h in case["half"][1:]], fontsize=11,
                        color="0.35")
    axp.set_xticklabels([])
    axp.grid(color="0.85", lw=0.6)
    axp.set_title("Same directions, seen along $\\mathbf{R}$", fontsize=13,
                  color=C_INK, pad=14)
    # Legend below and on three columns: in a column on the right it stole width from the
    # panels, which on the page are already reduced by a third and became unreadable.
    axp.legend(loc="upper center", bbox_to_anchor=(-0.08, -0.02), ncol=3, fontsize=11,
               frameon=False, handletextpad=0.4, columnspacing=1.6)

    fig.tight_layout()
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"  + {out}")


def _equal_area_radius(half_deg) -> np.ndarray:
    """Lambert azimuthal equal-area projection around R: rho = 2 sin(theta/2).

    It is the only choice that makes the figure readable as what it claims to be: with
    this radius the drawn area of a region IS its solid angle in steradians
    (rho drho = sin theta dtheta), so a cell twice as large is a ray standing for twice
    as much sky.  The polar view of cone_rings.png, where the radius is the angle from R,
    does not have this property and here it would be a figure that lies.
    """
    return 2.0 * np.sin(np.radians(np.asarray(half_deg, dtype=np.float64)) * 0.5)


def _project_disk(u: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """(..., 3) directions -> (..., 2) in the equal-area disc around `axis`.

    rho = 2 sin(gamma/2) = sqrt(2(1 - cos gamma)), azimuth around the axis.
    """
    t, b = onb(axis)
    c = np.clip(u @ axis, -1.0, 1.0)
    rho = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - c)))
    az = np.arctan2(u @ b, u @ t)
    return np.stack([rho * np.cos(az), rho * np.sin(az)], axis=-1)


def _slerp_ring(v: np.ndarray, steps: int = 6) -> np.ndarray:
    """Sides of a spherical cell sampled along the geodesic instead of in a straight line.

    A cell is about fifteen degrees wide: joining the vertices with segments in the disc
    would make the borders visibly straighter than they are and adjacent cells would not
    line up.
    """
    out = []
    for k in range(len(v)):
        a, b = v[k], v[(k + 1) % len(v)]
        w = np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)),
                              -1.0, 1.0))
        if w < 1e-9:
            out.append(a[None, :])
            continue
        s = np.linspace(0.0, 1.0, steps, endpoint=False)[:, None]
        out.append((np.sin((1.0 - s) * w) * a + np.sin(s * w) * b) / np.sin(w))
    return np.concatenate(out, axis=0)


def _dot_color(color) -> str:
    """White or black dot depending on the cell: the ramp spans the whole range, and a
    white dot vanishes on yellow just as a black one vanishes on black."""
    lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    return C_INK if lum > 0.55 else "white"


def _weights_chrome(ax, case: dict, rows: str) -> None:
    """Ring circles, apertures, mirror ray, horizon, list below the panel.
    Identical in the two panels: the only thing that changes between them must be N_i."""
    rho = _equal_area_radius(case["half"])
    a = np.linspace(0, 2 * np.pi, 400)
    for i in range(len(APERTURES) - 1):
        # Same per-ring colour as cone_rings.png and cone_geometry.png: the three figures
        # have to read as the same object.
        ax.plot(rho[i + 1] * np.cos(a), rho[i + 1] * np.sin(a),
                color=RING_COLORS[i], lw=1.6, zorder=5)
        ax.text(0.0, rho[i + 1] + 0.035, f"{APERTURES[i + 1]:.0f}$^\\circ$",
                ha="center", va="bottom", fontsize=11, color=C_INK, zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.8, alpha=0.85))

    # The horizon: the great circle perpendicular to n, seen from here.  The widest cone
    # crosses it, and that is where the shared set ends.
    e1, e2 = onb(case["n"])
    ang = np.linspace(0, 2 * np.pi, 721)
    xy = _project_disk(np.cos(ang)[:, None] * e1 + np.sin(ang)[:, None] * e2, case["r"])
    ax.plot(xy[:, 0], xy[:, 1], color="0.45", lw=1.2, ls="--", zorder=5,
            clip_path=Circle((0, 0), rho[-1], transform=ax.transData))

    ax.scatter([0], [0], marker="+", s=120, color=C_MIRROR, linewidth=2.0, zorder=6)
    ax.text(0.10, -0.13, r"$\mathbf{R}$", color=C_MIRROR, fontsize=14, zorder=6)

    lim = rho[-1] * 1.16
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.text(0.5, -0.04, rows, transform=ax.transAxes, ha="center", va="top",
            fontsize=11, color=C_INK, family="monospace", linespacing=1.5)


def _panel_shared(ax, case: dict, norm, cmap) -> None:
    """The real shared set, each direction with its Voronoi cell."""
    rho = _equal_area_radius(case["half"])
    clip = Circle((0, 0), rho[-1], transform=ax.transData)
    # A single hue for every cell, without distinguishing which ring they fall in: that is
    # exactly what the panel claims, a ray is worth 2*pi/S whichever ring takes it in.
    # The binning is in the circles drawn on top, and it is the subject of cone_rings.png,
    # not of this one.
    color = cmap(norm(case["w_shared"]))
    for cell in case["cells"]:
        p = Polygon(_project_disk(_slerp_ring(cell), case["r"]), closed=True,
                    facecolor=color, edgecolor="white", lw=0.7, zorder=2)
        p.set_clip_path(clip)
        ax.add_patch(p)

    pts = _project_disk(case["dirs"], case["r"])
    s = ax.scatter(pts[:, 0], pts[:, 1], s=9, color=_dot_color(color), zorder=4,
                   linewidths=0)
    s.set_clip_path(clip)


def _panel_aimed(ax, case: dict, norm, cmap) -> None:
    """Flat budget: ring i cut into N equal sectors, drawn to scale."""
    rho = _equal_area_radius(case["half"])
    for i in range(len(APERTURES) - 1):
        n_i = int(case["n_aimed"][i])
        color = cmap(norm(case["w_draw_aimed"][i]))
        r_in, r_out = rho[i], rho[i + 1]
        # Sectors with no offset between one ring and the next: the radial borders line up
        # and read as ten rays crossing every ring, which is exactly what the panel is
        # telling.
        edges = np.linspace(0.0, 360.0, n_i + 1)
        for k in range(n_i):
            ax.add_patch(Wedge((0.0, 0.0), r_out, edges[k], edges[k + 1],
                               width=r_out - r_in, facecolor=color,
                               edgecolor="white", lw=0.7, zorder=2))
        # One dot per cell, on the radius that halves its area: it is the traced ray, and
        # a reminder that the cell is its patch of sky and not a decoration.
        mid_a = np.radians(0.5 * (edges[:-1] + edges[1:]))
        mid_r = np.sqrt(0.5 * (r_in ** 2 + r_out ** 2))
        ax.scatter(mid_r * np.cos(mid_a), mid_r * np.sin(mid_a), s=9,
                   color=_dot_color(color), zorder=4, linewidths=0)


def fig_weights(case: dict, out: Path) -> None:
    fig = plt.figure(figsize=(10.0, 6.4))
    # Three rows and not two: the middle one stays empty and makes room for the ring list,
    # which lives in the panel coordinates and would otherwise end up under the bar.
    gs = fig.add_gridspec(3, 2, height_ratios=(1.0, 0.24, 0.04), hspace=0.10, wspace=0.04)

    w_all = np.append(case["w_draw_aimed"], case["w_shared"])
    # Logarithmic scale: between the smallest and the largest cell there is a factor of
    # ten, and in linear the bottom two thirds would all land in the same dark tint.  The
    # size stays the main channel, the colour reinforces it.
    norm = LogNorm(vmin=w_all.min(), vmax=w_all.max())
    cmap = plt.get_cmap("magma_r")

    # On the left the list has one column more: the panel shows the real rays, so next to
    # the budget N_i (an EXPECTED value, as the decimals say) one can read the count n_i
    # that actually arrived.  On the right there are no real rays.
    rows_shared = "\n".join(
        f"ring {i + 1}:  N = {case['n_nom'][i]:4.1f}   n = {int(case['ring_valid'][i + 1]):2d}"
        f"   W = {case['w_shared']:.3f} sr" for i in range(len(APERTURES) - 1))
    rows_aimed = "\n".join(
        f"ring {i + 1}:  N = {int(case['n_aimed'][i]):2d}   W = {case['w_draw_aimed'][i]:.3f} sr"
        for i in range(len(APERTURES) - 1))

    panels = ((_panel_shared, rows_shared,
               f"The {S} shared directions, uniform in solid angle",
               f"each owns a patch of $2\\pi/S$ = {case['w_shared']:.3f} sr"),
              (_panel_aimed, rows_aimed,
               f"Flat budget, $N_i$ = {AIMED_PER_RING} on every ring",
               "what a ray carries spans "
               f"{case['w_draw_aimed'].max() / case['w_draw_aimed'].min():.1f}$\\times$"))
    for col, (draw, rows, title, sub) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        draw(ax, case, norm, cmap)
        _weights_chrome(ax, case, rows)
        ax.set_title(f"{title}\n{sub}", fontsize=13, color=C_INK, pad=10)

    cax = fig.add_subplot(gs[2, :])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation="horizontal")
    cb.set_label("solid angle one ray stands for, $W_i = \\Omega_i / N_i$  [sr]",
                 fontsize=12, color=C_INK)
    # Ticks on the values the cells really have, not on the decades (within a factor of
    # ten none of them falls): the four of the flat-budget panel, which cover the whole
    # range.  The shared panel's weight, 2*pi/S, falls between the second and the third
    # and need not be repeated: it is already in the title and in the list.
    ticks = np.sort(case["w_draw_aimed"])
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:.3f}" for t in ticks], fontsize=10)
    cb.ax.minorticks_off()

    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"  + {out}")


def print_table(case: dict) -> None:
    print("\n% rows of the worked example's table (generated, not hand-written)")
    print("% ring & Theta_i & Omega_i & N_i & n_i & b_i & W_i & L(Theta_k) & exact")
    print(f"mirror & $0^\\circ$ & \\dash & \\dash & 1 & "
          f"{case['ring_sum'][0]:.3f} & \\dash & {case['cones'][0]:.4f} & "
          f"{case['exact'][0]:.4f} \\\\")
    for i in range(len(APERTURES) - 1):
        ex = (f"{case['exact'][i + 1]:.4f}" if case["unclipped"][i + 1] else "\\dash")
        print(f"{i + 1} & ${APERTURES[i]:.0f}^\\circ$ to ${APERTURES[i + 1]:.0f}^\\circ$ & "
              f"{case['omega'][i]:.3f} & {case['n_nom'][i]:.1f} & "
              f"{int(case['ring_valid'][i + 1])} & {case['ring_sum'][i + 1]:.3f} & "
              f"{case['w'][i]:.4f} & {case['cones'][i + 1]:.4f} & {ex} \\\\")

    print(f"\n% R = ({case['r'][0]:.3f}, {case['r'][1]:.3f}, {case['r'][2]:.3f}), "
          f"{np.degrees(np.arccos(case['r'][2])):.1f} deg from the normal")
    print(f"% W_i constant? min {case['w'].min():.6f} max {case['w'].max():.6f} "
          f"(2*pi/S = {2 * np.pi / S:.6f})")
    good = case["unclipped"][1:]
    if good.any():
        err = np.abs(case["cones"][1:][good] - case["exact"][1:][good])
        print(f"% gap from the closed form on the untruncated cones: max {err.max():.4f}")


def print_weights(case: dict) -> None:
    """The numbers the caption of cone_weights.png quotes, printed and not hand-written."""
    print("\n% weights of the two allocations (figure cone_weights)")
    print("% ring & Omega_i & N_i expected & n_i real & W_i shared & N_i aimed & W_i aimed")
    for i in range(len(APERTURES) - 1):
        print(f"{i + 1} & {case['omega'][i]:.3f} & {case['n_nom'][i]:.2f} & "
              f"{int(case['ring_valid'][i + 1])} & {case['w_shared']:.4f} & "
              f"{int(case['n_aimed'][i])} & {case['w_draw_aimed'][i]:.4f} \\\\")
    wa = case["w_draw_aimed"]
    print(f"% shared: W = 2*pi/S = {case['w_shared']:.4f} on every ring")
    print(f"% aimed:  W between {wa.min():.4f} and {wa.max():.4f}, "
          f"ratio {wa.max() / wa.min():.2f}")
    inside = case["ring"] < len(APERTURES) - 1
    print(f"% rays inside the widest cone: {int(inside.sum())} shared "
          f"(out of S = {S} over the hemisphere), {int(case['n_aimed'].sum())} aimed")

    # The Voronoi cells are not all equal and the caption says so: here is the number.
    a = case["cell_area"] / case["w_shared"]
    print(f"% Voronoi cells (in units of 2*pi/S): sum {case['cell_area'].sum():.5f} "
          f"= 2*pi ({2 * np.pi:.5f}), median {np.median(a):.3f}, "
          f"quartiles {np.percentile(a, 25):.3f}/{np.percentile(a, 75):.3f}, "
          f"min {a.min():.3f} (at the horizon) max {a.max():.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    case = build_case()
    fig_geometry(case, out / "cone_geometry.png")
    fig_rings(case, out / "cone_rings.png")
    fig_weights(case, out / "cone_weights.png")
    print_table(case)
    print_weights(case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
