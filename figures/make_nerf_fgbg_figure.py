#!/usr/bin/env python
"""make_nerf_fgbg_figure.py -- what training does with a ray that hits and one that does not.

    python make_nerf_fgbg_figure.py --out ../Doc/images/diagrams

Writes `nerf_fg_bg_rays.png`, the figure of Section 3.4.2.

The text says background rays are "treated as a purely directional background: each such
ray is evaluated against a large sphere enclosing the scene, so the environment becomes a
function of direction alone, with no parallax between views".  Every word of that is
carried by ONE operation, and it is the one the figure has to show: the background ray is
RE-ANCHORED AT THE WORLD ORIGIN.  Its origin is discarded and only its direction survives,
which is exactly why the environment cannot have parallax, and it is also what later lets
the whole background be baked into an equirectangular skybox by pointing `render_bg` at a
grid of directions (Section 3.3.7).  Drawn as two independent branches the figure would
say "two cases handled separately", which is the thing to avoid: `render_unified` is a
single `torch.where` over three quantities, and there is one network and one loss.

Panel (a) is TO SCALE, but the frame does not hold the whole sphere: at 5 x scene_radius
the shell encloses twenty-five times the area of the geometry, so a view containing all of
it makes the object a speck in an empty circle and leaves no room for the labels.  The
frame is cropped instead, and the shell enters it as an arc.  Nothing is distorted, only
framed: distances on the page are still proportional to distances in the scene, which is
what lets the arc's own label state the ratio without qualification.

Panel (b) has to be a magnification, and says so: the sampling window is +-0.05 against a
sphere radius of about 5, i.e. one percent of panel (a).  Its subject is that the two
branches have the SAME structure -- same number of samples, same window width, same
compositing -- and differ only in where the window is centred and in what the ray's origin
is.  The slab itself is Figure 3.17's subject and is not re-explained here.

The numbers are the operational ones (images_generator.py, the __main__ block), not the
NerfConfig defaults, which are different and are not what the thesis trained with.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update({"font.size": 13})

DPI = 190

# ── The operational configuration (images_generator.py, __main__) ────────────
# NOT the NerfConfig defaults: those are bg_radius_mult 6.0 and windows of 0.5, and the
# thesis did not train with them.  Same convention as make_nerf_sampling_figure.py, which
# hard-codes DEPTH_WINDOW and N_GUIDED from the same block.
BG_RADIUS_MULT = 5.0
DEPTH_WINDOW = 0.05          # foreground, both sides
BG_DEPTH_WINDOW = 0.05       # background, both sides
N_SAMPLES = 5                # depth_window_samples, shared by the two branches
DEPTH_EPS = 1e-6             # in_mask = depth > 1e-6 (dataset.py)

# ── Scene layout of panel (a), in scene_radius units ─────────────────────────
SCENE_RADIUS = 1.0
CAM = np.array([-2.70, -1.70])
MESH_C = np.array([0.0, 0.0])            # the blob standing in for the geometry
# Direction of the ray that misses.  Chosen so it clears the geometry by a wide margin
# (`build` asserts it) and so that, once re-anchored at the origin, the two copies stay
# far enough apart to read as two rays rather than as one thick one.
D_BG_DEG = 60.0
BG_FADE_LEN = 4.60                       # how far the discarded copy is drawn

C_GEOM = "#b9c3cd"
C_EDGE = "#5f6c79"
C_CAM = "#33404d"
C_FG = "#2ca02c"
C_BG = "#8452c9"
C_SPHERE = "#7c8894"
C_INK = "#222222"
HALO = dict(fc="white", ec="none", alpha=0.85, pad=1.2)


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def blob(center: np.ndarray, r: float, n: int = 240) -> np.ndarray:
    """A closed lumpy outline standing in for the mesh.  Deliberately not a circle: a
    circle at the centre of a circle reads as a second sphere rather than as geometry."""
    a = np.linspace(0, 2 * np.pi, n)
    rr = r * (1.0 + 0.20 * np.cos(3 * a + 0.7) + 0.09 * np.cos(5 * a - 1.1))
    return center[None, :] + np.stack([rr * np.cos(a), rr * np.sin(a)], axis=-1)


def ray_blob_hit(o: np.ndarray, d: np.ndarray, poly: np.ndarray) -> float | None:
    """First intersection distance of the ray with the closed polyline, or None.

    Computed rather than placed by hand: the surface point has to be where the drawn
    outline actually is, otherwise the figure shows a t_hit that does not belong to the
    geometry it also draws.
    """
    best = None
    for k in range(len(poly) - 1):
        a, b = poly[k], poly[k + 1]
        e = b - a
        den = d[0] * (-e[1]) - d[1] * (-e[0])
        if abs(den) < 1e-12:
            continue
        rhs = a - o
        t = (rhs[0] * (-e[1]) - rhs[1] * (-e[0])) / den
        u = (d[0] * rhs[1] - d[1] * rhs[0]) / den
        if t > 1e-9 and 0.0 <= u <= 1.0 and (best is None or t < best):
            best = t
    return best


def arrow(ax, p0, p1, *, color, lw=1.8, ls="-", zorder=4, head=0.20):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=f"-|>,head_width={head},head_length={head * 2.1}",
        connectionstyle="arc3,rad=0.0", color=color, lw=lw, linestyle=ls,
        shrinkA=0, shrinkB=0, zorder=zorder))


def camera_glyph(ax, p, look, *, size=0.30):
    """A small camera body pointing along `look`."""
    d = unit(look)
    n = np.array([-d[1], d[0]])
    body = np.array([p - n * size * 0.5, p + n * size * 0.5,
                     p + d * size * 0.9 + n * size * 0.5,
                     p + d * size * 0.9 - n * size * 0.5])
    ax.fill(body[:, 0], body[:, 1], color=C_CAM, zorder=6)


def build() -> dict:
    """Geometry of panel (a), with the two rays and the hit resolved on the outline."""
    r_sphere = BG_RADIUS_MULT * SCENE_RADIUS
    # 0.76 and not 0.84: `blob` adds up to 29% on top of its nominal radius, and the
    # assert in `report` is on the reach, which is what "scene radius" means.
    poly = blob(MESH_C, 0.76)

    # The foreground ray is aimed at the blob; the background one passes clear of it.
    d_fg = unit(MESH_C + np.array([-0.10, -0.06]) - CAM)
    t_hit = ray_blob_hit(CAM, d_fg, poly)
    assert t_hit is not None, "the foreground ray misses the geometry it is aimed at"

    a = np.radians(D_BG_DEG)
    d_bg = np.array([np.cos(a), np.sin(a)])
    assert ray_blob_hit(CAM, d_bg, poly) is None, \
        "the background ray hits the geometry: it would not take the background branch"

    return dict(r_sphere=r_sphere, poly=poly, d_fg=d_fg, t_hit=t_hit, d_bg=d_bg,
                p_hit=CAM + d_fg * t_hit)


def panel_scene(ax, case: dict) -> None:
    r = case["r_sphere"]
    d_bg = case["d_bg"]

    # Frame first: the labels are placed against these limits, and the shell is drawn as
    # a full circle that the limits then crop to the arc crossing the upper right.
    x0, x1 = -3.45, 3.65
    y0, y1 = -2.45, 4.80
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")

    a = np.linspace(0, 2 * np.pi, 720)
    ax.plot(r * np.cos(a), r * np.sin(a), color=C_SPHERE, lw=1.7, ls=(0, (7, 4)),
            zorder=1)
    # Horizontal, in the empty wedge the arc cuts off at the top left.  Set along the arc
    # it came out at 62 degrees, which is a rotation a reader has to tilt their head for,
    # and it crossed the ray that ends on the arc.
    ax.text(x0 + 0.18, y1 - 0.30, "background sphere",
            ha="left", va="top", color=C_SPHERE, fontsize=11.5, zorder=7)

    ax.fill(case["poly"][:, 0], case["poly"][:, 1], color=C_GEOM, ec=C_EDGE, lw=1.3,
            zorder=2)
    ax.text(MESH_C[0] + 1.05, MESH_C[1] - 0.70, "scene geometry", ha="left", va="top",
            color=C_EDGE, fontsize=11.5)

    # World origin: the anchor the background branch substitutes for the camera.  Marked
    # because it is a specific point and not "the middle of the object": it is the world
    # origin, fixed by convention so the bake is reproducible.
    ax.plot([0], [0], marker="+", ms=15, mew=2.2, color=C_INK, zorder=7)
    ax.text(0.20, -0.16, "world origin", ha="left", va="top", color=C_INK,
            fontsize=11.5, bbox=HALO, zorder=8)

    camera_glyph(ax, CAM, case["d_fg"], size=0.36)
    # Just "camera": that this ray keeps it as its origin is shown, not told, by the ray
    # leaving from here while the other one leaves from the marked origin.
    ax.text(CAM[0] - 0.02, CAM[1] - 0.32, "camera", ha="center", va="top", color=C_CAM,
            fontsize=12)

    # ── the ray that hits ──
    p_hit = case["p_hit"]
    arrow(ax, CAM, p_hit, color=C_FG)
    ax.plot([p_hit[0]], [p_hit[1]], marker="o", ms=8, color=C_FG, zorder=6)
    d = case["d_fg"]
    ang = np.degrees(np.arctan2(d[1], d[0]))
    # One short line, centred on the ray and offset perpendicular to it.  Anything longer
    # ran back past the camera glyph, because a centred rotated label grows in BOTH
    # directions along the ray and the ray is only 2.3 units long; what the origin is
    # gets said next to the camera instead.
    # The bare symbol: "from the depth pass" is in the caption, and spelled out here it
    # was long enough to run back over the camera glyph whatever the anchor.
    off = np.array([d[1], -d[0]]) * 0.30
    mid = CAM + d * case["t_hit"] * 0.55 + off
    ax.text(mid[0], mid[1], r"$t_{hit}$", ha="center", va="top", color=C_FG,
            fontsize=12, rotation=ang, rotation_mode="anchor", bbox=HALO, zorder=7)

    # ── the ray that misses ──
    # Faded from the camera, then drawn again solid from the origin, parallel: the
    # re-anchoring is the whole content of the panel, so the two segments have to be
    # visibly the same direction and visibly different origins.
    exit_pt = CAM + d_bg * BG_FADE_LEN
    ax.plot([CAM[0], exit_pt[0]], [CAM[1], exit_pt[1]], color=C_BG, lw=1.7,
            ls=(0, (3, 3)), alpha=0.50, zorder=3)
    ax.text(exit_pt[0] - 0.16, exit_pt[1] + 0.10,
            "origin discarded", ha="right", va="bottom", color=C_BG,
            fontsize=11, alpha=0.9, bbox=HALO, zorder=7)

    arrow(ax, np.zeros(2), d_bg * r, color=C_BG)
    ax.plot([(d_bg * r)[0]], [(d_bg * r)[1]], marker="o", ms=8, color=C_BG, zorder=6)
    lab = d_bg * r * 0.58
    # Short tags, not sentences: the prose belongs in the caption, and when it was in here
    # it set the width of the whole figure and pushed every label down to five points on
    # the printed page.
    ax.text(lab[0] + 0.20, lab[1],
            "same direction,\nre-anchored\nat the origin\n"
            r"$t_{hit} = R$",
            ha="left", va="center", color=C_BG, fontsize=11, linespacing=1.3,
            bbox=HALO, zorder=7)

    ax.axis("off")


def panel_window(ax, case: dict) -> None:
    """The two sampling windows side by side, magnified."""
    rows = [
        dict(y=1.0, color=C_FG, center=r"$t_{hit}$",
             head="on the geometry", origin="from the camera",
             win=DEPTH_WINDOW),
        # Well below the first row: at y = 0 this row's heading sat on the line above's
        # "from the camera", since each row now carries a label on both sides of its axis.
        dict(y=-0.45, color=C_BG, center="$R$",
             head="to the skybox", origin="from the world origin",
             win=BG_DEPTH_WINDOW),
    ]
    half = 1.0                       # the window is drawn one unit each side
    for r in rows:
        y = r["y"]
        ax.annotate("", xy=(half + 0.72, y), xytext=(-half - 0.85, y),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.16,head_length=0.42",
                                    color=r["color"], lw=1.7))
        ax.axvspan(-half, half, ymin=0.0, ymax=1.0, alpha=0.0)   # keep limits honest
        ax.fill_between([-half, half], y - 0.145, y + 0.145, color=r["color"],
                        alpha=0.14, linewidth=0)
        for e in (-half, half):
            ax.plot([e, e], [y - 0.145, y + 0.145], color=r["color"], lw=1.2)
        xs = np.linspace(-half, half, N_SAMPLES)
        ax.plot(xs, np.full(N_SAMPLES, y), ls="none", marker="o", ms=8,
                color=r["color"], zorder=4)
        ax.plot([0, 0], [y - 0.30, y + 0.30], color=r["color"], lw=1.3,
                ls=(0, (4, 2.5)))
        ax.text(0.0, y + 0.36, r["center"], ha="center", va="bottom", color=r["color"],
                fontsize=12)
        # Clear of the centre mark: on the same line the two ran into each other, since
        # the head label is left-aligned from outside the axis and reaches the middle.
        ax.text(-half - 0.95, y + 0.62, r["head"], ha="left", va="bottom",
                color=r["color"], fontsize=11.5)
        ax.text(-half - 0.95, y - 0.34, r["origin"], ha="left", va="top",
                color="0.42", fontsize=11)
        ax.text(half + 0.80, y, "$t$", ha="left", va="center", color=r["color"],
                fontsize=12)

    # The one measurement the panel is making: both windows are the same width and hold
    # the same number of samples.  Written once, under the pair, because it is a property
    # of the pair and not of either row.
    ax.annotate("", xy=(half, -1.07), xytext=(-half, -1.07),
                arrowprops=dict(arrowstyle="<|-|>,head_width=0.14,head_length=0.34",
                                color="0.42", lw=1.2))
    ax.text(0.0, -1.17,
            rf"both: {N_SAMPLES} samples, $\pm{DEPTH_WINDOW:g}$",
            ha="center", va="top", color="0.42", fontsize=11.5)

    ax.set_xlim(-half - 1.05, half + 1.35)
    ax.set_ylim(-1.70, 1.95)
    ax.axis("off")


def figure(case: dict, out: Path) -> None:
    # SIZED FOR THE PAGE, NOT FOR THE SCREEN.  The figure goes in at \linewidth, about
    # 5.7 in, so a label ends up on paper at its point size times 5.7 / (figure width in
    # inches).  At the 12.4 in this started from, 12 pt text arrived at under 5 pt and was
    # illegible in print while looking perfectly fine on screen.  Shrinking the figure
    # with the font sizes left alone is the whole lever: the drawing scales with the axes
    # and the text does not, so the text grows relative to it.
    fig = plt.figure(figsize=(6.8, 3.10))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.62, 1.0], wspace=0.04,
                          left=0.01, right=0.99, top=0.99, bottom=0.06)

    ax_a = fig.add_subplot(gs[0])
    panel_scene(ax_a, case)
    ax_a.set_title("(a)  the two rays, to scale", fontsize=12, color=C_INK, pad=2,
                   loc="left")

    ax_b = fig.add_subplot(gs[1])
    panel_window(ax_b, case)
    ax_b.set_title("(b)  the samples, magnified", fontsize=12,
                   color=C_INK, pad=2, loc="left")

    path = out / "nerf_fg_bg_rays.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  + {path}")


def report(case: dict) -> None:
    """Checks on the two claims the figure makes that could go stale."""
    print(f"\n  background sphere radius = {BG_RADIUS_MULT:g} x scene radius "
          f"= {case['r_sphere']:.2f}")
    print(f"  foreground t_hit on the drawn outline = {case['t_hit']:.3f}")
    print(f"  windows: fg +-{DEPTH_WINDOW:g}, bg +-{BG_DEPTH_WINDOW:g}; "
          f"{N_SAMPLES} samples each")
    print(f"  panel (b) magnification vs panel (a) = "
          f"{case['r_sphere'] / DEPTH_WINDOW:.0f}x the window half-width")

    # Panel (b) states in so many words that the two windows match.  If the operational
    # configuration ever splits them, the panel becomes false and this stops the run.
    assert DEPTH_WINDOW == BG_DEPTH_WINDOW, \
        f"the two windows differ ({DEPTH_WINDOW} against {BG_DEPTH_WINDOW}) and panel " \
        "(b) claims they are the same"
    # The shell has to enclose the geometry, which is what train.py itself enforces.
    assert BG_RADIUS_MULT > 1.0, \
        "the background sphere would cut through the geometry"
    # The blob has to fit inside the scene radius, or panel (a) is not to scale.
    reach = float(np.max(np.linalg.norm(case["poly"], axis=1)))
    assert reach <= SCENE_RADIUS + 1e-9, \
        f"the drawn geometry reaches {reach:.2f} against a scene radius of " \
        f"{SCENE_RADIUS:g}: panel (a) is no longer to scale"
    print(f"  drawn geometry reaches {reach:.2f} of the {SCENE_RADIUS:g} scene radius")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="../Doc/images/diagrams")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    case = build()
    figure(case, out)
    report(case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
