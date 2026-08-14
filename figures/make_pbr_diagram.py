"""Generate the illustrative diagram of the multi-view cone PBR approach:

    C_j = X * D + (1 - X) * L_j(r)

Four panels: (A) diffuse term, (B) specular cone with hit/miss,
(C) multi-camera system, (D) scan over r + closed form for X.

NOTE: this is the older diagram. It illustrates the pre-2026-07-16 model
(D approximated by color_min); the current model is the one drawn by
make_pbr_model_diagram.py.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Rectangle, FancyArrowPatch, Circle

P = np.array([5.0, 0.0])  # texel
SKY_R = 5.0

C_SKY0 = np.array([0.45, 0.65, 0.95])   # light blue
C_SUN = np.array([1.00, 0.78, 0.25])    # warm
C_HIT = "#d62728"
C_MISS = "#1f77b4"
C_CONE = "#9467bd"
C_DIFF = "#e8a33d"

OCCL = (7.6, 1.2, 1.1, 1.5)  # x, y, w, h


def deg2dir(a):
    a = np.radians(a)
    return np.array([np.cos(a), np.sin(a)])


def ray_hits_box(p, d, box, tmax=8.0):
    x0, y0, w, h = box
    for t in np.linspace(0.05, tmax, 400):
        q = p + t * d
        if x0 <= q[0] <= x0 + w and y0 <= q[1] <= y0 + h:
            return t
    return None


def draw_surface(ax):
    ax.plot([0.6, 9.4], [0, 0], color="0.25", lw=3, zorder=3)
    for x in np.arange(0.8, 9.4, 0.45):
        ax.plot([x, x - 0.3], [0, -0.35], color="0.55", lw=1)
    ax.plot(*P, marker="s", ms=7, color="black", zorder=6)
    ax.annotate("texel", P + (-0.05, -0.62), ha="center", fontsize=9)


def draw_normal(ax):
    ax.add_patch(FancyArrowPatch(P, P + (0, 2.0), arrowstyle="-|>",
                                 mutation_scale=14, color="0.2", lw=1.6, zorder=5))
    ax.annotate("n", P + (0.14, 2.0), fontsize=11, style="italic")


def draw_sky(ax, sun_deg=115):
    for a in np.arange(8, 172, 4):
        w = np.exp(-((a + 2 - sun_deg) / 22.0) ** 2)
        col = np.clip((1 - w) * C_SKY0 + w * C_SUN, 0, 1)
        ax.add_patch(Wedge(P, SKY_R + 0.4, a, a + 4.2, width=0.38,
                           color=col, lw=0, zorder=1))
    s = P + (SKY_R + 0.21) * deg2dir(sun_deg)
    ax.add_patch(Circle(s, 0.16, color=C_SUN, zorder=2))
    ax.annotate("envmap (skybox)", P + (SKY_R + 0.75) * deg2dir(90),
                ha="center", fontsize=9, color="0.35")


def draw_occluder(ax):
    x0, y0, w, h = OCCL
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor="0.75",
                           edgecolor="0.3", lw=1.2, zorder=4))
    ax.annotate("geometry\n(scene)", (x0 + w / 2, y0 + h + 0.12),
                ha="center", va="bottom", fontsize=8, color="0.3")


def setup(ax, title):
    ax.set_xlim(-0.3, 10.3)
    ax.set_ylim(-1.1, 7.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11, pad=6)


# ----------------------------------------------------------------- Panel A
def panel_diffuse(ax):
    setup(ax, "A — Diffuse term  $D$  (view-independent, already in the pipeline)")
    draw_sky(ax)
    draw_occluder(ax)
    draw_surface(ax)
    draw_normal(ax)
    for a in np.arange(18, 168, 15):
        d = deg2dir(a)
        t = ray_hits_box(P, d, OCCL)
        if t is not None:
            q = P + t * d
            ax.plot([P[0], q[0]], [P[1], q[1]], color=C_HIT, lw=1.3,
                    ls="--", zorder=2)
            ax.plot(*q, marker="x", ms=7, color=C_HIT, mew=2, zorder=5)
        else:
            q = P + (SKY_R - 0.05) * d
            ax.plot([P[0], q[0]], [P[1], q[1]], color=C_DIFF, lw=1.3, zorder=2)
    ax.annotate("miss → Irradiance_Generator\n(envmap + shadow ray)",
                (1.0, 4.6), fontsize=8.5, color=C_DIFF)
    ax.annotate("hit → Indirect_Generator\n(query NeRF)",
                (7.0, 0.55), fontsize=8.5, color=C_HIT, ha="left")
    ax.annotate(r"$D = d \cdot E$,   $E=\int L\,\cos\theta\, d\omega$"
                "   over the whole hemisphere",
                (5.0, 6.45), ha="center", fontsize=10)
    ax.annotate(r"in practice:  $D \approx$ color_min  (minimum across cameras)",
                (5.0, 5.9), ha="center", fontsize=8.5, color="0.35")


# ----------------------------------------------------------------- Panel B
def panel_cone(ax):
    setup(ax, "B — Specular term  $L_j(r)$ :  cone around the reflected ray")
    draw_sky(ax)
    draw_occluder(ax)
    draw_surface(ax)
    draw_normal(ax)

    cam = np.array([1.4, 3.6])
    ax.add_patch(Rectangle(cam - (0.28, 0.2), 0.56, 0.4, facecolor="0.2", zorder=5))
    ax.annotate("camera $j$", cam + (0, 0.42), ha="center", fontsize=9)
    ax.add_patch(FancyArrowPatch(cam, P + 0.12 * (cam - P), arrowstyle="-|>",
                                 mutation_scale=12, color="0.2", lw=1.4, zorder=4))
    ax.annotate("$v_j$", (2.9, 2.2), fontsize=10, style="italic")

    refl = 45.0  # reflected direction (mirror of v_j about n)
    half1, half2 = 13.0, 34.0
    ax.add_patch(Wedge(P, 4.4, refl - half1, refl + half1, color=C_CONE,
                       alpha=0.30, lw=0, zorder=2))
    ax.add_patch(Wedge(P, 4.4, refl - half2, refl + half2, facecolor="none",
                       edgecolor=C_CONE, ls="--", lw=1.4, zorder=2))
    Rj = P + 4.4 * deg2dir(refl)
    ax.add_patch(FancyArrowPatch(P, Rj, arrowstyle="-|>", mutation_scale=13,
                                 color=C_CONE, lw=2.0, zorder=4))
    ax.annotate("$R_j$", Rj + (0.15, 0.1), fontsize=11, color=C_CONE)
    ax.annotate("small $r$", P + 2.45 * deg2dir(refl + half1 + 4),
                fontsize=8.5, color=C_CONE, rotation=refl)
    ax.annotate("large $r$", P + 4.05 * deg2dir(refl + half2 + 5),
                fontsize=8.5, color=C_CONE, rotation=refl + 18)

    for a in (refl - 9, refl - 4.5, refl, refl + 4.5, refl + 9):
        d = deg2dir(a)
        t = ray_hits_box(P, d, OCCL)
        if t is not None:
            q = P + t * d
            ax.plot([P[0], q[0]], [P[1], q[1]], color=C_HIT, lw=1.1,
                    ls="--", zorder=3)
            ax.plot(*q, marker="x", ms=6, color=C_HIT, mew=2, zorder=5)
        else:
            q = P + (SKY_R - 0.05) * d
            ax.plot([P[0], q[0]], [P[1], q[1]], color=C_MISS, lw=1.1, zorder=3)

    ax.annotate("miss → envmap", (5.6, 5.0), fontsize=8.5, color=C_MISS)
    ax.annotate("hit → query NeRF", (8.2, 0.5), fontsize=8.5, color=C_HIT)
    ax.annotate(r"$L_j(r)$ = mean radiance over the cone"
                "   (r = aperture, 0°–180°)",
                (5.0, 6.45), ha="center", fontsize=10)
    ax.annotate("r = 0 → single mirror ray (sharp reflection);"
                "  large r → blurred reflection",
                (5.0, 5.9), ha="center", fontsize=8.5, color="0.35")


# ----------------------------------------------------------------- Panel C
def panel_multicam(ax):
    setup(ax, "C — Multi-view:  same $D$, $X$, $r$  —  $L_j$ changes with the camera")
    draw_surface(ax)
    draw_normal(ax)

    cams = [((1.3, 3.0), "tab:blue"), ((4.2, 5.2), "tab:green"),
            ((8.8, 2.9), "tab:red")]
    for k, (c, col) in enumerate(cams, start=1):
        c = np.array(c)
        ax.add_patch(Rectangle(c - (0.25, 0.18), 0.5, 0.36, facecolor=col, zorder=5))
        ax.annotate(f"cam {k}", c + (0, 0.38), ha="center", fontsize=8.5, color=col)
        ax.add_patch(FancyArrowPatch(c, P + 0.10 * (c - P), arrowstyle="-|>",
                                     mutation_scale=10, color=col, lw=1.3, zorder=4))
        d_in = P - c
        refl = np.degrees(np.arctan2(-d_in[1], d_in[0]))
        ax.add_patch(Wedge(P, 3.0, refl - 10, refl + 10, color=col,
                           alpha=0.28, lw=0, zorder=2))
        q = P + 3.1 * deg2dir(refl)
        ax.annotate(f"$L_{k}(r)$", q + (0.1, 0.05), fontsize=9, color=col)

    ax.annotate(r"$C_1 = X\,D + (1-X)\,L_1(r)$" "\n"
                r"$C_2 = X\,D + (1-X)\,L_2(r)$" "\n"
                r"$C_3 = X\,D + (1-X)\,L_3(r)$",
                (0.4, 5.4), fontsize=10.5, va="top",
                bbox=dict(boxstyle="round,pad=0.45", fc="#f5f0e8", ec="0.6"))
    ax.annotate("3 cameras × 3 RGB channels = 9 equations,\n2 unknowns (X, r)"
                " → heavily overdetermined",
                (9.8, 5.9), fontsize=8.5, ha="right", color="0.3")


# ----------------------------------------------------------------- Panel D
def panel_solve(ax):
    ax.set_title("D — Solution:  scan over $r$,  $X$ in closed form", fontsize=11, pad=6)
    r = np.linspace(0, 180, 400)
    res = 0.05 + 0.40 * ((r - 55) / 110) ** 2 + 0.22 * np.exp(-r / 14)
    ax.plot(r, res, color="0.25", lw=1.8)
    grid = np.array([0, 10, 25, 50, 90, 130, 180])
    resg = 0.05 + 0.40 * ((grid - 55) / 110) ** 2 + 0.22 * np.exp(-grid / 14)
    ax.plot(grid, resg, "o", color=C_CONE, ms=7, zorder=5)
    k = np.argmin(resg)
    ax.axvline(grid[k], color=C_CONE, ls="--", lw=1.2)
    ax.annotate("$r^*$ (minimum residual)", (grid[k] + 5, resg[k] + 0.015),
                fontsize=9.5, color=C_CONE)
    ax.annotate("for every candidate $r$:", (98, 0.275), fontsize=9.5)
    ax.annotate(r"$X^*(r)=\dfrac{\sum_{j,c}(C-L)(D-L)}{\sum_{j,c}(D-L)^2}$"
                r"$\;\in[0,1]$",
                (98, 0.255), fontsize=11, va="top",
                bbox=dict(boxstyle="round,pad=0.45", fc="#f5f0e8", ec="0.6"))
    ax.set_xlabel("cone aperture  $r$  (degrees)", fontsize=10)
    ax.set_ylabel("fit residual", fontsize=10)
    ax.set_xlim(-5, 185)
    ax.set_ylim(0.0, 0.33)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    fig, axs = plt.subplots(2, 2, figsize=(14.5, 10.5))
    panel_diffuse(axs[0, 0])
    panel_cone(axs[0, 1])
    panel_multicam(axs[1, 0])
    panel_solve(axs[1, 1])
    fig.suptitle(r"Multi-view PBR estimation:   $C_j = X \cdot D + (1-X)\cdot L_j(r)$"
                 "    —    $X$ = metallic (closed form),  $r$ = cone aperture (scan)",
                 fontsize=13.5, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = "pbr_cone_approach.png"
    fig.savefig(out, dpi=140)
    print("saved:", out)


if __name__ == "__main__":
    main()
