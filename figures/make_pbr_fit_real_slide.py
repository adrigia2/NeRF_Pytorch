#!/usr/bin/env python
"""make_pbr_fit_real_slide.py -- where the metallic comes from, on two real texels.

    python make_pbr_fit_real_slide.py --run <run dir> --out ../../PresentationImages

Writes `metallic_from_slope.png`: two columns, one texel each, both read from the run's own
artefacts.  Left the steel of the cube, right the top of the wooden table.

THE SLIDE MAKES ONE CLAIM, and it is made CAMERA BY CAMERA.  The top panel is what the
cone around each camera's reflected ray collects; the one below it is what that same
camera recorded, against what the fitted model says it should have.  On the steel the
environment swings by 9214x and the recorded colour follows it, 4420x.  On the wood the
environment swings by 29x and the colour does not move: 1.18x, three flat bands.  That
gap between the two factors IS the metallic, because m = V_CL / V_LL is exactly the ratio
between how much the colour follows and how much the environment offers.

WHY NOT A SCATTER OF dC AGAINST dL.  It draws m as the angle of one line, which is
compact, but it pools 22 cameras into an anonymous cloud -- and the camera is the unit of
this measurement: each one is an equation, each one either sees the reflection or does
not.  Per camera, one also sees what the pooled view hides: on the steel three cameras
out of 22 (2, 7 and 34) carry the whole fit, because they are the ones whose reflected
ray lands on a light.

THE TWO PANELS OF A COLUMN SHARE A VERTICAL SPAN, in decades (`_log_window`).  Autoscaled,
the wood's flat colour would fill its panel and be drawn as a mountain range, saying the
opposite of what it means.  And the factors quoted are per channel, never over the
flattened array: see `camera_factor`.

Below them, the residual over the whole grid of candidates -- the quantity that actually
selects the aperture -- and the slope candidate by candidate, which cannot select it: on
the steel it saturates at 1 across four candidates where the residual varies by 6x.

m AND r DO NOT PROMISE THE SAME THING.  The metallic is determined on both texels, the
aperture only on the steel.  On the wood the residual is flat end to end, so the winning
aperture is where the noise fell, not where the surface is -- the slide says so by drawing
that minimum hollow and greying the r it produces.  That is measured, not declared:
`determined()` compares the depth of the well against RATIO_GATE.

NOTHING HERE IS TYPED IN.  `read_texel` reads the same files the solver reads and keeps
the same cameras the solver keeps (`mask & visibility & n_valid > 0`); the fit is
`make_pbr_fit_figures.fit_moments`, whose docstring already ties it to pbr_solver.py; and
the model markers use the run's own intercept, not one re-derived here.  Before drawing,
`check_against_maps` compares metallic, roughness, lobe_param, residual, n_views and
diffuse_term against the maps the run already holds, and raises if they disagree.  A slide
that quoted a number the reconstruction did not produce would be worse than no slide.

The two texels are those of `make_texel_views_video.py`, chosen there on the run's own
statistics (distance from the UV seams, +Z normal on the wood, measured spread on the
steel) rather than by eye.

The authored Blender maps are deliberately NOT read.  This slide explains where the number
comes from, not how close it lands; mixing the two questions into one picture would answer
neither.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import _paths  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402

from pbr_solver import _ExrBandReader, _read_band, read_cones          # noqa: E402
from make_pbr_fit_figures import CH_COLORS, fit_moments                # noqa: E402
from make_geometry_diagrams import C_CAM, C_EDGE                       # noqa: E402

# ── Palette: the slides', not the thesis figure's ────────────────────────────
C_INK   = C_CAM
C_MUTED = C_EDGE
C_SPEC  = "#2171b5"        # the specular term, as in make_full_model_slide.py
C_RESID = "#eb6834"        # what the fit leaves behind
C_GRID  = "#dcdcdc"
C_DEAD  = "#9aa3ab"        # a number the data does not determine
C_WIN   = "#eef4fb"        # ground of the winning panel

# (heading, subheading, IUM texel).
COLUMNS = [("Steel", "the cube",      (648, 1335)),
           ("Wood",  "the table top", (3090, 3752))]

# Depth of the residual well, below which the aperture is not considered determined.
# The two regimes are far apart -- 328x on the steel against 1.9x on the wood -- so the
# gate only has to sit in the gap between them, and nothing on this slide is near it.
RATIO_GATE = 10.0

DPI = 200
FIGSIZE = (13.34, 10.0)    # 4:3


# ──────────────────────────────────────────────────────────────────────────────
# Reading one texel out of a run
# ──────────────────────────────────────────────────────────────────────────────

def read_texel(run: Path, y: int, x: int, source: str = "gt") -> dict:
    """Everything the fit needs at one texel: C_j, L_j(all candidates), the cameras.

    One scanline per file rather than the whole image: the cone atlas of a single camera
    is 2.6 GiB at 4096 and sixty of them are wanted for two numbers.
    """
    with open(run / "spec_cone" / "spec_cone_meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    if meta.get("format") != "cones":
        raise SystemExit(f"{run}: spec_cone is in format {meta.get('format')!r}; this "
                         f"slide needs the 'cones' format the solver reads")
    with open(run / "transforms_extended.json", encoding="utf-8") as fh:
        tjson = json.load(fh)

    apertures = np.asarray(meta["apertures_deg"], dtype=np.float32)
    cams = meta["cameras"]
    stems = [Path(f["file_path"]).stem for f in tjson["frames"]]

    if not _read_band(run / "ium" / "ium_masks.exr", y, 1)[x] > 0.5:
        raise SystemExit(f"texel ({y}, {x}) is outside the IUM mask: it is not a surface "
                         f"point, and there is nothing to fit there")

    # Column j of the visibility is camera j, but the channel NAME depends on how many
    # cameras there are (_channel_order): with 3 or 4 there are no Cam* channels at all.
    vis_path = run / "visibility" / "visibility.exr"
    with _ExrBandReader(vis_path) as rd:
        vis_cols = [rd.names[j] for j in cams]
    vis = _read_band(vis_path, y, 1, vis_cols)[x] > 0.5

    C, L, kept = [], [], []
    for jj, j in enumerate(cams):
        cones, n_valid = read_cones(run / "spec_cone" / meta["cam_file_pattern"].format(cam=j),
                                    apertures, y, 1)
        # The solver's own w_j: seen by the camera and reached by at least one ray above
        # the horizon.
        if not (vis[jj] and n_valid[x] > 0):
            continue
        C.append(_read_band(run / "sources" / source / "camera_texture" / f"{stems[j]}.exr",
                            y, 1)[x])
        L.append(cones[x])
        kept.append(j)

    if len(kept) < 2:
        raise SystemExit(f"texel ({y}, {x}) is seen by {len(kept)} camera(s): the centred "
                         f"regression needs at least two")

    return dict(C=np.asarray(C, dtype=np.float64), L=np.asarray(L, dtype=np.float64),
                cams=kept, apertures=apertures, texel=(y, x))


def determined(res: np.ndarray) -> "tuple[bool, float]":
    """Is the aperture chosen by the data?  The depth of the well says so."""
    ratio = float(res.max() / max(res.min(), 1e-30))
    return ratio >= RATIO_GATE, ratio


def intercept(data: dict, fit: dict, k: int) -> np.ndarray:
    """alpha_c = x*a_c*E_c/pi, the diffuse radiance, exactly as the solver forms it.

    With unit weights `(SC - beta*SL)/Sw` is the mean of `C - beta*L` over the cameras,
    and the clamp at zero is the solver's (pbr_solver.py:~449): an unconstrained
    intercept can come out slightly negative on a texel whose colour is nearly all
    specular, as it does here on the steel's red channel.
    """
    m = float(fit["beta"][k])
    return np.maximum(data["C"].mean(0) - m * data["L"][:, k].mean(0), 0.0)


def check_against_maps(run: Path, data: dict, fit: dict, k: int, source: str = "gt") -> None:
    """The slide and the run's maps have to hold the same numbers.

    Same spirit as _check_against_table in make_pbr_fit_figures.py, against measurements
    instead of a table.  fit_moments reaches the same quantities by a different route than
    the solver (centre first, versus raw sums minus SL^2/Sw), so the tolerances are those
    of two float64 paths through a cancelling difference, not of a bit comparison.
    """
    y, x = data["texel"]
    src = run / "sources" / source
    got = {name: float(np.atleast_1d(_read_band(src / rel, y, 1)[x])[0]) for name, rel in (
        ("metallic",   Path("metallic/metallic.exr")),
        ("roughness",  Path("roughness/roughness.exr")),
        ("lobe_param", Path("pbr/lobe_param.exr")),
        ("residual",   Path("pbr/residual.exr")),
        ("n_views",    Path("pbr/n_views.exr")))}

    ap = float(data["apertures"][k])
    mine = dict(metallic=float(fit["beta"][k]), lobe_param=ap,
                residual=float(fit["res"][k]), n_views=float(len(data["cams"])))

    print(f"    check against the run's maps, texel ({y}, {x})")
    for name, tol in (("metallic", 1e-4), ("residual", 5e-3),
                      ("lobe_param", 0.0), ("n_views", 0.0)):
        a, b = mine[name], got[name]
        rel = abs(a - b) / max(abs(b), 1e-12)
        print(f"      {name:<11} {a:>12.6f}  map {b:>12.6f}   rel {rel:.2e}")
        if rel > tol:
            raise SystemExit(
                f"{name} disagrees with {src}: recomputed {a}, the map holds {b} "
                f"(relative {rel:.2e} > {tol:.0e}). The slide would quote a number the "
                f"reconstruction did not produce.")

    # roughness carries the spec_threshold gate: either the fitted aperture/180, or 1.0
    # when the texel was gated out as too diffuse. Both are consistent; anything else
    # means this is not the map that fit produced.
    fitted_r = ap / 180.0
    if not (abs(got["roughness"] - fitted_r) <= 1e-4 or abs(got["roughness"] - 1.0) <= 1e-6):
        raise SystemExit(f"roughness map holds {got['roughness']}, which is neither the "
                         f"fitted {fitted_r} nor the gated 1.0")
    gated = abs(got["roughness"] - 1.0) <= 1e-6 and abs(fitted_r - 1.0) > 1e-6
    print(f"      roughness   {fitted_r:>12.6f}  map {got['roughness']:>12.6f}"
          f"{'   (gated to 1.0 by spec_threshold)' if gated else ''}")

    # alpha is drawn, not just quoted: the model markers of the lower panel are
    # alpha + m*L_j, so it has to be the run's alpha and not one re-derived here.
    a_mine = intercept(data, fit, k)
    a_map = np.atleast_1d(_read_band(src / "pbr" / "diffuse_term.exr", y, 1)[x]).ravel()[:3]
    rel = float(np.max(np.abs(a_mine - a_map) / np.maximum(np.abs(a_map), 1e-12)))
    print(f"      diffuse_term {np.round(a_mine, 6)}  map {np.round(a_map, 6)}"
          f"   rel {rel:.2e}")
    if rel > 1e-4:
        raise SystemExit(f"the intercept disagrees with {src}/pbr/diffuse_term.exr: "
                         f"{a_mine} against {a_map}")


# ──────────────────────────────────────────────────────────────────────────────
# Panels
# ──────────────────────────────────────────────────────────────────────────────

def _frame(ax, winner: bool) -> None:
    ax.set_axisbelow(True)
    ax.grid(color=C_GRID, lw=0.7)
    ax.tick_params(length=0, labelsize=10, colors=C_MUTED)
    if winner:
        ax.set_facecolor(C_WIN)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(C_SPEC)
            s.set_linewidth(1.8)
    else:
        for name, s in ax.spines.items():
            s.set_visible(name in ("left", "bottom"))
            s.set_color(C_GRID)


# The three channels of one camera are drawn side by side inside that camera's slot, so
# a camera reads as one group of three and not as three loose points.
CH_DX = (-0.24, 0.0, 0.24)


def camera_factor(values: np.ndarray) -> float:
    """How much a quantity varies ACROSS CAMERAS, as a factor.

    Per channel and then the worst channel, never over the flattened array: the three
    channels sit at different levels (the wood reads 0.65 in red against 0.22 in blue),
    so a global max/min would report 3.28x on a texel whose every channel is flat to
    within 1.18x -- a number about the colour of the wood, printed where the reader is
    being told how much the cameras disagree.
    """
    return float(np.max(values.max(axis=0) / values.min(axis=0)))


def _log_window(ax, values, span: float) -> None:
    """Put `values` on a log axis whose height is exactly `span` decades.

    The reason this is not left to autoscaling: the wood's colour varies by 1.18x across
    cameras and its environment by 32x.  Autoscaled, both fill their panel and the flat
    one is drawn as a mountain range.  Giving the two panels of a column the same number
    of decades makes a 1.18x variation look like 1.18x, which is the whole claim of the
    picture.
    """
    lo, hi = float(values.min()), float(values.max())
    centre = np.sqrt(lo * hi)
    half = 10.0 ** (span / 2.0)
    ax.set_yscale("log")
    ax.set_ylim(centre / half, centre * half)


def _percam_frame(ax, cams, span: float, label: str, factor: float, note_colour) -> None:
    ax.set_xlim(-0.7, len(cams) - 0.3)
    ax.set_xticks(range(len(cams)))
    ax.set_xticklabels([str(c) for c in cams], fontsize=8.5)
    ax.set_ylabel(label, fontsize=12, color=C_MUTED)
    txt = f"{factor:.0f}×" if factor >= 10 else f"{factor:.2f}×"
    # On white: the dots reach the top of the panel on both texels, and a caption sitting
    # on a data point is worse than one sitting on the frame.
    ax.text(0.995, 0.96, txt + "  across cameras", transform=ax.transAxes,
            ha="right", va="top", fontsize=15, color=note_colour,
            zorder=12,
            bbox=dict(facecolor="white", edgecolor="none", pad=2.0, alpha=0.88))
    _frame(ax, False)
    ax.grid(axis="x", color=C_GRID, lw=0.5, alpha=0.6)


def panel_percam_env(ax, cams, L, span: float) -> float:
    """What the cone around each camera's reflected ray collects, camera by camera."""
    for c in range(3):
        ax.plot(np.arange(len(cams)) + CH_DX[c], L[:, c], marker="o", ms=5.5,
                color=CH_COLORS[c], mec="white", mew=0.8, ls="none", zorder=4 + c)
    _log_window(ax, L, span)
    factor = camera_factor(L)
    _percam_frame(ax, cams, span, "environment  $L_j$", factor, C_INK)
    ax.tick_params(labelbottom=False)
    return factor


def panel_percam_obs(ax, cams, C, pred, span: float) -> float:
    """What each camera recorded, against what the fitted model says it should have.

    Filled marker: the measurement.  Hollow: alpha + m*L_j, the model with the run's own
    alpha and m.  The segment between them is that camera's residual, so the quality of
    the fit is legible one camera at a time -- which the pooled scatter could not show.
    """
    xs = np.arange(len(cams))
    for c in range(3):
        x = xs + CH_DX[c]
        for i in range(len(cams)):
            ax.plot([x[i], x[i]], [C[i, c], pred[i, c]], color=C_RESID, lw=1.3,
                    zorder=3, solid_capstyle="butt")
        ax.plot(x, pred[:, c], marker="o", ms=7.0, color="white", mec=CH_COLORS[c],
                mew=1.6, ls="none", zorder=4 + c)
        ax.plot(x, C[:, c], marker="o", ms=5.5, color=CH_COLORS[c], mec="white",
                mew=0.8, ls="none", zorder=7 + c)
    _log_window(ax, np.concatenate([C, pred]), span)
    factor = camera_factor(C)
    _percam_frame(ax, cams, span, "recorded  $C_j$", factor, C_INK)
    ax.set_xlabel("camera", fontsize=12, color=C_MUTED, labelpad=2)
    return factor


def panel_residual(ax, apertures, res, k: int, ok: bool, ratio: float) -> None:
    ax.plot(apertures, res, color=C_RESID, lw=2.2, marker="o", ms=5.5,
            mec="white", mew=1.0, zorder=4)
    # Filled when the well selects the aperture, hollow when it does not: the whole
    # difference between the two columns, said without a sentence.
    ax.plot(apertures[k], res[k], marker="o", ms=13, zorder=6, mew=2.4,
            color=C_RESID if ok else "white", mec=C_RESID if ok else C_DEAD)
    ax.set_yscale("log")
    ax.set_ylim(res.min() * 0.5, res.max() * 3.2)
    ax.set_xlim(-6, 186)
    ax.set_ylabel("residual", fontsize=12, color=C_MUTED)
    # Top left: on both texels the curve is well below the top of the axes there, and
    # the headroom of set_ylim leaves the corner empty.
    ax.text(0.025, 0.94, (f"well {ratio:.0f}× deep" if ok
                          else f"flat: {ratio:.1f}× end to end"),
            transform=ax.transAxes, ha="left", va="top", fontsize=14,
            color=C_INK if ok else C_DEAD)
    _frame(ax, False)
    ax.tick_params(labelbottom=False)


def panel_beta(ax, apertures, beta, k: int) -> None:
    ax.plot(apertures, beta, color=C_INK, lw=2.0, marker="o", ms=4.5,
            mec="white", mew=0.9, zorder=4)
    ax.axhline(1.0, color=C_GRID, lw=1.0, ls=(0, (3, 3)))
    ax.plot(apertures[k], beta[k], marker="o", ms=11, color=C_SPEC, mec="white",
            mew=2.0, zorder=6)
    ax.set_ylim(-0.06, 1.16)
    ax.set_xlim(-6, 186)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylabel("slope", fontsize=12, color=C_MUTED)
    ax.set_xlabel("candidate aperture  [degrees]", fontsize=12, color=C_MUTED)
    ax.set_xticks([0, 45, 90, 135, 180])
    _frame(ax, False)


def panel_readout(ax, m: float, aperture: float, ok: bool) -> None:
    """The two numbers that reach the maps.  r borrows the idiom of full_model.png."""
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # Everything sits low in the row: the axis label of the panel above is drawn into the
    # gap between them, and at the top of this axes the two would collide.
    ax.text(0.02, 0.40, f"m = {m:.2f}", fontsize=36, color=C_SPEC, ha="left", va="center",
            fontweight="bold")
    col = C_INK if ok else C_DEAD
    ax.text(0.60, 0.55, f"r = {aperture / 180.0:.2f}", fontsize=22, color=col,
            ha="left", va="center")
    ax.text(0.60, 0.27, "mirror" if aperture == 0.0 else f"{aperture:g}°", fontsize=19,
            color=C_SPEC if ok else C_DEAD, ha="left", va="center")
    if not ok:
        ax.text(0.60, 0.03, "not determined", fontsize=14, color=C_DEAD,
                ha="left", va="center", style="italic")


# ──────────────────────────────────────────────────────────────────────────────
# The slide
# ──────────────────────────────────────────────────────────────────────────────

def build(run: Path, out: Path, columns, source: str) -> None:
    fig = plt.figure(figsize=FIGSIZE)
    outer = fig.add_gridspec(1, 2, wspace=0.20, left=0.055, right=0.985,
                             top=0.845, bottom=0.075)

    for ci, (head, sub, (y, x)) in enumerate(columns):
        print(f"  {head} ({sub}), texel ({y}, {x})")
        data = read_texel(run, y, x, source)
        fit = fit_moments(data["C"], data["L"])
        ap = data["apertures"]
        k = int(np.argmin(fit["res"]))
        ok, ratio = determined(fit["res"])

        check_against_maps(run, data, fit, k, source)
        print(f"      {len(data['cams'])} views, winner {ap[k]:g}°, m {fit['beta'][k]:.3f}, "
              f"r {ap[k] / 180.0:.3f}, well {ratio:.1f}x -> "
              f"{'determined' if ok else 'NOT determined'}")

        gs = outer[0, ci].subgridspec(5, 1, height_ratios=(0.86, 0.86, 0.74, 0.34, 0.54),
                                      hspace=0.16)
        # The environment and the colour share a vertical span, in decades: see
        # _log_window.  The margin keeps the markers off the frame.
        C, L = data["C"], data["L"][:, k]
        pred = intercept(data, fit, k) + float(fit["beta"][k]) * L
        span = 1.22 * max(np.log10(L.max() / L.min()),
                          np.log10(max(C.max(), pred.max()) / min(C.min(), pred.min())))
        fL = panel_percam_env(fig.add_subplot(gs[0]), data["cams"], L, span)
        fC = panel_percam_obs(fig.add_subplot(gs[1]), data["cams"], C, pred, span)
        print(f"      per camera: L varies {fL:.1f}x, C varies {fC:.2f}x, "
              f"span {span:.2f} decades")

        panel_residual(fig.add_subplot(gs[2]), ap, fit["res"], k, ok, ratio)
        panel_beta(fig.add_subplot(gs[3]), ap, fit["beta"], k)
        panel_readout(fig.add_subplot(gs[4]), float(fit["beta"][k]), float(ap[k]), ok)

        # Heading of the column, centred over the pair of scatter panels. The position
        # comes from the SubplotSpec and not from an axes added for the purpose: such an
        # axes is drawn, and its white ground would cover the two panels under it.
        box = gs[0].get_position(fig)
        fig.text((box.x0 + box.x1) / 2.0, 0.955, head, fontsize=27, color=C_INK,
                 ha="center", va="center", fontweight="bold")
        fig.text((box.x0 + box.x1) / 2.0, 0.917, f"{sub}  ·  {len(data['cams'])} views",
                 fontsize=14, color=C_MUTED, ha="center", va="center")

    # The two keys the picture cannot do without. Three coloured dots could as easily be
    # three cameras as three channels, and the hollow markers of the lower panel mean
    # nothing at all until it is said that they are the model.
    lg = fig.add_axes([0.055, 0.004, 0.34, 0.034])
    lg.axis("off"); lg.set_xlim(0, 1); lg.set_ylim(0, 1)
    for i, c in enumerate(CH_COLORS):
        lg.plot(0.02 + i * 0.055, 0.5, marker="o", ms=7.5, color=c, mec="white", mew=0.9)
    lg.text(0.19, 0.5, "one point per camera and colour channel", fontsize=12.5,
            color=C_MUTED, ha="left", va="center")

    lg2 = fig.add_axes([0.455, 0.004, 0.42, 0.034])
    lg2.axis("off"); lg2.set_xlim(0, 1); lg2.set_ylim(0, 1)
    lg2.plot(0.015, 0.5, marker="o", ms=6.5, color=C_MUTED, mec="white", mew=0.8)
    lg2.text(0.05, 0.5, "recorded", fontsize=12.5, color=C_MUTED, ha="left", va="center")
    lg2.plot(0.30, 0.5, marker="o", ms=8.0, color="white", mec=C_MUTED, mew=1.6)
    lg2.text(0.34, 0.5, r"model  $\alpha + m\,L_j$", fontsize=12.5, color=C_MUTED,
             ha="left", va="center")
    lg2.plot([0.73, 0.73], [0.18, 0.82], color=C_RESID, lw=1.8)
    lg2.text(0.76, 0.5, "residual", fontsize=12.5, color=C_MUTED, ha="left", va="center")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"  + {out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="the run folder")
    ap.add_argument("--out", required=True, help="destination folder")
    ap.add_argument("--name", default="metallic_from_slope",
                    help="output stem, without .png")
    ap.add_argument("--source", default="gt",
                    help="colour source under sources/ (default: %(default)s)")
    ap.add_argument("--texel-metal", nargs=2, type=int, default=list(COLUMNS[0][2]),
                    metavar=("ROW", "COL"))
    ap.add_argument("--texel-wood", nargs=2, type=int, default=list(COLUMNS[1][2]),
                    metavar=("ROW", "COL"))
    a = ap.parse_args(argv)

    columns = [(*COLUMNS[0][:2], tuple(a.texel_metal)),
               (*COLUMNS[1][:2], tuple(a.texel_wood))]
    run, out = Path(a.run), Path(a.out) / f"{a.name}.png"
    print(f"{run.name} -> {out.resolve()}")
    build(run, out, columns, a.source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
