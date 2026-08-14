#!/usr/bin/env python
"""make_pbr_fit_figures.py -- figures for the worked example of the PBR fit.

    python make_pbr_fit_figures.py --out ../figure_review/pbr-fit
    python make_pbr_fit_figures.py --out ../Doc/images/pbr-fit \
           --figures worked_combined,moments_per_camera        (the two already in the thesis)

Generates the plots of "A Worked Example" (Doc/chapters/implementation.tex, the
subsubsection after eq:pbr-residual): one texel, three cameras, the regression
C_jc = alpha_c + beta*L_jc and the choice of aperture.

  fit_color_raw.png         C_j per camera: swatch + R/G/B bars, with the per-channel mean
  fit_color_centered.png    the same after centering: Delta C
  fit_cone_raw.png          L_j at the candidate shown in the table
  fit_cone_centered.png     Delta L
  fit_worked_combined.png   the four above in a single grid (the page alternative)
  fit_scatter.png           Delta L vs Delta C, the 9 points, the line of slope beta*
  fit_aperture_selection.png  why one aperture wins and another does not
  fit_residual_materials.png  res(Theta) for metal, glossy dielectric, diffuse

The C and L numbers at the candidate shown are those of tab:pbr-worked-example, written
here once as INPUTS: means, deviations, V_LL, V_CL, V_CC, beta* and res are always
recomputed and compared against the table by _check_against_table(), so that if one day
the table changes the comparison fails and no figure is left telling different numbers.
--print-table regenerates the LaTeX rows for a side-by-side check.

Apertures other than the one in the table do not exist in the thesis and have to be
modelled.  The model is that of a widening cone: the mean over the cone decays towards
the hemisphere mean H_c at a rate kappa_j that depends on the camera, because it depends
on how compact the source is around its reflected ray,

    u(Theta) = 1 - cos(Theta/2)                       (cone solid angle / 2pi)
    L_jc(Theta) = H_c + (L*_jc - H_c) * phi_j(Theta)
    phi_j = (1 + kappa_j*u*) / (1 + kappa_j*u)        (phi_j(Theta*) = 1)

Three properties make it usable: at the table's candidate it reproduces its values
EXACTLY; as Theta -> 180 every camera converges on H_c, which is what a widening cone
does; and since kappa_j changes with the camera, widening the cone changes the SHAPE of
Delta L between cameras and not only its scale.  That last point is the whole figure:
with kappa the same for all, Delta L would be rescaled as a block, beta would swallow
the factor and the residual curve would be flat -- which is then the degenerate case of
a diffuse texel.

H_c and kappa_j are not free: they were chosen by a grid search so that
argmin_Theta res(Theta) falls on the table's candidate and the well is visible
(neighbours ~12x the minimum, extremes ~120x).  _check_against_table() re-checks that too.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Data of the example: inputs, not results ─────────────────────────────────
# tab:pbr-worked-example, upper block.  (camera, channel)
C_OBS = np.array([[1.2, 2.1, 2.0],
                  [2.2, 3.2, 2.1],
                  [2.8, 3.7, 1.5]])
L_STAR = np.array([[1.0, 2.0, 3.0],
                   [3.0, 4.0, 3.0],
                   [4.0, 5.0, 2.0]])

# Operational aperture grid (images_generator.py, __main__): K = 14 candidates,
# index 0 = mirror ray.  The table's candidate is 45 degrees.
APERTURES = np.array([0., 5., 10., 15., 20., 30., 45., 60., 80., 100., 120., 140., 160., 180.])
K_STAR = 6

H_ENV = np.array([1.5, 2.3, 1.3])     # hemisphere mean, per channel
KAPPA = np.array([20.0, 30.0, 3.0])   # compactness of the source around R_j

# Materials of the three-curve figure: same texel, same envmap, only x and the lobe
# change.  The noise is ABSOLUTE and shared (same bake, same sensor): it is what makes
# the diffuse curve flat without having to declare it flat by hand.
NOISE_SIGMA = 0.045
NOISE_SEED = 19
MATERIALS = (("metal",     0.90, 2),
             ("glossy",    0.50, K_STAR),
             ("diffuse",   0.03, K_STAR))
ALBEDO_TERM = np.array([0.22, 0.26, 0.20])   # a_c*E_c/pi, the texel's diffuse colour

# ── Colours ──────────────────────────────────────────────────────────────────
# The channels are entities, not palette slots: they stay red/green/blue.  With these
# steps the worst pair sits at Delta E 7.2 under deuteranopia, inside the 6-8 band that
# is allowed only with a secondary encoding -- which is there: the channels are always in
# R, G, B order and labelled on the axis, so identity never rests on colour alone.
CH_COLORS = ("#e34948", "#008300", "#2a78d6")
CH_NAMES = ("R", "G", "B")
CAM_MARKERS = ("o", "s", "^")
# Categorical series (the three materials): slots 1-3 of the validated palette, every
# pair passes.  The aqua sits below 3:1 on the light background, so the curves are
# labelled in place and not only in the legend.
MAT_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
# The moments have a palette of their own: with blue/orange/green, three bars side by
# side in a plot where elsewhere three side-by-side bars ARE the channels would read as
# an RGB.  Violet, ochre and magenta (slots 7, 4, 5 of the same palette) contain neither
# blue nor green, so the confusion is simply not possible.  Ochre and magenta sit below
# 3:1 on the light background: here the relief is there, every bar carries its value.
MOMENT_COLORS = ("#4a3aa7", "#eda100", "#e87ba4")

# Names accepted by --figures, without the files' 'fit_' prefix.  The two published in
# the thesis are worked_combined and moments_per_camera: Doc/images/pbr-fit keeps only
# those, the rest live in figure_review until approved.
FIGURES = ("color_raw", "color_centered", "cone_raw", "cone_centered",
           "worked_combined", "scatter", "moments_per_camera", "beta",
           "aperture_selection", "residual_materials")

C_INK = "#222222"
C_SOFT = "#6b6b6b"
C_GRID = "#dcdcdc"
C_FIT = "#c0392b"        # the fit line, like the mirror ray in make_cone_diagram
C_RESID = "#eb6834"

# On the page the figure is shrunk: the sizes are chosen so that the text stays readable
# AFTER the reduction.
plt.rcParams.update({"font.size": 13, "axes.edgecolor": C_SOFT,
                     "axes.labelcolor": C_INK, "text.color": C_INK,
                     "xtick.color": C_SOFT, "ytick.color": C_SOFT})


# ── Mathematics of the fit ───────────────────────────────────────────────────
def fit_moments(C: np.ndarray, L: np.ndarray) -> dict:
    """Centred statistics and fit of one texel, for one or more candidates.

    Mirrors pbr_solver.py:270-283: the sums are POOLED over the three channels (beta is
    shared, alpha_c is not), beta is clipped to [0,1] and the residual is divided by the
    number of scalar equations 3*n_views.  C (n_cam, 3); L (n_cam, 3) or
    (n_cam, n_cand, 3).
    """
    single = L.ndim == 2
    Lx = L[:, None, :] if single else L
    dC = C - C.mean(axis=0, keepdims=True)
    dL = Lx - Lx.mean(axis=0, keepdims=True)
    VLL = (dL * dL).sum(axis=(0, 2))
    VCL = (dC[:, None, :] * dL).sum(axis=(0, 2))
    VCC = float((dC * dC).sum())
    beta = np.clip(VCL / np.maximum(VLL, 1e-12), 0.0, 1.0)
    res = (VCC - 2.0 * beta * VCL + beta ** 2 * VLL) / (3.0 * C.shape[0])
    # The two decompositions of the same sum: _c per channel (summed over the cameras),
    # _j per camera (summed over the channels).  Both give back the total when re-summed,
    # and that is what _check_against_table verifies.
    out = dict(dC=dC, dL=dL[:, 0] if single else dL, VLL=VLL, VCL=VCL, VCC=VCC,
               beta=beta, res=res,
               VLL_c=(dL * dL).sum(axis=0), VCL_c=(dC[:, None, :] * dL).sum(axis=0),
               VLL_j=(dL * dL).sum(axis=2), VCL_j=(dC[:, None, :] * dL).sum(axis=2),
               VCC_j=(dC * dC).sum(axis=1))
    if single:
        for k in ("VLL", "VCL", "beta", "res"):
            out[k] = float(out[k][0])
        for k in ("VLL_c", "VCL_c"):
            out[k] = out[k][0]
        for k in ("VLL_j", "VCL_j"):
            out[k] = out[k][:, 0]
    return out


def cone_curve() -> np.ndarray:
    """(n_cam, n_cand, 3): L_jc at every candidate aperture.  See the docstring."""
    u = 1.0 - np.cos(np.radians(APERTURES) * 0.5)
    phi = (1.0 + KAPPA[:, None] * u[K_STAR]) / (1.0 + KAPPA[:, None] * u[None, :])
    return H_ENV[None, None, :] + (L_STAR - H_ENV[None, :])[:, None, :] * phi[:, :, None]


def material_colors(L: np.ndarray) -> list[tuple]:
    """Synthetic observed colours for the three materials, from the same forward model."""
    out = []
    for name, beta_true, k_true in MATERIALS:
        rng = np.random.default_rng(NOISE_SEED)
        Cs = (ALBEDO_TERM[None, :] * (1.0 - beta_true) + beta_true * L[:, k_true, :]
              + NOISE_SIGMA * rng.standard_normal((L.shape[0], 3)))
        out.append((name, beta_true, k_true, Cs))
    return out


def _check_against_table(fit: dict, res_curve: np.ndarray) -> None:
    """The thesis table and the figures have to tell the same numbers."""
    exp = {"VLL": 10.0, "VCL": 16.0 / 3.0, "VCC": 2.853333, "beta": 8.0 / 15.0,
           "res": 9.876543e-4}
    for k, v in exp.items():
        got = float(fit[k])
        assert abs(got - v) < 5e-6, f"{k}: expected {v} from tab:pbr-worked-example, got {got}"
    assert np.allclose(fit["VLL_c"], [14 / 3, 14 / 3, 2 / 3], atol=5e-6), fit["VLL_c"]
    assert np.allclose(fit["VCL_c"], [2.466667, 2.5, 0.366667], atol=5e-6), fit["VCL_c"]
    # A decomposition that does not re-sum to the total is the only error the per-camera
    # figures could hide: the wrong bar would just look like data.
    for k in ("VLL", "VCL", "VCC"):
        for suffix in ("_c", "_j"):
            if k + suffix in fit:
                s = float(np.sum(fit[k + suffix]))
                assert abs(s - float(fit[k])) < 5e-6, f"{k}{suffix} sums to {s}, not {fit[k]}"
    w = fit["VLL_j"] / fit["VLL"]
    assert abs(float((w * (fit["VCL_j"] / fit["VLL_j"])).sum()) - fit["beta"]) < 5e-6, \
        "beta is not the V_LL_j-weighted mean of beta_j"
    k = int(np.argmin(res_curve))
    assert k == K_STAR, (f"the residual minimum falls at {APERTURES[k]:.0f} degrees and not "
                         f"on the table's candidate ({APERTURES[K_STAR]:.0f}): "
                         f"H_ENV/KAPPA need retuning")


# ── Drawing utilities ────────────────────────────────────────────────────────
def _tonemap(rgb: np.ndarray, scale: float) -> np.ndarray:
    """HDR -> sRGB with a shared exposure: the C and L swatches are comparable."""
    return np.clip(rgb / scale, 0.0, 1.0) ** (1.0 / 2.2)


def _bars(ax, values: np.ndarray, ylim: tuple, means: np.ndarray | None,
          fmt: str = "{:.3f}") -> None:
    """Three R/G/B bars with a direct label, grid in the background."""
    x = np.arange(3)
    ax.axhline(0.0, color=C_SOFT, lw=1.0, zorder=2)
    ax.bar(x, values, width=0.62, color=CH_COLORS, zorder=3)
    if means is not None:
        for i, m in enumerate(means):
            ax.plot([i - 0.42, i + 0.42], [m, m], color=C_INK, lw=1.4, ls=(0, (4, 2)),
                    zorder=4)
    span = ylim[1] - ylim[0]
    for i, v in enumerate(values):
        if means is None:
            # Centred bars: above the bar, which is where the number is looked for
            off = 0.028 * span * (1 if v >= 0 else -1)
            ax.text(i, v + off, fmt.format(v), ha="center", color=C_INK, fontsize=11,
                    va="bottom" if v >= 0 else "top", zorder=5)
        else:
            # Raw bars: inside the bar.  Above it would straddle the mean's dash when the
            # bar is lower than the mean, and the number would look like the mean's label
            # rather than the bar's.
            ax.text(i, v - 0.035 * span, fmt.format(v), ha="center", color="white",
                    fontsize=11, va="top", zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(CH_NAMES)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(*ylim)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=C_GRID, lw=0.7)
    ax.tick_params(length=0)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)


def _swatch(ax, rgb: np.ndarray, scale: float, label: str) -> None:
    ax.imshow(_tonemap(rgb, scale)[None, None, :], aspect="auto",
              extent=(0, 1, 0, 1), interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(C_SOFT); s.set_linewidth(0.8)
    ax.set_title(label, fontsize=13, color=C_INK, pad=6)


def _panel_grid(fig, specs, values, means, ylim, scale, titles, fmt="{:.3f}"):
    """A row of three panels (one camera each), with a swatch when scale is not None."""
    axes = []
    for j in range(3):
        if scale is not None:
            sub = specs[j].subgridspec(2, 1, height_ratios=(0.62, 5.0), hspace=0.10)
            _swatch(fig.add_subplot(sub[0]), values[j], scale, titles[j])
            ax = fig.add_subplot(sub[1])
        else:
            ax = fig.add_subplot(specs[j])
            ax.set_title(titles[j], fontsize=13, color=C_INK, pad=6)
        _bars(ax, values[j], ylim, means, fmt)
        if j:
            ax.set_yticklabels([])
        axes.append(ax)
    return axes


def _finish(fig, out: Path) -> None:
    fig.savefig(out, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  + {out.name}")


# ── Figures 1-4: per-camera bars, raw and centred ────────────────────────────
def fig_bars(values, means, ylim, scale, out: Path, suptitle: str, note: str,
             fmt: str = "{:.3f}") -> None:
    fig = plt.figure(figsize=(9.6, 4.6 if scale is not None else 3.9))
    gs = fig.add_gridspec(1, 3, wspace=0.12)
    titles = [f"Camera {j + 1}" for j in range(3)]
    axes = _panel_grid(fig, [gs[0, j] for j in range(3)], values, means, ylim, scale,
                       titles, fmt)
    axes[0].set_ylabel("radiance", color=C_INK)
    fig.suptitle(suptitle, fontsize=15, color=C_INK, y=1.02)
    fig.text(0.5, -0.035, note, ha="center", fontsize=11.5, color=C_SOFT)
    _finish(fig, out)


def fig_combined(fit: dict, L_star_mean, C_mean, ylim_raw, ylim_cen, scale,
                 out: Path) -> None:
    """The four panels in a single grid: this is the figure that goes in the thesis, so
    the spacing is tight -- at \\linewidth the wide version ate almost a whole page."""

    fig = plt.figure(figsize=(9.6, 10.6))
    gs = fig.add_gridspec(4, 3, hspace=0.42, wspace=0.12)
    rows = ((C_OBS, C_mean, ylim_raw, scale,
             r"Observed color $C_{jc}$"),
            (fit["dC"], None, ylim_cen, None,
             r"Centered color $\Delta C_{jc} = C_{jc} - \bar{C}_c$"),
            (L_STAR, L_star_mean, ylim_raw, scale,
             r"Cone radiance $L_{jc}$ at $\Theta = 45^\circ$"),
            (fit["dL"], None, ylim_cen, None,
             r"Centered radiance $\Delta L_{jc} = L_{jc} - \bar{L}_c$"))
    for r, (vals, means, ylim, sc, title) in enumerate(rows):
        titles = [f"Camera {j + 1}" for j in range(3)] if r == 0 else ["", "", ""]
        axes = _panel_grid(fig, [gs[r, j] for j in range(3)], vals, means, ylim, sc,
                           titles)
        axes[0].set_ylabel("radiance", color=C_INK)
        # Above the swatch band when there is one: the row title is long and at the height
        # of the bar axis it would lie across camera 1's swatch.
        # In the first row the camera names are above the swatch too.
        title_y = (1.46 if r == 0 else 1.32) if sc is not None else 1.04
        axes[0].text(-0.34, title_y, title, transform=axes[0].transAxes, fontsize=14,
                     color=C_INK, ha="left", va="bottom")
    _finish(fig, out)


# ── Figure 5: the centred regression ─────────────────────────────────────────
def _scatter_fit(ax, dL, dC, beta, res, xlim=None, drops=True, legend=False) -> None:
    lim = xlim or (np.abs(dL).max() * 1.35)
    xs = np.array([-lim, lim])
    ax.axhline(0, color=C_GRID, lw=1.0, zorder=1)
    ax.axvline(0, color=C_GRID, lw=1.0, zorder=1)
    ax.plot(xs, beta * xs, color=C_FIT, lw=2.0, zorder=3,
            label=fr"$\Delta C = \beta^\star \Delta L$")
    if drops:
        for j in range(dL.shape[0]):
            for c in range(3):
                ax.plot([dL[j, c], dL[j, c]], [dC[j, c], beta * dL[j, c]],
                        color=C_RESID, lw=1.6, zorder=4, solid_capstyle="butt")
    # Smaller markers on the later channels: two points can nearly coincide (camera 1 in
    # R and G differ by 0.03) and at equal size the second would vanish under the first.
    # The size is redundant, not informative: the channel stays the colour.
    #
    for c in range(3):
        for j in range(dL.shape[0]):
            ax.plot(dL[j, c], dC[j, c], marker=CAM_MARKERS[j], ms=(12.0, 9.0, 6.5)[c],
                    color=CH_COLORS[c], mec="white", mew=1.3, ls="none", zorder=5 + c)
    ax.set_xlim(-lim, lim)
    ax.set_axisbelow(True)
    ax.grid(color=C_GRID, lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if legend:
        handles = [Line2D([], [], color=CH_COLORS[c], marker="o", ls="none", ms=9,
                          mec="white", mew=1.2, label=f"channel {CH_NAMES[c]}")
                   for c in range(3)]
        handles += [Line2D([], [], color=C_SOFT, marker=CAM_MARKERS[j], ls="none",
                           ms=9, mec="white", mew=1.2, label=f"camera {j + 1}")
                    for j in range(3)]
        handles += [Line2D([], [], color=C_FIT, lw=2.0, label="fitted slope")]
        if drops:
            handles += [Line2D([], [], color=C_RESID, lw=1.8, label="residual")]
        ax.legend(handles=handles, loc="upper left", fontsize=11, frameon=False,
                  ncol=2, handletextpad=0.5, columnspacing=1.4, labelspacing=0.35)


def fig_scatter(fit: dict, out: Path) -> None:
    """The regression and, next to it, what it leaves behind: without the second panel
    this texel's residuals (third decimal) are below the width of the line and the figure
    would only say 'the points are aligned'."""
    fig = plt.figure(figsize=(12.6, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.55, 1.0), wspace=0.26)

    ax = fig.add_subplot(gs[0])
    _scatter_fit(ax, fit["dL"], fit["dC"], fit["beta"], fit["res"], drops=False,
                 legend=True)
    ax.set_ylim(-1.22, 1.22)
    ax.set_xlabel(r"centered cone radiance $\Delta L_{jc}$")
    ax.set_ylabel(r"centered color $\Delta C_{jc}$")
    ax.set_title("One slope for nine equations", fontsize=14, color=C_INK, pad=10)
    ax.text(0.985, 0.05,
            fr"$\beta^\star = V_{{CL}}/V_{{LL}} = {fit['VCL']:.3f}/{fit['VLL']:.3f}"
            fr" = {fit['beta']:.3f}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=12.5, color=C_INK)

    axr = fig.add_subplot(gs[1])
    resid = fit["dC"] - fit["beta"] * fit["dL"]
    # Separated groups: the labels of the two values straddling one camera and the next
    # would touch.
    x = np.array([4.4 * j + 1.15 * c for j in range(3) for c in range(3)])
    axr.axhline(0.0, color=C_SOFT, lw=1.0, zorder=2)
    axr.bar(x, resid.ravel(), width=0.66, color=list(CH_COLORS) * 3, zorder=3)
    lim = np.abs(resid).max() * 1.75
    for i, v in enumerate(resid.ravel()):
        axr.text(x[i], v + 0.03 * lim * (1 if v >= 0 else -1), f"{v:+.3f}",
                 ha="center", va="bottom" if v >= 0 else "top", fontsize=10,
                 color=C_INK, zorder=5)
    axr.set_xticks(x)
    axr.set_xticklabels(list(CH_NAMES) * 3)
    for j in range(3):
        axr.text(4.4 * j + 1.15, -lim * 0.93, f"camera {j + 1}", ha="center",
                 va="bottom", fontsize=11.5, color=C_SOFT)
        if j:
            axr.axvline(4.4 * j - 1.05, color=C_GRID, lw=1.0)
    axr.set_ylim(-lim, lim)
    axr.set_xlim(-1.0, 11.4)
    axr.set_ylabel(r"$\Delta C_{jc} - \beta^\star \Delta L_{jc}$")
    axr.set_title("What the slope leaves behind", fontsize=14, color=C_INK, pad=10)
    axr.text(0.5, 0.985,
             fr"$\sum \mathrm{{residual}}^2 = {fit['res'] * 9:.5f}$   over 9 equations"
             "\n"
             fr"$\mathrm{{res}} = {fit['res'] * 9:.5f}/9 = {fit['res']:.2e}$",
             transform=axr.transAxes, ha="center", va="top", fontsize=12,
             color=C_INK, linespacing=1.6)
    axr.set_axisbelow(True)
    axr.grid(axis="y", color=C_GRID, lw=0.7)
    axr.tick_params(length=0)
    for s in ("top", "right", "bottom"):
        axr.spines[s].set_visible(False)

    fig.suptitle(r"The centered regression of a single texel, at $\Theta = 45^\circ$",
                 fontsize=15, color=C_INK, y=1.03)
    _finish(fig, out)


# ── Figure 6: the moments, camera by camera ──────────────────────────────────
def fig_moments_per_camera(fit: dict, out: Path) -> None:
    """V_LL, V_CL, V_CC per camera plus the group of totals.

    The 'total' group is not decoration: they are the three numbers that really enter
    beta, and having them as a fourth column avoids annotating separately what sums to
    what.  The labels on EVERY bar are mandatory here: on a linear scale camera 2 is 0.33
    against camera 1's 5.67 and without a number it would be an unreadable sliver -- which
    is then exactly the thing to show.
    """
    vals = np.stack([fit["VLL_j"], fit["VCL_j"], fit["VCC_j"]], axis=1)   # (3 cam, 3)
    tot = np.array([fit["VLL"], fit["VCL"], fit["VCC"]])
    data = np.vstack([vals, tot])                                          # (4, 3)
    names = (r"$V_{LL}$", r"$V_{CL}$", r"$V_{CC}$")

    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    width = 0.26
    for m in range(3):
        x = np.arange(4) + (m - 1) * width
        ax.bar(x, data[:, m], width=width * 0.92, color=MOMENT_COLORS[m], zorder=3,
               label=names[m])
        for i, v in enumerate(data[:, m]):
            ax.text(x[i], v + 0.13, f"{v:.3f}", ha="center", va="bottom", fontsize=10.5,
                    color=C_INK, rotation=90, zorder=5)
    ax.axvline(2.5, color=C_SOFT, lw=1.0)
    ax.axvspan(2.5, 3.6, color="#f2f2f2", zorder=0)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(["camera 1", "camera 2", "camera 3", "total\n(what $\\beta$ uses)"])
    ax.set_xlim(-0.55, 3.6)
    ax.set_ylim(0, data.max() * 1.30)
    ax.set_ylabel("sum over the three channels")
    ax.set_title(r"Where $V_{LL}$, $V_{CL}$ and $V_{CC}$ come from, camera by camera",
                 fontsize=14, color=C_INK, pad=10)
    ax.legend(loc="upper left", fontsize=12, frameon=False, ncol=3)
    ax.annotate("camera 2 sits at the mean of the other two:\n"
                f"{fit['VLL_j'][1] / fit['VLL'] * 100:.0f}% of $V_{{LL}}$, so it barely "
                "constrains the slope",
                xy=(1 - width, fit["VLL_j"][1] + 0.9), xytext=(0.255, 0.40),
                textcoords="axes fraction", fontsize=11, color=C_SOFT,
                ha="left", va="bottom", linespacing=1.5,
                arrowprops=dict(arrowstyle="-", color=C_SOFT, lw=1.0, shrinkB=2))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=C_GRID, lw=0.7)
    ax.tick_params(length=0)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    _finish(fig, out)


# ── Figure 7: where beta comes from ──────────────────────────────────────────
def fig_beta(fit: dict, out: Path) -> None:
    trials = (0.20, fit["beta"], 0.90)
    # Two different greys and not one: the residual segments have no dash pattern, so at
    # equal colour one could not tell which slope they belong to.
    t_col = ("#9a9a9a", C_FIT, "#4d4d4d")
    t_ls = ((0, (5, 2)), "-", (0, (1, 2)))
    fig = plt.figure(figsize=(14.6, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=(1.15, 1.0, 1.0), wspace=0.38)

    # (a) three trial slopes on the same nine points
    ax = fig.add_subplot(gs[0])
    lim = np.abs(fit["dL"]).max() * 1.35
    xs = np.array([-lim, lim])
    ax.axhline(0, color=C_GRID, lw=1.0)
    ax.axvline(0, color=C_GRID, lw=1.0)
    for b, col, ls in zip(trials, t_col, t_ls):
        sse = fit["VCC"] - 2 * b * fit["VCL"] + b ** 2 * fit["VLL"]
        ax.plot(xs, b * xs, color=col, lw=2.0, ls=ls, zorder=3,
                label=fr"$b = {b:.2f}$,  SSE $= {sse:.3f}$")
        for j in range(3):
            for c in range(3):
                ax.plot([fit["dL"][j, c]] * 2, [fit["dC"][j, c], b * fit["dL"][j, c]],
                        color=col, lw=1.2, alpha=0.55, zorder=2)
    for c in range(3):
        for j in range(3):
            ax.plot(fit["dL"][j, c], fit["dC"][j, c], marker=CAM_MARKERS[j],
                    ms=(11.0, 8.5, 6.0)[c], color=CH_COLORS[c], mec="white", mew=1.3,
                    ls="none", zorder=5 + c)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-1.35, 1.35)
    ax.set_xlabel(r"$\Delta L_{jc}$")
    ax.set_ylabel(r"$\Delta C_{jc}$")
    ax.set_title("Try a slope, measure what is left", fontsize=13.5, color=C_INK, pad=8)
    ax.legend(loc="upper left", fontsize=10.5, frameon=False, labelspacing=0.3)

    # (b) the parabola: this is where beta comes out
    axp = fig.add_subplot(gs[1])
    b = np.linspace(-0.15, 1.25, 400)
    sse = fit["VCC"] - 2 * b * fit["VCL"] + b ** 2 * fit["VLL"]
    axp.axvspan(0.0, 1.0, color="#f2f2f2", zorder=0)
    axp.plot(b, sse, color=C_INK, lw=2.0, zorder=4)
    for t, col in zip(trials, t_col):
        s = fit["VCC"] - 2 * t * fit["VCL"] + t ** 2 * fit["VLL"]
        axp.plot(t, s, marker="o", ms=9, color=col, mec="white", mew=1.4, zorder=6)
    axp.plot([fit["beta"]] * 2, [0, fit["res"] * 9], color=C_FIT, lw=1.2,
             ls=(0, (3, 3)), zorder=3)
    axp.annotate(fr"$\beta^\star = \dfrac{{V_{{CL}}}}{{V_{{LL}}}} = {fit['beta']:.3f}$",
                 xy=(fit["beta"], fit["res"] * 9), xytext=(0.30, 0.52),
                 textcoords="axes fraction", fontsize=13, color=C_FIT, ha="left",
                 va="bottom", arrowprops=dict(arrowstyle="-", color=C_FIT, lw=1.0,
                                              shrinkB=4))
    axp.text(0.5, 0.985,
             r"SSE$(b) = V_{CC} - 2\,b\,V_{CL} + b^2 V_{LL}$",
             transform=axp.transAxes, ha="center", va="top", fontsize=12.5, color=C_INK)
    axp.text(0.98, 0.06, "shaded: $b$ is clamped\nto $[0, 1]$",
             transform=axp.transAxes, ha="right", va="bottom", fontsize=10.5,
             color=C_SOFT, linespacing=1.4)
    axp.set_ylim(0, sse.max() * 1.12)
    axp.set_xlim(-0.15, 1.25)
    axp.set_xlabel(r"trial slope $b$")
    axp.set_ylabel(r"SSE$(b)$   ($9 \times \mathrm{res}$)")
    axp.set_title(r"$\beta^\star$ is where the parabola bottoms out", fontsize=13.5,
                  color=C_INK, pad=8)

    # (c) beta as a weighted mean of the per-camera slopes
    axw = fig.add_subplot(gs[2])
    w = fit["VLL_j"] / fit["VLL"]
    beta_j = fit["VCL_j"] / fit["VLL_j"]
    # Dots and not bars: the three slopes differ by 0.04 out of 0.53, so they have to be
    # looked at closely, and a zoomed axis with bars would start from a fake zero.
    # The dot's area is the weight, and that is how one sees beta* fall next to camera 1
    # (57%) and far from camera 2 (3%).
    axw.axvline(fit["beta"], color=C_FIT, lw=2.0, zorder=4)
    for j in range(3):
        axw.plot([min(beta_j[j], fit["beta"]), max(beta_j[j], fit["beta"])],
                 [2 - j] * 2, color=C_GRID, lw=1.4, zorder=2)
        axw.scatter(beta_j[j], 2 - j, s=max(1500.0 * w[j], 55.0), color=C_SOFT,
                    edgecolor="white", linewidth=1.5, zorder=5)
    axw.text(fit["beta"], -0.66, fr"$\beta^\star = {fit['beta']:.3f}$", ha="center",
             va="bottom", fontsize=12.5, color=C_FIT)
    # Values on the axis labels and not next to the dots: the axis is narrow and any text
    # inside the panel would straddle the beta* line.
    axw.set_yticks([2, 1, 0])
    axw.set_yticklabels([f"camera {j + 1}\n$\\beta_{{{j + 1}}}$ = {beta_j[j]:.3f}\n"
                         f"$w_{{{j + 1}}}$ = {w[j] * 100:.0f}%"
                         for j in range(3)], fontsize=11.5)
    axw.set_xlim(min(beta_j.min(), fit["beta"]) - 0.025, beta_j.max() + 0.025)
    axw.set_ylim(-0.80, 2.85)
    axw.set_xlabel(r"per-camera slope $\beta_j = V_{CL}^{(j)} / V_{LL}^{(j)}$"
                   "\n" r"(marker area = weight)")
    axw.set_title(r"$\beta^\star = \sum_j w_j\, \beta_j$,   $w_j = V_{LL}^{(j)}/V_{LL}$",
                  fontsize=13.5, color=C_INK, pad=8)

    for a in (ax, axp, axw):
        a.set_axisbelow(True)
        a.grid(color=C_GRID, lw=0.7)
        a.tick_params(length=0)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    axw.grid(axis="x", color=C_GRID, lw=0.7)
    axw.yaxis.grid(False)
    _finish(fig, out)


# ── Figure 8: why that aperture ──────────────────────────────────────────────
def fig_aperture_selection(L: np.ndarray, curve: dict, out: Path) -> None:
    show = (2, K_STAR, 10)          # 10 degrees (narrow), 45 (winner), 120 (wide)
    tags = ("A", "B", "C")
    fig = plt.figure(figsize=(10.4, 10.0))
    # Two nested gridspecs and not one: between the scatter row and the residual there
    # needs to be air (the Delta L labels are down there), between residual and beta
    # there does not, because they share the x axis and must read as one block.
    outer = fig.add_gridspec(2, 1, height_ratios=(1.30, 1.55), hspace=0.30)
    gs = outer[0].subgridspec(1, 3, wspace=0.16)
    gs_bot = outer[1].subgridspec(2, 1, height_ratios=(1.0, 0.42), hspace=0.12)

    dC = curve["dC"]
    ylim = np.abs(dC).max() * 1.42
    for p, k in enumerate(show):
        ax = fig.add_subplot(gs[0, p])
        dL = curve["dL"][:, k]
        _scatter_fit(ax, dL, dC, curve["beta"][k], curve["res"][k],
                     xlim=np.abs(dL).max() * 1.45)
        ax.set_ylim(-ylim, ylim)
        if p:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(r"$\Delta C_{jc}$")
        ax.set_xlabel(r"$\Delta L_{jc}$")
        kind = ("too narrow", "best fit", "too wide")[p]
        ax.set_title(f"{tags[p]}   $\\Theta = {APERTURES[k]:.0f}^\\circ$ ({kind})",
                     fontsize=13.5, color=C_INK, pad=8)
        ax.text(0.035, 0.965, fr"$\beta^\star = {curve['beta'][k]:.3f}$" "\n"
                              fr"$\mathrm{{res}} = {curve['res'][k]:.2e}$",
                transform=ax.transAxes, ha="left", va="top", fontsize=11.5,
                color=C_INK, linespacing=1.5)

    # residual: this is what chooses
    axr = fig.add_subplot(gs_bot[0])
    axr.plot(APERTURES, curve["res"], color=C_FIT, lw=2.0, marker="o", ms=6.5,
             mec="white", mew=1.2, zorder=4)
    kmin = int(np.argmin(curve["res"]))
    axr.plot(APERTURES[kmin], curve["res"][kmin], marker="o", ms=13, color=C_FIT,
             mec="white", mew=2.0, zorder=6)
    axr.annotate(f"winner: {APERTURES[kmin]:.0f}$^\\circ$\n"
                 f"metallic = $\\beta^\\star$ = {curve['beta'][kmin]:.3f}\n"
                 f"roughness = {APERTURES[kmin]:.0f}/180 = "
                 f"{APERTURES[kmin] / 180.0:.3f}",
                 xy=(APERTURES[kmin], curve["res"][kmin]), xycoords="data",
                 xytext=(0.63, 0.10), textcoords="axes fraction", fontsize=12,
                 color=C_INK, ha="left", va="bottom", linespacing=1.6,
                 arrowprops=dict(arrowstyle="-", color=C_SOFT, lw=1.0,
                                 shrinkA=6, shrinkB=10))
    blend = axr.get_xaxis_transform()
    for p, k in enumerate(show):
        axr.axvline(APERTURES[k], color=C_SOFT, lw=1.0, ls=(0, (3, 3)), zorder=1)
        axr.text(APERTURES[k], 0.97, tags[p], ha="center", va="top", fontsize=13,
                 color=C_SOFT, transform=blend)
    axr.set_yscale("log")
    axr.set_ylim(curve["res"].min() * 0.55, curve["res"].max() * 2.4)
    axr.set_ylabel(r"residual  $\mathrm{res}(\Theta)$")
    axr.set_title(r"The residual selects the aperture", fontsize=14, color=C_INK, pad=8)

    # beta: it only grows, it has no minimum -- it cannot choose
    axb = fig.add_subplot(gs_bot[1], sharex=axr)
    axb.plot(APERTURES, curve["beta"], color=C_INK, lw=2.0, marker="o", ms=5.5,
             mec="white", mew=1.0, zorder=4)
    axb.axhline(1.0, color=C_SOFT, lw=1.0, ls=(0, (3, 3)))
    axb.set_ylim(0.0, 1.18)
    axb.set_ylabel(r"$\beta^\star(\Theta)$")
    axb.set_xlabel(r"candidate total aperture $\Theta$  [degrees]")
    axb.text(0.5, 0.08, "monotone in $\\Theta$, and clamped at 1: no minimum to pick",
             transform=axb.transAxes, ha="center", fontsize=11.5, color=C_SOFT)
    for ax in (axr, axb):
        ax.set_axisbelow(True)
        ax.grid(color=C_GRID, lw=0.7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axr.tick_params(labelbottom=False)
    axr.set_xlim(-6, 186)
    axb.set_xticks(APERTURES[::2])
    _finish(fig, out)


# ── Figure 7: metal, glossy, diffuse ─────────────────────────────────────────
def fig_residual_materials(L: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    lines, rmax = [], 1.0
    blend = ax.get_xaxis_transform()
    for i, (name, beta_true, k_true, Cs) in enumerate(material_colors(L)):
        f = fit_moments(Cs, L)
        r = f["res"] / f["res"].min()
        k = int(np.argmin(f["res"]))
        rmax = max(rmax, r.max())
        ax.plot(APERTURES, r, color=MAT_COLORS[i], lw=2.0, marker="o", ms=5.5,
                mec="white", mew=1.0, zorder=4)
        ax.plot(APERTURES[k], 1.0, marker="o", ms=13, color=MAT_COLORS[i],
                mec="white", mew=2.0, zorder=6)
        # Label in place on the right tail: the legend at the top forced the three minimum
        # captions to stack on the x axis.  It also acts as relief for the aqua, which on
        # the light background sits below 3:1.
        ax.annotate(f"{name}\npicks {APERTURES[k]:.0f}$^\\circ$, "
                    f"$\\beta^\\star$ = {f['beta'][k]:.2f}",
                    xy=(APERTURES[-1], r[-1]), xytext=(-4, 12),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=12, color=MAT_COLORS[i], linespacing=1.4, zorder=7)
        # coloured tick on the chosen candidate: it reads without following the curve
        ax.plot([APERTURES[k]] * 2, [0.0, 0.045], color=MAT_COLORS[i], lw=3.0,
                transform=blend, clip_on=False, zorder=7)
        lines.append((name, beta_true, k_true, k, f))
    ax.set_yscale("log")
    ax.set_xlim(-6, 186)
    ax.set_xticks(APERTURES[::2])
    ax.set_ylim(0.5, rmax * 7.0)
    ax.set_xlabel(r"candidate total aperture $\Theta$  [degrees]")
    ax.set_ylabel(r"residual, relative to its own minimum")
    ax.set_title("How much the data constrains the lobe, by material",
                 fontsize=14, color=C_INK, pad=10)
    ax.set_axisbelow(True)
    ax.grid(color=C_GRID, lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _finish(fig, out)
    return lines


# ── Numeric summary ──────────────────────────────────────────────────────────
def print_table(fit: dict) -> None:
    """Rows of tab:pbr-worked-example regenerated from the inputs, for comparison."""
    print("\n% tab:pbr-worked-example, regenerated (not transcribed)")
    for j in range(3):
        print(f"        Camera {j + 1} & " + " & ".join(
            f"${v:.3f}$" for v in np.r_[C_OBS[j], L_STAR[j]]) + r" \\")
    print("        Mean over cameras & " + " & ".join(
        f"${v:.3f}$" for v in np.r_[C_OBS.mean(0), L_STAR.mean(0)]) + r" \\")
    for j in range(3):
        print(f"        Camera {j + 1}, centered & " + " & ".join(
            f"${v:.3f}$" for v in np.r_[fit["dC"][j], fit["dL"][j]]) + r" \\")
    print("        Per-channel $V_{LL}$ & & & & " + " & ".join(
        f"${v:.3f}$" for v in fit["VLL_c"]) + r" \\")
    print("        Per-channel $V_{CL}$ & " + " & ".join(
        f"${v:.3f}$" for v in fit["VCL_c"]) + r" & & & \\")
    print(f"        $V_{{LL}} = {fit['VLL']:.3f}$ \\quad $V_{{CL}} = {fit['VCL']:.3f}$ "
          f"\\quad $\\beta^\\star = {fit['beta']:.3f}$ \\quad "
          f"res $= {fit['res']:.2e}$")
    print(f"% (V_CC = {fit['VCC']:.4f}, sum of squared residuals = "
          f"{fit['res'] * 9:.5f} over 9 equations)")


def print_curve(curve: dict, L: np.ndarray) -> None:
    print("\n% res(Theta) on the table's texel")
    print("  Theta    VLL     VCL    beta      res    res/min")
    rmin = curve["res"].min()
    for a in range(len(APERTURES)):
        star = " <-- table" if a == K_STAR else ""
        print(f"  {APERTURES[a]:5.0f}  {curve['VLL'][a]:6.2f} {curve['VCL'][a]:6.2f} "
              f"{curve['beta'][a]:6.3f}  {curve['res'][a]:.3e} {curve['res'][a]/rmin:8.1f}"
              f"{star}")
    print(f"  L min {L.min():.3f}  max {L.max():.3f}   (mirror level: "
          f"{np.round(L[:, 0, :], 2).tolist()})")


def print_per_camera(fit: dict) -> None:
    """Per-camera decomposition: the numbers of the two new figures."""
    w = fit["VLL_j"] / fit["VLL"]
    beta_j = fit["VCL_j"] / fit["VLL_j"]
    print("\n% per-camera moments (summed over the three channels)")
    print("           V_LL    V_CL    V_CC   beta_j   weight")
    for j in range(3):
        print(f"  camera {j + 1} {fit['VLL_j'][j]:6.3f} {fit['VCL_j'][j]:7.3f} "
              f"{fit['VCC_j'][j]:7.3f}  {beta_j[j]:6.3f} {w[j] * 100:6.1f}%")
    print(f"  total    {fit['VLL']:6.3f} {fit['VCL']:7.3f} {fit['VCC']:7.3f}  "
          f"{fit['beta']:6.3f} {w.sum() * 100:6.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--figures", default=None,
                    help="subset to generate, comma-separated names without the "
                         "'fit_' prefix (default: all).  Needed for the copy in "
                         "Doc/images/pbr-fit, which keeps only the published figures")
    ap.add_argument("--print-table", action="store_true",
                    help="regenerate the LaTeX table rows and the numeric detail")
    args = ap.parse_args()
    out = Path(args.out)

    if args.figures is None:
        want = lambda name: True                                  # noqa: E731
    else:
        asked = {s.strip() for s in args.figures.split(",") if s.strip()}
        unknown = asked - set(FIGURES)
        # A wrong name has to stop the script: otherwise the copy in the thesis ends up
        # empty-handed and nobody notices until the build.
        if unknown:
            raise SystemExit(f"--figures: unknown names {sorted(unknown)}; "
                             f"available {sorted(FIGURES)}")
        want = asked.__contains__
    out.mkdir(parents=True, exist_ok=True)   # after the check: a typo must not create folders

    L = cone_curve()
    fit = fit_moments(C_OBS, L_STAR)
    curve = fit_moments(C_OBS, L)
    _check_against_table(fit, curve["res"])

    # A single exposure for every swatch: C and L have to stay comparable
    scale = float(max(C_OBS.max(), L_STAR.max()))
    ylim_raw = (0.0, scale * 1.16)
    ylim_cen = float(max(np.abs(fit["dC"]).max(), np.abs(fit["dL"]).max())) * 1.30
    ylim_cen = (-ylim_cen, ylim_cen)

    print(f"[figure] {out}")
    if want("color_raw"):
        fig_bars(C_OBS, C_OBS.mean(0), ylim_raw, scale, out / "fit_color_raw.png",
                 r"Observed color $C_{jc}$, one texel seen by three cameras",
                 "dashed: the per-channel mean over the cameras, which centering subtracts"
                 f"  |  swatches tonemapped at a shared exposure (divide by {scale:.1f},"
                 " then gamma 1/2.2)")
    if want("color_centered"):
        fig_bars(fit["dC"], None, ylim_cen, None, out / "fit_color_centered.png",
                 r"After centering: $\Delta C_{jc} = C_{jc} - \bar{C}_c$",
                 "the diffuse term is gone: it contributed equally to all three cameras")
    if want("cone_raw"):
        fig_bars(L_STAR, L_STAR.mean(0), ylim_raw, scale, out / "fit_cone_raw.png",
                 r"Cone radiance $L_{jc}$ at the candidate $\Theta = 45^\circ$",
                 "same texel, same three cameras, same exposure as the observed color")
    if want("cone_centered"):
        fig_bars(fit["dL"], None, ylim_cen, None, out / "fit_cone_centered.png",
                 r"After centering: $\Delta L_{jc} = L_{jc} - \bar{L}_c$",
                 "blue varies seven times less than red and green: why the channels are "
                 "pooled")
    if want("worked_combined"):
        fig_combined(fit, L_STAR.mean(0), C_OBS.mean(0), ylim_raw, ylim_cen, scale,
                     out / "fit_worked_combined.png")
    if want("scatter"):
        fig_scatter(fit, out / "fit_scatter.png")
    if want("moments_per_camera"):
        fig_moments_per_camera(fit, out / "fit_moments_per_camera.png")
    if want("beta"):
        fig_beta(fit, out / "fit_beta.png")
    if want("aperture_selection"):
        fig_aperture_selection(L, curve, out / "fit_aperture_selection.png")
    mats = (fig_residual_materials(L, out / "fit_residual_materials.png")
            if want("residual_materials") else None)

    print_per_camera(fit)
    print_curve(curve, L)
    if mats is not None:
        print("\n% materials (shared absolute sigma "
              f"{NOISE_SIGMA}, seed {NOISE_SEED})")
        for name, beta_true, k_true, k, f in mats:
            print(f"  {name:8s} true beta {beta_true:.2f} at {APERTURES[k_true]:5.0f} "
                  f"degrees  ->  picks {APERTURES[k]:5.0f}, beta* {f['beta'][k]:.3f}, "
                  f"res {f['res'].min():.2e}, curve max/min "
                  f"{f['res'].max() / f['res'].min():8.1f}")
    if args.print_table:
        print_table(fit)
    print(f"\n[ok] swatch exposure = {scale:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
