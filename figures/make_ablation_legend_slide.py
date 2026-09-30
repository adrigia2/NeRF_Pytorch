#!/usr/bin/env python
"""make_ablation_legend_slide.py -- the two activations and the three losses of the NeRF ablation.

    python make_ablation_legend_slide.py --out ../../PresentationImages

Writes `nerf_ablation_legend.png`: the two output activations as curves on the top row,
the three training losses as curves on the bottom row (each in the hue the ablation
figures use for it, `compare_runs.COLORS`), under them the same 10 % error evaluated at
two scales, and a one-line key for x, y, p and t.  It is the slide that makes the
`exp/l1 ... softplus/rel_mse_raw` legend of the ablation curves readable.

The functions are the ones the code runs, not sketches of them:

  * activations, `nerf/rays.py`: `exp(x)` and `softplus(x) = log(1 + e^x)` on the raw
    RGB output, next to the sigmoid of a standard NeRF drawn in grey as the one *not*
    used, flat against the y = 1 line the other two cross.  Both are unbounded above (HDR), both are strictly positive; they differ
    in how fast they grow and in how they reach zero - exp only asymptotically, which is
    what makes its dead zone a one-way street;
  * losses, `nerf/train.py`: `|p - t|`, `(p - t)^2` and `(p - t)^2 / (p + eps)^2` with
    eps = 1e-3 (the RawNeRF relative squared error, prediction detached).  The eps is
    used in the numbers but left out of the printed formula, so the slide does not
    invite a question about it.  The curves are drawn against the prediction `p` for a
    fixed target `t = 1`, so the V, the parabola and the asymmetry of the relative error
    are visible at a glance: the relative error punishes an underestimate without bound
    and is nearly indifferent to an overestimate.

The two-scale row is the HDR argument in one line: for the same 10 % error, L1 grows
by 10^4 between a shadow pixel and a light source, the squared error by 10^8, the
relative squared error not at all.  With a mean over the batch, that ratio is who
drives the gradient.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _paths  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

from compare_runs import COLORS                       # noqa: E402
from make_loss_curves_figure import ACT_NAME          # noqa: E402

C_INK = "#33404d"
C_MUTED = "#5f6c79"
C_GHOST = "#c9d1d9"
C_LINE = "#d5dbe1"

ACTS = ("exp", "softplus")
LOSSES = ("l1", "mse", "rel_mse_raw")
ACT_FORMULA = {"exp": r"$y = e^{x}$", "softplus": r"$y = \log(1 + e^{x})$"}
LOSS_FORMULA = {"l1": r"$|p - t|$", "mse": r"$(p - t)^2$",
                "rel_mse_raw": r"$\dfrac{(p - t)^2}{p^2}$"}     # eps = 1e-3 in the code, left out here
LOSS_TITLE = {"l1": "L1", "mse": "squared", "rel_mse_raw": "relative squared"}
EPS = 1e-3


def act_fn(name, x):
    return np.exp(x) if name == "exp" else np.log1p(np.exp(x))


def loss_fn(name, p, t=1.0):
    d = p - t
    if name == "l1":
        return np.abs(d)
    if name == "mse":
        return d ** 2
    return d ** 2 / (p + EPS) ** 2


def _hex_to_rgb(h):
    return tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))


def _mix(a, b):
    """Midpoint of two hex colours: the hue of a loss, between its two shades."""
    ra, rb = _hex_to_rgb(a), _hex_to_rgb(b)
    return tuple((x + y) / 2 for x, y in zip(ra, rb))


# ── the three kinds of panel ──────────────────────────────────────────────────

def draw_activation(ax, name):
    """One activation, the other in ghost grey behind it, the LDR ceiling dashed."""
    x = np.linspace(-4.0, 3.0, 400)
    other = ACTS[1 - ACTS.index(name)]
    ax.plot(x, act_fn(other, x), color=C_GHOST, lw=2.0, zorder=1)
    ax.plot(x, act_fn(name, x), color=C_INK, lw=3.0, zorder=3)
    ax.axhline(0.0, color=C_MUTED, lw=1.0, zorder=0)
    ax.axhline(1.0, color=C_MUTED, lw=1.0, ls=(0, (4, 4)), zorder=0)
    ax.text(x[0] + 0.1, 1.15, "1", fontsize=12.5, color=C_MUTED, ha="left", va="bottom")
    ax.axvline(0.0, color=C_LINE, lw=1.0, zorder=0)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-0.4, 8.0)
    ax.axis("off")
    ax.text(0.03, 0.97, ACT_NAME[name], transform=ax.transAxes, fontsize=18.75,
            color=C_INK, ha="left", va="top", weight="bold")
    ax.text(0.03, 0.82, ACT_FORMULA[name], transform=ax.transAxes, fontsize=16.25,
            color=C_MUTED, ha="left", va="top")


def draw_sigmoid(ax):
    """The standard NeRF activation, not used here: it never leaves [0, 1]."""
    x = np.linspace(-4.0, 3.0, 400)
    ax.plot(x, 1.0 / (1.0 + np.exp(-x)), color=C_MUTED, lw=3.0, zorder=3)
    ax.axhline(0.0, color=C_MUTED, lw=1.0, zorder=0)
    ax.axhline(1.0, color=C_MUTED, lw=1.0, ls=(0, (4, 4)), zorder=0)
    ax.text(x[0] + 0.1, 1.15, "1", fontsize=12.5, color=C_MUTED, ha="left", va="bottom")
    ax.axvline(0.0, color=C_LINE, lw=1.0, zorder=0)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-0.4, 8.0)
    ax.axis("off")
    ax.text(0.03, 0.97, "sigmoid", transform=ax.transAxes, fontsize=18.75,
            color=C_MUTED, ha="left", va="top", weight="bold")
    ax.text(0.03, 0.82, r"$y = 1 / (1 + e^{-x})$", transform=ax.transAxes, fontsize=16.25,
            color=C_MUTED, ha="left", va="top")
    ax.text(0.03, 0.67, "standard NeRF, not used:\nbounded to [0, 1]", transform=ax.transAxes,
            fontsize=13.5, color=C_MUTED, ha="left", va="top", style="italic",
            linespacing=1.4)


def draw_loss(ax, name):
    """One loss against the prediction for a fixed target t = 1."""
    p = np.linspace(0.0, 3.0, 600)
    col = _mix(COLORS[("exp", name)], COLORS[("softplus", name)])
    y = loss_fn(name, p)
    y = np.where(y <= 4.5, y, np.nan)          # leave the frame instead of plateauing
    ax.plot(p, y, color=col, lw=3.0, zorder=3)
    ax.axhline(0.0, color=C_MUTED, lw=1.0, zorder=0)
    ax.plot([1.0, 1.0], [0.0, 4.6], color=C_MUTED, lw=1.0, ls=(0, (4, 4)), zorder=0)
    ax.text(1.0, -0.15, "t", fontsize=13.75, color=C_MUTED, ha="center", va="top")
    ax.text(3.0, -0.15, "p", fontsize=13.75, color=C_MUTED, ha="right", va="top")
    ax.set_xlim(-0.05, 3.05)
    ax.set_ylim(-0.6, 5.8)
    ax.axis("off")
    ax.text(0.5, 1.0, LOSS_TITLE[name], transform=ax.transAxes, fontsize=18.75,
            color=col, ha="center", va="top", weight="bold")
    ax.text(0.97, 0.84, LOSS_FORMULA[name], transform=ax.transAxes, fontsize=16.25,
            color=C_MUTED, ha="right", va="top")


# ── layout ────────────────────────────────────────────────────────────────────

def build(out: Path) -> None:
    """Two rows: the two activations, then the three losses, then a one-line key."""
    W, H = 15.0, 10.5
    fig = plt.figure(figsize=(W, H))

    left, right = 0.09, 0.985
    top_row = (0.62, 0.33)                            # y0, height of the activation row
    bot_row = (0.26, 0.31)                            # y0, height of the loss row
    gap = 0.02

    def row_axes(n, y0, h):
        w = (right - left - (n - 1) * gap) / n
        return [fig.add_axes([left + k * (w + gap), y0, w, h]) for k in range(n)]

    fig.text(0.035, top_row[0] + top_row[1] / 2, "activation", fontsize=18.75, color=C_INK,
             ha="center", va="center", rotation=90, weight="bold")
    fig.text(0.035, bot_row[0] + bot_row[1] / 2, "loss", fontsize=18.75, color=C_INK,
             ha="center", va="center", rotation=90, weight="bold")
    fig.add_artist(plt.Line2D([left, right], [top_row[0] - gap / 2] * 2,
                              color=C_LINE, lw=1.0))

    # the two activations used, then the one every standard NeRF uses and this one does not
    w = (right - left - 2 * gap) / 3
    top_axes = row_axes(3, *top_row)
    for ax, act in zip(top_axes, ACTS):
        draw_activation(ax, act)
    draw_sigmoid(top_axes[2])

    for ax, loss in zip(row_axes(3, *bot_row), LOSSES):
        draw_loss(ax, loss)

    # the same 10 % error at two scales, one value per loss
    cases = ((0.9, 1.0), (9000.0, 10000.0))
    y_tab = (0.19, 0.145)
    fig.text(0.045, sum(y_tab) / 2, "same\n10 % error", fontsize=14, color=C_INK,
             ha="center", va="center", weight="bold", linespacing=1.4)
    for k, loss in enumerate(LOSSES):
        col = _mix(COLORS[("exp", loss)], COLORS[("softplus", loss)])
        x0 = left + k * (w + gap)
        for (pv, tv), y in zip(cases, y_tab):
            val = float(loss_fn(loss, np.array(pv), tv))
            txt = f"{val:,.0f}".replace(",", " ") if val >= 1 else f"{val:.3g}"
            lab = f"$p$ = {pv:,g}   $t$ = {tv:,g}".replace(",", " ")
            fig.text(x0 + 0.012, y, lab, fontsize=13.5, color=C_MUTED, ha="left",
                     va="center")
            fig.text(x0 + w - 0.012, y, txt, fontsize=16, color=col, ha="right",
                     va="center", weight="bold")
        fig.add_artist(plt.Line2D([x0 + 0.012, x0 + w - 0.012], [y_tab[0] + 0.03] * 2,
                                  color=C_LINE, lw=1.0))

    # the key
    key = (r"$x$ = raw network output,  $y$ = radiance,  dashed line: $y = 1$"
           "\n"
           r"$p$ = predicted radiance,  $t$ = target radiance (curves drawn for $t = 1$)")
    fig.text((left + right) / 2, 0.05, key, fontsize=16.25, color=C_MUTED,
             ha="center", va="center", linespacing=1.7)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  + {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="../../PresentationImages")
    a = ap.parse_args()
    build(Path(a.out) / "nerf_ablation_legend.png")


if __name__ == "__main__":
    main()
