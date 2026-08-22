#!/usr/bin/env python
"""make_loss_curves_figure.py -- the training losses of the activation/loss sweep.

    python make_loss_curves_figure.py --out ../../Doc/images/nerf-ablation

Writes one PNG, `train_loss_grid.png`: a 3x2 grid with the training loss of each of the
six configurations of the sweep, loss functions down the rows and activations across the
columns, ONE RUN PER PANEL.

The figure exists to give the reader a reference for what a loss that is actually
descending looks like, because the only training curve the chapter shows otherwise is the
flat one of the run that does not train.  Hence one run per panel and no overlays: two
runs in one panel would read as a comparison between them, and the comparison is not the
point here, the shape of each curve is.

Three things the layout is doing on purpose:

  * the y axis is logarithmic and its limits are shared BY ROW, that is between the two
    activations of the same loss, and never between rows.  The six losses run from 0.017
    to 88071 because each is in the units of its own objective; a shared scale would say
    they are comparable, and they are not;

  * the panel titles use the vocabulary of the sweep table in the thesis (exponential /
    softplus, L1 / sq. / rel. sq.) rather than the internal keys of the CSV
    (l1 / mse / rel_mse_raw), so the figure and the table can be read against each other;

  * the first and last value are printed inside each panel, straight from the data, so the
    claim the figure makes is a number the reader can check and not an impression of a
    slope.

The colours come from `compare_runs.COLORS`, the palette the rest of this sweep's figures
already use: the colour family identifies the loss, the shade the activation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _paths  # noqa: F401

# compare_runs is self-contained (stdlib, numpy, matplotlib: it imports neither the
# pipeline nor nerf.metrics), so this costs nothing and keeps run discovery, the CSV
# reader and the palette in one place.
from compare_runs import (COLORS, DEFAULT_ROOT, col_float, discover_runs, read_csv_rows)

import matplotlib.pyplot as plt

# The names the thesis uses, against the keys the CSV carries.
ACT_NAME = {"exp": "exponential", "softplus": "softplus"}
LOSS_NAME = {"l1": "$L_1$", "mse": "sq.", "rel_mse_raw": "rel. sq."}

ROWS = ("l1", "mse", "rel_mse_raw")             # top to bottom: one loss per row
COLS = ("exp", "softplus")                      # left to right: one activation per column

# Three rows of two and not two of three.  The section ends just before a `\section`, which
# this document starts on a new page, so the float is flushed onto a page of its own
# whatever its size; a portrait block fills that page instead of leaving two thirds of it
# blank, and it buys the panels half as much width again.
#
# The figure prints at \linewidth, 5.795 in, so a 7.0 in canvas is reduced to 0.83 and an
# 11 pt label lands at 9.1 pt on the page.  Same reasoning as the other thesis figures:
# the size is chosen on what it becomes after the reduction, not on screen.
FIGSIZE = (7.0, 7.2)
DPI = 200
FS = 11.0

C_INK = "#222222"
C_NOTE = "0.35"


def series(run) -> tuple[np.ndarray, np.ndarray]:
    """(iteration, loss) of one run, in file order.

    No dedup and no `epoch` column: these six CSVs predate that column and none of the
    runs was ever resumed, so the iteration grid is already strictly increasing.  `report`
    checks both of those rather than leaving them assumed.
    """
    rows = read_csv_rows(run.train_csv)
    return col_float(rows, "iter"), col_float(rows, "loss")


def figure(runs: dict, data: dict, out: Path) -> Path:
    fig, axes = plt.subplots(len(ROWS), len(COLS), figsize=FIGSIZE,
                             sharex=True, squeeze=False)

    # One pair of limits per row, taken over both activations of that loss: within a row
    # the two panels have to be readable against each other, between rows they must not
    # invite it.
    ylim = {}
    for lt in ROWS:
        vals = np.concatenate([data[(act, lt)][1] for act in COLS])
        vals = vals[np.isfinite(vals) & (vals > 0)]
        ylim[lt] = (vals.min() / 2.0, vals.max() * 2.0)

    for i, lt in enumerate(ROWS):
        for j, act in enumerate(COLS):
            ax = axes[i][j]
            it, loss = data[(act, lt)]
            ax.plot(it, loss, color=COLORS[(act, lt)], lw=1.4)
            ax.set_yscale("log")
            ax.set_ylim(*ylim[lt])
            ax.set_xlim(0, 75000)
            ax.set_xticks([0, 25000, 50000, 75000])
            ax.set_xticklabels(["0", "25k", "50k", "75k"])
            ax.grid(alpha=0.25, which="both", lw=0.4)
            ax.set_title(f"{ACT_NAME[act]}, {LOSS_NAME[lt]}", fontsize=FS, color=C_INK)
            if j == 0:
                ax.set_ylabel("loss", fontsize=FS - 0.5)
            if i == len(ROWS) - 1:
                ax.set_xlabel("iteration", fontsize=FS - 0.5)
            ax.tick_params(labelsize=FS - 2)
            # First and last value, from the data.  The curves descend from the top left,
            # so the top right corner is the one place free in every panel.
            # Four significant digits and not three: at three, 0.66993 prints as "0.67"
            # next to a "0.576" in the panel above it and reads as a rounding slip.
            ax.text(0.97, 0.93, f"{loss[0]:.4g} $\\rightarrow$ {loss[-1]:.4g}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=FS - 2, color=C_NOTE)

    fig.tight_layout()
    path = out / "train_loss_grid.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def report(runs: dict, data: dict) -> None:
    """The numbers the thesis prose quotes, and the checks that they mean what it says."""
    print()
    for lt in ROWS:
        for act in COLS:
            it, loss = data[(act, lt)]
            print(f"  {ACT_NAME[act]:<12s} {lt:<12s} "
                  f"iter {int(it[0]):>6d}..{int(it[-1]):<6d}  "
                  f"loss {loss[0]:>11.5g} -> {loss[-1]:<11.5g}  "
                  f"factor {loss[0] / loss[-1]:>8.1f}x")
            assert np.all(np.isfinite(loss)), f"{act}/{lt}: a non-finite loss on record"
            assert np.all(np.diff(it) > 0), \
                f"{act}/{lt}: the iteration grid is not increasing, the run was resumed"
            # The figure's whole claim in one line: every one of the six descends.  A run
            # that ended above where it started would make the section say the opposite of
            # what it shows.
            assert loss[-1] < loss[0], f"{act}/{lt}: the loss did not descend"

    iters = {tuple(data[k][0]) for k in data}
    assert len(iters) == 1, "the six runs are not on the same iteration grid"
    print(f"\n  all six on the same grid, {len(next(iter(iters)))} points, "
          f"{int(min(next(iter(iters))))}..{int(max(next(iter(iters))))}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True,
                    help="folder for the PNG, i.e. ../../Doc/images/nerf-ablation")
    ap.add_argument("--sweep", default=DEFAULT_ROOT,
                    help=f"sweep root (default: {DEFAULT_ROOT})")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    found = discover_runs(Path(args.sweep), None)
    runs = {(r.activation, r.loss): r for r in found}
    missing = [(a, l) for a in COLS for l in ROWS if (a, l) not in runs]
    if missing:
        ap.error(f"the sweep at {args.sweep} is missing: {missing}")

    data = {k: series(runs[k]) for k in ((a, l) for a in COLS for l in ROWS)}
    path = figure(runs, data, out)
    print(f"  + {path}")
    report(runs, data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
