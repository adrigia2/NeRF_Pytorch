#!/usr/bin/env python
"""make_aperture_selection_slide.py -- the candidate grid alone, on two real texels.

    python make_aperture_selection_slide.py --run <run dir> --out ../../PresentationImages

Writes `aperture_selection.png`: the three lower rows of `metallic_from_slope.png` and
nothing else, at the same panel geometry, so the two slides can follow one another in a
build without the curves changing shape under the audience.

WHAT IS LEFT OUT AND WHY.  The parent slide argues camera by camera: the environment each
camera's cone collects, against what that camera recorded.  That argument establishes the
slope.  This one starts after it and asks the second question only: given the slope at
every candidate aperture, which candidate does the data actually choose?  The residual
answers, the slope cannot -- on the steel it saturates at 1 across four candidates where
the residual varies by 6x -- and that separation is the whole point of showing the two
curves stacked on one axis of apertures.  The axis carries a tick per
candidate rather than the parent's 0/45/90/135/180: r is quantized to exactly these
fourteen values, the grid is dense near the mirror and coarse at the wide end, and that
density is what lets the steel land on 5 degrees.

The two columns are the two regimes.  On the steel the residual has a well 328x deep and
the aperture is chosen; on the wood it is flat to 1.9x end to end, so the winning 120
degrees is where the noise fell, not where the surface is.  The slide says so by drawing
that minimum hollow and greying the r it produces, exactly as the parent does
(`determined()` against RATIO_GATE, measured rather than declared).

Everything is imported from `make_pbr_fit_real_slide`: the reader, the fit, the panels and
`check_against_maps`, which compares metallic, roughness, lobe_param, residual and n_views
against the maps the run already holds and raises if they disagree.  Nothing is typed in
here either, and the two files cannot drift apart.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _paths  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402

from make_pbr_fit_figures import fit_moments                           # noqa: E402
from make_pbr_fit_real_slide import (                                  # noqa: E402
    COLUMNS, C_GRID, C_INK, C_MUTED, DPI, check_against_maps, determined,
    panel_beta, panel_readout, panel_residual, read_texel)

# The panels keep the parent slide's geometry, so the curves have the same aspect on both
# and a build can move from one to the other without them changing shape.  Margins are
# therefore given in INCHES and the figure height follows from them: with the ratios below
# and hspace=0.20 the three rows plus their two gaps come to PANELS_IN, of which the
# residual panel takes 1.50 in against the parent's 1.51.  Driving this from a fixed
# figsize instead would silently restyle the panels whenever a margin changed.
WIDTH_IN = 13.34
PANELS_IN = 3.726          # the three rows and the two gaps between them
BOTTOM_IN = 0.30
TOP_IN = {True: 1.05, False: 0.25}      # by `headings`
HEIGHT_RATIOS = (0.74, 0.34, 0.54)


def build(run: Path, out: Path, columns, source: str, headings: bool) -> None:
    height = PANELS_IN + BOTTOM_IN + TOP_IN[headings]
    fig = plt.figure(figsize=(WIDTH_IN, height))
    outer = fig.add_gridspec(1, 2, wspace=0.20, left=0.055, right=0.985,
                             top=1.0 - TOP_IN[headings] / height,
                             bottom=BOTTOM_IN / height)
    grids = []

    for ci, (head, sub, (y, x)) in enumerate(columns):
        print(f"  {head} ({sub}), texel ({y}, {x})")
        data = read_texel(run, y, x, source)
        fit = fit_moments(data["C"], data["L"])
        ap = data["apertures"]
        k = int(np.argmin(fit["res"]))
        ok, ratio = determined(fit["res"])

        check_against_maps(run, data, fit, k, source)
        print(f"      {len(data['cams'])} views, winner {ap[k]:g} deg, "
              f"m {fit['beta'][k]:.3f}, r {ap[k] / 180.0:.3f}, well {ratio:.1f}x -> "
              f"{'determined' if ok else 'NOT determined'}")

        gs = outer[0, ci].subgridspec(3, 1, height_ratios=HEIGHT_RATIOS, hspace=0.20)
        ax_res = fig.add_subplot(gs[0])
        panel_residual(ax_res, ap, fit["res"], k, ok, ratio)
        # Same tick positions as the panel below, labels only there: the two share an x
        # axis and are read as one stack, so two different sets of vertical grid lines
        # would be an artefact of the drawing rather than of the data.
        ax_res.set_xticks(ap)
        ax_res.tick_params(axis="x", length=0)
        ax_beta = fig.add_subplot(gs[1])
        panel_beta(ax_beta, ap, fit["beta"], k)
        # The parent ticks 0/45/90/135/180, which reads as a continuous axis and hides that
        # the grid is dense near the mirror and coarse at the wide end.  Here every
        # candidate carries its own tick: r is quantized to exactly these fourteen values,
        # and the reader has to be able to see which.  Rotated because 5, 10, 15 and 20 sit
        # 0.14 in apart and no horizontal label fits in that.
        ax_beta.set_xticks(ap)
        ax_beta.set_xticklabels([f"{a:g}" for a in ap], fontsize=8.5, rotation=90)
        ax_beta.tick_params(axis="x", length=3.5, color=C_GRID, pad=2)
        panel_readout(fig.add_subplot(gs[2]), float(fit["beta"][k]), float(ap[k]), ok)

        grids.append(ap)
        if headings:
            # From the SubplotSpec, not from an axes added for the purpose: such an axes
            # is drawn, and its white ground would cover the panel under it.
            box = gs[0].get_position(fig)
            fig.text((box.x0 + box.x1) / 2.0, 1.0 - 0.30 / height, head, fontsize=22,
                     color=C_INK, ha="center", va="center", fontweight="bold")
            fig.text((box.x0 + box.x1) / 2.0, 1.0 - 0.675 / height,
                     f"{sub}  ·  {len(data['cams'])} views", fontsize=12.5,
                     color=C_MUTED, ha="center", va="center")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"  + {out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="the run folder")
    ap.add_argument("--out", required=True, help="destination folder")
    ap.add_argument("--name", default="aperture_selection",
                    help="output stem, without .png")
    ap.add_argument("--source", default="gt",
                    help="colour source under sources/ (default: %(default)s)")
    ap.add_argument("--no-headings", dest="headings", action="store_false",
                    help="drop the column titles, for a build that already showed them")
    ap.add_argument("--texel-metal", nargs=2, type=int, default=list(COLUMNS[0][2]),
                    metavar=("ROW", "COL"))
    ap.add_argument("--texel-wood", nargs=2, type=int, default=list(COLUMNS[1][2]),
                    metavar=("ROW", "COL"))
    a = ap.parse_args(argv)

    columns = [(*COLUMNS[0][:2], tuple(a.texel_metal)),
               (*COLUMNS[1][:2], tuple(a.texel_wood))]
    run, out = Path(a.run), Path(a.out) / f"{a.name}.png"
    print(f"{run.name} -> {out.resolve()}")
    build(run, out, columns, a.source, a.headings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
