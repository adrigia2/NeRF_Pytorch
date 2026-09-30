#!/usr/bin/env python
"""make_irradiance_sum_slide.py -- direct + indirect = the hemisphere, at ONE exposure.

    python make_irradiance_sum_slide.py --out ../../PresentationImages

Writes `irradiance_sum.png` for the slide that says: the passes give two irradiance
textures, one carrying what comes from the environment map and one what comes back from the
indirect bounces, and the irradiance of the hemisphere is the sum of the two.

The slide has a second job, which is the reason this script exists.  Shown side by side,
each exposed on its own, the two textures look like equal partners and they are not: the
indirect is 13.2% of the direct.  Three devices carry that, and they are meant to be read
together because each one alone is not enough.

  ONE EXPOSURE FOR ALL THREE.  The panels come from `make_presentation_magnitude.py`, which
  normalises the three irradiances by a single factor, the p99.5 of the DIRECT (k = 5.783).
  The same colour is then the same radiance in all three, and the indirect is dark because
  it is dark.  Exposed one by one instead, it is shown 4.83x larger than it is and no
  reader can tell.

  A BAR.  One exposure is necessary and not sufficient: gamma 2.2 pulls the ratio back up.
  Measured on these very files, the mean of the indirect is 13.1% of the direct in linear
  radiance but 37.9% of it in displayed value, so even the honest panel understates the gap
  by about three times.  No tone-mapped panel can do better, which is why the number is
  written rather than shown: the bar is the only thing here that carries the ratio exactly.

  A GAINED COPY.  Being honest about the level costs the structure: at the shared exposure
  the indirect is nearly unreadable, and it is worth looking at.  It is repeated at its own
  exposure with the factor stated, the convention every rendering paper uses for this.

`SHARE` is the fraction measured on the EXRs and documented in `PresentationImages/README.md`;
`check()` recomputes it from the PNGs actually drawn, by undoing the gamma and the factor,
and refuses to draw if the two disagree by more than half a point.  A slide whose numbers
and whose pictures come apart is worse than one with no numbers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _paths  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.patches import Rectangle                               # noqa: E402

from make_geometry_diagrams import C_EDGE, C_CAM, C_DIRECT, C_INDIR    # noqa: E402

C_INK = C_CAM
C_MUTED = C_EDGE

# -- The measurements, all documented in PresentationImages/README.md ---------
K_SHARED = 5.783        # p99.5 of the direct: the one factor the three panels share
K_OWN = 1.196           # p99.5 of the indirect, the factor its own-exposure copy uses
RATIO = 0.1320          # mean(indirect) / mean(direct), measured on the EXRs
SHARE = RATIO / (1.0 + RATIO)          # the indirect as a fraction of the total
GAIN = K_SHARED / K_OWN                # how much the own-exposure copy is lifted by

SRC = "PresentationImages/"
PANELS = (("irradiance_p995shared_gamma22.png", "from the environment map"),
          ("irradiance_indirect_p995shared_gamma22.png", "from the NeRF"),
          ("irradiance_full_p995shared_gamma22.png", "the hemisphere"))
GAINED = "thesis_format/irradiance_indirect_p995_gamma22.png"

# -- Layout, in axes units (10 to the inch) -----------------------------------
MARGIN = 3.0
DPI = 200
MAXPX = 1100

S = 40.0                       # side of a map panel
COL = (6.0, 58.0, 110.0)       # left edge of each panel
ROW_Y = 53.0                   # bottom edge of the row of panels
OPS = (52.0, 104.0)            # x of the + and the = signs

# Three squares in a row are 3.5 wide for 1 tall whatever else is on the slide, and there is
# no arrangement of three EQUAL squares that lands near the 1.5 of a titled page: the widest
# is the row, the narrowest is two over one at about 0.9, and everything between costs a gap
# with nothing in it.  The full slide reaches 1.43 because the bar and the gained copy fill
# that gap; with the panels alone the choice is the two ends.  `--panels` draws the narrow
# one, which is also the arrangement the slide already had.
TRI = ((6.0, 66.0), (62.0, 66.0), (34.0, 8.0))     # A, B on top, C below
TRI_OPS = (((55.0, 87.0), "+"), ((55.0, 56.0), "="))
# The bar goes last, at the bottom: it is the punchline, and between the panels and the
# gained copy it cut the copy off from the panel it repeats.
BAR = (6.0, 4.0, 150.0, 13.0)  # x0, y0, x1, y1 of the energy bar
INSET = (66.0, 20.0, 90.0, 44.0)   # centred under the indirect panel


def load(path: Path) -> np.ndarray:
    img = np.asarray(plt.imread(path), dtype=np.float32)[..., :3]
    step = max(1, int(np.ceil(max(img.shape[:2]) / MAXPX)))
    return img[::step, ::step]


def check(root: Path) -> float:
    """Recover the linear ratio from the panels that are actually drawn.

    The panels are `clip(L / k) ** (1/2.2)`, so `(png ** 2.2) * k` is the radiance back
    again, up to what the clip took at the very top.  Returns the reconstructed ratio and
    raises if it disagrees with the documented one.
    """
    lin = []
    for name, _ in PANELS[:2]:
        a = np.asarray(plt.imread(root / SRC / name), dtype=np.float64)[..., :3]
        lin.append(float(((a ** 2.2) * K_SHARED).mean()))
    got = lin[1] / lin[0]
    if abs(got / (1.0 + got) - SHARE) > 0.005:
        raise SystemExit(f"the panels say {got:.4f}, the constant says {RATIO:.4f}")
    return got


def draw_panels(ax, root: Path, pos, ops):
    for (x0, y0), (name, label) in zip(pos, PANELS):
        ax.imshow(load(root / SRC / name), extent=(x0, x0 + S, y0, y0 + S),
                  aspect="auto", zorder=3, interpolation="antialiased")
        ax.add_patch(Rectangle((x0, y0), S, S, facecolor="none", edgecolor=C_INK,
                               lw=1.5, zorder=4))
        ax.text(x0 + S / 2.0, y0 - 3.5, label, fontsize=16, color=C_INK,
                ha="center", va="top")

    for (x, y), sign in ops:
        ax.text(x, y, sign, fontsize=40, color=C_INK, ha="center", va="center")


def draw_bar(ax):
    """The one thing on the slide that carries the ratio exactly."""
    x0, y0, x1, y1 = BAR
    cut = x0 + (x1 - x0) * (1.0 - SHARE)
    ax.add_patch(Rectangle((x0, y0), cut - x0, y1 - y0, facecolor=C_DIRECT,
                           edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((cut, y0), x1 - cut, y1 - y0, facecolor=C_INDIR,
                           edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none",
                           edgecolor=C_INK, lw=1.3, zorder=4))

    mid = 0.5 * (y0 + y1)
    ax.text(0.5 * (x0 + cut), mid, f"{100 * (1 - SHARE):.1f} %", fontsize=20,
            color="white", ha="center", va="center", fontweight="bold", zorder=5)
    # Inside the purple and not above it: above, it lands on the caption of the third
    # panel, and a number that touches a caption is read as part of it.
    ax.text(cut + 0.5 * (x1 - cut), mid, f"{100 * SHARE:.1f} %", fontsize=16,
            color="white", ha="center", va="center", fontweight="bold", zorder=5)
    ax.text(x0, y1 + 2.5, "share of the irradiance each one carries",
            fontsize=15, color=C_MUTED, ha="left", va="bottom")


def draw_inset(ax, root: Path):
    """The indirect again, on its own exposure, with the lift stated."""
    x0, y0, x1, y1 = INSET
    ax.imshow(load(root / SRC / GAINED), extent=(x0, x1, y0, y1), aspect="auto",
              zorder=3, interpolation="antialiased")
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none",
                           edgecolor=C_INDIR, lw=1.8, zorder=4))
    # No arrow up to the panel it repeats: it sits directly under it, in its colour, and an
    # arrow across that gap would have had to cross the bar to say what the position says.
    ax.text(x1 + 5.0, y1 - 2.0, f"$\\times$ {GAIN:.1f}", fontsize=26, color=C_INDIR,
            ha="left", va="top", fontweight="bold")
    ax.text(x1 + 5.0, y1 - 11.0,
            "the same map on its own exposure,\nso that what is in it can be seen",
            fontsize=15, color=C_MUTED, ha="left", va="top")


def build(root: Path, out: Path, panels_only: bool) -> None:
    got = check(root)

    fig = plt.figure(figsize=(20.0, 14.0))       # scratch canvas, cropped below
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 140)

    if panels_only:
        draw_panels(ax, root, TRI, TRI_OPS)
        xs = [x for x, _ in TRI] + [x + S for x, _ in TRI]
        ys = [y for _, y in TRI] + [y + S for _, y in TRI]
        x0, x1 = min(xs) - MARGIN, max(xs) + MARGIN
        y0, y1 = min(ys) - 5.0 - MARGIN, max(ys) + MARGIN
    else:
        draw_panels(ax, root, [(x, ROW_Y) for x in COL],
                    tuple(((x, ROW_Y + S / 2.0), sg) for x, sg in zip(OPS, ("+", "="))))
        draw_bar(ax)
        draw_inset(ax, root)

        top = ROW_Y + S + 6.0
        ax.text(COL[0], top, "one exposure for all three: same colour, same radiance",
                fontsize=16, color=C_INK, ha="left", va="bottom")

        x0 = min(COL[0], BAR[0], INSET[0]) - MARGIN
        x1 = max(COL[2] + S, BAR[2], 150.0) + MARGIN
        y0 = BAR[1] - MARGIN
        y1 = top + 4.0 + MARGIN
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    fig.set_size_inches((x1 - x0) / 10.0, (y1 - y0) / 10.0)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"  ratio from the panels {got:.4f} vs documented {RATIO:.4f}")
    print(f"  share {100 * SHARE:.1f} %   gain {GAIN:.2f}x")
    print(f"  frame {x1 - x0:.1f} x {y1 - y0:.1f} units, ratio {(x1 - x0) / (y1 - y0):.3f}")
    print(f"  + {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="../../PresentationImages")
    ap.add_argument("--assets", default=None,
                    help="folder the asset paths are relative to (default: the repo root)")
    ap.add_argument("--panels", action="store_true",
                    help="only the three panels, two over one, without bar or gained copy")
    a = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]
    root = Path(a.assets) if a.assets else repo
    name = "irradiance_sum_panels.png" if a.panels else "irradiance_sum.png"
    build(root, Path(a.out).resolve() / name, a.panels)


if __name__ == "__main__":
    main()
