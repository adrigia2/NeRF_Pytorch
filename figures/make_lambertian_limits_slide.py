#!/usr/bin/env python
"""make_lambertian_limits_slide.py -- original against Lambertian, and WHY it differs.

    python make_lambertian_limits_slide.py --out ../../PresentationImages

Writes `lambertian_limits.png`: the authored base colour, the albedo the Lambertian
inversion recovers from it, and a third panel that says where the two part company and for
which of two entirely different reasons.

The slide it replaces put the original metallic map beside the comparison and left the room
to do the matching.  Two things were being asked of the audience at once, and they are not
the same kind of claim:

  A LIMIT OF THE MODEL.  Where the surface is metal there is no diffuse albedo to recover:
  `C = a E / pi` has nothing to say about a conductor, and the inversion returns whatever
  makes the arithmetic work.  This is the argument for a model that carries the specular
  term, and it is the point of the slide.

  A LIMIT OF THE DATA.  Where no camera ever saw a texel the colour texture is empty, so
  the albedo is zero and the texel comes out black.  Nothing about the model would fix it.

Drawn the same way, as "the picture looks wrong here", the two are indistinguishable, and a
question from the room about one of them lands on the other.  The third panel separates
them by colour.  How much of the surface each one covers is printed on stdout and kept in
the README rather than on the slide: written there it is read instead of the panel beside
it, and the panel is what says WHERE.  `--bare` drops the legend and the captions too, for
a slide that carries all of its words in the speaker.

The masks are MEASURED, not drawn: metal is the authored metallic above 0.5, and unseen is
the reconstruction being exactly zero inside the IUM mask.  That second one is not a proxy:
the texels where the albedo is black and the texels where the colour texture is black are
the same set, agreement 1.000, which is what "no camera saw it" means in this pipeline,
since the colour texture is the mean over the cameras that see a texel.

The two comparison panels go through the SAME transfer, plain sRGB with no exposure, the
`thesis_format` convention for a quantity already in [0, 1].  A comparison of two
reflectances under two different curves would be an argument about tone mapping.
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

from make_geometry_diagrams import C_EDGE, C_CAM, C_HILITE, C_RAY      # noqa: E402

C_INK = C_CAM
C_MUTED = C_EDGE
C_METAL = C_HILITE             # the model cannot explain it
C_UNSEEN = C_RAY               # there is nothing to explain it from
C_MESH = "#c9cfd6"             # the rest of the surface
C_VOID = "#101418"             # outside the atlas islands

TF = "PresentationImages/thesis_format/"
ORIGINAL = TF + "interiorSpecularOriginal_base_color_srgb.png"
LAMBERT = TF + "albedo_lambertian_srgb.png"
METALLIC = TF + "interiorSpecularOriginal_metallic_linear01.png"
IUM = "Doc/images/ium/ium_mask.png"

METAL_T = 0.5                  # the authored metallic is all but binary; any cut in the
BLACK_T = 0.02                 # middle gives the same set

# -- Layout, in axes units (10 to the inch) -----------------------------------
MARGIN = 3.0
DPI = 200
MAXPX = 1100

S = 44.0                       # side of a comparison panel
TOP = (6.0, 56.0), (60.0, 56.0)
W = 34.0                       # side of the third panel
WHY = (6.0, 10.0)              # with the legend beside it
WHY_BARE = (38.0, 10.0)        # without: centred under the two above
LEG_X = 48.0


def load(path: Path, gray: bool = False) -> np.ndarray:
    a = np.asarray(plt.imread(path), dtype=np.float32)
    return a[..., 0] if gray else a[..., :3]


def decimate(a: np.ndarray) -> np.ndarray:
    step = max(1, int(np.ceil(max(a.shape[:2]) / MAXPX)))
    return a[::step, ::step]


def masks(root: Path):
    """Where the model has nothing to say, and where the data has nothing to say."""
    ium = load(root / IUM, gray=True) > 0.5
    metal = (load(root / METALLIC, gray=True) > METAL_T) & ium
    unseen = (load(root / LAMBERT).max(axis=-1) < BLACK_T) & ium
    return ium, metal, unseen


def why_panel(ium, metal, unseen) -> np.ndarray:
    """A categorical map, unseen drawn over metal.

    Where a texel is both, the missing data is the binding one: with no observation there
    is no question about the model yet.
    """
    out = np.zeros(ium.shape + (3,), dtype=np.float32)
    out[:] = to_rgb(C_VOID)
    out[ium] = to_rgb(C_MESH)
    out[metal] = to_rgb(C_METAL)
    out[unseen] = to_rgb(C_UNSEEN)
    return out


def to_rgb(c: str) -> np.ndarray:
    return np.array(matplotlib.colors.to_rgb(c), dtype=np.float32)


def panel(ax, img, x0, y0, side, label, edge=C_INK):
    ax.imshow(decimate(img), extent=(x0, x0 + side, y0, y0 + side), aspect="auto",
              zorder=3, interpolation="antialiased")
    ax.add_patch(Rectangle((x0, y0), side, side, facecolor="none", edgecolor=edge,
                           lw=1.5, zorder=4))
    if label:
        ax.text(x0 + side / 2.0, y0 - 3.5, label, fontsize=16, color=C_INK,
                ha="center", va="top")


def legend(ax):
    """Two lines and no numbers.

    The shares (printed on stdout, and in the README) used to be here.  They belong to the
    speaker: written on the slide they are read instead of the panel beside them, and the
    panel is the thing that says WHERE, which is what the sentence needs.
    """
    entries = ((C_METAL, "metallic part"),
               (C_UNSEEN, "part seen by no camera"))
    y = 32.0
    for col, label in entries:
        ax.add_patch(Rectangle((LEG_X, y - 2.2), 6.0, 4.6, facecolor=col,
                               edgecolor="none", zorder=3))
        ax.text(LEG_X + 8.5, y, label, fontsize=19, color=C_INK, ha="left", va="center")
        y -= 11.0


def build(root: Path, out: Path, bare: bool) -> None:
    ium, metal, unseen = masks(root)
    n = float(ium.sum())
    pct_metal, pct_unseen = 100.0 * metal.sum() / n, 100.0 * unseen.sum() / n

    fig = plt.figure(figsize=(20.0, 14.0))       # scratch canvas, cropped below
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 140)

    panel(ax, load(root / ORIGINAL), *TOP[0], S,
          None if bare else "the original, as it was authored")
    panel(ax, load(root / LAMBERT), *TOP[1], S,
          None if bare else "recovered by the Lambertian inversion")
    why = WHY_BARE if bare else WHY
    panel(ax, why_panel(ium, metal, unseen), *why, W,
          None if bare else "where they part company, and why")
    if not bare:
        legend(ax)

    x0 = TOP[0][0] - MARGIN
    x1 = max(TOP[1][0] + S, LEG_X + 40.0 if not bare else 0.0) + MARGIN
    y0 = (why[1] if bare else 4.0) - MARGIN
    y1 = TOP[0][1] + S + MARGIN
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    fig.set_size_inches((x1 - x0) / 10.0, (y1 - y0) / 10.0)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"  metal {pct_metal:.2f} %   unseen {pct_unseen:.2f} %   "
          f"both {100.0 * (metal & unseen).sum() / n:.2f} %")
    print(f"  frame {x1 - x0:.1f} x {y1 - y0:.1f} units, ratio {(x1 - x0) / (y1 - y0):.3f}")
    print(f"  + {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="../../PresentationImages")
    ap.add_argument("--assets", default=None,
                    help="folder the asset paths are relative to (default: the repo root)")
    ap.add_argument("--bare", action="store_true",
                    help="no legend and no captions: the three panels alone")
    a = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]
    root = Path(a.assets) if a.assets else repo
    name = "lambertian_limits_bare.png" if a.bare else "lambertian_limits.png"
    build(root, Path(a.out).resolve() / name, a.bare)


if __name__ == "__main__":
    main()
