#!/usr/bin/env python
"""make_maps_slides.py -- authored / recovered / difference maps plus camera coverage.

    python make_maps_slides.py [--out ../../PresentationImages] [--only interior_night]

One figure per reconstructed scene (interior studio, interior night, sword & shield
studio, sword & shield night): the three PBR maps in rows (albedo, metallic, roughness),
the authored texture, the recovered one and their difference in columns, each difference
with the colour bar of its own scale, and, on the right, how many cameras cover each texel
of the atlas.  The coverage is the context the difference column is read against: where
few cameras see a texel the per-texel fit is under-determined, and below two cameras it
does not run at all.

Nothing is computed here.  Every panel is a PNG the thesis already prints, taken from
`Doc/images/` exactly as `results.tex` includes it:

  * authored     `table-and-other-specular-png/` and `sword-shield-png/BakedMaterial_*.png`
  * recovered    `results/{interior,sword}/{albedo,metallic,roughness}_{studio,night}.png`
  * difference   `results/{interior,sword}/mapdiff/{map}_{sky}_diff.png` + `{map}_cbar.png`
  * coverage     `visibility/camera_count.png` (interior), `results/sword/camera_count.png`

so the slides and the document show the same pixels.  The maps come from the run
`test_sword_shield_after_fix_irradiance`; the night sword ones are from its softplus
configuration, as the chapter states.  The coverage depends on geometry and rig only, so
one map serves the studio and the night capture of the same scene, as in the thesis.

The authored textures (4096 px for the interior, 1012 px for the sword) are reduced to the
cell size with a box filter, i.e. a block mean, the same treatment the recovered maps got
in `make_results_figures.do_maps`; nothing is renormalised.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from PIL import Image                                 # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
IMG = ROOT / "Doc" / "images"

C_INK = "#33404d"
C_MUTED = "#5f6c79"

MAPS = ("albedo", "metallic", "roughness")
AUTHORED_NAME = {"albedo": "base_color", "metallic": "metallic", "roughness": "roughness"}
FAMILIES = {
    "interior": dict(authored="table-and-other-specular-png", results="results/interior",
                     coverage="visibility/camera_count.png",
                     title="Interior"),
    "sword": dict(authored="sword-shield-png", results="results/sword",
                  coverage="results/sword/camera_count.png",
                  title="Sword & shield"),
}
SKIES = ("studio", "night")
CELL = 1024                       # px of the square cells after the box reduction


def paths(family: str, sky: str) -> dict:
    f = FAMILIES[family]
    p = {"coverage": IMG / f["coverage"]}
    for m in MAPS:
        p[(m, "authored")] = IMG / f["authored"] / f"BakedMaterial_{AUTHORED_NAME[m]}.png"
        p[(m, "recovered")] = IMG / f["results"] / f"{m}_{sky}.png"
        p[(m, "difference")] = IMG / f["results"] / "mapdiff" / f"{m}_{sky}_diff.png"
        p[(m, "cbar")] = IMG / f["results"] / "mapdiff" / f"{m}_cbar.png"
    missing = [str(v) for v in p.values() if not v.exists()]
    if missing:
        raise SystemExit("missing:\n  " + "\n  ".join(missing))
    return p


def load_cell(path: Path, size: int = CELL) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.BOX)
    return np.asarray(im)


def load_rgba_on_white(path: Path) -> np.ndarray:
    """The colour bars and the coverage map carry transparency; flatten it on white."""
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(bg, im).convert("RGB"))


def build(family: str, sky: str, out: Path) -> None:
    p = paths(family, sky)
    cov = load_rgba_on_white(p["coverage"])
    cov_aspect = cov.shape[1] / cov.shape[0]

    # -- layout in figure-width units ---------------------------------------------------
    cell = 0.185                    # width of a square cell
    cbar_w = cell * 173 / 838       # the bars are 173 x 838 px, drawn cell-high
    gap = 0.010
    left = 0.045                    # room for the row labels
    header = 0.028                  # room for the column titles
    top_pad = 0.010
    grid_w = 3 * cell + 2 * gap + cbar_w + 0.004
    cov_h = 0.82 * cell             # the coverage panel: smaller than a cell, the
                                    # textures are what the slide is about
    cov_w = cov_h * cov_aspect
    width_units = left + grid_w + 2.0 * gap + cov_w + 0.006
    body = top_pad + header + 3 * cell + 2 * gap + 0.008
    fig_w = 16.0
    fig_h = fig_w * body / width_units
    fig = plt.figure(figsize=(fig_w, fig_h))
    fw2h = width_units / body       # width units -> height fraction: divide by body

    def add(x, y_top, w, h):
        """x, w in width units [0, width_units]; y_top, h likewise, measured from the top."""
        return fig.add_axes([x / width_units, 1.0 - (y_top + h) / body,
                             w / width_units, h / body])

    y0 = top_pad + header
    cols = (("authored", "authored"), ("recovered", "recovered"), ("difference", "difference"))
    for r, m in enumerate(MAPS):
        y = y0 + r * (cell + gap)
        fig.text((left - 0.01) / width_units, 1.0 - (y + cell / 2) / body, m,
                 fontsize=17, color=C_INK, ha="right", va="center", rotation=90,
                 weight="bold")
        for c, (key, title) in enumerate(cols):
            x = left + c * (cell + gap)
            ax = add(x, y, cell, cell)
            ax.imshow(load_cell(p[(m, key)]), interpolation="lanczos")
            ax.axis("off")
            if r == 0:
                ax.set_title(title, fontsize=17, color=C_INK, pad=8)
        ax = add(left + 3 * cell + 2 * gap + 0.004, y, cbar_w, cell)
        ax.imshow(load_rgba_on_white(p[(m, "cbar")]), interpolation="lanczos")
        ax.axis("off")

    # coverage, centred on the three rows
    x_cov = left + grid_w + 2.0 * gap
    y_cov = y0 + (3 * cell + 2 * gap - cov_h) / 2
    ax = add(x_cov, y_cov, cov_w, cov_h)
    ax.imshow(cov, interpolation="lanczos")
    ax.axis("off")
    ax.set_title("cameras per texel", fontsize=15, color=C_INK, pad=6)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  + {out}   ({width_units / body:.2f}:1)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="../../PresentationImages")
    ap.add_argument("--only", default=None,
                    help="one of interior_studio, interior_night, sword_studio, sword_night")
    a = ap.parse_args()
    out = Path(a.out)
    for family in FAMILIES:
        for sky in SKIES:
            key = f"{family}_{sky}"
            if a.only and key != a.only:
                continue
            build(family, sky, out / f"maps_{key}.png")


if __name__ == "__main__":
    main()
