#!/usr/bin/env python
"""make_lighting_figure.py -- the two lighting conditions of the Results chapter.

    python make_lighting_figure.py --out ../Doc/images/lighting

Writes four PNGs:

  skybox_studio.png / skybox_night.png     the two environment maps, tonemapped
  heat_studio.png   / heat_night.png       the norm ||c|| of every pixel, in false colour

Two conventions, each chosen for a definite reason.

**The two tonemapped panels do NOT share an exposure.**  They are two different
environments, not two versions of the same scene: a common exposure would turn the night
one into a black rectangle and say nothing about what it looks like.  Each brings its own
median to the reference level, exactly as the rest of the chapter does.

**The two heatmaps do share one, and it is logarithmic.**  Here the subject is the
comparison: it is by looking at the same scale that one sees the night map is overall
dimmer and far more concentrated.  The scale is logarithmic because the dynamic range of
these envmaps spans several orders of magnitude; in linear the heatmap would be black with
a few white dots, so the information that matters — where the energy is — would be lost
exactly where it is interesting.

The mapped value is the pixel's Euclidean norm, not its luminance: what matters here is
how much radiance arrives from that direction, not how an eye would perceive it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import _paths  # noqa: F401

from make_skybox_figure import LUMA_COEFF, block_mean, load_exr, tonemap  # noqa: E402

plt.rcParams.update({"font.size": 13})

DPI = 190
DOWNSAMPLE = 2          # the sources are 4k equirectangular
KEY = 0.5               # level the median is brought to before the Reinhard
FLOOR_PCTL = 1.0        # percentile that sets the floor of the shared scale
CMAP = "inferno"

HDR_DIR = Path("C:/Users/adria/Documents/GitHub/Tesi/OptixProjectCMake/Scenes/"
               "SwordShield Thesis/Blender/assets/hdrs")
MAPS = [("studio", HDR_DIR / "wooden_studio_13_4k.exr"),
        ("night",  HDR_DIR / "cobblestone_street_night_4k.exr")]


def own_exposure(img: np.ndarray) -> float:
    """Exposure that brings the median luminance to KEY.  The median and not the maximum:
    in an envmap the peak sits on the light sources and would dictate the tonemap alone."""
    med = max(float(np.median((img * LUMA_COEFF).sum(-1))), 1e-6)
    return KEY / med


def heat_png(norm: np.ndarray, vmin: float, vmax: float, out: Path, label: str) -> None:
    h, w = norm.shape
    fig, ax = plt.subplots(figsize=(7.2, 7.2 * h / w + 0.9))
    im = ax.imshow(np.maximum(norm, vmin), cmap=CMAP,
                   norm=LogNorm(vmin=vmin, vmax=vmax), interpolation="nearest")
    ax.axis("off")
    cb = fig.colorbar(im, ax=ax, fraction=0.031, pad=0.02, orientation="horizontal")
    cb.set_label(label)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  + {out.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="../Doc/images/lighting")
    ap.add_argument("--downsample", type=int, default=DOWNSAMPLE)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    imgs, norms = {}, {}
    for name, p in MAPS:
        if not p.exists():
            raise SystemExit(f"✗ not found: {p}")
        a = block_mean(load_exr(p), args.downsample)
        imgs[name] = a
        norms[name] = np.linalg.norm(a, axis=-1)
        n = norms[name]
        print(f"{name:7s} {a.shape[1]}x{a.shape[0]}  ||c||: p50={np.median(n):.4f}  "
              f"p99.9={np.percentile(n, 99.9):9.3f}  max={n.max():10.3f}  "
              f"mean={n.mean():.4f}")

    # Shared scale.  The ceiling is the maximum of the two, because the night map's peak
    # is precisely the phenomenon to show; the floor is a low percentile and not a fixed
    # number of decades below the ceiling.  With fixed decades that peak (~7·10^4, two
    # hundred times the studio maximum) would drag the bottom of the scale above the
    # median of both maps, flattening everything else into a single colour: the bar would
    # cover a range the data is almost never in.
    allv = np.concatenate([norms[n].ravel() for n, _ in MAPS])
    vmax = float(allv.max())
    vmin = max(float(np.percentile(allv, FLOOR_PCTL)), vmax * 1e-9)
    print(f"\nshared heatmap scale: [{vmin:.3e}, {vmax:.3e}]  "
          f"({np.log10(vmax / vmin):.1f} decades, logarithmic, "
          f"floor at the joint p{FLOOR_PCTL})")

    for name, _ in MAPS:
        save = out / f"skybox_{name}.png"
        expo = own_exposure(imgs[name])
        plt.imsave(save, np.clip(tonemap(imgs[name], expo), 0.0, 1.0))
        print(f"  + {save.name}  (own exposure {expo:.4g})")
        heat_png(norms[name], vmin, vmax, out / f"heat_{name}.png",
                 r"$\|\mathbf{c}\|$  (log scale, shared)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
