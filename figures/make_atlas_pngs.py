#!/usr/bin/env python
"""make_atlas_pngs.py -- convert the channels of a Blender bake into PNGs for the thesis.

    python make_atlas_pngs.py <bake folder> --out <folder> [--channels roughness]

The conventions are not arbitrary: they reproduce those of the PNGs already in
Doc/images/, measured against the source EXRs.

  1. The base colour is the only COLOUR channel and has to be sRGB encoded (the exact
     curve, not gamma 2.2: that is what reproduces the existing files to within 0.0065).
     The other three are DATA, not colour, and must be written linear: applying a gamma
     to a roughness shows a value that is not the one the renderer used.

  2. The 8192 -> 4096 downsample is a block mean in linear space.  Point subsampling
     would throw away three quarters of the authored signal and alias the rest, which is
     the same reason the ground truth is reduced this way.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _paths  # noqa: F401

from make_skybox_figure import block_mean, load_exr

CHANNELS = ("base_color", "metallic", "roughness", "normal")
# Only the base colour is colour: the others are data and stay linear.
SRGB_CHANNELS = {"base_color"}


def srgb(x: np.ndarray) -> np.ndarray:
    """sRGB encoding (IEC 61966-2-1), not gamma 2.2."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bake_dir", help="folder holding the BakedMaterial_<channel>.exr files")
    ap.add_argument("--out", required=True, help="destination folder for the PNGs")
    ap.add_argument("--channels", nargs="+", default=list(CHANNELS),
                    help=f"channels to convert (default: {' '.join(CHANNELS)})")
    ap.add_argument("--downsample", type=int, default=2,
                    help="block mean (default 2: 8192x8192 -> 4096x4096)")
    args = ap.parse_args()

    bake, out = Path(args.bake_dir), Path(args.out)
    if not bake.is_dir():
        print(f"ERROR: {bake} is not a folder")
        return 2
    out.mkdir(parents=True, exist_ok=True)

    for ch in args.channels:
        src = bake / f"BakedMaterial_{ch}.exr"
        if not src.exists():
            print(f"ERROR: {src} does not exist")
            return 2
        a = block_mean(load_exr(src), args.downsample)
        rgb = srgb(a) if ch in SRGB_CHANNELS else np.clip(a, 0.0, 1.0)
        dst = out / f"BakedMaterial_{ch}.png"
        plt.imsave(dst, rgb)
        print(f"  + {dst}  {rgb.shape[1]}x{rgb.shape[0]}  "
              f"range [{a.min():.3f}, {a.max():.3f}]  "
              f"{'sRGB' if ch in SRGB_CHANNELS else 'linear'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
