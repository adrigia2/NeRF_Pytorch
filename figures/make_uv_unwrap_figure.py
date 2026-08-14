#!/usr/bin/env python
"""make_uv_unwrap_figure.py -- the two panels of figure 3.3.

    python make_uv_unwrap_figure.py <run_dir> --world C:/.../positional_image.exr \
        --out ../Doc/images/ium

Writes two PNGs:

  uv_unwrap_world.png   the mesh drawn from the world-space positions of its vertices
  uv_unwrap_uv.png      the same mesh at its UV coordinates, i.e. the run's ium_positions,
                        which is the world position written into texture space

**The mapping is the identity, not a normalization.**  The two panels show a coordinate,
not a radiance: rescaling per channel over the extent of the geometry, as `position_rgb`
in make_depth_figure.py does, would produce two images with two different factors, and
the reader would compare them believing the same colour meant the same point.  Here the
value goes to the pixel as it is: negatives are brought to zero and anything above 1
saturates, which is all an 8-bit PNG can do.  No gamma, for the same reason there is
none in the depth-layer figure.

A consequence to keep in mind when reading the figure: the parts of the scene at a
negative coordinate come out black in that channel, and those beyond 1 saturate.  That is
deliberate: it says where the geometry sits relative to the origin, which a rescaled
version hides.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import _paths  # noqa: F401

from make_depth_figure import position_rgb, save_png   # noqa: E402
from make_skybox_figure import load_exr                # noqa: E402

# The mapping lives in `position_rgb` (make_depth_figure.py) and is not duplicated here:
# it is the same convention as every position map in the thesis, and having two
# implementations would mean being able to let them diverge unnoticed.


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="run to take ium/ium_positions.exr from")
    ap.add_argument("--world", required=True,
                    help="EXR of the mesh drawn from its vertex positions")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    run, out = Path(args.run_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{run.name} → {out.resolve()}")

    print("world space:")
    world = load_exr(Path(args.world))
    save_png(position_rgb(world), out / "uv_unwrap_world.png")

    print("UV space:")
    ium = load_exr(run / "ium" / "ium_positions.exr")
    mask = load_exr(run / "ium" / "ium_masks.exr")[..., 0] > 0.5
    save_png(position_rgb(ium, mask), out / "uv_unwrap_uv.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
