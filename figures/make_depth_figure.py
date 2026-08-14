#!/usr/bin/env python
"""make_depth_figure.py -- the geometric layers of a run, as PNGs for the thesis.

    python make_depth_figure.py <run_dir> --camera render_Camera_Shell21_38 --out DIR
    python make_depth_figure.py <run_dir> --ium --out DIR

The first form writes the three per-camera panels of the depth pass: depth.png (false
colour), position.png, mask.png.  The second writes the two texture-space panels of the
IUM pass: ium_position.png and ium_mask.png.

Two choices are not a matter of taste:

  1. The depth normalization uses the MASK, not a threshold on the value.  On the
     background the file carries 1e20, the tracer's miss value: a min/max over the whole
     image would push the entire foreground into a single colour.

  2. The background of the depth panel is not a depth value and must not look like one.
     It is rendered in a neutral grey, distinguishable from both ends of the ramp and
     from the white of the page, rather than crushed onto one end of the colormap.

The colormap is viridis: perceptually uniform, so an equal colour difference corresponds
to an equal depth difference, which on a depth map is exactly what one wants to read.

"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import _paths  # noqa: F401

from make_skybox_figure import load_exr

# Background grey of the depth panel: lighter than viridis' yellow end would be
# unreadable, darker would be confused with the purple end.
BG_GREY = 0.75


def save_png(rgb: np.ndarray, out: Path) -> None:
    plt.imsave(out, np.clip(rgb, 0.0, 1.0))
    print(f"  + {out}  ({rgb.shape[1]}x{rgb.shape[0]})")


def content_box(mask: np.ndarray, margin: float) -> tuple[slice, slice]:
    """Bounding box of the foreground plus a margin, ONE for all the panels.

    Printed at a third of a line, the three full-frame panels show the subject over a
    quarter of the height.  The crop has to be identical in all three, otherwise they are
    no longer comparable pixel by pixel, which is the whole reason they sit side by side.
    """
    ys, xs = np.where(mask)
    h, w = mask.shape
    my = int(margin * (ys.max() - ys.min() + 1))
    mx = int(margin * (xs.max() - xs.min() + 1))
    return (slice(max(ys.min() - my, 0), min(ys.max() + 1 + my, h)),
            slice(max(xs.min() - mx, 0), min(xs.max() + 1 + mx, w)))


def position_rgb(pos: np.ndarray, mask: "np.ndarray | None" = None) -> np.ndarray:
    """Position mapped to RGB by the IDENTITY, black background.

    No rescaling and no gamma: the value goes to the pixel as it is, negatives are brought
    to zero and anything above one saturates.  This function used to rescale per channel
    over the extent of the geometry: that made each panel readable on its own, but with a
    different factor from every other, so two figures showing the same scene were not
    comparable and the same colour did not mean the same point.  With the identity the
    colour is the coordinate, which is the only reading anyone looking at a position map
    needs.

    `mask` is optional: without it the whole frame is mapped (the world-space panel has no
    mask to apply).  It prints the extremes and how much gets clipped: those are the
    numbers the caption has to quote.
    """
    sel = mask if mask is not None else np.ones(pos.shape[:2], bool)
    lo, hi = pos[sel].min(axis=0), pos[sel].max(axis=0)
    print("  position " + "  ".join(f"{a}[{lo[i]:.2f}, {hi[i]:.2f}]"
                                    for i, a in enumerate("xyz")))
    print(f"  clamp: brought to 0 {100.0 * (pos[sel] < 0).mean():.2f}%, "
          f"saturated at 1 {100.0 * (pos[sel] > 1).mean():.2f}%")
    rgb = np.clip(pos, 0.0, 1.0).astype(np.float32)
    if mask is not None:
        rgb[~mask] = 0.0
    return rgb


def normal_rgb(nrm: np.ndarray, mask: "np.ndarray | None" = None) -> np.ndarray:
    """Normal mapped to RGB with the normal-map encoding, black background.

    It takes $[-1, 1]$ to $[0, 1]$ with 0.5 + 0.5*n, which is how normal maps are written
    and how this pipeline reads the external one (`external_normal_range="0_1"`), so the
    panel is comparable with the map that replaces it.  The position clamp is NOT used:
    it would zero every negative component and make every face pointing towards -x, -y or
    -z disappear, i.e. half the geometry.

    It normalises before encoding.  The kernel builds the normal as the cross product of
    the triangle edges and the consumers normalise it at the point of use
    (deviceProgramsIrradiance.cu does so explicitly), so the buffer is not guaranteed to
    be unit length: without normalising, the colour would say the triangle's area instead
    of its direction.
    """
    n = np.linalg.norm(nrm, axis=-1, keepdims=True)
    unit = np.divide(nrm, n, out=np.zeros_like(nrm), where=n > 1e-8)
    degenerate = float((n[..., 0] <= 1e-8).mean())
    print(f"  normal   |n| in [{n.min():.3f}, {n.max():.3f}], "
          f"degenerate {100.0 * degenerate:.2f}%")
    rgb = np.clip(0.5 + 0.5 * unit, 0.0, 1.0).astype(np.float32)
    if mask is not None:
        rgb[~mask] = 0.0
    return rgb


def ium_panels(run: Path, out: Path) -> int:
    """The texture-space layers: per-texel position and coverage mask.

    The normal is NOT written: when the pipeline is given an external normal map, that map
    overwrites the geometric normal inside the pass buffer before saving, so
    ium_normals.exr holds the supplied map, not the computed one.
    """
    paths = {"position": run / "ium" / "ium_positions.exr",
             "mask":     run / "ium" / "ium_masks.exr"}
    for p in paths.values():
        if not p.exists():
            print(f"ERROR: {p} does not exist")
            return 2

    pos = load_exr(paths["position"])
    mask = load_exr(paths["mask"])[..., 0] > 0.5
    print(f"atlas {mask.shape[1]}x{mask.shape[0]}, coverage {100 * mask.mean():.2f}% "
          f"({mask.sum():,} texels)")

    save_png(position_rgb(pos, mask), out / "ium_position.png")
    save_png(np.repeat(mask[..., None].astype(np.float32), 3, axis=-1),
             out / "ium_mask.png")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="the run folder (holds depth/, position/, mask/)")
    ap.add_argument("--camera", default=None, help="frame name, without the suffix")
    ap.add_argument("--out", required=True, help="destination folder for the PNGs")
    ap.add_argument("--ium", action="store_true",
                    help="write the texture-space layers instead of the per-camera ones")
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--margin", type=float, default=0.06,
                    help="margin around the foreground, as a fraction of the box "
                         "(default 0.06; 0 disables the crop)")
    args = ap.parse_args()

    run, out, cam = Path(args.run_dir), Path(args.out), args.camera
    out.mkdir(parents=True, exist_ok=True)

    if args.ium:
        return ium_panels(run, out)
    if not cam:
        print("ERROR: --camera is required for the per-camera layers")
        return 2

    paths = {
        "depth":    run / "depth" / f"{cam}_depth.exr",
        "position": run / "position" / f"{cam}_position.exr",
        "mask":     run / "mask" / f"{cam}_mask.png",
    }
    for k, p in paths.items():
        if not p.exists():
            print(f"ERROR: {p} does not exist")
            return 2

    depth = load_exr(paths["depth"])[..., 0]
    pos = load_exr(paths["position"])
    mask_raw = mpimg.imread(paths["mask"])
    mask = (mask_raw if mask_raw.ndim == 2 else mask_raw[..., 0]) > 0.5

    print(f"{cam}: {mask.shape[1]}x{mask.shape[0]}, foreground {100 * mask.mean():.1f}%")

    d0, d1 = float(depth[mask].min()), float(depth[mask].max())
    # Extremes and crop are computed on the whole frame: the crop only decides what is
    # visible, not how the values are mapped.
    if args.margin > 0:
        rows, cols = content_box(mask, args.margin)
        print(f"  crop [{cols.start}:{cols.stop}, {rows.start}:{rows.stop}] "
              f"= {cols.stop - cols.start}x{rows.stop - rows.start}")
        depth, pos, mask = depth[rows, cols], pos[rows, cols], mask[rows, cols]

    print(f"  depth  [{d0:.3f}, {d1:.3f}]  median {float(np.median(depth[mask])):.3f}")
    norm = np.clip((depth - d0) / max(d1 - d0, 1e-9), 0.0, 1.0)
    rgb = matplotlib.colormaps[args.cmap](norm)[..., :3]
    rgb[~mask] = BG_GREY
    save_png(rgb, out / "depth.png")

    save_png(position_rgb(pos, mask), out / "position.png")

    save_png(np.repeat(mask[..., None].astype(np.float32), 3, axis=-1), out / "mask.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
