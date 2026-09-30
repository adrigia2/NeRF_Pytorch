#!/usr/bin/env python
"""make_presentation_pngs.py -- turn HDR texture-space atlases into slide-ready PNGs.

    python make_presentation_pngs.py <file.exr|dir> [...] --out DIR
                                     [--mask ium_masks.exr] [--key 0.18]
                                     [--name NAME] [--sum NAME] [--downsample 1]
                                     [--transfer gamma22|srgb]
                                     [--variants tonemapped clamp01] [--force]

A directory argument expands to the ``*.exr`` files it contains, so a whole
``irradiance/`` folder converts with a single argument.  Sources are opened read only and
nothing is ever written outside ``--out``.

Two variants per source, because they answer two different questions and therefore cannot
share a convention:

  <stem>_tonemapped.png  Reinhard + gamma 2.2 with an exposure derived FROM THAT IMAGE, so
                         every map shows all of its detail.  Brightness is no longer
                         comparable between maps -- that is what the other variant is for.

  <stem>_clamp01.png     np.clip(x, 0, 1) and nothing else: no exposure, no gamma, no mask.
                         Every map goes through the identical absolute transfer, so the
                         difference in MAGNITUDE between them is what the image shows.  Any
                         normalisation here would destroy the only reason this variant
                         exists.

The exposure is the median luminance of the ACTIVE, POSITIVE texels, not of the whole
square: a UV atlas is roughly half empty, that median would be 0, and the exposure would
blow the panel to white.  This is ``make_roi_preview.exposure_on``, reused as is.

``--transfer`` picks the encoding of the tonemapped variant, applied to the Reinhard
output: ``gamma22`` (default, ``x ** (1/2.2)``) or ``srgb``, the exact IEC 61966-2-1 curve
of ``make_atlas_pngs.srgb``.  The two agree to within a few thousandths above middle grey
and differ only near black, which sRGB lifts through its linear segment; sRGB is the one a
colour-managed viewer or a projector assumes, so it is the safer choice for slides.  The
file is named ``<stem>_reinhard_srgb.png`` against ``<stem>_reinhard_gamma22.png``, so the
two encodings coexist and the name says which is which.  The
exposure does not depend on the curve: both encodings of a map share it.

The clamp variant is deliberately left out of this: it is a raw linear clip whose whole
point is the absolute magnitude, and a transfer curve on top of it would be a second
nonlinearity between the data and the reader.

``--name NAME`` renames the output of a single input, for when the file name on disk is
not the one the slide wants (``albedo.exr`` -> ``albedo_lambertian_*.png``).

``--sum NAME`` adds the inputs together and emits that single map instead of one per file
(``--sum irradiance_full`` on the ``irradiance/`` folder gives E_dir + E_ind).  The sum is
taken on the LINEAR radiometric values, before exposure and tonemap: irradiance is
additive only there, and adding tonemapped images would be adding two different nonlinear
remappings of it.  Same reason the optional ``--downsample`` block mean runs in linear
space before the tonemap -- resampling after it, or with a non-conservative filter,
changes the mean radiance of the small bright texels, the ones carrying the energy.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _paths  # noqa: F401

from make_skybox_figure import LUMA_COEFF, block_mean, load_exr, tonemap
from make_roi_preview import exposure_on
from make_atlas_pngs import srgb

VARIANTS = ("tonemapped", "clamp01")
TRANSFERS = ("gamma22", "srgb")
KEY = 0.18          # middle grey, same value make_roi_preview uses on these atlases

# The file name states the OPERATOR, not the intent: "tonemapped" says nothing about which
# curve produced the pixels, and someone opening the folder months later has to be able to
# tell the two encodings apart without going back to the script.
OUTPUT_TAG = {("tonemapped", "gamma22"): "reinhard_gamma22",
              ("tonemapped", "srgb"):    "reinhard_srgb",
              ("clamp01",    "gamma22"): "clamp01",
              ("clamp01",    "srgb"):    "clamp01"}


def display(img: np.ndarray, exposure: float, transfer: str) -> np.ndarray:
    """Reinhard at `exposure`, then the requested encoding.  `gamma22` goes through
    make_skybox_figure.tonemap unchanged, so that path stays bit-identical to the rest of
    the figures; `srgb` splits the two steps because the sRGB curve is not a power."""
    if transfer == "gamma22":
        return tonemap(img, exposure)
    y = img * exposure
    return srgb(y / (1.0 + y))


def expand(inputs: list[str]) -> list[Path]:
    """Files as given, directories expanded to the .exr they contain, sorted, no dups."""
    out: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            out.extend(sorted(p.glob("*.exr")))
        elif p.is_file():
            out.append(p)
        else:
            raise SystemExit(f"ERROR: {p} does not exist")
    seen, uniq = set(), []
    for p in out:
        r = p.resolve()
        if r not in seen:
            seen.add(r)
            uniq.append(p)
    return uniq


def load_mask(path: Path | None, shape: tuple[int, int]) -> np.ndarray:
    """(H, W) bool.  Without a mask every texel is active and exposure_on's own `v > 0`
    filter already drops the empty ones."""
    if path is None:
        return np.ones(shape, dtype=bool)
    m = load_exr(path)[..., 0] > 0.5
    if m.shape != shape:
        raise SystemExit(f"ERROR: mask is {m.shape}, atlas is {shape}")
    return m


def sources(srcs: list[Path], sum_name: str | None,
            name: str | None) -> Iterator[tuple[str, np.ndarray]]:
    """(output stem, linear HxWx3) per map to emit."""
    if not sum_name:
        if name and len(srcs) != 1:
            raise SystemExit(f"ERROR: --name takes one input, got {len(srcs)}")
        for p in srcs:
            yield (name or p.stem), load_exr(p)
        return
    total = None
    for p in srcs:
        a = load_exr(p)
        if total is None:
            total = a
        elif a.shape != total.shape:
            raise SystemExit(f"ERROR: {p.name} is {a.shape}, expected {total.shape}")
        else:
            total = total + a
        print(f"  + {p.name}")
    yield sum_name, total


def emit(name: str, img: np.ndarray, mask: np.ndarray, out: Path,
         variants: list[str], key: float, transfer: str, force: bool) -> None:
    for variant in variants:
        dst = out / f"{name}_{OUTPUT_TAG[(variant, transfer)]}.png"
        if dst.exists() and not force:
            print(f"  = {dst.name} already there, skipped (--force to redo)")
            continue
        if variant == "tonemapped":
            expo = exposure_on(img, mask, key)
            rgb = display(img, expo, transfer)
            # Reported on the ACTIVE texels, the same set exposure_on works on: an atlas is
            # about half empty, and those black texels would drag the median well below the
            # level the exposure actually placed the signal at.
            lum = (rgb * LUMA_COEFF).sum(-1)[mask & (img > 0.0).any(-1)]
            note = (f"{transfer}, exposure {expo:.4g}, median luminance in the PNG "
                    f"{float(np.median(lum)):.3f}")
        else:
            rgb = np.clip(img, 0.0, 1.0)
            clipped = (img > 1.0).any(-1)
            note = (f"linear clip, {100.0 * clipped.mean():5.2f}% of the atlas "
                    f"saturated ({100.0 * clipped[mask].mean():.2f}% of the active texels)")
        plt.imsave(dst, rgb)
        print(f"  + {dst.name}  {rgb.shape[1]}x{rgb.shape[0]}  {note}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="EXR files, or folders holding them")
    ap.add_argument("--out", required=True, help="destination folder for the PNGs")
    ap.add_argument("--mask", default=None,
                    help="ium_masks.exr; restricts the exposure to the active texels")
    ap.add_argument("--key", type=float, default=KEY,
                    help=f"level the median luminance is brought to (default {KEY})")
    ap.add_argument("--name", default=None, metavar="NAME",
                    help="output stem for a single input (default: the file name)")
    ap.add_argument("--sum", dest="sum_name", default=None, metavar="NAME",
                    help="add the inputs in linear space and emit that map as NAME")
    ap.add_argument("--downsample", type=int, default=1,
                    help="block mean in linear space before the tonemap (default 1: none)")
    ap.add_argument("--transfer", default=TRANSFERS[0], choices=TRANSFERS,
                    help="encoding of the tonemapped variant (default gamma22)")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--force", action="store_true", help="overwrite existing PNGs")
    a = ap.parse_args(argv)

    srcs = expand(a.inputs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mask_path = Path(a.mask) if a.mask else None
    if mask_path is not None and not mask_path.exists():
        raise SystemExit(f"ERROR: {mask_path} does not exist")

    what = f"{len(srcs)} source(s)" + (f" summed into '{a.sum_name}'" if a.sum_name else "")
    print(f"{what} -> {out}")
    for name, img in sources(srcs, a.sum_name, a.name):
        print(f"\n{name}")
        h, w = img.shape[:2]
        m = load_mask(mask_path, (h, w))
        act = m & (img > 0.0).any(-1)
        v = img[act]
        print(f"  {w}x{h}  active {100.0 * act.mean():5.2f}%  "
              f"range [{float(v.min()):.4g}, {float(v.max()):.4g}]")

        if a.downsample > 1:
            img = block_mean(img, a.downsample)
            m = block_mean(m.astype(np.float32)[..., None], a.downsample)[..., 0] > 0.5
            print(f"  block mean x{a.downsample} -> {img.shape[1]}x{img.shape[0]}")

        emit(name, img, m, out, a.variants, a.key, a.transfer, a.force)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
