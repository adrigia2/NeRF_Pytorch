#!/usr/bin/env python
"""envmap_source_size.py -- how many samples it takes to resolve the lights of an envmap.

The irradiance pass draws S = N x N directions uniform in SOLID ANGLE over the
hemisphere (deviceProgramsIrradiance.cu: cos(theta) is exactly equispaced) and weights
each one by cos(theta). Each sample therefore stands for 2*pi/S steradians, and a source
subtending Omega steradians is hit by

    E[hits] = S * Omega / (2*pi)

rays on average, whatever its brightness. Brightness only decides how badly a miss hurts:
a source carrying a fraction f of the incident flux and hit by n rays contributes with a
relative standard error of about f/sqrt(n). So the sample count needed to resolve a light
is set by its ANGULAR SIZE, and the sample count needed for a given accuracy is set by
its size and its share of the energy together.

This script measures both from an actual environment map and prints the N that follows.

    python envmap_source_size.py <envmap.exr> [<envmap2.exr> ...] [--hits 1 10 30]

Comparing the authored HDRI with the one baked from the NeRF is the interesting use: the
network smooths the small bright sources, which spreads their solid angle and makes them
cheaper to hit while also removing the energy that made them worth hitting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import _paths  # noqa: F401
from regen_heatmaps import _load_exr_hw3  # noqa: E402

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
QUANTILES = (0.25, 0.50, 0.75, 0.90, 0.99)


def solid_angles(h: int, w: int) -> np.ndarray:
    """Solid angle of each pixel of an equirectangular map, in steradians.

    Rows span theta in [0, pi], columns phi in [0, 2*pi]; a pixel covers
    (2*pi/w) * (cos(theta_top) - cos(theta_bottom)).
    """
    edges = np.pi * np.arange(h + 1, dtype=np.float64) / h
    band = (np.cos(edges[:-1]) - np.cos(edges[1:])) * (2.0 * np.pi / w)
    return np.repeat(band[:, None], w, axis=1)


def report(path: Path, hits: list[int]) -> None:
    img = _load_exr_hw3(str(path)).astype(np.float64)
    h, w = img.shape[:2]
    dw = solid_angles(h, w)
    lum = img @ LUMA

    flux = lum * dw
    total = float(flux.sum())
    order = np.argsort(lum, axis=None)[::-1]
    cum = np.cumsum(flux.ravel()[order]) / total
    om = np.cumsum(dw.ravel()[order])

    print()
    print(f"{path.name}   {w}x{h}   peak luminance {lum.max():.4g}   "
          f"total flux {total:.4g} W/sr-equivalent")
    print(f"{'energy':>8}{'pixels':>10}{'Omega [sr]':>13}{'as a disc':>12}"
          f"{'peak share':>12}   " + "".join(f"{'N for ' + str(k) + ' hits':>16}"
                                             for k in hits))
    for q in QUANTILES:
        k = int(np.searchsorted(cum, q)) + 1
        omega = float(om[k - 1])
        # Half-angle of the disc with the same solid angle: Omega = 2*pi*(1 - cos(a)).
        alpha = np.degrees(np.arccos(max(1.0 - omega / (2.0 * np.pi), -1.0)))
        ns = []
        for nh in hits:
            s = nh * 2.0 * np.pi / omega
            ns.append(f"{int(np.ceil(np.sqrt(s))):>16d}")
        print(f"{100 * q:>7.0f}%{k:>10d}{omega:>13.3e}{alpha:>11.2f} deg"
              f"{lum.ravel()[order][k - 1] / lum.max():>12.2e}   " + "".join(ns))

    # The single brightest region, defined without an arbitrary energy target: the
    # pixels within a factor of ten of the peak.
    sel = lum >= lum.max() / 10.0
    omega = float(dw[sel].sum())
    share = float((lum[sel] * dw[sel]).sum() / total)
    alpha = np.degrees(np.arccos(max(1.0 - omega / (2.0 * np.pi), -1.0)))
    print(f"  pixels within 10x of the peak: {int(sel.sum())} px, "
          f"Omega {omega:.3e} sr ({alpha:.2f} deg as a disc), {100 * share:.1f}% of the "
          f"flux")
    for nh in hits:
        s = nh * 2.0 * np.pi / max(omega, 1e-12)
        print(f"    {nh:>3} hits on it needs S = {s:.4g} samples, i.e. N = "
              f"{int(np.ceil(np.sqrt(s)))}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("envmaps", nargs="+")
    ap.add_argument("--hits", nargs="*", type=int, default=[1, 10, 30])
    a = ap.parse_args()
    for p in a.envmaps:
        report(Path(p), a.hits)
    return 0


if __name__ == "__main__":
    sys.exit(main())
