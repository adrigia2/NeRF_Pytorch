#!/usr/bin/env python
"""make_time_to_quality_figure.py -- convergence grids for the "Time to Quality" section.

Reads the sandboxes ttq_sweep.py produced, crops them to the ROI and writes one PNG per
panel of two grids. Each row is a refinement step: the map at N, the relative change it
undergoes when N grows, and the map at the higher N. The middle panel of a row is the
left panel of the next one, so a reader follows the same texel down the ladder.

    python make_time_to_quality_figure.py <run_dir> [--out DIR]
           [--irr-tags ttq_irr512 ttq_irr1024 ttq_irr2048 ttq_irr4096]
           [--ind-tags ttq_ind32 ttq_ind64 ttq_ind128]
           [--decades 3] [--top-pctl 99.5] [--max-side 1024]

Three conventions are what make the figure readable, and none of them is optional:

  1. ONE exposure for every image panel of a grid, taken from the finest level and from
     the active texels only. With one exposure per panel the tonemap would rescale each
     level back to the same apparent brightness and the figure would show a convergence
     that has already been divided out. Same rule as make_scenes_figure.column_exposure.

  2. ONE normalisation for every heatmap of a grid, with a single bar. The claim of the
     figure is that each refinement changes LESS than the one before it, and two
     independent colour scales would erase exactly that comparison.

  3. The heatmap is a RELATIVE difference, ||A - B|| / max(||B||, floor), on the linear
     values. The albedo divides by the irradiance, so a relative error on E is, to first
     order, the same relative error on the delivered map; an absolute difference would be
     dominated by the texels that see the lamp and would say nothing about the ones in
     shadow, which are most of the ROI. The floor is albedo_eps read from the manifest:
     below it the pipeline clamps and the albedo stops depending on E at all.

The script also prints the rows of the convergence table: percentiles of the relative
change, the fraction of texels still moving by more than one per cent, and the measured
cost of each level taken from run_timings.jsonl.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import _paths  # noqa: F401
from make_results_figures import _colorbar_png, _heat_png, diff_norm  # noqa: E402
from make_skybox_figure import LUMA_COEFF, block_mean, load_exr, tonemap  # noqa: E402

KEY = 0.18
EPS = 1e-12

FAMILY = {
    "irr": ("irradiance.exr", "direct irradiance", "N"),
    "ind": ("irradiance_indirect.exr", "indirect irradiance", "N"),
}


def exposure_on(img: np.ndarray, mask: np.ndarray, key: float = KEY) -> float:
    """Reinhard exposure from the median luminance of the ACTIVE texels.

    An atlas crop is partly empty and the empty texels are absence of surface, not dark
    surface: including them in the median drags the exposure to the clamp and washes the
    panel out.
    """
    lum = (img * LUMA_COEFF).sum(-1)
    v = lum[mask]
    v = v[v > 0.0]
    med = max(float(np.median(v)) if v.size else 0.0, 1e-6)
    return key / med


def nice_ceiling(x: float) -> float:
    """Round up to 1, 2 or 5 times a power of ten.

    Rounding to the next decade instead would throw away most of a decade of the scale
    whenever the data stop just above one: a ceiling of 1 on a p99.5 of 0.15 spends two
    thirds of the colour ramp on values that never occur.
    """
    if x <= 0:
        return 1.0
    e = np.floor(np.log10(x))
    for m in (1.0, 2.0, 5.0):
        if x <= m * 10.0 ** e:
            return m * 10.0 ** e
    return 10.0 ** (e + 1)


def load_level(run: Path, tag: str, name: str):
    """(crop of the map, crop of the IUM mask, rect) for one sandbox."""
    sandbox = run / "roi" / tag
    rect = json.loads((sandbox / "roi.json").read_text(encoding="utf-8"))["rect"]
    x0, y0, w, h = rect
    sl = (slice(y0, y0 + h), slice(x0, x0 + w))
    img = load_exr(sandbox / "irradiance" / name)[sl]
    mask = load_exr(sandbox / "ium" / "ium_masks.exr")[..., 0][sl] > 0.5
    return img, mask, rect


def pass_seconds(run: Path, tag: str, key: str) -> float | None:
    p = run / "roi" / tag / "run_timings.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        return None
    return ok[-1]["substeps"].get(key)


def level_value(tag: str) -> int:
    """The sample side from the tag: ttq2_irr2048 -> 2048, ttq2_irr512b -> 512.

    The trailing run of digits, not every digit in the string: a tag prefix may carry a
    number of its own (ttq2), and collecting all of them turned 512 into 2512.
    """
    m = re.search(r"(\d+)b?$", tag)
    if not m:
        raise ValueError(f"no sample count in the tag {tag!r}")
    return int(m.group(1))


def albedo_eps(run: Path) -> float:
    m = run / "run_manifest.json"
    if not m.exists():
        return 1e-3
    cfg = json.loads(m.read_text(encoding="utf-8"))["config"]["render"]
    return float(cfg.get("albedo_eps", 1e-3))


def grid(run: Path, out: Path, tags: list[str], family: str, args) -> None:
    name, label, _ = FAMILY[family]
    substep = "step3/irradiance" if family == "irr" else "step3/indirect"
    floor = albedo_eps(run)

    levels, mask, rect = [], None, None
    for t in tags:
        img, m, rect = load_level(run, t, name)
        levels.append(img)
        mask = m if mask is None else (mask & m)
    assert mask is not None

    down = 1
    while max(mask.shape) // down > args.max_side:
        down *= 2
    if down > 1:
        print(f"  block mean x{down}: {mask.shape[0]} -> {mask.shape[0] // down}")

    # Rule 1: one exposure, from the finest level.
    expo = exposure_on(levels[-1], mask)
    print(f"  exposure {expo:.4g} (from {tags[-1]}, active texels only)")

    def reduce(a):
        return block_mean(a, down) if down > 1 else a

    small_mask = (block_mean(mask.astype(np.float32)[..., None], down)[..., 0] > 0.5
                  if down > 1 else mask)

    for t, img in zip(tags, levels):
        rgb = tonemap(reduce(img * mask[..., None]), expo)
        rgb[~small_mask] = 1.0                        # empty texels white, not black
        plt.imsave(out / f"{family}_n{level_value(t)}.png", np.clip(rgb, 0, 1))
        print(f"  + {family}_n{level_value(t)}.png")

    # Rule 3: relative change between consecutive levels, on the linear values.
    diffs = []
    for a, b in zip(levels, levels[1:]):
        d = diff_norm(a, b) / np.maximum(np.linalg.norm(b, axis=-1), floor)
        diffs.append(reduce(d[..., None])[..., 0] if down > 1 else d)

    # Rule 2: one normalisation for the whole grid.
    pooled = np.concatenate([d[small_mask].ravel() for d in diffs])
    top = float(np.percentile(pooled, args.top_pctl))
    vmax = nice_ceiling(top)
    vmin = vmax / 10.0 ** args.decades
    print(f"  heat scale [{vmin:.3g}, {vmax:.3g}]  "
          f"(p{args.top_pctl:g} of the pooled differences = {top:.3g})")

    for (a, b), d in zip(zip(tags, tags[1:]), diffs):
        _heat_png(d, vmin, vmax,
                  out / f"{family}_diff_{level_value(a)}_{level_value(b)}.png",
                  log=True, bad=~small_mask)
    _colorbar_png(vmin, vmax, out / f"{family}_cbar.png",
                  r"relative change  $\|\Delta E\|_2 / \|E\|_2$", log=True)

    # ── table ────────────────────────────────────────────────────────────────
    print()
    print(f"  {label}: convergence table (ROI {rect}, "
          f"{int(mask.sum())} active texels)")
    hdr = (f"    {'N':>6}{'samples':>12}{'pass [s]':>10}{'x cost':>8}"
           f"{'p50':>10}{'p90':>10}{'p99':>10}{'>1%':>8}{'p50 vs finest':>16}")
    print(hdr)
    t0 = pass_seconds(run, tags[0], substep)
    finest = levels[-1]
    for i, t in enumerate(tags):
        n = level_value(t)
        secs = pass_seconds(run, t, substep)
        cost = f"{secs / t0:.1f}" if (secs and t0) else "-"
        if i < len(diffs):
            r = diffs[i][small_mask]
            cols = (f"{np.median(r):>10.2e}{np.percentile(r, 90):>10.2e}"
                    f"{np.percentile(r, 99):>10.2e}{100 * (r > 0.01).mean():>7.1f}%")
        else:
            cols = f"{'-':>10}{'-':>10}{'-':>10}{'-':>8}"
        if i < len(tags) - 1:
            dv = (diff_norm(levels[i], finest)
                  / np.maximum(np.linalg.norm(finest, axis=-1), floor))[mask]
            vs = f"{np.median(dv):>16.2e}"
        else:
            vs = f"{'0 (reference)':>16}"
        print(f"    {n:>6}{n * n:>12}"
              f"{(f'{secs:.1f}' if secs else '-'):>10}{cost:>8}{cols}{vs}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--irr-tags", nargs="*", default=[])
    ap.add_argument("--ind-tags", nargs="*", default=[])
    ap.add_argument("--decades", type=float, default=3.0)
    ap.add_argument("--top-pctl", type=float, default=99.5)
    ap.add_argument("--max-side", type=int, default=1024)
    a = ap.parse_args(argv)

    run = Path(a.run_dir).resolve()
    out = Path(a.out) if a.out else (Path(__file__).resolve().parents[2]
                                     / "Doc" / "images" / "results" / "time-to-quality")
    out.mkdir(parents=True, exist_ok=True)
    print(f"run: {run}")
    print(f"out: {out}")

    for family, tags in (("irr", a.irr_tags), ("ind", a.ind_tags)):
        if len(tags) < 2:
            continue
        print()
        print(f"== {FAMILY[family][1]}: {' -> '.join(tags)}")
        grid(run, out, tags, family, a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
