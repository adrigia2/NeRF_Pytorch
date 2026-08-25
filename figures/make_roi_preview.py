#!/usr/bin/env python
"""make_roi_preview.py -- browse the irradiance atlas to choose a test ROI.

Writes a set of labelled PNGs of a run's texture-space irradiance so that a rectangle
can be picked by eye and then handed to the ROI machinery as ``--rect X0 Y0 W H``.  The
axes are in IUM texels with the same convention the pipeline uses: ``_load_roi`` builds
``rect[y0:y1, x0:x1]`` on an ``(ium_h, ium_w)`` array and flattens it row-major, so y is
counted from the TOP of the atlas, exactly as the images below are displayed.

    python make_roi_preview.py <run_dir> [--out DIR] [--zoom X0 Y0 W H]
                               [--rect X0 Y0 W H [--rect ...]] [--label NAME ...]

Four panels, because two of them decide whether raising the sample count can help at all:

  direct     E_dir, tonemapped.  What the OptiX skybox quadrature produces.
  indirect   E_ind, tonemapped.  What the NeRF bounce pass produces.
  structure  mean |grad log10 E_dir| over a 3x3 neighbourhood.  A quadrature error shows
             up where the integrand has structure; on a region where the irradiance is
             smooth, more samples change nothing and the residual error is bias, not
             variance.
  share      E_ind / (E_dir + E_ind).  Where this is small the indirect term is a few per
             cent of the denominator of the albedo, and refining it cannot move the
             delivered map however noisy it is.

Each map carries its OWN exposure, printed on the panel: this is a selection tool, not a
thesis figure, and readability beats comparability here.  The thesis panels are produced
by make_time_to_quality_figure.py, which shares one exposure by design.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import _paths  # noqa: F401
from make_skybox_figure import block_mean, load_exr, tonemap, LUMA_COEFF

EPS = 1e-8
KEY = 0.18


def exposure_on(img: np.ndarray, mask: np.ndarray, key: float = KEY) -> float:
    """Reinhard exposure from the median luminance of the ACTIVE texels only.

    make_scenes_figure.exposure_of takes the median of the whole frame, which is right
    for a render: there every pixel carries signal.  An atlas crop is mostly empty, so
    that median is 0, it gets clamped to the 1e-4 floor and the exposure blows every
    panel to white.  The empty texels are not dark texture, they are absence of texture.
    """
    lum = (img * LUMA_COEFF).sum(-1)
    v = lum[mask] if mask.any() else lum.ravel()
    v = v[v > 0.0]
    med = max(float(np.median(v)) if v.size else 0.0, 1e-6)
    return key / med


def _load(run: Path, name: str) -> np.ndarray:
    p = run / "irradiance" / name
    if not p.exists():
        raise SystemExit(f"missing: {p}")
    print(f"  reading {p.name} ...", flush=True)
    return load_exr(p)


def _mask(run: Path, shape: tuple[int, int]) -> np.ndarray:
    p = run / "ium" / "ium_masks.exr"
    if not p.exists():
        print(f"  ! {p} missing, no mask applied")
        return np.ones(shape, dtype=bool)
    return load_exr(p)[..., 0] > 0.5


def structure(e: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """mean |grad log10 E| over the 4-neighbourhood, on the luminance.

    log10 and not the linear value: the quantity that matters is the RELATIVE variation,
    because the albedo divides by E and a fixed absolute step means something different
    on a texel in shadow and on one facing the lamp.
    """
    lum = np.log10(np.maximum(e.mean(-1), 1e-6))
    lum = np.where(mask, lum, np.nan)
    gy = np.abs(np.diff(lum, axis=0, prepend=lum[:1]))
    gx = np.abs(np.diff(lum, axis=1, prepend=lum[:, :1]))
    with np.errstate(invalid="ignore"):
        g = np.nanmean(np.stack([gx, gy]), axis=0)
    return np.where(mask, np.nan_to_num(g), np.nan)


def panel(ax, img, extent, title, *, cmap=None, vmin=None, vmax=None):
    im = ax.imshow(img, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest", origin="upper")
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(True, color="w", alpha=0.25, lw=0.4)
    return im


def draw(maps, extent, rects, labels, out: Path, grid: int, suptitle: str) -> None:
    x0, x1, y1, y0 = extent          # imshow extent is (left, right, bottom, top)
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    ims = []
    for ax, (title, img, kw) in zip(axes.ravel(), maps):
        ims.append((ax, panel(ax, img, extent, title, **kw), kw))
        ax.set_xticks(np.arange(x0, x1 + 1, grid))
        ax.set_yticks(np.arange(y0, y1 + 1, grid))
        for r, lab in zip(rects, labels):
            ax.add_patch(Rectangle((r[0], r[1]), r[2], r[3], fill=False,
                                   ec="#00ff88", lw=1.6))
            ax.text(r[0] + 6, r[1] + 6, lab, color="#00ff88", fontsize=8,
                    va="top", ha="left", fontweight="bold")
    for ax, im, kw in ims:
        if kw.get("cmap") is not None:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  + {out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--zoom", nargs=4, type=int, default=None,
                    metavar=("X0", "Y0", "W", "H"))
    ap.add_argument("--rect", nargs=4, type=int, action="append", default=[],
                    metavar=("X0", "Y0", "W", "H"))
    ap.add_argument("--label", action="append", default=[])
    ap.add_argument("--grid", type=int, default=256)
    ap.add_argument("--zoom-grid", type=int, default=128)
    ap.add_argument("--down", type=int, default=4, help="block mean for the full atlas")
    a = ap.parse_args(argv)

    run = Path(a.run_dir)
    out = Path(a.out) if a.out else run / "roi_preview"
    out.mkdir(parents=True, exist_ok=True)

    direct = _load(run, "irradiance.exr")
    indirect = _load(run, "irradiance_indirect.exr")
    h, w = direct.shape[:2]
    m = _mask(run, (h, w))
    print(f"  atlas {w}x{h}, {int(m.sum())} active texels "
          f"({100.0 * m.mean():.1f}% of the atlas)")

    labels = list(a.label) + [f"R{i}" for i in range(len(a.label), len(a.rect))]

    def build(sl_y, sl_x, down):
        mm = block_mean(m[sl_y, sl_x].astype(np.float32)[..., None], down)[..., 0] > 0.5
        d = block_mean(direct[sl_y, sl_x] * m[sl_y, sl_x, None], down)
        i = block_mean(indirect[sl_y, sl_x] * m[sl_y, sl_x, None], down)
        s = block_mean(structure(direct, m)[sl_y, sl_x][..., None], down)[..., 0]
        s = np.where(mm, s, np.nan)
        tot = d + i
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(mm & (tot.mean(-1) > EPS),
                             i.mean(-1) / np.maximum(tot.mean(-1), EPS), np.nan)
        ed = exposure_on(d, mm)
        ei = exposure_on(i, mm)
        return [
            (f"direct  E_dir   (exposure {ed:.3g})", tonemap(d, ed), {}),
            (f"indirect  E_ind   (exposure {ei:.3g})", tonemap(i, ei), {}),
            ("structure  mean |grad log10 E_dir|",
             s, dict(cmap="magma", vmin=0.0, vmax=float(np.nanpercentile(s, 99)))),
            ("share  E_ind / (E_dir + E_ind)",
             share, dict(cmap="viridis", vmin=0.0, vmax=1.0)),
        ]

    maps = build(slice(None), slice(None), a.down)
    draw(maps, (0, w, h, 0), a.rect, labels, out / "atlas_overview.png", a.grid,
         f"{run.name} -- full atlas, block mean x{a.down}, axes in IUM texels")

    if a.zoom:
        zx, zy, zw, zh = a.zoom
        zx1, zy1 = min(zx + zw, w), min(zy + zh, h)
        maps = build(slice(zy, zy1), slice(zx, zx1), 1)
        draw(maps, (zx, zx1, zy1, zy), a.rect, labels, out / "atlas_zoom.png",
             a.zoom_grid,
             f"{run.name} -- zoom [{zx},{zy}] {zx1 - zx}x{zy1 - zy}, native resolution")
    return 0


if __name__ == "__main__":
    sys.exit(main())
