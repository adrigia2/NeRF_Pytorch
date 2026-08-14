#!/usr/bin/env python
"""make_skybox_figure.py -- thesis figures on the skyboxes baked from a sweep's NeRFs.

Produces two PNGs:

  skybox_grid.png    grid with the original envmap and the one baked by each run,
                     with IDENTICAL tonemap and EXPOSURE in every panel
  skybox_detail.png  original, baked and log2(baked/GT) map for the chosen model only

The `skybox_compare/skybox_heatmap.png` files bake_skyboxes.py produces per run do not
serve this purpose: they clip to [0,1] with neither tonemap nor gamma, so the skybox comes
out nearly black except for the light sources.

    python make_skybox_figure.py <sweep_root> --gt GT.exr --out DIR [--selected RUN]

Three choices are not negotiable, and are the reason this script exists:

  1. The downsample happens in LINEAR space, as a block mean, before the tonemap.
     Resizing after the tonemap, or with a non-conservative filter, alters the mean
     radiance of the small bright pixels, which are the ones carrying the energy.

  2. There is ONE exposure for every panel, derived from the GT.  If each panel had its
     own, a brightness difference between the models would vanish from the figure, which
     is exactly what the figure is there to show.

  3. The difference is a RATIO in log2, not a subtraction.  The measured gap is a nearly
     uniform factor of a few per cent: on a linear scale it would be invisible everywhere
     except on the light sources.

The script also prints the ratio of the solid-angle-weighted linear means,
<||baked||> / <||GT||>, which is the number that enters the irradiance integral.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import _paths  # noqa: F401

BAKED_NAME = "skybox_nerf_baked.exr"
# Rec.709, identical to LUMA_COEFF in compare_runs.py
LUMA_COEFF = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
EPS = 1e-6

# Readable labels: the folder name is not presentable in a figure
LOSS_LABEL = {"l1": "L1", "mse": "sq.", "relmseraw": "rel. sq.",
              "rel_mse_raw": "rel. sq."}


def load_exr(path: Path) -> np.ndarray:
    """(H, W, 3) float32.  Same loader as bake_skyboxes.py, one reader for the
    skyboxes."""
    from regen_heatmaps import _load_exr_hw3
    return _load_exr_hw3(str(path))


def block_mean(img: np.ndarray, factor: int) -> np.ndarray:
    """Downsample by a factor x factor block mean, in linear space.

    It preserves the mean radiance, which interpolating resampling does not guarantee: on
    an envmap with small, very bright sources the difference is not cosmetic.

    The channel count is read from the array instead of being fixed at 3: the heatmaps of
    make_results_figures come through here single-channel, and with 3 hard-coded the
    reshape used to fail.  On RGB the behaviour is identical.
    """
    if factor <= 1:
        return img
    h, w = img.shape[:2]
    c = img.shape[2] if img.ndim == 3 else 1
    h2, w2 = (h // factor) * factor, (w // factor) * factor
    a = img[:h2, :w2].reshape(h2, w2, c)
    out = a.reshape(h2 // factor, factor, w2 // factor, factor, c).mean(axis=(1, 3))
    return out if img.ndim == 3 else out[..., 0]


def tonemap(x: np.ndarray, exposure: float) -> np.ndarray:
    """Reinhard plus gamma 2.2.  Same formula as _tonemap_srgb in compare_runs.py."""
    y = x * exposure
    y = y / (1.0 + y)
    return np.clip(y, 0.0, 1.0) ** (1.0 / 2.2)


def solid_angle_weights(h: int, w: int) -> np.ndarray:
    """sin(theta) per row of an equirectangular map.  Without this weight the poles, where
    pixels cover a tiny solid angle, would count as much as the equator."""
    theta = np.pi * (np.arange(h, dtype=np.float64) + 0.5) / h
    return np.repeat(np.sin(theta)[:, None], w, axis=1)


def discover(root: Path) -> list[tuple[str, Path]]:
    """[(run name, path of the baked skybox)], sorted by name."""
    out = []
    for p in sorted(root.glob(f"*/*/{BAKED_NAME}")):
        out.append((p.parents[1].name, p))
    return out


def pretty(run_key: str) -> str:
    """exp_relmseraw_d02 -> exp / rel. sq."""
    parts = run_key.split("_")
    act = parts[0]
    loss = "_".join(parts[1:-1]) if len(parts) > 2 else parts[1]
    return f"{act} / {LOSS_LABEL.get(loss, loss)}"


def _panel(ax, rgb: np.ndarray, label: str, bold: bool = False) -> None:
    ax.imshow(rgb)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(label, fontsize=10, fontweight="bold" if bold else "normal")


def fig_grid(gt_tm: np.ndarray, baked_tm: dict[str, np.ndarray], selected: str | None,
             out: Path, ncols: int = 2) -> None:
    items = [("original", gt_tm, True)] + [(pretty(k), v, k == selected)
                                           for k, v in baked_tm.items()]
    nrows = (len(items) + ncols - 1) // ncols
    h, w = gt_tm.shape[:2]
    # each panel is 2:1; the figure follows that ratio, otherwise matplotlib would leave
    # white bands between the rows
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.6 * ncols, 5.6 * (h / w) * nrows))
    flat = np.atleast_1d(axes).ravel()
    for ax, (label, img, bold) in zip(flat, items):
        _panel(ax, img, label, bold)
    for ax in flat[len(items):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  + {out}")


def fig_detail(gt_lin: np.ndarray, baked_lin: np.ndarray, gt_tm: np.ndarray,
               baked_tm: np.ndarray, label: str, out: Path) -> tuple[float, float]:
    """Detail panel.  Returns (colormap limit, median of the log2)."""
    ratio = np.log2((baked_lin.mean(-1) + EPS) / (gt_lin.mean(-1) + EPS))
    # symmetric extremes from a percentile: the absolute maximum falls on a few isolated
    # pixels at the edge of the sources and would crush everything else to grey
    lim = float(np.percentile(np.abs(ratio), 98.0))
    lim = max(lim, 0.05)
    med = float(np.median(ratio))

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 3.1))
    _panel(axes[0], gt_tm, "original")
    _panel(axes[1], baked_tm, f"baked, {label}")
    im = axes[2].imshow(ratio, cmap="RdBu_r",
                        norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim))
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].set_title(r"$\log_2$(baked / original)", fontsize=10)
    cb = fig.colorbar(im, ax=axes[2], fraction=0.032, pad=0.02)
    cb.set_label("stops", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  + {out}")
    return lim, med


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_root")
    ap.add_argument("--gt", required=True, help="the original equirectangular EXR")
    ap.add_argument("--out", required=True, help="destination folder for the PNGs")
    ap.add_argument("--selected", default=None,
                    help="run for the detail panel (default: the first one found)")
    ap.add_argument("--downsample", type=int, default=4,
                    help="block-mean factor (default 4: 4096x2048 -> 1024x512)")
    ap.add_argument("--key", type=float, default=0.5,
                    help="level the GT median is brought to in the tonemap")
    args = ap.parse_args()

    root, gt_path, out = Path(args.sweep_root), Path(args.gt), Path(args.out)
    if not root.is_dir():
        print(f"ERROR: {root} is not a folder")
        return 2
    if not gt_path.exists():
        print(f"ERROR: {gt_path} does not exist")
        return 2
    out.mkdir(parents=True, exist_ok=True)

    runs = discover(root)
    if not runs:
        print(f"ERROR: no {BAKED_NAME} under {root}")
        return 2
    selected = args.selected or runs[0][0]
    if selected not in dict(runs):
        print(f"ERROR: {selected} is not among the runs found ({[k for k, _ in runs]})")
        return 2

    print(f"GT: {gt_path.name}")
    gt_full = load_exr(gt_path)
    gt = block_mean(gt_full, args.downsample)
    print(f"  {gt_full.shape[1]}x{gt_full.shape[0]} -> {gt.shape[1]}x{gt.shape[0]}")

    # Shared exposure, from the GT: the median luminance lands at mid scale, so the
    # tonemap is not dictated by the HDR peak of the light sources
    lum = (gt * LUMA_COEFF).sum(-1)
    expo = args.key / max(float(np.median(lum)), 1e-4)
    print(f"  shared exposure = {expo:.4f} (GT median luminance "
          f"{float(np.median(lum)):.4f})")

    wts = solid_angle_weights(*gt.shape[:2])
    gt_norm = np.sqrt((gt.astype(np.float64) ** 2).sum(-1))
    gt_mean = float((gt_norm * wts).sum() / wts.sum())

    gt_tm = tonemap(gt, expo)
    baked_lin: dict[str, np.ndarray] = {}
    baked_tm: dict[str, np.ndarray] = {}
    print("\nratio of the solid-angle-weighted linear means, "
          "<||baked||> / <||GT||>:")
    for key, path in runs:
        a = block_mean(load_exr(path), args.downsample)
        if a.shape != gt.shape:
            print(f"  [skip] {key}: shape {a.shape} != GT {gt.shape}")
            continue
        baked_lin[key] = a
        baked_tm[key] = tonemap(a, expo)
        n = np.sqrt((a.astype(np.float64) ** 2).sum(-1))
        ratio = float((n * wts).sum() / wts.sum()) / gt_mean
        mark = "  <- selected" if key == selected else ""
        print(f"  {pretty(key):22s} {ratio:7.4f}   ({100 * (ratio - 1):+.2f}%){mark}")

    print()
    fig_grid(gt_tm, baked_tm, selected, out / "skybox_grid.png")
    lim, med = fig_detail(gt, baked_lin[selected], gt_tm, baked_tm[selected],
                          pretty(selected), out / "skybox_detail.png")
    print(f"\ndetail ({pretty(selected)}): median log2(baked/GT) = {med:+.4f} stops "
          f"({100 * (2.0 ** med - 1):+.2f}%), colormap extremes +/-{lim:.3f} stops")
    return 0


if __name__ == "__main__":
    sys.exit(main())
