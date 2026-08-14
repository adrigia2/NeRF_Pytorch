"""Pixel-by-pixel comparison of two sets of EXR renders.

For every pair of images with the same name it produces a three-panel figure:

    Original (NerfOpenEXRSmooth) | Computed (NerfOpenEXRSmoothRerender) | heatmap

The heatmap shows the L2 norm of the RGB difference, ||A - B||_2, on a logarithmic
scale with a GLOBAL normalization over every pair: the same colour means the same
error in every frame.

The difference is always computed on the original linear values of the EXR.
Tone mapping (Reinhard + sRGB gamma) is applied only to the two left panels, for
display only, and identically to both.

Requires an environment with working opencv, numpy and matplotlib; in Anaconda's base
env matplotlib is broken (NumPy 1.x vs 2.x ABI), so use instead:

    C:\\Users\\adria\\anaconda3\\envs\\nerfpytorch\\python.exe compare_exr.py
"""

from __future__ import annotations

# Must be set BEFORE importing cv2, otherwise imread() returns None on EXRs.
import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import argparse
import csv
import math
import sys
import time

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterSciNotation, LogLocator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
DIR_ORIGINAL = os.path.join(ROOT, "NerfOpenEXRSmooth", "images")
DIR_COMPUTED = os.path.join(ROOT, "NerfOpenEXRSmoothRerender", "images")
DIR_OUTPUT = os.path.join(ROOT, "output")

# Figure geometry, in pixels. At dpi=100 each panel gets exactly its native
# 1920x1080 pixels, with no rescaling.
PANEL_W, PANEL_H = 1920, 1080
MARGIN = 20
PANEL_GAP = 20
CBAR_GAP = 30
CBAR_W = 30
CBAR_LABEL_W = 140
TOP_BAR = 170  # room for the suptitle and the panel titles


# --------------------------------------------------------------------------- #
# Tone mapping (display only)
# --------------------------------------------------------------------------- #


def srgb_encode(y: np.ndarray) -> np.ndarray:
    """Standard sRGB encoding on values already in [0, 1]."""
    y = np.clip(y, 0.0, 1.0)
    return np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(y, 1.0 / 2.4) - 0.055)


def tonemap(bgr: np.ndarray) -> np.ndarray:
    """Reinhard x/(1+x) + sRGB gamma. Returns float RGB in [0, 1].

    The input is BGR, as cv2.imread returns it; the order is reversed here.
    """
    x = np.maximum(bgr.astype(np.float32), 0.0)  # EXRs can hold slightly negative values
    y = x / (1.0 + x)
    return srgb_encode(y)[:, :, ::-1]


# --------------------------------------------------------------------------- #
# I/O and difference
# --------------------------------------------------------------------------- #


def read_exr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"cannot read the EXR: {path}")
    if img.ndim != 3 or img.shape[2] < 3:
        raise RuntimeError(f"expected 3 channels, got shape {img.shape}: {path}")
    return img[:, :, :3].astype(np.float32)


def diff_norm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel L2 norm on the original linear values.

    The channel order is irrelevant: the L2 norm is invariant to it.
    """
    d = a - b
    return np.sqrt(np.einsum("ijk,ijk->ij", d, d, optimize=True))


def frame_stats(d: np.ndarray, vmin: float) -> dict:
    d64 = d.astype(np.float64)
    p50, p99 = np.percentile(d64, [50.0, 99.0])
    return {
        "max": float(d64.max()),
        "mean": float(d64.mean()),
        "rmse": float(np.sqrt(np.mean(d64 * d64))),
        "p50": float(p50),
        "p99": float(p99),
        "frac_above_vmin": float((d64 > vmin).mean()),
    }


def find_pairs(dir_a: str, dir_b: str) -> tuple[list[str], list[str], list[str]]:
    """Returns (common, only_in_a, only_in_b), in natural order."""
    for d in (dir_a, dir_b):
        if not os.path.isdir(d):
            raise SystemExit(f"folder not found: {d}")

    def exrs(d):
        return {f for f in os.listdir(d) if f.lower().endswith(".exr")}

    a, b = exrs(dir_a), exrs(dir_b)

    def natural_key(name: str):
        stem = os.path.splitext(name)[0]
        parts = stem.replace("_", " ").split()
        return [(int(p), "") if p.isdigit() else (math.inf, p) for p in parts]

    return sorted(a & b, key=natural_key), sorted(a - b), sorted(b - a)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def render_figure(
    name: str,
    orig_bgr: np.ndarray,
    comp_bgr: np.ndarray,
    d: np.ndarray,
    stats: dict,
    vmin: float,
    vmax: float,
    cmap_name: str,
    dpi: int,
    out_path: str,
) -> None:
    h, w = d.shape
    # The geometry is computed on the real image resolution, so the script also
    # handles renders at a resolution other than 1920x1080.
    total_w = MARGIN + 3 * w + 2 * PANEL_GAP + CBAR_GAP + CBAR_W + CBAR_LABEL_W
    total_h = TOP_BAR + h + MARGIN

    fig = plt.figure(figsize=(total_w / dpi, total_h / dpi), dpi=dpi, facecolor="white")

    cmap = plt.get_cmap(cmap_name).copy()
    # Below the floor = numerically irrelevant difference -> solid black.
    cmap.set_under("black")
    cmap.set_bad("black")

    panels = [
        ("Original", tonemap(orig_bgr), None),
        ("Computed", tonemap(comp_bgr), None),
        (r"Difference  $\||\Delta$RGB$\||_2$  (log)", d, LogNorm(vmin=vmin, vmax=vmax)),
    ]

    im_diff = None
    for i, (title, data, norm) in enumerate(panels):
        left = MARGIN + i * (w + PANEL_GAP)
        ax = fig.add_axes(
            [left / total_w, MARGIN / total_h, w / total_w, h / total_h]
        )
        if norm is None:
            ax.imshow(data, interpolation="nearest", resample=False)
        else:
            im_diff = ax.imshow(
                data,
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
                resample=False,
            )
        ax.set_title(title, fontsize=30, pad=18)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#999999")

    cbar_left = MARGIN + 3 * w + 2 * PANEL_GAP + CBAR_GAP
    cax = fig.add_axes(
        [cbar_left / total_w, MARGIN / total_h, CBAR_W / total_w, h / total_h]
    )
    cbar = fig.colorbar(
        im_diff,
        cax=cax,
        extend="min",
        ticks=LogLocator(base=10.0),
        format=LogFormatterSciNotation(base=10.0),
    )
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(
        r"$\||A - B\||_2$   (linear radiance, no tonemap)", fontsize=22, labelpad=16
    )

    fig.suptitle(
        f"{name}    "
        f"max {stats['max']:.4g}   mean {stats['mean']:.4g}   "
        f"RMSE {stats['rmse']:.4g}   "
        f"px > {vmin:g}: {stats['frac_above_vmin'] * 100:.1f}%",
        fontsize=32,
        y=1.0 - 14.0 / total_h,
        va="top",
    )

    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="EXR comparison with a logarithmic heatmap of the difference."
    )
    ap.add_argument("--original", default=DIR_ORIGINAL, help="Original images folder")
    ap.add_argument("--computed", default=DIR_COMPUTED, help="Computed images folder")
    ap.add_argument("--output", default=DIR_OUTPUT, help="output folder")
    ap.add_argument(
        "--vmin",
        type=float,
        default=1e-4,
        help="floor of the log scale; below this value the difference is "
        "float noise (default: 1e-4)",
    )
    ap.add_argument(
        "--vmax-exact",
        action="store_true",
        help="use the exact global maximum instead of rounding it to the decade",
    )
    ap.add_argument("--cmap", default="inferno", help="colormap matplotlib (default: inferno)")
    ap.add_argument("--dpi", type=int, default=100, help="figure dpi (default: 100)")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N pairs")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="list the pairs found and check they match, without writing anything",
    )
    args = ap.parse_args(argv)

    names, only_a, only_b = find_pairs(args.original, args.computed)
    if only_a or only_b:
        print(f"WARNING: {len(only_a)} files only in Original, {len(only_b)} only in Computed")
        for n in only_a[:10]:
            print(f"  Original only: {n}")
        for n in only_b[:10]:
            print(f"  Computed only: {n}")
    if not names:
        print("no pair found.", file=sys.stderr)
        return 1
    if args.limit:
        names = names[: args.limit]

    print(f"Original : {args.original}")
    print(f"Computed : {args.computed}")
    print(f"Coppie   : {len(names)}")

    if args.dry_run:
        for n in names:
            print(f"  {n}")
        print("\n--dry-run: no file written.")
        return 0

    os.makedirs(args.output, exist_ok=True)

    # ---------------- Pass 1: global statistics ------------------ #
    print("\n[1/2] global statistics...")
    t0 = time.time()
    stats_all: dict[str, dict] = {}
    vmax_global = 0.0
    for i, n in enumerate(names, 1):
        a = read_exr(os.path.join(args.original, n))
        b = read_exr(os.path.join(args.computed, n))
        if a.shape != b.shape:
            raise SystemExit(f"different sizes for {n}: {a.shape} vs {b.shape}")
        s = frame_stats(diff_norm(a, b), args.vmin)
        stats_all[n] = s
        vmax_global = max(vmax_global, s["max"])
        print(f"\r  {i}/{len(names)}  {n}  max={s['max']:.4g}   ", end="", flush=True)
    print(f"\n  done in {time.time() - t0:.1f}s   global max = {vmax_global:.6g}")

    if vmax_global <= args.vmin:
        raise SystemExit(
            f"the global maximum ({vmax_global:.3g}) does not exceed --vmin ({args.vmin:g}): "
            "the images are identical, or the floor is too high."
        )
    vmax = vmax_global if args.vmax_exact else 10.0 ** math.ceil(math.log10(vmax_global))
    print(f"  scala log: vmin={args.vmin:g}  vmax={vmax:g}")

    # ---------------- Pass 2: rendering ---------------- #
    print(f"\n[2/2] rendering in {args.output} ...")
    t0 = time.time()
    for i, n in enumerate(names, 1):
        a = read_exr(os.path.join(args.original, n))
        b = read_exr(os.path.join(args.computed, n))
        d = diff_norm(a, b)
        out_path = os.path.join(args.output, os.path.splitext(n)[0] + ".png")
        render_figure(
            n, a, b, d, stats_all[n], args.vmin, vmax, args.cmap, args.dpi, out_path
        )
        print(f"\r  {i}/{len(names)}  {os.path.basename(out_path)}   ", end="", flush=True)
    print(f"\n  done in {time.time() - t0:.1f}s")

    # ---------------- Report ---------------- #
    csv_path = os.path.join(args.output, "diff_stats.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wtr = csv.writer(f)
        wtr.writerow(["frame", "max", "mean", "rmse", "p50", "p99", "frac_above_vmin"])
        for n in names:
            s = stats_all[n]
            wtr.writerow(
                [n]
                + [
                    f"{s[k]:.8g}"
                    for k in ("max", "mean", "rmse", "p50", "p99", "frac_above_vmin")
                ]
            )

    worst = sorted(names, key=lambda n: stats_all[n]["rmse"], reverse=True)
    txt_path = os.path.join(args.output, "diff_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("EXR comparison: Original vs Computed\n")
        f.write(f"  Original      : {args.original}\n")
        f.write(f"  Computed      : {args.computed}\n")
        f.write(f"  pairs         : {len(names)}\n\n")
        f.write("Difference = ||A - B||_2 per pixel, on the original linear values\n")
        f.write("(no tone mapping, no clipping).\n\n")
        f.write(f"  max globale   : {vmax_global:.8g}\n")
        f.write(f"  scala log vmin: {args.vmin:g}\n")
        f.write(f"  scala log vmax: {vmax:g}\n")
        f.write(f"  colormap      : {args.cmap}\n\n")
        mean_rmse = sum(stats_all[n]["rmse"] for n in names) / len(names)
        f.write(f"  Mean RMSE over every frame: {mean_rmse:.8g}\n\n")
        f.write("Frames sorted by decreasing RMSE (first 10):\n")
        for n in worst[:10]:
            s = stats_all[n]
            f.write(f"  {n:34s} rmse={s['rmse']:.6g}  max={s['max']:.6g}  mean={s['mean']:.6g}\n")
        f.write("\nMost similar frames (last 10):\n")
        for n in worst[-10:][::-1]:
            s = stats_all[n]
            f.write(f"  {n:34s} rmse={s['rmse']:.6g}  max={s['max']:.6g}  mean={s['mean']:.6g}\n")

    print(f"\nScritti {len(names)} PNG")
    print(f"  {csv_path}")
    print(f"  {txt_path}")
    print(f"\nFrame with the highest RMSE: {worst[0]}  (rmse={stats_all[worst[0]]['rmse']:.6g})")
    print(f"Frame with the lowest RMSE : {worst[-1]}  (rmse={stats_all[worst[-1]]['rmse']:.6g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
