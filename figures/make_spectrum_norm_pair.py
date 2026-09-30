#!/usr/bin/env python
"""make_spectrum_norm_pair.py -- the two `norm` spectra, one panel each, for a slide.

    python make_spectrum_norm_pair.py --skybox-gt D:/tesi_output/wooden_studio_13_4k.exr

Writes three PNGs into `PresentationImages/`:

  spectrum_norm_frames.png   the pooled spectrum of the rendered frames
  spectrum_norm_skybox.png   the same analysis on the baked skyboxes
  spectrum_norm_pair.png     the two side by side, to see what they do to the page

The envmap panel counts pixels by default, which makes both panels the same quantity, a
fraction of pixels per bin.  `--skybox-solid-angle` weights them by sin(theta) instead and
adds `_solidangle` to the two skybox file names; that is the physically right one for a map
the pipeline integrates over the sphere, and the one that would survive a change of envmap
resolution, but it makes the y axes of the two panels two different fractions.  BOTH are
computed from one pass over the EXRs whatever is drawn, so the check below always runs.

`compare_runs.py` already computes both, but only as 2x2 grids over R, G, B and norm
(`spectrum_global.png`, `spectrum_skybox.png`), at the 8 to 10 point type of an analysis
figure.  This draws the `norm` panel of each on its own, at the size of a projection.

It also fixes something in the passage.  In both 2x2 figures the legend is drawn once, on
`axs[0]`, so the W1 one reads there is the one of the R CHANNEL even while looking at the
norm panel.  Here the number beside each run is the W1 of the channel actually plotted,
which is also the channel `compare_runs.py` ranks the runs on (`NORM_CH` in
`spectrum_metrics`).

ONE SCALE FOR BOTH PANELS, x and y, computed on the union of the two datasets before either
is drawn.  It is legitimate because both are fractions per bin over the same 162 log-spaced
bins and both sum to one; it is also the only way two panels read side by side.  What the
two fractions are OF is not the same thing, pixels on one side and solid angle on the other,
and that is what the axis labels are for.

WHERE THE DATA COMES FROM.  The frame spectra cost no EXR at all: the histograms are already
in `<sweep>/_comparison/_cache_metrics.npz`, and summing the frame axis gives exactly the
`spec_gt_tot` and `spec_run_tot` of `fig_spectrum_global`.  The skybox spectra are not cached
by anything, so they are computed once from the seven envmaps (the original plus one
`skybox_nerf_baked.exr` per run) and cached here, because iterating on a figure that costs
660 MB of I/O per attempt is not iterating.

Both panels are CHECKED against tables `compare_runs.py` already wrote: the W1 recomputed
here must match `spectrum_distance.csv` and `spectrum_skybox.csv` on the norm channel to
1e-6.  Same functions on the same data, so a disagreement is an indexing bug and not noise,
and the script refuses to draw.

A note on the cache on disk.  It is a version-2 one, written before the pixel-set axis
existed, so its arrays are (F, 4, NS) instead of (F, 3, 4, NS).  `load_cache` would reject
it and `compare_runs.py --reuse-cache` would quietly re-read all 420 EXRs; this script reads
the npz itself and accepts both layouts, which is sound because the axis is absent exactly
when the only set stored was `full`, the one wanted here.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import _paths  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402

from compare_runs import (                                             # noqa: E402
    CACHE_NAME, DEFAULT_ROOT, NORM_CH, Run, discover_runs, load_exr_rgb,
    resolve_skybox_gt, spec_w1_dex, spectrum_hist,
    _draw_spectrum, _equirect_solid_angle_weights, _spec_x, _spec_xlim, _spec_ylim,
)

SKY_CACHE = {True: "_cache_spectrum_skybox.npz",
             False: "_cache_spectrum_skybox_pixels.npz"}
DPI = 200
FIG_W, FIG_H = 9.0, 6.7        # inches; 1.34, so the pair lands near 2.7 on a 4:3 page
SLIDE_W, SLIDE_H = 13.33, 10.0  # the page itself, 4:3
TOL = 1e-6

FS_TITLE, FS_LABEL, FS_TICK, FS_LEG = 15, 13, 11, 12
FS_SLIDE = 19, 17, 15, 15      # the same four, for a figure that IS the page


# ── the two sources ──────────────────────────────────────────────────────────
def frame_spectra(out: Path, runs: list[Run]):
    """(gt (NS,), {key: (NS,)}) on the norm channel, pooled over the frames.

    Accepts both cache layouts: (F, 3, 4, NS) with the pixel-set axis, and the older
    (F, 4, NS) without it.  The axis is missing exactly when `full` was the only set
    stored, which is the set this figure wants, so the two are the same quantity.
    """
    path = out / CACHE_NAME
    if not path.exists():
        raise SystemExit(f"{path} is missing: run compare_runs.py on the sweep first")
    z = np.load(path, allow_pickle=True)

    def full(a: np.ndarray) -> np.ndarray:
        if a.ndim == 4:
            return a[:, 0]
        if a.ndim == 3:
            return a
        raise SystemExit(f"{path}: unexpected spectrum shape {a.shape}")

    gt = full(z["__spec_gt__"]).sum(axis=0)[NORM_CH]
    per = {}
    for r in runs:
        key = f"{r.key}//spec"
        if key in z.files:
            per[r.key] = full(z[key]).sum(axis=0)[NORM_CH]
    if not per:
        raise SystemExit(f"{path}: none of the runs {[r.key for r in runs]} is in the cache")
    return gt, per


def _skybox_hists_both(runs: list[Run], gt_path: Path):
    """Both weightings of the envmap spectrum, from ONE pass over the EXRs.

    The two differ only in the `weights` argument of `spectrum_hist`, so reading the seven
    files twice would be paying twice for nothing.  Computing both also keeps the check
    alive whichever one is drawn: only the weighted one has a table on disk to be verified
    against, and it is the same reading and the same binning as the other.

    An equirectangular map does not sample the sphere uniformly: a pixel's solid angle goes
    as sin(theta), so at 4096x2048 a pole pixel covers about 1/1300 of an equatorial one.
    Counting pixels asks whether the two EXR files hold the same values, weighting asks
    whether the two spheres carry the same radiance.
    """
    both: dict[bool, dict[str, np.ndarray]] = {True: {}, False: {}}
    gt = load_exr_rgb(str(gt_path))
    w = _equirect_solid_angle_weights(*gt.shape[:2])
    both[True]["__spec_gt__"], both[False]["__spec_gt__"] = spectrum_hist(gt, w), spectrum_hist(gt)
    del gt, w
    for r in runs:
        path = r.scene_dir / "skybox_nerf_baked.exr"
        if not path.exists():
            print(f"      [skip] {r.key}: skybox_nerf_baked.exr missing "
                  f"(generate it with bake_skyboxes.py)")
            continue
        a = load_exr_rgb(str(path))
        w = _equirect_solid_angle_weights(*a.shape[:2])
        both[True][r.key], both[False][r.key] = spectrum_hist(a, w), spectrum_hist(a)
        del a, w
    return both


def _from_cache(path: Path, runs: list[Run], gt_path: Path):
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    if str(z["__gt__"]) != gt_path.as_posix():
        print(f"  {path.name} was built from another reference: recomputing")
        return None
    d = {k: z[k] for k in z.files if k != "__gt__"}
    return d if any(r.key in d for r in runs) else None


def skybox_spectra(out: Path, runs: list[Run], gt_path: Path):
    """{weighted: (gt (NS,), {key: (NS,)})} on the norm channel, both variants."""
    cached = {w: _from_cache(out / SKY_CACHE[w], runs, gt_path) for w in (True, False)}
    if any(v is None for v in cached.values()):
        print("  reading the envmaps, both weightings (the only expensive step)…")
        both = _skybox_hists_both(runs, gt_path)
        for w, d in both.items():
            np.savez_compressed(out / SKY_CACHE[w], __gt__=gt_path.as_posix(), **d)
            print(f"  + {out / SKY_CACHE[w]}")
        cached = both
    else:
        print(f"  skybox spectra from {SKY_CACHE[True]} and {SKY_CACHE[False]}")

    return {w: (d["__spec_gt__"][NORM_CH],
                {r.key: d[r.key][NORM_CH] for r in runs if r.key in d})
            for w, d in cached.items()}


# ── the checks ───────────────────────────────────────────────────────────────
def _csv_norm_w1(path: Path, column: str) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["run"]: float(row[column]) for row in csv.DictReader(fh)
                if row.get("channel") == "norm"}


def check(name: str, gt: np.ndarray, per: dict[str, np.ndarray],
          table: Path, column: str) -> dict[str, float]:
    """W1 of every run, verified against the table compare_runs.py already wrote."""
    w1 = {k: float(spec_w1_dex(h, gt)) for k, h in per.items()}
    ref = _csv_norm_w1(table, column)
    checked = 0
    for k, v in w1.items():
        if k in ref:
            if abs(v - ref[k]) > TOL:
                raise SystemExit(f"{name}: {k} gives W1={v:.7f}, {table.name} says "
                                 f"{ref[k]:.7f}. Same data and same function, so this is "
                                 f"an indexing bug, not noise.")
            checked += 1
    print(f"  {name}: {len(w1)} runs, {checked} checked against {table.name}"
          if checked else f"  {name}: {len(w1)} runs, {table.name} not on disk, NOT checked")
    return w1


# ── drawing ──────────────────────────────────────────────────────────────────
def panel(ax, gt: np.ndarray, per: dict[str, np.ndarray], w1: dict[str, float],
          runs: list[Run], xlim, ylim, title: str, xlabel: str, ylabel: str,
          fonts=None, loc: str = "upper left") -> None:
    # Sorted by W1, best first, so the legend of a panel is the ranking of that panel.
    # Each panel by its own: a shared order would read as unsorted, because the two are
    # not the same order (see the note in the README).  Best first also puts the worst
    # curve on top, which is the one with something to show.
    ordered = sorted((r for r in runs if r.key in per), key=lambda r: w1[r.key])
    curves = [(f"{r.label}  W1={w1[r.key]:.3f}", r.color, r.linestyle, per[r.key])
              for r in ordered]
    ft, fl, fk, fg = fonts or (FS_TITLE, FS_LABEL, FS_TICK, FS_LEG)
    _draw_spectrum(ax, _spec_x(), gt, curves, ylim, xlim)
    ax.set_title(title, fontsize=ft)
    ax.set_xlabel(xlabel, fontsize=fl)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fl)
    ax.tick_params(labelsize=fk)
    # Lower right on the tall panels: up there the box lands on the peak, which is where
    # the mass is; down there it lands on the inside of the GT area, whose information is
    # its outline, and that runs two decades higher.
    ax.legend(fontsize=fg, framealpha=0.9, loc=loc)


def save(fig, path: Path) -> None:
    """No bbox_inches='tight': the two single panels must come out the same size, and a
    tight box is the size of the ink, which the legend can change."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.set_layout_engine("constrained")
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"  + {path}  ({int(fig.get_figwidth() * DPI)}x{int(fig.get_figheight() * DPI)})")


def build(root: Path, out: Path, figdir: Path, sky_gt: str | None, no_sky: bool,
          weighted: bool) -> None:
    runs = discover_runs(root, None)
    if not runs:
        raise SystemExit(f"no run found under {root}")
    print(f"  {len(runs)} runs: {', '.join(r.key for r in runs)}")

    fr_gt, fr_per = frame_spectra(out, runs)
    fr_w1 = check("frames", fr_gt, fr_per, out / "spectrum_distance.csv", "w1_dex_pooled")

    sk = None
    if not no_sky:
        gt_path = resolve_skybox_gt(runs, sky_gt)
        if gt_path is not None:
            variants = skybox_spectra(out, runs, gt_path)
            # The check runs on the WEIGHTED variant whichever one is drawn: it is the one
            # `spectrum_skybox.csv` holds, and the two share the reading and the binning,
            # so verifying it verifies the machinery under both.
            check("skybox (solid-angle, the checkable one)", *variants[True],
                  out / "spectrum_skybox.csv", "w1_dex")
            sk_gt, sk_per = variants[weighted]
            sk_w1 = {k: float(spec_w1_dex(h, sk_gt)) for k, h in sk_per.items()}
            print(f"  skybox drawn: {'solid-angle weighted' if weighted else 'per pixel'}")
            sk = (sk_gt, sk_per, sk_w1, gt_path)

    # One scale for both, computed on everything before anything is drawn.
    stack = [fr_gt] + list(fr_per.values())
    if sk is not None:
        stack += [sk[0]] + list(sk[1].values())
    xlim = _spec_xlim(*stack)
    ylim = _spec_ylim(*stack)

    # One name for the x axis of both.  compare_runs calls it "pixel value (HDR)" in the
    # frame figure and "radiance" in the envmap one, which is a habit of context and not a
    # distinction: a pixel of a frame is the radiance reaching the camera along that ray, a
    # texel of the envmap the radiance arriving from that direction, same units and the
    # same range (both span 1e-3 to 1e2, the GT frames being renders lit by that very
    # HDRI).  Two names on one shared scale would suggest a difference that is not there,
    # and the shared scale is only legitimate because it is not.
    #
    # The real asymmetry is on y, and it keeps its two names: a fraction of pixels on one
    # side, of solid angle on the other.
    a_title, a_xlab = "rendered frames, all pooled", "radiance (HDR)"
    a_ylab = "fraction of pixels per bin"
    b_title, b_xlab = ("baked skyboxes, solid-angle weighted" if weighted
                       else "baked skyboxes, one count per pixel"), "radiance (HDR)"
    b_ylab = ("fraction of solid angle per bin" if weighted
              else "fraction of pixels per bin")

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    panel(ax, fr_gt, fr_per, fr_w1, runs, xlim, ylim, a_title, a_xlab, a_ylab)
    save(fig, figdir / "spectrum_norm_frames.png")

    if sk is None:
        print("  skybox panel skipped, so no pair figure either")
        return

    # The per-pixel variant owns the plain names, being the default; the weighted one
    # takes the suffix.  Whoever changes the default has to move the suffix with it, or
    # a name already placed on a slide would quietly start meaning the other thing.
    tag = "_solidangle" if weighted else ""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    panel(ax, sk[0], sk[1], sk[2], runs, xlim, ylim, b_title, b_xlab, b_ylab)
    save(fig, figdir / f"spectrum_norm_skybox{tag}.png")

    fig, axs = plt.subplots(1, 2, figsize=(2 * FIG_W, FIG_H))
    panel(axs[0], fr_gt, fr_per, fr_w1, runs, xlim, ylim, a_title, a_xlab, a_ylab)
    panel(axs[1], sk[0], sk[1], sk[2], runs, xlim, ylim, b_title, b_xlab, b_ylab)
    save(fig, figdir / f"spectrum_norm_pair{tag}.png")

    # The page itself.  `spectrum_norm_pair` is 2.7 wide for 1 tall, so on a 4:3 slide it
    # fills the width and leaves nearly 40% of the height empty; this one is the page, so
    # the two panels come out taller than wide and use all of it.
    #
    # The y axis is SHARED when the two panels measure the same fraction, which they do
    # unless --skybox-solid-angle is on: the scale is common anyway, so the second copy of
    # the ticks and of the label was only spending width.  When the two labels differ they
    # both stay, because then dropping one would be claiming they are the same quantity.
    share = a_ylab == b_ylab
    fig, axs = plt.subplots(1, 2, figsize=(SLIDE_W, SLIDE_H), sharey=share)
    panel(axs[0], fr_gt, fr_per, fr_w1, runs, xlim, ylim, a_title, a_xlab, a_ylab,
          fonts=FS_SLIDE, loc="lower right")
    panel(axs[1], sk[0], sk[1], sk[2], runs, xlim, ylim, b_title, b_xlab,
          "" if share else b_ylab, fonts=FS_SLIDE, loc="lower right")
    save(fig, figdir / f"spectrum_norm_slide{tag}.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sweep_root", nargs="?", default=DEFAULT_ROOT)
    ap.add_argument("--comparison", default=None,
                    help="folder holding _cache_metrics.npz and the spectrum CSVs "
                         "(default: <sweep_root>/_comparison)")
    ap.add_argument("--out", default="../../PresentationImages")
    ap.add_argument("--skybox-gt", default=None,
                    help="equirectangular EXR of the original skybox; in this sweep the "
                         "field in run_manifest.json is empty, so it has to be given")
    ap.add_argument("--no-skybox", action="store_true",
                    help="only the frames panel, to iterate without paying the envmap I/O")
    ap.add_argument("--skybox-solid-angle", action="store_true",
                    help="weight the envmap pixels by their solid angle instead of "
                         "counting them: the physically right one for a map that gets "
                         "integrated over the sphere, and the one that survives a change "
                         "of envmap resolution. Default is the per-pixel count, which "
                         "makes the two panels the same quantity")
    a = ap.parse_args()

    root = Path(a.sweep_root)
    out = Path(a.comparison) if a.comparison else root / "_comparison"
    build(root, out, Path(a.out).resolve(), a.skybox_gt, a.no_skybox,
          a.skybox_solid_angle)


if __name__ == "__main__":
    main()
