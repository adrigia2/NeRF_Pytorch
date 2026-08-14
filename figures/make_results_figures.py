#!/usr/bin/env python
"""make_results_figures.py -- the Results-chapter panels that come from the runs.

Seven modes, one per group of figures:

    python make_results_figures.py maps      --out ../Doc/images/results
    python make_results_figures.py views     --out ../Doc/images/results
    python make_results_figures.py highfreq  --out ../Doc/images/results
    python make_results_figures.py curves    --out ../Doc/images/results
    python make_results_figures.py grids     --out ../Doc/images/results
    python make_results_figures.py mapdiff   --out ../Doc/images/results
    python make_results_figures.py spectrum  --out ../Doc/images/results

  maps      the Studio/Night columns of the two map grids (albedo from the fit,
            metallic, roughness) plus the diffuse-cube variant
  views     the "NeRF render" row of the old three-row preview grids
  highfreq  the panels of the detail section: the stone sphere, the emissive sphere,
            and the night sword on which the exponential emits zero
  curves    MSE against iteration for the run that does not train (night sword,
            exponential with L1): one panel, one curve
  spectrum  the distribution of the whole-frame values, original render against that
            column's NeRF, two panels per scene (studio | night)
  grids     the eight preview grids in the new form: four views per row, and for each
            original | reconstruction | difference heatmap, with one table per skybox
            and one per reconstruction (NeRF, re-render)
  mapdiff   the heatmap column of the map grids: the authored reference reduced to the
            atlas resolution, minus the recovered map

`grids` requires rerender_run.py to have been run on the relevant runs already: it reads
<run>/rerender/pbr_gt/images/ and skips the column when it is not there.

Three conventions, each there so as not to falsify a reading the thesis makes on the figures:

  1. The recovered maps are encoded like the authored ones of make_atlas_pngs.py: sRGB on
     the albedo alone, which is colour, and LINEAR on metallic and roughness, which are
     data.  Applying a gamma to a roughness would show a value different from the one the
     fit wrote, and the grid row is there precisely to compare the values between the
     authored column and the recovered ones.

  2. A panel's exposure is never computed on the panel itself: it comes from
     make_scenes_figure.column_exposure(), i.e. from the same median that tonemapped the
     original render at the top of the column.  If every row normalised on its own
     median, a NeRF that gets the mean level wrong would be brought back into scale by
     the tonemap and the figure would show an error that is no longer there.

  3. Where two reconstructions are compared against a reference (the stone sphere, the
     emissive sphere, the night sword) the three panels share ONE exposure, taken from
     the reference.  The difference between the reconstructions is the subject: giving
     each its own would erase it.

The crops are taken from the full-resolution linear image and tonemapped afterwards, for
the same reason make_scenes_figure.py does it: cropping after the downsample would throw
away the detail the panel is meant to show.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _paths  # noqa: F401

from make_atlas_pngs import srgb
from make_skybox_figure import block_mean, load_exr, tonemap
from make_scenes_figure import (
    FAMILIES, SPHERE_CROP, column_exposure, exposure_of, frame_path, save_png,
)

RUNS_ROOT = Path("D:/tesi_output")

# The run that covers all five scenes under both configurations.  Keeping every column
# on a single run is what makes the grids comparable: between two runs, parameters that
# are not the subject of the figure would change too.
#
# Since 2026-08-13 the copy regenerated after the double-precision azimuth fix in
# deviceProgramsIrradiance.cu is used: in `test_sword_shield` the maps derived from the
# irradiance (the irradiance itself, albedo, albedo_pbr) are stale.  Everything else is
# bit-identical between the two trees, verified, so changing the source here touches in
# practice only the albedo row of the two grids.
SWEEP = RUNS_ROOT / "test_sword_shield_after_fix_irradiance"
EXP, SOFT = "exp_l1_d02", "softplus_relmseraw_d02"

# The high-frequency variant stops at the NeRF and lives in a run of its own.
HIGHFREQ_ROOT = RUNS_ROOT / "test_high_details_new_batches"
HIGHFREQ_SCENE = "TableAndOtherInteriorWithSpecularHighDetails"

# The recovered maps, with the encoding each one requires.  albedo_pbr and not albedo:
# the grid is the fit's output, whereas the Lambertian albedo has its own figure in
# Supporting Material and the two are not the same quantity.
MAPS: list[tuple[str, str, bool]] = [
    # (PNG name, path under sources/<source>/, encode in sRGB)
    ("albedo",    "albedo_pbr/albedo_pbr.exr", True),
    ("metallic",  "metallic/metallic.exr",     False),
    ("roughness", "roughness/roughness.exr",   False),
]


@dataclass(frozen=True)
class Column:
    """One column of the grids: one scene under one configuration."""
    folder: str      # subfolder of images/results/
    suffix: str      # "_studio", "_night", or "" when the scene has a single column
    run: Path
    family: str      # make_scenes_figure family, for the camera and the exposure
    scene_key: str   # variant inside the family


COLUMNS: list[Column] = [
    Column("interior", "_studio", SWEEP/EXP/"TableAndOtherInteriorWithSpecular",
           "interior", "specular"),
    Column("interior", "_night",  SWEEP/EXP/"TableAndOtherInteriorWithSpecularNight",
           "interior", "night"),
    Column("sword",    "_studio", SWEEP/EXP/"SwordShieldStudio",
           "sword",    "sword_studio"),
    # The only column of the chapter produced with softplus: under the night map the
    # exponential does not train at all (see sec:results-collapse).
    Column("sword",    "_night",  SWEEP/SOFT/"SwordShieldNight",
           "sword",    "sword_night"),
    Column("diffusecube", "",     SWEEP/EXP/"TableAndOtherInteriorNoSpecular",
           "interior", "diffusecube"),
]


# ──────────────────────────────────────────────────────────────────────────────
# Common utilities
# ──────────────────────────────────────────────────────────────────────────────

def latest_render_dir(run: Path) -> Path:
    """The latest iter_* of nerf_render_images/."""
    dirs = sorted((run / "nerf_render_images").glob("iter_*"))
    if not dirs:
        raise SystemExit(f"ERROR: no iter_* in {run / 'nerf_render_images'}")
    return dirs[-1]


def frame_index(run: Path, camera: str) -> int:
    """Position of the camera in transforms_extended.json, which is the index the frames
    are numbered by in nerf_render_images/.  Resolved rather than written by hand: the
    frame order depends on the dataset, and a wrong index does not give an error, it
    gives another camera's view."""
    frames = json.loads((run / "transforms_extended.json").read_text())["frames"]
    names = [Path(f["file_path"]).stem for f in frames]
    if camera not in names:
        raise SystemExit(f"ERROR: {camera} is not among the frames of {run.name}")
    return names.index(camera)


def rendered_pair(run: Path, camera: str) -> tuple[np.ndarray, np.ndarray]:
    """(gt, pred) of that camera's frame, at full resolution and linear."""
    d = latest_render_dir(run)
    i = frame_index(run, camera)
    return load_exr(d / f"frame_{i:03d}_gt.exr"), load_exr(d / f"frame_{i:03d}_pred.exr")


def check_gt_matches_dataset(col: Column, gt: np.ndarray) -> None:
    """The GT saved by the run has to be the dataset frame make_scenes_figure derived the
    column's exposure from.  If it were not, the exposure taken from there would not be
    this image's and the NeRF render row would not be comparable with the one above: it
    is a silent error, so it is checked and not assumed."""
    fam = FAMILIES[col.family]
    scene_dir = {k: d for k, d, _, _ in fam.scenes}[col.scene_key]
    p = frame_path(scene_dir, fam.camera, fam.root)
    if not p.exists():
        print(f"    ! {p.name} not found in the dataset, check skipped")
        return
    ref = load_exr(p)
    if ref.shape != gt.shape:
        raise SystemExit(f"ERROR: {col.folder}{col.suffix}: the run's GT is "
                         f"{gt.shape} and the dataset frame {ref.shape}")
    d = float(np.abs(ref - gt).max())
    if d > 1e-4:
        raise SystemExit(f"ERROR: {col.folder}{col.suffix}: the run's GT is not the "
                         f"dataset frame (maximum gap {d:.3g})")


def panels(out: Path, names: list[str], images: list[np.ndarray], expo: float) -> None:
    """Write n panels tonemapped with a single exposure."""
    for name, img in zip(names, images):
        save_png(tonemap(img, expo), out / f"{name}.png")


# ──────────────────────────────────────────────────────────────────────────────
# maps
# ──────────────────────────────────────────────────────────────────────────────

def do_maps(out: Path, source: str = "gt", atlas_size: int = 1024) -> None:
    for col in COLUMNS:
        src = col.run / "sources" / source
        print(f"{col.folder}{col.suffix or ' (only)'}  <- {col.run.parent.name}/{col.run.name}")
        dst = out / col.folder
        dst.mkdir(parents=True, exist_ok=True)
        for name, rel, as_srgb in MAPS:
            p = src / rel
            if not p.exists():
                print(f"    ! {rel} does not exist, skipped")
                continue
            a = load_exr(p)
            # In the grids a map is 0.27\linewidth wide, i.e. about 4.3 cm: at 4096
            # texels that is 24000 dpi, and the thesis PDF reached 369 MB with the
            # recovered columns alone.  1024 texels at that size are still 600 dpi,
            # above print resolution.  The block mean has to happen BEFORE the sRGB
            # encoding: averaging already-gammated values would give a different colour.
            k = max(1, a.shape[0] // atlas_size)
            a = block_mean(a, k)
            rgb = srgb(a) if as_srgb else np.clip(a, 0.0, 1.0)
            f = dst / f"{name}{col.suffix}.png"
            plt.imsave(f, rgb)
            print(f"    + {f.name}  {rgb.shape[1]}x{rgb.shape[0]}  "
                  f"range [{a.min():.3f}, {a.max():.3f}]  "
                  f"{'sRGB' if as_srgb else 'linear'}")


# ──────────────────────────────────────────────────────────────────────────────
# views
# ──────────────────────────────────────────────────────────────────────────────

def do_views(out: Path, downsample: int = 2) -> None:
    for col in COLUMNS:
        if not col.suffix:          # the diffuse-cube variant has no preview grid
            continue
        fam = FAMILIES[col.family]
        gt, pred = rendered_pair(col.run, fam.camera)
        check_gt_matches_dataset(col, gt)
        expo = column_exposure(col.family, col.scene_key)
        dst = out / col.folder
        dst.mkdir(parents=True, exist_ok=True)
        name = f"{col.suffix.lstrip('_')}_nerf.png"
        save_png(tonemap(block_mean(pred, downsample), expo), dst / name)
        print(f"    exposure {expo:.4f} from {col.scene_key}, "
              f"median pred/gt {np.median(pred):.5f}/{np.median(gt):.5f}")


# ──────────────────────────────────────────────────────────────────────────────
# highfreq
# ──────────────────────────────────────────────────────────────────────────────

def _detail_stats(name: str, crop: np.ndarray) -> None:
    """The measurements quoted in sec:results-collapse.  The gradient is what separates a
    reconstructed surface from a smoothed one: mean and spread can be right while the
    detail is gone."""
    from make_skybox_figure import LUMA_COEFF
    lum = (crop * LUMA_COEFF).sum(-1)
    print(f"      {name:10s} mean {lum.mean():8.4f}  std {lum.std():8.4f}  "
          f"peak {lum.max():9.2f}  gradient {np.abs(np.diff(lum, axis=1)).mean():.5f}")


def do_highfreq(out: Path, downsample: int = 2) -> None:
    x0, y0, w, h = SPHERE_CROP

    # 1. The stone sphere of the studio interior: neither configuration fails, and the
    #    fine detail is already lost by both.
    print("stone sphere (studio interior)")
    dst = out / "stone"
    dst.mkdir(parents=True, exist_ok=True)
    gt, pred_exp = rendered_pair(SWEEP/EXP/"TableAndOtherInteriorWithSpecular",
                                 FAMILIES["interior"].camera)
    _, pred_soft = rendered_pair(SWEEP/SOFT/"TableAndOtherInteriorWithSpecular",
                                 FAMILIES["interior"].camera)
    crops = [im[y0:y0+h, x0:x0+w] for im in (gt, pred_exp, pred_soft)]
    expo, _ = exposure_of(crops[0])
    for n, c in zip(("gt", "exp", "soft"), crops):
        _detail_stats(n, c)
    panels(dst, ["sphere_gt", "sphere_exp", "sphere_soft"], crops, expo)

    # 2. The emissive sphere: same question, details an order of magnitude brighter than
    #    their base.  Here the two configurations part ways.
    print("emissive sphere (high-frequency variant)")
    dst = out / "highfreq"
    dst.mkdir(parents=True, exist_ok=True)
    gt, pred_exp = rendered_pair(HIGHFREQ_ROOT/EXP/HIGHFREQ_SCENE,
                                 FAMILIES["interior"].camera)
    _, pred_soft = rendered_pair(HIGHFREQ_ROOT/SOFT/HIGHFREQ_SCENE,
                                 FAMILIES["interior"].camera)
    views = [gt, pred_exp, pred_soft]
    crops = [im[y0:y0+h, x0:x0+w] for im in views]
    for n, c in zip(("gt", "exp", "soft"), crops):
        _detail_stats(n, c)
    # The whole view and the crop have two different exposures, both taken from the GT:
    # the emissive curves are so much brighter than the rest that the exposure that makes
    # the table readable burns them out, and vice versa.
    expo_view, _ = exposure_of(views[0])
    expo_crop, _ = exposure_of(crops[0])
    panels(dst, ["view_gt", "view_exp", "view_soft"],
           [block_mean(v, downsample) for v in views], expo_view)
    panels(dst, ["sphere_gt", "sphere_exp", "sphere_soft"], crops, expo_crop)

    # 3. The night sword: the limiting case, the exponential emits zero everywhere.
    print("night sword (the exponential emits zero)")
    dst = out / "collapse"
    dst.mkdir(parents=True, exist_ok=True)
    fam = FAMILIES["sword"]
    gt, pred_exp = rendered_pair(SWEEP/EXP/"SwordShieldNight", fam.camera)
    _, pred_soft = rendered_pair(SWEEP/SOFT/"SwordShieldNight", fam.camera)
    print(f"      exp: maximum {pred_exp.max():.6g}, "
          f"fraction below 1e-3 {(pred_exp < 1e-3).mean():.4f}")
    expo = column_exposure("sword", "sword_night")
    panels(dst, ["swordnight_gt", "swordnight_exp", "swordnight_soft"],
           [block_mean(v, downsample) for v in (gt, pred_exp, pred_soft)], expo)


# ──────────────────────────────────────────────────────────────────────────────
# curves
# ──────────────────────────────────────────────────────────────────────────────

# One panel only, one curve only: the batch MSE of the run that does not train.
# Since 2026-08-13 neither the softplus run alongside nor the loss on a second axis are
# drawn any more.  Putting two runs trained on different losses in the same panel invites
# reading them as a comparison between losses, which is not what they are; the final
# values of the other runs are in the chapter's table, where they are labelled.
CURVE_RUN = SWEEP/EXP/"SwordShieldNight"
CURVE_NAME = "curves_swordnight"
CURVE_TITLE = "Sword and shield, night: \\texttt{exp} with $L_1$"


def _metrics(run: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(iter, mse, loss) from nerf_train/training_metrics.csv."""
    import csv
    rows = list(csv.DictReader((run / "nerf_train" / "training_metrics.csv").open()))
    it = np.array([int(r["iter"]) for r in rows])
    return it, np.array([float(r["mse"]) for r in rows]), \
        np.array([float(r["loss"]) for r in rows])


def do_curves(out: Path) -> None:
    dst = out / "collapse"
    dst.mkdir(parents=True, exist_ok=True)
    it, mse, _ = _metrics(CURVE_RUN)

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(it, mse, color="#c0392b", lw=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("MSE on the training batch")
    ax.set_xlim(0, 75000)
    # Limits on the percentiles and not on the extremes: eight batches out of 808 shoot up
    # to 7e12 and come back within one display interval, and on a scale that contains them
    # all the plateau, which is the subject, becomes a single line.  This way they leave
    # the frame as vertical strokes, visible but not decisive for the scale.
    lo, hi = np.percentile(mse, 1), np.percentile(mse, 98)
    ax.set_ylim(lo / 1.6, hi * 1.6)
    ax.grid(alpha=0.25, which="both", lw=0.4)

    fig.tight_layout()
    fig.savefig(dst / f"{CURVE_NAME}.png", dpi=200)
    plt.close(fig)
    print(f"  + {dst / CURVE_NAME}.png   {mse[0]:.6g} -> {mse[-1]:.6g}")


# ──────────────────────────────────────────────────────────────────────────────
# spectrum -- distribution of the HDR values, original against that column's NeRF
# ──────────────────────────────────────────────────────────────────────────────

# The statistics come from compare_runs.py rather than being rewritten: if the two
# diverged, the scene-section figure and the ablation one would say different things
# about the same quantity.  spectrum_hist is already on the whole frame and without a
# mask, which is exactly the cut needed here.
SPECTRA: list[tuple[str, list[tuple[str, Path, str]]]] = [
    ("interior", [
        ("Studio",  SWEEP/EXP/"TableAndOtherInteriorWithSpecular",     "exp / $L_1$"),
        ("Night",   SWEEP/EXP/"TableAndOtherInteriorWithSpecularNight", "exp / $L_1$"),
    ]),
    ("sword", [
        ("Studio",  SWEEP/EXP/"SwordShieldStudio",  "exp / $L_1$"),
        ("Night",   SWEEP/SOFT/"SwordShieldNight",  "softplus / rel. sq."),
    ]),
]


def _pooled_spectrum(run: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """(gt histogram, pred histogram, number of frames) on the ||RGB|| channel.

    Every frame of the run in a single histogram, whole frame: the NeRF is consumed as a
    radiance source over the entire sphere, not on the subject alone, and separating
    foreground and background is the ablation's cut, not this figure's.
    """
    from compare_runs import spectrum_hist, NORM_CH

    d = latest_render_dir(run)
    gts = sorted(d.glob("frame_*_gt.exr"))
    if not gts:
        raise SystemExit(f"ERROR: no frame_*_gt.exr in {d}")
    h_gt = h_pr = None
    for p in gts:
        pred = p.with_name(p.name.replace("_gt.exr", "_pred.exr"))
        if not pred.exists():
            raise SystemExit(f"ERROR: {pred.name} missing next to {p.name}")
        a = spectrum_hist(load_exr(p))[NORM_CH]
        b = spectrum_hist(load_exr(pred))[NORM_CH]
        h_gt = a if h_gt is None else h_gt + a
        h_pr = b if h_pr is None else h_pr + b
    return h_gt, h_pr, len(gts)


def do_spectrum(out: Path) -> None:
    from compare_runs import (spec_density, spec_w1_dex, _spec_x, _spec_xlim,
                              _spec_ylim)

    x = _spec_x()
    for folder, panels in SPECTRA:
        dst = out / folder
        dst.mkdir(parents=True, exist_ok=True)
        fig, axs = plt.subplots(1, len(panels), figsize=(9.0, 3.4))
        for ax, (title, run, label) in zip(np.atleast_1d(axs), panels):
            h_gt, h_pr, n = _pooled_spectrum(run)
            w1 = float(spec_w1_dex(h_pr, h_gt))
            # Per-panel limits: the two lighting conditions occupy different radiance
            # ranges, and a common axis would crush the narrower one.
            xlim = _spec_xlim(h_gt, h_pr)
            ylim = _spec_ylim(h_gt, h_pr)
            ax.fill_between(x, np.maximum(spec_density(h_gt), 1e-12), 1e-12,
                            step="mid", color="#B0B0B0", alpha=0.75, lw=0,
                            label="original render", zorder=1)
            ax.plot(x, np.maximum(spec_density(h_pr), 1e-12), color="#12436D",
                    lw=1.4, label=f"{label}   $W_1$={w1:.3f} dex", zorder=3)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.grid(alpha=0.25, which="both", lw=0.4)
            ax.set_title(title, fontsize=10)
            # \lVert does not exist in mathtext: the double bar is written \|
            ax.set_xlabel(r"$\|\mathrm{RGB}\|_2$ (linear radiance)", fontsize=8)
            ax.set_ylabel("fraction of pixels per bin", fontsize=8)
            ax.legend(fontsize=7, loc="lower center", framealpha=0.9)
            print(f"    {folder}/{title}: {n} frames, W1 {w1:.4f} dex")
        fig.tight_layout()
        fig.savefig(dst / "spectrum.png", dpi=200)
        plt.close(fig)
        print(f"  + {dst / 'spectrum.png'}")


# ──────────────────────────────────────────────────────────────────────────────
# grids -- le otto griglie preview: originale | ricostruzione | heatmap, su 4 viste
# ──────────────────────────────────────────────────────────────────────────────

# Threshold below which a difference is numerically irrelevant and the panel goes solid
# black.  Same value as compare_exr.py, which is the three-panel figure this grid derives
# from: two different conventions for the same quantity would make the two readings
# incomparable.
DIFF_FLOOR = 1e-4
DIFF_DECADES = 4.0       # maximum width of the log scale, below the ceiling
DIFF_TOP_PCTL = 99.9     # the ceiling, so a single pixel does not dictate it
HEAT_CMAP = "inferno"

N_VIEWS = 4
COV_PCTL = 60.0          # coverage percentile below which a camera is not a candidate


def diff_norm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """||A - B||_2 per pixel on the linear values.  Copied from compare_exr.py."""
    d = a - b
    return np.sqrt(np.einsum("ijk,ijk->ij", d, d, optimize=True))


def _heat_png(d: np.ndarray, vmin: float, vmax: float, out: Path,
              log: bool = True, bad: np.ndarray | None = None) -> None:
    """One heatmap panel, with no axes and no bar: there is one bar per figure."""
    from matplotlib.colors import LogNorm, Normalize
    cmap = plt.get_cmap(HEAT_CMAP).copy()
    cmap.set_under("black")
    cmap.set_bad("#f0f2f4")          # outside the mask: neutral grey, not black
    norm = LogNorm(vmin=vmin, vmax=vmax) if log else Normalize(vmin=vmin, vmax=vmax)
    x = np.ma.masked_where(bad, d) if bad is not None else d
    plt.imsave(out, cmap(norm(x)))
    print(f"  + {out.name}")


def _colorbar_png(vmin: float, vmax: float, out: Path, label: str,
                  log: bool = True, vertical: bool = False) -> None:
    """The figure's scale bar, thin, to be placed next to or below the table.

    One per figure instead of one per panel: in a 4x3 grid four identical bars would be
    nothing but noise.  Horizontal under the preview grids, where the scale is one for the
    whole figure; vertical in the map grids, where every row has its own and the bar has
    to sit beside the row it belongs to.
    """
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.cm import ScalarMappable
    norm = LogNorm(vmin=vmin, vmax=vmax) if log else Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(HEAT_CMAP).copy()
    cmap.set_under("black")
    orient = "vertical" if vertical else "horizontal"
    fig, ax = plt.subplots(figsize=(0.30, 5.0) if vertical else (6.0, 0.42))
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation=orient)
    if vertical:
        # Three values only and no label: the bar sits in a column of the table, and every
        # extra digit is width taken from the panels.  Which quantity it is is said by the
        # caption.
        cb.set_ticks([vmin, 0.5 * (vmin + vmax), vmax])
        ax.set_yticklabels([f"{v:.2f}" for v in
                            (vmin, 0.5 * (vmin + vmax), vmax)], fontsize=11)
    else:
        cb.set_label(label, fontsize=8)
        ax.tick_params(labelsize=7)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  + {out.name}   [{vmin:.3g}, {vmax:.3g}]")


def _camera_table(run: Path) -> list[tuple[str, float, np.ndarray]]:
    """(stem, foreground coverage, unit direction) for every frame of the run.

    The coverage comes from the masks Step 1 already wrote, the direction from the pose:
    both live in the run, so the choice of views does not depend on anything that has to
    be recomputed."""
    frames = json.loads((run / "transforms_extended.json").read_text())["frames"]
    rows = []
    for f in frames:
        m = np.asarray(Image.open(run / f["mask_path"]))
        pos = np.array(f["transform_matrix"], dtype=np.float64)[:3, 3]
        rows.append((Path(f["file_path"]).stem, float((m > 127).mean()),
                     pos / max(np.linalg.norm(pos), 1e-9)))
    return rows


def select_cameras(run: Path, seed: str, n: int = N_VIEWS,
                   cov_pctl: float = COV_PCTL) -> list[str]:
    """The grid's n views, chosen by a rule instead of by hand.

    Seeded on the camera the chapter already uses in the scene figures, so the grid stays
    anchored to a view the reader has already seen; the others by farthest-point sampling
    on the camera direction, restricted to those that frame enough subject.  Without the
    coverage constraint the sampling would pick the views farthest from each other, which
    are also the ones looking at the back or at nothing; without the farthest-point rule
    the four views would be four variants of the same one."""
    rows = _camera_table(run)
    cov = np.array([r[1] for r in rows])
    thr = float(np.percentile(cov, cov_pctl))
    cand = [r for r in rows if r[1] >= thr or r[0] == seed]
    if seed not in [r[0] for r in cand]:
        raise SystemExit(f"ERROR: the seed camera {seed} is not among the frames of {run.name}")

    chosen = [next(r for r in cand if r[0] == seed)]
    while len(chosen) < n:
        best, best_d = None, -1.0
        for r in cand:
            if r[0] in [c[0] for c in chosen]:
                continue
            d = min(float(np.arccos(np.clip(r[2] @ c[2], -1, 1))) for c in chosen)
            if d > best_d:
                best, best_d = r, d
        chosen.append(best)
    print(f"  views (minimum coverage {100 * thr:.1f}%, {len(cand)} candidates):")
    for stem, c, _ in chosen:
        print(f"     {stem:28s} coverage {100 * c:5.2f}%")
    return [c[0] for c in chosen]


def _grid_columns() -> list[Column]:
    return [c for c in COLUMNS if c.suffix]


def do_grids(out: Path, downsample: int = 3, cameras: list[str] | None = None) -> None:
    for col in _grid_columns():
        fam = FAMILIES[col.family]
        sky = col.suffix.lstrip("_")
        print(f"\n{col.folder} / {sky}  <- {col.run.parent.name}/{col.run.name}")
        rer_dir = col.run / "rerender" / "pbr_gt" / "images"
        if not rer_dir.is_dir():
            print(f"  ! {rer_dir} does not exist: run rerender_run.py first")
            continue

        views = cameras or select_cameras(col.run, fam.camera)
        rdir = latest_render_dir(col.run)
        missing = [c for c in views if not (rer_dir / f"{c}.exr").exists()]
        if missing:
            print(f"  ! incomplete re-render, {len(missing)} views missing "
                  f"({', '.join(missing)}): column skipped")
            continue

        # First pass: every image is read and the differences computed, so that ONE scale
        # can be fixed over all eight (4 views x 2 reconstructions).  It is that shared
        # scale that makes the comparison between the NeRF table and the re-render one
        # readable: with one scale per table the question "which of the two is closer to
        # the original" would have no answer in the figure.
        data = {}
        for cam in views:
            i = frame_index(col.run, cam)
            orig = load_exr(col.run / "images" / f"{cam}.exr")
            gt = load_exr(rdir / f"frame_{i:03d}_gt.exr")
            # The frame index is resolved by name, and a wrong one gives no error: it
            # gives another camera's view, which a heatmap full of structure would make
            # look like a reconstruction error.  So it is checked.
            d = float(np.abs(orig - gt).max())
            if d > 1e-4:
                raise SystemExit(f"ERROR: {cam} -> frame_{i:03d}: the run's GT is not "
                                 f"the dataset frame (maximum gap {d:.3g})")
            nerf = load_exr(rdir / f"frame_{i:03d}_pred.exr")
            rer = load_exr(rer_dir / f"{cam}.exr")
            fg = np.asarray(Image.open(col.run / "mask" /
                                       f"{cam}_mask.png")) > 127
            data[cam] = (orig, {"nerf": nerf, "rerender": rer},
                         {k: diff_norm(orig, v) for k, v in
                          (("nerf", nerf), ("rerender", rer))}, fg)

        pooled = np.concatenate([d.ravel() for _, _, dd, _ in data.values()
                                 for d in dd.values()])
        vmax = float(np.percentile(pooled, DIFF_TOP_PCTL))
        vmin = max(DIFF_FLOOR, vmax / 10.0 ** DIFF_DECADES)
        print(f"  shared scale [{vmin:.3g}, {vmax:.3g}] "
              f"({np.log10(vmax / vmin):.1f} decades, log, ceiling at p{DIFF_TOP_PCTL})")

        dst = out / col.folder / "grid"
        dst.mkdir(parents=True, exist_ok=True)
        for k, cam in enumerate(views):
            orig, recon, diffs, fg = data[cam]
            # One exposure per row, taken from that row's ORIGINAL: it is rule 2 of the
            # docstring.  Different rows have different exposures because they are
            # different views, but the panels of one row do not, or the tonemap would put
            # back into scale the mean-level error the row is meant to show.
            expo, med = exposure_of(orig)
            save_png(tonemap(block_mean(orig, downsample), expo),
                     dst / f"{sky}_v{k}_orig.png")
            for which in ("nerf", "rerender"):
                save_png(tonemap(block_mean(recon[which], downsample), expo),
                         dst / f"{sky}_v{k}_{which}.png")
                _heat_png(block_mean(diffs[which][..., None], downsample)[..., 0],
                          vmin, vmax, dst / f"{sky}_v{k}_{which}_heat.png")
            # Two medians, and the second is the one that counts.  On a view with a lot of
            # background the whole-frame median mostly measures the environment, which the
            # re-render reproduces exactly and the NeRF does not: on its own it would say
            # the re-render is nearly perfect even where it gets all the geometry wrong.
            print(f"    {cam}: exposure {expo:.4f} (median {med:.4f}); p50 |diff| "
                  f"all  nerf {np.median(diffs['nerf']):.4f} / "
                  f"rerender {np.median(diffs['rerender']):.4f}   "
                  f"foreground  nerf {np.median(diffs['nerf'][fg]):.4f} / "
                  f"rerender {np.median(diffs['rerender'][fg]):.4f}")
        _colorbar_png(vmin, vmax, dst / f"{sky}_cbar.png",
                      r"$\|\Delta$RGB$\|_2$  (linear radiance, log scale)")


# ──────────────────────────────────────────────────────────────────────────────
# mapdiff -- la colonna heatmap delle griglie delle mappe
# ──────────────────────────────────────────────────────────────────────────────

# (name, authored file, recovered path under sources/<source>/, is it scalar?)
#
# The last field is NOT a convenience detail.  metallic and roughness are single-channel
# quantities, but `load_exr` replicates the single channel onto three for uniformity:
# taking their L2 norm would multiply the difference by sqrt(3), and indeed the maximum
# came out 1.7321 on maps that live in [0,1].  On the scalar quantities |delta| on a
# single channel is used; on the albedo, which really is a colour, the norm over three.
#
# Note for the caption: two rows out of three are not an error against a reference.  The
# recovered roughness is the cone-aperture index and not a GGX roughness
# (sec:metallic-roughness), and on the cube's island the authored base colour is the
# specular tint F0 and not a diffuse albedo (sec:results-gt): in both cases the map shows
# a DISAGREEMENT, not a gap from a correct value.
MAPDIFF: list[tuple[str, str, str, bool]] = [
    ("albedo",    "BakedMaterial_base_color.exr", "albedo_pbr/albedo_pbr.exr", False),
    ("metallic",  "BakedMaterial_metallic.exr",   "metallic/metallic.exr",     True),
    ("roughness", "BakedMaterial_roughness.exr",  "roughness/roughness.exr",   True),
]

MAPDIFF_TOP_PCTL = 99.0


def _axis_weights(n: int, m: int):
    """Sparse (m, n) matrix of the area mean along one axis.

    Generalises the block mean to a non-integer ratio, which is needed because the sword's
    authored bake is 8096 and not 8192: 8096/2 = 4048 cannot be subtracted from a 4096
    atlas.  Each row has two or three non-zero elements, so the sum stays exact in float32
    (unlike a running sum, which over 8096 terms would lose exactly the digits the
    difference has to show).
    """
    import scipy.sparse as sp
    s = n / m
    rows, cols, vals = [], [], []
    for j in range(m):
        a0, a1 = j * s, (j + 1) * s
        for i in range(int(np.floor(a0)), min(int(np.ceil(a1)), n)):
            w = min(a1, i + 1.0) - max(a0, float(i))
            if w > 0:
                rows.append(j)
                cols.append(i)
                vals.append(w / s)
    return sp.csr_matrix((vals, (rows, cols)), shape=(m, n), dtype=np.float32)


def reduce_to_atlas(img: np.ndarray, side: int) -> np.ndarray:
    """The authored reference reduced to the atlas resolution, by area mean.

    The protocol of sec:results-gt: never by point sampling, which would throw away three
    quarters of the authored signal and count the resulting aliasing as a reconstruction
    error.  With an integer ratio it is exactly the block mean.
    """
    h, w = img.shape[:2]
    if h == w and h % side == 0:
        return block_mean(img, h // side)
    a = img.astype(np.float32)
    t = (_axis_weights(h, side) @ a.reshape(h, -1)).reshape(side, w, -1)
    t = (_axis_weights(w, side) @ t.transpose(1, 0, 2).reshape(w, -1))
    return t.reshape(side, side, -1).transpose(1, 0, 2)


def _authored_dir(run: Path) -> Path:
    """The authored bake folder, from the run's manifest: it is the external normal's, the
    only one of the four files the pipeline receives as input."""
    m = json.loads((run / "run_manifest.json").read_text())
    p = m["scene"].get("external_normal_path")
    if not p:
        raise SystemExit(f"ERROR: {run.name} does not record scene.external_normal_path")
    return Path(p).parent


def do_mapdiff(out: Path, source: str = "gt", atlas_size: int = 1024) -> None:
    for scene in ("interior", "sword"):
        cols = [c for c in _grid_columns() if c.folder == scene]
        print(f"\n{scene}: {' e '.join(c.suffix.lstrip('_') for c in cols)}")
        for name, authored_name, rel, scalar in MAPDIFF:
            # One scale per map, shared between studio and night: the authored file is the
            # same in both, so the comparison the figure is for is precisely which
            # lighting recovers better, and with two scales it would disappear.
            diffs, masks = {}, {}
            for col in cols:
                ref_p = _authored_dir(col.run) / authored_name
                rec_p = col.run / "sources" / source / rel
                if not ref_p.exists() or not rec_p.exists():
                    print(f"  ! {name}: missing {ref_p if not ref_p.exists() else rec_p}")
                    break
                rec = load_exr(rec_p)
                raw = load_exr(ref_p)
                ref = reduce_to_atlas(raw, rec.shape[0])
                mask = load_exr(col.run / "ium" / "ium_masks.exr")[..., 0] > 0.5
                d = (np.abs(ref[..., 0] - rec[..., 0]) if scalar
                     else diff_norm(ref, rec))
                diffs[col.suffix] = d
                masks[col.suffix] = mask
                print(f"  {name:9s} {col.suffix.lstrip('_'):6s} authored "
                      f"{raw.shape[0]} -> {rec.shape[0]}; "
                      f"|diff| over the covered texels: p50 {np.median(d[mask]):.4f}  "
                      f"p99 {np.percentile(d[mask], 99):.4f}")
                del raw
            if len(diffs) != len(cols):
                continue

            pooled = np.concatenate([d[masks[k]].ravel() for k, d in diffs.items()])
            vmax = float(np.percentile(pooled, MAPDIFF_TOP_PCTL))
            # Linear and not logarithmic: here the two quantities live in [0,1] and the
            # difference is bounded, whereas in the preview grids it is HDR radiance.
            for col in cols:
                dst = out / col.folder / "mapdiff"
                dst.mkdir(parents=True, exist_ok=True)
                d, mask = diffs[col.suffix], masks[col.suffix]
                k = max(1, d.shape[0] // atlas_size)
                _heat_png(block_mean(d[..., None], k)[..., 0], 0.0, vmax,
                          dst / f"{name}{col.suffix}_diff.png", log=False,
                          bad=~(block_mean(mask[..., None].astype(np.float32), k)[..., 0]
                                > 0.5))
            # Vertical: here the scale is one per row, not one per figure, and the bar has
            # to sit beside the row it belongs to.
            _colorbar_png(0.0, vmax, out / scene / "mapdiff" / f"{name}_cbar.png",
                          r"$|\Delta|$" if scalar else r"$\|\Delta\|_2$",
                          log=False, vertical=True)


# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("maps", "views", "highfreq", "curves",
                                     "grids", "mapdiff", "spectrum", "all"))
    ap.add_argument("--out", required=True, help="the Doc/images/results folder")
    ap.add_argument("--downsample", type=int, default=2,
                    help="block mean on the whole views (default 2)")
    ap.add_argument("--atlas-size", type=int, default=1024,
                    help="side the texture-space maps are reduced to (default 1024)")
    ap.add_argument("--cameras", nargs="+", default=None,
                    help="force the grid views instead of choosing them by coverage "
                         "and angular spacing (grids mode only)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    modes = (("maps", "views", "highfreq", "curves", "grids", "mapdiff", "spectrum")
             if args.mode == "all" else (args.mode,))
    for m in modes:
        print(f"\n=== {m} ===")
        if m == "maps":
            do_maps(out, atlas_size=args.atlas_size)
        elif m == "views":
            do_views(out, args.downsample)
        elif m == "highfreq":
            do_highfreq(out, args.downsample)
        elif m == "grids":
            # 3 and not 2: with 96 new panels the PDF weight matters, and 640x360 at
            # 0.30\linewidth are still ~340 dpi, above print resolution.
            do_grids(out, downsample=3, cameras=args.cameras)
        elif m == "mapdiff":
            do_mapdiff(out, atlas_size=args.atlas_size)
        elif m == "spectrum":
            do_spectrum(out)
        else:
            do_curves(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
