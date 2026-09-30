#!/usr/bin/env python
"""make_nerf_configs_view.py -- one view of the studio interior, rendered by each NeRF.

    python make_nerf_configs_view.py [--out ../../PresentationImages] [--crop X0 Y0 W H]

Reads the Step 2b renders already saved by the ablation run
(`D:/tesi_output/sweep_nerf_activation_loss_decay_find_better_nerf`, six configurations =
two activations x three losses, scene TableAndOtherInteriorWithSpecular) and writes
`nerf_configs_view.png`: the original render on the left and, on the right, the same
camera rendered by the six networks in the 2 x 3 grid the ablation figures use (rows =
activation, columns = loss).  It is the slide that shows *what* changes between the
models before any number is shown.

With `--what skybox` the same layout is filled with the environment maps instead: the
original `wooden_studio_13_4k.exr` on top and, in the grid, the `skybox_nerf_baked.exr`
each run baked from its own NeRF (`nerf_configs_skybox.png`), tonemapped with the key
0.5 that `make_skybox_figure.py` uses for the thesis figure.

Nothing under the run is written, ever: the script only opens the EXRs.

Choices that matter:

  * The camera is `render_Camera_Shell21_38`, the one the Results chapter uses for this
    scene (sphere, rabbit and cube do not overlap).  Its frame index is resolved from
    `transforms_extended.json`, as in `make_results_figures.frame_index`, because a wrong
    index does not fail, it shows another camera.
  * One exposure for all seven panels, taken from the median luminance of the ground
    truth (`make_scenes_figure.exposure_of`, key 0.20) and applied with the Reinhard +
    gamma 2.2 of `make_skybox_figure.tonemap`.  Per-panel exposures would pull a network
    that gets the mean level wrong back into scale and hide exactly the error the slide
    is meant to show.
  * The six runs share the frame order and their `gt.exr` are bit-identical (checked
    here, not assumed), so a single ground-truth panel is honest.
  * Downsampling is a block mean in linear space (`block_mean`), which preserves the
    radiance of small bright sources; a resampling filter would not.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import _paths  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402

from compare_runs import COLORS                                   # noqa: E402
from make_loss_curves_figure import ACT_NAME                      # noqa: E402
from make_scenes_figure import exposure_of                        # noqa: E402
from make_skybox_figure import block_mean, load_exr, tonemap      # noqa: E402

RUN = Path("D:/tesi_output/sweep_nerf_activation_loss_decay_find_better_nerf")
SCENE = "TableAndOtherInteriorWithSpecular"
CAMERA = "render_Camera_Shell21_38"

ACTS = ("exp", "softplus")
LOSSES = ("l1", "mse", "rel_mse_raw")
RUN_DIR = {"l1": "l1", "mse": "mse", "rel_mse_raw": "relmseraw"}   # folder naming
LOSS_TITLE = {"l1": "L1", "mse": "squared", "rel_mse_raw": "relative squared"}

SKYBOX_GT = (Path(__file__).resolve().parents[2] / "OptixProjectCMake" / "Scenes"
             / "TableAndOtherInterior" / "Blender" / "assets" / "hdri"
             / "wooden_studio_13_4k.exr")
BAKED_NAME = "skybox_nerf_baked.exr"

C_INK = "#33404d"
C_MUTED = "#5f6c79"
C_LINE = "#d5dbe1"


def scene_dir(act: str, loss: str) -> Path:
    return RUN / f"{act}_{RUN_DIR[loss]}_d02" / SCENE


def latest_render_dir(run: Path) -> Path:
    dirs = sorted((run / "nerf_render_images").glob("iter_*"))
    if not dirs:
        raise SystemExit(f"no iter_* in {run / 'nerf_render_images'}")
    return dirs[-1]


def frame_index(run: Path, camera: str) -> int:
    frames = json.loads((run / "transforms_extended.json").read_text())["frames"]
    names = [Path(f["file_path"]).stem for f in frames]
    if camera not in names:
        raise SystemExit(f"{camera} is not among the frames of {run}")
    return names.index(camera)


def load_all(camera: str) -> tuple[np.ndarray, dict]:
    """(gt, {(act, loss): pred}) at full resolution, linear; checks the GTs agree."""
    gt = None
    preds = {}
    for act in ACTS:
        for loss in LOSSES:
            run = scene_dir(act, loss)
            d = latest_render_dir(run)
            i = frame_index(run, camera)
            g = load_exr(d / f"frame_{i:03d}_gt.exr")
            p = load_exr(d / f"frame_{i:03d}_pred.exr")
            if gt is None:
                gt = g
            elif not np.array_equal(gt, g):
                raise SystemExit(f"GT of {run} differs from the first run's: "
                                 "the panels would not share a reference")
            if p.shape != gt.shape:
                raise SystemExit(f"{run}: pred {p.shape} vs gt {gt.shape}")
            preds[(act, loss)] = p
            print(f"  {act:>8s}/{loss:<12s} {d.name}  frame {i:03d}")
    return gt, preds


def load_skyboxes() -> tuple[np.ndarray, dict]:
    """(original envmap, {(act, loss): envmap baked from that NeRF}), linear."""
    gt = load_exr(SKYBOX_GT)
    preds = {}
    for act in ACTS:
        for loss in LOSSES:
            p = scene_dir(act, loss) / BAKED_NAME
            a = load_exr(p)
            if a.shape != gt.shape:
                raise SystemExit(f"{p}: {a.shape} vs original {gt.shape}")
            preds[(act, loss)] = a
            print(f"  {act:>8s}/{loss:<12s} {p.name}")
    return gt, preds


def build(out: Path, camera: str, crop: tuple[int, int, int, int] | None,
          downsample: int, what: str = "view", key: float | None = None) -> None:
    gt, preds = load_all(camera) if what == "view" else load_skyboxes()
    if crop is not None:
        x0, y0, w, h = crop
        sl = (slice(y0, y0 + h), slice(x0, x0 + w))
        gt = gt[sl]
        preds = {k: v[sl] for k, v in preds.items()}
        downsample = 1
    expo, med = exposure_of(gt) if key is None else exposure_of(gt, key)
    print(f"  exposure {expo:.3f} from GT median luminance {med:.4f}")

    def show(img):
        return tonemap(block_mean(img, downsample) if downsample > 1 else img, expo)

    H, W = gt.shape[:2]
    aspect = W / H

    # -- layout: GT on top, centred; the 2 x 3 grid under it; every panel 16:9 ---------
    # Units: figure width = 1.  Heights are converted with fw2h once the figure size is
    # known, so the panels keep the frame's aspect whatever the final canvas is.
    fig_w = 16.0
    gap = 0.012
    left, right = 0.012, 0.975
    grid_w = right - left
    cell_w = (grid_w - 2 * gap) / 3
    cell_h = cell_w / aspect
    gt_w = 1.7 * cell_w
    gt_h = gt_w / aspect
    title_h = 0.035                     # room for the panel titles, width units
    body = title_h + gt_h + 0.02 + title_h + 2 * cell_h + gap
    fig_h = fig_w * (body + 0.02)
    fig = plt.figure(figsize=(fig_w, fig_h))
    fw2h = fig_w / fig_h

    def add(x, y_top, w, h):
        return fig.add_axes([x, 1.0 - (y_top + h) * fw2h, w, h * fw2h])

    y = 0.01 + title_h
    ax = add(left + (grid_w - gt_w) / 2, y, gt_w, gt_h)
    ax.imshow(show(gt), interpolation="lanczos")
    ax.axis("off")
    ax.set_title("original render" if what == "view" else "original skybox",
                 fontsize=18, color=C_INK, weight="bold", pad=8)

    y_grid = y + gt_h + 0.02 + title_h
    for r, act in enumerate(ACTS):
        for c, loss in enumerate(LOSSES):
            x = left + c * (cell_w + gap)
            ax = add(x, y_grid + r * (cell_h + gap), cell_w, cell_h)
            ax.imshow(show(preds[(act, loss)]), interpolation="lanczos")
            ax.axis("off")
            col = COLORS[(act, loss)]
            ax.text(0.02, 0.96, f"{act}/{loss}", transform=ax.transAxes, fontsize=13,
                    color="white", ha="left", va="top", family="monospace", weight="bold",
                    bbox=dict(boxstyle="round,pad=0.35", fc=col, ec="none", alpha=0.95))
            if r == 0:
                ax.set_title(LOSS_TITLE[loss], fontsize=16, color=C_INK, pad=6)
    for r, act in enumerate(ACTS):
        y_mid = 1.0 - (y_grid + r * (cell_h + gap) + cell_h / 2) * fw2h
        fig.text(right + 0.001, y_mid, ACT_NAME[act], fontsize=15, color=C_INK,
                 rotation=90, ha="left", va="center")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  + {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="../../PresentationImages")
    ap.add_argument("--camera", default=CAMERA)
    ap.add_argument("--crop", nargs=4, type=int, metavar=("X0", "Y0", "W", "H"),
                    default=None, help="full-resolution pixel crop, same for all panels")
    ap.add_argument("--downsample", type=int, default=None,
                    help="block-mean factor (default: 2 for the view, 4 for the skybox)")
    ap.add_argument("--what", choices=["view", "skybox"], default="view",
                    help="'view': the Shell21_38 render; 'skybox': the baked envmap")
    ap.add_argument("--key", type=float, default=None,
                    help="tonemap key for the GT median (default 0.20 view, 0.5 skybox, "
                         "the values the thesis figures use)")
    a = ap.parse_args()
    if a.what == "view":
        name = "nerf_configs_view.png" if a.crop is None else "nerf_configs_view_crop.png"
        ds, key = a.downsample or 2, a.key
    else:
        name = "nerf_configs_skybox.png"
        ds, key = a.downsample or 4, (0.5 if a.key is None else a.key)
    build(Path(a.out) / name, a.camera, tuple(a.crop) if a.crop else None, ds,
          what=a.what, key=key)


if __name__ == "__main__":
    main()
