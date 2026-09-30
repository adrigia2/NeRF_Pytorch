#!/usr/bin/env python
"""make_spec_cone_strip.py -- one camera's view, and the cones baked for that camera.

    python make_spec_cone_strip.py <run_dir> --camera 38 --out ../../PresentationImages

Writes `spec_cone_strip.png`: the render of one camera beside the texture the specular cone
bake produced for it at several apertures, for the bottom of the SpecCone slide.

The panels are read from the run's own artefacts, `spec_cone/cam_{j:03d}.exr`, one RGB
channel group per level, named for the aperture by `spec_cone_level_name` and read with
`_ExrBandReader` in bands: at full resolution one camera's file is a few GB, and the
solver's own reader mounts the framebuffer once instead of decompressing the file per
channel.

ONE EXPOSURE FOR ALL THE PANELS, the p95 of the luminance of the MIRROR level over the
texels the bake marked valid.  It is the whole point of the strip: what changes from panel
to panel is the width of the cone the radiance was averaged over, so the panels have to be
comparable to each other or the smoothing they show is an artefact of six different
exposures.  The mirror sets the scale because it is the brightest of them, being the one
average that never smooths anything.

The reduction to a readable size is a MASKED block mean: a plain mean over a block that
straddles the edge of an island mixes in the zeros outside it and draws a dark rim around
every island that is not in the data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import _paths  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.patches import Rectangle                               # noqa: E402

from images_generator import spec_cone_level_name                      # noqa: E402
from pbr_solver import _ExrBandReader                                  # noqa: E402
from make_geometry_diagrams import C_EDGE, C_CAM                       # noqa: E402

C_INK = C_CAM
C_MUTED = C_EDGE

SHOWN_DEG = [0.0, 15.0, 30.0, 60.0, 120.0, 180.0]
PCTL = 95.0                    # of the mirror level, the factor every panel is divided by
DOWN = 4                       # 4096 -> 1024
BAND = 512                     # scanlines per read

VIEW = "Doc/images/scenes/specular_view.png"     # the render of camera 38

# -- Layout, in axes units (10 to the inch) -----------------------------------
MARGIN = 3.0
DPI = 200

VIEW_BOX = (6.0, 14.0, 50.0, 47.0)     # x0, y0, x1, y1
CELL = 24.0
COLS = (56.0, 83.0, 110.0)
ROWS = (32.0, 5.0)                     # bottom edge of the top and the bottom row


def to_ar(img, ar):
    h, w = img.shape[:2]
    nw = int(round(h * ar))
    if w >= nw:
        x0 = (w - nw) // 2
        return img[:, x0:x0 + nw]
    out = np.zeros((h, nw, 3), dtype=img.dtype)
    out[:, (nw - w) // 2:(nw - w) // 2 + w] = img
    return out


def masked_block_mean(v: np.ndarray, m: np.ndarray, f: int):
    """Mean of `v` over f x f blocks, counting only the texels `m` marks valid."""
    h, w = m.shape
    h, w = (h // f) * f, (w // f) * f
    c = v.shape[2]
    vb = (v[:h, :w] * m[:h, :w, None]).reshape(h // f, f, w // f, f, c).sum(axis=(1, 3))
    mb = m[:h, :w].reshape(h // f, f, w // f, f).sum(axis=(1, 3))
    out = np.zeros_like(vb)
    ok = mb > 0
    out[ok] = vb[ok] / mb[ok, None]
    return out, ok


def read_levels(run: Path, cam: int, apertures):
    """The chosen levels of one camera's cone file, reduced and masked."""
    names = [spec_cone_level_name(apertures, apertures.index(d)) for d in SHOWN_DEG]
    chans = [f"{n}.{c}" for n in names for c in "RGB"] + ["valid"]

    with _ExrBandReader(run / "spec_cone" / f"cam_{cam:03d}.exr") as r:
        h, w = r.height, r.width
        tiles, masks = [], []
        for y0 in range(0, h, BAND):
            rows = min(BAND, h - y0)
            a = r.read(y0, rows, chans).reshape(rows, w, len(chans))
            v = a[..., :-1].reshape(rows, w, len(names), 3)
            m = a[..., -1] > 0.0
            red, ok = masked_block_mean(v.reshape(rows, w, len(names) * 3), m, DOWN)
            tiles.append(red.reshape(-1, w // DOWN, len(names), 3))
            masks.append(ok)
    return np.concatenate(tiles, axis=0), np.concatenate(masks, axis=0), names


def tonemap(levels: np.ndarray, mask: np.ndarray):
    """One factor for every panel: the p95 of the mirror, which is the brightest level."""
    lum = levels[..., 0, :] @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    k = float(np.percentile(lum[mask], PCTL))
    out = np.clip(levels / max(k, 1e-8), 0.0, 1.0) ** (1.0 / 2.2)
    out[~mask] = 0.0
    return out, k


def build(run: Path, root: Path, cam: int, out: Path) -> None:
    meta = json.loads((run / "spec_cone" / "spec_cone_meta.json").read_text())
    if meta.get("format") != "cones":
        raise SystemExit(f"{run}: spec_cone is format {meta.get('format')!r}, not 'cones'")
    apertures = [float(a) for a in meta["apertures_deg"]]
    missing = [d for d in SHOWN_DEG if d not in apertures]
    if missing:
        raise SystemExit(f"apertures {missing} are not in the bake: {apertures}")

    levels, mask, names = read_levels(run, cam, apertures)
    panels, k = tonemap(levels, mask)

    fig = plt.figure(figsize=(20.0, 14.0))       # scratch canvas, cropped below
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 140)

    view = np.asarray(plt.imread(root / VIEW), dtype=np.float32)[..., :3]
    x0, y0, x1, y1 = VIEW_BOX
    ax.imshow(to_ar(view, (x1 - x0) / (y1 - y0)), extent=(x0, x1, y0, y1),
              aspect="auto", zorder=3, interpolation="antialiased")
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none",
                           edgecolor=C_INK, lw=1.5, zorder=4))

    for i, deg in enumerate(SHOWN_DEG):
        cx, cy = COLS[i % 3], ROWS[i // 3]
        ax.imshow(panels[..., i, :], extent=(cx, cx + CELL, cy, cy + CELL),
                  aspect="auto", zorder=3, interpolation="antialiased")
        ax.add_patch(Rectangle((cx, cy), CELL, CELL, facecolor="none", edgecolor=C_INK,
                               lw=1.2, zorder=4))
        # inside the panel: the atlas is black there, and a caption under every cell
        # would cost a row of height the strip does not have
        label = "mirror" if deg == 0.0 else f"{deg:g}°"
        ax.text(cx + 1.4, cy + CELL - 1.4, label, fontsize=15, color="white",
                ha="left", va="top", zorder=5)

    ax.text(VIEW_BOX[0], 1.0,
            f"one exposure for all six, the p{PCTL:g} of the mirror",
            fontsize=14, color=C_MUTED, ha="left", va="center")

    fx0, fx1 = VIEW_BOX[0] - MARGIN, COLS[-1] + CELL + MARGIN
    fy0, fy1 = 1.0 - 2.0 - MARGIN, ROWS[0] + CELL + MARGIN
    ax.set_xlim(fx0, fx1)
    ax.set_ylim(fy0, fy1)
    fig.set_size_inches((fx1 - fx0) / 10.0, (fy1 - fy0) / 10.0)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"  camera {cam}, levels {names}")
    print(f"  shared factor k = {k:.4f}   valid texels {100 * mask.mean():.1f} % of the atlas")
    print(f"  frame {fx1 - fx0:.1f} x {fy1 - fy0:.1f} units, "
          f"ratio {(fx1 - fx0) / (fy1 - fy0):.3f}")
    print(f"  + {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", help="the run folder (holds spec_cone/)")
    ap.add_argument("--camera", type=int, default=38,
                    help="index of the camera, as ordered in transforms_extended.json")
    ap.add_argument("--out", default="../../PresentationImages")
    ap.add_argument("--assets", default=None,
                    help="folder the view path is relative to (default: the repo root)")
    a = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]
    root = Path(a.assets) if a.assets else repo
    build(Path(a.run_dir), root, a.camera, Path(a.out).resolve() / "spec_cone_strip.png")


if __name__ == "__main__":
    main()
