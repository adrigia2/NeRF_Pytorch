#!/usr/bin/env python
"""make_camera_texture_figure.py -- the per-camera colour textures, figure 3.7.

    python make_camera_texture_figure.py <run_dir> \
        --cameras render_Camera_Shell21_38 render_Camera_Shell10_12 \
        --out ../../Doc/images/camera_texture

Four panels, two per camera:

  view_a.png / view_b.png    the photograph the camera took
  atlas_a.png / atlas_b.png  the texture the colour-texture pass produced from it,
                             i.e. those same pixels moved into the UV atlas, black
                             on the texels that camera does not see

The point of the figure is that the atlas holds THE SAME VALUES as the photograph,
only reordered, so the conversion to PNG is a plain clamp to [0, 1]: no percentile
normalization and no gamma, unlike make_visibility_figure, because any exposure
factor would be one more transformation standing between the two panels of a row.
The data is HDR and unbounded above, so the bright parts saturate to white; the
caption has to say so, and this script prints the per-channel maxima and the
fraction of clipped pixels to write that sentence on measured numbers.

The two atlases are cropped to the used area of the atlas with the same
`content_box` as every other texture-space figure, so they stay comparable with
figures 3.8 and 3.9.  The photographs are left at full frame.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")

import _paths  # noqa: F401

from make_depth_figure import content_box, save_png            # noqa: E402
from make_skybox_figure import load_exr                        # noqa: E402
from make_visibility_figure import crop_to_atlas               # noqa: E402

SLOTS = ("a", "b")


def find_photo(run: Path, stem: str) -> Path:
    """The source image of a frame.  The pipeline copies it into <run>/images/ keeping
    the extension of the capture, which is EXR here but need not be."""
    hits = sorted((run / "images").glob(f"{stem}.*"))
    hits = [p for p in hits if p.is_file()]
    if not hits:
        raise SystemExit(f"x photograph not found: {run / 'images' / stem}.*")
    return hits[0]


def report(name: str, img: np.ndarray) -> None:
    """What the clamp costs on this panel: the maxima it cuts and how much it cuts."""
    mx = img.reshape(-1, img.shape[-1]).max(axis=0)
    clipped = 100.0 * (img > 1.0).any(axis=-1).mean()
    print(f"  {name:14s} max per channel "
          f"({mx[0]:.3f}, {mx[1]:.3f}, {mx[2]:.3f})  "
          f"clamped {clipped:.2f}% of the pixels")


def coverage(run: Path, stem: str, mask: np.ndarray) -> np.ndarray:
    """The texels this camera really sees, from its camera_mask (occlusion, frustum and
    grazing together).  Returned so the caller can also report the overlap of the pair,
    which is what decides whether two cameras are worth putting in the figure."""
    p = run / "camera_mask" / f"{stem}.exr"
    if not p.exists():
        raise SystemExit(f"x per-camera mask not found: {p}")
    cam = load_exr(p)[..., 0] > 0.5
    print(f"  {stem:32s} sees {100.0 * cam[mask].mean():5.1f}% of the mesh texels")
    return cam


def disagreement(a: np.ndarray, b: np.ndarray, both: np.ndarray) -> None:
    """How much the two atlases differ ON THE TEXELS BOTH CAMERAS SEE.

    Restricted to the shared texels on purpose.  Read over the whole atlas the difference
    is dominated by the texels only one camera reaches, where the other is simply black,
    and that is a difference in coverage, not one between two views of the same surface.
    The caption of the figure quotes these numbers, so they are printed rather than
    eyeballed off the panels: on the interior scene the largest visible feature of the
    pair, the saturated red block, turned out to be a coverage difference.
    """
    d = np.abs(a - b).max(axis=-1)[both]
    print(f"  |A - B| over the {both.sum()} shared texels: "
          f"p50 {np.percentile(d, 50):.4f}  p90 {np.percentile(d, 90):.4f}  "
          f"p99 {np.percentile(d, 99):.4f}  max {d.max():.4f}")
    print(f"  {100.0 * (d > 0.1).mean():.1f}% of them differ by more than 0.1 in radiance; "
          f"that spread is the signal the material fit reads")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="the run folder (holds images/, ium/, sources/, ...)")
    ap.add_argument("--cameras", nargs=2, metavar=("A", "B"),
                    default=["render_Camera_Shell10_8", "render_Camera_Shell21_35"],
                    help="stems of the two cameras to show.  The default pair is the one "
                         "with the largest shared coverage of the interior scene, 29.4%% "
                         "of the masked texels against 7.1%% for an arbitrary pair: the "
                         "figure says nothing about what changes between viewpoints on "
                         "the texels only one of the two reaches")
    ap.add_argument("--source", default="gt",
                    help="image source, i.e. sources/{source}/camera_texture/")
    ap.add_argument("--out", required=True, help="destination folder for the PNGs")
    args = ap.parse_args()

    run, out = Path(args.run_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{run.name} → {out.resolve()}")

    box, mask = crop_to_atlas(run)
    ys, xs = box

    seen = [coverage(run, stem, mask) for stem in args.cameras]
    both = seen[0] & seen[1]
    print(f"  the two together cover {100.0 * both[mask].mean():5.1f}% of the mesh texels; "
          f"that overlap is the part of the figure the fit can actually read")

    atlases = []
    for slot, stem in zip(SLOTS, args.cameras):
        photo = load_exr(find_photo(run, stem))
        report(f"view_{slot}", photo)
        save_png(photo, out / f"view_{slot}.png")

        atlas_path = run / "sources" / args.source / "camera_texture" / f"{stem}.exr"
        if not atlas_path.exists():
            raise SystemExit(f"x per-camera texture not found: {atlas_path}")
        atlas = load_exr(atlas_path)
        atlases.append(atlas)
        report(f"atlas_{slot}", atlas)
        save_png(atlas[ys, xs], out / f"atlas_{slot}.png")

    disagreement(atlases[0], atlases[1], both)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
