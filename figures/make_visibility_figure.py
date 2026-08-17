#!/usr/bin/env python
"""make_visibility_figure.py -- texture-space maps for `fig:visibility-maps` and the
irradiance figures of Section `sec:irradiance`.

    python make_visibility_figure.py <run_dir> --camera render_Camera_Shell21_38 \
        --out ../Doc/images/visibility
    python make_visibility_figure.py <run_dir> --count-only --out ../Doc/images/results/sword
    python make_visibility_figure.py <run_dir> --irradiance --out ../Doc/images/irradiance

`--visibility` mode (the default), `fig:visibility-maps`:

  camera_mask.png    the mask of ONE camera in texel space: the texels that camera
                     really sees, i.e. occlusion AND frustum AND grazing
  camera_count.png   how many cameras cover each texel, summing the 60 channels of
                     visibility.exr, on a colour scale with a bar

`--count-only` writes just the second one and never looks at `--camera`.  The Results
chapter repeats the coverage map per scene (`fig:res-interior-coverage`,
`fig:res-sword-coverage`) without the single-camera panel, and the camera stems are not
shared between scenes anyway: the interior rig is `render_Camera_Shell21_*` and the sword
one `render_Camera_Shell10_*`, so the default `--camera` does not exist outside the
interior run.

`--irradiance` mode:

  irradiance.png           the direct component from the environment map
  irradiance_indirect.png  the indirect component queried from the NeRF
  irradiance_total.png     their sum, the quantity the albedo is divided by

The three are HDR and have very different ranges (the indirect one is typically an order
of magnitude lower), so each is normalised **on its own percentile** and the factor is
printed: on a shared scale the indirect one would be a black rectangle and say nothing.
The normalization is on a percentile, not on the maximum, because the HDR tail of a scene
with a concentrated source would crush everything else.

The price of three separate factors has to be stated wherever these panels are used: the
relation total = direct + indirect is NOT readable off the images, which are three
qualitative maps and not a sum anyone can check by eye.  What carries that information is
the ratio of the means, printed here, and it belongs in the caption.

`--albedo` mode:

  albedo_lambert.png       the Lambertian albedo, pi * colour / total irradiance

Encoded in sRGB with NO exposure normalization, unlike everything above: the albedo is a
reflectance the pipeline has already clipped to [0, 1], and rescaling it would contradict
the one claim the figure exists to support, that on the diffuse parts the division returns
something close to the surface colour.  The whole atlas is kept and only block-averaged,
the same treatment `make_results_figures.do_maps` gives the recovered maps, so this figure
and the albedo row of the Results grids can be read against each other texel by texel.

MIND THE SOURCE: after the azimuth fix in `deviceProgramsIrradiance.cu` (2026-08-13) the
`irradiance.exr` files of `test_sword_shield` are stale.  The thesis figure must use the
regenerated tree `test_sword_shield_after_fix_irradiance`.  The visibility maps and the
per-camera masks do not depend on the irradiance and are bit-identical between the two
trees, so either works for the visibility figures; use the regenerated tree anyway, since
that is the one every other figure of the Results chapter comes from.

In that regenerated tree `irradiance.exr` is post-fix while `irradiance_indirect.exr` was
deliberately kept from the earlier bake (`irradiance_fix.json`).  The two are still
comparable: the drift the fix removed comes from `i * goldenAngle` losing precision in
float32, and the indirect pass runs at 32x32 = 1024 directions, where the worst-case error
is about 0.01 degrees, against the 3.6 degrees the direct pass reached at 512x512.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _paths  # noqa: F401

from make_atlas_pngs import srgb                          # noqa: E402
from make_depth_figure import content_box, save_png       # noqa: E402
from make_skybox_figure import block_mean, load_exr       # noqa: E402

plt.rcParams.update({"font.size": 13})

DPI = 190
PCTL = 99.5          # percentile for the normalization of the HDR maps
MARGIN = 0.02        # crop margin around the used area of the atlas
ATLAS_SIZE = 1024    # texels a texture-space panel is reduced to before writing


def read_channels(path: Path) -> dict:
    """Every channel of an EXR as float32.  `load_exr` assumes RGB and visibility.exr has
    one channel per camera, so here the header is read and everything is taken."""
    import OpenEXR
    import Imath
    fh = OpenEXR.InputFile(str(path))
    head = fh.header()
    dw = head["dataWindow"]
    w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    ft = Imath.PixelType(Imath.PixelType.FLOAT)
    return {c: np.frombuffer(fh.channel(c, ft), np.float32).reshape(h, w)
            for c in head["channels"]}


def crop_to_atlas(run: Path):
    """The used box of the atlas, from the IUM mask.  The same `content_box`
    make_depth_figure uses, so the crops of the figures are comparable."""
    mask = load_exr(run / "ium" / "ium_masks.exr")[..., 0] > 0.5
    return content_box(mask, MARGIN), mask


def heat_png(data: np.ndarray, mask: np.ndarray, out: Path, label: str,
             vmax: float | None = None, cmap: str = "magma") -> None:
    """False-colour map with a bar, neutral grey outside the mask."""
    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    shown = np.where(mask, data, np.nan)
    im = ax.imshow(shown, cmap=cmap, vmin=0.0, vmax=vmax, interpolation="nearest")
    ax.set_facecolor("#f0f2f4")
    ax.axis("off")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(label)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def do_visibility(run: Path, camera: str, out: Path, count_only: bool = False) -> None:
    box, mask = crop_to_atlas(run)
    ys, xs = box

    if not count_only:
        cam_path = run / "camera_mask" / f"{camera}.exr"
        if not cam_path.exists():
            raise SystemExit(f"✗ per-camera mask not found: {cam_path}")
        cam = load_exr(cam_path)[..., 0] > 0.5
        save_png(np.repeat(cam[ys, xs, None].astype(np.float32), 3, axis=-1),
                 out / "camera_mask.png")
        print(f"  camera_mask.png       {camera}: {100 * cam[mask].mean():.1f}% of the mesh "
              f"texels seen by this camera")

    vis = read_channels(run / "visibility" / "visibility.exr")
    cams = sorted(vis, key=lambda c: int(c.replace("Cam", "")))
    count = np.zeros_like(vis[cams[0]])
    for c in cams:
        count += (vis[c] > 0.5)
    heat_png(count[ys, xs], mask[ys, xs], out / "camera_count.png",
             f"cameras covering the texel (out of {len(cams)})",
             vmax=float(np.percentile(count[mask], 99.9)), cmap="viridis")
    v = count[mask]
    print(f"  camera_count.png      {len(cams)} cameras; per texel: "
          f"mean {v.mean():.1f}, median {np.median(v):.0f}, "
          f"p10 {np.percentile(v, 10):.0f}, p90 {np.percentile(v, 90):.0f}, "
          f"never seen {100 * (v == 0).mean():.2f}%")


def tonemap(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalise on a percentile of the luminance and apply a 1/2.2 gamma.
    Also returns the factor, which has to be stated in the caption."""
    lum = img @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    k = float(np.percentile(lum[mask], PCTL))
    k = k if k > 1e-8 else 1.0
    o = np.clip(img / k, 0.0, 1.0) ** (1.0 / 2.2)
    o[~mask] = 0.0
    return o.astype(np.float32), k


def do_irradiance(run: Path, out: Path, atlas_size: int = ATLAS_SIZE) -> None:
    box, mask = crop_to_atlas(run)
    ys, xs = box
    lum_w = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

    maps = {}
    for name, fname in (("irradiance", "irradiance.exr"),
                        ("irradiance_indirect", "irradiance_indirect.exr")):
        p = run / "irradiance" / fname
        if not p.exists():
            print(f"  ⚠  {p} missing, skipped")
            continue
        maps[name] = load_exr(p)

    # The sum is computed at FULL resolution, before any block averaging: it is the
    # quantity Equation eq:albedo divides by, and averaging the two terms first and adding
    # afterwards would only agree by accident.
    if len(maps) == 2:
        maps["irradiance_total"] = maps["irradiance"] + maps["irradiance_indirect"]

    for name in ("irradiance", "irradiance_indirect", "irradiance_total"):
        if name not in maps:
            continue
        img = maps[name]
        # Block mean BEFORE the tonemap, and on the crop only.  At 4096 texels three
        # panels of a single figure carry more pixels than the rest of the chapter put
        # together; the reason is the one documented in make_results_figures.do_maps.
        sub, msk = img[ys, xs], mask[ys, xs]
        k_ds = max(1, sub.shape[0] // atlas_size)
        if k_ds > 1:
            sub = block_mean(sub, k_ds)
            msk = block_mean(msk.astype(np.float32), k_ds) > 0.5
        rgb, k = tonemap(sub, msk)
        save_png(rgb, out / f"{name}.png")
        lum = img @ lum_w
        print(f"  {name + '.png':24s} normalised on p{PCTL} = {k:.4f}; "
              f"mean {lum[mask].mean():.4f}, max {lum[mask].max():.3f}"
              + (f"; downsampled /{k_ds}" if k_ds > 1 else ""))

    # The number the caption has to carry, because the three separate exposures make the
    # comparison impossible to do by eye.
    if "irradiance_total" in maps:
        d = (maps["irradiance"] @ lum_w)[mask].mean()
        i = (maps["irradiance_indirect"] @ lum_w)[mask].mean()
        print(f"  indirect / direct, mean over the masked texels: {i / d:.4f} "
              f"({100.0 * i / (d + i):.1f}% of the total)")


def do_albedo(run: Path, source: str, out: Path, atlas_size: int = ATLAS_SIZE) -> None:
    """The Lambertian albedo of `sources/<source>/albedo/albedo.exr`.

    The WHOLE atlas, not the content crop the other texture-space panels use, and no
    exposure: this is the same treatment the recovered maps get in the Results grids, so
    the Lambertian albedo and the one that comes out of the fit sit on the same footing
    and can be compared texel by texel.  It is also the only encoding under which the
    figure's claim survives, since the values ARE reflectances in [0, 1] and rescaling
    them would mean the colours shown are not the ones the equation produced.
    """
    p = run / "sources" / source / "albedo" / "albedo.exr"
    if not p.exists():
        raise SystemExit(f"✗ albedo not found: {p}")
    a = load_exr(p)
    mask = load_exr(run / "ium" / "ium_masks.exr")[..., 0] > 0.5

    k = max(1, a.shape[0] // atlas_size)
    # The block mean has to happen BEFORE the sRGB encoding: averaging already-gammated
    # values gives a different colour.  Same order as make_results_figures.do_maps.
    small = block_mean(a, k)
    save_png(srgb(small), out / "albedo_lambert.png")

    m = a[mask]
    print(f"  albedo_lambert.png     {small.shape[1]}x{small.shape[0]} "
          f"(from {a.shape[1]}, block mean /{k}), sRGB, no exposure")
    print(f"  per-channel median over the mesh: "
          + ", ".join(f"{v:.3f}" for v in np.median(m, axis=0)))
    # The number behind "any highlight survives in the albedo as if it were pigment": a
    # reflectance pinned at 1 is a texel where the division did not return a material.
    print(f"  texels clipped at 1.0 in at least one channel: "
          f"{100.0 * (m.max(axis=1) >= 1.0).mean():.2f}%")
    print(f"  texels at 0.0 in every channel: "
          f"{100.0 * (m.max(axis=1) <= 0.0).mean():.2f}%")


def do_pixel_change(run: Path, source: str, out: Path) -> None:
    """The four statistics of the colour-texture pass, on ONE shared exposure.

    Shared because that is the point of the figure: `range` is `max - min`, and with four
    different normalizations that relation disappears.  The factor comes from a percentile
    of `color_max`, the brightest of the four.

    The variance is shown as its SQUARE ROOT: it is in radiance squared, and on the shared
    scale it would be a black rectangle.  The root brings it back into radiance units and
    makes it comparable with the range.
    """
    box, mask = crop_to_atlas(run)
    ys, xs = box
    pc = run / "sources" / source / "pixel_change"
    lum_w = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

    cmax = load_exr(pc / "color_max.exr")
    k = float(np.percentile((cmax @ lum_w)[mask], PCTL))
    k = k if k > 1e-8 else 1.0
    print(f"  shared exposure: p{PCTL} of color_max = {k:.4f}")

    for name in ("color_min", "color_max", "color_range", "color_variance"):
        p = pc / f"{name}.exr"
        if not p.exists():
            print(f"  ⚠  {p} missing, skipped")
            continue
        img = load_exr(p)
        if name == "color_variance":
            img = np.sqrt(np.maximum(img, 0.0))     # -> radiance units
        rgb = np.clip(img / k, 0.0, 1.0) ** (1.0 / 2.2)
        rgb[~mask] = 0.0
        save_png(rgb[ys, xs].astype(np.float32), out / f"{name}.png")
        L = (img @ lum_w)[mask]
        print(f"  {name + '.png':22s} p50={np.median(L):8.4f}  p99={np.percentile(L, 99):9.4f}"
              f"  saturated {100.0 * (L > k).mean():5.2f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="the run folder (holds ium/, visibility/, ...)")
    ap.add_argument("--camera", default="render_Camera_Shell21_38",
                    help="stem of the reference camera for the mask")
    ap.add_argument("--count-only", action="store_true",
                    help="write only camera_count.png, ignoring --camera")
    ap.add_argument("--irradiance", action="store_true",
                    help="produce the irradiance maps instead of the visibility ones")
    ap.add_argument("--albedo", action="store_true",
                    help="produce the Lambertian albedo map")
    ap.add_argument("--pixel-change", action="store_true",
                    help="produce the four colour-texture statistics")
    ap.add_argument("--source", default="gt",
                    help="source for --albedo and --pixel-change (sources/{source}/)")
    ap.add_argument("--out", required=True, help="destination folder for the PNGs")
    args = ap.parse_args()

    run, out = Path(args.run_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{run.name} → {out.resolve()}")
    if args.pixel_change:
        do_pixel_change(run, args.source, out)
    elif args.albedo:
        do_albedo(run, args.source, out)
    elif args.irradiance:
        do_irradiance(run, out)
    else:
        do_visibility(run, args.camera, out, args.count_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
