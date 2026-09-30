#!/usr/bin/env python
"""make_presentation_thesis_format.py -- the presentation maps in the thesis's conventions.

    python make_presentation_thesis_format.py <run_dir> --bake <blender bake dir>
                                              --out DIR [--atlas-size 4096]

The thesis does not use one tone mapping: it uses a different rendition for each KIND of
map, each chosen so that a specific reading of the figure stays true.  This script puts
every map produced for the presentation through the rendition its thesis counterpart
already has, importing the conventions from the scripts that made those figures instead of
restating them, so the two cannot drift apart.

  map                    thesis figure                         rendition
  ---------------------  ------------------------------------  ---------------------------
  irradiance             Doc/images/irradiance/irradiance      content crop, normalised on
  irradiance_indirect      ..._indirect                        the p99.5 of the luminance
  irradiance_full          ..._total                           over the masked texels, then
                                                               gamma 1/2.2, black outside
                                                               the mask; EACH map its own
                                                               factor
  albedo_lambertian      Doc/images/albedo/albedo_lambert      WHOLE atlas, sRGB, no
                                                               exposure
  interiorSpecular*      Doc/images/table-and-other-specular   block mean to atlas_size,
                                                               sRGB on the base colour,
                                                               linear clip on the others
  color_texture          (none)                                content crop and a plain
                                                               clamp: the convention of
                                                               fig. 3.7, the only other
                                                               place a colour-texture
                                                               atlas is shown

Every file name carries the operator that produced it, since a folder holding six
different renditions is unreadable otherwise: `_p995_gamma22`, `_srgb`, `_clamp01` for an
HDR map clipped to [0, 1], `_linear01` for data that was already inside it and went through
no curve at all.  The distinction between the last two is not cosmetic: `_clamp01` loses
information and the fraction it cuts is printed, `_linear01` loses none.

Three points that are not free choices:

  1. `irradiance_full` is summed at FULL resolution, before any block averaging, exactly as
     make_visibility_figure.do_irradiance does: it is the quantity the albedo divides by,
     and averaging the two terms first would agree with it only by accident.

  2. The three irradiance panels have three DIFFERENT normalisation factors, so
     total = direct + indirect is not readable off the images.  The factors and the ratio
     of the means are printed, because that is the information the caption has to carry.

  3. The block mean always happens BEFORE the transfer curve: averaging already-gammated
     values gives a different colour.

`--atlas-size` defaults to 4096 rather than the thesis's 1024.  The reduction to 1024 is
documented in make_visibility_figure and make_results_figures as a limit of the PDF, which
had reached 369 MB, not as a property of the figure; a slide has no such limit.  Everything
else is identical.  Note the normalisation percentile is measured after the block mean, so
these panels are not a pure upscale of the ones in Doc/images.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")

import _paths  # noqa: F401

from make_atlas_pngs import SRGB_CHANNELS, srgb
from make_depth_figure import save_png
from make_skybox_figure import block_mean, load_exr
from make_visibility_figure import PCTL, crop_to_atlas, tonemap as pctl_gamma

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

# (our name, file under the bake folder) -- the four authored Blender maps.
AUTHORED = [("base_color", "BakedMaterial_base_color.exr"),
            ("metallic",   "BakedMaterial_metallic.exr"),
            ("roughness",  "BakedMaterial_roughness.exr"),
            ("normal",     "BakedMaterial_normal.exr")]


def do_irradiance(run: Path, out: Path, box, mask, atlas_size: int) -> None:
    """The three irradiance panels, as make_visibility_figure.do_irradiance renders them."""
    ys, xs = box
    direct = load_exr(run / "irradiance" / "irradiance.exr")
    indirect = load_exr(run / "irradiance" / "irradiance_indirect.exr")
    maps = [("irradiance", direct),
            ("irradiance_indirect", indirect),
            ("irradiance_full", direct + indirect)]   # summed at full resolution

    for name, img in maps:
        sub, msk = img[ys, xs], mask[ys, xs]
        k_ds = max(1, sub.shape[0] // atlas_size)
        if k_ds > 1:
            sub = block_mean(sub, k_ds)
            msk = block_mean(msk.astype(np.float32), k_ds) > 0.5
        rgb, k = pctl_gamma(sub, msk)
        save_png(rgb, out / f"{name}_p995_gamma22.png")
        lum = img @ LUMA
        print(f"      normalised on p{PCTL} = {k:.4f}; mean {lum[mask].mean():.4f}, "
              f"max {lum[mask].max():.3f}" + (f"; downsampled /{k_ds}" if k_ds > 1 else ""))

    d = (direct @ LUMA)[mask].mean()
    i = (indirect @ LUMA)[mask].mean()
    print(f"      indirect / direct over the masked texels: {i / d:.4f} "
          f"({100.0 * i / (d + i):.1f}% of the total)")


def do_albedo(run: Path, out: Path, source: str, atlas_size: int) -> None:
    """The Lambertian albedo, as make_visibility_figure.do_albedo renders it: the WHOLE
    atlas and no exposure, so it can be read against the recovered maps texel by texel."""
    a = load_exr(run / "sources" / source / "albedo" / "albedo.exr")
    mask = load_exr(run / "ium" / "ium_masks.exr")[..., 0] > 0.5
    k = max(1, a.shape[0] // atlas_size)
    save_png(srgb(block_mean(a, k)), out / "albedo_lambertian_srgb.png")
    m = a[mask]
    print(f"      sRGB, no exposure, block mean /{k}; per-channel median over the mesh "
          + ", ".join(f"{v:.3f}" for v in np.median(m, axis=0)))
    print(f"      texels clipped at 1.0 in at least one channel: "
          f"{100.0 * (m.max(axis=1) >= 1.0).mean():.2f}%")


def do_color_texture(run: Path, out: Path, source: str, box) -> None:
    """The aggregated colour texture.  No thesis figure shows it; fig. 3.7 shows the
    per-camera ones, and its rendition is a content crop and a plain clamp, with neither
    normalisation nor gamma, because the atlas holds the same values as the photograph and
    any factor would stand between the two panels.  The data is HDR and unbounded above,
    so the bright parts saturate: the fraction is printed for the caption."""
    ys, xs = box
    a = load_exr(run / "sources" / source / "color_texture" / "color_texture.exr")
    save_png(a[ys, xs], out / "color_texture_clamp01.png")
    mx = a.reshape(-1, 3).max(axis=0)
    print(f"      plain clamp; max per channel ({mx[0]:.3f}, {mx[1]:.3f}, {mx[2]:.3f}), "
          f"clamped {100.0 * (a > 1.0).any(-1).mean():.2f}% of the atlas")


def do_authored(bake: Path, out: Path, prefix: str, atlas_size: int) -> None:
    """The authored Blender maps, as make_atlas_pngs renders them: sRGB on the base colour
    alone, which is colour, and linear on the rest, which is data.  The block mean has to
    precede the encoding."""
    for name, fname in AUTHORED:
        p = bake / fname
        if not p.exists():
            print(f"    ! {p.name} does not exist, skipped")
            continue
        a = load_exr(p)
        k = max(1, a.shape[0] // atlas_size)
        a = block_mean(a, k)
        as_srgb = name in SRGB_CHANNELS
        save_png(srgb(a) if as_srgb else np.clip(a, 0.0, 1.0),
                 out / f"{prefix}_{name}_{'srgb' if as_srgb else 'linear01'}.png")
        print(f"      {'sRGB' if as_srgb else 'linear'}, block mean /{k}, "
              f"range [{a.min():.3f}, {a.max():.3f}]")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="the run folder (holds ium/, irradiance/, sources/)")
    ap.add_argument("--bake", default=None,
                    help="folder with the authored BakedMaterial_*.exr")
    ap.add_argument("--prefix", default="interiorSpecularOriginal",
                    help="name prefix for the authored maps")
    ap.add_argument("--source", default="gt", help="sources/{source}/ to read")
    ap.add_argument("--atlas-size", type=int, default=4096,
                    help="texels a panel is reduced to (the thesis uses 1024)")
    ap.add_argument("--out", required=True, help="destination folder")
    a = ap.parse_args(argv)

    run, out = Path(a.run_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{run.name} -> {out.resolve()}   (atlas size {a.atlas_size})")

    box, mask = crop_to_atlas(run)
    ys, xs = box
    print(f"  content crop [{xs.start}:{xs.stop}, {ys.start}:{ys.stop}] "
          f"= {xs.stop - xs.start}x{ys.stop - ys.start} of {mask.shape[1]}x{mask.shape[0]}")

    print("  irradiance (p99.5 + gamma 1/2.2, cropped)")
    do_irradiance(run, out, box, mask, a.atlas_size)
    print("  albedo (sRGB, no exposure, whole atlas)")
    do_albedo(run, out, a.source, a.atlas_size)
    print("  colour texture (plain clamp, cropped)")
    do_color_texture(run, out, a.source, box)
    if a.bake:
        print("  authored Blender maps")
        do_authored(Path(a.bake), out, a.prefix, a.atlas_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
