#!/usr/bin/env python
"""make_presentation_magnitude.py -- panels that show HOW MUCH, not what it looks like.

    python make_presentation_magnitude.py <run_dir> --out DIR [--source gt]
                                          [--atlas-size 4096]

A tonemapped panel answers "what does this map look like"; by construction it cannot answer
"how big is it", because the exposure that made it readable is the very thing that removed
the scale.  The three renditions here give up the first question to answer the second, and
none of them applies a tone mapping.

  1. `*_p995shared_gamma22`  the direct, indirect and total irradiance through ONE
     normalisation factor, the p99.5 of the DIRECT.  Same colour means the same radiance in
     all three, so the indirect looks dark because it is dark.  With one factor each -- what
     Doc/images/irradiance and thesis_format/ do, for a different and equally good reason --
     the indirect is shown about five times brighter than it is and no reader can tell.

  2. `irradiance_indirect_share_viridis01`  the per-texel ratio E_ind / (E_dir + E_ind), in
     false colour with a bar, fixed to [0, 1].  This is the panel that answers the question
     texel by texel: it is a dimensionless ratio, so there is no exposure and no curve to
     argue about, and it shows WHERE the bounce matters instead of only how much it weighs
     on average.  The quantity is the one make_roi_preview already calls `share`, computed
     the same way, on the channel mean.

  3. `color_texture_clamp01_marked`  the colour texture clipped to [0, 1], with the texels
     the clip destroyed painted in a flat marker colour.  The clip keeps 97.9% of the atlas
     untouched, which is why it beats a tone mapping here, but the part it does destroy
     becomes invisible -- a saturated texel is indistinguishable from a legitimately white
     one.  Marking it makes the loss explicit instead of silent.

The marker colour is CHOSEN FROM THE DATA, not fixed: this scene has bright pink walls and a
green statue, so a hardcoded magenta or green would read as content.  The candidate that
maximises the distance from the colours actually present is picked and printed.

The block mean, where it applies, precedes everything else: averaging already-gammated
values gives a different colour, and the mean of a ratio is not the ratio of the means, so
the two irradiances are reduced FIRST and the share is computed from the reduced maps, the
order make_roi_preview.build uses.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _paths  # noqa: F401

from make_depth_figure import save_png
from make_skybox_figure import block_mean, load_exr
from make_visibility_figure import PCTL, crop_to_atlas, tonemap as pctl_gamma

EPS = 1e-8
CMAP = "viridis"
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

# A slide is projected at 1920 or more; make_visibility_figure.heat_png is sized for a page
# (6.0x5.6 at 190 dpi, about 1140 px) and would be soft full screen.  Same styling, more
# pixels.
FIGSIZE = (9.0, 8.0)
DPI = 240

# Flat, fully saturated candidates; the one furthest from the content wins.
MARKERS = {"magenta": (1.0, 0.0, 1.0), "green": (0.0, 1.0, 0.0),
           "red": (1.0, 0.0, 0.0), "blue": (0.0, 0.0, 1.0),
           "cyan": (0.0, 1.0, 1.0), "yellow": (1.0, 1.0, 0.0)}


def apply_k(img: np.ndarray, mask: np.ndarray, k: float) -> np.ndarray:
    """The tail of make_visibility_figure.tonemap with the factor GIVEN instead of measured:
    that function normalises every map on its own percentile, which is exactly what has to
    be avoided here."""
    o = np.clip(img / k, 0.0, 1.0) ** (1.0 / 2.2)
    o[~mask] = 0.0
    return o.astype(np.float32)


def reduce_to(img: np.ndarray, mask: np.ndarray, atlas_size: int):
    """Block mean to at most `atlas_size` texels, mask included.  Returns the factor too."""
    k_ds = max(1, img.shape[0] // atlas_size)
    if k_ds == 1:
        return img, mask, 1
    return (block_mean(img, k_ds),
            block_mean(mask.astype(np.float32), k_ds) > 0.5, k_ds)


def do_shared(run: Path, out: Path, box, mask, atlas_size: int) -> None:
    ys, xs = box
    direct = load_exr(run / "irradiance" / "irradiance.exr")
    indirect = load_exr(run / "irradiance" / "irradiance_indirect.exr")
    maps = [("irradiance", direct),
            ("irradiance_indirect", indirect),
            ("irradiance_full", direct + indirect)]   # summed at full resolution

    # The factor comes from the DIRECT and is then imposed on the other two.  Taken through
    # the thesis function so the percentile and the gamma cannot drift from the panels of
    # Doc/images/irradiance.
    sub_d, msk_d, _ = reduce_to(direct[ys, xs], mask[ys, xs], atlas_size)
    _, k = pctl_gamma(sub_d, msk_d)
    print(f"      shared factor = p{PCTL} of the direct = {k:.4f}")

    for name, img in maps:
        sub, msk, k_ds = reduce_to(img[ys, xs], mask[ys, xs], atlas_size)
        save_png(apply_k(sub, msk, k), out / f"{name}_p995shared_gamma22.png")
        lum = (sub @ LUMA)[msk]
        print(f"      mean radiance {lum.mean():.4f} = {lum.mean() / k:.4f} of the shared "
              f"scale; {100.0 * (lum > k).mean():.2f}% of the texels above it"
              + (f"; downsampled /{k_ds}" if k_ds > 1 else ""))


def share_png(data: np.ndarray, mask: np.ndarray, out: Path, label: str) -> None:
    """False colour with a bar, neutral grey outside the mask.  Mirrors the styling of
    make_visibility_figure.heat_png, at slide resolution rather than page resolution."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    im = ax.imshow(np.where(mask, data, np.nan), cmap=CMAP, vmin=0.0, vmax=1.0,
                   interpolation="nearest")
    ax.set_facecolor("#f0f2f4")
    ax.axis("off")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(label)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  + {out}")


def do_share(run: Path, out: Path, box, mask, atlas_size: int) -> None:
    ys, xs = box
    msk = mask[ys, xs]
    # Reduce the two energies FIRST and take the ratio of the reduced maps: the mean of a
    # ratio is not the ratio of the means.
    d, msk_r, _ = reduce_to(load_exr(run / "irradiance" / "irradiance.exr")[ys, xs],
                            msk, atlas_size)
    i, _, _ = reduce_to(load_exr(run / "irradiance" / "irradiance_indirect.exr")[ys, xs],
                        msk, atlas_size)
    tot = d.mean(-1) + i.mean(-1)
    ok = msk_r & (tot > EPS)
    share = np.where(ok, i.mean(-1) / np.maximum(tot, EPS), np.nan)
    share_png(share, ok, out / "irradiance_indirect_share_viridis01.png",
              r"$E_{\mathrm{ind}} / (E_{\mathrm{dir}} + E_{\mathrm{ind}})$")
    v = share[ok]
    print(f"      indirect share per texel: p10 {np.percentile(v, 10):.3f}  "
          f"p50 {np.median(v):.3f}  p90 {np.percentile(v, 90):.3f}  max {v.max():.3f}")
    print(f"      texels where the bounce is more than a quarter of the light: "
          f"{100.0 * (v > 0.25).mean():.1f}%")


def pick_marker(rgb: np.ndarray, over: np.ndarray) -> tuple[str, tuple]:
    """The candidate furthest from the colours actually in the image.

    Measured on the texels the marker does NOT cover, since those are the ones it must not
    be confused with, and on a subsample with a fixed seed: the answer is a minimum over
    millions of pixels and does not move with the sample size.
    """
    content = rgb[~over]
    if content.size == 0:
        return "magenta", MARKERS["magenta"]
    rng = np.random.default_rng(0)
    n = min(400_000, content.shape[0])
    sample = content[rng.choice(content.shape[0], size=n, replace=False)]
    best = None
    for name, c in MARKERS.items():
        dmin = float(np.linalg.norm(sample - np.array(c, np.float32), axis=1).min())
        print(f"      marker {name:8s} closest content colour at {dmin:.4f}")
        if best is None or dmin > best[0]:
            best = (dmin, name, c)
    print(f"      -> {best[1]} chosen, nothing in the atlas closer than {best[0]:.4f}")
    return best[1], best[2]


def do_marked(run: Path, out: Path, source: str, box) -> None:
    ys, xs = box
    a = load_exr(run / "sources" / source / "color_texture" / "color_texture.exr")[ys, xs]
    rgb = np.clip(a, 0.0, 1.0)
    over = (a > 1.0).any(-1)
    _, colour = pick_marker(rgb.reshape(-1, 3), over.reshape(-1))
    rgb[over] = colour
    save_png(rgb, out / "color_texture_clamp01_marked.png")
    print(f"      {100.0 * over.mean():.2f}% of the atlas marked as destroyed by the clip")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="the run folder (holds ium/, irradiance/, sources/)")
    ap.add_argument("--source", default="gt", help="sources/{source}/ to read")
    ap.add_argument("--atlas-size", type=int, default=4096,
                    help="texels a panel is reduced to")
    ap.add_argument("--out", required=True, help="destination folder")
    a = ap.parse_args(argv)

    run, out = Path(a.run_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{run.name} -> {out.resolve()}")

    box, mask = crop_to_atlas(run)
    ys, xs = box
    print(f"  content crop = {xs.stop - xs.start}x{ys.stop - ys.start}")

    print("  irradiance on ONE shared scale")
    do_shared(run, out, box, mask, a.atlas_size)
    print("  indirect share, false colour with a bar")
    do_share(run, out, box, mask, a.atlas_size)
    print("  colour texture, clipped texels marked")
    do_marked(run, out, a.source, box)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
