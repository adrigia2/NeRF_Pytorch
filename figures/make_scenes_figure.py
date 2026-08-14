#!/usr/bin/env python
"""make_scenes_figure.py -- thesis figures on the scenes: the interior, and the sword and shield.

For each variant it produces two PNGs:

  <key>_view.png    the rendered view, tonemapped
  <key>_detail.png  a full-resolution crop on the element that sets the variant apart
                    variant (the metallic cube, the emissive sphere)

    python make_scenes_figure.py --out ../Doc/images/scenes
    python make_scenes_figure.py --family sword --out ../Doc/images/scenes
    python make_scenes_figure.py --contact-sheet specular --out <temporary folder>

Three choices are not negotiable, and are the reason this script exists:

  1. The camera is the same for every variant of a family.  On the interior it is taken
     from the `render_Camera_Shell21_*` set: the `render_config.json` files differ, the
     high-frequency variant has `center_offset` 0 instead of 0.5 on shell 1, so only the
     30 frames of shell 2 have identical extrinsics across all the scenes.  With a shell 1
     camera the views would not be comparable.

  2. Variants that share the same lighting share ONE exposure, derived from the median
     luminance of a reference variant.  If each had its own, the cube that looks matte in
     the diffuse variant might look so because of the exposure rather than the material,
     which is exactly what the figure is there to show.  The night variant has its own,
     because at the daytime exposure it would be black.
     nera.

  3. The crop is taken from the full-resolution linear image and tonemapped with the same
     exposure as the view it comes from.  Cropping after the downsample would throw away
     precisely the detail the panel is meant to show.

There are two families and they share nothing but these three rules: the interior has four
variants and two exposure groups, the sword has two which are also the two groups.
`column_exposure()` is where rule 2 is written down once:
make_results_figures.py calls it to tonemap the "NeRF render" row of the grids with the
exposure of the original row of the same column, which is what the caption promises.
didascalia promette al lettore.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _paths  # noqa: F401

from make_skybox_figure import LUMA_COEFF, block_mean, load_exr, tonemap

SCENES_ROOT = Path("C:/Users/adria/Documents/GitHub/Tesi/OptixProjectCMake/Scenes"
                   "/TableAndOtherInterior")
SWORD_ROOT = Path("C:/Users/adria/Documents/GitHub/Tesi/OptixProjectCMake/Scenes"
                  "/SwordShield Thesis")

# Shared camera: only shell 2 has identical extrinsics across every variant.
# Number 38 is the only one where the three objects (sphere, bunny, cube) do not overlap
# and the environment is also visible, which is what explains the tint of the light.
CAMERA = "render_Camera_Shell21_38"

# The sword has a single shell of 60 cameras, so the extrinsics constraint that governs
# the interior does not apply: the choice is only one of framing.  Coverage alone is not
# enough as a criterion, and that is why this constant has a comment: the cameras that
# maximise the covered area (number 40 leading) frame the BACK of the shield, where only
# the two grips are visible.  Number 23 is among the best for coverage (12.9 % of the
# pixels) and centring, and on top of that it is one of the few that shows the three
# things the text talks about together: the wooden boards of the face, the steel boss and
# the whole blade, from pommel to tip.
SWORD_CAMERA = "render_Camera_Shell10_23"

# (key, dataset folder, label, exposure group)
# The "day" group shares a single exposure; "night" has its own.
SCENES: list[tuple[str, str, str, str]] = [
    ("specular",    "NerfOpenEXRSmooth",          "specular variant (base)", "day"),
    ("highfreq",    "NerfOpenEXRHighDetails",     "high-frequency variant",  "day"),
    ("night",       "NerfOpenEXRSmoothNight",     "night variant",           "night"),
    ("diffusecube", "NerfOpenExrSmoothNoDiffuse", "diffuse-cube variant",    "day"),
]

SWORD_SCENES: list[tuple[str, str, str, str]] = [
    ("sword_studio", "NerfStudio", "sword and shield, studio", "studio"),
    ("sword_night",  "NerfNight",  "sword and shield, night",  "night"),
]

# Crop (x0, y0, width, height) in full-resolution pixels, per variant.
# The three variants that differ by the cube use the same rectangle, so comparing the
# crops is direct; the high-frequency one frames the sphere.
# Same 16:9 as the view: in the figure the two panels sit side by side at the same width,
# and with different proportions they would have different heights.
CROPS: dict[str, tuple[int, int, int, int]] = {
    "specular":    (860, 265, 640, 360),
    "highfreq":    (460, 340, 640, 360),
    "night":       (860, 265, 640, 360),
    "diffusecube": (860, 265, 640, 360),
}

# The crop of the stone sphere: the same rectangle as "highfreq", because the sphere
# sits in the same place in every variant, and it is the detail panel of the section on
# lost detail (fig:res-highfreq-stone).
SPHERE_CROP = CROPS["highfreq"]

# The exposure brings the median luminance to this level before the Reinhard.
# 0.2 and not 0.5: the median of this framing falls on the studio's dark floor, and
# bringing it to mid scale burned out the table, which is almost all that matters.
KEY = 0.20

# Previews of the two envmaps for the asset table: (key, path relative to
# SCENES_ROOT).  The night one lives in the bake folder, not in assets/hdri.
SKYBOXES: list[tuple[str, str]] = [
    ("skybox_studio", "Blender/assets/hdri/wooden_studio_13_4k.exr"),
    ("skybox_night",  "BlenderBakedSmoothNight/cobblestone_street_night_4k.exr"),
]


@dataclass(frozen=True)
class Family:
    """A family of scenes: same geometry, same camera, same exposure rules.
    `reference` says, for each exposure group, which variant dictates its median: it is
    rule 2 of the docstring made explicit instead of being written inside the flow of
    main()."""
    root: Path
    camera: str
    scenes: list[tuple[str, str, str, str]]
    reference: dict[str, str]
    crops: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)


FAMILIES: dict[str, Family] = {
    "interior": Family(
        root=SCENES_ROOT, camera=CAMERA, scenes=SCENES,
        reference={"day": "specular", "night": "night"}, crops=CROPS),
    # Studio and night do not share an exposure: they are two different environments, not
    # two versions of the same one, and with a single exposure one of them would be unreadable.
    "sword": Family(
        root=SWORD_ROOT, camera=SWORD_CAMERA, scenes=SWORD_SCENES,
        reference={"studio": "sword_studio", "night": "sword_night"}),
}


def frame_path(scene_dir: str, camera: str, root: Path = SCENES_ROOT) -> Path:
    return root / scene_dir / "images" / f"{camera}.exr"


def exposure_of(img: np.ndarray, key: float = KEY) -> tuple[float, float]:
    """(exposure, median luminance).  The median, and not the maximum, because in an HDR
    scene the peak sits on the light sources and would dictate the tonemap alone."""
    lum = (img * LUMA_COEFF).sum(-1)
    med = max(float(np.median(lum)), 1e-4)
    return key / med, med


def column_exposure(family: str, scene_key: str, camera: str | None = None,
                    key: float = KEY) -> float:
    """The exposure a column of the Results grids has to be tonemapped with.

    This is where rule 2 lives: the variant's exposure group decides which frame dictates
    its median, and every panel of that column inherits it.
    It is needed outside this module because the preview grids stack the original render,
    the NeRF one and the re-render in one column, and the caption promises the three share
    a single exposure: if each recomputed it on its own median, a NeRF that gets the mean
    level wrong would be brought back into scale by the tonemap and the figure would show
    an error that is no longer there."""
    fam = FAMILIES[family]
    group = {k: g for k, _, _, g in fam.scenes}[scene_key]
    ref_key = fam.reference[group]
    ref_dir = {k: d for k, d, _, _ in fam.scenes}[ref_key]
    expo, _ = exposure_of(load_exr(frame_path(ref_dir, camera or fam.camera, fam.root)),
                          key)
    return expo


def save_png(rgb: np.ndarray, out: Path) -> None:
    plt.imsave(out, np.clip(rgb, 0.0, 1.0))
    print(f"  + {out}  ({rgb.shape[1]}x{rgb.shape[0]})")


def skybox_previews(out: Path, key: float, downsample: int = 4) -> None:
    """The two envmaps, tonemapped, for the asset table.

    Each with its own exposure: they are two different environments, not two versions of
    the same scene, and here the figure only has to make them readable.  The downsample is
    a block mean in linear space, before the tonemap, so as not to alter the mean radiance
    of the small sources (same reason as in make_skybox_figure).
    """
    for name, rel in SKYBOXES:
        p = SCENES_ROOT / rel
        if not p.exists():
            raise SystemExit(f"ERROR: {p} does not exist")
        a = block_mean(load_exr(p), downsample)
        expo, med = exposure_of(a, key)
        print(f"{name}: {p.name}  exposure {expo:.3f} "
              f"(median luminance {med:.4f})")
        save_png(tonemap(a, expo), out / f"{name}.png")


def contact_sheet(scene_dir: str, out: Path, downsample: int = 8,
                  ncols: int = 6, root: Path = SCENES_ROOT) -> None:
    """Grid of every frame of the scene, to pick the camera by eye."""
    paths = sorted((root / scene_dir / "images").glob("*.exr"))
    if not paths:
        raise SystemExit(f"ERROR: no EXR in {root / scene_dir / 'images'}")
    print(f"contact sheet of {scene_dir}: {len(paths)} frames")

    thumbs = [block_mean(load_exr(p), downsample) for p in paths]
    expo, med = exposure_of(np.concatenate([t.reshape(-1, 3) for t in thumbs])[None])
    print(f"  exposure = {expo:.4f} (median luminance {med:.4f})")

    nrows = (len(thumbs) + ncols - 1) // ncols
    h, w = thumbs[0].shape[:2]
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.0 * ncols, 3.0 * (h / w) * nrows))
    flat = np.atleast_1d(axes).ravel()
    for ax, path, t in zip(flat, paths, thumbs):
        ax.imshow(tonemap(t, expo))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(path.stem.replace("render_Camera_", ""), fontsize=7)
    for ax in flat[len(thumbs):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  + {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="destination folder for the PNGs")
    ap.add_argument("--family", default="interior", choices=sorted(FAMILIES),
                    help="scene family to generate (default interior)")
    ap.add_argument("--camera", default=None,
                    help="frame to use (default: the family's own)")
    ap.add_argument("--downsample", type=int, default=2,
                    help="block mean on the view (default 2: 1920x1080 -> 960x540)")
    ap.add_argument("--key", type=float, default=KEY,
                    help=f"level the median luminance is brought to (default {KEY})")
    ap.add_argument("--contact-sheet", default=None, metavar="KEY",
                    help="write only the frame grid of the given variant")
    ap.add_argument("--skyboxes", action="store_true",
                    help="write only the previews of the two envmaps for the asset table")
    ap.add_argument("--no-crop", action="store_true",
                    help="skip the crops (useful while choosing the rectangles)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fam = FAMILIES[args.family]
    camera = args.camera or fam.camera
    by_key = {k: (d, lab, grp) for k, d, lab, grp in fam.scenes}

    if args.skyboxes:
        skybox_previews(out, args.key)
        return 0

    if args.contact_sheet:
        if args.contact_sheet not in by_key:
            print(f"ERROR: {args.contact_sheet} is not among {list(by_key)}")
            return 2
        contact_sheet(by_key[args.contact_sheet][0],
                      out / f"contact_{args.contact_sheet}.png", root=fam.root)
        return 0

    # Load every linear view first: a group's exposure is shared and has to be computed
    # on the reference variant before anything is tonemapped.
    linear: dict[str, np.ndarray] = {}
    for key, scene_dir, _, _ in fam.scenes:
        p = frame_path(scene_dir, camera, fam.root)
        if not p.exists():
            print(f"ERROR: {p} does not exist")
            return 2
        linear[key] = load_exr(p)
        print(f"{key:14s} {p.name}  {linear[key].shape[1]}x{linear[key].shape[0]}")

    # One exposure per group, dictated by the group's reference variant.
    expo: dict[str, float] = {}
    print()
    for group, ref_key in fam.reference.items():
        expo[group], med = exposure_of(linear[ref_key], args.key)
        print(f"exposure {group:8s} = {expo[group]:9.4f}  "
              f"(median luminance of {ref_key}: {med:.5f})")

    print()
    for key, _, _, group in fam.scenes:
        lin = linear[key]
        save_png(tonemap(block_mean(lin, args.downsample), expo[group]),
                 out / f"{key}_view.png")
        if args.no_crop or key not in fam.crops:
            continue
        x0, y0, w, h = fam.crops[key]
        save_png(tonemap(lin[y0:y0 + h, x0:x0 + w], expo[group]),
                 out / f"{key}_detail.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
