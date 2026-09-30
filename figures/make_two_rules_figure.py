#!/usr/bin/env python
"""make_two_rules_figure.py -- the two rules the reconstruction rests on, in one picture.

    python make_two_rules_figure.py --run <run dir> --out ../../PresentationImages

Writes `two_rules.png` for the Overview slide, which states two rules aloud before it gets
to the coverage map and the video:

  1. one point of the texture is one point of the model, and the other way round;
  2. every texel is solved on its own, and solving one must not move another.

Two panels, because the two rules are about the same object -- a texel -- and are meant to
be read one after the other.

LEFT, the correspondence.  The atlas beside a render, THREE texels marked in three colours
and joined by three lines to the points they stand for on the surface.  Three and not one:
with one it looks like an example, with three it looks like a rule.  The points are not
placed by hand, they are the world positions the inverse UV mapping stores for those texels
(`ium/ium_positions.exr`), projected with the inverse of `nerf/rays.py:get_rays_np`.

RIGHT, the independence.  The same three colours, three lanes side by side, each with the
cameras that see ITS texel and its own empty (a, m, r) box -- empty because at this point of
the talk nothing has been recovered yet.  Nothing crosses between the lanes, which is the
whole point: the visual grammar of "independent" is "not connected", and drawing arrows
between the texels to say they do not affect each other would say the opposite.  The three
sets of cameras really are different -- 22, 15 and 16 of the 60, sharing between 5 and 11 --
and that difference is also the ramp into the next sentence of the talk and into the video.

The rule is true of this data, which was worth checking rather than assuming.  No texel
serves two surface points: the 7260 triangles sum to 0.559896 of the unit square in UV while
the IUM mask covers 0.559895, a ratio of 1.000002, so no two islands overlap.  And no point
sits on two texels: not one pair of masked texels holds the same float32 position.  The seam
duplication a cube atlas shows (`uv_cube_atlas.png`, where two vertices appear three times)
lives on the cut curve, which has measure zero and which texel centres never land on.

One convention.  A texel of a 4096 atlas drawn at panel size is about a tenth of a pixel, so
the marks on the atlas are POINTERS, not the texel drawn to scale; a zoom box would say what
is there, which is the job of `pbr_texel_zoom.png` and of the video, not of this figure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.patches import Circle, FancyArrow, Rectangle      # noqa: E402
from matplotlib.patches import ConnectionPatch                    # noqa: E402

import _paths  # noqa: F401,E402

from make_skybox_figure import block_mean, load_exr, tonemap      # noqa: E402
from make_pbr_texel_zoom_figure import project                    # noqa: E402
from make_texel_views_video import measure_exposure, row_pixels   # noqa: E402
from make_geometry_diagrams import DPI, C_CAM, C_EDGE             # noqa: E402

# (label, IUM texel, colour).  One per object, so the three lines end in three clearly
# different places.  The first two are the texels of the video, so the slide and the video
# are about the same points.
TEXELS = [("metal", (648, 1335), "#00d8ff"),
          ("wood",  (3090, 3752), "#ffd400"),
          ("bunny", (1290, 2114), "#ff5ba8")]

KEY = 0.18            # the exposure of the GIF and the video of this scene
ATLAS_DS = 8
RENDER_DS = 2
DEPTH_TOL = 0.005     # metres of agreement between the depth map and the distance
GRID_COLS = 10        # the 60 cameras of a lane, as a 10 x 6 grid

C_INK = C_CAM
C_MUTED = C_EDGE
C_EMPTY = "#e9ecef"


def visibility(run: Path, meta: dict, texel) -> list[int]:
    """The cameras that see this texel, from the per-camera masks, which are the
    authoritative artefact (they carry occlusion AND frustum AND grazing)."""
    y, x = texel
    out = []
    for i, f in enumerate(meta["frames"]):
        stem = Path(f["file_path"]).name
        if row_pixels(run / "camera_mask" / stem, y, [x], ["Z"])[0, 0] > 0.5:
            out.append(i)
    return out


def pick_camera(run: Path, meta: dict, intr: dict, world, sets, forced):
    """A camera that sees all three texels, unoccluded, and spreads them in the frame.

    Spread and not just visibility: three marks bunched into one corner would make the
    three lines cross and say nothing about three different places.
    """
    common = set(sets[0]) & set(sets[1]) & set(sets[2])
    if forced is not None:
        if forced not in common:
            raise SystemExit(f"camera {forced} does not see all three texels "
                             f"(the ones that do: {sorted(common)})")
        common = {forced}
    if not common:
        raise SystemExit("no camera sees all three texels: the correspondence panel "
                         "cannot draw them on one render")
    best = None
    for i in sorted(common):
        f = meta["frames"][i]
        c2w = np.array(f["transform_matrix"], dtype=np.float64)
        proj = [project(p, c2w, intr) for p in world]
        if any(q is None for q in proj):
            continue
        dep = load_exr(run / f["depth_path"])[..., 0]
        d = [abs(float(dep[int(round(v)), int(round(u))]) - dist)
             for u, v, dist in proj]
        clear = all(x < DEPTH_TOL for x in d)
        spread = float(np.std([q[0] for q in proj]) + np.std([q[1] for q in proj]))
        print(f"    camera {i:>2}: spread {spread:6.1f} px, depth agrees to "
              f"{1e3 * max(d):.2f} mm{'' if clear else '   OCCLUDED'}")
        if clear and (best is None or spread > best[0]):
            best = (spread, i, proj, max(d))
    if best is None:
        raise SystemExit("every camera that frames all three has one of them occluded")
    return best


def save(fig, out: Path, name: str, fig_w: float, fig_h: float) -> Path:
    path = out / f"{name}.png"
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"  + {path}  ({int(fig_w * DPI)}x{int(fig_h * DPI)})")
    return path


def srgb(x: np.ndarray) -> np.ndarray:
    """sRGB encoding, the convention make_atlas_pngs uses for an albedo."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def draw(run: Path, meta: dict, intr: dict, cam, proj, sets, out: Path, expo: float,
         bare: bool, atlas_path: Path, panel: str, name: str, side: int) -> Path:
    n_cams = len(meta["frames"])
    # An albedo and not the colour texture: the aggregated colour still carries the
    # studio's magenta and green, and reads as a stain rather than as a texture.  An
    # albedo is in [0, 1] by construction, so it takes the sRGB curve and no exposure.
    #
    # The atlas shown may be a different resolution from the IUM (the authored bake is
    # 8192 against the run's 4096), so the axes are put in IUM texels through `extent`
    # and the marks land on the same coordinates whatever is underneath.  The two share
    # an orientation, measured: their masks agree on 98.4% of the atlas as they are, and
    # on 62% with v flipped.
    raw = load_exr(atlas_path)
    atlas = srgb(block_mean(raw, max(1, raw.shape[0] // 512)))
    print(f"    atlas {atlas_path.name} {raw.shape[1]}x{raw.shape[0]} "
          f"-> shown at {atlas.shape[0]} over {side} IUM texels")
    render = np.clip(tonemap(block_mean(
        load_exr(run / meta["frames"][cam]["file_path"]), RENDER_DS), expo), 0, 1)

    # -- layout, in inches ---------------------------------------------------
    m_side, m_bot = 0.42, 0.30
    cap_h = 0.0 if bare else 0.46      # the caption over each panel
    img_h = 3.10
    gap_ai = 0.42                      # atlas to render
    render_w = img_h * intr["w"] / intr["h"]
    left_w = img_h + gap_ai + render_w
    lane_w, lane_gap = 1.62, 0.30
    right_w = 3 * lane_w + 2 * lane_gap
    gap_panels = 0.95
    body = {"both": left_w + gap_panels + right_w, "left": left_w,
            "right": right_w}[panel]
    fig_w = 2 * m_side + body
    fig_h = m_bot + img_h + cap_h + (0.10 if bare else 0.22)
    fig = plt.figure(figsize=(fig_w, fig_h))
    y0 = m_bot / fig_h
    hh = img_h / fig_h

    def caption(xc, text):
        if not bare:
            fig.text(xc / fig_w, (m_bot + img_h + 0.16) / fig_h, text, ha="center",
                     va="bottom", fontsize=15, color=C_INK)

    # -- left panel: the correspondence --------------------------------------
    if panel != "right":
        ax_a = fig.add_axes([m_side / fig_w, y0, img_h / fig_w, hh])
        ax_a.imshow(atlas, extent=(0, side, side, 0), interpolation="nearest")
        ax_r = fig.add_axes([(m_side + img_h + gap_ai) / fig_w, y0,
                             render_w / fig_w, hh])
        ax_r.imshow(render, extent=(0, intr["w"], intr["h"], 0),
                    interpolation="bilinear")
        for ax in (ax_a, ax_r):
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(C_MUTED); s.set_linewidth(1.0)
        caption(m_side + left_w / 2, "one texel  ↔  one point of the surface")

        for ((_, (ty, tx), col), (u, v, _)) in zip(TEXELS, proj):
            # a square on the atlas, a circle on the surface: the two are different
            # kinds of thing, and the shape says so before the line does
            ax_a.add_patch(Rectangle((tx - 62, ty - 62), 124, 124, fill=False, ec=col,
                                     lw=2.6, zorder=6))
            ax_r.add_patch(Circle((u, v), 26, fill=False, ec=col, lw=2.6, zorder=6))
            cp = ConnectionPatch(xyA=(tx + 62, ty), coordsA=ax_a.transData,
                                 xyB=(u - 26, v), coordsB=ax_r.transData,
                                 color=col, lw=1.8, alpha=0.9, zorder=5)
            cp.set_annotation_clip(False)
            fig.add_artist(cp)

    # -- right panel: the independence ---------------------------------------
    if panel == "left":
        return save(fig, out, name, fig_w, fig_h)
    x_right = m_side + (left_w + gap_panels if panel == "both" else 0.0)
    caption(x_right + right_w / 2, "each texel is solved on its own")
    rows = int(np.ceil(n_cams / GRID_COLS))
    for j, ((_, _, col), seen) in enumerate(zip(TEXELS, sets)):
        xb = x_right + j * (lane_w + lane_gap)
        ax = fig.add_axes([xb / fig_w, y0, lane_w / fig_w, hh])
        ax.set_xlim(0, lane_w)
        ax.set_ylim(0, img_h)
        ax.axis("off")

        ax.add_patch(Rectangle((lane_w / 2 - 0.13, img_h - 0.30), 0.26, 0.26,
                               facecolor=col, edgecolor=C_INK, lw=1.2))

        cell = min((lane_w - 0.10) / GRID_COLS, 0.145)
        gx0 = (lane_w - GRID_COLS * cell) / 2
        gy1 = img_h - 0.62
        for c in range(n_cams):
            r, k = divmod(c, GRID_COLS)
            ax.add_patch(Rectangle((gx0 + k * cell, gy1 - (r + 1) * cell),
                                   cell * 0.84, cell * 0.84,
                                   facecolor=col if c in seen else C_EMPTY,
                                   edgecolor="none"))
        gy0 = gy1 - rows * cell
        if not bare:
            ax.text(lane_w / 2, gy0 - 0.20, f"{len(seen)} of {n_cams} cameras",
                    ha="center", va="top", fontsize=11, color=C_MUTED)

        ax.add_patch(FancyArrow(lane_w / 2, gy0 - 0.52, 0, -0.34, width=0.012,
                                head_width=0.10, head_length=0.13,
                                length_includes_head=True, color=C_MUTED))
        bw, bh = lane_w - 0.34, 0.46
        ax.add_patch(Rectangle(((lane_w - bw) / 2, gy0 - 1.44), bw, bh, fill=False,
                               ec=col, lw=2.0))
        ax.text(lane_w / 2, gy0 - 1.44 + bh / 2, "a,  m,  r", ha="center", va="center",
                fontsize=13, color=C_INK)

    return save(fig, out, name, fig_w, fig_h)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="the run folder")
    ap.add_argument("--out", required=True, help="destination folder")
    ap.add_argument("--camera", type=int, default=None,
                    help="force a frame index instead of choosing one")
    ap.add_argument("--key", type=float, default=KEY)
    ap.add_argument("--atlas", default="albedo/albedo.exr",
                    help="the map shown as the atlas, relative to sources/gt/")
    ap.add_argument("--bake", default=None,
                    help="folder with the authored BakedMaterial_*.exr; with it the "
                         "atlas is the ORIGINAL base colour instead of the recovered "
                         "albedo")
    ap.add_argument("--panel", choices=("both", "left", "right"), default="both",
                    help="which rule to draw: left is the correspondence, right the "
                         "independence")
    ap.add_argument("--bare", action="store_true", help="drop the captions, for a slide")
    ap.add_argument("--name", default="two_rules", help="output stem, without .png")
    a = ap.parse_args(argv)

    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((run / "transforms_extended.json").read_text())
    intr = dict(fx=meta["fl_x"], fy=meta["fl_y"], cx=meta["cx"], cy=meta["cy"],
                w=meta["w"], h=meta["h"])
    print(f"{run.name} -> {out.resolve()}")

    pos = load_exr(run / "ium" / "ium_positions.exr")
    msk = load_exr(run / "ium" / "ium_masks.exr")[..., 0] > 0.5
    side = pos.shape[0]
    world, sets = [], []
    for lab, (y, x), _ in TEXELS:
        if not msk[y, x]:
            raise SystemExit(f"the {lab} texel ({y}, {x}) is empty atlas in the IUM mask")
        world.append(pos[y, x].astype(np.float64))
        sets.append(visibility(run, meta, (y, x)))
        print(f"  {lab:<6} texel ({y}, {x})  p = {np.round(world[-1], 4)}  "
              f"{len(sets[-1])} of {len(meta['frames'])} cameras")
    del pos, msk

    for i in range(3):
        for j in range(i + 1, 3):
            shared = len(set(sets[i]) & set(sets[j]))
            if set(sets[i]) == set(sets[j]):
                raise SystemExit(
                    f"{TEXELS[i][0]} and {TEXELS[j][0]} are seen by exactly the same "
                    "cameras: the independence panel would be illustrating three "
                    "separate problems with two identical ones")
            print(f"  {TEXELS[i][0]} & {TEXELS[j][0]}: {shared} cameras in common")

    print("  choosing the camera")
    spread, cam, proj, worst = pick_camera(run, meta, intr, world, sets, a.camera)
    print(f"  chose camera {cam}: spread {spread:.1f} px, "
          f"every depth agrees to {1e3 * worst:.2f} mm")

    expo = measure_exposure(run, meta, a.key, 4)
    atlas_path = (Path(a.bake) / "BakedMaterial_base_color.exr" if a.bake
                  else run / "sources" / "gt" / a.atlas)
    if not atlas_path.exists():
        raise SystemExit(f"{atlas_path} does not exist")
    draw(run, meta, intr, cam, proj, sets, out, expo, a.bare, atlas_path, a.panel,
         a.name, side)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
