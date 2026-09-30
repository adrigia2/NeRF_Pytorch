#!/usr/bin/env python
"""make_texel_views_video.py -- one texel, sixty cameras, two very different answers.

    python make_texel_views_video.py --run <run dir> --out ../../PresentationImages

Writes an MP4.  Each frame is one camera: on the left the render it took, on the right the
`camera_texture` it produced, which is the atlas filled with what THAT camera reads and
left at zero everywhere it sees nothing.  Two texels are marked in both panels, and along
the bottom two strips grow, one per texel, gaining a square per camera: the colour that
camera recorded, or a square with an X where that camera does not see the texel.

The video makes two claims and both are visible without a word.  A texel on a specular
surface is a different colour from every camera; a texel on a diffuse one is the same
colour from all of them.  And neither is seen by most of the cameras.

THE TWO TEXELS are chosen from the run's own statistics, not by eye:

  metal, the steel of the cube    IUM (648, 1335)   22 of 60 views   luminance 0.012 - 14.475
  wood, the table top             IUM (3090, 3752)  15 of 60 views   luminance 0.410 -  0.457

A factor of 1181 against a factor of 1.11, on two points of the same scene, from the same
sixty cameras.  That is the whole video.  They share eleven cameras out of sixty, so a
square with an X is the normal case and not a curiosity.

Both texels are chosen against a measured neighbourhood, not just a good number.  The metal
one sits **32 texels from the nearest non-metal**, on clean brushed steel: the obvious
candidate was the texel of `pbr_texel_zoom_metal.png`, which has an even better view of the
argument at 142x, but it lies 5 texels from the rust, and its zoom box is more rust than
steel -- fine for a figure whose subject IS the rust boundary, wrong for a video that says
"this is the metal".  The wood one is on the table's TOP face, checked on the IUM normal
(+Z): the first candidate the statistics offered sat on the vertical rim, where the magenta
key light washes the recorded colour to nearly pure red, and a strip of red squares is a
poor way to say "this is wood and it looks the same from everywhere".  Both are also at
least forty texels from any seam of their UV island.

The metal texel is ranked on the SPREAD OF THE SQUARES AS DRAWN, not on the luminance
ratio.  The two disagree: the ratio is a number about the radiances, and after the exposure
and the Reinhard a strip can have a huge ratio and still read as a row of dark browns, which
is what the first clean-metal candidate did.  The chosen one is the candidate whose squares
are farthest apart in the very colours the strip paints, so the criterion is measured on the
thing the viewer actually judges.

ONE EXPOSURE for the whole video, and the SAME one for both panels.  The render and the
camera texture hold the same radiances, so sharing the exposure makes the colour of the
marked texel on the right the very pixel seen on the left, and the square added to the
strip that colour through the same curve.  A per-frame exposure would make the loop
flicker and, worse, would rescale exactly the variation the video exists to show.  The
tonemap is the shared `make_skybox_figure.tonemap`, and the level is set the way
`make_input_views_gif` sets it, from the median luminance of the subject.

THE ORDER IS BY AZIMUTH, not by index.  The capture shells are spirals: consecutive indices
jump about 105 degrees, so index order strobes while azimuth order walks around the scene.

Nothing here is asserted without being measured.  `report()` prints the view counts and the
luminance ratios that the narration quotes, and refuses to run when a texel is not surface
in the IUM mask, or when `camera_mask` and a non-zero `camera_texture` disagree about
whether a camera sees a texel -- two files that have to be saying the same thing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.patches import Circle, Rectangle                  # noqa: E402

import _paths  # noqa: F401,E402

from make_skybox_figure import LUMA_COEFF, block_mean, load_exr, tonemap   # noqa: E402
from make_pbr_texel_zoom_figure import project                             # noqa: E402
from make_input_views_gif import load_mask                                 # noqa: E402
from make_geometry_diagrams import C_CAM, C_EDGE                           # noqa: E402

# (label, IUM texel, marker colour, label colour).  See the module docstring for where the
# two texels come from.  The marker has to stay visible on a
# magenta-and-green studio, the label on white paper, and no single colour does both:
# yellow reads on the scene and vanishes on the page.
TEXELS = [("metal, the cube", (648, 1335), "#00d8ff", "#0086a8"),
          ("wood, the table", (3090, 3752), "#ffd400", "#a07e00")]

KEY = 0.18            # the subject's median luminance is brought here before the Reinhard
FPS = 8               # the sweep is meant to be fast; the strips are what one reads
HOLD = 2.5            # seconds the last frame is held, with both strips complete
WIDTH, HEIGHT = 1600, 900
DPI = 100
EXPO_STRIDE = 4       # frames sampled for the exposure; the median is stable well before 60
RENDER_DS = 2         # the render is shown at 1/this
ATLAS_DS = 8          # the atlas is 4096 and the panel is ~500 px
INSET = 64            # side of the atlas zoom, in texels: any wider and the metal
                      # texel's box would reach the rust 34 texels away
INSET_FRAC = 0.30     # side of the zoom box, as a fraction of the atlas panel

C_INK = C_CAM
C_MUTED = C_EDGE
C_EMPTY = "#f2f2f2"   # a strip square for a camera that does not see the texel
C_X = "#9aa3ab"


def row_pixels(path: Path, y: int, xs, names) -> np.ndarray:
    """(len(xs), len(names)) read from one scanline.

    The per-camera masks are 4096 x 4096 and only two of their texels are wanted; reading
    the whole file sixty times would cost minutes for 32 values.
    """
    import OpenEXR, Imath

    f = OpenEXR.InputFile(Path(path).as_posix())
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    have = set(f.header()["channels"].keys())
    names = [n for n in names if n in have] or [next(iter(have))]
    bufs = f.channels(names, pt, y, y)
    rows = [np.frombuffer(b, dtype=np.float32) for b in bufs]
    return np.array([[r[x] for r in rows] for x in xs])


def frame_order(meta: dict) -> list[int]:
    """Frame indices sorted by azimuth.  See the module docstring."""
    p = np.array([np.array(f["transform_matrix"])[:3, 3] for f in meta["frames"]])
    az = np.degrees(np.arctan2(p[:, 1], p[:, 0])) % 360.0
    return list(np.argsort(az))


def measure_exposure(run: Path, meta: dict, key: float, stride: int) -> float:
    """One exposure, from the median luminance of the subject over sampled frames."""
    lum = []
    for f in meta["frames"][::stride]:
        img = load_exr(run / f["file_path"])
        lum.append((img @ LUMA_COEFF)[load_mask(run / f["mask_path"])])
    med = float(np.median(np.concatenate(lum)))
    print(f"  subject median luminance {med:.4f} over {len(lum)} sampled frames "
          f"-> exposure {key / med:.3f}")
    return key / med


def gather(run: Path, meta: dict, order, expo: float, intr: dict, points):
    """One pass over the run: everything each frame of the video needs.

    Read once and reduced immediately.  Holding sixty 4096 x 4096 atlases would be 12 GB;
    what is kept is their 8-bit thumbnails and, per texel, two numbers.
    """
    out = []
    for k, i in enumerate(order):
        f = meta["frames"][i]
        stem = Path(f["file_path"]).name
        c2w = np.array(f["transform_matrix"], dtype=np.float64)

        img = load_exr(run / f["file_path"])
        render = to_u8(tonemap(block_mean(img, RENDER_DS), expo))
        del img

        atlas_lin = load_exr(run / "sources" / "gt" / "camera_texture" / stem)
        values = [np.array(atlas_lin[y, x]) for (y, x) in points]
        h = INSET // 2
        crops = [to_u8(tonemap(atlas_lin[y - h:y + h, x - h:x + h], expo))
                 for (y, x) in points]
        atlas = to_u8(tonemap(block_mean(atlas_lin, ATLAS_DS), expo))
        del atlas_lin

        seen = []
        for (y, x) in points:
            seen.append(bool(row_pixels(run / "camera_mask" / stem, y, [x], ["Z"])[0, 0]
                             > 0.5))

        proj = []
        for (y, x), _ in zip(points, values):
            pr = project(world[(y, x)], c2w, intr)
            proj.append(None if pr is None or not (0 <= pr[0] < intr["w"]
                                                   and 0 <= pr[1] < intr["h"])
                        else (pr[0], pr[1]))

        out.append(dict(idx=int(i), render=render, atlas=atlas, crops=crops,
                        values=values, seen=seen, proj=proj))
        if k % 10 == 0 or k == len(order) - 1:
            print(f"    {k + 1:>3}/{len(order)}  camera {i:>2}  "
                  f"seen {['no', 'yes'][seen[0]]:>3}/{['no', 'yes'][seen[1]]:>3}")
    return out


def to_u8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def report(frames, points, labels) -> None:
    """The numbers the narration quotes, measured here so they cannot be typed wrong."""
    sets = []
    for j, lab in enumerate(labels):
        vis = [fr for fr in frames if fr["seen"][j]]
        lum = np.array([float(fr["values"][j] @ LUMA_COEFF) for fr in vis])
        if len(lum) == 0:
            raise SystemExit(f"no camera sees the {lab} texel {points[j]}: "
                             "there is nothing to show")
        bad = [fr["idx"] for fr in frames
               if not fr["seen"][j] and float(fr["values"][j].max()) > 0.0]
        if bad:
            raise SystemExit(
                f"camera_mask says the {lab} texel is unseen in cameras {bad} but "
                "camera_texture holds a colour there: the two files disagree about the "
                "same fact and the video would paint a square from a value the mask "
                "calls absent")
        print(f"  {lab:<16} texel {points[j]}: seen by {len(lum)} of {len(frames)}, "
              f"luminance {lum.min():.4f} - {lum.max():.4f}, "
              f"ratio {lum.max() / max(lum.min(), 1e-9):.2f}x")
        sets.append({fr["idx"] for fr in vis})
    print(f"  the two texels share {len(sets[0] & sets[1])} camera(s) of {len(frames)}")


def draw_frame(fig, fr, upto, frames, points, labels, colours, ink, intr, atlas_side):
    """One video frame, drawn from scratch: 60 figures is a few seconds."""
    fig.clf()
    fig.patch.set_facecolor("white")
    w_in, h_in = WIDTH / DPI, HEIGHT / DPI

    # -- layout, in inches ---------------------------------------------------
    # The panels are limited by the WIDTH, not the height: a 16:9 render beside a square
    # atlas is 2.78 panel-heights wide, so the height follows from the room across.
    m_side, m_top, m_bot = 0.40, 0.34, 0.55
    gap_panels = 0.42
    aspect = intr["w"] / intr["h"]
    panel_h = (w_in - 2 * m_side - gap_panels) / (1.0 + aspect)
    render_w = panel_h * aspect
    block_w = render_w + gap_panels + panel_h
    x0 = (w_in - block_w) / 2.0
    y_panels = h_in - m_top - panel_h

    ax_r = fig.add_axes([x0 / w_in, y_panels / h_in, render_w / w_in, panel_h / h_in])
    ax_r.imshow(fr["render"], extent=(0, intr["w"], intr["h"], 0), interpolation="bilinear")
    ax_r.set_xlim(0, intr["w"])
    ax_r.set_ylim(intr["h"], 0)
    for ax in (ax_r,):
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(C_MUTED); s.set_linewidth(1.0)

    ax_a = fig.add_axes([(x0 + render_w + gap_panels) / w_in, y_panels / h_in,
                         panel_h / w_in, panel_h / h_in])
    ax_a.imshow(fr["atlas"], extent=(0, atlas_side, atlas_side, 0), interpolation="nearest")
    ax_a.set_xticks([]); ax_a.set_yticks([])
    for s in ax_a.spines.values():
        s.set_edgecolor(C_MUTED); s.set_linewidth(1.0)

    # -- the two markers -----------------------------------------------------
    # On the render a circle, solid when the camera sees the texel and dashed when it does
    # not: the point is still there, the camera just cannot reach it.  On the atlas a zoom
    # box instead, because a circle 105 texels wide on a 4096 atlas says "somewhere around
    # here" and the question the box answers is exactly "around here where".
    for j, ((ty, tx), col) in enumerate(zip(points, colours)):
        style = dict(fill=False, lw=2.6, ec=col, zorder=6)
        soft = dict(fill=False, lw=1.3, ec=col, alpha=0.55, ls=(0, (3, 3)), zorder=6)
        if fr["proj"][j] is not None:
            u, v = fr["proj"][j]
            ax_r.add_patch(Circle((u, v), 38, **(style if fr["seen"][j] else soft)))

        # The zoom goes in the corner on the texel's OWN side horizontally and the opposite
        # one vertically: the connector then runs short and near an edge instead of across
        # the atlas, and the two of them cannot cross each other.  Note the axes fraction
        # counts from the bottom while the atlas is drawn with y downwards, so a texel high
        # in the atlas wants a box low in the axes.
        half = atlas_side / 2
        fx = 0.035 if tx < half else 1.0 - 0.035 - INSET_FRAC
        fy = 0.035 if ty < half else 1.0 - 0.035 - INSET_FRAC
        ax_i = ax_a.inset_axes([fx, fy, INSET_FRAC, INSET_FRAC])
        ax_i.imshow(fr["crops"][j], interpolation="nearest",
                    extent=(0, INSET, INSET, 0))
        ax_i.set_xticks([]); ax_i.set_yticks([])
        for sp in ax_i.spines.values():
            sp.set_edgecolor(col); sp.set_linewidth(2.2)
        c = INSET / 2
        for ec, lw in ((C_INK, 3.2), (col, 1.6)):     # a dark halo, then the key colour
            ax_i.add_patch(Rectangle((c - 4.5, c - 4.5), 9, 9, fill=False, ec=ec, lw=lw,
                                     zorder=7))
        ax_a.indicate_inset((tx - INSET / 2, ty - INSET / 2, INSET, INSET),
                            inset_ax=ax_i, edgecolor=col, linewidth=1.6, alpha=1.0)

    # -- the two strips ------------------------------------------------------
    lab_w, cnt_w = 1.70, 1.25
    strip_h, strip_gap = 0.70, 0.34
    strip_w = w_in - 2 * m_side - lab_w - cnt_w
    n = len(frames)
    cell = strip_w / n
    y_block = m_bot
    for j, (lab, col, txt) in enumerate(zip(labels, colours, ink)):
        yb = y_block + (len(labels) - 1 - j) * (strip_h + strip_gap)
        ax_s = fig.add_axes([m_side / w_in, yb / h_in,
                             (lab_w + strip_w + cnt_w) / w_in, strip_h / h_in])
        ax_s.set_xlim(0, lab_w + strip_w + cnt_w)
        ax_s.set_ylim(0, strip_h)
        ax_s.axis("off")
        ax_s.text(lab_w - 0.14, strip_h / 2, lab, ha="right", va="center",
                  fontsize=13.5, color=txt)
        seen_n = 0
        for k in range(upto + 1):
            g = frames[k]
            x = lab_w + k * cell
            now = k == upto
            if g["seen"][j]:
                seen_n += 1
                rgb = np.clip(tonemap(g["values"][j], EXPO[0]), 0.0, 1.0)
                ax_s.add_patch(Rectangle((x, 0), cell * 0.86, strip_h,
                                         facecolor=tuple(rgb),
                                         edgecolor=C_INK if now else C_MUTED,
                                         lw=2.0 if now else 0.4, zorder=3 if now else 2))
            else:
                ax_s.add_patch(Rectangle((x, 0), cell * 0.86, strip_h,
                                         facecolor=C_EMPTY,
                                         edgecolor=C_INK if now else C_X,
                                         lw=2.0 if now else 0.5, zorder=3 if now else 2))
                p = cell * 0.86 * 0.26
                q = strip_h * 0.26
                ax_s.plot([x + p, x + cell * 0.86 - p], [q, strip_h - q],
                          color=C_X, lw=0.9)
                ax_s.plot([x + p, x + cell * 0.86 - p], [strip_h - q, q],
                          color=C_X, lw=0.9)
        ax_s.text(lab_w + strip_w + 0.14, strip_h / 2,
                  f"seen {seen_n} / {upto + 1}", ha="left", va="center",
                  fontsize=12.5, color=C_MUTED)

    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3]


EXPO = [1.0]      # the one exposure, shared by the panels and the strips
world = {}        # texel -> world position, filled in main()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="the run folder")
    ap.add_argument("--out", required=True, help="destination folder")
    ap.add_argument("--name", default="texel_views", help="output stem, without .mp4")
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--hold", type=float, default=HOLD,
                    help="seconds the last frame is held")
    ap.add_argument("--slow", type=int, default=1,
                    help="hold each camera for this many container frames: 2 is half "
                         "speed, 3 a third, and the container fps stays sane")
    ap.add_argument("--key", type=float, default=KEY)
    ap.add_argument("--texel-a", type=int, nargs=2, default=list(TEXELS[0][1]),
                    metavar=("ROW", "COL"))
    ap.add_argument("--texel-b", type=int, nargs=2, default=list(TEXELS[1][1]),
                    metavar=("ROW", "COL"))
    a = ap.parse_args(argv)

    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((run / "transforms_extended.json").read_text())
    intr = dict(fx=meta["fl_x"], fy=meta["fl_y"], cx=meta["cx"], cy=meta["cy"],
                w=meta["w"], h=meta["h"])
    points = [tuple(a.texel_a), tuple(a.texel_b)]
    labels = [TEXELS[0][0], TEXELS[1][0]]
    colours = [TEXELS[0][2], TEXELS[1][2]]
    ink = [TEXELS[0][3], TEXELS[1][3]]
    print(f"{run.name} -> {(out / (a.name + '.mp4')).resolve()}")

    pos = load_exr(run / "ium" / "ium_positions.exr")
    msk = load_exr(run / "ium" / "ium_masks.exr")[..., 0] > 0.5
    atlas_side = pos.shape[0]
    for (y, x), lab in zip(points, labels):
        if not msk[y, x]:
            raise SystemExit(f"the {lab} texel ({y}, {x}) is empty atlas in the IUM mask, "
                             "not surface: it has no world position to project")
        world[(y, x)] = pos[y, x].astype(np.float64)
        print(f"  {lab:<16} texel ({y}, {x})  p = {np.round(world[(y, x)], 4)}")
    del pos, msk

    EXPO[0] = measure_exposure(run, meta, a.key, EXPO_STRIDE)
    order = frame_order(meta)
    print(f"  {len(order)} cameras, ordered by azimuth")
    frames = gather(run, meta, order, EXPO[0], intr, points)
    report(frames, points, labels)

    import cv2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    path = out / f"{a.name}.mp4"
    vw = cv2.VideoWriter(path.as_posix(), fourcc, a.fps, (WIDTH, HEIGHT))
    if not vw.isOpened():
        raise SystemExit(f"OpenCV could not open a writer for {path}")
    fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI)
    last = None
    for k, fr in enumerate(frames):
        rgb = draw_frame(fig, fr, k, frames, points, labels, colours, ink,
                         intr, atlas_side)
        last = rgb
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # Slowed down by repeating each camera, not by lowering the container fps: a 2.7
        # fps MP4 is legal and some players still stutter on it.
        for _ in range(max(1, a.slow)):
            vw.write(bgr)
    held = int(round(a.hold * a.fps))
    for _ in range(held):
        vw.write(cv2.cvtColor(last, cv2.COLOR_RGB2BGR))
    vw.release()
    plt.close(fig)

    total = len(frames) * max(1, a.slow) + held
    kb = path.stat().st_size / 1024
    print(f"  + {path}  ({WIDTH}x{HEIGHT}, {len(frames)} cameras x{a.slow} + {held} held "
          f"= {total} frames at {a.fps} fps = {total / a.fps:.1f} s, "
          f"{1000 * a.slow / a.fps:.0f} ms per camera, {kb:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
