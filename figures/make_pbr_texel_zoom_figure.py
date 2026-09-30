#!/usr/bin/env python
"""make_pbr_texel_zoom_figure.py -- one texel, in the scene and in the three maps.

    python make_pbr_texel_zoom_figure.py --bake <bake dir> --run <run dir> \
                                         --out ../../PresentationImages --bare
    python make_pbr_texel_zoom_figure.py --bake <bake dir> --run <run dir> --out ... \
                                         --texel 866 1579 --name pbr_texel_zoom_metal --bare

Writes one PNG: a render of the scene and the three authored Blender maps of it (base
colour, metallic, roughness) side by side, each zoomed onto the SAME atlas texel, with the
value that texel holds printed underneath.

The figure exists to make one sentence of the presentation visible: the three maps are not
three pictures, they are three coordinates of one function of the surface point.  A slide
that shows the atlases alone leaves the viewer to take that on trust; here the same (u, v)
is followed down from the whole atlas to a 256-texel window to a 9x9 block in which single
texels are drawn as squares, and what comes out at the bottom is the triple
(albedo, metallic, roughness) the BRDF is evaluated with at that point.  The render column
closes the loop at the other end: that texel is not an address in an image, it is a patch
of a surface a camera can see, and the red outline in the render is the projection of the
same 256-texel window the map columns are showing.

The default texel sits on the rust of the metal cross, and it is chosen rather than found:
inside one 256-texel window the maps disagree about the material in a way a flat patch
could never show.  The rust is a dielectric (metallic 0) and rough, the bare steel a
millimetre away is a conductor (metallic 1) and smooth, while the base colour turns from
white to brown.  The three zoom panels therefore carry the same shape and three different
readings of it, which is the point being made.  `--texel 866 1579` marks the bare steel in
the same window instead, for the slide that wants a conductor under the marker.

THE RENDER COLUMN (optional, `--run`; without it the figure is the three maps alone).

The world position of the texel is not derived, it is read: the run's inverse UV mapping
(`ium/ium_positions.exr`) is the pass whose whole job is to answer "which surface point
does this texel cover".  The projection into a camera is the inverse of
`nerf/rays.py:get_rays_np`, written out from it rather than restated, so the two cannot
drift: with d_c = R^T (p - o) and s = -d_c.z, the pixel is
(cx + fx d_c.x / s, cy - fy d_c.y / s).

The camera is CHOSEN, by three tests in order: the point has to be in front of the camera
and inside the frame with room for the crop, it has to be UNOCCLUDED, and among what
survives the most frontal view wins.  The occlusion test is not a formality -- it compares
the distance to the point against the run's own depth map at that pixel, and on this scene
it throws out a camera (frame 23) that has the point squarely inside its frame with
something else 1.3 m in front of it.  Without it the figure would have zoomed onto a
different object with full confidence.

The run's atlas is 4096 while the bake's is 8192, so the world position is that of the 2x2
block of bake texels containing the marked one: the point is located to within 0.74 mm,
which at 5.2 m is 0.4 pixels of the render.  The render is tonemapped with the convention
the Results figures use, `make_skybox_figure.tonemap` (Reinhard, then gamma 2.2) on the
exposure `make_scenes_figure.exposure_of` derives from the median luminance of the frame.

`--bare` drops every small caption -- the line above the columns, the two lines under them,
and the camera line beneath the render -- and closes the margins that carried them, leaving
the four headings and the three values.  It is what the copies in PresentationImages are
made with: at the size a projector gives a figure this wide, a 11.5 pt caption is not small
text, it is noise, and everything it said belongs in what the speaker says instead.  The
default keeps the captions, for a figure that has to stand on its own on a page.

Display conventions for the maps, the ones make_atlas_pngs already uses for these files,
so a viewer comparing this figure with the atlas PNGs in PresentationImages sees the same
pixels:

  base colour   sRGB encoded (IEC 61966-2-1).  It is colour, and 0.22 linear would read
                as almost black on a slide.
  metallic      linear, clipped to [0, 1].  It is data, and a curve would misstate it.
  roughness     linear, clipped to [0, 1].  Same reason.

Neither map is normalised nor exposed: metallic and roughness live in [0, 1] by
construction, and the base colour of this bake peaks at 0.854, so nothing is being cut.
The swatch under each column is filled with the texel's own value through the same
encoding as its map, and the numbers printed beside it are the linear ones.

Nothing in the figure is typed by hand.  The three values come from the EXRs at the marked
coordinate, and the run makes the figure refuse to draw when any of the assumptions it
asserts is not true of the data: that the three maps share one resolution (otherwise "the
same texel" means nothing), that the marked texel carries surface and not empty atlas in
both the bake and the run's IUM, that metallic and roughness hold one value across R, G
and B (the Blender bake stores them as three channels, and printing a single number for
them is only honest if they agree), and that some camera sees the point without anything
in the way.

The last of those checks is the one with teeth, and it is not about a single point.  A
projection can be wrong in a way that still lands somewhere plausible -- a flipped axis
puts the marker on the same object, at a mirrored place -- so `pattern_agreement` projects
a grid of texels from across the whole window and correlates the hue the albedo map gives
each one against the hue the render shows at its pixel.  On the default texel that
correlation is 0.96, while every flip of the mapping (v, u, transposed) falls to 0.16 or
below: the figure is asserting a correspondence, and this measures it rather than trusting
it.  Where the window is uniform in hue there is nothing to correlate, and the check says
so and stands down instead of failing on noise.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.patches import Polygon, Rectangle                 # noqa: E402

import _paths  # noqa: F401,E402

from make_skybox_figure import block_mean, load_exr, tonemap      # noqa: E402
from make_scenes_figure import KEY as SCENE_KEY, exposure_of      # noqa: E402
from make_geometry_diagrams import DPI, C_CAM, C_EDGE, C_HILITE   # noqa: E402

# (label, file, is it colour?) -- the three maps the Disney BRDF slide shows.
MAPS = [("Albedo",    "BakedMaterial_base_color.exr", True),
        ("Metallic",  "BakedMaterial_metallic.exr",   False),
        ("Roughness", "BakedMaterial_roughness.exr",  False)]

# A texel on the rust of the metal cross: dielectric and rough, a millimetre from bare
# steel.  Full-resolution atlas coordinates, (row, column).
TEXEL = (855, 1619)
WINDOW = 256      # side of the first zoom, in atlas texels
MICRO = 9         # side of the second zoom, in atlas texels (odd: it has a centre)
PANEL = 1024      # the whole atlas is shown reduced to this many texels

# The render column.
DEPTH_TOL = 0.005     # metres of agreement between the depth map and the distance
FRAME_MARGIN = 140    # pixels of room the crop needs inside the frame
RENDER_ZOOM = 2.4     # crop side, as a multiple of the projected window
EXAMINED = 8          # candidate cameras reported before the pick
PATTERN_GRID = 64     # texels per side sampled by the pattern-agreement check
MIN_CORR = 0.50       # the correlation it demands; the right mapping gives 0.96
MIN_HUE_SPREAD = 0.05  # relative spread below which the window has no pattern to match

C_INK = C_CAM
C_MUTED = C_EDGE

C_GRID = "#8a8a8a"     # the texel grid of the second zoom
BADGE_FS = 17.0        # the printed value
BADGE_GAP = 0.16       # inches between the swatch and the number
MONO_ADV = 0.6023      # advance width of DejaVu Sans Mono, in em


def srgb(x: np.ndarray) -> np.ndarray:
    """sRGB encoding (IEC 61966-2-1), not gamma 2.2.  Same curve as make_atlas_pngs."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def encode(a: np.ndarray, is_colour: bool) -> np.ndarray:
    """A map to displayable RGB, by the convention its kind already has."""
    return srgb(a) if is_colour else np.clip(a, 0.0, 1.0)


def read_map(path: Path, texel, window: int, micro: int, panel: int):
    """One pass over an 800 MB EXR: the reduced atlas, the two crops, the value.

    The file is read once and released before the next one is opened; at 8192 x 8192 x 3
    float32 the three maps together would be 2.4 GB resident for no reason.
    """
    a = load_exr(path)
    y, x = texel
    h, w = a.shape[:2]
    k = max(1, h // panel)
    small = block_mean(a, k)
    half = window // 2
    crop = np.array(a[y - half:y + half, x - half:x + half])
    m = micro // 2
    tiny = np.array(a[y - m:y + m + 1, x - m:x + m + 1])
    value = np.array(a[y, x])
    shape = (h, w)
    del a
    return small, crop, tiny, value, shape, k


def check(shapes, values, on_surface: bool, texel, window: int, micro: int) -> None:
    """Refuse to draw a figure that asserts something the data does not support."""
    if len(set(shapes)) != 1:
        raise SystemExit(f"the maps have different resolutions {shapes}: "
                         "'the same texel' would not mean the same surface point")
    h, w = shapes[0]
    y, x = texel
    half = max(window // 2, micro // 2 + 1)
    if not (half <= y < h - half and half <= x < w - half):
        raise SystemExit(f"texel ({y}, {x}) is closer than {half} to the atlas edge: "
                         "the zoom window would fall outside the map")
    if not on_surface:
        raise SystemExit(f"texel ({y}, {x}) is empty atlas in the base colour, not "
                         "surface: the figure would zoom onto nothing")
    for (label, _, is_colour), v in zip(MAPS, values):
        if not is_colour and float(v.max() - v.min()) > 1e-6:
            raise SystemExit(f"{label} does not hold one value across R, G, B at this "
                             f"texel ({v}): printing a single number would be a lie")


# --------------------------------------------------------------- the render column
def to_ium(texel, bake_shape, ium_shape) -> tuple[int, int]:
    """A bake texel to the run's IUM texel covering the same (u, v).

    Through the UV coordinate and not through a ratio of resolutions, so the two atlases
    are only required to agree on their aspect, not to be a power of two apart.  Both
    store row 0 at v = 1 -- the IUM raygen flips v against the row index
    (deviceProgramsIUM.cu) and a Blender-baked EXR puts the top of the image on the first
    scanline -- so the row fraction maps straight across.  Measured, not assumed: the two
    masks agree on 98.4% of the atlas as they are, and on 62% with v flipped.
    """
    (bh, bw), (ih, iw) = bake_shape, ium_shape
    if abs(bw / bh - iw / ih) > 1e-6:
        raise SystemExit(f"the bake ({bw}x{bh}) and the run's IUM ({iw}x{ih}) have "
                         "different aspect ratios: one texel of one is not a region "
                         "of the other")
    return (int((texel[0] + 0.5) / bh * ih), int((texel[1] + 0.5) / bw * iw))


def project(p: np.ndarray, c2w: np.ndarray, intr: dict):
    """World point to pixel, the inverse of nerf/rays.py:get_rays_np.

    That function builds a camera-space direction ((i-cx)/fx, -(j-cy)/fy, -1) and rotates
    it by c2w; inverting it gives what is below.  Returns (u, v, distance), or None when
    the point is behind the camera.  The distance is Euclidean, which is what the run's
    depth maps hold (the OptiX kernel normalises the direction before tracing).
    """
    rot, o = c2w[:3, :3], c2w[:3, 3]
    dc = rot.T @ (p - o)
    s = -dc[2]
    if s <= 1e-6:
        return None
    return (intr["cx"] + intr["fx"] * dc[0] / s,
            intr["cy"] - intr["fy"] * dc[1] / s,
            float(np.linalg.norm(p - o)))


def pick_camera(run: Path, frames, intr, p, n, camera, margin: int):
    """The camera the render column uses, and why.

    Frontality orders the candidates, occlusion decides them.  The depth test is the one
    that matters: a point can sit in the middle of a frame and be behind something else,
    and on this scene one of the most frontal cameras is exactly that case.
    """
    cands = []
    for k, f in enumerate(frames):
        if camera is not None and k != camera:
            continue
        pr = project(p, np.array(f["transform_matrix"], dtype=np.float64), intr)
        if pr is None:
            continue
        u, v, dist = pr
        if not (margin <= u < intr["w"] - margin and margin <= v < intr["h"] - margin):
            continue
        cands.append((float(n @ ((np.array(f["transform_matrix"])[:3, 3] - p) / dist)),
                      k, u, v, dist))
    cands.sort(reverse=True)
    if not cands:
        raise SystemExit("no camera has the point in front of it and inside the frame "
                         f"with {margin} px to spare"
                         + (f" (--camera {camera})" if camera is not None else ""))

    chosen, examined = None, 0
    for cos, k, u, v, dist in cands:
        depth = load_exr(run / frames[k]["depth_path"])[int(round(v)), int(round(u)), 0]
        delta = float(depth) - dist
        clear = abs(delta) < DEPTH_TOL
        if examined < EXAMINED:
            print(f"    frame {k:>3}  cos {cos:.3f}  px ({u:7.1f}, {v:7.1f})  "
                  f"dist {dist:.4f}  depth {float(depth):.4f}  "
                  f"{'clear' if clear else 'OCCLUDED'}")
            examined += 1
        if clear and chosen is None:
            chosen = dict(idx=k, frame=frames[k], u=u, v=v, dist=dist, cos=cos,
                          delta=delta)
            if examined >= EXAMINED:
                break
    if chosen is None:
        raise SystemExit("every camera that frames the point has something in front of "
                         "it: there is no view to zoom into"
                         + (f" (--camera {camera})" if camera is not None else ""))
    return chosen


def window_quad(pos, mask, texel, window, bake_shape, c2w, intr):
    """The 256-texel window's outline, projected into the chosen frame.

    The four corners are read from the IUM and projected, which is exact for a window
    lying on one planar island.  When a corner falls outside the mask the window straddles
    a seam and its projection is not a quadrilateral at all; the bounding box of the valid
    texels is used instead and the substitution is printed, because it is a different
    claim about the same region.
    """
    half = window // 2
    corners = [(texel[0] - half, texel[1] - half), (texel[0] - half, texel[1] + half - 1),
               (texel[0] + half - 1, texel[1] + half - 1),
               (texel[0] + half - 1, texel[1] - half)]
    pts, valid = [], True
    for c in corners:
        iy, ix = to_ium(c, bake_shape, pos.shape[:2])
        if not mask[iy, ix]:
            valid = False
            break
        pr = project(pos[iy, ix], c2w, intr)
        if pr is None:
            valid = False
            break
        pts.append(pr[:2])
    if valid:
        return np.array(pts), True

    y0, x0 = to_ium((texel[0] - half, texel[1] - half), bake_shape, pos.shape[:2])
    y1, x1 = to_ium((texel[0] + half - 1, texel[1] + half - 1), bake_shape,
                    pos.shape[:2])
    sub_p = pos[y0:y1 + 1, x0:x1 + 1].reshape(-1, 3)
    sub_m = mask[y0:y1 + 1, x0:x1 + 1].reshape(-1)
    proj = [project(q, c2w, intr) for q in sub_p[sub_m]]
    proj = np.array([q[:2] for q in proj if q is not None])
    if len(proj) < 3:
        raise SystemExit("the window projects to fewer than three valid points: "
                         "nothing to outline in the render")
    lo, hi = proj.min(axis=0), proj.max(axis=0)
    return np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]]), False


def pattern_agreement(pos, mask, texel, window, bake_shape, c2w, intr, img, albedo_crop):
    """Does the render show, texel by texel, the pattern the albedo map holds?

    The single marked pixel cannot answer that: a mapping flipped in u or in v still lands
    on the same object and still reports a plausible colour.  A grid spanning the window
    can, because a flip scrambles which texel goes to which pixel, and the hue that the
    rust and the bare steel differ in stops lining up.  Returns (correlation, n, spread),
    with the correlation None when the window is too uniform in hue to discriminate.
    """
    step = max(1, window // PATTERN_GRID)
    g = np.arange(-(window // 2), window // 2, step)
    by, bx = np.meshgrid(texel[0] + g, texel[1] + g, indexing="ij")
    a = albedo_crop[np.ix_(g + window // 2, g + window // 2)]
    a_rg = a[..., 0] / np.maximum(a[..., 1], 1e-9)

    iy = ((by + 0.5) / bake_shape[0] * pos.shape[0]).astype(int)
    ix = ((bx + 0.5) / bake_shape[1] * pos.shape[1]).astype(int)
    p, m = pos[iy, ix], mask[iy, ix]
    rot, o = c2w[:3, :3], c2w[:3, 3]
    dc = (p - o) @ rot                       # rot.T @ (p - o), one row per texel
    s = -dc[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = intr["cx"] + intr["fx"] * dc[..., 0] / s
        v = intr["cy"] - intr["fy"] * dc[..., 1] / s
    ok = (m & (s > 1e-6) & np.isfinite(u) & np.isfinite(v)
          & (u >= 0) & (u < intr["w"]) & (v >= 0) & (v < intr["h"]))
    if ok.sum() < 64:
        raise SystemExit("fewer than 64 texels of the window project into the frame: "
                         "there is not enough overlap to check the correspondence")
    c = img[np.clip(v, 0, intr["h"] - 1).astype(int)[ok],
            np.clip(u, 0, intr["w"] - 1).astype(int)[ok]]
    r_rg = c[..., 0] / np.maximum(c[..., 1], 1e-9)
    ref = a_rg[ok]
    spread = float(ref.std() / max(abs(ref.mean()), 1e-9))
    if spread < MIN_HUE_SPREAD:
        return None, int(ok.sum()), spread
    return float(np.corrcoef(ref, r_rg)[0, 1]), int(ok.sum()), spread


def load_render(run: Path, texel, bake_shape, camera, zoom: float, margin: int,
                albedo_crop: np.ndarray, window: int):
    """Everything the render column draws, read from the run."""
    meta = json.loads((run / "transforms_extended.json").read_text())
    intr = dict(fx=meta["fl_x"], fy=meta["fl_y"], cx=meta["cx"], cy=meta["cy"],
                w=meta["w"], h=meta["h"])

    pos = load_exr(run / "ium" / "ium_positions.exr")
    nor = load_exr(run / "ium" / "ium_normals.exr")
    msk = load_exr(run / "ium" / "ium_masks.exr")[..., 0] > 0.5
    iy, ix = to_ium(texel, bake_shape, pos.shape[:2])
    if not msk[iy, ix]:
        raise SystemExit(f"the texel maps to IUM ({iy}, {ix}), which the run's mask "
                         "calls empty atlas: there is no surface point to project")
    p, n = pos[iy, ix].astype(np.float64), nor[iy, ix].astype(np.float64)
    n = n / max(np.linalg.norm(n), 1e-12)
    print(f"  IUM ({iy}, {ix}) of {pos.shape[1]}x{pos.shape[0]}: "
          f"p = {np.round(p, 4)}, n = {np.round(n, 3)}")

    cam = pick_camera(run, meta["frames"], intr, p, n, camera, margin)
    f = cam["frame"]
    c2w = np.array(f["transform_matrix"], dtype=np.float64)
    print(f"  chose frame {cam['idx']} ({Path(f['file_path']).stem}): "
          f"cos {cam['cos']:.3f}, {cam['dist']:.3f} m, "
          f"depth agrees to {abs(cam['delta']) * 1e3:.2f} mm")

    quad, exact = window_quad(pos, msk, texel, window, bake_shape, c2w, intr)
    if not exact:
        print("    ! the window straddles a UV seam; outlining its bounding box instead")
    side = float(max(quad[:, 0].max() - quad[:, 0].min(),
                     quad[:, 1].max() - quad[:, 1].min()))
    px_per_texel = side / window
    crop = max(int(round(zoom * side)), 64)
    print(f"    the {window}-texel window spans {side:.1f} px "
          f"({px_per_texel:.2f} px per texel); crop {crop} px")

    img = load_exr(run / f["file_path"])
    expo, med = exposure_of(img, SCENE_KEY)
    print(f"    exposure {expo:.3f} (median luminance {med:.4f})")
    rgb = tonemap(img, expo)

    corr, n_pts, spread = pattern_agreement(pos, msk, texel, window, bake_shape, c2w,
                                            intr, img, albedo_crop)
    if corr is None:
        print(f"    the window's hue spread is {spread:.3f}: too uniform to correlate, "
              "the correspondence check stands down")
    else:
        print(f"    albedo-vs-render hue correlation over {n_pts} texels of the window: "
              f"{corr:+.4f}")
        if corr < MIN_CORR:
            raise SystemExit(
                f"the render does not show the pattern the albedo holds (correlation "
                f"{corr:+.4f} < {MIN_CORR}): the projection is landing somewhere else, "
                "and the figure would claim a correspondence that is not there")
    raw = img[int(round(cam["v"])), int(round(cam["u"]))]
    alb = albedo_crop[window // 2, window // 2]
    print(f"    at the marked pixel: render R/G {float(raw[0] / max(raw[1], 1e-9)):.2f}, "
          f"albedo R/G {float(alb[0] / max(alb[1], 1e-9)):.2f}")

    half = crop / 2.0
    cu = float(np.clip(cam["u"], half, intr["w"] - half))
    cv = float(np.clip(cam["v"], half, intr["h"] - half))
    box = (cu - half, cv - half, crop, crop)
    return dict(rgb=rgb, quad=quad, box=box, cam=cam, intr=intr,
                px_per_texel=px_per_texel, name=Path(f["file_path"]).stem)


def draw_crosshair(ax, u: float, v: float, span: float) -> None:
    """A gap-centred cross on the marked point.  A filled marker would cover the very
    thing it points at: one texel is a fraction of a pixel here."""
    gap, arm = span * 0.035, span * 0.075
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ax.plot([u + dx * gap, u + dx * (gap + arm)],
                [v + dy * gap, v + dy * (gap + arm)],
                color=C_HILITE, lw=1.6, solid_capstyle="butt", zorder=6)


def draw_micro(ax, tiny_rgb: np.ndarray, micro: int) -> None:
    """The second zoom: single texels as squares, the marked one outlined."""
    ax.imshow(tiny_rgb, interpolation="nearest", extent=(0, micro, micro, 0))
    # A mid grey grid, not white: the block is black in the metallic map and almost
    # white in the roughness one, and one colour has to stay visible on both.
    for i in range(micro + 1):                       # the grid that makes them texels
        ax.axhline(i, color=C_GRID, lw=0.6, alpha=0.85)
        ax.axvline(i, color=C_GRID, lw=0.6, alpha=0.85)
    c = micro // 2
    ax.add_patch(Rectangle((c, c), 1, 1, fill=False, ec=C_HILITE, lw=2.4, zorder=5))
    ax.set_xlim(0, micro)
    ax.set_ylim(micro, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(C_MUTED)
        s.set_linewidth(1.0)


def frame(ax, colour=C_MUTED, lw=1.0) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(colour)
        s.set_linewidth(lw)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bake", required=True,
                    help="folder holding the BakedMaterial_*.exr of the scene")
    ap.add_argument("--out", required=True, help="destination folder")
    ap.add_argument("--name", default="pbr_texel_zoom", help="output stem, without .png")
    ap.add_argument("--texel", nargs=2, type=int, default=list(TEXEL),
                    metavar=("ROW", "COL"), help="the marked texel, full-resolution atlas")
    ap.add_argument("--window", type=int, default=WINDOW, help="side of the first zoom")
    ap.add_argument("--micro", type=int, default=MICRO, help="side of the second zoom")
    ap.add_argument("--panel", type=int, default=PANEL, help="atlas reduced to this side")
    ap.add_argument("--run", default=None,
                    help="run folder (ium/, depth/, images/, transforms_extended.json); "
                         "adds the render column")
    ap.add_argument("--camera", type=int, default=None,
                    help="force a frame index instead of choosing one")
    ap.add_argument("--render-zoom", type=float, default=RENDER_ZOOM,
                    help="crop side, as a multiple of the projected window")
    ap.add_argument("--bare", action="store_true",
                    help="drop every small caption, for a slide")
    ap.add_argument("--dpi", type=int, default=DPI)
    a = ap.parse_args(argv)

    bake, out = Path(a.bake), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    texel = (int(a.texel[0]), int(a.texel[1]))
    if a.micro % 2 == 0:
        raise SystemExit("--micro must be odd: the second zoom needs a centre texel")

    print(f"{bake.name} -> {(out / (a.name + '.png')).resolve()}")
    print(f"  texel (row {texel[0]}, col {texel[1]}), window {a.window}, "
          f"micro {a.micro}")

    data, shapes = [], []
    for label, fname, is_colour in MAPS:
        p = bake / fname
        if not p.exists():
            raise SystemExit(f"{p} does not exist")
        small, crop, tiny, value, shape, k = read_map(p, texel, a.window, a.micro,
                                                      a.panel)
        shapes.append(shape)
        data.append(dict(label=label, is_colour=is_colour, small=small, crop=crop,
                         tiny=tiny, value=value, k=k))
        print(f"  {label:<9} {shape[1]}x{shape[0]}, reduced /{k}, "
              f"value {np.round(value, 4)}")

    on_surface = float(data[0]["value"].max()) > 1e-4
    check(shapes, [d["value"] for d in data], on_surface, texel, a.window, a.micro)

    h, w = shapes[0]
    u, v = (texel[1] + 0.5) / w, (texel[0] + 0.5) / h
    print(f"  uv = ({u:.5f}, {v:.5f}) of a {w}x{h} atlas")

    ren = None
    if a.run:
        ren = load_render(Path(a.run), texel, (h, w), a.camera, a.render_zoom,
                          FRAME_MARGIN, data[0]["crop"], a.window)

    # -- layout, in inches ---------------------------------------------------
    # The margins carry the two caption lines; with --bare there is nothing to carry and
    # they close up, otherwise the figure would keep a band of white where the words were.
    m_side = 0.46
    m_top, m_bot = (0.50, 0.28) if a.bare else (0.95, 0.62)
    gap_col, col_w, gap_rule = 0.50, 3.90, 0.34
    n_extra = 1 if ren else 0
    fig_w = (2 * m_side + (3 + n_extra) * col_w + (2 + n_extra) * gap_col
             + n_extra * gap_rule)
    atlas_w, badge_h = 0.60 * col_w, 0.98
    gap_az, gap_zb = 0.42, 0.30
    fig_h = m_top + atlas_w + gap_az + col_w + gap_zb + badge_h + m_bot
    fig = plt.figure(figsize=(fig_w, fig_h))

    y_badge = m_bot / fig_h
    y_zoom = (m_bot + badge_h + gap_zb) / fig_h
    y_atlas = (m_bot + badge_h + gap_zb + col_w + gap_az) / fig_h
    cw = col_w / fig_w

    # x of the first map column: the render column, when there is one, sits to its left
    # behind a rule, because it is not a map and should not read as a fourth one.
    x_maps = m_side + (col_w + gap_col + gap_rule if ren else 0.0)

    if ren:
        # -- the whole frame -------------------------------------------------
        rh = col_w * ren["rgb"].shape[0] / ren["rgb"].shape[1]     # the render is 16:9
        ax_r = fig.add_axes([m_side / fig_w, y_atlas + (atlas_w - rh) / 2 / fig_h,
                             cw, rh / fig_h])
        ax_r.imshow(np.clip(block_mean(ren["rgb"], 2), 0.0, 1.0),
                    extent=(0, ren["intr"]["w"], ren["intr"]["h"], 0))
        frame(ax_r, lw=0.9)
        ax_r.set_title("Render", fontsize=18, color=C_INK, pad=10)

        # -- the crop around the point ---------------------------------------
        x0, y0, side, _ = ren["box"]
        ax_c = fig.add_axes([m_side / fig_w, y_zoom, cw, col_w / fig_h])
        ax_c.imshow(np.clip(ren["rgb"], 0.0, 1.0),
                    extent=(0, ren["intr"]["w"], ren["intr"]["h"], 0))
        ax_c.set_xlim(x0, x0 + side)
        ax_c.set_ylim(y0 + side, y0)
        frame(ax_c)
        ax_r.indicate_inset((x0, y0, side, side), inset_ax=ax_c,
                            edgecolor=C_HILITE, linewidth=1.4, alpha=1.0)
        ax_c.add_patch(Polygon(ren["quad"], closed=True, fill=False, ec=C_HILITE,
                               lw=1.6, zorder=5))
        draw_crosshair(ax_c, ren["cam"]["u"], ren["cam"]["v"], side)

        # -- which camera, where the value badges are ------------------------
        if not a.bare:
            ax_t = fig.add_axes([m_side / fig_w, y_badge, cw, badge_h / fig_h])
            ax_t.set_xlim(0, 1)
            ax_t.set_ylim(0, 1)
            ax_t.axis("off")
            ax_t.text(0.5, 0.62, f"camera {ren['cam']['idx']}  ·  {ren['name']}",
                      ha="center", va="center", fontsize=12.5, color=C_INK)
            ax_t.text(0.5, 0.26, f"{ren['cam']['dist']:.2f} m away  ·  one texel is "
                                 f"{ren['px_per_texel']:.2f} px here",
                      ha="center", va="center", fontsize=12, color=C_MUTED)

        # The rule is anchored to the content and not to the margins, so that closing
        # the margins for --bare does not send it off the top of the page.
        x_rule = (m_side + col_w + gap_col + gap_rule / 2) / fig_w
        fig.add_artist(plt.Line2D(
            [x_rule, x_rule],
            [(m_bot - 0.10) / fig_h,
             (m_bot + badge_h + gap_zb + col_w + gap_az + atlas_w + 0.42) / fig_h],
            color=C_MUTED, lw=0.8, alpha=0.35, transform=fig.transFigure))

    for i, d in enumerate(data):
        x0 = (x_maps + i * (col_w + gap_col)) / fig_w
        aw = atlas_w / fig_w

        # -- the whole atlas ------------------------------------------------
        ax_a = fig.add_axes([x0 + (cw - aw) / 2, y_atlas, aw, atlas_w / fig_h])
        ax_a.imshow(encode(d["small"], d["is_colour"]), interpolation="nearest")
        frame(ax_a, lw=0.9)
        ax_a.set_title(d["label"], fontsize=18, color=C_INK, pad=10)

        # -- the 256-texel window -------------------------------------------
        ax_z = fig.add_axes([x0, y_zoom, cw, col_w / fig_h])
        ax_z.imshow(encode(d["crop"], d["is_colour"]), interpolation="nearest",
                    extent=(0, a.window, a.window, 0))
        frame(ax_z)

        k = d["k"]
        bx = (texel[1] - a.window // 2) / k
        by = (texel[0] - a.window // 2) / k
        ax_a.indicate_inset((bx, by, a.window / k, a.window / k), inset_ax=ax_z,
                            edgecolor=C_HILITE, linewidth=1.4, alpha=1.0)

        # -- the 9x9 block, inset in the corner of the window ---------------
        ax_m = ax_z.inset_axes([0.035, 0.035, 0.33, 0.33])
        draw_micro(ax_m, encode(d["tiny"], d["is_colour"]), a.micro)
        c = a.window // 2
        ax_z.indicate_inset((c - a.micro // 2, c - a.micro // 2, a.micro, a.micro),
                            inset_ax=ax_m, edgecolor=C_HILITE, linewidth=1.4, alpha=1.0)

        # -- the value ------------------------------------------------------
        # Swatch and number are laid out as one group and centred under the column:
        # the albedo triple is three times as wide as a scalar, and a fixed left
        # margin would push it into the next column.
        val = d["value"]
        ax_b = fig.add_axes([x0, y_badge, cw, badge_h / fig_h])
        ax_b.set_xlim(0, 1)
        ax_b.set_ylim(0, 1)
        ax_b.axis("off")
        txt = (f"[{val[0]:.3f}, {val[1]:.3f}, {val[2]:.3f}]" if d["is_colour"]
               else f"[{val[0]:.3f}]")
        sw_in = badge_h * 0.68                           # a square swatch, in inches
        txt_in = len(txt) * MONO_ADV * BADGE_FS / 72.0
        group = (sw_in + BADGE_GAP + txt_in) / col_w
        if group > 1.0:
            raise SystemExit(f"the {d['label']} value does not fit its column "
                             f"({group * col_w:.2f} in of {col_w:.2f}): widen col_w "
                             "or lower BADGE_FS")
        left = (1.0 - group) / 2.0
        ax_b.add_patch(Rectangle((left, 0.16), sw_in / col_w, 0.68,
                                 facecolor=tuple(encode(val, d["is_colour"])),
                                 edgecolor=C_MUTED, lw=1.0))
        ax_b.text(left + (sw_in + BADGE_GAP) / col_w, 0.50, txt, ha="left",
                  va="center", fontsize=BADGE_FS, color=C_INK,
                  family="DejaVu Sans Mono")

    if not a.bare:
        seen = ("seen in the scene and read in the three maps" if ren
                else "read in the three maps")
        fig.text(0.5, 1.0 - 0.30 / fig_h,
                 f"one texel of the atlas, (u, v) = ({u:.4f}, {v:.4f}), {seen}",
                 ha="center", va="top", fontsize=14, color=C_MUTED)
        note = ("the swatch carries the texel's own value: the albedo sRGB encoded, "
                "metallic and roughness as linear grey")
        if ren:
            note += ("\nin the render, the red outline is the projection of that same "
                     f"{a.window}-texel window; Reinhard and gamma 2.2 on the median "
                     "luminance of the frame")
        fig.text(0.5, 0.14 / fig_h, note, ha="center", va="bottom", fontsize=11.5,
                 color=C_MUTED, linespacing=1.6)

    path = out / f"{a.name}.png"
    fig.savefig(path, dpi=a.dpi, facecolor="white")
    plt.close(fig)
    print(f"  + {path}  ({int(fig_w * a.dpi)}x{int(fig_h * a.dpi)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
