#!/usr/bin/env python
"""make_idea_figure.py -- the six-panel storyboard of Section 3.1 (Idea).

    python make_idea_figure.py --out ../../Doc/images
    python make_idea_figure.py --out ../../figure_review/idea --register 2d

Writes the same six panels twice, once per drawing register, into `<out>/idea-2d/`
and `<out>/idea-3d/`, so the two can be compared on the compiled page and one picked:

  idea_diffuse.png       one texel, two cameras, the two recorded colours EQUAL
  idea_glossy.png        the same scene with x lowered, the two colours DIFFERENT
  idea_hemisphere.png    the light the patch gathers from the whole sky, no cameras
  idea_cones.png         the light gathered around each camera's mirror direction
  idea_recap_diffuse.png the diffuse patch with the hemisphere: the whole story
  idea_recap_both.png    the other patch: hemisphere and cones, both needed

The section asks its question in words and the figure has to answer in the same
register, so there is not a symbol on it.  What keeps it honest is that every colour
comes from `make_pbr_model_diagram.build_case`, i.e. from the real integrals of the
model closed by the same functions the bake uses; `report` prints them and asserts the
properties the panels claim.

Three things that are easy to get wrong and that the code deliberately fixes:

  * the six panels are saved on the FULL canvas (`bbox_inches=None`) and then cropped
    with ONE shared box.  `bbox_inches="tight"` crops on the axes box and returns six
    images of different size in unrelated pixel frames, and LaTeX scaling each of them
    to the same width would put the six drawings at six different scales;

  * the first pair differs in the material and in NOTHING else.  Both materials come
    out of one case (neither E nor L_j depends on x) and every panel of both registers
    shares one exposure, otherwise a comparison between panels would be comparing
    exposures;

  * in the plane, the two cameras lose the azimuth, which is the freedom that keeps
    each camera's view ray out of the other camera's cone.  `report` measures that
    clearance instead of trusting the angles.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _paths  # noqa: F401

from make_pbr_model_diagram import (CAMS, CONE_S, C_DIFF, C_INK, HALO, THETA,
                                    X_DIFFUSE, build_case, circle_on_sphere,
                                    cone_directions, draw_scene_base, label3d,
                                    luminance, sky, terms)

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge

# ── The canvas ───────────────────────────────────────────────────────────────
# 4.2 inches and not 7.0: `pbr_model.png` is a 7.0 in canvas printed at
# 0.8\linewidth, i.e. reduced to 0.662, which is what puts its 13 pt labels at 8.6 pt
# on the page.  These panels print at 0.48\linewidth, so the canvas that reproduces the
# SAME reduction, and with it the same on-page type size, is 7.0 * 0.48/0.8.  Taking it
# from the ratio instead of by eye is the only way the storyboard and the figure it
# comes from stay typographically consistent.  Only the width sets the reduction, so
# the canvas is free to be as tall as the drawing needs.
WIDTH = 4.2
DPI = 300                       # integer pixel sizes: no rounding between panels

# Within a register the canvas, the axes rect and the strip band are the SAME in all
# six panels, including the two that draw no strip: a panel that reclaimed the free
# band would put its drawing at a different scale from its neighbours, which is the one
# thing a storyboard of six subfigures cannot afford.  Between registers they differ,
# because a half-disc and a projected 3D box do not have the same shape, and the crop
# that keeps the scale consistent is per register anyway.
LAYOUT = {
    "2d": dict(figsize=(WIDTH, 3.2), axes=(0.0, 0.0, 1.0, 0.76),
               bleed=("left", "right", "bottom"),
               strip=(0.30, 0.70, 0.40, 0.095)),
    # The axes is far larger than the canvas and mostly off it.  mplot3d fits the whole
    # projected BOX inside the frame, and the drawing is inscribed in that box: with a
    # rect of ordinary size the dome came out filling 56% of the width and 44% of the
    # height, the rest being the corners of the box.  Those corners are empty, so the
    # fix is to push them off the canvas.  The numbers come from measuring the ink of a
    # rendered panel, not from taste; `report` checks that nothing ends up cut.  It is
    # the same trick as `make_pbr_model_diagram.figure`, which oversizes by 10%, only
    # here it has to work without `bbox_inches="tight"` to lean on afterwards.
    "3d": dict(figsize=(WIDTH, 3.7), axes=(-0.309, -0.282, 1.583, 1.541), bleed=(),
               strip=(0.30, 0.815, 0.40, 0.085)),
}

# The drawing inside that canvas is 0.6 times the one of `pbr_model.png` while points
# are absolute, so the scene furniture would come out relatively fatter and the dome
# 2.8 times denser.  These bring it back.  The TEXT is deliberately not scaled: that is
# the whole point of choosing the canvas above.
SCENE_SCALE = 0.85
N_DOME = 320                    # dome dots in 3D, against the 520 of the full figure
N_SKY = 180                     # segments of the 2D sky band
N_GATHER = 17                   # candidate gather arrows, before the two exclusions
FS = 10.5                       # label size
R_SKY = 1.10                    # outer radius of the 2D sky band
W_SKY = 0.10                    # its thickness

# ── The two camera sets ──────────────────────────────────────────────────────
# The 3D register reuses the constant of make_pbr_model_diagram, so Section 3.1 and
# Section 3.5 show the same configuration.  The 2D register cannot: a section needs the
# two cameras coplanar with the normal, and coplanar cameras lose a degree of freedom.
# In the plane, with camera 1 at -theta_1 and camera 2 at +theta_2, the mirror
# directions land at +theta_1 and -theta_2, so the view ray of one camera sits
# |theta_1 - theta_2| away from the axis of the other's cone: with the angles of the 3D
# set (55 and 45) that is 10 degrees against a half-aperture of 20, i.e. straight
# through it.  Three constraints have to hold at once, and `report` checks them:
#   |theta_1 - theta_2| > Theta/2 + margin      a view ray clears the other cone
#   theta_j + Theta/2 <= 90                     the cone stays above the horizon
#   theta_j > Theta/2                           the cone does not cross the zenith
CAMS_2D = (
    dict(theta=64.0, phi=180.0, name="camera 1", color="#c0392b"),
    dict(theta=32.0, phi=0.0,   name="camera 2", color="#12776b"),
)
VIEW_CONE_MARGIN = 8.0          # degrees of clearance the checks require

X_MATTE = 1.0                   # panels (a) and (e): C_1 == C_2 by construction
X_GLOSSY = X_DIFFUSE            # panels (b) and (f): the case pbr_model.png draws

REGISTERS = ("2d", "3d")

# What each panel puts on the scene.  A table and not six functions, because the six
# panels ARE the same drawing with things switched on and off, which is exactly what
# the storyboard is telling the reader.
#   material  which pair of recorded colours the strip shows, None for no strip
#   gather    the incoming arrows of the light collected over the whole hemisphere
#   mirror    the reflected ray and the cone around it
#   views     the dashed ray from the texel to each camera
# Every panel with a cone keeps its view rays: the reflected ray of a camera points
# away from it, so without the dashed ray back to the glyph the reader has only the
# colour to tie a cone to the camera it belongs to, and the pair of rays meeting at the
# texel is what makes the drawing read as a reflection at all.  What that costs is a
# view ray passing near the other camera's cone, and `report` measures the clearance
# instead of leaving it to a comment.
PANEL_SPEC = {
    "diffuse":       dict(material="matte",  views=True),
    "glossy":        dict(material="glossy", views=True),
    "hemisphere":    dict(gather=True, cams=False),
    "cones":         dict(mirror=True, views=True),
    "recap_diffuse": dict(material="matte",  gather=True, views=True),
    "recap_both":    dict(material="glossy", gather=True, views=True, mirror=True),
}
PANELS = tuple(PANEL_SPEC)


def spec(name: str) -> dict:
    """Panel settings with the defaults filled in."""
    return {"cams": True, "gather": False, "mirror": False, "views": False,
            "material": None, **PANEL_SPEC[name]}


# ────────────────────────────────────────────────────────── cases and materials
def build_registers() -> dict:
    """One case per register: the 3D one, and the same scene cut into a plane."""
    return {"2d": build_case(CAMS_2D), "3d": build_case(CAMS)}


def materials(case: dict) -> dict:
    """The two materials of the storyboard, on one case.

    Neither E nor L_j depends on x, so the fully diffuse patch and the glossy one are
    the same scene, under the same light, seen by the same cameras: x is the only thing
    that changes between the first two panels, which is exactly what their caption
    claims.  With x = 1 the specular term vanishes and both cameras record a E / pi, so
    the equality of the two swatches is a property of the model and not something the
    drawing arranges.
    """
    return {name: [terms(case, c, x=x)[2] for c in case["cams"]]
            for name, x in (("matte", X_MATTE), ("glossy", X_GLOSSY))}


def shared_tonemap(cases: dict):
    """One exposure for every radiance any panel of any register draws.

    `make_pbr_model_diagram.make_tonemap` reads the peak of a single case, which would
    put the two registers, and the two materials, on two different exposures: the
    reader would then be comparing exposures instead of colours.  The sun is left out
    of the peak on purpose and stays the only thing allowed to clip, as in the figure
    this scene comes from.
    """
    peak = 0.0
    for case in cases.values():
        peak = max(peak, float(np.max(case["E"] / np.pi)))
        peak = max(peak, max(float(np.max(c["L"])) for c in case["cams"]))
        for cols in materials(case).values():
            peak = max(peak, max(float(np.max(c)) for c in cols))
    scale = 0.95 / peak

    def tm(rgb):
        return np.clip(np.asarray(rgb) * scale, 0.0, 1.0) ** (1.0 / 2.2)

    return tm


def u8(rgb) -> np.ndarray:
    """What the reader actually sees: the tonemapped colour, quantised."""
    return np.round(np.asarray(rgb) * 255.0).astype(int)


# ─────────────────────────────────────────── the colour strip, shared by both
def draw_strip(fig, case: dict, cols, tm, *, equal: bool, register: str) -> None:
    """The two recorded colours as two rectangles sharing an edge.

    The device carries the panel's whole argument: if the two colours are the same the
    strip is one uniform block and the seam does not exist, and if they differ it is
    impossible to miss.  So the two halves are drawn WITHOUT an edge of their own and a
    single outline is put around the pair; bordering each half in its camera's colour
    would draw the seam even when there is nothing to see, which is precisely the claim
    the panel is making.

    It lives on its own axes in figure coordinates, so it comes out identical in the
    two registers and they can at least be compared on that.
    """
    ax = fig.add_axes(LAYOUT[register]["strip"])
    ax.set_axis_off()
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 1.0)
    for i, (col, cam) in enumerate(zip(cols, case["cams"])):
        ax.add_patch(Rectangle((i, 0.0), 1.0, 1.0, facecolor=tm(col),
                               edgecolor="none"))
        # The names go ABOVE the strip: below they would land on the apex of the sky,
        # which is the one part of the scene that reaches the top of the panel.  They
        # are also the only place the cameras are named on a panel that has a strip,
        # so the colour of the word is what ties a half to its camera.
        ax.text(i + 0.5, 1.30, f"camera {i + 1}", color=cam["color"], fontsize=FS - 1,
                ha="center", va="bottom")
    ax.add_patch(Rectangle((0.0, 0.0), 2.0, 1.0, facecolor="none", edgecolor=C_INK,
                           linewidth=1.1))
    ax.text(2.32, 0.46, "=" if equal else r"$\neq$", color=C_INK, fontsize=16,
            ha="center", va="center")


# ────────────────────────────────────────────────────────────── the 2D register
def ax_2d(fig):
    """Axes of the section, framed so that `equal` does not letterbox the drawing."""
    rect, fs = LAYOUT["2d"]["axes"], LAYOUT["2d"]["figsize"]
    ax = fig.add_axes(rect)
    # Wide enough for the camera names, which stick out past the glyphs at the two
    # ends of the half-circle.  Whatever no panel uses, the shared crop takes back.
    x = 1.42
    # The vertical span follows the axes box: otherwise `set_aspect("equal")` pads one
    # direction and the drawing stops filling the panel at the same scale as its
    # neighbours.
    span = 2.0 * x * (fs[1] * rect[3]) / (fs[0] * rect[2])
    ax.set_xlim(-x, x)
    ax.set_ylim(-0.26, span - 0.26)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    return ax


def p2(d) -> np.ndarray:
    """A direction of the case, read in the plane the 2D set was built to lie in."""
    return np.array([d[0], d[2]])


def arrow2d(ax, a, b, *, color, lw, zorder=6, head=0.26):
    """The 2D arrow idiom of make_geometry_diagrams.

    `annotate` and not `FancyArrowPatch`: an arrowstyle's head is measured in
    mutation units, which `annotate` scales by the font size while a bare patch leaves
    at 1, so the same numbers give a visible head here and none there.
    """
    ax.annotate("", xy=tuple(b), xytext=tuple(a), zorder=zorder,
                arrowprops=dict(arrowstyle=f"-|>,head_width={head},"
                                           f"head_length={head * 2.0}",
                                color=color, lw=lw, shrinkA=0, shrinkB=0))


def camera_2d(ax, p, look, *, color, size, zorder=8) -> None:
    """A small camera body with its lens turned toward the texel.

    Not `make_nerf_fgbg_figure.camera_glyph`: that one is a square of side `size`, which
    at this scale is a coloured blob with no direction to it, and here the direction is
    half of what the glyph has to say.
    """
    d = np.asarray(look, float)
    d = d / np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    body = np.array([p - d * 0.55 * size + n * 0.5 * size,
                     p - d * 0.55 * size - n * 0.5 * size,
                     p + d * 0.15 * size - n * 0.5 * size,
                     p + d * 0.15 * size + n * 0.5 * size])
    lens = np.array([p + d * 0.15 * size + n * 0.30 * size,
                     p + d * 0.15 * size - n * 0.30 * size,
                     p + d * 0.62 * size - n * 0.46 * size,
                     p + d * 0.62 * size + n * 0.46 * size])
    for poly in (body, lens):
        ax.fill(poly[:, 0], poly[:, 1], color=color, zorder=zorder, lw=0)


def scene_2d(ax, case: dict, tm, *, gather: bool, cones: bool) -> None:
    """Surface, texel, normal, the sky band and the light gathered over the hemisphere.

    The environment is a continuous band and not a row of dots weighted by cos(theta)
    as the 3D dome is.  A section shows one direction per angle instead of a solid
    angle, so a dot carrying its weight would be a rule the drawing cannot honour
    anyway; the band says what this panel needs to say, that the sky is out there and
    that one part of it is much brighter, and it says it without leaving the bright
    spot as white ink on white paper, which is what a thin row of dots does.
    """
    ax.add_patch(Rectangle((-1.45, -0.26), 2.9, 0.26, facecolor="0.82",
                           edgecolor="none", zorder=0))
    ax.plot([-1.45, 1.45], [0.0, 0.0], color="0.5", lw=0.9 * SCENE_SCALE, zorder=1)

    edges = np.linspace(0.0, 180.0, N_SKY + 1)
    mid = np.radians(0.5 * (edges[:-1] + edges[1:]))
    d = np.stack([np.cos(mid), np.zeros_like(mid), np.sin(mid)], axis=-1)
    col = tm(sky(d, case["sun_dir"]))
    for i in range(N_SKY):
        # The edge is the face colour: with any other, 180 segments would read as a
        # dashed band instead of a sky.
        ax.add_patch(Wedge((0.0, 0.0), R_SKY, edges[i], edges[i + 1], width=W_SKY,
                           facecolor=col[i], edgecolor=col[i], lw=0.4, zorder=2))
    # A thin rule on both rims.  Where the sun saturates, the band is white, and
    # without the rule that stretch reads as a gap in the band rather than as its
    # brightest part.
    a = np.radians(np.linspace(0.0, 180.0, 240))
    for r in (R_SKY, R_SKY - W_SKY):
        ax.plot(np.cos(a) * r, np.sin(a) * r, color="0.72", lw=0.6, zorder=3)
    s = p2(case["sun_dir"]) * (R_SKY - W_SKY * 0.5)
    ax.scatter([s[0]], [s[1]], s=170 * SCENE_SCALE ** 2, marker="o",
               color=tm(sky(case["sun_dir"], case["sun_dir"])), edgecolor=C_DIFF,
               linewidth=1.5 * SCENE_SCALE, zorder=3)

    if gather:
        # Same rule as `draw_scene_base`: arrows spread over the whole hemisphere,
        # skipping the ones a cone already fills.
        for a in np.linspace(np.radians(12.0), np.radians(168.0), N_GATHER):
            u = np.array([np.cos(a), 0.0, np.sin(a)])
            # Around the zenith the normal and its label already occupy the wedge, and
            # inside a cone the drawing is full: the light does arrive from there too,
            # the arrows just have nowhere to go.
            if abs(np.degrees(a) - 90.0) < 15.0:
                continue
            if cones and any(np.degrees(np.arccos(np.clip(u @ c["r"], -1.0, 1.0)))
                             < case["half"] + 4.0 for c in case["cams"]):
                continue
            q = p2(u)
            arrow2d(ax, q * 0.92, q * 0.62, color=C_DIFF, lw=1.4 * SCENE_SCALE,
                    zorder=3, head=0.22)

    # Shorter than the unit normal of the 3D panels: the label has to sit above the
    # arrow, in the wedge around the zenith that no cone reaches, and at full length
    # the tip is already inside the sky band.
    arrow2d(ax, (0.0, 0.0), (0.0, 0.76), color=C_INK, lw=2.0 * SCENE_SCALE, zorder=6)
    ax.text(0.0, 0.80, "normal", color=C_INK, fontsize=FS, ha="center", va="bottom",
            bbox=HALO, zorder=8)
    ax.plot([-0.11, 0.11], [0.0, 0.0], color=C_INK, lw=3.6 * SCENE_SCALE, zorder=7,
            solid_capstyle="butt")
    ax.text(0.15, -0.12, "texel", color=C_INK, fontsize=FS, ha="left", va="center",
            zorder=8)


def cams_2d(ax, case: dict, *, views: bool, mirror: bool, label: bool) -> None:
    """The two cameras, their view rays and, when asked, their cones."""
    for k, c in enumerate(case["cams"], start=1):
        v, r, col = p2(c["v"]), p2(c["r"]), c["color"]
        if views:
            ax.plot([0.0, v[0]], [0.0, v[1]], color=col, lw=1.5 * SCENE_SCALE,
                    ls=(0, (4, 2)), zorder=6)
        if mirror:
            a = np.degrees(np.arctan2(r[1], r[0]))
            lo, hi = a - case["half"], a + case["half"]
            ax.add_patch(Wedge((0.0, 0.0), 0.99, lo, hi, facecolor=col,
                               edgecolor="none", alpha=0.16, zorder=4))
            for e in (lo, hi):
                u = np.radians(e)
                ax.plot([0.0, np.cos(u) * 0.99], [0.0, np.sin(u) * 0.99], color=col,
                        lw=1.4 * SCENE_SCALE, zorder=5)
            # The slice of sky that cone averages, marked on the outside of the band:
            # this is what makes the two cameras collect different light visible, and
            # it says it without covering the band it is pointing at.
            ax.add_patch(Wedge((0.0, 0.0), R_SKY + 0.055, lo, hi, width=0.05,
                               facecolor=col, edgecolor="none", zorder=5))
            arrow2d(ax, (0.0, 0.0), r * 0.99, color=col, lw=2.2 * SCENE_SCALE,
                    zorder=6)
        pos = v * (R_SKY + 0.13)
        camera_2d(ax, pos, -v, size=0.19, color=col)
        if label:
            ax.text(pos[0], pos[1] + 0.12, f"camera {k}", color=col, fontsize=FS,
                    ha="center", va="bottom", bbox=HALO, zorder=9)


# ────────────────────────────────────────────────────────────── the 3D register
def ax_3d(fig):
    return fig.add_axes(LAYOUT["3d"]["axes"], projection="3d")


def scene_3d(ax, case: dict, tm, *, gather: bool, cones: bool) -> None:
    """The scene of `pbr_model.png`, thinned for the panel and stripped of symbols."""
    draw_scene_base(ax, case, tm, arrows=gather, scale=SCENE_SCALE, n_dome=N_DOME)
    n = case["n"]
    ax.quiver(0, 0, 0, *n, color=C_INK, lw=2.0 * SCENE_SCALE, arrow_length_ratio=0.12,
              zorder=6)
    # After `draw_scene_base`, never before: `label3d` projects by hand and reads the
    # projection matrix, which is what the limits, the box aspect and the view angles
    # set at the top of that function.
    # To the left and not straight above: at panel size the zenith is where camera 2's
    # name comes down to meet it.
    label3d(ax, n * 1.04 + np.array([0.0, -0.34, 0.06]), "normal", color=C_INK,
            fontsize=FS, ha="right", va="center", bbox=HALO)
    ax.scatter([0], [0], [0], color=C_INK, s=22 * SCENE_SCALE ** 2, zorder=7)
    label3d(ax, (0.20, -0.12, 0.02), "texel", color=C_INK, fontsize=FS, bbox=HALO)


def cams_3d(ax, case: dict, *, views: bool, mirror: bool, label: bool) -> None:
    for k, c in enumerate(case["cams"], start=1):
        v, r, col = c["v"], c["r"], c["color"]
        if views:
            ax.plot([0, v[0]], [0, v[1]], [0, v[2]], color=col, lw=1.5 * SCENE_SCALE,
                    ls=(0, (4, 2)), zorder=6)
        if mirror:
            ax.quiver(0, 0, 0, *r, color=col, lw=2.2 * SCENE_SCALE,
                      arrow_length_ratio=0.12, zorder=6)
            cd = cone_directions(r, case["half"], 26)
            ax.scatter(cd[:, 0] * 1.03, cd[:, 1] * 1.03, cd[:, 2] * 1.03,
                       s=CONE_S * SCENE_SCALE ** 2, color=col, edgecolor="white",
                       linewidth=0.5, depthshade=False, zorder=5)
            rim = circle_on_sphere(r, case["half"], 90)
            t = np.linspace(0.0, 1.0, 2)[:, None, None]
            cone = t * rim[None, :, :]
            ax.plot_surface(cone[..., 0], cone[..., 1], cone[..., 2], color=col,
                            alpha=0.15, linewidth=0, shade=False, zorder=4)
            ax.plot(rim[:, 0], rim[:, 1], rim[:, 2], color=col, lw=1.6 * SCENE_SCALE,
                    zorder=5)
        ax.scatter([v[0] * 1.13], [v[1] * 1.13], [v[2] * 1.13], marker="s",
                   s=130 * SCENE_SCALE ** 2, color=col, depthshade=False, zorder=8)
        if label:
            label3d(ax, v * 1.13 + np.array([0.0, 0.0, 0.18]), f"camera {k}",
                    color=col, fontsize=FS, ha="center", va="bottom", bbox=HALO)


KITS = {"2d": dict(ax=ax_2d, scene=scene_2d, cams=cams_2d),
        "3d": dict(ax=ax_3d, scene=scene_3d, cams=cams_3d)}


# ──────────────────────────────────────────────────────────── writing the panels
def write_panel(name: str, register: str, case: dict, tm, out: Path) -> Path:
    """One panel, on the shared canvas.  `bbox_inches` is left out on purpose."""
    s = spec(name)
    kit = KITS[register]
    fig = plt.figure(figsize=LAYOUT[register]["figsize"])
    ax = kit["ax"](fig)
    kit["scene"](ax, case, tm, gather=s["gather"], cones=s["mirror"])
    if s["cams"]:
        # The cameras are named once per panel: where the strip is drawn it already
        # names them, in their own colours, and repeating it at the glyph only adds
        # two more words to a panel that has to be read in a glance.
        kit["cams"](ax, case, views=s["views"], mirror=s["mirror"],
                    label=not s["material"])
    if s["material"]:
        draw_strip(fig, case, materials(case)[s["material"]], tm, register=register,
                   equal=s["material"] == "matte")
    path = out / f"idea_{name}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def crop_shared(paths, bleed=(), pad: int = 16) -> tuple:
    """Crop every panel with ONE box, so the six drawings end up at one scale.

    The six canvases are identical in size and framing, so a data unit is the same
    number of pixels in all of them and a given point of the scene falls on the same
    pixel: cropping moves the origin, never the scale.  That only holds because the
    panels were saved on the full canvas; `bbox_inches="tight"` would have cropped each
    one on its own axes box first, leaving six unrelated pixel frames whose bounding
    boxes cannot be compared, let alone unioned.

    The price is that the sparse panels keep the whitespace the full ones need.  That
    is the registration, not waste, and it must not be trimmed away per panel.
    """
    from PIL import Image, ImageChops

    ims = [Image.open(p).convert("RGB") for p in paths]
    size = ims[0].size
    assert all(im.size == size for im in ims), \
        "the panels are not the same size: the shared crop would change their scale"
    box = None
    for im, p in zip(ims, paths):
        b = ImageChops.difference(im, Image.new("RGB", size, (255, 255, 255))).getbbox()
        assert b is not None, "a panel is blank"
        # Nothing may touch a border it is not meant to run off.  The 3D register puts
        # the axes deliberately larger than the canvas to push the empty corners of the
        # projected box off it, and the failure mode of that trick is a label or a rim
        # quietly walking off the edge: the panel still looks plausible, it is only
        # missing a piece.  The surface of the 2D register, on the other hand, is meant
        # to run off the sides and the bottom, the way a ground plane does.
        for side, inside in (("left", b[0] > 0), ("top", b[1] > 0),
                             ("right", b[2] < size[0]), ("bottom", b[3] < size[1])):
            assert inside or side in bleed, \
                f"{p.name}: the drawing reaches the {side} border, something is cut off"
        box = b if box is None else (min(box[0], b[0]), min(box[1], b[1]),
                                     max(box[2], b[2]), max(box[3], b[3]))
    box = (max(box[0] - pad, 0), max(box[1] - pad, 0),
           min(box[2] + pad, size[0]), min(box[3] + pad, size[1]))
    for im, p in zip(ims, paths):
        im.crop(box).save(p)
    return box


# ──────────────────────────────────────────────────────────────────── the checks
def report(cases: dict, tm, written: dict) -> None:
    """Print the numbers and assert that the panels show what their captions claim."""
    half = THETA / 2.0
    for reg, case in cases.items():
        print(f"\n  register {reg}")
        for k, c in enumerate(case["cams"], start=1):
            err = float(np.max(np.abs(c["L"] - c["L_check"])) / np.max(c["L_check"]))
            print(f"    L_{k} = " + ", ".join(f"{v:.4f}" for v in c["L"])
                  + f"   lum {luminance(c['L']):.4f}   ({c['n_rays']} rays, "
                    f"cone {'whole' if c['unclipped'] else 'TRUNCATED'}, "
                    f"gap {err:.1e})")
            assert c["unclipped"], f"{reg}: the cone leaves the horizon"
            assert err < 1e-2, f"{reg}: the bake's closing does not match the cone"

        # Each camera's view ray has to stay clear of the OTHER camera's cone, or the
        # panels read as if a camera were looking into a reflection that is not its
        # own.  In 3D this is held by the azimuths and documented only in a comment;
        # here it is measured, because the 2D set has no azimuth left to play with.
        for j, cj in enumerate(case["cams"]):
            for k, ck in enumerate(case["cams"]):
                if j == k:
                    continue
                ang = np.degrees(np.arccos(np.clip(cj["v"] @ ck["r"], -1.0, 1.0)))
                print(f"    v_{j + 1} to R_{k + 1}: {ang:.1f} deg "
                      f"(needs > {half + VIEW_CONE_MARGIN:.0f})")
                assert ang > half + VIEW_CONE_MARGIN, \
                    f"{reg}: camera {j + 1} looks into camera {k + 1}'s cone"

        mat = materials(case)
        m1, m2 = mat["matte"]
        g1, g2 = mat["glossy"]
        print(f"    matte  C_1 lum {luminance(m1):.4f}   C_2 lum {luminance(m2):.4f}")
        print(f"    glossy C_1 lum {luminance(g1):.4f}   C_2 lum {luminance(g2):.4f}"
              f"   ratio {luminance(g1) / luminance(g2):.2f}")

        # The seam of the matte strip has to be invisible ON THE PAGE, which is a
        # statement about the quantised pixel and not about the float behind it.
        assert np.array_equal(u8(tm(m1)), u8(tm(m2))), \
            f"{reg}: the diffuse pair is not identical once quantised"
        # And the glossy seam has to be unmissable.  The linear ratio is not enough on
        # its own: the gamma compresses it, 1.3 linear becomes 1.13 after **(1/2.2).
        d = int(np.max(np.abs(u8(tm(g1)) - u8(tm(g2)))))
        print(f"    glossy pair, largest channel gap on the page: {d}/255")
        assert luminance(g1) / luminance(g2) > 1.3, \
            f"{reg}: the two cameras record almost the same colour"
        assert d >= 15, f"{reg}: the two colours differ in the data but not on the page"

        if reg == "2d":
            # The section is only honest if the sun and both views lie in the plane it
            # cuts.  `build_case` derives the sun from R_1 and n, so this holds when the
            # azimuths are 0 and 180, but it is cheaper to check than to reason about.
            assert abs(case["sun_dir"][1]) < 1e-12, "2d: the sun is off the plane"
            for c in case["cams"]:
                assert abs(c["v"][1]) < 1e-12, "2d: a camera is off the plane"
                t = np.degrees(np.arccos(np.clip(c["v"][2], -1.0, 1.0)))
                assert t + half <= 90.0, "2d: the cone crosses the horizon"
                assert t > half, "2d: the cone crosses the zenith"

        if reg in written:
            box, paths = written[reg]
            print(f"    {len(paths)} panels, shared crop {box[2] - box[0]}"
                  f"x{box[3] - box[1]} px")

    # The number the 2D caption has to quote: a wedge of aperture Theta takes Theta/180
    # of the visible half-circle, while the cone covers 1 - cos(Theta/2) of the
    # hemisphere.  Printed rather than written by hand, so the caption cannot go stale.
    on_page = THETA / 180.0
    in_fact = 1.0 - np.cos(np.radians(half))
    print(f"\n  a cone of {THETA:.0f} deg: {on_page * 100:.1f}% of the half-circle on "
          f"the page, {in_fact * 100:.1f}% of the hemisphere in fact "
          f"(factor {on_page / in_fact:.1f})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True,
                    help="parent folder: idea-2d/ and idea-3d/ are created inside it")
    ap.add_argument("--register", default="2d,3d", help="2d, 3d or both")
    ap.add_argument("--panels", default=",".join(PANELS))
    args = ap.parse_args()

    want = tuple(r.strip() for r in args.register.split(",") if r.strip())
    if not set(want) <= set(REGISTERS):
        ap.error(f"unknown register: {sorted(set(want) - set(REGISTERS))}")
    panels = tuple(p.strip() for p in args.panels.split(",") if p.strip())
    if not set(panels) <= set(PANELS):
        ap.error(f"unknown panel: {sorted(set(panels) - set(PANELS))}")

    cases = build_registers()
    tm = shared_tonemap(cases)

    written = {}
    for reg in want:
        out = Path(args.out) / f"idea-{reg}"
        out.mkdir(parents=True, exist_ok=True)
        paths = [write_panel(name, reg, cases[reg], tm, out) for name in panels]
        box = crop_shared(paths, LAYOUT[reg]["bleed"])
        written[reg] = (box, paths)
        for p in paths:
            print(f"  + {p}")

    report(cases, tm, written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
