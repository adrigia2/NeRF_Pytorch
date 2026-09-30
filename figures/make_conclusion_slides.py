#!/usr/bin/env python
"""make_conclusion_slides.py -- the two closing slides: conclusions and future work.

    python make_conclusion_slides.py --out ../../PresentationImages

Writes `conclusions.png` and `future_work.png`, in the same house style as
`make_full_model_slide.py` (white ground, DejaVu, amber for the diffuse side and blue for
the specular one, ink and muted grey for everything else).

CONCLUSIONS.  One line tells the whole story, left to right: the three inputs, the hybrid
box in which OptiX and the NeRF each do their part, the three maps that come out of it
(real crops of the shield boss from the night bake of SwordShield), and the closing check,
the scene re-rendered with those maps beside the original render.  Under it four cards,
one per claim of the chapter, so the speaker points at a card and says a sentence.

FUTURE WORK.  Four cards in a 2x2 grid, each with a drawn glyph rather than an icon: a
small bar chart for the comparison with the state of the art, a blurred and a sharp tile
for the radiance source, a Fresnel curve per channel for the richer specular model, and
the residual over the aperture grid with the interpolated minimum for the roughness
calibration.  The middle of the slide carries the one sentence the chapter insists on:
the pipeline itself does not change, better neural rendering and faster GPUs arrive as
better maps.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _paths  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Wedge    # noqa: E402
from PIL import Image                                                    # noqa: E402

from make_geometry_diagrams import C_EDGE, C_CAM, C_INDIR, C_ACCEPT      # noqa: E402

C_INK = C_CAM
C_MUTED = C_EDGE
C_DIFF = "#c98a20"             # diffuse / amber, as in the thesis PBR diagram
C_SPEC = "#2171b5"             # specular / blue, as in the cone ramp
C_MARK = "#ffe27a"             # highlight
C_CARD = "#f4f6f8"             # card ground
C_LINE = "#d5dbe1"             # card border

DPI = 200
MARGIN = 3.0

ROOT = Path(__file__).resolve().parents[2]
IMG = ROOT / "Doc" / "images"

# Crops of the reconstructed maps (SwordShield, night capture): the shield boss.
MAP_CROP = (195, 340, 690, 590)                          # x0, y0, x1, y1 in the 1024 atlas
MAPS = (("albedo", "results/sword/albedo_night.png"),
        ("metallic", "results/sword/metallic_night.png"),
        ("roughness", "results/sword/roughness_night.png"))
PAIR = (("original render", "results/sword/grid/night_v0_orig.png"),
        ("re-rendered with the recovered materials", "results/sword/grid/night_v0_rerender.png"))


# ── helpers ───────────────────────────────────────────────────────────────────

def card(ax, x, y, w, h, fc=C_CARD, ec=C_LINE, lw=1.2, r=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=1))


def arrow(ax, x0, y0, x1, y1, col=C_MUTED, lw=2.2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=22,
                                 color=col, lw=lw, zorder=3))


def image(ax, path, x, y, w, h, crop=None):
    im = Image.open(path).convert("RGB")
    if crop is not None:
        im = im.crop(crop)
    ax.imshow(np.asarray(im), extent=(x, x + w, y, y + h), interpolation="lanczos",
              zorder=2, aspect="auto")
    ax.add_patch(plt.Rectangle((x, y), w, h, fc="none", ec=C_LINE, lw=1.0, zorder=3))


def wrap_text(ax, x, y, lines, size, col=C_INK, dy=None, ha="left", weight="normal"):
    dy = dy or size * 0.24
    for i, line in enumerate(lines):
        ax.text(x, y - i * dy, line, fontsize=size, color=col, ha=ha, va="top",
                weight=weight)


def finish(fig, ax, x0, y0, x1, y1, out: Path):
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    fig.set_size_inches((x1 - x0) / 10.0, (y1 - y0) / 10.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"  frame {x1 - x0:.1f} x {y1 - y0:.1f} units, ratio {(x1 - x0) / (y1 - y0):.3f}")
    print(f"  + {out}")


def canvas():
    fig = plt.figure(figsize=(26.0, 14.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


# ── conclusions ───────────────────────────────────────────────────────────────

# The three panels of nerf-skybox/skybox_detail.png (2697x515): original, baked, ratio.
SKY_ORIG = (14, 46, 856, 466)
SKY_BAKED = (884, 46, 1722, 466)


def label(ax, x, y, text, size=13.0, col=C_MUTED, ha="center", va="top", **kw):
    ax.text(x, y, text, fontsize=size, color=col, ha=ha, va=va, **kw)


def conclusions(out: Path) -> None:
    """Three rows on a 4:3 slide: the pipeline, the closing check, the three panels."""
    fig, ax = canvas()
    W = 180.0
    X0 = 4.0

    # -- row 1: inputs -> OptiX / NeRF -> the three maps -------------------------------
    TOP, H = 142.0, 46.0
    ymid = TOP - H / 2

    x = X0
    card(ax, x, TOP - H, 38.0, H)
    wrap_text(ax, x + 19.0, TOP - 9.0, ["mesh", "normal map", "HDR photos"],
              17, ha="center", dy=10.5)
    arrow(ax, x + 39.5, ymid, x + 48.5, ymid)

    x = X0 + 50.0
    card(ax, x, TOP - H, 44.0, H, fc="white", ec=C_INK, lw=1.6)
    ax.text(x + 22.0, ymid + 5.0, "OptiX", fontsize=22, color=C_INK, ha="center",
            va="center", weight="bold")
    ax.plot([x + 9, x + 35], [ymid, ymid], color=C_LINE, lw=1.2)
    ax.text(x + 22.0, ymid - 5.0, "NeRF", fontsize=22, color=C_SPEC, ha="center",
            va="center", weight="bold")
    label(ax, x + 22.0, ymid - 11.5, "HDR", size=13, col=C_SPEC)
    arrow(ax, x + 45.5, ymid, x + 54.5, ymid)

    x = X0 + 106.0
    tw, th, gap = 34.0, 14.0, 2.0
    ytop = TOP
    for i, (name, rel) in enumerate(MAPS):
        y = ytop - th - i * (th + gap)
        image(ax, IMG / rel, x, y, tw, th, crop=MAP_CROP)
        ax.text(x + tw + 1.5, y + th / 2, name, fontsize=15, color=C_INK,
                ha="left", va="center")
    maps_cx = x + tw / 2

    # -- row 2: the closing check, original vs re-rendered -----------------------------
    pw = 62.0
    ph = pw * 9 / 16
    R2T = TOP - H - 11.0
    arrow(ax, maps_cx, TOP - H - 1.5, maps_cx, R2T + 5.0)
    xp = X0 + (W - (2 * pw + 6.0)) / 2
    for i, (_, rel) in enumerate(PAIR):
        image(ax, IMG / rel, xp + i * (pw + 6.0), R2T - ph, pw, ph)
    label(ax, xp + pw / 2, R2T + 1.0, "original", size=14, va="bottom")
    label(ax, xp + pw + 6.0 + pw / 2, R2T + 1.0, "re-rendered", size=14, va="bottom",
          col=C_INK)
    ax.text(xp + pw + 3.0, R2T - ph / 2, "≈", fontsize=30, color=C_INK, ha="center",
            va="center")

    # -- row 3: three panels of evidence, images only ----------------------------------
    PT = R2T - ph - 7.0
    gap = 4.0
    pw3 = (W - 2 * gap) / 3.0
    iw = (pw3 - 2.0 - 1.5) / 2
    ih = iw * 9 / 16
    PH = ih + 8.0

    def panel(k, caption, col=C_MUTED):
        x0 = X0 + k * (pw3 + gap)
        card(ax, x0, PT - PH, pw3, PH)
        label(ax, x0 + pw3 / 2, PT - PH + 1.2, caption, size=13.5, col=col, va="bottom")
        return x0, PT - 1.5 - ih

    x0, y0 = panel(0, "original  ≈  re-rendered", C_INK)
    image(ax, IMG / "results/interior/grid/studio_v0_orig.png", x0 + 1.0, y0, iw, ih)
    image(ax, IMG / "results/interior/grid/studio_v0_rerender.png",
          x0 + 1.0 + iw + 1.5, y0, iw, ih)

    x0, y0 = panel(1, "skybox: original  /  HDR NeRF", C_SPEC)
    image(ax, IMG / "nerf-skybox/skybox_detail.png", x0 + 1.0, y0, iw, ih, crop=SKY_ORIG)
    image(ax, IMG / "nerf-skybox/skybox_detail.png", x0 + 1.0 + iw + 1.5, y0, iw, ih,
          crop=SKY_BAKED)

    x0, y0 = panel(2, "studio  /  night", C_DIFF)
    band = (0, 128, 2048, 128 + 1152 // 2)
    image(ax, IMG / "lighting/skybox_studio.png", x0 + 1.0, y0, iw, ih, crop=band)
    image(ax, IMG / "lighting/skybox_night.png", x0 + 1.0 + iw + 1.5, y0, iw, ih,
          crop=band)

    finish(fig, ax, X0 - MARGIN, PT - PH - MARGIN, X0 + W + MARGIN, TOP + MARGIN, out)


# ── future work ───────────────────────────────────────────────────────────────

def glyph_bars(ax, x, y, w, h):
    """Three bars, ours highlighted: the comparison still to be made."""
    vals = (0.55, 0.8, 0.68)
    cols = (C_MUTED, C_SPEC, C_MUTED)
    names = ("[A]", "ours", "[B]")
    bw = w / 5.0
    for i, (v, c, nm) in enumerate(zip(vals, cols, names)):
        xi = x + w * 0.08 + i * bw * 1.35
        ax.add_patch(plt.Rectangle((xi, y), bw, h * v, fc=c, ec="none", zorder=2,
                                   alpha=0.9 if c == C_SPEC else 0.5))
        ax.text(xi + bw / 2, y - 1.0, nm, fontsize=12, color=c, ha="center", va="top")
    ax.plot([x, x + w * 0.85], [y, y], color=C_INK, lw=1.2)


def glyph_sharpen(ax, x, y, w, h):
    """A blurred tile becomes a sharp one: the radiance source is the lever."""
    n = 96
    u = np.linspace(0, 1, n)
    U, V = np.meshgrid(u, u)
    sharp = (((U * 6).astype(int) + (V * 6).astype(int)) % 2).astype(float)
    k = 15
    blur = sharp.copy()
    for _ in range(2):
        pad = np.pad(blur, k // 2, mode="edge")
        blur = sum(pad[i:i + n, j:j + n] for i in range(k) for j in range(k)) / (k * k)
    tw = min(h, w * 0.4)
    for arr, xi in ((blur, x), (sharp, x + w - tw)):
        ax.imshow(arr, extent=(xi, xi + tw, y, y + tw), cmap="Blues", vmin=-0.3, vmax=1.4,
                  interpolation="bilinear", zorder=2, aspect="auto")
        ax.add_patch(plt.Rectangle((xi, y), tw, tw, fc="none", ec=C_LINE, lw=1.0, zorder=3))
    arrow(ax, x + tw + 1.5, y + tw / 2, x + w - tw - 1.5, y + tw / 2, col=C_SPEC, lw=2.6)
    ax.text(x + w / 2, y + tw / 2 + 2.0, "NeRF", fontsize=13, color=C_SPEC, ha="center",
            va="bottom", weight="bold")


def _sphere(ax, cx, cy, d, f0, fresnel):
    """A mirror ball under a sky/ground environment. `f0` is the per-channel reflectance
    at normal incidence; with `fresnel` the Schlick term lifts it towards white at the
    rim, without it the reflectance is the same constant everywhere."""
    n = 240
    u = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(u, -u)
    R2 = X ** 2 + Y ** 2
    inside = R2 <= 1.0
    Z = np.sqrt(np.clip(1.0 - R2, 0.0, 1.0))
    f0 = np.asarray(f0, dtype=float)
    if fresnel:
        F = f0[None, None, :] + (1.0 - f0)[None, None, :] * (1.0 - Z[..., None]) ** 5
    else:
        F = np.broadcast_to(f0[None, None, :], (n, n, 3))
    ry = 2.0 * Z * Y                                  # y of reflect(v, n) for v = +z
    sky = np.clip(0.5 + 0.5 * ry, 0, 1) ** 3.0
    env = (0.06 + 0.94 * sky)[..., None]
    rgb = np.clip(F * env, 0, 1) ** (1 / 2.2)
    rgba = np.concatenate([rgb, inside[..., None].astype(float)], axis=-1)
    ax.imshow(rgba, extent=(cx - d / 2, cx + d / 2, cy - d / 2, cy + d / 2),
              interpolation="bilinear", zorder=2, aspect="auto")


def glyph_fresnel(ax, x, y, w, h):
    """The current model (one constant specular fraction, grey) becomes a per-channel
    Fresnel one (copper tint, bright rim). No words: two balls and an arrow."""
    d = min(h, w * 0.38)
    cy = y + h / 2
    _sphere(ax, x + d / 2, cy, d, (0.45, 0.45, 0.45), fresnel=False)
    _sphere(ax, x + w - d / 2, cy, d, (0.95, 0.64, 0.54), fresnel=True)
    arrow(ax, x + d + 2.0, cy, x + w - d - 2.0, cy, col=C_DIFF, lw=2.6)


def glyph_residual(ax, x, y, w, h):
    """The residual over the aperture grid, and the minimum between two candidates."""
    xs = np.array([0.0, 0.14, 0.28, 0.42, 0.56, 0.70, 0.84, 1.0])
    xmin = 0.47

    def f(u):
        return np.clip(0.15 + 0.85 * (u - xmin) ** 2 / 0.3, 0, 1)

    rs = f(xs)
    ax.plot([x, x + w], [y, y], color=C_INK, lw=1.2)
    u = np.linspace(0, 1, 200)
    ax.plot(x + w * u, y + h * f(u), color=C_LINE, lw=1.6, zorder=2)
    ax.scatter(x + w * xs, y + h * rs, s=40, color=C_SPEC, zorder=3)
    k = int(np.argmin(rs))
    ax.scatter([x + w * xs[k]], [y + h * rs[k]], s=90, color=C_DIFF, zorder=4)
    ax.plot([x + w * xmin, x + w * xmin], [y, y + h * 0.15], color=C_DIFF, lw=1.8,
            ls="--", zorder=3)
    ax.text(x + w * xmin, y - 1.0, "r", fontsize=13, color=C_DIFF, ha="center", va="top")
    ax.text(x + w + 1.0, y, "aperture", fontsize=12, color=C_MUTED, ha="left", va="center")


def future_work(out: Path) -> None:
    fig, ax = canvas()

    cards = (
        (0, 0, C_INK, glyph_bars, "Broader evaluation"),
        (1, 0, C_SPEC, glyph_sharpen, "Better radiance source"),
        (0, 1, C_DIFF, glyph_fresnel, "Better PBR model"),
        (1, 1, C_INDIR, glyph_residual, "Roughness GGX"),
    )
    CW, CH, GX, GY = 70.0, 38.0, 5.0, 5.0
    X0, Y1 = 4.0, 140.0
    for cx, cy, col, glyph, head in cards:
        x = X0 + cx * (CW + GX)
        ytop = Y1 - cy * (CH + GY)
        card(ax, x, ytop - CH, CW, CH)
        ax.add_patch(plt.Rectangle((x, ytop - 1.0), CW, 1.0, fc=col, ec="none", zorder=2))
        ax.text(x + CW / 2, ytop - 4.0, head, fontsize=17, color=col, ha="center",
                va="top", weight="bold")
        glyph(ax, x + 13.0, ytop - CH + 7.5, 40.0, 21.0)

    finish(fig, ax, X0 - MARGIN, Y1 - 2 * CH - GY - MARGIN, X0 + 2 * CW + GX + MARGIN,
           Y1 + MARGIN, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="../../PresentationImages")
    a = ap.parse_args()
    out = Path(a.out)
    conclusions(out / "conclusions.png")
    future_work(out / "future_work.png")


if __name__ == "__main__":
    main()
