"""Per-texel camera count of a run, as a full-resolution PNG (1 pixel = 1 texel).

Reads sources/{source}/pbr/n_views.exr — the cameras the PBR fit actually used for each
texel (refined visibility ∧ cones valid) — and paints it with a single-hue sequential
ramp, light (2 cameras) to dark (--cmax). Texels the fit cannot solve are flagged apart:
0 cameras red, 1 camera orange. Outside the IUM mask (or the ROI of a sandbox) the page
colour shows through.

Pass the same --cmax to two runs to compare them on one scale.

Usage:
    python make_view_count_map.py <scene_or_sandbox_dir> [--roi X0 Y0 W H]
                                  [--cmax N] [--title TEXT] [--out PNG] [--source gt]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import _paths  # noqa: F401
from pbr_solver import _read_exr  # noqa: E402

# Sequential blue ramp 100 → 700 and the status colours of the dataviz palette.
RAMP = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5',
        '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
SURFACE = (0xfc, 0xfc, 0xfb)
CRITICAL = (0xd0, 0x3b, 0x3b)
SERIOUS = (0xec, 0x83, 0x5a)
INK, INK2 = (0x0b, 0x0b, 0x0b), (0x52, 0x51, 0x4e)
_STOPS = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in RAMP], dtype=np.float64)


def ramp(t: np.ndarray) -> np.ndarray:
    """t in [0, 1] → RGB along the ramp (linear between the stops)."""
    x = np.clip(t, 0.0, 1.0) * (len(_STOPS) - 1)
    i = np.minimum(x.astype(int), len(_STOPS) - 2)
    f = (x - i)[..., None]
    return _STOPS[i] * (1 - f) + _STOPS[i + 1] * f


def _font(size: int, bold: bool = False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    try:
        return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)
    except OSError:
        return ImageFont.load_default()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("scene_dir")
    ap.add_argument("--roi", nargs=4, type=int, metavar=("X0", "Y0", "W", "H"))
    ap.add_argument("--cmax", type=int, default=None, help="top of the scale (default: data max)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", default=None, help="default: <scene_dir>/view_count.png")
    ap.add_argument("--source", default="gt")
    args = ap.parse_args()

    root = Path(args.scene_dir)
    mk = _read_exr(root / "ium" / "ium_masks.exr")
    mk = (mk[..., 0] if mk.ndim == 3 else mk) > 0.5
    nv = np.rint(_read_exr(root / "sources" / args.source / "pbr" / "n_views.exr")).astype(np.int32)
    if args.roi:
        x0, y0, w, h = args.roi
        mk, nv = mk[y0:y0 + h, x0:x0 + w], nv[y0:y0 + h, x0:x0 + w]
    H, W = nv.shape
    cmax = args.cmax or int(nv[mk].max())
    n0, n1 = int((mk & (nv == 0)).sum()), int((mk & (nv == 1)).sum())
    print(f"texels {int(mk.sum())}, max cameras {int(nv[mk].max())}, scale top {cmax}, "
          f"0 cameras {n0}, 1 camera {n1}")

    img = np.empty((H, W, 3), dtype=np.float64)
    img[:] = SURFACE
    seen = mk & (nv >= 2)
    img[seen] = ramp((nv[seen] - 2) / max(cmax - 2, 1))
    img[mk & (nv == 0)] = CRITICAL
    img[mk & (nv == 1)] = SERIOUS
    img = img.astype(np.uint8)

    # Title band and legend scale with the map, which stays at 1 texel = 1 pixel.
    u = max(H, 1024) / 4096
    TOP, RIGHT = int(190 * u) + 20, int(820 * u) + 180
    canvas = Image.new("RGB", (W + RIGHT, H + TOP), SURFACE)
    canvas.paste(Image.fromarray(img), (0, TOP))
    d = ImageDraw.Draw(canvas)
    f_t, f_s, f_l = _font(int(64 * u) + 8, True), _font(int(44 * u) + 6), _font(int(48 * u) + 6)
    title = args.title or f"{root.name}: da quante camere è visto ogni texel"
    d.text((int(40 * u), int(30 * u)), title, font=f_t, fill=INK)
    roi_txt = f" · ROI {args.roi}" if args.roi else ""
    d.text((int(40 * u), int(115 * u) + 4),
           f"n_views del fit (visibilità e coni validi) · 1 pixel = 1 texel{roi_txt} · "
           f"massimo {int(nv[mk].max())}", font=f_s, fill=INK2)

    bw, bh = int(110 * u) + 20, int(min(2600 * u, H * 0.62))
    x0, y0 = W + int(90 * u) + 20, TOP + int(120 * u)
    d.text((x0 - 10, TOP + int(30 * u)), "camere", font=f_l, fill=INK)
    col = ramp(np.linspace(1, 0, bh))[:, None, :].repeat(bw, axis=1).astype(np.uint8)
    canvas.paste(Image.fromarray(col), (x0, y0))
    step = 2 if cmax <= 30 else 5
    for c in range(2, cmax + 1):
        if c in (2, cmax) or c % step == 0:
            y = y0 + round((1 - (c - 2) / max(cmax - 2, 1)) * (bh - 1))
            d.line([(x0 + bw, y), (x0 + bw + int(22 * u) + 4, y)], fill=INK2, width=max(1, int(4 * u)))
            d.text((x0 + bw + int(34 * u) + 8, y - int(26 * u) - 4), str(c), font=f_l, fill=INK)
    ys = y0 + bh + int(120 * u)
    for colr, lab in [(CRITICAL, "0 camere"), (SERIOUS, "1 camera")]:
        d.rectangle([x0, ys, x0 + bw, ys + int(80 * u) + 10], fill=colr)
        d.text((x0 + bw + int(30 * u) + 6, ys + int(12 * u)), lab, font=f_l, fill=INK)
        ys += int(120 * u) + 16
    d.text((x0, ys + 10), "non risolvibile", font=f_s, fill=INK2)
    d.text((x0, ys + int(64 * u) + 12), "(servono ≥ 2 camere)", font=f_s, fill=INK2)

    out = Path(args.out) if args.out else root / "view_count.png"
    canvas.save(out, optimize=True)
    print(f"✓ {out}  {canvas.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
