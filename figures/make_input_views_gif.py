#!/usr/bin/env python
"""make_input_views_gif.py -- the input captures of a scene, as a looping GIF.

    python make_input_views_gif.py --run <run dir> --out ../../PresentationImages
    python make_input_views_gif.py --run <run dir> --out ... --frames 8 --duration 500

Writes `<name>.gif`: a handful of the frames the pipeline is given as input, cycling.  It
is the "many views of one scene" panel of the Goal slide, which a stack of stills can only
suggest.

WHICH FRAMES.  Not the first N, and not every Nth: the capture shells are spirals, so
consecutive indices jump about 105 degrees in azimuth while the elevation drifts steadily
from above the object to below it.  Taking a slice of the index would give a set biased to
one elevation, and taking them in index order would strobe.  So the frames are filtered
first -- the subject fully inside the frame with a margin, covering a sane fraction of it,
and not from a near-polar view that reads as a floor plan -- and then spread evenly in
AZIMUTH, so that the loop walks around the object instead of hopping.  The filter runs on
the run's own per-frame masks, so "the subject is inside the frame" is measured, not
assumed.

EXPOSURE.  One exposure for the whole GIF, otherwise the loop flickers as each frame
re-normalises itself.  And it is measured on the SUBJECT, not on the frame: in the studio
scene the median luminance of a whole frame is 0.767 while the median over the sword and
shield is 0.021, thirty-seven times darker, because most of the frame is the lit studio
behind them.  Exposing on the frame is what the Results figures do and it is right there,
where the frame is the subject; here it puts the object at the bottom of the curve and the
GIF comes out nearly black.  The tonemap itself is the shared one,
`make_skybox_figure.tonemap`: Reinhard, then gamma 2.2.

PALETTE.  A GIF holds 256 colours.  Quantising each frame on its own gives each one a
different palette, and the loop then shimmers on the flat backgrounds even where nothing
moves.  One palette is built from all the frames together and every frame is mapped
through it, so the only thing that changes between frames is what actually changed.
Dithering is left on: the studio backgrounds are smooth gradients, and without it they
band into visible steps.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import _paths  # noqa: F401,E402

from make_skybox_figure import LUMA_COEFF, load_exr, tonemap    # noqa: E402

KEY = 0.15            # the subject's median luminance is brought here before the Reinhard
FRAMES = 12           # frames in the loop
WIDTH = 720           # width of the GIF, in pixels
DURATION = 420        # milliseconds a frame is held
BORDER = 40           # pixels the subject has to keep clear of the frame edge
AREA = (0.04, 0.11)   # fraction of the frame the subject may cover
ELEV = (-35.0, 45.0)  # degrees; outside this the view reads as a floor plan or a ceiling
BLOWN_WARN = 0.02     # fraction of clipped pixels worth a warning


def load_mask(path: Path) -> np.ndarray:
    a = np.asarray(Image.open(path))
    return (a if a.ndim == 2 else a[..., 0]) > 127


def survey(run: Path, meta: dict) -> list[dict]:
    """Every frame, with where its subject sits and where its camera does."""
    rows = []
    for i, f in enumerate(meta["frames"]):
        m = load_mask(run / f["mask_path"])
        ys, xs = np.where(m)
        if len(ys) == 0:
            continue
        p = np.array(f["transform_matrix"], dtype=np.float64)[:3, 3]
        h, w = m.shape
        rows.append(dict(
            i=i, frame=f, area=len(ys) / m.size,
            inside=(xs.min() > BORDER and xs.max() < w - BORDER
                    and ys.min() > BORDER and ys.max() < h - BORDER),
            az=float(np.degrees(np.arctan2(p[1], p[0])) % 360.0),
            el=float(np.degrees(np.arcsin(p[2] / np.linalg.norm(p))))))
    return rows


def choose(rows: list[dict], n: int, area, elev) -> list[dict]:
    """The frames of the loop: filtered, then spread evenly in azimuth."""
    ok = [r for r in rows
          if r["inside"] and area[0] <= r["area"] <= area[1]
          and elev[0] <= r["el"] <= elev[1]]
    if len(ok) < n:
        raise SystemExit(
            f"only {len(ok)} of {len(rows)} frames pass the filter (subject inside with "
            f"{BORDER} px to spare, covering {100 * area[0]:.0f}-{100 * area[1]:.0f}% of "
            f"the frame, elevation in {elev}), fewer than the {n} asked for: widen the "
            "ranges or ask for fewer frames")
    ok.sort(key=lambda r: r["az"])
    return [ok[int(round(k))] for k in np.linspace(0, len(ok) - 1, n)]


def exposure(run: Path, picked: list[dict], key: float) -> float:
    """One exposure, from the median luminance of the subject across the chosen frames."""
    lum = []
    for r in picked:
        img = load_exr(run / r["frame"]["file_path"])
        lum.append((img @ LUMA_COEFF)[load_mask(run / r["frame"]["mask_path"])])
    med = float(np.median(np.concatenate(lum)))
    print(f"  subject median luminance {med:.4f} over {len(picked)} frames "
          f"-> exposure {key / med:.2f}")
    return key / med


def render(run: Path, picked: list[dict], expo: float, width: int) -> list[Image.Image]:
    out = []
    for r in picked:
        img = load_exr(run / r["frame"]["file_path"])
        tm = np.clip(tonemap(img, expo), 0.0, 1.0)
        blown = float((tm > 0.99).mean())
        if blown > BLOWN_WARN:
            print(f"    ! frame {r['i']}: {100 * blown:.1f}% of the pixels clip")
        im = Image.fromarray((tm * 255.0 + 0.5).astype(np.uint8))
        h = int(round(width * im.height / im.width))
        out.append(im.resize((width, h), Image.LANCZOS))
        print(f"    frame {r['i']:>3}  az {r['az']:6.1f}  el {r['el']:6.1f}  "
              f"subject {100 * r['area']:5.2f}%")
    return out


def one_palette(frames: list[Image.Image]) -> Image.Image:
    """A single palette for the whole loop.  See the module docstring: per-frame palettes
    make the still parts of the picture shimmer."""
    tile = int(np.ceil(np.sqrt(len(frames))))
    small = [f.resize((f.width // 4, f.height // 4), Image.LANCZOS) for f in frames]
    w, h = small[0].size
    sheet = Image.new("RGB", (tile * w, tile * h))
    for k, f in enumerate(small):
        sheet.paste(f, ((k % tile) * w, (k // tile) * h))
    return sheet.convert("P", palette=Image.ADAPTIVE, colors=256)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True,
                    help="run folder (images/, mask/, transforms_extended.json)")
    ap.add_argument("--out", required=True, help="destination folder")
    ap.add_argument("--name", default=None, help="output stem (default: the run's name)")
    ap.add_argument("--frames", type=int, default=FRAMES)
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--duration", type=int, default=DURATION,
                    help="milliseconds a frame is held")
    ap.add_argument("--area", type=float, nargs=2, default=list(AREA),
                    metavar=("MIN", "MAX"), help="fraction of the frame the subject covers")
    ap.add_argument("--elev", type=float, nargs=2, default=list(ELEV),
                    metavar=("MIN", "MAX"), help="camera elevation, in degrees")
    ap.add_argument("--key", type=float, default=KEY,
                    help="the subject's median luminance is brought here before the "
                         "Reinhard")
    a = ap.parse_args(argv)

    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = a.name or run.name
    meta = json.loads((run / "transforms_extended.json").read_text())
    print(f"{run.name} -> {(out / (name + '.gif')).resolve()}")

    rows = survey(run, meta)
    picked = choose(rows, a.frames, tuple(a.area), tuple(a.elev))
    expo = exposure(run, picked, a.key)
    frames = render(run, picked, expo, a.width)

    pal = one_palette(frames)
    quant = [f.quantize(palette=pal, dither=Image.FLOYDSTEINBERG) for f in frames]
    path = out / f"{name}.gif"
    quant[0].save(path, save_all=True, append_images=quant[1:], loop=0,
                  duration=a.duration, optimize=True, disposal=1)
    kb = path.stat().st_size / 1024
    print(f"  + {path}  ({frames[0].width}x{frames[0].height}, {len(frames)} frames, "
          f"{a.duration} ms each, {len(frames) * a.duration / 1000:.1f} s loop, "
          f"{kb:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
