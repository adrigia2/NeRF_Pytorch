#!/usr/bin/env python
"""make_uv_unwrap_figure.py -- I due pannelli della figura 3.3.

    python make_uv_unwrap_figure.py <run_dir> --world C:/.../positional_image.exr \
        --out ../Doc/images/ium

Scrive due PNG:

  uv_unwrap_world.png   la mesh disegnata dalle posizioni dei suoi vertici in world space
  uv_unwrap_uv.png      la stessa mesh alle sue coordinate UV, cioe' la ium_positions
                        della run, che e' la posizione world scritta nello spazio texture

**La mappatura e' l'identita', non una normalizzazione.**  I due pannelli mostrano una
coordinata, non una radianza: riscalare per canale sull'estensione della geometria, come
fa `position_rgb` in make_depth_figure.py, produrrebbe due immagini con due fattori
diversi, e il lettore le confronterebbe credendo di vedere lo stesso colore per lo stesso
punto.  Qui il valore va sul pixel com'e': i negativi vengono portati a zero e quello che
supera 1 satura, che e' l'unica cosa che un PNG a 8 bit possa fare.  Nessuna gamma, per
la stessa ragione per cui non ce n'e' nella figura dei layer di depth.

Conseguenza da tenere presente leggendo la figura: le parti della scena a coordinata
negativa risultano nere in quel canale, e quelle oltre 1 saturano.  E' voluto: dice dove
sta la geometria rispetto all'origine, cosa che una versione riscalata nasconde.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import _paths  # noqa: F401

from make_depth_figure import position_rgb, save_png   # noqa: E402
from make_skybox_figure import load_exr                # noqa: E402

# La mappatura vive in `position_rgb` (make_depth_figure.py) e non e' duplicata qui:
# e' la stessa convenzione di tutte le mappe di posizione della tesi, e averne due
# implementazioni vorrebbe dire poterle far divergere senza accorgersene.


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="run da cui prendere ium/ium_positions.exr")
    ap.add_argument("--world", required=True,
                    help="EXR della mesh disegnata dalle posizioni dei vertici")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    run, out = Path(args.run_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{run.name} → {out.resolve()}")

    print("world space:")
    world = load_exr(Path(args.world))
    save_png(position_rgb(world), out / "uv_unwrap_world.png")

    print("UV space:")
    ium = load_exr(run / "ium" / "ium_positions.exr")
    mask = load_exr(run / "ium" / "ium_masks.exr")[..., 0] > 0.5
    save_png(position_rgb(ium, mask), out / "uv_unwrap_uv.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
