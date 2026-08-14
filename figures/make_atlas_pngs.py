#!/usr/bin/env python
"""make_atlas_pngs.py -- Converte i canali di un bake Blender in PNG per la tesi.

    python make_atlas_pngs.py <cartella del bake> --out <cartella> [--channels roughness]

Le convenzioni non sono arbitrarie: riproducono quelle dei PNG gia' presenti in
Doc/images/, misurate contro gli EXR sorgente.

  1. Il base color e' l'unico canale di COLORE e va codificato in sRGB (la curva esatta,
     non gamma 2.2: e' quella che riproduce i file esistenti con scarto 0.0065).  Gli
     altri tre sono DATI, non colore, e vanno scritti lineari: applicare una gamma a una
     roughness significa mostrare un valore che non e' quello che il renderer ha usato.

  2. Il downsample 8192 -> 4096 e' una media di blocchi in spazio lineare.  Un
     sottocampionamento puntuale butterebbe tre quarti del segnale autoriale e aliaserebbe
     il resto, che e' lo stesso motivo per cui la ground truth viene ridotta cosi'.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _paths  # noqa: F401

from make_skybox_figure import block_mean, load_exr

CHANNELS = ("base_color", "metallic", "roughness", "normal")
# Solo il base color e' colore: gli altri sono dati e restano lineari.
SRGB_CHANNELS = {"base_color"}


def srgb(x: np.ndarray) -> np.ndarray:
    """Codifica sRGB (IEC 61966-2-1), non gamma 2.2."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bake_dir", help="cartella con i BakedMaterial_<canale>.exr")
    ap.add_argument("--out", required=True, help="cartella di destinazione dei PNG")
    ap.add_argument("--channels", nargs="+", default=list(CHANNELS),
                    help=f"canali da convertire (default: {' '.join(CHANNELS)})")
    ap.add_argument("--downsample", type=int, default=2,
                    help="media di blocchi (default 2: 8192x8192 -> 4096x4096)")
    args = ap.parse_args()

    bake, out = Path(args.bake_dir), Path(args.out)
    if not bake.is_dir():
        print(f"ERRORE: {bake} non e' una cartella")
        return 2
    out.mkdir(parents=True, exist_ok=True)

    for ch in args.channels:
        src = bake / f"BakedMaterial_{ch}.exr"
        if not src.exists():
            print(f"ERRORE: {src} non esiste")
            return 2
        a = block_mean(load_exr(src), args.downsample)
        rgb = srgb(a) if ch in SRGB_CHANNELS else np.clip(a, 0.0, 1.0)
        dst = out / f"BakedMaterial_{ch}.png"
        plt.imsave(dst, rgb)
        print(f"  + {dst}  {rgb.shape[1]}x{rgb.shape[0]}  "
              f"range [{a.min():.3f}, {a.max():.3f}]  "
              f"{'sRGB' if ch in SRGB_CHANNELS else 'lineare'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
