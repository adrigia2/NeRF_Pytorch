#!/usr/bin/env python
"""make_lighting_figure.py -- Le due condizioni di luce del capitolo Results.

    python make_lighting_figure.py --out ../Doc/images/lighting

Scrive quattro PNG:

  skybox_studio.png / skybox_night.png     le due environment map, tonemappate
  heat_studio.png   / heat_night.png       la norma ||c|| di ogni pixel, in falsi colori

Due convenzioni, ognuna scelta per una ragione precisa.

**I due pannelli tonemappati NON condividono l'esposizione.**  Sono due ambienti diversi,
non due versioni della stessa scena: una esposizione comune renderebbe la notturna un
rettangolo nero e non direbbe nulla su come e' fatta.  Ognuna porta la propria mediana al
livello di riferimento, esattamente come fa il resto del capitolo.

**Le due heatmap la condividono, ed e' logaritmica.**  Qui il soggetto e' il confronto: e'
guardando la stessa scala che si vede che la notturna e' complessivamente piu' fioca e
molto piu' concentrata.  La scala e' logaritmica perche' la dinamica di questi envmap
copre parecchi ordini di grandezza; in lineare la heatmap sarebbe nera con qualche punto
bianco, cioe' l'informazione che serve, dove sta l'energia, andrebbe persa proprio dove e'
interessante.

Il valore mappato e' la norma euclidea del pixel, non la luminanza: qui interessa quanta
radianza arriva da quella direzione, non come la percepirebbe un occhio.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).parent))

from make_skybox_figure import LUMA_COEFF, block_mean, load_exr, tonemap  # noqa: E402

plt.rcParams.update({"font.size": 13})

DPI = 190
DOWNSAMPLE = 2          # le sorgenti sono 4k equirettangolari
KEY = 0.5               # livello a cui portare la mediana prima del Reinhard
FLOOR_PCTL = 1.0        # percentile che fissa il fondo della scala condivisa
CMAP = "inferno"

HDR_DIR = Path("C:/Users/adria/Documents/GitHub/Tesi/OptixProjectCMake/Scenes/"
               "SwordShield Thesis/Blender/assets/hdrs")
MAPS = [("studio", HDR_DIR / "wooden_studio_13_4k.exr"),
        ("night",  HDR_DIR / "cobblestone_street_night_4k.exr")]


def own_exposure(img: np.ndarray) -> float:
    """Esposizione che porta la mediana della luminanza a KEY.  La mediana e non il
    massimo: in una envmap il picco sta sulle sorgenti e detterebbe da solo il tonemap."""
    med = max(float(np.median((img * LUMA_COEFF).sum(-1))), 1e-6)
    return KEY / med


def heat_png(norm: np.ndarray, vmin: float, vmax: float, out: Path, label: str) -> None:
    h, w = norm.shape
    fig, ax = plt.subplots(figsize=(7.2, 7.2 * h / w + 0.9))
    im = ax.imshow(np.maximum(norm, vmin), cmap=CMAP,
                   norm=LogNorm(vmin=vmin, vmax=vmax), interpolation="nearest")
    ax.axis("off")
    cb = fig.colorbar(im, ax=ax, fraction=0.031, pad=0.02, orientation="horizontal")
    cb.set_label(label)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  + {out.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="../Doc/images/lighting")
    ap.add_argument("--downsample", type=int, default=DOWNSAMPLE)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    imgs, norms = {}, {}
    for name, p in MAPS:
        if not p.exists():
            raise SystemExit(f"✗ non trovata: {p}")
        a = block_mean(load_exr(p), args.downsample)
        imgs[name] = a
        norms[name] = np.linalg.norm(a, axis=-1)
        n = norms[name]
        print(f"{name:7s} {a.shape[1]}x{a.shape[0]}  ||c||: p50={np.median(n):.4f}  "
              f"p99.9={np.percentile(n, 99.9):9.3f}  max={n.max():10.3f}  "
              f"media={n.mean():.4f}")

    # Scala condivisa.  Il tetto e' il massimo fra le due, perche' il picco della
    # notturna e' proprio il fenomeno da mostrare; il fondo e' un percentile basso e non
    # un numero fisso di decadi sotto il tetto.  Con le decadi fisse quel picco
    # (~7·10^4, duecento volte il massimo dello studio) trascinerebbe il fondo scala
    # sopra la mediana di entrambe le mappe, appiattendo tutto il resto in un colore
    # solo: la barra coprirebbe un intervallo in cui i dati quasi non stanno.
    allv = np.concatenate([norms[n].ravel() for n, _ in MAPS])
    vmax = float(allv.max())
    vmin = max(float(np.percentile(allv, FLOOR_PCTL)), vmax * 1e-9)
    print(f"\nscala condivisa delle heatmap: [{vmin:.3e}, {vmax:.3e}]  "
          f"({np.log10(vmax / vmin):.1f} decadi, logaritmica, "
          f"fondo al p{FLOOR_PCTL} congiunto)")

    for name, _ in MAPS:
        save = out / f"skybox_{name}.png"
        expo = own_exposure(imgs[name])
        plt.imsave(save, np.clip(tonemap(imgs[name], expo), 0.0, 1.0))
        print(f"  + {save.name}  (esposizione propria {expo:.4g})")
        heat_png(norms[name], vmin, vmax, out / f"heat_{name}.png",
                 r"$\|\mathbf{c}\|$  (log scale, shared)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
