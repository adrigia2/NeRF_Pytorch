#!/usr/bin/env python
"""make_depth_figure.py -- I layer geometrici di una run, come PNG per la tesi.

    python make_depth_figure.py <run_dir> --camera render_Camera_Shell21_38 --out DIR
    python make_depth_figure.py <run_dir> --ium --out DIR

Nel primo modo scrive i tre pannelli per camera del depth pass: depth.png (falsi colori),
position.png, mask.png.  Nel secondo i due pannelli in spazio texture del pass IUM:
ium_position.png e ium_mask.png.

Due scelte non sono di gusto:

  1. La normalizzazione del depth usa la MASCHERA, non una soglia sul valore.  Sullo
     sfondo il file porta 1e20, il valore di miss del tracer: un min/max sull'immagine
     intera manderebbe tutto il primo piano nello stesso colore.

  2. Lo sfondo del pannello depth non e' un valore di profondita' e non deve sembrarlo.
     Va reso con un grigio neutro, distinguibile dai due estremi della rampa e dal bianco
     della pagina, invece di essere schiacciato su un capo della colormap.

La colormap e' viridis: percettivamente uniforme, quindi una differenza di colore uguale
corrisponde a una differenza di profondita' uguale, che su una mappa di profondita' e'
esattamente cio' che si vuole leggere.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sys.path.insert(0, str(Path(__file__).parent))

from make_skybox_figure import load_exr

# Grigio dello sfondo nel pannello depth: piu' chiaro dell'estremo giallo di viridis
# sarebbe illeggibile, piu' scuro si confonderebbe con l'estremo viola.
BG_GREY = 0.75


def save_png(rgb: np.ndarray, out: Path) -> None:
    plt.imsave(out, np.clip(rgb, 0.0, 1.0))
    print(f"  + {out}  ({rgb.shape[1]}x{rgb.shape[0]})")


def content_box(mask: np.ndarray, margin: float) -> tuple[slice, slice]:
    """Riquadro del primo piano piu' un margine, UNO SOLO per tutti i pannelli.

    Stampati a un terzo di riga, i tre pannelli a pieno fotogramma mostrano il soggetto
    su un quarto dell'altezza.  Il ritaglio dev'essere identico nei tre, altrimenti non
    sono piu' confrontabili pixel a pixel, che e' il motivo per cui stanno accanto.
    """
    ys, xs = np.where(mask)
    h, w = mask.shape
    my = int(margin * (ys.max() - ys.min() + 1))
    mx = int(margin * (xs.max() - xs.min() + 1))
    return (slice(max(ys.min() - my, 0), min(ys.max() + 1 + my, h)),
            slice(max(xs.min() - mx, 0), min(xs.max() + 1 + mx, w)))


def position_rgb(pos: np.ndarray, mask: "np.ndarray | None" = None) -> np.ndarray:
    """Posizione mappata su RGB con l'IDENTITA', sfondo nero.

    Nessuna riscalatura e nessuna gamma: il valore finisce sul pixel com'e', i negativi
    vengono portati a zero e quello che supera uno satura.  Prima questa funzione
    riscalava per canale sull'estensione della geometria: rendeva ogni pannello
    leggibile per conto suo, ma con un fattore diverso da quello di ogni altro, per cui
    due figure che mostrano la stessa scena non erano confrontabili e lo stesso colore
    non significava lo stesso punto.  Con l'identita' il colore e' la coordinata, che e'
    l'unica lettura che serva a chi guarda una mappa di posizioni.

    `mask` e' opzionale: senza, si mappa tutto il fotogramma (il pannello in world space
    non ha una maschera da applicare).  Stampa gli estremi e quanto viene tagliato:
    sono i numeri che la didascalia deve riportare.
    """
    sel = mask if mask is not None else np.ones(pos.shape[:2], bool)
    lo, hi = pos[sel].min(axis=0), pos[sel].max(axis=0)
    print("  position " + "  ".join(f"{a}[{lo[i]:.2f}, {hi[i]:.2f}]"
                                    for i, a in enumerate("xyz")))
    print(f"  clamp: portati a 0 {100.0 * (pos[sel] < 0).mean():.2f}%, "
          f"saturati a 1 {100.0 * (pos[sel] > 1).mean():.2f}%")
    rgb = np.clip(pos, 0.0, 1.0).astype(np.float32)
    if mask is not None:
        rgb[~mask] = 0.0
    return rgb


def normal_rgb(nrm: np.ndarray, mask: "np.ndarray | None" = None) -> np.ndarray:
    """Normale mappata su RGB con la codifica delle normal map, sfondo nero.

    Porta $[-1, 1]$ in $[0, 1]$ con 0.5 + 0.5*n, che e' come le normal map sono scritte e
    come questa pipeline legge quella esterna (`external_normal_range="0_1"`), quindi il
    pannello e' confrontabile con la mappa che lo sostituisce.  NON si usa il clamp delle
    posizioni: azzererebbe ogni componente negativa e farebbe sparire tutte le facce
    rivolte verso -x, -y o -z, cioe' meta' della geometria.

    Normalizza prima di codificare.  Il kernel costruisce la normale come prodotto
    vettoriale degli spigoli del triangolo e i consumatori la normalizzano al momento
    dell'uso (deviceProgramsIrradiance.cu lo fa esplicitamente), quindi il buffer non e'
    garantito unitario: senza normalizzare il colore direbbe l'area del triangolo invece
    della direzione.
    """
    n = np.linalg.norm(nrm, axis=-1, keepdims=True)
    unit = np.divide(nrm, n, out=np.zeros_like(nrm), where=n > 1e-8)
    degenerate = float((n[..., 0] <= 1e-8).mean())
    print(f"  normal   |n| in [{n.min():.3f}, {n.max():.3f}], "
          f"degeneri {100.0 * degenerate:.2f}%")
    rgb = np.clip(0.5 + 0.5 * unit, 0.0, 1.0).astype(np.float32)
    if mask is not None:
        rgb[~mask] = 0.0
    return rgb


def ium_panels(run: Path, out: Path) -> int:
    """I layer in spazio texture: posizione per texel e maschera di copertura.

    La normale NON viene scritta: quando la pipeline riceve una normal map esterna, questa
    sovrascrive la normale geometrica dentro il buffer del pass prima del salvataggio, e
    ium_normals.exr contiene quindi la mappa fornita, non quella calcolata.
    """
    paths = {"position": run / "ium" / "ium_positions.exr",
             "mask":     run / "ium" / "ium_masks.exr"}
    for p in paths.values():
        if not p.exists():
            print(f"ERRORE: {p} non esiste")
            return 2

    pos = load_exr(paths["position"])
    mask = load_exr(paths["mask"])[..., 0] > 0.5
    print(f"atlante {mask.shape[1]}x{mask.shape[0]}, copertura {100 * mask.mean():.2f}% "
          f"({mask.sum():,} texel)")

    save_png(position_rgb(pos, mask), out / "ium_position.png")
    save_png(np.repeat(mask[..., None].astype(np.float32), 3, axis=-1),
             out / "ium_mask.png")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="cartella della run (contiene depth/, position/, mask/)")
    ap.add_argument("--camera", default=None, help="nome del frame, senza suffisso")
    ap.add_argument("--out", required=True, help="cartella di destinazione dei PNG")
    ap.add_argument("--ium", action="store_true",
                    help="scrive i layer in spazio texture invece di quelli per camera")
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--margin", type=float, default=0.06,
                    help="margine attorno al primo piano, in frazione del riquadro "
                         "(default 0.06; 0 disattiva il ritaglio)")
    args = ap.parse_args()

    run, out, cam = Path(args.run_dir), Path(args.out), args.camera
    out.mkdir(parents=True, exist_ok=True)

    if args.ium:
        return ium_panels(run, out)
    if not cam:
        print("ERRORE: serve --camera per i layer per camera")
        return 2

    paths = {
        "depth":    run / "depth" / f"{cam}_depth.exr",
        "position": run / "position" / f"{cam}_position.exr",
        "mask":     run / "mask" / f"{cam}_mask.png",
    }
    for k, p in paths.items():
        if not p.exists():
            print(f"ERRORE: {p} non esiste")
            return 2

    depth = load_exr(paths["depth"])[..., 0]
    pos = load_exr(paths["position"])
    mask_raw = mpimg.imread(paths["mask"])
    mask = (mask_raw if mask_raw.ndim == 2 else mask_raw[..., 0]) > 0.5

    print(f"{cam}: {mask.shape[1]}x{mask.shape[0]}, primo piano {100 * mask.mean():.1f}%")

    d0, d1 = float(depth[mask].min()), float(depth[mask].max())
    # Estremi e ritaglio si calcolano sul fotogramma intero: il ritaglio decide solo cosa
    # si vede, non come i valori vengono mappati.
    if args.margin > 0:
        rows, cols = content_box(mask, args.margin)
        print(f"  ritaglio [{cols.start}:{cols.stop}, {rows.start}:{rows.stop}] "
              f"= {cols.stop - cols.start}x{rows.stop - rows.start}")
        depth, pos, mask = depth[rows, cols], pos[rows, cols], mask[rows, cols]

    print(f"  depth  [{d0:.3f}, {d1:.3f}]  mediana {float(np.median(depth[mask])):.3f}")
    norm = np.clip((depth - d0) / max(d1 - d0, 1e-9), 0.0, 1.0)
    rgb = matplotlib.colormaps[args.cmap](norm)[..., :3]
    rgb[~mask] = BG_GREY
    save_png(rgb, out / "depth.png")

    save_png(position_rgb(pos, mask), out / "position.png")

    save_png(np.repeat(mask[..., None].astype(np.float32), 3, axis=-1), out / "mask.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
