#!/usr/bin/env python
"""make_skybox_figure.py -- Figure di tesi sulle skybox bakate dai NeRF di uno sweep.

Produce due PNG:

  skybox_grid.png    griglia con l'envmap originale e quella bakata da ogni run,
                     tonemap ed ESPOSIZIONE IDENTICI in ogni pannello
  skybox_detail.png  originale, bakata e mappa log2(baked/GT) per il solo modello scelto

Le `skybox_compare/skybox_heatmap.png` che bake_skyboxes.py produce per ogni run non
servono a questo scopo: clippano a [0,1] senza tonemap ne' gamma, quindi la skybox
appare quasi nera tranne le sorgenti.

    python make_skybox_figure.py <sweep_root> --gt GT.exr --out DIR [--selected RUN]

Tre scelte non sono negoziabili e sono il motivo per cui questo script esiste:

  1. Il downsample avviene in spazio LINEARE e per media di blocchi, prima del tonemap.
     Ridimensionare dopo il tonemap, o con un filtro non conservativo, altera la
     radianza media dei pixel piccoli e brillanti, che sono quelli che portano l'energia.

  2. L'esposizione e' UNA SOLA per tutti i pannelli, derivata dalla GT.  Se ogni
     pannello avesse la sua, una differenza di luminosita' fra i modelli sparirebbe
     dalla figura, che e' esattamente cio' che la figura deve mostrare.

  3. La differenza e' un RAPPORTO in log2, non una sottrazione.  Lo scarto misurato e'
     un fattore quasi uniforme di pochi punti percentuali: in scala lineare sarebbe
     invisibile ovunque tranne che sulle sorgenti.

Lo script stampa anche il rapporto delle medie lineari pesate per angolo solido,
<||baked||> / <||GT||>, che e' il numero che entra nell'integrale di irradianza.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import _paths  # noqa: F401

BAKED_NAME = "skybox_nerf_baked.exr"
# Rec.709, identico a LUMA_COEFF di compare_runs.py
LUMA_COEFF = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
EPS = 1e-6

# Etichette leggibili: il nome della cartella non e' presentabile in una figura
LOSS_LABEL = {"l1": "L1", "mse": "sq.", "relmseraw": "rel. sq.",
              "rel_mse_raw": "rel. sq."}


def load_exr(path: Path) -> np.ndarray:
    """(H, W, 3) float32.  Stesso loader di bake_skyboxes.py, un solo lettore per le
    skybox."""
    from regen_heatmaps import _load_exr_hw3
    return _load_exr_hw3(str(path))


def block_mean(img: np.ndarray, factor: int) -> np.ndarray:
    """Downsample per media di blocchi factor x factor, in spazio lineare.

    Conserva la radianza media, che un resampling per interpolazione non garantisce:
    su un envmap con sorgenti piccole e molto brillanti la differenza non e' cosmetica.

    Il numero di canali si legge dall'array invece di essere fissato a 3: le heatmap di
    make_results_figures passano di qui a canale singolo, e con il 3 cablato la reshape
    falliva.  Sugli RGB il comportamento e' identico.
    """
    if factor <= 1:
        return img
    h, w = img.shape[:2]
    c = img.shape[2] if img.ndim == 3 else 1
    h2, w2 = (h // factor) * factor, (w // factor) * factor
    a = img[:h2, :w2].reshape(h2, w2, c)
    out = a.reshape(h2 // factor, factor, w2 // factor, factor, c).mean(axis=(1, 3))
    return out if img.ndim == 3 else out[..., 0]


def tonemap(x: np.ndarray, exposure: float) -> np.ndarray:
    """Reinhard piu' gamma 2.2.  Stessa formula di _tonemap_srgb in compare_runs.py."""
    y = x * exposure
    y = y / (1.0 + y)
    return np.clip(y, 0.0, 1.0) ** (1.0 / 2.2)


def solid_angle_weights(h: int, w: int) -> np.ndarray:
    """sin(theta) per riga di una equirettangolare.  Senza questo peso i poli, dove i
    pixel coprono un angolo solido minuscolo, conterebbero quanto l'equatore."""
    theta = np.pi * (np.arange(h, dtype=np.float64) + 0.5) / h
    return np.repeat(np.sin(theta)[:, None], w, axis=1)


def discover(root: Path) -> list[tuple[str, Path]]:
    """[(nome del run, percorso della skybox bakata)], ordinati per nome."""
    out = []
    for p in sorted(root.glob(f"*/*/{BAKED_NAME}")):
        out.append((p.parents[1].name, p))
    return out


def pretty(run_key: str) -> str:
    """exp_relmseraw_d02 -> exp / rel. sq."""
    parts = run_key.split("_")
    act = parts[0]
    loss = "_".join(parts[1:-1]) if len(parts) > 2 else parts[1]
    return f"{act} / {LOSS_LABEL.get(loss, loss)}"


def _panel(ax, rgb: np.ndarray, label: str, bold: bool = False) -> None:
    ax.imshow(rgb)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(label, fontsize=10, fontweight="bold" if bold else "normal")


def fig_grid(gt_tm: np.ndarray, baked_tm: dict[str, np.ndarray], selected: str | None,
             out: Path, ncols: int = 2) -> None:
    items = [("original", gt_tm, True)] + [(pretty(k), v, k == selected)
                                           for k, v in baked_tm.items()]
    nrows = (len(items) + ncols - 1) // ncols
    h, w = gt_tm.shape[:2]
    # ogni pannello e' 2:1; la figura segue quel rapporto, altrimenti matplotlib
    # lascerebbe bande bianche fra le righe
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.6 * ncols, 5.6 * (h / w) * nrows))
    flat = np.atleast_1d(axes).ravel()
    for ax, (label, img, bold) in zip(flat, items):
        _panel(ax, img, label, bold)
    for ax in flat[len(items):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  + {out}")


def fig_detail(gt_lin: np.ndarray, baked_lin: np.ndarray, gt_tm: np.ndarray,
               baked_tm: np.ndarray, label: str, out: Path) -> tuple[float, float]:
    """Pannello di dettaglio.  Restituisce (limite della colormap, mediana del log2)."""
    ratio = np.log2((baked_lin.mean(-1) + EPS) / (gt_lin.mean(-1) + EPS))
    # estremi simmetrici da un percentile: il massimo assoluto cade su qualche pixel
    # isolato del bordo delle sorgenti e schiaccerebbe a grigio tutto il resto
    lim = float(np.percentile(np.abs(ratio), 98.0))
    lim = max(lim, 0.05)
    med = float(np.median(ratio))

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 3.1))
    _panel(axes[0], gt_tm, "original")
    _panel(axes[1], baked_tm, f"baked, {label}")
    im = axes[2].imshow(ratio, cmap="RdBu_r",
                        norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim))
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].set_title(r"$\log_2$(baked / original)", fontsize=10)
    cb = fig.colorbar(im, ax=axes[2], fraction=0.032, pad=0.02)
    cb.set_label("stops", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  + {out}")
    return lim, med


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_root")
    ap.add_argument("--gt", required=True, help="EXR equirettangolare originale")
    ap.add_argument("--out", required=True, help="cartella di destinazione dei PNG")
    ap.add_argument("--selected", default=None,
                    help="run del pannello di dettaglio (default: il primo trovato)")
    ap.add_argument("--downsample", type=int, default=4,
                    help="fattore di media di blocchi (default 4: 4096x2048 -> 1024x512)")
    ap.add_argument("--key", type=float, default=0.5,
                    help="livello a cui portare la mediana della GT nel tonemap")
    args = ap.parse_args()

    root, gt_path, out = Path(args.sweep_root), Path(args.gt), Path(args.out)
    if not root.is_dir():
        print(f"ERRORE: {root} non e' una cartella")
        return 2
    if not gt_path.exists():
        print(f"ERRORE: {gt_path} non esiste")
        return 2
    out.mkdir(parents=True, exist_ok=True)

    runs = discover(root)
    if not runs:
        print(f"ERRORE: nessun {BAKED_NAME} sotto {root}")
        return 2
    selected = args.selected or runs[0][0]
    if selected not in dict(runs):
        print(f"ERRORE: {selected} non e' fra i run trovati ({[k for k, _ in runs]})")
        return 2

    print(f"GT: {gt_path.name}")
    gt_full = load_exr(gt_path)
    gt = block_mean(gt_full, args.downsample)
    print(f"  {gt_full.shape[1]}x{gt_full.shape[0]} -> {gt.shape[1]}x{gt.shape[0]}")

    # Esposizione condivisa, dalla GT: la mediana della luminanza finisce a meta' scala,
    # cosi' il tonemap non e' dettato dal picco HDR delle sorgenti
    lum = (gt * LUMA_COEFF).sum(-1)
    expo = args.key / max(float(np.median(lum)), 1e-4)
    print(f"  esposizione condivisa = {expo:.4f} (mediana luminanza GT "
          f"{float(np.median(lum)):.4f})")

    wts = solid_angle_weights(*gt.shape[:2])
    gt_norm = np.sqrt((gt.astype(np.float64) ** 2).sum(-1))
    gt_mean = float((gt_norm * wts).sum() / wts.sum())

    gt_tm = tonemap(gt, expo)
    baked_lin: dict[str, np.ndarray] = {}
    baked_tm: dict[str, np.ndarray] = {}
    print("\nrapporto delle medie lineari pesate per angolo solido, "
          "<||baked||> / <||GT||>:")
    for key, path in runs:
        a = block_mean(load_exr(path), args.downsample)
        if a.shape != gt.shape:
            print(f"  [skip] {key}: shape {a.shape} != GT {gt.shape}")
            continue
        baked_lin[key] = a
        baked_tm[key] = tonemap(a, expo)
        n = np.sqrt((a.astype(np.float64) ** 2).sum(-1))
        ratio = float((n * wts).sum() / wts.sum()) / gt_mean
        mark = "  <- selezionato" if key == selected else ""
        print(f"  {pretty(key):22s} {ratio:7.4f}   ({100 * (ratio - 1):+.2f}%){mark}")

    print()
    fig_grid(gt_tm, baked_tm, selected, out / "skybox_grid.png")
    lim, med = fig_detail(gt, baked_lin[selected], gt_tm, baked_tm[selected],
                          pretty(selected), out / "skybox_detail.png")
    print(f"\ndettaglio ({pretty(selected)}): mediana log2(baked/GT) = {med:+.4f} stop "
          f"({100 * (2.0 ** med - 1):+.2f}%), estremi colormap +/-{lim:.3f} stop")
    return 0


if __name__ == "__main__":
    sys.exit(main())
