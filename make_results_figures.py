#!/usr/bin/env python
"""make_results_figures.py -- I pannelli del capitolo Results che vengono dalle run.

Sei modalita', una per gruppo di figure:

    python make_results_figures.py maps      --out ../Doc/images/results
    python make_results_figures.py views     --out ../Doc/images/results
    python make_results_figures.py highfreq  --out ../Doc/images/results
    python make_results_figures.py curves    --out ../Doc/images/results
    python make_results_figures.py grids     --out ../Doc/images/results
    python make_results_figures.py mapdiff   --out ../Doc/images/results
    python make_results_figures.py spectrum  --out ../Doc/images/results

  maps      le colonne Studio/Night delle due griglie delle mappe (albedo dal fit,
            metallic, roughness) piu' la variante a cubo diffuso
  views     la riga "NeRF render" delle vecchie griglie preview a tre righe
  highfreq  i pannelli della sezione sul dettaglio: la sfera di pietra, la sfera
            emissiva, e la spada notturna su cui l'esponenziale emette zero
  curves    l'MSE contro l'iterazione della run che non addestra (spada notturna,
            esponenziale con L1): un pannello, una curva
  spectrum  la distribuzione dei valori del frame intero, render originale contro il
            NeRF di quella colonna, due pannelli per scena (studio | night)
  grids     le otto griglie preview nella forma nuova: quattro viste in riga, e per
            ognuna originale | ricostruzione | heatmap della differenza, con una
            tabella per skybox e una per ricostruzione (NeRF, re-render)
  mapdiff   la colonna heatmap delle griglie delle mappe: il riferimento autoriale
            ridotto alla risoluzione dell'atlante, meno la mappa recuperata

`grids` richiede che rerender_run.py sia gia' stato lanciato sulle run interessate:
legge <run>/rerender/pbr_gt/images/ e salta la colonna se non c'e'.

Tre convenzioni, ognuna per non falsare una lettura che la tesi fa sulle figure:

  1. Le mappe recuperate sono codificate come quelle autoriali di make_atlas_pngs.py:
     sRGB solo sull'albedo, che e' colore, e LINEARE su metallic e roughness, che sono
     dati.  Applicare una gamma a una roughness mostrerebbe un valore diverso da quello
     che il fit ha scritto, e la riga della griglia serve proprio a confrontare i valori
     fra colonna autoriale e colonne recuperate.

  2. L'esposizione di un pannello non si calcola mai sul pannello stesso: viene da
     make_scenes_figure.column_exposure(), cioe' dalla stessa mediana che ha tonemappato
     il render originale in cima alla colonna.  Se ogni riga si normalizzasse sulla
     propria mediana, un NeRF che sbaglia il livello medio verrebbe riportato in scala
     dal tonemap e la figura mostrerebbe un errore che non c'e' piu'.

  3. Dove si confrontano due ricostruzioni contro un riferimento (la sfera di pietra, la
     sfera emissiva, la spada notturna) i tre pannelli condividono UNA esposizione,
     presa dal riferimento.  E' la differenza fra le ricostruzioni il soggetto: dare a
     ognuna la sua la cancellerebbe.

I ritagli sono presi dall'immagine lineare a piena risoluzione e tonemappati dopo, per
lo stesso motivo per cui lo fa make_scenes_figure.py: ritagliare dopo il downsample
butterebbe via il dettaglio che il pannello deve mostrare.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from make_atlas_pngs import srgb
from make_skybox_figure import block_mean, load_exr, tonemap
from make_scenes_figure import (
    FAMILIES, SPHERE_CROP, column_exposure, exposure_of, frame_path, save_png,
)

RUNS_ROOT = Path("D:/tesi_output")

# La run che copre tutte e cinque le scene sotto entrambe le configurazioni.  Tenere
# tutte le colonne su una sola run e' cio' che rende confrontabili le griglie: fra due
# run cambierebbero anche i parametri che non sono il soggetto della figura.
#
# Dal 13/08/2026 si legge la copia rigenerata dopo il fix dell'azimut in doppia
# precisione di deviceProgramsIrradiance.cu: in `test_sword_shield` le mappe derivate
# dall'irradiance (irradiance stessa, albedo, albedo_pbr) sono stale.  Tutto il resto
# e' bit-identico fra i due alberi, verificato, quindi cambiare qui la sorgente tocca
# in pratica solo la riga dell'albedo delle due griglie.
SWEEP = RUNS_ROOT / "test_sword_shield_after_fix_irradiance"
EXP, SOFT = "exp_l1_d02", "softplus_relmseraw_d02"

# La variante ad alta frequenza si ferma al NeRF e vive in una run sua.
HIGHFREQ_ROOT = RUNS_ROOT / "test_high_details_new_batches"
HIGHFREQ_SCENE = "TableAndOtherInteriorWithSpecularHighDetails"

# Le mappe recuperate, con la codifica che ognuna richiede.  albedo_pbr e non albedo:
# la griglia e' l'output del fit, mentre l'albedo lambertiano ha la sua figura in
# Supporting Material e i due non sono la stessa quantita'.
MAPS: list[tuple[str, str, bool]] = [
    # (nome del PNG, percorso sotto sources/<source>/, codificare in sRGB)
    ("albedo",    "albedo_pbr/albedo_pbr.exr", True),
    ("metallic",  "metallic/metallic.exr",     False),
    ("roughness", "roughness/roughness.exr",   False),
]


@dataclass(frozen=True)
class Column:
    """Una colonna delle griglie: una scena sotto una configurazione."""
    folder: str      # sottocartella di images/results/
    suffix: str      # "_studio", "_night" o "" quando la scena ha una colonna sola
    run: Path
    family: str      # famiglia di make_scenes_figure, per la camera e l'esposizione
    scene_key: str   # variante dentro la famiglia


COLUMNS: list[Column] = [
    Column("interior", "_studio", SWEEP/EXP/"TableAndOtherInteriorWithSpecular",
           "interior", "specular"),
    Column("interior", "_night",  SWEEP/EXP/"TableAndOtherInteriorWithSpecularNight",
           "interior", "night"),
    Column("sword",    "_studio", SWEEP/EXP/"SwordShieldStudio",
           "sword",    "sword_studio"),
    # L'unica colonna del capitolo prodotta con softplus: sotto la mappa notturna
    # l'esponenziale non addestra affatto (vedi sec:results-collapse).
    Column("sword",    "_night",  SWEEP/SOFT/"SwordShieldNight",
           "sword",    "sword_night"),
    Column("diffusecube", "",     SWEEP/EXP/"TableAndOtherInteriorNoSpecular",
           "interior", "diffusecube"),
]


# ──────────────────────────────────────────────────────────────────────────────
# Utilita' comuni
# ──────────────────────────────────────────────────────────────────────────────

def latest_render_dir(run: Path) -> Path:
    """L'ultima iter_* di nerf_render_images/."""
    dirs = sorted((run / "nerf_render_images").glob("iter_*"))
    if not dirs:
        raise SystemExit(f"ERRORE: nessuna iter_* in {run / 'nerf_render_images'}")
    return dirs[-1]


def frame_index(run: Path, camera: str) -> int:
    """Posizione della camera in transforms_extended.json, che e' l'indice con cui i
    frame sono numerati in nerf_render_images/.  Risolto invece che scritto a mano:
    l'ordine dei frame dipende dal dataset, e un indice sbagliato non da' errore, da'
    la vista di un'altra camera."""
    frames = json.loads((run / "transforms_extended.json").read_text())["frames"]
    names = [Path(f["file_path"]).stem for f in frames]
    if camera not in names:
        raise SystemExit(f"ERRORE: {camera} non e' fra i frame di {run.name}")
    return names.index(camera)


def rendered_pair(run: Path, camera: str) -> tuple[np.ndarray, np.ndarray]:
    """(gt, pred) del frame di quella camera, a piena risoluzione e lineari."""
    d = latest_render_dir(run)
    i = frame_index(run, camera)
    return load_exr(d / f"frame_{i:03d}_gt.exr"), load_exr(d / f"frame_{i:03d}_pred.exr")


def check_gt_matches_dataset(col: Column, gt: np.ndarray) -> None:
    """La GT salvata dalla run deve essere il frame del dataset da cui make_scenes_figure
    ha ricavato l'esposizione della colonna.  Se non lo fosse, l'esposizione presa da li'
    non sarebbe quella di questa immagine e la riga NeRF render non sarebbe confrontabile
    con quella sopra: e' un errore silenzioso, quindi va controllato e non assunto."""
    fam = FAMILIES[col.family]
    scene_dir = {k: d for k, d, _, _ in fam.scenes}[col.scene_key]
    p = frame_path(scene_dir, fam.camera, fam.root)
    if not p.exists():
        print(f"    ! {p.name} non trovato nel dataset, controllo saltato")
        return
    ref = load_exr(p)
    if ref.shape != gt.shape:
        raise SystemExit(f"ERRORE: {col.folder}{col.suffix}: la GT della run e' "
                         f"{gt.shape} e il frame del dataset {ref.shape}")
    d = float(np.abs(ref - gt).max())
    if d > 1e-4:
        raise SystemExit(f"ERRORE: {col.folder}{col.suffix}: la GT della run non e' il "
                         f"frame del dataset (scarto massimo {d:.3g})")


def panels(out: Path, names: list[str], images: list[np.ndarray], expo: float) -> None:
    """Scrive n pannelli tonemappati con una sola esposizione."""
    for name, img in zip(names, images):
        save_png(tonemap(img, expo), out / f"{name}.png")


# ──────────────────────────────────────────────────────────────────────────────
# maps
# ──────────────────────────────────────────────────────────────────────────────

def do_maps(out: Path, source: str = "gt", atlas_size: int = 1024) -> None:
    for col in COLUMNS:
        src = col.run / "sources" / source
        print(f"{col.folder}{col.suffix or ' (unica)'}  <- {col.run.parent.name}/{col.run.name}")
        dst = out / col.folder
        dst.mkdir(parents=True, exist_ok=True)
        for name, rel, as_srgb in MAPS:
            p = src / rel
            if not p.exists():
                print(f"    ! {rel} non esiste, saltata")
                continue
            a = load_exr(p)
            # Nelle griglie una mappa e' larga 0.27\linewidth, cioe' circa 4.3 cm: a
            # 4096 texel sono 24000 dpi, e il PDF della tesi arrivava a 369 MB con le
            # sole colonne recuperate.  1024 texel su quella misura sono ancora 600 dpi,
            # sopra la risoluzione di stampa.  La media di blocchi va fatta PRIMA della
            # codifica sRGB: mediare valori gia' gammati darebbe un colore diverso.
            k = max(1, a.shape[0] // atlas_size)
            a = block_mean(a, k)
            rgb = srgb(a) if as_srgb else np.clip(a, 0.0, 1.0)
            f = dst / f"{name}{col.suffix}.png"
            plt.imsave(f, rgb)
            print(f"    + {f.name}  {rgb.shape[1]}x{rgb.shape[0]}  "
                  f"range [{a.min():.3f}, {a.max():.3f}]  "
                  f"{'sRGB' if as_srgb else 'lineare'}")


# ──────────────────────────────────────────────────────────────────────────────
# views
# ──────────────────────────────────────────────────────────────────────────────

def do_views(out: Path, downsample: int = 2) -> None:
    for col in COLUMNS:
        if not col.suffix:          # la variante a cubo diffuso non ha griglia preview
            continue
        fam = FAMILIES[col.family]
        gt, pred = rendered_pair(col.run, fam.camera)
        check_gt_matches_dataset(col, gt)
        expo = column_exposure(col.family, col.scene_key)
        dst = out / col.folder
        dst.mkdir(parents=True, exist_ok=True)
        name = f"{col.suffix.lstrip('_')}_nerf.png"
        save_png(tonemap(block_mean(pred, downsample), expo), dst / name)
        print(f"    esposizione {expo:.4f} da {col.scene_key}, "
              f"mediana pred/gt {np.median(pred):.5f}/{np.median(gt):.5f}")


# ──────────────────────────────────────────────────────────────────────────────
# highfreq
# ──────────────────────────────────────────────────────────────────────────────

def _detail_stats(name: str, crop: np.ndarray) -> None:
    """Le misure citate in sec:results-collapse.  Il gradiente e' cio' che separa una
    superficie ricostruita da una superficie levigata: media e dispersione possono
    essere giuste mentre il dettaglio e' sparito."""
    from make_skybox_figure import LUMA_COEFF
    lum = (crop * LUMA_COEFF).sum(-1)
    print(f"      {name:10s} media {lum.mean():8.4f}  std {lum.std():8.4f}  "
          f"picco {lum.max():9.2f}  gradiente {np.abs(np.diff(lum, axis=1)).mean():.5f}")


def do_highfreq(out: Path, downsample: int = 2) -> None:
    x0, y0, w, h = SPHERE_CROP

    # 1. La sfera di pietra dell'interno studio: nessuna delle due configurazioni
    #    fallisce, e il dettaglio fine e' gia' perso da entrambe.
    print("sfera di pietra (interno studio)")
    dst = out / "stone"
    dst.mkdir(parents=True, exist_ok=True)
    gt, pred_exp = rendered_pair(SWEEP/EXP/"TableAndOtherInteriorWithSpecular",
                                 FAMILIES["interior"].camera)
    _, pred_soft = rendered_pair(SWEEP/SOFT/"TableAndOtherInteriorWithSpecular",
                                 FAMILIES["interior"].camera)
    crops = [im[y0:y0+h, x0:x0+w] for im in (gt, pred_exp, pred_soft)]
    expo, _ = exposure_of(crops[0])
    for n, c in zip(("gt", "exp", "soft"), crops):
        _detail_stats(n, c)
    panels(dst, ["sphere_gt", "sphere_exp", "sphere_soft"], crops, expo)

    # 2. La sfera emissiva: stessa domanda, dettagli un ordine di grandezza piu'
    #    luminosi della loro base.  Qui le due configurazioni si separano.
    print("sfera emissiva (variante ad alta frequenza)")
    dst = out / "highfreq"
    dst.mkdir(parents=True, exist_ok=True)
    gt, pred_exp = rendered_pair(HIGHFREQ_ROOT/EXP/HIGHFREQ_SCENE,
                                 FAMILIES["interior"].camera)
    _, pred_soft = rendered_pair(HIGHFREQ_ROOT/SOFT/HIGHFREQ_SCENE,
                                 FAMILIES["interior"].camera)
    views = [gt, pred_exp, pred_soft]
    crops = [im[y0:y0+h, x0:x0+w] for im in views]
    for n, c in zip(("gt", "exp", "soft"), crops):
        _detail_stats(n, c)
    # Vista intera e ritaglio hanno due esposizioni diverse, entrambe prese dalla GT:
    # le curve emissive sono cosi' piu' brillanti del resto che l'esposizione che rende
    # leggibile il tavolo le brucia, e viceversa.
    expo_view, _ = exposure_of(views[0])
    expo_crop, _ = exposure_of(crops[0])
    panels(dst, ["view_gt", "view_exp", "view_soft"],
           [block_mean(v, downsample) for v in views], expo_view)
    panels(dst, ["sphere_gt", "sphere_exp", "sphere_soft"], crops, expo_crop)

    # 3. La spada notturna: il caso limite, l'esponenziale emette zero ovunque.
    print("spada notturna (l'esponenziale emette zero)")
    dst = out / "collapse"
    dst.mkdir(parents=True, exist_ok=True)
    fam = FAMILIES["sword"]
    gt, pred_exp = rendered_pair(SWEEP/EXP/"SwordShieldNight", fam.camera)
    _, pred_soft = rendered_pair(SWEEP/SOFT/"SwordShieldNight", fam.camera)
    print(f"      exp: massimo {pred_exp.max():.6g}, "
          f"frazione sotto 1e-3 {(pred_exp < 1e-3).mean():.4f}")
    expo = column_exposure("sword", "sword_night")
    panels(dst, ["swordnight_gt", "swordnight_exp", "swordnight_soft"],
           [block_mean(v, downsample) for v in (gt, pred_exp, pred_soft)], expo)


# ──────────────────────────────────────────────────────────────────────────────
# curves
# ──────────────────────────────────────────────────────────────────────────────

# Un pannello solo, una curva sola: l'MSE sul batch della run che non addestra.
# Dal 13/08/2026 non si disegnano piu' ne' la run softplus accanto ne' la loss su un
# secondo asse.  Mettere due run addestrate su loss diverse nello stesso pannello
# invita a leggerle come un confronto fra loss, che non e' quello che sono; i valori
# finali delle altre run stanno nella tabella del capitolo, dove sono etichettati.
CURVE_RUN = SWEEP/EXP/"SwordShieldNight"
CURVE_NAME = "curves_swordnight"
CURVE_TITLE = "Sword and shield, night: \\texttt{exp} with $L_1$"


def _metrics(run: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(iter, mse, loss) da nerf_train/training_metrics.csv."""
    import csv
    rows = list(csv.DictReader((run / "nerf_train" / "training_metrics.csv").open()))
    it = np.array([int(r["iter"]) for r in rows])
    return it, np.array([float(r["mse"]) for r in rows]), \
        np.array([float(r["loss"]) for r in rows])


def do_curves(out: Path) -> None:
    dst = out / "collapse"
    dst.mkdir(parents=True, exist_ok=True)
    it, mse, _ = _metrics(CURVE_RUN)

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(it, mse, color="#c0392b", lw=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("MSE on the training batch")
    ax.set_xlim(0, 75000)
    # Limiti sui percentili e non sugli estremi: otto batch su 808 schizzano fino a
    # 7e12 e tornano indietro entro un intervallo di display, e su una scala che li
    # contiene tutti il plateau, che e' il soggetto, diventa una riga sola.  Cosi'
    # escono dal riquadro come tratti verticali, visibili ma non decisivi per la scala.
    lo, hi = np.percentile(mse, 1), np.percentile(mse, 98)
    ax.set_ylim(lo / 1.6, hi * 1.6)
    ax.grid(alpha=0.25, which="both", lw=0.4)

    fig.tight_layout()
    fig.savefig(dst / f"{CURVE_NAME}.png", dpi=200)
    plt.close(fig)
    print(f"  + {dst / CURVE_NAME}.png   {mse[0]:.6g} -> {mse[-1]:.6g}")


# ──────────────────────────────────────────────────────────────────────────────
# spectrum -- distribuzione dei valori HDR, originale contro il NeRF di quella colonna
# ──────────────────────────────────────────────────────────────────────────────

# Le statistiche vengono da compare_runs.py invece di essere riscritte: se le due
# divergessero, la figura delle sezioni di scena e quella dell'ablation direbbero cose
# diverse sulla stessa quantita'.  spectrum_hist e' gia' sul frame intero e senza
# maschera, che e' esattamente il taglio che serve qui.
SPECTRA: list[tuple[str, list[tuple[str, Path, str]]]] = [
    ("interior", [
        ("Studio",  SWEEP/EXP/"TableAndOtherInteriorWithSpecular",     "exp / $L_1$"),
        ("Night",   SWEEP/EXP/"TableAndOtherInteriorWithSpecularNight", "exp / $L_1$"),
    ]),
    ("sword", [
        ("Studio",  SWEEP/EXP/"SwordShieldStudio",  "exp / $L_1$"),
        ("Night",   SWEEP/SOFT/"SwordShieldNight",  "softplus / rel. sq."),
    ]),
]


def _pooled_spectrum(run: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """(istogramma gt, istogramma pred, numero di frame) sul canale ||RGB||.

    Tutti i frame della run messi in un unico istogramma, frame intero: il NeRF viene
    consumato come sorgente di radianza su tutta la sfera, non solo sul soggetto, e
    separare foreground e background e' il taglio dell'ablation, non di questa figura.
    """
    from compare_runs import spectrum_hist, NORM_CH

    d = latest_render_dir(run)
    gts = sorted(d.glob("frame_*_gt.exr"))
    if not gts:
        raise SystemExit(f"ERRORE: nessun frame_*_gt.exr in {d}")
    h_gt = h_pr = None
    for p in gts:
        pred = p.with_name(p.name.replace("_gt.exr", "_pred.exr"))
        if not pred.exists():
            raise SystemExit(f"ERRORE: manca {pred.name} accanto a {p.name}")
        a = spectrum_hist(load_exr(p))[NORM_CH]
        b = spectrum_hist(load_exr(pred))[NORM_CH]
        h_gt = a if h_gt is None else h_gt + a
        h_pr = b if h_pr is None else h_pr + b
    return h_gt, h_pr, len(gts)


def do_spectrum(out: Path) -> None:
    from compare_runs import (spec_density, spec_w1_dex, _spec_x, _spec_xlim,
                              _spec_ylim)

    x = _spec_x()
    for folder, panels in SPECTRA:
        dst = out / folder
        dst.mkdir(parents=True, exist_ok=True)
        fig, axs = plt.subplots(1, len(panels), figsize=(9.0, 3.4))
        for ax, (title, run, label) in zip(np.atleast_1d(axs), panels):
            h_gt, h_pr, n = _pooled_spectrum(run)
            w1 = float(spec_w1_dex(h_pr, h_gt))
            # Limiti per pannello: le due condizioni di luce occupano intervalli di
            # radianza diversi, e un asse comune schiaccerebbe quella piu' stretta.
            xlim = _spec_xlim(h_gt, h_pr)
            ylim = _spec_ylim(h_gt, h_pr)
            ax.fill_between(x, np.maximum(spec_density(h_gt), 1e-12), 1e-12,
                            step="mid", color="#B0B0B0", alpha=0.75, lw=0,
                            label="original render", zorder=1)
            ax.plot(x, np.maximum(spec_density(h_pr), 1e-12), color="#12436D",
                    lw=1.4, label=f"{label}   $W_1$={w1:.3f} dex", zorder=3)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.grid(alpha=0.25, which="both", lw=0.4)
            ax.set_title(title, fontsize=10)
            # \lVert non esiste in mathtext: la doppia barra si scrive \|
            ax.set_xlabel(r"$\|\mathrm{RGB}\|_2$ (linear radiance)", fontsize=8)
            ax.set_ylabel("fraction of pixels per bin", fontsize=8)
            ax.legend(fontsize=7, loc="lower center", framealpha=0.9)
            print(f"    {folder}/{title}: {n} frames, W1 {w1:.4f} dex")
        fig.tight_layout()
        fig.savefig(dst / "spectrum.png", dpi=200)
        plt.close(fig)
        print(f"  + {dst / 'spectrum.png'}")


# ──────────────────────────────────────────────────────────────────────────────
# grids -- le otto griglie preview: originale | ricostruzione | heatmap, su 4 viste
# ──────────────────────────────────────────────────────────────────────────────

# Soglia sotto la quale una differenza e' numericamente irrilevante e il pannello va a
# nero pieno.  Stesso valore di compare_exr.py, che e' la figura a tre pannelli da cui
# questa griglia deriva: due convenzioni diverse per la stessa quantita' renderebbero
# incomparabili le due letture.
DIFF_FLOOR = 1e-4
DIFF_DECADES = 4.0       # ampiezza massima della scala log, sotto il tetto
DIFF_TOP_PCTL = 99.9     # il tetto, per non farlo dettare da un pixel solo
HEAT_CMAP = "inferno"

N_VIEWS = 4
COV_PCTL = 60.0          # percentile di copertura sotto cui una camera non e' candidata


def diff_norm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """||A - B||_2 per pixel sui valori lineari.  Copiata da compare_exr.py."""
    d = a - b
    return np.sqrt(np.einsum("ijk,ijk->ij", d, d, optimize=True))


def _heat_png(d: np.ndarray, vmin: float, vmax: float, out: Path,
              log: bool = True, bad: np.ndarray | None = None) -> None:
    """Un pannello di heatmap, senza assi ne' barra: la barra e' una sola per figura."""
    from matplotlib.colors import LogNorm, Normalize
    cmap = plt.get_cmap(HEAT_CMAP).copy()
    cmap.set_under("black")
    cmap.set_bad("#f0f2f4")          # fuori maschera: grigio neutro, non nero
    norm = LogNorm(vmin=vmin, vmax=vmax) if log else Normalize(vmin=vmin, vmax=vmax)
    x = np.ma.masked_where(bad, d) if bad is not None else d
    plt.imsave(out, cmap(norm(x)))
    print(f"  + {out.name}")


def _colorbar_png(vmin: float, vmax: float, out: Path, label: str,
                  log: bool = True, vertical: bool = False) -> None:
    """La barra di scala della figura, sottile, da mettere accanto o sotto la tabella.

    Una sola per figura invece di una per pannello: in una griglia 4x3 quattro barre
    identiche sarebbero solo rumore.  Orizzontale sotto le griglie preview, dove la scala
    e' una sola per tutta la figura; verticale nelle griglie delle mappe, dove ogni riga
    ha la sua e la barra deve stare a fianco della riga a cui appartiene.
    """
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.cm import ScalarMappable
    norm = LogNorm(vmin=vmin, vmax=vmax) if log else Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(HEAT_CMAP).copy()
    cmap.set_under("black")
    orient = "vertical" if vertical else "horizontal"
    fig, ax = plt.subplots(figsize=(0.30, 5.0) if vertical else (6.0, 0.42))
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation=orient)
    if vertical:
        # Tre soli valori e nessuna etichetta: la barra sta in una colonna della tabella,
        # e ogni cifra in piu' e' larghezza tolta ai pannelli.  Che quantita' sia lo dice
        # la didascalia.
        cb.set_ticks([vmin, 0.5 * (vmin + vmax), vmax])
        ax.set_yticklabels([f"{v:.2f}" for v in
                            (vmin, 0.5 * (vmin + vmax), vmax)], fontsize=11)
    else:
        cb.set_label(label, fontsize=8)
        ax.tick_params(labelsize=7)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  + {out.name}   [{vmin:.3g}, {vmax:.3g}]")


def _camera_table(run: Path) -> list[tuple[str, float, np.ndarray]]:
    """(stem, copertura del foreground, direzione unitaria) per ogni frame della run.

    La copertura viene dalle maschere che lo Step 1 ha gia' scritto, la direzione dalla
    posa: entrambe stanno nella run, quindi la scelta delle viste non dipende da nulla
    che debba essere ricalcolato."""
    frames = json.loads((run / "transforms_extended.json").read_text())["frames"]
    rows = []
    for f in frames:
        m = np.asarray(Image.open(run / f["mask_path"]))
        pos = np.array(f["transform_matrix"], dtype=np.float64)[:3, 3]
        rows.append((Path(f["file_path"]).stem, float((m > 127).mean()),
                     pos / max(np.linalg.norm(pos), 1e-9)))
    return rows


def select_cameras(run: Path, seed: str, n: int = N_VIEWS,
                   cov_pctl: float = COV_PCTL) -> list[str]:
    """Le n viste della griglia, scelte da una regola invece che a mano.

    Seme sulla camera che il capitolo usa gia' nelle figure di scena, cosi' la griglia
    resta ancorata a una vista che il lettore ha gia' visto; le altre per farthest-point
    sampling sulla direzione della camera, ristretto a quelle che inquadrano abbastanza
    soggetto.  Senza il vincolo di copertura il campionamento sceglierebbe le viste piu'
    lontane fra loro, che sono anche quelle che guardano il retro o il vuoto; senza il
    farthest-point le quattro viste sarebbero quattro varianti della stessa.
    """
    rows = _camera_table(run)
    cov = np.array([r[1] for r in rows])
    thr = float(np.percentile(cov, cov_pctl))
    cand = [r for r in rows if r[1] >= thr or r[0] == seed]
    if seed not in [r[0] for r in cand]:
        raise SystemExit(f"ERRORE: la camera seme {seed} non e' fra i frame di {run.name}")

    chosen = [next(r for r in cand if r[0] == seed)]
    while len(chosen) < n:
        best, best_d = None, -1.0
        for r in cand:
            if r[0] in [c[0] for c in chosen]:
                continue
            d = min(float(np.arccos(np.clip(r[2] @ c[2], -1, 1))) for c in chosen)
            if d > best_d:
                best, best_d = r, d
        chosen.append(best)
    print(f"  viste (copertura minima {100 * thr:.1f}%, {len(cand)} candidate):")
    for stem, c, _ in chosen:
        print(f"     {stem:28s} copertura {100 * c:5.2f}%")
    return [c[0] for c in chosen]


def _grid_columns() -> list[Column]:
    return [c for c in COLUMNS if c.suffix]


def do_grids(out: Path, downsample: int = 3, cameras: list[str] | None = None) -> None:
    for col in _grid_columns():
        fam = FAMILIES[col.family]
        sky = col.suffix.lstrip("_")
        print(f"\n{col.folder} / {sky}  <- {col.run.parent.name}/{col.run.name}")
        rer_dir = col.run / "rerender" / "pbr_gt" / "images"
        if not rer_dir.is_dir():
            print(f"  ! {rer_dir} non esiste: lancia prima rerender_run.py")
            continue

        views = cameras or select_cameras(col.run, fam.camera)
        rdir = latest_render_dir(col.run)
        missing = [c for c in views if not (rer_dir / f"{c}.exr").exists()]
        if missing:
            print(f"  ! re-render incompleto, mancano {len(missing)} viste "
                  f"({', '.join(missing)}): colonna saltata")
            continue

        # Prima passata: si leggono tutte le immagini e si calcolano le differenze, per
        # poter fissare UNA scala su tutte e otto (4 viste x 2 ricostruzioni).  E' quella
        # scala condivisa a rendere leggibile il confronto fra la tabella NeRF e quella
        # re-render: con una scala per tabella la domanda "quale delle due sta piu'
        # vicino all'originale" non avrebbe risposta nella figura.
        data = {}
        for cam in views:
            i = frame_index(col.run, cam)
            orig = load_exr(col.run / "images" / f"{cam}.exr")
            gt = load_exr(rdir / f"frame_{i:03d}_gt.exr")
            # L'indice del frame e' risolto per nome, e uno sbagliato non da' errore: da'
            # la vista di un'altra camera, che una heatmap piena di struttura farebbe
            # passare per errore di ricostruzione.  Quindi si controlla.
            d = float(np.abs(orig - gt).max())
            if d > 1e-4:
                raise SystemExit(f"ERRORE: {cam} -> frame_{i:03d}: la GT della run non e' "
                                 f"il frame del dataset (scarto massimo {d:.3g})")
            nerf = load_exr(rdir / f"frame_{i:03d}_pred.exr")
            rer = load_exr(rer_dir / f"{cam}.exr")
            fg = np.asarray(Image.open(col.run / "mask" /
                                       f"{cam}_mask.png")) > 127
            data[cam] = (orig, {"nerf": nerf, "rerender": rer},
                         {k: diff_norm(orig, v) for k, v in
                          (("nerf", nerf), ("rerender", rer))}, fg)

        pooled = np.concatenate([d.ravel() for _, _, dd, _ in data.values()
                                 for d in dd.values()])
        vmax = float(np.percentile(pooled, DIFF_TOP_PCTL))
        vmin = max(DIFF_FLOOR, vmax / 10.0 ** DIFF_DECADES)
        print(f"  scala condivisa [{vmin:.3g}, {vmax:.3g}] "
              f"({np.log10(vmax / vmin):.1f} decadi, log, tetto al p{DIFF_TOP_PCTL})")

        dst = out / col.folder / "grid"
        dst.mkdir(parents=True, exist_ok=True)
        for k, cam in enumerate(views):
            orig, recon, diffs, fg = data[cam]
            # Una esposizione per riga, presa dall'ORIGINALE di quella riga: e' la regola
            # 2 del docstring.  Righe diverse hanno esposizioni diverse perche' sono viste
            # diverse, ma i pannelli di una riga no, o il tonemap rimetterebbe in scala
            # l'errore di livello medio che la riga deve mostrare.
            expo, med = exposure_of(orig)
            save_png(tonemap(block_mean(orig, downsample), expo),
                     dst / f"{sky}_v{k}_orig.png")
            for which in ("nerf", "rerender"):
                save_png(tonemap(block_mean(recon[which], downsample), expo),
                         dst / f"{sky}_v{k}_{which}.png")
                _heat_png(block_mean(diffs[which][..., None], downsample)[..., 0],
                          vmin, vmax, dst / f"{sky}_v{k}_{which}_heat.png")
            # Due mediane, e la seconda e' quella che conta.  Su una vista con molto
            # sfondo la mediana sull'intero fotogramma misura soprattutto l'ambiente,
            # che il re-render riproduce esatto e il NeRF no: da sola direbbe che il
            # re-render e' quasi perfetto anche dove sbaglia tutta la geometria.
            print(f"    {cam}: esposizione {expo:.4f} (mediana {med:.4f}); p50 |diff| "
                  f"tutto  nerf {np.median(diffs['nerf']):.4f} / "
                  f"rerender {np.median(diffs['rerender']):.4f}   "
                  f"foreground  nerf {np.median(diffs['nerf'][fg]):.4f} / "
                  f"rerender {np.median(diffs['rerender'][fg]):.4f}")
        _colorbar_png(vmin, vmax, dst / f"{sky}_cbar.png",
                      r"$\|\Delta$RGB$\|_2$  (linear radiance, log scale)")


# ──────────────────────────────────────────────────────────────────────────────
# mapdiff -- la colonna heatmap delle griglie delle mappe
# ──────────────────────────────────────────────────────────────────────────────

# (nome, file autoriale, percorso recuperato sotto sources/<source>/, e' scalare?)
#
# L'ultimo campo NON e' un dettaglio di comodo.  metallic e roughness sono grandezze a
# un canale, ma `load_exr` replica il canale unico su tre per uniformita': prendendone
# la norma L2 la differenza verrebbe moltiplicata per sqrt(3), e infatti il massimo
# usciva 1.7321 su mappe che stanno in [0,1].  Sulle grandezze scalari si usa |delta|
# su un canale solo; sull'albedo, che e' davvero un colore, la norma sui tre.
#
# Nota per la didascalia: due righe su tre non sono un errore contro un riferimento.  La
# roughness recuperata e' l'indice di apertura del cono e non una roughness GGX
# (sec:metallic-roughness), e sull'isola del cubo il base color autoriale e' la tinta
# speculare F0 e non un albedo diffuso (sec:results-gt): in entrambi i casi la mappa
# mostra un DISACCORDO, non uno scarto da un valore giusto.
MAPDIFF: list[tuple[str, str, str, bool]] = [
    ("albedo",    "BakedMaterial_base_color.exr", "albedo_pbr/albedo_pbr.exr", False),
    ("metallic",  "BakedMaterial_metallic.exr",   "metallic/metallic.exr",     True),
    ("roughness", "BakedMaterial_roughness.exr",  "roughness/roughness.exr",   True),
]

MAPDIFF_TOP_PCTL = 99.0


def _axis_weights(n: int, m: int):
    """Matrice sparsa (m, n) della media d'area lungo un asse.

    Generalizza la media di blocchi al rapporto non intero, che serve perche' il bake
    autoriale della spada e' 8096 e non 8192: 8096/2 = 4048 non si puo' sottrarre da un
    atlante 4096.  Ogni riga ha due o tre elementi non nulli, quindi la somma resta
    esatta in float32 (a differenza di una cumulata, che su 8096 termini perderebbe
    proprio le cifre che la differenza deve mostrare).
    """
    import scipy.sparse as sp
    s = n / m
    rows, cols, vals = [], [], []
    for j in range(m):
        a0, a1 = j * s, (j + 1) * s
        for i in range(int(np.floor(a0)), min(int(np.ceil(a1)), n)):
            w = min(a1, i + 1.0) - max(a0, float(i))
            if w > 0:
                rows.append(j)
                cols.append(i)
                vals.append(w / s)
    return sp.csr_matrix((vals, (rows, cols)), shape=(m, n), dtype=np.float32)


def reduce_to_atlas(img: np.ndarray, side: int) -> np.ndarray:
    """Il riferimento autoriale ridotto alla risoluzione dell'atlante, per media d'area.

    Il protocollo di sec:results-gt: mai per campionamento puntuale, che butterebbe tre
    quarti del segnale autoriale e conterebbe l'aliasing risultante come errore di
    ricostruzione.  Con rapporto intero e' esattamente la media di blocchi.
    """
    h, w = img.shape[:2]
    if h == w and h % side == 0:
        return block_mean(img, h // side)
    a = img.astype(np.float32)
    t = (_axis_weights(h, side) @ a.reshape(h, -1)).reshape(side, w, -1)
    t = (_axis_weights(w, side) @ t.transpose(1, 0, 2).reshape(w, -1))
    return t.reshape(side, side, -1).transpose(1, 0, 2)


def _authored_dir(run: Path) -> Path:
    """La cartella del bake autoriale, dal manifest della run: e' quella della normale
    esterna, l'unico dei quattro file che la pipeline riceve in ingresso."""
    m = json.loads((run / "run_manifest.json").read_text())
    p = m["scene"].get("external_normal_path")
    if not p:
        raise SystemExit(f"ERRORE: {run.name} non registra scene.external_normal_path")
    return Path(p).parent


def do_mapdiff(out: Path, source: str = "gt", atlas_size: int = 1024) -> None:
    for scene in ("interior", "sword"):
        cols = [c for c in _grid_columns() if c.folder == scene]
        print(f"\n{scene}: {' e '.join(c.suffix.lstrip('_') for c in cols)}")
        for name, authored_name, rel, scalar in MAPDIFF:
            # Una scala per mappa, condivisa fra studio e notte: l'autoriale e' lo stesso
            # file nelle due, quindi il confronto che la figura serve a fare e' proprio
            # quale illuminazione recupera meglio, e con due scale sparirebbe.
            diffs, masks = {}, {}
            for col in cols:
                ref_p = _authored_dir(col.run) / authored_name
                rec_p = col.run / "sources" / source / rel
                if not ref_p.exists() or not rec_p.exists():
                    print(f"  ! {name}: manca {ref_p if not ref_p.exists() else rec_p}")
                    break
                rec = load_exr(rec_p)
                raw = load_exr(ref_p)
                ref = reduce_to_atlas(raw, rec.shape[0])
                mask = load_exr(col.run / "ium" / "ium_masks.exr")[..., 0] > 0.5
                d = (np.abs(ref[..., 0] - rec[..., 0]) if scalar
                     else diff_norm(ref, rec))
                diffs[col.suffix] = d
                masks[col.suffix] = mask
                print(f"  {name:9s} {col.suffix.lstrip('_'):6s} autoriale "
                      f"{raw.shape[0]} -> {rec.shape[0]}; "
                      f"|diff| sui texel coperti: p50 {np.median(d[mask]):.4f}  "
                      f"p99 {np.percentile(d[mask], 99):.4f}")
                del raw
            if len(diffs) != len(cols):
                continue

            pooled = np.concatenate([d[masks[k]].ravel() for k, d in diffs.items()])
            vmax = float(np.percentile(pooled, MAPDIFF_TOP_PCTL))
            # Lineare e non logaritmica: qui le due grandezze stanno in [0,1] e la
            # differenza e' limitata, mentre nelle griglie preview e' radianza HDR.
            for col in cols:
                dst = out / col.folder / "mapdiff"
                dst.mkdir(parents=True, exist_ok=True)
                d, mask = diffs[col.suffix], masks[col.suffix]
                k = max(1, d.shape[0] // atlas_size)
                _heat_png(block_mean(d[..., None], k)[..., 0], 0.0, vmax,
                          dst / f"{name}{col.suffix}_diff.png", log=False,
                          bad=~(block_mean(mask[..., None].astype(np.float32), k)[..., 0]
                                > 0.5))
            # Verticale: qui la scala e' una per riga, non una per figura, e la barra
            # deve stare a fianco della riga a cui appartiene.
            _colorbar_png(0.0, vmax, out / scene / "mapdiff" / f"{name}_cbar.png",
                          r"$|\Delta|$" if scalar else r"$\|\Delta\|_2$",
                          log=False, vertical=True)


# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("maps", "views", "highfreq", "curves",
                                     "grids", "mapdiff", "spectrum", "all"))
    ap.add_argument("--out", required=True, help="cartella Doc/images/results")
    ap.add_argument("--downsample", type=int, default=2,
                    help="media di blocchi sulle viste intere (default 2)")
    ap.add_argument("--atlas-size", type=int, default=1024,
                    help="lato a cui ridurre le mappe in texture space (default 1024)")
    ap.add_argument("--cameras", nargs="+", default=None,
                    help="forza le viste delle griglie invece di sceglierle per copertura "
                         "e spaziatura angolare (solo per il modo grids)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    modes = (("maps", "views", "highfreq", "curves", "grids", "mapdiff", "spectrum")
             if args.mode == "all" else (args.mode,))
    for m in modes:
        print(f"\n=== {m} ===")
        if m == "maps":
            do_maps(out, atlas_size=args.atlas_size)
        elif m == "views":
            do_views(out, args.downsample)
        elif m == "highfreq":
            do_highfreq(out, args.downsample)
        elif m == "grids":
            # 3 e non 2: con 96 pannelli nuovi il peso del PDF conta, e 640x360 su
            # 0.30\linewidth sono ancora ~340 dpi, sopra la risoluzione di stampa.
            do_grids(out, downsample=3, cameras=args.cameras)
        elif m == "mapdiff":
            do_mapdiff(out, atlas_size=args.atlas_size)
        elif m == "spectrum":
            do_spectrum(out)
        else:
            do_curves(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
