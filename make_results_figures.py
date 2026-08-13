#!/usr/bin/env python
"""make_results_figures.py -- I pannelli del capitolo Results che vengono dalle run.

Quattro modalita', una per gruppo di figure:

    python make_results_figures.py maps      --out ../Doc/images/results
    python make_results_figures.py views     --out ../Doc/images/results
    python make_results_figures.py highfreq  --out ../Doc/images/results
    python make_results_figures.py curves    --out ../Doc/images/results

  maps      le colonne Studio/Night delle due griglie delle mappe (albedo dal fit,
            metallic, roughness) piu' la variante a cubo diffuso
  views     la riga "NeRF render" delle due griglie preview
  highfreq  i pannelli della sezione sul dettaglio: la sfera di pietra, la sfera
            emissiva, e la spada notturna su cui l'esponenziale emette zero
  curves    l'MSE contro l'iterazione sulle tre scene dove le due configurazioni
            non sono d'accordo

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

# (nome del file, titolo, run esponenziale, run softplus, mostrare anche la loss L1)
CURVES: list[tuple[str, str, Path, Path, bool]] = [
    ("curves_nightinterior", "Interior, night",
     SWEEP/EXP/"TableAndOtherInteriorWithSpecularNight",
     SWEEP/SOFT/"TableAndOtherInteriorWithSpecularNight", False),
    ("curves_highfreq", "Interior, high frequency",
     HIGHFREQ_ROOT/EXP/HIGHFREQ_SCENE, HIGHFREQ_ROOT/SOFT/HIGHFREQ_SCENE, True),
    ("curves_swordnight", "Sword and shield, night",
     SWEEP/EXP/"SwordShieldNight", SWEEP/SOFT/"SwordShieldNight", False),
]


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
    for name, title, run_exp, run_soft, with_loss in CURVES:
        it_e, mse_e, loss_e = _metrics(run_exp)
        it_s, mse_s, _ = _metrics(run_soft)

        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        ax.plot(it_e, mse_e, color="#c0392b", lw=1.4, label="exp / $L_1$")
        ax.plot(it_s, mse_s, color="#2471a3", lw=1.4, label="softplus / rel. sq.")
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("MSE on the training batch")
        ax.set_title(title, fontsize=10)
        # Stesso intervallo di iterazioni sui tre pannelli: e' cio' che rende
        # confrontabile a vista la lunghezza dei plateau.
        ax.set_xlim(0, 75000)
        ax.grid(alpha=0.25, which="both", lw=0.4)

        if with_loss:
            # L'unica curva che continua a scendere mentre l'MSE e' piatto: senza di
            # essa il pannello sembrerebbe un ottimizzatore fermo, e non lo e'.
            ax2 = ax.twinx()
            ax2.plot(it_e, loss_e, color="#c0392b", lw=1.0, ls="--", alpha=0.7,
                     label="exp / $L_1$: the loss")
            ax2.set_yscale("log")
            ax2.set_ylabel("$L_1$ loss", color="#c0392b")
            ax2.tick_params(axis="y", labelcolor="#c0392b")
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="best")
        else:
            ax.legend(fontsize=8, loc="best")

        fig.tight_layout()
        fig.savefig(dst / f"{name}.png", dpi=200)
        plt.close(fig)
        print(f"  + {dst / name}.png   exp {mse_e[-1]:.4g} -> softplus {mse_s[-1]:.4g}")


# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("maps", "views", "highfreq", "curves", "all"))
    ap.add_argument("--out", required=True, help="cartella Doc/images/results")
    ap.add_argument("--downsample", type=int, default=2,
                    help="media di blocchi sulle viste intere (default 2)")
    ap.add_argument("--atlas-size", type=int, default=1024,
                    help="lato a cui ridurre le mappe in texture space (default 1024)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    modes = ("maps", "views", "highfreq", "curves") if args.mode == "all" else (args.mode,)
    for m in modes:
        print(f"\n=== {m} ===")
        if m == "maps":
            do_maps(out, atlas_size=args.atlas_size)
        elif m == "views":
            do_views(out, args.downsample)
        elif m == "highfreq":
            do_highfreq(out, args.downsample)
        else:
            do_curves(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
