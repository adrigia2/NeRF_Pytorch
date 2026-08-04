#!/usr/bin/env python
"""make_scenes_figure.py -- Figure di tesi sulle quattro varianti della scena interna.

Per ogni variante produce due PNG:

  <key>_view.png    la vista renderizzata, tonemappata
  <key>_detail.png  un ritaglio a piena risoluzione sull'elemento che distingue la
                    variante (il cubo metallico, la sfera emissiva)

    python make_scenes_figure.py --out ../Doc/images/scenes
    python make_scenes_figure.py --contact-sheet specular --out <cartella temporanea>

Tre scelte non sono negoziabili e sono il motivo per cui questo script esiste:

  1. La camera e' la stessa per tutte e quattro le varianti, presa fra le
     `render_Camera_Shell21_*`.  I `render_config.json` differiscono: la variante ad
     alta frequenza ha `center_offset` 0 invece di 0.5 sullo shell 1, quindi solo i 30
     frame dello shell 2 hanno estrinseche identiche in tutte le scene.  Con una camera
     dello shell 1 le viste non sarebbero confrontabili.

  2. Le tre varianti diurne condividono UNA esposizione, derivata dalla mediana della
     luminanza della variante speculare.  Se ognuna avesse la sua, il cubo che nella
     variante diffusa appare opaco potrebbe esserlo per via dell'esposizione e non del
     materiale, che e' esattamente cio' che la figura deve mostrare.  La variante
     notturna ha la sua, perche' con l'esposizione diurna sarebbe nera.

  3. Il ritaglio e' preso dall'immagine lineare a piena risoluzione e tonemappato con
     la stessa esposizione della vista da cui proviene.  Ritagliare dopo il downsample
     butterebbe via proprio il dettaglio che il pannello deve mostrare.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from make_skybox_figure import LUMA_COEFF, block_mean, load_exr, tonemap

SCENES_ROOT = Path("C:/Users/adria/Documents/GitHub/Tesi/OptixProjectCMake/Scenes"
                   "/TableAndOtherInterior")

# Camera condivisa: solo lo shell 2 ha estrinseche identiche in tutte le varianti.
# La 38 e' l'unica in cui i tre oggetti (sfera, coniglio, cubo) non si sovrappongono e
# si vede anche l'ambiente, che e' cio' che spiega la tinta della luce.
CAMERA = "render_Camera_Shell21_38"

# (key, cartella del dataset, etichetta, gruppo di esposizione)
# Il gruppo "day" condivide una sola esposizione; "night" ne ha una propria.
SCENES: list[tuple[str, str, str, str]] = [
    ("specular",    "NerfOpenEXRSmooth",          "specular variant (base)", "day"),
    ("highfreq",    "NerfOpenEXRHighDetails",     "high-frequency variant",  "day"),
    ("night",       "NerfOpenEXRSmoothNight",     "night variant",           "night"),
    ("diffusecube", "NerfOpenExrSmoothNoDiffuse", "diffuse-cube variant",    "day"),
]

# Ritaglio (x0, y0, larghezza, altezza) in pixel a piena risoluzione, per variante.
# Le tre varianti che si distinguono per il cubo usano lo stesso rettangolo, cosi' il
# confronto fra i ritagli e' diretto; quella ad alta frequenza inquadra la sfera.
# Stesso 16:9 della vista: nella figura i due pannelli stanno affiancati alla stessa
# larghezza, e con proporzioni diverse avrebbero altezze diverse.
CROPS: dict[str, tuple[int, int, int, int]] = {
    "specular":    (860, 265, 640, 360),
    "highfreq":    (460, 340, 640, 360),
    "night":       (860, 265, 640, 360),
    "diffusecube": (860, 265, 640, 360),
}

# L'esposizione porta la mediana della luminanza a questo livello prima del Reinhard.
# 0.2 e non 0.5: la mediana di questa inquadratura cade sul pavimento scuro dello studio,
# e portarla a meta' scala bruciava il tavolo, che e' quasi tutto cio' che conta.
KEY = 0.20

# Preview delle due envmap per la tabella degli asset: (key, percorso relativo a
# SCENES_ROOT).  La notturna vive nella cartella del bake, non in assets/hdri.
SKYBOXES: list[tuple[str, str]] = [
    ("skybox_studio", "Blender/assets/hdri/wooden_studio_13_4k.exr"),
    ("skybox_night",  "BlenderBakedSmoothNight/cobblestone_street_night_4k.exr"),
]


def frame_path(scene_dir: str, camera: str) -> Path:
    return SCENES_ROOT / scene_dir / "images" / f"{camera}.exr"


def exposure_of(img: np.ndarray, key: float = KEY) -> tuple[float, float]:
    """(esposizione, mediana della luminanza).  La mediana, e non il massimo, perche'
    in una scena HDR il picco sta sulle sorgenti e detterebbe da solo il tonemap."""
    lum = (img * LUMA_COEFF).sum(-1)
    med = max(float(np.median(lum)), 1e-4)
    return key / med, med


def save_png(rgb: np.ndarray, out: Path) -> None:
    plt.imsave(out, np.clip(rgb, 0.0, 1.0))
    print(f"  + {out}  ({rgb.shape[1]}x{rgb.shape[0]})")


def skybox_previews(out: Path, key: float, downsample: int = 4) -> None:
    """Le due envmap tonemappate, per la tabella degli asset.

    Ognuna con la propria esposizione: sono due ambienti diversi, non due versioni della
    stessa scena, e qui la figura deve solo renderle leggibili.  Il downsample e' una
    media di blocchi in spazio lineare, prima del tonemap, per non alterare la radianza
    media delle sorgenti piccole (stesso motivo di make_skybox_figure).
    """
    for name, rel in SKYBOXES:
        p = SCENES_ROOT / rel
        if not p.exists():
            raise SystemExit(f"ERRORE: {p} non esiste")
        a = block_mean(load_exr(p), downsample)
        expo, med = exposure_of(a, key)
        print(f"{name}: {p.name}  esposizione {expo:.3f} "
              f"(mediana luminanza {med:.4f})")
        save_png(tonemap(a, expo), out / f"{name}.png")


def contact_sheet(scene_dir: str, out: Path, downsample: int = 8,
                  ncols: int = 6) -> None:
    """Griglia di tutti i frame della scena, per scegliere la camera a occhio."""
    paths = sorted((SCENES_ROOT / scene_dir / "images").glob("*.exr"))
    if not paths:
        raise SystemExit(f"ERRORE: nessun EXR in {SCENES_ROOT / scene_dir / 'images'}")
    print(f"contact sheet di {scene_dir}: {len(paths)} frame")

    thumbs = [block_mean(load_exr(p), downsample) for p in paths]
    expo, med = exposure_of(np.concatenate([t.reshape(-1, 3) for t in thumbs])[None])
    print(f"  esposizione = {expo:.4f} (mediana luminanza {med:.4f})")

    nrows = (len(thumbs) + ncols - 1) // ncols
    h, w = thumbs[0].shape[:2]
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.0 * ncols, 3.0 * (h / w) * nrows))
    flat = np.atleast_1d(axes).ravel()
    for ax, path, t in zip(flat, paths, thumbs):
        ax.imshow(tonemap(t, expo))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(path.stem.replace("render_Camera_", ""), fontsize=7)
    for ax in flat[len(thumbs):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  + {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="cartella di destinazione dei PNG")
    ap.add_argument("--camera", default=CAMERA, help=f"frame da usare (default {CAMERA})")
    ap.add_argument("--downsample", type=int, default=2,
                    help="media di blocchi sulla vista (default 2: 1920x1080 -> 960x540)")
    ap.add_argument("--key", type=float, default=KEY,
                    help=f"livello a cui portare la mediana della luminanza (default {KEY})")
    ap.add_argument("--contact-sheet", default=None, metavar="KEY",
                    help="scrive solo la griglia dei frame della variante indicata")
    ap.add_argument("--skyboxes", action="store_true",
                    help="scrive solo le preview delle due envmap per la tabella asset")
    ap.add_argument("--no-crop", action="store_true",
                    help="salta i ritagli (utile mentre si scelgono i rettangoli)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    by_key = {k: (d, lab, grp) for k, d, lab, grp in SCENES}

    if args.skyboxes:
        skybox_previews(out, args.key)
        return 0

    if args.contact_sheet:
        if args.contact_sheet not in by_key:
            print(f"ERRORE: {args.contact_sheet} non e' fra {list(by_key)}")
            return 2
        contact_sheet(by_key[args.contact_sheet][0],
                      out / f"contact_{args.contact_sheet}.png")
        return 0

    # Carica prima tutte le viste lineari: l'esposizione diurna e' condivisa e va
    # calcolata sulla variante speculare prima di tonemappare qualsiasi cosa.
    linear: dict[str, np.ndarray] = {}
    for key, scene_dir, _, _ in SCENES:
        p = frame_path(scene_dir, args.camera)
        if not p.exists():
            print(f"ERRORE: {p} non esiste")
            return 2
        linear[key] = load_exr(p)
        print(f"{key:12s} {p.name}  {linear[key].shape[1]}x{linear[key].shape[0]}")

    expo_day, med_day = exposure_of(linear["specular"], args.key)
    expo_night, med_night = exposure_of(linear["night"], args.key)
    print(f"\nesposizione diurna condivisa = {expo_day:.4f} "
          f"(mediana luminanza speculare {med_day:.5f})")
    print(f"esposizione notturna         = {expo_night:.4f} "
          f"(mediana luminanza notte {med_night:.5f}, "
          f"{med_day / max(med_night, 1e-9):.1f}x piu' scura)")

    print()
    for key, _, _, group in SCENES:
        expo = expo_night if group == "night" else expo_day
        lin = linear[key]
        save_png(tonemap(block_mean(lin, args.downsample), expo),
                 out / f"{key}_view.png")
        if args.no_crop:
            continue
        x0, y0, w, h = CROPS[key]
        save_png(tonemap(lin[y0:y0 + h, x0:x0 + w], expo), out / f"{key}_detail.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
