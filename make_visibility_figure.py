#!/usr/bin/env python
"""make_visibility_figure.py -- Mappe in spazio texture per le figure 3.8 e 3.3.7.

    python make_visibility_figure.py <run_dir> --camera render_Camera_Shell21_38 \
        --out ../Doc/images/visibility
    python make_visibility_figure.py <run_dir> --irradiance --out ../Doc/images/irradiance

Modalita' `--visibility` (default), figura 3.8:

  camera_mask.png    la maschera di UNA camera in spazio texel: i texel che quella
                     camera vede davvero, cioe' occlusione AND frustum AND grazing
  camera_count.png   quante camere coprono ogni texel, sommando i 60 canali di
                     visibility.exr, su scala di colore con barra

Modalita' `--irradiance`, sezione 3.3.7:

  irradiance.png           la componente diretta dalla environment map
  irradiance_indirect.png  la componente indiretta interrogata dal NeRF

Le due componenti dell'irradiance sono HDR e hanno intervalli molto diversi (l'indiretta
e' tipicamente un ordine di grandezza sotto), quindi si normalizzano **separatamente** e
si scrive il fattore usato: una scala comune renderebbe l'indiretta un rettangolo nero e
non direbbe niente.  La normalizzazione e' su un percentile, non sul massimo, perche' la
coda HDR di una scena con sorgente concentrata schiaccerebbe tutto il resto.

ATTENZIONE alla sorgente: dopo il fix dell'azimut in `deviceProgramsIrradiance.cu`
(13/08/2026) le `irradiance.exr` di `test_sword_shield` sono stale.  Per la figura della
tesi va usato l'albero rigenerato `test_sword_shield_after_fix_irradiance`.  Le mappe di
visibilita' e le maschere per-camera invece non dipendono dall'irradiance e sono
bit-identiche fra i due alberi, quindi per la 3.8 va bene entrambi.
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

from make_depth_figure import content_box, save_png       # noqa: E402
from make_skybox_figure import load_exr                   # noqa: E402

plt.rcParams.update({"font.size": 13})

DPI = 190
PCTL = 99.5          # percentile per la normalizzazione delle mappe HDR
MARGIN = 0.02        # margine del ritaglio sull'area utile dell'atlante


def read_channels(path: Path) -> dict:
    """Tutti i canali di un EXR come float32.  `load_exr` assume RGB e visibility.exr
    ha un canale per camera, quindi qui si legge l'header e si prende tutto."""
    import OpenEXR
    import Imath
    fh = OpenEXR.InputFile(str(path))
    head = fh.header()
    dw = head["dataWindow"]
    w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    ft = Imath.PixelType(Imath.PixelType.FLOAT)
    return {c: np.frombuffer(fh.channel(c, ft), np.float32).reshape(h, w)
            for c in head["channels"]}


def crop_to_atlas(run: Path):
    """Il riquadro utile dell'atlante, dalla maschera IUM.  Lo stesso `content_box` che
    usa make_depth_figure, cosi' i ritagli delle figure sono confrontabili fra loro."""
    mask = load_exr(run / "ium" / "ium_masks.exr")[..., 0] > 0.5
    return content_box(mask, MARGIN), mask


def heat_png(data: np.ndarray, mask: np.ndarray, out: Path, label: str,
             vmax: float | None = None, cmap: str = "magma") -> None:
    """Mappa in falsi colori con barra, fuori maschera in grigio neutro."""
    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    shown = np.where(mask, data, np.nan)
    im = ax.imshow(shown, cmap=cmap, vmin=0.0, vmax=vmax, interpolation="nearest")
    ax.set_facecolor("#f0f2f4")
    ax.axis("off")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(label)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def do_visibility(run: Path, camera: str, out: Path) -> None:
    box, mask = crop_to_atlas(run)
    ys, xs = box

    cam_path = run / "camera_mask" / f"{camera}.exr"
    if not cam_path.exists():
        raise SystemExit(f"✗ maschera per-camera non trovata: {cam_path}")
    cam = load_exr(cam_path)[..., 0] > 0.5
    save_png(np.repeat(cam[ys, xs, None].astype(np.float32), 3, axis=-1),
             out / "camera_mask.png")
    print(f"  camera_mask.png       {camera}: {100 * cam[mask].mean():.1f}% dei texel "
          f"della mesh visti da questa camera")

    vis = read_channels(run / "visibility" / "visibility.exr")
    cams = sorted(vis, key=lambda c: int(c.replace("Cam", "")))
    count = np.zeros_like(vis[cams[0]])
    for c in cams:
        count += (vis[c] > 0.5)
    heat_png(count[ys, xs], mask[ys, xs], out / "camera_count.png",
             f"cameras covering the texel (out of {len(cams)})",
             vmax=float(np.percentile(count[mask], 99.9)), cmap="viridis")
    v = count[mask]
    print(f"  camera_count.png      {len(cams)} camere; per texel: "
          f"media {v.mean():.1f}, mediana {np.median(v):.0f}, "
          f"p10 {np.percentile(v, 10):.0f}, p90 {np.percentile(v, 90):.0f}, "
          f"mai visti {100 * (v == 0).mean():.2f}%")


def tonemap(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalizza su un percentile della luminanza e applica una gamma 1/2.2.
    Restituisce anche il fattore, che va dichiarato in didascalia."""
    lum = img @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    k = float(np.percentile(lum[mask], PCTL))
    k = k if k > 1e-8 else 1.0
    o = np.clip(img / k, 0.0, 1.0) ** (1.0 / 2.2)
    o[~mask] = 0.0
    return o.astype(np.float32), k


def do_irradiance(run: Path, out: Path) -> None:
    box, mask = crop_to_atlas(run)
    ys, xs = box
    for name, fname in (("irradiance", "irradiance.exr"),
                        ("irradiance_indirect", "irradiance_indirect.exr")):
        p = run / "irradiance" / fname
        if not p.exists():
            print(f"  ⚠  manca {p}, salto")
            continue
        img = load_exr(p)
        rgb, k = tonemap(img, mask)
        save_png(rgb[ys, xs], out / f"{name}.png")
        lum = img @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        print(f"  {name + '.png':24s} normalizzata su p{PCTL} = {k:.4f}; "
              f"media {lum[mask].mean():.4f}, max {lum[mask].max():.3f}")


def do_pixel_change(run: Path, source: str, out: Path) -> None:
    """Le quattro statistiche del pass di color texture, su UNA esposizione condivisa.

    Condivisa perche' e' il punto della figura: `range` e' `max - min`, e con quattro
    normalizzazioni diverse quella relazione sparisce.  Il fattore si prende da un
    percentile di `color_max`, che e' la piu' luminosa delle quattro.

    La varianza viene mostrata come la sua RADICE: e' in radianza al quadrato, e sulla
    scala condivisa sarebbe un rettangolo nero.  La radice la riporta in unita' di
    radianza e la rende confrontabile con il range.
    """
    box, mask = crop_to_atlas(run)
    ys, xs = box
    pc = run / "sources" / source / "pixel_change"
    lum_w = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

    cmax = load_exr(pc / "color_max.exr")
    k = float(np.percentile((cmax @ lum_w)[mask], PCTL))
    k = k if k > 1e-8 else 1.0
    print(f"  esposizione condivisa: p{PCTL} di color_max = {k:.4f}")

    for name in ("color_min", "color_max", "color_range", "color_variance"):
        p = pc / f"{name}.exr"
        if not p.exists():
            print(f"  ⚠  manca {p}, salto")
            continue
        img = load_exr(p)
        if name == "color_variance":
            img = np.sqrt(np.maximum(img, 0.0))     # -> unita' di radianza
        rgb = np.clip(img / k, 0.0, 1.0) ** (1.0 / 2.2)
        rgb[~mask] = 0.0
        save_png(rgb[ys, xs].astype(np.float32), out / f"{name}.png")
        L = (img @ lum_w)[mask]
        print(f"  {name + '.png':22s} p50={np.median(L):8.4f}  p99={np.percentile(L, 99):9.4f}"
              f"  saturati {100.0 * (L > k).mean():5.2f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="cartella della run (contiene ium/, visibility/, ...)")
    ap.add_argument("--camera", default="render_Camera_Shell21_38",
                    help="stem della camera di riferimento per la maschera")
    ap.add_argument("--irradiance", action="store_true",
                    help="genera le mappe di irradiance invece di quelle di visibilita'")
    ap.add_argument("--pixel-change", action="store_true",
                    help="genera le quattro statistiche di color texture")
    ap.add_argument("--source", default="gt",
                    help="sorgente per --pixel-change (sources/{source}/)")
    ap.add_argument("--out", required=True, help="cartella di destinazione dei PNG")
    args = ap.parse_args()

    run, out = Path(args.run_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{run.name} → {out.resolve()}")
    if args.pixel_change:
        do_pixel_change(run, args.source, out)
    elif args.irradiance:
        do_irradiance(run, out)
    else:
        do_visibility(run, args.camera, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
