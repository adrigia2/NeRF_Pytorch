"""inspect_spec_cone.py — Sfoglia i coni speculari già baked in spec_cone/.

Il bake scrive per ogni camera un `spec_cone/cam_{j:03d}.exr` con un canale RGB
per candidato, chiamato con la sua apertura (`cone_000_mirror`, `cone_005deg`,
… `cone_180deg`), che è direttamente la L_j(r) usata dal fit PBR. Il file è già
ispezionabile con un viewer che raggruppa i canali per prefisso (tev mostra un
layer per apertura, in ordine di apertura); questo script serve quando si vuole:

  - un colpo d'occhio su tutte le aperture insieme (`contact_sheet.png`), con
    ESPOSIZIONE CONDIVISA, così la progressione specchio → cono largo si legge
    davvero invece di essere riscalata via apertura per apertura;
  - i coni spacchettati in un EXR per apertura (`--unpack`), per aprirli con
    strumenti che non gestiscono i canali multipli.

Uso:
    python inspect_spec_cone.py <output_dir> [--cams 0,3,7] [--unpack]
                                [--no-preview] [--preview-size 1024]
                                [--preview-percentile 95]

Uscite in <output_dir>/spec_cone_view/cam_{j:03d}/:
    contact_sheet.png            griglia di tutte le aperture
    cone_000_mirror.exr,         solo con --unpack, un file per apertura,
    cone_005deg.exr, …           con lo stesso nome del canale nel bake

Non richiede GPU, OptiX né il checkpoint NeRF: legge solo gli EXR del bake.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import _paths  # noqa: F401
from images_generator import (  # noqa: E402
    DataLayer, ImageFormat, _save_layer, spec_cone_level_name,
)
from pbr_solver import read_cones  # noqa: E402


def save_contact_sheet(cones: np.ndarray, valid: np.ndarray, apertures: np.ndarray,
                       shape: "tuple[int, int]", path: Path, cam: int,
                       size: int = 1024, percentile: float = 95.0) -> None:
    """Griglia di tutte le aperture con ESPOSIZIONE CONDIVISA.

    La scala viene da un percentile del livello specchio sui texel validi e resta
    la stessa per tutti i pannelli: rinormalizzare ogni apertura nasconderebbe
    proprio la progressione che si vuole guardare (L è una media, quindi la
    luminosità media deve restare confrontabile a ogni apertura).

    Il default è il percentile 95 e non il 99 perché la distribuzione di L ha una
    coda HDR ripidissima (in una scena tipica p95 ≈ 0.8 e p99 ≈ 18):
    normalizzare sui picchi delle luci schiaccia nel nero tutto il resto della
    texture. Con 95 i highlight saturano, e sono highlight.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, W = shape
    K = cones.shape[1]
    stride = max(1, -(-H // size))
    imgs = [cones[:, k].reshape(H, W, 3)[::stride, ::stride] for k in range(K)]

    mirror = cones[valid, 0]
    nz = mirror[mirror > 0]
    scale = max(float(np.percentile(nz, percentile)) if nz.size else 1.0, 1e-6)

    ncols = min(5, K)
    nrows = -(-K // ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.4 * nrows))
    axs = np.atleast_1d(axs).ravel()
    for k, ax in enumerate(axs):
        if k >= K:
            ax.axis("off")
            continue
        ax.imshow(np.clip(imgs[k] / scale, 0.0, 1.0) ** (1 / 2.2))
        ax.set_title("specchio" if k == 0 else f"{apertures[k]:g}°", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"cam {cam:03d} — L_j(r), esposizione condivisa "
                 f"(p{percentile:g} specchio = {scale:.4g})", fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output_dir", help="cartella di una scena (contiene spec_cone/)")
    ap.add_argument("--cams", default="", help="indici camera separati da virgola (default: tutte)")
    ap.add_argument("--unpack", action="store_true",
                    help="scrive anche un EXR per apertura (full-res, ~39 MB l'uno)")
    ap.add_argument("--no-preview", action="store_true", help="niente contact sheet")
    ap.add_argument("--preview-size", type=int, default=1024,
                    help="lato massimo delle preview nel contact sheet (default 1024)")
    ap.add_argument("--preview-percentile", type=float, default=95.0,
                    help="percentile del livello specchio che fissa l'esposizione "
                         "condivisa del contact sheet (default 95)")
    args = ap.parse_args()

    out = Path(args.output_dir)
    spec_dir = out / "spec_cone"
    meta_path = spec_dir / "spec_cone_meta.json"
    if not meta_path.exists():
        sys.exit(f"{meta_path} non trovato: il bake spec_cone non è finito "
                 f"(il meta viene scritto solo alla fine) oppure output_dir è sbagliata.")

    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    if meta.get("format") != "cones":
        sys.exit(f"formato {meta.get('format')!r} non supportato: serve un bake "
                 f"'cones'. I bake 'rings'/'rings_shared' (medie per anello) sono "
                 f"precedenti allo spostamento della chiusura dei coni nel bake e "
                 f"vanno rifatti.")

    cams = ([int(x) for x in args.cams.split(",") if x.strip()]
            if args.cams else list(meta["cameras"]))
    unknown = set(cams) - set(meta["cameras"])
    if unknown:
        sys.exit(f"camere non presenti nel bake: {sorted(unknown)}")

    K = int(meta["num_levels"])
    apertures = np.asarray(meta["apertures_deg"], dtype=np.float64)
    # stessi nomi dei canali nel bake: un solo nome per livello, ovunque
    labels = [spec_cone_level_name(apertures, k) for k in range(K)]
    view_root = out / "spec_cone_view"

    # la risoluzione della texture viene dall'IUM: il bake ci scrive dentro
    ium_mask = out / "ium" / "ium_masks.exr"
    if not ium_mask.exists():
        sys.exit(f"{ium_mask} non trovato: serve per conoscere la risoluzione IUM.")
    from pbr_solver import _read_exr
    H, W = _read_exr(ium_mask).shape[:2]

    print(f"[spec_cone] scheme={meta.get('scheme')}, {K} livelli "
          f"(specchio + {K - 1} coni), aperture {apertures[1:].tolist()}°")
    print(f"[spec_cone] {len(cams)} camere, texture {W}×{H} → {view_root}")

    from tqdm import tqdm
    for cam in tqdm(cams, unit="cam", desc="coni"):
        cones, n_valid = read_cones(
            spec_dir / meta["cam_file_pattern"].format(cam=cam), apertures)
        valid = n_valid > 0
        cam_dir = view_root / f"cam_{cam:03d}"
        if args.unpack:
            cam_dir.mkdir(parents=True, exist_ok=True)
            for k in range(K):
                _save_layer(cones[:, k].reshape(H, W, 3),
                            (cam_dir / f"{labels[k]}.exr").as_posix(),
                            ImageFormat.OPENEXR, DataLayer.SPEC_CONE)
        if not args.no_preview:
            save_contact_sheet(cones, valid, apertures, (H, W),
                               cam_dir / "contact_sheet.png", cam,
                               size=args.preview_size,
                               percentile=args.preview_percentile)

    print(f"✓ fatto: {view_root}")


if __name__ == "__main__":
    main()
