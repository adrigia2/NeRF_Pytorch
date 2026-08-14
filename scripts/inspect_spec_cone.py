"""inspect_spec_cone.py — browse the specular cones already baked into spec_cone/.

For each camera the bake writes a `spec_cone/cam_{j:03d}.exr` with one RGB channel per
candidate, named after its aperture (`cone_000_mirror`, `cone_005deg`, … `cone_180deg`),
which is directly the L_j(r) the PBR fit uses. The file can already be inspected with a
viewer that groups channels by prefix (tev shows one layer per aperture, in aperture
order); this script is for when one wants:

  - a single glance at every aperture together (`contact_sheet.png`), with a SHARED
    EXPOSURE, so that the mirror → wide-cone progression really reads instead of being
    rescaled away aperture by aperture;
  - the cones unpacked into one EXR per aperture (`--unpack`), to open them with tools
    that do not handle multiple channels.

Uso:
    python inspect_spec_cone.py <output_dir> [--cams 0,3,7] [--unpack]
                                [--no-preview] [--preview-size 1024]
                                [--preview-percentile 95]

Uscite in <output_dir>/spec_cone_view/cam_{j:03d}/:
    contact_sheet.png            grid of every aperture
    cone_000_mirror.exr,         only with --unpack, one file per aperture,
    cone_005deg.exr, …           with the same name as the channel in the bake

Needs no GPU, no OptiX and no NeRF checkpoint: it only reads the bake's EXRs.
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
    """Grid of every aperture with a SHARED EXPOSURE.

    The scale comes from a percentile of the mirror level over the valid texels and stays
    the same for every panel: renormalising each aperture would hide precisely the
    progression one wants to look at (L is a mean, so the mean brightness has to stay
    comparable at every aperture).

    The default is the 95th percentile and not the 99th because the distribution of L has
    a very steep HDR tail (in a typical scene p95 ≈ 0.8 and p99 ≈ 18): normalising on the
    peaks of the lights crushes all the rest of the texture into black. At 95 the
    highlights saturate, and they are highlights."""
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
    ap.add_argument("output_dir", help="a scene folder (holds spec_cone/)")
    ap.add_argument("--cams", default="", help="comma-separated camera indices (default: all)")
    ap.add_argument("--unpack", action="store_true",
                    help="also write one EXR per aperture (full-res, ~39 MB each)")
    ap.add_argument("--no-preview", action="store_true", help="no contact sheet")
    ap.add_argument("--preview-size", type=int, default=1024,
                    help="maximum side of the previews in the contact sheet (default 1024)")
    ap.add_argument("--preview-percentile", type=float, default=95.0,
                    help="percentile of the mirror level that sets the shared "
                         "exposure of the contact sheet (default 95)")
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
        sys.exit(f"format {meta.get('format')!r} not supported: a bake in the "
                 f"'cones' format is required. The 'rings'/'rings_shared' bakes "
                 f"(per-ring means) predate the move of the cone closing into the "
                 f"bake and have to be redone.")

    cams = ([int(x) for x in args.cams.split(",") if x.strip()]
            if args.cams else list(meta["cameras"]))
    unknown = set(cams) - set(meta["cameras"])
    if unknown:
        sys.exit(f"camere non presenti nel bake: {sorted(unknown)}")

    K = int(meta["num_levels"])
    apertures = np.asarray(meta["apertures_deg"], dtype=np.float64)
    # the same channel names as in the bake: one name per level, everywhere
    labels = [spec_cone_level_name(apertures, k) for k in range(K)]
    view_root = out / "spec_cone_view"

    # the texture resolution comes from the IUM: the bake writes into it
    ium_mask = out / "ium" / "ium_masks.exr"
    if not ium_mask.exists():
        sys.exit(f"{ium_mask} non trovato: serve per conoscere la risoluzione IUM.")
    from pbr_solver import _read_exr
    H, W = _read_exr(ium_mask).shape[:2]

    print(f"[spec_cone] scheme={meta.get('scheme')}, {K} levels "
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
