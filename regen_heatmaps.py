#!/usr/bin/env python3
"""regen_heatmaps.py — Rigenera le heatmap PNG dagli EXR già su disco.

Non richiede GPU, training o OptiX. Legge direttamente i file EXR prodotti da una
run precedente (nerf_render_images/iter_*/ per le prospettive NeRF, skybox_nerf_baked.exr
per il confronto skybox) e riscrive i PNG con le funzioni di nerf.metrics aggiornate.

Uso:
  python regen_heatmaps.py <run_dir> [gt_skybox.exr]

  <run_dir>      : cartella di una scena, es.
                   D:/tesi_output/heatmap_norm_diff/TableAndOtherInterior
  gt_skybox.exr  : (opzionale) path all'HDR GT per il confronto skybox; se omesso,
                   il confronto skybox viene saltato.

Esempi:
  conda run --no-capture-output -n nerfpytorch python -u regen_heatmaps.py ^
      D:/tesi_output/heatmap_norm_diff/TableAndOtherInterior ^
      C:/Users/adria/Documents/GitHub/Tesi/OptixProjectCMake/Scenes/TableAndOtherInterior/Blender/assets/hdri/wooden_studio_13_4k.exr

  # Solo per-vista (senza skybox):
  conda run --no-capture-output -n nerfpytorch python -u regen_heatmaps.py ^
      D:/tesi_output/heatmap_norm_diff/TableAndOtherInterior
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Loader EXR locale — replica _load_image_hw3_native senza dipendere da
# images_generator (che a import-time inizializza OptiX).
# ---------------------------------------------------------------------------

def _load_exr_hw3(path: str) -> np.ndarray:
    """Carica un EXR come (H, W, 3) float32, leggendo i canali R/G/B per nome."""
    import OpenEXR
    import Imath

    exr = OpenEXR.InputFile(path)
    dw  = exr.header()["dataWindow"]
    w   = dw.max.x - dw.min.x + 1
    h   = dw.max.y - dw.min.y + 1
    pt  = Imath.PixelType(Imath.PixelType.FLOAT)
    chs = exr.header()["channels"]

    if "R" in chs and "G" in chs and "B" in chs:
        r = np.frombuffer(exr.channel("R", pt), dtype=np.float32).reshape(h, w)
        g = np.frombuffer(exr.channel("G", pt), dtype=np.float32).reshape(h, w)
        b = np.frombuffer(exr.channel("B", pt), dtype=np.float32).reshape(h, w)
    else:
        key = next(iter(chs))
        ch  = np.frombuffer(exr.channel(key, pt), dtype=np.float32).reshape(h, w)
        r = g = b = ch

    return np.stack([r, g, b], axis=-1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    run_dir        = Path(sys.argv[1]).resolve()
    gt_skybox_path = Path(sys.argv[2]).resolve() if len(sys.argv) >= 3 else None

    if not run_dir.is_dir():
        print(f"Errore: {run_dir} non esiste o non è una directory.", flush=True)
        sys.exit(1)

    # Importa le funzioni di plot da nerf.metrics (non da images_generator)
    sys.path.insert(0, str(Path(__file__).parent))
    from nerf.metrics import plot_error_heatmap, plot_skybox_compare

    # ── Per-vista: ultima iter_* ─────────────────────────────────────────────
    render_root = run_dir / "nerf_render_images"
    if render_root.is_dir():
        iter_dirs = sorted(render_root.glob("iter_*"))
        if iter_dirs:
            iter_dir = iter_dirs[-1]   # usa solo l'ultima iterazione
            gt_files = sorted(iter_dir.glob("frame_*_gt.exr"))
            print(f"[regen] Per-vista: {len(gt_files)} frame in {iter_dir.name}", flush=True)

            for gt_path in gt_files:
                stem      = gt_path.stem.replace("_gt", "")   # "frame_NNN"
                pred_path = iter_dir / f"{stem}_pred.exr"
                if not pred_path.exists():
                    print(f"  ⚠  {pred_path.name} non trovato, skip", flush=True)
                    continue

                gt_np   = _load_exr_hw3(str(gt_path))
                pred_np = _load_exr_hw3(str(pred_path))
                out_png = str(iter_dir / f"{stem}_heatmap.png")
                plot_error_heatmap(pred_np, gt_np, out_png, title=stem)
                print(f"  ✓  {stem}_heatmap.png", flush=True)
        else:
            print(f"[regen] Nessuna iter_* trovata in {render_root}", flush=True)
    else:
        print(f"[regen] nerf_render_images/ non trovata in {run_dir} — skip per-vista", flush=True)

    # ── Skybox: skybox_nerf_baked.exr vs GT HDR ─────────────────────────────
    baked_path = run_dir / "skybox_nerf_baked.exr"
    if not baked_path.exists():
        print(f"[regen] skybox_nerf_baked.exr non trovato — skip skybox compare", flush=True)
    elif gt_skybox_path is None:
        print("[regen] GT skybox non specificato (argomento 2) — skip skybox compare", flush=True)
    elif not gt_skybox_path.exists():
        print(f"[regen] GT skybox non trovato: {gt_skybox_path} — skip", flush=True)
    else:
        print(f"[regen] Skybox compare: {baked_path.name} vs {gt_skybox_path.name}", flush=True)
        gt_sky    = _load_exr_hw3(str(gt_skybox_path))
        baked_sky = _load_exr_hw3(str(baked_path))
        sky_cmp_dir = run_dir / "skybox_compare"
        sky_cmp_dir.mkdir(exist_ok=True)
        out_png   = str(sky_cmp_dir / "skybox_heatmap.png")
        sky_title = (
            f"Skybox  baked NeRF ({baked_sky.shape[1]}x{baked_sky.shape[0]}) "
            f"-> GT ({gt_sky.shape[1]}x{gt_sky.shape[0]})"
        )
        plot_skybox_compare(gt_sky, baked_sky, out_png, title=sky_title)
        print(f"  ✓  skybox_compare/skybox_heatmap.png", flush=True)

    print("[regen] Fatto.", flush=True)


if __name__ == "__main__":
    main()
