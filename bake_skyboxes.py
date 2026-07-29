#!/usr/bin/env python3
"""bake_skyboxes.py — Bake della skybox NeRF per tutte le run di uno sweep.

Produce lo stesso `skybox_nerf_baked.exr` dello Step 3 senza eseguire nulla del resto
della pipeline texture-space (niente OptiX, niente mesh, niente transforms.json):
serve solo il checkpoint `<run_dir>/model/nerf_model_cache.pt`, da cui si legge la
bg-sphere del NeRF. Il bake è di qualche decina di secondi per run a 4096x2048.

Se una skybox HDR ground-truth è disponibile (opzione --gt oppure skybox_path nel
run_manifest.json della run), genera anche `<run_dir>/skybox_compare/skybox_heatmap.png`,
identico a quello prodotto dallo Step 3 con compare_skybox_to_gt=True.

Uso:
  python bake_skyboxes.py <root> [--gt GT.exr] [--force] [--size W H] [--yaw DEG] [--dry-run]

  <root>    : root di uno sweep (contiene <tag>/<scena>/model/...) oppure una singola run dir.
  --gt      : HDR equirettangolare di riferimento per la heatmap di confronto.
  --force   : rifà bake e heatmap anche dove i file esistono già (default: skip).
  --size    : override di skybox_size; default = quello del run_manifest.json (o 4096 2048).
  --yaw     : override di skybox_yaw_degrees; default = quello del manifest (o 0.0).
  --dry-run : elenca le run trovate e cosa verrebbe fatto, senza caricare torch.

Esempi:
  conda run --no-capture-output -n nerfpytorch python -u bake_skyboxes.py ^
      D:/tesi_output/sweep_nerf_activation_loss_decay_find_better_nerf ^
      --gt C:/Users/adria/Documents/GitHub/Tesi/OptixProjectCMake/Scenes/TableAndOtherInterior/Blender/assets/hdri/wooden_studio_13_4k.exr

  python bake_skyboxes.py D:/tesi_output/sweep_.../exp_l1_d02/TableAndOtherInteriorWithSpecular
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

CKPT_REL       = Path("model") / "nerf_model_cache.pt"
BAKED_NAME     = "skybox_nerf_baked.exr"
HEATMAP_REL    = Path("skybox_compare") / "skybox_heatmap.png"
DEFAULT_SIZE   = [4096, 2048]
DEFAULT_YAW    = 0.0


# ---------------------------------------------------------------------------
# Discovery e parametri per-run
# ---------------------------------------------------------------------------

def find_run_dirs(root: Path) -> list[Path]:
    """Run dir (parent di model/nerf_model_cache.pt) sotto root, a profondità 0/1/2."""
    found = set()
    for pattern in (CKPT_REL, Path("*") / CKPT_REL, Path("*") / "*" / CKPT_REL):
        for ckpt in root.glob(pattern.as_posix()):
            found.add(ckpt.parent.parent)
    return sorted(found)


def _read_manifest(run_dir: Path) -> dict:
    path = run_dir / "run_manifest.json"
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  ⚠  run_manifest.json illeggibile ({exc}) — uso i default", flush=True)
        return {}


def resolve_params(run_dir: Path, size_override, yaw_override) -> tuple[int, int, float, str]:
    """(width, height, yaw_degrees, gt_path) per la run, dal manifest con override CLI."""
    manifest = _read_manifest(run_dir)
    render   = manifest.get("config", {}).get("render", {})
    scene    = manifest.get("scene", {})

    size = size_override or render.get("skybox_size") or DEFAULT_SIZE
    yaw  = yaw_override if yaw_override is not None else render.get("skybox_yaw_degrees", DEFAULT_YAW)
    gt   = render.get("skybox_path") or scene.get("skybox_path") or ""

    return int(size[0]), int(size[1]), float(yaw), gt


# ---------------------------------------------------------------------------
# Passi per-run
# ---------------------------------------------------------------------------

def bake_one(run_dir: Path, width: int, height: int, yaw: float) -> None:
    """Scrive <run_dir>/skybox_nerf_baked.exr riusando il bake della pipeline."""
    from images_generator import RenderConfig, _bake_skybox_from_nerf

    rc = RenderConfig(
        transforms_path    = "",          # non letti sul percorso di bake
        model_path         = "",
        output_dir         = str(run_dir),
        skybox_source      = "nerf",
        skybox_size        = [width, height],
        skybox_yaw_degrees = yaw,
    )
    _bake_skybox_from_nerf(rc, width, height, run_dir)


def compare_one(run_dir: Path, gt_path: Path) -> None:
    """Scrive <run_dir>/skybox_compare/skybox_heatmap.png (baked vs GT HDR)."""
    from nerf.metrics import plot_skybox_compare
    from regen_heatmaps import _load_exr_hw3

    baked_sky = _load_exr_hw3(str(run_dir / BAKED_NAME))
    gt_sky    = _load_exr_hw3(str(gt_path))

    out_png = run_dir / HEATMAP_REL
    out_png.parent.mkdir(exist_ok=True)
    title = (
        f"Skybox  baked NeRF ({baked_sky.shape[1]}x{baked_sky.shape[0]}) "
        f"-> GT ({gt_sky.shape[1]}x{gt_sky.shape[0]})"
    )
    plot_skybox_compare(gt_sky, baked_sky, str(out_png), title=title)
    print(f"  ✓  {HEATMAP_REL.as_posix()}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bake della skybox NeRF per tutte le run di uno sweep (solo checkpoint, niente Step 3).")
    parser.add_argument("root", type=Path, help="root dello sweep o singola run dir")
    parser.add_argument("--gt", type=Path, default=None,
                        help="HDR ground-truth per la heatmap di confronto")
    parser.add_argument("--force", action="store_true",
                        help="rifà bake e heatmap anche se i file esistono")
    parser.add_argument("--size", type=int, nargs=2, metavar=("W", "H"), default=None,
                        help="override di skybox_size (default: dal run_manifest.json)")
    parser.add_argument("--yaw", type=float, default=None,
                        help="override di skybox_yaw_degrees (default: dal run_manifest.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="elenca cosa verrebbe fatto senza caricare torch")
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        print(f"Errore: {root} non esiste o non è una directory.", flush=True)
        return 1

    if args.gt is not None and not args.gt.exists():
        print(f"Errore: GT skybox non trovato: {args.gt}", flush=True)
        return 1

    run_dirs = find_run_dirs(root)
    if not run_dirs:
        print(f"Nessun checkpoint {CKPT_REL.as_posix()} trovato sotto {root}", flush=True)
        return 1

    print(f"[bake] {len(run_dirs)} run trovate sotto {root}", flush=True)

    # Le funzioni della pipeline stanno accanto a questo file.
    sys.path.insert(0, str(Path(__file__).parent))

    n_baked = n_skipped = n_failed = 0

    for run_dir in run_dirs:
        label = run_dir.relative_to(root).as_posix() if run_dir != root else run_dir.name
        width, height, yaw, gt_manifest = resolve_params(run_dir, args.size, args.yaw)

        gt_path = args.gt if args.gt is not None else (Path(gt_manifest) if gt_manifest else None)
        gt_ok   = gt_path is not None and gt_path.exists()

        baked_exists   = (run_dir / BAKED_NAME).exists()
        heatmap_exists = (run_dir / HEATMAP_REL).exists()

        print(f"\n[bake] {label}  ({width}x{height}, yaw={yaw}°)", flush=True)

        if args.dry_run:
            print(f"  bake:    {'skip (esiste)' if baked_exists and not args.force else 'da fare'}", flush=True)
            if gt_ok:
                print(f"  heatmap: {'skip (esiste)' if heatmap_exists and not args.force else 'da fare'}"
                      f"  vs {gt_path.name}", flush=True)
            elif gt_path is not None:
                print(f"  heatmap: skip (GT non trovato: {gt_path})", flush=True)
            else:
                print("  heatmap: skip (nessuna GT: usa --gt)", flush=True)
            continue

        try:
            if baked_exists and not args.force:
                print(f"  ↻  {BAKED_NAME} già presente — skip bake (--force per rifarlo)", flush=True)
                n_skipped += 1
            else:
                bake_one(run_dir, width, height, yaw)
                n_baked += 1

            if not gt_ok:
                if gt_path is not None:
                    print(f"  ⚠  GT non trovato ({gt_path}) — skip heatmap", flush=True)
                else:
                    print("  •  nessuna GT disponibile — skip heatmap (usa --gt)", flush=True)
            elif heatmap_exists and not args.force:
                print(f"  ↻  {HEATMAP_REL.as_posix()} già presente — skip", flush=True)
            else:
                compare_one(run_dir, gt_path)

        except Exception as exc:      # una run rotta non deve fermare le altre
            n_failed += 1
            print(f"  ✗  FALLITA: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()

    if args.dry_run:
        print("\n[bake] dry-run: nessun file scritto.", flush=True)
        return 0

    print(f"\n[bake] Riepilogo: {n_baked} bakate, {n_skipped} saltate, {n_failed} fallite "
          f"su {len(run_dirs)} run.", flush=True)
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
