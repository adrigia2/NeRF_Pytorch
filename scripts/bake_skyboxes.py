#!/usr/bin/env python3
"""bake_skyboxes.py — bake the NeRF skybox for every run of a sweep.

Produces the same `skybox_nerf_baked.exr` as Step 3 without running any of the rest of
the texture-space pipeline (no OptiX, no mesh, no transforms.json): all it needs is the
checkpoint `<run_dir>/model/nerf_model_cache.pt`, from which the NeRF background sphere
is read. The bake takes a few tens of seconds per run at 4096x2048.

When a ground-truth HDR skybox is available (the --gt option, or skybox_path in the
run's run_manifest.json), it also writes `<run_dir>/skybox_compare/skybox_heatmap.png`,
identical to the one Step 3 produces with compare_skybox_to_gt=True.

Usage:
  python bake_skyboxes.py <root> [--gt GT.exr] [--force] [--size W H] [--yaw DEG] [--dry-run]

  <root>    : the root of a sweep (holding <tag>/<scene>/model/...) or a single run dir.
  --gt      : reference equirectangular HDR for the comparison heatmap.
  --force   : redo bake and heatmap even where the files exist (default: skip).
  --size    : override skybox_size; default = the one in run_manifest.json (or 4096 2048).
  --yaw     : override skybox_yaw_degrees; default = the manifest's (or 0.0).
  --dry-run : list the runs found and what would be done, without importing torch.

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
# Discovery and per-run parameters
# ---------------------------------------------------------------------------

def find_run_dirs(root: Path) -> list[Path]:
    """Run dirs (the parent of model/nerf_model_cache.pt) under root, at depth 0/1/2."""
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
    """(width, height, yaw_degrees, gt_path) for the run, from the manifest with CLI overrides."""
    manifest = _read_manifest(run_dir)
    render   = manifest.get("config", {}).get("render", {})
    scene    = manifest.get("scene", {})

    size = size_override or render.get("skybox_size") or DEFAULT_SIZE
    yaw  = yaw_override if yaw_override is not None else render.get("skybox_yaw_degrees", DEFAULT_YAW)
    gt   = render.get("skybox_path") or scene.get("skybox_path") or ""

    return int(size[0]), int(size[1]), float(yaw), gt


# ---------------------------------------------------------------------------
# Per-run steps
# ---------------------------------------------------------------------------

def bake_one(run_dir: Path, width: int, height: int, yaw: float) -> None:
    """Write <run_dir>/skybox_nerf_baked.exr, reusing the pipeline's bake."""
    from images_generator import RenderConfig, _bake_skybox_from_nerf

    rc = RenderConfig(
        transforms_path    = "",          # not read on the bake path
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
        description="Bake the NeRF skybox for every run of a sweep (checkpoint only, no Step 3).")
    parser.add_argument("root", type=Path, help="root dello sweep o singola run dir")
    parser.add_argument("--gt", type=Path, default=None,
                        help="ground-truth HDR for the comparison heatmap")
    parser.add_argument("--force", action="store_true",
                        help="redo bake and heatmap even when the files exist")
    parser.add_argument("--size", type=int, nargs=2, metavar=("W", "H"), default=None,
                        help="override skybox_size (default: from run_manifest.json)")
    parser.add_argument("--yaw", type=float, default=None,
                        help="override skybox_yaw_degrees (default: from run_manifest.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be done without importing torch")
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        print(f"Error: {root} does not exist, or is not a directory.", flush=True)
        return 1

    if args.gt is not None and not args.gt.exists():
        print(f"Error: GT skybox not found: {args.gt}", flush=True)
        return 1

    run_dirs = find_run_dirs(root)
    if not run_dirs:
        print(f"No {CKPT_REL.as_posix()} checkpoint found under {root}", flush=True)
        return 1

    print(f"[bake] {len(run_dirs)} runs found under {root}", flush=True)

    # The pipeline functions live next to this file.
    import _paths  # noqa: F401

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
            print(f"  bake:    {'skip (exists)' if baked_exists and not args.force else 'to do'}", flush=True)
            if gt_ok:
                print(f"  heatmap: {'skip (exists)' if heatmap_exists and not args.force else 'to do'}"
                      f"  vs {gt_path.name}", flush=True)
            elif gt_path is not None:
                print(f"  heatmap: skip (GT not found: {gt_path})", flush=True)
            else:
                print("  heatmap: skip (nessuna GT: usa --gt)", flush=True)
            continue

        try:
            if baked_exists and not args.force:
                print(f"  ↻  {BAKED_NAME} already present — skipping the bake (--force to redo it)", flush=True)
                n_skipped += 1
            else:
                bake_one(run_dir, width, height, yaw)
                n_baked += 1

            if not gt_ok:
                if gt_path is not None:
                    print(f"  ⚠  GT not found ({gt_path}) — skipping the heatmap", flush=True)
                else:
                    print("  •  nessuna GT disponibile — skip heatmap (usa --gt)", flush=True)
            elif heatmap_exists and not args.force:
                print(f"  ↻  {HEATMAP_REL.as_posix()} already present — skipped", flush=True)
            else:
                compare_one(run_dir, gt_path)

        except Exception as exc:      # one broken run must not stop the others
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
