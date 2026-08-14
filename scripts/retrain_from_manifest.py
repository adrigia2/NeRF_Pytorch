#!/usr/bin/env python
"""Retrain the scenes of an existing sweep into a new folder, changing a few parameters.

It answers questions of the form "does the NeRF still train without this
hyper-parameter?" without touching the reference runs: the configuration is rebuilt
from each scene's `run_manifest.json`, so the only difference between the new arm and
the old one is whatever is overridden explicitly on the command line.
line. Transcribing the config by hand would invalidate the comparison.

Only Step 1 and Step 2 are run (plus Step 2b when the manifest calls for it): the
texture-space reconstruction is beside the point and would cost hours.

Usage:
    python retrain_from_manifest.py <src_root> <dst_root> \
        --iters 10000 --raw-noise-std 0.0 [--configs exp_l1_d02 ...] [--scenes ...]

`src_root` is the root of a sweep, i.e. <root>/<config>/<scene>/run_manifest.json;
the same structure is replicated under `dst_root`. The runs go in SEQUENCE: there is
only one GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import _paths  # noqa: F401

from images_generator import (  # noqa: E402
    SceneConfig,
    _console_to_file,
    _write_run_manifest,
    run_pipeline,
)
from roi_rerun import config_from_manifest  # noqa: E402


def find_runs(src: Path, configs, scenes) -> "list[tuple[str, str, Path]]":
    """(config, scene, manifest) for every run found under src_root."""
    out = []
    for cfg_dir in sorted(p for p in src.glob("*") if p.is_dir()):
        if configs and cfg_dir.name not in configs:
            continue
        for scene_dir in sorted(p for p in cfg_dir.glob("*") if p.is_dir()):
            if scenes and scene_dir.name not in scenes:
                continue
            man = scene_dir / "run_manifest.json"
            if man.exists():
                out.append((cfg_dir.name, scene_dir.name, man))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src_root", help="root of the reference sweep")
    ap.add_argument("dst_root", help="the new root (it gets created)")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--raw-noise-std", type=float, default=0.0)
    ap.add_argument("--configs", nargs="*", default=None, help="filter the configs")
    ap.add_argument("--scenes", nargs="*", default=None, help="filter the scenes")
    ap.add_argument("--step2b", choices=("keep", "on", "off"), default="keep",
                    help="render the training frames with the model (default: as in the manifest)")
    ap.add_argument("--note", default="", help="note to store in the new manifest")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src, dst = Path(args.src_root).resolve(), Path(args.dst_root).resolve()
    runs = find_runs(src, set(args.configs or []), set(args.scenes or []))
    if not runs:
        print(f"✗ no run_manifest.json under {src}")
        return 2

    note = args.note or (f"{args.iters}iter | raw_noise_std={args.raw_noise_std} "
                         f"| rerun da {src.name}")
    print(f"{len(runs)} run da riallenare, {args.iters} iterazioni, "
          f"raw_noise_std={args.raw_noise_std}")
    for c, s, _ in runs:
        print(f"    {c}/{s}")
    print(f"  destinazione: {dst}\n")

    statuses: dict[str, str] = {}
    t_start = time.perf_counter()

    for idx, (conf, scene_name, man) in enumerate(runs, 1):
        key = f"{conf}/{scene_name}"
        out_dir = dst / conf / scene_name
        cfg, manifest, notes = config_from_manifest(man)

        # Override: only the steps, the length, the hyper-parameter under test and the paths.
        cfg.run_step1 = cfg.run_step2 = True
        cfg.run_step3 = cfg.run_step4 = False
        cfg.resume_skip_step2_if_ckpt = False
        cfg.nerf_interactive_loop = False
        cfg.nerf_num_iters = args.iters
        cfg.nerf_raw_noise_std = args.raw_noise_std
        if args.step2b != "keep":
            cfg.enable_nerf_render_train_images = (args.step2b == "on")
        # The derived paths must start from output_dir, not from the old manifest.
        cfg.nerf_ckpt_path = ""
        cfg.nerf_train_output_dir = ""
        cfg.nerf_render_train_images_dir = ""
        cfg.render.output_dir = str(out_dir)

        print(f"{'=' * 70}")
        print(f"  [{idx}/{len(runs)}] {key}")
        print(f"  output   : {out_dir}")
        print(f"  iters={cfg.nerf_num_iters}  raw_noise_std={cfg.nerf_raw_noise_std}  "
              f"act={cfg.nerf_rgb_activation}  loss={cfg.nerf_loss_type}  "
              f"lr_decay_steps={cfg.nerf_lr_decay_steps}  step2b={cfg.enable_nerf_render_train_images}")
        for n in notes:
            print(n)
        print(f"{'=' * 70}")
        if args.dry_run:
            statuses[key] = "dry-run"
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        scene = SceneConfig(**manifest["scene"])
        _write_run_manifest(cfg, scene, note)

        t0 = time.perf_counter()
        try:
            with _console_to_file(str(out_dir / "console.log")):
                run_pipeline(cfg, tb_enabled=False)
            statuses[key] = f"ok ({(time.perf_counter() - t0) / 60:.1f} min)"
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            statuses[key] = f"error: {exc}"

    print(f"\n{'=' * 70}")
    print(f"  Riepilogo ({(time.perf_counter() - t_start) / 60:.1f} min totali):")
    for k, v in statuses.items():
        print(f"    {'✓' if v.startswith(('ok', 'dry')) else '✗'} {k}: {v}")
    print(f"{'=' * 70}")
    return 0 if all(v.startswith(("ok", "dry")) for v in statuses.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
