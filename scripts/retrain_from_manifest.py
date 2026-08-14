#!/usr/bin/env python
"""Riallena le scene di uno sweep esistente in una cartella nuova, cambiando pochi parametri.

Serve a rispondere a domande del tipo "il NeRF si allena anche senza questo
iperparametro?" senza toccare le run di riferimento: la configurazione viene
ricostruita dal `run_manifest.json` di ogni scena, così l'unica differenza fra il
braccio nuovo e quello vecchio è ciò che si sovrascrive esplicitamente da riga di
comando. Ritrascrivere la config a mano invaliderebbe il confronto.

Esegue solo Step 1 e Step 2 (più lo Step 2b se il manifest lo prevede): la
ricostruzione texture-space non c'entra e costerebbe ore.

Uso:
    python retrain_from_manifest.py <src_root> <dst_root> \
        --iters 10000 --raw-noise-std 0.0 [--configs exp_l1_d02 ...] [--scenes ...]

`src_root` è la radice di uno sweep, cioè <root>/<config>/<scena>/run_manifest.json;
la stessa struttura viene replicata sotto `dst_root`. Le run girano in SEQUENZA:
la GPU è una sola.
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
    """(config, scena, manifest) per ogni run trovata sotto src_root."""
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
    ap.add_argument("src_root", help="radice dello sweep di riferimento")
    ap.add_argument("dst_root", help="radice nuova (viene creata)")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--raw-noise-std", type=float, default=0.0)
    ap.add_argument("--configs", nargs="*", default=None, help="filtra le config")
    ap.add_argument("--scenes", nargs="*", default=None, help="filtra le scene")
    ap.add_argument("--step2b", choices=("keep", "on", "off"), default="keep",
                    help="render dei frame di training col modello (default: come il manifest)")
    ap.add_argument("--note", default="", help="nota da salvare nel manifest nuovo")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src, dst = Path(args.src_root).resolve(), Path(args.dst_root).resolve()
    runs = find_runs(src, set(args.configs or []), set(args.scenes or []))
    if not runs:
        print(f"✗ nessun run_manifest.json sotto {src}")
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

        # Override: solo gli step, la durata, l'iperparametro in prova e i path.
        cfg.run_step1 = cfg.run_step2 = True
        cfg.run_step3 = cfg.run_step4 = False
        cfg.resume_skip_step2_if_ckpt = False
        cfg.nerf_interactive_loop = False
        cfg.nerf_num_iters = args.iters
        cfg.nerf_raw_noise_std = args.raw_noise_std
        if args.step2b != "keep":
            cfg.enable_nerf_render_train_images = (args.step2b == "on")
        # I path derivati devono ripartire da output_dir, non dal manifest vecchio.
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
            statuses[key] = f"errore: {exc}"

    print(f"\n{'=' * 70}")
    print(f"  Riepilogo ({(time.perf_counter() - t_start) / 60:.1f} min totali):")
    for k, v in statuses.items():
        print(f"    {'✓' if v.startswith(('ok', 'dry')) else '✗'} {k}: {v}")
    print(f"{'=' * 70}")
    return 0 if all(v.startswith(("ok", "dry")) for v in statuses.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
