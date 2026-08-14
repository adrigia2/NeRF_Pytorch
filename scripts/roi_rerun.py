#!/usr/bin/env python
"""Rilancia Step 3 + Step 4 di una run già esistente restringendoli a una ROI.

Serve a verificare la ROI in spazio texture: la run piena su disco fa da
riferimento, questo script la ri-esegue su una porzione e lascia il risultato in
`<run_dir>/roi/<tag>/`, dopodiché `compare_roi_run.py` confronta i due alberi.

La configurazione **non** viene ritrascritta a mano ma ricostruita da
`run_manifest.json`: una sola differenza (un'apertura, un numero di campioni, la
finestra di profondità) renderebbe il confronto privo di significato. Lo script
verifica esplicitamente che la config ricostruita coincida con quella del
manifest su ogni chiave tranne quelle che deve cambiare, e si ferma se non è così.

Uso:
    python roi_rerun.py <run_dir> --rect X0 Y0 W H [--tag NOME] [--mask FILE]

Chiama `run_pipeline` direttamente e non `run_pipeline_multi`, che riscriverebbe
`run_manifest.json` e farebbe append sul `console.log` della run di riferimento.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, fields
from enum import Enum
from pathlib import Path

import _paths  # noqa: F401

from images_generator import (  # noqa: E402
    ImageFormat,
    PipelineConfig,
    RenderConfig,
    _console_to_file,
    _roi_assets_dir,
    run_pipeline,
)

# Chiavi che questo script cambia di proposito: sono le uniche differenze
# ammesse rispetto al manifest della run di riferimento.
_EXPECTED_DIFFS = {
    "run_step1", "run_step2", "run_step3", "run_step4",
    "render.output_dir",
    "render.roi_rect", "render.roi_mask_path", "render.roi_mask_threshold",
    "render.roi_tag",
    # Campo CANCELLATO dalla config, non un override: il gate diffuso del solver
    # e' stato rimosso, ma i manifest gia' scritti lo contengono ancora e senza
    # questa voce ogni run precedente verrebbe rifiutata.
    "render.pbr_diffuse_cv_gate",
}


def _enc(o: object) -> object:
    """Stessa codifica di _write_run_manifest, per poter confrontare i dict."""
    if isinstance(o, Enum):
        return o.name
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _encode_config(cfg: PipelineConfig) -> dict:
    return json.loads(json.dumps(asdict(cfg), default=_enc))


def _build(cls, data: dict, path: str) -> tuple[object, list[str]]:
    """Istanzia la dataclass `cls` dai valori del manifest.

    Gli ImageFormat tornano dal nome (il manifest li serializza con Enum.name).
    Le chiavi del manifest che la dataclass non ha più, e i campi aggiunti dopo
    quella run (tipicamente i roi_*), vengono segnalati invece che ignorati in
    silenzio: la prima categoria significa che la config è cambiata sotto i piedi.
    """
    notes: list[str] = []
    known = {f.name: f for f in fields(cls)}
    kwargs = {}
    for name, value in data.items():
        f = known.get(name)
        if f is None:
            notes.append(f"    ⚠  {path}{name}: presente nel manifest ma non in {cls.__name__}")
            continue
        if isinstance(f.default, ImageFormat):
            value = ImageFormat[value] if isinstance(value, str) else value
        kwargs[name] = value
    for name in known:
        if name not in data and name != "render":
            notes.append(f"    ·  {path}{name}: assente dal manifest, uso il default")
    return cls(**kwargs), notes


def config_from_manifest(manifest_path: Path) -> tuple[PipelineConfig, dict, list[str]]:
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    raw = dict(manifest["config"])
    render_raw = raw.pop("render")
    rc, notes_r = _build(RenderConfig, render_raw, "render.")
    # `render` va passato al costruttore: il suo default_factory è RenderConfig,
    # che ha tre campi obbligatori e quindi non è istanziabile a vuoto.
    raw["render"] = rc
    cfg, notes_c = _build(PipelineConfig, raw, "")
    return cfg, manifest, notes_c + notes_r


def check_against_manifest(cfg: PipelineConfig, manifest: dict,
                           expected_diffs: "set[str] | None" = None) -> list[str]:
    """Differenze fra la config ricostruita e quella del manifest, esclusi gli override.

    `expected_diffs` è l'insieme delle chiavi che il chiamante cambia di proposito;
    di default quelle della ROI. Lo parametrizza `rerun_irradiance.py`, che riusa
    questa validazione con un insieme di override diverso.
    """
    if expected_diffs is None:
        expected_diffs = _EXPECTED_DIFFS
    got, want = _encode_config(cfg), manifest["config"]
    diffs = []

    def walk(a: dict, b: dict, prefix: str) -> None:
        for k in sorted(set(a) | set(b)):
            key = f"{prefix}{k}"
            va, vb = a.get(k, "<assente>"), b.get(k, "<assente>")
            if isinstance(va, dict) and isinstance(vb, dict):
                walk(va, vb, f"{key}.")
            elif vb == "<assente>":
                # Campo aggiunto alla dataclass DOPO quella run: il manifest non
                # può contenerlo e la ricostruzione usa il default, cosa che
                # `_build` ha già segnalato fra le notes. Contarlo come divergenza
                # renderebbe inutilizzabile lo script su ogni run precedente
                # all'ultimo campo introdotto, cioè proprio il caso d'uso.
                continue
            elif va != vb and key not in expected_diffs:
                diffs.append(f"{key}: ricostruita={va!r}  manifest={vb!r}")

    walk(got, want, "")
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="cartella della run di riferimento (contiene run_manifest.json)")
    ap.add_argument("--rect", nargs=4, type=int, metavar=("X0", "Y0", "W", "H"),
                    help="ROI rettangolare in texel IUM")
    ap.add_argument("--mask", default="", help="immagine maschera della ROI (opzionale)")
    ap.add_argument("--mask-threshold", type=float, default=0.5)
    ap.add_argument("--tag", default="", help="nome della sandbox (default: derivato)")
    ap.add_argument("--dry-run", action="store_true",
                    help="ricostruisce e verifica la config, poi si ferma")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        print(f"✗ manifest non trovato: {manifest_path}")
        return 2
    if not args.rect and not args.mask:
        print("✗ serve almeno --rect o --mask")
        return 2

    cfg, manifest, notes = config_from_manifest(manifest_path)
    print(f"Config ricostruita da {manifest_path}")
    print(f"  run originale: {manifest['timestamp']}  |  {manifest.get('run_note', '')}")
    for n in notes:
        print(n)

    # Override: solo gli step di ricostruzione, la ROI, e la cartella (la run di
    # riferimento è stata copiata altrove, quindi il path del manifest è stale).
    cfg.run_step1 = cfg.run_step2 = False
    cfg.run_step3 = cfg.run_step4 = True
    cfg.render.output_dir = str(run_dir)
    cfg.render.roi_rect = list(args.rect) if args.rect else None
    cfg.render.roi_mask_path = args.mask
    cfg.render.roi_mask_threshold = args.mask_threshold
    cfg.render.roi_tag = args.tag

    diffs = check_against_manifest(cfg, manifest)
    if diffs:
        print("\n✗ la config ricostruita diverge dal manifest su chiavi non previste:")
        for d in diffs:
            print(f"    {d}")
        print("  Il confronto con la run piena non sarebbe valido: interrotto.")
        return 1
    print("  ✓ config identica al manifest (a meno di step, output_dir e ROI)")

    assets_dir, tag = _roi_assets_dir(cfg.render, run_dir)
    print(f"  sandbox: {assets_dir}")
    if args.dry_run:
        return 0

    log_path = str(assets_dir / "console.log")
    with _console_to_file(log_path):
        print(f"{'=' * 70}")
        print(f"  ROI rerun  : {tag}")
        print(f"  Riferimento: {run_dir}")
        print(f"  Sandbox    : {assets_dir}")
        print(f"  rect={cfg.render.roi_rect}  maschera={cfg.render.roi_mask_path or '-'}")
        print(f"{'=' * 70}")
        run_pipeline(cfg, tb_enabled=False)
    print(f"\n✓ fatto. Confronta con:\n    python compare_roi_run.py {run_dir} --tag {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
