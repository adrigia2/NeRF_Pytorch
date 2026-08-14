#!/usr/bin/env python
"""Rigenera irradiance + Step 4 su un albero di run già esistente.

Serve dopo il fix dell'azimut in `deviceProgramsIrradiance.cu`, che calcolava
`(float)i * goldenAngle` in float32 senza mai ridurre mod 2pi: a
`irradiance_sample_side = 512` (S = 262144) phi arriva a 6.3e5 rad, dove un ULP
float32 vale 3.58 gradi e la sequenza aurea perde la bassa discrepanza. Tutte le
mappe derivate dall'irradiance vanno rifatte, ma NON il training NeRF e NON il
bake degli spec cone, che dall'irradiance non dipendono.

Cosa viene rifatto, e cosa no:

    irradiance/irradiance.exr              rigenerato   (Step 3)
    sources/*/albedo/                      rigenerato   (Step 4, usa E)
    sources/*/albedo_pbr/                  rigenerato   (Step 4, usa E)
    sources/*/metallic|roughness|pbr/      rigenerati, ma devono venire IDENTICI:
                                           la regressione C_jc = a_c + b*L_jc non
                                           usa l'irradiance da nessuna parte
    spec_cone/, irradiance_indirect.exr    intoccati
    skybox_nerf_baked.exr, nerf_train/     intoccati
    ium/, visibility/, color_texture/      ricalcolati o letti da cache, deterministici

La config NON viene ritrascritta a mano ma ricostruita da `run_manifest.json` e
validata contro di esso: una sola differenza non voluta renderebbe il confronto
con l'albero originale privo di significato.

Uso:
    python rerun_irradiance.py <root> [--only TAG/SCENA ...] [--dry-run]
    python rerun_irradiance.py <root> --verify <root_originale>

Chiama `run_pipeline` e non `run_pipeline_multi`, che riscriverebbe
`run_manifest.json` (la fonte di verita' da cui questo script ricostruisce la
config, e che va lasciata intatta perche' il rerun resti rieseguibile).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import _paths  # noqa: F401

from images_generator import _console_to_file, run_pipeline  # noqa: E402
from roi_rerun import check_against_manifest, config_from_manifest  # noqa: E402

# Alberi di run che questo script si rifiuta di modificare: sono i riferimenti
# del capitolo Results e vanno rigenerati solo su copia.
PROTECTED_ROOTS = {"test_sword_shield"}

# Chiavi che questo script cambia di proposito: le uniche differenze ammesse
# rispetto al manifest della run. `precompute_indirect` NON e' qui perche' viene
# lasciato com'e': se True lo Step 3 trova irradiance_indirect.exr su disco e lo
# salta, se False lo Step 4 lo rilegge comunque dal path. In nessun caso viene
# ricalcolato, quindi non serve toccarlo.
EXPECTED_DIFFS = {
    "run_step1", "run_step2", "run_step3", "run_step4",
    "render.output_dir",
    "render.precompute_spec_cone",
    # Campo CANCELLATO dalla config, non un override: il gate diffuso del solver
    # e' stato rimosso, ma i manifest gia' scritti lo contengono ancora.
    "render.pbr_diffuse_cv_gate",
}

# Cartelle di Step 4 da azzerare, relative a sources/{src}/. Si cancella la
# cartella intera e non i singoli file per non lasciare in giro i *_rgb.exr e le
# preview della run precedente.
STALE_SOURCE_DIRS = ("albedo", "albedo_pbr", "metallic", "roughness", "pbr")

# Artefatti costosi che il rerun non deve toccare: se ne registra (size, mtime)
# prima di partire, e --verify controlla che siano rimasti identici.
def protected_files(run_dir: Path) -> list[Path]:
    out = [run_dir / "irradiance" / "irradiance_indirect.exr",
           run_dir / "skybox_nerf_baked.exr"]
    out += sorted((run_dir / "spec_cone").glob("*"))
    return [p for p in out if p.is_file()]


# ── Confronti per --verify ────────────────────────────────────────────────────
# Il fit PBR non usa l'irradiance e i suoi ingressi non sono cambiati: se metallic
# differisce, e' cambiato qualcosa di non voluto.
MUST_MATCH = [
    "sources/{src}/metallic/metallic.exr",
    "sources/{src}/roughness/roughness.exr",
    "sources/{src}/pbr/diffuse_weight.exr",
    "sources/{src}/pbr/diffuse_term.exr",
    "sources/{src}/pbr/lobe_param.exr",
    "sources/{src}/pbr/residual.exr",
    "sources/{src}/pbr/n_views.exr",
]
# Un'uguaglianza qui significherebbe che Step 3 o Step 4 ha saltato il lavoro.
MUST_DIFFER = [
    "irradiance/irradiance.exr",
    "sources/{src}/albedo/albedo.exr",
    "sources/{src}/albedo_pbr/albedo_pbr.exr",
]


def read_exr_channels(path: Path) -> "dict[str, object]":
    """Tutti i canali di un EXR come float32, senza assumerne il numero.

    Generico di proposito: visibility.exr ha un canale per camera, gli EXR dei
    coni uno per apertura piu' `valid`, le mappe finali un solo `Z`.
    """
    import numpy as np
    import OpenEXR
    import Imath
    fh = OpenEXR.InputFile(str(path))
    header = fh.header()
    dw = header["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1
    ftype = Imath.PixelType(Imath.PixelType.FLOAT)
    return {c: np.frombuffer(fh.channel(c, ftype), dtype=np.float32).reshape(h, w)
            for c in header["channels"]}


def arrays_equal(a: dict, b: dict) -> bool:
    import numpy as np
    if set(a) != set(b):
        return False
    return all(np.array_equal(a[c], b[c]) for c in a)


def snapshot(paths: "list[Path]") -> dict:
    return {p.as_posix(): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}


# ── Scoperta delle run ────────────────────────────────────────────────────────

def discover_runs(root: Path, only: "list[str] | None") -> "list[Path]":
    runs = sorted(p.parent for p in root.glob("*/*/run_manifest.json"))
    if only:
        wanted = {o.replace("\\", "/").strip("/") for o in only}
        runs = [r for r in runs
                if f"{r.parent.name}/{r.name}" in wanted or r.name in wanted]
    return runs


def guard_root(root: Path) -> None:
    if root.name in PROTECTED_ROOTS:
        raise SystemExit(
            f"✗ {root} e' un albero di riferimento protetto.\n"
            f"  Copiarlo altrove e lanciare lo script sulla copia: il rerun "
            f"sovrascrive irradiance e le mappe di Step 4.")
    if not root.is_dir():
        raise SystemExit(f"✗ root inesistente: {root}")


# ── Esecuzione di una run ─────────────────────────────────────────────────────

def stale_paths(run_dir: Path, sources: "list[str]") -> "list[Path]":
    out = [run_dir / "irradiance" / "irradiance.exr"]
    for src in sources:
        out += [run_dir / "sources" / src / d for d in STALE_SOURCE_DIRS]
    return [p for p in out if p.exists()]


def process_run(run_dir: Path, root: Path, dry_run: bool) -> None:
    manifest_path = run_dir / "run_manifest.json"
    cfg, manifest, notes = config_from_manifest(manifest_path)
    for n in notes:
        print(n)

    # La run deve stare sotto la root richiesta: il manifest copiato porta ancora
    # l'output_dir dell'albero originale, ed e' esattamente l'errore da evitare.
    run_dir = run_dir.resolve()
    if root.resolve() not in run_dir.parents:
        raise SystemExit(f"✗ {run_dir} non e' sotto {root}")

    original_out = cfg.render.output_dir
    cfg.run_step1 = cfg.run_step2 = False
    cfg.run_step3 = cfg.run_step4 = True
    cfg.render.output_dir = str(run_dir)
    cfg.render.precompute_spec_cone = False

    diffs = check_against_manifest(cfg, manifest, EXPECTED_DIFFS)
    if diffs:
        print("  ✗ la config ricostruita diverge dal manifest su chiavi non previste:")
        for d in diffs:
            print(f"      {d}")
        raise SystemExit("  Il confronto con l'albero originale non sarebbe valido: interrotto.")

    sources = list(cfg.render.color_texture_image_sources)
    stale = stale_paths(run_dir, sources)
    keep = protected_files(run_dir)

    print(f"  output_dir : {original_out}")
    print(f"           -> {cfg.render.output_dir}")
    print(f"  sorgenti   : {sources}")
    print(f"  intoccati  : {len(keep)} file (spec_cone/, irradiance_indirect, skybox)")
    print(f"  da azzerare: {len(stale)}")
    for p in stale:
        print(f"      - {p.relative_to(run_dir).as_posix()}")

    if dry_run:
        print("  (dry-run: niente cancellato, niente eseguito)")
        return

    before = snapshot(keep)
    for p in stale:
        shutil.rmtree(p) if p.is_dir() else p.unlink()

    t0 = time.perf_counter()
    log_path = run_dir / "console_irradiance_fix.log"
    with _console_to_file(str(log_path)):
        print(f"{'=' * 70}")
        print(f"  Rerun irradiance + Step 4")
        print(f"  Run        : {run_dir}")
        print(f"  Originale  : {original_out}")
        print(f"{'=' * 70}")
        run_pipeline(cfg, tb_enabled=False)
    elapsed = time.perf_counter() - t0

    after = snapshot([p for p in keep if p.is_file()])
    touched = [k for k, v in before.items() if after.get(k) != v]
    if touched:
        print(f"  ⚠  {len(touched)} file protetti sono stati modificati:")
        for k in touched[:10]:
            print(f"      {k}")

    with open(run_dir / "irradiance_fix.json", "w", encoding="utf-8") as fh:
        json.dump({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "reason": "fix azimut float32 -> float64 in deviceProgramsIrradiance.cu",
            "original_output_dir": original_out,
            "sources": sources,
            "deleted": [p.relative_to(run_dir).as_posix() for p in stale],
            "protected_before": before,
            "protected_touched": touched,
            "elapsed_s": round(elapsed, 1),
        }, fh, indent=2)
    print(f"  ✓ fatto in {elapsed / 60:.1f} min → {log_path.name}")


# ── Verifica ──────────────────────────────────────────────────────────────────

def albedo_is_pinned(run_dir: Path) -> "tuple[bool, float]":
    """L'albedo puo' cambiare, su questa run, al variare dell'irradiance?

    L'albedo e' `pi * color / max(irr + irr_indirect, albedo_eps)`.  Se la somma delle due
    irradiance sta sotto `albedo_eps` OVUNQUE, il denominatore e' la costante eps e
    l'albedo non dipende piu' dall'irradiance: resta identico bit a bit anche dopo un
    ri-bake, e segnalarlo come "step saltato" sarebbe un falso allarme.  Succede davvero:
    su SwordShieldNight con attivazione esponenziale e loss L1 il NeRF non emette nulla e
    l'irradiance vale ~1e-18, quindici ordini di grandezza sotto eps.
    """
    import numpy as np
    try:
        with open(run_dir / "run_manifest.json", encoding="utf-8") as fh:
            eps = float(json.load(fh)["config"]["render"].get("albedo_eps", 1e-3))
        irr = read_exr_channels(run_dir / "irradiance" / "irradiance.exr")
        ind_p = run_dir / "irradiance" / "irradiance_indirect.exr"
        tot = sum(irr[c] for c in irr)
        if ind_p.exists():
            ind = read_exr_channels(ind_p)
            tot = tot + sum(ind[c] for c in ind)
        msk = read_exr_channels(run_dir / "ium" / "ium_masks.exr")
        m = next(iter(msk.values())) > 0.5
        frac = float((tot[m] <= eps).mean())
        return frac >= 1.0, frac
    except Exception:
        return False, float("nan")


def verify_run(run_dir: Path, ref_dir: Path, label: str) -> "list[str]":
    problems: list[str] = []
    stamp_path = run_dir / "irradiance_fix.json"
    if not stamp_path.exists():
        return [f"{label}: manca irradiance_fix.json (rerun non eseguito?)"]
    with open(stamp_path, encoding="utf-8") as fh:
        stamp = json.load(fh)
    pinned, frac = albedo_is_pinned(run_dir)
    if pinned:
        print(f"    nota: irradiance sotto albedo_eps su tutti i texel "
              f"({100 * frac:.2f}%), l'albedo non puo' cambiare: atteso identico")

    # 1. gli artefatti costosi non devono essere stati toccati
    now = snapshot([Path(k) for k in stamp["protected_before"] if Path(k).is_file()])
    for k, v in stamp["protected_before"].items():
        if now.get(k) != v:
            problems.append(f"{label}: file protetto modificato: {k}")

    # 2. il fit PBR non usa l'irradiance → deve venire identico all'originale
    for src in stamp["sources"]:
        for pat in MUST_MATCH:
            rel = pat.format(src=src)
            a, b = run_dir / rel, ref_dir / rel
            if not (a.exists() and b.exists()):
                continue
            if not arrays_equal(read_exr_channels(a), read_exr_channels(b)):
                problems.append(f"{label}: {rel} DIFFERISCE dall'originale "
                                f"(il fit PBR non dipende da E: indagare)")
        # 3. queste invece devono essere cambiate, salvo che l'albedo sia bloccato
        # dal clamp: in quel caso restare identico e' la risposta giusta.
        for pat in MUST_DIFFER:
            rel = pat.format(src=src)
            a, b = run_dir / rel, ref_dir / rel
            if not (a.exists() and b.exists()):
                continue
            if arrays_equal(read_exr_channels(a), read_exr_channels(b)):
                if pinned and "albedo" in rel:
                    continue
                problems.append(f"{label}: {rel} IDENTICA all'originale "
                                f"(step saltato?)")
    return problems


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="albero di run da rigenerare (una COPIA)")
    ap.add_argument("--only", nargs="+", metavar="TAG/SCENA",
                    help="limita a queste run (es. exp_l1_d02/SwordShieldStudio)")
    ap.add_argument("--dry-run", action="store_true",
                    help="mostra config, override e cancellazioni, poi si ferma")
    ap.add_argument("--keep-going", action="store_true",
                    help="prosegue con le run successive se una fallisce")
    ap.add_argument("--verify", metavar="ROOT_ORIGINALE",
                    help="non esegue nulla: confronta il risultato con l'albero originale")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    guard_root(root)
    runs = discover_runs(root, args.only)
    if not runs:
        print(f"✗ nessuna run trovata sotto {root}" +
              (f" con --only {args.only}" if args.only else ""))
        return 2

    if args.verify:
        ref_root = Path(args.verify).resolve()
        problems: list[str] = []
        for run_dir in runs:
            rel = run_dir.relative_to(root)
            ref_dir = ref_root / rel
            print(f"[verify] {rel.as_posix()}")
            if not ref_dir.is_dir():
                problems.append(f"{rel}: manca la run di riferimento {ref_dir}")
                continue
            problems += verify_run(run_dir, ref_dir, rel.as_posix())
        print(f"\n{'=' * 70}\nRiepilogo verifica: {len(runs)} run")
        if problems:
            for p in problems:
                print(f"  ✗ {p}")
            return 1
        print("  ✓ tutti i controlli passati: il fit PBR e' invariato, "
              "irradiance/albedo sono cambiate, gli artefatti costosi sono intatti")
        return 0

    print(f"Root      : {root}")
    print(f"Run       : {len(runs)}")
    failures: list[tuple[str, str]] = []
    for i, run_dir in enumerate(runs, 1):
        rel = run_dir.relative_to(root).as_posix()
        print(f"\n{'=' * 70}\n[{i}/{len(runs)}] {rel}\n{'=' * 70}")
        try:
            process_run(run_dir, root, args.dry_run)
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            failures.append((rel, str(exc)))
            if not args.keep_going:
                print("\n✗ interrotto. Usare --keep-going per proseguire comunque.")
                break

    print(f"\n{'=' * 70}\nRiepilogo rerun_irradiance: {len(runs)} run")
    for rel, err in failures:
        print(f"  ✗ {rel}: {err}")
    if not failures:
        print("  ✓ nessun errore")
        if not args.dry_run:
            print(f"\nVerifica con:\n    python rerun_irradiance.py {root} "
                  f"--verify D:/tesi_output/test_sword_shield")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
