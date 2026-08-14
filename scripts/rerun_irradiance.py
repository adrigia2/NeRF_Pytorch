#!/usr/bin/env python
"""Regenerate the irradiance + Step 4 over an existing run tree.

Needed after the azimuth fix in `deviceProgramsIrradiance.cu`, which computed
`(float)i * goldenAngle` in float32 without ever reducing mod 2pi: at
`irradiance_sample_side = 512` (S = 262144) phi reaches 6.3e5 rad, where one float32
ULP is 3.58 degrees and the golden sequence loses its low discrepancy. Every map
derived from the irradiance has to be redone, but NOT the NeRF training and NOT the
spec cone bake, which do not depend on the irradiance.

What is redone, and what is not:

    irradiance/irradiance.exr              regenerated  (Step 3)
    sources/*/albedo/                      regenerated  (Step 4, uses E)
    sources/*/albedo_pbr/                  regenerated  (Step 4, uses E)
    sources/*/metallic|roughness|pbr/      regenerated, but must come out IDENTICAL:
                                           the regression C_jc = a_c + b*L_jc does not
                                           use the irradiance anywhere
    spec_cone/, irradiance_indirect.exr    untouched
    skybox_nerf_baked.exr, nerf_train/     untouched
    ium/, visibility/, color_texture/      recomputed or read from cache, deterministic

The config is NOT transcribed by hand but rebuilt from `run_manifest.json` and
validated against it: a single unintended difference would make the comparison with
the original tree meaningless.

Usage:
    python rerun_irradiance.py <root> [--only TAG/SCENE ...] [--dry-run]
    python rerun_irradiance.py <root> --verify <original_root>

It calls `run_pipeline` and not `run_pipeline_multi`, which would rewrite
`run_manifest.json` (the source of truth this script rebuilds the config from, and
which has to stay intact for the rerun to remain repeatable).
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

# Run trees this script refuses to modify: they are the references of the Results
# chapter and must only be regenerated on a copy.
PROTECTED_ROOTS = {"test_sword_shield"}

# Keys this script changes on purpose: the only differences allowed with respect to
# the run's manifest. `precompute_indirect` is NOT here because it is left as it is:
# if True, Step 3 finds irradiance_indirect.exr on disk and skips it; if False,
# Step 4 re-reads it from the path anyway. In neither case is it recomputed, so
# there is no need to touch it.
EXPECTED_DIFFS = {
    "run_step1", "run_step2", "run_step3", "run_step4",
    "render.output_dir",
    "render.precompute_spec_cone",
    # A field DELETED from the config, not an override: the solver's diffuse gate was
    # removed, but the manifests already written still contain it.
    "render.pbr_diffuse_cv_gate",
}

# Step 4 folders to clear, relative to sources/{src}/. The whole folder is deleted
# rather than the individual files, so no *_rgb.exr or preview from the previous run
# is left behind.
STALE_SOURCE_DIRS = ("albedo", "albedo_pbr", "metallic", "roughness", "pbr")

# Expensive artefacts the rerun must not touch: their (size, mtime) is recorded
# before starting, and --verify checks they stayed identical.
def protected_files(run_dir: Path) -> list[Path]:
    out = [run_dir / "irradiance" / "irradiance_indirect.exr",
           run_dir / "skybox_nerf_baked.exr"]
    out += sorted((run_dir / "spec_cone").glob("*"))
    return [p for p in out if p.is_file()]


# ── Comparisons for --verify ─────────────────────────────────────────────────
# The PBR fit does not use the irradiance and its inputs have not changed: if metallic
# differs, something unintended changed.
MUST_MATCH = [
    "sources/{src}/metallic/metallic.exr",
    "sources/{src}/roughness/roughness.exr",
    "sources/{src}/pbr/diffuse_weight.exr",
    "sources/{src}/pbr/diffuse_term.exr",
    "sources/{src}/pbr/lobe_param.exr",
    "sources/{src}/pbr/residual.exr",
    "sources/{src}/pbr/n_views.exr",
]
# Equality here would mean Step 3 or Step 4 skipped the work.
MUST_DIFFER = [
    "irradiance/irradiance.exr",
    "sources/{src}/albedo/albedo.exr",
    "sources/{src}/albedo_pbr/albedo_pbr.exr",
]


def read_exr_channels(path: Path) -> "dict[str, object]":
    """Every channel of an EXR as float32, without assuming how many there are.

    Generic on purpose: visibility.exr has one channel per camera, the cone EXRs one
    per aperture plus `valid`, and the final maps a single `Z`.
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


# ── Run discovery ────────────────────────────────────────────────────────────

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
            f"✗ {root} is a protected reference tree.\n"
            f"  Copy it elsewhere and run the script on the copy: the rerun "
            f"overwrites the irradiance and the Step 4 maps.")
    if not root.is_dir():
        raise SystemExit(f"✗ root does not exist: {root}")


# ── Running one run ──────────────────────────────────────────────────────────

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

    # The run has to live under the requested root: a copied manifest still carries the
    # output_dir of the original tree, and that is exactly the mistake to avoid.
    run_dir = run_dir.resolve()
    if root.resolve() not in run_dir.parents:
        raise SystemExit(f"✗ {run_dir} is not under {root}")

    original_out = cfg.render.output_dir
    cfg.run_step1 = cfg.run_step2 = False
    cfg.run_step3 = cfg.run_step4 = True
    cfg.render.output_dir = str(run_dir)
    cfg.render.precompute_spec_cone = False

    diffs = check_against_manifest(cfg, manifest, EXPECTED_DIFFS)
    if diffs:
        print("  ✗ the rebuilt config diverges from the manifest on unexpected keys:")
        for d in diffs:
            print(f"      {d}")
        raise SystemExit("  The comparison with the original tree would not be valid: aborted.")

    sources = list(cfg.render.color_texture_image_sources)
    stale = stale_paths(run_dir, sources)
    keep = protected_files(run_dir)

    print(f"  output_dir : {original_out}")
    print(f"           -> {cfg.render.output_dir}")
    print(f"  sources    : {sources}")
    print(f"  untouched  : {len(keep)} files (spec_cone/, irradiance_indirect, skybox)")
    print(f"  to clear   : {len(stale)}")
    for p in stale:
        print(f"      - {p.relative_to(run_dir).as_posix()}")

    if dry_run:
        print("  (dry-run: nothing deleted, nothing run)")
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
        print(f"  Original   : {original_out}")
        print(f"{'=' * 70}")
        run_pipeline(cfg, tb_enabled=False)
    elapsed = time.perf_counter() - t0

    after = snapshot([p for p in keep if p.is_file()])
    touched = [k for k, v in before.items() if after.get(k) != v]
    if touched:
        print(f"  ⚠  {len(touched)} protected files were modified:")
        for k in touched[:10]:
            print(f"      {k}")

    with open(run_dir / "irradiance_fix.json", "w", encoding="utf-8") as fh:
        json.dump({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "reason": "float32 -> float64 azimuth fix in deviceProgramsIrradiance.cu",
            "original_output_dir": original_out,
            "sources": sources,
            "deleted": [p.relative_to(run_dir).as_posix() for p in stale],
            "protected_before": before,
            "protected_touched": touched,
            "elapsed_s": round(elapsed, 1),
        }, fh, indent=2)
    print(f"  ✓ done in {elapsed / 60:.1f} min → {log_path.name}")


# ── Verification ─────────────────────────────────────────────────────────────

def albedo_is_pinned(run_dir: Path) -> "tuple[bool, float]":
    """Can the albedo change, on this run, when the irradiance changes?

    The albedo is `pi * color / max(irr + irr_indirect, albedo_eps)`.  If the sum of the
    two irradiances is below `albedo_eps` EVERYWHERE, the denominator is the constant eps
    and the albedo no longer depends on the irradiance: it stays bit-identical even after
    a re-bake, and flagging that as a \"skipped step\" would be a false alarm.  It really
    happens: on SwordShieldNight with exponential activation and L1 loss the NeRF emits
    nothing and the irradiance is ~1e-18, fifteen orders of magnitude below eps.
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
        return [f"{label}: irradiance_fix.json missing (rerun not executed?)"]
    with open(stamp_path, encoding="utf-8") as fh:
        stamp = json.load(fh)
    pinned, frac = albedo_is_pinned(run_dir)
    if pinned:
        print(f"    note: irradiance below albedo_eps on every texel "
              f"({100 * frac:.2f}%), the albedo cannot change: identical is expected")

    # 1. the expensive artefacts must not have been touched
    now = snapshot([Path(k) for k in stamp["protected_before"] if Path(k).is_file()])
    for k, v in stamp["protected_before"].items():
        if now.get(k) != v:
            problems.append(f"{label}: protected file modified: {k}")

    # 2. the PBR fit does not use the irradiance → it must match the original exactly
    for src in stamp["sources"]:
        for pat in MUST_MATCH:
            rel = pat.format(src=src)
            a, b = run_dir / rel, ref_dir / rel
            if not (a.exists() and b.exists()):
                continue
            if not arrays_equal(read_exr_channels(a), read_exr_channels(b)):
                problems.append(f"{label}: {rel} DIFFERS from the original "
                                f"(the PBR fit does not depend on E: investigate)")
        # 3. these must have changed instead, unless the albedo is pinned by the
        # clamp: in that case staying identical is the right answer.
        for pat in MUST_DIFFER:
            rel = pat.format(src=src)
            a, b = run_dir / rel, ref_dir / rel
            if not (a.exists() and b.exists()):
                continue
            if arrays_equal(read_exr_channels(a), read_exr_channels(b)):
                if pinned and "albedo" in rel:
                    continue
                problems.append(f"{label}: {rel} IDENTICAL to the original "
                                f"(step skipped?)")
    return problems


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="run tree to regenerate (a COPY)")
    ap.add_argument("--only", nargs="+", metavar="TAG/SCENE",
                    help="limit to these runs (e.g. exp_l1_d02/SwordShieldStudio)")
    ap.add_argument("--dry-run", action="store_true",
                    help="show config, overrides and deletions, then stop")
    ap.add_argument("--keep-going", action="store_true",
                    help="carry on with the next runs when one fails")
    ap.add_argument("--verify", metavar="ORIGINAL_ROOT",
                    help="run nothing: compare the result with the original tree")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    guard_root(root)
    runs = discover_runs(root, args.only)
    if not runs:
        print(f"✗ no run found under {root}" +
              (f" with --only {args.only}" if args.only else ""))
        return 2

    if args.verify:
        ref_root = Path(args.verify).resolve()
        problems: list[str] = []
        for run_dir in runs:
            rel = run_dir.relative_to(root)
            ref_dir = ref_root / rel
            print(f"[verify] {rel.as_posix()}")
            if not ref_dir.is_dir():
                problems.append(f"{rel}: the reference run {ref_dir} is missing")
                continue
            problems += verify_run(run_dir, ref_dir, rel.as_posix())
        print(f"\n{'=' * 70}\nVerification summary: {len(runs)} runs")
        if problems:
            for p in problems:
                print(f"  ✗ {p}")
            return 1
        print("  ✓ all checks passed: the PBR fit is unchanged, "
              "irradiance/albedo changed, and the expensive artefacts are intact")
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
                print("\n✗ aborted. Use --keep-going to carry on anyway.")
                break

    print(f"\n{'=' * 70}\nrerun_irradiance summary: {len(runs)} runs")
    for rel, err in failures:
        print(f"  ✗ {rel}: {err}")
    if not failures:
        print("  ✓ no error")
        if not args.dry_run:
            print(f"\nVerify with:\n    python rerun_irradiance.py {root} "
                  f"--verify D:/tesi_output/test_sword_shield")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
