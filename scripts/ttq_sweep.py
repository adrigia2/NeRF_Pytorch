#!/usr/bin/env python
"""ttq_sweep.py -- sweep the two sampling knobs of the bake inside one texture-space ROI.

Backs the "Time to Quality" section of the thesis. The pipeline has exactly two places
where the number of samples is a free parameter and the quantity being estimated is an
integral by quadrature: the direct irradiance over the skybox (``irradiance_sample_side``)
and the indirect one queried against the NeRF (``indirect_sample_side``). This script
bakes the same ROI at several values of one of them, leaving each variant in its own
sandbox, and records how long each bake took.

    python ttq_sweep.py <run_dir> (--rect X0 Y0 W H | --full)
                        [--tag-prefix ttq] [--irradiance 512 768 1024]
                        [--indirect 32 64 96] [--repeat-baseline]
                        [--force] [--dry-run] [--keep-going]

``--full`` bakes the whole atlas into the run root instead of a sandbox. It is there for
one job: after a kernel changes, the reference map has to be rebuilt with the SAME binary
the sandboxes use, otherwise the bit-identity check compares a code change instead of the
ROI.

Design notes, in decreasing order of how easily they are got wrong:

  * The indirect pass HAS an on-disk cache: Step 3 skips it when
    ``irradiance_indirect.exr`` already exists. Without deleting it first the N=64
    variant would read back the file of N=32 and every difference in the figure would be
    exactly zero, which is the quietest possible way to get the experiment wrong. The
    direct irradiance has no cache and is always recomputed.
  * The configuration is rebuilt from ``run_manifest.json`` and validated against it, as
    in ``roi_rerun.py``: only the keys listed in ``EXPECTED_DIFFS`` may differ.
    Transcribing the config by hand would void any comparison with the reference run.
  * Every pass neither of the two needs is switched off. Visibility, colour texture and
    the specular cone are not inputs of either irradiance pass (see the guards in
    ``_step3_posttrain_assets``) and the cone bake alone is half the cost of a full run.
  * ``render_ium`` stays True: the ROI is applied inside that block and nowhere else, so
    switching the IUM off would silently switch the ROI off with it.
  * If ``skybox_nerf_baked.exr`` were missing from the run root, ``_resolve_skybox_flat``
    would re-bake it and write into the root. The (size, mtime) of the root artefacts is
    snapshotted before each variant and checked after it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import _paths  # noqa: F401

from images_generator import (  # noqa: E402
    _console_to_file,
    _roi_assets_dir,
    run_pipeline,
)
from roi_rerun import (  # noqa: E402
    _EXPECTED_DIFFS,
    check_against_manifest,
    config_from_manifest,
)

# Run trees this script refuses to touch: they are the references of the Results
# chapter. Same guard as rerun_irradiance.PROTECTED_ROOTS, widened to the copy the
# irradiance fix produced, which is just as much a reference.
PROTECTED_ROOTS = {"test_sword_shield", "test_sword_shield_after_fix_irradiance"}

# On top of the ROI keys roi_rerun already declares: the passes switched off, the two
# switches that select which family of variants runs, and the subject of the experiment.
EXPECTED_DIFFS = _EXPECTED_DIFFS | {
    "render.render_visibility",
    "render.render_color_texture",
    "render.render_pixel_change",
    "render.precompute_spec_cone",
    "render.render_pbr_maps",
    "render.render_albedo",
    "render.render_irradiance",
    "render.precompute_indirect",
    "render.irradiance_sample_side",
    "render.indirect_sample_side",
}

# (folder, file) produced by each family, relative to the sandbox.
OUTPUT = {"irradiance": ("irradiance", "irradiance.exr"),
          "indirect": ("irradiance", "irradiance_indirect.exr")}

SUBSTEP = {"irradiance": "step3/irradiance", "indirect": "step3/indirect"}


def protected_root_files(run_dir: Path) -> list[Path]:
    """Root artefacts a ROI variant must never rewrite."""
    out = [run_dir / "skybox_nerf_baked.exr",
           run_dir / "irradiance" / "irradiance.exr",
           run_dir / "irradiance" / "irradiance_indirect.exr",
           run_dir / "model" / "nerf_model_cache.pt"]
    out += sorted((run_dir / "ium").glob("*.exr"))
    return [p for p in out if p.is_file()]


def snapshot(paths: list[Path]) -> dict:
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}


def active_texels(sandbox: Path) -> int:
    """Texels the ROI actually bakes: the rect intersected with the IUM mask.

    roi.json counts the rect alone, before the intersection with the mesh, and that is
    not the number the cost is proportional to.
    """
    p = sandbox / "ium" / "ium_masks.exr"
    if not p.exists():
        return -1
    from regen_heatmaps import _load_exr_hw3
    return int((_load_exr_hw3(str(p))[..., 0] > 0.5).sum())


def build_cfg(manifest_path: Path, run_dir: Path, rect, tag: str,
              family: str, value: int):
    cfg, manifest, notes = config_from_manifest(manifest_path)
    rc = cfg.render

    cfg.run_step1 = cfg.run_step2 = cfg.run_step4 = False
    cfg.run_step3 = True
    rc.output_dir = str(run_dir)
    # rect None is the whole atlas: _roi_is_active goes False and _roi_assets_dir gives
    # back the run root, so the bake lands where the reference map lives. That is how the
    # reference is rebuilt after a change to the kernel, and roi_tag is then ignored.
    rc.roi_rect = list(rect) if rect else None
    rc.roi_mask_path = ""
    rc.roi_mask_threshold = 0.5
    rc.roi_tag = tag if rect else ""

    rc.render_ium = True            # the ROI lives inside this block
    rc.render_visibility = False
    rc.render_color_texture = False
    rc.render_pixel_change = False
    rc.precompute_spec_cone = False
    rc.render_pbr_maps = False
    rc.render_albedo = False

    rc.render_irradiance = family == "irradiance"
    rc.precompute_indirect = family == "indirect"
    if family == "irradiance":
        rc.irradiance_sample_side = value
    else:
        rc.indirect_sample_side = value

    diffs = check_against_manifest(cfg, manifest, EXPECTED_DIFFS)
    return cfg, manifest, notes, diffs


def run_variant(run_dir: Path, manifest_path: Path, rect, family: str, value: int,
                tag: str, force: bool, dry: bool) -> dict:
    cfg, manifest, notes, diffs = build_cfg(manifest_path, run_dir, rect, tag,
                                            family, value)
    for n in notes:
        print(n)
    if diffs:
        print("  x config diverges from the manifest on unexpected keys:")
        for d in diffs:
            print(f"      {d}")
        raise SystemExit(1)

    sandbox, resolved = _roi_assets_dir(cfg.render, run_dir)
    sub, name = OUTPUT[family]
    target = sandbox / sub / name
    print(f"  sandbox : {sandbox}")
    print(f"  target  : {sub}/{name}")

    if target.exists():
        if not force:
            print(f"  x {target} already exists. Use --force to redo this variant, "
                  "or pick another --tag-prefix.")
            raise SystemExit(1)
        print(f"  - deleting the previous {name} (the indirect pass would skip it)")
        target.unlink()

    if dry:
        return {"tag": tag, "family": family, "value": value, "status": "dry-run"}

    full_atlas = sandbox == run_dir
    before = {} if full_atlas else snapshot(protected_root_files(run_dir))
    log = sandbox / ("console_ttq_full.log" if full_atlas else "console.log")
    t0 = time.perf_counter()
    with _console_to_file(str(log)):
        print("=" * 70)
        print(f"  ttq_sweep : {family} = {value}")
        print(f"  sandbox   : {sandbox}")
        print(f"  rect      : {cfg.render.roi_rect}")
        print("=" * 70)
        run_pipeline(cfg, tb_enabled=False)
    wall = time.perf_counter() - t0
    after = snapshot(protected_root_files(run_dir))
    touched = [k for k in before if before[k] != after.get(k)]
    if touched:
        print("  ! the variant modified files in the run root:")
        for k in touched:
            print(f"      {k}")

    rec = {"tag": tag, "family": family, "value": value,
           "samples_per_texel": value * value,
           "wall_s": round(wall, 2),
           "status": "ok" if target.exists() else "missing-output",
           "root_touched": touched,
           "active_texels": active_texels(sandbox)}

    timings = sandbox / "run_timings.jsonl"
    if timings.exists():
        last = json.loads(timings.read_text(encoding="utf-8").strip().splitlines()[-1])
        rec["pass_s"] = last["substeps"].get(SUBSTEP[family])
        rec["ium_s"] = last["substeps"].get("step3/ium")
        rec["step3_s"] = last["steps"].get("step3")
    print(f"  = {family} {value}: pass {rec.get('pass_s')} s, "
          f"wall {rec['wall_s']} s, {rec['active_texels']} active texels")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--rect", nargs=4, type=int, default=None,
                    metavar=("X0", "Y0", "W", "H"))
    ap.add_argument("--full", action="store_true",
                    help="bake the WHOLE atlas into the run root instead of a sandbox: "
                         "this is how the reference map is rebuilt after a change to a "
                         "kernel, so that the ROI check compares two bakes of the same "
                         "binary")
    ap.add_argument("--tag-prefix", default="ttq")
    ap.add_argument("--irradiance", nargs="*", type=int, default=[])
    ap.add_argument("--indirect", nargs="*", type=int, default=[])
    ap.add_argument("--repeat-baseline", action="store_true",
                    help="bake the first value of each family a second time in a twin "
                         "sandbox, to measure the repeatability of the bake itself")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-going", action="store_true")
    a = ap.parse_args()

    run_dir = Path(a.run_dir).resolve()
    if {p.name for p in run_dir.parents} & PROTECTED_ROOTS:
        print(f"x {run_dir} lives under a protected reference tree "
              f"({sorted(PROTECTED_ROOTS)}). Work on a copy.")
        return 2
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        print(f"x manifest not found: {manifest_path}")
        return 2
    if not a.irradiance and not a.indirect:
        print("x nothing to do: pass --irradiance and/or --indirect")
        return 2
    if not a.rect and not a.full:
        print("x pass --rect X0 Y0 W H, or --full to bake the whole atlas")
        return 2
    if a.rect and a.full:
        print("x --rect and --full are exclusive")
        return 2

    for need, why in ((run_dir / "transforms_extended.json", "every pass"),
                      (run_dir / "skybox_nerf_baked.exr", "the direct irradiance"),
                      (run_dir / "model" / "nerf_model_cache.pt", "the indirect pass")):
        if not need.exists():
            print(f"  ! missing {need.name}, needed by {why}")

    jobs = []
    for v in a.irradiance:
        jobs.append(("irradiance", v, f"{a.tag_prefix}_irr{v}"))
    if a.repeat_baseline and a.irradiance:
        v = a.irradiance[0]
        jobs.append(("irradiance", v, f"{a.tag_prefix}_irr{v}b"))
    for v in a.indirect:
        jobs.append(("indirect", v, f"{a.tag_prefix}_ind{v}"))
    if a.repeat_baseline and a.indirect:
        v = a.indirect[0]
        jobs.append(("indirect", v, f"{a.tag_prefix}_ind{v}b"))

    print("=" * 70)
    print(f"  run      : {run_dir}")
    if a.rect:
        print(f"  rect     : {a.rect}  ({a.rect[2] * a.rect[3]} texels in the rectangle)")
    else:
        print("  rect     : none, WHOLE ATLAS into the run root")
    print(f"  variants : {[j[2] for j in jobs]}")
    print("=" * 70)

    out = {"run_dir": str(run_dir), "rect": list(a.rect) if a.rect else None,
           "tag_prefix": a.tag_prefix, "variants": []}
    failed = 0
    for family, value, tag in jobs:
        print()
        print(f"-- {tag}  ({family} = {value}, {value * value} samples per texel)")
        try:
            out["variants"].append(
                run_variant(run_dir, manifest_path, a.rect, family, value, tag,
                            a.force, a.dry_run))
        except SystemExit:
            raise
        except Exception as exc:                                  # noqa: BLE001
            failed += 1
            print(f"  x {tag} failed: {type(exc).__name__}: {exc}")
            out["variants"].append({"tag": tag, "family": family, "value": value,
                                    "status": f"failed:{type(exc).__name__}"})
            if not a.keep_going:
                break

    if not a.dry_run:
        dest = run_dir / "roi" / f"{a.tag_prefix}_sweep.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print()
        print(f"  summary -> {dest}")

    print()
    print("  verify the ROI is not an approximation with:")
    base = a.irradiance[0] if a.irradiance else None
    if base is not None:
        print(f"    python compare_roi_run.py {run_dir} "
              f"--tag {a.tag_prefix}_irr{base}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
