#!/usr/bin/env python
"""Re-run Step 3 + Step 4 of an existing run, restricted to a ROI.

It is there to verify the texture-space ROI: the full run on disk is the reference, this
script re-runs it on a portion and leaves the result in
`<run_dir>/roi/<tag>/`, after which `compare_roi_run.py` compares the two trees.

The configuration is **not** transcribed by hand but rebuilt from `run_manifest.json`: a
single difference (an aperture, a sample count, the
depth window) would make the comparison meaningless.
It checks explicitly that the rebuilt config matches the manifest's on every key except
those it has to change, and stops if it does not.

Usage:
    python roi_rerun.py <run_dir> --rect X0 Y0 W H [--tag NAME] [--mask FILE]

It calls `run_pipeline` directly rather than `run_pipeline_multi`, which would rewrite
`run_manifest.json` and append to the reference run's `console.log`.
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

# Keys this script changes on purpose: they are the only differences allowed with
# respect to the reference run's manifest.
_EXPECTED_DIFFS = {
    "run_step1", "run_step2", "run_step3", "run_step4",
    "render.output_dir",
    "render.roi_rect", "render.roi_mask_path", "render.roi_mask_threshold",
    "render.roi_tag",
    # A field DELETED from the config, not an override: the solver's diffuse gate was
    # removed, but the manifests already written still contain it, and without this
    # entry every earlier run would be rejected.
    "render.pbr_diffuse_cv_gate",
}


def _enc(o: object) -> object:
    """Same encoding as _write_run_manifest, so the dicts can be compared."""
    if isinstance(o, Enum):
        return o.name
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _encode_config(cfg: PipelineConfig) -> dict:
    return json.loads(json.dumps(asdict(cfg), default=_enc))


def _build(cls, data: dict, path: str) -> tuple[object, list[str]]:
    """Instantiate the dataclass `cls` from the manifest's values.

    The ImageFormats come back from their name (the manifest serialises them with Enum.name).
    Manifest keys the dataclass no longer has, and fields added after that run (typically
    the roi_* ones), are reported rather than ignored silently: the first category means
    the config changed underfoot.
    """
    notes: list[str] = []
    known = {f.name: f for f in fields(cls)}
    kwargs = {}
    for name, value in data.items():
        f = known.get(name)
        if f is None:
            notes.append(f"    ⚠  {path}{name}: present in the manifest but not in {cls.__name__}")
            continue
        if isinstance(f.default, ImageFormat):
            value = ImageFormat[value] if isinstance(value, str) else value
        kwargs[name] = value
    for name in known:
        if name not in data and name != "render":
            notes.append(f"    ·  {path}{name}: absent from the manifest, using the default")
    return cls(**kwargs), notes


def config_from_manifest(manifest_path: Path) -> tuple[PipelineConfig, dict, list[str]]:
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    raw = dict(manifest["config"])
    render_raw = raw.pop("render")
    rc, notes_r = _build(RenderConfig, render_raw, "render.")
    # `render` has to be passed to the constructor: its default_factory is RenderConfig,
    # which has three mandatory fields and therefore cannot be instantiated empty.
    raw["render"] = rc
    cfg, notes_c = _build(PipelineConfig, raw, "")
    return cfg, manifest, notes_c + notes_r


def check_against_manifest(cfg: PipelineConfig, manifest: dict,
                           expected_diffs: "set[str] | None" = None) -> list[str]:
    """Differences between the rebuilt config and the manifest's, overrides excluded.

    `expected_diffs` is the set of keys the caller changes on purpose; by default the ROI
    ones. `rerun_irradiance.py` parametrises it, reusing this validation with a different
    set of overrides."""
    if expected_diffs is None:
        expected_diffs = _EXPECTED_DIFFS
    got, want = _encode_config(cfg), manifest["config"]
    diffs = []

    def walk(a: dict, b: dict, prefix: str) -> None:
        for k in sorted(set(a) | set(b)):
            key = f"{prefix}{k}"
            va, vb = a.get(k, "<absent>"), b.get(k, "<absent>")
            if isinstance(va, dict) and isinstance(vb, dict):
                walk(va, vb, f"{key}.")
            elif vb == "<absent>":
                # A field added to the dataclass AFTER that run: the manifest cannot
                # contain it and the rebuild uses the default, which `_build` has already
                # reported among the notes. Counting it as a divergence would make the
                # script unusable on every run older than the latest field introduced,
                # i.e. exactly the use case.
                continue
            elif va != vb and key not in expected_diffs:
                diffs.append(f"{key}: rebuilt={va!r}  manifest={vb!r}")

    walk(got, want, "")
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="the reference run folder (holds run_manifest.json)")
    ap.add_argument("--rect", nargs=4, type=int, metavar=("X0", "Y0", "W", "H"),
                    help="rectangular ROI in IUM texels")
    ap.add_argument("--mask", default="", help="ROI mask image (optional)")
    ap.add_argument("--mask-threshold", type=float, default=0.5)
    ap.add_argument("--tag", default="", help="sandbox name (default: derived)")
    ap.add_argument("--dry-run", action="store_true",
                    help="rebuild and check the config, then stop")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        print(f"✗ manifest not found: {manifest_path}")
        return 2
    if not args.rect and not args.mask:
        print("✗ at least one of --rect or --mask is required")
        return 2

    cfg, manifest, notes = config_from_manifest(manifest_path)
    print(f"Config rebuilt from {manifest_path}")
    print(f"  original run: {manifest['timestamp']}  |  {manifest.get('run_note', '')}")
    for n in notes:
        print(n)

    # Override: only the reconstruction steps, the ROI, and the folder (the reference run
    # was copied elsewhere, so the manifest's path is stale).
    cfg.run_step1 = cfg.run_step2 = False
    cfg.run_step3 = cfg.run_step4 = True
    cfg.render.output_dir = str(run_dir)
    cfg.render.roi_rect = list(args.rect) if args.rect else None
    cfg.render.roi_mask_path = args.mask
    cfg.render.roi_mask_threshold = args.mask_threshold
    cfg.render.roi_tag = args.tag

    diffs = check_against_manifest(cfg, manifest)
    if diffs:
        print("\n✗ the rebuilt config diverges from the manifest on unexpected keys:")
        for d in diffs:
            print(f"    {d}")
        print("  The comparison with the full run would not be valid: aborted.")
        return 1
    print("  ✓ config identical to the manifest (up to steps, output_dir and ROI)")

    assets_dir, tag = _roi_assets_dir(cfg.render, run_dir)
    print(f"  sandbox: {assets_dir}")
    if args.dry_run:
        return 0

    log_path = str(assets_dir / "console.log")
    with _console_to_file(log_path):
        print(f"{'=' * 70}")
        print(f"  ROI rerun  : {tag}")
        print(f"  Reference:   {run_dir}")
        print(f"  Sandbox    : {assets_dir}")
        print(f"  rect={cfg.render.roi_rect}  mask={cfg.render.roi_mask_path or '-'}")
        print(f"{'=' * 70}")
        run_pipeline(cfg, tb_enabled=False)
    print(f"\n✓ done. Compare with:\n    python compare_roi_run.py {run_dir} --tag {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
