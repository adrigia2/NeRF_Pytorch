#!/usr/bin/env python
"""export_rerender_scenes.py -- one Blender scene per run and material mode.

    python export_rerender_scenes.py [--run-root <sweep>] [--out D:/rerender] [--dry-run]

Turns the maps a finished sweep reconstructed into Blender scenes that can be opened and
orbited: one `.blend` per (config, scene, material mode).  It renders nothing.  Every
invocation of `rerender_run.py` carries `--no-render --save-blend --asset-dir`, so what
lands on disk is the scene plus the files it links to, and nothing else.

Layout produced under `--out`:

    _assets/<scene>/                     normal, HDR, GT maps: shared by both configs
                                         and by every mode of that scene
    gt_control/<scene>/                  the ORIGINAL textures in the same rig
    <config>/<scene>/pbr_<source>/       albedo_pbr + metallic + roughness
    <config>/<scene>/lambert_<source>/   diffuse albedo, metallic 0, roughness 1
    export_manifest.json, README.md

`gt_control` is built once per scene, not once per config: it mounts the bake's original
textures, which owe nothing to the NeRF, so the two configs would write identical files.
It is the control that separates the two possible causes of a difference: if the GT scene
matches the training images, then whatever the `pbr_` and `lambert_` scenes get wrong is
reconstruction and not assembly.

The sweep folder is read and never written: every invocation gets an explicit `--out`
outside it, which is where `rerender_run.py` puts the job, the dilated textures and the
blend.

The real work (map resolution, dilation of the UV island borders, the material, the World)
all stays in `rerender_run.py`, which is imported here for the dry run and spawned as a
subprocess for the real thing.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rerender_run import _resolve_maps  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:    # noqa: BLE001  -- stream not reconfigurable: never mind
        pass


DEFAULT_RUN_ROOT = "D:/tesi_output/test_sword_shield_after_fix_irradiance"
DEFAULT_OUT = "D:/rerender"
DEFAULT_REPO = "C:/Users/adria/Documents/GitHub/Tesi/OptixProjectCMake"

CONFIGS = ["exp_l1_d02", "softplus_relmseraw_d02"]
SCENES = [
    "SwordShieldStudio",
    "SwordShieldNight",
    "TableAndOtherInteriorWithSpecular",
    "TableAndOtherInteriorWithSpecularNight",
    "TableAndOtherInteriorNoSpecular",
]
MODES = ["pbr", "lambert", "gt"]

# Scene -> HDR of the original bake, relative to the repo's `Scenes` folder.  Same table as
# the docstring of rerender_run.py, which also carries the reason it has to be right: night
# and studio share model, cameras and layout, so a scene lit by the wrong one *looks* fine.
SKYBOX = {
    "SwordShieldStudio":
        "SwordShield Thesis/Blender/assets/hdrs/wooden_studio_13_4k.exr",
    "SwordShieldNight":
        "SwordShield Thesis/Blender/assets/hdrs/cobblestone_street_night_4k.exr",
    "TableAndOtherInteriorWithSpecular":
        "TableAndOtherInterior/Blender/assets/hdri/wooden_studio_13_4k.exr",
    "TableAndOtherInteriorNoSpecular":
        "TableAndOtherInterior/Blender/assets/hdri/wooden_studio_13_4k.exr",
    "TableAndOtherInteriorWithSpecularNight":
        "TableAndOtherInterior/BlenderBakedSmoothNight/cobblestone_street_night_4k.exr",
}


def _tag(mode: str, source: str) -> str:
    return "gt_control" if mode == "gt" else f"{mode}_{source}"


def _out_dir(out_root: Path, config: str, scene: str, mode: str, source: str) -> Path:
    """Where a scene goes.

    `gt_control` hangs off the root and not off a config: it does not depend on one, and
    filing it under `exp_l1_d02` would suggest that it does.
    """
    if mode == "gt":
        return out_root / "gt_control" / scene
    return out_root / config / scene / _tag(mode, source)


def _jobs(args, out_root: Path) -> list:
    """The (config, scene, mode) list to build, with `gt` deduplicated across configs."""
    jobs = []
    for scene in args.scenes:
        for mode in args.modes:
            configs = args.configs[:1] if mode == "gt" else args.configs
            for config in configs:
                jobs.append({
                    "config": config,
                    "scene": scene,
                    "mode": mode,
                    "source": args.source,
                    "run_dir": Path(args.run_root) / config / scene,
                    "out_dir": _out_dir(out_root, config, scene, mode, args.source),
                    "asset_dir": out_root / "_assets" / scene,
                    "skybox": Path(args.repo) / "Scenes" / SKYBOX[scene],
                })
    return jobs


def _resolve(job: dict) -> dict:
    """Resolve a job's inputs without touching a byte of them.

    Reuses `rerender_run._resolve_maps`, so the dry run checks exactly the paths the real
    run will use, instead of a hand-written copy of the same rules that could drift away
    from them.
    """
    run_dir = job["run_dir"]
    manifest_path = run_dir / "run_manifest.json"
    tf_path = run_dir / "transforms_extended.json"
    for p in (manifest_path, tf_path):
        if not p.exists():
            raise FileNotFoundError(str(p))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tf_ext = json.loads(tf_path.read_text(encoding="utf-8"))
    model = Path(manifest["scene"]["model_path"])
    if not model.exists():
        raise FileNotFoundError(str(model))
    if not job["skybox"].exists():
        raise FileNotFoundError(str(job["skybox"]))
    resolved = _resolve_maps(run_dir, tf_ext, manifest, job["mode"], job["source"])
    return {"model": model, "maps": resolved["maps"],
            "n_frames": len(tf_ext.get("frames", []))}


def _command(job: dict, args) -> list:
    cmd = [args.python, str(Path(__file__).resolve().parent / "rerender_run.py"),
           str(job["run_dir"]),
           "--skybox", str(job["skybox"]),
           "--materials", job["mode"],
           "--out", str(job["out_dir"]),
           "--asset-dir", str(job["asset_dir"]),
           "--no-render", "--save-blend",
           "--dilate", str(args.dilate)]
    if job["mode"] != "gt":
        cmd += ["--source", job["source"]]
    if args.blender:
        cmd += ["--blender", args.blender]
    return cmd


README = """# Rerendered scenes -- {n} Blender scenes with the reconstructed textures

Generated by `NeRF_Pytorch/scripts/export_rerender_scenes.py` from the run
`{run_root}` on {when}.  Every scene is a `.blend` holding the real geometry, the World
with that scene's original HDR, and a `RerenderMaterial` replicating the graph of the
`BakedMaterial` of the source bake:

    base color -> Base Color      metallic -> Metallic
    roughness  -> Roughness       normal   -> Normal Map (space OBJECT) -> Normal

## Layout

    _assets/<scene>/                     normal, HDR and GT maps, shared
    gt_control/<scene>/                  ORIGINAL textures: a control, not a reconstruction
    <config>/<scene>/pbr_{source}/       albedo_pbr + metallic + roughness (the PBR fit)
    <config>/<scene>/lambert_{source}/   diffuse albedo, metallic 0, roughness 1

The paths inside the blends are relative, so this folder can be moved as one piece.
Moving a single scene out of it breaks its links into `_assets/`.

`gt_control` exists once per scene because it does not depend on the NeRF.  It is what
separates the two possible causes of a difference: if the GT scene matches the training
images, then whatever the other scenes get wrong is reconstruction, not assembly.

## Reading these scenes without being misled

- **Roughness is `cone_aperture / 180`, not a GGX alpha.**  The extremes agree (0 is a
  mirror, 1 is maximally rough), the middle does not.  It goes into the Principled as it
  is, deliberately: an aperture-to-alpha calibration belongs between the texture and the
  input, and this pipeline does not do one.
- **Roughness 1.0 is ambiguous**: it means both "the 180 degree cone won" and "the fit was
  not reliable here".  What tells them apart is
  `<run>/sources/{source}/pbr/r_valid.png`.
- **Black patches on metal are a convention, not a bug.**  `pbr_solver` writes albedo 0
  where a texel comes out fully specular, because a diffuse albedo is undefined there;
  Blender's Principled reads Base Color as the metal's reflection colour, so those texels
  become mirrors reflecting nothing.  `rerender_run.py` counts them and says so.
- **The light is the original HDR, not the one the pipeline used.**  The reconstruction ran
  on `skybox_nerf_baked.exr`, baked from the NeRF; these scenes show the maps under
  ground-truth light.  For the other reading, rerun with
  `--skybox <run>/skybox_nerf_baked.exr` into a different folder.
- **`exp_l1_d02/SwordShieldNight` has degenerate maps** (an albedo of 1.3 MB against the
  ~100 MB of every other scene, metallic and roughness nearly empty).  That is the known
  behaviour of exp + L1 on the night sword, where the NeRF emits zero radiance, and not a
  failure of this export.  For that scene, read the softplus config.
- The maps come from source `{source}`.  `sources/nerf` exists in only 3 of the 10 runs;
  to add it, rerun with `--source nerf --scenes ...` on those.

## Rebuilding one scene

    python NeRF_Pytorch/scripts/export_rerender_scenes.py --configs <config>
        --scenes <scene> --modes pbr --force

`export_manifest.json` records, for every scene, which run it was read from and which map
files went into it.
"""


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="export_rerender_scenes",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", default=DEFAULT_RUN_ROOT, dest="run_root",
                    help=f"sweep folder, the one holding the configs "
                         f"(default: {DEFAULT_RUN_ROOT})")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"output root (default: {DEFAULT_OUT})")
    ap.add_argument("--repo", default=DEFAULT_REPO,
                    help="OptixProjectCMake folder, where the HDRs are looked up")
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    ap.add_argument("--scenes", nargs="+", default=SCENES, choices=SCENES)
    ap.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    ap.add_argument("--source", default="gt", help="source of the maps: gt | nerf (default: gt)")
    ap.add_argument("--dilate", type=float, default=8.0,
                    help="radius in texels of the outward fill of the reconstructed maps")
    ap.add_argument("--force", action="store_true", help="rebuild the scenes already on disk")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="resolve and check every input, print what would be built, "
                         "copy nothing and launch nothing")
    ap.add_argument("--blender", default=None, help="passed through to rerender_run.py")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter used to run rerender_run.py (needs OpenEXR and scipy)")
    args = ap.parse_args()

    out_root = Path(args.out).resolve()
    jobs = _jobs(args, out_root)
    print(f"run root  {args.run_root}")
    print(f"output    {out_root}")
    print(f"scenes    {len(jobs)} to build "
          f"({len(args.scenes)} scenes x {len(args.modes)} modes, gt deduplicated)")
    print()

    results, failures, skipped = [], 0, 0
    t0 = time.time()
    for i, job in enumerate(jobs, 1):
        blend = job["out_dir"] / "scene.blend"
        head = (f"[{i}/{len(jobs)}] {job['config']} / {job['scene']} / "
                f"{_tag(job['mode'], job['source'])}")
        if blend.exists() and not args.force:
            print(f"{head}  -- already on disk, skipped (--force to redo it)")
            skipped += 1
            continue
        bar = "=" * 78
        print(f"{bar}\n{head}\n{bar}", flush=True)

        try:
            info = _resolve(job)
        except (FileNotFoundError, SystemExit) as exc:
            print(f"  X  input not usable: {exc}")
            results.append({**{k: str(v) for k, v in job.items()},
                            "status": "unresolved", "error": str(exc)})
            failures += 1
            continue

        if args.dry_run:
            print(f"  model   {info['model']}")
            print(f"  skybox  {job['skybox']}")
            for k, v in info["maps"].items():
                print(f"  {k:11s} {v}")
            print(f"  -> {blend}")
            print(f"  $ {' '.join(_command(job, args))}")
            results.append({**{k: str(v) for k, v in job.items()},
                            "status": "dry-run", "maps": info["maps"]})
            continue

        rc = subprocess.call(_command(job, args))
        ok = rc == 0 and blend.exists()
        if not ok:
            print(f"  X  rerender_run exited with {rc}, blend "
                  f"{'present' if blend.exists() else 'missing'}")
            failures += 1
        results.append({**{k: str(v) for k, v in job.items()},
                        "status": "ok" if ok else "failed",
                        "returncode": rc,
                        "blend": str(blend) if blend.exists() else None,
                        "blend_bytes": blend.stat().st_size if blend.exists() else None,
                        "maps": info["maps"]})
        print(flush=True)

    built = sum(1 for r in results if r["status"] == "ok")
    print("=" * 78)
    print(f"built {built}, skipped {skipped}, failed {failures}, "
          f"in {(time.time() - t0) / 60.0:.1f} min")

    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "run_root": str(args.run_root),
            "source": args.source,
            "dilate": args.dilate,
            "built": built, "skipped": skipped, "failed": failures,
            "scenes": results,
        }
        mpath = out_root / "export_manifest.json"
        # Merge with what a previous invocation recorded: partial reruns are the norm here,
        # and a manifest that only knows about the last --scenes would be worse than none.
        if mpath.exists():
            try:
                old = json.loads(mpath.read_text(encoding="utf-8"))
                keep = [r for r in old.get("scenes", [])
                        if not any(r.get("out_dir") == n.get("out_dir") for n in results)]
                manifest["scenes"] = keep + results
            except (json.JSONDecodeError, OSError) as exc:
                print(f"  !  could not merge the previous export_manifest.json: {exc}")
        mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (out_root / "README.md").write_text(
            README.format(n=len(manifest["scenes"]), run_root=args.run_root,
                          when=time.strftime("%Y-%m-%d"), source=args.source),
            encoding="utf-8")
        print(f"manifest  {mpath}")
        print(f"readme    {out_root / 'README.md'}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
