#!/usr/bin/env python
"""turntable_video.py -- a 360 degree turntable mp4 of one rerendered Blender scene.

    python turntable_video.py <scene.blend | scene_dir> [--engine cycles|eevee] ...

`scene.blend` is one of the scenes built by `export_rerender_scenes.py` under
`D:/rerender`, for example:

    python turntable_video.py D:/rerender/gt_control/SwordShieldStudio
    python turntable_video.py D:/rerender/exp_l1_d02/SwordShieldStudio/pbr_gt

The camera orbits the object once around the world Z axis, slightly above the horizon,
and Blender writes the frames straight into an H.264 mp4.  Pivot, radius, elevation,
start azimuth, direction, frame count, frame rate, engine and the environment the object
is lit by are all flags, so the same shot can be reproduced across the variants of a
scene and the videos of `gt_control`, `pbr_gt` and `lambert_gt` line up frame by frame.

-------------------------------------------------------------------------------
The source scenes are read-only
-------------------------------------------------------------------------------
`D:/rerender` is an 18 GB artefact that took a full export run to build, and nothing in
this script may touch it.  Three things enforce that, in order of how much they would
have to fail together:

  * nothing is written inside the tree.  `--out` defaults *outside* it: the launcher
    walks up to the folder holding `export_manifest.json` and mirrors the relative
    subpath into a sibling `<root>_video/`, so
    `D:/rerender/gt_control/SwordShieldStudio/scene.blend` produces
    `D:/rerender_video/gt_control/SwordShieldStudio/turntable.mp4`;
  * this file contains no call able to write a blend.  No `wm.save_mainfile`, no
    `wm.save_as_mainfile`, no `file.make_paths_relative`, no `file.pack_all`, no
    `outliner.orphans_purge` - the list `rerender_run.py` legitimately uses and this
    script must not.  Every edit (pivot, camera parent, world nodes, render settings)
    lives in RAM and dies with the process;
  * the launcher fingerprints `scene.blend` (size, mtime, sha1) before starting Blender
    and re-checks it afterwards, aborting if anything moved.

-------------------------------------------------------------------------------
How it runs
-------------------------------------------------------------------------------
A single file in two modes depending on where it is executed (`try: import bpy`), the
same split as `rerender_run.py`:

  * with the normal python -> launcher: resolves the blend, estimates the orbit from the
    run's camera rig with numpy, writes a JSON job and invokes `blender --background
    --factory-startup <scene.blend> --python <itself>`;
  * inside Blender          -> reads the job, builds the orbit and renders.

-------------------------------------------------------------------------------
The neutral environment, and what it is for
-------------------------------------------------------------------------------
`--env-color R G B` replaces the scene's HDRI with a uniform environment.  With colour
(1,1,1) and strength S the incident radiance is a constant S from every direction, so a
background pixel is exactly S, a Lambertian surface returns rho*S and a metal returns
base_color*S: at S = 1 the render *is* the albedo map, modulated only by ambient
occlusion.  That is the mode for comparing ground-truth against reconstructed textures
without the HDRI's directional lighting.  Through AgX the background lands at 197/255
and the GT textures of SwordShieldStudio put the object at 43-164, so the two never
collide; `--env-strength 2` lifts the background to 217/255 if the object looks lost.

The caveat is structural, not a bug: a uniform environment has no direction, so there
are no shadows, and the GGX directional albedo is very nearly roughness-independent - a
mirror and a matte metal both return F0*S.  Albedo differences show crisply, roughness
and metallic differences essentially do not.  On the sword/shield asset that matters,
since metallic p50 is 0.572 and over half the surface is metal.  It is a deliberate
ablation; it is not the mode for judging the specular fit.

-------------------------------------------------------------------------------
Things that would go wrong without a symptom
-------------------------------------------------------------------------------
1. In Blender 5.0 `image_settings.file_format = "FFMPEG"` raises `TypeError` unless
   `media_type = "VIDEO"` is set first, and the file comes out named
   `turntable0001-0120.mp4` unless `render.filepath` already ends in `.mp4`.
2. The blends ship `view_transform = "Raw"`, correct for the 32-bit EXR they were built
   to produce and a linear passthrough with no sRGB encode for an 8-bit video: left
   alone, the mp4 comes out crushed and dark.
3. EEVEE has `use_raytracing = False` in these blends.  Off, there are no screen-space
   reflections at all and the metallic/roughness maps are invisible - the render looks
   plausible and shows nothing of what is being compared.  EEVEE is a fast preview
   either way: its reflections cannot contain what is off-screen, so a metal turntable
   will not agree with Cycles.
4. With a bad blend path Blender prints an error, carries on, and runs this script
   against the factory startup scene.  Hence the `bpy.data.filepath` guard.
5. Blender 5.0 renamed the EEVEE engine id back to `BLENDER_EEVEE` and removed
   `Action.fcurves`.  Both are resolved defensively here; do not copy
   `blender_renderer.py:443`, which still says `BLENDER_EEVEE_NEXT`.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

try:                     # inside Blender
    import bpy
    IN_BLENDER = True
except ImportError:      # launcher, normal python
    IN_BLENDER = False

# On Windows stdout arrives as cp1252 and the characters used in the messages blow it up
# halfway through a run.  Setting PYTHONIOENCODING inside the script would be too late:
# the stream already exists when the module is imported.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:    # noqa: BLE001  -- stream not reconfigurable: never mind
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_BLENDER = r"C:/Program Files/Blender Foundation/Blender 5.0/blender.exe"

DEFAULT_FRAMES = 120
DEFAULT_FPS = 30                    # 120/30 -> a 4 s loop at 3 deg per frame
DEFAULT_ELEVATION = 20.0
DEFAULT_START_AZIMUTH = 0.0
DEFAULT_VIEW_TRANSFORM = "AgX"
DEFAULT_ENV_STRENGTH = 1.0          # only applied when --env-color is given
DEFAULT_SAMPLES = {"cycles": 128, "eevee": 64}   # different properties, different scales

FIT_MARGIN = 1.05                   # slack around the bounding sphere when auto-framing
MAX_DEG_PER_FRAME = 3.0             # above this the turntable visibly strobes
FLICKER_SAMPLES = 256               # below this, per-frame denoising can shimmer
COND_MAX = 1e6                      # least-squares rig solve: give up past this

VIDEO_NAME = "turntable.mp4"        # the .mp4 in the path is what kills the frame suffix
PREVIEW_NAME = "preview"            # deliberately extensionless
JOB_NAME = "turntable_job.json"
META_NAME = "turntable_meta.json"
ROOT_MARKER = "export_manifest.json"

FFMPEG = dict(format="MPEG4", codec="H264", constant_rate_factor="HIGH",
              ffmpeg_preset="GOOD", audio_codec="NONE")

CS_FLOAT = "Linear Rec.709"         # EXR/HDR environments: scene-linear, and honest
CS_BYTE = "sRGB"                    # 8-bit environments: display-encoded

VIEW_TRANSFORMS = ("Standard", "Filmic", "Filmic Log", "AgX", "Khronos PBR Neutral",
                   "Raw", "False Color")


# ══════════════════════════════════════════════════════════════════════════════
# LAUNCHER  (normal python: numpy available)
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_blend(arg: str) -> Path:
    """Accept either a .blend or a folder holding one, so a scene directory works."""
    p = Path(arg).expanduser()
    if not p.exists():
        raise SystemExit(f"✗ not found: {p}")
    p = p.resolve()
    if p.is_dir():
        cands = sorted(p.glob("*.blend"))          # *.blend1 backups do not match
        if not cands:
            raise SystemExit(f"✗ no .blend in {p}")
        if len(cands) > 1:
            named = [c for c in cands if c.name == "scene.blend"]
            if len(named) != 1:
                names = ", ".join(c.name for c in cands)
                raise SystemExit(f"✗ {p} holds several blends ({names}): name one")
            cands = named
        p = cands[0]
    if p.suffix.lower() != ".blend":
        raise SystemExit(f"✗ not a .blend: {p}")
    return p


def _fingerprint(path: Path) -> dict:
    """Size, mtime and sha1 of the source blend.

    Taken before Blender starts and re-checked after it exits.  The scenes are the
    expensive artefact here and this script has no business modifying one, so the claim
    is verified rather than asserted.  The files are ~250 kB: the hash is free.
    """
    import hashlib
    st = path.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns,
            "sha1": hashlib.sha1(path.read_bytes()).hexdigest()}


def _default_out(blend: Path) -> Path:
    """Mirror the scene's position under a sibling `<root>_video/`, outside the tree.

    The rerender root is the folder holding `export_manifest.json`.  Without one (a
    scene copied elsewhere) fall back to a `turntable/` next to the blend.
    """
    for root in blend.parents:
        if (root / ROOT_MARKER).exists():
            return root.parent / f"{root.name}_video" / blend.parent.relative_to(root)
    return blend.parent / "turntable"


def _camera_rays(transforms: Path):
    """Centres and unit view directions of the run's cameras, in the Blender frame.

    `rerender_run.py` assigns the NeRF c2w straight to `matrix_world` with
    `apply_axis_correction=False`, so the matrices are already Blender-frame: the centre
    is column 3 and the view direction is -column 2 (a camera looks down its local -Z).
    """
    import numpy as np
    tf = json.loads(transforms.read_text(encoding="utf-8"))
    frames = tf.get("frames") or []
    mats = [f["transform_matrix"] for f in frames if "transform_matrix" in f]
    if len(mats) < 2:
        return None
    m = np.asarray(mats, dtype=np.float64)
    if m.shape[1:] != (4, 4):
        return None
    centres = m[:, :3, 3]
    dirs = -m[:, :3, 2]
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    if not np.all(norms > 1e-9):
        return None
    return centres, dirs / norms


def _orbit_from_rig(centres, dirs) -> dict | None:
    """Least-squares closest point to every camera view ray, plus the orbit radius.

    A = sum(I - d dT), b = sum((I - d dT) p).  This is the point the capture rig actually
    converges on, which for the interior scenes is a far better pivot than the mesh
    bounding box - that bbox contains the room.  Degenerate when the rays are parallel,
    hence the conditioning guard.
    """
    import numpy as np
    proj = np.eye(3) - dirs[:, :, None] * dirs[:, None, :]
    a = proj.sum(axis=0)
    b = np.einsum("nij,nj->i", proj, centres)
    if not np.all(np.isfinite(a)) or np.linalg.cond(a) > COND_MAX:
        return None
    try:
        target = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return None
    radii = np.linalg.norm(centres - target, axis=1)
    return {"target": [float(v) for v in target],
            "radius": float(radii.mean()),
            "radius_min": float(radii.min()),
            "radius_max": float(radii.max()),
            "n_cameras": int(len(radii))}


def _estimate_orbit(blend: Path) -> dict | None:
    """Default orbit from the sibling `rerender_job.json`, or None with the reason why."""
    job_path = blend.parent / "rerender_job.json"
    if not job_path.exists():
        print(f"[turntable] no {job_path.name} next to the blend: "
              f"orbit will come from the mesh bounding box")
        return None
    try:
        transforms = Path(json.loads(job_path.read_text(encoding="utf-8"))["transforms"])
    except Exception as exc:                       # noqa: BLE001
        print(f"[turntable] could not read {job_path.name} ({exc}): "
              f"orbit will come from the mesh bounding box")
        return None
    if not transforms.exists():
        print(f"[turntable] {transforms} is gone: "
              f"orbit will come from the mesh bounding box")
        return None
    rays = _camera_rays(transforms)
    rig = _orbit_from_rig(*rays) if rays else None
    if rig is None:
        print("[turntable] the camera rig does not converge: "
              "orbit will come from the mesh bounding box")
        return None
    tx, ty, tz = rig["target"]
    print(f"[turntable] rig: {rig['n_cameras']} cameras converge on "
          f"({tx:.3f}, {ty:.3f}, {tz:.3f}), radius "
          f"{rig['radius_min']:.2f} / {rig['radius']:.2f} / {rig['radius_max']:.2f} "
          f"(min/mean/max)")
    return rig


def _build_job(args, blend: Path, out_dir: Path, rig: dict | None) -> dict:
    target = list(args.target) if args.target else (rig["target"] if rig else None)
    radius = args.radius if args.radius else (rig["radius"] if rig else None)
    return {
        "blend": str(blend),
        "blend_sha1": args.fingerprint["sha1"],
        "out_dir": str(out_dir),
        "video": str(out_dir / VIDEO_NAME),
        "preview": bool(args.preview),
        "target": target,
        "target_from": "cli" if args.target else ("rig" if rig else "bbox"),
        "radius": radius,
        "radius_from": "cli" if args.radius else ("rig" if rig else "bbox"),
        "rig": rig,
        "elevation": args.elevation,
        "start_azimuth": args.start_azimuth,
        "direction": args.direction,
        "fov": args.fov,
        "frames": args.frames,
        "fps": args.fps,
        "engine": args.engine,
        "samples": args.samples,
        "device": args.device,
        "resolution": list(args.resolution) if args.resolution else None,
        "percentage": args.percentage,
        "view_transform": args.view_transform,
        "exposure": args.exposure,
        "skybox": str(Path(args.skybox).resolve()) if args.skybox else None,
        "env_color": list(args.env_color) if args.env_color else None,
        "env_strength": args.env_strength,
        "script_dir": str(Path(__file__).resolve().parent),
    }


def launcher() -> int:
    import argparse
    import subprocess
    import time

    ap = argparse.ArgumentParser(
        prog="turntable_video",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("blend", help="scene.blend, or the folder holding one")

    shot = ap.add_argument_group("shot")
    shot.add_argument("--target", nargs=3, type=float, metavar=("X", "Y", "Z"),
                      help="pivot of the orbit (default: where the run's cameras "
                           "converge, else the mesh bounding-box centre)")
    shot.add_argument("--radius", type=float,
                      help="orbit radius (default: the run's mean camera distance, else "
                           "the distance that frames the bounding sphere)")
    shot.add_argument("--elevation", type=float, default=DEFAULT_ELEVATION,
                      help=f"degrees above the horizon (default: {DEFAULT_ELEVATION})")
    shot.add_argument("--start-azimuth", type=float, default=DEFAULT_START_AZIMUTH,
                      dest="start_azimuth",
                      help=f"azimuth of frame 1, degrees "
                           f"(default: {DEFAULT_START_AZIMUTH})")
    shot.add_argument("--direction", default="ccw", choices=["ccw", "cw"],
                      help="turn direction seen from above (default: ccw)")
    shot.add_argument("--fov", type=float,
                      help="horizontal field of view in degrees (default: the camera's "
                           "own, 39.6 in these scenes)")

    timing = ap.add_argument_group("timing")
    timing.add_argument("--frames", type=int, default=DEFAULT_FRAMES,
                        help=f"frames for the full turn (default: {DEFAULT_FRAMES})")
    timing.add_argument("--fps", type=int, default=DEFAULT_FPS,
                        help=f"frame rate; sets the playback speed only, the orbit is "
                             f"keyed on frames (default: {DEFAULT_FPS})")

    engine = ap.add_argument_group("engine")
    engine.add_argument("--engine", default="cycles", choices=["cycles", "eevee"],
                        help="cycles: the reference. eevee: seconds instead of minutes, "
                             "but screen-space reflections cannot show what is "
                             "off-screen, so metals will not agree (default: cycles)")
    engine.add_argument("--samples", type=int,
                        help=f"cycles.samples or eevee.taa_render_samples depending on "
                             f"--engine (default: {DEFAULT_SAMPLES['cycles']} cycles, "
                             f"{DEFAULT_SAMPLES['eevee']} eevee)")
    engine.add_argument("--device", default="GPU", choices=["GPU", "CPU"],
                        help="Cycles only; EEVEE always renders on the GPU")

    env = ap.add_argument_group("environment")
    world = env.add_mutually_exclusive_group()
    world.add_argument("--skybox", help="replace the scene's HDRI with this .exr/.hdr")
    world.add_argument("--env-color", nargs=3, type=float, metavar=("R", "G", "B"),
                       dest="env_color",
                       help="replace the environment with a uniform colour. `--env-color "
                            "1 1 1` is the neutral backdrop: no directional light, so the "
                            "render reads as the albedo map (see the module docstring)")
    env.add_argument("--env-strength", type=float, dest="env_strength",
                     help=f"Background strength (default: {DEFAULT_ENV_STRENGTH} with "
                          f"--env-color, otherwise the scene's own)")

    img = ap.add_argument_group("image")
    img.add_argument("--resolution", nargs=2, type=int, metavar=("W", "H"),
                     help="default: the scene's own, 1920x1080")
    img.add_argument("--percentage", type=int, default=100,
                     help="resolution scale in %% (default: 100)")
    img.add_argument("--view-transform", default=DEFAULT_VIEW_TRANSFORM,
                     dest="view_transform",
                     help=f"the scenes ship 'Raw', which is wrong for 8-bit output "
                          f"(default: {DEFAULT_VIEW_TRANSFORM}; one of "
                          f"{', '.join(VIEW_TRANSFORMS)})")
    img.add_argument("--exposure", type=float, default=0.0,
                     help="view-transform exposure in stops (default: 0)")

    ap.add_argument("--out", default=None,
                    help="output folder (default: mirrored under a sibling "
                         "<rerender_root>_video/, so nothing is written in the tree)")
    ap.add_argument("--blender", default=DEFAULT_BLENDER)
    ap.add_argument("--preview", action="store_true",
                    help="render only frame 1, as preview.png: the cheap way to check "
                         "framing and lighting before committing to a whole turn")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing turntable.mp4")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="resolve everything and print the job without launching Blender")
    args = ap.parse_args()

    blend = _resolve_blend(args.blend)
    blender = Path(args.blender)
    if not blender.exists():
        raise SystemExit(f"✗ Blender not found: {blender}  (use --blender)")
    if args.frames < 2:
        raise SystemExit(f"✗ --frames {args.frames}: a turn needs at least 2")
    if args.fps < 1:
        raise SystemExit(f"✗ --fps {args.fps}: must be >= 1")
    if args.samples is None:
        args.samples = DEFAULT_SAMPLES[args.engine]
    if args.samples < 1:
        raise SystemExit(f"✗ --samples {args.samples}: must be >= 1")
    if args.skybox and not Path(args.skybox).exists():
        raise SystemExit(f"✗ skybox not found: {args.skybox}")
    if args.env_strength is None and args.env_color:
        args.env_strength = DEFAULT_ENV_STRENGTH

    out_dir = Path(args.out).resolve() if args.out else _default_out(blend)
    video = out_dir / VIDEO_NAME
    if video.exists() and not args.preview and not args.force:
        raise SystemExit(f"✗ {video} already exists (--force to overwrite)")

    args.fingerprint = _fingerprint(blend)
    rig = _estimate_orbit(blend)
    job = _build_job(args, blend, out_dir, rig)

    step = 360.0 / args.frames
    print(f"[turntable] {blend}")
    print(f"[turntable] {'preview' if args.preview else 'video'} -> {out_dir}")
    print(f"[turntable] {args.frames} frames @ {args.fps} fps = "
          f"{args.frames / args.fps:.1f} s, {step:.2f} deg/frame, "
          f"{args.engine} {args.samples} samples")
    if step > MAX_DEG_PER_FRAME:
        print(f"[turntable] ⚠  {step:.1f} deg/frame: expect strobing, "
              f"raise --frames or lower --fps")
    if args.env_color:
        print(f"[turntable] uniform environment {tuple(args.env_color)} "
              f"strength {args.env_strength}: no directional light, so roughness and "
              f"metallic differences will be nearly invisible (see --help)")
        if args.env_strength >= 1.0 and args.view_transform == "Standard":
            print(f"[turntable] ⚠  'Standard' clips at linear 1.0: a strength of "
                  f"{args.env_strength} gives a blown-out background. Use AgX.")
    elif args.skybox:
        print(f"[turntable] environment replaced with {Path(args.skybox).name}")

    out_dir.mkdir(parents=True, exist_ok=True)
    job_path = out_dir / JOB_NAME
    job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps(job, indent=2))
        print(f"\n--dry-run: job written to {job_path}, Blender not launched.")
        return 0

    # --factory-startup: no user add-ons and no local preferences influencing the render.
    # The blend is passed on the command line so that `bpy.data.filepath` is already set
    # when the script runs: that is what makes its relative //.. texture links resolve.
    cmd = [str(blender), "--background", "--factory-startup", str(blend),
           "--python", str(Path(__file__).resolve()), "--", str(job_path)]
    t0 = time.time()
    rc = subprocess.call(cmd)
    dt = (time.time() - t0) / 60.0

    after = _fingerprint(blend)
    if after != args.fingerprint:
        raise SystemExit(f"✗ {blend} CHANGED during the render.\n"
                         f"  before: {args.fingerprint}\n"
                         f"  after:  {after}\n"
                         f"  This script must never write a blend. Restore it from "
                         f"scene.blend1 and report this.")
    print(f"[turntable] source blend unchanged (sha1 {after['sha1'][:12]})")

    if rc != 0:
        raise SystemExit(f"✗ Blender exited with {rc}")
    produced = out_dir / (f"{PREVIEW_NAME}.png" if args.preview else VIDEO_NAME)
    if not produced.exists():
        raise SystemExit(f"✗ Blender finished but {produced} is missing")
    print(f"[turntable] {produced}  ({produced.stat().st_size / 1e6:.1f} MB, "
          f"{dt:.1f} min)")
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# INSIDE BLENDER
# ══════════════════════════════════════════════════════════════════════════════

def _set(owner, name: str, value) -> bool:
    """Assign a property that may not exist in this Blender version, and say so if not.

    EEVEE's property set moves between releases; a turntable is not worth aborting over
    one missing knob, but a silent skip would hide a version mismatch.
    """
    if not hasattr(owner, name):
        print(f"[turntable] ⚠  no {name} in this Blender: skipped")
        return False
    try:
        setattr(owner, name, value)
    except (TypeError, ValueError) as exc:
        print(f"[turntable] ⚠  {name} = {value!r} rejected ({exc}): left as it was")
        return False
    return True


def _check_loaded(job: dict) -> None:
    """Blender loads the factory scene and carries on if the .blend path is bad.

    Without this guard a typo renders 120 frames of the default cube, successfully.
    """
    loaded = Path(bpy.data.filepath).resolve() if bpy.data.filepath else None
    want = Path(job["blend"]).resolve()
    if loaded != want:
        raise SystemExit(f"✗ Blender did not load {want}\n"
                         f"  bpy.data.filepath = {bpy.data.filepath!r}")


def scene_bbox() -> dict:
    """World-space bounds of the mesh objects, and their bounding sphere."""
    from mathutils import Vector
    corners = [obj.matrix_world @ Vector(c)
               for obj in bpy.context.scene.objects if obj.type == "MESH"
               for c in obj.bound_box]
    if not corners:
        raise SystemExit("✗ no mesh in the scene: nothing to turn around")
    lo = Vector((min(c.x for c in corners), min(c.y for c in corners),
                 min(c.z for c in corners)))
    hi = Vector((max(c.x for c in corners), max(c.y for c in corners),
                 max(c.z for c in corners)))
    centre = (lo + hi) * 0.5
    radius = max((c - centre).length for c in corners)
    print(f"[turntable] bbox ({lo.x:.3f}, {lo.y:.3f}, {lo.z:.3f}) .. "
          f"({hi.x:.3f}, {hi.y:.3f}, {hi.z:.3f})  centre "
          f"({centre.x:.3f}, {centre.y:.3f}, {centre.z:.3f})  sphere {radius:.3f}")
    return {"min": list(lo), "max": list(hi), "centre": list(centre),
            "radius": float(radius)}


def resolve_resolution(scene, job: dict) -> dict:
    """Fold --resolution and --percentage into an even effective size.

    H.264 rejects odd dimensions, and Blender computes the render size as
    `resolution * percentage // 100` (integer truncation), so rounding `resolution_x`
    alone is not enough: the fold has to happen on the product.
    """
    r = scene.render
    w = job["resolution"][0] if job["resolution"] else r.resolution_x
    h = job["resolution"][1] if job["resolution"] else r.resolution_y
    pct = job["percentage"] or r.resolution_percentage
    want_w, want_h = (w * pct) // 100, (h * pct) // 100
    eff_w, eff_h = want_w - want_w % 2, want_h - want_h % 2
    if eff_w < 2 or eff_h < 2:
        raise SystemExit(f"✗ {want_w}x{want_h} is too small to encode")
    if (eff_w, eff_h) != (want_w, want_h):
        print(f"[turntable] ⚠  {want_w}x{want_h} -> {eff_w}x{eff_h}: "
              f"H.264 needs even dimensions")
    r.resolution_x, r.resolution_y = eff_w, eff_h
    r.resolution_percentage = 100
    if abs(r.pixel_aspect_x - r.pixel_aspect_y) > 1e-6:
        print(f"[turntable] ⚠  non-square pixels "
              f"({r.pixel_aspect_x}/{r.pixel_aspect_y}): framing will be off")
    return {"requested": [want_w, want_h], "effective": [eff_w, eff_h]}


def configure_camera(scene, job: dict):
    """The scene's own RenderCamera, with --fov applied if given."""
    cam = scene.camera or next(
        (o for o in scene.objects if o.type == "CAMERA"), None)
    if cam is None:
        raise SystemExit("✗ no camera in the scene")
    scene.camera = cam
    if job["fov"]:
        cam.data.sensor_fit = "HORIZONTAL"
        cam.data.angle = math.radians(job["fov"])
    return cam


def resolve_orbit(cam, job: dict, bbox: dict, res: dict) -> dict:
    """Fill in whatever the launcher left null, from the bounding box.

    The framing distance uses the *smaller* of the two fields of view.  At 16:9 the
    vertical FOV is 22.9 degrees against 39.6 horizontal, so fitting on the horizontal
    one would crop the object top and bottom.
    """
    hfov = cam.data.angle
    w, h = res["effective"]
    vfov = 2.0 * math.atan(math.tan(hfov * 0.5) * h / w)
    target = job["target"] or bbox["centre"]
    radius = job["radius"] or FIT_MARGIN * bbox["radius"] / math.sin(min(hfov, vfov) * 0.5)
    orbit = {"target": [float(v) for v in target], "radius": float(radius),
             "elevation": job["elevation"], "start_azimuth": job["start_azimuth"],
             "direction": job["direction"],
             "target_from": job["target_from"], "radius_from": job["radius_from"],
             "hfov_deg": math.degrees(hfov), "vfov_deg": math.degrees(vfov)}
    tx, ty, tz = orbit["target"]
    print(f"[turntable] orbit: --target {tx:.3f} {ty:.3f} {tz:.3f} "
          f"--radius {radius:.3f} --elevation {orbit['elevation']:.1f}  "
          f"(target from {orbit['target_from']}, radius from {orbit['radius_from']})")
    if radius <= bbox["radius"]:
        print(f"[turntable] ⚠  radius {radius:.3f} is inside the bounding sphere "
              f"{bbox['radius']:.3f}: the camera will pass through the object")
    return orbit


def build_rig(scene, cam, orbit: dict):
    """Empty at the pivot, camera parented to it at a fixed local pose.

    No TRACK_TO constraint: the camera's pose relative to the pivot is a constant, so a
    fixed local transform is exact and stays readable from Python before the depsgraph
    is evaluated.  With XYZ euler (pi/2 - e, 0, pi/2) the camera's -Z points exactly at
    the pivot origin and its +Y is the in-plane up.
    """
    from mathutils import Matrix, Vector

    pivot = bpy.data.objects.new("TurntablePivot", None)
    pivot.empty_display_type = "PLAIN_AXES"
    scene.collection.objects.link(pivot)
    pivot.location = Vector(orbit["target"])
    pivot.rotation_mode = "XYZ"
    pivot.rotation_euler = (0.0, 0.0, 0.0)

    for con in list(cam.constraints):
        cam.constraints.remove(con)
    cam.animation_data_clear()
    cam.parent = pivot
    # Not bpy.ops.object.parent_set: that bakes the current world matrix into the parent
    # inverse, and the local pose below would then be applied on top of it.
    cam.matrix_parent_inverse = Matrix.Identity(4)

    elev = math.radians(orbit["elevation"])
    cam.rotation_mode = "XYZ"
    cam.location = (orbit["radius"] * math.cos(elev), 0.0,
                    orbit["radius"] * math.sin(elev))
    cam.rotation_euler = (math.pi / 2.0 - elev, 0.0, math.pi / 2.0)
    return pivot


def _action_fcurves(anim_data):
    """The F-Curves of an object's action.

    Blender 5.0 removed `Action.fcurves`: actions are slotted, and the curves live in
    the channelbag of the slot the datablock is bound to.
    """
    action = getattr(anim_data, "action", None)
    if action is None:
        return []
    if hasattr(action, "fcurves"):                       # Blender <= 4.3
        return list(action.fcurves)
    curves = []
    for layer in action.layers:
        for strip in layer.strips:
            bag = strip.channelbag(anim_data.action_slot)
            if bag is not None:
                curves.extend(bag.fcurves)
    return curves


def keyframe_spin(scene, pivot, job: dict) -> None:
    """One full turn, keyed at frame 1 and frame N+1, rendering 1..N.

    Frame N+1 would repeat frame 1 exactly, so leaving it out is what makes the loop
    seamless.  The interpolation has to be forced: the factory default is BEZIER, which
    with two keys and auto-clamped handles would ease the turntable in and out.
    """
    n = job["frames"]
    sign = 1.0 if job["direction"] == "ccw" else -1.0
    a0 = job["start_azimuth"]
    pivot.animation_data_clear()
    for frame, az in ((1, a0), (n + 1, a0 + sign * 360.0)):
        pivot.rotation_euler = (0.0, 0.0, math.radians(az))
        pivot.keyframe_insert("rotation_euler", index=2, frame=frame)
    curves = _action_fcurves(pivot.animation_data)
    if not curves:
        raise SystemExit("✗ the spin keyframes did not land on an F-Curve")
    for fcurve in curves:
        for key in fcurve.keyframe_points:
            key.interpolation = "LINEAR"
        fcurve.update()
    scene.frame_start = 1
    scene.frame_end = n
    scene.frame_step = 1


def override_world(scene, job: dict) -> dict:
    """Swap the HDRI, or replace the environment with a uniform colour.

    The nodes are looked up by `bl_idname`, not by name: `rerender_run.setup_world`
    created them with `nodes.new()`, so their names are Blender's defaults and would
    pick up a numeric suffix if anything else were ever added to the tree.
    """
    world = scene.world
    # `world.use_nodes` is deprecated in 5.0; the node tree's existence says the same.
    if world is None or world.node_tree is None:
        if job["skybox"] or job["env_color"]:
            raise SystemExit("✗ the scene has no node-based world to override")
        return {"mode": "scene"}
    tree = world.node_tree
    bg = next((n for n in tree.nodes if n.bl_idname == "ShaderNodeBackground"), None)
    env = next((n for n in tree.nodes if n.bl_idname == "ShaderNodeTexEnvironment"), None)
    if bg is None:
        if job["skybox"] or job["env_color"]:
            raise SystemExit("✗ the world has no Background node to override")
        return {"mode": "scene"}

    desc: dict = {"mode": "scene"}
    if job["env_color"]:
        for link in list(tree.links):
            if link.to_socket == bg.inputs["Color"]:
                tree.links.remove(link)
        bg.inputs["Color"].default_value = (*job["env_color"], 1.0)
        if env is not None:
            # Unlinking already keeps the texture off the GPU (both engines compile the
            # graph backwards from the output), but the host-side Image datablock stays
            # loaded: ~100 MB for a 4k float HDRI.  Remove both.
            image = env.image
            tree.nodes.remove(env)
            if image is not None and image.users == 0:
                bpy.data.images.remove(image)
        desc = {"mode": "uniform", "color": list(job["env_color"])}
        print(f"[turntable] world: uniform {tuple(job['env_color'])}")
    elif job["skybox"]:
        if env is None:
            raise SystemExit("✗ the world has no Environment Texture to replace")
        old = env.image
        env.image = bpy.data.images.load(job["skybox"], check_existing=True)
        env.image.colorspace_settings.name = (
            CS_FLOAT if env.image.is_float else CS_BYTE)
        if not env.image.is_float:
            print("[turntable] ⚠  8-bit environment: about 2 stops of range, "
                  "it will not light like an HDRI")
        if old is not None and old is not env.image and old.users == 0:
            bpy.data.images.remove(old)
        desc = {"mode": "hdri", "image": job["skybox"],
                "colorspace": env.image.colorspace_settings.name}
        print(f"[turntable] world: {Path(job['skybox']).name} "
              f"({desc['colorspace']})")
    elif env is not None and env.image is not None:
        desc = {"mode": "hdri", "image": bpy.path.abspath(env.image.filepath),
                "colorspace": env.image.colorspace_settings.name}

    if job["env_strength"] is not None:
        bg.inputs["Strength"].default_value = job["env_strength"]
    desc["strength"] = float(bg.inputs["Strength"].default_value)
    return desc


def configure_colors(scene, job: dict) -> str:
    """A display transform, because the scenes ship 'Raw'.

    'Raw' is a linear passthrough with no sRGB encode: right for the 32-bit EXR these
    blends were built to produce, and crushed and dark once quantised to 8 bit.  The
    enum cannot be introspected - it is OCIO-populated and `enum_items` returns
    ['NONE'] - so the guard has to be a try/except on the assignment.
    """
    view = scene.view_settings
    view.look = "None"          # a look is scoped to its view transform: neutralise first
    want = job["view_transform"]
    try:
        view.view_transform = want
    except TypeError:
        print(f"[turntable] ⚠  unknown view transform {want!r}, using "
              f"{DEFAULT_VIEW_TRANSFORM} (valid: {', '.join(VIEW_TRANSFORMS)})")
        view.view_transform = DEFAULT_VIEW_TRANSFORM
    view.exposure = job["exposure"]
    view.gamma = 1.0
    _set(scene.display_settings, "display_device", "sRGB")
    # dither_intensity is 1.0 in these blends and stays there: it is the noise applied
    # at the float -> 8-bit conversion, and the reason smooth gradients do not band.
    return view.view_transform


def resolve_engine(scene, engine: str) -> str:
    """`engine` is a dynamic enum, so probe it by assignment.

    `enum_items` lists only BLENDER_EEVEE even while Cycles is the active engine, and
    Blender 5.0 renamed BLENDER_EEVEE_NEXT (4.2-4.5) back to BLENDER_EEVEE.
    """
    ids = ("CYCLES",) if engine == "cycles" else ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT")
    for ident in ids:
        try:
            scene.render.engine = ident
            return ident
        except TypeError:
            continue
    raise SystemExit(f"✗ no render engine id for {engine!r} in Blender "
                     f"{bpy.app.version_string}")


def configure_cycles(scene, job: dict) -> str:
    """Cycles: samples, and the one setting that makes a turntable cheap.

    `use_persistent_data` keeps the BVH and the textures between frames.  Measured on
    these scenes at 1080p/128 spp: 2.07 s per frame without it, 0.22 s with - a
    120-frame turn goes from about 4 minutes to under 30 seconds.  It costs a couple of
    GB of resident memory, which is what the geometry and the 8096^2 maps would be
    rebuilt into every frame anyway.
    """
    sys.path.insert(0, job["script_dir"])
    from blender_renderer import enable_gpu_rendering       # noqa: E402

    scene.cycles.samples = job["samples"]
    scene.render.use_persistent_data = True
    _set(scene.cycles, "denoising_use_gpu", True)
    if job["samples"] < FLICKER_SAMPLES:
        print(f"[turntable] ⚠  {job['samples']} samples: per-frame denoising can "
              f"shimmer on glossy areas, far more visible in a video than in a still "
              f"(>= {FLICKER_SAMPLES} to be safe)")
    if job["device"] == "GPU":
        # Mandatory under --factory-startup: compute_device_type starts at NONE.
        enable_gpu_rendering()
    else:
        scene.cycles.device = "CPU"
    return "cycles.samples"


def configure_eevee(scene, job: dict) -> str:
    """EEVEE: the settings without which the comparison shows nothing.

    `use_raytracing` is False in these blends.  Off, EEVEE has no screen-space
    reflections at all and falls back to the world probe, so the metallic and roughness
    maps - the whole point of looking at these scenes - are invisible.  The rest is
    turning off temporal reprojection (it ghosts under a moving camera) and giving the
    tracer enough resolution and roughness range to matter.
    """
    eevee = scene.eevee
    _set(eevee, "taa_render_samples", job["samples"])
    _set(eevee, "use_raytracing", True)
    _set(eevee, "ray_tracing_method", "SCREEN")
    _set(eevee, "use_shadows", True)
    _set(eevee, "shadow_ray_count", 2)
    _set(eevee, "shadow_step_count", 12)
    _set(eevee, "use_fast_gi", True)
    _set(eevee, "fast_gi_resolution", "1")
    _set(eevee, "fast_gi_ray_count", 4)
    _set(eevee, "fast_gi_step_count", 16)
    # The HDRI is baked into a cubemap before it lights anything; at the stock 512 the
    # studio window's highlight is blurred and the night lamps are smeared away.
    _set(eevee, "gi_cubemap_resolution", "1024")
    _set(eevee, "use_taa_reprojection", False)
    # Screen-space rays have no data outside the frame, and pop at the edges while the
    # camera orbits.  Overscan is the standard fix.
    if _set(eevee, "use_overscan", True):
        _set(eevee, "overscan_size", 3.0)
    options = getattr(eevee, "ray_tracing_options", None)
    if options is not None:
        _set(options, "resolution_scale", "1")
        _set(options, "trace_max_roughness", 1.0)   # 0.5 drops the rough half of the asset
        _set(options, "denoise_temporal", False)
    if job["device"] != "GPU":
        print("[turntable] ⚠  --device is Cycles-only: EEVEE always renders on the GPU")
    print("[turntable] EEVEE is a preview: screen-space reflections cannot contain what "
          "is off-screen, so metals will not match Cycles")
    return "eevee.taa_render_samples"


def configure_video(scene, job: dict) -> None:
    """FFMPEG output, in the order Blender 5.0 requires.

    `file_format = "FFMPEG"` raises TypeError unless `media_type` is VIDEO first, and
    the file is named `turntable0001-0120.mp4` unless `filepath` already ends in `.mp4`
    (only `.mp4` is a valid extension for the MPEG4 container - `.mov` would give
    `turntable.mov0001-0120.mp4`).
    """
    render = scene.render
    settings = render.image_settings
    _set(settings, "media_type", "VIDEO")
    settings.file_format = "FFMPEG"
    settings.color_mode = "RGB"       # H.264 carries no alpha
    settings.color_depth = "8"

    ffmpeg = render.ffmpeg
    for name, value in FFMPEG.items():
        _set(ffmpeg, name, value)
    _set(ffmpeg, "gopsize", max(1, job["fps"]))
    _set(ffmpeg, "use_autosplit", False)   # True appends _001 and destroys the name

    render.fps = job["fps"]
    render.fps_base = 1.0
    render.film_transparent = False        # with a uniform world the background *is* it
    render.filepath = job["video"]         # absolute, and ending in .mp4
    render.use_file_extension = True


def configure_preview(scene, job: dict) -> Path:
    """Frame 1 as a PNG: same rig, only the output differs."""
    settings = scene.render.image_settings
    _set(settings, "media_type", "IMAGE")
    settings.file_format = "PNG"
    settings.color_mode = "RGB"
    settings.color_depth = "8"
    _set(settings, "compression", 15)
    out = Path(job["out_dir"]) / PREVIEW_NAME
    scene.render.filepath = str(out)       # extensionless: write_still appends it
    scene.render.use_file_extension = True
    return out.with_suffix(".png")


def _install_progress(n_frames: int) -> None:
    import time
    state = {"i": 0, "t0": time.time()}

    def _post(_scene, _depsgraph=None):
        state["i"] += 1
        done = state["i"]
        elapsed = time.time() - state["t0"]
        eta = elapsed / done * (n_frames - done)
        print(f"[turntable] {done}/{n_frames}  ({elapsed / 60.0:.1f} min, "
              f"ETA {eta / 60.0:.1f} min)", flush=True)

    bpy.app.handlers.render_post.append(_post)


def blender_main(job_path: str) -> None:
    import time
    from datetime import datetime

    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    _check_loaded(job)
    scene = bpy.context.scene
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    bbox = scene_bbox()
    res = resolve_resolution(scene, job)
    cam = configure_camera(scene, job)
    orbit = resolve_orbit(cam, job, bbox, res)
    pivot = build_rig(scene, cam, orbit)
    keyframe_spin(scene, pivot, job)
    world = override_world(scene, job)
    view_transform = configure_colors(scene, job)
    engine_id = resolve_engine(scene, job["engine"])
    samples_property = (configure_cycles(scene, job) if job["engine"] == "cycles"
                        else configure_eevee(scene, job))

    scene.frame_set(1)
    evaluated = cam.evaluated_get(bpy.context.evaluated_depsgraph_get())
    cam_at_start = list(evaluated.matrix_world.translation)

    t0 = time.time()
    if job["preview"]:
        produced = configure_preview(scene, job)
        print(f"[turntable] preview, {res['effective'][0]}x{res['effective'][1]}, "
              f"{engine_id}", flush=True)
        bpy.ops.render.render(write_still=True)
    else:
        configure_video(scene, job)
        produced = Path(job["video"])
        print(f"[turntable] {job['frames']} frames, "
              f"{res['effective'][0]}x{res['effective'][1]}, {engine_id}, "
              f"{job['samples']} samples", flush=True)
        _install_progress(job["frames"])
        bpy.ops.render.render(animation=True)

    meta = dict(job)
    meta.pop("script_dir", None)
    meta.update({
        "blender": bpy.app.version_string,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "wall_min": round((time.time() - t0) / 60.0, 2),
        "output": str(produced),
        "orbit": orbit,
        "degrees_per_frame": 360.0 / job["frames"],
        "duration_s": job["frames"] / job["fps"],
        "bbox": bbox,
        "resolution_resolved": res,
        "engine_id": engine_id,
        "samples_property": samples_property,
        "world": world,
        "view_transform_applied": view_transform,
        "camera_at_frame_1": cam_at_start,
    })
    (out_dir / META_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[turntable] meta -> {out_dir / META_NAME}")


if __name__ == "__main__":
    if IN_BLENDER:
        argv = sys.argv
        argv = argv[argv.index("--") + 1:] if "--" in argv else []
        if len(argv) != 1:
            raise SystemExit("internal use: blender --background --python "
                             "turntable_video.py -- <job.json>")
        blender_main(argv[0])
    else:
        raise SystemExit(launcher())
