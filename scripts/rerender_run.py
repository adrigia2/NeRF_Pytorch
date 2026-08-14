#!/usr/bin/env python
"""rerender_run.py -- re-render a finished run in Blender with its reconstructed textures.

    python rerender_run.py <run_dir> --skybox <hdr.exr> [--materials pbr|lambert|gt] ...

`<run_dir>` is the scene folder of a run (the one holding `run_manifest.json`), for
example:

    D:/tesi_output/test_sword_shield_after_fix_irradiance/exp_l1_d02/TableAndOtherInteriorWithSpecular

It produces the same images as the training dataset - same names, same resolution,
same format - but with the material reconstructed by the pipeline in place of the
original one, so that

    compare_exr.py --original <run_dir>/images --computed <out>/images

pairs them up on its own and compares them pixel by pixel.

Outside the pipeline: like `roi_rerun.py` and `rerun_irradiance.py`, it touches neither
OptiX nor the NeRF checkpoint, and reads only what the run already wrote to disk.

-------------------------------------------------------------------------------
How it runs
-------------------------------------------------------------------------------
A single file, in two modes depending on where it is executed (`try: import bpy`):

  * with the normal python -> it acts as a launcher: resolves the paths by reading the
    run, prepares the textures, writes a JSON job and invokes `blender --background
    --python <itself>`;
  * inside Blender          -> it reads the job, builds the scene and renders.

The split is not cosmetic: the maps have to be read and rewritten as EXR, and Blender's
Python has numpy but NOT OpenEXR, whereas the `tesi-nerf` env has both plus scipy.

-------------------------------------------------------------------------------
The reference material
-------------------------------------------------------------------------------
The graph replicated is that of `BakedMaterial` in the source `Baked.blend` files, which
are the files that produced the training images (their `render.filepath` points to the
last frame of the dataset):

    base color -> Base Color
    metallic   -> Metallic
    roughness  -> Roughness
    normal     -> Normal Map (space=OBJECT, strength 1) -> Normal

Colour space: the base colour is tagged `Linear Rec.709`, the other three `Non-Color`.  In
the original blend the base colour was `Non-Color` too; the difference does not change a
single pixel (measured: maximum gap 0.0, because the scene reference space *is* Linear
Rec.709 and both transforms are therefore the identity), but `Non-Color` means "this is
not a colour, never convert it" and is the right tag only for data.  On an albedo, in a
configuration with a different reference (ACEScg), it would pass Rec.709 numbers off as
ACEScg, going wrong silently.

-------------------------------------------------------------------------------
Three things that would break everything without a symptom
-------------------------------------------------------------------------------
1. `metallic.exr` and `roughness.exr` have a single channel, called `Z`, which Blender does
   not read as an image: the `_rgb` copies the pipeline writes next to them must be used.
   If they are missing this script stops and points at `exr_to_blender_rgb.py` instead of
   loading a black image.
2. The skybox has to be passed by hand and getting it wrong is easy: night and studio
   share model, cameras and layout, and a render with the wrong skybox *looks* right.
   Hence the guard that compares against the environment of the source `Baked.blend`.
   The correct pairings:

     TableAndOtherInteriorWithSpecular / ...NoSpecular
         Scenes/TableAndOtherInterior/Blender/assets/hdri/wooden_studio_13_4k.exr
     TableAndOtherInteriorWithSpecularNight
         Scenes/TableAndOtherInterior/BlenderBakedSmoothNight/cobblestone_street_night_4k.exr
     SwordShieldStudio
         Scenes/SwordShield Thesis/Blender/assets/hdrs/wooden_studio_13_4k.exr
     SwordShieldNight
         Scenes/SwordShield Thesis/Blender/assets/hdrs/cobblestone_street_night_4k.exr
3. The OBJ is already in the Blender frame: `wm.obj_import(forward_axis='Y', up_axis='Z')`
   gives an identity `matrix_world`.  No axis correction on the camera poses.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:                     # inside Blender
    import bpy
    IN_BLENDER = True
except ImportError:      # launcher, normal python
    IN_BLENDER = False

# On Windows stdout arrives as cp1252 and the characters used in the messages (⚠, ✗) blow
# it up halfway through a run.  Setting PYTHONIOENCODING inside the script would be too
# late: the stream already exists when the module is imported.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:    # noqa: BLE001  -- stream not reconfigurable: never mind
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Render constants, read from the five source Baked.blend files.
#
# They are identical in all five, which is why they can be fixed here instead of being
# re-read scene by scene.  The only one that varies is `view_transform` (Raw in some, AgX
# in others) and it does not matter: Blender writes linear to float formats, as shown by
# the GT images of the AgX blends holding values up to 1.54.  It is forced to Raw so that
# the output is unambiguous.
# ──────────────────────────────────────────────────────────────────────────────

GT_SAMPLES = 4096
GT_ADAPTIVE_THRESHOLD = 0.01
GT_BOUNCES = dict(max_bounces=12, diffuse_bounces=4, glossy_bounces=4,
                  transmission_bounces=12, volume_bounces=0, transparent_max_bounces=8)
GT_CLAMP_DIRECT = 0.0
GT_CLAMP_INDIRECT = 10.0
GT_CLIP_START = 0.1
GT_CLIP_END = 1000.0
GT_BG_STRENGTH = 1.0

DEFAULT_BLENDER = r"C:/Program Files/Blender Foundation/Blender 5.0/blender.exe"
DEFAULT_DILATE = 8          # texels at 4096^2, equivalent to the margin 16 of the GT bake at 8192^2

CS_COLOR = "Linear Rec.709"
CS_DATA = "Non-Color"


# ══════════════════════════════════════════════════════════════════════════════
# LAUNCHER  (normal python: OpenEXR + scipy available)
# ══════════════════════════════════════════════════════════════════════════════

def _load_exr_hw3(path: Path):
    """(H, W, 3) float32.  Same loader as regen_heatmaps: reads R/G/B by name and
    replicates the single channel when the EXR has only one."""
    import numpy as np
    import OpenEXR
    import Imath

    exr = OpenEXR.InputFile(str(path))
    dw = exr.header()["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    chs = exr.header()["channels"]
    if "R" in chs and "G" in chs and "B" in chs:
        ch = [np.frombuffer(exr.channel(c, pt), np.float32).reshape(h, w) for c in "RGB"]
    else:
        key = next(iter(chs))
        one = np.frombuffer(exr.channel(key, pt), np.float32).reshape(h, w)
        ch = [one, one, one]
    return np.stack(ch, axis=-1)


def _write_exr_rgb(path: Path, img) -> None:
    """Write an (H, W, 3) float32 array to three float R/G/B channels, ZIP compression."""
    import numpy as np
    import OpenEXR
    import Imath

    h, w = img.shape[:2]
    header = OpenEXR.Header(w, h)
    ft = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header["channels"] = {c: ft for c in "RGB"}
    header["compression"] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
    out = OpenEXR.OutputFile(str(path), header)
    a = np.ascontiguousarray(img, dtype=np.float32)
    out.writePixels({c: a[..., i].tobytes() for i, c in enumerate("RGB")})
    out.close()


def _nearest_valid_indices(mask):
    """For every invalid texel, the index of the nearest valid texel plus the distance.

    Computed once and reused by every map of the same run: the Euclidean distance
    transform on 4096^2 is not free and the mask is the same for all of them.
    """
    from scipy.ndimage import distance_transform_edt
    dist, idx = distance_transform_edt(~mask, return_indices=True)
    return dist, idx


def _dilate(img, mask, dist, idx, radius: float):
    """Fill the invalid texels within `radius` outwards with the value of the nearest
    valid texel.

    Needed because the reconstructed maps are 0 outside the IUM mask, UV island borders
    included, and with bilinear filtering Blender picks up that zero just inside the
    edge.  The resulting artefact is not the dark seam one would expect: on the roughness,
    zero means MIRROR, so the border becomes a BRIGHT specular line (measured: up to 0.45
    of difference on a line a couple of pixels wide, 0.24 % of the image's pixels).  The
    GT textures do not have it because Blender's baker applies a margin; this is the
    equivalent.
    """
    out = img.copy()
    fill = (~mask) & (dist <= radius)
    iy, ix = idx[0][fill], idx[1][fill]
    out[fill] = img[iy, ix]
    return out, int(fill.sum())


def _resolve_maps(run_dir: Path, tf_ext: dict, manifest: dict,
                  materials: str, source: str) -> dict:
    """The absolute paths of the maps, and which of them are reconstructed (i.e. to dilate).

    In `lambert` mode there is only the base colour: metallic and roughness become scalars,
    fixed by `build_material`.

    No path is built by hand: the reconstructed ones live in the `ium` block of
    `transforms_extended.json`, the GT ones sit next to the external normal the manifest
    records.
    """
    ium = tf_ext.get("ium", {})
    scene = manifest.get("scene", {})
    ext_normal = scene.get("external_normal_path")
    if not ext_normal:
        raise SystemExit("✗ the manifest does not record `scene.external_normal_path`: "
                         "without the normal I cannot replicate the reference material.")
    normal = Path(ext_normal)

    if materials == "gt":
        # The original maps live in the bake folder, i.e. the normal's.
        gt_dir = normal.parent
        maps = {
            "base_color": gt_dir / "BakedMaterial_base_color.exr",
            "metallic": gt_dir / "BakedMaterial_metallic.exr",
            "roughness": gt_dir / "BakedMaterial_roughness.exr",
        }
        reconstructed = []
    else:
        def need(key: str) -> Path:
            rel = ium.get(f"{key}_{source}")
            if not rel:
                raise SystemExit(
                    f"✗ `transforms_extended.json` has no `{key}_{source}`. "
                    f"Sources available: "
                    f"{sorted({k.rsplit('_', 1)[-1] for k in ium if k.startswith('albedo')})}")
            return run_dir / rel

        if materials == "pbr":
            base = need("albedo_pbr_path")
            # The single-channel (`Z`) files are not readable by Blender: the _rgb copies
            # that `exr_to_blender_rgb` writes next to them are needed.
            met = need("metallic_path").with_name("metallic_rgb.exr")
            rou = need("roughness_path").with_name("roughness_rgb.exr")
            for p, orig in ((met, "metallic"), (rou, "roughness")):
                if not p.exists():
                    raise SystemExit(
                        f"✗ {p} is missing.\n"
                        f"  {orig}.exr has a single channel, called `Z`, which Blender "
                        f"does not read as an image.\n"
                        f"  Generate the RGB copy with:  python exr_to_blender_rgb.py "
                        f"{p.parent}")
            maps = {"base_color": base, "metallic": met, "roughness": rou}
        else:  # lambert
            maps = {"base_color": need("albedo_path")}
        reconstructed = [k for k in maps]

    maps["normal"] = normal
    for k, p in maps.items():
        if not p.exists():
            raise SystemExit(f"✗ map `{k}` not found: {p}")
    return {"maps": {k: str(v) for k, v in maps.items()},
            "reconstructed": reconstructed}


def _report_material_stats(run_dir: Path, maps: dict) -> None:
    """Statistics of the three reconstructed maps, plus a warning on the "black metal" texels.

    Not cosmetic.  `pbr_solver` writes albedo = 0 where the texel comes out fully specular
    (x < X_EPS), by the metal convention: the diffuse albedo is undefined there.
    Blender's Principled, however, reads Base Color as the metal's REFLECTION colour when
    Metallic = 1, so those same texels become mirrors that reflect nothing, i.e. black
    patches.  It is a disagreement between two conventions, not a rendering error: without
    this count it would be mistaken for a bug in the rerender pipeline.
    """
    import numpy as np

    mask_path = run_dir / "ium" / "ium_masks.exr"
    if "metallic" not in maps or not mask_path.exists():
        return
    m = _load_exr_hw3(mask_path)[..., 0] > 0.5
    alb = _load_exr_hw3(Path(maps["base_color"]))[m].max(-1)
    met = _load_exr_hw3(Path(maps["metallic"]))[m][..., 0]
    rou = _load_exr_hw3(Path(maps["roughness"]))[m][..., 0]
    print(f"  {m.sum():,} valid texels")
    print(f"  albedo    p50={np.median(alb):.4f}   black (<1e-3) {100 * (alb < 1e-3).mean():.2f}%")
    print(f"  metallic  p50={np.median(met):.4f}   >0.5 {100 * (met > 0.5).mean():.2f}%")
    print(f"  roughness p50={np.median(rou):.4f}   ==0 (mirror) "
          f"{100 * (rou < 1e-6).mean():.2f}%   ==1 {100 * (rou > 0.999).mean():.2f}%")
    bad = float(((met > 0.5) & (alb < 1e-3)).mean())
    if bad > 1e-4:
        print(f"  ⚠  {100 * bad:.2f}% of the texels have metallic>0.5 with albedo~0: in the "
              f"Principled they are mirrors reflecting nothing,\n"
              f"     so they will look BLACK.  It is pbr_solver's metal convention "
              f"(albedo=0 where x<X_EPS), not a rendering error.")


def _prepare_textures(run_dir: Path, out_dir: Path, maps: dict,
                      reconstructed: list, radius: float) -> dict:
    """Dilate the reconstructed maps and rewrite them into `<out>/textures/`.

    The normal and the GT maps are untouched: they are the original bake files, which
    already have the margin.
    """
    if not reconstructed:
        print("  no reconstructed map to prepare (GT materials, the original bake "
              "files are used)")
        return maps
    if radius <= 0:
        print("  dilation disabled (--dilate 0): expect bright specular lines on the "
              "UV island borders")
        return maps

    mask_path = run_dir / "ium" / "ium_masks.exr"
    if not mask_path.exists():
        print(f"  ⚠  {mask_path} not found: skipping the dilation")
        return maps

    mask = _load_exr_hw3(mask_path)[..., 0] > 0.5
    print(f"  IUM mask {mask.shape[1]}x{mask.shape[0]}, "
          f"{100.0 * mask.mean():.1f}% of the texels valid")
    dist, idx = _nearest_valid_indices(mask)

    tex_dir = out_dir / "textures"
    tex_dir.mkdir(parents=True, exist_ok=True)
    resolved = dict(maps)
    for key in reconstructed:
        src = Path(maps[key])
        img = _load_exr_hw3(src)
        if img.shape[:2] != mask.shape:
            print(f"  ⚠  {src.name} is {img.shape[1]}x{img.shape[0]} but the mask is "
                  f"{mask.shape[1]}x{mask.shape[0]}: skipping the dilation of this map")
            continue
        out, n = _dilate(img, mask, dist, idx, radius)
        dst = tex_dir / src.name
        _write_exr_rgb(dst, out)
        resolved[key] = str(dst)
        print(f"  + textures/{src.name}  ({n} texels filled within {radius:g})")
    del dist, idx, mask
    return resolved


def launcher() -> int:
    import argparse
    import subprocess
    import time

    ap = argparse.ArgumentParser(
        prog="rerender_run",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="scene folder of the run (holds run_manifest.json)")
    ap.add_argument("--skybox", required=True,
                    help="the original HDR to use as environment (NOT the reconstructed one)")
    ap.add_argument("--materials", default="pbr", choices=["pbr", "lambert", "gt"],
                    help="pbr: albedo_pbr+metallic+roughness | lambert: diffuse albedo | "
                         "gt: original textures, a check on the rig (default: pbr)")
    ap.add_argument("--source", default="gt", help="source of the maps: gt | nerf (default: gt)")
    ap.add_argument("--out", default=None, help="output folder (default: <run>/rerender/<mode>)")
    ap.add_argument("--samples", type=int, default=GT_SAMPLES,
                    help=f"Cycles samples (default: {GT_SAMPLES}, as in the GT render; "
                         f"lowering it is how to make a quick test)")
    ap.add_argument("--frames", nargs="+", default=None,
                    help="stems of the only frames to render, e.g. render_Camera_Shell10_0")
    ap.add_argument("--limit", type=int, default=0, help="render only the first N frames")
    ap.add_argument("--force", action="store_true", help="redo the frames already on disk too")
    ap.add_argument("--dilate", type=float, default=DEFAULT_DILATE,
                    help=f"radius, in texels, of the outward fill of the reconstructed "
                         f"maps (default: {DEFAULT_DILATE}, 0 disables it)")
    ap.add_argument("--no-normal", action="store_true", dest="no_normal",
                    help="do not connect the normal map")
    ap.add_argument("--device", default="GPU", choices=["GPU", "CPU"])
    ap.add_argument("--blender", default=DEFAULT_BLENDER)
    ap.add_argument("--save-blend", action="store_true", dest="save_blend",
                    help="also save scene.blend (unpacked: the textures stay linked)")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="resolve everything and print the job without launching Blender")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    manifest_path = run_dir / "run_manifest.json"
    tf_path = run_dir / "transforms_extended.json"
    for p in (manifest_path, tf_path):
        if not p.exists():
            raise SystemExit(f"✗ not found: {p}\n  `run_dir` has to be the scene folder "
                             f"of a finished run.")
    skybox = Path(args.skybox).resolve()
    if not skybox.exists():
        raise SystemExit(f"✗ skybox not found: {skybox}")
    blender = Path(args.blender)
    if not blender.exists():
        raise SystemExit(f"✗ Blender not found: {blender}  (use --blender)")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tf_ext = json.loads(tf_path.read_text(encoding="utf-8"))
    model = Path(manifest["scene"]["model_path"])
    if not model.exists():
        raise SystemExit(f"✗ model not found: {model}")

    tag = "gt_control" if args.materials == "gt" else f"{args.materials}_{args.source}"
    out_dir = Path(args.out).resolve() if args.out else run_dir / "rerender" / tag
    out_images = out_dir / "images"

    print(f"run       {run_dir}")
    print(f"model     {model}")
    print(f"skybox    {skybox}")
    print(f"materials {args.materials}" + ("" if args.materials == "gt" else f" (source {args.source})"))
    print(f"output    {out_dir}")

    resolved = _resolve_maps(run_dir, tf_ext, manifest, args.materials, args.source)
    print("maps:")
    for k, v in resolved["maps"].items():
        print(f"  {k:11s} {v}")

    # Which frames are really missing.  The decision lives here and not inside Blender,
    # which merely renders the list it receives.
    stems = [Path(f["file_path"]).stem for f in tf_ext["frames"]]
    if args.frames:
        unknown = sorted(set(args.frames) - set(stems))
        if unknown:
            raise SystemExit(f"✗ stems not present in transforms_extended.json: {unknown}")
        stems = [s for s in stems if s in set(args.frames)]
    if args.limit > 0:
        stems = stems[:args.limit]
    todo = stems if args.force else [s for s in stems if not (out_images / f"{s}.exr").exists()]
    skipped = len(stems) - len(todo)
    print(f"frames    {len(todo)} to render"
          + (f", {skipped} already on disk (--force to redo them)" if skipped else ""))
    if not todo:
        print("nothing to do.")
        return 0

    out_images.mkdir(parents=True, exist_ok=True)
    if args.materials == "pbr":
        print("reconstructed material:")
        _report_material_stats(run_dir, resolved["maps"])
    print("textures:")
    maps = _prepare_textures(run_dir, out_dir, resolved["maps"],
                             resolved["reconstructed"], args.dilate)

    job = {
        "run_dir": str(run_dir),
        "transforms": str(tf_path),
        "model": str(model),
        "skybox": str(skybox),
        "materials": args.materials,
        "source": args.source,
        "maps": maps,
        "maps_original": resolved["maps"],
        "dilate": args.dilate,
        "samples": args.samples,
        "device": args.device,
        "no_normal": args.no_normal,
        "save_blend": args.save_blend,
        "out_dir": str(out_dir),
        "frames": todo,
        "script_dir": str(Path(__file__).resolve().parent),
    }
    job_path = out_dir / "rerender_job.json"
    job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")

    if args.dry_run:
        print(f"\n--dry-run: job written to {job_path}, Blender not launched.")
        return 0

    # --factory-startup: no user add-ons (blenderkit prints a lot) and no local
    # preferences, so the render depends on this script alone.  The GPU devices are
    # re-enabled explicitly by enable_gpu_rendering().
    cmd = [str(blender), "--background", "--factory-startup",
           "--python", str(Path(__file__).resolve()), "--", str(job_path)]
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd)
    dt = time.time() - t0
    print(f"\nBlender exited with {rc} after {dt / 60.0:.1f} min")
    if rc != 0:
        return rc

    done = sorted(p.name for p in out_images.glob("*.exr"))
    print(f"{len(done)} images in {out_images}")
    print(f"\nComparison:\n  python compare_exr.py --original {run_dir / 'images'} "
          f"--computed {out_images} --output {out_dir / 'compare'}")
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# INSIDE BLENDER
# ══════════════════════════════════════════════════════════════════════════════

def _principled(nt):
    """The tree's Principled node, creating and connecting it when missing."""
    bsdf = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)
    if bsdf is not None:
        return bsdf
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    out = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial"), None)
    if out is None:
        out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return bsdf


def build_material(job: dict):
    """Replicate the `BakedMaterial` graph with the maps named in the job.

    Normal Map in OBJECT space: that is how the normal is baked, and it is the same one
    the pipeline consumed as input.  Keeping it identical to the GT makes the comparison
    isolate the material error, instead of mixing it with the detail lost by a normal the
    pipeline does not reconstruct.
    """
    maps = job["maps"]
    mat = bpy.data.materials.new("RerenderMaterial")
    if mat.node_tree is None:
        mat.use_nodes = True
    nt = mat.node_tree
    bsdf = _principled(nt)

    def tex(path: str, colorspace: str, y: int):
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = bpy.data.images.load(path, check_existing=True)
        n.image.colorspace_settings.name = colorspace
        n.location = (-600, y)
        return n

    # Base colour: tagged as colour, not as data.  See the docstring at the top.
    nt.links.new(tex(maps["base_color"], CS_COLOR, 400).outputs["Color"],
                 bsdf.inputs["Base Color"])

    if "metallic" in maps:
        nt.links.new(tex(maps["metallic"], CS_DATA, 100).outputs["Color"],
                     bsdf.inputs["Metallic"])
        # The reconstructed roughness goes in AS IT IS.  That is a choice, not a
        # derivation: the file holds cone_aperture/180 of the winning cone, while Blender
        # reads this input as a GGX roughness.  The extremes coincide (0 = mirror,
        # 1 = maximally rough), the middle does not.  Any aperture -> alpha calibration
        # belongs here, between the texture and the input.
        nt.links.new(tex(maps["roughness"], CS_DATA, -200).outputs["Color"],
                     bsdf.inputs["Roughness"])
    else:
        # lambert: no specular component to reconstruct
        bsdf.inputs["Metallic"].default_value = 0.0
        bsdf.inputs["Roughness"].default_value = 1.0

    if not job["no_normal"]:
        t = tex(maps["normal"], CS_DATA, -500)
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.space = "OBJECT"
        nm.inputs["Strength"].default_value = 1.0
        nm.location = (-300, -500)
        nt.links.new(t.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def setup_world(skybox: str) -> None:
    """World with the environment texture alone, strength 1, no mapping node:
    exactly the World of the source blends."""
    world = bpy.data.worlds.new("RerenderWorld")
    bpy.context.scene.world = world
    if world.node_tree is None:
        world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    bg = nt.nodes.new("ShaderNodeBackground")
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    out = nt.nodes.new("ShaderNodeOutputWorld")
    env.image = bpy.data.images.load(skybox, check_existing=True)
    env.image.colorspace_settings.name = CS_DATA
    bg.inputs["Strength"].default_value = GT_BG_STRENGTH
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])


def check_skybox(job: dict) -> None:
    """Warn when the skybox passed is not the one in the source `Baked.blend`.

    Getting it wrong is the easiest way to invalidate the comparison without noticing:
    night and studio share model, cameras and layout, and the render *looks* right.
    Only the basename is compared, because in the blend the path is relative to its folder.
    It only reports, it does not decide for the user.
    """
    from pathlib import Path as P
    cands = [P(job["maps_original"]["normal"]).parent / "Baked.blend",
             P(job["model"]).parent / "Baked.blend"]
    blend = next((c for c in cands if c.exists()), None)
    if blend is None:
        print("[skybox] no source Baked.blend found: guard skipped")
        return
    try:
        # After the `with` block, dst.worlds holds the datablocks just loaded (inside they
        # are still names).  They have to be read from there and not looked up by name in
        # bpy.data: a World already called "World" would be renamed to "World.001".
        with bpy.data.libraries.load(str(blend), link=False) as (src, dst):
            dst.worlds = list(src.worlds)[:1]
        loaded = [w for w in dst.worlds if w is not None]
    except Exception as e:                        # noqa: BLE001
        print(f"[skybox] could not read {blend}: {e}")
        return
    found = None
    for world in loaded:
        if world.node_tree is not None:
            for n in world.node_tree.nodes:
                if n.bl_idname == "ShaderNodeTexEnvironment" and n.image:
                    found = P(n.image.filepath.replace("\\", "/")).name
        bpy.data.worlds.remove(world)
    want = P(job["skybox"]).name
    if found is None:
        print(f"[skybox] {blend.name} has no environment texture: guard ineffective")
    elif found.lower() == want.lower():
        print(f"[skybox] ok, matches the one in {blend.name}: {found}")
    else:
        print(f"[skybox] ⚠  WARNING: {blend.name} uses `{found}`, "
              f"you passed `{want}`.\n"
              f"[skybox] ⚠  If that is not intended, the comparison with the training "
              f"images is meaningless.")


def configure_render(scene, intr, job: dict) -> None:
    """The parameters of the source Baked.blend files, identical in all five scenes."""
    r = scene.render
    r.resolution_x = intr.w
    r.resolution_y = intr.h
    r.resolution_percentage = 100
    r.engine = "CYCLES"
    r.film_transparent = False
    r.use_file_extension = True

    c = scene.cycles
    c.samples = job["samples"]
    c.use_adaptive_sampling = True
    c.adaptive_threshold = GT_ADAPTIVE_THRESHOLD
    c.use_denoising = True
    for k, v in GT_BOUNCES.items():
        setattr(c, k, v)
    c.sample_clamp_direct = GT_CLAMP_DIRECT
    c.sample_clamp_indirect = GT_CLAMP_INDIRECT

    fmt = r.image_settings
    fmt.file_format = "OPEN_EXR"
    fmt.color_mode = "RGB"
    fmt.color_depth = "32"
    fmt.exr_codec = "NONE"

    # Irrelevant on float formats (Blender writes linear), but fixing it makes the output
    # unambiguous: the source blends do not agree with each other on this field.
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0


def blender_main(job_path: str) -> None:
    import time
    from datetime import datetime

    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    sys.path.insert(0, job["script_dir"])
    from blender_renderer import (create_camera, enable_gpu_rendering,  # noqa: E402
                                  import_model, load_transforms, set_camera_pose)

    out_dir = Path(job["out_dir"])
    out_images = out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    tf = load_transforms(job["transforms"])
    intr = tf.intrinsics
    wanted = set(job["frames"])
    frames = [f for f in tf.frames if Path(f.file_path).stem in wanted]
    print(f"[rerender] {len(frames)} frames, {intr.w}x{intr.h}, "
          f"{job['samples']} samples, materials {job['materials']}")

    check_skybox(job)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    import_model(job["model"])

    # The import prints WARNINGs about the .mtl textures: they are absolute paths Blender
    # re-anchors to the OBJ's folder and therefore does not find.  Harmless, the imported
    # material is thrown away below anyway.
    mat = build_material(job)
    n_mesh = 0
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            obj.data.materials.clear()
            obj.data.materials.append(mat)
            n_mesh += 1
    print(f"[rerender] material applied to {n_mesh} meshes")

    setup_world(job["skybox"])

    scene = bpy.context.scene
    cam_obj = create_camera(intr)
    # create_camera leaves clip_start at 0.001; the source blends use 0.1.
    cam_obj.data.clip_start = GT_CLIP_START
    cam_obj.data.clip_end = GT_CLIP_END
    scene.camera = cam_obj
    configure_render(scene, intr, job)
    if job["device"] == "GPU":
        enable_gpu_rendering()
    else:
        scene.cycles.device = "CPU"

    t0 = time.time()
    for i, frame in enumerate(frames, 1):
        stem = Path(frame.file_path).stem
        set_camera_pose(cam_obj, frame.transform_matrix, apply_axis_correction=False)
        bpy.context.view_layer.update()
        scene.render.filepath = str(out_images / stem)
        bpy.ops.render.render(write_still=True)
        el = time.time() - t0
        eta = el / i * (len(frames) - i)
        print(f"[rerender] {i}/{len(frames)} {stem}  "
              f"({el / 60.0:.1f} min, ETA {eta / 60.0:.1f} min)", flush=True)

    meta = dict(job)
    meta.pop("script_dir", None)
    meta.update({
        "blender": bpy.app.version_string,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "n_frames": len(frames),
        "wall_min": round((time.time() - t0) / 60.0, 2),
        "render": {
            "engine": "CYCLES", "samples": job["samples"],
            "adaptive_threshold": GT_ADAPTIVE_THRESHOLD, "denoising": True,
            "bounces": GT_BOUNCES,
            "clamp_direct": GT_CLAMP_DIRECT, "clamp_indirect": GT_CLAMP_INDIRECT,
            "resolution": [intr.w, intr.h], "format": "OPEN_EXR RGB 32 NONE",
            "view_transform": "Raw",
            "colorspace": {"base_color": CS_COLOR, "metallic": CS_DATA,
                           "roughness": CS_DATA, "normal": CS_DATA},
            "normal_map_space": "OBJECT",
        },
    })
    (out_dir / "rerender_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[rerender] meta -> {out_dir / 'rerender_meta.json'}")

    if job["save_blend"]:
        blend = str(out_dir / "scene.blend")
        bpy.ops.wm.save_as_mainfile(filepath=blend, check_existing=False)
        print(f"[rerender] scene.blend -> {blend}")


if __name__ == "__main__":
    if IN_BLENDER:
        argv = sys.argv
        argv = argv[argv.index("--") + 1:] if "--" in argv else []
        if len(argv) != 1:
            raise SystemExit("internal use: blender --background --python rerender_run.py "
                             "-- <job.json>")
        blender_main(argv[0])
    else:
        raise SystemExit(launcher())
