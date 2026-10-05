"""
bake_normal_variants.py
-----------------------
Bake a scene's normal map at several resolutions, onto exactly the BakeUV layout that
`bake_unified_material` produced for that scene, so the pipeline never has to resample a
normal map (it refuses one whose size differs from ium_texture_size).

Run headless, from the source .blend (the one bake_unified_material was run on):

    blender -b <source.blend> --factory-startup --python-exit-code 1 \
            --python bake_normal_variants.py -- \
            --out <dir> --sizes 4096 2048 1024 512 [--samples 256]
            [--scene Scene] [--reference-scene Baked] [--force]

It reuses bake_unified_material's own steps, in the order of `run()`, up to the packed
BakeUV (duplicate, apply transforms, copy the UVs, join, pack), then calls `_bake_all`
once per size with only the normal channel, in OBJECT space. Every size is baked on that
one BakeUV.

UV guard: before baking anything, the new BakeUV is compared loop by loop with the BakeUV
of the mesh in --reference-scene (the "Baked" scene bake_unified_material left in the
file, the mesh the pipeline's OBJ was exported from). Polygon and loop order are the same
by construction; any difference above 1e-6 means the maps would not line up with the OBJ,
and the script stops without writing.

Why a bake per size instead of resizing one map: Cycles evaluates the shading normal
from the materials' own textures, inside each triangle (no bleeding across UV islands);
a resize of an already-baked map filters baked values and mixes in the background at the
island borders.

What a texel holds (measured on both scenes at 1024, 2026-10-01): the mean of the
shading normal over a footprint NARROWER than the texel, equivalent to a Gaussian of
σ ≈ 0.15 texel (a full-texel box has σ ≈ 0.29). Sample 0 sits at the texel centre and
the others are jittered around it; 1 and 256 samples differ (p99 6.7°), and the film's
pixel filter has no effect (BOX 1.0 and Blackman-Harris 1.5 bake bit-identical maps). So
at low resolutions the bake keeps sub-texel relief that a texel-area average would
smooth out: it is close to point sampling, not to a box over the texel.

Each size lands in <out>/normal_<N>/BakedMaterial_normal.exr with a bake_info.json next
to it. The .blend is never saved and an existing EXR is never overwritten without --force.
"""

import argparse
import dataclasses
import json
import os
import sys
import time

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bake_unified_material as bum  # noqa: E402

UV_TOL = 1e-6


def _parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(prog="bake_normal_variants.py")
    ap.add_argument("--out", required=True, help="root folder: <out>/normal_<N>/")
    ap.add_argument("--sizes", type=int, nargs="+", required=True)
    ap.add_argument("--samples", type=int, default=256,
                    help="Cycles samples per texel, jittered around its centre (default 256)")
    ap.add_argument("--scene", default="Scene", help="the source scene (original materials)")
    ap.add_argument("--reference-scene", default="Baked",
                    help="scene holding the mesh with the reference BakeUV; '' skips the guard")
    ap.add_argument("--force", action="store_true", help="overwrite existing EXRs")
    return ap.parse_args(argv)


def _uv_array(mesh: bpy.types.Mesh) -> np.ndarray:
    layer = mesh.uv_layers[bum.UV_LAYER_NAME]
    uv = np.empty(len(layer.data) * 2, dtype=np.float64)
    layer.data.foreach_get("uv", uv)
    return uv.reshape(-1, 2)


def _uv_guard(merged: bpy.types.Object, ref_scene_name: str) -> dict:
    """Compare the new BakeUV with the reference one, loop by loop."""
    if not ref_scene_name:
        print("bake_normal_variants: WARNING - no reference scene, UV guard skipped: "
              "nothing proves these maps line up with the pipeline's OBJ.")
        return {"reference_scene": None}
    ref_scene = bpy.data.scenes.get(ref_scene_name)
    if ref_scene is None:
        raise RuntimeError(f"reference scene {ref_scene_name!r} not found "
                           f"(scenes: {[s.name for s in bpy.data.scenes]}); "
                           f"pass --reference-scene '' to skip the guard")
    ref_meshes = [o for o in ref_scene.objects if o.type == "MESH"]
    if len(ref_meshes) != 1:
        raise RuntimeError(f"reference scene {ref_scene_name!r} holds {len(ref_meshes)} "
                           f"meshes, expected the single joined one")
    ref = ref_meshes[0]
    if bum.UV_LAYER_NAME not in ref.data.uv_layers:
        raise RuntimeError(f"{ref.name!r} has no {bum.UV_LAYER_NAME} layer")
    a, b = _uv_array(merged.data), _uv_array(ref.data)
    if a.shape != b.shape:
        raise RuntimeError(f"UV guard: {a.shape[0]} loops against {b.shape[0]} in "
                           f"{ref.name!r}: not the same mesh")
    d = float(np.abs(a - b).max()) if a.size else 0.0
    print(f"bake_normal_variants: UV guard against {ref_scene_name}/{ref.name}: "
          f"{a.shape[0]} loops, max |duv| = {d:.3e}")
    if d >= UV_TOL:
        raise RuntimeError(f"UV guard failed: max |duv| = {d:.3e} >= {UV_TOL:g}. The new "
                           f"BakeUV does not match the reference layout; nothing written.")
    return {"reference_scene": ref_scene_name, "reference_object": ref.name,
            "loops": int(a.shape[0]), "max_abs_duv": d}


def main() -> None:
    args = _parse_args()
    if any(n <= 0 for n in args.sizes):
        raise ValueError(f"sizes must be positive: {args.sizes}")
    out_root = os.path.abspath(args.out)
    targets = {n: os.path.join(out_root, f"normal_{n}") for n in args.sizes}
    if not args.force:
        existing = [os.path.join(d, "BakedMaterial_normal.exr") for d in targets.values()
                    if os.path.exists(os.path.join(d, "BakedMaterial_normal.exr"))]
        if existing:
            raise RuntimeError(f"already present (use --force): {existing}")

    scene = bpy.data.scenes.get(args.scene)
    if scene is None:
        raise RuntimeError(f"scene {args.scene!r} not found "
                           f"(scenes: {[s.name for s in bpy.data.scenes]})")
    bpy.context.window.scene = scene
    view_layer = bpy.context.view_layer

    cfg = bum.BakeConfig(
        output_dir=out_root,
        channels={"base_color": False, "metallic": False, "roughness": False, "normal": True},
        normal_space="OBJECT",
        samples=args.samples,
        save_blend=False,
    )

    # The steps of bum.run(), in its order, up to the packed BakeUV.
    bum._setup_render_settings(scene, cfg)
    source_objects = [o for o in scene.objects if o.type == "MESH"]
    if not source_objects:
        raise RuntimeError(f"no MESH object in scene {scene.name!r}")
    duplicates = bum._duplicate_objects(source_objects, view_layer, cfg)
    if cfg.apply_transform:
        bum._apply_transforms(duplicates, view_layer)
    for dup in duplicates:
        bum._prepare_bake_uv_per_object(dup, view_layer, cfg)
    merged = bum._join_objects(duplicates, view_layer, cfg)
    bum._ensure_default_material(merged)
    bum._pack_bake_uv(merged, view_layer, cfg)

    guard = _uv_guard(merged, args.reference_scene)

    for n in sorted(args.sizes, reverse=True):
        cfg_n = dataclasses.replace(cfg, output_dir=targets[n], resolutions={"normal": (n, n)})
        t0 = time.perf_counter()
        baked = bum._bake_all(merged, cfg_n)
        dt = time.perf_counter() - t0
        image, path = baked["normal"]
        bpy.data.images.remove(image)  # a 4096² float image is ~270 MB
        info = {
            "source_blend": bpy.data.filepath,
            "scene": scene.name,
            "size": [n, n],
            "samples": cfg.samples,
            "bake_margin": cfg.bake_margin,
            "margin_type": scene.render.bake.margin_type,
            "normal_space": cfg.normal_space,
            "swizzle": [cfg.normal_r, cfg.normal_g, cfg.normal_b],
            "uv_island_margin": cfg.uv_island_margin,
            "encoding": "0_1 (n*0.5+0.5), world space (transforms applied)",
            "blender": bpy.app.version_string,
            "uv_guard": guard,
            "bake_seconds": round(dt, 1),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(os.path.join(targets[n], "bake_info.json"), "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=2)
        print(f"bake_normal_variants: {n}x{n} in {dt:.1f} s -> {path}")

    print("bake_normal_variants: done (the .blend was not saved)")


if __name__ == "__main__":
    main()
