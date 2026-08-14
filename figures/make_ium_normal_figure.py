#!/usr/bin/env python
"""make_ium_normal_figure.py -- recover the geometric normal of the IUM pass (figure 3.7b).

    python make_ium_normal_figure.py <run_dir> --out ../Doc/images/ium

Writes `ium_normal.png`: the face normal `IUM_Generator` computes on the GPU.

**Why this needs a script of its own.**  `ium_normals.exr` on disk does NOT hold that
map.  When the run declares an external normal map, `_apply_external_normal`
(images_generator.py:638, called at :3098) overwrites it host-side *after* the render:
the final file carries the supplied map, not the one the tracer computed.  The geometric
normal therefore exists only for the duration of one call and is then discarded.  Here
the IUM pass alone is re-run, with no NeRF and no external map, and read before anything
overwrites it.

Mesh and atlas resolution come from the `run_manifest.json` of the given run, not
transcribed by hand: if they did not match the run's, the panel would not be aligned with
the other two of the same figure and the comparison would be false.

How to tell the result really is the geometric one: the pass computes FACE normals, so
the cube must show flat, sharp tints and the sphere a visible faceting.  The external map
is baked smooth and would give continuous gradients everywhere.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import _paths  # noqa: F401

from make_depth_figure import normal_rgb, save_png     # noqa: E402
from make_skybox_figure import load_exr                # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="run to read the mesh and atlas resolution from")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    run, out = Path(args.run_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(run / "run_manifest.json", encoding="utf-8") as fh:
        manifest = json.load(fh)
    model_path = manifest["scene"]["model_path"]
    ium_w, ium_h = manifest["config"]["render"]["ium_texture_size"]
    ext = manifest["scene"].get("external_normal_path")
    print(f"mesh      : {model_path}")
    print(f"atlas     : {ium_w}x{ium_h}")
    print(f"run's external map (NOT applied here): {ext}")

    import OptixProgrammablePasses as optix

    model = optix.TriangleMesh()
    model.add_from_obj_file(model_path)

    gen = optix.IUMGenerator()
    gen.set_traversable(model)
    gen.set_texture_size([ium_w, ium_h])
    gen.render()
    res = gen.get_result()          # keep it alive: the *_np views are zero-copy

    nrm = np.array(res.normals_np, dtype=np.float32).reshape(ium_h, ium_w, 3)
    mask = load_exr(run / "ium" / "ium_masks.exr")[..., 0] > 0.5
    print(f"coverage  : {100.0 * mask.mean():.2f}% of the texels")

    save_png(normal_rgb(nrm, mask), out / "ium_normal.png")

    # The external map, i.e. the one that overwrites the geometric normal and that every
    # consumer actually reads.  On disk it is already decoded (per channel in [-1,1],
    # |n| = 1 on the mask), so it goes through the same `normal_rgb` and the two panels
    # are comparable.  It is also the version the pipeline uses, after resampling and
    # range conversion, not the source file in the scene folder.
    disk = load_exr(run / "ium" / "ium_normals.exr")
    save_png(normal_rgb(disk, mask), out / "ium_normal_external.png")

    # That the two panels are not the same file is not a detail: confusing them would
    # make the figure a lie that is hard to spot.
    d = np.abs(disk - nrm)[mask]
    print(f"gap between geometric and external: max {d.max():.4f}, mean {d.mean():.4f}  "
          f"→ {'DIFFERENT (expected)' if d.max() > 1e-3 else 'IDENTICAL (suspicious)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
