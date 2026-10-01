"""Export the IUM mesh normals of an OBJ for visual inspection, in both modes.

Runs IUMGenerator on the mesh alone (no external normal map, nothing else of the
pipeline) with set_mesh_normals("triangle") and ("obj"), and writes to <out_dir>:

  mesh_normals_triangle.exr / mesh_normals_obj.exr
      world-space normal per texel, raw float RGB in [-1, 1] (0 outside the mesh).
      Open them in tev or in Blender's image editor.
  mesh_normals_triangle.png / mesh_normals_obj.png
      the same, encoded as a normal map (n·0.5 + 0.5), for a quick look.
  angle_obj_vs_triangle.png
      heatmap of the angle between the two modes: where the OBJ says "smooth".
  angle_obj_vs_reference.png   (with --reference)
      heatmap of the angle between the "obj" normals and a reference normal map,
      e.g. Blender's own flat normal map. The reference is only read.

The heatmaps use one blue ramp from 0° to --max-deg (clipped above), the page colour
outside the mesh; a legend strip on the right gives the scale.

Usage:
    python export_mesh_normals.py <model.obj> <out_dir> [--size 4096]
                                  [--reference NORMAL_MAP.exr] [--reference-range 0_1|-1_1]
                                  [--max-deg 20]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import _paths  # noqa: F401
import OptixProgrammablePasses as opp  # noqa: E402
from images_generator import DataLayer, ImageFormat, _save_layer  # noqa: E402
from make_view_count_map import SURFACE, INK, INK2, ramp, _font  # noqa: E402
from pbr_solver import _read_exr  # noqa: E402


def ium_normals(obj: str, size: int, mode: str):
    mesh = opp.TriangleMesh()
    mesh.add_from_obj_file(obj)
    if mode == "obj" and not mesh.has_normals:
        raise SystemExit(f"{obj} has no vertex normals (vn): nothing to compare")
    gen = opp.IUMGenerator()
    gen.set_traversable(mesh)
    gen.set_texture_size([size, size])
    gen.set_mesh_normals(mode)
    gen.render()
    res = gen.get_result()
    n = np.array(res.normals_np, dtype=np.float32).reshape(size, size, 3)
    m = np.array(res.masks_np, dtype=np.uint8).reshape(size, size) > 0
    return n, m


def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    c = (a * b).sum(-1) / np.maximum(na * nb, 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def heatmap(ang: np.ndarray, mask: np.ndarray, max_deg: float, title: str, out: Path) -> None:
    H, W = ang.shape
    img = np.empty((H, W, 3), dtype=np.float64)
    img[:] = SURFACE
    img[mask] = ramp(ang[mask] / max_deg)
    img = img.astype(np.uint8)
    u = H / 4096
    TOP, RIGHT = int(150 * u) + 20, int(560 * u) + 120
    canvas = Image.new("RGB", (W + RIGHT, H + TOP), SURFACE)
    canvas.paste(Image.fromarray(img), (0, TOP))
    d = ImageDraw.Draw(canvas)
    d.text((int(30 * u), int(24 * u)), title, font=_font(int(60 * u) + 8, True), fill=INK)
    a = ang[mask]
    d.text((int(30 * u), int(100 * u) + 4),
           f"p50 {np.median(a):.2f}°   p90 {np.percentile(a, 90):.2f}°   p99 {np.percentile(a, 99):.2f}°   "
           f"texel oltre 1°: {100 * (a > 1).mean():.1f}%", font=_font(int(42 * u) + 6), fill=INK2)
    bw, bh = int(100 * u) + 16, int(H * 0.6)
    x0, y0 = W + int(70 * u) + 16, TOP + int(60 * u)
    col = ramp(np.linspace(1, 0, bh))[:, None, :].repeat(bw, axis=1).astype(np.uint8)
    canvas.paste(Image.fromarray(col), (x0, y0))
    f = _font(int(44 * u) + 6)
    for t in np.linspace(0, max_deg, 5):
        y = y0 + round((1 - t / max_deg) * (bh - 1))
        d.line([(x0 + bw, y), (x0 + bw + 14, y)], fill=INK2, width=2)
        d.text((x0 + bw + 22, y - int(22 * u) - 4), f"{t:g}°" + ("+" if t == max_deg else ""), font=f, fill=INK)
    canvas.save(out, optimize=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("obj")
    ap.add_argument("out_dir")
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--reference", default=None, help="normal map to compare with (only read)")
    ap.add_argument("--reference-range", choices=("0_1", "-1_1"), default="0_1")
    ap.add_argument("--max-deg", type=float, default=20.0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    maps = {}
    for mode in ("triangle", "obj"):
        n, m = ium_normals(args.obj, args.size, mode)
        maps[mode] = n
        _save_layer(n, (out / f"mesh_normals_{mode}.exr").as_posix(), ImageFormat.OPENEXR, DataLayer.NORMAL)
        enc = np.where(m[..., None], n * 0.5 + 0.5, 0.0)
        Image.fromarray((np.clip(enc, 0, 1) * 255 + 0.5).astype(np.uint8)).save(out / f"mesh_normals_{mode}.png")
    mask = m & (cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    heatmap(angle_deg(maps["obj"], maps["triangle"]), mask, args.max_deg,
            "Normale della mesh: \"obj\" contro \"triangle\"", out / "angle_obj_vs_triangle.png")
    if args.reference:
        ref = _read_exr(Path(args.reference)).astype(np.float32)[..., :3]
        if ref.shape[:2] != (args.size, args.size):
            ref = cv2.resize(ref, (args.size, args.size), interpolation=cv2.INTER_LANCZOS4)
        if args.reference_range == "0_1":
            ref = ref * 2.0 - 1.0
        heatmap(angle_deg(maps["obj"], ref), mask, args.max_deg,
                f"Normale \"obj\" contro {Path(args.reference).name}", out / "angle_obj_vs_reference.png")
    print(f"✓ written to {out}")
    for p in sorted(out.iterdir()):
        print("   ", p.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
