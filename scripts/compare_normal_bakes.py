"""Compare Blender's per-size normal bakes with the reductions of the full-size map that
the pipeline applied until 2026-10-01.

For every size N three normal maps at N×N:

  bake        <bake_root>/normal_<N>/BakedMaterial_normal.exr, from bake_normal_variants.py:
              the shading normal from the materials' own textures, averaged by Cycles over
              a footprint narrower than the texel (≈ Gaussian σ 0.15 texel), inside each
              triangle
  lanczos     the path of every earlier run: images_generator._load_image_as_vec3 resizes
              the ENCODED full-size map with PIL LANCZOS per channel, no mask, then the
              decode and renormalisation of _apply_external_normal
  masked_avg  the low-pass of the former external_normal_downsample_to: area average of
              the decoded normals weighted by a validity mask, renormalised

measured on the texels the pipeline shades, the IUM mask at N computed by OptiX on the
OBJ. Three subsets:

  interior    the mask eroded by 2 texels
  border      the 2 texels along the edge of the mask (UV island borders): an unmasked
              resize pulls in the (0,0,0)-encoded background once its support outruns the
              16 px bake margin of the full-size map
  detail      the 15% of texels with the most sub-texel relief, measured as the Toksvig
              deficit 1 − |mean of the full-size unit normals over the texel|

Per subset: angle bake↔lanczos and bake↔masked_avg (p50/p90/p99/max, share above 1°/5°/15°).
Coverage: texels of the mask where a map has no usable vector.

Writes to --out: normal_compare.json, angle heatmaps per N (angle_<N>_vs_<map>.png, one
scale) and normals_<N>.png, two crops (the most relief, the worst border) with bake |
lanczos | masked_avg as normal-map RGB and the two angle maps. Everything else is only read.

Usage:
    python compare_normal_bakes.py <bake_root> --source-normal <full-size normal EXR>
                                   --model <Baked.obj> --sizes 4096 2048 1024 512
                                   --out <dir> [--max-deg 15]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

import _paths  # noqa: F401
from export_mesh_normals import angle_deg, heatmap, ium_normals  # noqa: E402
from images_generator import _load_image_as_vec3, _load_image_hw3_native  # noqa: E402
from make_view_count_map import INK, SURFACE, _font, ramp  # noqa: E402

MAPS = ("lanczos", "masked_avg")


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 1e-8)


def _valid(dec: np.ndarray) -> np.ndarray:
    """A decoded normal is usable when it is close to unit length: the background of a
    baked map is (0,0,0) encoded, i.e. |n| = √3 once decoded."""
    return np.abs(np.linalg.norm(dec, axis=-1) - 1.0) < 0.1


def _stats(a: np.ndarray) -> dict:
    if a.size == 0:
        return {"texels": 0}
    r = lambda x: round(float(x), 3)  # noqa: E731
    return {"texels": int(a.size), "p50": r(np.median(a)), "p90": r(np.percentile(a, 90)),
            "p99": r(np.percentile(a, 99)), "max": r(a.max()),
            "over_1deg": r((a > 1).mean()), "over_5deg": r((a > 5).mean()),
            "over_15deg": r((a > 15).mean())}


def _window(score: np.ndarray, weight: np.ndarray, t: int, min_frac: float) -> tuple[int, int]:
    """Top-left corner of the t×t window with the highest mean score over `weight`,
    among the windows where `weight` covers at least `min_frac` of the area."""
    num = cv2.boxFilter((score * weight).astype(np.float32), -1, (t, t), normalize=False,
                        anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
    den = cv2.boxFilter(weight.astype(np.float32), -1, (t, t), normalize=False,
                        anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
    H, W = score.shape
    m = np.where(den > min_frac * t * t, num / np.maximum(den, 1e-6), -1.0)[:H - t + 1, :W - t + 1]
    y, x = np.unravel_index(int(np.argmax(m)), m.shape)
    return int(x), int(y)


def _crop_panel(title: str, x: int, y: int, t: int, normals: dict, angles: dict,
                mask: np.ndarray, max_deg: float, out: Path, px: int = 384) -> None:
    sl = (slice(y, y + t), slice(x, x + t))
    tiles, labels = [], []
    for name, n in normals.items():
        enc = np.where(mask[sl][..., None], np.clip(n[sl] * 0.5 + 0.5, 0, 1), 0.0)
        tiles.append((enc * 255 + 0.5).astype(np.uint8))
        labels.append(name)
    for name, a in angles.items():
        img = np.empty((t, t, 3), dtype=np.float64)
        img[:] = SURFACE
        img[mask[sl]] = ramp(a[sl][mask[sl]] / max_deg)
        tiles.append(img.astype(np.uint8))
        labels.append(f"angolo bake↔{name} (0–{max_deg:g}°)")
    gap, top = 12, 64
    canvas = Image.new("RGB", (len(tiles) * (px + gap) + gap, px + top + gap), SURFACE)
    d = ImageDraw.Draw(canvas)
    d.text((gap, 8), title, font=_font(20, True), fill=INK)
    for i, (tile, lab) in enumerate(zip(tiles, labels)):
        im = Image.fromarray(tile).resize((px, px), Image.NEAREST)
        canvas.paste(im, (gap + i * (px + gap), top))
        d.text((gap + i * (px + gap), top - 24), lab, font=_font(15), fill=INK)
    canvas.save(out, optimize=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bake_root", help="folder holding normal_<N>/BakedMaterial_normal.exr")
    ap.add_argument("--source-normal", required=True,
                    help="the full-size normal map the pipeline used to resize (0_1 encoded)")
    ap.add_argument("--model", required=True, help="the scene OBJ (for the IUM mask)")
    ap.add_argument("--sizes", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-deg", type=float, default=15.0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src_path = Path(args.source_normal).as_posix()
    print(f"source {src_path}")
    src = _load_image_hw3_native(src_path)[..., :3] * 2.0 - 1.0
    src_valid = _valid(src).astype(np.float32)
    src_unit = _unit(src) * src_valid[..., None]
    print(f"  {src.shape[1]}×{src.shape[0]}, valid {src_valid.mean():.3f}")

    summary = {"bake_root": str(Path(args.bake_root)), "source_normal": src_path,
               "model": args.model, "max_deg": args.max_deg, "sizes": {}}
    for N in args.sizes:
        bake_path = Path(args.bake_root) / f"normal_{N}" / "BakedMaterial_normal.exr"
        bake = _load_image_hw3_native(bake_path.as_posix())[..., :3] * 2.0 - 1.0
        if bake.shape[:2] != (N, N):
            raise SystemExit(f"{bake_path} is {bake.shape[1]}×{bake.shape[0]}, not {N}×{N}")
        _, mask = ium_normals(args.model, N, "triangle")

        # lanczos: exactly what _apply_external_normal did with a map of another size
        lz = _load_image_as_vec3(src_path, N, N).reshape(N, N, 3) * 2.0 - 1.0
        # masked_avg: the former external_normal_downsample_to, straight to N
        num = cv2.resize(src * src_valid[..., None], (N, N), interpolation=cv2.INTER_AREA)
        den = cv2.resize(src_valid, (N, N), interpolation=cv2.INTER_AREA)
        ma = np.divide(num, den[..., None], out=np.zeros_like(num), where=den[..., None] > 1e-6)
        # Toksvig deficit of the full-size unit normals over each texel
        mean_unit = np.divide(cv2.resize(src_unit, (N, N), interpolation=cv2.INTER_AREA),
                              den[..., None], out=np.zeros_like(num), where=den[..., None] > 1e-6)
        deficit = 1.0 - np.linalg.norm(mean_unit, axis=-1)

        coverage = {"mask_texels": int(mask.sum()),
                    "bake_unusable": round(float((~_valid(bake) & mask).sum() / mask.sum()), 6),
                    "lanczos_off_unit": round(float((~_valid(lz) & mask).sum() / mask.sum()), 6),
                    "masked_avg_empty": round(float(((den <= 1e-6) & mask).sum() / mask.sum()), 6)}
        bake_u, maps = _unit(bake), {"lanczos": _unit(lz), "masked_avg": _unit(ma)}
        ok = mask & _valid(bake)
        interior = ok & (cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
        border = ok & ~interior
        thr = float(np.percentile(deficit[ok], 85))
        detail = ok & (deficit > thr)
        subsets = {"all": ok, "interior": interior, "border": border, "detail": detail}

        angles = {k: angle_deg(bake_u, v) for k, v in maps.items()}
        res = {"coverage": coverage, "detail_deficit_threshold": round(thr, 6), "angle_deg": {}}
        print(f"\n── {N}×{N}  mask {mask.sum()} texels · coverage {coverage}")
        print(f"   {'subset':10s} {'vs':11s} {'texels':>8s} {'p50':>7s} {'p90':>7s} {'p99':>7s} "
              f"{'max':>7s} {'>1°':>6s} {'>5°':>6s} {'>15°':>6s}")
        for sname, s in subsets.items():
            res["angle_deg"][sname] = {}
            for k in MAPS:
                a = angles[k][s & (np.linalg.norm(maps[k], axis=-1) > 0)]
                st = _stats(a)
                res["angle_deg"][sname][k] = st
                if st["texels"]:
                    print(f"   {sname:10s} {k:11s} {st['texels']:8d} {st['p50']:7.3f} {st['p90']:7.3f} "
                          f"{st['p99']:7.3f} {st['max']:7.2f} {st['over_1deg']:6.3f} "
                          f"{st['over_5deg']:6.3f} {st['over_15deg']:6.3f}")
        summary["sizes"][str(N)] = res

        for k in MAPS:
            heatmap(angles[k], ok, args.max_deg, f"{N}×{N}: bake di Blender contro {k}",
                    out / f"angle_{N}_vs_{k}.png")
        t = max(32, N // 16)
        x, y = _window(deficit, ok.astype(np.float32), t, 0.5)
        _crop_panel(f"{N}×{N} · più rilievo sotto il texel · finestra x={x} y={y} lato {t}", x, y, t,
                    {"bake": bake_u, **maps}, angles, mask, args.max_deg, out / f"crop_{N}_detail.png")
        x, y = _window(angles["lanczos"], border.astype(np.float32), t, 0.02)
        _crop_panel(f"{N}×{N} · bordo d'isola peggiore per lanczos · finestra x={x} y={y} lato {t}", x, y, t,
                    {"bake": bake_u, **maps}, angles, mask, args.max_deg, out / f"crop_{N}_border.png")

    with open(out / "normal_compare.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n✓ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
