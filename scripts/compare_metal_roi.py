"""Compare the metal/roughness export of pbr_solver with the maps it replaced and with
the Blender ground truth, ROI by ROI (a ROI may be the whole texture).

Three sets of maps, all in IUM texel space (4096², no flip: the GT bakes share the
UV layout of the OBJ the pipeline renders):

  new     sources/{source}/metallic, base_color, pbr/{spec_color, diffuse_albedo,
          slope_se, consistency, residual, lobe_param, n_views}
          (pbr_solver after 2026-09-30)
  before  reference_before/{source}/metallic (= the shared slope β),
          albedo_pbr (= a), pbr/{residual, lobe_param}   (snapshot of the old solver)
  gt      <gt-dir>/BakedMaterial_{metallic,base_color,roughness}.exr, resampled to
          the IUM resolution with INTER_AREA

Besides metallic and base colour it compares the two reflectances the fit actually
measures against their GT counterparts, S ↔ F0 = 0.04(1−m)+m·b and
D ↔ (1−m)·b, and the reflection Blender renders from the OLD maps,
β·a + 0.04(1−β): the old solver meant a white reflection of strength β, Blender
tints it by a.

Metrics are computed on the SEEN texels (n_views ≥ 2): an unseen texel is written as
diffuse by convention and says nothing about the model. The metal/dielectric
classification is reported as the share of GT-metal texels read as metal (m > 0.5)
and of GT-dielectric texels wrongly read as metal.

An analysis-only column, "variant", recomputes (m, b) with the magnitude of S taken
from the old shared slope β (the VLL-weighted mean of the channels, which
down-weights a channel whose L barely varies) and only the chroma from the new
per-channel slopes. It is not something the solver writes.

Checks printed per ROI (and saved to <out>/summary.json):
  - the new residual is ≤ the old one (more parameters per texel), up to roundoff;
  - every new map is exactly 0 outside the union of the ROIs.
Side-by-side crops (GT | before | new) go to <out>/<roi>.png, downsampled to at most
--panel-max pixels per side.

Usage:
    python compare_metal_roi.py <scene_dir> --gt-dir <Blender bake dir>
                                --roi NAME X0 Y0 W H [--roi ...] [--source gt]
                                [--out DIR] [--panel-max 1024]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import _paths  # noqa: F401
from pbr_solver import F0_DIELECTRIC, _metal_from_reflectances, _read_exr  # noqa: E402


def _load(path: Path) -> np.ndarray:
    return _read_exr(path).astype(np.float32)


def _gt(gt_dir: Path, name: str, H: int, W: int) -> np.ndarray:
    a = _read_exr(gt_dir / f"BakedMaterial_{name}.exr")
    a = cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)
    return a if name == "base_color" else a[..., 0]     # metallic/roughness: R = G = B


def _r(v, nd=4):
    return round(float(v), nd)


def _pct(a) -> list:
    return [_r(v) for v in np.percentile(a, [10, 50, 90])] if a.size else []


def _mean3(a) -> list:
    return [_r(v) for v in a.mean(axis=0)] if len(a) else []


def _mae(a, b) -> float:
    return _r(np.abs(a - b).mean()) if a.size else float("nan")


def _classif(m_hat, m_gt) -> dict:
    metal, diel = m_gt > 0.5, m_gt <= 0.5
    return {"metal_recall": _r((m_hat[metal] > 0.5).mean()) if metal.any() else None,
            "false_metal": _r((m_hat[diel] > 0.5).mean()) if diel.any() else None}


def _to_png(a: np.ndarray, colour: bool) -> np.ndarray:
    """Linear [0,1] → 8-bit sRGB-ish (gamma 2.2), grey maps replicated to RGB."""
    a = np.clip(a, 0.0, 1.0)
    if colour:
        a = a ** (1 / 2.2)
    else:
        a = np.repeat(a[..., None], 3, axis=-1)
    return (a * 255 + 0.5).astype(np.uint8)


def _panel(rows: "list[list[np.ndarray]]", labels: "list[str]", scale: int) -> np.ndarray:
    from PIL import Image, ImageDraw
    tiles = [np.concatenate([np.pad(t, ((0, 0), (0, 4), (0, 0)), constant_values=255)
                             for t in r], axis=1) for r in rows]
    grid = np.concatenate([np.pad(t, ((0, 4), (0, 0), (0, 0)), constant_values=255)
                           for t in tiles], axis=0)
    grid = np.repeat(np.repeat(grid, scale, axis=0), scale, axis=1)
    img = Image.fromarray(np.pad(grid, ((22, 0), (0, 0), (0, 0)), constant_values=255))
    draw = ImageDraw.Draw(img)
    col_w = (rows[0][0].shape[1] + 4) * scale
    for i, lab in enumerate(labels):
        draw.text((i * col_w + 4, 4), lab, fill=(0, 0, 0))
    return np.asarray(img)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("scene_dir")
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--roi", nargs=5, action="append", required=True,
                    metavar=("NAME", "X0", "Y0", "W", "H"))
    ap.add_argument("--source", default="gt")
    ap.add_argument("--out", default=None, help="default: <scene_dir>/metal_test")
    ap.add_argument("--panel-max", type=int, default=1024,
                    help="crops larger than this (pixels per side) are downsampled")
    args = ap.parse_args()

    root = Path(args.scene_dir)
    new = root / "sources" / args.source
    old = root / "reference_before" / args.source
    gt_dir = Path(args.gt_dir)
    out = Path(args.out) if args.out else root / "metal_test"
    out.mkdir(parents=True, exist_ok=True)
    rois = [(r[0], *(int(v) for v in r[1:])) for r in args.roi]

    mask = _load(root / "ium" / "ium_masks.exr")
    mask = (mask[..., 0] if mask.ndim == 3 else mask) > 0.5
    H, W = mask.shape

    n = {"m": _load(new / "metallic" / "metallic.exr"),
         "b": _load(new / "base_color" / "base_color.exr"),
         "S": _load(new / "pbr" / "spec_color.exr"),
         "D": _load(new / "pbr" / "diffuse_albedo.exr"),
         "se": _load(new / "pbr" / "slope_se.exr"),
         "cons": _load(new / "pbr" / "consistency.exr"),
         "res": _load(new / "pbr" / "residual.exr"),
         "lobe": _load(new / "pbr" / "lobe_param.exr")}
    nv = _load(new / "pbr" / "n_views.exr")
    o = {"beta": _load(old / "metallic" / "metallic.exr"),
         "a": _load(old / "albedo_pbr" / "albedo_pbr.exr"),
         "res": _load(old / "pbr" / "residual.exr"),
         "lobe": _load(old / "pbr" / "lobe_param.exr")}
    g = {"m": _gt(gt_dir, "metallic", H, W),
         "b": _gt(gt_dir, "base_color", H, W),
         "r": _gt(gt_dir, "roughness", H, W)}
    g["F0"] = F0_DIELECTRIC * (1 - g["m"])[..., None] + g["m"][..., None] * g["b"]
    g["D"] = (1 - g["m"])[..., None] * g["b"]
    # What Blender renders from the OLD maps: Base Color = a, Metallic = β.
    o["refl"] = F0_DIELECTRIC * (1 - o["beta"])[..., None] + o["beta"][..., None] * o["a"]

    # Variant (analysis only): magnitude of S from the old shared slope, chroma from
    # the new per-channel slopes.
    S_mean = n["S"].mean(axis=-1, keepdims=True)
    S_var = np.where(S_mean > 1e-9, n["S"] * (o["beta"][..., None] / np.maximum(S_mean, 1e-9)),
                     o["beta"][..., None])
    m_var, b_var, _, _ = _metal_from_reflectances(np.clip(n["D"], 0.0, 1.0), S_var)
    v = {"m": m_var.astype(np.float32), "b": b_var.astype(np.float32)}
    del S_mean, S_var, m_var, b_var

    # ── every new map must be 0 outside the union of the ROIs ────────────────
    union = np.zeros((H, W), dtype=bool)
    for _, x0, y0, w, h in rois:
        union[y0:y0 + h, x0:x0 + w] = True
    leak = {k: int(np.count_nonzero(a[~union])) for k, a in n.items()}
    print(f"non-zero texels outside the ROIs: {leak}")
    summary = {"scene_dir": str(root), "gt_dir": str(gt_dir),
               "outside_roi_nonzero": leak, "rois": {}}

    for name, x0, y0, w, h in rois:
        sl = (slice(y0, y0 + h), slice(x0, x0 + w))
        mk = mask[sl]
        seen = mk & (nv[sl] >= 2)
        sel = {k: a[sl][seen] for k, a in n.items()}
        sel_o = {k: a[sl][seen] for k, a in o.items()}
        sel_g = {k: a[sl][seen] for k, a in g.items()}
        sel_v = {k: a[sl][seen] for k, a in v.items()}

        # the old fit had fewer parameters: its residual bounds the new one
        # (up to the per-channel clip and float32 storage)
        tol = 1e-6 * np.maximum(sel_o["res"], 1e-12) + 1e-9
        res_ok = _r((sel["res"] <= sel_o["res"] + tol).mean()) if seen.any() else None

        r = {
            "rect": [x0, y0, w, h], "texels": int(mk.sum()), "seen": int(seen.sum()),
            "seen_frac": _r(seen.sum() / max(mk.sum(), 1)),
            "gt": {"metallic": _pct(sel_g["m"]), "base_color": _mean3(sel_g["b"]),
                   "roughness": _pct(sel_g["r"]), "F0": _mean3(sel_g["F0"]),
                   "D": _mean3(sel_g["D"]),
                   "frac_metal": _r((sel_g["m"] > 0.5).mean()) if seen.any() else None},
            "before": {"beta": _pct(sel_o["beta"]), "albedo_pbr": _mean3(sel_o["a"]),
                       "blender_reflection": _mean3(sel_o["refl"]),
                       "mae_metallic_vs_gt": _mae(sel_o["beta"], sel_g["m"]),
                       "mae_base_vs_gt": _mae(sel_o["a"], sel_g["b"]),
                       "mae_reflection_vs_F0": _mae(sel_o["refl"], sel_g["F0"]),
                       **_classif(sel_o["beta"], sel_g["m"]),
                       "lobe_median_deg": float(np.median(sel_o["lobe"])) if seen.any() else None},
            "new": {"metallic": _pct(sel["m"]), "base_color": _mean3(sel["b"]),
                    "S": _mean3(sel["S"]), "D": _mean3(sel["D"]),
                    "mae_metallic_vs_gt": _mae(sel["m"], sel_g["m"]),
                    "mae_base_vs_gt": _mae(sel["b"], sel_g["b"]),
                    "mae_S_vs_F0": _mae(sel["S"], sel_g["F0"]),
                    **_classif(sel["m"], sel_g["m"]),
                    "slope_se_median": [_r(x) for x in np.median(sel["se"], axis=0)]
                    if seen.any() else [],
                    "consistency_median": _r(np.median(sel["cons"])) if seen.any() else None,
                    "lobe_median_deg": float(np.median(sel["lobe"])) if seen.any() else None},
            "variant_shared_magnitude": {
                    "mae_metallic_vs_gt": _mae(sel_v["m"], sel_g["m"]),
                    "mae_base_vs_gt": _mae(sel_v["b"], sel_g["b"]),
                    **_classif(sel_v["m"], sel_g["m"])},
            "residual_new_le_old": res_ok,
        }
        summary["rois"][name] = r
        B, N_, V = r["before"], r["new"], r["variant_shared_magnitude"]

        print(f"\n══ {name} {r['rect']}  ({r['texels']} texels, seen by ≥2 cameras: "
              f"{r['seen']} = {100 * r['seen_frac']:.1f}%)")
        print(f"  metallic p10/50/90   GT {r['gt']['metallic']}   before β {B['beta']}   "
              f"new m {N_['metallic']}")
        print(f"  metallic MAE vs GT   before {B['mae_metallic_vs_gt']}   new {N_['mae_metallic_vs_gt']}"
              f"   variant {V['mae_metallic_vs_gt']}")
        print(f"  GT metal read metal  before {B['metal_recall']}   new {N_['metal_recall']}   "
              f"variant {V['metal_recall']}   (GT metal share {r['gt']['frac_metal']})")
        print(f"  false metal          before {B['false_metal']}   new {N_['false_metal']}   "
              f"variant {V['false_metal']}")
        print(f"  base colour (mean)   GT {r['gt']['base_color']}   before a {B['albedo_pbr']}   "
              f"new b {N_['base_color']}")
        print(f"  base colour MAE      before {B['mae_base_vs_gt']}   new {N_['mae_base_vs_gt']}   "
              f"variant {V['mae_base_vs_gt']}")
        print(f"  reflection vs GT F0  Blender from old maps: MAE {B['mae_reflection_vs_F0']}   "
              f"new S: MAE {N_['mae_S_vs_F0']}   (means: GT {r['gt']['F0']}, new {N_['S']})")
        print(f"  diffuse              GT (1−m)b {r['gt']['D']}   new D {N_['D']}")
        print(f"  lobe median          GT roughness {r['gt']['roughness'][1] if seen.any() else None}   "
              f"before {B['lobe_median_deg']}°   new {N_['lobe_median_deg']}°")
        print(f"  diagnostics          SE median {N_['slope_se_median']}   consistency "
              f"{N_['consistency_median']}   residual new≤old {r['residual_new_le_old']}")

        # ── crops: GT | before | new ────────────────────────────────────────
        def crop(a):
            c = a[sl]
            s = max(h, w) / args.panel_max
            if s > 1:
                c = cv2.resize(c, (round(w / s), round(h / s)), interpolation=cv2.INTER_AREA)
            return c

        rows = [[_to_png(crop(g["m"]), False), _to_png(crop(o["beta"]), False),
                 _to_png(crop(n["m"]), False)],
                [_to_png(crop(g["b"]), True), _to_png(crop(o["a"]), True),
                 _to_png(crop(n["b"]), True)],
                [_to_png(crop(g["F0"]), True), _to_png(crop(o["refl"]), True),
                 _to_png(crop(n["S"]), True)]]
        scale = max(1, 256 // max(h, 1)) if h < 256 else 1
        from PIL import Image
        Image.fromarray(_panel(rows, ["GT", "before (Blender view)", "new"], scale)
                        ).save(out / f"{name}.png")

    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n✓ summary and crops in {out}")
    print("  rows of each crop: metallic | base colour | reflection "
          "(GT F0, what Blender renders from the old maps, new S)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
