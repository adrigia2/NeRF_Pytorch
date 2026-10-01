"""Side-by-side comparison of runs that differ in how the texel normal is handled
(grazing test on the normal-map or the mesh normal, raw or filtered normal map), on the
PBR maps of the metal/roughness solver.

Every run is a folder holding the maps (metallic, base_color, roughness, pbr/n_views):
a full run or its test copy, or a roi_rerun.py sandbox of it; a folder without
sources/{source}/ is read as the source folder itself, and a missing base_color/ falls
back to albedo_pbr/, so the OLD solver's maps (e.g. reference_before/gt) compare too.
Roughness is the aperture index (winning cone / 180), not a GGX α: its MAE against the
GT is indicative, the Spearman rank correlation is the fair measure. Everything
is measured inside --rect ∩ the IUM mask of the LAST run (a sandbox mask carries the
ROI), on four subsets:

  all       every texel of the ROI
  wood      GT metallic < 0.1 (clear dielectrics)
  grooves   wood texels whose GT normal-map normal tilts more than the 85th percentile
            of the wood from its 31×31 local mean: sub-pixel bump detail
  metal     GT metallic > 0.5

For each subset and run: cameras per texel (n_views p10/50/90, share below the 2 the fit
needs), share read as metal (m > 0.5: false metal on wood/grooves, recall on metal),
MAE of metallic and base colour against the GT, and the CHROMA error of the base colour
(mean |(b − mean_c b) − (b_gt − mean_c b_gt)|): the coloured artefacts, separated from a
plain brightness error. Unsolvable texels count with what the solver writes for them
(diffuse), so every run is scored on the same texels.

With --crop a PNG shows GT | run 1 | run 2 | … for base colour (top) and metallic
(bottom) on that window, at 1 texel = 1 pixel.

Usage:
    python compare_grazing_ab.py <run_dir> [<run_dir> ...] --labels A B …
                                 --gt-dir <Blender bake dir> --rect X0 Y0 W H
                                 [--crop X0 Y0 W H] [--out summary.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import _paths  # noqa: F401
from compare_metal_roi import _gt, _load, _r, _to_png  # noqa: E402
from pbr_solver import _read_exr  # noqa: E402


def _chroma(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=-1, keepdims=True)


def _spearman(a: np.ndarray, b: np.ndarray, max_n: int = 1_000_000) -> float:
    """Rank correlation (ties averaged), on a fixed subsample for large sets."""
    from scipy.stats import spearmanr
    if a.size > max_n:
        idx = np.random.default_rng(0).choice(a.size, max_n, replace=False)
        a, b = a[idx], b[idx]
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def _stats(nv, m, b, r, gm, gb, gr) -> dict:
    if nv.size == 0:
        return {}
    return {"texels": int(nv.size),
            "n_views_p10_50_90": [int(v) for v in np.percentile(nv, [10, 50, 90])],
            "n_views_mean": _r(nv.mean(), 2),
            "below_2_views": _r((nv < 2).mean()),
            "read_as_metal": _r((m > 0.5).mean()),
            "mae_metallic": _r(np.abs(m - gm).mean()),
            "mae_base_color": _r(np.abs(b - gb).mean()),
            "chroma_err_base_color": _r(np.abs(_chroma(b) - _chroma(gb)).mean()),
            # roughness here is the aperture index (winning cone /180), not a GGX α:
            # the MAE is indicative, the rank correlation is the fair comparison
            "mae_roughness": _r(np.abs(r - gr).mean()),
            "spearman_roughness": _r(_spearman(r, gr), 3)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="run folders, in display order")
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--rect", nargs=4, type=int, required=True, metavar=("X0", "Y0", "W", "H"))
    ap.add_argument("--crop", nargs=4, type=int, default=None, metavar=("X0", "Y0", "W", "H"))
    ap.add_argument("--crop-scale", type=float, default=1.0,
                    help="downscale factor for the crop tiles (e.g. 0.25 for a whole atlas)")
    ap.add_argument("--source", default="gt")
    ap.add_argument("--out", default=None, help="default: <last run>/normal_ab.json")
    args = ap.parse_args()

    dirs = [Path(d) for d in args.runs]
    labels = args.labels or [d.name for d in dirs]
    if len(labels) != len(dirs):
        raise SystemExit("--labels must name every run")
    gt_dir = Path(args.gt_dir)
    x0, y0, w, h = args.rect
    sl = (slice(y0, y0 + h), slice(x0, x0 + w))

    mk_full = _load(dirs[-1] / "ium" / "ium_masks.exr")
    mk_full = (mk_full[..., 0] if mk_full.ndim == 3 else mk_full) > 0.5
    H, W = mk_full.shape
    mk = mk_full[sl]

    def maps(root: Path, window) -> dict:
        # A run/sandbox folder holds sources/{source}/; a folder without it is taken as
        # the source folder itself (e.g. reference_before/gt with the old solver's maps).
        s = root / "sources" / args.source
        if not s.is_dir():
            s = root
        # The old solver wrote albedo_pbr (a = D/x), the current one base_color (b).
        bc = s / "base_color" / "base_color.exr"
        if not bc.exists():
            bc = s / "albedo_pbr" / "albedo_pbr.exr"
        return {"nv": np.rint(_load(s / "pbr" / "n_views.exr")[window]).astype(np.int32),
                "m": _load(s / "metallic" / "metallic.exr")[window],
                "b": _load(bc)[window],
                "r": _load(s / "roughness" / "roughness.exr")[window]}

    runs = {lab: maps(d, sl) for lab, d in zip(labels, dirs)}
    gm_full = _gt(gt_dir, "metallic", H, W)
    gb_full = _gt(gt_dir, "base_color", H, W)
    gr_full = _gt(gt_dir, "roughness", H, W)
    gm, gb, gr = gm_full[sl], gb_full[sl], gr_full[sl]

    # Groove detector on the ORIGINAL GT normal map: tilt from the local mean normal.
    N = cv2.resize(_read_exr(gt_dir / "BakedMaterial_normal.exr"), (W, H),
                   interpolation=cv2.INTER_AREA)[sl] * 2.0 - 1.0
    N /= np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-6)
    Nb = cv2.blur(N, (31, 31))
    Nb /= np.maximum(np.linalg.norm(Nb, axis=-1, keepdims=True), 1e-6)
    tilt = np.degrees(np.arccos(np.clip((N * Nb).sum(-1), -1.0, 1.0)))

    wood = mk & (gm < 0.1)
    thr = float(np.percentile(tilt[wood], 85)) if wood.any() else 90.0
    subsets = {"all": mk, "wood": wood, "grooves": wood & (tilt > thr), "metal": mk & (gm > 0.5)}

    summary = {"runs": dict(zip(labels, map(str, dirs))), "rect": args.rect,
               "groove_tilt_threshold_deg": _r(thr, 2), "subsets": {}}
    print(f"ROI {args.rect}: {int(mk.sum())} texels · grooves = wood tilt > {thr:.1f}°")
    col = max(14, max(len(lab) for lab in labels) + 2)
    rows_def = [("n_views p10/50/90", "n_views_p10_50_90"), ("n_views mean", "n_views_mean"),
                ("share below 2 views", "below_2_views"), (None, "read_as_metal"),
                ("MAE metallic vs GT", "mae_metallic"), ("MAE base colour vs GT", "mae_base_color"),
                ("chroma error base colour", "chroma_err_base_color"),
                ("MAE roughness (aperture)", "mae_roughness"),
                ("Spearman roughness vs GT", "spearman_roughness")]
    for name, s in subsets.items():
        rows = {lab: _stats(v["nv"][s], v["m"][s], v["b"][s], v["r"][s], gm[s], gb[s], gr[s])
                for lab, v in runs.items()}
        summary["subsets"][name] = rows
        if not rows[labels[0]]:
            continue
        print(f"\n── {name}  ({rows[labels[0]]['texels']} texels)")
        print("   " + " " * 26 + "".join(f"{lab:>{col}s}" for lab in labels))
        for title, key in rows_def:
            if title is None:
                title = "read as metal (recall)" if name == "metal" else "read as metal (false)"
            print(f"   {title:26s}" + "".join(f"{str(rows[lab][key]):>{col}s}" for lab in labels))

    if args.crop:
        cx, cy, cw, ch = args.crop
        win = (slice(cy, cy + ch), slice(cx, cx + cw))
        sc = args.crop_scale

        def small(a):
            if sc == 1.0:
                return a
            return cv2.resize(a, (max(1, round(cw * sc)), max(1, round(ch * sc))),
                              interpolation=cv2.INTER_AREA)

        per_run = [maps(d, win) for d in dirs]
        tiles_b = [_to_png(small(gb_full[win]), True)] + [_to_png(small(m["b"]), True) for m in per_run]
        tiles_m = [_to_png(small(gm_full[win]), False)] + [_to_png(small(m["m"]), False) for m in per_run]
        tiles_r = [_to_png(small(gr_full[win]), False)] + [_to_png(small(m["r"]), False) for m in per_run]
        gap = lambda t: np.pad(t, ((0, 6), (0, 6), (0, 0)), constant_values=255)  # noqa: E731
        grid = np.concatenate([np.concatenate([gap(t) for t in tiles], 1)
                               for tiles in (tiles_b, tiles_m, tiles_r)], 0)
        from PIL import Image, ImageDraw
        img = Image.fromarray(np.pad(grid, ((26, 0), (0, 0), (0, 0)), constant_values=255))
        d = ImageDraw.Draw(img)
        tw = tiles_b[0].shape[1] + 6
        for i, lab in enumerate(["GT"] + labels):
            d.text((i * tw + 4, 6), lab, fill=(0, 0, 0))
        stem = Path(args.out).stem if args.out else "normal_ab"
        crop_png = (Path(args.out).parent if args.out else dirs[-1]) / f"{stem}_crop.png"
        img.save(crop_png)
        print(f"\n✓ crop (rows: base colour, metallic, roughness): {crop_png}")

    out = Path(args.out) if args.out else dirs[-1] / "normal_ab.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"✓ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
