"""Fit PBR multi-vista (prima versione, modello a cono):

    C_j = X · D + (1 - X) · L_j(r)

    C_j    colore osservato dalla camera j   (camera_texture/<stem>.exr)
    D      termine diffuso vista-indipendente (pixel_change/color_min.exr)
    L_j(r) radianza media del cono attorno al raggio riflesso
           (spec_cone/cam_jjj_rkk.exr, precompute di images_generator.py)
    X      peso del termine DIFFUSO (X=1 → nessuna dipendenza dalla vista);
           la specularità/metallic è 1−X. Forma chiusa per ogni r della griglia.
    r      apertura del cono: scelta per scan (residuo minimo)

Per ogni livello r_k il residuo è una parabola in X:
    res(X) = S_cc - 2·X·S_cd + X²·S_dd
    S_cc = Σ_{j,c} w_j (C-L)²,  S_cd = Σ w_j (C-L)(D-L),  S_dd = Σ w_j (D-L)²
quindi X*(r_k) = clip(S_cd / S_dd, 0, 1) e il residuo si valuta senza
ripassare sui dati. Si tiene il livello con residuo minimo per texel.

Gate e validità:
  - gate diffuso scale-invariant sul coefficiente di variazione tra camere:
    sqrt(var) < cv_gate · luminanza(D)  →  X=1, metallic=0, r non vincolato;
  - r è attendibile solo dove la specularità supera spec_threshold: sotto,
    roughness viene posta a 1.0 (nessun riflesso nitido) e il texel è
    marcato in pbr/r_valid.png.

Mappe finali (come l'albedo, cartelle dedicate, EXR):
  <out>/metallic/metallic.exr    = 1−X   (0=diffuso, 1=tutto speculare)
  <out>/roughness/roughness.exr  = r/180 dove valido, 1.0 altrove
Diagnostica in <out>/pbr/: diffuse_weight.exr (X), spec_cone_r.exr (gradi),
residual.exr, n_views.exr, r_valid.png + preview PNG a scala assoluta.

Uso:
    python pbr_solver.py <output_dir> [--cv-gate 0.05] [--spec-threshold 0.2]
                         [--min-views 2]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from images_generator import (  # noqa: E402
    DataLayer, ImageFormat, _save_layer,
)


# ──────────────────────────────────────────────────────────────────────────────
# IO helpers
# ──────────────────────────────────────────────────────────────────────────────

def _read_exr(path: Path) -> np.ndarray:
    """EXR → (H, W) float32 [canale Z] oppure (H, W, C) [R,G,B(,A) o Cam*]."""
    import OpenEXR, Imath

    exr = OpenEXR.InputFile(path.as_posix())
    header = exr.header()
    dw = header["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    names = list(header["channels"].keys())

    def chan(name):
        return np.frombuffer(exr.channel(name, pt), dtype=np.float32).reshape(h, w)

    if names == ["Z"] or names == ["Y"]:
        return chan(names[0])
    if set("RGB").issubset(names):
        chans = ["R", "G", "B"] + (["A"] if "A" in names else [])
    else:  # canali arbitrari (es. Cam0, Cam1, …) in ordine numerico
        chans = sorted(names, key=lambda n: (len(n), n))
    return np.stack([chan(c) for c in chans], axis=-1)


def _read_mask_png(path: Path) -> np.ndarray:
    from PIL import Image
    return (np.asarray(Image.open(path)) > 0)


# ──────────────────────────────────────────────────────────────────────────────
# Solver
# ──────────────────────────────────────────────────────────────────────────────

def solve_pbr(output_dir: str,
              cv_gate: float = 0.05,
              spec_threshold: float = 0.2,
              min_views: int = 2,
              eps: float = 1e-12) -> dict:
    out = Path(output_dir)
    spec_dir = out / "spec_cone"

    with open(spec_dir / "spec_cone_meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    with open(out / "transforms_extended.json", encoding="utf-8") as fh:
        tjson = json.load(fh)

    apertures = np.asarray(meta["apertures_deg"], dtype=np.float32)
    cams      = meta["cameras"]
    K         = meta["num_levels"]
    stems     = [Path(f["file_path"]).stem for f in tjson["frames"]]

    # ── Input vista-indipendenti ─────────────────────────────────────────────
    D_img = _read_exr(out / "pixel_change" / "color_min.exr")          # (H, W, 3)
    var   = _read_exr(out / "pixel_change" / "color_variance.exr")     # (H, W, 3)
    H, W  = D_img.shape[:2]
    N     = H * W
    D     = D_img.reshape(N, 3).astype(np.float64)

    mask_path = out / "ium" / "ium_masks.exr"
    if mask_path.exists():
        mask = _read_exr(mask_path).reshape(N) > 0.5
    else:
        mask = np.ones(N, dtype=bool)

    vis_path = out / "visibility" / "visibility.exr"
    vis = _read_exr(vis_path).reshape(N, -1) > 0.5 if vis_path.exists() else None

    # ── Osservazioni e pesi per camera (costanti rispetto a r) ───────────────
    C_list, w_list = [], []
    for j in cams:
        C_j = _read_exr(out / "camera_texture" / f"{stems[j]}.exr")
        C_list.append(C_j.reshape(N, 3).astype(np.float64))
        w_j = mask.copy()
        if vis is not None:
            w_j &= vis[:, j]
        w_j &= _read_mask_png(spec_dir / meta["valid_file_pattern"].format(cam=j)).reshape(N)
        w_list.append(w_j)

    n_views = np.sum(np.stack(w_list), axis=0).astype(np.int32)        # (N,)
    solvable = mask & (n_views >= min_views)

    # Gate diffuso scale-invariant: variazione tra camere piccola rispetto
    # alla luminanza del texel → X=1, metallic=0, r non vincolato
    lum_D   = D.mean(axis=-1)
    lum_std = np.sqrt(np.maximum(var.reshape(N, 3).mean(axis=-1), 0.0))
    diffuse_gate = solvable & (lum_std < cv_gate * np.maximum(lum_D, 1e-6))

    print(f"[pbr_solver] {N} texel, {len(cams)} camere, {K} livelli r")
    print(f"  texel risolvibili: {int(solvable.sum())}, "
          f"di cui gated come diffusi (CV<{cv_gate}): {int(diffuse_gate.sum())}")

    # ── Scan sui livelli r ────────────────────────────────────────────────────
    best_res = np.full(N, np.inf)
    best_X   = np.ones(N)
    best_k   = np.zeros(N, dtype=np.int32)

    target = solvable & ~diffuse_gate
    for k in range(K):
        S_cc = np.zeros(N); S_cd = np.zeros(N); S_dd = np.zeros(N)
        for j, C_j, w_j in zip(cams, C_list, w_list):
            L = _read_exr(spec_dir / meta["level_file_pattern"].format(cam=j, level=k))
            L = L.reshape(N, 3).astype(np.float64)
            dC = (C_j - L); dD = (D - L)
            wf = w_j.astype(np.float64)[:, None]
            S_cc += (wf * dC * dC).sum(axis=-1)
            S_cd += (wf * dC * dD).sum(axis=-1)
            S_dd += (wf * dD * dD).sum(axis=-1)

        X_k = np.clip(S_cd / np.maximum(S_dd, eps), 0.0, 1.0)
        res_k = S_cc - 2.0 * X_k * S_cd + X_k * X_k * S_dd
        res_k /= np.maximum(3.0 * n_views, 1.0)   # residuo medio per equazione

        better = target & (res_k < best_res)
        best_res[better] = res_k[better]
        best_X[better]   = X_k[better]
        best_k[better]   = k
        print(f"  livello r={apertures[k]:6.1f}° → migliore per "
              f"{int(better.sum())} texel")

    # ── Composizione output ───────────────────────────────────────────────────
    fitted = target & np.isfinite(best_res)

    diffuse_w = np.zeros(N, dtype=np.float32)     # X
    diffuse_w[diffuse_gate] = 1.0
    diffuse_w[fitted] = best_X[fitted].astype(np.float32)

    metallic = np.zeros(N, dtype=np.float32)      # 1−X (specularità)
    metallic[fitted] = (1.0 - best_X[fitted]).astype(np.float32)

    cone_r = np.zeros(N, dtype=np.float32)
    cone_r[fitted] = apertures[best_k[fitted]]

    # r attendibile solo con specularità sufficiente; altrove roughness=1
    r_valid = fitted & (metallic >= spec_threshold)
    roughness = np.where(mask, 1.0, 0.0).astype(np.float32)
    roughness[r_valid] = cone_r[r_valid] / 180.0

    residual = np.zeros(N, dtype=np.float32)
    residual[fitted] = best_res[fitted].astype(np.float32)

    print(f"  texel con r attendibile (metallic≥{spec_threshold}): "
          f"{int(r_valid.sum())}")

    # ── Mappe finali (cartelle dedicate, come l'albedo) ───────────────────────
    fmt = ImageFormat.OPENEXR
    met_dir = out / "metallic";  met_dir.mkdir(parents=True, exist_ok=True)
    rgh_dir = out / "roughness"; rgh_dir.mkdir(parents=True, exist_ok=True)
    metallic_path  = (met_dir / "metallic.exr").resolve().as_posix()
    roughness_path = (rgh_dir / "roughness.exr").resolve().as_posix()
    _save_layer(metallic.reshape(H, W), metallic_path, fmt, DataLayer.METALLIC)
    _save_layer(roughness.reshape(H, W), roughness_path, fmt, DataLayer.ROUGHNESS)

    # ── Diagnostica ───────────────────────────────────────────────────────────
    pbr_dir = out / "pbr"
    pbr_dir.mkdir(parents=True, exist_ok=True)
    _save_layer(diffuse_w.reshape(H, W), (pbr_dir / "diffuse_weight.exr").as_posix(),
                fmt, DataLayer.METALLIC)
    _save_layer(cone_r.reshape(H, W), (pbr_dir / "spec_cone_r.exr").as_posix(),
                fmt, DataLayer.SPEC_CONE_R)
    _save_layer(residual.reshape(H, W), (pbr_dir / "residual.exr").as_posix(),
                fmt, DataLayer.SPEC_CONE_R)
    _save_layer(n_views.reshape(H, W).astype(np.float32),
                (pbr_dir / "n_views.exr").as_posix(), fmt, DataLayer.SPEC_CONE_R)
    _save_layer(r_valid.reshape(H, W).astype(np.uint8),
                (pbr_dir / "r_valid.png").as_posix(), ImageFormat.PNG,
                DataLayer.MASK)

    # Preview PNG a scala assoluta
    from PIL import Image
    Image.fromarray((np.clip(metallic, 0, 1).reshape(H, W) * 255).astype(np.uint8)
                    ).save(pbr_dir / "metallic_preview.png")
    Image.fromarray((np.clip(roughness, 0, 1).reshape(H, W) * 255).astype(np.uint8)
                    ).save(pbr_dir / "roughness_preview.png")

    print(f"✓ metallic:  {metallic_path}")
    print(f"✓ roughness: {roughness_path}")
    print(f"✓ diagnostica in {pbr_dir}")
    return {
        "metallic_path": metallic_path,
        "roughness_path": roughness_path,
        "metallic": metallic.reshape(H, W),
        "roughness": roughness.reshape(H, W),
        "diffuse_weight": diffuse_w.reshape(H, W),
        "cone_r": cone_r.reshape(H, W),
        "residual": residual.reshape(H, W),
        "n_views": n_views.reshape(H, W),
        "r_valid": r_valid.reshape(H, W),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Fit PBR C_j = X·D + (1-X)·L_j(r) → metallic/roughness")
    ap.add_argument("output_dir", help="output_dir del pipeline (contiene spec_cone/, ...)")
    ap.add_argument("--cv-gate", type=float, default=0.05,
                    help="gate diffuso: std tra camere < cv_gate·luminanza → metallic=0")
    ap.add_argument("--spec-threshold", type=float, default=0.2,
                    help="metallic minimo perché r sia attendibile (sotto: roughness=1)")
    ap.add_argument("--min-views", type=int, default=2,
                    help="minimo di camere valide per texel")
    args = ap.parse_args()
    solve_pbr(args.output_dir, args.cv_gate, args.spec_threshold, args.min_views)
