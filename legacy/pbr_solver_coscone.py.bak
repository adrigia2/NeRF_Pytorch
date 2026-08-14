"""Fit PBR multi-vista con coni speculari a taglio netto (schema unico "coscone"):

    C_j = X · D + (1 - X) · L_j

    C_j   colore osservato dalla camera j    (sources/{source}/camera_texture/<stem>.exr)
    D     termine diffuso vista-indipendente (sources/{source}/pixel_change/color_min.exr)
    L_j   INTEGRALE PURO ∫L·cosθ_R dω dell'ambiente sul cono attorno al raggio
          riflesso, ricostruito dalle medie per-anello del bake
          (spec_cone/cam_jjj_ringkk.exr + counts, format "rings" di
          images_generator.py); candidati = specchio + un cono per apertura
          della griglia; roughness = apertura/180
    X     peso del termine DIFFUSO (X=1 → nessuna dipendenza dalla vista);
          la specularità/metallic è 1−X. Forma chiusa per ogni candidato.

Pesi per anello del cono, in forma chiusa (b = semi-aperture, c = cos b ≥ 0):
    W_i = π · (c_{i-1}² − c_i²)  per i ≤ k, 0 oltre
    L = Σ_i W_i·mean_i·valid_i / M  (integrale puro; M = samples_per_ring dal
        meta: denominatore fisso come nei pass irradiance, i raggi scartati
        sotto l'orizzonte contribuiscono 0. L scala col fattore geometrico
        π·sin²(semi-apertura) → X assorbe la scala e metallic = 1−X NON è la
        frazione speculare assoluta; il candidato specchio resta in unità di
        radianza, non di integrale.)
Nota: la risoluzione sulla larghezza del cono è quantizzata dalla griglia di
aperture del bake (spec_cone_apertures_deg).

Per ogni candidato il residuo è una parabola in X:
    res(X) = S_cc - 2·X·S_cd + X²·S_dd
    S_cc = Σ_{j,c} w_j (C-L)²,  S_cd = Σ w_j (C-L)(D-L),  S_dd = Σ w_j (D-L)²
quindi X* = clip(S_cd / S_dd, 0, 1). Il loop è streaming per camera: carica
C_j + anelli, accumula S_cc/S_cd/S_dd per tutti i candidati, libera — la RAM
non scala col numero di camere.

Gate e validità:
  - gate diffuso scale-invariant sul coefficiente di variazione tra camere:
    sqrt(var) < cv_gate · luminanza(D)  →  X=1, metallic=0, lobo non vincolato;
  - il lobo è attendibile solo dove la specularità supera spec_threshold:
    sotto, roughness viene posta a 1.0 e il texel è marcato in pbr/r_valid.png.

Mappe finali, per la sorgente `source` (chiamare una volta per ogni sorgente da
processare: le uscite vivono sotto sources/{source}/, i nomi interni non sono
suffissati):
  <out>/sources/{source}/metallic/metallic.exr      = 1−X   (0=diffuso, 1=tutto speculare)
  <out>/sources/{source}/roughness/roughness.exr    = apertura/180 del cono vincente
    (0 = specchio), 1.0 altrove — indice di larghezza del cono, NON α GGX:
    per texture Disney-compliant serve una calibrazione apertura→α a valle
  <out>/sources/{source}/albedo_pbr/albedo_pbr.exr  = π·(X·D)/max(E_sky+E_ind, albedo_eps):
    albedo diffusa depurata dallo speculare (coesiste con l'albedo Lambertiana
    classica in <out>/sources/{source}/albedo/). Richiede irradiance/irradiance.exr
    su disco (irradiance_indirect.exr sommata se presente; input condiviso, non
    per-source); se manca, la mappa è saltata con warning. Sui texel non
    risolvibili si assume X=1 (diffuso).
Diagnostica in <out>/sources/{source}/pbr/: diffuse_weight.exr (X), lobe_param.exr
(apertura del cono vincente in gradi, 0 = specchio), residual.exr,
n_views.exr, r_valid.png + preview PNG a scala assoluta.

Input condivisi (source-indipendenti, non sotto sources/{source}/): spec_cone/,
ium/ium_masks.exr, visibility/visibility.exr, irradiance/.

Uso:
    python pbr_solver.py <output_dir> [--source gt] [--cv-gate 0.05]
                         [--spec-threshold 0.2] [--min-views 2] [--albedo-eps 1e-3]
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
# Pesi per anello del cono, in forma chiusa
# ──────────────────────────────────────────────────────────────────────────────

def ring_weights_coscone(cos_edges: np.ndarray, k: int) -> np.ndarray:
    """Cono troncato all'anello k con peso cosθ_R (integrale puro, non media):
    W_i = π(c_{i-1}² − c_i²) per i ≤ k, 0 oltre, con c = max(cos b, 0).
    Σ_{i≤k} W_i = π(1 − c_k²) = π·sin²(semi-apertura_k)."""
    c = np.clip(np.asarray(cos_edges, dtype=np.float64), 0.0, 1.0)
    w = np.pi * (c[:-1] ** 2 - c[1:] ** 2)
    w[k:] = 0.0
    return w


# ──────────────────────────────────────────────────────────────────────────────
# Solver
# ──────────────────────────────────────────────────────────────────────────────

def solve_pbr(output_dir: str,
              source: str = "gt",
              cv_gate: float = 0.05,
              spec_threshold: float = 0.2,
              min_views: int = 2,
              albedo_eps: float = 1e-3,
              eps: float = 1e-12) -> dict:
    out = Path(output_dir)
    src_dir = out / "sources" / source     # artefatti source-dipendenti (camera_texture/, pixel_change/, uscite PBR)
    spec_dir = out / "spec_cone"           # condiviso (source-indipendente)

    with open(spec_dir / "spec_cone_meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    with open(out / "transforms_extended.json", encoding="utf-8") as fh:
        tjson = json.load(fh)

    if meta.get("format") != "rings":
        raise ValueError(
            "spec_cone_meta.json in formato legacy (coni cumulativi): il solver "
            "richiede il formato 'rings' (medie per-anello + counts). "
            "Ri-eseguire il precompute spec_cone.")

    apertures = np.asarray(meta["apertures_deg"], dtype=np.float32)
    cams      = meta["cameras"]
    K         = meta["num_levels"]
    stems     = [Path(f["file_path"]).stem for f in tjson["frames"]]

    # ── Candidati: [specchio] + un cono coscone per apertura ─────────────────
    # Tupla: (label, pesi per anello — None per lo specchio, roughness, param)
    cos_edges = np.asarray(meta.get("ring_edges_cos")
                           or np.cos(np.radians(apertures) * 0.5),
                           dtype=np.float64)
    candidates = [("mirror", None, 0.0, 0.0)]
    candidates += [(f"r={apertures[k]:g}°∫cos", ring_weights_coscone(cos_edges, k),
                    float(apertures[k]) / 180.0, float(apertures[k]))
                   for k in range(1, K)]
    n_cand = len(candidates)
    # L = integrale puro con denominatore fisso (campioni lanciati per anello),
    # non media pesata — vedi docstring
    M_ring = float(meta["samples_per_ring"])

    # ── Input vista-indipendenti (per la sorgente `source`) ──────────────────
    D_img = _read_exr(src_dir / "pixel_change" / "color_min.exr")          # (H, W, 3)
    var   = _read_exr(src_dir / "pixel_change" / "color_variance.exr")     # (H, W, 3)
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

    print(f"[pbr_solver] {N} texel, {len(cams)} camere, "
          f"{n_cand} candidati ({K - 1} anelli + specchio)")

    # ── Scan streaming: una camera alla volta, statistiche per candidato ─────
    S_cc = np.zeros((N, n_cand))
    S_cd = np.zeros((N, n_cand))
    S_dd = np.zeros((N, n_cand))
    n_views = np.zeros(N, dtype=np.int32)

    for j in cams:
        C_j = _read_exr(src_dir / "camera_texture" / f"{stems[j]}.exr")
        C_j = C_j.reshape(N, 3).astype(np.float64)
        w_j = mask.copy()
        if vis is not None:
            w_j &= vis[:, j]
        w_j &= _read_mask_png(spec_dir / meta["valid_file_pattern"].format(cam=j)).reshape(N)
        n_views += w_j

        means = np.stack([_read_exr(spec_dir / meta["ring_file_pattern"]
                                    .format(cam=j, level=k)).reshape(N, 3)
                          for k in range(K)], axis=1)               # (N, K, 3)
        counts = _read_exr(spec_dir / meta["counts_file_pattern"].format(cam=j))
        counts = counts.reshape(N, K).astype(np.float64)

        wf = w_j.astype(np.float64)[:, None]
        for c_idx, (_label, w_rings, _rough, _param) in enumerate(candidates):
            if w_rings is None:                    # specchio puro (livello 0)
                L = means[:, 0].astype(np.float64)
            else:                                  # cono: anelli 1..K-1 pesati
                wc  = w_rings[None, :] * counts[:, 1:]              # (N, K-1)
                num = np.einsum("nk,nkc->nc", wc,
                                means[:, 1:].astype(np.float64))
                L = num / M_ring    # integrale puro: denominatore fisso
            dC = C_j - L; dD = D - L
            S_cc[:, c_idx] += (wf * dC * dC).sum(axis=-1)
            S_cd[:, c_idx] += (wf * dC * dD).sum(axis=-1)
            S_dd[:, c_idx] += (wf * dD * dD).sum(axis=-1)
        print(f"  cam {j}: statistiche accumulate")

    solvable = mask & (n_views >= min_views)

    # Gate diffuso scale-invariant: variazione tra camere piccola rispetto
    # alla luminanza del texel → X=1, metallic=0, lobo non vincolato
    lum_D   = D.mean(axis=-1)
    lum_std = np.sqrt(np.maximum(var.reshape(N, 3).mean(axis=-1), 0.0))
    diffuse_gate = solvable & (lum_std < cv_gate * np.maximum(lum_D, 1e-6))

    print(f"  texel risolvibili: {int(solvable.sum())}, "
          f"di cui gated come diffusi (CV<{cv_gate}): {int(diffuse_gate.sum())}")

    # ── Selezione del candidato a residuo minimo ─────────────────────────────
    X_all   = np.clip(S_cd / np.maximum(S_dd, eps), 0.0, 1.0)       # (N, n_cand)
    res_all = S_cc - 2.0 * X_all * S_cd + X_all * X_all * S_dd
    res_all /= np.maximum(3.0 * n_views, 1.0)[:, None]  # residuo medio per equazione

    target   = solvable & ~diffuse_gate
    best_k   = np.argmin(res_all, axis=1).astype(np.int32)
    _idx     = np.arange(N)
    best_res = res_all[_idx, best_k]
    best_X   = X_all[_idx, best_k]

    for c_idx, (label, _w, rough, _param) in enumerate(candidates):
        n_best = int(((best_k == c_idx) & target & np.isfinite(best_res)).sum())
        print(f"  candidato {label:>9} (roughness={rough:.3f}) → migliore per "
              f"{n_best} texel")

    # ── Composizione output ───────────────────────────────────────────────────
    fitted = target & np.isfinite(best_res)

    diffuse_w = np.zeros(N, dtype=np.float32)     # X
    diffuse_w[diffuse_gate] = 1.0
    diffuse_w[fitted] = best_X[fitted].astype(np.float32)

    metallic = np.zeros(N, dtype=np.float32)      # 1−X (specularità)
    metallic[fitted] = (1.0 - best_X[fitted]).astype(np.float32)

    rough_vals = np.asarray([c[2] for c in candidates], dtype=np.float32)
    param_vals = np.asarray([c[3] for c in candidates], dtype=np.float32)

    lobe_param = np.zeros(N, dtype=np.float32)
    lobe_param[fitted] = param_vals[best_k[fitted]]

    # lobo attendibile solo con specularità sufficiente; altrove roughness=1
    r_valid = fitted & (metallic >= spec_threshold)
    roughness = np.where(mask, 1.0, 0.0).astype(np.float32)
    roughness[r_valid] = rough_vals[best_k[r_valid]]

    residual = np.zeros(N, dtype=np.float32)
    residual[fitted] = best_res[fitted].astype(np.float32)

    print(f"  texel con r attendibile (metallic≥{spec_threshold}): "
          f"{int(r_valid.sum())}")

    # ── Mappe finali (cartelle dedicate, come l'albedo) ───────────────────────
    fmt = ImageFormat.OPENEXR
    met_dir = src_dir / "metallic";  met_dir.mkdir(parents=True, exist_ok=True)
    rgh_dir = src_dir / "roughness"; rgh_dir.mkdir(parents=True, exist_ok=True)
    metallic_path  = (met_dir / "metallic.exr").resolve().as_posix()
    roughness_path = (rgh_dir / "roughness.exr").resolve().as_posix()
    _save_layer(metallic.reshape(H, W), metallic_path, fmt, DataLayer.METALLIC)
    _save_layer(roughness.reshape(H, W), roughness_path, fmt, DataLayer.ROUGHNESS)

    # ── Diagnostica ───────────────────────────────────────────────────────────
    pbr_dir = src_dir / "pbr"
    pbr_dir.mkdir(parents=True, exist_ok=True)
    _save_layer(diffuse_w.reshape(H, W), (pbr_dir / "diffuse_weight.exr").as_posix(),
                fmt, DataLayer.METALLIC)
    _save_layer(lobe_param.reshape(H, W), (pbr_dir / "lobe_param.exr").as_posix(),
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

    # ── Albedo PBR: π·(X·D)/max(E_sky+E_ind, albedo_eps) ─────────────────────
    # Numeratore = radianza diffusa stimata dal fit (vista-indipendente, depurata
    # dallo speculare); coesiste con l'albedo Lambertiana classica in <out>/albedo/.
    albedo_pbr_path = None
    albedo_pbr = None
    irr_path = out / "irradiance" / "irradiance.exr"
    if irr_path.exists():
        E = _read_exr(irr_path).reshape(N, 3).astype(np.float64)
        ind_path = out / "irradiance" / "irradiance_indirect.exr"
        if ind_path.exists():
            E += _read_exr(ind_path).reshape(N, 3).astype(np.float64)
        denom = np.maximum(E, albedo_eps)

        # X completo: dove il fit non è possibile si assume diffuso (X=1),
        # come roughness=1 altrove, così la mappa è definita su tutta la mask
        X_full = diffuse_w.astype(np.float64)
        X_full[mask & ~fitted & ~diffuse_gate] = 1.0

        alb_flat = np.clip(np.pi * X_full[:, None] * D / denom, 0.0, 1.0)
        alb_flat[~mask] = 0.0
        albedo_pbr = alb_flat.reshape(H, W, 3).astype(np.float32)

        alb_dir = src_dir / "albedo_pbr"; alb_dir.mkdir(parents=True, exist_ok=True)
        albedo_pbr_path = (alb_dir / "albedo_pbr.exr").resolve().as_posix()
        _save_layer(albedo_pbr, albedo_pbr_path, fmt, DataLayer.ALBEDO)
        Image.fromarray((albedo_pbr * 255).astype(np.uint8)
                        ).save(pbr_dir / "albedo_pbr_preview.png")
        print(f"✓ albedo_pbr: {albedo_pbr_path} (indirect: "
              f"{'sì' if ind_path.exists() else 'no'})")
    else:
        print(f"    ⚠  albedo_pbr saltata: {irr_path} non trovata "
              "(serve il pass irradiance)")

    print(f"✓ metallic:  {metallic_path}")
    print(f"✓ roughness: {roughness_path}")
    print(f"✓ diagnostica in {pbr_dir}")
    return {
        "metallic_path": metallic_path,
        "roughness_path": roughness_path,
        "albedo_pbr_path": albedo_pbr_path,
        "albedo_pbr": albedo_pbr,
        "metallic": metallic.reshape(H, W),
        "roughness": roughness.reshape(H, W),
        "diffuse_weight": diffuse_w.reshape(H, W),
        "lobe_param": lobe_param.reshape(H, W),
        "residual": residual.reshape(H, W),
        "n_views": n_views.reshape(H, W),
        "r_valid": r_valid.reshape(H, W),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Fit PBR C_j = X·D + (1-X)·L_j(r) → metallic/roughness")
    ap.add_argument("output_dir", help="output_dir del pipeline (contiene spec_cone/, ...)")
    ap.add_argument("--source", type=str, default="gt",
                    help="sorgente da processare (sources/{source}/, es. gt o nerf)")
    ap.add_argument("--cv-gate", type=float, default=0.05,
                    help="gate diffuso: std tra camere < cv_gate·luminanza → metallic=0")
    ap.add_argument("--spec-threshold", type=float, default=0.2,
                    help="metallic minimo perché r sia attendibile (sotto: roughness=1)")
    ap.add_argument("--min-views", type=int, default=2,
                    help="minimo di camere valide per texel")
    ap.add_argument("--albedo-eps", type=float, default=1e-3,
                    help="clamp minimo dell'irradiance nell'albedo_pbr")
    args = ap.parse_args()
    solve_pbr(args.output_dir, source=args.source,
              cv_gate=args.cv_gate, spec_threshold=args.spec_threshold,
              min_views=args.min_views, albedo_eps=args.albedo_eps)
