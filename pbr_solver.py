"""Fit PBR multi-vista con coni speculari a taglio netto (media pura sul cono):

    C_j = (a·x/π) · E + (1 − x) · L_j        (per canale c)

    C_j   colore osservato dalla camera j    (sources/{source}/camera_texture/<stem>.exr)
    E     irradianza emisferica vista-indipendente (irradiance/irradiance.exr
          + irradiance_indirect.exr se presente; input condiviso, non per-source).
          Serve SOLO per l'albedo: metallic e roughness non la richiedono.
    a     albedo diffusa (incognita, RGB, in [0,1])
    x     peso del termine DIFFUSO (x=1 → nessuna dipendenza dalla vista);
          la specularità/metallic è 1−x
    L_j   MEDIA PURA ad angolo solido della radianza ambiente sul cono attorno
          al raggio riflesso (nessun peso cosθ_R). Il solver NON la ricostruisce:
          la legge dal bake, che scrive un canale RGB per candidato in
          spec_cone/cam_{j:03d}.exr (format "cones" di images_generator.py).
          Candidati = specchio + un cono per apertura della griglia;
          roughness = apertura/180. Essendo una media, L resta in unità di
          radianza a OGNI apertura (specchio compreso): metallic = 1−x è una
          frazione speculare omogenea tra i candidati.

La media sul cono la chiude il bake (images_generator._cones_from_rings_*):
qui basta sapere che il canale di un livello (cone_045deg, dal nome si legge
l'apertura) è la media dei soli raggi validi entro
l'apertura k, pesata per angolo solido, e che `valid` = raggi sopra l'orizzonte
del texel (0 → il texel non è utilizzabile per quella camera).
Nota: la risoluzione sulla larghezza del cono è quantizzata dalla griglia di
aperture del bake (spec_cone_apertures_deg) — scelta deliberata, niente
raffinamento sub-griglia in questa versione.

Per ogni candidato r il modello è una regressione lineare per texel:
    C_jc = α_c + β·L_jc        α_c = x·a_c·E_c/π  (intercetta, per canale)
                               β   = 1−x          (pendenza, condivisa sui canali)
Il termine diffuso è identico per tutte le viste, quindi ogni variazione di C
tra le camere è attribuibile solo a L: la centratura sulle medie tra camere
elimina α (equivale a lavorare sulle differenze C_i − C_j) e dà β in forma
chiusa. Statistiche sufficienti, accumulate in streaming camera per camera:
    Sw = Σ w_j     SC_c = Σ w_j·C     SCC = Σ_{j,c} w_j·C²
    SL_c(r) = Σ w_j·L     SLL(r) = Σ_{j,c} w_j·L²     SCL(r) = Σ_{j,c} w_j·C·L
    VLL = SLL − Σ_c SL²/Sw    VCL = SCL − Σ_c SC·SL/Sw    VCC = SCC − Σ_c SC²/Sw
    β*(r) = clip(VCL / VLL, 0, 1)
    res(r) = (VCC − 2β·VCL + β²·VLL) / (3·n_views)   →   argmin su r
Dal candidato vincente: x = 1−β,  α_c = (SC_c − β·SL_c)/Sw (≥ 0),
    a_c = clip(π·α_c / (max(E_c, albedo_eps)·x), 0, 1);  x < X_EPS → a = 0
(convenzione metallo: un texel completamente speculare non ha albedo diffusa).
La RAM non scala col numero di camere (loop streaming, come sempre).

Gate e validità:
  - gate diffuso scale-invariant sul coefficiente di variazione tra camere
    (da pixel_change/, usato SOLO per il gate — color_min non entra più nel
    fit): sqrt(var) < cv_gate · luminanza(color_min) → x=1, metallic=0;
  - il lobo è attendibile solo dove la specularità supera spec_threshold:
    sotto, roughness viene posta a 1.0 e il texel è marcato in pbr/r_valid.png.

Mappe finali, per la sorgente `source` (chiamare una volta per ogni sorgente da
processare: le uscite vivono sotto sources/{source}/, i nomi interni non sono
suffissati):
  <out>/sources/{source}/metallic/metallic.exr      = 1−x   (0=diffuso, 1=tutto speculare)
  <out>/sources/{source}/roughness/roughness.exr    = apertura/180 del cono vincente
    (0 = specchio), 1.0 altrove — indice di larghezza del cono, NON α GGX:
    per texture Disney-compliant serve una calibrazione apertura→α a valle
  <out>/sources/{source}/albedo_pbr/albedo_pbr.exr  = a dal fit (albedo diffusa
    depurata dallo speculare per-vista; coesiste con l'albedo Lambertiana
    classica in <out>/sources/{source}/albedo/). Richiede irradiance/irradiance.exr
    su disco (irradiance_indirect.exr sommata se presente); se manca, la mappa
    è saltata con warning. Sui texel non risolvibili si assume x=1 (diffuso,
    α = media tra camere → a ≡ albedo Lambertiana classica).
Diagnostica in <out>/sources/{source}/pbr/: diffuse_weight.exr (x),
diffuse_term.exr (α, radianza diffusa stimata), lobe_param.exr (apertura del
cono vincente in gradi, 0 = specchio), residual.exr, n_views.exr,
r_valid.png + preview PNG a scala assoluta.

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
    DataLayer, ImageFormat, _save_layer, spec_cone_level_name,
)

# Sotto questo x il texel è considerato completamente speculare: l'albedo
# diffusa non è definita (α/x → 0/0) e viene scritta a 0 (convenzione metallo).
X_EPS = 1e-3


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


def read_cones(path: Path, apertures_deg) -> "tuple[np.ndarray, np.ndarray]":
    """EXR dei coni di una camera → (L (N, K, 3), n_valid (N,)).

    Un canale RGB per livello, con la media pura sul cono già chiusa dal bake,
    più `valid`, il numero di raggi validi del texel. I nomi dei livelli portano
    l'apertura (`cone_045deg`) e vengono generati da spec_cone_level_name a
    partire dalle aperture del meta: writer e reader usano la stessa funzione,
    quindi non possono divergere. Un solo file per camera e non uno per
    apertura: il bake condiviso ha il loop sui tile all'esterno, quindi i writer
    di tutte le camere restano aperti insieme e K+1 file per camera
    supererebbero il limite stdio di MSVC.
    """
    import OpenEXR, Imath

    exr = OpenEXR.InputFile(path.as_posix())
    header = exr.header()
    dw = header["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)   # half convertito in lettura
    names = set(header["channels"].keys())

    def chan(name):
        if name not in names:
            raise ValueError(f"{path}: canale {name!r} mancante "
                             f"(bake incompleto o num_levels sbagliato nel meta)")
        return np.frombuffer(exr.channel(name, pt), dtype=np.float32).reshape(h * w)

    cones = np.stack(
        [np.stack([chan(f"{spec_cone_level_name(apertures_deg, k)}.{c}") for c in "RGB"],
                  axis=-1)
         for k in range(len(apertures_deg))], axis=1)      # (N, K, 3)
    return cones, chan("valid")


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

    # Il bake scrive i coni già chiusi (un canale RGB per candidato), quindi qui
    # non si ricostruisce nulla: L_j(r) si legge e basta. I bake in formato
    # "rings"/"rings_shared" (medie per anello) non sono più supportati.
    if meta.get("format") != "cones":
        raise ValueError(
            f"spec_cone_meta.json in formato {meta.get('format')!r}: il solver "
            f"richiede il formato 'cones' (L_j(r) già chiusa dal bake). "
            f"Ri-eseguire il precompute spec_cone.")

    apertures = np.asarray(meta["apertures_deg"], dtype=np.float32)
    cams      = meta["cameras"]
    K         = meta["num_levels"]
    stems     = [Path(f["file_path"]).stem for f in tjson["frames"]]

    # ── Candidati: [specchio] + un cono (media pura) per apertura ────────────
    # Sono i K canali del bake nell'ordine in cui li ha scritti: l'indice del
    # candidato È l'indice del canale. Tupla: (label, roughness, param).
    candidates = [("mirror", 0.0, 0.0)]
    candidates += [(f"r={apertures[k]:g}°mean",
                    float(apertures[k]) / 180.0, float(apertures[k]))
                   for k in range(1, K)]
    n_cand = len(candidates)

    # ── Input per il gate diffuso (color_min NON entra più nel fit) ──────────
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
          f"{n_cand} candidati ({K - 1} coni-media + specchio)")

    # ── Scan streaming: una camera alla volta, statistiche sufficienti ───────
    SC  = np.zeros((N, 3))                 # Σ w·C          (indip. dal candidato)
    SCC = np.zeros(N)                      # Σ w·C² pooled  (indip. dal candidato)
    SL  = np.zeros((N, n_cand, 3))         # Σ w·L          per candidato
    SLL = np.zeros((N, n_cand))            # Σ w·L² pooled  per candidato
    SCL = np.zeros((N, n_cand))            # Σ w·C·L pooled per candidato
    n_views = np.zeros(N, dtype=np.int32)

    for j in cams:
        C_j = _read_exr(src_dir / "camera_texture" / f"{stems[j]}.exr")
        C_j = C_j.reshape(N, 3).astype(np.float64)
        w_j = mask.copy()
        if vis is not None:
            w_j &= vis[:, j]

        cones, n_valid = read_cones(
            spec_dir / meta["cam_file_pattern"].format(cam=j), apertures)
        cones = cones.astype(np.float64)                    # (N, n_cand, 3)
        # Un texel è valido per questa camera se almeno un raggio è finito sopra
        # l'orizzonte: è la stessa maschera che il bake usa per azzerare i coni.
        w_j &= n_valid > 0

        n_views += w_j

        wf = w_j.astype(np.float64)
        SC  += wf[:, None] * C_j
        SCC += wf * (C_j * C_j).sum(axis=-1)
        for c_idx in range(n_cand):
            L = cones[:, c_idx]
            SL[:, c_idx]  += wf[:, None] * L
            SLL[:, c_idx] += wf * (L * L).sum(axis=-1)
            SCL[:, c_idx] += wf * (C_j * L).sum(axis=-1)
        print(f"  cam {j}: statistiche accumulate")

    solvable = mask & (n_views >= min_views)

    # Gate diffuso scale-invariant: variazione tra camere piccola rispetto
    # alla luminanza del texel → x=1, metallic=0, lobo non vincolato
    lum_D   = D.mean(axis=-1)
    lum_std = np.sqrt(np.maximum(var.reshape(N, 3).mean(axis=-1), 0.0))
    diffuse_gate = solvable & (lum_std < cv_gate * np.maximum(lum_D, 1e-6))

    print(f"  texel risolvibili: {int(solvable.sum())}, "
          f"di cui gated come diffusi (CV<{cv_gate}): {int(diffuse_gate.sum())}")

    # ── Fit per candidato: regressione centrata in forma chiusa ─────────────
    Sw  = np.maximum(n_views.astype(np.float64), 1.0)
    VLL = np.maximum(SLL - (SL ** 2).sum(axis=-1) / Sw[:, None], 0.0)
    VCL = SCL - np.einsum("nc,nkc->nk", SC, SL) / Sw[:, None]
    VCC = np.maximum(SCC - (SC ** 2).sum(axis=-1) / Sw, 0.0)

    beta_all = np.clip(VCL / np.maximum(VLL, eps), 0.0, 1.0)        # β = 1−x
    res_all  = VCC[:, None] - 2.0 * beta_all * VCL + beta_all ** 2 * VLL
    res_all /= np.maximum(3.0 * n_views, 1.0)[:, None]  # residuo medio per equazione

    target    = solvable & ~diffuse_gate
    best_k    = np.argmin(res_all, axis=1).astype(np.int32)
    _idx      = np.arange(N)
    best_res  = res_all[_idx, best_k]
    best_beta = beta_all[_idx, best_k]

    for c_idx, (label, rough, _param) in enumerate(candidates):
        n_best = int(((best_k == c_idx) & target & np.isfinite(best_res)).sum())
        print(f"  candidato {label:>11} (roughness={rough:.3f}) → migliore per "
              f"{n_best} texel")

    # ── Composizione output ───────────────────────────────────────────────────
    fitted = target & np.isfinite(best_res)

    diffuse_w = np.zeros(N, dtype=np.float32)     # x
    diffuse_w[diffuse_gate] = 1.0
    diffuse_w[fitted] = (1.0 - best_beta[fitted]).astype(np.float32)

    metallic = np.zeros(N, dtype=np.float32)      # β = 1−x (specularità)
    metallic[fitted] = best_beta[fitted].astype(np.float32)

    rough_vals = np.asarray([c[1] for c in candidates], dtype=np.float32)
    param_vals = np.asarray([c[2] for c in candidates], dtype=np.float32)

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

    # ── Intercetta α = x·a·E/π (radianza diffusa stimata) ────────────────────
    # x e α completi su tutta la mask: texel gated/non fittabili → diffusi
    # (x=1, α = media di C tra le camere ⇒ a ≡ albedo Lambertiana classica)
    C_bar  = SC / Sw[:, None]
    X_full = np.ones(N)
    X_full[fitted] = 1.0 - best_beta[fitted]
    SL_best = SL[_idx, best_k]                                       # (N, 3)
    alpha = C_bar.copy()
    alpha[fitted] = ((SC[fitted] - best_beta[fitted, None] * SL_best[fitted])
                     / Sw[fitted, None])
    alpha = np.maximum(alpha, 0.0)
    alpha[~mask] = 0.0

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
    _save_layer(alpha.reshape(H, W, 3).astype(np.float32),
                (pbr_dir / "diffuse_term.exr").as_posix(), fmt, DataLayer.ALBEDO)
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

    # ── Albedo PBR: a = π·α / (max(E_sky+E_ind, albedo_eps)·x) ───────────────
    # L'albedo esce direttamente dal fit, già depurata dallo speculare per-vista;
    # coesiste con l'albedo Lambertiana classica in <out>/sources/{source}/albedo/.
    albedo_pbr_path = None
    albedo_pbr = None
    irr_path = out / "irradiance" / "irradiance.exr"
    if irr_path.exists():
        E = _read_exr(irr_path).reshape(N, 3).astype(np.float64)
        ind_path = out / "irradiance" / "irradiance_indirect.exr"
        if ind_path.exists():
            E += _read_exr(ind_path).reshape(N, 3).astype(np.float64)
        denom = np.maximum(E, albedo_eps)

        x_col = np.maximum(X_full, X_EPS)[:, None]
        alb_flat = np.clip(np.pi * alpha / (denom * x_col), 0.0, 1.0)
        alb_flat[X_full < X_EPS] = 0.0    # completamente speculare: albedo nulla
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
        "diffuse_term": alpha.reshape(H, W, 3),
        "lobe_param": lobe_param.reshape(H, W),
        "residual": residual.reshape(H, W),
        "n_views": n_views.reshape(H, W),
        "r_valid": r_valid.reshape(H, W),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Fit PBR C_j = (a·x/π)·E + (1-x)·L_j(r) → "
                    "metallic/roughness/albedo_pbr")
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
