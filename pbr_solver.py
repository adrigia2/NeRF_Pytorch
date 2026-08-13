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

La RAM non scala né col numero di camere né con la risoluzione della texture: il
loop esterno è sulle BANDE di texel (blocchi di scanline intere, `tile_texels`),
quello interno sulle camere. Gli accumulatori vivono solo per la banda corrente e
il fit — che è puramente per-texel, nessuna operazione mette in relazione texel
diversi — viene chiuso banda per banda; a piena risoluzione restano solo le mappe
di uscita, in float32. Ogni lettura EXR è una `channels()` su un intervallo di
scanline: una sola decompressione per banda invece di una per canale (a 4096²
con 14 aperture il file dei coni ha 43 canali, quindi 43 decompressioni
dell'intero file per camera).

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
metallic e roughness vengono scritte anche come metallic_rgb.exr /
roughness_rgb.exr accanto agli originali (stessi valori su tre canali R/G/B
float32, convenzione dei bake di Blender) salvo blender_rgb=False.
Diagnostica in <out>/sources/{source}/pbr/: diffuse_weight.exr (x),
diffuse_term.exr (α, radianza diffusa stimata), lobe_param.exr (apertura del
cono vincente in gradi, 0 = specchio), residual.exr, n_views.exr,
r_valid.png + preview PNG a scala assoluta.

Input condivisi (source-indipendenti, non sotto sources/{source}/): spec_cone/,
ium/ium_masks.exr, visibility/visibility.exr, irradiance/.

Uso:
    python pbr_solver.py <output_dir> [--source gt] [--cv-gate 0.05]
                         [--spec-threshold 0.2] [--min-views 2] [--albedo-eps 1e-3]
                         [--tile-texels 1048576] [--no-blender-rgb]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from images_generator import (  # noqa: E402
    DataLayer, ImageFormat, _save_layer, _tile_bar, spec_cone_level_name,
)
from exr_to_blender_rgb import write_blender_rgb  # noqa: E402

# Sotto questo x il texel è considerato completamente speculare: l'albedo
# diffusa non è definita (α/x → 0/0) e viene scritta a 0 (convenzione metallo).
X_EPS = 1e-3


# ──────────────────────────────────────────────────────────────────────────────
# IO helpers
# ──────────────────────────────────────────────────────────────────────────────

def _channel_order(names: "list[str]") -> "tuple[list[str], bool]":
    """Ordine canonico dei canali di un EXR → (nomi, canale_singolo).

    Regola condivisa da _read_exr e _ExrBandReader: la colonna j di un array a
    piena immagine e quella di una banda devono riferirsi allo stesso canale.
    Non è banale perché ExrWriter (images_generator.py:100-106) nomina i canali
    in base al loro numero: 1 → Z, 3 → R,G,B, 4 → R,G,B,A e solo da 5 in su
    Cam0, Cam1, … — quindi una visibility con 3 o 4 camere NON ha canali Cam*.
    """
    names = list(names)
    if names == ["Z"] or names == ["Y"]:
        return names, True
    if set("RGB").issubset(names):
        return ["R", "G", "B"] + (["A"] if "A" in names else []), False
    return sorted(names, key=lambda n: (len(n), n)), False   # Cam0, Cam1, … in ordine numerico


class _ExrBandReader:
    """EXR letto per blocchi di scanline consecutive: la controparte in lettura
    di images_generator.IncrementalExrWriter.

    Serve al solver, che ha il loop sulle bande di texel all'esterno e quello
    sulle camere all'interno: a piena risoluzione i coni di una camera sono già
    2.6 GiB e gli accumulatori quasi 10, mentre una banda costa poche decine di MB.

    Una `channels()` sola per blocco e non una `channel()` per canale: il
    framebuffer viene montato una volta e i blocchi ZIP si decomprimono una
    volta sola. Con i 43 canali del file dei coni la differenza misurata è ~34x.
    """

    def __init__(self, path: Path):
        import OpenEXR, Imath

        self.path = Path(path)
        self._exr = OpenEXR.InputFile(self.path.as_posix())
        header = self._exr.header()
        dw = header["dataWindow"]
        self.width  = dw.max.x - dw.min.x + 1
        self.height = dw.max.y - dw.min.y + 1
        self._pt    = Imath.PixelType(Imath.PixelType.FLOAT)  # half convertito in lettura
        self._avail = set(header["channels"].keys())
        self.names, self.single = _channel_order(list(header["channels"].keys()))

    # ── lettura ──────────────────────────────────────────────────────────────
    def read_raw(self, y0: int, rows: "int | None",
                 names: "list[str]") -> "list[bytes]":
        """Buffer grezzi dei canali richiesti, nell'ordine richiesto."""
        rows = self.height - y0 if rows is None else rows
        if y0 < 0 or rows <= 0 or y0 + rows > self.height:
            raise ValueError(f"{self.path}: banda [{y0}, {y0 + rows}) fuori "
                             f"dalle {self.height} scanline del file")
        missing = [n for n in names if n not in self._avail]
        if missing:
            raise ValueError(f"{self.path}: canali mancanti {missing} "
                             f"(disponibili: {sorted(self._avail)[:8]}…)")
        return self._exr.channels(names, self._pt, y0, y0 + rows - 1)

    def read(self, y0: int = 0, rows: "int | None" = None,
             names=None) -> np.ndarray:
        """Banda → (rows·width, C) float32, oppure (rows·width,) se `names` è una
        stringa o se il file ha un canale singolo (Z/Y) e `names` è None."""
        squeeze = False
        if names is None:
            names, squeeze = self.names, self.single
        elif isinstance(names, str):
            names, squeeze = [names], True
        else:
            names = list(names)

        bufs = self.read_raw(y0, rows, names)
        n = (self.height - y0 if rows is None else rows) * self.width
        if squeeze:
            return np.frombuffer(bufs[0], dtype=np.float32).copy()   # scrivibile
        out = np.empty((n, len(names)), dtype=np.float32)
        for i, buf in enumerate(bufs):
            out[:, i] = np.frombuffer(buf, dtype=np.float32)
        return out

    def close(self) -> None:
        if self._exr is not None:
            self._exr.close()
            self._exr = None

    def __enter__(self): return self

    def __exit__(self, exc_type, exc, tb): self.close()


def _read_band(path: Path, y0: int, rows: "int | None" = None, names=None) -> np.ndarray:
    """Una banda di scanline da `path` (apre, legge, chiude)."""
    with _ExrBandReader(path) as rd:
        return rd.read(y0, rows, names)


def _read_exr(path: Path) -> np.ndarray:
    """EXR → (H, W) float32 [canale Z] oppure (H, W, C) [R,G,B(,A) o Cam*]."""
    with _ExrBandReader(path) as rd:
        arr = rd.read()
        return arr.reshape(rd.height, rd.width) if arr.ndim == 1 else \
            arr.reshape(rd.height, rd.width, -1)


def read_cones(path: Path, apertures_deg, y0: int = 0,
               rows: "int | None" = None) -> "tuple[np.ndarray, np.ndarray]":
    """EXR dei coni di una camera → (L (n, K, 3), n_valid (n,)).

    Un canale RGB per livello, con la media pura sul cono già chiusa dal bake,
    più `valid`, il numero di raggi validi del texel. I nomi dei livelli portano
    l'apertura (`cone_045deg`) e vengono generati da spec_cone_level_name a
    partire dalle aperture del meta: writer e reader usano la stessa funzione,
    quindi non possono divergere. Un solo file per camera e non uno per
    apertura: il bake condiviso ha il loop sui tile all'esterno, quindi i writer
    di tutte le camere restano aperti insieme e K+1 file per camera
    supererebbero il limite stdio di MSVC.

    Con y0/rows si legge solo una banda di scanline (n = rows·W); il default
    resta l'immagine intera (n = N).
    """
    K = len(apertures_deg)
    want = [f"{spec_cone_level_name(apertures_deg, k)}.{c}"
            for k in range(K) for c in "RGB"] + ["valid"]

    with _ExrBandReader(path) as rd:
        try:
            bufs = rd.read_raw(y0, rows, want)
        except ValueError as err:
            raise ValueError(f"{err} — bake incompleto o num_levels sbagliato "
                             f"nel meta") from None
        n = (rd.height - y0 if rows is None else rows) * rd.width
        # Riempita canale per canale: uno np.stack annidato terrebbe in vita le
        # K fette (n,3) *e* il risultato, raddoppiando il picco a immagine piena.
        cones = np.empty((n, K, 3), dtype=np.float32)
        for k in range(K):
            for ci in range(3):
                cones[:, k, ci] = np.frombuffer(bufs[3 * k + ci], dtype=np.float32)
        return cones, np.frombuffer(bufs[-1], dtype=np.float32).copy()


# ──────────────────────────────────────────────────────────────────────────────
# Solver
# ──────────────────────────────────────────────────────────────────────────────

def solve_pbr(output_dir: str,
              source: str = "gt",
              cv_gate: float = 0.05,
              spec_threshold: float = 0.2,
              min_views: int = 2,
              albedo_eps: float = 1e-3,
              blender_rgb: bool = True,
              tile_texels: int = 1 << 20,
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

    rough_vals = np.asarray([c[1] for c in candidates], dtype=np.float32)
    param_vals = np.asarray([c[2] for c in candidates], dtype=np.float32)

    # ── Input: solo geometria dagli header, i pixel si leggono per banda ─────
    cmin_path = src_dir / "pixel_change" / "color_min.exr"
    cvar_path = src_dir / "pixel_change" / "color_variance.exr"
    with _ExrBandReader(cmin_path) as rd:
        H, W = rd.height, rd.width
    N = H * W

    mask_path = out / "ium" / "ium_masks.exr"
    has_mask  = mask_path.exists()

    vis_path = out / "visibility" / "visibility.exr"
    has_vis  = vis_path.exists()
    if has_vis:
        # La colonna j dell'array che _read_exr costruirebbe è la camera j:
        # l'ordine canonico dei canali dà il nome corrispondente, e leggere per
        # nome evita di decomprimere i canali delle camere che non servono.
        with _ExrBandReader(vis_path) as rd:
            vis_cols = [rd.names[j] for j in cams]

    # Serve solo all'albedo: il controllo sta fuori dal loop così le bande non
    # allocano nulla quando l'irradiance manca.
    irr_path = out / "irradiance" / "irradiance.exr"
    ind_path = out / "irradiance" / "irradiance_indirect.exr"
    has_irr  = irr_path.exists()
    has_ind  = has_irr and ind_path.exists()

    cam_paths  = [src_dir / "camera_texture" / f"{stems[j]}.exr" for j in cams]
    cone_paths = [spec_dir / meta["cam_file_pattern"].format(cam=j) for j in cams]

    # ── Partizione in bande di scanline intere ───────────────────────────────
    # Il fit è puramente per-texel, quindi partizionare la texture non cambia il
    # risultato di un bit; cambia solo il picco di RAM, che diventa
    # ~tile_texels·n_cand·8B·20 invece di N·n_cand·8B·20.
    rows = max(1, int(tile_texels) // W)
    if rows >= 16:
        rows = (rows // 16) * 16      # allineate al blocco ZIP: niente decompressioni doppie
    rows = min(rows, H)
    n_tiles = (H + rows - 1) // rows

    print(f"[pbr_solver] {N} texel, {len(cams)} camere, "
          f"{n_cand} candidati ({K - 1} coni-media + specchio)")
    print(f"             {n_tiles} bande da {rows} scanline ({rows * W} texel)")

    # ── Uscite a piena risoluzione (float32/int32/bool: ~870 MiB a 4096²) ────
    n_views    = np.zeros(N, dtype=np.int32)
    diffuse_w  = np.zeros(N, dtype=np.float32)     # x
    metallic   = np.zeros(N, dtype=np.float32)     # β = 1−x (specularità)
    lobe_param = np.zeros(N, dtype=np.float32)
    roughness  = np.zeros(N, dtype=np.float32)
    residual   = np.zeros(N, dtype=np.float32)
    r_valid    = np.zeros(N, dtype=bool)
    alpha      = np.zeros((N, 3), dtype=np.float32)
    albedo_flat = np.zeros((N, 3), dtype=np.float32) if has_irr else None

    tot_solvable = tot_gated = tot_rvalid = 0
    best_counts  = np.zeros(n_cand, dtype=np.int64)

    bar = _tile_bar(n_tiles, f"Fit PBR ({source})")
    for y0 in range(0, H, rows):
        r   = min(rows, H - y0)
        off = y0 * W
        T   = r * W
        sl  = slice(off, off + T)

        mask_t = (_read_band(mask_path, y0, r) > 0.5) if has_mask \
            else np.ones(T, dtype=bool)

        # Banda interamente fuori dalla maschera IUM: senza texel validi ogni
        # uscita della banda varrebbe zero, e gli array sono già zero-inizializzati,
        # quindi saltarla è bit-identico. Evita le due letture di pixel_change e
        # soprattutto il loop sulle camere, che è la parte cara (una banda di
        # camera_texture + una di cam_XXX.exr per camera). Vale sempre — l'atlante
        # UV ha zone vuote — ma è ciò che rende quasi gratis una ricostruzione
        # ristretta a una ROI.
        if not mask_t.any():
            bar.update(1)
            continue

        D_t   = _read_band(cmin_path, y0, r).astype(np.float64)   # (T, 3)
        var_t = _read_band(cvar_path, y0, r)                      # (T, 3) f32
        # Una sola lettura per banda invece di una per camera.
        vis_t = (_read_band(vis_path, y0, r, vis_cols) > 0.5) if has_vis else None

        # ── Scan streaming: una camera alla volta, statistiche sufficienti ───
        SC  = np.zeros((T, 3))             # Σ w·C          (indip. dal candidato)
        SCC = np.zeros(T)                  # Σ w·C² pooled  (indip. dal candidato)
        SL  = np.zeros((T, n_cand, 3))     # Σ w·L          per candidato
        SLL = np.zeros((T, n_cand))        # Σ w·L² pooled  per candidato
        SCL = np.zeros((T, n_cand))        # Σ w·C·L pooled per candidato
        nv_t = np.zeros(T, dtype=np.int32)

        for jj in range(len(cams)):
            C_j = _read_band(cam_paths[jj], y0, r).astype(np.float64)   # (T, 3)
            w_j = mask_t.copy()
            if vis_t is not None:
                w_j &= vis_t[:, jj]

            cones, n_valid = read_cones(cone_paths[jj], apertures, y0, r)
            cones = cones.astype(np.float64)                # (T, n_cand, 3)
            # Un texel è valido per questa camera se almeno un raggio è finito sopra
            # l'orizzonte: è la stessa maschera che il bake usa per azzerare i coni.
            w_j &= n_valid > 0

            nv_t += w_j

            wf = w_j.astype(np.float64)
            SC  += wf[:, None] * C_j
            SCC += wf * (C_j * C_j).sum(axis=-1)
            for c_idx in range(n_cand):
                L = cones[:, c_idx]
                SL[:, c_idx]  += wf[:, None] * L
                SLL[:, c_idx] += wf * (L * L).sum(axis=-1)
                SCL[:, c_idx] += wf * (C_j * L).sum(axis=-1)

        solvable = mask_t & (nv_t >= min_views)

        # Gate diffuso scale-invariant: variazione tra camere piccola rispetto
        # alla luminanza del texel → x=1, metallic=0, lobo non vincolato
        lum_D   = D_t.mean(axis=-1)
        lum_std = np.sqrt(np.maximum(var_t.mean(axis=-1), 0.0))
        diffuse_gate = solvable & (lum_std < cv_gate * np.maximum(lum_D, 1e-6))

        # ── Fit per candidato: regressione centrata in forma chiusa ─────────
        Sw  = np.maximum(nv_t.astype(np.float64), 1.0)
        VLL = np.maximum(SLL - (SL ** 2).sum(axis=-1) / Sw[:, None], 0.0)
        VCL = SCL - np.einsum("nc,nkc->nk", SC, SL) / Sw[:, None]
        VCC = np.maximum(SCC - (SC ** 2).sum(axis=-1) / Sw, 0.0)

        beta_all = np.clip(VCL / np.maximum(VLL, eps), 0.0, 1.0)    # β = 1−x
        res_all  = VCC[:, None] - 2.0 * beta_all * VCL + beta_all ** 2 * VLL
        res_all /= np.maximum(3.0 * nv_t, 1.0)[:, None]  # residuo medio per equazione

        target    = solvable & ~diffuse_gate
        best_k    = np.argmin(res_all, axis=1).astype(np.int32)
        _idx      = np.arange(T)
        best_res  = res_all[_idx, best_k]
        best_beta = beta_all[_idx, best_k]

        # ── Composizione output della banda ─────────────────────────────────
        fitted = target & np.isfinite(best_res)

        dw_t = np.zeros(T, dtype=np.float32)
        dw_t[diffuse_gate] = 1.0
        dw_t[fitted] = (1.0 - best_beta[fitted]).astype(np.float32)

        met_t = np.zeros(T, dtype=np.float32)
        met_t[fitted] = best_beta[fitted].astype(np.float32)

        lobe_t = np.zeros(T, dtype=np.float32)
        lobe_t[fitted] = param_vals[best_k[fitted]]

        # lobo attendibile solo con specularità sufficiente; altrove roughness=1
        rval_t = fitted & (met_t >= spec_threshold)
        rgh_t  = np.where(mask_t, 1.0, 0.0).astype(np.float32)
        rgh_t[rval_t] = rough_vals[best_k[rval_t]]

        res_t = np.zeros(T, dtype=np.float32)
        res_t[fitted] = best_res[fitted].astype(np.float32)

        # ── Intercetta α = x·a·E/π (radianza diffusa stimata) ───────────────
        # x e α completi su tutta la mask: texel gated/non fittabili → diffusi
        # (x=1, α = media di C tra le camere ⇒ a ≡ albedo Lambertiana classica)
        C_bar = SC / Sw[:, None]
        X_t   = np.ones(T)
        X_t[fitted] = 1.0 - best_beta[fitted]
        SL_best = SL[_idx, best_k]                                   # (T, 3)
        alpha_t = C_bar.copy()
        alpha_t[fitted] = ((SC[fitted] - best_beta[fitted, None] * SL_best[fitted])
                           / Sw[fitted, None])
        alpha_t = np.maximum(alpha_t, 0.0)
        alpha_t[~mask_t] = 0.0

        n_views[sl]    = nv_t
        diffuse_w[sl]  = dw_t
        metallic[sl]   = met_t
        lobe_param[sl] = lobe_t
        roughness[sl]  = rgh_t
        residual[sl]   = res_t
        r_valid[sl]    = rval_t
        alpha[sl]      = alpha_t.astype(np.float32)

        # ── Albedo PBR: a = π·α / (max(E_sky+E_ind, albedo_eps)·x) ──────────
        # Calcolata qui dall'α/x in float64 della banda: rifarla a valle dall'α
        # float32 salvata cambierebbe gli ultimi bit.
        if has_irr:
            E = _read_band(irr_path, y0, r).astype(np.float64)
            if has_ind:
                E += _read_band(ind_path, y0, r).astype(np.float64)
            denom = np.maximum(E, albedo_eps)
            x_col = np.maximum(X_t, X_EPS)[:, None]
            alb_t = np.clip(np.pi * alpha_t / (denom * x_col), 0.0, 1.0)
            alb_t[X_t < X_EPS] = 0.0    # completamente speculare: albedo nulla
            alb_t[~mask_t] = 0.0
            albedo_flat[sl] = alb_t.astype(np.float32)

        tot_solvable += int(solvable.sum())
        tot_gated    += int(diffuse_gate.sum())
        tot_rvalid   += int(rval_t.sum())
        best_counts  += np.bincount(best_k[fitted], minlength=n_cand)
        bar.update(1)
    bar.close()

    print(f"  texel risolvibili: {tot_solvable}, "
          f"di cui gated come diffusi (CV<{cv_gate}): {tot_gated}")
    for c_idx, (label, rough, _param) in enumerate(candidates):
        print(f"  candidato {label:>11} (roughness={rough:.3f}) → migliore per "
              f"{int(best_counts[c_idx])} texel")
    print(f"  texel con r attendibile (metallic≥{spec_threshold}): {tot_rvalid}")

    # ── Mappe finali (cartelle dedicate, come l'albedo) ───────────────────────
    fmt = ImageFormat.OPENEXR
    met_dir = src_dir / "metallic";  met_dir.mkdir(parents=True, exist_ok=True)
    rgh_dir = src_dir / "roughness"; rgh_dir.mkdir(parents=True, exist_ok=True)
    metallic_path  = (met_dir / "metallic.exr").resolve().as_posix()
    roughness_path = (rgh_dir / "roughness.exr").resolve().as_posix()
    _save_layer(metallic.reshape(H, W), metallic_path, fmt, DataLayer.METALLIC)
    _save_layer(roughness.reshape(H, W), roughness_path, fmt, DataLayer.ROUGHNESS)

    # Variante R/G/B nella convenzione dei bake di Blender, accanto agli
    # originali a canale singolo (che restano l'input dei lettori interni).
    # force=True: le mappe sono appena state riscritte, una _rgb rimasta da un
    # run precedente sarebbe stantia.
    metallic_rgb_path = roughness_rgb_path = None
    if blender_rgb:
        metallic_rgb_path = write_blender_rgb(metallic_path, force=True, quiet=True)
        roughness_rgb_path = write_blender_rgb(roughness_path, force=True, quiet=True)
        metallic_rgb_path = metallic_rgb_path and metallic_rgb_path.as_posix()
        roughness_rgb_path = roughness_rgb_path and roughness_rgb_path.as_posix()

    # ── Diagnostica ───────────────────────────────────────────────────────────
    pbr_dir = src_dir / "pbr"
    pbr_dir.mkdir(parents=True, exist_ok=True)
    _save_layer(diffuse_w.reshape(H, W), (pbr_dir / "diffuse_weight.exr").as_posix(),
                fmt, DataLayer.METALLIC)
    _save_layer(alpha.reshape(H, W, 3),
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
    # Il calcolo è già stato fatto banda per banda: qui si scrive soltanto.
    albedo_pbr_path = None
    albedo_pbr = None
    if has_irr:
        albedo_pbr = albedo_flat.reshape(H, W, 3)

        alb_dir = src_dir / "albedo_pbr"; alb_dir.mkdir(parents=True, exist_ok=True)
        albedo_pbr_path = (alb_dir / "albedo_pbr.exr").resolve().as_posix()
        _save_layer(albedo_pbr, albedo_pbr_path, fmt, DataLayer.ALBEDO)
        Image.fromarray((albedo_pbr * 255).astype(np.uint8)
                        ).save(pbr_dir / "albedo_pbr_preview.png")
        print(f"✓ albedo_pbr: {albedo_pbr_path} (indirect: "
              f"{'sì' if has_ind else 'no'})")
    else:
        print(f"    ⚠  albedo_pbr saltata: {irr_path} non trovata "
              "(serve il pass irradiance)")

    print(f"✓ metallic:  {metallic_path}")
    print(f"✓ roughness: {roughness_path}")
    if metallic_rgb_path and roughness_rgb_path:
        print(f"✓ variante RGB per Blender: {Path(metallic_rgb_path).name}, "
              f"{Path(roughness_rgb_path).name}")
    print(f"✓ diagnostica in {pbr_dir}")
    return {
        "metallic_path": metallic_path,
        "roughness_path": roughness_path,
        "metallic_rgb_path": metallic_rgb_path,
        "roughness_rgb_path": roughness_rgb_path,
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
    ap.add_argument("--tile-texels", type=int, default=1 << 20,
                    help="texel per banda (arrotondati a scanline intere): il "
                         "picco di RAM scala con questo, non con la risoluzione")
    ap.add_argument("--no-blender-rgb", action="store_true",
                    help="non scrivere le varianti metallic_rgb/roughness_rgb "
                         "(EXR R/G/B nella convenzione dei bake di Blender)")
    args = ap.parse_args()
    solve_pbr(args.output_dir, source=args.source,
              cv_gate=args.cv_gate, spec_threshold=args.spec_threshold,
              min_views=args.min_views, albedo_eps=args.albedo_eps,
              blender_rgb=not args.no_blender_rgb,
              tile_texels=args.tile_texels)
