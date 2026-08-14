"""Multi-view PBR fit with hard-cutoff specular cones (a pure mean over the cone):

    C_j = (a·x/π) · E + (1 − x) · L_j        (per channel c)

    C_j   colour observed by camera j        (sources/{source}/camera_texture/<stem>.exr)
    E     view-independent hemispherical irradiance (irradiance/irradiance.exr
          + irradiance_indirect.exr when present; a shared input, not per-source).
          Needed ONLY for the albedo: metallic and roughness do not require it.
    a     diffuse albedo (unknown, RGB, in [0,1])
    x     weight of the DIFFUSE term (x=1 → no view dependence);
          the specularity/metallic is 1−x
    L_j   PURE solid-angle MEAN of the ambient radiance over the cone around the
          reflected ray (no cosθ_R weight). The solver does NOT reconstruct it: it
          reads it from the bake, which writes one RGB channel per candidate into
          spec_cone/cam_{j:03d}.exr (the "cones" format of images_generator.py).
          Candidates = mirror + one cone per aperture of the grid;
          roughness = aperture/180. Being a mean, L stays in radiance units at
          EVERY aperture (the mirror included): metallic = 1−x is a specular
          fraction homogeneous across the candidates.

The mean over the cone is closed by the bake (images_generator._cones_from_rings_*):
here it is enough to know that a level's channel (cone_045deg — the name carries
the aperture) is the solid-angle-weighted mean of the valid rays alone within
aperture k, and that `valid` = the rays above the texel's horizon (0 → the texel
is unusable for that camera).
Note: the resolution on the cone width is quantized by the bake's aperture grid
(spec_cone_apertures_deg) — a deliberate choice, with no sub-grid refinement in
this version.

For each candidate r the model is a per-texel linear regression:
    C_jc = α_c + β·L_jc        α_c = x·a_c·E_c/π  (intercept, per channel)
                               β   = 1−x          (slope, shared across channels)
The diffuse term is identical for every view, so any variation of C between the
cameras is attributable to L alone: centring on the per-camera means eliminates α
(equivalent to working on the differences C_i − C_j) and gives β in closed form.
Sufficient statistics, accumulated streaming, one camera at a time:
    Sw = Σ w_j     SC_c = Σ w_j·C     SCC = Σ_{j,c} w_j·C²
    SL_c(r) = Σ w_j·L     SLL(r) = Σ_{j,c} w_j·L²     SCL(r) = Σ_{j,c} w_j·C·L
    VLL = SLL − Σ_c SL²/Sw    VCL = SCL − Σ_c SC·SL/Sw    VCC = SCC − Σ_c SC²/Sw
    β*(r) = clip(VCL / VLL, 0, 1)
    res(r) = (VCC − 2β·VCL + β²·VLL) / (3·n_views)   →   argmin over r
From the winning candidate: x = 1−β,  α_c = (SC_c − β·SL_c)/Sw (≥ 0),
    a_c = clip(π·α_c / (max(E_c, albedo_eps)·x), 0, 1);  x < X_EPS → a = 0
(metal convention: a fully specular texel has no diffuse albedo).

RAM scales with neither the camera count nor the texture resolution: the outer
loop is over BANDS of texels (blocks of whole scanlines, `tile_texels`) and the
inner one over the cameras. The accumulators live only for the current band and
the fit — which is purely per-texel, no operation relates different texels — is
closed band by band; only the output maps stay at full resolution, in float32.
Every EXR read is one `channels()` over a scanline range: a single decompression
per band instead of one per channel (at 4096² with 14 apertures the cone file has
43 channels, hence 43 decompressions of the whole file per camera).


Validity:
  - the lobe is reliable only where the specularity exceeds spec_threshold: below
    that, roughness is set to 1.0 and the texel is flagged in pbr/r_valid.png.

Final maps, for the given `source` (call once per source to process: the outputs
live under sources/{source}/ and the inner names are not suffixed):

  <out>/sources/{source}/metallic/metallic.exr      = 1−x   (0=diffuse, 1=fully specular)
  <out>/sources/{source}/roughness/roughness.exr    = aperture/180 of the winning cone
    (0 = mirror), 1.0 elsewhere — a cone-width index, NOT a GGX α: a
    Disney-compliant texture needs an aperture→α calibration downstream
  <out>/sources/{source}/albedo_pbr/albedo_pbr.exr  = a from the fit (diffuse albedo
    with the per-view specular removed; it coexists with the classic Lambertian
    albedo in <out>/sources/{source}/albedo/). Requires irradiance/irradiance.exr
    on disk (irradiance_indirect.exr is added when present); if it is missing the
    map is skipped with a warning. On unsolvable texels x=1 is assumed (diffuse,
    α = the mean across cameras → a ≡ the classic Lambertian albedo).
metallic and roughness are also written as metallic_rgb.exr / roughness_rgb.exr
next to the originals (the same values on three float32 R/G/B channels, the
convention of Blender's bakes) unless blender_rgb=False.
Diagnostics in <out>/sources/{source}/pbr/: diffuse_weight.exr (x),
diffuse_term.exr (α, the estimated diffuse radiance), lobe_param.exr (aperture of
the winning cone in degrees, 0 = mirror), residual.exr, n_views.exr,
r_valid.png, plus absolute-scale PNG previews.

Shared inputs (source-independent, not under sources/{source}/): spec_cone/,
ium/ium_masks.exr, visibility/visibility.exr, irradiance/.

Usage:
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

# Below this x the texel counts as fully specular: the diffuse albedo is not
# defined (α/x → 0/0) and is written as 0 (the metal convention).
X_EPS = 1e-3


# ──────────────────────────────────────────────────────────────────────────────
# IO helpers
# ──────────────────────────────────────────────────────────────────────────────

def _channel_order(names: "list[str]") -> "tuple[list[str], bool]":
    """Canonical channel order of an EXR → (names, single_channel).

    A rule shared by _read_exr and _ExrBandReader: column j of a full-image array
    and column j of a band have to refer to the same channel. It is not trivial,
    because ExrWriter (images_generator.py:100-106) names channels by their count:
    1 → Z, 3 → R,G,B, 4 → R,G,B,A and only from 5 upwards Cam0, Cam1, … — so a
    visibility with 3 or 4 cameras has NO Cam* channels.
    """
    names = list(names)
    if names == ["Z"] or names == ["Y"]:
        return names, True
    if set("RGB").issubset(names):
        return ["R", "G", "B"] + (["A"] if "A" in names else []), False
    return sorted(names, key=lambda n: (len(n), n)), False   # Cam0, Cam1, … in numeric order


class _ExrBandReader:
    """EXR read in blocks of consecutive scanlines: the read-side counterpart of
    images_generator.IncrementalExrWriter.

    The solver needs it: its outer loop is over bands of texels and the inner one
    over cameras, and at full resolution one camera's cones are already 2.6 GiB
    with the accumulators near 10, while a band costs a few tens of MB.

    One `channels()` per block rather than one `channel()` per channel: the
    framebuffer is mounted once and the ZIP blocks are decompressed once. With the
    43 channels of the cone file the measured difference is ~34x.
    """

    def __init__(self, path: Path):
        import OpenEXR, Imath

        self.path = Path(path)
        self._exr = OpenEXR.InputFile(self.path.as_posix())
        header = self._exr.header()
        dw = header["dataWindow"]
        self.width  = dw.max.x - dw.min.x + 1
        self.height = dw.max.y - dw.min.y + 1
        self._pt    = Imath.PixelType(Imath.PixelType.FLOAT)  # half is converted on read
        self._avail = set(header["channels"].keys())
        self.names, self.single = _channel_order(list(header["channels"].keys()))

    # ── reading ──────────────────────────────────────────────────────────────
    def read_raw(self, y0: int, rows: "int | None",
                 names: "list[str]") -> "list[bytes]":
        """Raw buffers of the requested channels, in the requested order."""
        rows = self.height - y0 if rows is None else rows
        if y0 < 0 or rows <= 0 or y0 + rows > self.height:
            raise ValueError(f"{self.path}: band [{y0}, {y0 + rows}) outside "
                             f"the {self.height} scanlines of the file")
        missing = [n for n in names if n not in self._avail]
        if missing:
            raise ValueError(f"{self.path}: missing channels {missing} "
                             f"(available: {sorted(self._avail)[:8]}…)")
        return self._exr.channels(names, self._pt, y0, y0 + rows - 1)

    def read(self, y0: int = 0, rows: "int | None" = None,
             names=None) -> np.ndarray:
        """Band → (rows·width, C) float32, or (rows·width,) when `names` is a string
        or the file has a single channel (Z/Y) and `names` is None."""
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
            return np.frombuffer(bufs[0], dtype=np.float32).copy()   # writable
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
    """One band of scanlines from `path` (open, read, close)."""
    with _ExrBandReader(path) as rd:
        return rd.read(y0, rows, names)


def _read_exr(path: Path) -> np.ndarray:
    """EXR → (H, W) float32 [Z channel] or (H, W, C) [R,G,B(,A) or Cam*]."""
    with _ExrBandReader(path) as rd:
        arr = rd.read()
        return arr.reshape(rd.height, rd.width) if arr.ndim == 1 else \
            arr.reshape(rd.height, rd.width, -1)


def read_cones(path: Path, apertures_deg, y0: int = 0,
               rows: "int | None" = None) -> "tuple[np.ndarray, np.ndarray]":
    """A camera's cone EXR → (L (n, K, 3), n_valid (n,)).

    One RGB channel per level, with the pure mean over the cone already closed by
    the bake, plus `valid`, the texel's number of valid rays. Level names carry the
    aperture (`cone_045deg`) and are generated by spec_cone_level_name from the
    apertures in the meta: writer and reader use the same function, so they cannot
    diverge. One file per camera rather than one per aperture: the shared bake has
    the tile loop on the outside, so the writers of every camera stay open together
    and K+1 files per camera would exceed the MSVC stdio limit.


    With y0/rows only one band of scanlines is read (n = rows·W); the default is
    still the whole image (n = N).
    """
    K = len(apertures_deg)
    want = [f"{spec_cone_level_name(apertures_deg, k)}.{c}"
            for k in range(K) for c in "RGB"] + ["valid"]

    with _ExrBandReader(path) as rd:
        try:
            bufs = rd.read_raw(y0, rows, want)
        except ValueError as err:
            raise ValueError(f"{err} — incomplete bake, or the wrong num_levels "
                             f"in the meta") from None
        n = (rd.height - y0 if rows is None else rows) * rd.width
        # Filled channel by channel: a nested np.stack would keep the K (n,3) slices
        # alive *and* the result, doubling the peak on a full image.
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
              spec_threshold: float = 0.2,
              min_views: int = 2,
              albedo_eps: float = 1e-3,
              blender_rgb: bool = True,
              tile_texels: int = 1 << 20,
              eps: float = 1e-12) -> dict:
    out = Path(output_dir)
    src_dir = out / "sources" / source     # source-dependent artefacts (camera_texture/, pixel_change/, PBR outputs)
    spec_dir = out / "spec_cone"           # shared (source-independent)

    with open(spec_dir / "spec_cone_meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    with open(out / "transforms_extended.json", encoding="utf-8") as fh:
        tjson = json.load(fh)

    # The bake writes the cones already closed (one RGB channel per candidate), so
    # nothing is reconstructed here: L_j(r) is simply read. Bakes in the
    # "rings"/"rings_shared" formats (per-ring means) are no longer supported.
    if meta.get("format") != "cones":
        raise ValueError(
            f"spec_cone_meta.json is in format {meta.get('format')!r}: the solver "
            f"requires the 'cones' format (L_j(r) already closed by the bake). "
            f"Re-run the spec_cone precompute.")

    apertures = np.asarray(meta["apertures_deg"], dtype=np.float32)
    cams      = meta["cameras"]
    K         = meta["num_levels"]
    stems     = [Path(f["file_path"]).stem for f in tjson["frames"]]

    # ── Candidates: [mirror] + one cone (pure mean) per aperture ─────────────
    # These are the bake's K channels in the order it wrote them: the candidate
    # index IS the channel index. Tuple: (label, roughness, param).
    candidates = [("mirror", 0.0, 0.0)]
    candidates += [(f"r={apertures[k]:g}°mean",
                    float(apertures[k]) / 180.0, float(apertures[k]))
                   for k in range(1, K)]
    n_cand = len(candidates)

    rough_vals = np.asarray([c[1] for c in candidates], dtype=np.float32)
    param_vals = np.asarray([c[2] for c in candidates], dtype=np.float32)

    # ── Input: geometry from the headers only, pixels are read band by band ──
    # The resolution comes from the first cone EXR: a file the solver has to read
    # for every camera anyway, whereas the IUM mask is optional and pixel_change is
    # no longer opened at all.
    cone_paths = [spec_dir / meta["cam_file_pattern"].format(cam=j) for j in cams]
    with _ExrBandReader(cone_paths[0]) as rd:
        H, W = rd.height, rd.width
    N = H * W

    mask_path = out / "ium" / "ium_masks.exr"
    has_mask  = mask_path.exists()

    vis_path = out / "visibility" / "visibility.exr"
    has_vis  = vis_path.exists()
    if has_vis:
        # Column j of the array _read_exr would build is camera j: the canonical
        # channel order gives the matching name, and reading by name avoids
        # decompressing the channels of cameras that are not needed.
        with _ExrBandReader(vis_path) as rd:
            vis_cols = [rd.names[j] for j in cams]

    # Only the albedo needs this: the check sits outside the loop so the bands
    # allocate nothing when the irradiance is missing.
    irr_path = out / "irradiance" / "irradiance.exr"
    ind_path = out / "irradiance" / "irradiance_indirect.exr"
    has_irr  = irr_path.exists()
    has_ind  = has_irr and ind_path.exists()

    cam_paths  = [src_dir / "camera_texture" / f"{stems[j]}.exr" for j in cams]

    # ── Partition into bands of whole scanlines ──────────────────────────────
    # The fit is purely per-texel, so partitioning the texture does not change the
    # result by a single bit; it changes only the peak RAM, which becomes
    # ~tile_texels·n_cand·8B·20 instead of N·n_cand·8B·20.
    rows = max(1, int(tile_texels) // W)
    if rows >= 16:
        rows = (rows // 16) * 16      # aligned to the ZIP block: no double decompressions
    rows = min(rows, H)
    n_tiles = (H + rows - 1) // rows

    print(f"[pbr_solver] {N} texels, {len(cams)} cameras, "
          f"{n_cand} candidates ({K - 1} mean-cones + mirror)")
    print(f"             {n_tiles} bands of {rows} scanlines ({rows * W} texels)")

    # ── Full-resolution outputs (float32/int32/bool: ~870 MiB at 4096²) ──────
    n_views    = np.zeros(N, dtype=np.int32)
    diffuse_w  = np.zeros(N, dtype=np.float32)     # x
    metallic   = np.zeros(N, dtype=np.float32)     # β = 1−x (specularity)
    lobe_param = np.zeros(N, dtype=np.float32)
    roughness  = np.zeros(N, dtype=np.float32)
    residual   = np.zeros(N, dtype=np.float32)
    r_valid    = np.zeros(N, dtype=bool)
    alpha      = np.zeros((N, 3), dtype=np.float32)
    albedo_flat = np.zeros((N, 3), dtype=np.float32) if has_irr else None

    tot_solvable = tot_rvalid = 0
    best_counts  = np.zeros(n_cand, dtype=np.int64)

    bar = _tile_bar(n_tiles, f"Fit PBR ({source})")
    for y0 in range(0, H, rows):
        r   = min(rows, H - y0)
        off = y0 * W
        T   = r * W
        sl  = slice(off, off + T)

        mask_t = (_read_band(mask_path, y0, r) > 0.5) if has_mask \
            else np.ones(T, dtype=bool)

        # Band entirely outside the IUM mask: with no valid texel every output of
        # the band would be zero, and the arrays are already zero-initialised, so
        # skipping it is bit-identical. It avoids the two pixel_change reads and,
        # above all, the loop over cameras, which is the expensive part (one band of
        # camera_texture plus one of cam_XXX.exr per camera). It always pays off —
        # the UV atlas has empty regions — but it is what makes a reconstruction
        # restricted to a ROI nearly free.
        if not mask_t.any():
            bar.update(1)
            continue

        # A single read per band instead of one per camera.
        vis_t = (_read_band(vis_path, y0, r, vis_cols) > 0.5) if has_vis else None

        # ── Streaming scan: one camera at a time, sufficient statistics ──────
        SC  = np.zeros((T, 3))             # Σ w·C          (candidate-independent)
        SCC = np.zeros(T)                  # Σ w·C² pooled  (candidate-independent)
        SL  = np.zeros((T, n_cand, 3))     # Σ w·L          per candidate
        SLL = np.zeros((T, n_cand))        # Σ w·L² pooled  per candidate
        SCL = np.zeros((T, n_cand))        # Σ w·C·L pooled per candidate
        nv_t = np.zeros(T, dtype=np.int32)

        for jj in range(len(cams)):
            C_j = _read_band(cam_paths[jj], y0, r).astype(np.float64)   # (T, 3)
            w_j = mask_t.copy()
            if vis_t is not None:
                w_j &= vis_t[:, jj]

            cones, n_valid = read_cones(cone_paths[jj], apertures, y0, r)
            cones = cones.astype(np.float64)                # (T, n_cand, 3)
            # A texel is valid for this camera if at least one ray landed above the
            # horizon: the same mask the bake uses to zero the cones.
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

        # ── Fit per candidate: centred regression in closed form ────────────
        Sw  = np.maximum(nv_t.astype(np.float64), 1.0)
        VLL = np.maximum(SLL - (SL ** 2).sum(axis=-1) / Sw[:, None], 0.0)
        VCL = SCL - np.einsum("nc,nkc->nk", SC, SL) / Sw[:, None]
        VCC = np.maximum(SCC - (SC ** 2).sum(axis=-1) / Sw, 0.0)

        beta_all = np.clip(VCL / np.maximum(VLL, eps), 0.0, 1.0)    # β = 1−x
        res_all  = VCC[:, None] - 2.0 * beta_all * VCL + beta_all ** 2 * VLL
        res_all /= np.maximum(3.0 * nv_t, 1.0)[:, None]  # mean residual per equation

        target    = solvable
        best_k    = np.argmin(res_all, axis=1).astype(np.int32)
        _idx      = np.arange(T)
        best_res  = res_all[_idx, best_k]
        best_beta = beta_all[_idx, best_k]

        # ── Compose the band's outputs ──────────────────────────────────────
        fitted = target & np.isfinite(best_res)

        dw_t = np.zeros(T, dtype=np.float32)
        dw_t[fitted] = (1.0 - best_beta[fitted]).astype(np.float32)

        met_t = np.zeros(T, dtype=np.float32)
        met_t[fitted] = best_beta[fitted].astype(np.float32)

        lobe_t = np.zeros(T, dtype=np.float32)
        lobe_t[fitted] = param_vals[best_k[fitted]]

        # the lobe is reliable only with enough specularity; elsewhere roughness=1
        rval_t = fitted & (met_t >= spec_threshold)
        rgh_t  = np.where(mask_t, 1.0, 0.0).astype(np.float32)
        rgh_t[rval_t] = rough_vals[best_k[rval_t]]

        res_t = np.zeros(T, dtype=np.float32)
        res_t[fitted] = best_res[fitted].astype(np.float32)

        # ── Intercept α = x·a·E/π (the estimated diffuse radiance) ──────────
        # x and α are complete over the whole mask: gated/unfittable texels are
        # diffuse (x=1, α = mean of C across cameras ⇒ a ≡ classic Lambertian albedo)
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

        # ── PBR albedo: a = π·α / (max(E_sky+E_ind, albedo_eps)·x) ──────────
        # Computed here from the band's float64 α/x: redoing it downstream from the
        # saved float32 α would change the last bits.
        if has_irr:
            E = _read_band(irr_path, y0, r).astype(np.float64)
            if has_ind:
                E += _read_band(ind_path, y0, r).astype(np.float64)
            denom = np.maximum(E, albedo_eps)
            x_col = np.maximum(X_t, X_EPS)[:, None]
            alb_t = np.clip(np.pi * alpha_t / (denom * x_col), 0.0, 1.0)
            alb_t[X_t < X_EPS] = 0.0    # fully specular: zero albedo
            alb_t[~mask_t] = 0.0
            albedo_flat[sl] = alb_t.astype(np.float32)

        tot_solvable += int(solvable.sum())
        tot_rvalid   += int(rval_t.sum())
        best_counts  += np.bincount(best_k[fitted], minlength=n_cand)
        bar.update(1)
    bar.close()

    print(f"  solvable texels: {tot_solvable}")
    for c_idx, (label, rough, _param) in enumerate(candidates):
        print(f"  candidate {label:>11} (roughness={rough:.3f}) → best for "
              f"{int(best_counts[c_idx])} texels")
    print(f"  texels with a reliable r (metallic≥{spec_threshold}): {tot_rvalid}")

    # ── Final maps (dedicated folders, like the albedo) ───────────────────────
    fmt = ImageFormat.OPENEXR
    met_dir = src_dir / "metallic";  met_dir.mkdir(parents=True, exist_ok=True)
    rgh_dir = src_dir / "roughness"; rgh_dir.mkdir(parents=True, exist_ok=True)
    metallic_path  = (met_dir / "metallic.exr").resolve().as_posix()
    roughness_path = (rgh_dir / "roughness.exr").resolve().as_posix()
    _save_layer(metallic.reshape(H, W), metallic_path, fmt, DataLayer.METALLIC)
    _save_layer(roughness.reshape(H, W), roughness_path, fmt, DataLayer.ROUGHNESS)

    # R/G/B variant in the convention of Blender's bakes, next to the
    # single-channel originals (which stay the input of the internal readers).
    # force=True: the maps were just rewritten, so an _rgb left over from an
    # earlier run would be stale.
    metallic_rgb_path = roughness_rgb_path = None
    if blender_rgb:
        metallic_rgb_path = write_blender_rgb(metallic_path, force=True, quiet=True)
        roughness_rgb_path = write_blender_rgb(roughness_path, force=True, quiet=True)
        metallic_rgb_path = metallic_rgb_path and metallic_rgb_path.as_posix()
        roughness_rgb_path = roughness_rgb_path and roughness_rgb_path.as_posix()

    # ── Diagnostics ───────────────────────────────────────────────────────────
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

    # Absolute-scale PNG previews
    from PIL import Image
    Image.fromarray((np.clip(metallic, 0, 1).reshape(H, W) * 255).astype(np.uint8)
                    ).save(pbr_dir / "metallic_preview.png")
    Image.fromarray((np.clip(roughness, 0, 1).reshape(H, W) * 255).astype(np.uint8)
                    ).save(pbr_dir / "roughness_preview.png")

    # ── PBR albedo: a = π·α / (max(E_sky+E_ind, albedo_eps)·x) ───────────────
    # The albedo comes straight out of the fit, with the per-view specular already
    # removed; it coexists with the classic Lambertian albedo in
    # <out>/sources/{source}/albedo/. It was computed band by band: here it is only written.
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
              f"{'yes' if has_ind else 'no'})")
    else:
        print(f"    ⚠  albedo_pbr skipped: {irr_path} not found "
              "(the irradiance pass is required)")

    print(f"✓ metallic:  {metallic_path}")
    print(f"✓ roughness: {roughness_path}")
    if metallic_rgb_path and roughness_rgb_path:
        print(f"✓ Blender RGB variants: {Path(metallic_rgb_path).name}, "
              f"{Path(roughness_rgb_path).name}")
    print(f"✓ diagnostics in {pbr_dir}")
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
        description="PBR fit C_j = (a·x/π)·E + (1-x)·L_j(r) → "
                    "metallic/roughness/albedo_pbr")
    ap.add_argument("output_dir", help="the pipeline output_dir (holds spec_cone/, ...)")
    ap.add_argument("--source", type=str, default="gt",
                    help="source to process (sources/{source}/, e.g. gt or nerf)")
    ap.add_argument("--spec-threshold", type=float, default=0.2,
                    help="minimum metallic for r to be reliable (below it: roughness=1)")
    ap.add_argument("--min-views", type=int, default=2,
                    help="minimum number of valid cameras per texel")
    ap.add_argument("--albedo-eps", type=float, default=1e-3,
                    help="lower clamp on the irradiance in albedo_pbr")
    ap.add_argument("--tile-texels", type=int, default=1 << 20,
                    help="texels per band (rounded to whole scanlines): the peak "
                         "RAM scales with this, not with the resolution")
    ap.add_argument("--no-blender-rgb", action="store_true",
                    help="do not write the metallic_rgb/roughness_rgb variants "
                         "(R/G/B EXRs in the convention of Blender's bakes)")
    args = ap.parse_args()
    solve_pbr(args.output_dir, source=args.source,
              spec_threshold=args.spec_threshold,
              min_views=args.min_views, albedo_eps=args.albedo_eps,
              blender_rgb=not args.no_blender_rgb,
              tile_texels=args.tile_texels)
