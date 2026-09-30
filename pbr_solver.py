"""Multi-view PBR fit with hard-cutoff specular cones (a pure mean over the cone),
exported in the metal/roughness convention of Blender's Principled BSDF:

    C_jc = D_c · E_c / π + S_c · L_jc          (per channel c)

    C_j   colour observed by camera j        (sources/{source}/camera_texture/<stem>.exr)
    E     view-independent hemispherical irradiance (irradiance/irradiance.exr
          + irradiance_indirect.exr when present; a shared input, not per-source).
          Required: D, hence metallic and base colour, depend on it.
    D     diffuse reflectance (RGB): the fraction of E re-emitted diffusely
    S     specular reflectance (RGB): the colour of the reflection
    L_j   PURE solid-angle MEAN of the ambient radiance over the cone around the
          reflected ray (no cosθ_R weight). The solver does NOT reconstruct it: it
          reads it from the bake, which writes one RGB channel per candidate into
          spec_cone/cam_{j:03d}.exr (the "cones" format of images_generator.py).
          Candidates = mirror + one cone per aperture of the grid;
          roughness = aperture/180. Being a mean, L stays in radiance units at
          EVERY aperture (the mirror included), so S is a reflectance homogeneous
          across the candidates.

(D, S) are six numbers per texel; the metal/roughness convention compresses them
into four, a base colour b and a metallic m, through
    D_c = (1−m) · b_c        S_c = F0_d · (1−m) + m · b_c        (F0_d = 0.04)
i.e. a dielectric has a coloured diffuse and a grey 4% reflection, a metal no
diffuse and a reflection tinted by b. These are the equations Blender renders, so
(m, b) reproduces (D, S) there. m is solved on the channel means D̄, S̄ (metalness
has no colour; the equations are linear in b, so they survive the averaging):
    S̄ ≤ F0_d → m = 0,  else  u = 2·D̄ / (s + sqrt(s² − 4·F0_d·D̄)),  s = S̄+D̄,  m = 1−u
(the stable form of the smaller root of F0_d·u² − s·u + D̄ = 0, the only one in
[0,1]). With m known, b_c is the least-squares solution of its two equations:
    b_c = [(1−m)·D_c + m·(S_c − F0_d·(1−m))] / [(1−m)² + m²]
which reduces to D at m=0 and to S at m=1.

The mean over the cone is closed by the bake (images_generator._cones_from_rings_*):
here it is enough to know that a level's channel (cone_045deg — the name carries
the aperture) is the solid-angle-weighted mean of the valid rays alone within
aperture k, and that `valid` = the rays above the texel's horizon (0 → the texel
is unusable for that camera).
Note: the resolution on the cone width is quantized by the bake's aperture grid
(spec_cone_apertures_deg) — a deliberate choice, with no sub-grid refinement in
this version.

For each candidate r the model is a per-texel, per-channel linear regression:
    C_jc = α_c + β_c·L_jc      α_c = D_c·E_c/π  (intercept: the diffuse radiance)
                               β_c = S_c        (slope: the specular reflectance)
The diffuse term is identical for every view, so any variation of C between the
cameras is attributable to L alone: centring on the per-camera means eliminates α
(equivalent to working on the differences C_i − C_j) and gives β_c in closed form,
without E. Sufficient statistics, accumulated streaming, one camera at a time:
    Sw = Σ w_j     SC_c = Σ w_j·C     SCC_c = Σ w_j·C²
    SL_c(r) = Σ w_j·L     SLL_c(r) = Σ w_j·L²     SCL_c(r) = Σ w_j·C·L
    VLL_c = SLL_c − SL_c²/Sw    VCL_c = SCL_c − SC_c·SL_c/Sw    VCC_c = SCC_c − SC_c²/Sw
    β_c*(r) = clip(VCL_c / VLL_c, 0, 1)
    res(r) = Σ_c (VCC_c − 2β_c·VCL_c + β_c²·VLL_c) / (3·n_views)   →   argmin over r
The aperture is one per texel (the lobe shape is geometry, not colour), so the
argmin runs on the residual summed over the channels. From the winning candidate:
    α_c = max((SC_c − β_c·SL_c)/Sw, 0),   D_c = π·α_c / max(E_c, albedo_eps),
    S_c = β_c,   SE_c = sqrt(RSS_c / max(n_views−2, 1) / VLL_c)
where SE_c is the standard error of the slope: large where L barely changes
between the views in that channel, i.e. where the specular colour is not measured.
Until 2026-09-30 the slope was one β shared by the three channels (a VLL-weighted
mean of the β_c) and was written as metallic: it was the achromatic specular
reflectance, not the metalness, and the tint of a coloured metal leaked into α.

RAM scales with neither the camera count nor the texture resolution: the outer
loop is over BANDS of texels (blocks of whole scanlines, `tile_texels`) and the
inner one over the cameras. The accumulators live only for the current band and
the fit — which is purely per-texel, no operation relates different texels — is
closed band by band; only the output maps stay at full resolution, in float32.
Every EXR read is one `channels()` over a scanline range: a single decompression
per band instead of one per channel (at 4096² with 14 apertures the cone file has
43 channels, hence 43 decompressions of the whole file per camera).


Validity:
  - the lobe is reliable only where the mean specular reflectance S̄ exceeds
    spec_threshold: below that, roughness is set to 1.0 and the texel is flagged
    in pbr/r_valid.png.
  - unsolvable texels (fewer than min_views cameras) are taken as diffuse: β = 0,
    α = the mean of C across cameras, so D ≡ the classic Lambertian albedo,
    m = 0 and b = D.

Final maps, for the given `source` (call once per source to process: the outputs
live under sources/{source}/ and the inner names are not suffixed):

  <out>/sources/{source}/metallic/metallic.exr      = m  (Principled "Metallic")
  <out>/sources/{source}/base_color/base_color.exr  = b  (Principled "Base Color")
  <out>/sources/{source}/roughness/roughness.exr    = aperture/180 of the winning cone
    (0 = mirror), 1.0 elsewhere — a cone-width index, NOT a GGX α: a
    Disney-compliant texture needs an aperture→α calibration downstream
metallic and roughness are also written as metallic_rgb.exr / roughness_rgb.exr
next to the originals (the same values on three float32 R/G/B channels, the
convention of Blender's bakes) unless blender_rgb=False.
Diagnostics in <out>/sources/{source}/pbr/: spec_color.exr (S), diffuse_albedo.exr
(D, not clipped above 1, so an underestimated E stays visible), diffuse_term.exr
(α, the estimated diffuse radiance), slope_se.exr (SE_c), consistency.exr
(mean_c |D−D̂| + |S−Ŝ|, how far (D, S) is from what (m, b) can represent: mixed
materials, borders, a wrong E), lobe_param.exr (aperture of the winning cone in
degrees, 0 = mirror), residual.exr, n_views.exr, r_valid.png, plus
absolute-scale PNG previews.
Before 2026-09-30 the solver wrote albedo_pbr/ (a = D/x) and pbr/diffuse_weight.exr
(x) instead of base_color/: x no longer exists once the slope is per channel.

Shared inputs (source-independent, not under sources/{source}/): spec_cone/,
ium/ium_masks.exr, visibility/visibility.exr, irradiance/ (required).

`roi_rects` (CLI --roi, repeatable) restricts the fit to the union of rectangles
[x0, y0, w, h] in IUM texels, ANDed with the IUM mask: bands with no active texel
are skipped and every output is 0 outside. Being a per-texel fit, the texels inside
get bit-identical values to a full run. Unlike the pipeline ROI (roi/<tag>/
sandbox, Step 3 re-run), this reads the full-resolution Step 3 cache in place.

Usage:
    python pbr_solver.py <output_dir> [--source gt] [--spec-threshold 0.2]
                         [--min-views 2] [--albedo-eps 1e-3]
                         [--tile-texels 1048576] [--no-blender-rgb]
                         [--roi X0 Y0 W H] [--roi ...]
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

# Normal-incidence reflectance of a dielectric in the metal/roughness convention:
# IOR 1.5, the default of Blender's Principled BSDF.
F0_DIELECTRIC = 0.04


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

def _fit_band(SC: np.ndarray, SCC: np.ndarray, SL: np.ndarray, SLL: np.ndarray,
              SCL: np.ndarray, nv: np.ndarray, eps: float = 1e-12) -> dict:
    """Closed-form per-channel fit of C_jc = α_c + β_c·L_jc(r) for every candidate r,
    then the argmin over r of the residual summed over the channels.

    Inputs are the band's sufficient statistics (float64): SC, SCC (T, 3);
    SL, SLL, SCL (T, K, 3); nv (T,) the number of valid cameras. Pure, so the
    synthetic test can drive it without EXRs. The (T, K, 3) temporaries are built
    in place: at 1 M texels and K=14 each one is 336 MB.

    Returns best_k (T,), beta (T, 3), alpha (T, 3), res (T,) the mean residual per
    equation, se (T, 3) the standard error of each slope.
    """
    T = SC.shape[0]
    Sw = np.maximum(nv.astype(np.float64), 1.0)[:, None, None]      # (T, 1, 1)

    VLL = SL * SL
    VLL /= Sw
    np.subtract(SLL, VLL, out=VLL)
    np.maximum(VLL, 0.0, out=VLL)

    VCL = SC[:, None, :] * SL
    VCL /= Sw
    np.subtract(SCL, VCL, out=VCL)

    VCC = np.maximum(SCC - SC * SC / Sw[:, 0], 0.0)                  # (T, 3)

    beta = VCL / np.maximum(VLL, eps)
    np.clip(beta, 0.0, 1.0, out=beta)

    # RSS_c(r) = VCC_c − 2β_c·VCL_c + β_c²·VLL_c, clamped: cancellation can leave
    # it a few ULP below zero.
    rss = beta * VCL
    rss *= -2.0
    rss += beta * beta * VLL
    rss += VCC[:, None, :]
    np.maximum(rss, 0.0, out=rss)
    del VCL

    res_all = rss.sum(axis=-1) / np.maximum(3.0 * nv, 1.0)[:, None]  # (T, K)
    best_k = np.argmin(res_all, axis=1).astype(np.int32)
    idx = np.arange(T)

    beta_b = beta[idx, best_k]                                       # (T, 3)
    alpha = np.maximum((SC - beta_b * SL[idx, best_k]) / Sw[:, 0], 0.0)
    dof = np.maximum(nv.astype(np.float64) - 2.0, 1.0)[:, None]
    se = np.sqrt(rss[idx, best_k] / dof / np.maximum(VLL[idx, best_k], eps))
    return {"best_k": best_k, "beta": beta_b, "alpha": alpha,
            "res": res_all[idx, best_k], "se": se}


def _metal_from_reflectances(D: np.ndarray, S: np.ndarray,
                             f0_dielectric: float = F0_DIELECTRIC):
    """(D, S) → (m, b, D̂, Ŝ) in the metal/roughness convention.

    D, S: (..., 3) diffuse and specular reflectances. The model is
        D_c = (1−m)·b_c        S_c = f0·(1−m) + m·b_c
    m is solved once on the channel means (metalness has no colour, and averaging
    the channels before solving keeps a dark channel of a coloured dielectric from
    passing for metal): with u = 1−m, f0·u² − (S̄+D̄)·u + D̄ = 0. For S̄ > f0 exactly
    one root lies in [0, 1] (f(0) = D̄ ≥ 0, f(1) = f0 − S̄ < 0), the smaller one,
    written as 2·D̄ / (s + √Δ) to avoid the cancellation of (s − √Δ)/(2·f0) when D̄
    is small. For S̄ ≤ f0 the reflection is no stronger than a dielectric's: m = 0.
    b_c is then the least-squares solution of its two equations, clipped to [0,1].
    D̂, Ŝ are what (m, b) reproduces: their distance from (D, S) is how far the
    texel is from anything the convention can represent.
    """
    D = np.asarray(D, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    Dm = D.mean(axis=-1)
    Sm = S.mean(axis=-1)
    s = Sm + Dm
    disc = np.maximum(s * s - 4.0 * f0_dielectric * Dm, 0.0)
    u = 2.0 * Dm / np.maximum(s + np.sqrt(disc), 1e-30)
    m = np.where(Sm > f0_dielectric, 1.0 - np.clip(u, 0.0, 1.0), 0.0)

    um = (1.0 - m)[..., None]
    mm = m[..., None]
    b = (um * D + mm * (S - f0_dielectric * um)) / (um * um + mm * mm)
    b = np.clip(b, 0.0, 1.0)
    D_hat = um * b
    S_hat = f0_dielectric * um + mm * b
    return m, b, D_hat, S_hat


def _roi_band_mask(roi_rects, y0: int, rows: int, W: int) -> "np.ndarray | None":
    """Union of the ROI rectangles restricted to scanlines [y0, y0+rows) → (rows·W,)
    bool, or None when no ROI is given."""
    if not roi_rects:
        return None
    m = np.zeros((rows, W), dtype=bool)
    for x0, ry0, w, h in roi_rects:
        a, b = max(ry0, y0), min(ry0 + h, y0 + rows)
        if a < b:
            m[a - y0:b - y0, max(x0, 0):min(x0 + w, W)] = True
    return m.reshape(-1)


def solve_pbr(output_dir: str,
              source: str = "gt",
              spec_threshold: float = 0.2,
              min_views: int = 2,
              albedo_eps: float = 1e-3,
              blender_rgb: bool = True,
              tile_texels: int = 1 << 20,
              roi_rects=None,
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

    # D = π·α/E, and metallic and base colour are built from D: without the
    # irradiance nothing of the metal/roughness export can be computed.
    irr_path = out / "irradiance" / "irradiance.exr"
    ind_path = out / "irradiance" / "irradiance_indirect.exr"
    if not irr_path.exists():
        raise FileNotFoundError(
            f"{irr_path} not found: metallic and base colour need the irradiance "
            f"(D = π·α/E). Run the irradiance pass of Step 3 first.")
    has_ind  = ind_path.exists()

    cam_paths  = [src_dir / "camera_texture" / f"{stems[j]}.exr" for j in cams]

    if roi_rects:
        roi_rects = [tuple(int(v) for v in r) for r in roi_rects]
        for r in roi_rects:
            x0, y0, w, h = r
            if len(r) != 4 or w <= 0 or h <= 0:
                raise ValueError(f"ROI must be [x0, y0, w, h] with w, h > 0: {r!r}")
            if x0 >= W or y0 >= H or x0 + w <= 0 or y0 + h <= 0:
                raise ValueError(f"ROI {r!r} lies entirely outside the {W}x{H} texture")

    # ── Partition into bands of whole scanlines ──────────────────────────────
    # The fit is purely per-texel, so partitioning the texture does not change the
    # result by a single bit; it changes only the peak RAM. With the statistics
    # kept per channel a band costs ~tile_texels·n_cand·3·8B·~10 (≈3-4 GiB at 1 M
    # texels and 14 candidates) instead of N·n_cand·3·8B·~10.
    rows = max(1, int(tile_texels) // W)
    if rows >= 16:
        rows = (rows // 16) * 16      # aligned to the ZIP block: no double decompressions
    rows = min(rows, H)
    n_tiles = (H + rows - 1) // rows

    print(f"[pbr_solver] {N} texels, {len(cams)} cameras, "
          f"{n_cand} candidates ({K - 1} mean-cones + mirror)")
    print(f"             {n_tiles} bands of {rows} scanlines ({rows * W} texels)")
    if roi_rects:
        print(f"             ROI: {len(roi_rects)} rectangle(s) {roi_rects}")

    # ── Full-resolution outputs (float32/int32/bool) ─────────────────────────
    n_views    = np.zeros(N, dtype=np.int32)
    metallic   = np.zeros(N, dtype=np.float32)       # m
    base_color = np.zeros((N, 3), dtype=np.float32)  # b
    spec_color = np.zeros((N, 3), dtype=np.float32)  # S = β_c
    diff_alb   = np.zeros((N, 3), dtype=np.float32)  # D = π·α/E (not clipped above)
    slope_se   = np.zeros((N, 3), dtype=np.float32)  # SE_c
    consist    = np.zeros(N, dtype=np.float32)
    lobe_param = np.zeros(N, dtype=np.float32)
    roughness  = np.zeros(N, dtype=np.float32)
    residual   = np.zeros(N, dtype=np.float32)
    r_valid    = np.zeros(N, dtype=bool)
    alpha      = np.zeros((N, 3), dtype=np.float32)

    tot_active = tot_solvable = tot_rvalid = tot_metal = 0
    best_counts = np.zeros(n_cand, dtype=np.int64)

    bar = _tile_bar(n_tiles, f"Fit PBR ({source})")
    for y0 in range(0, H, rows):
        r   = min(rows, H - y0)
        off = y0 * W
        T   = r * W
        sl  = slice(off, off + T)

        mask_t = (_read_band(mask_path, y0, r) > 0.5) if has_mask \
            else np.ones(T, dtype=bool)
        roi_t = _roi_band_mask(roi_rects, y0, r, W)
        if roi_t is not None:
            mask_t &= roi_t

        # Band with no active texel (outside the IUM mask or the ROI): every output
        # of the band would be zero, and the arrays are already zero-initialised, so
        # skipping it is bit-identical. It avoids the loop over cameras, which is
        # the expensive part (one band of camera_texture plus one of cam_XXX.exr per
        # camera). It always pays off — the UV atlas has empty regions — but it is
        # what makes a reconstruction restricted to a ROI nearly free.
        if not mask_t.any():
            bar.update(1)
            continue
        tot_active += int(mask_t.sum())

        # A single read per band instead of one per camera.
        vis_t = (_read_band(vis_path, y0, r, vis_cols) > 0.5) if has_vis else None

        # ── Streaming scan: one camera at a time, per-channel statistics ─────
        SC  = np.zeros((T, 3))             # Σ w·C          (candidate-independent)
        SCC = np.zeros((T, 3))             # Σ w·C²         (candidate-independent)
        SL  = np.zeros((T, n_cand, 3))     # Σ w·L          per candidate
        SLL = np.zeros((T, n_cand, 3))     # Σ w·L²         per candidate
        SCL = np.zeros((T, n_cand, 3))     # Σ w·C·L        per candidate
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

            wf = w_j.astype(np.float64)[:, None]
            SC  += wf * C_j
            SCC += wf * C_j * C_j
            for c_idx in range(n_cand):
                L = cones[:, c_idx]
                SL[:, c_idx]  += wf * L
                SLL[:, c_idx] += wf * L * L
                SCL[:, c_idx] += wf * C_j * L
            del cones

        solvable = mask_t & (nv_t >= min_views)

        # ── Fit per candidate and channel, argmin over candidates ───────────
        fit = _fit_band(SC, SCC, SL, SLL, SCL, nv_t, eps)
        del SL, SLL, SCL
        best_k = fit["best_k"]
        fitted = solvable & np.isfinite(fit["res"])

        # Unsolvable texels of the mask are diffuse: β = 0, α = mean of C across
        # cameras ⇒ D ≡ the classic Lambertian albedo, m = 0, b = D.
        Sw = np.maximum(nv_t.astype(np.float64), 1.0)[:, None]
        beta_t  = np.where(fitted[:, None], fit["beta"], 0.0)
        alpha_t = np.where(fitted[:, None], fit["alpha"], SC / Sw)
        se_t    = np.where(fitted[:, None], fit["se"], 0.0)
        alpha_t[~mask_t] = 0.0
        beta_t[~mask_t]  = 0.0
        se_t[~mask_t]    = 0.0

        # ── Reflectances → metal/roughness convention ───────────────────────
        # Computed here from the band's float64 α/β: redoing it downstream from the
        # saved float32 maps would change the last bits.
        E = _read_band(irr_path, y0, r).astype(np.float64)
        if has_ind:
            E += _read_band(ind_path, y0, r).astype(np.float64)
        D_t = np.pi * alpha_t / np.maximum(E, albedo_eps)
        D_c = np.clip(D_t, 0.0, 1.0)
        m_t, b_t, D_hat, S_hat = _metal_from_reflectances(D_c, beta_t)
        cons_t = (np.abs(D_c - D_hat) + np.abs(beta_t - S_hat)).mean(axis=-1)
        m_t[~mask_t] = 0.0
        b_t[~mask_t] = 0.0
        cons_t[~mask_t] = 0.0

        lobe_t = np.zeros(T, dtype=np.float32)
        lobe_t[fitted] = param_vals[best_k[fitted]]

        # the lobe is reliable only with enough specular reflectance; elsewhere roughness=1
        rval_t = fitted & (beta_t.mean(axis=-1) >= spec_threshold)
        rgh_t  = np.where(mask_t, 1.0, 0.0).astype(np.float32)
        rgh_t[rval_t] = rough_vals[best_k[rval_t]]

        res_t = np.zeros(T, dtype=np.float32)
        res_t[fitted] = fit["res"][fitted].astype(np.float32)

        n_views[sl]    = nv_t
        metallic[sl]   = m_t.astype(np.float32)
        base_color[sl] = b_t.astype(np.float32)
        spec_color[sl] = beta_t.astype(np.float32)
        diff_alb[sl]   = D_t.astype(np.float32)
        slope_se[sl]   = se_t.astype(np.float32)
        consist[sl]    = cons_t.astype(np.float32)
        lobe_param[sl] = lobe_t
        roughness[sl]  = rgh_t
        residual[sl]   = res_t
        r_valid[sl]    = rval_t
        alpha[sl]      = alpha_t.astype(np.float32)

        tot_solvable += int(solvable.sum())
        tot_rvalid   += int(rval_t.sum())
        tot_metal    += int((fitted & (m_t > 0.5)).sum())
        best_counts  += np.bincount(best_k[fitted], minlength=n_cand)
        bar.update(1)
    bar.close()

    if roi_rects:
        print(f"  active texels (IUM mask ∩ ROI): {tot_active}")
    print(f"  solvable texels: {tot_solvable}")
    for c_idx, (label, rough, _param) in enumerate(candidates):
        print(f"  candidate {label:>11} (roughness={rough:.3f}) → best for "
              f"{int(best_counts[c_idx])} texels")
    print(f"  texels with a reliable r (mean S≥{spec_threshold}): {tot_rvalid}")
    print(f"  texels with metallic > 0.5: {tot_metal}")

    # ── Final maps (dedicated folders) ───────────────────────────────────────
    fmt = ImageFormat.OPENEXR
    met_dir = src_dir / "metallic";   met_dir.mkdir(parents=True, exist_ok=True)
    rgh_dir = src_dir / "roughness";  rgh_dir.mkdir(parents=True, exist_ok=True)
    bc_dir  = src_dir / "base_color"; bc_dir.mkdir(parents=True, exist_ok=True)
    metallic_path   = (met_dir / "metallic.exr").resolve().as_posix()
    roughness_path  = (rgh_dir / "roughness.exr").resolve().as_posix()
    base_color_path = (bc_dir / "base_color.exr").resolve().as_posix()
    _save_layer(metallic.reshape(H, W), metallic_path, fmt, DataLayer.METALLIC)
    _save_layer(roughness.reshape(H, W), roughness_path, fmt, DataLayer.ROUGHNESS)
    _save_layer(base_color.reshape(H, W, 3), base_color_path, fmt, DataLayer.ALBEDO)

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
    _save_layer(spec_color.reshape(H, W, 3),
                (pbr_dir / "spec_color.exr").as_posix(), fmt, DataLayer.ALBEDO)
    _save_layer(diff_alb.reshape(H, W, 3),
                (pbr_dir / "diffuse_albedo.exr").as_posix(), fmt, DataLayer.ALBEDO)
    _save_layer(alpha.reshape(H, W, 3),
                (pbr_dir / "diffuse_term.exr").as_posix(), fmt, DataLayer.ALBEDO)
    _save_layer(slope_se.reshape(H, W, 3),
                (pbr_dir / "slope_se.exr").as_posix(), fmt, DataLayer.ALBEDO)
    _save_layer(consist.reshape(H, W), (pbr_dir / "consistency.exr").as_posix(),
                fmt, DataLayer.SPEC_CONE_R)
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
    Image.fromarray((np.clip(base_color, 0, 1).reshape(H, W, 3) * 255).astype(np.uint8)
                    ).save(pbr_dir / "base_color_preview.png")

    print(f"✓ metallic:   {metallic_path}")
    print(f"✓ base_color: {base_color_path} (indirect: {'yes' if has_ind else 'no'})")
    print(f"✓ roughness:  {roughness_path}")
    if metallic_rgb_path and roughness_rgb_path:
        print(f"✓ Blender RGB variants: {Path(metallic_rgb_path).name}, "
              f"{Path(roughness_rgb_path).name}")
    print(f"✓ diagnostics in {pbr_dir}")
    return {
        "metallic_path": metallic_path,
        "roughness_path": roughness_path,
        "base_color_path": base_color_path,
        "metallic_rgb_path": metallic_rgb_path,
        "roughness_rgb_path": roughness_rgb_path,
        "metallic": metallic.reshape(H, W),
        "roughness": roughness.reshape(H, W),
        "base_color": base_color.reshape(H, W, 3),
        "spec_color": spec_color.reshape(H, W, 3),
        "diffuse_albedo": diff_alb.reshape(H, W, 3),
        "diffuse_term": alpha.reshape(H, W, 3),
        "slope_se": slope_se.reshape(H, W, 3),
        "consistency": consist.reshape(H, W),
        "lobe_param": lobe_param.reshape(H, W),
        "residual": residual.reshape(H, W),
        "n_views": n_views.reshape(H, W),
        "r_valid": r_valid.reshape(H, W),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="PBR fit C_jc = D_c·E_c/π + S_c·L_jc(r) → "
                    "metallic/base_color/roughness")
    ap.add_argument("output_dir", help="the pipeline output_dir (holds spec_cone/, ...)")
    ap.add_argument("--source", type=str, default="gt",
                    help="source to process (sources/{source}/, e.g. gt or nerf)")
    ap.add_argument("--spec-threshold", type=float, default=0.2,
                    help="minimum mean specular reflectance for r to be reliable "
                         "(below it: roughness=1). The pipeline uses 0")
    ap.add_argument("--min-views", type=int, default=2,
                    help="minimum number of valid cameras per texel")
    ap.add_argument("--albedo-eps", type=float, default=1e-3,
                    help="lower clamp on the irradiance in D = π·α/E")
    ap.add_argument("--tile-texels", type=int, default=1 << 20,
                    help="texels per band (rounded to whole scanlines): the peak "
                         "RAM scales with this, not with the resolution")
    ap.add_argument("--no-blender-rgb", action="store_true",
                    help="do not write the metallic_rgb/roughness_rgb variants "
                         "(R/G/B EXRs in the convention of Blender's bakes)")
    ap.add_argument("--roi", type=int, nargs=4, action="append",
                    metavar=("X0", "Y0", "W", "H"),
                    help="restrict the fit to this rectangle of IUM texels; repeatable "
                         "(union). Every output is 0 outside")
    args = ap.parse_args()
    solve_pbr(args.output_dir, source=args.source,
              spec_threshold=args.spec_threshold,
              min_views=args.min_views, albedo_eps=args.albedo_eps,
              blender_rgb=not args.no_blender_rgb,
              tile_texels=args.tile_texels,
              roi_rects=args.roi)
