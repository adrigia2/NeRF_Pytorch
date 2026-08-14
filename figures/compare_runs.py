#!/usr/bin/env python
"""compare_runs.py -- comparison of the runs of a NeRF sweep (activation x loss).

Reads the artefacts images_generator.py already produced for every run of the sweep
(training_metrics.csv, nerf_render_images/iter_*/metrics_per_frame.csv,
bias_bins.csv, and the frame_NNN_{gt,pred}.exr pairs) and produces comparative figures,
aggregate tables and a text report into a NEW folder.  No file of the runs is opened
for writing: the script is purely read-only on the artefacts.

    python compare_runs.py [sweep_root] [-o OUT] [--no-recompute] [--reuse-cache]
                           [--runs NAME ...] [--visual-frame N]
                           [--skybox-gt GT.exr] [--no-spectrum-frames]

The methodological point, which is why this script exists:

  Every run is by construction the minimum of its OWN loss on this data.
  Ranking them with the MSE rewards the runs trained with mse, with the MAE those
  trained with l1, and the relMSE those trained with rel_mse_raw.  A third-party
  arbiter is needed, so those three metrics are EXCLUDED from the matrix and from the
  rankings (they stay in the raw CSVs): marking them is not enough, because a
  self-evaluated cell weighs the same at a glance.  The headline metric is the
  SMAPE, flanked by the mu-law PSNR and the log-RMSE.

  Moreover the NeRF, in this pipeline, is not consumed as an image but as a radiance
  source inside hemispherical integrals (indirect irradiance, specular cones).  There
  the systematic bias accumulates while the noise cancels: that is the reason why the
  signed bias per luminance band is reported next to the per-pixel error, and not as a
  footnote.

  For the same reason the matrix does not hold per-pixel errors alone: the last two
  columns measure the SPECTRUM, i.e. how much the run's distribution of HDR values
  resembles the GT's (W1 distance in decades and mean tonal shift), computed on the
  same set of pixels as the matrix.  A run can have the lowest per-pixel error and the
  most skewed spectrum.  The final section decomposes that same spectrum per channel,
  per frame and on the skybox.
  (originale contro quelle bakate dai NeRF).

Formulas replicated from other modules are flagged in the comments with their source
file, so that a future divergence is findable: the script is standalone by choice (it
imports neither nerf.metrics nor images_generator) so as not to depend on the pipeline
while the thesis is in progress.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ──────────────────────────────────────────────────────────────────────────────
# Costanti
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_ROOT = "D:/tesi_output/sweep_nerf_activation_loss_decay_find_better_nerf"

# Rec.709 -- identical to _LUMA_COEFF in nerf/metrics.py
LUMA_COEFF = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

EPS = 1e-3          # same eps as highlight_percentile_error (nerf/metrics.py)
PSNR_EPS = 1e-10    # same addend used in images_generator.py:2609
MU = 5000.0         # mu-law, Kalantari & Ramamoorthi 2017

# Luminance grid: 15 bands of 1/3 decade between 1e-3 and 1e2, plus underflow and
# overflow.  The 1/3-decade steps land EXACTLY on 0.1, 1 and 10, so the four
# partitions (shadow/midtone/highlight/extreme) are sums of contiguous bands and no
# separate accumulators are needed.
DEC_LO_EXP, DEC_HI_EXP = -3.0, 2.0
DEC_PER_DECADE = 3
N_DEC_INNER = int((DEC_HI_EXP - DEC_LO_EXP) * DEC_PER_DECADE)   # 15
N_DEC = N_DEC_INNER + 2                                          # + underflow/overflow
DEC_EDGES = 10.0 ** np.linspace(DEC_LO_EXP, DEC_HI_EXP, N_DEC_INNER + 1)

# Partitions: indices of the bands that make them up (bin 0 = underflow, 16 = overflow)
BANDS: dict[str, tuple[int, int]] = {
    "shadow":    (0, 6),     # L <= 0.1
    "midtone":   (7, 9),     # 0.1 < L <= 1
    "highlight": (10, 12),   # 1 < L <= 10
    "extreme":   (13, 16),   # L > 10
}
BAND_ORDER = ("shadow", "midtone", "highlight", "extreme")
# Headline binary split
SPLIT2 = {"rest": (0, 9), "highlight_all": (10, 16)}

PIXEL_SETS = ("full", "fg", "bg")

# Histogram of the log-ratios, for the median pred/gt without sampling
NR = 201
RATIO_LO, RATIO_HI = -2.0, 2.0
RATIO_STEP = (RATIO_HI - RATIO_LO) / (NR - 1)

# ── Value spectrum ───────────────────────────────────────────────────────────
# Distribution of the HDR values (not of the errors): it says whether a run reproduces
# the shape of the GT's histogram, which no per-pixel metric measures.
# Two runs with the same MSE can have very different spectra, and in the pipeline the
# NeRF ends up inside hemispherical integrals, where how the energy is distributed
# matters.  Log-spaced grid like DEC_EDGES but finer and wider: 20 bins per decade
# from 1e-5 to 1e3, plus underflow and overflow.
SPEC_LO_EXP, SPEC_HI_EXP = -5.0, 3.0
SPEC_PER_DECADE = 20
SPEC_STEP = 1.0 / SPEC_PER_DECADE                                  # dex per bin
NS_INNER = int((SPEC_HI_EXP - SPEC_LO_EXP) * SPEC_PER_DECADE)      # 160
NS = NS_INNER + 2                                                  # + underflow/overflow
SPEC_EDGES = 10.0 ** np.linspace(SPEC_LO_EXP, SPEC_HI_EXP, NS_INNER + 1)
# centre of each bin in log10; bins 0 and NS-1 (open) take the same nominal width as
# the others: an approximation, but it keeps them on the axis
SPEC_CENTERS = SPEC_LO_EXP + (np.arange(NS) - 0.5) * SPEC_STEP
# upper end of each bin in log10, for the quantiles: bin 0 -> SPEC_LO_EXP,
# bin k -> SPEC_LO_EXP + k*step, last bin -> SPEC_HI_EXP + step (open)
SPEC_UPPER = SPEC_LO_EXP + np.arange(NS) * SPEC_STEP

# norm = ||RGB||_2 of the pixel, not the Rec.709 luminance: it is the same quantity
# the heatmaps of nerf/metrics.py use, and it does not privilege green
SPEC_CHANNELS = ("R", "G", "B", "norm")
SPEC_CH_COLORS = ("#E63946", "#2A9D8F", "#457B9D", "#9B59B6")   # as _RGB_HIST_COLORS
N_SPEC_CH = len(SPEC_CHANNELS)
NORM_CH = N_SPEC_CH - 1   # index of the "norm" channel

# Quantities accumulated per cell (set x band).  All are means of per-sample
# quantities, hence additive over the partitions: that is what makes the error-budget
# decomposition exact.
QUANTS = ("sq", "absd", "smape", "rel", "logsq", "musq",
          "tmclip_sq", "tmrein_sq", "p", "g")

# Metrics shown in the matrix, with their direction and family.
#
# No column coincides with one of the sweep's losses: linear MSE/PSNR, MAE and relMSE
# were removed precisely because each is the loss of one of the runs, which is
# therefore its minimum by construction.  They stay in the raw CSVs
# (metrics_global.csv), where nobody reads them as a ranking.
@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    higher_is_better: bool
    fmt: str = "{:.4f}"
    # True = measures the DISTRIBUTION of the values, not the per-pixel error.  It is
    # there to separate the two families in the figures and in the report: they are
    # different questions, and the interesting case is when they contradict each other.
    spectrum: bool = False


METRIC_SPECS = (
    MetricSpec("smape",            "SMAPE",                 False, "{:.4f}"),
    MetricSpec("mu_psnr",          "mu-PSNR [dB]",          True,  "{:.2f}"),
    MetricSpec("log_rmse",         "log-RMSE",              False, "{:.4f}"),
    MetricSpec("psnr_tm_clip",     "PSNR tm-clip [dB]",     True,  "{:.2f}"),
    MetricSpec("psnr_tm_reinhard", "PSNR tm-Reinhard [dB]", True,  "{:.2f}"),
    # label with no vertical bars: it ends up inside a markdown table
    MetricSpec("abs_energy_bias",  "abs. energy bias",      False, "{:.5f}"),
    MetricSpec("spec_w1",          "W1 spectrum [dex]",     False, "{:.4f}", spectrum=True),
    MetricSpec("spec_absdmean",    "abs. dmean [dex]",      False, "{:.4f}", spectrum=True),
)

# index of the first distribution column: the separator in the figures
FIRST_SPEC_COL = next(j for j, sp in enumerate(METRIC_SPECS) if sp.spectrum)

# Colours: the colour family identifies the loss, the shade and the stroke
# identify the activation (exp = dark solid, softplus = light dashed)
COLORS = {
    ("exp", "l1"):               "#12436D",
    ("softplus", "l1"):          "#6BAED6",
    ("exp", "mse"):              "#A63603",
    ("softplus", "mse"):         "#FD8D3C",
    ("exp", "rel_mse_raw"):      "#0F6B3C",
    ("softplus", "rel_mse_raw"): "#74C476",
}
FALLBACK_COLORS = ["#12436D", "#A63603", "#0F6B3C", "#6BAED6", "#FD8D3C", "#74C476"]


# ──────────────────────────────────────────────────────────────────────────────
# IO
# ──────────────────────────────────────────────────────────────────────────────

def load_exr_rgb(path: str) -> np.ndarray:
    """EXR -> (H, W, 3) float32.  Replica of _load_image_hw3_native
    (images_generator.py:482), EXR branch."""
    import OpenEXR, Imath
    exr = OpenEXR.InputFile(path)
    dw = exr.header()["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    chs = exr.header()["channels"]
    if "R" in chs and "G" in chs and "B" in chs:
        r, g, b = (np.frombuffer(exr.channel(c, pt), dtype=np.float32).reshape(h, w)
                   for c in ("R", "G", "B"))
    else:
        key = next(iter(chs))
        r = g = b = np.frombuffer(exr.channel(key, pt), dtype=np.float32).reshape(h, w)
    return np.stack([r, g, b], axis=-1)


def load_mask_bool(path: str) -> np.ndarray:
    """Mask PNG -> (H, W) bool.  Threshold 127, as _load_mask_bool in the pipeline."""
    from PIL import Image
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 127


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def col_float(rows: list[dict], key: str) -> np.ndarray:
    out = np.empty(len(rows), dtype=np.float64)
    for i, r in enumerate(rows):
        try:
            out[i] = float(r[key])
        except (TypeError, ValueError):
            out[i] = np.nan
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Run discovery
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Run:
    key: str                 # name of the run's folder
    run_dir: Path
    scene_dir: Path
    train_csv: Path
    iter_dir: Path | None
    activation: str
    loss: str
    decay: float
    n_frames: int = 0
    color: str = "#333333"
    linestyle: str = "-"

    @property
    def label(self) -> str:
        return f"{self.activation}/{self.loss}"


def discover_runs(root: Path, only: list[str] | None) -> list[Run]:
    runs: list[Run] = []
    for csv_path in sorted(root.glob("*/*/nerf_train/training_metrics.csv")):
        scene_dir = csv_path.parents[1]
        run_dir = csv_path.parents[2]
        if only and run_dir.name not in only:
            continue
        rows = read_csv_rows(csv_path)
        if not rows:
            print(f"  [skip] {run_dir.name}: training_metrics.csv vuoto")
            continue
        last = rows[-1]
        # Activation/loss/decay are read from the CSV columns, not from the folder
        # name: the last four columns are there for exactly that (train.py:65-68).
        iter_dirs = sorted((scene_dir / "nerf_render_images").glob("iter_*"))
        runs.append(Run(
            key=run_dir.name,
            run_dir=run_dir,
            scene_dir=scene_dir,
            train_csv=csv_path,
            iter_dir=iter_dirs[-1] if iter_dirs else None,
            activation=last.get("rgb_activation", "?"),
            loss=last.get("loss_type", "?"),
            decay=float(last.get("lr_decay_factor", "nan") or "nan"),
        ))

    for i, r in enumerate(runs):
        r.color = COLORS.get((r.activation, r.loss), FALLBACK_COLORS[i % len(FALLBACK_COLORS)])
        r.linestyle = "-" if r.activation == "exp" else "--"
    return runs


# ──────────────────────────────────────────────────────────────────────────────
# Accumulatori
# ──────────────────────────────────────────────────────────────────────────────

N_CELLS = 2 * N_DEC     # set (0=fg, 1=bg) x luminance band


class Accum:
    """Sums per cell (set x band).  All in float64."""

    def __init__(self) -> None:
        self.n = np.zeros(N_CELLS, dtype=np.float64)
        self.s = {q: np.zeros(N_CELLS, dtype=np.float64) for q in QUANTS}
        self.hist = np.zeros(N_CELLS * NR, dtype=np.float64)

    def add(self, other: "Accum") -> None:
        self.n += other.n
        for q in QUANTS:
            self.s[q] += other.s[q]
        self.hist += other.hist


def _cells_for(pixel_set: str, lo: int, hi: int) -> np.ndarray:
    """Cell indices for a pixel set and a range of bands."""
    bands = np.arange(lo, hi + 1)
    if pixel_set == "fg":
        return bands
    if pixel_set == "bg":
        return N_DEC + bands
    return np.concatenate([bands, N_DEC + bands])


def _slice(acc: Accum, pixel_set: str, lo: int, hi: int) -> tuple[float, dict[str, float]]:
    cells = _cells_for(pixel_set, lo, hi)
    n = float(acc.n[cells].sum())
    sums = {q: float(acc.s[q][cells].sum()) for q in QUANTS}
    return n, sums


def _hist_slice(acc: Accum, pixel_set: str, lo: int, hi: int) -> np.ndarray:
    cells = _cells_for(pixel_set, lo, hi)
    h = acc.hist.reshape(N_CELLS, NR)[cells].sum(axis=0)
    return h


def hist_quantile(hist: np.ndarray, q: float) -> float:
    """Quantile of log10(ratio) from a histogram, interpolated inside the bin."""
    total = hist.sum()
    if total <= 0:
        return float("nan")
    cum = np.cumsum(hist)
    target = q * total
    k = int(np.searchsorted(cum, target, side="left"))
    k = min(k, NR - 1)
    prev = cum[k - 1] if k > 0 else 0.0
    frac = (target - prev) / hist[k] if hist[k] > 0 else 0.5
    return RATIO_LO + (k - 0.5 + frac) * RATIO_STEP


# ──────────────────────────────────────────────────────────────────────────────
# Value spectrum: histogram and distances between distributions
# ──────────────────────────────────────────────────────────────────────────────

def spectrum_hist(arr_hw3: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """(H, W, 3) -> (4, NS) counts per log bin: R, G, B, ||RGB||_2.

    Whole frame, no mask.  Bin 0 collects everything below 1e-5 (zero and negatives
    included), the last one everything above 1e3: no pixel is lost, and the sum of the
    histogram is exactly H*W.
    `weights` (H, W) is for the envmap, where the pixels do not have the same solid
    angle; None = weight 1 per pixel.
    """
    a = arr_hw3.astype(np.float32, copy=False)
    vals = (a[..., 0], a[..., 1], a[..., 2], np.sqrt((a * a).sum(-1)))
    w = None if weights is None else weights.reshape(-1).astype(np.float64)
    out = np.zeros((N_SPEC_CH, NS), dtype=np.float64)
    for c, v in enumerate(vals):
        # side="right": index 0 below the first edge, NS-1 above the last.
        # NaNs land in the overflow (searchsorted puts them at the end): they are
        # still counted, so the check on the sum catches them.
        idx = np.clip(np.searchsorted(SPEC_EDGES, v.reshape(-1), side="right"), 0, NS - 1)
        out[c] = np.bincount(idx, weights=w, minlength=NS)
    return out


def spectrum_hist_sets(arr_hw3: np.ndarray, fg: np.ndarray) -> np.ndarray:
    """(H, W, 3) + fg mask -> (3, 4, NS): full, fg, bg in the order of PIXEL_SETS.

    The bin index is computed once and re-binned three times: searchsorted is the
    expensive part, the bincounts are not.  The three histograms are computed
    independently (full unweighted, fg with the mask, bg with its complement) and not
    by difference: that way verify_spectrum's `full == fg + bg` check really has power
    over the mask handling instead of being true by construction.

    Why the separation is needed: the pipeline consumes the NeRF in two different ways,
    the foreground feeds the PBR fit and the background the envmap bake, and they are
    two different radiance distributions.
    """
    a = arr_hw3.astype(np.float32, copy=False)
    vals = (a[..., 0], a[..., 1], a[..., 2], np.sqrt((a * a).sum(-1)))
    w_fg = fg.reshape(-1).astype(np.float64)
    w_bg = 1.0 - w_fg
    out = np.zeros((len(PIXEL_SETS), N_SPEC_CH, NS), dtype=np.float64)
    for c, v in enumerate(vals):
        # side="right" and clip: same convention as spectrum_hist, no pixel lost
        idx = np.clip(np.searchsorted(SPEC_EDGES, v.reshape(-1), side="right"), 0, NS - 1)
        out[0, c] = np.bincount(idx, minlength=NS)
        out[1, c] = np.bincount(idx, weights=w_fg, minlength=NS)
        out[2, c] = np.bincount(idx, weights=w_bg, minlength=NS)
    return out


def spec_density(hist: np.ndarray) -> np.ndarray:
    """Histogram (..., NS) -> density summing to 1 along the last axis."""
    tot = hist.sum(axis=-1, keepdims=True)
    return np.divide(hist, tot, out=np.zeros_like(hist, dtype=np.float64),
                     where=tot > 0)


def spec_w1_dex(h_pred: np.ndarray, h_gt: np.ndarray) -> np.ndarray:
    """Wasserstein-1 between two spectra in the log10 domain, in decades.

    With bins equispaced in log10, W1 = sum(|F_pred - F_gt|) * step: it is the mean
    shift of the quantiles measured in decades.  Zero only when the two distributions
    coincide bin by bin.  No scipy required.
    """
    fp = np.cumsum(spec_density(h_pred), axis=-1)
    fg = np.cumsum(spec_density(h_gt), axis=-1)
    return np.abs(fp - fg).sum(axis=-1) * SPEC_STEP


def spec_dmean_dex(h_pred: np.ndarray, h_gt: np.ndarray) -> np.ndarray:
    """Signed version: mean(log10 pred) - mean(log10 gt), in decades.
    Positive = spectrum shifted towards the high values (brighter run)."""
    dp = spec_density(h_pred) @ SPEC_CENTERS
    dg = spec_density(h_gt) @ SPEC_CENTERS
    return dp - dg


def spec_quantile(hist: np.ndarray, qs: np.ndarray) -> np.ndarray:
    """Quantiles of the spectrum, in log10, interpolated on the bin edges."""
    cdf = np.cumsum(spec_density(hist))
    return np.interp(qs, cdf, SPEC_UPPER)


METRIC_KEYS = ("n", "mse_lin", "psnr_lin", "mae_lin", "smape", "rel_mse_gt",
               "log_rmse", "mu_psnr", "psnr_tm_clip", "psnr_tm_reinhard",
               "residual_mean", "energy_bias", "abs_energy_bias", "mean_gt")


def metrics_from(n: float, s: dict[str, float], mu_scale: float) -> dict[str, float]:
    """Turn the sums of a cell (or a union of cells) into metrics."""
    if n <= 0:
        out = {k: float("nan") for k in METRIC_KEYS}
        out["n"] = 0.0
        return out
    mse = s["sq"] / n
    mae = s["absd"] / n
    smape = s["smape"] / n
    rel = s["rel"] / n
    logrmse = float(np.sqrt(s["logsq"] / n))
    mu_mse = s["musq"] / n
    tmc = s["tmclip_sq"] / n
    tmr = s["tmrein_sq"] / n
    energy_bias = (s["p"] - s["g"]) / s["g"] if s["g"] > 0 else float("nan")
    return {
        "n": n,
        "mse_lin": mse,
        "psnr_lin": -10.0 * np.log10(mse + PSNR_EPS),
        "mae_lin": mae,
        "smape": smape,
        "rel_mse_gt": rel,
        "log_rmse": logrmse,
        "mu_psnr": -10.0 * np.log10(mu_mse + PSNR_EPS),
        "psnr_tm_clip": -10.0 * np.log10(tmc + PSNR_EPS),
        "psnr_tm_reinhard": -10.0 * np.log10(tmr + PSNR_EPS),
        "residual_mean": (s["p"] - s["g"]) / n,
        "energy_bias": energy_bias,
        "abs_energy_bias": abs(energy_bias),
        "mean_gt": s["g"] / n,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Per-frame GT context (computed once, reused by every run)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GtCtx:
    g: np.ndarray            # (H, W, 3) float32
    cell_flat: np.ndarray    # (H*W*3,) int32  -> set*N_DEC + band
    n_cell: np.ndarray       # (N_CELLS,) conteggi
    log_g: np.ndarray        # (H, W, 3) float32  log(g + EPS)
    t_g: np.ndarray          # (H, W, 3) float32  mu-law of the GT
    clip_g: np.ndarray
    rein_g: np.ndarray
    den_rel: np.ndarray      # (g + EPS)^2
    sel99: np.ndarray        # (H, W) bool, highlight p99 su fg
    sel999: np.ndarray
    fg: np.ndarray           # (H, W) bool


def mu_law(x: np.ndarray, scale: float) -> np.ndarray:
    """T(x) = log(1 + mu * clip(x/scale, 0, 1)) / log(1 + mu)."""
    xs = np.clip(x / scale, 0.0, 1.0)
    return np.log1p(MU * xs, dtype=np.float32) / np.float32(np.log1p(MU))


def build_gt_ctx(gt: np.ndarray, fg: np.ndarray, mu_scale: float) -> GtCtx:
    lum = (gt * LUMA_COEFF).sum(-1)                      # (H, W)
    # band: 0 = underflow, 1..15 = 1/3 decade, 16 = overflow
    band = np.searchsorted(DEC_EDGES, lum, side="right").astype(np.int32)
    band = np.clip(band, 0, N_DEC - 1)
    cell = np.where(fg, band, band + N_DEC).astype(np.int32)
    cell_flat = np.repeat(cell.ravel(), 3)
    n_cell = np.bincount(cell_flat, minlength=N_CELLS).astype(np.float64)

    # highlight thresholds identical to highlight_percentile_error (nerf/metrics.py):
    # percentile of the GT luminance computed on the foreground pixels alone
    lum_fg = lum[fg]
    if lum_fg.size:
        t99 = float(np.percentile(lum_fg, 99.0))
        t999 = float(np.percentile(lum_fg, 99.9))
        sel99 = fg & (lum >= t99)
        sel999 = fg & (lum >= t999)
    else:
        sel99 = sel999 = np.zeros_like(fg)

    return GtCtx(
        g=gt,
        cell_flat=cell_flat,
        n_cell=n_cell,
        log_g=np.log(gt + np.float32(EPS)),
        t_g=mu_law(gt, mu_scale),
        clip_g=np.clip(gt, 0.0, 1.0),
        rein_g=gt / (1.0 + gt),
        den_rel=(gt + np.float32(EPS)) ** 2,
        sel99=sel99,
        sel999=sel999,
        fg=fg,
    )


def accumulate_frame(pred: np.ndarray, ctx: GtCtx, mu_scale: float) -> Accum:
    """Per-cell sums for a single frame of a single run."""
    acc = Accum()
    acc.n = ctx.n_cell.copy()

    g = ctx.g
    d = pred - g
    idx = ctx.cell_flat

    def bc(w: np.ndarray) -> np.ndarray:
        return np.bincount(idx, weights=w.reshape(-1), minlength=N_CELLS)

    acc.s["sq"] = bc(d * d)
    absd = np.abs(d)
    acc.s["absd"] = bc(absd)
    # SMAPE: symmetric, scale-invariant, bounded in [0, 1]
    acc.s["smape"] = bc(absd / (np.abs(pred) + np.abs(g) + np.float32(EPS)))
    acc.s["rel"] = bc((d * d) / ctx.den_rel)
    dlog = np.log(np.maximum(pred, 0.0) + np.float32(EPS)) - ctx.log_g
    acc.s["logsq"] = bc(dlog * dlog)
    dmu = mu_law(pred, mu_scale) - ctx.t_g
    acc.s["musq"] = bc(dmu * dmu)
    dtc = np.clip(pred, 0.0, 1.0) - ctx.clip_g
    acc.s["tmclip_sq"] = bc(dtc * dtc)
    dtr = pred / (1.0 + pred) - ctx.rein_g
    acc.s["tmrein_sq"] = bc(dtr * dtr)
    acc.s["p"] = bc(pred)
    acc.s["g"] = bc(g)

    # histogram of the log10(ratio), for the median pred/gt without sampling
    lr = np.log10((pred + np.float32(EPS)) / (g + np.float32(EPS)))
    rb = np.clip(np.round((lr - RATIO_LO) / RATIO_STEP), 0, NR - 1).astype(np.int32)
    acc.hist = np.bincount(idx * NR + rb.reshape(-1),
                           minlength=N_CELLS * NR).astype(np.float64)
    return acc


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: statistics on the GTs (global constants shared by every run)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GtStats:
    mu_scale: float
    per_frame_p999: np.ndarray
    per_frame_max: np.ndarray
    band_frac: dict[str, float]
    fg_frac: float
    hl_in_fg: float          # share of the highlights (L>1) that falls in the foreground
    percentiles: dict[float, float]
    # (F, 3, 4, NS): frame x pixel set (PIXEL_SETS) x canale x bin
    spec_gt: np.ndarray = field(
        default_factory=lambda: np.zeros((0, len(PIXEL_SETS), N_SPEC_CH, NS)))
    n_pixels: int = 0        # pixels per frame, for the histogram check


def gt_prepass(gt_paths: list[Path], mask_paths: list[Path]) -> GtStats:
    print(f"[1/3] Pre-pass over the {len(gt_paths)} GTs (global constants)...")
    samples: list[np.ndarray] = []
    p999 = np.zeros(len(gt_paths))
    gmax = np.zeros(len(gt_paths))
    n_cell_tot = np.zeros(N_CELLS)
    # GT spectrum: computed here because this pass already loads every GT and its mask,
    # so the three pixel sets do not cost one extra EXR
    spec_gt = np.zeros((len(gt_paths), len(PIXEL_SETS), N_SPEC_CH, NS), dtype=np.float64)
    n_pixels = 0
    t0 = time.perf_counter()
    for i, (gp, mp) in enumerate(zip(gt_paths, mask_paths)):
        g = load_exr_rgb(str(gp))
        fg = load_mask_bool(str(mp))
        spec_gt[i] = spectrum_hist_sets(g, fg)
        n_pixels = g.shape[0] * g.shape[1]
        lum = (g * LUMA_COEFF).sum(-1)
        p999[i] = float(np.percentile(lum, 99.9))
        gmax[i] = float(lum.max())
        band = np.clip(np.searchsorted(DEC_EDGES, lum, side="right"), 0, N_DEC - 1)
        cell = np.where(fg, band, band + N_DEC).astype(np.int32)
        n_cell_tot += np.bincount(cell.ravel(), minlength=N_CELLS)
        # deterministic subsample (no RNG): fixed stride
        flat = lum.ravel()
        stride = max(1, flat.size // 50_000)
        samples.append(flat[::stride].copy())
        print(f"\r      frame {i + 1}/{len(gt_paths)}", end="", flush=True)
    print(f"   ({time.perf_counter() - t0:.1f}s)")

    v = np.concatenate(samples)
    pct = {p: float(np.percentile(v, p)) for p in (0.1, 1, 25, 50, 75, 90, 99, 99.9, 99.99)}
    mu_scale = max(pct[99.99], 1e-3)

    n_tot = n_cell_tot.sum()
    band_frac = {}
    for name, (lo, hi) in BANDS.items():
        cells = _cells_for("full", lo, hi)
        band_frac[name] = float(n_cell_tot[cells].sum() / n_tot)
    fg_frac = float(n_cell_tot[:N_DEC].sum() / n_tot)
    hl_cells_fg = _cells_for("fg", *SPLIT2["highlight_all"])
    hl_cells_all = _cells_for("full", *SPLIT2["highlight_all"])
    hl_tot = n_cell_tot[hl_cells_all].sum()
    hl_in_fg = float(n_cell_tot[hl_cells_fg].sum() / hl_tot) if hl_tot > 0 else float("nan")

    print(f"      mu-law scale X = p99.99 of the GT luminance = {mu_scale:.4f}")
    print(f"      quote: " + ", ".join(f"{k} {100 * v_:.2f}%" for k, v_ in band_frac.items()))
    print(f"      foreground {100 * fg_frac:.1f}% of the pixels, "
          f"holding {100 * hl_in_fg:.1f}% of the highlights (L>1)")
    return GtStats(mu_scale, p999, gmax, band_frac, fg_frac, hl_in_fg, pct,
                   spec_gt=spec_gt, n_pixels=n_pixels)


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: per-run metrics
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    run: Run
    acc: Accum
    per_frame: list[dict] = field(default_factory=list)
    # (F, 3, 4, NS), same layout as GtStats.spec_gt
    spec: np.ndarray = field(
        default_factory=lambda: np.zeros((0, len(PIXEL_SETS), N_SPEC_CH, NS)))


def compute_all(runs: list[Run], gt_paths: list[Path], mask_paths: list[Path],
                stats: GtStats) -> dict[str, RunResult]:
    results = {r.key: RunResult(
                   run=r, acc=Accum(),
                   spec=np.zeros((len(gt_paths), len(PIXEL_SETS), N_SPEC_CH, NS)))
               for r in runs}
    n_frames = len(gt_paths)
    total = n_frames * len(runs)
    done = 0
    t0 = time.perf_counter()
    print(f"[2/3] Ricalcolo metriche: {n_frames} frame x {len(runs)} run "
          f"= {total} images...")

    for i, (gp, mp) in enumerate(zip(gt_paths, mask_paths)):
        gt = load_exr_rgb(str(gp))
        fg = load_mask_bool(str(mp))
        ctx = build_gt_ctx(gt, fg, stats.mu_scale)

        for r in runs:
            pred = load_exr_rgb(str(r.iter_dir / f"frame_{i:03d}_pred.exr"))
            if pred.shape != gt.shape:
                raise ValueError(f"{r.key} frame {i}: shape {pred.shape} != GT {gt.shape}")
            facc = accumulate_frame(pred, ctx, stats.mu_scale)
            results[r.key].acc.add(facc)
            # spectrum on the three pixel sets: the mask is already at hand, no extra EXR
            results[r.key].spec[i] = spectrum_hist_sets(pred, fg)

            row = {"run": r.key, "frame": i}
            for ps in PIXEL_SETS:
                n, s = _slice(facc, ps, 0, N_DEC - 1)
                m = metrics_from(n, s, stats.mu_scale)
                for k in ("psnr_lin", "smape", "mu_psnr", "psnr_tm_clip",
                          "psnr_tm_reinhard", "energy_bias", "mae_lin"):
                    row[f"{k}_{ps}"] = m[k]
            # highlight/rest on the complete set
            for name, (lo, hi) in SPLIT2.items():
                n, s = _slice(facc, "full", lo, hi)
                m = metrics_from(n, s, stats.mu_scale)
                row[f"smape_{name}"] = m["smape"]
                row[f"energy_bias_{name}"] = m["energy_bias"]
            # relative error on the highlights, definition from nerf/metrics.py
            for tag, sel in (("p99", ctx.sel99), ("p999", ctx.sel999)):
                if sel.any():
                    ps_, gs_ = pred[sel], gt[sel]
                    row[f"rel_err_{tag}"] = float(
                        (np.abs(ps_ - gs_) / (np.abs(gs_) + EPS)).mean())
                else:
                    row[f"rel_err_{tag}"] = float("nan")
            results[r.key].per_frame.append(row)

            done += 1
            if done % 5 == 0 or done == total:
                el = time.perf_counter() - t0
                eta = el / done * (total - done)
                print(f"\r      {done}/{total}  ({el:.0f}s, ETA {eta:.0f}s)",
                      end="", flush=True)
        del ctx, gt, fg
    print()
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Result cache
# ──────────────────────────────────────────────────────────────────────────────
#
# Recomputing reads 420 EXRs of 24 MB and costs a few minutes, whereas tweaking a
# figure costs seconds.  The accumulators are a few thousand floats, so they are saved
# whole: iterating on the figures then does not pay the I/O cost every time.

CACHE_NAME = "_cache_metrics.npz"
# v2 = the spectra were added (per run and for the GT).  A v1 cache does not have the
# matching keys: it has to be ignored rather than blow up with a KeyError.
# v3 = the spectra have one axis more (pixel set): same key, different shape, so this
# one too has to be ignored and not re-read.
CACHE_VERSION = 3


def _spec_grid() -> dict:
    return {"lo": SPEC_LO_EXP, "hi": SPEC_HI_EXP, "per_decade": SPEC_PER_DECADE,
            "channels": list(SPEC_CHANNELS), "pixel_sets": list(PIXEL_SETS)}


def save_cache(out: Path, results: dict[str, RunResult], stats: GtStats) -> None:
    blob: dict[str, np.ndarray] = {}
    for key, res in results.items():
        blob[f"{key}//n"] = res.acc.n
        blob[f"{key}//hist"] = res.acc.hist
        blob[f"{key}//spec"] = res.spec
        for q in QUANTS:
            blob[f"{key}//s//{q}"] = res.acc.s[q]
    blob["__spec_gt__"] = stats.spec_gt
    meta = {
        "version": CACHE_VERSION,
        "spec_grid": _spec_grid(),
        "runs": list(results.keys()),
        "per_frame": {k: v.per_frame for k, v in results.items()},
        "stats": {
            "mu_scale": stats.mu_scale,
            "per_frame_p999": stats.per_frame_p999.tolist(),
            "per_frame_max": stats.per_frame_max.tolist(),
            "band_frac": stats.band_frac,
            "fg_frac": stats.fg_frac,
            "hl_in_fg": stats.hl_in_fg,
            "percentiles": {str(k): v for k, v in stats.percentiles.items()},
            "n_pixels": stats.n_pixels,
        },
    }
    blob["__meta__"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    np.savez_compressed(out / CACHE_NAME, **blob)


def load_cache(out: Path, runs: list[Run]) -> tuple[dict[str, RunResult], GtStats] | None:
    path = out / CACHE_NAME
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    meta = json.loads(bytes(z["__meta__"]).decode("utf-8"))
    by_key = {r.key: r for r in runs}
    if set(meta["runs"]) != set(by_key):
        print(f"  cache present but for different runs ({meta['runs']}), ignoring it")
        return None
    if meta.get("version", 1) != CACHE_VERSION:
        print(f"  cache of version {meta.get('version', 1)} (current {CACHE_VERSION}), "
              "ignoring it: recomputing from the EXRs")
        return None
    if meta.get("spec_grid") != _spec_grid():
        print("  cache with a different spectrum grid, ignoring it: recomputing")
        return None
    results: dict[str, RunResult] = {}
    for key in meta["runs"]:
        acc = Accum()
        acc.n = z[f"{key}//n"]
        acc.hist = z[f"{key}//hist"]
        for q in QUANTS:
            acc.s[q] = z[f"{key}//s//{q}"]
        results[key] = RunResult(run=by_key[key], acc=acc,
                                 per_frame=meta["per_frame"][key],
                                 spec=z[f"{key}//spec"])
    st = meta["stats"]
    stats = GtStats(
        mu_scale=st["mu_scale"],
        per_frame_p999=np.array(st["per_frame_p999"]),
        per_frame_max=np.array(st["per_frame_max"]),
        band_frac=st["band_frac"],
        fg_frac=st["fg_frac"],
        hl_in_fg=st["hl_in_fg"],
        percentiles={float(k): v for k, v in st["percentiles"].items()},
        spec_gt=z["__spec_gt__"],
        n_pixels=int(st.get("n_pixels", 0)),
    )
    return results, stats


# ──────────────────────────────────────────────────────────────────────────────
# Check against the existing artefacts
# ──────────────────────────────────────────────────────────────────────────────

def verify_against_artifacts(results: dict[str, RunResult], tol: float = 1e-3) -> list[str]:
    """The PSNR recomputed on `full` has to match the `psnr` column of
    metrics_per_frame.csv, and the tonemap-clip PSNR on `fg` that of psnr_tonemap_clip.
    It is the proof that loader, frame order and pixel set match those the pipeline
    used (images_generator.py:2609 and :2628)."""
    lines = []
    ok = True
    for key, res in results.items():
        csv_path = res.run.iter_dir / "metrics_per_frame.csv"
        if not csv_path.exists():
            lines.append(f"  {key:26s} metrics_per_frame.csv absent, check skipped")
            continue
        rows = read_csv_rows(csv_path)
        ref_psnr = col_float(rows, "psnr")
        ref_tm = col_float(rows, "psnr_tonemap_clip")
        got_psnr = np.array([r["psnr_lin_full"] for r in res.per_frame])
        got_tm = np.array([r["psnr_tm_clip_fg"] for r in res.per_frame])
        n = min(len(ref_psnr), len(got_psnr))
        d1 = float(np.nanmax(np.abs(ref_psnr[:n] - got_psnr[:n])))
        d2 = float(np.nanmax(np.abs(ref_tm[:n] - got_tm[:n])))
        flag = "OK " if (d1 <= tol and d2 <= tol) else "FAIL"
        if flag == "FAIL":
            ok = False
        lines.append(f"  [{flag}] {key:26s} max|dPSNR|={d1:.2e} dB   "
                     f"max|dPSNR_tm_fg|={d2:.2e} dB")
    if not ok:
        lines.append("  WARNING: deviation beyond tolerance. Loader, frame order or "
                     "pixel set do not match the pipeline.")
    return lines


def verify_decomposition(results: dict[str, RunResult], mu_scale: float) -> list[str]:
    """The recomposition sum_p (n_p/n_tot)*M_p has to reproduce M_tot, and the sum of
    the err_shares has to be 1.  It is what makes the sentence readable.
    'the highlights account for X% of the error' readable."""
    lines = []
    worst = 0.0
    for key, res in results.items():
        for ps in PIXEL_SETS:
            n_tot, s_tot = _slice(res.acc, ps, 0, N_DEC - 1)
            tot = metrics_from(n_tot, s_tot, mu_scale)
            for mk, sk in (("mse_lin", "sq"), ("mae_lin", "absd"), ("smape", "smape")):
                recon = 0.0
                share = 0.0
                for lo, hi in BANDS.values():
                    n_b, s_b = _slice(res.acc, ps, lo, hi)
                    if n_b <= 0:
                        continue
                    recon += (n_b / n_tot) * (s_b[sk] / n_b)
                    share += s_b[sk] / s_tot[sk] if s_tot[sk] > 0 else 0.0
                rel = abs(recon - tot[mk]) / max(abs(tot[mk]), 1e-30)
                worst = max(worst, rel, abs(share - 1.0))
    lines.append(f"  [{'OK ' if worst < 1e-6 else 'FAIL'}] decomposition by band: "
                 f"maximum relative error {worst:.2e} (expected < 1e-6)")
    return lines


# ──────────────────────────────────────────────────────────────────────────────
# Scrittura tabelle
# ──────────────────────────────────────────────────────────────────────────────

def write_metrics_global(out: Path, results: dict[str, RunResult], mu_scale: float) -> None:
    fields = ["run", "activation", "loss", "decay", "pixel_set", "n_samples",
              "smape", "mu_psnr", "log_rmse", "psnr_lin", "mse_lin", "mae_lin",
              "rel_mse_gt", "psnr_tm_clip", "psnr_tm_reinhard",
              "residual_mean", "energy_bias", "abs_energy_bias",
              "smape_highlight_all", "smape_rest",
              "energy_bias_highlight_all", "energy_bias_rest"]
    with open(out / "metrics_global.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for key, res in results.items():
            r = res.run
            for ps in PIXEL_SETS:
                n, s = _slice(res.acc, ps, 0, N_DEC - 1)
                m = metrics_from(n, s, mu_scale)
                row = {"run": key, "activation": r.activation, "loss": r.loss,
                       "decay": r.decay, "pixel_set": ps, "n_samples": int(n)}
                for k in ("smape", "mu_psnr", "log_rmse", "psnr_lin", "mse_lin",
                          "mae_lin", "rel_mse_gt", "psnr_tm_clip", "psnr_tm_reinhard",
                          "residual_mean", "energy_bias", "abs_energy_bias"):
                    row[k] = f"{m[k]:.8g}"
                for name, (lo, hi) in SPLIT2.items():
                    nb, sb = _slice(res.acc, ps, lo, hi)
                    mb = metrics_from(nb, sb, mu_scale)
                    row[f"smape_{name}"] = f"{mb['smape']:.8g}"
                    row[f"energy_bias_{name}"] = f"{mb['energy_bias']:.8g}"
                w.writerow(row)


def band_table(res: RunResult, pixel_set: str, mu_scale: float) -> list[dict]:
    """Rows per (band) of a run: error level + share of the budget."""
    n_tot, s_tot = _slice(res.acc, pixel_set, 0, N_DEC - 1)
    rows = []
    for name, (lo, hi) in BANDS.items():
        n_b, s_b = _slice(res.acc, pixel_set, lo, hi)
        m = metrics_from(n_b, s_b, mu_scale)
        h = _hist_slice(res.acc, pixel_set, lo, hi)
        rows.append({
            "band": name,
            "n_frac": (n_b / n_tot) if n_tot else float("nan"),
            "smape": m["smape"],
            "mae": m["mae_lin"],
            "mse": m["mse_lin"],
            "median_ratio": 10.0 ** hist_quantile(h, 0.5),
            "rel_bias_signed": m["energy_bias"],
            "err_share_mse": (s_b["sq"] / s_tot["sq"]) if s_tot["sq"] > 0 else float("nan"),
            "err_share_mae": (s_b["absd"] / s_tot["absd"]) if s_tot["absd"] > 0 else float("nan"),
            "err_share_smape": (s_b["smape"] / s_tot["smape"]) if s_tot["smape"] > 0 else float("nan"),
        })
    return rows


def write_metrics_by_band(out: Path, results: dict[str, RunResult], mu_scale: float) -> None:
    fields = ["run", "activation", "loss", "pixel_set", "band", "n_frac",
              "smape", "mae", "mse", "median_ratio", "rel_bias_signed",
              "err_share_mse", "err_share_mae", "err_share_smape"]
    with open(out / "metrics_by_band.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for key, res in results.items():
            for ps in PIXEL_SETS:
                for row in band_table(res, ps, mu_scale):
                    w.writerow({"run": key, "activation": res.run.activation,
                                "loss": res.run.loss, "pixel_set": ps,
                                **{k: (f"{v:.8g}" if isinstance(v, float) else v)
                                   for k, v in row.items()}})


def decade_table(res: RunResult, pixel_set: str, mu_scale: float) -> list[dict]:
    rows = []
    for b in range(N_DEC):
        n_b, s_b = _slice(res.acc, pixel_set, b, b)
        if n_b <= 0:
            continue
        m = metrics_from(n_b, s_b, mu_scale)
        h = _hist_slice(res.acc, pixel_set, b, b)
        lo = 0.0 if b == 0 else DEC_EDGES[b - 1]
        hi = float("inf") if b == N_DEC - 1 else DEC_EDGES[b]
        rows.append({
            "bin": b, "lum_lo": lo, "lum_hi": hi,
            "lum_center": float(np.sqrt(max(lo, 1e-4) * (hi if np.isfinite(hi) else lo * 3))),
            "count": n_b,
            "median_ratio": 10.0 ** hist_quantile(h, 0.5),
            "p25_ratio": 10.0 ** hist_quantile(h, 0.25),
            "p75_ratio": 10.0 ** hist_quantile(h, 0.75),
            "smape": m["smape"],
            "rel_bias_signed": m["energy_bias"],
        })
    return rows


def write_bias_by_decade(out: Path, results: dict[str, RunResult], mu_scale: float) -> None:
    fields = ["run", "pixel_set", "bin", "lum_lo", "lum_hi", "lum_center", "count",
              "median_ratio", "p25_ratio", "p75_ratio", "smape", "rel_bias_signed"]
    with open(out / "bias_by_decade.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for key, res in results.items():
            for ps in PIXEL_SETS:
                for row in decade_table(res, ps, mu_scale):
                    w.writerow({"run": key, "pixel_set": ps,
                                **{k: (f"{v:.8g}" if isinstance(v, float) else v)
                                   for k, v in row.items()}})


def write_per_frame(out: Path, results: dict[str, RunResult]) -> None:
    rows = [r for res in results.values() for r in res.per_frame]
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(out / "metrics_per_frame_all_runs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.8g}" if isinstance(v, float) else v) for k, v in r.items()})


# ──────────────────────────────────────────────────────────────────────────────
# Figure -- helper
# ──────────────────────────────────────────────────────────────────────────────

def _save(fig, path: Path, dpi: int = 150, constrained: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if constrained:
        # keeps the panel titles from landing on the labels of the row above;
        # bbox_inches="tight" trims the borders but does not fix it
        # collisioni interne
        fig.set_layout_engine("constrained")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"      + {path.name}")


def _legend_runs(ax, *_unused, **kw) -> None:
    ax.legend(fontsize=8, framealpha=0.9, **kw)


def rank_of(values: np.ndarray, higher_is_better: bool) -> np.ndarray:
    """Rank 1 = migliore.  NaN in fondo."""
    v = values.copy()
    if higher_is_better:
        v = -v
    order = np.argsort(np.where(np.isnan(v), np.inf, v), kind="stable")
    ranks = np.empty(len(v), dtype=int)
    ranks[order] = np.arange(1, len(v) + 1)
    return ranks


# ──────────────────────────────────────────────────────────────────────────────
# Figure -- dai training_metrics.csv
# ──────────────────────────────────────────────────────────────────────────────

def fig_training(runs: list[Run], figdir: Path) -> None:
    data = {}
    for r in runs:
        rows = read_csv_rows(r.train_csv)
        # defensive dedup: a crash between a display block and a checkpoint can leave
        # duplicate/non-monotone iterations (train.py:92-96)
        seen: dict[int, dict] = {}
        for row in rows:
            try:
                seen[int(float(row["iter"]))] = row
            except (TypeError, ValueError):
                continue
        rows = [seen[k] for k in sorted(seen)]
        data[r.key] = {c: col_float(rows, c) for c in
                       ("iter", "loss", "mse", "psnr_db", "lr", "iters_per_s",
                        "rays_per_s", "acc_fg", "wall_s")}

    # 1. training PSNR
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in runs:
        d = data[r.key]
        ax.plot(d["iter"], d["psnr_db"], color=r.color, ls=r.linestyle, lw=1.4,
                label=r.label)
    ax.set_xlabel("iteration")
    ax.set_ylabel("PSNR [dB] on the training batch")
    ax.set_title("Training PSNR (training batch, not held out)")
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=max(-5.0, min(float(np.nanmin(data[r.key]["psnr_db"]))
                                     for r in runs)))
    _legend_runs(ax, loc="upper left")
    axin = ax.inset_axes([0.52, 0.08, 0.45, 0.42])
    for r in runs:
        d = data[r.key]
        m = d["iter"] >= d["iter"].max() * 0.8
        axin.plot(d["iter"][m], d["psnr_db"][m], color=r.color, ls=r.linestyle, lw=1.2)
    axin.grid(alpha=0.3)
    axin.set_title("last 20% of training", fontsize=8, pad=2)
    axin.tick_params(labelsize=7)
    axin.patch.set_alpha(0.95)
    _save(fig, figdir / "train_psnr.png", constrained=False)

    # 2. loss per type (units not comparable between panels)
    losses = sorted({r.loss for r in runs})
    fig, axes = plt.subplots(1, len(losses), figsize=(4.2 * len(losses), 4), squeeze=False)
    for ax, lt in zip(axes[0], losses):
        for r in [x for x in runs if x.loss == lt]:
            d = data[r.key]
            ax.plot(d["iter"], d["loss"], color=r.color, ls=r.linestyle, lw=1.4,
                    label=r.activation)
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel(f"loss [{lt} units]")
        ax.set_title(lt)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle("Training loss, one panel per loss type "
                 "(values are in the units of each loss and are not comparable across panels)",
                 fontsize=10)
    _save(fig, figdir / "train_loss_by_type.png")

    # 3. training MSE
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in runs:
        d = data[r.key]
        ax.plot(d["iter"], d["mse"], color=r.color, ls=r.linestyle, lw=1.4, label=r.label)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("MSE on the training batch")
    ax.set_title("Training MSE (the mse runs optimise this quantity directly)")
    ax.grid(alpha=0.3, which="both")
    _legend_runs(ax)
    _save(fig, figdir / "train_mse.png")

    # 4. diagnostica
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    panels = [("lr", "learning rate", True), ("acc_fg", "accumulated opacity (fg)", False),
              ("iters_per_s", "iterations / s", False), ("wall_s", "wall clock [s]", False)]
    for ax, (col, lab, logy) in zip(axes.ravel(), panels):
        for r in runs:
            d = data[r.key]
            ax.plot(d["iter"], d[col], color=r.color, ls=r.linestyle, lw=1.3, label=r.label)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Training diagnostics: schedule, geometry convergence, throughput")
    _save(fig, figdir / "train_diagnostics.png")

    # 5. quality at equal time
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in runs:
        d = data[r.key]
        ax.plot(d["wall_s"] / 60.0, d["mse"], color=r.color, ls=r.linestyle, lw=1.4,
                label=r.label)
    ax.set_yscale("log")
    ax.set_xlabel("wall clock [min]")
    ax.set_ylabel("MSE on the training batch")
    ax.set_title("Quality at equal wall-clock time")
    ax.grid(alpha=0.3, which="both")
    _legend_runs(ax)
    _save(fig, figdir / "train_efficiency.png")


def fig_bias_bins_existing(runs: list[Run], figdir: Path) -> None:
    """Direct replot of the bias_bins.csv files already on disk."""
    channels = ["R", "G", "B", "Luma"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    any_data = False
    for ax, ch in zip(axes.ravel(), channels):
        for r in runs:
            if r.iter_dir is None:
                continue
            p = r.iter_dir / "bias_bins.csv"
            if not p.exists():
                continue
            rows = [x for x in read_csv_rows(p) if x["channel"] == ch]
            if not rows:
                continue
            x = col_float(rows, "center_gt")
            y = col_float(rows, "ratio")
            ax.plot(x, y, color=r.color, ls=r.linestyle, marker="o", ms=3, lw=1.2,
                    label=r.label)
            any_data = True
        ax.axhline(1.0, color="k", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("GT value (bin centre)")
        ax.set_ylabel("median pred / GT")
        ax.set_title(ch)
        ax.grid(alpha=0.3, which="both")
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Existing bias_bins.csv replotted: median pred/GT ratio per GT band\n"
                 "(above 1 = over-estimate, below 1 = under-estimate)", fontsize=10)
    if any_data:
        _save(fig, figdir / "bias_bins_existing.png")
    else:
        plt.close(fig)


def fig_existing_per_frame(runs: list[Run], figdir: Path) -> None:
    """Boxplot of the columns already present in metrics_per_frame.csv."""
    cols = [("psnr", "PSNR linear [dB] (full frame)", True),
            ("psnr_tonemap_clip", "PSNR tonemap-clip [dB] (fg only)", True),
            ("rel_err_p99", "relative error above p99 (fg only)", False)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ok = False
    for ax, (col, lab, _hib) in zip(axes, cols):
        vals, labels, colors = [], [], []
        for r in runs:
            if r.iter_dir is None:
                continue
            p = r.iter_dir / "metrics_per_frame.csv"
            if not p.exists():
                continue
            v = col_float(read_csv_rows(p), col)
            v = v[~np.isnan(v)]
            if v.size == 0:
                continue
            vals.append(v)
            labels.append(r.label)
            colors.append(r.color)
            ok = True
        if not vals:
            continue
        bp = ax.boxplot(vals, patch_artist=True, widths=0.6)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.65)
        for med in bp["medians"]:
            med.set_color("k")
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(lab, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Per-frame distributions from the existing metrics_per_frame.csv "
                 "(60 frames per run)", fontsize=10)
    if ok:
        _save(fig, figdir / "quality_boxplots_existing.png")
    else:
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Figure -- dalle metriche ricalcolate
# ──────────────────────────────────────────────────────────────────────────────

def fig_quality_boxplots(results: dict[str, RunResult], runs: list[Run], figdir: Path) -> None:
    cols = [("psnr_lin_full", "PSNR linear [dB], full frame"),
            ("psnr_tm_clip_fg", "PSNR tonemap-clip [dB], foreground"),
            ("smape_full", "SMAPE, full frame (lower is better)")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (col, lab) in zip(axes, cols):
        vals = [np.array([r[col] for r in results[x.key].per_frame]) for x in runs]
        bp = ax.boxplot(vals, patch_artist=True, widths=0.6)
        for patch, x in zip(bp["boxes"], runs):
            patch.set_facecolor(x.color)
            patch.set_alpha(0.65)
        for med in bp["medians"]:
            med.set_color("k")
        ax.set_xticklabels([x.label for x in runs], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(lab, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Per-frame distributions, recomputed consistently on the stated pixel set",
                 fontsize=10)
    _save(fig, figdir / "quality_boxplots.png")


def fig_per_frame_referee(results: dict[str, RunResult], runs: list[Run],
                          stats: GtStats, figdir: Path) -> None:
    order = np.argsort(stats.per_frame_p999)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    for r in runs:
        v = np.array([x["smape_full"] for x in results[r.key].per_frame])
        axes[0].plot(np.arange(len(order)), v[order], color=r.color, ls=r.linestyle,
                     lw=1.3, marker="o", ms=2.5, label=r.label)
    axes[0].set_ylabel("SMAPE (full frame)")
    axes[0].set_title("Referee metric per frame, frames sorted by GT dynamic range\n"
                      "(if the curves move together, the hard frames are hard for everybody)",
                      fontsize=10)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].plot(np.arange(len(order)), stats.per_frame_p999[order], color="k", lw=1.2)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("GT luminance p99.9")
    axes[1].set_xlabel("frames, sorted by GT p99.9")
    axes[1].grid(alpha=0.3, which="both")
    _save(fig, figdir / "per_frame_referee.png")


def _matrix_values(results: dict[str, RunResult], runs: list[Run], pixel_set: str,
                   mu_scale: float, an: "SpectrumAnalysis"
                   ) -> tuple[np.ndarray, np.ndarray]:
    vals = np.zeros((len(runs), len(METRIC_SPECS)))
    ranks = np.zeros_like(vals, dtype=int)
    for i, r in enumerate(runs):
        n, s = _slice(results[r.key].acc, pixel_set, 0, N_DEC - 1)
        m = metrics_from(n, s, mu_scale)
        # the distribution columns come from the spectrum of the SAME pixel set
        m.update(spectrum_metrics(an, r.key, pixel_set))
        for j, sp in enumerate(METRIC_SPECS):
            vals[i, j] = m[sp.key]
    for j, sp in enumerate(METRIC_SPECS):
        ranks[:, j] = rank_of(vals[:, j], sp.higher_is_better)
    return vals, ranks


def fig_metric_matrix(results: dict[str, RunResult], runs: list[Run], mu_scale: float,
                      figdir: Path, pixel_set: str, an: "SpectrumAnalysis") -> None:
    vals, ranks = _matrix_values(results, runs, pixel_set, mu_scale, an)
    cmap = plt.get_cmap("RdYlGn_r")
    fig, ax = plt.subplots(figsize=(1.35 * len(METRIC_SPECS) + 3, 0.75 * len(runs) + 3))
    im = ax.imshow(ranks, cmap="RdYlGn_r", vmin=1, vmax=len(runs), aspect="auto")
    span = max(len(runs) - 1, 1)
    for i, r in enumerate(runs):
        for j, sp in enumerate(METRIC_SPECS):
            # white text on the dark cells (extreme ranks): dark grey on saturated green
            # or red is unreadable
            cr, cg, cb, _ = cmap((ranks[i, j] - 1) / span)
            fgc = "white" if (0.2126 * cr + 0.7152 * cg + 0.0722 * cb) < 0.5 else "black"
            ax.text(j, i - 0.16, sp.fmt.format(vals[i, j]), ha="center", va="center",
                    fontsize=8, color=fgc,
                    fontweight="bold" if ranks[i, j] == 1 else "normal")
            ax.text(j, i + 0.24, f"#{ranks[i, j]}", ha="center", va="center",
                    fontsize=6.5, color=fgc)
    # separator between per-pixel error and distribution: they measure different things
    ax.axvline(FIRST_SPEC_COL - 0.5, color="k", lw=1.6)
    ax.set_xticks(range(len(METRIC_SPECS)))
    ax.set_xticklabels([s.label for s in METRIC_SPECS], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels([r.label for r in runs], fontsize=9)
    ax.set_title(f"Run x metric, pixel set = {pixel_set}. Colour = rank (green best).\n"
                 "No column is one of the sweep's training losses: no cell is self-graded.\n"
                 "Right of the line: distribution of the values on the same pixel set, "
                 "not per-pixel error.", fontsize=10)
    fig.colorbar(im, ax=ax, label="rank", shrink=0.7)
    _save(fig, figdir / f"metric_matrix_{pixel_set}.png")


def fig_rank_bump(results: dict[str, RunResult], runs: list[Run], mu_scale: float,
                  figdir: Path, an: "SpectrumAnalysis") -> None:
    _, ranks = _matrix_values(results, runs, "full", mu_scale, an)
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(METRIC_SPECS))
    for i, r in enumerate(runs):
        ax.plot(x, ranks[i], color=r.color, ls=r.linestyle, lw=2, marker="o", ms=6,
                label=r.label)
    # The distribution columns are on a shaded background instead of labelled: with six
    # curves crossing every rank there is no point in the frame where a label would not
    # cover a line, and above the frame there is the title.
    ax.axvspan(FIRST_SPEC_COL - 0.5, len(METRIC_SPECS) - 0.5, color="#8C8CB4",
               alpha=0.13, lw=0, zorder=0)
    ax.axvline(FIRST_SPEC_COL - 0.5, color="k", lw=1.4, alpha=0.7, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([s.label for s in METRIC_SPECS], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(1, len(runs) + 1))
    ax.invert_yaxis()
    ax.set_ylabel("rank (1 = best)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Rank stability across metrics, none of which is a training loss.\n"
                 "A run that changes rank between columns is telling you the ranking "
                 "depends on the metric, not on the model.\n"
                 "Left of the divider: per-pixel error. Shaded: distribution of the "
                 "values.", fontsize=10)
    _save(fig, figdir / "rank_bump.png")


def fig_bias_by_decade(results: dict[str, RunResult], runs: list[Run], mu_scale: float,
                       figdir: Path) -> None:
    # two rows: log scale on top, because in the darker bands the ratio reaches 20x and
    # would crush everything else onto the line at 1; a narrow linear zoom around 1 at
    # the bottom, which is the regime that matters for the integrals
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for col, ps in enumerate(PIXEL_SETS):
        for row in (0, 1):
            ax = axes[row][col]
            for r in runs:
                rows = decade_table(results[r.key], ps, mu_scale)
                if not rows:
                    continue
                x = np.array([q["lum_center"] for q in rows])
                y = np.array([q["median_ratio"] for q in rows])
                ax.plot(x, y, color=r.color, ls=r.linestyle, lw=1.4, marker="o", ms=3,
                        label=r.label)
            ax.axhline(1.0, color="k", lw=0.9, ls=":")
            for t in (0.1, 1.0, 10.0):
                ax.axvline(t, color="#999999", lw=0.7, ls="-.")
            ax.set_xscale("log")
            ax.grid(alpha=0.3, which="both")
            if row == 0:
                ax.set_yscale("log")
                ax.set_title(f"pixel set: {ps}")
            else:
                ax.set_ylim(0.9, 1.1)
                ax.set_xlabel("GT luminance")
    axes[0][0].set_ylabel("median pred / GT (log scale)")
    axes[1][0].set_ylabel("same, zoom on +/- 10%")
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Multiplicative bias per luminance band. Vertical lines mark the band "
                 "thresholds 0.1 / 1 / 10.\nBias survives hemispherical integration, "
                 "zero-mean noise does not.", fontsize=10)
    _save(fig, figdir / "bias_by_decade.png")


def fig_smape_by_decade(results: dict[str, RunResult], runs: list[Run], mu_scale: float,
                        figdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, ps in zip(axes, PIXEL_SETS):
        for r in runs:
            rows = decade_table(results[r.key], ps, mu_scale)
            if not rows:
                continue
            x = np.array([q["lum_center"] for q in rows])
            y = np.array([q["smape"] for q in rows])
            ax.plot(x, y, color=r.color, ls=r.linestyle, lw=1.4, marker="o", ms=3,
                    label=r.label)
        for t in (0.1, 1.0, 10.0):
            ax.axvline(t, color="#999999", lw=0.7, ls="-.")
        ax.set_xscale("log")
        ax.set_xlabel("GT luminance")
        ax.set_title(f"pixel set: {ps}")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("SMAPE within the band")
    axes[0].legend(fontsize=7)
    fig.suptitle("Where each model errs: SMAPE per luminance band "
                 "(shadows on the left, highlights on the right)", fontsize=10)
    _save(fig, figdir / "smape_by_decade.png")


def fig_highlight_bias(results: dict[str, RunResult], runs: list[Run], mu_scale: float,
                       figdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(runs))
    for ax, ps in zip(axes, ("full", "fg")):
        hl = []
        rest = []
        for r in runs:
            n, s = _slice(results[r.key].acc, ps, *SPLIT2["highlight_all"])
            hl.append(metrics_from(n, s, mu_scale)["energy_bias"])
            n, s = _slice(results[r.key].acc, ps, *SPLIT2["rest"])
            rest.append(metrics_from(n, s, mu_scale)["energy_bias"])
        ax.bar(x - 0.2, rest, 0.38, label="rest (L <= 1)", color="#9ecae1",
               edgecolor="k", lw=0.5)
        ax.bar(x + 0.2, hl, 0.38, label="highlight (L > 1)", color="#e6550d",
               edgecolor="k", lw=0.5)
        ax.axhline(0, color="k", lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([r.label for r in runs], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("signed relative bias  (sum p - sum g) / sum g")
        ax.set_title(f"pixel set: {ps}")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.suptitle("Signed energy bias, highlights vs the rest. "
                 "Negative = the model under-estimates that zone.", fontsize=10)
    _save(fig, figdir / "highlight_bias.png")


BAND_COLORS = {"shadow": "#3182bd", "midtone": "#9ecae1",
               "highlight": "#fd8d3c", "extreme": "#a63603"}


def _error_split_axes(axes, results, runs, mu_scale, pixel_set: str) -> None:
    tables = {r.key: {q["band"]: q for q in band_table(results[r.key], pixel_set, mu_scale)}
              for r in runs}
    x = np.arange(len(runs))
    w = 0.2

    ax = axes[0]
    for k, band in enumerate(BAND_ORDER):
        ax.bar(x + (k - 1.5) * w, [tables[r.key][band]["smape"] for r in runs], w,
               color=BAND_COLORS[band], edgecolor="k", lw=0.4, label=band)
    ax.set_ylabel("SMAPE within the band")
    ax.set_title(f"Error level inside each zone ({pixel_set})", fontsize=9)
    ax.margins(y=0.22)
    ax.legend(fontsize=7, ncol=4, loc="upper center", framealpha=0.9)

    ax = axes[1]
    for k, band in enumerate(BAND_ORDER):
        ax.bar(x + (k - 1.5) * w, [tables[r.key][band]["rel_bias_signed"] for r in runs], w,
               color=BAND_COLORS[band], edgecolor="k", lw=0.4, label=band)
    ax.axhline(0, color="k", lw=0.9)
    ax.set_ylabel("signed relative bias")
    ax.set_title(f"Signed bias inside each zone ({pixel_set})", fontsize=9)

    for ax, key, lab in ((axes[2], "err_share_mse", "share of total MSE"),
                         (axes[3], "err_share_smape", "share of total SMAPE")):
        bottom = np.zeros(len(runs))
        for band in BAND_ORDER:
            v = np.array([tables[r.key][band][key] for r in runs])
            ax.bar(x, v, 0.62, bottom=bottom, color=BAND_COLORS[band],
                   edgecolor="k", lw=0.4, label=band)
            bottom += v
        # reference: cumulative share of pixels of each band.  It is there to read the
        # gap between "how many pixels" and "how much error": if a band's bar overshoots
        # its line, that band produces more error than it weighs.
        cum = 0.0
        for bi, band in enumerate(BAND_ORDER[:-1]):
            # n_frac depends on the GT alone (the bands are defined on the ground-truth
            # luminance), so it is identical for every run: taking it from the first
            # introduces no asymmetry
            cum += tables[runs[0].key][band]["n_frac"]
            ax.axhline(cum, ls="--", lw=1.0, color="k")
            # labels alternating above/below: the last two cumulative lines are
            # a few percentage points apart and would overlap
            above = bi % 2 == 0
            ax.text(-0.45, cum + (0.008 if above else -0.008),
                    f"{band} pixels, cumulative {100 * cum:.1f}%",
                    fontsize=6, va="bottom" if above else "top", ha="left",
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.8))
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(lab)
        ax.set_title(f"Error budget: which zone produces the error ({pixel_set})", fontsize=9)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([r.label for r in runs], rotation=30, ha="right", fontsize=7)
        ax.grid(alpha=0.3, axis="y")


def fig_error_split(results: dict[str, RunResult], runs: list[Run], mu_scale: float,
                    figdir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    _error_split_axes(axes.ravel(), results, runs, mu_scale, "full")
    fig.suptitle("How much error on the highlights, how much on the rest (full frame)\n"
                 "Top: error level inside each zone. Bottom: how the total error is "
                 "split between zones, with the dashed lines giving the cumulative "
                 "share of pixels.", fontsize=10)
    _save(fig, figdir / "error_split.png")


def fig_error_split_fg_bg(results: dict[str, RunResult], runs: list[Run], mu_scale: float,
                          figdir: Path, hl_in_fg: float) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    _error_split_axes(axes[0], results, runs, mu_scale, "fg")
    _error_split_axes(axes[1], results, runs, mu_scale, "bg")
    fig.suptitle("The same split separated into foreground and background. "
                 f"Only {100 * hl_in_fg:.1f}% of the highlights lie in the foreground: "
                 "without this separation,\n'highlight error' and 'environment map error' "
                 "are the same number.", fontsize=10)
    _save(fig, figdir / "error_split_fg_bg.png")


def _tonemap_srgb(x: np.ndarray, exposure: float) -> np.ndarray:
    """Reinhard + gamma 2.2, exposure shared by every image."""
    y = x * exposure
    y = y / (1.0 + y)
    return np.clip(y, 0, 1) ** (1 / 2.2)


def _best_window(score_map: np.ndarray, crop: int, valid: np.ndarray | None = None
                 ) -> tuple[slice, slice]:
    """crop x crop window that maximises the mean of score_map.

    Searched on a grid of step crop//2: a pointwise argmax would take the brightest
    pixel, which in a scene with a visible source falls in the middle of a saturated,
    structureless surface, i.e. exactly where nothing of the models' behaviour can be
    seen.
    """
    h, w = score_map.shape
    step = max(crop // 2, 1)
    best, best_yx = -np.inf, (0, 0)
    for y0 in range(0, max(h - crop, 0) + 1, step):
        for x0 in range(0, max(w - crop, 0) + 1, step):
            blk = score_map[y0:y0 + crop, x0:x0 + crop]
            if valid is not None and valid[y0:y0 + crop, x0:x0 + crop].mean() < 0.25:
                continue
            v = float(blk.mean())
            if v > best:
                best, best_yx = v, (y0, x0)
    y0, x0 = best_yx
    return slice(y0, y0 + crop), slice(x0, x0 + crop)


def fig_visual(runs: list[Run], gt_paths: list[Path], mask_paths: list[Path],
               frame_idx: int, figdir: Path, crop: int = 260) -> None:
    gt = load_exr_rgb(str(gt_paths[frame_idx]))
    fg = load_mask_bool(str(mask_paths[frame_idx]))
    lum = (gt * LUMA_COEFF).sum(-1)
    loglum = np.log10(lum + 1e-3)

    preds = {r.key: load_exr_rgb(str(r.iter_dir / f"frame_{frame_idx:03d}_pred.exr"))
             for r in runs}

    # 1) bright AND structured area: the fraction above 1 weighs the brightness, the
    #    local gradient guarantees there is something to look at
    grad = np.abs(np.gradient(loglum)[0]) + np.abs(np.gradient(loglum)[1])
    bright = (lum > 1.0).astype(np.float32)
    score_bright = grad * (0.25 + bright)
    # 2) area of maximum disagreement between models: that is what discriminates
    if len(runs) > 1:
        stack = np.stack([np.log10((preds[r.key] * LUMA_COEFF).sum(-1) + 1e-3)
                          for r in runs])
        disagree = stack.std(axis=0)
    else:
        disagree = np.abs(np.log10((preds[runs[0].key] * LUMA_COEFF).sum(-1) + 1e-3)
                          - loglum)

    crops = [("bright, structured region", _best_window(score_bright, crop)),
             ("largest disagreement between models"
              if len(runs) > 1 else "largest deviation from GT",
              _best_window(disagree, crop, valid=fg))]

    ncol = len(runs) + 1
    fig, axes = plt.subplots(2 * len(crops), ncol,
                             figsize=(2.3 * ncol, 2.6 * 2 * len(crops)))
    for ci, (cname, (sy, sx)) in enumerate(crops):
        gc = gt[sy, sx]
        # "key" exposure: the crop's median lands at mid scale, so the tonemap is not
        # dictated by the HDR peak
        med = float(np.median((gc * LUMA_COEFF).sum(-1)))
        expo = 0.5 / max(med, 1e-4)
        diffs = {r.key: np.abs(preds[r.key][sy, sx] - gc).mean(-1) for r in runs}
        vmax = max(float(np.percentile(np.concatenate([d.ravel() for d in diffs.values()]),
                                       99.0)), 1e-4)

        axes[2 * ci][0].imshow(_tonemap_srgb(gc, expo))
        axes[2 * ci][0].set_title(f"ground truth\n{cname}", fontsize=8)
        axes[2 * ci + 1][0].axis("off")
        axes[2 * ci + 1][0].text(0.5, 0.5, "|pred - GT|\nshared scale,\nclipped at p99",
                                 ha="center", va="center", fontsize=8)
        for k, r in enumerate(runs):
            pc = preds[r.key][sy, sx]
            smape_c = float((np.abs(pc - gc) / (np.abs(pc) + np.abs(gc) + EPS)).mean())
            axes[2 * ci][k + 1].imshow(_tonemap_srgb(pc, expo))
            axes[2 * ci][k + 1].set_title(f"{r.label}\nSMAPE {smape_c:.4f}",
                                          fontsize=8, color=r.color)
            im = axes[2 * ci + 1][k + 1].imshow(diffs[r.key], cmap="inferno",
                                                vmin=0, vmax=vmax)
        fig.colorbar(im, ax=axes[2 * ci + 1][1:].tolist(), shrink=0.85, pad=0.01)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Frame {frame_idx}: identical tonemapping and exposure in every panel, "
                 "so what differs on screen differs in the data", fontsize=10)
    _save(fig, figdir / "visual_bright_region.png")


# ──────────────────────────────────────────────────────────────────────────────
# Colour spectrum
# ──────────────────────────────────────────────────────────────────────────────
#
# The question this section answers: which run reproduces the DISTRIBUTION of the GT's
# values.  It is a different question from "which run errs least per pixel": a run can
# have the lowest per-pixel error and the most skewed spectrum (for instance by
# compressing the highlights and lifting the black floor).  Always the whole frame,
# never fg/bg separately.

@dataclass
class SpectrumAnalysis:
    keys: list[str]                  # run, nell'ordine di `runs`
    # The per-frame quantities stay on the WHOLE FRAME: they are what feeds the figures
    # and the tables of this section.  The split by pixel set lives only in the
    # `*_pooled_set` arrays, which feed the three run x metric matrices.
    w1: np.ndarray                   # (n_run, n_frame, 4)  distanza per frame
    dmean: np.ndarray                # (n_run, n_frame, 4)  signed bias
    w1_pooled: np.ndarray            # (n_run, 4)   over every frame together, full
    dmean_pooled: np.ndarray         # (n_run, 4)
    w1_pooled_set: np.ndarray        # (n_run, 3, 4)  per pixel set (PIXEL_SETS)
    dmean_pooled_set: np.ndarray     # (n_run, 3, 4)
    spec_gt_tot: np.ndarray          # (4, NS)      full
    spec_run_tot: np.ndarray         # (n_run, 4, NS)
    notable: dict[str, int]          # label -> frame index
    order: list[int]                 # runs sorted by w1 on the norm channel


def analyze_spectrum(results: dict[str, RunResult], runs: list[Run],
                     stats: GtStats) -> SpectrumAnalysis:
    keys = [r.key for r in runs]
    spec_gt_set = stats.spec_gt                                   # (F, 3, 4, NS)
    spec_runs_set = np.stack([results[k].spec for k in keys])     # (R, F, 3, 4, NS)
    spec_gt = spec_gt_set[:, 0]                                   # (F, 4, NS)  full
    spec_runs = spec_runs_set[:, :, 0]                            # (R, F, 4, NS)
    n_frames = spec_gt.shape[0]

    w1 = np.zeros((len(keys), n_frames, N_SPEC_CH))
    dmean = np.zeros_like(w1)
    for i in range(len(keys)):
        w1[i] = spec_w1_dex(spec_runs[i], spec_gt)
        dmean[i] = spec_dmean_dex(spec_runs[i], spec_gt)

    spec_gt_tot = spec_gt.sum(0)                              # (4, NS)
    spec_run_tot = spec_runs.sum(1)                           # (R, 4, NS)

    # pooling sui frame per ciascun pixel set: (R, 3, 4, NS) contro (1, 3, 4, NS)
    gt_tot_set = spec_gt_set.sum(0)                           # (3, 4, NS)
    run_tot_set = spec_runs_set.sum(1)                        # (R, 3, 4, NS)
    w1_pooled_set = spec_w1_dex(run_tot_set, gt_tot_set[None])
    dmean_pooled_set = spec_dmean_dex(run_tot_set, gt_tot_set[None])
    w1_pooled = w1_pooled_set[:, 0]
    dmean_pooled = dmean_pooled_set[:, 0]

    # ── frame notevoli ───────────────────────────────────────────────────────
    # Reference: the `norm` channel, which is the quantity the rest of the pipeline
    # consumes (the pixel's radiance, not the single channel).
    per_frame_mean = w1[:, :, NORM_CH].mean(axis=0)           # (F,) media sui run
    spread = w1[:, :, NORM_CH].max(axis=0) - w1[:, :, NORM_CH].min(axis=0)
    order_frames = np.argsort(per_frame_mean)
    notable = {
        "worst": int(order_frames[-1]),
        "best": int(order_frames[0]),
        "median": int(order_frames[len(order_frames) // 2]),
        # the frame that separates the runs the most: that is the one to look at to see
        # WHERE the difference shows, not the one with the largest error
        "most_discriminant": int(np.argmax(spread)),
        # same criterion as fig_visual: p99.9 and not the maximum
        "most_hdr": int(np.argmax(stats.per_frame_p999)),
    }

    # Ranking on the `norm` channel and not on the mean of the four: it is the same
    # quantity the run x metric matrix puts in a column, so the two rankings cannot
    # diverge.  Averaging the four channels would count the same information twice
    # anyway, since norm is a function of RGB.
    order = list(np.argsort(w1_pooled[:, NORM_CH]))
    return SpectrumAnalysis(keys, w1, dmean, w1_pooled, dmean_pooled,
                            w1_pooled_set, dmean_pooled_set,
                            spec_gt_tot, spec_run_tot, notable, order)


def spectrum_metrics(an: SpectrumAnalysis, key: str, pixel_set: str) -> dict[str, float]:
    """The matrix's two spectrum columns, norm channel (= ||RGB||_2).

    `dmean` enters in absolute value because the rank needs a direction; the sign stays
    readable in the spectrum section and in spectrum_distance.csv."""
    if key not in an.keys:
        raise KeyError(f"{key} is not in the spectrum analysis ({an.keys})")
    i, s = an.keys.index(key), PIXEL_SETS.index(pixel_set)
    return {"spec_w1": float(an.w1_pooled_set[i, s, NORM_CH]),
            "spec_absdmean": float(abs(an.dmean_pooled_set[i, s, NORM_CH]))}


def verify_spectrum(results: dict[str, RunResult], stats: GtStats) -> list[str]:
    """No pixel lost in the binning, exact fg/bg partition, GT at zero distance from
    itself."""
    lines = []
    n_pix = stats.n_pixels
    all_spec = [stats.spec_gt] + [res.spec for res in results.values()]

    tot = np.concatenate([s[:, 0].sum(axis=-1).ravel() for s in all_spec])   # full
    bad = 0 if n_pix <= 0 else int(np.count_nonzero(np.abs(tot - n_pix) > 0.5))
    lines.append(f"  [{'OK ' if bad == 0 else 'FAIL'}] spectrum: every histogram sums "
                 f"to {n_pix} pixels ({bad} violations)")

    # The three sets are binned independently (spectrum_hist_sets), so this equality is
    # a real check on the mask: it proves no pixel ends up
    # in due insiemi o in nessuno.
    worst = max(float(np.abs(s[:, 0] - s[:, 1] - s[:, 2]).max()) for s in all_spec)
    lines.append(f"  [{'OK ' if worst == 0.0 else 'FAIL'}] spectrum: full == fg + bg bin "
                 f"by bin (max gap {worst:.3e})")

    self_w1 = float(np.abs(spec_w1_dex(stats.spec_gt[:, 0], stats.spec_gt[:, 0])).max())
    lines.append(f"  [{'OK ' if self_w1 == 0.0 else 'FAIL'}] spectrum: W1(GT, GT) = "
                 f"{self_w1:.3e}")

    dens = spec_density(stats.spec_gt[:, 0])
    err = float(np.abs(dens.sum(axis=-1) - 1.0).max())
    lines.append(f"  [{'OK ' if err < 1e-12 else 'FAIL'}] spectrum: densities sum to 1 "
                 f"(scarto max {err:.2e})")
    return lines


# ── figure ────────────────────────────────────────────────────────────────────

def _spec_x() -> np.ndarray:
    return 10.0 ** SPEC_CENTERS


def _spec_xlim(*hists: np.ndarray, floor: float = 1e-7) -> tuple[float, float]:
    """x limits common to every figure: bins where mass exists."""
    m = np.zeros(NS)
    for h in hists:
        m += h.reshape(-1, NS).sum(axis=0)
    d = m / max(m.sum(), 1.0)
    nz = np.nonzero(d > floor)[0]
    if nz.size == 0:
        return 10.0 ** SPEC_LO_EXP, 10.0 ** SPEC_HI_EXP
    lo = max(int(nz[0]) - 1, 0)
    hi = min(int(nz[-1]) + 1, NS - 1)
    return float(10.0 ** SPEC_CENTERS[lo]), float(10.0 ** SPEC_CENTERS[hi])


def _spec_panel_grid(figsize=(12, 8), **kw):
    fig, axs = plt.subplots(2, 2, figsize=figsize, **kw)
    return fig, list(axs.flat)


def _draw_spectrum(ax, x, spec_gt_ch, curves, ylim, xlim):
    """One panel: GT as a grey area, one profile per run."""
    g = spec_density(spec_gt_ch)
    ax.fill_between(x, np.maximum(g, 1e-12), 1e-12, step="mid",
                    color="#B0B0B0", alpha=0.75, lw=0, label="GT", zorder=1)
    for label, color, ls, h in curves:
        ax.plot(x, np.maximum(spec_density(h), 1e-12), color=color, ls=ls,
                lw=1.3, label=label, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.25)


def _spec_ylim(*hists: np.ndarray) -> tuple[float, float]:
    top = 0.0
    for h in hists:
        d = spec_density(h.reshape(-1, NS))
        top = max(top, float(d.max()))
    return 1e-7, top * 3.0


def fig_spectrum_global(an: SpectrumAnalysis, runs: list[Run], figdir: Path) -> None:
    x = _spec_x()
    xlim = _spec_xlim(an.spec_gt_tot, an.spec_run_tot)
    ylim = _spec_ylim(an.spec_gt_tot, an.spec_run_tot)
    fig, axs = _spec_panel_grid(sharex=True, sharey=True)
    for c, ch in enumerate(SPEC_CHANNELS):
        curves = [(f"{r.label}  W1={an.w1_pooled[i, c]:.3f}", r.color, r.linestyle,
                   an.spec_run_tot[i, c]) for i, r in enumerate(runs)]
        _draw_spectrum(axs[c], x, an.spec_gt_tot[c], curves, ylim, xlim)
        axs[c].set_title(f"{ch}", fontsize=10)
        axs[c].set_xlabel("pixel value (HDR)", fontsize=8)
        axs[c].set_ylabel("fraction of pixels per bin", fontsize=8)
    _legend_runs(axs[0], loc="upper left")
    fig.suptitle("Value spectrum, all frames pooled -- shared axes across runs "
                 f"({SPEC_PER_DECADE} bins/decade; leftmost bin = values below "
                 f"1e{SPEC_LO_EXP:.0f}, zero included)", fontsize=10)
    _save(fig, figdir / "spectrum_global.png")


def fig_spectrum_ratio(an: SpectrumAnalysis, runs: list[Run], figdir: Path) -> None:
    """log2(run density / GT density): the differences the histogram flattens."""
    x = _spec_x()
    xlim = _spec_xlim(an.spec_gt_tot, an.spec_run_tot)
    fig, axs = _spec_panel_grid(sharex=True, sharey=True)
    lim = 0.0
    for c, ch in enumerate(SPEC_CHANNELS):
        ax = axs[c]
        g = spec_density(an.spec_gt_tot[c])
        # below this density the GT has too few pixels: the ratio is noise
        keep = g > 1e-6
        for i, r in enumerate(runs):
            p = spec_density(an.spec_run_tot[i, c])
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.log2(np.where(keep & (p > 0), p / np.maximum(g, 1e-300),
                                         np.nan))
            ax.plot(x, ratio, color=r.color, ls=r.linestyle, lw=1.2, label=r.label)
            fin = ratio[np.isfinite(ratio)]
            if fin.size:
                lim = max(lim, float(np.abs(fin).max()))
        ax.axhline(0.0, color="k", lw=1.0, alpha=0.6)
        ax.set_xscale("log")
        ax.set_xlim(*xlim)
        ax.grid(alpha=0.25)
        ax.set_title(ch, fontsize=10)
        ax.set_xlabel("pixel value (HDR)", fontsize=8)
        ax.set_ylabel("log2(run / GT) density", fontsize=8)
    for ax in axs:
        ax.set_ylim(-min(lim, 6.0) * 1.05, min(lim, 6.0) * 1.05)
    _legend_runs(axs[0], loc="upper left")
    fig.suptitle("Spectrum residual: above 0 the run puts more pixels in that "
                 "intensity band than the GT does", fontsize=10)
    _save(fig, figdir / "spectrum_global_ratio.png")


def fig_spectrum_cdf(an: SpectrumAnalysis, runs: list[Run], figdir: Path) -> None:
    x = _spec_x()
    xlim = _spec_xlim(an.spec_gt_tot, an.spec_run_tot)
    fig, axs = _spec_panel_grid(sharex=True, sharey=True)
    for c, ch in enumerate(SPEC_CHANNELS):
        ax = axs[c]
        ax.plot(x, np.cumsum(spec_density(an.spec_gt_tot[c])), color="#444444",
                lw=2.5, alpha=0.8, label="GT")
        for i, r in enumerate(runs):
            ax.plot(x, np.cumsum(spec_density(an.spec_run_tot[i, c])),
                    color=r.color, ls=r.linestyle, lw=1.3, label=r.label)
        ax.set_xscale("log")
        ax.set_xlim(*xlim)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.set_title(ch, fontsize=10)
        ax.set_xlabel("pixel value (HDR)", fontsize=8)
        ax.set_ylabel("cumulative fraction", fontsize=8)
    _legend_runs(axs[0], loc="lower right")
    fig.suptitle("Cumulative spectrum -- the area between a run and the GT curve "
                 "is exactly its W1 distance [dex]", fontsize=10)
    _save(fig, figdir / "spectrum_cdf.png")


def fig_spectrum_qq(an: SpectrumAnalysis, runs: list[Run], figdir: Path) -> None:
    qs = np.linspace(0.005, 0.995, 199)
    # no sharey: in a Q-Q the two axes must have the same range inside each panel,
    # otherwise the bisector is no longer at 45 degrees.  The runs that collapse into
    # the underflow bin leave the frame, and that is deliberate: their distance reads
    # in the other figures
    fig, axs = _spec_panel_grid()
    for c, ch in enumerate(SPEC_CHANNELS):
        ax = axs[c]
        qg = spec_quantile(an.spec_gt_tot[c], qs)
        for i, r in enumerate(runs):
            qp = spec_quantile(an.spec_run_tot[i, c], qs)
            ax.plot(10.0 ** qg, 10.0 ** qp, color=r.color, ls=r.linestyle,
                    lw=1.3, label=r.label)
        lo = float(10.0 ** qg.min())
        hi = float(10.0 ** qg.max())
        ax.plot([lo, hi], [lo, hi], color="k", lw=1.0, alpha=0.6, zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo / 1.6, hi * 1.6)
        ax.set_ylim(lo / 1.6, hi * 1.6)
        ax.grid(alpha=0.25)
        ax.set_title(ch, fontsize=10)
        ax.set_xlabel("GT quantile", fontsize=8)
        ax.set_ylabel("run quantile", fontsize=8)
    _legend_runs(axs[0], loc="upper left")
    fig.suptitle("Q-Q plot of the value spectrum (q from 0.005 to 0.995): distance "
                 "from the diagonal is the tone mismatch, decade by decade", fontsize=10)
    _save(fig, figdir / "spectrum_qq.png")


def fig_spectrum_per_frame(an: SpectrumAnalysis, runs: list[Run], figdir: Path) -> None:
    n_frames = an.w1.shape[1]
    xs = np.arange(n_frames)
    # scale shared between the runs (it is the same panel), not between the channels:
    # G has peaks 6 times higher and would flatten the other three into a line
    fig, axs = _spec_panel_grid(figsize=(13, 8), sharex=True)
    for c, ch in enumerate(SPEC_CHANNELS):
        ax = axs[c]
        for i, r in enumerate(runs):
            ax.plot(xs, an.w1[i, :, c], color=r.color, ls=r.linestyle, lw=1.2,
                    label=r.label)
        for name, f in an.notable.items():
            ax.axvline(f, color="#777777", lw=0.8, ls=":", zorder=0)
            ax.annotate(f"{name} ({f})", (f, 1.0), xycoords=("data", "axes fraction"),
                        fontsize=6, rotation=90, va="top", ha="right", color="#555555")
        ax.grid(alpha=0.25)
        ax.set_title(ch, fontsize=10)
        ax.set_xlabel("frame", fontsize=8)
        ax.set_ylabel("W1 spectrum distance [dex]", fontsize=8)
    _legend_runs(axs[0], loc="upper left")
    fig.suptitle("Per-frame spectrum distance from the GT (lower is closer) -- "
                 "scale shared across runs; per-channel y range", fontsize=10)
    _save(fig, figdir / "spectrum_w1_per_frame.png")


def fig_spectrum_box(an: SpectrumAnalysis, runs: list[Run], figdir: Path) -> None:
    fig, axs = _spec_panel_grid(figsize=(12, 8))
    for c, ch in enumerate(SPEC_CHANNELS):
        ax = axs[c]
        bp = ax.boxplot([an.w1[i, :, c] for i in range(len(runs))],
                        patch_artist=True, widths=0.6)
        for patch, r in zip(bp["boxes"], runs):
            patch.set_facecolor(r.color)
            patch.set_alpha(0.65)
        for med in bp["medians"]:
            med.set_color("k")
        ax.set_xticklabels([r.label for r in runs], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("W1 [dex]", fontsize=9)
        ax.set_title(ch, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Distribution over the frames of the spectrum distance from the GT",
                 fontsize=10)
    _save(fig, figdir / "spectrum_w1_box.png")


def fig_spectrum_frames(an: SpectrumAnalysis, runs: list[Run], results: dict[str, RunResult],
                        stats: GtStats, figdir: Path) -> None:
    """One spectrum per frame, with identical axes in every file of the folder."""
    outdir = figdir / "spectrum_frames"
    outdir.mkdir(parents=True, exist_ok=True)
    x = _spec_x()
    xlim = _spec_xlim(an.spec_gt_tot, an.spec_run_tot)
    # y limits from the individual frames (higher than the pooled: fewer pixels per bin).
    # `full` slice: mixing the three pixel sets would count the same pixels twice.
    ylim = _spec_ylim(stats.spec_gt[:, 0], *[results[k].spec[:, 0] for k in an.keys])
    by_frame: dict[int, list[str]] = {}
    for name, f in an.notable.items():
        by_frame.setdefault(f, []).append(name)

    n_frames = an.w1.shape[1]
    print(f"      spectrum_frames/: {n_frames} figure...")
    for fidx in range(n_frames):
        fig, axs = _spec_panel_grid(sharex=True, sharey=True)
        for c, ch in enumerate(SPEC_CHANNELS):
            curves = [(f"{r.label}  W1={an.w1[i, fidx, c]:.3f}", r.color, r.linestyle,
                       results[r.key].spec[fidx, 0, c]) for i, r in enumerate(runs)]
            _draw_spectrum(axs[c], x, stats.spec_gt[fidx, 0, c], curves, ylim, xlim)
            axs[c].set_title(ch, fontsize=10)
            axs[c].set_xlabel("pixel value (HDR)", fontsize=8)
            axs[c].set_ylabel("fraction of pixels per bin", fontsize=8)
        _legend_runs(axs[0], loc="upper left")
        tag = f"  [{', '.join(by_frame[fidx])}]" if fidx in by_frame else ""
        fig.suptitle(f"Value spectrum -- frame {fidx:03d}{tag}   "
                     f"(axes identical in every file of this folder)", fontsize=10)
        fig.savefig(outdir / f"frame_{fidx:03d}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"\r        {fidx + 1}/{n_frames}", end="", flush=True)
    print(f"\r      + spectrum_frames/ ({n_frames} figure)")


# ── skybox: original vs baked ─────────────────────────────────────────────────

def _equirect_solid_angle_weights(h: int, w: int) -> np.ndarray:
    """Per-pixel weight of an equirectangular map: sin(theta) per row.

    Without this weight the poles, where pixels cover a tiny solid angle,
    would count as much as the equator and the envmap spectrum would be distorted.
    """
    theta = np.pi * (np.arange(h, dtype=np.float64) + 0.5) / h
    return np.repeat(np.sin(theta)[:, None], w, axis=1)


def resolve_skybox_gt(runs: list[Run], cli: str | None) -> Path | None:
    """Original skybox: CLI argument, otherwise the field in run_manifest.json.

    Same resolution order as bake_skyboxes.py.  In the current sweep the manifest has
    the field empty (skybox_path is commented out in the SceneConfig), so without
    --skybox-gt the panel is skipped."""
    if cli:
        p = Path(cli)
        if not p.exists():
            print(f"  [skybox] {p} does not exist: skybox panel skipped")
            return None
        return p
    for r in runs:
        mf = r.scene_dir / "run_manifest.json"
        if not mf.exists():
            continue
        try:
            meta = json.loads(mf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        cand = (meta.get("config", {}).get("render", {}).get("skybox_path")
                or meta.get("scene", {}).get("skybox_path") or "")
        if cand and Path(cand).exists():
            return Path(cand)
    print("  [skybox] no reference skybox (use --skybox-gt): "
          "skybox panel skipped")
    return None


def spectrum_skybox(runs: list[Run], gt_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """(spectrum of the original skybox, {run: spectrum of the baked one})."""
    gt = load_exr_rgb(str(gt_path))
    spec_gt = spectrum_hist(gt, _equirect_solid_angle_weights(*gt.shape[:2]))
    del gt
    per_run: dict[str, np.ndarray] = {}
    for r in runs:
        p = r.scene_dir / "skybox_nerf_baked.exr"
        if not p.exists():
            print(f"      [skip] {r.key}: skybox_nerf_baked.exr missing "
                  f"(generate it with bake_skyboxes.py)")
            continue
        a = load_exr_rgb(str(p))
        per_run[r.key] = spectrum_hist(a, _equirect_solid_angle_weights(*a.shape[:2]))
        del a
    return spec_gt, per_run


def fig_spectrum_skybox(spec_gt: np.ndarray, per_run: dict[str, np.ndarray],
                        runs: list[Run], gt_path: Path, figdir: Path) -> None:
    x = _spec_x()
    have = [r for r in runs if r.key in per_run]
    if not have:
        return
    stack = np.stack([per_run[r.key] for r in have])
    xlim = _spec_xlim(spec_gt, stack)
    ylim = _spec_ylim(spec_gt, stack)
    fig, axs = _spec_panel_grid(sharex=True, sharey=True)
    for c, ch in enumerate(SPEC_CHANNELS):
        w1 = {r.key: float(spec_w1_dex(per_run[r.key][c], spec_gt[c])) for r in have}
        curves = [(f"{r.label}  W1={w1[r.key]:.3f}", r.color, r.linestyle,
                   per_run[r.key][c]) for r in have]
        _draw_spectrum(axs[c], x, spec_gt[c], curves, ylim, xlim)
        axs[c].set_title(ch, fontsize=10)
        axs[c].set_xlabel("radiance", fontsize=8)
        axs[c].set_ylabel("fraction of solid angle per bin", fontsize=8)
    _legend_runs(axs[0], loc="upper left")
    fig.suptitle(f"Envmap spectrum: original skybox ({gt_path.name}, grey) vs the "
                 "skybox baked from each NeRF -- solid-angle weighted", fontsize=10)
    _save(fig, figdir / "spectrum_skybox.png")


# ── tabelle ───────────────────────────────────────────────────────────────────

def write_spectrum_tables(out: Path, an: SpectrumAnalysis, runs: list[Run]) -> None:
    rows = []
    for i, r in enumerate(runs):
        for c, ch in enumerate(SPEC_CHANNELS):
            v = an.w1[i, :, c]
            rows.append({
                "run": r.key, "activation": r.activation, "loss": r.loss,
                "channel": ch,
                "w1_dex_pooled": an.w1_pooled[i, c],
                "dmean_dex_pooled": an.dmean_pooled[i, c],
                "w1_dex_frame_mean": float(v.mean()),
                "w1_dex_frame_median": float(np.median(v)),
                "w1_dex_frame_p90": float(np.percentile(v, 90)),
                "worst_frame": int(np.argmax(v)),
            })
    with open(out / "spectrum_distance.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v)
                        for k, v in row.items()})

    by_frame: dict[int, list[str]] = {}
    for name, f in an.notable.items():
        by_frame.setdefault(f, []).append(name)
    rows = []
    for i, r in enumerate(runs):
        for f in range(an.w1.shape[1]):
            row = {"run": r.key, "frame": f}
            for c, ch in enumerate(SPEC_CHANNELS):
                row[f"w1_{ch}"] = an.w1[i, f, c]
            row["dmean_norm"] = an.dmean[i, f, NORM_CH]
            row["notable"] = "|".join(by_frame.get(f, []))
            rows.append(row)
    with open(out / "spectrum_per_frame.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v)
                        for k, v in row.items()})


def write_spectrum_skybox_table(out: Path, spec_gt: np.ndarray,
                                per_run: dict[str, np.ndarray], runs: list[Run]) -> None:
    rows = []
    for r in runs:
        if r.key not in per_run:
            continue
        for c, ch in enumerate(SPEC_CHANNELS):
            rows.append({
                "run": r.key, "activation": r.activation, "loss": r.loss, "channel": ch,
                "w1_dex": float(spec_w1_dex(per_run[r.key][c], spec_gt[c])),
                "dmean_dex": float(spec_dmean_dex(per_run[r.key][c], spec_gt[c])),
            })
    if not rows:
        return
    with open(out / "spectrum_skybox.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v)
                        for k, v in row.items()})


# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────

def _md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _report_spectrum(A, runs: list[Run], an: SpectrumAnalysis,
                     sky: tuple[np.ndarray, dict[str, np.ndarray], Path] | None) -> None:
    A("## Colour spectrum")
    A("")
    A("Per-pixel metrics say how much a run errs, not whether it reproduces the "
      "**distribution** of the GT's values. They are different questions: a run can have "
      "the lowest error and the most skewed spectrum, for instance by compressing the "
      "highlights and lifting the black floor. Inside the pipeline's hemispherical "
      "integrals it is the second one that counts.")
    A("")
    A("The two `W1 spectrum` and `abs. dmean` columns of the matrix come from here, "
      "there split by pixel set. This section stays on the **whole frame** and "
      "decomposes it: per channel, per frame, and on the skybox.")
    A("")
    A(f"The spectrum is the histogram of the HDR values on a log grid ({SPEC_PER_DECADE} "
      f"bins per decade from 1e{SPEC_LO_EXP:.0f} to 1e{SPEC_HI_EXP:.0f}), computed here on "
      "the **whole frame**, for the R, G, B channels and for the norm "
      "`||RGB||` of the pixel. The distance from the GT is the **W1 in the log10 "
      "domain**, i.e. the mean shift of the quantiles measured in **decades**: 0 = "
      "identical distributions, 0.1 = the quantiles are shifted on average by a tenth "
      "of a decade (+26%). `dmean` is the same thing with a sign: positive = run brighter than the GT.")
    A("")
    rows = []
    for i in an.order:
        r = runs[i]
        rows.append([f"**{list(an.order).index(i) + 1}**", r.key, r.activation, r.loss]
                    + [f"{an.w1_pooled[i, c]:.4f}" for c in range(N_SPEC_CH)]
                    + [f"{an.w1_pooled[i].mean():.4f}",
                       f"{an.dmean_pooled[i, NORM_CH]:+.4f}"])
    A(_md_table(["#", "run", "act.", "loss"] + [f"W1 {c}" for c in SPEC_CHANNELS]
                + ["W1 mean", "dmean norm"], rows))
    A("")
    A("Ranking on the `norm` channel, over every frame together: it is the pixel's "
      "radiance, i.e. the quantity the pipeline consumes, and it is the same column "
      "that appears in the matrix, so the two rankings cannot diverge. The "
      "`W1 mean` column stays as a diagnostic, but averaging the four channels would "
      "count the same information twice, since `norm` is a function of R, G, B.")
    A("")
    A("Per-frame statistics in `spectrum_distance.csv`. Figures: "
      "`spectrum_global.png` (histograms), `spectrum_global_ratio.png` (residual, "
      "where the differences show), `spectrum_cdf.png` (the area between the curves "
      "**is** the W1), `spectrum_qq.png` (quantile by quantile).")
    A("")
    A("### Notable frames")
    A("")
    rows = []
    labels = {
        "worst": "spectrum furthest from the GT (mean over the runs)",
        "best": "spectrum closest to the GT (mean over the runs)",
        "median": "median frame",
        "most_discriminant": "largest difference between the runs: the frame to look at",
        "most_hdr": "highest p99.9 of the GT luminance",
    }
    for name, f in an.notable.items():
        vals = an.w1[:, f, NORM_CH]
        rows.append([f"`{name}`", str(f), labels.get(name, ""),
                     f"{vals.mean():.4f}", f"{vals.min():.4f}", f"{vals.max():.4f}"])
    A(_md_table(["label", "frame", "criterion", "W1 norm mean", "min", "max"], rows))
    A("")
    A("They are marked in `spectrum_w1_per_frame.png`, annotated in the title of the "
      "corresponding file in `figures/spectrum_frames/` and in the `notable` column of "
      "`spectrum_per_frame.csv`. The 60 per-frame figures have identical axes, "
      "so scrolling through them is a valid comparison.")
    A("")

    if sky is not None:
        spec_sky_gt, per_run, sky_path = sky
        A("### Skybox: original vs baked from the NeRFs")
        A("")
        A(f"Same analysis on the envmap: `{sky_path.name}` against the "
          "`skybox_nerf_baked.exr` produced by `bake_skyboxes.py`. The histograms are "
          "**weighted by solid angle** (sin(theta) per row of the equirectangular map): "
          "without the weight the poles would count as much as the equator.")
        A("")
        rows = []
        pairs = [(r, per_run[r.key]) for r in runs if r.key in per_run]
        # same criterion as the per-frame ranking: norm channel, not the mean of the 4
        pairs.sort(key=lambda t: float(spec_w1_dex(t[1], spec_sky_gt)[NORM_CH]))
        for rank, (r, h) in enumerate(pairs, start=1):
            w1 = spec_w1_dex(h, spec_sky_gt)
            dm = spec_dmean_dex(h, spec_sky_gt)
            rows.append([f"**{rank}**", r.key, r.activation, r.loss]
                        + [f"{w1[c]:.4f}" for c in range(N_SPEC_CH)]
                        + [f"{w1.mean():.4f}", f"{dm[NORM_CH]:+.4f}"])
        A(_md_table(["#", "run", "act.", "loss"] + [f"W1 {c}" for c in SPEC_CHANNELS]
                    + ["W1 mean", "dmean norm"], rows))
        A("")
        A("Figure: `spectrum_skybox.png`, full table in `spectrum_skybox.csv`. "
          "This ranking is independent of the per-frame one: it measures how well the "
          "**background** of the NeRF, which is what ends up in the envmap bake and "
          "hence in the irradiance, reproduces the source's radiance distribution.")
        A("")
    A("`dmean` (mean of the logs) and `energy_bias` in `metrics_global.csv` (linear "
      "mean) **can have different signs**, and that happens here for the "
      "`rel_mse_raw` runs: the total energy is slightly underestimated while the "
      "geometric mean rises. It is not a contradiction, it is the signature of a "
      "spectrum that compresses the highlights and lifts the black floor, i.e. exactly "
      "what a relative loss rewards. The linear mean is seen by the brightest pixels, "
      "the mean of the logs by the bulk of the distribution.")
    A("")
    A("Note on the extreme bins: the first collects every value below "
      f"1e{SPEC_LO_EXP:.0f} (**zero included**) and the last every value above "
      f"1e{SPEC_HI_EXP:.0f}; in the W1 they enter with the nominal width of the "
      "other bins. It is an approximation that touches the tails alone.")
    A("")


def write_report(out: Path, runs: list[Run], results: dict[str, RunResult],
                 stats: GtStats, checks: list[str], root: Path, n_frames: int,
                 an: SpectrumAnalysis,
                 sky: tuple[np.ndarray, dict[str, np.ndarray], Path] | None = None) -> None:
    mu_scale = stats.mu_scale
    vals, ranks = _matrix_values(results, runs, "full", mu_scale, an)
    order = np.argsort([r for r in ranks[:, 0]])   # rank on the SMAPE (first column)

    L: list[str] = []
    A = L.append
    A("# Comparison of the runs of the activation x loss sweep")
    A("")
    A(f"Source: `{root}`  ")
    A(f"Runs compared: {len(runs)}, frames per run: {n_frames}, "
      f"generated on {time.strftime('%Y-%m-%d %H:%M')}")
    A("")
    A("## How to read this ranking")
    A("")
    A("Every run is by construction the minimum of its **own** loss on this data. "
      "Ranking them with the MSE rewards the `mse` runs, with the MAE the `l1` runs, "
      "with the relMSE the `rel_mse_raw` runs. That is why the three metrics that "
      "coincide with the sweep's losses, i.e. **linear MSE/PSNR, MAE and relMSE**, were "
      "**removed from the matrix** rather than marked: a self-evaluated cell stays "
      "misleading even when labelled, and marking some but not all would give home "
      "advantage to a single loss. They stay computed in `metrics_global.csv`, where "
      "nobody reads them as a ranking.")
    A("")
    A("The headline metric is the **SMAPE**, `mean(|p-g| / (|p|+|g|+1e-3))`: it is "
      "symmetric, scale-invariant (in HDR every decade weighs the same) and bounded in "
      "[0,1], so a single incandescent pixel does not decide the ranking. mu-PSNR and "
      "log-RMSE flank it as a control. One clarification, so as not to sell more than "
      "is there: no column of the matrix **is** one of the losses, but `PSNR tm-clip` "
      "and `PSNR tm-Reinhard` stay quadratic and `log-RMSE` stays relative, so the "
      "family resemblance with `mse` and `rel_mse_raw` does exist. It is much weaker "
      "than exact coincidence, and it is declared rather than "
      "hidden.")
    A("")
    A("The second criterion is not a per-pixel error. The NeRF, here, is consumed inside "
      "hemispherical integrals (indirect irradiance, specular cones): the **signed bias** "
      "survives the integration, zero-mean noise cancels. A model that is slightly "
      "noisier but centred is preferable to one that is smoother but systematically "
      "underestimates the highlights.")
    A("")
    A("The third criterion is the **spectrum**, i.e. the distribution of the values, and "
      "it is no longer an appendix: the last two columns of the matrix are the W1 "
      "distance from the GT's distribution and the mean tonal shift, both in decades "
      "and computed on the **same pixel set** as the column next to them. The "
      "\"Colour spectrum\" section decomposes them per channel and per frame.")
    A("")

    A("## Ranking by the headline metric (SMAPE, full frame)")
    A("")
    rows = []
    for i in order:
        r = runs[i]
        n, s = _slice(results[r.key].acc, "full", 0, N_DEC - 1)
        m = metrics_from(n, s, mu_scale)
        nh, sh = _slice(results[r.key].acc, "full", *SPLIT2["highlight_all"])
        mh = metrics_from(nh, sh, mu_scale)
        nr_, sr_ = _slice(results[r.key].acc, "full", *SPLIT2["rest"])
        mr = metrics_from(nr_, sr_, mu_scale)
        rows.append([f"**{ranks[i, 0]}**", r.key, r.activation, r.loss,
                     f"{m['smape']:.4f}", f"{mr['smape']:.4f}", f"{mh['smape']:.4f}",
                     f"{mr['energy_bias']:+.4f}", f"{mh['energy_bias']:+.4f}",
                     f"{m['mu_psnr']:.2f}",
                     f"{spectrum_metrics(an, r.key, 'full')['spec_w1']:.4f}"])
    A(_md_table(["#", "run", "act.", "loss", "SMAPE", "SMAPE rest", "SMAPE highlight",
                 "bias rest", "bias highlight", "mu-PSNR", "W1 spectrum"], rows))
    A("")
    A("`rest` = GT luminance <= 1, `highlight` = GT luminance > 1. The bias is "
      "`(sum pred - sum gt) / sum gt` over the zone: negative = underestimate. `W1 spectrum` "
      "is the distance between the run's value distribution and the GT's, in "
      "decades, on the whole frame: it is not split between `rest` and `highlight` "
      "because it is a property of the whole distribution, not of one of its bands.")
    A("")
    A("The same ranking on the three pixel sets, because the pipeline consumes the "
      "NeRF in two different ways: the foreground feeds the PBR fit, the background "
      "feeds the envmap bake.")
    A("")
    by_set = {ps: _matrix_values(results, runs, ps, mu_scale, an) for ps in PIXEL_SETS}
    j_w1 = next(j for j, sp in enumerate(METRIC_SPECS) if sp.key == "spec_w1")
    rows = []
    for i, r in enumerate(runs):
        cells = [r.label]
        for j in (0, j_w1):
            for ps in PIXEL_SETS:
                v_ps, rk_ps = by_set[ps]
                cells.append(f"{v_ps[i, j]:.4f} (#{rk_ps[i, j]})")
        rows.append(cells)
    A(_md_table(["run", "SMAPE full", "SMAPE foreground", "SMAPE background",
                 "W1 full", "W1 foreground", "W1 background"], rows))
    A("")
    A("The `W1` columns are the spectrum computed on the same pixel set, not the "
      "whole-frame value repeated three times: the distribution of the object's "
      "radiance and that of the envmap are two different things, and a run can "
      "reproduce one well and the other badly.")
    A("")

    # ── The trade-off, computed from the data and not asserted ───────────────
    A("## What is gained and what is lost")
    A("")
    top = [runs[i] for i in order]
    win = int(order[0])
    # the ranks are a permutation of 1..n, so argmin gives the index of the first.
    # The partition is no longer neutral/diagonal (there are no diagonals any more) but
    # by family: per-pixel error against distribution of the values.
    per_pixel = [(j, sp) for j, sp in enumerate(METRIC_SPECS) if not sp.spectrum]
    spec_fam = [(j, sp) for j, sp in enumerate(METRIC_SPECS) if sp.spectrum]
    p_agree = [sp.label for j, sp in per_pixel if int(np.argmin(ranks[:, j])) == win]
    p_dis = [sp.label for j, sp in per_pixel if sp.label not in p_agree]
    s_agree = [sp.label for j, sp in spec_fam if int(np.argmin(ranks[:, j])) == win]
    s_dis = [sp.label for j, sp in spec_fam if sp.label not in s_agree]

    A(f"There are {len(per_pixel)} **per-pixel error** metrics, none of which "
      f"coincides with a loss of the sweep. Of those, {len(p_agree)} point to the same "
      f"winner as the headline metric"
      + (f": {', '.join(p_agree)}" if p_agree else "")
      + (f"; the remaining ones do not: {', '.join(p_dis)}." if p_dis else "."))
    A("")
    A(f"There are {len(spec_fam)} **distribution** metrics. "
      + (f"{', '.join(s_agree)} " + ("point" if len(s_agree) > 1 else "points")
         + " to the same winner. " if s_agree else "")
      + (f"{', '.join(s_dis)} " + ("point" if len(s_dis) > 1 else "points")
         + " to a different winner: the run that errs least per pixel is not the one "
           "that best reproduces the distribution of the values, and it is the second "
           "that counts inside the hemispherical integrals." if s_dis else ""))
    A("")
    if not p_dis and not s_dis:
        A("No metric contradicts the headline ranking: the ranking does not depend on "
          "the metric chosen, and none of the metrics is one of the losses.")
    elif not s_dis:
        A("The two families agree on the winner, but at least one per-pixel metric "
          "does not: the result should be presented as a trade-off and not as an "
          "absolute winner.")
    else:
        A("The two families do not agree. This is the interesting case: per-pixel error "
          "and distribution fidelity are different questions, and the choice has to be "
          "justified on which of the two the pipeline actually consumes.")
    A("")

    # paired comparison between the top two, over the 60 frames
    if len(top) >= 2:
        a = np.array([x["smape_full"] for x in results[top[0].key].per_frame])
        b = np.array([x["smape_full"] for x in results[top[1].key].per_frame])
        d = b - a                      # >0 = the first is better on that frame
        won = int((d > 0).sum())
        A(f"**{top[0].label} against {top[1].label}**, paired comparison over the "
          f"{len(d)} frames: mean SMAPE difference {d.mean():+.5f} "
          f"(standard deviation {d.std(ddof=1):.5f}), the first wins on "
          f"{won}/{len(d)} frames. " +
          ("The gap is of the same order as the spread between frames: on the "
           "headline metric the two are equivalent and the choice has to be made on "
           "another criterion."
           if abs(d.mean()) < d.std(ddof=1) else
           "The gap exceeds the spread between frames: the difference is systematic."))
        A("")

    # the criterion that decides when the headline metric does not separate: the bias
    # that survives the hemispherical integration
    A("Tie-breaker, the signed bias that survives the hemispherical integration "
      "(the two brightest background bands are the envmap, which the irradiance bake "
      "integrates in linear units):")
    A("")
    rows = []
    for r in runs:
        cells = [r.label]
        for ps, (lo, hi) in (("bg", BANDS["extreme"]), ("bg", BANDS["highlight"]),
                             ("fg", BANDS["shadow"])):
            n_b, s_b = _slice(results[r.key].acc, ps, lo, hi)
            m = metrics_from(n_b, s_b, mu_scale)
            cells.append(f"{m['energy_bias']:+.4f}")
        n_b, s_b = _slice(results[r.key].acc, "bg", *BANDS["extreme"])
        cells.append(f"{metrics_from(n_b, s_b, mu_scale)['mse_lin']:.3f}")
        rows.append(cells)
    A(_md_table(["run", "bias bg extreme (L>10)", "bias bg highlight (1<L<=10)",
                 "bias fg shadow (L<=0.1)", "MSE bg extreme"], rows))
    A("")
    A("The last column is the absolute squared error on the brightest pixels of the "
      "background. That is where the relative losses pay for their advantage: the "
      "typical relative error is lower, but the tail in absolute value is heavier, "
      "and it is the absolute value that enters the irradiance integral.")
    A("")

    A("## Full matrix (full frame)")
    A("")
    hdr = ["run"] + [s.label for s in METRIC_SPECS]
    rows = []
    for i, r in enumerate(runs):
        cells = []
        for j, sp in enumerate(METRIC_SPECS):
            txt = sp.fmt.format(vals[i, j]) + f" (#{ranks[i, j]})"
            if ranks[i, j] == 1:
                txt = f"**{txt}**"
            cells.append(txt)
        rows.append([r.label] + cells)
    A(_md_table(hdr, rows))
    A("")
    A("No column is one of the sweep's losses, so no cell is self-evaluated. The last "
      "two measure the distribution and not the per-pixel error: they are the W1 from "
      "the GT's distribution and the mean tonal shift in absolute value, on the "
      "`norm` channel and on the same pixel set as the matrix. The same two columns, "
      "computed on `fg` and `bg`, are in "
      "`metric_matrix_fg.png` e `metric_matrix_bg.png`.")
    A("")

    A("## Error on the highlights and on the rest")
    A("")
    for ps in ("full", "fg", "bg"):
        A(f"### pixel set: {ps}")
        A("")
        rows = []
        for r in runs:
            tb = {q["band"]: q for q in band_table(results[r.key], ps, mu_scale)}
            for band in BAND_ORDER:
                q = tb[band]
                rows.append([r.label, band, f"{100 * q['n_frac']:.2f}%",
                             f"{q['smape']:.4f}", f"{q['median_ratio']:.4f}",
                             f"{q['rel_bias_signed']:+.4f}",
                             f"{100 * q['err_share_mse']:.1f}%",
                             f"{100 * q['err_share_smape']:.1f}%"])
        A(_md_table(["run", "band", "pixels", "SMAPE", "median pred/gt", "signed bias",
                     "share of MSE", "share of SMAPE"], rows))
        A("")
    A("The last two columns are the exact decomposition of the error budget: inside "
      "each block (one run, one pixel set) they sum to 100%. The comparison between "
      "`share of MSE` and `share of SMAPE` says how much of the ranking depends on "
      "the metric chosen rather than on the model.")
    A("")

    if an is not None:
        _report_spectrum(A, runs, an, sky)

    A("## Global constants used")
    A("")
    A(f"- mu-law scale `X` = p99.99 of the GT luminance over every frame = **{mu_scale:.4f}**, "
      f"mu = {MU:.0f}. The encoding saturates above `X`: the brightest 0.01% of pixels "
      f"enter the mu-PSNR with a compressed error. That is why the mu-PSNR "
      f"flanks the SMAPE instead of replacing it.")
    A(f"- partition thresholds: shadow L<=0.1, midtone 0.1<L<=1, highlight 1<L<=10, "
      f"extreme L>10")
    A(f"- band grid: {N_DEC_INNER} bins of 1/3 decade between 1e-3 and 1e2, "
      f"plus underflow and overflow")
    A(f"- percentiles of the GT luminance: " +
      ", ".join(f"p{k}={v:.4f}" for k, v in stats.percentiles.items()))
    A(f"- pixel shares: " + ", ".join(f"{k} {100 * v:.2f}%" for k, v in stats.band_frac.items()))
    A(f"- foreground = {100 * stats.fg_frac:.1f}% of the pixels, but it holds only "
      f"**{100 * stats.hl_in_fg:.1f}%** of the highlights (L>1): without separating fg and bg, "
      f"\"error on the highlights\" effectively means \"error on the envmap\"")
    A("")

    A("## Checks performed")
    A("")
    for c in checks:
        A(f"- `{c.strip()}`")
    A("")

    A("## Caveats")
    A("")
    A("1. **Circularity.** Every run is the minimum of its own loss on this data. The "
      "three metrics that coincide with the sweep's losses (linear MSE/PSNR, MAE, "
      "relMSE) are therefore excluded from the matrix and from the rankings; they "
      "survive only in `metrics_global.csv` and `metrics_per_frame_all_runs.csv`, which "
      "are raw dumps. If they are used to compare the runs, the comparison is circular.")
    A("2. **They are all training views.** `hold_out_preview` is `False` by default "
      "(`nerf/dataset.py`), so every frame evaluated here was seen during "
      "training. What is being measured is the quality of the fit, not generalisation.")
    A("3. **`psnr_db` in `training_metrics.csv`** is measured on the iteration's "
      "training batch (fg and bg mixed) and is `-10*log10(mse)` with `MAX_I=1` on HDR "
      "targets: a monotone remapping of the MSE, comparable between runs on the same "
      "data but not with the PSNR figures of the literature.")
    A("4. **The `loss` column** is in the units of its own loss: it only makes sense "
      "within a run, never between different runs. That is why the figure "
      "`train_loss_by_type.png` uses a separate panel per loss type.")
    A("5. **The existing artefacts mix pixel sets**: in "
      "`metrics_per_frame.csv` the `psnr` column is computed on the whole frame, while "
      "`psnr_tonemap_*`, `rel_err_*` and `residual_*` only on the foreground "
      "(`images_generator.py:2609` against `:2628-2631`). Here every metric is reported "
      "explicitly on `full`, `fg` and `bg`.")
    A("6. **`metrics_summary.txt` averages the per-frame PSNRs.** With a dynamic range "
      "going from 1.0 to 56 depending on the frame, that average is dominated by a few "
      "frames. The tables above instead aggregate the sums over every pixel and convert "
      "at the end; the per-frame distributions are in the boxplots.")
    A("7. **The sweep on disk has 6 configurations** (3 losses x 2 activations, a single "
      "decay), while `tab:results-nerf-ablation` in `Doc/chapters/results.tex` expects "
      "8 (2 losses x 2 activations x 2 decays). The thesis table has to be realigned.")
    A("")

    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"      + report.md")


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_root", nargs="?", default=DEFAULT_ROOT)
    ap.add_argument("-o", "--out", default=None,
                    help="output folder (default: <sweep_root>/_comparison)")
    ap.add_argument("--no-recompute", action="store_true",
                    help="only the figures derived from the CSVs already on disk")
    ap.add_argument("--reuse-cache", action="store_true",
                    help=f"reuse <out>/{CACHE_NAME} instead of re-reading the EXRs "
                         f"(to iterate on the figures without paying the I/O again)")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="restrict to these folder names")
    ap.add_argument("--visual-frame", type=int, default=None,
                    help="frame for the crop figure (default: the most HDR one)")
    ap.add_argument("--skybox-gt", default=None,
                    help="equirectangular EXR of the original skybox, for the "
                         "comparison with the runs' skybox_nerf_baked.exr "
                         "(default: the skybox_path field of run_manifest.json)")
    ap.add_argument("--no-spectrum-frames", action="store_true",
                    help="skip the per-frame spectrum figures "
                         "(one per frame in figures/spectrum_frames/)")
    args = ap.parse_args()

    root = Path(args.sweep_root)
    if not root.is_dir():
        print(f"ERROR: {root} is not a folder")
        return 2
    out = Path(args.out) if args.out else root / "_comparison"
    figdir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(root, args.runs)
    if not runs:
        print(f"ERROR: no run found under {root}")
        return 2
    print(f"Runs found ({len(runs)}):")
    for r in runs:
        it = r.iter_dir.name if r.iter_dir else "(no render)"
        print(f"  {r.key:28s} act={r.activation:9s} loss={r.loss:12s} "
              f"decay={r.decay:g}  {it}")
    print(f"Output: {out}")
    print()

    print("[figures from the CSVs]")
    fig_training(runs, figdir)
    fig_bias_bins_existing(runs, figdir)
    fig_existing_per_frame(runs, figdir)

    if args.no_recompute:
        print("\n--no-recompute: skipping the recomputation from the EXRs.")
        return 0

    usable = [r for r in runs if r.iter_dir is not None]
    if not usable:
        print("No run with renders on disk: nothing to recompute.")
        return 0

    ref = usable[0]
    tf_path = ref.scene_dir / "transforms_extended.json"
    frames = json.loads(tf_path.read_text(encoding="utf-8"))["frames"]
    gt_paths = sorted(ref.iter_dir.glob("frame_*_gt.exr"))
    n_frames = min(len(gt_paths), len(frames))
    gt_paths = [ref.iter_dir / f"frame_{i:03d}_gt.exr" for i in range(n_frames)]
    mask_paths = [ref.scene_dir / frames[i]["mask_path"] for i in range(n_frames)]
    missing = [p for p in gt_paths + mask_paths if not p.exists()]
    if missing:
        print(f"ERROR: missing files, for instance {missing[0]}")
        return 2
    for r in usable:
        r.n_frames = n_frames

    print()
    cached = load_cache(out, usable) if args.reuse_cache else None
    if cached is not None:
        results, stats = cached
        print(f"[1-2/3] Cache reused from {out / CACHE_NAME}: no EXR re-read.")
    else:
        stats = gt_prepass(gt_paths, mask_paths)
        print()
        results = compute_all(usable, gt_paths, mask_paths, stats)
        save_cache(out, results, stats)

    print("\n[3/3] Checks, tables and figures")
    checks = verify_against_artifacts(results)
    checks += verify_decomposition(results, stats.mu_scale)
    checks += verify_spectrum(results, stats)
    for c in checks:
        print(c)
    smax = max(max(x["smape_full"] for x in res.per_frame) for res in results.values())
    print(f"  [{'OK ' if 0 <= smax <= 1 else 'FAIL'}] SMAPE inside [0,1]: max={smax:.4f}")

    # Before the tables and the figures: two columns of the matrix come from here.
    an = analyze_spectrum(results, usable, stats)
    print("      notable frames (norm channel): " +
          ", ".join(f"{k}={v}" for k, v in an.notable.items()))
    print("      spectrum ranking (W1 on the norm channel, in decades): " +
          ", ".join(f"{usable[i].label} {an.w1_pooled[i, NORM_CH]:.4f}" for i in an.order))

    write_metrics_global(out, results, stats.mu_scale)
    write_metrics_by_band(out, results, stats.mu_scale)
    write_bias_by_decade(out, results, stats.mu_scale)
    write_per_frame(out, results)
    print(f"      + metrics_global.csv, metrics_by_band.csv, bias_by_decade.csv, "
          f"metrics_per_frame_all_runs.csv")

    fig_quality_boxplots(results, usable, figdir)
    fig_per_frame_referee(results, usable, stats, figdir)
    for ps in PIXEL_SETS:
        fig_metric_matrix(results, usable, stats.mu_scale, figdir, ps, an)
    fig_rank_bump(results, usable, stats.mu_scale, figdir, an)
    fig_bias_by_decade(results, usable, stats.mu_scale, figdir)
    fig_smape_by_decade(results, usable, stats.mu_scale, figdir)
    fig_highlight_bias(results, usable, stats.mu_scale, figdir)
    fig_error_split(results, usable, stats.mu_scale, figdir)
    fig_error_split_fg_bg(results, usable, stats.mu_scale, figdir, stats.hl_in_fg)
    # most HDR frame by p99.9 and not by the maximum: the maximum is a single pixel
    # and would pick a frame with a point source in view
    vf = (args.visual_frame if args.visual_frame is not None
          else int(np.argmax(stats.per_frame_p999)))
    fig_visual(usable, gt_paths, mask_paths, vf, figdir)

    # ── colour spectrum ──────────────────────────────────────────────────────
    write_spectrum_tables(out, an, usable)
    print("      + spectrum_distance.csv, spectrum_per_frame.csv")
    fig_spectrum_global(an, usable, figdir)
    fig_spectrum_ratio(an, usable, figdir)
    fig_spectrum_cdf(an, usable, figdir)
    fig_spectrum_qq(an, usable, figdir)
    fig_spectrum_per_frame(an, usable, figdir)
    fig_spectrum_box(an, usable, figdir)

    sky = None
    sky_path = resolve_skybox_gt(usable, args.skybox_gt)
    if sky_path is not None:
        spec_sky_gt, per_run_sky = spectrum_skybox(usable, sky_path)
        if per_run_sky:
            fig_spectrum_skybox(spec_sky_gt, per_run_sky, usable, sky_path, figdir)
            write_spectrum_skybox_table(out, spec_sky_gt, per_run_sky, usable)
            print("      + spectrum_skybox.csv")
            sky = (spec_sky_gt, per_run_sky, sky_path)

    if not args.no_spectrum_frames:
        fig_spectrum_frames(an, usable, results, stats, figdir)

    write_report(out, usable, results, stats, checks, root, n_frames, an=an, sky=sky)
    print(f"\nDone. Open: {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
