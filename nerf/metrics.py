"""nerf/metrics.py — systematic analysis of the NeRF's colour bias.

Pure NumPy functions: pred-vs-GT comparison per luminance band, tonemapped
PSNR, relative error on the highlights.
plot_bias_scatter requires matplotlib (Agg backend).

Compatible with every training combination (rgb_activation × loss_type): both
activations ("exp", "softplus") produce HDR output (values > 1 are allowed).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_LUMA_COEFF = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)  # Rec.709

_PANEL_LABELS = ["R", "G", "B", "Luma"]
_PANEL_COLORS = ["#E63946", "#2A9D8F", "#457B9D", "#9B59B6"]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _luminance(arr: np.ndarray) -> np.ndarray:
    """(..., 3) float32 → (...) float32  Rec.709 luminance."""
    return (arr.astype(np.float32) * _LUMA_COEFF).sum(-1)


def _signed_log_norm(diff: np.ndarray):
    """Symmetric AsinhNorm normalization for signed heatmaps.

    Linear near 0, logarithmic in the tails on both signs, so outliers stay visible
    and distinguishable (no clamping, no saturation).

    Parameters are chosen automatically from the data:
    - vmin = −max|diff|, vmax = +max|diff|  (true maximum, no clamp)
    - linear_width = max(median|diff|, 1e-4) (linear/log threshold)

    Falls back to SymLogNorm on matplotlib < 3.5 (no AsinhNorm there).
    """
    import matplotlib.colors as mcolors

    M = max(float(np.abs(diff).max()), 1e-5)
    linear_width = max(float(np.median(np.abs(diff[diff != 0]))) if (diff != 0).any()
                       else 1e-4, 1e-4)
    try:
        return mcolors.AsinhNorm(linear_width=linear_width, vmin=-M, vmax=M)
    except AttributeError:
        # matplotlib < 3.5: fallback
        return mcolors.SymLogNorm(linthresh=linear_width, vmin=-M, vmax=M, base=10)


def _mask_flatten(arr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Flatten arr with an optional boolean mask.

    arr : (...) — any shape
    mask: same spatial shape as arr (or None)
    Returns (N,) float32
    """
    a = arr.astype(np.float32)
    if mask is not None:
        return a[mask.astype(bool)]
    return a.ravel()


# ---------------------------------------------------------------------------
# binned_median_curve
# ---------------------------------------------------------------------------

def binned_median_curve(
    pred_flat: np.ndarray,
    gt_flat: np.ndarray,
    n_bins: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Median of pred per gt band (bins adaptive by percentile).

    The bins are defined as percentiles of gt_flat, so this works correctly on any
    HDR dynamic range, without absolute thresholds.

    Parameters
    ----------
    pred_flat, gt_flat : (N,) float32 — already masked, single channel or luminance.

    Returns
    -------
    centers      : (K,) — median gt in each bin (X axis of the curve)
    pred_medians : (K,) — median pred in each bin (Y axis of the curve)
    gt_medians   : (K,) — identical to centers, kept for interface consistency
    counts       : (K,) int64 — pixels per bin
    """
    p = pred_flat.astype(np.float32)
    g = gt_flat.astype(np.float32)

    pct_edges = np.linspace(0.0, 100.0, n_bins + 1)
    edges = np.unique(np.percentile(g, pct_edges))
    K = len(edges) - 1
    if K < 1:
        empty = np.array([], dtype=np.float32)
        return empty, empty, empty, np.array([], dtype=np.int64)

    centers_out:   list[float] = []
    pred_meds_out: list[float] = []
    gt_meds_out:   list[float] = []
    counts_out:    list[int]   = []

    for k in range(K):
        lo, hi = edges[k], edges[k + 1]
        if k < K - 1:
            sel = (g >= lo) & (g < hi)
        else:
            sel = (g >= lo) & (g <= hi)
        n = int(sel.sum())
        counts_out.append(n)
        if n == 0:
            centers_out.append(float(0.5 * (lo + hi)))
            pred_meds_out.append(float("nan"))
            gt_meds_out.append(float(0.5 * (lo + hi)))
        else:
            centers_out.append(float(np.median(g[sel])))
            pred_meds_out.append(float(np.median(p[sel])))
            gt_meds_out.append(float(np.median(g[sel])))

    return (
        np.array(centers_out,   dtype=np.float32),
        np.array(pred_meds_out, dtype=np.float32),
        np.array(gt_meds_out,   dtype=np.float32),
        np.array(counts_out,    dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# tonemapped_psnr
# ---------------------------------------------------------------------------

def tonemapped_psnr(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray | None = None,
    mode: str = "clip",
) -> float:
    """PSNR after tonemapping, to judge the diffuse range without HDR contamination.

    mode: "clip"     → clip to [0, 1]
          "reinhard" → x / (1 + x)
    mask: (H, W) bool, foreground; None = all pixels.
    """
    p = pred.astype(np.float32)
    g = gt.astype(np.float32)
    if mask is not None:
        m = mask.astype(bool)
        p = p[m]
        g = g[m]
    else:
        p = p.ravel()
        g = g.ravel()

    if mode == "reinhard":
        p = p / (1.0 + p)
        g = g / (1.0 + g)
    else:  # "clip"
        p = np.clip(p, 0.0, 1.0)
        g = np.clip(g, 0.0, 1.0)

    mse = float(np.mean((p - g) ** 2))
    return -10.0 * np.log10(mse + 1e-10)


# ---------------------------------------------------------------------------
# highlight_percentile_error
# ---------------------------------------------------------------------------

def highlight_percentile_error(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray | None = None,
    pcts: tuple[float, ...] = (99.0, 99.9),
) -> dict[float, float]:
    """Mean relative error |pred-gt|/|gt| over the pixels above the GT luminance percentile.

    A robust measure of highlight reproduction; NaN when there are not enough pixels.
    """
    p = pred.astype(np.float32)
    g = gt.astype(np.float32)

    # luminance for the percentile threshold
    luma_g = _luminance(g) if g.ndim == 3 else g

    if mask is not None:
        m = mask.astype(bool)
        luma_flat = luma_g[m]
        p_flat    = p[m].reshape(-1, 3) if p.ndim == 3 else p[m].ravel()
        g_flat    = g[m].reshape(-1, 3) if g.ndim == 3 else g[m].ravel()
    else:
        luma_flat = luma_g.ravel()
        p_flat    = p.reshape(-1, 3) if p.ndim == 3 else p.ravel()
        g_flat    = g.reshape(-1, 3) if g.ndim == 3 else g.ravel()

    result: dict[float, float] = {}
    for pct in pcts:
        thresh = float(np.percentile(luma_flat, pct))
        sel = luma_flat >= thresh
        if sel.sum() == 0:
            result[pct] = float("nan")
            continue
        rel = np.abs(p_flat[sel] - g_flat[sel]) / (np.abs(g_flat[sel]) + 1e-3)
        result[pct] = float(rel.mean())
    return result


# ---------------------------------------------------------------------------
# signed_residual_stats
# ---------------------------------------------------------------------------

def signed_residual_stats(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Signed residual (pred - gt) statistics: global, and highlights only (gt > 1).

    Keys returned: mean, median, mean_highlight, median_highlight.
    """
    p = pred.astype(np.float32)
    g = gt.astype(np.float32)

    if mask is not None:
        m = mask.astype(bool)
        p = p[m]
        g = g[m]
    else:
        p = p.ravel()
        g = g.ravel()

    res     = p - g
    hl_sel  = g > 1.0  # highlights: values outside the LDR range

    return {
        "mean":             float(res.mean()),
        "median":           float(np.median(res)),
        "mean_highlight":   float(res[hl_sel].mean())   if hl_sel.any() else float("nan"),
        "median_highlight": float(np.median(res[hl_sel])) if hl_sel.any() else float("nan"),
    }


# ---------------------------------------------------------------------------
# plot_bias_scatter
# ---------------------------------------------------------------------------

def _log10_label(t: float) -> str:
    """Turn a log10 exponent into a readable label (e.g. -2 → '0.01', 0 → '1', 1 → '10')."""
    v = 10.0 ** t
    if t >= 0:
        return str(int(round(v)))
    return f"{v:.4g}"


def plot_bias_scatter(
    pred_hw3: np.ndarray,
    gt_hw3: np.ndarray,
    mask_hw: np.ndarray | None,
    out_png: str,
    title: str = "",
    n_bins: int = 20,
    eps: float = 1e-3,
    gridsize: int = 60,
) -> None:
    """Save a log-log pred-vs-gt density scatter with the y=x bisector and a median curve.

    Four panels side by side: R, G, B, Luminance.
    Density below the diagonal = underestimation; above = overestimation.
    The orange curve = median(pred) per gt bin (adaptive to the percentiles).

    Parameters
    ----------
    pred_hw3, gt_hw3 : (..., 3) float32 — any spatial shape
    mask_hw          : boolean foreground mask (same spatial shape as pred/gt)
                       None → all pixels
    out_png          : output path (intermediate directories are created)
    title            : plot suptitle (e.g. "frame_003  PSNR=25.4 dB / tonemap=30.1 dB")
    n_bins           : adaptive bins for the median curve
    eps              : lower clamp before log10 (avoids log(0))
    gridsize         : hexbin resolution
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("    ⚠  matplotlib not available: bias scatter skipped")
        return

    p = pred_hw3.astype(np.float32)
    g = gt_hw3.astype(np.float32)
    luma_p = _luminance(p)
    luma_g = _luminance(g)

    # Extract the channels through the mask
    if mask_hw is not None:
        m = mask_hw.astype(bool)
        ch_p = [p[m, 0], p[m, 1], p[m, 2], luma_p[m]]
        ch_g = [g[m, 0], g[m, 1], g[m, 2], luma_g[m]]
    else:
        ch_p = [p[..., 0].ravel(), p[..., 1].ravel(), p[..., 2].ravel(), luma_p.ravel()]
        ch_g = [g[..., 0].ravel(), g[..., 1].ravel(), g[..., 2].ravel(), luma_g.ravel()]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    for ax, label, color, cp, cg in zip(axes, _PANEL_LABELS, _PANEL_COLORS, ch_p, ch_g):
        # log10 transform with clamping
        log_g = np.log10(np.clip(cg, eps, None))
        log_p = np.log10(np.clip(cp, eps, None))

        lim_lo = float(np.log10(eps)) - 0.05
        lim_hi = float(max(log_g.max(), log_p.max(), 0.1)) + 0.15

        # Hexbin: colour = log10(count), for legibility over a wide dynamic range
        hb = ax.hexbin(
            log_g, log_p,
            gridsize=gridsize,
            cmap="Blues",
            bins="log",
            mincnt=1,
            extent=[lim_lo, lim_hi, lim_lo, lim_hi],
        )
        fig.colorbar(hb, ax=ax, pad=0.02, label="log₁₀(count)")

        # y = x bisector
        ax.plot(
            [lim_lo, lim_hi], [lim_lo, lim_hi],
            "--", color="#E74C3C", linewidth=1.3, alpha=0.9, label="y = x",
        )

        # Per-bin median curve (computed in linear space → projected to log10)
        centers, pred_meds, _, counts = binned_median_curve(cp, cg, n_bins=n_bins)
        valid = (~np.isnan(pred_meds)) & (counts > max(5, len(cp) // (n_bins * 10)))
        if valid.sum() > 1:
            lc = np.log10(np.clip(centers[valid], eps, None))
            lp = np.log10(np.clip(pred_meds[valid], eps, None))
            ax.plot(lc, lp, "-o", color="#F39C12", linewidth=1.8,
                    markersize=3, alpha=0.9, label="median pred")

        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.set_xlabel("log₁₀(gt)")
        ax.set_ylabel("log₁₀(pred)")
        ax.set_title(label, color=color, fontweight="bold")
        ax.legend(fontsize=7, loc="upper left")

        # Axis labels as linear values
        tick_range = np.arange(int(np.ceil(lim_lo)), int(np.floor(lim_hi)) + 1, dtype=float)
        if len(tick_range) > 0:
            ax.set_xticks(tick_range)
            ax.set_yticks(tick_range)
            labels_str = [_log10_label(t) for t in tick_range]
            ax.set_xticklabels(labels_str, fontsize=7)
            ax.set_yticklabels(labels_str, fontsize=7)

    if title:
        fig.suptitle(title, fontsize=9, y=1.02)

    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# plot_error_heatmap
# ---------------------------------------------------------------------------

def plot_error_heatmap(
    pred_hw3: np.ndarray,
    gt_hw3: np.ndarray,
    out_png: str,
    title: str = "",
) -> None:
    """Per-frame diagnostic heatmap: GT, Pred (clipped to [0,1]) and the signed norm difference.

    No mask: the whole frame (model + background/skybox) goes into the heatmap. The
    visual contrast between skybox (error ≈0) and model (concentrated bias) is immediate.

    Panels (1×3):
    1. GT  clip [0, 1]
    2. Pred clip [0, 1]
    3. ‖GT‖ − ‖pred‖  (signed per-pixel RGB L2 norm) — magma colormap, AsinhNorm scale

    Positive values = the NeRF underestimates the colour magnitude.
    Negative values = the NeRF overestimates.

    Parameters
    ----------
    pred_hw3, gt_hw3 : (H, W, 3) float32
    out_png          : output path (intermediate directories are created)
    title            : plot suptitle
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("    ⚠  matplotlib not available: heatmap skipped")
        return

    p = pred_hw3.astype(np.float32)
    g = gt_hw3.astype(np.float32)

    norm_gt   = np.linalg.norm(g, axis=-1)   # (H, W)
    norm_pred = np.linalg.norm(p, axis=-1)   # (H, W)
    diff = norm_gt - norm_pred               # signed: positive where the NeRF underestimates

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 1. GT clip
    axes[0].imshow(np.clip(g, 0.0, 1.0))
    axes[0].set_title("GT  (clip [0,1])", fontsize=9)
    axes[0].axis("off")

    # 2. Pred clip
    axes[1].imshow(np.clip(p, 0.0, 1.0))
    axes[1].set_title("Pred NeRF  (clip [0,1])", fontsize=9)
    axes[1].axis("off")

    # 3. signed ‖GT‖ − ‖pred‖ — AsinhNorm scale: preserves outliers, linear near 0
    im = axes[2].imshow(diff, cmap="magma", norm=_signed_log_norm(diff), interpolation="nearest")
    axes[2].set_title("‖GT‖ − ‖pred‖  (positive = NeRF underestimates)", fontsize=9)
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, fontsize=9, y=1.01)

    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# plot_skybox_compare
# ---------------------------------------------------------------------------

def plot_skybox_compare(
    gt_hw3: np.ndarray,
    baked_hw3: np.ndarray,
    out_png: str,
    title: str = "",
) -> None:
    """Comparison heatmap of the GT skybox (native HDR) against the NeRF-baked one.

    The baked map is brought to GT resolution (Lanczos upsample, channel by channel via
    PIL) before the difference is computed. The baked skybox is assumed to be visually
    aligned with the GT already (yaw=0, the bake inverts sampleEnvmap exactly): no rotation search.

    Panels (1×3):
    1. GT  clip [0, 1]
    2. NeRF baked, upsampled to GT resolution  clip [0, 1]
    3. ‖GT‖ − ‖baked‖  (signed per-pixel RGB L2 norm) — magma colormap, AsinhNorm scale

    Positive values = the NeRF bake underestimates the magnitude.
    In the title: mean ratio of the norms (≈1 = correct mean scale) and mean difference.

    Parameters
    ----------
    gt_hw3    : (H_gt, W_gt, 3) float32 — GT skybox at native resolution
    baked_hw3 : (H_b,  W_b,  3) float32 — NeRF-baked skybox (lower resolution)
    out_png   : output path
    title     : base suptitle (statistics are appended automatically)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("    ⚠  matplotlib not available: skybox compare skipped")
        return

    from PIL import Image

    g = gt_hw3.astype(np.float32)
    b = baked_hw3.astype(np.float32)
    h_gt, w_gt = g.shape[:2]
    h_b,  w_b  = b.shape[:2]

    # Upsample baked → GT resolution (Lanczos, channel by channel on float32)
    if (h_b, w_b) != (h_gt, w_gt):
        b_up = np.stack([
            np.array(Image.fromarray(b[..., c]).resize((w_gt, h_gt), Image.LANCZOS))
            for c in range(3)
        ], axis=-1)
    else:
        b_up = b.copy()

    # Title statistics: mean ratio of the norms and mean of the signed difference
    norm_gt   = np.linalg.norm(g,    axis=-1)
    norm_baked = np.linalg.norm(b_up, axis=-1)
    diff = norm_gt - norm_baked  # signed: positive where the bake underestimates

    eps = 1e-5
    ratio_norm = float(norm_baked.mean() / max(norm_gt.mean(), eps))
    mean_diff  = float(diff.mean())
    stat_str   = f"norm_ratio={ratio_norm:.3f}  mean(‖gt‖-‖baked‖)={mean_diff:+.4f}"
    full_title = f"{title}  [{stat_str}]" if title else stat_str

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 1. GT clip
    axes[0].imshow(np.clip(g, 0.0, 1.0))
    axes[0].set_title(f"GT  ({w_gt}x{h_gt})", fontsize=9)
    axes[0].axis("off")

    # 2. Baked clip (upsampled)
    axes[1].imshow(np.clip(b_up, 0.0, 1.0))
    axes[1].set_title(f"Baked NeRF  ({w_b}x{h_b} → {w_gt}x{h_gt})", fontsize=9)
    axes[1].axis("off")

    # 3. signed ‖GT‖ − ‖baked‖ — AsinhNorm scale: preserves outliers, linear near 0
    im = axes[2].imshow(diff, cmap="magma", norm=_signed_log_norm(diff), interpolation="nearest")
    axes[2].set_title("‖GT‖ − ‖baked‖  (positive = bake underestimates)", fontsize=9)
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(full_title, fontsize=9, y=1.01)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
