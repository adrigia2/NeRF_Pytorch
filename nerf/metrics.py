"""nerf/metrics.py — Analisi sistematica del bias colore del NeRF.

Funzioni pure NumPy: confronto pred-vs-GT per fascia di luminanza,
PSNR tonemappato, errore relativo agli highlight.
plot_bias_scatter richiede matplotlib (backend Agg).

Compatibile con tutte le combinazioni di training (rgb_activation × loss_type):
entrambe le attivazioni ("exp", "softplus") producono output HDR (valori > 1 ammessi).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------
_LUMA_COEFF = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)  # Rec.709

_PANEL_LABELS = ["R", "G", "B", "Luma"]
_PANEL_COLORS = ["#E63946", "#2A9D8F", "#457B9D", "#9B59B6"]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _luminance(arr: np.ndarray) -> np.ndarray:
    """(..., 3) float32 → (...) float32  luminanza Rec.709."""
    return (arr.astype(np.float32) * _LUMA_COEFF).sum(-1)


def _mask_flatten(arr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Appiattisce arr con maschera booleana opzionale.

    arr : (...) — qualsiasi forma
    mask: stessa forma spaziale di arr (o None)
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
    """Mediana di pred per fascia di gt (bin adattativi per percentili).

    I bin sono definiti come percentili del gt_flat: così funziona
    correttamente su qualsiasi dinamica HDR, senza soglie assolute.

    Parameters
    ----------
    pred_flat, gt_flat : (N,) float32 — già mascherati, singolo canale o luminanza.

    Returns
    -------
    centers      : (K,) — mediana gt in ogni bin (asse X della curva)
    pred_medians : (K,) — mediana pred in ogni bin (asse Y della curva)
    gt_medians   : (K,) — identica a centers per coerenza d'interfaccia
    counts       : (K,) int64 — pixel per bin
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
    """PSNR dopo tonemapping, per valutare il range diffuso senza inquinamento HDR.

    mode: "clip"     → clip a [0, 1]
          "reinhard" → x / (1 + x)
    mask: (H, W) bool, foreground; None = tutti i pixel.
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
    """Errore relativo medio |pred-gt|/|gt| sui pixel sopra il percentile di luminanza GT.

    Misura robusta della resa degli highlight; NaN se non ci sono pixel sufficienti.
    """
    p = pred.astype(np.float32)
    g = gt.astype(np.float32)

    # luminanza per la soglia percentile
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
    """Statistiche del residuo con segno (pred - gt): globale e solo highlight (gt > 1).

    Keys restituiti: mean, median, mean_highlight, median_highlight.
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
    hl_sel  = g > 1.0  # highlight: valori fuori range LDR

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
    """Converte un esponente log10 in etichetta leggibile (es. -2 → '0.01', 0 → '1', 1 → '10')."""
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
    """Salva scatter di densità pred-vs-gt in log-log con bisettrice y=x e curva mediana.

    4 pannelli affiancati: R, G, B, Luminanza.
    Densità sotto la diagonale = sottostima; sopra = sovrastima.
    La curva arancio = mediana(pred) per bin di gt (adattiva ai percentili).

    Parameters
    ----------
    pred_hw3, gt_hw3 : (..., 3) float32 — qualsiasi forma spaziale
    mask_hw          : maschera foreground booleana (stesso shape spaziale di pred/gt)
                       None → tutti i pixel
    out_png          : percorso output (le directory intermedie vengono create)
    title            : suptitle del plot (es. "frame_003  PSNR=25.4 dB / tonemap=30.1 dB")
    n_bins           : bin adattativi per la curva mediana
    eps              : clamp minimo prima di log10 (evita log(0))
    gridsize         : risoluzione hexbin
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("    ⚠  matplotlib non disponibile: scatter bias saltato")
        return

    p = pred_hw3.astype(np.float32)
    g = gt_hw3.astype(np.float32)
    luma_p = _luminance(p)
    luma_g = _luminance(g)

    # Estrae i canali con maschera
    if mask_hw is not None:
        m = mask_hw.astype(bool)
        ch_p = [p[m, 0], p[m, 1], p[m, 2], luma_p[m]]
        ch_g = [g[m, 0], g[m, 1], g[m, 2], luma_g[m]]
    else:
        ch_p = [p[..., 0].ravel(), p[..., 1].ravel(), p[..., 2].ravel(), luma_p.ravel()]
        ch_g = [g[..., 0].ravel(), g[..., 1].ravel(), g[..., 2].ravel(), luma_g.ravel()]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    for ax, label, color, cp, cg in zip(axes, _PANEL_LABELS, _PANEL_COLORS, ch_p, ch_g):
        # Trasformazione in log10 con clamp
        log_g = np.log10(np.clip(cg, eps, None))
        log_p = np.log10(np.clip(cp, eps, None))

        lim_lo = float(np.log10(eps)) - 0.05
        lim_hi = float(max(log_g.max(), log_p.max(), 0.1)) + 0.15

        # Hexbin: colore = log10(conteggio) per leggibilità su dinamica ampia
        hb = ax.hexbin(
            log_g, log_p,
            gridsize=gridsize,
            cmap="Blues",
            bins="log",
            mincnt=1,
            extent=[lim_lo, lim_hi, lim_lo, lim_hi],
        )
        fig.colorbar(hb, ax=ax, pad=0.02, label="log₁₀(count)")

        # Bisettrice y = x
        ax.plot(
            [lim_lo, lim_hi], [lim_lo, lim_hi],
            "--", color="#E74C3C", linewidth=1.3, alpha=0.9, label="y = x",
        )

        # Curva mediana per bin (in spazio lineare → proiettata in log10)
        centers, pred_meds, _, counts = binned_median_curve(cp, cg, n_bins=n_bins)
        valid = (~np.isnan(pred_meds)) & (counts > max(5, len(cp) // (n_bins * 10)))
        if valid.sum() > 1:
            lc = np.log10(np.clip(centers[valid], eps, None))
            lp = np.log10(np.clip(pred_meds[valid], eps, None))
            ax.plot(lc, lp, "-o", color="#F39C12", linewidth=1.8,
                    markersize=3, alpha=0.9, label="mediana pred")

        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.set_xlabel("log₁₀(gt)")
        ax.set_ylabel("log₁₀(pred)")
        ax.set_title(label, color=color, fontweight="bold")
        ax.legend(fontsize=7, loc="upper left")

        # Etichette degli assi come valori lineari
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
