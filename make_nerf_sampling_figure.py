#!/usr/bin/env python
"""make_nerf_sampling_figure.py -- Le due strategie di campionamento (figura 3.14).

    python make_nerf_sampling_figure.py --out ../Doc/images/diagrams

Scrive due PNG, uno per sottofigura:

  sampling_traditional.png   NeRF classico: coarse uniforme + fine per importance
  sampling_depth_guided.png  il nostro: pochi campioni in una fetta attorno alla depth

Il pannello di sinistra riproduce il campionamento gerarchico del NeRF originale, non un
generico "tanti punti": prima una passata coarse stratificata su tutto l'intervallo
[t_near, t_far], poi una fine ridistribuita secondo i pesi della coarse, che si addensa
dove la densita' e' alta.  Disegnarne solo una delle due darebbe l'idea sbagliata, cioe'
che il NeRF sprechi campioni per ingenuita' invece che per non sapere dove sia la
superficie.

Il pannello di destra usa i numeri veri della configurazione operativa: finestra
`depth_window` = 0.05 attorno alla distanza restituita dal pass di depth, e 5 campioni
per raggio, gli stessi che finiscono in `tab:results-config`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 13})

DPI = 190

# ── Parametri ────────────────────────────────────────────────────────────────
T_NEAR, T_FAR = 0.0, 1.0       # intervallo del raggio, in unita' arbitrarie
T_SURF = 0.62                  # dove sta la superficie
N_COARSE, N_FINE = 32, 24      # NeRF originale: due passate
DEPTH_WINDOW = 0.05            # RenderConfig / NerfConfig: +- attorno alla depth
N_GUIDED = 5                   # campioni per raggio della nostra pipeline
SIGMA_FINE = 0.055             # larghezza della gaussiana dei pesi della coarse

C_RAY   = "#33404d"
C_CO    = "#5b9bd5"
C_FI    = "#8452c9"
C_OURS  = "#2ca02c"
C_SURF  = "#b9c3cd"
C_DEPTH = "#d62728"


def ray_axis(ax, y=0.0, label_surface=True):
    ax.annotate("", xy=(T_FAR + 0.08, y), xytext=(T_NEAR - 0.06, y),
                arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.5",
                                color=C_RAY, lw=1.8))
    ax.scatter([T_NEAR - 0.06], [y], s=70, color=C_RAY, zorder=5)
    ax.text(T_NEAR - 0.09, y + 0.10, r"$\mathbf{o}$", ha="center", va="bottom",
            fontsize=15, fontweight="bold", color=C_RAY)
    ax.text(T_FAR + 0.10, y, r"$t$", ha="left", va="center", fontsize=15, color=C_RAY)
    # La superficie: una banda, non una linea, perche' e' spessa quanto la geometria
    ax.axvspan(T_SURF, T_SURF + 0.035, color=C_SURF, zorder=1)
    if label_surface:
        ax.text(T_SURF + 0.018, -0.88, "surface", ha="center", va="bottom",
                fontsize=11, color="#5f6c79")


def frame(ax, title):
    ax.set_xlim(T_NEAR - 0.20, T_FAR + 0.24)
    ax.set_ylim(-0.95, 0.95)
    ax.set_title(title, fontsize=13, pad=10)
    ax.axis("off")


def fig_traditional(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    ray_axis(ax)

    # Coarse: stratificato uniforme su tutto l'intervallo
    edges = np.linspace(T_NEAR, T_FAR, N_COARSE + 1)
    t_coarse = 0.5 * (edges[:-1] + edges[1:])
    ax.scatter(t_coarse, np.full_like(t_coarse, 0.44), s=26, color=C_CO,
               zorder=4, edgecolors="white", linewidths=0.5)
    ax.text(T_NEAR - 0.16, 0.44, "coarse", ha="right", va="center",
            fontsize=11, color=C_CO)

    # Fine: ridistribuita sui pesi della coarse, quindi addensata sulla superficie.
    # Si campiona la CDF di una gaussiana centrata sulla superficie: e' la forma che i
    # pesi assumono una volta che la coarse ha trovato dove sta la densita'.
    q = (np.arange(N_FINE) + 0.5) / N_FINE
    t_fine = np.clip(T_SURF + SIGMA_FINE * np.sqrt(2) * _erfinv(2 * q - 1),
                     T_NEAR, T_FAR)
    ax.scatter(t_fine, np.full_like(t_fine, -0.44), s=26, color=C_FI,
               zorder=4, edgecolors="white", linewidths=0.5)
    ax.text(T_NEAR - 0.16, -0.44, "fine", ha="right", va="center",
            fontsize=11, color=C_FI)

    frame(ax, f"{N_COARSE} coarse over the whole ray, then {N_FINE} fine")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}  ({N_COARSE}+{N_FINE} campioni)")


def _erfinv(y):
    """Inversa della erf, senza tirarsi dietro scipy per due dozzine di punti."""
    a = 0.147
    ln = np.log(1 - y * y)
    tt1 = 2 / (np.pi * a) + ln / 2
    return np.sign(y) * np.sqrt(np.sqrt(tt1 * tt1 - ln / a) - tt1)


def fig_depth_guided(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    ray_axis(ax)

    lo, hi = T_SURF - DEPTH_WINDOW, T_SURF + DEPTH_WINDOW
    ax.axvspan(lo, hi, color=C_OURS, alpha=0.14, zorder=1)
    for x in (lo, hi):
        ax.plot([x, x], [-0.45, 0.45], color=C_OURS, lw=1.0, ls=(0, (3, 3)), zorder=2)

    t = lo + (hi - lo) * (np.arange(N_GUIDED) + 0.5) / N_GUIDED
    ax.scatter(t, np.zeros_like(t), s=44, color=C_OURS, zorder=5,
               edgecolors="white", linewidths=0.7)

    ax.annotate("", xy=(hi, 0.60), xytext=(lo, 0.60),
                arrowprops=dict(arrowstyle="<|-|>,head_width=0.18,head_length=0.4",
                                color=C_OURS, lw=1.3))
    ax.text(T_SURF, 0.68, rf"$\pm{DEPTH_WINDOW}$", ha="center", va="bottom",
            fontsize=12, color=C_OURS)

    ax.plot([T_SURF, T_SURF], [-0.45, -0.05], color=C_DEPTH, lw=1.6, zorder=3)
    ax.text(T_SURF, -0.52, r"$t$ from the depth pass", ha="center", va="top",
            fontsize=11, color=C_DEPTH)

    frame(ax, f"{N_GUIDED} samples in a slab around the surface")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}  ({N_GUIDED} campioni, finestra ±{DEPTH_WINDOW})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="../Doc/images/diagrams")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Campionamento NeRF → {out.resolve()}")
    fig_traditional(out / "sampling_traditional.png")
    fig_depth_guided(out / "sampling_depth_guided.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
