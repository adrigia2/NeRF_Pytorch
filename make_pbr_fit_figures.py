#!/usr/bin/env python
"""make_pbr_fit_figures.py -- Figure dell'esempio numerico del fit PBR.

    python make_pbr_fit_figures.py --out ../figure_review/pbr-fit
    python make_pbr_fit_figures.py --out ../Doc/images/pbr-fit \
           --figures worked_combined,moments_per_camera        (le due gia' in tesi)

Genera i grafici di "A Worked Example" (Doc/chapters/implementation.tex, subsubsection
dopo eq:pbr-residual): un texel, tre camere, la regressione C_jc = alpha_c + beta*L_jc
e la scelta dell'apertura.

  fit_color_raw.png         C_j per camera: swatch + barre R/G/B, con la media per canale
  fit_color_centered.png    lo stesso dopo il centramento: Delta C
  fit_cone_raw.png          L_j al candidato mostrato in tabella
  fit_cone_centered.png     Delta L
  fit_worked_combined.png   i quattro sopra in una griglia sola (alternativa per la pagina)
  fit_scatter.png           Delta L vs Delta C, i 9 punti, la retta di pendenza beta*
  fit_aperture_selection.png  perche' vince un'apertura e non un'altra
  fit_residual_materials.png  res(Theta) per metallo, dielettrico lucido, diffuso

I numeri di C e L al candidato mostrato sono quelli di tab:pbr-worked-example, scritti
qui una volta sola come INPUT: medie, deviazioni, V_LL, V_CL, V_CC, beta* e res sono
sempre ricalcolati e confrontati con la tabella da _check_against_table(), cosi' se un
giorno la tabella cambia il confronto salta e non restano figure che raccontano altri
numeri.  --print-table rigenera le righe LaTeX per il confronto a vista.

Le aperture diverse da quella in tabella non esistono nella tesi e vanno modellate.
Il modello e' quello di un cono che si allarga: la media sul cono decade verso la media
sull'emisfero H_c con una rapidita' kappa_j che dipende dalla camera, perche' dipende da
quanto e' compatta la sorgente attorno al suo raggio riflesso,

    u(Theta) = 1 - cos(Theta/2)                       (angolo solido del cono / 2pi)
    L_jc(Theta) = H_c + (L*_jc - H_c) * phi_j(Theta)
    phi_j = (1 + kappa_j*u*) / (1 + kappa_j*u)        (phi_j(Theta*) = 1)

Tre proprieta' lo rendono utilizzabile: al candidato della tabella riproduce ESATTAMENTE
i suoi valori; per Theta -> 180 tutte le camere convergono su H_c, che e' quello che fa
un cono che si allarga; e siccome kappa_j cambia con la camera, allargando il cono cambia
la FORMA di Delta L fra camere e non solo la sua scala.  L'ultimo punto e' tutta la
figura: con kappa uguale per tutte, Delta L verrebbe riscalato in blocco, beta si
mangerebbe il fattore e la curva del residuo sarebbe piatta -- che e' poi il caso
degenere del texel diffuso.

H_c e kappa_j non sono liberi: sono stati scelti con una ricerca su griglia perche'
argmin_Theta res(Theta) cada sul candidato della tabella e il pozzo sia visibile
(vicini ~12x il minimo, estremi ~120x).  _check_against_table() riverifica anche questo.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Dati dell'esempio: input, non risultati ───────────────────────────────────
# tab:pbr-worked-example, blocco superiore.  (camera, canale)
C_OBS = np.array([[1.2, 2.1, 2.0],
                  [2.2, 3.2, 2.1],
                  [2.8, 3.7, 1.5]])
L_STAR = np.array([[1.0, 2.0, 3.0],
                   [3.0, 4.0, 3.0],
                   [4.0, 5.0, 2.0]])

# Griglia operativa delle aperture (images_generator.py, __main__): K = 14 candidati,
# indice 0 = raggio specchio.  Il candidato della tabella e' 45 gradi.
APERTURES = np.array([0., 5., 10., 15., 20., 30., 45., 60., 80., 100., 120., 140., 160., 180.])
K_STAR = 6

H_ENV = np.array([1.5, 2.3, 1.3])     # media sull'emisfero, per canale
KAPPA = np.array([20.0, 30.0, 3.0])   # compattezza della sorgente attorno a R_j

# Materiali della figura delle tre curve: stesso texel, stessa envmap, solo x e il lobo
# cambiano.  Il rumore e' ASSOLUTO e comune (stesso bake, stesso sensore): e' cio' che
# rende piatta la curva del diffuso senza doverla dichiarare piatta a mano.
NOISE_SIGMA = 0.045
NOISE_SEED = 19
MATERIALS = (("metal",     0.90, 2),
             ("glossy",    0.50, K_STAR),
             ("diffuse",   0.03, K_STAR))
ALBEDO_TERM = np.array([0.22, 0.26, 0.20])   # a_c*E_c/pi, il colore diffuso del texel

# ── Colori ────────────────────────────────────────────────────────────────────
# I canali sono entita', non slot di palette: restano rosso/verde/blu.  Con questi passi
# la coppia peggiore sta a Delta E 7.2 in deuteranopia, dentro la fascia 6-8 ammessa solo
# con codifica secondaria -- che c'e': i canali sono sempre in ordine R, G, B e etichettati
# sull'asse, quindi l'identita' non e' mai affidata al solo colore.
CH_COLORS = ("#e34948", "#008300", "#2a78d6")
CH_NAMES = ("R", "G", "B")
CAM_MARKERS = ("o", "s", "^")
# Serie categoriche (i tre materiali): slot 1-3 della palette validata, passano tutte le
# coppie.  L'aqua sta sotto 3:1 sul fondo chiaro, quindi le curve sono etichettate in
# posto e non solo in legenda.
MAT_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
# I momenti hanno una palette loro: con blu/arancio/verde, tre barre affiancate in un
# grafico dove altrove tre barre affiancate SONO i canali si leggono come un RGB.
# Viola, ocra e magenta (slot 7, 4, 5 della stessa palette) non contengono ne' blu ne'
# verde, quindi l'equivoco non e' proprio possibile.  Ocra e magenta stanno sotto 3:1
# sul fondo chiaro: qui il rilievo c'e', ogni barra porta il suo valore scritto.
MOMENT_COLORS = ("#4a3aa7", "#eda100", "#e87ba4")

# Nomi accettati da --figures, senza il prefisso 'fit_' del file.  Le due pubblicate in
# tesi sono worked_combined e moments_per_camera: Doc/images/pbr-fit tiene solo quelle,
# il resto vive in figure_review finche' non e' approvato.
FIGURES = ("color_raw", "color_centered", "cone_raw", "cone_centered",
           "worked_combined", "scatter", "moments_per_camera", "beta",
           "aperture_selection", "residual_materials")

C_INK = "#222222"
C_SOFT = "#6b6b6b"
C_GRID = "#dcdcdc"
C_FIT = "#c0392b"        # la retta del fit, come il raggio specchio in make_cone_diagram
C_RESID = "#eb6834"

# In pagina la figura viene rimpicciolita: le dimensioni sono scelte perche' il testo
# resti leggibile DOPO la riduzione.
plt.rcParams.update({"font.size": 13, "axes.edgecolor": C_SOFT,
                     "axes.labelcolor": C_INK, "text.color": C_INK,
                     "xtick.color": C_SOFT, "ytick.color": C_SOFT})


# ── Matematica del fit ────────────────────────────────────────────────────────
def fit_moments(C: np.ndarray, L: np.ndarray) -> dict:
    """Statistiche centrate e fit di un texel, per uno o piu' candidati.

    Rispecchia pbr_solver.py:270-283: le somme sono POOLED sui tre canali (beta e'
    condiviso, alpha_c no), beta e' clippato in [0,1] e il residuo e' diviso per il
    numero di equazioni scalari 3*n_views.  C (n_cam, 3); L (n_cam, 3) oppure
    (n_cam, n_cand, 3).
    """
    single = L.ndim == 2
    Lx = L[:, None, :] if single else L
    dC = C - C.mean(axis=0, keepdims=True)
    dL = Lx - Lx.mean(axis=0, keepdims=True)
    VLL = (dL * dL).sum(axis=(0, 2))
    VCL = (dC[:, None, :] * dL).sum(axis=(0, 2))
    VCC = float((dC * dC).sum())
    beta = np.clip(VCL / np.maximum(VLL, 1e-12), 0.0, 1.0)
    res = (VCC - 2.0 * beta * VCL + beta ** 2 * VLL) / (3.0 * C.shape[0])
    # Le due decomposizioni della stessa somma: _c per canale (somma sulle camere),
    # _j per camera (somma sui canali).  Entrambe ritornano il totale se risommate,
    # ed e' quello che _check_against_table verifica.
    out = dict(dC=dC, dL=dL[:, 0] if single else dL, VLL=VLL, VCL=VCL, VCC=VCC,
               beta=beta, res=res,
               VLL_c=(dL * dL).sum(axis=0), VCL_c=(dC[:, None, :] * dL).sum(axis=0),
               VLL_j=(dL * dL).sum(axis=2), VCL_j=(dC[:, None, :] * dL).sum(axis=2),
               VCC_j=(dC * dC).sum(axis=1))
    if single:
        for k in ("VLL", "VCL", "beta", "res"):
            out[k] = float(out[k][0])
        for k in ("VLL_c", "VCL_c"):
            out[k] = out[k][0]
        for k in ("VLL_j", "VCL_j"):
            out[k] = out[k][:, 0]
    return out


def cone_curve() -> np.ndarray:
    """(n_cam, n_cand, 3): L_jc a ogni apertura candidata.  Vedi il docstring."""
    u = 1.0 - np.cos(np.radians(APERTURES) * 0.5)
    phi = (1.0 + KAPPA[:, None] * u[K_STAR]) / (1.0 + KAPPA[:, None] * u[None, :])
    return H_ENV[None, None, :] + (L_STAR - H_ENV[None, :])[:, None, :] * phi[:, :, None]


def material_colors(L: np.ndarray) -> list[tuple]:
    """Colori osservati sintetici per i tre materiali, dallo stesso modello diretto."""
    out = []
    for name, beta_true, k_true in MATERIALS:
        rng = np.random.default_rng(NOISE_SEED)
        Cs = (ALBEDO_TERM[None, :] * (1.0 - beta_true) + beta_true * L[:, k_true, :]
              + NOISE_SIGMA * rng.standard_normal((L.shape[0], 3)))
        out.append((name, beta_true, k_true, Cs))
    return out


def _check_against_table(fit: dict, res_curve: np.ndarray) -> None:
    """La tabella della tesi e le figure devono raccontare gli stessi numeri."""
    exp = {"VLL": 10.0, "VCL": 16.0 / 3.0, "VCC": 2.853333, "beta": 8.0 / 15.0,
           "res": 9.876543e-4}
    for k, v in exp.items():
        got = float(fit[k])
        assert abs(got - v) < 5e-6, f"{k}: atteso {v} da tab:pbr-worked-example, ottenuto {got}"
    assert np.allclose(fit["VLL_c"], [14 / 3, 14 / 3, 2 / 3], atol=5e-6), fit["VLL_c"]
    assert np.allclose(fit["VCL_c"], [2.466667, 2.5, 0.366667], atol=5e-6), fit["VCL_c"]
    # Una decomposizione che non risomma al totale e' l'unico errore che le figure
    # per camera potrebbero nascondere: la barra sbagliata sembrerebbe solo un dato.
    for k in ("VLL", "VCL", "VCC"):
        for suffix in ("_c", "_j"):
            if k + suffix in fit:
                s = float(np.sum(fit[k + suffix]))
                assert abs(s - float(fit[k])) < 5e-6, f"{k}{suffix} somma {s}, non {fit[k]}"
    w = fit["VLL_j"] / fit["VLL"]
    assert abs(float((w * (fit["VCL_j"] / fit["VLL_j"])).sum()) - fit["beta"]) < 5e-6, \
        "beta non e' la media di beta_j pesata su V_LL_j"
    k = int(np.argmin(res_curve))
    assert k == K_STAR, (f"il minimo del residuo cade a {APERTURES[k]:.0f} gradi e non "
                         f"sul candidato della tabella ({APERTURES[K_STAR]:.0f}): "
                         f"H_ENV/KAPPA vanno ritarati")


# ── Utility di disegno ────────────────────────────────────────────────────────
def _tonemap(rgb: np.ndarray, scale: float) -> np.ndarray:
    """HDR -> sRGB con esposizione condivisa: le swatch di C e L sono confrontabili."""
    return np.clip(rgb / scale, 0.0, 1.0) ** (1.0 / 2.2)


def _bars(ax, values: np.ndarray, ylim: tuple, means: np.ndarray | None,
          fmt: str = "{:.3f}") -> None:
    """Tre barre R/G/B con etichetta diretta, griglia sullo sfondo."""
    x = np.arange(3)
    ax.axhline(0.0, color=C_SOFT, lw=1.0, zorder=2)
    ax.bar(x, values, width=0.62, color=CH_COLORS, zorder=3)
    if means is not None:
        for i, m in enumerate(means):
            ax.plot([i - 0.42, i + 0.42], [m, m], color=C_INK, lw=1.4, ls=(0, (4, 2)),
                    zorder=4)
    span = ylim[1] - ylim[0]
    for i, v in enumerate(values):
        if means is None:
            # Barre centrate: sopra la barra, che e' dove il numero si cerca
            off = 0.028 * span * (1 if v >= 0 else -1)
            ax.text(i, v + off, fmt.format(v), ha="center", color=C_INK, fontsize=11,
                    va="bottom" if v >= 0 else "top", zorder=5)
        else:
            # Barre grezze: dentro la barra.  Sopra finirebbe a cavallo del trattino
            # della media quando la barra e' piu' bassa della media, e il numero
            # sembrerebbe l'etichetta della media invece che della barra.
            ax.text(i, v - 0.035 * span, fmt.format(v), ha="center", color="white",
                    fontsize=11, va="top", zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(CH_NAMES)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(*ylim)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=C_GRID, lw=0.7)
    ax.tick_params(length=0)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)


def _swatch(ax, rgb: np.ndarray, scale: float, label: str) -> None:
    ax.imshow(_tonemap(rgb, scale)[None, None, :], aspect="auto",
              extent=(0, 1, 0, 1), interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(C_SOFT); s.set_linewidth(0.8)
    ax.set_title(label, fontsize=13, color=C_INK, pad=6)


def _panel_grid(fig, specs, values, means, ylim, scale, titles, fmt="{:.3f}"):
    """Una riga di tre pannelli (una camera ciascuno), con swatch se scale non e' None."""
    axes = []
    for j in range(3):
        if scale is not None:
            sub = specs[j].subgridspec(2, 1, height_ratios=(0.62, 5.0), hspace=0.10)
            _swatch(fig.add_subplot(sub[0]), values[j], scale, titles[j])
            ax = fig.add_subplot(sub[1])
        else:
            ax = fig.add_subplot(specs[j])
            ax.set_title(titles[j], fontsize=13, color=C_INK, pad=6)
        _bars(ax, values[j], ylim, means, fmt)
        if j:
            ax.set_yticklabels([])
        axes.append(ax)
    return axes


def _finish(fig, out: Path) -> None:
    fig.savefig(out, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  + {out.name}")


# ── Figure 1-4: barre per camera, grezze e centrate ───────────────────────────
def fig_bars(values, means, ylim, scale, out: Path, suptitle: str, note: str,
             fmt: str = "{:.3f}") -> None:
    fig = plt.figure(figsize=(9.6, 4.6 if scale is not None else 3.9))
    gs = fig.add_gridspec(1, 3, wspace=0.12)
    titles = [f"Camera {j + 1}" for j in range(3)]
    axes = _panel_grid(fig, [gs[0, j] for j in range(3)], values, means, ylim, scale,
                       titles, fmt)
    axes[0].set_ylabel("radiance", color=C_INK)
    fig.suptitle(suptitle, fontsize=15, color=C_INK, y=1.02)
    fig.text(0.5, -0.035, note, ha="center", fontsize=11.5, color=C_SOFT)
    _finish(fig, out)


def fig_combined(fit: dict, L_star_mean, C_mean, ylim_raw, ylim_cen, scale,
                 out: Path) -> None:
    """I quattro pannelli in una griglia sola: e' questa la figura che va in tesi,
    quindi la spaziatura e' stretta -- a \\linewidth la versione larga si mangiava
    quasi una pagina intera."""
    fig = plt.figure(figsize=(9.6, 10.6))
    gs = fig.add_gridspec(4, 3, hspace=0.42, wspace=0.12)
    rows = ((C_OBS, C_mean, ylim_raw, scale,
             r"Observed color $C_{jc}$"),
            (fit["dC"], None, ylim_cen, None,
             r"Centered color $\Delta C_{jc} = C_{jc} - \bar{C}_c$"),
            (L_STAR, L_star_mean, ylim_raw, scale,
             r"Cone radiance $L_{jc}$ at $\Theta = 45^\circ$"),
            (fit["dL"], None, ylim_cen, None,
             r"Centered radiance $\Delta L_{jc} = L_{jc} - \bar{L}_c$"))
    for r, (vals, means, ylim, sc, title) in enumerate(rows):
        titles = [f"Camera {j + 1}" for j in range(3)] if r == 0 else ["", "", ""]
        axes = _panel_grid(fig, [gs[r, j] for j in range(3)], vals, means, ylim, sc,
                           titles)
        axes[0].set_ylabel("radiance", color=C_INK)
        # Sopra la fascia della swatch quando c'e': il titolo di riga e' lungo e alla
        # quota dell'asse delle barre finirebbe steso sulla swatch della camera 1.
        # Nella prima riga sopra la swatch ci sono anche i nomi delle camere.
        title_y = (1.46 if r == 0 else 1.32) if sc is not None else 1.04
        axes[0].text(-0.34, title_y, title, transform=axes[0].transAxes, fontsize=14,
                     color=C_INK, ha="left", va="bottom")
    _finish(fig, out)


# ── Figura 5: la regressione centrata ─────────────────────────────────────────
def _scatter_fit(ax, dL, dC, beta, res, xlim=None, drops=True, legend=False) -> None:
    lim = xlim or (np.abs(dL).max() * 1.35)
    xs = np.array([-lim, lim])
    ax.axhline(0, color=C_GRID, lw=1.0, zorder=1)
    ax.axvline(0, color=C_GRID, lw=1.0, zorder=1)
    ax.plot(xs, beta * xs, color=C_FIT, lw=2.0, zorder=3,
            label=fr"$\Delta C = \beta^\star \Delta L$")
    if drops:
        for j in range(dL.shape[0]):
            for c in range(3):
                ax.plot([dL[j, c], dL[j, c]], [dC[j, c], beta * dL[j, c]],
                        color=C_RESID, lw=1.6, zorder=4, solid_capstyle="butt")
    # Marker piu' piccoli sui canali successivi: due punti possono coincidere quasi
    # esattamente (camera 1 in R e G differiscono di 0.03) e a parita' di taglia il
    # secondo sparirebbe sotto il primo.  La taglia e' ridondante, non informativa:
    # il canale resta il colore.
    for c in range(3):
        for j in range(dL.shape[0]):
            ax.plot(dL[j, c], dC[j, c], marker=CAM_MARKERS[j], ms=(12.0, 9.0, 6.5)[c],
                    color=CH_COLORS[c], mec="white", mew=1.3, ls="none", zorder=5 + c)
    ax.set_xlim(-lim, lim)
    ax.set_axisbelow(True)
    ax.grid(color=C_GRID, lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if legend:
        handles = [Line2D([], [], color=CH_COLORS[c], marker="o", ls="none", ms=9,
                          mec="white", mew=1.2, label=f"channel {CH_NAMES[c]}")
                   for c in range(3)]
        handles += [Line2D([], [], color=C_SOFT, marker=CAM_MARKERS[j], ls="none",
                           ms=9, mec="white", mew=1.2, label=f"camera {j + 1}")
                    for j in range(3)]
        handles += [Line2D([], [], color=C_FIT, lw=2.0, label="fitted slope")]
        if drops:
            handles += [Line2D([], [], color=C_RESID, lw=1.8, label="residual")]
        ax.legend(handles=handles, loc="upper left", fontsize=11, frameon=False,
                  ncol=2, handletextpad=0.5, columnspacing=1.4, labelspacing=0.35)


def fig_scatter(fit: dict, out: Path) -> None:
    """La regressione e, accanto, quello che le resta: senza il secondo pannello i
    residui di questo texel (terza cifra decimale) sono sotto la larghezza della retta
    e la figura direbbe solo 'i punti sono allineati'."""
    fig = plt.figure(figsize=(12.6, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.55, 1.0), wspace=0.26)

    ax = fig.add_subplot(gs[0])
    _scatter_fit(ax, fit["dL"], fit["dC"], fit["beta"], fit["res"], drops=False,
                 legend=True)
    ax.set_ylim(-1.22, 1.22)
    ax.set_xlabel(r"centered cone radiance $\Delta L_{jc}$")
    ax.set_ylabel(r"centered color $\Delta C_{jc}$")
    ax.set_title("One slope for nine equations", fontsize=14, color=C_INK, pad=10)
    ax.text(0.985, 0.05,
            fr"$\beta^\star = V_{{CL}}/V_{{LL}} = {fit['VCL']:.3f}/{fit['VLL']:.3f}"
            fr" = {fit['beta']:.3f}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=12.5, color=C_INK)

    axr = fig.add_subplot(gs[1])
    resid = fit["dC"] - fit["beta"] * fit["dL"]
    # Gruppi staccati: le etichette dei due valori a cavallo fra una camera e la
    # successiva si toccherebbero.
    x = np.array([4.4 * j + 1.15 * c for j in range(3) for c in range(3)])
    axr.axhline(0.0, color=C_SOFT, lw=1.0, zorder=2)
    axr.bar(x, resid.ravel(), width=0.66, color=list(CH_COLORS) * 3, zorder=3)
    lim = np.abs(resid).max() * 1.75
    for i, v in enumerate(resid.ravel()):
        axr.text(x[i], v + 0.03 * lim * (1 if v >= 0 else -1), f"{v:+.3f}",
                 ha="center", va="bottom" if v >= 0 else "top", fontsize=10,
                 color=C_INK, zorder=5)
    axr.set_xticks(x)
    axr.set_xticklabels(list(CH_NAMES) * 3)
    for j in range(3):
        axr.text(4.4 * j + 1.15, -lim * 0.93, f"camera {j + 1}", ha="center",
                 va="bottom", fontsize=11.5, color=C_SOFT)
        if j:
            axr.axvline(4.4 * j - 1.05, color=C_GRID, lw=1.0)
    axr.set_ylim(-lim, lim)
    axr.set_xlim(-1.0, 11.4)
    axr.set_ylabel(r"$\Delta C_{jc} - \beta^\star \Delta L_{jc}$")
    axr.set_title("What the slope leaves behind", fontsize=14, color=C_INK, pad=10)
    axr.text(0.5, 0.985,
             fr"$\sum \mathrm{{residual}}^2 = {fit['res'] * 9:.5f}$   over 9 equations"
             "\n"
             fr"$\mathrm{{res}} = {fit['res'] * 9:.5f}/9 = {fit['res']:.2e}$",
             transform=axr.transAxes, ha="center", va="top", fontsize=12,
             color=C_INK, linespacing=1.6)
    axr.set_axisbelow(True)
    axr.grid(axis="y", color=C_GRID, lw=0.7)
    axr.tick_params(length=0)
    for s in ("top", "right", "bottom"):
        axr.spines[s].set_visible(False)

    fig.suptitle(r"The centered regression of a single texel, at $\Theta = 45^\circ$",
                 fontsize=15, color=C_INK, y=1.03)
    _finish(fig, out)


# ── Figura 6: i momenti, camera per camera ────────────────────────────────────
def fig_moments_per_camera(fit: dict, out: Path) -> None:
    """V_LL, V_CL, V_CC per camera piu' il gruppo dei totali.

    Il gruppo 'total' non e' decorazione: sono i tre numeri che entrano davvero in
    beta, e averli come quarta colonna evita di dover annotare a parte cosa somma a
    cosa.  Le etichette su OGNI barra sono obbligatorie qui: su scala lineare la
    camera 2 vale 0.33 contro i 5.67 della camera 1 e senza numero sarebbe un
    filetto illeggibile -- che poi e' esattamente la cosa da far vedere.
    """
    vals = np.stack([fit["VLL_j"], fit["VCL_j"], fit["VCC_j"]], axis=1)   # (3 cam, 3)
    tot = np.array([fit["VLL"], fit["VCL"], fit["VCC"]])
    data = np.vstack([vals, tot])                                          # (4, 3)
    names = (r"$V_{LL}$", r"$V_{CL}$", r"$V_{CC}$")

    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    width = 0.26
    for m in range(3):
        x = np.arange(4) + (m - 1) * width
        ax.bar(x, data[:, m], width=width * 0.92, color=MOMENT_COLORS[m], zorder=3,
               label=names[m])
        for i, v in enumerate(data[:, m]):
            ax.text(x[i], v + 0.13, f"{v:.3f}", ha="center", va="bottom", fontsize=10.5,
                    color=C_INK, rotation=90, zorder=5)
    ax.axvline(2.5, color=C_SOFT, lw=1.0)
    ax.axvspan(2.5, 3.6, color="#f2f2f2", zorder=0)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(["camera 1", "camera 2", "camera 3", "total\n(what $\\beta$ uses)"])
    ax.set_xlim(-0.55, 3.6)
    ax.set_ylim(0, data.max() * 1.30)
    ax.set_ylabel("sum over the three channels")
    ax.set_title(r"Where $V_{LL}$, $V_{CL}$ and $V_{CC}$ come from, camera by camera",
                 fontsize=14, color=C_INK, pad=10)
    ax.legend(loc="upper left", fontsize=12, frameon=False, ncol=3)
    ax.annotate("camera 2 sits at the mean of the other two:\n"
                f"{fit['VLL_j'][1] / fit['VLL'] * 100:.0f}% of $V_{{LL}}$, so it barely "
                "constrains the slope",
                xy=(1 - width, fit["VLL_j"][1] + 0.9), xytext=(0.255, 0.40),
                textcoords="axes fraction", fontsize=11, color=C_SOFT,
                ha="left", va="bottom", linespacing=1.5,
                arrowprops=dict(arrowstyle="-", color=C_SOFT, lw=1.0, shrinkB=2))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=C_GRID, lw=0.7)
    ax.tick_params(length=0)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    _finish(fig, out)


# ── Figura 7: da dove esce beta ───────────────────────────────────────────────
def fig_beta(fit: dict, out: Path) -> None:
    trials = (0.20, fit["beta"], 0.90)
    # Due grigi diversi e non uno solo: i segmenti dei residui non hanno tratteggio,
    # quindi a parita' di colore non si saprebbe a quale pendenza appartengono.
    t_col = ("#9a9a9a", C_FIT, "#4d4d4d")
    t_ls = ((0, (5, 2)), "-", (0, (1, 2)))
    fig = plt.figure(figsize=(14.6, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=(1.15, 1.0, 1.0), wspace=0.38)

    # (a) tre pendenze di prova sugli stessi nove punti
    ax = fig.add_subplot(gs[0])
    lim = np.abs(fit["dL"]).max() * 1.35
    xs = np.array([-lim, lim])
    ax.axhline(0, color=C_GRID, lw=1.0)
    ax.axvline(0, color=C_GRID, lw=1.0)
    for b, col, ls in zip(trials, t_col, t_ls):
        sse = fit["VCC"] - 2 * b * fit["VCL"] + b ** 2 * fit["VLL"]
        ax.plot(xs, b * xs, color=col, lw=2.0, ls=ls, zorder=3,
                label=fr"$b = {b:.2f}$,  SSE $= {sse:.3f}$")
        for j in range(3):
            for c in range(3):
                ax.plot([fit["dL"][j, c]] * 2, [fit["dC"][j, c], b * fit["dL"][j, c]],
                        color=col, lw=1.2, alpha=0.55, zorder=2)
    for c in range(3):
        for j in range(3):
            ax.plot(fit["dL"][j, c], fit["dC"][j, c], marker=CAM_MARKERS[j],
                    ms=(11.0, 8.5, 6.0)[c], color=CH_COLORS[c], mec="white", mew=1.3,
                    ls="none", zorder=5 + c)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-1.35, 1.35)
    ax.set_xlabel(r"$\Delta L_{jc}$")
    ax.set_ylabel(r"$\Delta C_{jc}$")
    ax.set_title("Try a slope, measure what is left", fontsize=13.5, color=C_INK, pad=8)
    ax.legend(loc="upper left", fontsize=10.5, frameon=False, labelspacing=0.3)

    # (b) la parabola: e' qui che beta viene fuori
    axp = fig.add_subplot(gs[1])
    b = np.linspace(-0.15, 1.25, 400)
    sse = fit["VCC"] - 2 * b * fit["VCL"] + b ** 2 * fit["VLL"]
    axp.axvspan(0.0, 1.0, color="#f2f2f2", zorder=0)
    axp.plot(b, sse, color=C_INK, lw=2.0, zorder=4)
    for t, col in zip(trials, t_col):
        s = fit["VCC"] - 2 * t * fit["VCL"] + t ** 2 * fit["VLL"]
        axp.plot(t, s, marker="o", ms=9, color=col, mec="white", mew=1.4, zorder=6)
    axp.plot([fit["beta"]] * 2, [0, fit["res"] * 9], color=C_FIT, lw=1.2,
             ls=(0, (3, 3)), zorder=3)
    axp.annotate(fr"$\beta^\star = \dfrac{{V_{{CL}}}}{{V_{{LL}}}} = {fit['beta']:.3f}$",
                 xy=(fit["beta"], fit["res"] * 9), xytext=(0.30, 0.52),
                 textcoords="axes fraction", fontsize=13, color=C_FIT, ha="left",
                 va="bottom", arrowprops=dict(arrowstyle="-", color=C_FIT, lw=1.0,
                                              shrinkB=4))
    axp.text(0.5, 0.985,
             r"SSE$(b) = V_{CC} - 2\,b\,V_{CL} + b^2 V_{LL}$",
             transform=axp.transAxes, ha="center", va="top", fontsize=12.5, color=C_INK)
    axp.text(0.98, 0.06, "shaded: $b$ is clamped\nto $[0, 1]$",
             transform=axp.transAxes, ha="right", va="bottom", fontsize=10.5,
             color=C_SOFT, linespacing=1.4)
    axp.set_ylim(0, sse.max() * 1.12)
    axp.set_xlim(-0.15, 1.25)
    axp.set_xlabel(r"trial slope $b$")
    axp.set_ylabel(r"SSE$(b)$   ($9 \times \mathrm{res}$)")
    axp.set_title(r"$\beta^\star$ is where the parabola bottoms out", fontsize=13.5,
                  color=C_INK, pad=8)

    # (c) beta come media pesata delle pendenze delle singole camere
    axw = fig.add_subplot(gs[2])
    w = fit["VLL_j"] / fit["VLL"]
    beta_j = fit["VCL_j"] / fit["VLL_j"]
    # Punti e non barre: le tre pendenze differiscono di 0.04 su 0.53, quindi vanno
    # guardate da vicino, e un asse zoomato con le barre partirebbe da un finto zero.
    # L'area del punto e' il peso, ed e' cosi' che si vede beta* cadere accanto alla
    # camera 1 (57%) e lontano dalla camera 2 (3%).
    axw.axvline(fit["beta"], color=C_FIT, lw=2.0, zorder=4)
    for j in range(3):
        axw.plot([min(beta_j[j], fit["beta"]), max(beta_j[j], fit["beta"])],
                 [2 - j] * 2, color=C_GRID, lw=1.4, zorder=2)
        axw.scatter(beta_j[j], 2 - j, s=max(1500.0 * w[j], 55.0), color=C_SOFT,
                    edgecolor="white", linewidth=1.5, zorder=5)
    axw.text(fit["beta"], -0.66, fr"$\beta^\star = {fit['beta']:.3f}$", ha="center",
             va="bottom", fontsize=12.5, color=C_FIT)
    # Valori sulle etichette dell'asse e non accanto ai punti: l'asse e' stretto e
    # qualunque testo dentro il pannello finirebbe a cavallo della riga di beta*.
    axw.set_yticks([2, 1, 0])
    axw.set_yticklabels([f"camera {j + 1}\n$\\beta_{{{j + 1}}}$ = {beta_j[j]:.3f}\n"
                         f"$w_{{{j + 1}}}$ = {w[j] * 100:.0f}%"
                         for j in range(3)], fontsize=11.5)
    axw.set_xlim(min(beta_j.min(), fit["beta"]) - 0.025, beta_j.max() + 0.025)
    axw.set_ylim(-0.80, 2.85)
    axw.set_xlabel(r"per-camera slope $\beta_j = V_{CL}^{(j)} / V_{LL}^{(j)}$"
                   "\n" r"(marker area = weight)")
    axw.set_title(r"$\beta^\star = \sum_j w_j\, \beta_j$,   $w_j = V_{LL}^{(j)}/V_{LL}$",
                  fontsize=13.5, color=C_INK, pad=8)

    for a in (ax, axp, axw):
        a.set_axisbelow(True)
        a.grid(color=C_GRID, lw=0.7)
        a.tick_params(length=0)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    axw.grid(axis="x", color=C_GRID, lw=0.7)
    axw.yaxis.grid(False)
    _finish(fig, out)


# ── Figura 8: perche' quell'apertura ──────────────────────────────────────────
def fig_aperture_selection(L: np.ndarray, curve: dict, out: Path) -> None:
    show = (2, K_STAR, 10)          # 10 gradi (stretto), 45 (vincitore), 120 (largo)
    tags = ("A", "B", "C")
    fig = plt.figure(figsize=(10.4, 10.0))
    # Due gridspec annidati e non uno solo: fra la riga degli scatter e il residuo
    # serve aria (li' sotto ci sono le etichette di Delta L), fra residuo e beta no,
    # perche' condividono l'asse x e devono leggersi come un blocco.
    outer = fig.add_gridspec(2, 1, height_ratios=(1.30, 1.55), hspace=0.30)
    gs = outer[0].subgridspec(1, 3, wspace=0.16)
    gs_bot = outer[1].subgridspec(2, 1, height_ratios=(1.0, 0.42), hspace=0.12)

    dC = curve["dC"]
    ylim = np.abs(dC).max() * 1.42
    for p, k in enumerate(show):
        ax = fig.add_subplot(gs[0, p])
        dL = curve["dL"][:, k]
        _scatter_fit(ax, dL, dC, curve["beta"][k], curve["res"][k],
                     xlim=np.abs(dL).max() * 1.45)
        ax.set_ylim(-ylim, ylim)
        if p:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(r"$\Delta C_{jc}$")
        ax.set_xlabel(r"$\Delta L_{jc}$")
        kind = ("too narrow", "best fit", "too wide")[p]
        ax.set_title(f"{tags[p]}   $\\Theta = {APERTURES[k]:.0f}^\\circ$ ({kind})",
                     fontsize=13.5, color=C_INK, pad=8)
        ax.text(0.035, 0.965, fr"$\beta^\star = {curve['beta'][k]:.3f}$" "\n"
                              fr"$\mathrm{{res}} = {curve['res'][k]:.2e}$",
                transform=ax.transAxes, ha="left", va="top", fontsize=11.5,
                color=C_INK, linespacing=1.5)

    # residuo: e' questo che sceglie
    axr = fig.add_subplot(gs_bot[0])
    axr.plot(APERTURES, curve["res"], color=C_FIT, lw=2.0, marker="o", ms=6.5,
             mec="white", mew=1.2, zorder=4)
    kmin = int(np.argmin(curve["res"]))
    axr.plot(APERTURES[kmin], curve["res"][kmin], marker="o", ms=13, color=C_FIT,
             mec="white", mew=2.0, zorder=6)
    axr.annotate(f"winner: {APERTURES[kmin]:.0f}$^\\circ$\n"
                 f"metallic = $\\beta^\\star$ = {curve['beta'][kmin]:.3f}\n"
                 f"roughness = {APERTURES[kmin]:.0f}/180 = "
                 f"{APERTURES[kmin] / 180.0:.3f}",
                 xy=(APERTURES[kmin], curve["res"][kmin]), xycoords="data",
                 xytext=(0.63, 0.10), textcoords="axes fraction", fontsize=12,
                 color=C_INK, ha="left", va="bottom", linespacing=1.6,
                 arrowprops=dict(arrowstyle="-", color=C_SOFT, lw=1.0,
                                 shrinkA=6, shrinkB=10))
    blend = axr.get_xaxis_transform()
    for p, k in enumerate(show):
        axr.axvline(APERTURES[k], color=C_SOFT, lw=1.0, ls=(0, (3, 3)), zorder=1)
        axr.text(APERTURES[k], 0.97, tags[p], ha="center", va="top", fontsize=13,
                 color=C_SOFT, transform=blend)
    axr.set_yscale("log")
    axr.set_ylim(curve["res"].min() * 0.55, curve["res"].max() * 2.4)
    axr.set_ylabel(r"residual  $\mathrm{res}(\Theta)$")
    axr.set_title(r"The residual selects the aperture", fontsize=14, color=C_INK, pad=8)

    # beta: cresce e basta, non ha un minimo -- non puo' scegliere
    axb = fig.add_subplot(gs_bot[1], sharex=axr)
    axb.plot(APERTURES, curve["beta"], color=C_INK, lw=2.0, marker="o", ms=5.5,
             mec="white", mew=1.0, zorder=4)
    axb.axhline(1.0, color=C_SOFT, lw=1.0, ls=(0, (3, 3)))
    axb.set_ylim(0.0, 1.18)
    axb.set_ylabel(r"$\beta^\star(\Theta)$")
    axb.set_xlabel(r"candidate total aperture $\Theta$  [degrees]")
    axb.text(0.5, 0.08, "monotone in $\\Theta$, and clamped at 1: no minimum to pick",
             transform=axb.transAxes, ha="center", fontsize=11.5, color=C_SOFT)
    for ax in (axr, axb):
        ax.set_axisbelow(True)
        ax.grid(color=C_GRID, lw=0.7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axr.tick_params(labelbottom=False)
    axr.set_xlim(-6, 186)
    axb.set_xticks(APERTURES[::2])
    _finish(fig, out)


# ── Figura 7: metallo, lucido, diffuso ────────────────────────────────────────
def fig_residual_materials(L: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    lines, rmax = [], 1.0
    blend = ax.get_xaxis_transform()
    for i, (name, beta_true, k_true, Cs) in enumerate(material_colors(L)):
        f = fit_moments(Cs, L)
        r = f["res"] / f["res"].min()
        k = int(np.argmin(f["res"]))
        rmax = max(rmax, r.max())
        ax.plot(APERTURES, r, color=MAT_COLORS[i], lw=2.0, marker="o", ms=5.5,
                mec="white", mew=1.0, zorder=4)
        ax.plot(APERTURES[k], 1.0, marker="o", ms=13, color=MAT_COLORS[i],
                mec="white", mew=2.0, zorder=6)
        # Etichetta in posto sulla coda destra: la legenda in alto costringeva le tre
        # scritte del minimo a impilarsi sull'asse x.  Serve anche come rilievo per
        # l'aqua, che sul fondo chiaro sta sotto 3:1.
        ax.annotate(f"{name}\npicks {APERTURES[k]:.0f}$^\\circ$, "
                    f"$\\beta^\\star$ = {f['beta'][k]:.2f}",
                    xy=(APERTURES[-1], r[-1]), xytext=(-4, 12),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=12, color=MAT_COLORS[i], linespacing=1.4, zorder=7)
        # tacca colorata sul candidato scelto: si legge senza inseguire la curva
        ax.plot([APERTURES[k]] * 2, [0.0, 0.045], color=MAT_COLORS[i], lw=3.0,
                transform=blend, clip_on=False, zorder=7)
        lines.append((name, beta_true, k_true, k, f))
    ax.set_yscale("log")
    ax.set_xlim(-6, 186)
    ax.set_xticks(APERTURES[::2])
    ax.set_ylim(0.5, rmax * 7.0)
    ax.set_xlabel(r"candidate total aperture $\Theta$  [degrees]")
    ax.set_ylabel(r"residual, relative to its own minimum")
    ax.set_title("How much the data constrains the lobe, by material",
                 fontsize=14, color=C_INK, pad=10)
    ax.set_axisbelow(True)
    ax.grid(color=C_GRID, lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _finish(fig, out)
    return lines


# ── Riepilogo numerico ────────────────────────────────────────────────────────
def print_table(fit: dict) -> None:
    """Righe di tab:pbr-worked-example rigenerate dagli input, per il confronto."""
    print("\n% tab:pbr-worked-example, rigenerata (non trascritta)")
    for j in range(3):
        print(f"        Camera {j + 1} & " + " & ".join(
            f"${v:.3f}$" for v in np.r_[C_OBS[j], L_STAR[j]]) + r" \\")
    print("        Mean over cameras & " + " & ".join(
        f"${v:.3f}$" for v in np.r_[C_OBS.mean(0), L_STAR.mean(0)]) + r" \\")
    for j in range(3):
        print(f"        Camera {j + 1}, centered & " + " & ".join(
            f"${v:.3f}$" for v in np.r_[fit["dC"][j], fit["dL"][j]]) + r" \\")
    print("        Per-channel $V_{LL}$ & & & & " + " & ".join(
        f"${v:.3f}$" for v in fit["VLL_c"]) + r" \\")
    print("        Per-channel $V_{CL}$ & " + " & ".join(
        f"${v:.3f}$" for v in fit["VCL_c"]) + r" & & & \\")
    print(f"        $V_{{LL}} = {fit['VLL']:.3f}$ \\quad $V_{{CL}} = {fit['VCL']:.3f}$ "
          f"\\quad $\\beta^\\star = {fit['beta']:.3f}$ \\quad "
          f"res $= {fit['res']:.2e}$")
    print(f"% (V_CC = {fit['VCC']:.4f}, somma dei quadrati dei residui = "
          f"{fit['res'] * 9:.5f} su 9 equazioni)")


def print_curve(curve: dict, L: np.ndarray) -> None:
    print("\n% res(Theta) sul texel della tabella")
    print("  Theta    VLL     VCL    beta      res    res/min")
    rmin = curve["res"].min()
    for a in range(len(APERTURES)):
        star = " <-- tabella" if a == K_STAR else ""
        print(f"  {APERTURES[a]:5.0f}  {curve['VLL'][a]:6.2f} {curve['VCL'][a]:6.2f} "
              f"{curve['beta'][a]:6.3f}  {curve['res'][a]:.3e} {curve['res'][a]/rmin:8.1f}"
              f"{star}")
    print(f"  L min {L.min():.3f}  max {L.max():.3f}   (livello specchio: "
          f"{np.round(L[:, 0, :], 2).tolist()})")


def print_per_camera(fit: dict) -> None:
    """Decomposizione per camera: i numeri delle due figure nuove."""
    w = fit["VLL_j"] / fit["VLL"]
    beta_j = fit["VCL_j"] / fit["VLL_j"]
    print("\n% momenti per camera (somma sui tre canali)")
    print("           V_LL    V_CL    V_CC   beta_j   peso")
    for j in range(3):
        print(f"  camera {j + 1} {fit['VLL_j'][j]:6.3f} {fit['VCL_j'][j]:7.3f} "
              f"{fit['VCC_j'][j]:7.3f}  {beta_j[j]:6.3f} {w[j] * 100:6.1f}%")
    print(f"  totale   {fit['VLL']:6.3f} {fit['VCL']:7.3f} {fit['VCC']:7.3f}  "
          f"{fit['beta']:6.3f} {w.sum() * 100:6.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--figures", default=None,
                    help="sottoinsieme da generare, nomi separati da virgola senza il "
                         "prefisso 'fit_' (default: tutte).  Serve per la copia in "
                         "Doc/images/pbr-fit, che tiene solo le figure pubblicate")
    ap.add_argument("--print-table", action="store_true",
                    help="rigenera le righe LaTeX della tabella e il dettaglio numerico")
    args = ap.parse_args()
    out = Path(args.out)

    if args.figures is None:
        want = lambda name: True                                  # noqa: E731
    else:
        asked = {s.strip() for s in args.figures.split(",") if s.strip()}
        unknown = asked - set(FIGURES)
        # Un nome sbagliato deve fermare lo script: altrimenti la copia in tesi
        # finisce a mani vuote e nessuno se ne accorge fino alla compilazione.
        if unknown:
            raise SystemExit(f"--figures: nomi sconosciuti {sorted(unknown)}; "
                             f"disponibili {sorted(FIGURES)}")
        want = asked.__contains__
    out.mkdir(parents=True, exist_ok=True)   # dopo il controllo: un refuso non crea cartelle

    L = cone_curve()
    fit = fit_moments(C_OBS, L_STAR)
    curve = fit_moments(C_OBS, L)
    _check_against_table(fit, curve["res"])

    # Un'unica esposizione per tutte le swatch: C e L devono restare confrontabili
    scale = float(max(C_OBS.max(), L_STAR.max()))
    ylim_raw = (0.0, scale * 1.16)
    ylim_cen = float(max(np.abs(fit["dC"]).max(), np.abs(fit["dL"]).max())) * 1.30
    ylim_cen = (-ylim_cen, ylim_cen)

    print(f"[figure] {out}")
    if want("color_raw"):
        fig_bars(C_OBS, C_OBS.mean(0), ylim_raw, scale, out / "fit_color_raw.png",
                 r"Observed color $C_{jc}$, one texel seen by three cameras",
                 "dashed: the per-channel mean over the cameras, which centering subtracts"
                 f"  |  swatches tonemapped at a shared exposure (divide by {scale:.1f},"
                 " then gamma 1/2.2)")
    if want("color_centered"):
        fig_bars(fit["dC"], None, ylim_cen, None, out / "fit_color_centered.png",
                 r"After centering: $\Delta C_{jc} = C_{jc} - \bar{C}_c$",
                 "the diffuse term is gone: it contributed equally to all three cameras")
    if want("cone_raw"):
        fig_bars(L_STAR, L_STAR.mean(0), ylim_raw, scale, out / "fit_cone_raw.png",
                 r"Cone radiance $L_{jc}$ at the candidate $\Theta = 45^\circ$",
                 "same texel, same three cameras, same exposure as the observed color")
    if want("cone_centered"):
        fig_bars(fit["dL"], None, ylim_cen, None, out / "fit_cone_centered.png",
                 r"After centering: $\Delta L_{jc} = L_{jc} - \bar{L}_c$",
                 "blue varies seven times less than red and green: why the channels are "
                 "pooled")
    if want("worked_combined"):
        fig_combined(fit, L_STAR.mean(0), C_OBS.mean(0), ylim_raw, ylim_cen, scale,
                     out / "fit_worked_combined.png")
    if want("scatter"):
        fig_scatter(fit, out / "fit_scatter.png")
    if want("moments_per_camera"):
        fig_moments_per_camera(fit, out / "fit_moments_per_camera.png")
    if want("beta"):
        fig_beta(fit, out / "fit_beta.png")
    if want("aperture_selection"):
        fig_aperture_selection(L, curve, out / "fit_aperture_selection.png")
    mats = (fig_residual_materials(L, out / "fit_residual_materials.png")
            if want("residual_materials") else None)

    print_per_camera(fit)
    print_curve(curve, L)
    if mats is not None:
        print("\n% materiali (sigma assoluto comune "
              f"{NOISE_SIGMA}, seme {NOISE_SEED})")
        for name, beta_true, k_true, k, f in mats:
            print(f"  {name:8s} beta vero {beta_true:.2f} a {APERTURES[k_true]:5.0f} "
                  f"gradi  ->  sceglie {APERTURES[k]:5.0f}, beta* {f['beta'][k]:.3f}, "
                  f"res {f['res'].min():.2e}, curva max/min "
                  f"{f['res'].max() / f['res'].min():8.1f}")
    if args.print_table:
        print_table(fit)
    print(f"\n[ok] esposizione swatch = {scale:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
