#!/usr/bin/env python
"""make_cone_diagram.py -- Figure ed esempio numerico per l'equazione dei coni.

    python make_cone_diagram.py --out ../Doc/images/cone

Scrive tre PNG e stampa le righe della tabella LaTeX dell'esempio:

  cone_geometry.png  texel, normale, camera, raggio riflesso e i coni candidati
  cone_rings.png     il set condiviso di Fibonacci, colorato per anello, in 3D e
                     nella vista lungo R dove il binning si conta a occhio
  cone_weights.png   che cosa vale un raggio: la stessa regione in proiezione
                     equivalente, a sinistra il set condiviso vero con le sue celle di
                     Voronoi, a destra un budget piatto di N raggi per anello

I pesi e la chiusura dei coni NON sono riscritti qui: arrivano da images_generator,
gli stessi usati dal bake.  Se un giorno divergono, diverge anche la figura della tesi
e ce ne si accorge.

Il caso e' didattico ma non finto: direzioni di Fibonacci come quelle del kernel
condiviso (cos(theta) equispaziato, azimut sull'angolo aureo) e radianza analitica
L(d) = 0.5 + 0.5*d_z, la stessa envmap 'gradient' con cui test_hemivis_shared.py
verifica il bake.  Su quella envmap la media sul cono di semiapertura b attorno a R,
se il cono sta tutto sopra l'orizzonte, vale in forma chiusa

    L = 0.5 + 0.5 * R_z * (1 + cos b) / 2

quindi la tabella puo' mostrare la stima accanto al valore esatto invece di chiedere
al lettore di fidarsi.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Wedge
from scipy.spatial import SphericalVoronoi

import _paths  # noqa: F401

from images_generator import (_cones_from_rings_np, ring_weights_mean,
                              spec_cone_ring_samples,
                              spec_cone_shared_ring_samples)

# Caso didattico: pochi raggi, contabili a occhio, e una griglia di aperture ridotta
# a quattro anelli invece dei tredici della configurazione operativa.
APERTURES = [0.0, 30.0, 60.0, 90.0, 140.0]   # aperture TOTALI, in gradi
S = 96                                        # direzioni condivise sull'emisfero
THETA_V = 30.0                                # inclinazione della camera dalla normale
PHI_V = 200.0                                 # suo azimut, scelto solo per l'inquadratura

# Budget piatto della figura sui pesi: un numero arbitrario di raggi per anello, uguale
# per tutti.  E' l'esagerazione didattica dell'allocazione mirata, dove il floor sui primi
# anelli lascia comunque una ventina di volte fra il raggio piu' denso e il piu' rado.
AIMED_PER_RING = 10

# Rampa sequenziale a tinta unica: gli anelli sono ordinati, quindi il colore deve
# crescere con l'indice.  Niente arcobaleno.  Grigio per i raggi fuori da ogni cono.
RING_COLORS = plt.get_cmap("Blues")(np.linspace(0.45, 0.92, len(APERTURES) - 1))
C_OUT = "#b8b8b8"
C_INK = "#222222"
C_MIRROR = "#c0392b"

# In pagina la figura viene rimpicciolita di circa un terzo: le dimensioni qui sono
# scelte perche' il testo resti leggibile DOPO quella riduzione, non sullo schermo.
plt.rcParams.update({"font.size": 13})


def onb(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Base ortonormale attorno ad `a`."""
    t = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(t, a)
    u /= np.linalg.norm(u)
    return u, np.cross(a, u)


def fibonacci_hemisphere(s: int) -> np.ndarray:
    """(S, 3) direzioni sopra z, uniformi in angolo solido.

    Stessa costruzione del kernel condiviso: cos(theta) equispaziato e azimut
    sull'angolo aureo.  La rotazione per texel del kernel qui non serve, c'e' un
    texel solo.
    """
    i = np.arange(s)
    cos_t = 1.0 - (i + 0.5) / s
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t ** 2))
    phi = i * np.pi * (3.0 - np.sqrt(5.0))
    return np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=-1)


def radiance(d: np.ndarray) -> np.ndarray:
    """Envmap 'gradient' dei test: lineare nella componente verticale."""
    return 0.5 + 0.5 * d[..., 2]


def circle_on_sphere(axis: np.ndarray, half_deg: float, n: int = 240) -> np.ndarray:
    """Direzioni a `half_deg` gradi da `axis`: il bordo di un cono."""
    t, b = onb(axis)
    a = np.radians(half_deg)
    ang = np.linspace(0, 2 * np.pi, n)
    return (np.cos(a) * axis[None, :]
            + np.sin(a) * (np.cos(ang)[:, None] * t + np.sin(ang)[:, None] * b))


def _hemisphere_voronoi(dirs: np.ndarray) -> tuple[list, np.ndarray]:
    """Celle di Voronoi sferiche del set condiviso, chiuse esattamente sull'orizzonte.

    La toppa di cielo che un raggio rappresenta e' la regione delle direzioni piu' vicine
    a lui che a ogni altro raggio: e' quella, non un settore disegnato a tavolino, che
    dice quanto vale un campione.

    Il set vive solo sopra n e un Voronoi su mezza sfera lascerebbe le celle di bordo
    aperte fin sotto la superficie.  Aggiungere i punti specchiati sotto l'orizzonte le
    chiude su z = 0 senza approssimare niente: per una direzione u con u_z > 0 e una
    coppia q, q' = (q_x, q_y, -q_z) vale sempre u.q > u.q', quindi nessuna cella superiore
    attraversa l'equatore.  La somma delle prime S aree torna 2*pi esatto.

    Le aree NON sono tutte uguali: meta' sta entro lo 0.6% da 2*pi/S, ma sul bordo del
    reticolo (i punti a un decimo di grado dall'equatore) la cella e' tagliata dalla
    superficie e vale circa meta'.  E' geometria del reticolo, non del bake: l'equazione
    dei coni usa il peso nominale 2*pi/S e con W_i costante quel fattore si semplifica,
    quindi la stima resta la media dei campioni.  Per questo la figura colora le celle col
    peso nominale e ne disegna la forma vera.
    """
    mirrored = dirs * np.array([1.0, 1.0, -1.0])
    sv = SphericalVoronoi(np.concatenate([dirs, mirrored], axis=0),
                          radius=1.0, center=np.zeros(3))
    sv.sort_vertices_of_regions()
    cells = [sv.vertices[r] for r in sv.regions[:len(dirs)]]
    return cells, sv.calculate_areas()[:len(dirs)]


def build_case() -> dict:
    """Geometria, binning e tutte le quantita' dell'equazione."""
    n = np.array([0.0, 0.0, 1.0])
    tv, pv = np.radians(THETA_V), np.radians(PHI_V)
    v = np.array([np.sin(tv) * np.cos(pv), np.sin(tv) * np.sin(pv), np.cos(tv)])
    r = 2.0 * np.dot(n, v) * n - v          # Equazione del raggio riflesso

    dirs = fibonacci_hemisphere(S)
    ang = np.degrees(np.arccos(np.clip(dirs @ r, -1.0, 1.0)))   # angolo da R
    half = np.array(APERTURES) / 2.0
    ring = np.digitize(ang, half[1:], right=True)               # 0..K-2 dentro, K-1 fuori
    ring[ang > half[-1]] = len(half) - 1                        # fuori dal cono piu' largo

    k = len(APERTURES)
    ring_sum = np.zeros((1, k, 3))
    ring_valid = np.zeros((1, k))
    lum = radiance(dirs)
    for i in range(k - 1):
        m = ring == i
        ring_sum[0, i + 1] = lum[m].sum()
        ring_valid[0, i + 1] = m.sum()
    ring_sum[0, 0] = radiance(r)            # livello specchio: un raggio solo
    ring_valid[0, 0] = 1.0

    cos_b = np.cos(np.radians(np.asarray(APERTURES)) * 0.5)
    omega = 2.0 * np.pi * (cos_b[:-1] - cos_b[1:])
    n_nom = np.asarray(spec_cone_shared_ring_samples(APERTURES, S))
    w = ring_weights_mean(cos_b, k - 1, n_nom)
    cones = _cones_from_rings_np(ring_sum, ring_valid, w)[0, :, 0]

    # Le due allocazioni del budget messe a confronto in fig_weights.  Quella mirata passa
    # dalla funzione di produzione, cosi' se un giorno cambia allocazione cambia la figura.
    n_aimed = np.asarray(spec_cone_ring_samples(APERTURES, AIMED_PER_RING, alloc="uniform"),
                         dtype=np.float64)
    # I pesi disegnati si dividono a mano e NON passano da ring_weights_mean: con
    # ring_samples uniforme quella salta la divisione di proposito, perche' un fattore
    # costante si semplifica fra numeratore e denominatore dell'equazione dei coni.  Esatto
    # per il valore del cono, sbagliato di un fattore N per una figura il cui soggetto e'
    # proprio quanto vale una singola toppa.
    w_draw_aimed = omega / n_aimed
    w_shared = 2.0 * np.pi / S          # = Omega_i/N_i con N_i = S*Omega_i/2pi, per ogni i

    cells, cell_area = _hemisphere_voronoi(dirs)

    # Forma chiusa, valida solo dove il cono non tocca l'orizzonte
    exact = 0.5 + 0.5 * r[2] * (1.0 + cos_b) / 2.0
    unclipped = np.degrees(np.arccos(r[2])) + half <= 90.0

    return dict(n=n, v=v, r=r, dirs=dirs, ang=ang, ring=ring, lum=lum,
                omega=omega, n_nom=n_nom, w=w, ring_sum=ring_sum[0, :, 0],
                ring_valid=ring_valid[0], cones=cones, exact=exact,
                unclipped=unclipped, half=half, cos_b=cos_b,
                n_aimed=n_aimed, w_draw_aimed=w_draw_aimed, w_shared=w_shared,
                cells=cells, cell_area=cell_area)


def _frame(ax, case: dict, arrows: bool = True) -> None:
    """Piano della superficie, cerchio dell'orizzonte, normale, camera, raggio riflesso."""
    # Il quadrato della superficie sta dentro l'equatore: piu' grande, coprirebbe le
    # direzioni radenti e le farebbe sembrare sotto la superficie, che e' impossibile.
    g = np.linspace(-0.62, 0.62, 2)
    gx, gy = np.meshgrid(g, g)
    ax.plot_surface(gx, gy, np.zeros_like(gx), color="0.9", alpha=0.45,
                    edgecolor="0.65", linewidth=0.6, zorder=0)
    # l'equatore dell'emisfero: senza, il 3D si legge come un disegno piatto
    a = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(a), np.sin(a), np.zeros_like(a), color="0.6", lw=0.9, zorder=1)
    if arrows:
        # Le etichette sono spostate di lato: sopra la punta ci finiscono anche quelle
        # delle aperture, che stanno in cima ai rispettivi cerchi.
        for vec, lab, col, off in (
                (case["n"], r"$\mathbf{n}$", C_INK, (-0.12, 0.12, 0.06)),
                (case["v"], r"$\mathbf{v}$", C_INK, (-0.10, 0.10, 0.06)),
                (case["r"], r"$\mathbf{R}$", C_MIRROR, (0.10, -0.10, 0.06))):
            ax.quiver(0, 0, 0, *vec, color=col, lw=2.0, arrow_length_ratio=0.12)
            ax.text(*(vec * 1.06 + np.array(off)), lab, color=col, fontsize=15)
    ax.scatter([0], [0], [0], color=C_INK, s=22, zorder=5)
    # Il rapporto della scatola DEVE seguire le estensioni dei dati, altrimenti la scala
    # verticale e' diversa da quella orizzontale e gli angoli non si leggono piu': i
    # cerchi dei coni, che stanno su piani perpendicolari a R, appaiono ruotati male e
    # n sembra piu' lungo di R benche' siano entrambi unitari.  In un diagramma il cui
    # contenuto sono gli angoli e' un errore, non una scelta di inquadratura.
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(0, 1.1)
    ax.set_box_aspect((2.1, 2.1, 1.1))
    ax.set_axis_off()
    ax.view_init(elev=32, azim=-72)


def fig_geometry(case: dict, out: Path) -> None:
    fig = plt.figure(figsize=(6.2, 4.6))
    ax = fig.add_subplot(111, projection="3d")
    _frame(ax, case)

    ax.text(0.26, -0.06, 0.0, "texel", color=C_INK, fontsize=12)
    # Arco fra R e il bordo del cono a 60 gradi, disegnato sulla sfera unitaria in modo
    # che i suoi estremi cadano esattamente su R e sul cerchio: e' li' che si vede che
    # l'apertura e' TOTALE e il bordo sta a meta'.  Lo si apre dalla parte opposta alla
    # normale, dove non c'e' nient'altro da leggere.
    hd = case["half"][2]
    u = case["n"] - np.dot(case["n"], case["r"]) * case["r"]
    u = -u / np.linalg.norm(u)
    arc = np.array([np.cos(np.radians(g)) * case["r"] + np.sin(np.radians(g)) * u
                    for g in np.linspace(0, hd, 60)])
    ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color=C_INK, lw=1.1)
    ax.text(*(arc[len(arc) // 2] * 1.06), r"$\Theta/2$", color=C_INK, fontsize=14)

    for i, hd in enumerate(case["half"][1:]):
        c = circle_on_sphere(case["r"], hd)
        keep = c[:, 2] >= 0                      # sotto l'orizzonte non esistono direzioni
        cc = c.copy()
        cc[~keep] = np.nan
        ax.plot(cc[:, 0], cc[:, 1], cc[:, 2], color=RING_COLORS[i], lw=2.0)
        # Etichetta sul fianco del cerchio, nella direzione perpendicolare al piano che
        # contiene n e R: e' l'unica zona dove i cerchi non si accavallano fra loro ne'
        # con le frecce, e i raggi diversi separano le etichette da soli.
        _, side = onb(case["r"])
        p = np.cos(np.radians(hd)) * case["r"] + np.sin(np.radians(hd)) * side
        ax.text(*(p * 1.08), f"{APERTURES[i + 1]:.0f}$^\\circ$",
                color=C_INK, fontsize=12, ha="center")

    # generatrici del cono piu' largo, per far leggere il solido invece del bordo
    for t in np.linspace(0, 2 * np.pi, 13)[:-1]:
        e = circle_on_sphere(case["r"], case["half"][-1], 240)
        p = e[int(t / (2 * np.pi) * 239)]
        if p[2] < 0:
            continue
        ax.plot([0, p[0]], [0, p[1]], [0, p[2]], color=RING_COLORS[-1],
                lw=0.6, alpha=0.35)

    # Titolo come testo di figura e non dell'asse: gli assi 3D lasciano una fascia vuota
    # sopra il contenuto, e un titolo agganciato all'asse la trasformerebbe in margine
    # dentro la pagina.
    fig.text(0.5, 0.9, "Candidate cones share the reflected ray as their axis",
             ha="center", fontsize=14, color=C_INK)
    ax.set_position([0.0, -0.06, 1.0, 1.06])
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"  + {out}")


def fig_rings(case: dict, out: Path) -> None:
    fig = plt.figure(figsize=(10.0, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.05), wspace=0.02)

    ax = fig.add_subplot(gs[0], projection="3d")
    _frame(ax, case)
    d, ring = case["dirs"], case["ring"]
    for i in range(len(APERTURES) - 1):
        m = ring == i
        ax.scatter(d[m, 0], d[m, 1], d[m, 2], s=30, color=RING_COLORS[i],
                   depthshade=False, edgecolor="white", linewidth=0.4, zorder=4)
    m = ring == len(APERTURES) - 1
    ax.scatter(d[m, 0], d[m, 1], d[m, 2], s=14, color=C_OUT, depthshade=False,
               alpha=0.6, zorder=3)
    for i, hd in enumerate(case["half"][1:]):
        c = circle_on_sphere(case["r"], hd)
        c[c[:, 2] < 0] = np.nan
        ax.plot(c[:, 0], c[:, 1], c[:, 2], color=RING_COLORS[i], lw=1.2, alpha=0.85)
    ax.set_title(f"The {S} shared directions, binned by angle to $\\mathbf{{R}}$",
                 fontsize=13, color=C_INK, pad=-8)

    # Vista lungo R: raggio = angolo da R, azimut attorno a R.  E' qui che i punti
    # per anello si contano.
    axp = fig.add_subplot(gs[1], projection="polar")
    t, b = onb(case["r"])
    az = np.arctan2(d @ b, d @ t)
    for i in range(len(APERTURES) - 1):
        m = ring == i
        axp.scatter(az[m], case["ang"][m], s=30, color=RING_COLORS[i],
                    edgecolor="white", linewidth=0.4, zorder=4,
                    label=f"ring {i + 1}: {APERTURES[i]:.0f}$^\\circ$ to "
                          f"{APERTURES[i + 1]:.0f}$^\\circ$  ($n_{i + 1}$ = "
                          f"{int(case['ring_valid'][i + 1])})")
    m = ring == len(APERTURES) - 1
    axp.scatter(az[m], case["ang"][m], s=18, color=C_OUT, zorder=3,
                label="outside every candidate")
    axp.scatter([0], [0], marker="+", s=110, color=C_MIRROR, zorder=6,
                linewidth=2.0, label="mirror ray $\\mathbf{R}$")

    # L'orizzonte visto da qui: dove una direzione a quell'angolo da R affonda sotto
    # la superficie.  E' la ragione per cui gli anelli esterni non si riempiono mai.
    # Angolo massimo dall'asse R che resta sopra l'orizzonte, per ogni azimut: la
    # direzione e' cos(g)*R + sin(g)*(cos(az)*T + sin(az)*B) e affonda dove la sua
    # componente verticale si annulla, cioe' tan(g) = -R_z / k_z.  Dove k_z >= 0 la
    # soluzione cade oltre i 90 gradi e finisce fuori dal grafico, che e' corretto:
    # da quel lato il cono non incontra mai la superficie.
    aa = np.linspace(0, 2 * np.pi, 361)
    kz = np.cos(aa) * t[2] + np.sin(aa) * b[2]
    horiz = np.degrees(np.arctan2(-case["r"][2], kz)) % 180.0
    axp.plot(aa, horiz, color="0.45", lw=1.2, ls="--", zorder=2,
             label="surface horizon")

    for hd in case["half"][1:]:
        axp.plot(np.linspace(0, 2 * np.pi, 200), np.full(200, hd),
                 color="0.55", lw=0.8, zorder=1)
    axp.set_rmax(90)
    axp.set_rticks(list(case["half"][1:]))
    axp.set_rlabel_position(112)     # fuori dalla zona dove si addensano i punti
    axp.set_yticklabels([f"{h:.0f}$^\\circ$" for h in case["half"][1:]], fontsize=11,
                        color="0.35")
    axp.set_xticklabels([])
    axp.grid(color="0.85", lw=0.6)
    axp.set_title("Same directions, seen along $\\mathbf{R}$", fontsize=13,
                  color=C_INK, pad=14)
    # Legenda sotto e su tre colonne: in colonna a destra rubava larghezza ai pannelli,
    # che in pagina sono gia' ridotti di un terzo e diventavano illeggibili.
    axp.legend(loc="upper center", bbox_to_anchor=(-0.08, -0.02), ncol=3, fontsize=11,
               frameon=False, handletextpad=0.4, columnspacing=1.6)

    fig.tight_layout()
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"  + {out}")


def _equal_area_radius(half_deg) -> np.ndarray:
    """Proiezione azimutale equivalente di Lambert attorno a R: rho = 2 sin(theta/2).

    E' l'unica scelta che rende la figura leggibile come dice di essere: con questo
    raggio l'area disegnata di una regione VALE il suo angolo solido in steradianti
    (rho drho = sin theta dtheta), quindi una cella grande il doppio e' un raggio che
    sta per il doppio di cielo.  La vista polare di cone_rings.png, dove il raggio e'
    l'angolo da R, non ha questa proprieta' e qui sarebbe una figura che mente.
    """
    return 2.0 * np.sin(np.radians(np.asarray(half_deg, dtype=np.float64)) * 0.5)


def _project_disk(u: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """(..., 3) direzioni -> (..., 2) nel disco equivalente attorno ad `axis`.

    rho = 2 sin(gamma/2) = sqrt(2(1 - cos gamma)), azimut attorno all'asse.
    """
    t, b = onb(axis)
    c = np.clip(u @ axis, -1.0, 1.0)
    rho = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - c)))
    az = np.arctan2(u @ b, u @ t)
    return np.stack([rho * np.cos(az), rho * np.sin(az)], axis=-1)


def _slerp_ring(v: np.ndarray, steps: int = 6) -> np.ndarray:
    """Lati di una cella sferica campionati lungo la geodetica invece che in linea retta.

    Una cella e' larga una quindicina di gradi: unendo i vertici con segmenti nel disco i
    bordi risulterebbero visibilmente piu' dritti del vero e le celle adiacenti non
    combacerebbero.
    """
    out = []
    for k in range(len(v)):
        a, b = v[k], v[(k + 1) % len(v)]
        w = np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)),
                              -1.0, 1.0))
        if w < 1e-9:
            out.append(a[None, :])
            continue
        s = np.linspace(0.0, 1.0, steps, endpoint=False)[:, None]
        out.append((np.sin((1.0 - s) * w) * a + np.sin(s * w) * b) / np.sin(w))
    return np.concatenate(out, axis=0)


def _dot_color(color) -> str:
    """Punto bianco o nero a seconda della cella: la rampa attraversa tutto l'intervallo,
    e un punto bianco sparisce sul giallo tanto quanto uno nero sparisce sul nero."""
    lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    return C_INK if lum > 0.55 else "white"


def _weights_chrome(ax, case: dict, rows: str) -> None:
    """Cerchi degli anelli, aperture, raggio specchio, orizzonte, elenco sotto il pannello.
    Identico nei due pannelli: l'unica cosa che cambia fra loro deve essere N_i."""
    rho = _equal_area_radius(case["half"])
    a = np.linspace(0, 2 * np.pi, 400)
    for i in range(len(APERTURES) - 1):
        # Stesso colore per anello di cone_rings.png e cone_geometry.png: le tre figure
        # devono leggersi come lo stesso oggetto.
        ax.plot(rho[i + 1] * np.cos(a), rho[i + 1] * np.sin(a),
                color=RING_COLORS[i], lw=1.6, zorder=5)
        ax.text(0.0, rho[i + 1] + 0.035, f"{APERTURES[i + 1]:.0f}$^\\circ$",
                ha="center", va="bottom", fontsize=11, color=C_INK, zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.8, alpha=0.85))

    # L'orizzonte: il cerchio massimo perpendicolare a n, visto da qui.  Il cono piu' largo
    # lo attraversa, ed e' li' che il set condiviso finisce.
    e1, e2 = onb(case["n"])
    ang = np.linspace(0, 2 * np.pi, 721)
    xy = _project_disk(np.cos(ang)[:, None] * e1 + np.sin(ang)[:, None] * e2, case["r"])
    ax.plot(xy[:, 0], xy[:, 1], color="0.45", lw=1.2, ls="--", zorder=5,
            clip_path=Circle((0, 0), rho[-1], transform=ax.transData))

    ax.scatter([0], [0], marker="+", s=120, color=C_MIRROR, linewidth=2.0, zorder=6)
    ax.text(0.10, -0.13, r"$\mathbf{R}$", color=C_MIRROR, fontsize=14, zorder=6)

    lim = rho[-1] * 1.16
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.text(0.5, -0.04, rows, transform=ax.transAxes, ha="center", va="top",
            fontsize=11, color=C_INK, family="monospace", linespacing=1.5)


def _panel_shared(ax, case: dict, norm, cmap) -> None:
    """Il set condiviso vero, ogni direzione con la sua cella di Voronoi."""
    rho = _equal_area_radius(case["half"])
    clip = Circle((0, 0), rho[-1], transform=ax.transData)
    # Una tinta sola per tutte le celle, senza distinguere in quale anello cadono: e'
    # esattamente cio' che il pannello afferma, un raggio vale 2*pi/S qualunque anello lo
    # accolga.  Il binning sta nei cerchi disegnati sopra, ed e' il soggetto di
    # cone_rings.png, non di questo.
    color = cmap(norm(case["w_shared"]))
    for cell in case["cells"]:
        p = Polygon(_project_disk(_slerp_ring(cell), case["r"]), closed=True,
                    facecolor=color, edgecolor="white", lw=0.7, zorder=2)
        p.set_clip_path(clip)
        ax.add_patch(p)

    pts = _project_disk(case["dirs"], case["r"])
    s = ax.scatter(pts[:, 0], pts[:, 1], s=9, color=_dot_color(color), zorder=4,
                   linewidths=0)
    s.set_clip_path(clip)


def _panel_aimed(ax, case: dict, norm, cmap) -> None:
    """Budget piatto: l'anello i tagliato in N settori uguali, disegnati in scala."""
    rho = _equal_area_radius(case["half"])
    for i in range(len(APERTURES) - 1):
        n_i = int(case["n_aimed"][i])
        color = cmap(norm(case["w_draw_aimed"][i]))
        r_in, r_out = rho[i], rho[i + 1]
        # Settori senza sfasamento fra un anello e l'altro: i bordi radiali si allineano e
        # si leggono come dieci raggi che attraversano tutti gli anelli, che e' esattamente
        # quello che il pannello racconta.
        edges = np.linspace(0.0, 360.0, n_i + 1)
        for k in range(n_i):
            ax.add_patch(Wedge((0.0, 0.0), r_out, edges[k], edges[k + 1],
                               width=r_out - r_in, facecolor=color,
                               edgecolor="white", lw=0.7, zorder=2))
        # Un punto per cella, sul raggio che ne dimezza l'area: e' il raggio tracciato, e
        # ricorda che la cella e' la sua toppa di cielo e non una decorazione.
        mid_a = np.radians(0.5 * (edges[:-1] + edges[1:]))
        mid_r = np.sqrt(0.5 * (r_in ** 2 + r_out ** 2))
        ax.scatter(mid_r * np.cos(mid_a), mid_r * np.sin(mid_a), s=9,
                   color=_dot_color(color), zorder=4, linewidths=0)


def fig_weights(case: dict, out: Path) -> None:
    fig = plt.figure(figsize=(10.0, 6.4))
    # Tre righe e non due: quella di mezzo resta vuota e fa spazio all'elenco degli anelli,
    # che vive nelle coordinate dei pannelli e altrimenti finisce sotto la barra.
    gs = fig.add_gridspec(3, 2, height_ratios=(1.0, 0.24, 0.04), hspace=0.10, wspace=0.04)

    w_all = np.append(case["w_draw_aimed"], case["w_shared"])
    # Scala logaritmica: fra la cella piu' piccola e la piu' grande c'e' un fattore dieci,
    # e in lineare i due terzi bassi finirebbero tutti nella stessa tinta scura.  La
    # dimensione resta comunque il canale principale, il colore la rinforza.
    norm = LogNorm(vmin=w_all.min(), vmax=w_all.max())
    cmap = plt.get_cmap("magma_r")

    # A sinistra l'elenco ha una colonna in piu': il pannello mostra i raggi veri, quindi
    # accanto al budget N_i (un valore ATTESO, e i decimali lo dicono) si puo' leggere il
    # conteggio n_i che e' arrivato davvero.  A destra di raggi veri non ce ne sono.
    rows_shared = "\n".join(
        f"ring {i + 1}:  N = {case['n_nom'][i]:4.1f}   n = {int(case['ring_valid'][i + 1]):2d}"
        f"   W = {case['w_shared']:.3f} sr" for i in range(len(APERTURES) - 1))
    rows_aimed = "\n".join(
        f"ring {i + 1}:  N = {int(case['n_aimed'][i]):2d}   W = {case['w_draw_aimed'][i]:.3f} sr"
        for i in range(len(APERTURES) - 1))

    panels = ((_panel_shared, rows_shared,
               f"The {S} shared directions, uniform in solid angle",
               f"each owns a patch of $2\\pi/S$ = {case['w_shared']:.3f} sr"),
              (_panel_aimed, rows_aimed,
               f"Flat budget, $N_i$ = {AIMED_PER_RING} on every ring",
               "what a ray carries spans "
               f"{case['w_draw_aimed'].max() / case['w_draw_aimed'].min():.1f}$\\times$"))
    for col, (draw, rows, title, sub) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        draw(ax, case, norm, cmap)
        _weights_chrome(ax, case, rows)
        ax.set_title(f"{title}\n{sub}", fontsize=13, color=C_INK, pad=10)

    cax = fig.add_subplot(gs[2, :])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation="horizontal")
    cb.set_label("solid angle one ray stands for, $W_i = \\Omega_i / N_i$  [sr]",
                 fontsize=12, color=C_INK)
    # Tick sui valori che le celle hanno davvero, non sulle decadi (in un fattore dieci non
    # ne cade nessuna): i quattro del pannello a budget piatto, che coprono tutto
    # l'intervallo.  Il peso del pannello condiviso, 2*pi/S, cade fra il secondo e il terzo
    # e non va ripetuto: e' gia' scritto nel titolo e nell'elenco.
    ticks = np.sort(case["w_draw_aimed"])
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:.3f}" for t in ticks], fontsize=10)
    cb.ax.minorticks_off()

    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"  + {out}")


def print_table(case: dict) -> None:
    print("\n% righe della tabella dell'esempio (generate, non scritte a mano)")
    print("% ring & Theta_i & Omega_i & N_i & n_i & b_i & W_i & L(Theta_k) & exact")
    print(f"mirror & $0^\\circ$ & \\dash & \\dash & 1 & "
          f"{case['ring_sum'][0]:.3f} & \\dash & {case['cones'][0]:.4f} & "
          f"{case['exact'][0]:.4f} \\\\")
    for i in range(len(APERTURES) - 1):
        ex = (f"{case['exact'][i + 1]:.4f}" if case["unclipped"][i + 1] else "\\dash")
        print(f"{i + 1} & ${APERTURES[i]:.0f}^\\circ$ to ${APERTURES[i + 1]:.0f}^\\circ$ & "
              f"{case['omega'][i]:.3f} & {case['n_nom'][i]:.1f} & "
              f"{int(case['ring_valid'][i + 1])} & {case['ring_sum'][i + 1]:.3f} & "
              f"{case['w'][i]:.4f} & {case['cones'][i + 1]:.4f} & {ex} \\\\")

    print(f"\n% R = ({case['r'][0]:.3f}, {case['r'][1]:.3f}, {case['r'][2]:.3f}), "
          f"{np.degrees(np.arccos(case['r'][2])):.1f} deg dalla normale")
    print(f"% W_i costante? min {case['w'].min():.6f} max {case['w'].max():.6f} "
          f"(2*pi/S = {2 * np.pi / S:.6f})")
    good = case["unclipped"][1:]
    if good.any():
        err = np.abs(case["cones"][1:][good] - case["exact"][1:][good])
        print(f"% scarto dalla forma chiusa sui coni non tagliati: max {err.max():.4f}")


def print_weights(case: dict) -> None:
    """I numeri che la didascalia di cone_weights.png cita, stampati e non scritti a mano."""
    print("\n% pesi delle due allocazioni (figura cone_weights)")
    print("% ring & Omega_i & N_i atteso & n_i vero & W_i cond. & N_i mirato & W_i mirato")
    for i in range(len(APERTURES) - 1):
        print(f"{i + 1} & {case['omega'][i]:.3f} & {case['n_nom'][i]:.2f} & "
              f"{int(case['ring_valid'][i + 1])} & {case['w_shared']:.4f} & "
              f"{int(case['n_aimed'][i])} & {case['w_draw_aimed'][i]:.4f} \\\\")
    wa = case["w_draw_aimed"]
    print(f"% condiviso: W = 2*pi/S = {case['w_shared']:.4f} su ogni anello")
    print(f"% mirato:    W fra {wa.min():.4f} e {wa.max():.4f}, "
          f"rapporto {wa.max() / wa.min():.2f}")
    inside = case["ring"] < len(APERTURES) - 1
    print(f"% raggi dentro il cono piu' largo: {int(inside.sum())} condivisi "
          f"(su S = {S} sull'emisfero), {int(case['n_aimed'].sum())} mirati")

    # Le celle di Voronoi non sono tutte uguali e la didascalia lo dice: qui il numero.
    a = case["cell_area"] / case["w_shared"]
    print(f"% celle di Voronoi (in unita' di 2*pi/S): somma {case['cell_area'].sum():.5f} "
          f"= 2*pi ({2 * np.pi:.5f}), mediana {np.median(a):.3f}, "
          f"quartili {np.percentile(a, 25):.3f}/{np.percentile(a, 75):.3f}, "
          f"min {a.min():.3f} (all'orizzonte) max {a.max():.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    case = build_case()
    fig_geometry(case, out / "cone_geometry.png")
    fig_rings(case, out / "cone_rings.png")
    fig_weights(case, out / "cone_weights.png")
    print_table(case)
    print_weights(case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
