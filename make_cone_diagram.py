#!/usr/bin/env python
"""make_cone_diagram.py -- Figure ed esempio numerico per l'equazione dei coni.

    python make_cone_diagram.py --out ../Doc/images/cone

Scrive due PNG e stampa le righe della tabella LaTeX dell'esempio:

  cone_geometry.png  texel, normale, camera, raggio riflesso e i coni candidati
  cone_rings.png     il set condiviso di Fibonacci, colorato per anello, in 3D e
                     nella vista lungo R dove il binning si conta a occhio

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
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))

from images_generator import (_cones_from_rings_np, ring_weights_mean,
                              spec_cone_shared_ring_samples)

# Caso didattico: pochi raggi, contabili a occhio, e una griglia di aperture ridotta
# a quattro anelli invece dei tredici della configurazione operativa.
APERTURES = [0.0, 30.0, 60.0, 90.0, 140.0]   # aperture TOTALI, in gradi
S = 96                                        # direzioni condivise sull'emisfero
THETA_V = 30.0                                # inclinazione della camera dalla normale
PHI_V = 200.0                                 # suo azimut, scelto solo per l'inquadratura

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

    # Forma chiusa, valida solo dove il cono non tocca l'orizzonte
    exact = 0.5 + 0.5 * r[2] * (1.0 + cos_b) / 2.0
    unclipped = np.degrees(np.arccos(r[2])) + half <= 90.0

    return dict(n=n, v=v, r=r, dirs=dirs, ang=ang, ring=ring, lum=lum,
                omega=omega, n_nom=n_nom, w=w, ring_sum=ring_sum[0, :, 0],
                ring_valid=ring_valid[0], cones=cones, exact=exact,
                unclipped=unclipped, half=half)


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
    print_table(case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
