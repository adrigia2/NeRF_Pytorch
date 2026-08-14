#!/usr/bin/env python
"""make_pbr_model_diagram.py -- Figure dell'equazione del modello PBR.

    python make_pbr_model_diagram.py --out ../Doc/images/pbr-model
    python make_pbr_model_diagram.py --out ../figure_review/pbr-model --figures kernel

Scrive due PNG:

  pbr_model.png       un texel, la sua normale, l'emisfero da cui arriva
                      l'irradianza diffusa e due camere con il proprio raggio
                      riflesso e il cono di apertura Theta attorno ad esso.

  tophat_vs_ggx.png   gli STESSI raggi visti lungo la direzione riflessa, pesati
                      nei due modi: piatto dentro il cono (top-hat) e con la NDF
                      GGX.  Non e' una figura sulla forma del kernel ma sulla sua
                      MISURABILITA': con il top-hat i pesi valgono 1 e si
                      conoscono prima di tracciare, quindi la stima e' la media
                      dei raggi; con il GGX ogni peso dipende da alpha, cioe'
                      dalla larghezza che il fit deve ancora ricavare.

La figura e' volutamente muta sui numeri: la scomposizione del colore nei due
termini la fa il testo della tesi, qui si disegna solo la geometria che rende i
due termini diversi fra loro.  I conti restano pero' nello script, stampati a
fine run, perche' sono la verifica che il disegno stia dicendo la verita': il
sole cade dentro il cono della camera 1 e fuori da quello della camera 2, quindi
L_1 e L_2 devono venire diversi, ed e' esattamente cio' su cui il fit lavora.

L'ambiente e' un cielo analitico (gradiente piu' un sole caldo), l'irradianza e'
la sua integrazione pesata sul coseno sull'emisfero e L_j e' la media pura ad
angolo solido sul cono, chiusa dalle STESSE funzioni del bake
(`ring_weights_mean` e `_cones_from_rings_np` di images_generator), cosi' i
numeri stampati non possono divergere dalla matematica della pipeline.  Un
controllo indipendente per campionamento diretto del cono viene stampato accanto.

I punti della cupola sono la radianza incidente dopo esposizione condivisa e
gamma, cioe' come apparirebbe a schermo; l'esposizione e' scelta sulle medie sui
coni e sull'irradianza, cosi' il sole satura in bianco e il resto del cielo resta
leggibile.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d

import _paths  # noqa: F401

from images_generator import (_cones_from_rings_np, ring_weights_mean,
                              spec_cone_shared_ring_samples)

# ── Parametri del caso didattico ──────────────────────────────────────────────
# Una sola apertura, non la griglia di candidati: quella e' il soggetto di
# fig:cone-geometry.  Qui Theta e' gia' stata scelta dal fit ed e' la stessa per
# le due camere, perche' nel modello e' una proprieta' del texel e non della vista.
THETA = 40.0                      # apertura TOTALE del cono, in gradi
X_DIFFUSE = 0.6                   # peso del termine diffuso; metallic = 1 - x
ALBEDO = np.array([0.62, 0.45, 0.31])     # a, riflettanza diffusa in [0,1]

# Le camere stanno basse sull'orizzonte per due ragioni: i loro coni si aprono a
# ventaglio invece di accavallarsi attorno alla normale, e la calotta in alto
# resta libera per la normale e per le didascalie dell'emisfero.  Il vincolo da
# rispettare e' theta + Theta/2 <= 90 gradi, altrimenti il cono taglia l'orizzonte
# e la media non e' piu' quella di un cono intero (`report` lo verifica).
# Gli azimut sono separati di piu' di Theta anche fra la camera di uno e il cono
# dell'altra (R_j sta all'azimut opposto a v_j): con 170 e 20 gradi il glifo della
# camera 1 finiva dentro il bordo del cono della camera 2.
CAMS = (                          # (theta dalla normale, azimut) in gradi
    # Rosso e verde acqua e non rosso e blu: i pallini della cupola sono azzurrini
    # (e' il colore del cielo), e una camera blu si confondeva con loro proprio
    # dentro il suo cono, dove le due popolazioni si sovrappongono.
    dict(theta=55.0, phi=160.0, name="camera 1", color="#c0392b"),
    dict(theta=45.0, phi=30.0,  name="camera 2", color="#12776b"),
)

S_INT = 200_000                   # direzioni per gli integrali (E e L_j)
S_DOME = 520                      # direzioni disegnate come cupola
DOME_S = 22                       # area del pallino della cupola a theta = 0
CONE_DOTS = 34                    # direzioni disegnate dentro ciascun cono
CONE_S = 24                       # area, uguale per tutte: nel cono non c'e' peso

# Cielo analitico: gradiente orizzonte-zenit piu' un sole gaussiano.  Il sole e'
# messo vicino al raggio riflesso della camera 1 e lontano da quello della
# camera 2: e' quello che rende L_1 e L_2 diversi, che e' il punto della figura.
# Il cielo e' tenuto poco saturo perche' il termine diffuso, che e' l'albedo
# moltiplicata per questa luce, resti riconoscibile come colore del materiale.
SKY_HORIZON = np.array([0.78, 0.80, 0.84])
SKY_ZENITH = np.array([0.24, 0.36, 0.60])
SUN_RGB = np.array([1.00, 0.80, 0.52]) * 2.2
SUN_SIGMA = 16.0                  # gradi
SUN_OFFSET = 8.0                  # scostamento del sole da R_1, in gradi

# ── Parametri della figura sul kernel (top-hat contro lobo GGX) ───────────────
ALPHA_GGX = 0.30                  # rugosita' GGX del lobo disegnato (alpha = r^2)
S_KERNEL = 520                    # direzioni tracciate, LE STESSE nei due pannelli
R_TILT = 40.0                     # inclinazione di R dalla normale, in gradi
KERNEL_CONTAIN = 0.90             # frazione di peso del lobo che il cono deve tenere
KERNEL_S = 78.0                   # area del pallino di peso massimo, scala condivisa

C_INK = "#222222"
C_DIFF = "#e8a33d"                # ambra: tutto cio' che appartiene al termine diffuso
C_KERNEL = "#2171b5"              # blu della rampa dei coni, uguale nei due pannelli
C_OUT = "#b8b8b8"                 # raggi che il top-hat non traccia nemmeno

# Alone bianco per le etichette del 3D.  Serve perche' negli assi 3D lo zorder non
# decide l'ordine di disegno: matplotlib riordina gli artisti per profondita', e
# un'etichetta che sta dietro alla campitura di un cono ci finisce sotto comunque.
HALO = dict(fc="white", ec="none", alpha=0.75, pad=1.0)

# In pagina la figura viene rimpicciolita: le dimensioni sono scelte perche' il
# testo resti leggibile DOPO la riduzione, non sullo schermo.
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
    sull'angolo aureo.
    """
    i = np.arange(s)
    cos_t = 1.0 - (i + 0.5) / s
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t ** 2))
    phi = i * np.pi * (3.0 - np.sqrt(5.0))
    return np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=-1)


def cone_directions(axis: np.ndarray, half_deg: float, n: int) -> np.ndarray:
    """(n, 3) direzioni uniformi in angolo solido DENTRO il cono attorno ad `axis`.

    Uniformi in angolo solido significa cos(angolo dall'asse) equispaziato, non
    l'angolo: e' la stessa condizione che rende la media sul cono una media
    semplice dei campioni, senza pesi.
    """
    t, b = onb(axis)
    i = np.arange(n)
    cz = 1.0 - (i + 0.5) / n * (1.0 - np.cos(np.radians(half_deg)))
    sz = np.sqrt(np.maximum(0.0, 1.0 - cz ** 2))
    az = i * np.pi * (3.0 - np.sqrt(5.0))
    return (cz[:, None] * axis
            + sz[:, None] * (np.cos(az)[:, None] * t + np.sin(az)[:, None] * b))


def circle_on_sphere(axis: np.ndarray, half_deg: float, n: int = 240) -> np.ndarray:
    """Direzioni a `half_deg` gradi da `axis`: il bordo di un cono."""
    t, b = onb(axis)
    a = np.radians(half_deg)
    ang = np.linspace(0, 2 * np.pi, n)
    return (np.cos(a) * axis[None, :]
            + np.sin(a) * (np.cos(ang)[:, None] * t + np.sin(ang)[:, None] * b))


def label3d(ax, p, text: str, **kw) -> None:
    """Etichetta ancorata al punto 3D `p` ma disegnata come testo 2D.

    Negli assi 3D lo zorder non decide l'ordine di disegno: matplotlib riordina
    gli artisti 3D per profondita', quindi un'etichetta finisce sotto la
    campitura di un cono ogni volta che un pezzo del cono e' piu' vicino
    dell'ancora, anche se l'ancora sta fuori dal solido.  Proiettando a mano e
    disegnando un Text 2D si torna a un ordine deterministico.  Va chiamata dopo
    aver fissato limiti, box_aspect e view_init, che sono quello che
    `get_proj()` legge.
    """
    x, y, _ = proj3d.proj_transform(*p, ax.get_proj())
    # zorder alto e non 20: gli artisti 3D ricevono uno zorder calcolato dalla
    # profondita', che con molte collezioni supera i valori piccoli.
    ax.text2D(x, y, text, transform=ax.transData, zorder=200, **kw)


def sph(theta_deg: float, phi_deg: float) -> np.ndarray:
    t, p = np.radians(theta_deg), np.radians(phi_deg)
    return np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])


def sky(d: np.ndarray, sun_dir: np.ndarray) -> np.ndarray:
    """(..., 3) -> (..., 3): radianza dell'ambiente lungo `d`."""
    t = np.clip(d[..., 2], 0.0, 1.0)[..., None] ** 0.6
    base = SKY_HORIZON * (1.0 - t) + SKY_ZENITH * t
    ang = np.degrees(np.arccos(np.clip(d @ sun_dir, -1.0, 1.0)))
    glow = np.exp(-(ang / SUN_SIGMA) ** 2)[..., None]
    return base + SUN_RGB * glow


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.2126, 0.7152, 0.0722])


def build_case() -> dict:
    """Geometria, integrali e i tre termini dell'equazione."""
    n = np.array([0.0, 0.0, 1.0])
    dirs = fibonacci_hemisphere(S_INT)
    half = THETA / 2.0

    cams = []
    for spec in CAMS:
        v = sph(spec["theta"], spec["phi"])
        r = 2.0 * np.dot(n, v) * n - v          # Equazione del raggio riflesso
        cams.append(dict(spec, v=v, r=r))

    # Il sole sta vicino a R_1, scostato di SUN_OFFSET verso la normale.
    r1 = cams[0]["r"]
    u = n - np.dot(n, r1) * r1
    u /= np.linalg.norm(u)
    a = np.radians(SUN_OFFSET)
    sun_dir = np.cos(a) * r1 + np.sin(a) * u

    rad = sky(dirs, sun_dir)                    # radianza su tutto l'emisfero

    # E = integrale pesato sul coseno: ogni direzione del set vale 2*pi/S sr.
    e_irr = (rad * dirs[:, 2:3]).sum(axis=0) * (2.0 * np.pi / S_INT)

    # L_j: media pura ad angolo solido sul cono, chiusa dal codice del bake.
    # Griglia di aperture con un solo anello, [0, Theta]: il livello 1 e' il cono.
    apertures = [0.0, THETA]
    cos_b = np.cos(np.radians(np.asarray(apertures)) * 0.5)
    w = ring_weights_mean(cos_b, 1,
                          np.asarray(spec_cone_shared_ring_samples(apertures, S_INT)))
    for c in cams:
        ang = np.degrees(np.arccos(np.clip(dirs @ c["r"], -1.0, 1.0)))
        m = ang <= half
        ring_sum = np.zeros((1, 2, 3))
        ring_valid = np.zeros((1, 2))
        ring_sum[0, 1] = rad[m].sum(axis=0)
        ring_valid[0, 1] = m.sum()
        ring_sum[0, 0] = sky(c["r"], sun_dir)   # livello specchio: un raggio solo
        ring_valid[0, 0] = 1.0
        c["L"] = _cones_from_rings_np(ring_sum, ring_valid, w)[0, 1]
        c["n_rays"] = int(m.sum())
        # Il cono resta tutto sopra l'orizzonte? Se no la media e' troncata e il
        # confronto con il campionamento diretto non avrebbe senso.
        c["unclipped"] = np.degrees(np.arccos(c["r"][2])) + half <= 90.0
        # Controllo indipendente: media uniforme in angolo solido dentro il cono,
        # campionata direttamente invece che filtrando l'emisfero.  Essendo i
        # campioni uniformi in angolo solido, la media e' quella semplice.
        c["L_check"] = sky(cone_directions(c["r"], half, 20_000), sun_dir).mean(axis=0)

    diffuse = ALBEDO * X_DIFFUSE / np.pi * e_irr
    for c in cams:
        c["spec"] = (1.0 - X_DIFFUSE) * c["L"]
        c["C"] = diffuse + c["spec"]

    return dict(n=n, cams=cams, sun_dir=sun_dir, E=e_irr, diffuse=diffuse, half=half)


def make_tonemap(case: dict):
    """Esposizione condivisa da tutti gli swatch di radianza, piu' gamma."""
    peak = max(float(np.max(case["E"] / np.pi)),
               max(float(np.max(c["C"])) for c in case["cams"]),
               max(float(np.max(c["L"])) for c in case["cams"]))
    scale = 0.95 / peak

    def tm(rgb):
        return np.clip(np.asarray(rgb) * scale, 0.0, 1.0) ** (1.0 / 2.2)

    return tm


# ─────────────────────────────────────────────────────────── pannello (a), 3D
def panel_geometry(ax, case: dict, tm) -> None:
    n = case["n"]

    # Inquadratura fissata PRIMA di disegnare, perche' `label3d` proietta a mano e
    # legge la matrice di proiezione: cambiarla dopo lascerebbe le etichette dove
    # erano.  Il rapporto della scatola DEVE seguire le estensioni dei dati,
    # altrimenti la scala verticale differisce da quella orizzontale e gli angoli
    # non si leggono piu': i cerchi dei coni appaiono ruotati e n sembra piu'
    # lunga di R benche' siano entrambe unitarie.  In un diagramma di angoli e'
    # un errore, non una scelta di inquadratura.
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(0, 1.1)
    ax.set_box_aspect((2.1, 2.1, 1.1))
    ax.set_axis_off()
    ax.view_init(elev=26, azim=-72)

    # Il quadrato della superficie sta dentro l'equatore: piu' grande, coprirebbe
    # le direzioni radenti e le farebbe sembrare sotto la superficie.
    g = np.linspace(-0.62, 0.62, 2)
    gx, gy = np.meshgrid(g, g)
    ax.plot_surface(gx, gy, np.zeros_like(gx), color="0.82", alpha=0.6,
                    edgecolor="0.5", linewidth=0.7, zorder=0)
    a = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(a), np.sin(a), np.zeros_like(a), color="0.6", lw=0.9, zorder=1)

    # La cupola: le direzioni dell'emisfero colorate con l'ambiente da cui la
    # radianza arriva.  E' la stessa costruzione con cui e' integrata E, e rende
    # visibile perche' i due coni raccolgono colori diversi.
    #
    # La DIMENSIONE del pallino e' il peso con cui quella direzione entra
    # nell'irradianza, cioe' cos(theta): area esattamente proporzionale, senza
    # minimo.  I pallini che si spengono contro l'orizzonte non sono un difetto di
    # resa, sono direzioni radenti che non portano quasi nulla; la silhouette
    # dell'emisfero la tiene comunque il cerchio dell'equatore.
    dome = fibonacci_hemisphere(S_DOME)
    ax.scatter(dome[:, 0] * 1.02, dome[:, 1] * 1.02, dome[:, 2] * 1.02,
               s=DOME_S * dome[:, 2], c=tm(sky(dome, case["sun_dir"])),
               depthshade=False, alpha=0.9, linewidths=0, zorder=2)
    # Il sole disegnato a parte: fra i punti della cupola sarebbe un dettaglio di
    # pochi campioni, e invece e' la ragione per cui L_1 e L_2 non coincidono.
    # Raggio maggiore di quello dei campioni del cono, che gli cadono sopra: e'
    # l'ordinamento per profondita' del 3D a decidere chi copre chi.
    ax.scatter(*(case["sun_dir"] * 1.07)[:, None], s=300, marker="o",
               color=tm(sky(case["sun_dir"], case["sun_dir"])),
               edgecolor=C_DIFF, linewidth=1.6, depthshade=False, zorder=3)

    # Frecce entranti sparse sulla cupola: l'irradianza raccoglie tutto l'emisfero.
    # Quelle che cadrebbero dentro un cono sono saltate, la' il disegno e' gia' pieno.
    for d in fibonacci_hemisphere(13):
        if d[2] < 0.25 or any(d @ c["r"] > np.cos(np.radians(case["half"] + 14))
                              for c in case["cams"]):
            continue
        ax.quiver(*(d * 0.90), *(-d * 0.26), color=C_DIFF, lw=1.4,
                  arrow_length_ratio=0.40, alpha=0.95, zorder=3)
    # Le due didascalie stanno in coordinate di assi e non nella scena: dentro la
    # cupola ogni posizione libera in 3D finisce addosso a n o a una delle R.
    ax.text2D(0.12, 0.755, r"$E$:  the whole hemisphere", color=C_DIFF,
              fontsize=12.5, transform=ax.transAxes)
    ax.text2D(0.20, 0.155, "incident radiance", color="0.45", fontsize=11.5,
              transform=ax.transAxes)

    ax.quiver(0, 0, 0, *n, color=C_INK, lw=2.0, arrow_length_ratio=0.12, zorder=6)
    label3d(ax, n * 1.04 + np.array([0.02, 0.10, 0.02]), r"$\mathbf{n}$",
            color=C_INK, fontsize=15, bbox=HALO)
    ax.scatter([0], [0], [0], color=C_INK, s=22, zorder=7)
    label3d(ax, (0.20, -0.12, 0.02), "texel", color=C_INK, fontsize=12, bbox=HALO)

    for k, c in enumerate(case["cams"], start=1):
        v, r, col = c["v"], c["r"], c["color"]
        # La vista e' tratteggiata e sottile, il raggio riflesso pieno e spesso:
        # e' il secondo a portare il cono, e la coppia si legge come una riflessione.
        ax.plot([0, v[0]], [0, v[1]], [0, v[2]], color=col, lw=1.5, ls=(0, (4, 2)),
                zorder=6)
        ax.quiver(0, 0, 0, *r, color=col, lw=2.2, arrow_length_ratio=0.12, zorder=6)
        ax.scatter([v[0] * 1.13], [v[1] * 1.13], [v[2] * 1.13], marker="s", s=85,
                   color=col, depthshade=False, zorder=8)
        label3d(ax, v * 1.13 + np.array([0.0, 0.0, 0.24]), c["name"], color=col,
                fontsize=12, ha="center", va="bottom", bbox=HALO)
        label3d(ax, v * 0.62 + np.array([0.0, 0.0, -0.13]), rf"$\mathbf{{v}}_{k}$",
                color=col, fontsize=13, ha="center", bbox=HALO)
        # Etichetta di R oltre il bordo del cono, sull'asse: piu' vicino finirebbe
        # sotto il cerchio del bordo, e di lato finirebbe addosso alla zenit.
        label3d(ax, r * 1.30, rf"$\mathbf{{R}}_{k}$", color=col, fontsize=15,
                ha="center", va="center", bbox=HALO)

        # I campioni del cono, tutti della STESSA dimensione, sopra quelli della
        # cupola che invece rimpiccioliscono: e' la differenza fra i due termini
        # del modello resa in un solo colpo d'occhio.  L_j e' una media pura ad
        # angolo solido, senza peso cos(theta), quindi dentro al cono ogni
        # direzione conta uguale.
        #
        # Tinta piena nel colore della camera invece che la radianza campionata:
        # colorati come l'ambiente si confondevano con i pallini della cupola
        # proprio dove le due popolazioni si sovrappongono, e nel cono che guarda
        # il sole saturavano in bianco.  Che radianza vedano lo dicono comunque i
        # pallini della cupola sotto, che restano.
        cd = cone_directions(r, case["half"], CONE_DOTS)
        ax.scatter(cd[:, 0] * 1.03, cd[:, 1] * 1.03, cd[:, 2] * 1.03, s=CONE_S,
                   color=col, edgecolor="white", linewidth=0.5,
                   depthshade=False, zorder=5)

        # Superficie laterale del cono, dal texel al bordo sulla sfera unitaria.
        # Alpha basso: il sole e i punti della cupola devono restare visibili
        # attraverso il cono, altrimenti sparisce la ragione dei due colori.
        rim = circle_on_sphere(r, case["half"], 90)
        t = np.linspace(0.0, 1.0, 2)[:, None, None]
        cone = t * rim[None, :, :]
        ax.plot_surface(cone[..., 0], cone[..., 1], cone[..., 2], color=col,
                        alpha=0.15, linewidth=0, shade=False, zorder=4)
        ax.plot(rim[:, 0], rim[:, 1], rim[:, 2], color=col, lw=1.6, zorder=5)

    # L'arco Theta/2 su una camera sola: raddoppiarlo raddoppierebbe solo le
    # etichette.  Aperto dalla parte opposta alla normale, dove non c'e' altro.
    c0 = case["cams"][0]
    u = n - np.dot(n, c0["r"]) * c0["r"]
    u = -u / np.linalg.norm(u)
    arc = np.array([np.cos(np.radians(g)) * c0["r"] + np.sin(np.radians(g)) * u
                    for g in np.linspace(0, case["half"], 60)])
    ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color=C_INK, lw=1.1, zorder=6)
    label3d(ax, arc[len(arc) // 2] * 1.16, r"$\Theta/2$", color=C_INK, fontsize=14,
            va="center", bbox=HALO)


# ───────────────────────────── figura 2: kernel top-hat contro lobo GGX
def ggx_d(cos_h: np.ndarray, alpha: float) -> np.ndarray:
    """\\gls{ndf} GGX, la formula di background.tex: D = a^2 / (pi (c^2(a^2-1)+1)^2)."""
    a2 = alpha * alpha
    return a2 / (np.pi * (cos_h ** 2 * (a2 - 1.0) + 1.0) ** 2)


def build_kernel_case() -> dict:
    """I due kernel sullo STESSO insieme di direzioni, entrambi a integrale 1.

    Il kernel del prefiltraggio dipende dal solo angolo dalla direzione riflessa
    soltanto sotto l'ipotesi n = v = R, quella che rende possibile prefiltrare
    l'ambiente una volta sola invece che per ogni vista.  Sotto quell'ipotesi il
    mezzo vettore di un campione a gamma da R sta a gamma/2, e il peso di Karis
    (D per il coseno, il suo N.L) diventa

        w(gamma) = D(cos(gamma/2); alpha) * cos(gamma)

    che si annulla a 90 gradi: sotto l'ipotesi, oltre i 90 gradi da R si e' sotto
    la superficie.  Il coseno non e' un dettaglio estetico: senza, la coda pesa
    tanto da rendere il cono equivalente piu' largo dell'emisfero.

    L'apertura del top-hat non e' scelta: e' quella che tiene KERNEL_CONTAIN del
    peso del lobo.  Accostare i due pannelli con una larghezza decisa a mano
    sarebbe un confronto truccato.
    """
    n = np.array([0.0, 0.0, 1.0])
    r = sph(R_TILT, 0.0)
    dirs = fibonacci_hemisphere(S_KERNEL)
    gamma = np.degrees(np.arccos(np.clip(dirs @ r, -1.0, 1.0)))
    dw = 2.0 * np.pi / S_KERNEL          # angolo solido per campione, uniforme

    w_ggx = (ggx_d(np.cos(np.radians(gamma * 0.5)), ALPHA_GGX)
             * np.maximum(np.cos(np.radians(gamma)), 0.0))
    w_ggx /= w_ggx.sum() * dw            # integrale 1 sul dominio campionato

    order = np.argsort(gamma)
    cum = np.cumsum(w_ggx[order]) * dw
    half = float(np.interp(KERNEL_CONTAIN, cum, gamma[order]))
    omega = 2.0 * np.pi * (1.0 - np.cos(np.radians(half)))
    inside = gamma <= half
    w_top = np.where(inside, 1.0 / omega, 0.0)

    t, b = onb(r)
    return dict(n=n, r=r, dirs=dirs, gamma=gamma, dw=dw, azim=np.arctan2(dirs @ b,
                dirs @ t), w_ggx=w_ggx, w_top=w_top, inside=inside, half=half,
                omega=omega, t=t, b=b)


def panel_kernel(ax, case: dict, weights: np.ndarray, *, scale: float,
                 title: str, edge_solid: bool, note: str,
                 note_at: tuple[float, float] = (139.0, 76.0)) -> None:
    """Vista lungo R: raggio = angolo da R, area del pallino = peso del raggio."""
    ax.scatter(case["azim"], case["gamma"], s=weights * scale, color=C_KERNEL,
               edgecolor="white", linewidth=0.3, zorder=4)
    if edge_solid:
        # I raggi a peso nullo il top-hat non li traccia affatto: e' meta' di cio'
        # che lo rende misurabile, e vanno mostrati come esclusi, non come assenti.
        out = ~case["inside"]
        ax.scatter(case["azim"][out], case["gamma"][out], s=7, color=C_OUT,
                   alpha=0.75, zorder=3)
    # Il bordo del cono resta blu anche nel pannello del lobo, dove e' solo un
    # riferimento: grigio tratteggiato si confonderebbe con l'orizzonte.
    ang = np.linspace(0, 2 * np.pi, 200)
    ax.plot(ang, np.full(200, case["half"]), color=C_KERNEL,
            lw=1.8 if edge_solid else 1.2, ls="-" if edge_solid else (0, (5, 3)),
            zorder=5)
    ax.text(np.radians(note_at[0]), note_at[1], note, color="0.4", fontsize=11,
            ha="center", va="center", zorder=6, bbox=HALO)

    # L'orizzonte visto lungo R, stessa costruzione del pannello polare di
    # make_cone_diagram: una direzione a gamma da R affonda sotto la superficie
    # dove la sua componente verticale si annulla, cioe' tan(g) = -R_z / k_z.
    aa = np.linspace(0, 2 * np.pi, 361)
    kz = np.cos(aa) * case["t"][2] + np.sin(aa) * case["b"][2]
    ax.plot(aa, np.degrees(np.arctan2(-case["r"][2], kz)) % 180.0, color="0.45",
            lw=1.1, ls="--", zorder=2)

    ax.set_rmax(90)
    ax.set_rticks([30, 60, 90])
    ax.set_rlabel_position(112)
    ax.set_yticklabels(["30$^\\circ$", "60$^\\circ$", "90$^\\circ$"], fontsize=10.5,
                       color="0.35")
    ax.set_xticklabels([])
    ax.grid(color="0.85", lw=0.6)
    ax.set_title(title, fontsize=13, color=C_INK, pad=12)


def figure_kernel(case: dict, out: Path) -> None:
    scale = KERNEL_S / case["w_ggx"].max()      # scala CONDIVISA fra i pannelli:
    fig = plt.figure(figsize=(9.6, 5.9))        # e' l'unico modo di confrontarli
    gs = fig.add_gridspec(1, 2, wspace=0.06, left=0.01, right=0.99,
                          top=0.90, bottom=0.30)

    tail = case["w_ggx"][~case["inside"]].sum() * case["dw"]
    panel_kernel(fig.add_subplot(gs[0], projection="polar"), case, case["w_top"],
                 scale=scale, edge_solid=True, note="outside:\nnever traced",
                 title=f"Top-hat cone, aperture $\\Theta = {2 * case['half']:.0f}"
                       f"^\\circ$")
    panel_kernel(fig.add_subplot(gs[1], projection="polar"), case, case["w_ggx"],
                 scale=scale, edge_solid=False, note_at=(148.0, 82.0),
                 note=f"no cut, only fade:\n{tail * 100:.0f}% is past the edge",
                 title=f"GGX lobe, roughness $\\alpha = {ALPHA_GGX:.2f}$")

    # Gli stimatori sotto ai rispettivi pannelli.  La riga grigia e' il punto della
    # figura: non la forma del kernel, ma cosa bisogna sapere per poterlo valutare.
    for x, expr, note in (
            (0.25, r"$L(\Theta) = \dfrac{1}{N}\,\sum_s L_s$",
             "every weight is 1, and known\nbefore a single ray is traced"),
            (0.75, r"$L = \dfrac{\sum_s w_s L_s}{\sum_s w_s}$,"
                   r"$\quad w_s = D(\mathbf{h}_s;\,\alpha)\,\cos\gamma_s$",
             "every weight is set by $\\alpha$,\nthe width the fit has to recover")):
        fig.text(x, 0.21, expr, ha="center", va="center", fontsize=15, color=C_INK)
        fig.text(x, 0.06, note, ha="center", va="center", fontsize=11.5,
                 color="0.35", linespacing=1.25)

    fig.text(0.5, 0.965, "The same rays, weighted in two ways", ha="center",
             fontsize=14, color=C_INK)
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)
    trim_white(out)
    print(f"  + {out}")


def report_kernel(case: dict) -> None:
    """Controlli sui due kernel, piu' i numeri che finiscono in caption."""
    i_top = float(case["w_top"].sum() * case["dw"])
    i_ggx = float(case["w_ggx"].sum() * case["dw"])
    tail = float(case["w_ggx"][~case["inside"]].sum() * case["dw"])
    print(f"\n  alpha = {ALPHA_GGX:.2f}   ->   Theta = {2 * case['half']:.1f} deg "
          f"(mezza apertura {case['half']:.1f}, Omega {case['omega']:.3f} sr)")
    print(f"  integrale dei kernel: top-hat {i_top:.4f}, GGX {i_ggx:.4f}")
    print(f"  peso GGX oltre il bordo del cono: {tail * 100:.1f} %")
    print(f"  picco GGX / livello del top-hat: "
          f"{case['w_ggx'].max() / (1.0 / case['omega']):.2f}x")
    print("  Theta della stessa regola per altre rugosita': "
          + ", ".join(f"alpha {a:.2f} -> {_theta_for_alpha(a):.0f} deg"
                      for a in (0.1, 0.2, 0.5)))
    assert abs(i_top - 1.0) < 0.05 and abs(i_ggx - 1.0) < 1e-6, \
        "i due kernel non sono normalizzati allo stesso modo: le aree non si confrontano"
    assert tail > 0.02, "la coda oltre il cono e' invisibile: la figura non dice nulla"
    assert case["w_ggx"].max() * case["omega"] > 1.5, \
        "il lobo non e' piu' piccato del top-hat: rivedere alpha o la regola di larghezza"


def _theta_for_alpha(alpha: float) -> float:
    """L'apertura che la stessa regola darebbe a un'altra rugosita'."""
    g = np.degrees(np.arccos(np.clip(fibonacci_hemisphere(S_KERNEL)
                                     @ sph(R_TILT, 0.0), -1.0, 1.0)))
    w = (ggx_d(np.cos(np.radians(g * 0.5)), alpha)
         * np.maximum(np.cos(np.radians(g)), 0.0))
    o = np.argsort(g)
    cum = np.cumsum(w[o]) / w.sum()
    return 2.0 * float(np.interp(KERNEL_CONTAIN, cum, g[o]))


def trim_white(path: Path, pad: int = 14) -> None:
    """Ritaglia il bordo bianco uniforme del PNG.

    `bbox_inches="tight"` ritaglia sul RIQUADRO degli assi, non sul contenuto, e
    un asse 3D lo riempie solo in parte: la scatola proiettata sporge con i suoi
    spigoli oltre la cupola, che le e' inscritta.  Restano due fasce bianche sopra
    e sotto che in pagina sarebbero margine pagato a peso d'oro, e che nessun
    parametro di matplotlib toglie senza rimpicciolire anche il disegno.
    """
    from PIL import Image, ImageChops

    im = Image.open(path).convert("RGB")
    box = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))
                                ).getbbox()
    if box is None:                       # immagine tutta bianca: niente da fare
        return
    left, top, right, bottom = box
    im.crop((max(left - pad, 0), max(top - pad, 0),
             min(right + pad, im.width), min(bottom + pad, im.height))).save(path)


def figure(case: dict, out: Path) -> None:
    tm = make_tonemap(case)
    fig = plt.figure(figsize=(7.0, 5.0))

    # Assi piu' alti della figura stessa: la scatola 3D e' larga e bassa
    # (box_aspect 2.1 x 2.1 x 1.1) e matplotlib la fa stare tutta dentro il
    # riquadro, lasciando due fasce vuote sopra e sotto il contenuto.  Sono margine
    # interno all'asse, quindi `bbox_inches="tight"` non le vede e resterebbero
    # dentro la figura in pagina; sovradimensionare l'asse se le mangia.
    ax = fig.add_axes([0.0, -0.07, 1.0, 1.10], projection="3d")
    panel_geometry(ax, case, tm)

    # Titolo come testo di figura e non dell'asse: gli assi 3D lasciano una fascia
    # vuota sopra il contenuto e un titolo agganciato la trasformerebbe in margine
    # dentro la pagina.
    # Il titolo va tenuto basso, vicino al contenuto: sopra la cupola l'asse 3D ha
    # una fascia vuota alta (gli spigoli della scatola proiettata) e un titolo in
    # cima alla figura resterebbe separato dal disegno anche dopo il ritaglio.
    fig.text(0.5, 0.84, "One hemisphere for the diffuse term, one cone per camera",
             ha="center", fontsize=14, color=C_INK)
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)
    trim_white(out)
    print(f"  + {out}")


def report(case: dict) -> None:
    """Controlli che la figura stia mostrando quello che dice di mostrare."""
    e = case["E"]
    print(f"\n  E            = ({e[0]:.4f}, {e[1]:.4f}, {e[2]:.4f})   "
          f"lum {luminance(e):.4f}")
    print(f"  (a x/pi) E   = " + ", ".join(f"{v:.4f}" for v in case["diffuse"])
          + f"   lum {luminance(case['diffuse']):.4f}")
    for k, c in enumerate(case["cams"], start=1):
        err = float(np.max(np.abs(c["L"] - c["L_check"])) / np.max(c["L_check"]))
        print(f"  L_{k}({THETA:.0f} deg) = " + ", ".join(f"{v:.4f}" for v in c["L"])
              + f"   lum {luminance(c['L']):.4f}   ({c['n_rays']} raggi, "
                f"cono {'intero' if c['unclipped'] else 'TRONCATO'}, "
                f"scarto dal campionamento diretto {err:.2e})")
        assert c["unclipped"], "il cono esce dall'orizzonte: media troncata"
        assert err < 1e-2, "la chiusura del bake non torna col cono campionato"
    c1, c2 = (c["C"] for c in case["cams"])
    print(f"  C_1          = " + ", ".join(f"{v:.4f}" for v in c1)
          + f"   lum {luminance(c1):.4f}")
    print(f"  C_2          = " + ", ".join(f"{v:.4f}" for v in c2)
          + f"   lum {luminance(c2):.4f}")
    ratio = luminance(c1) / luminance(c2)
    print(f"  C_1 / C_2 (lum) = {ratio:.2f}   "
          f"(tutto lo scarto viene dal termine speculare)")
    assert ratio > 1.3, "le due camere registrano quasi lo stesso colore: " \
                        "la figura perde il suo punto, spostare le camere o il sole"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--figures", default="model,kernel",
                    help="quali figure scrivere, separate da virgola: model, kernel")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    want = {f.strip() for f in args.figures.split(",") if f.strip()}
    if not want <= {"model", "kernel"}:
        ap.error(f"figure sconosciute: {sorted(want - {'model', 'kernel'})}")

    if "model" in want:
        case = build_case()
        figure(case, out / "pbr_model.png")
        report(case)
    if "kernel" in want:
        kcase = build_kernel_case()
        figure_kernel(kcase, out / "tophat_vs_ggx.png")
        report_kernel(kcase)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
