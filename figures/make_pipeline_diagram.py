#!/usr/bin/env python
"""make_pipeline_diagram.py -- Schema a blocchi della pipeline (figura 3.1).

    python make_pipeline_diagram.py --out ../Doc/images/diagrams

Scrive `pipeline_overview.png`.

Lo schema e' DATI, non disegno: si modifica editando `NODES` e `EDGES` qui sotto, e il
layout viene fuori da solo.  Ogni nodo dichiara la colonna e la riga in cui sta, la
larghezza in colonne, e a quale gruppo appartiene (il gruppo decide solo il colore).  Le
frecce sono coppie di identificatori: non ci sono coordinate scritte a mano da nessuna
parte, quindi spostare un blocco non richiede di risistemare le frecce.

Le colonne sono equispaziate e le righe pure; se un blocco risulta stretto per il suo
testo, la cosa da cambiare e' `COL_W` o il testo, non la posizione.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({"font.size": 13})

DPI = 190

# ── Aspetto ──────────────────────────────────────────────────────────────────
# Le misure sono in POLLICI e la figura viene dimensionata perche' una unita' dati valga
# esattamente un pollice.  Serve a poter ragionare sulla larghezza di una scatola in
# punti tipografici: con unita' arbitrarie il testo sborda appena si cambia figsize, ed
# e' esattamente cio' che succedeva prima.
COL_W, COL_GAP = 2.05, 0.62      # larghezza di una colonna e spazio fra colonne
ROW_H, ROW_GAP = 0.88, 0.34      # altezza di una riga e spazio fra righe
FONT = 11                        # a 11pt una riga di ~20 caratteri sta in COL_W
MARGIN = 0.22

STYLE = {
    "input":  dict(fc="#eef2f6", ec="#8a99a8", tc="#25303a"),
    "nerf":   dict(fc="#e7dcf7", ec="#8452c9", tc="#3d2a63"),
    "optix":  dict(fc="#d9e8f7", ec="#2f7ec4", tc="#123a5c"),
    "fit":    dict(fc="#fdeacd", ec="#d9932a", tc="#5c3d0c"),
    "output": dict(fc="#dcf0dc", ec="#2ca02c", tc="#12470f"),
}

# id, colonna, riga, larghezza in colonne, gruppo, testo
NODES = [
    ("img",   0, 2.0, 1, "input",  "Multi-view images\nwith known poses"),
    ("mesh",  0, 1.0, 1, "input",  "Scene mesh\n(UV-unwrapped)"),
    ("nmap",  0, 0.0, 1, "input",  "Object-space\nnormal map"),

    ("nerf",  1, 2.0, 1, "nerf",   "NeRF training\n(Step 2)"),
    ("optix", 1, 0.5, 1, "optix",  "OptiX passes\n(Steps 1 and 3)"),

    ("rad",   2, 2.0, 1, "nerf",   "Radiance queries\n(indirect, sky)"),
    ("geo",   2, 1.0, 1, "optix",  "Geometry, visibility\nand color"),
    ("light", 2, 0.0, 1, "optix",  "Irradiance and\nspecular cones"),

    ("fit",   3, 1.0, 1, "fit",    "PBR fit\n(Step 4)"),

    ("base",  4, 2.0, 1, "output", "Base color"),
    ("met",   4, 1.0, 1, "output", "Metallic"),
    ("rough", 4, 0.0, 1, "output", "Roughness"),
]

EDGES = [
    ("img", "nerf"), ("mesh", "optix"), ("nmap", "optix"), ("img", "optix"),
    ("nerf", "rad"), ("optix", "geo"), ("optix", "light"), ("rad", "light"),
    ("rad", "fit"), ("geo", "fit"), ("light", "fit"),
    ("fit", "base"), ("fit", "met"), ("fit", "rough"),
]


def box_rect(col, row, span):
    x = col * (COL_W + COL_GAP)
    y = row * (ROW_H + ROW_GAP)
    return x, y, COL_W * span + COL_GAP * (span - 1), ROW_H


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="../Doc/images/diagrams")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rects = {}
    for nid, col, row, span, group, text in NODES:
        rects[nid] = (*box_rect(col, row, span), group, text)

    xs = [r[0] for r in rects.values()] + [r[0] + r[2] for r in rects.values()]
    ys = [r[1] for r in rects.values()] + [r[1] + r[3] for r in rects.values()]
    w, h = max(xs) - min(xs), max(ys) - min(ys)

    # Una unita' dati = un pollice: figsize e limiti sono coerenti per costruzione
    fig, ax = plt.subplots(figsize=(w + 2 * MARGIN, h + 2 * MARGIN))

    for a, b in EDGES:
        xa, ya, wa, ha, _, _ = rects[a]
        xb, yb, wb, hb, _, _ = rects[b]
        if abs(xa - xb) < 1e-9:
            # Stessa colonna: l'arco va verticale, altrimenti esce a destra e rientra
            # a sinistra facendo il giro attorno al blocco.
            up = yb > ya
            p0 = (xa + wa / 2, ya + (ha if up else 0.0))
            p1 = (xb + wb / 2, yb + (0.0 if up else hb))
            # Curvatura marcata: un connettore verticale dritto passerebbe dietro il
            # blocco interposto e si leggerebbe come due archi separati.
            rad = 0.55
        else:
            p0, p1, rad = (xa + wa, ya + ha / 2), (xb, yb + hb / 2), 0.06
        ax.add_patch(FancyArrowPatch(
            p0, p1, arrowstyle="-|>,head_width=0.18,head_length=0.36",
            connectionstyle=f"arc3,rad={rad}", color="#7c8894", lw=1.3,
            shrinkA=2, shrinkB=3, zorder=1))

    for nid, (x, y, bw, bh, group, text) in rects.items():
        s = STYLE[group]
        ax.add_patch(FancyBboxPatch(
            (x, y), bw, bh, boxstyle="round,pad=0.0,rounding_size=0.12",
            facecolor=s["fc"], edgecolor=s["ec"], linewidth=1.6, zorder=2))
        ax.text(x + bw / 2, y + bh / 2, text, ha="center", va="center",
                color=s["tc"], fontsize=FONT, zorder=3, linespacing=1.35)

    ax.set_xlim(min(xs) - MARGIN, max(xs) + MARGIN)
    ax.set_ylim(min(ys) - MARGIN, max(ys) + MARGIN)
    ax.set_aspect("equal")
    ax.axis("off")
    path = out / "pipeline_overview.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}  ({len(NODES)} blocchi, {len(EDGES)} archi)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
