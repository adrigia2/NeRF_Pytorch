#!/usr/bin/env python
"""make_geometry_diagrams.py -- Schemi geometrici del capitolo Implementation.

    python make_geometry_diagrams.py --out ../Doc/images/diagrams
    python make_geometry_diagrams.py --out DIR --only irradiance

Scrive cinque PNG, uno per figura della tesi:

  depth_ray.png                 3.4   camera, griglia di raggi dal FOV, o / d / t
  ium_trick.png                 3.6   il trucco dell'inverse UV mapping
  grazing_cull.png              3.9   scarto dei contributi ad angolo radente
  color_texture_projection.png  3.10  proiezione texel -> pixel attraverso il near plane
  irradiance_hemisphere.png     3.11  emisfero di Fibonacci, diretta contro indiretta

Due scelte che vale la pena conoscere prima di metterci le mani.

**Le intersezioni sono calcolate, non disegnate.**  Ogni raggio viene davvero intersecato
con la geometria (piano, box, sfera) e il punto di impatto e' quello che esce dal conto.
Un raggio che nella figura sembra colpire, colpisce; e cambiare un parametro in testa allo
script non produce una figura che mente.

**La scena e' 3D ma il disegno e' 2D.**  Si proietta a mano in ortografica (`Projector`) e
si disegna su assi normali, invece di usare mplot3d.  Con mplot3d l'inquadratura la decide
la libreria a partire dal cubo dei dati e la geometria finisce minuscola in mezzo al vuoto,
l'ordine di disegno non e' controllabile e le etichette ballano.  Qui l'ordine e' quello
delle chiamate (painter), i limiti si ricavano dai punti proiettati e l'inquadratura e'
sempre giusta.

Le direzioni della figura 3.11 usano la stessa formula del kernel
`deviceProgramsIrradiance.cu`: cos(theta) equispaziato e azimut sulla sequenza aurea come
frazione di giro, (3-sqrt(5))/2, ridotta in [0,1) prima della trigonometria.  Non e'
l'unica sequenza di Fibonacci della pipeline: HemiVis usa 1/phi, cioe' il verso opposto
(vedi `_hemivis_directions` in images_generator.py).  Qui serve quella dell'irradiance,
perche' e' il pass che la figura illustra.
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

# ── Palette: lo stesso concetto ha lo stesso colore in tutte le figure ───────
C_GEOM   = "#b9c3cd"   # geometria della scena
C_EDGE   = "#5f6c79"   # spigoli
C_RAY    = "#5b9bd5"   # raggio generico
C_HILITE = "#d62728"   # elemento evidenziato
C_ACCEPT = "#2ca02c"   # contributo accettato
C_REJECT = "#d62728"   # contributo scartato
C_DIRECT = "#e8a33d"   # irradiance diretta (cielo)
C_INDIR  = "#8452c9"   # irradiance indiretta (rimbalzo)
C_CAM    = "#33404d"   # corpo camera

# ── Parametri: tutto cio' che vale la pena ritoccare sta qui ─────────────────
# Nota sull'azimut di `view`: va tenuto a circa 90 gradi dall'azimut dell'asse della
# camera della scena.  Se i due sono allineati (o opposti) si guarda il frustum di
# infilata, il ventaglio di raggi collassa in una riga e la figura diventa illeggibile.
FIG_DEPTH_RAY = dict(fov_deg=34.0, grid=(5, 4), cam=(0.0, -1.70, 0.78),
                     look=(0.0, 0.0, 0.25), hilite=(2, 1), ground_half=1.05,
                     miss_len=0.85, view=(16.0, -20.0), size=(7.6, 4.6))
FIG_IUM       = dict(grid=(11, 11), z_start=0.95, z_end=-0.55,
                     view=(24.0, -60.0), size=(7.6, 5.4))
FIG_GRAZING   = dict(threshold_deg=75.0, theta_ok=20.0, theta_no=82.0,
                     view=(14.0, -62.0), size=(7.4, 4.6))
FIG_PROJ      = dict(fov_deg=36.0, near=0.62, cam=(1.30, -1.85, 1.15),
                     view=(20.0, 35.0), size=(7.8, 5.0))
FIG_IRR       = dict(n_samples=72, sphere_c=(0.62, 0.05, 0.92), sphere_r=0.50,
                     view=(16.0, -62.0), size=(7.4, 5.0))


# ── Proiezione ortografica ───────────────────────────────────────────────────

class Projector:
    """Mondo 3D -> piano, in ortografica.  `depth` serve per l'ordinamento painter."""

    def __init__(self, elev_deg: float, azim_deg: float):
        e, a = np.radians(elev_deg), np.radians(azim_deg)
        # v punta dalla scena verso l'osservatore
        self.v = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        self.right = np.array([-np.sin(a), np.cos(a), 0.0])
        self.up = np.cross(self.v, self.right)

    def __call__(self, p) -> np.ndarray:
        p = np.atleast_2d(np.asarray(p, float))
        out = np.stack([p @ self.right, p @ self.up], axis=-1)
        return out[0] if out.shape[0] == 1 else out

    def depth(self, p) -> float:
        return float(np.asarray(p, float).reshape(-1, 3).mean(axis=0) @ self.v)


class Scene:
    """Accumula i punti disegnati per poter incorniciare esattamente alla fine."""

    def __init__(self, ax, proj: Projector):
        self.ax, self.proj, self.pts = ax, proj, []

    def _add(self, xy):
        self.pts.append(np.atleast_2d(xy))

    def poly(self, corners, color, alpha=0.55, edge=C_EDGE, lw=1.0, z=1):
        xy = self.proj(np.asarray(corners, float))
        self.ax.fill(xy[:, 0], xy[:, 1], color=color, alpha=alpha, zorder=z,
                     edgecolor=edge, linewidth=lw)
        self._add(xy)

    def line(self, a, b, color, lw=1.5, ls="-", alpha=1.0, z=3):
        xy = self.proj(np.array([a, b], float))
        self.ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw, ls=ls, alpha=alpha, zorder=z)
        self._add(xy)

    def curve(self, pts3, color, lw=1.2, ls="-", alpha=1.0, z=3):
        xy = self.proj(np.asarray(pts3, float))
        self.ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw, ls=ls, alpha=alpha, zorder=z)
        self._add(xy)

    def arrow(self, o, d, color, lw=1.8, z=6):
        o = np.asarray(o, float)
        a, b = self.proj(o), self.proj(o + np.asarray(d, float))
        self.ax.annotate("", xy=b, xytext=a, zorder=z,
                         arrowprops=dict(arrowstyle="-|>,head_width=0.28,head_length=0.55",
                                         color=color, lw=lw, shrinkA=0, shrinkB=0))
        self._add(np.array([a, b]))

    def dot(self, p, color, s=38, z=8, edge="white", lw=0.8):
        xy = self.proj(np.asarray(p, float))
        self.ax.scatter([xy[0]], [xy[1]], s=s, color=color, zorder=z,
                        edgecolors=edge, linewidths=lw)
        self._add(xy[None, :])

    def text(self, p, s, color="black", dxy=(0, 0), size=None, weight="normal",
             ha="center", va="center", z=12):
        xy = self.proj(np.asarray(p, float))
        self.ax.text(xy[0] + dxy[0], xy[1] + dxy[1], s, color=color, fontsize=size,
                     fontweight=weight, ha=ha, va=va, zorder=z)
        self._add(np.array([[xy[0] + dxy[0], xy[1] + dxy[1]]]))

    def finish(self, pad=0.10):
        pts = np.vstack(self.pts)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        c, r = 0.5 * (lo + hi), 0.5 * (hi - lo).max() * (1.0 + pad)
        self.ax.set_xlim(c[0] - r, c[0] + r)
        self.ax.set_ylim(c[1] - r, c[1] + r)
        self.ax.set_aspect("equal")
        self.ax.axis("off")


def new_scene(view, size):
    fig, ax = plt.subplots(figsize=size)
    return fig, ax, Scene(ax, Projector(*view))


def save(fig, sc, out: Path, note: str = "") -> None:
    sc.finish()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}{('  ' + note) if note else ''}")


# ── Geometria: intersezioni vere ─────────────────────────────────────────────

def hit_plane(o, d, z=0.0, half=None):
    """Intersezione col piano z=cost.  `half` lo limita al quadrato effettivamente
    disegnato: senza, un raggio che atterra fuori dal terreno visibile risulta comunque
    un hit a distanza enorme, e la figura mostra un segmento che finisce nel nulla."""
    o, d = np.asarray(o, float), np.asarray(d, float)
    if abs(d[2]) < 1e-9:
        return np.inf
    t = (z - o[2]) / d[2]
    if t <= 1e-6:
        return np.inf
    if half is not None:
        p = o + d * t
        if abs(p[0]) > half or abs(p[1]) > half:
            return np.inf
    return t


def hit_box(o, d, lo, hi):
    """Slab method: t di ingresso, inf se il raggio manca il box."""
    o, d = np.asarray(o, float), np.asarray(d, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (np.asarray(lo, float) - o) / d, (np.asarray(hi, float) - o) / d
    tmin = np.nanmax(np.minimum(t1, t2))
    tmax = np.nanmin(np.maximum(t1, t2))
    if tmax < max(tmin, 0.0):
        return np.inf
    return tmin if tmin > 1e-6 else np.inf


def hit_sphere(o, d, c, r):
    o, d, c = np.asarray(o, float), np.asarray(d, float), np.asarray(c, float)
    m = o - c
    b, cc = 2.0 * m @ d, m @ m - r * r
    disc = b * b - 4.0 * (d @ d) * cc
    if disc < 0:
        return np.inf
    s = np.sqrt(disc)
    for t in ((-b - s) / (2 * d @ d), (-b + s) / (2 * d @ d)):
        if t > 1e-6:
            return t
    return np.inf


def ground(sc: Scene, half=1.5, z=0.0, color=C_GEOM, alpha=0.30):
    sc.poly([(-half, -half, z), (half, -half, z), (half, half, z), (-half, half, z)],
            color, alpha=alpha, edge="#96a3b0", lw=0.9, z=0)


def draw_box(sc: Scene, lo, hi, color=C_GEOM):
    """Le tre facce visibili, ordinate per profondita' cosi' non si sovrappongono male."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    faces = [
        [(lo[0], lo[1], hi[2]), (hi[0], lo[1], hi[2]), (hi[0], hi[1], hi[2]), (lo[0], hi[1], hi[2])],
        [(lo[0], lo[1], lo[2]), (hi[0], lo[1], lo[2]), (hi[0], lo[1], hi[2]), (lo[0], lo[1], hi[2])],
        [(hi[0], lo[1], lo[2]), (hi[0], hi[1], lo[2]), (hi[0], hi[1], hi[2]), (hi[0], lo[1], hi[2])],
        [(lo[0], lo[1], lo[2]), (lo[0], hi[1], lo[2]), (lo[0], hi[1], hi[2]), (lo[0], lo[1], hi[2])],
        [(lo[0], hi[1], lo[2]), (hi[0], hi[1], lo[2]), (hi[0], hi[1], hi[2]), (lo[0], hi[1], hi[2])],
    ]
    for f in sorted(faces, key=sc.proj.depth):
        sc.poly(f, color, alpha=0.92, edge=C_EDGE, lw=1.0, z=2)


def camera_basis(pos, look, up=(0, 0, 1)):
    pos, look = np.asarray(pos, float), np.asarray(look, float)
    f = look - pos
    f /= np.linalg.norm(f)
    r = np.cross(f, np.asarray(up, float))
    r /= np.linalg.norm(r)
    return f, r, np.cross(r, f)


def image_rect(pos, f, r, u, dist, fov_deg, aspect=1.5):
    h = np.tan(np.radians(fov_deg) * 0.5) * dist
    w = h * aspect
    c = np.asarray(pos, float) + f * dist
    return [c - r * w - u * h, c + r * w - u * h, c + r * w + u * h, c - r * w + u * h]


def draw_camera(sc: Scene, pos, f, r, u, size=0.13, z=7):
    """Piramide che identifica la camera senza rubare la scena."""
    c = np.asarray(pos, float)
    back = [c + (r * sx + u * sy) * size - f * size * 0.9
            for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
    sc.poly(back, C_CAM, alpha=0.85, edge=C_CAM, lw=0.8, z=z)
    for b in back:
        sc.line(c, b, C_CAM, lw=1.0, z=z)
    sc.dot(c, C_CAM, s=26, z=z + 1)


# ── 3.4 -- Il raggio del depth pass ──────────────────────────────────────────

def fig_depth_ray(out: Path) -> None:
    p = FIG_DEPTH_RAY
    cam = np.array(p["cam"], float)
    f, r, u = camera_basis(cam, p["look"])
    nx, ny = p["grid"]
    box_lo, box_hi = np.array([-0.38, -0.32, 0.0]), np.array([0.38, 0.32, 0.58])

    fig, ax, sc = new_scene(p["view"], p["size"])
    ground(sc, half=p["ground_half"])
    draw_box(sc, box_lo, box_hi)
    draw_camera(sc, cam, f, r, u)

    h = np.tan(np.radians(p["fov_deg"]) * 0.5)

    def direction(i, j):
        sx = (2.0 * (i + 0.5) / nx - 1.0) * h * 1.5
        sy = (2.0 * (j + 0.5) / ny - 1.0) * h
        d = f + r * sx + u * sy
        return d / np.linalg.norm(d)          # unitaria: convenzione della pipeline

    for j in range(ny):
        for i in range(nx):
            if (i, j) == tuple(p["hilite"]):
                continue
            d = direction(i, j)
            t = min(hit_plane(cam, d, half=p["ground_half"]),
                    hit_box(cam, d, box_lo, box_hi))
            if not np.isfinite(t):
                sc.line(cam, cam + d * p["miss_len"], C_RAY, lw=0.7, alpha=0.30, z=1)
            else:
                sc.line(cam, cam + d * t, C_RAY, lw=0.8, alpha=0.55, z=4)
                sc.dot(cam + d * t, C_RAY, s=8, z=5, edge="none")

    d = direction(*p["hilite"])
    t = min(hit_plane(cam, d, half=p["ground_half"]),
            hit_box(cam, d, box_lo, box_hi))
    hit = cam + d * t
    sc.line(cam, hit, C_HILITE, lw=2.6, z=9)
    sc.arrow(cam, d, C_HILITE, lw=2.2, z=10)
    sc.dot(cam, C_HILITE, s=60, z=11)
    sc.dot(hit, C_HILITE, s=60, z=11)
    sc.text(cam, r"$\mathbf{o}$", C_HILITE, dxy=(-0.13, 0.06), weight="bold", size=15)
    sc.text(cam + d * 0.55, r"$\mathbf{d}$", C_HILITE, dxy=(0.0, 0.14), weight="bold", size=15)
    sc.text(cam + d * (t * 0.80), r"$t$", C_HILITE, dxy=(0.0, -0.13), weight="bold", size=15)
    sc.text(hit, r"$\mathbf{o}+t\,\mathbf{d}$", C_HILITE, dxy=(0.10, -0.16),
            size=12, ha="left")

    ax.legend(handles=[
        Line2D([], [], color=C_HILITE, lw=2.4, label="the traced ray"),
        Line2D([], [], color=C_RAY, lw=1.2, alpha=0.6, label="one ray per pixel, same origin"),
    ], loc="upper left", frameon=False, fontsize=11)
    save(fig, sc, out)


# ── 3.6 -- Il trucco dell'inverse UV mapping ─────────────────────────────────

def _uv_islands():
    """Due isole appena irregolari: un atlante vero non e' un rettangolo, ed e' proprio
    il fatto che parte del dominio resti vuota a rendere utile la maschera."""
    return [np.array([(-0.82, -0.72), (-0.06, -0.74), (0.10, -0.10),
                      (-0.34, 0.30), (-0.88, -0.04)]),
            np.array([(0.20, 0.08), (0.84, 0.00), (0.88, 0.62),
                      (0.44, 0.82), (0.16, 0.50)])]


def _inside(poly, pt):
    """Punto in poligono, ray casting: la stessa domanda che il tracer pone al BVH."""
    x, y = pt
    ins = False
    for k in range(len(poly)):
        x0, y0 = poly[k]
        x1, y1 = poly[(k + 1) % len(poly)]
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            ins = not ins
    return ins


def fig_ium_trick(out: Path) -> None:
    p = FIG_IUM
    islands = _uv_islands()
    fig, ax, sc = new_scene(p["view"], p["size"])

    sc.poly([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)],
            "#eef2f6", alpha=0.95, edge="#9aa7b4", lw=1.1, z=0)
    for isl in islands:
        sc.poly([(x, y, 0.0) for x, y in isl], C_GEOM, alpha=0.95,
                edge=C_EDGE, lw=1.3, z=2)

    nx, ny = p["grid"]
    n_hit = 0
    for j in range(ny):
        for i in range(nx):
            x = -1.0 + 2.0 * (i + 0.5) / nx
            y = -1.0 + 2.0 * (j + 0.5) / ny
            if any(_inside(isl, (x, y)) for isl in islands):
                n_hit += 1
                sc.line((x, y, p["z_start"]), (x, y, 0.0), C_ACCEPT, lw=1.1, alpha=0.9, z=4)
                sc.dot((x, y, 0.0), C_ACCEPT, s=20, z=6, edge="none")
            else:
                sc.line((x, y, p["z_start"]), (x, y, p["z_end"]), C_REJECT, lw=0.8,
                        ls=(0, (2, 2)), alpha=0.45, z=1)

    sc.text((1.0, -1.0, 0.0), "UV domain", "#5f6c79", size=11,
            dxy=(0.22, -0.14), ha="left")

    ax.legend(handles=[
        Line2D([], [], color=C_ACCEPT, lw=1.8, label=r"hit: texel lies on the mesh, mask $=1$"),
        Line2D([], [], color=C_REJECT, lw=1.4, ls=(0, (2, 2)), label=r"miss: mask $=0$"),
    ], loc="upper left", frameon=False, fontsize=11)
    save(fig, sc, out, f"({n_hit}/{nx * ny} raggi colpiscono)")


# ── 3.9 -- Scarto ad angolo radente ──────────────────────────────────────────

def fig_grazing_cull(out: Path) -> None:
    p = FIG_GRAZING
    fig, ax, sc = new_scene(p["view"], p["size"])
    ground(sc, half=1.25)
    P = np.zeros(3)

    th = np.radians(p["threshold_deg"])
    sc.curve([(np.sin(th) * np.cos(a), np.sin(th) * np.sin(a), np.cos(th))
              for a in np.linspace(0, 2 * np.pi, 120)],
             "#8894a1", lw=1.0, ls=(0, (3, 3)), z=2)

    sc.arrow(P, np.array([0, 0, 1.15]), C_EDGE, lw=1.6, z=5)
    sc.text(P + np.array([0, 0, 1.15]), r"$\mathbf{n}$", "#3c4750",
            dxy=(-0.13, 0.05), weight="bold", size=15)

    for theta_deg, color, ls, phi_deg in ((p["theta_ok"], C_ACCEPT, "-", 40.0),
                                          (p["theta_no"], C_REJECT, (0, (4, 3)), 205.0)):
        t, phi = np.radians(theta_deg), np.radians(phi_deg)
        d = np.array([np.sin(t) * np.cos(phi), np.sin(t) * np.sin(phi), np.cos(t)])
        cam = P + d * 1.5
        f, r, u = camera_basis(cam, P)
        draw_camera(sc, cam, f, r, u, size=0.10)
        sc.line(P, cam, color, lw=2.2, ls=ls, z=6)
        sc.text(P + d * 0.72, rf"${theta_deg:.0f}^\circ$", color,
                dxy=(0.13, 0.10), weight="bold", size=13)
    sc.dot(P, "black", s=46, z=9)

    sc.text((np.sin(th), 0.0, np.cos(th)), rf"cull beyond ${p['threshold_deg']:.0f}^\circ$",
            "#5f6c79", size=11, dxy=(0.34, -0.10))

    ax.legend(handles=[
        Line2D([], [], color=C_ACCEPT, lw=2.2, label="near-normal view: contributes"),
        Line2D([], [], color=C_REJECT, lw=2.2, ls=(0, (4, 3)), label="grazing view: discarded"),
    ], loc="upper left", frameon=False, fontsize=11)
    save(fig, sc, out)


# ── 3.10 -- Proiezione del texel nel pixel ───────────────────────────────────

def fig_color_texture(out: Path) -> None:
    p = FIG_PROJ
    cam = np.array(p["cam"], float)
    f, r, u = camera_basis(cam, (0.0, 0.0, 0.10))
    rect = np.array(image_rect(cam, f, r, u, p["near"], p["fov_deg"]), float)
    c_near = rect.mean(axis=0)
    h = np.tan(np.radians(p["fov_deg"]) * 0.5) * p["near"]
    w = h * 1.5

    fig, ax, sc = new_scene(p["view"], p["size"])
    ground(sc, half=1.15)

    def pixel_of(P):
        """Dove il segmento P->camera buca il piano immagine, e se ci cade dentro.
        E' la domanda del kernel: il texel finisce in un pixel, o in nessuno."""
        d = cam - np.asarray(P, float)
        den = d @ f
        if abs(den) < 1e-9:
            return None
        X = np.asarray(P, float) + d * ((c_near - P) @ f / den)
        loc = X - c_near
        return X, bool(abs(loc @ r) <= w and abs(loc @ u) <= h)

    for P, color, tag in ((np.array([-0.05, 0.12, 0.0]), C_ACCEPT, "seen"),
                          (np.array([-1.00, -0.92, 0.0]), C_REJECT, "unseen")):
        X, ok = pixel_of(P)
        sc.dot(P, color, s=52, z=9)
        if ok:
            sc.line(P, X, color, lw=2.2, z=6)
            sc.line(X, cam, color, lw=1.1, ls=(0, (2, 2)), alpha=0.85, z=6)
            sc.dot(X, color, s=44, z=11)
            sc.text(X, "pixel", color, dxy=(0.02, 0.20), size=11, weight="bold")
        else:
            sc.line(P, cam, color, lw=1.8, ls=(0, (4, 3)), alpha=0.9, z=6)
            sc.text(P, "outside the frustum", color, dxy=(-0.05, -0.22), size=11)

    for k in range(4):
        sc.line(cam, rect[k], "#8894a1", lw=0.9, alpha=0.9, z=5)
    sc.poly(rect, "#cfd8e2", alpha=0.55, edge="#5f6c79", lw=1.3, z=7)
    draw_camera(sc, cam, f, r, u, size=0.11, z=8)
    sc.text(c_near, "near plane", "#3c4750", dxy=(0.0, 0.30), size=11)

    ax.legend(handles=[
        Line2D([], [], color=C_ACCEPT, lw=2.2, label="texel projects into a pixel"),
        Line2D([], [], color=C_REJECT, lw=2.0, ls=(0, (4, 3)), label="texel not seen"),
    ], loc="upper left", frameon=False, fontsize=11)
    save(fig, sc, out)


# ── 3.11 -- Emisfero di Fibonacci, diretta contro indiretta ──────────────────

IRR_GOLDEN_TURN = 0.3819660112501051518   # (3 - sqrt(5))/2, come nel kernel
TWO_PI = 6.283185307179586477


def irradiance_directions(n_samples: int) -> np.ndarray:
    """Le direzioni del kernel irradiance attorno a +z, uniformi in angolo solido.

    Identica a `__raygen__renderIrradiance`: z = 1 - (i+0.5)/S e azimut sulla sequenza
    aurea ridotta in [0,1) PRIMA della trigonometria.  La riduzione anticipata e' il fix
    del 13/08/2026: senza, in float32 e a S grande l'azimut perde la bassa discrepanza.
    """
    i = np.arange(n_samples)
    z = 1.0 - (i + 0.5) / n_samples
    rad = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    x = i * IRR_GOLDEN_TURN
    phi = (x - np.floor(x)) * TWO_PI
    return np.stack([rad * np.cos(phi), rad * np.sin(phi), z], axis=-1)


def fig_irradiance(out: Path) -> None:
    p = FIG_IRR
    sc_c, sr = np.array(p["sphere_c"], float), p["sphere_r"]
    P = np.zeros(3)
    dirs = irradiance_directions(p["n_samples"])

    fig, ax, sc = new_scene(p["view"], p["size"])
    ground(sc, half=1.30)

    n_ind = 0
    for d in dirs:
        t = hit_sphere(P, d, sc_c, sr)
        if np.isfinite(t):
            n_ind += 1
        else:
            sc.line(P, P + d * 1.30, C_DIRECT, lw=0.9, alpha=0.60, z=1)

    # La sfera occludente, sopra i raggi liberi e sotto quelli occlusi
    circ = np.linspace(0, 2 * np.pi, 90)
    disc = [sc_c + sr * (np.cos(a) * sc.proj.right + np.sin(a) * sc.proj.up) for a in circ]
    sc.poly(disc, C_GEOM, alpha=0.97, edge=C_EDGE, lw=1.1, z=3)

    for d in dirs:
        t = hit_sphere(P, d, sc_c, sr)
        if np.isfinite(t):
            sc.line(P, P + d * t, C_INDIR, lw=1.3, alpha=0.95, z=5)
            sc.dot(P + d * t, C_INDIR, s=13, z=6, edge="none")

    sc.arrow(P, np.array([0, 0, 0.62]), C_EDGE, lw=1.6, z=7)
    sc.text(P + np.array([0, 0, 0.62]), r"$\mathbf{n}$", "#3c4750",
            dxy=(-0.13, 0.04), weight="bold", size=15)
    sc.dot(P, "black", s=46, z=9)

    ax.legend(handles=[
        Line2D([], [], color=C_DIRECT, lw=1.8,
               label="unoccluded: direct irradiance from the environment map"),
        Line2D([], [], color=C_INDIR, lw=1.8,
               label="blocked: indirect, radiance queried from the NeRF"),
    ], loc="upper left", frameon=False, fontsize=11)
    save(fig, sc, out, f"({n_ind}/{len(dirs)} raggi occlusi)")


# ── main ─────────────────────────────────────────────────────────────────────

FIGURES = {
    "depth":      ("depth_ray.png",                fig_depth_ray),
    "ium":        ("ium_trick.png",                fig_ium_trick),
    "grazing":    ("grazing_cull.png",             fig_grazing_cull),
    "projection": ("color_texture_projection.png", fig_color_texture),
    "irradiance": ("irradiance_hemisphere.png",    fig_irradiance),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="../Doc/images/diagrams",
                    help="cartella di destinazione dei PNG")
    ap.add_argument("--only", nargs="+", choices=sorted(FIGURES),
                    help="genera solo queste figure")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Figure geometriche → {out.resolve()}")
    for key in (args.only or sorted(FIGURES)):
        name, fn = FIGURES[key]
        fn(out / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
