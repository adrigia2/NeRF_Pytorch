"""Toy numerico per verificare l'identificabilita' del sistema PBR multi-vista.

Modello (split-sum, workflow metallic, per texel, camera j):

    C_j = (1 - X) * (d / pi) * E  +  F_j * L_j(r)

    F_j  = F0 + (1 - F0) * (1 - cos(theta_j))^5        (Schlick)
    F0   = 0.04 * (1 - X) + d * X                       (accoppiamento spettrale)
    E    = irradianza coseno-pesata sull'emisfero (vista-indipendente)
    L_j(r) = radianza ambiente prefiltrata GGX attorno a reflect(v_j, n)
             (nel pipeline reale: query NeRF cone-traced, qui: envmap analitica + MC)

Incognite per texel: d (3), X (1), r (1).
Strategia: scan su griglia (X, r) + minimi quadrati chiusi su d (lineare!),
con L(r) interpolata da K livelli precalcolati (la "catena di mip").

Lo script verifica anche la degenerazione del modello naive
    C_j = X*d*E + (1-X)*s*L_j(r)
in cui X non e' identificabile (si assorbe in d ed s).
"""

import numpy as np

rng = np.random.default_rng(42)

# ---------------------------------------------------------------- environment


def _normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


SUN_DIR = _normalize(np.array([0.4, 0.3, 0.85]))
WALL_DIR = _normalize(np.array([-0.8, 0.4, 0.1]))
ZENITH = np.array([0.20, 0.45, 0.95])
HORIZON = np.array([0.55, 0.60, 0.70])
SUN_COLOR = np.array([60.0, 55.0, 40.0])
WALL_COLOR = np.array([3.0, 0.6, 0.3])


def env_radiance(w):
    """Radianza ambiente analitica, w: (...,3) -> (...,3). Colorata e
    angolarmente variata (sole caldo + parete rossa) cosi' direzioni di
    riflessione diverse vedono colori diversi."""
    t = 0.5 * (w[..., 2:3] + 1.0)
    sky = (1.0 - t) * HORIZON + t * ZENITH
    sun = SUN_COLOR * np.exp((np.sum(w * SUN_DIR, -1, keepdims=True) - 1.0) / 0.02)
    wall = WALL_COLOR * np.exp((np.sum(w * WALL_DIR, -1, keepdims=True) - 1.0) / 0.15)
    return sky + sun + wall


N_NORMAL = np.array([0.0, 0.0, 1.0])  # texel con normale +Z (Z-up come nel progetto)


def irradiance(n_samples=200_000):
    """E = integrale di L*cos sull'emisfero. Campionamento coseno: pdf=cos/pi
    quindi E = pi * media(L)."""
    u1, u2 = rng.random(n_samples), rng.random(n_samples)
    z = np.sqrt(u1)
    rxy = np.sqrt(1.0 - u1)
    phi = 2.0 * np.pi * u2
    w = np.stack([rxy * np.cos(phi), rxy * np.sin(phi), z], -1)
    return np.pi * env_radiance(w).mean(0)


def _frame(d):
    """Base ortonormale con terzo asse d."""
    a = np.array([1.0, 0.0, 0.0]) if abs(d[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t = _normalize(np.cross(a, d))
    b = np.cross(d, t)
    return t, b, d


def prefiltered_env(R, rough, n_samples=8192):
    """L(R, r): prefiltro split-sum con campionamento GGX (assunzione n=v=R).
    Nel pipeline reale questa e' la query NeRF con cono di apertura ~r."""
    if rough < 1e-3:
        return env_radiance(R)
    alpha = rough * rough
    u1, u2 = rng.random(n_samples), rng.random(n_samples)
    cos_h = np.sqrt((1.0 - u1) / (1.0 + (alpha * alpha - 1.0) * u1))
    sin_h = np.sqrt(np.maximum(0.0, 1.0 - cos_h * cos_h))
    phi = 2.0 * np.pi * u2
    t, b, d = _frame(R)
    h = (sin_h[:, None] * np.cos(phi)[:, None] * t
         + sin_h[:, None] * np.sin(phi)[:, None] * b
         + cos_h[:, None] * d)
    l = 2.0 * np.sum(R * h, -1, keepdims=True) * h - R
    w = np.maximum(np.sum(l * R, -1), 0.0)
    valid = w > 0
    return (env_radiance(l[valid]) * w[valid, None]).sum(0) / w[valid].sum()


# ------------------------------------------------------------------- cameras


def make_cameras(n):
    """n direzioni di vista sull'emisfero superiore (spirale di Fibonacci,
    elevazioni da ~radente a ~zenitale)."""
    i = np.arange(n)
    z = 0.15 + 0.8 * (i + 0.5) / n
    rxy = np.sqrt(1.0 - z * z)
    phi = i * np.pi * (3.0 - np.sqrt(5.0))
    return np.stack([rxy * np.cos(phi), rxy * np.sin(phi), z], -1)


# ------------------------------------------------------------- forward model


def schlick_g(cos_t):
    return (1.0 - cos_t) ** 5


def render(d, X, cos_t, E, L):
    """C_j per tutte le camere. d:(3,) X:scalar cos_t:(N,) E:(3,) L:(N,3)."""
    F0 = 0.04 * (1.0 - X) + d * X
    g = schlick_g(cos_t)[:, None]
    F = F0 * (1.0 - g) + g
    return (1.0 - X) * d * E / np.pi + F * L


# -------------------------------------------------------------------- solver

R_LEVELS = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.55, 0.7, 0.85, 1.0])


def lerp_levels(L_levels, r):
    """Interpola la catena di livelli L(r_k) -> L(r). L_levels: (K, N, 3)."""
    k = np.clip(np.searchsorted(R_LEVELS, r) - 1, 0, len(R_LEVELS) - 2)
    t = (r - R_LEVELS[k]) / (R_LEVELS[k + 1] - R_LEVELS[k])
    return (1.0 - t) * L_levels[k] + t * L_levels[k + 1]


def solve_texel(C_obs, cos_t, E, L_levels,
                r_grid=np.linspace(0.0, 1.0, 51),
                X_grid=np.linspace(0.0, 1.0, 41)):
    """Scan (X, r); per ciascuna coppia d ha soluzione chiusa per canale:

        C_jc = A_jc * d_c + b_jc
        A_jc = (1-X) E_c / pi + X (1-g_j) L_jc
        b_jc = (0.04 (1-X)(1-g_j) + g_j) L_jc
    """
    g = schlick_g(cos_t)[:, None]
    best = (np.inf, None)
    for r in r_grid:
        L = lerp_levels(L_levels, r)
        for X in X_grid:
            A = (1.0 - X) * E / np.pi + X * (1.0 - g) * L
            b = (0.04 * (1.0 - X) * (1.0 - g) + g) * L
            num = (A * (C_obs - b)).sum(0)
            den = (A * A).sum(0)
            d = np.clip(num / np.maximum(den, 1e-12), 0.0, 1.0)
            resid = ((A * d + b - C_obs) ** 2).sum()
            if resid < best[0]:
                best = (resid, (d, X, r))
    return best


def solve_naive(C_obs, E, L, X_grid):
    """Modello naive C = X d E + (1-X) s L: per X fissato e' lineare in (d,s).
    Mostra che il residuo e' piatto in X -> X non identificabile."""
    out = []
    for X in X_grid:
        resid, d_rec = 0.0, np.zeros(3)
        for c in range(3):
            M = np.stack([np.full(len(L), X * E[c]), (1.0 - X) * L[:, c]], -1)
            coef, rss, *_ = np.linalg.lstsq(M, C_obs[:, c], rcond=None)
            d_rec[c] = coef[0]
            resid += float(rss[0]) if len(rss) else float(
                ((M @ coef - C_obs[:, c]) ** 2).sum())
        out.append((X, resid, d_rec))
    return out


# ----------------------------------------------------------------- experiment


def main():
    E = irradiance()
    print(f"Irradianza E = {np.round(E, 3)}  (RGB)\n")

    materials = {
        "dielettrico opaco  (d rosso,  X=0.0, r=0.80)": (np.array([0.70, 0.15, 0.10]), 0.0, 0.80),
        "dielettrico lucido (d blu,    X=0.0, r=0.12)": (np.array([0.10, 0.20, 0.65]), 0.0, 0.12),
        "metallo oro        (d oro,    X=1.0, r=0.25)": (np.array([1.00, 0.71, 0.29]), 1.0, 0.25),
        "semi-metallo       (d grigio, X=0.5, r=0.45)": (np.array([0.50, 0.50, 0.50]), 0.5, 0.45),
    }

    for n_cams in (16, 6, 3):
        views = make_cameras(n_cams)
        cos_t = views[:, 2]
        refl = 2.0 * cos_t[:, None] * N_NORMAL - views
        L_levels = np.stack(
            [np.stack([prefiltered_env(R, rk) for R in refl]) for rk in R_LEVELS])

        for noise in (0.0, 0.02, 0.05):
            print(f"=== {n_cams} camere, rumore {noise:.0%} ===")
            for name, (d_gt, X_gt, r_gt) in materials.items():
                L_true = np.stack([prefiltered_env(R, r_gt) for R in refl])
                C = render(d_gt, X_gt, cos_t, E, L_true)
                C_obs = C * (1.0 + noise * rng.standard_normal(C.shape))
                resid, (d, X, r) = solve_texel(C_obs, cos_t, E, L_levels)
                print(f"  {name}")
                print(f"    GT  d={np.round(d_gt, 3)}  X={X_gt:.2f}  r={r_gt:.2f}")
                print(f"    rec d={np.round(d, 3)}  X={X:.2f}  r={r:.2f}"
                      f"   (residuo {resid:.2e})")
            print()

    # --- degenerazione del modello naive --------------------------------
    print("=== Degenerazione del blend lineare C = X*d*E + (1-X)*s*L ===")
    print("(metallo oro, 16 camere, r fissato al valore vero, nessun rumore)")
    views = make_cameras(16)
    cos_t = views[:, 2]
    refl = 2.0 * cos_t[:, None] * N_NORMAL - views
    d_gt, X_gt, r_gt = materials["metallo oro        (d oro,    X=1.0, r=0.25)"]
    L_true = np.stack([prefiltered_env(R, r_gt) for R in refl])
    C = render(d_gt, X_gt, cos_t, E, L_true)
    for X, resid, d_rec in solve_naive(C, E, L_true, np.array([0.1, 0.3, 0.5, 0.7, 0.9])):
        print(f"  X assunto={X:.1f} -> residuo {resid:.3e},"
              f"  d recuperato={np.round(d_rec, 3)}")
    print("  -> residuo identico per ogni X: il sistema non vincola X,"
          " e d scala di conseguenza.")


if __name__ == "__main__":
    main()