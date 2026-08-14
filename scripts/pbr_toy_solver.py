"""Numerical toy that checks the identifiability of the multi-view PBR system.

Model (split-sum, metallic workflow, per texel, camera j):

    C_j = (1 - X) * (d / pi) * E  +  F_j * L_j(r)

    F_j  = F0 + (1 - F0) * (1 - cos(theta_j))^5        (Schlick)
    F0   = 0.04 * (1 - X) + d * X                       (spectral coupling)
    E    = cosine-weighted irradiance over the hemisphere (view-independent)
    L_j(r) = GGX-prefiltered ambient radiance around reflect(v_j, n)
             (in the real pipeline: cone-traced NeRF query, here: analytic envmap + MC)

Unknowns per texel: d (3), X (1), r (1).
Strategy: grid scan over (X, r) + closed-form least squares on d (linear!),
with L(r) interpolated from K precomputed levels (the "mip chain").

The script also checks the degeneracy of the naive model
    C_j = X*d*E + (1-X)*s*L_j(r)
in which X is not identifiable (it is absorbed into d and s).

Finally it checks the identifiability of the model pbr_solver.py adopted
(2026-07-16, pure mean over the cone, s ≡ 1):

    C_j = (a*x/pi) * E + (1 - x) * L_j(r)      L_j = mean over the cone

Here the slope of C with respect to L across the cameras directly identifies
beta = 1-x (the diffuse term is view-independent) and the intercept identifies
a*x*E/pi: unlike the naive model there is no free scale on the specular, so
(a, x, r) are recoverable. The fit is the same centred closed-form regression
pbr_solver.py uses.
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
    """Analytic environment radiance, w: (...,3) -> (...,3). Coloured and
    angularly varied (warm sun + red wall) so that different reflection
    directions see different colours."""
    t = 0.5 * (w[..., 2:3] + 1.0)
    sky = (1.0 - t) * HORIZON + t * ZENITH
    sun = SUN_COLOR * np.exp((np.sum(w * SUN_DIR, -1, keepdims=True) - 1.0) / 0.02)
    wall = WALL_COLOR * np.exp((np.sum(w * WALL_DIR, -1, keepdims=True) - 1.0) / 0.15)
    return sky + sun + wall


N_NORMAL = np.array([0.0, 0.0, 1.0])  # texel with a +Z normal (Z-up, as in the project)


def irradiance(n_samples=200_000):
    """E = integral of L*cos over the hemisphere. Cosine sampling: pdf=cos/pi
    so E = pi * mean(L)."""
    u1, u2 = rng.random(n_samples), rng.random(n_samples)
    z = np.sqrt(u1)
    rxy = np.sqrt(1.0 - u1)
    phi = 2.0 * np.pi * u2
    w = np.stack([rxy * np.cos(phi), rxy * np.sin(phi), z], -1)
    return np.pi * env_radiance(w).mean(0)


def _frame(d):
    """Orthonormal basis with d as the third axis."""
    a = np.array([1.0, 0.0, 0.0]) if abs(d[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t = _normalize(np.cross(a, d))
    b = np.cross(d, t)
    return t, b, d


def prefiltered_env(R, rough, n_samples=8192):
    """L(R, r): split-sum prefilter with GGX sampling (assuming n=v=R).
    In the real pipeline this is the NeRF query with a cone of aperture ~r."""
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
    """n view directions over the upper hemisphere (Fibonacci spiral,
    elevations from ~grazing to ~zenithal)."""
    i = np.arange(n)
    z = 0.15 + 0.8 * (i + 0.5) / n
    rxy = np.sqrt(1.0 - z * z)
    phi = i * np.pi * (3.0 - np.sqrt(5.0))
    return np.stack([rxy * np.cos(phi), rxy * np.sin(phi), z], -1)


# ------------------------------------------------------------- forward model


def schlick_g(cos_t):
    return (1.0 - cos_t) ** 5


def render(d, X, cos_t, E, L):
    """C_j for every camera. d:(3,) X:scalar cos_t:(N,) E:(3,) L:(N,3)."""
    F0 = 0.04 * (1.0 - X) + d * X
    g = schlick_g(cos_t)[:, None]
    F = F0 * (1.0 - g) + g
    return (1.0 - X) * d * E / np.pi + F * L


# -------------------------------------------------------------------- solver

R_LEVELS = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.55, 0.7, 0.85, 1.0])


def lerp_levels(L_levels, r):
    """Interpolate the level chain L(r_k) -> L(r). L_levels: (K, N, 3)."""
    k = np.clip(np.searchsorted(R_LEVELS, r) - 1, 0, len(R_LEVELS) - 2)
    t = (r - R_LEVELS[k]) / (R_LEVELS[k + 1] - R_LEVELS[k])
    return (1.0 - t) * L_levels[k] + t * L_levels[k + 1]


def solve_texel(C_obs, cos_t, E, L_levels,
                r_grid=np.linspace(0.0, 1.0, 51),
                X_grid=np.linspace(0.0, 1.0, 41)):
    """Scan (X, r); for each pair, d has a closed-form solution per channel:

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


def cone_mean_env(R, aperture_deg, n_samples=8192):
    """Pure mean of the radiance over a cone of total aperture aperture_deg
    around R (uniform sampling in solid angle, no cos weight).
    In the real pipeline this is the ring_weights_mean reconstruction of pbr_solver.py."""
    if aperture_deg < 1e-3:
        return env_radiance(R)
    cos_b = np.cos(np.radians(aperture_deg) * 0.5)
    u1, u2 = rng.random(n_samples), rng.random(n_samples)
    z = 1.0 - u1 * (1.0 - cos_b)              # cos uniform in [cos_b, 1]
    s = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = 2.0 * np.pi * u2
    t, b, d = _frame(R)
    w = (s[:, None] * np.cos(phi)[:, None] * t
         + s[:, None] * np.sin(phi)[:, None] * b
         + z[:, None] * d)
    return env_radiance(w).mean(0)


def render_cone_model(a, x, E, L):
    """C_j = (a*x/pi)*E + (1-x)*L_j  (the pbr_solver model, pure mean over the cone)."""
    return (x * a * E / np.pi)[None, :] + (1.0 - x) * L


def solve_cone_model(C_obs, E, L_levels, apertures, x_eps=1e-3):
    """Closed-form fit of the pbr_solver model: for each candidate r, a regression
    centred, C_jc = alpha_c + beta*L_jc (beta shared across channels); argmin
    of the residual over r; then x = 1-beta, a = pi*alpha/(E*x). Identical to the
    real pipeline (centred sufficient statistics)."""
    best = (np.inf, None)
    for k, ap in enumerate(apertures):
        L = L_levels[k]
        dL = L - L.mean(0)
        dC = C_obs - C_obs.mean(0)
        vll = (dL * dL).sum()
        vcl = (dC * dL).sum()
        vcc = (dC * dC).sum()
        beta = np.clip(vcl / max(vll, 1e-12), 0.0, 1.0)
        res = vcc - 2.0 * beta * vcl + beta * beta * vll
        if res < best[0]:
            alpha = np.maximum(C_obs.mean(0) - beta * L.mean(0), 0.0)
            x = 1.0 - beta
            a = (np.clip(np.pi * alpha / (E * x), 0.0, 1.0)
                 if x >= x_eps else np.zeros(3))
            best = (res, (a, x, ap))
    return best


def solve_naive(C_obs, E, L, X_grid):
    """Naive model C = X d E + (1-X) s L: for fixed X it is linear in (d,s).
    Shows that the residual is flat in X -> X is not identifiable."""
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
        "matte dielectric   (d red,    X=0.0, r=0.80)": (np.array([0.70, 0.15, 0.10]), 0.0, 0.80),
        "glossy dielectric  (d blue,   X=0.0, r=0.12)": (np.array([0.10, 0.20, 0.65]), 0.0, 0.12),
        "gold metal         (d gold,   X=1.0, r=0.25)": (np.array([1.00, 0.71, 0.29]), 1.0, 0.25),
        "semi-metal         (d grey,   X=0.5, r=0.45)": (np.array([0.50, 0.50, 0.50]), 0.5, 0.45),
    }

    for n_cams in (16, 6, 3):
        views = make_cameras(n_cams)
        cos_t = views[:, 2]
        refl = 2.0 * cos_t[:, None] * N_NORMAL - views
        L_levels = np.stack(
            [np.stack([prefiltered_env(R, rk) for R in refl]) for rk in R_LEVELS])

        for noise in (0.0, 0.02, 0.05):
            print(f"=== {n_cams} cameras, noise {noise:.0%} ===")
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

    # --- degeneracy of the naive model ----------------------------------
    print("=== Degeneracy of the linear blend C = X*d*E + (1-X)*s*L ===")
    print("(gold metal, 16 cameras, r fixed at the true value, no noise)")
    views = make_cameras(16)
    cos_t = views[:, 2]
    refl = 2.0 * cos_t[:, None] * N_NORMAL - views
    d_gt, X_gt, r_gt = materials["gold metal         (d gold,   X=1.0, r=0.25)"]
    L_true = np.stack([prefiltered_env(R, r_gt) for R in refl])
    C = render(d_gt, X_gt, cos_t, E, L_true)
    for X, resid, d_rec in solve_naive(C, E, L_true, np.array([0.1, 0.3, 0.5, 0.7, 0.9])):
        print(f"  X assunto={X:.1f} -> residuo {resid:.3e},"
              f"  d recuperato={np.round(d_rec, 3)}")
    print("  -> identical residual for every X: the system does not constrain X,"
          " and d scales accordingly.")

    experiment_cone_model()


# -------------------------------------------- pbr_solver model (2026-07-16)

APERTURES_TOY = np.array([0.0, 10.0, 25.0, 50.0, 90.0, 130.0, 180.0])


def experiment_cone_model():
    """Identifiability of the adopted model: C = (a*x/pi)E + (1-x)*mean-cone(r).
    Same closed-form fit as pbr_solver.py; r quantized to the aperture grid
    (the 'off-grid' material has to fall on the adjacent level)."""
    E = irradiance()
    print("\n=== pbr_solver model: C = (a*x/pi)*E + (1-x)*mean-cone(r) ===")
    print(f"aperture grid: {APERTURES_TOY.astype(int).tolist()} degrees\n")

    materials = {
        "pure diffuse   (a red,    x=1.00, r=n/d)": (np.array([0.70, 0.15, 0.10]), 1.00, 90.0),
        "glossy         (a blue,   x=0.70, r=25)": (np.array([0.10, 0.20, 0.65]), 0.70, 25.0),
        "near mirror    (a grey,   x=0.20, r=0)": (np.array([0.50, 0.50, 0.50]), 0.20, 0.0),
        "off-grid       (a green,  x=0.50, r=35)": (np.array([0.20, 0.60, 0.25]), 0.50, 35.0),
    }

    for n_cams in (16, 6, 3):
        views = make_cameras(n_cams)
        cos_t = views[:, 2]
        refl = 2.0 * cos_t[:, None] * N_NORMAL - views
        L_levels = np.stack(
            [np.stack([cone_mean_env(R, ap) for R in refl])
             for ap in APERTURES_TOY])

        for noise in (0.0, 0.02, 0.05):
            print(f"--- {n_cams} cameras, noise {noise:.0%} ---")
            for name, (a_gt, x_gt, r_gt) in materials.items():
                L_true = np.stack([cone_mean_env(R, r_gt) for R in refl])
                C = render_cone_model(a_gt, x_gt, E, L_true)
                C_obs = C * (1.0 + noise * rng.standard_normal(C.shape))
                resid, (a, x, r) = solve_cone_model(C_obs, E, L_levels,
                                                    APERTURES_TOY)
                print(f"  {name}")
                print(f"    GT  a={np.round(a_gt, 3)}  x={x_gt:.2f}  r={r_gt:.0f}")
                print(f"    rec a={np.round(a, 3)}  x={x:.2f}  r={r:.0f}"
                      f"   (residuo {resid:.2e})")
            print()
    print("  -> with x=1 (pure diffuse) r is not constrained: any level"
          " gives the same residual, but a and x stay correct.")


if __name__ == "__main__":
    main()