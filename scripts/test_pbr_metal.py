"""Synthetic test of the metal/roughness export of pbr_solver (no EXR, no GPU).

The solver goes C_j → (α_c, β_c) → (D, S) → (m, b). The two pure functions that
carry the maths are tested here on data with a known answer, in order of
insidiousness:

  1. _fit_band recovers the winning aperture, the per-channel slopes β_c and the
     intercepts α_c exactly on noiseless data, including a GOLD texel whose three
     slopes differ: the shared slope of the old solver could not represent it.
  2. With noise, the standard error SE_c is calibrated: |β̂_c − β_c| ≤ 2·SE_c on
     about 95% of the texels. It is the map that says where the specular colour is
     measured and where it is noise, so it must not be off by a factor.
  3. _metal_from_reflectances on the two extremes of the convention: a metal
     (D = 0, coloured S) gives m = 1, b = S; a dielectric (S = 0.04 grey) gives
     m = 0, b = D.
  4. Round trip (m, b) → (D, S) → (m, b) exact wherever the convention is
     invertible, with D̂, Ŝ reproducing D, S (consistency 0).
  5. The two documented limits: a base colour darker than the dielectric 4%
     (b̄ < 0.04) makes S̄ < 0.04 and is read as m = 0 — the convention itself is
     ambiguous there; and the channels are averaged BEFORE solving, so a dark
     channel of a coloured dielectric does not pass for metal.
  6. The ROI band mask is the union of the rectangles, clipped to the band.

On success the process prints "✓ all pbr metal tests passed" and exits 0.

Usage:  python test_pbr_metal.py
"""

import numpy as np

import _paths  # noqa: F401
from pbr_solver import (  # noqa: E402
    F0_DIELECTRIC, _fit_band, _metal_from_reflectances, _roi_band_mask,
)

rng = np.random.default_rng(20260930)


def accumulate(C, L, w):
    """Sufficient statistics exactly as solve_pbr accumulates them.

    C (T, n, 3), L (T, n, K, 3), w (T, n) bool → SC, SCC, SL, SLL, SCL, nv."""
    wf = w.astype(np.float64)[..., None]                      # (T, n, 1)
    SC = (wf * C).sum(axis=1)
    SCC = (wf * C * C).sum(axis=1)
    wk = wf[:, :, None, :]                                     # (T, n, 1, 1)
    SL = (wk * L).sum(axis=1)
    SLL = (wk * L * L).sum(axis=1)
    SCL = (wk * C[:, :, None, :] * L).sum(axis=1)
    return SC, SCC, SL, SLL, SCL, w.sum(axis=1).astype(np.int32)


def synth(T, n, K, noise=0.0):
    """Texels with known (k*, β_c, α_c): C_j = α + β·L_j(k*) (+ noise)."""
    L = rng.uniform(0.05, 3.0, size=(T, n, K, 3))
    k_true = rng.integers(0, K, size=T)
    beta = rng.uniform(0.05, 1.0, size=(T, 3))
    alpha = rng.uniform(0.0, 0.5, size=(T, 3))
    Lk = L[np.arange(T), :, k_true]                            # (T, n, 3)
    C = alpha[:, None, :] + beta[:, None, :] * Lk
    if noise:
        C = C + rng.normal(0.0, noise, size=C.shape)
    w = rng.random((T, n)) > 0.2                               # some cameras miss the texel
    w[:, :3] = True                                            # ≥ 3 valid views everywhere
    return C, L, w, k_true, beta, alpha


# ── 1. exact recovery on noiseless data ─────────────────────────────────────
T, n, K = 4000, 24, 14
C, L, w, k_true, beta, alpha = synth(T, n, K)
# Texel 0 is gold: three different slopes, no diffuse.
beta[0] = [1.0, 0.78, 0.34]
alpha[0] = 0.0
Lk0 = L[0, :, k_true[0]]
C[0] = alpha[0] + beta[0] * Lk0

fit = _fit_band(*accumulate(C, L, w))
assert np.array_equal(fit["best_k"], k_true), \
    f"wrong aperture on {(fit['best_k'] != k_true).sum()} texels"
err_b = np.abs(fit["beta"] - beta).max()
err_a = np.abs(fit["alpha"] - alpha).max()
assert err_b < 1e-9, f"β_c not recovered: max err {err_b:.2e}"
assert err_a < 1e-9, f"α_c not recovered: max err {err_a:.2e}"
# The residual comes from a difference of sums of order 10-100: float64 roundoff
# leaves ~1e-15 there, anything above 1e-12 would be a real misfit.
assert fit["res"].max() < 1e-12, f"non-zero residual on exact data: {fit['res'].max():.2e}"
print(f"✓ exact fit: aperture, β_c, α_c recovered on {T} texels "
      f"(max err β {err_b:.1e}, α {err_a:.1e}); gold slopes {np.round(fit['beta'][0], 6)}")

# ── 2. the standard error is calibrated ────────────────────────────────────
T, n, K, sigma = 20000, 40, 6, 0.02
C, L, w, k_true, beta, alpha = synth(T, n, K, noise=sigma)
fit = _fit_band(*accumulate(C, L, w))
ok_k = fit["best_k"] == k_true
assert ok_k.mean() > 0.99, f"aperture recovered on only {ok_k.mean():.3f} of the texels"
# β is clipped to [0,1] and the truth is ≥ 0.05, so only texels far from the
# clip boundary enter the coverage count.
inner = ok_k[:, None] & (beta > 0.1) & (beta < 0.9)
z = np.abs(fit["beta"] - beta) / np.maximum(fit["se"], 1e-30)
cover = (z[inner] <= 2.0).mean()
assert 0.90 < cover < 0.99, f"|β̂−β| ≤ 2·SE on {cover:.3f} of the slopes (expected ≈0.95)"
print(f"✓ SE calibrated: |β̂−β| ≤ 2·SE on {cover:.3f} of {inner.sum()} slopes "
      f"(σ={sigma}, {n} views)")

# ── 3. the two extremes of the convention ───────────────────────────────────
gold_S = np.array([1.0, 0.78, 0.34])
m, b, Dh, Sh = _metal_from_reflectances(np.zeros(3), gold_S)
assert m == 1.0 and np.allclose(b, gold_S, atol=1e-15), f"gold: m={m}, b={b}"
red_D = np.array([0.6, 0.1, 0.1])
m, b, Dh, Sh = _metal_from_reflectances(red_D, np.full(3, F0_DIELECTRIC))
assert m == 0.0 and np.allclose(b, red_D, atol=1e-15), f"red plastic: m={m}, b={b}"
m, b, _, _ = _metal_from_reflectances(red_D, np.full(3, 0.02))
assert m == 0.0, f"S̄ below the dielectric 4% must give m=0, got {m}"
print("✓ extremes: gold → m=1, b=S; red plastic → m=0, b=D; S̄<0.04 → m=0")

# ── 4. round trip where the convention is invertible ───────────────────────
M = 200000
m_true = rng.uniform(0.0, 1.0, M)
m_true[:1000] = 0.0
m_true[1000:2000] = 1.0
b_true = rng.uniform(0.0, 1.0, (M, 3))
keep = b_true.mean(axis=1) > 0.06          # see point 5 for b̄ < 0.04
m_true, b_true = m_true[keep], b_true[keep]
um, mm = (1 - m_true)[:, None], m_true[:, None]
D = um * b_true
S = F0_DIELECTRIC * um + mm * b_true
m, b, Dh, Sh = _metal_from_reflectances(D, S)
err_m = np.abs(m - m_true).max()
err_bc = np.abs(b - b_true).max()
err_c = max(np.abs(Dh - D).max(), np.abs(Sh - S).max())
assert err_m < 1e-9, f"m not recovered: max err {err_m:.2e}"
assert err_bc < 1e-9, f"b not recovered: max err {err_bc:.2e}"
assert err_c < 1e-9, f"(D̂, Ŝ) ≠ (D, S): max err {err_c:.2e}"
print(f"✓ round trip on {keep.sum()} materials: max err m {err_m:.1e}, b {err_bc:.1e}, "
      f"consistency {err_c:.1e}")

# ── 5. documented limits ────────────────────────────────────────────────────
# A metal darker than the dielectric 4% reflects less than a plastic: S̄ < 0.04
# and the convention cannot tell it from a dark dielectric.
b_dark = np.full(3, 0.03)
m, _, _, _ = _metal_from_reflectances(0.2 * b_dark, F0_DIELECTRIC * 0.2 + 0.8 * b_dark)
assert m == 0.0, f"expected the ambiguous dark metal to read as m=0, got {m}"
# Blue plastic with a slightly inflated reflection: averaging the channels first
# keeps it a dielectric; solving per channel and averaging m would give ≈0.15.
blue_D = np.array([0.05, 0.10, 0.80])
m, _, _, _ = _metal_from_reflectances(blue_D, np.full(3, 0.05))
per_ch = np.mean([_metal_from_reflectances(np.full(3, d), np.full(3, 0.05))[0]
                  for d in blue_D])
assert abs(m - 0.0347) < 1e-3, f"blue plastic: m={m:.4f}, expected 0.0347"
assert per_ch > 0.14, f"per-channel mean of m should be ≈0.15, got {per_ch:.3f}"
print(f"✓ limits: dark metal (b̄<0.04) → m=0; blue plastic m={m:.3f} "
      f"(per-channel mean would be {per_ch:.3f})")

# ── 6. ROI band mask ────────────────────────────────────────────────────────
W = 16
assert _roi_band_mask(None, 0, 4, W) is None
mk = _roi_band_mask([(2, 1, 3, 2), (10, 3, 20, 5)], 2, 4, W).reshape(4, W)
ref = np.zeros((4, W), dtype=bool)
ref[0, 2:5] = True             # rect 1 spans rows 1-2 → only row 2 is in the band
ref[1:4, 10:16] = True         # rect 2 starts at row 3, clipped at the right edge
assert np.array_equal(mk, ref), "ROI band mask ≠ union of the clipped rectangles"
assert not _roi_band_mask([(0, 0, 4, 2)], 8, 4, W).any(), "ROI leaked into another band"
print("✓ ROI band mask: union, clipped to the band and to the width")

print("✓ all pbr metal tests passed")
