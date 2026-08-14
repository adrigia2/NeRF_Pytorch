"""Smoke test of the SpecCone pass, envmap-only (no NeRF required).

Checks:
  1. the pass runs over every tile and the buffers have consistent shapes/ranges;
  2. with a constant envmap, the mean of every ring of the texels with no hit is
     exactly the envmap value;
  3. with an envmap that is a function of elevation, level 0 (the mirror ray)
     matches the envmap evaluated along R = reflect(v, n) computed in NumPy
     (validating both the reflected direction and the equirectangular convention);
  4. the cone reconstruction: the solid-angle mean over the whole hemisphere
     (inline weights) matches the historical cumulative formula, and the smoothing
     is monotone from the mirror to ever wider cones (pure mean);
  5. the pure mean over the cone (ring_weights_mean of pbr_solver.py, normalised
     over the valid samples): on a constant envmap it is ≡ 1, and on rings not cut
     by the horizon it is exactly 1 at every aperture, like the mirror level
     (homogeneous radiance units across every candidate);
  6. NON-uniform per-ring sample counts: the kernel really launches N_i rays per
     ring, and the reconstruction with Ω_i/N_i weights reproduces the analytic
     reference on a linear envmap. That last one is the only check with power over
     a wrong weight: on a CONSTANT envmap every mean_i is 1, so any convex
     combination yields 1 and the weights are not observable.

Usage:  python test_spec_cone_smoke.py
"""

import sys
from pathlib import Path

import numpy as np

import _paths  # noqa: F401
import OptixProgrammablePasses as optix

REPO = Path(__file__).resolve().parents[2] / "OptixProjectCMake"
MODEL = REPO / "Scenes" / "SwordShield" / "Models" / "SwordShield.obj"

APERTURES = [0.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0]
K = len(APERTURES)
RS_UNIFORM = [16] * (K - 1)                              # the historical sampling
RS_VAR = [8, 8, 12, 16, 24, 32, 40, 48, 56, 64]          # increasing outwards
IUM_RES = 64
SKY_W, SKY_H = 256, 128


def make_envmap(mode: str) -> np.ndarray:
    """Equirectangular envmap (H*W, 3) float32. sampleEnvmap convention:
    u = 0.5 + atan2(dx, -dy)/2π,  v = 0.5 - asin(dz)/π  (Z-up)."""
    if mode == "constant":
        return np.full((SKY_H * SKY_W, 3), 1.0, dtype=np.float32)
    v = (np.arange(SKY_H) + 0.5) / SKY_H
    dz = np.sin((0.5 - v) * np.pi)                       # elevation per row
    rows = (0.5 + 0.5 * dz).astype(np.float32)           # f(dz) ∈ [0,1]
    env = np.repeat(rows[:, None], SKY_W, axis=1)
    return np.stack([env, env, env], axis=-1).reshape(-1, 3)


def env_value(d: np.ndarray) -> np.ndarray:
    """f(dz) of the 'gradient' envmap evaluated on directions (N,3) (continuous)."""
    return 0.5 + 0.5 * d[:, 2]


def run_pass(envmap, cam_pos, ium_res_obj, num_pix, ring_samples=None):
    gen = optix.SpecConeGenerator()
    gen.set_traversable(model)
    gen.set_inputs(ium_res_obj, APERTURES,
                   RS_UNIFORM if ring_samples is None else ring_samples, 1024)
    gen.set_envmap(envmap, [SKY_W, SKY_H], 0.0)
    gen.set_camera(list(cam_pos), np.empty(0, dtype=np.uint8))  # all visible

    ring_sum = np.zeros((num_pix, K, 3))
    valid    = np.zeros((num_pix, K), dtype=np.int64)
    hit_cnt  = np.zeros((num_pix, K), dtype=np.int64)

    for t in range(gen.num_tiles()):
        r = gen.render_tile(t)
        off = t * 1024
        tt = r.tile_texels
        assert r.num_levels == K
        ring_sum[off:off + tt] += r.sky_sum_np
        valid[off:off + tt]    += r.valid_count_np
        if r.count > 0:
            li = r.local_idx_np
            ri = r.ring_idx_np
            assert li.min() >= 0 and li.max() < tt, "local_idx out of range"
            assert ri.min() >= 0 and ri.max() < K, "ring_idx out of range"
            np.add.at(hit_cnt, (off + li, ri), 1)
    return ring_sum, valid, hit_cnt


print(f"Loading model: {MODEL}")
model = optix.TriangleMesh()
model.add_from_obj_file(MODEL.as_posix())

ium = optix.IUMGenerator()
ium.set_traversable(model)
ium.set_texture_size([IUM_RES, IUM_RES])
ium.render()
ium_res = ium.get_result()

pos   = ium_res.positions_np.astype(np.float64)
nrm   = ium_res.normals_np.astype(np.float64)
masks = ium_res.masks_np.astype(bool)
num_pix = pos.shape[0]
print(f"IUM {IUM_RES}×{IUM_RES}: {masks.sum()} valid texels")

center = pos[masks].mean(axis=0)
diag = np.linalg.norm(pos[masks].max(0) - pos[masks].min(0))
cam_pos = center + np.array([0.6, -1.2, 0.9]) / np.linalg.norm([0.6, -1.2, 0.9]) * (2.5 * diag)

# ── Test 2: constant envmap ───────────────────────────────────────────────────
ring_sum, valid, hit_cnt = run_pass(make_envmap("constant"), cam_pos, ium_res, num_pix)

total_expected = 1 + sum(RS_UNIFORM)
assert valid[~masks].sum() == 0, "masked texels have samples"
assert valid.sum(axis=1).max() <= total_expected, "valid_count beyond the samples launched"
assert (valid[:, 1:] <= np.asarray(RS_UNIFORM)[None, :]).all(), \
    "a ring has more valid samples than were launched"
assert (valid[:, 0] <= 1).all(), "level 0 is not a single ray"

sky_only = masks & (hit_cnt.sum(axis=1) == 0) & (valid[:, 0] > 0)
print(f"[constant] sky-only texels: {sky_only.sum()} / {masks.sum()}")
L_levels = ring_sum[sky_only] / np.maximum(valid[sky_only, :, None], 1)
err = np.abs(L_levels[valid[sky_only] > 0] - 1.0).max() if sky_only.any() else 0.0
print(f"[constant] max |L - 1| over the sky-only levels: {err:.2e}")
assert err < 1e-5, "constant envmap not reproduced exactly"

# ── Test 3: gradient envmap, level 0 vs NumPy reflect ────────────────────────
ring_sum, valid, hit_cnt = run_pass(make_envmap("gradient"), cam_pos, ium_res, num_pix)

mirror_sky = masks & (hit_cnt[:, 0] == 0) & (valid[:, 0] > 0)
n_len = np.linalg.norm(nrm, axis=1)
ok_n = n_len > 1e-8
sel = mirror_sky & ok_n

n_hat = nrm[sel] / n_len[sel, None]
v = cam_pos - pos[sel]
v /= np.linalg.norm(v, axis=1, keepdims=True)
nv = (n_hat * v).sum(axis=1)
sel2 = nv > 0
R = 2.0 * nv[sel2, None] * n_hat[sel2] - v[sel2]

L0 = ring_sum[sel][sel2][:, 0, 0] / valid[sel][sel2][:, 0]
expected = env_value(R)
err = np.abs(L0 - expected)
print(f"[gradient] mirror vs NumPy on {len(L0)} texels: "
      f"max err={err.max():.4f}, mean={err.mean():.4f}")
assert err.max() < 0.05, "level 0 does not match envmap(reflect(v,n))"

# ── Lobe reconstruction (pbr_solver helper) on the per-ring means ────────────
from images_generator import ring_weights_mean

cos_b = np.cos(np.radians(np.asarray(APERTURES)) * 0.5)
sky_all = masks & (hit_cnt.sum(axis=1) == 0) & (valid > 0).all(axis=1)
means = ring_sum[sky_all] / valid[sky_all][..., None]              # (n, K, 3)
cnts  = valid[sky_all].astype(np.float64)                          # (n, K)

def lobe(w):
    """L = Σ_i W_i·mean_i·valid_i / Σ_i W_i·valid_i over rings 1..K-1."""
    wc = w[None, :] * cnts[:, 1:]
    return (np.einsum("nk,nkc->nc", wc, means[:, 1:])
            / wc.sum(axis=1)[:, None])

# (a) bake consistency: solid-angle mean over the whole hemisphere
#     (inline weights = untruncated ring_weights_mean) ≡ historical cumulative formula
omega = ring_weights_mean(cos_b, K - 1)          # = 2π(c_{i-1} − c_i), no truncation
w_sum = np.cumsum(omega[None, :, None] * ring_sum[sky_all][:, 1:], axis=1)
w_cnt = np.cumsum(omega[None, :] * valid[sky_all][:, 1:], axis=1)
L_old = w_sum[:, -1] / w_cnt[:, -1, None]
L_new = lobe(omega)
err = np.abs(L_new - L_old).max()
print(f"[lobes] solid-angle mean (180°) vs historical cumulative: max err={err:.2e}")
assert err < 1e-9, "solid-angle mean ≠ historical cumulative formula"

# (b) monotone smoothing: mirror → ever wider cones (pure mean), decreasing std;
#     the mean is already normalised, so the units are the same at every aperture
spreads = [np.std(means[:, 0, 0])]
for k in range(1, K):
    spreads.append(np.std(lobe(ring_weights_mean(cos_b, k))[:, 0]))
print("[lobes] std mirror→180°: " + "  ".join(f"{x:.4f}" for x in spreads))
assert spreads[-1] < spreads[0], "the wide lobe should average (smaller std)"
assert all(spreads[i + 1] <= spreads[i] * 1.05 for i in range(len(spreads) - 1)), \
    "the smoothing should be ~monotone as the lobe widens"

# Per-texel cut depth: kmax = the last ring NOT cut by the horizon.
# Ring k is intact if it has exactly the N_k samples launched. Note: the outermost
# ring reaches 90° from R, so its outermost sample always falls on the horizon and
# kmax never reaches K-1: that is geometry, not a defect.
def cut_depth(valid_arr, rs):
    uncut = valid_arr[:, 1:] == np.asarray(rs)[None, :]
    return np.where(uncut.all(axis=1), K - 1, uncut.argmin(axis=1))


def lobe_of(w, means_sel, cnts_sel):
    """L = Σ_i W_i·mean_i·valid_i / Σ_i W_i·valid_i over rings 1..K-1."""
    wc = w[None, :] * cnts_sel[:, 1:]
    return (np.einsum("nk,nkc->nc", wc, means_sel[:, 1:])
            / wc.sum(axis=1)[:, None])


# (c) pure mean on a constant envmap ≡ 1: every truncated cone must be exactly 1,
#     like the mirror level (homogeneous radiance units).
#     NOTE: this does NOT check the weights — with mean_i ≡ 1 any convex
#     combination yields 1. It only checks the homogeneity of the units
#     (mean against integral). The weight check is block (d).
ring_sum_c, valid_c, hit_cnt_c = run_pass(make_envmap("constant"), cam_pos,
                                          ium_res, num_pix)
sky_c = masks & (hit_cnt_c.sum(axis=1) == 0) & (valid_c[:, 0] > 0)
kmax_c = cut_depth(valid_c, RS_UNIFORM)
means_c = ring_sum_c / np.maximum(valid_c[..., None], 1)
cnts_c  = valid_c.astype(np.float64)

err0 = np.abs(means_c[sky_c, 0] - 1.0).max()
print(f"[lobes] mirror on a constant envmap: max |L - 1| = {err0:.2e}")
assert err0 < 1e-5, "mirror ≠ constant envmap"

tested = 0
for k in range(1, K):
    sel = sky_c & (kmax_c >= k)
    if sel.sum() == 0:
        print(f"[lobes] pure mean r={APERTURES[k]:g}°: no texel with rings "
              f"1..{k} intact, skipped")
        continue
    err = np.abs(lobe_of(ring_weights_mean(cos_b, k), means_c[sel], cnts_c[sel])
                 - 1.0).max()
    print(f"[lobes] pure mean r={APERTURES[k]:g}° on {sel.sum()} texels: "
          f"expected 1, max err={err:.2e}")
    assert err < 1e-5, f"pure mean r={APERTURES[k]}° ≠ 1 on a constant envmap"
    tested += 1
assert tested >= 3, "the pure-mean check covered too few truncations"

# ── (d) NON-uniform per-ring samples + analytic reference ────────────────────
# On make_envmap("gradient") env(d) = 0.5 + 0.5·d_z, LINEAR in d, so the pure mean
# over the cap of half-aperture b around R has a closed form:
#     L(k) = 0.5 + 0.5 · R_z · (1 + cos b_k)/2
# The discrete reconstruction reproduces it exactly: midpoint stratification gives
# a mean of (c_{i-1}+c_i)/2 per ring, and with weights ∝ (c_{i-1}−c_i) the sum
# telescopes into (1−c_k²)/2 / (1−c_k) = (1+c_k)/2. With Ω_i·N_i weights (that is,
# without dividing by the samples launched) the telescoping breaks: this is the
# check that tells a correct weight from a wrong one.
ring_sum_v, valid_v, hit_cnt_v = run_pass(make_envmap("gradient"), cam_pos,
                                          ium_res, num_pix, RS_VAR)

rs_v = np.asarray(RS_VAR)
assert (valid_v[:, 1:] <= rs_v[None, :]).all(), \
    "valid_count beyond the samples launched, with variable N"
reached = valid_v[:, 1:].max(axis=0)
print(f"[var] N_i requested     : {RS_VAR}")
print(f"[var] max valid reached : {reached.tolist()}")
assert (reached[:-1] == rs_v[:-1]).all(), \
    "an inner ring does not reach N_i: the kernel is not reading ring_samples[i]"

# R = reflect(v, n) for every texel, for the analytic reference
n_len_all = np.linalg.norm(nrm, axis=1)
ok_all = masks & (n_len_all > 1e-8)
n_hat_all = np.zeros_like(nrm)
n_hat_all[ok_all] = nrm[ok_all] / n_len_all[ok_all, None]
v_all = cam_pos[None, :] - pos
v_all /= np.maximum(np.linalg.norm(v_all, axis=1, keepdims=True), 1e-12)
nv_all = (n_hat_all * v_all).sum(axis=1)
R_all = 2.0 * nv_all[:, None] * n_hat_all - v_all
Rz = R_all[:, 2]

sky_v  = ok_all & (nv_all > 0) & (hit_cnt_v.sum(axis=1) == 0) & (valid_v[:, 0] > 0)
kmax_v = cut_depth(valid_v, RS_VAR)
means_v = ring_sum_v / np.maximum(valid_v[..., None], 1)
cnts_v  = valid_v.astype(np.float64)

ratios = []
for k in range(1, K):
    sel = sky_v & (kmax_v >= k)
    if sel.sum() < 20:
        print(f"[analytic] r={APERTURES[k]:g}°: only {sel.sum()} texels, skipped")
        continue
    ref = 0.5 + 0.5 * Rz[sel] * (1.0 + cos_b[k]) / 2.0
    ok_w = lobe_of(ring_weights_mean(cos_b, k, RS_VAR),
                   means_v[sel], cnts_v[sel])[:, 0]
    bad_w = lobe_of(ring_weights_mean(cos_b, k, None),
                    means_v[sel], cnts_v[sel])[:, 0]
    e_ok, e_bad = np.abs(ok_w - ref).max(), np.abs(bad_w - ref).max()
    print(f"[analytic] r={APERTURES[k]:g}° on {sel.sum():5d} texels: "
          f"err(Ω/N)={e_ok:.2e}   err(Ω, wrong)={e_bad:.2e}   "
          f"ratio={e_bad / max(e_ok, 1e-12):.1f}×")
    assert e_ok < 3e-2, \
        f"pure mean r={APERTURES[k]}° ≠ analytic reference: wrong weights or " \
        f"stratification"
    assert e_ok <= e_bad * 1.05, \
        f"at r={APERTURES[k]}° the Ω_i/N_i weight is WORSE than the unnormalised one"
    ratios.append((k, e_bad / max(e_ok, 1e-12)))

assert len(ratios) >= 4, "the analytic reference covered too few truncations"
# On narrow cones the rings see almost identical radiances and the weight is nearly
# irrelevant: the discrimination exists only on the wide cones, where the outer ring
# weighs a lot and has many more samples than the inner ones.
k_wide, ratio_wide = ratios[-1]
print(f"[analytic] discrimination at the widest truncation tested "
      f"(r={APERTURES[k_wide]:g}°): {ratio_wide:.1f}×")
assert ratio_wide > 3.0, \
    "on the widest cone the N_i normalization does not improve things " \
    "significantly: the Ω_i/N_i correction is not active"

print("\n✓ All SpecCone pass checks passed")
