"""Test of the HemiVis pass and of the shared reconstruction, envmap-only (no NeRF).

The shared scheme traces ONE Fibonacci set per texel and lets each camera bin the
same rays into its own ring. The kernel returns only the t_hits, indexed by
position: the directions live on the Python side. The checks below cover, in order
of insidiousness:

  1. kernel↔torch PARITY of the directions (shared and mirror). It is the most
     dangerous failure of the whole scheme: if the two formulas diverge, every
     t_hit is paired with the wrong direction and there is no symptom other than a
     silently wrong L_j.
  2. Properties of the shared set: every direction above the horizon, cosθ exactly
     equispaced (uniformity in solid angle), and an azimuthal rotation that differs
     from texel to texel.
  3. Constant envmap: every per-ring mean and every reconstructed L(k) is exactly
     the envmap value, the mirror level included (homogeneous radiance units
     across every candidate).
  4. 'gradient' envmap with an ANALYTIC reference: on the cones entirely above the
     horizon, L(k) = 0.5 + 0.5·R_z·(1+cos b_k)/2 in closed form. It is the only
     check with power over the weights: on a constant envmap every ring mean is 1,
     so any convex combination yields 1 and a wrong weight stays invisible.
  5. The mirror level matches the envmap evaluated along R = reflect(v, n)
     computed in NumPy.
  6. Cone closing and the on-disk format: raw sums → _cones_from_rings_np →
     IncrementalExrWriter → read_cones must give the same L as the weighted formula
     validated in points 3-4.

Tests 3-5 run on a flat quad generated on the fly: on the real model, with thousands
of directions per texel, almost every texel finds something to hit and the occluded
rays would need the NeRF, leaving no closed-form reference. Above a planar quad every
direction lies in the hemisphere above the normal, so no ray can hit the quad itself.

Note: the process exits with a non-zero code at interpreter shutdown (a pre-existing
OptiX cleanup issue, not a test failure — as in test_spec_cone_smoke.py). The line
that counts is "✓ all HemiVis tests passed".

Usage:  python test_hemivis_shared.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

import _paths  # noqa: F401
import OptixProgrammablePasses as optix
from images_generator import (
    IncrementalExrWriter, _cones_from_rings_np, _hemivis_directions,
    _hemivis_rotation, _sample_envmap_torch, ring_weights_mean,
    spec_cone_channels, spec_cone_level_name, spec_cone_shared_ring_samples,
)
from pbr_solver import read_cones

REPO = Path(__file__).resolve().parents[2] / "OptixProjectCMake"
MODEL = REPO / "Scenes" / "SwordShield" / "Models" / "SwordShield.obj"

APERTURES = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 80.0,
             100.0, 120.0, 140.0, 160.0, 180.0]
K = len(APERTURES)
COS_B = np.cos(np.radians(np.asarray(APERTURES)) * 0.5)

S = 8192            # shared samples per texel
IUM_RES = 64
TILE = 1024
SKY_W, SKY_H = 256, 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_envmap(mode: str) -> np.ndarray:
    """Equirectangular envmap (H*W, 3) float32, sampleEnvmap convention."""
    if mode == "constant":
        return np.full((SKY_H * SKY_W, 3), 1.0, dtype=np.float32)
    v = (np.arange(SKY_H) + 0.5) / SKY_H
    dz = np.sin((0.5 - v) * np.pi)                   # elevation per row
    rows = (0.5 + 0.5 * dz).astype(np.float32)
    env = np.repeat(rows[:, None], SKY_W, axis=1)
    return np.stack([env, env, env], axis=-1).reshape(-1, 3)


# ──────────────────────────────────────────────────────────────────────────────
# Scene setup
# ──────────────────────────────────────────────────────────────────────────────

print(f"Loading model: {MODEL}")
model = optix.TriangleMesh()
model.add_from_obj_file(MODEL.as_posix())

ium = optix.IUMGenerator()
ium.set_traversable(model)
ium.set_texture_size([IUM_RES, IUM_RES])
ium.render()
ium_res = ium.get_result()

pos_np = ium_res.positions_np.astype(np.float32)
nrm_np = ium_res.normals_np.astype(np.float32)
masks  = ium_res.masks_np.astype(bool)
num_pix = pos_np.shape[0]
print(f"IUM {IUM_RES}×{IUM_RES}: {masks.sum()} valid texels")

center = pos_np[masks].mean(axis=0)
diag = np.linalg.norm(pos_np[masks].max(0) - pos_np[masks].min(0))
cam_dir = np.array([0.6, -1.2, 0.9]); cam_dir /= np.linalg.norm(cam_dir)
cam_pos = (center + cam_dir * (2.5 * diag)).astype(np.float32)

gen = optix.HemiVisGenerator()
gen.set_traversable(model)
gen.set_inputs(ium_res, S, TILE)
gen.set_cameras([cam_pos.tolist()])
gen.set_debug_directions(True)

n_tiles = gen.num_tiles()
t_hit = np.zeros((num_pix, S), dtype=np.float32)
t_mir = np.zeros((num_pix, 1), dtype=np.float32)

# ── Test 1: kernel ↔ torch parity of the directions ──────────────────────────
max_err_shared = 0.0
max_err_mirror = 0.0
for t in range(n_tiles):
    r = gen.render_tile(t)
    off, tt = t * TILE, r.tile_texels
    assert r.num_samples == S and r.num_cams == 1
    t_hit[off:off + tt] = r.t_hit_np
    t_mir[off:off + tt] = r.t_hit_mirror_np

    gidx = np.arange(off, off + tt, dtype=np.int64)
    live = masks[gidx] & (np.linalg.norm(nrm_np[gidx], axis=-1) > 1e-8)
    if not live.any():
        continue

    nrm_t = torch.as_tensor(nrm_np[gidx][live], device=device)
    pos_t = torch.as_tensor(pos_np[gidx][live], device=device)
    dirs_torch = _hemivis_directions(nrm_t, gidx[live], S).cpu().numpy()
    max_err_shared = max(max_err_shared,
                         np.abs(dirs_torch - r.dirs_np[live]).max())

    # Mirror ray: the same formula as the kernel, in torch
    v = torch.as_tensor(cam_pos, device=device)[None, :] - pos_t
    v = v / torch.linalg.norm(v, dim=-1, keepdim=True)
    n_unit = nrm_t / torch.linalg.norm(nrm_t, dim=-1, keepdim=True)
    nv = (n_unit * v).sum(-1, keepdim=True)
    R_torch = (n_unit * (2.0 * nv) - v).cpu().numpy()
    front = (nv[:, 0] > 0).cpu().numpy()
    if front.any():
        max_err_mirror = max(max_err_mirror,
                             np.abs(R_torch[front]
                                    - r.dirs_mirror_np[live][front, 0]).max())

assert max_err_shared < 1e-5, \
    f"shared directions diverge between kernel and torch: {max_err_shared:.2e}"
assert max_err_mirror < 1e-5, \
    f"mirror directions diverge between kernel and torch: {max_err_mirror:.2e}"
print(f"✓ kernel↔torch direction parity: max err shared {max_err_shared:.2e}, "
      f"mirror {max_err_mirror:.2e}")

# ── Test 2: properties of the shared set ────────────────────────────────────
probe = np.flatnonzero(masks & (np.linalg.norm(nrm_np, axis=-1) > 1e-8))[:64]
nrm_p = torch.as_tensor(nrm_np[probe], device=device)
dirs_p = _hemivis_directions(nrm_p, probe, S)
n_unit_p = nrm_p / torch.linalg.norm(nrm_p, dim=-1, keepdim=True)
cos_n = (dirs_p * n_unit_p[:, None, :]).sum(-1).cpu().numpy()

assert cos_n.min() > 0.0, "a shared sample falls below the horizon"
expected_cos = 1.0 - (np.arange(S) + 0.5) / S
assert np.abs(np.sort(cos_n, axis=1)[:, ::-1] - expected_cos).max() < 1e-5, \
    "cosθ not equispaced: the set is not uniform in solid angle"
norms = np.linalg.norm(dirs_p.cpu().numpy(), axis=-1)
assert np.abs(norms - 1.0).max() < 1e-5, "directions not normalised"

rots = _hemivis_rotation(probe)
assert 0.0 <= rots.min() and rots.max() < 1.0, "rotation outside [0,1)"
assert len(np.unique(rots)) > len(probe) * 0.9, \
    "the azimuthal rotation does not decorrelate the texels"
print(f"✓ shared set uniform in solid angle, {len(np.unique(rots))} "
      f"distinct rotations over {len(probe)} texels")


# ──────────────────────────────────────────────────────────────────────────────
# Cone reconstruction from the shared rays (the same one the bake does)
# ──────────────────────────────────────────────────────────────────────────────

def bin_shared(idx: np.ndarray, envmap: np.ndarray,
               pos_np, nrm_np, t_hit, t_mir, cam_pos):
    """(means (n, K, 3), counts (n, K), R (n, 3)) for the texels `idx`, single camera.

    Replicates the binning of _precompute_spec_cone_shared: missed rays take the
    envmap, level 0 is the mirror ray, and the samples beyond the widest cone are
    discarded.
    """
    env_t = torch.as_tensor(envmap, device=device)
    nrm_t = torch.as_tensor(nrm_np[idx], device=device)
    pos_t = torch.as_tensor(pos_np[idx], device=device)
    dirs = _hemivis_directions(nrm_t, idx, S)
    th = torch.as_tensor(t_hit[idx], device=device)

    rad = torch.zeros_like(dirs)
    miss = th == 0.0
    rad[miss] = _sample_envmap_torch(dirs[miss], env_t, [SKY_W, SKY_H], 0.0)

    n_unit = nrm_t / torch.linalg.norm(nrm_t, dim=-1, keepdim=True)
    v = torch.as_tensor(cam_pos, device=device)[None, :] - pos_t
    v = v / torch.linalg.norm(v, dim=-1, keepdim=True)
    nv = (n_unit * v).sum(-1, keepdim=True)
    R = n_unit * (2.0 * nv) - v

    cosang = (dirs * R[:, None, :]).sum(-1)
    asc = -torch.as_tensor(COS_B, device=device, dtype=torch.float32)
    ring = torch.searchsorted(asc, -cosang.contiguous(), right=True).clamp_(min=1, max=K)

    n = idx.size
    flat = (torch.arange(n, device=device)[:, None] * (K + 1) + ring).reshape(-1)
    acc_s = torch.zeros((n * (K + 1), 3), device=device)
    acc_c = torch.zeros(n * (K + 1), device=device)
    acc_s.index_add_(0, flat, rad.reshape(-1, 3))
    acc_c.index_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
    sums   = acc_s.view(n, K + 1, 3)[:, :K].clone()
    counts = acc_c.view(n, K + 1)[:, :K].clone()

    # Level 0: mirror ray
    thm = torch.as_tensor(t_mir[idx, 0], device=device)
    mir_ok = (thm == 0.0)                       # miss → envmap; hit → the NeRF is needed
    if bool(mir_ok.any()):
        sums[mir_ok, 0] = _sample_envmap_torch(R[mir_ok], env_t, [SKY_W, SKY_H], 0.0)
        counts[mir_ok, 0] = 1.0

    return (sums / torch.clamp(counts, min=1.0)[..., None]).cpu().numpy(), \
           counts.cpu().numpy(), R.cpu().numpy()


def cone_mean(means, counts, k, ring_samples):
    """L(k) as pbr_solver computes it: weighted mean of rings 1..k."""
    w = ring_weights_mean(COS_B, k, ring_samples)          # (K-1,)
    wc = w[None, :] * counts[:, 1:]
    num = np.einsum("nk,nkc->nc", wc, means[:, 1:])
    return num / np.maximum(wc.sum(axis=1), 1e-12)[:, None]


RING_NOMINAL = np.asarray(spec_cone_shared_ring_samples(APERTURES, S))

# ──────────────────────────────────────────────────────────────────────────────
# Flat scene for the analytic tests
#
# On the real model, with S=8192 directions per texel, practically every texel finds
# something to hit: the occluded rays would need the NeRF and there would be no
# closed-form reference left. A planar quad solves the problem at the root: every
# direction lies in the hemisphere above the normal, so NO ray can hit the quad
# itself and every sample sees the envmap.
# ──────────────────────────────────────────────────────────────────────────────

import tempfile

_tmpdir = tempfile.TemporaryDirectory()
quad_obj = Path(_tmpdir.name) / "quad.obj"
quad_obj.write_text(
    "v -1 -1 0\nv 1 -1 0\nv 1 1 0\nv -1 1 0\n"
    "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
    "f 1/1 2/2 3/3\nf 1/1 3/3 4/4\n", encoding="utf-8")

quad = optix.TriangleMesh()
quad.add_from_obj_file(quad_obj.as_posix())

ium_q = optix.IUMGenerator()
ium_q.set_traversable(quad)
ium_q.set_texture_size([IUM_RES, IUM_RES])
ium_q.render()
ium_q_res = ium_q.get_result()

qpos = ium_q_res.positions_np.astype(np.float32)
qnrm = ium_q_res.normals_np.astype(np.float32)
qmask = ium_q_res.masks_np.astype(bool)
# camera at a finite distance and off axis: R_z varies between the texels, otherwise
# the analytic reference would be constant and would discriminate nothing
qcam = np.array([1.6, 1.1, 1.4], dtype=np.float32)

gen_q = optix.HemiVisGenerator()
gen_q.set_traversable(quad)
gen_q.set_inputs(ium_q_res, S, TILE)
gen_q.set_cameras([qcam.tolist()])

q_t_hit = np.zeros((qpos.shape[0], S), dtype=np.float32)
q_t_mir = np.zeros((qpos.shape[0], 1), dtype=np.float32)
for t in range(gen_q.num_tiles()):
    r = gen_q.render_tile(t)
    off, tt = t * TILE, r.tile_texels
    q_t_hit[off:off + tt] = r.t_hit_np
    q_t_mir[off:off + tt] = r.t_hit_mirror_np

sky_only = np.flatnonzero(qmask & (q_t_hit <= 0.0).all(axis=1) & (q_t_mir[:, 0] == 0.0))
print(f"planar quad: {qmask.sum()} texels, {sky_only.size} of them unoccluded")
assert sky_only.size > 500, \
    f"the quad should be free of self-occlusion, found {sky_only.size} texels"

# ── Test 3: constant envmap ──────────────────────────────────────────────────
means_c, counts_c, _ = bin_shared(sky_only, make_envmap("constant"),
                                  qpos, qnrm, q_t_hit, q_t_mir, qcam)
has = counts_c > 0
assert np.abs(means_c[has] - 1.0).max() < 1e-3, \
    "ring mean differs from the constant envmap"
for k in range(1, K):
    L = cone_mean(means_c, counts_c, k, RING_NOMINAL)
    assert np.abs(L - 1.0).max() < 1e-3, \
        f"L(k={k}) = {L.min():.4f}..{L.max():.4f} on a constant envmap"
assert np.abs(means_c[:, 0] - 1.0).max() < 1e-3, "mirror level ≠ envmap"
print("✓ constant envmap: every ring, every cone and the mirror are 1.0")

# ── Tests 4/5: gradient envmap, analytic reference ───────────────────────────
env_grad = make_envmap("gradient")
means_g, counts_g, R_g = bin_shared(sky_only, env_grad,
                                    qpos, qnrm, q_t_hit, q_t_mir, qcam)

# Test 5: mirror level ≡ envmap along R (continuous, f(dz) = 0.5 + 0.5·dz)
mirror_ref = 0.5 + 0.5 * R_g[:, 2]
err_mirror = np.abs(means_g[:, 0, 0] - mirror_ref).max()
assert err_mirror < 2e-2, f"mirror ≠ envmap(R): max err {err_mirror:.4f}"
print(f"✓ mirror level ≡ envmap(reflect(v,n)): max err {err_mirror:.4f} "
      f"(discretization {SKY_W}×{SKY_H})")

# Test 4: L(k) = 0.5 + 0.5·R_z·(1+cos b_k)/2 on the cones not cut by the horizon.
# The truncation has to be excluded because the closed form holds on the whole cone;
# the real bake truncates at the horizon instead (and is right to do so).
n_unit_g = qnrm[sky_only] / np.linalg.norm(qnrm[sky_only], axis=-1, keepdims=True)
cos_nr = (n_unit_g * R_g).sum(-1)
theta_R = np.degrees(np.arccos(np.clip(cos_nr, -1, 1)))

checked = 0
widest = None
for k in range(1, K):
    b = APERTURES[k] / 2.0
    unclipped = theta_R + b < 88.0                    # margin at the edge
    if unclipped.sum() < 20:
        continue
    L = cone_mean(means_g, counts_g, k, RING_NOMINAL)[unclipped, 0]
    ref = 0.5 + 0.5 * R_g[unclipped, 2] * (1.0 + COS_B[k]) / 2.0
    err = np.abs(L - ref).max()
    n_samp = S * (1.0 - COS_B[k])
    assert err < 0.05, (f"L(aperture {APERTURES[k]}°) deviates from the analytic: "
                        f"max err {err:.4f} over {int(unclipped.sum())} texels")
    print(f"    aperture {APERTURES[k]:5.0f}° (~{n_samp:5.0f} samples): "
          f"max err {err:.4f} over {int(unclipped.sum())} texels")
    checked += 1
    widest = (k, unclipped, err)

assert checked >= 4, f"only {checked} apertures checked against the analytic reference"
print(f"✓ analytic reference verified on {checked} apertures")

# Negative control: without the nominal counts in the meta the solver would use
# W_i = Ω_i, which with samples ALREADY proportional to Ω_i weighs the solid angle
# twice. The analytic test has to notice, otherwise it is not checking the weights
# but merely that the mean of something does something. The discrimination grows
# with the aperture, so it is measured at the widest truncation reached.
k_w, unclipped_w, err_ok = widest
L_bad = cone_mean(means_g, counts_g, k_w, None)[unclipped_w, 0]
ref_w = 0.5 + 0.5 * R_g[unclipped_w, 2] * (1.0 + COS_B[k_w]) / 2.0
err_bad = np.abs(L_bad - ref_w).max()
assert err_bad > 20.0 * max(err_ok, 1e-6), (
    f"wrong weights indistinguishable at {APERTURES[k_w]}°: "
    f"correct err {err_ok:.5f} vs wrong err {err_bad:.5f}")
print(f"✓ negative control at {APERTURES[k_w]:.0f}°: Ω_i weights (wrong) "
      f"give err {err_bad:.4f} against {err_ok:.4f} → the test discriminates "
      f"({err_bad / max(err_ok, 1e-6):.0f}×)")

# ── Test 6: cone closing and round trip of the on-disk format ────────────────
# The bake no longer saves the rings: it closes the cones and writes one RGB channel
# per candidate. Here we check that this path (raw sums → _cones_from_rings_np →
# IncrementalExrWriter → read_cones) gives the same L as cone_mean's weighted
# formula, i.e. the one validated by the analytic tests above.
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "cam_000.exr"
    rng = np.random.default_rng(0)
    n_tex = IUM_RES * IUM_RES
    ref_means  = rng.random((n_tex, K, 3), dtype=np.float32).astype(np.float64)
    ref_counts = rng.integers(0, 500, (n_tex, K)).astype(np.float64)
    # level 0 is a single ray: 0 when the camera is behind the surface
    ref_counts[:, 0] = rng.integers(0, 2, n_tex).astype(np.float64)
    ref_counts[:37] = 0.0                       # texels never seen by the camera
    ref_sum = ref_means * ref_counts[..., None]

    cone_w = ring_weights_mean(COS_B, K - 1, RING_NOMINAL)
    cones = _cones_from_rings_np(ref_sum, ref_counts, cone_w).astype(np.float32)
    n_valid = ref_counts.sum(axis=1)
    cones[n_valid <= 0] = 0.0

    wr = IncrementalExrWriter(p.as_posix(), IUM_RES, IUM_RES,
                              spec_cone_channels(APERTURES))
    rows = 16
    cones_img = cones.reshape(IUM_RES, IUM_RES, K, 3)
    valid_img = n_valid.astype(np.float32).reshape(IUM_RES, IUM_RES)
    names = [spec_cone_level_name(APERTURES, k) for k in range(K)]
    # The level names are what one reads in the viewer: they must carry the aperture,
    # contain no dots (in EXR channels the dot separates the layer from the channel)
    # and be in ascending alphabetical order, so that tev lists the layers in
    # aperture order rather than 10°, 100°, 120°, 15°…
    assert all("." not in n for n in names), f"dot in the level names: {names}"
    assert names == sorted(names), f"alphabetical order ≠ angular order: {names}"
    for k in range(1, K):
        assert str(int(APERTURES[k])) in names[k], \
            f"level {k} does not carry its aperture: {names[k]}"
    assert spec_cone_level_name([0.0, 7.5], 1) == "cone_007p5deg", \
        "fractional apertures: 'p' is needed instead of the decimal point"
    for r0 in range(0, IUM_RES, rows):
        block = {}
        for k in range(K):
            for ci, c in enumerate("RGB"):
                block[f"{names[k]}.{c}"] = cones_img[r0:r0 + rows, :, k, ci]
        block["valid"] = valid_img[r0:r0 + rows]
        wr.write_block(block)
    wr.close()

    got_cones, got_valid = read_cones(p, APERTURES)
    assert np.array_equal(got_valid, n_valid.astype(np.float32)), \
        "valid counts altered by the round trip"
    seen = n_valid > 0
    # the bake's cumsum must reproduce the weighted formula, within the on-disk half
    for k in range(1, K):
        ref = cone_mean(ref_means[seen], ref_counts[seen], k, RING_NOMINAL)
        err = np.abs(got_cones[seen][:, k] - ref).max()
        assert err < 2e-3, f"cone {APERTURES[k]:g}°: gap of {err:.2e} from cone_mean"
    # The mirror exists only where the ray was launched: where it was not, level 0
    # stays 0 (the same convention as the old ring bake, which filled the means only
    # on the levels with at least one sample).
    has_mirror = seen & (ref_counts[:, 0] > 0)
    assert np.abs(got_cones[has_mirror][:, 0] - ref_means[has_mirror][:, 0]).max() < 2e-3, \
        "mirror altered"
    assert (got_cones[seen & (ref_counts[:, 0] == 0)][:, 0] == 0).all(), \
        "non-zero mirror with no ray launched"
    assert (got_cones[~seen] == 0).all(), "texels with no samples not zeroed"
print("✓ cone closing ≡ cone_mean + round trip IncrementalExrWriter → read_cones")

print("\n✓ all HemiVis tests passed")
