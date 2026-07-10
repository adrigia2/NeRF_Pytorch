"""Smoke test del pass SpecCone, envmap-only (nessun NeRF richiesto).

Verifiche:
  1. il pass gira su tutti i tile e i buffer hanno shape/range coerenti;
  2. con envmap costante, la media di ogni anello dei texel senza hit vale
     esattamente il valore dell'envmap;
  3. con envmap funzione dell'elevazione, il livello 0 (raggio specchio)
     coincide con l'envmap valutata lungo R = reflect(v, n) calcolato in
     NumPy (valida direzione riflessa + convenzione equirettangolare);
  4. la ricostruzione a lobi di pbr_solver (ring_weights_*): il top-hat
     completo coincide con la formula cumulativa storica, e lo smoothing
     è monotono dallo specchio al lobo s=0.

Uso:  python test_spec_cone_smoke.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import OptixProgrammablePasses as optix

REPO = Path(__file__).resolve().parents[1] / "OptixProjectCMake"
MODEL = REPO / "Scenes" / "SwordShield" / "Models" / "SwordShield.obj"

APERTURES = [0.0, 10.0, 25.0, 50.0, 90.0, 130.0, 180.0]
K = len(APERTURES)
M = 16
IUM_RES = 64
SKY_W, SKY_H = 256, 128


def make_envmap(mode: str) -> np.ndarray:
    """Envmap equirettangolare (H*W, 3) float32. Convenzione di sampleEnvmap:
    u = 0.5 + atan2(dx, -dy)/2π,  v = 0.5 - asin(dz)/π  (Z-up)."""
    if mode == "constant":
        return np.full((SKY_H * SKY_W, 3), 1.0, dtype=np.float32)
    v = (np.arange(SKY_H) + 0.5) / SKY_H
    dz = np.sin((0.5 - v) * np.pi)                       # elevazione per riga
    rows = (0.5 + 0.5 * dz).astype(np.float32)           # f(dz) ∈ [0,1]
    env = np.repeat(rows[:, None], SKY_W, axis=1)
    return np.stack([env, env, env], axis=-1).reshape(-1, 3)


def env_value(d: np.ndarray) -> np.ndarray:
    """f(dz) dell'envmap 'gradient' valutata in direzioni (N,3) (continua)."""
    return 0.5 + 0.5 * d[:, 2]


def run_pass(envmap, cam_pos, ium_res_obj, num_pix):
    gen = optix.SpecConeGenerator()
    gen.set_traversable(model)
    gen.set_inputs(ium_res_obj, APERTURES, M, 1024)
    gen.set_envmap(envmap, [SKY_W, SKY_H], 0.0)
    gen.set_camera(list(cam_pos), np.empty(0, dtype=np.uint8))  # tutti visibili

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
            assert li.min() >= 0 and li.max() < tt, "local_idx fuori range"
            assert ri.min() >= 0 and ri.max() < K, "ring_idx fuori range"
            np.add.at(hit_cnt, (off + li, ri), 1)
    return ring_sum, valid, hit_cnt


print(f"Carico modello: {MODEL}")
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
print(f"IUM {IUM_RES}×{IUM_RES}: {masks.sum()} texel validi")

center = pos[masks].mean(axis=0)
diag = np.linalg.norm(pos[masks].max(0) - pos[masks].min(0))
cam_pos = center + np.array([0.6, -1.2, 0.9]) / np.linalg.norm([0.6, -1.2, 0.9]) * (2.5 * diag)

# ── Test 2: envmap costante ───────────────────────────────────────────────────
ring_sum, valid, hit_cnt = run_pass(make_envmap("constant"), cam_pos, ium_res, num_pix)

total_expected = 1 + (K - 1) * M
assert valid[~masks].sum() == 0, "texel mascherati hanno campioni"
assert valid.max() <= total_expected, "valid_count oltre il numero di campioni"

sky_only = masks & (hit_cnt.sum(axis=1) == 0) & (valid[:, 0] > 0)
print(f"[constant] texel solo-cielo: {sky_only.sum()} / {masks.sum()}")
L_levels = ring_sum[sky_only] / np.maximum(valid[sky_only, :, None], 1)
err = np.abs(L_levels[valid[sky_only] > 0] - 1.0).max() if sky_only.any() else 0.0
print(f"[constant] max |L - 1| sui livelli solo-cielo: {err:.2e}")
assert err < 1e-5, "envmap costante non riprodotta esattamente"

# ── Test 3: envmap gradiente, livello 0 vs reflect NumPy ─────────────────────
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
print(f"[gradient] specchio vs NumPy su {len(L0)} texel: "
      f"max err={err.max():.4f}, mean={err.mean():.4f}")
assert err.max() < 0.05, "livello 0 non coincide con envmap(reflect(v,n))"

# ── Ricostruzione a lobi (helper pbr_solver) su medie per-anello ─────────────
from pbr_solver import ring_weights_phong, ring_weights_tophat

cos_b = np.cos(np.radians(np.asarray(APERTURES)) * 0.5)
sky_all = masks & (hit_cnt.sum(axis=1) == 0) & (valid > 0).all(axis=1)
means = ring_sum[sky_all] / valid[sky_all][..., None]              # (n, K, 3)
cnts  = valid[sky_all].astype(np.float64)                          # (n, K)

def lobe(w):
    """L = Σ_i W_i·mean_i·valid_i / Σ_i W_i·valid_i sugli anelli 1..K-1."""
    wc = w[None, :] * cnts[:, 1:]
    return (np.einsum("nk,nkc->nc", wc, means[:, 1:])
            / wc.sum(axis=1)[:, None])

# (a) top-hat completo ≡ formula cumulativa storica
omega = 2.0 * np.pi * (cos_b[:-1] - cos_b[1:])
w_sum = np.cumsum(omega[None, :, None] * ring_sum[sky_all][:, 1:], axis=1)
w_cnt = np.cumsum(omega[None, :] * valid[sky_all][:, 1:], axis=1)
L_old = w_sum[:, -1] / w_cnt[:, -1, None]
L_new = lobe(ring_weights_tophat(cos_b, K - 1))
err = np.abs(L_new - L_old).max()
print(f"[lobi] tophat(180°) vs cumulativa storica: max err={err:.2e}")
assert err < 1e-9, "helper top-hat ≠ formula cumulativa storica"

# (b) smoothing monotono: specchio → lobi via via più larghi (std decrescente)
spreads = [np.std(means[:, 0, 0])]
for s in [200.0, 20.0, 2.0, 0.0]:
    spreads.append(np.std(lobe(ring_weights_phong(cos_b, s))[:, 0]))
print("[lobi] std specchio→s=0: " + "  ".join(f"{x:.4f}" for x in spreads))
assert spreads[-1] < spreads[0], "il lobo largo dovrebbe mediare (std minore)"
assert all(spreads[i + 1] <= spreads[i] * 1.05 for i in range(len(spreads) - 1)), \
    "lo smoothing dovrebbe essere ~monotono con l'allargarsi del lobo"

print("\n✓ Tutti i check del pass SpecCone superati")
