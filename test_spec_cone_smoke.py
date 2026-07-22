"""Smoke test del pass SpecCone, envmap-only (nessun NeRF richiesto).

Verifiche:
  1. il pass gira su tutti i tile e i buffer hanno shape/range coerenti;
  2. con envmap costante, la media di ogni anello dei texel senza hit vale
     esattamente il valore dell'envmap;
  3. con envmap funzione dell'elevazione, il livello 0 (raggio specchio)
     coincide con l'envmap valutata lungo R = reflect(v, n) calcolato in
     NumPy (valida direzione riflessa + convenzione equirettangolare);
  4. la ricostruzione a coni: la media ad angolo solido sull'intera semisfera
     (pesi inline) coincide con la formula cumulativa storica, e lo smoothing
     è monotono dallo specchio ai coni (media pura) via via più larghi;
  5. la media pura sul cono (ring_weights_mean di pbr_solver.py, normalizzata
     sui campioni validi): su envmap costante ≡ 1 e anelli non tagliati
     dall'orizzonte vale esattamente 1 per ogni apertura, come il livello
     specchio (unità di radianza omogenee tra tutti i candidati);
  6. campioni per anello NON uniformi: il kernel lancia davvero N_i raggi per
     anello, e la ricostruzione con pesi Ω_i/N_i riproduce il riferimento
     analitico su envmap lineare. Quest'ultimo è l'unico check che smaschera
     un peso sbagliato: su envmap COSTANTE ogni mean_i vale 1, quindi qualsiasi
     combinazione convessa dà 1 e i pesi non sono osservabili.

Uso:  python test_spec_cone_smoke.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import OptixProgrammablePasses as optix

REPO = Path(__file__).resolve().parents[1] / "OptixProjectCMake"
MODEL = REPO / "Scenes" / "SwordShield" / "Models" / "SwordShield.obj"

APERTURES = [0.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0]
K = len(APERTURES)
RS_UNIFORM = [16] * (K - 1)                              # campionamento storico
RS_VAR = [8, 8, 12, 16, 24, 32, 40, 48, 56, 64]          # crescente verso l'esterno
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


def run_pass(envmap, cam_pos, ium_res_obj, num_pix, ring_samples=None):
    gen = optix.SpecConeGenerator()
    gen.set_traversable(model)
    gen.set_inputs(ium_res_obj, APERTURES,
                   RS_UNIFORM if ring_samples is None else ring_samples, 1024)
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

total_expected = 1 + sum(RS_UNIFORM)
assert valid[~masks].sum() == 0, "texel mascherati hanno campioni"
assert valid.sum(axis=1).max() <= total_expected, "valid_count oltre i campioni lanciati"
assert (valid[:, 1:] <= np.asarray(RS_UNIFORM)[None, :]).all(), \
    "un anello ha più campioni validi di quanti ne siano stati lanciati"
assert (valid[:, 0] <= 1).all(), "il livello 0 non è un singolo raggio"

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
from images_generator import ring_weights_mean

cos_b = np.cos(np.radians(np.asarray(APERTURES)) * 0.5)
sky_all = masks & (hit_cnt.sum(axis=1) == 0) & (valid > 0).all(axis=1)
means = ring_sum[sky_all] / valid[sky_all][..., None]              # (n, K, 3)
cnts  = valid[sky_all].astype(np.float64)                          # (n, K)

def lobe(w):
    """L = Σ_i W_i·mean_i·valid_i / Σ_i W_i·valid_i sugli anelli 1..K-1."""
    wc = w[None, :] * cnts[:, 1:]
    return (np.einsum("nk,nkc->nc", wc, means[:, 1:])
            / wc.sum(axis=1)[:, None])

# (a) consistenza del bake: media ad angolo solido sull'intera semisfera
#     (pesi inline = ring_weights_mean non troncato) ≡ formula cumulativa storica
omega = ring_weights_mean(cos_b, K - 1)          # = 2π(c_{i-1} − c_i), nessun taglio
w_sum = np.cumsum(omega[None, :, None] * ring_sum[sky_all][:, 1:], axis=1)
w_cnt = np.cumsum(omega[None, :] * valid[sky_all][:, 1:], axis=1)
L_old = w_sum[:, -1] / w_cnt[:, -1, None]
L_new = lobe(omega)
err = np.abs(L_new - L_old).max()
print(f"[lobi] media angolo solido (180°) vs cumulativa storica: max err={err:.2e}")
assert err < 1e-9, "media ad angolo solido ≠ formula cumulativa storica"

# (b) smoothing monotono: specchio → coni (media pura) via via più larghi (std
#     decrescente; la media è già normalizzata, stesse unità a ogni apertura)
spreads = [np.std(means[:, 0, 0])]
for k in range(1, K):
    spreads.append(np.std(lobe(ring_weights_mean(cos_b, k))[:, 0]))
print("[lobi] std specchio→180°: " + "  ".join(f"{x:.4f}" for x in spreads))
assert spreads[-1] < spreads[0], "il lobo largo dovrebbe mediare (std minore)"
assert all(spreads[i + 1] <= spreads[i] * 1.05 for i in range(len(spreads) - 1)), \
    "lo smoothing dovrebbe essere ~monotono con l'allargarsi del lobo"

# Profondità di taglio per texel: kmax = ultimo anello NON tagliato dall'orizzonte.
# L'anello k è integro se ha esattamente gli N_k campioni lanciati. Nota: l'anello
# più esterno arriva a 90° da R, quindi il suo campione più esterno cade sempre
# sull'orizzonte e kmax non raggiunge mai K-1: è geometria, non un difetto.
def cut_depth(valid_arr, rs):
    uncut = valid_arr[:, 1:] == np.asarray(rs)[None, :]
    return np.where(uncut.all(axis=1), K - 1, uncut.argmin(axis=1))


def lobe_of(w, means_sel, cnts_sel):
    """L = Σ_i W_i·mean_i·valid_i / Σ_i W_i·valid_i sugli anelli 1..K-1."""
    wc = w[None, :] * cnts_sel[:, 1:]
    return (np.einsum("nk,nkc->nc", wc, means_sel[:, 1:])
            / wc.sum(axis=1)[:, None])


# (c) media pura su envmap costante ≡ 1: ogni cono troncato deve valere
#     esattamente 1, come il livello specchio (unità di radianza omogenee).
#     ATTENZIONE: questo NON verifica i pesi — con mean_i ≡ 1 qualsiasi
#     combinazione convessa dà 1. Verifica solo l'omogeneità delle unità
#     (media contro integrale). Il check sui pesi è il blocco (d).
ring_sum_c, valid_c, hit_cnt_c = run_pass(make_envmap("constant"), cam_pos,
                                          ium_res, num_pix)
sky_c = masks & (hit_cnt_c.sum(axis=1) == 0) & (valid_c[:, 0] > 0)
kmax_c = cut_depth(valid_c, RS_UNIFORM)
means_c = ring_sum_c / np.maximum(valid_c[..., None], 1)
cnts_c  = valid_c.astype(np.float64)

err0 = np.abs(means_c[sky_c, 0] - 1.0).max()
print(f"[lobi] specchio su envmap costante: max |L - 1| = {err0:.2e}")
assert err0 < 1e-5, "specchio ≠ envmap costante"

tested = 0
for k in range(1, K):
    sel = sky_c & (kmax_c >= k)
    if sel.sum() == 0:
        print(f"[lobi] media pura r={APERTURES[k]:g}°: nessun texel con anelli "
              f"1..{k} integri, skip")
        continue
    err = np.abs(lobe_of(ring_weights_mean(cos_b, k), means_c[sel], cnts_c[sel])
                 - 1.0).max()
    print(f"[lobi] media pura r={APERTURES[k]:g}° su {sel.sum()} texel: "
          f"atteso 1, max err={err:.2e}")
    assert err < 1e-5, f"media pura r={APERTURES[k]}° ≠ 1 su envmap costante"
    tested += 1
assert tested >= 3, "il check sulla media pura ha coperto troppo pochi troncamenti"

# ── (d) Campioni per anello NON uniformi + riferimento analitico ─────────────
# Su make_envmap("gradient") vale env(d) = 0.5 + 0.5·d_z, LINEARE in d, quindi
# la media pura sulla calotta di semi-apertura b attorno a R ha forma chiusa:
#     L(k) = 0.5 + 0.5 · R_z · (1 + cos b_k)/2
# La ricostruzione discreta la riproduce esattamente: la stratificazione
# midpoint dà media (c_{i-1}+c_i)/2 per anello e con pesi ∝ (c_{i-1}−c_i) la
# somma telescopa in (1−c_k²)/2 / (1−c_k) = (1+c_k)/2. Con pesi Ω_i·N_i (cioè
# senza dividere per i campioni lanciati) la telescopia salta: è questo il
# check che distingue un peso corretto da uno sbagliato.
ring_sum_v, valid_v, hit_cnt_v = run_pass(make_envmap("gradient"), cam_pos,
                                          ium_res, num_pix, RS_VAR)

rs_v = np.asarray(RS_VAR)
assert (valid_v[:, 1:] <= rs_v[None, :]).all(), \
    "valid_count oltre i campioni lanciati con N variabile"
reached = valid_v[:, 1:].max(axis=0)
print(f"[var] N_i richiesti      : {RS_VAR}")
print(f"[var] max valid raggiunto: {reached.tolist()}")
assert (reached[:-1] == rs_v[:-1]).all(), \
    "un anello interno non raggiunge N_i: il kernel non legge ring_samples[i]"

# R = reflect(v, n) per tutti i texel, per il riferimento analitico
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
        print(f"[analitico] r={APERTURES[k]:g}°: solo {sel.sum()} texel, skip")
        continue
    ref = 0.5 + 0.5 * Rz[sel] * (1.0 + cos_b[k]) / 2.0
    ok_w = lobe_of(ring_weights_mean(cos_b, k, RS_VAR),
                   means_v[sel], cnts_v[sel])[:, 0]
    bad_w = lobe_of(ring_weights_mean(cos_b, k, None),
                    means_v[sel], cnts_v[sel])[:, 0]
    e_ok, e_bad = np.abs(ok_w - ref).max(), np.abs(bad_w - ref).max()
    print(f"[analitico] r={APERTURES[k]:g}° su {sel.sum():5d} texel: "
          f"err(Ω/N)={e_ok:.2e}   err(Ω, sbagliato)={e_bad:.2e}   "
          f"rapporto={e_bad / max(e_ok, 1e-12):.1f}×")
    assert e_ok < 3e-2, \
        f"media pura r={APERTURES[k]}° ≠ riferimento analitico: pesi o " \
        f"stratificazione errati"
    assert e_ok <= e_bad * 1.05, \
        f"a r={APERTURES[k]}° il peso Ω_i/N_i è PEGGIORE di quello non normalizzato"
    ratios.append((k, e_bad / max(e_ok, 1e-12)))

assert len(ratios) >= 4, "il riferimento analitico ha coperto troppo pochi troncamenti"
# A coni stretti gli anelli vedono radianze quasi identiche e il peso è quasi
# irrilevante: la discriminazione esiste solo sui coni larghi, dove l'anello
# esterno pesa molto e ha molti più campioni degli interni.
k_wide, ratio_wide = ratios[-1]
print(f"[analitico] discriminazione al troncamento più largo testato "
      f"(r={APERTURES[k_wide]:g}°): {ratio_wide:.1f}×")
assert ratio_wide > 3.0, \
    "sul cono più largo la normalizzazione per N_i non migliora in modo " \
    "significativo: la correzione Ω_i/N_i non è attiva"

print("\n✓ Tutti i check del pass SpecCone superati")
