"""Test del pass HemiVis e della ricostruzione condivisa, envmap-only (niente NeRF).

Lo schema condiviso traccia UN set Fibonacci per texel e lascia che ogni camera
classifichi gli stessi raggi nel proprio anello. Il kernel restituisce solo i
t_hit, indicizzati per posizione: le direzioni vivono lato Python. Le verifiche
qui sotto coprono, in ordine di insidiosità:

  1. PARITÀ kernel↔torch delle direzioni (condivise e specchio). È il fallimento
     più pericoloso dell'intero schema: se le due formule divergono, ogni t_hit
     viene appaiato alla direzione sbagliata e non c'è alcun sintomo se non una
     L_j silenziosamente errata.
  2. Proprietà del set condiviso: tutte le direzioni sopra l'orizzonte, cosθ
     esattamente equispaziato (uniformità in angolo solido), rotazione azimutale
     diversa da texel a texel.
  3. Envmap costante: ogni media per anello e ogni L(k) ricostruita valgono
     esattamente il valore dell'envmap, livello specchio compreso (unità di
     radianza omogenee tra tutti i candidati).
  4. Envmap 'gradient' con riferimento ANALITICO: sui coni interamente sopra
     l'orizzonte L(k) = 0.5 + 0.5·R_z·(1+cos b_k)/2 in forma chiusa. È l'unico
     check con potere discriminante sui pesi: su envmap costante ogni media di
     anello vale 1, quindi qualsiasi combinazione convessa dà 1 e un peso
     sbagliato resta invisibile.
  5. Il livello specchio coincide con l'envmap valutata lungo R = reflect(v, n)
     calcolato in NumPy.
  6. Chiusura dei coni e formato su disco: somme grezze → _cones_from_rings_np →
     IncrementalExrWriter → read_cones deve dare la stessa L della formula
     pesata validata ai punti 3-4.

I test 3-5 girano su un quad planare generato al volo: sul modello reale, con
migliaia di direzioni per texel, quasi ogni texel trova qualcosa da colpire e i
raggi occlusi richiederebbero il NeRF, senza più un riferimento in forma chiusa.
Sul quad le direzioni stanno tutte nell'emisfero sopra la normale, quindi nessun
raggio può colpire il quad stesso.

Nota: il processo esce con codice diverso da zero allo shutdown dell'interprete
(problema di cleanup OptiX preesistente, non un fallimento del test — come in
test_spec_cone_smoke.py). Fa fede la riga "✓ tutti i test HemiVis passati".

Uso:  python test_hemivis_shared.py
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

REPO = Path(__file__).resolve().parents[1] / "OptixProjectCMake"
MODEL = REPO / "Scenes" / "SwordShield" / "Models" / "SwordShield.obj"

APERTURES = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 80.0,
             100.0, 120.0, 140.0, 160.0, 180.0]
K = len(APERTURES)
COS_B = np.cos(np.radians(np.asarray(APERTURES)) * 0.5)

S = 8192            # campioni condivisi per texel
IUM_RES = 64
TILE = 1024
SKY_W, SKY_H = 256, 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_envmap(mode: str) -> np.ndarray:
    """Envmap equirettangolare (H*W, 3) float32, convenzione di sampleEnvmap."""
    if mode == "constant":
        return np.full((SKY_H * SKY_W, 3), 1.0, dtype=np.float32)
    v = (np.arange(SKY_H) + 0.5) / SKY_H
    dz = np.sin((0.5 - v) * np.pi)                   # elevazione per riga
    rows = (0.5 + 0.5 * dz).astype(np.float32)
    env = np.repeat(rows[:, None], SKY_W, axis=1)
    return np.stack([env, env, env], axis=-1).reshape(-1, 3)


# ──────────────────────────────────────────────────────────────────────────────
# Setup scena
# ──────────────────────────────────────────────────────────────────────────────

print(f"Carico modello: {MODEL}")
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
print(f"IUM {IUM_RES}×{IUM_RES}: {masks.sum()} texel validi")

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

# ── Test 1: parità kernel ↔ torch delle direzioni ────────────────────────────
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

    # Raggio specchio: stessa formula del kernel, in torch
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
    f"direzioni condivise divergono tra kernel e torch: {max_err_shared:.2e}"
assert max_err_mirror < 1e-5, \
    f"direzioni specchio divergono tra kernel e torch: {max_err_mirror:.2e}"
print(f"✓ parità direzioni kernel↔torch: max err condivise {max_err_shared:.2e}, "
      f"specchio {max_err_mirror:.2e}")

# ── Test 2: proprietà del set condiviso ──────────────────────────────────────
probe = np.flatnonzero(masks & (np.linalg.norm(nrm_np, axis=-1) > 1e-8))[:64]
nrm_p = torch.as_tensor(nrm_np[probe], device=device)
dirs_p = _hemivis_directions(nrm_p, probe, S)
n_unit_p = nrm_p / torch.linalg.norm(nrm_p, dim=-1, keepdim=True)
cos_n = (dirs_p * n_unit_p[:, None, :]).sum(-1).cpu().numpy()

assert cos_n.min() > 0.0, "un campione condiviso cade sotto l'orizzonte"
expected_cos = 1.0 - (np.arange(S) + 0.5) / S
assert np.abs(np.sort(cos_n, axis=1)[:, ::-1] - expected_cos).max() < 1e-5, \
    "cosθ non equispaziato: il set non è uniforme in angolo solido"
norms = np.linalg.norm(dirs_p.cpu().numpy(), axis=-1)
assert np.abs(norms - 1.0).max() < 1e-5, "direzioni non normalizzate"

rots = _hemivis_rotation(probe)
assert 0.0 <= rots.min() and rots.max() < 1.0, "rotazione fuori da [0,1)"
assert len(np.unique(rots)) > len(probe) * 0.9, \
    "la rotazione azimutale non decorrela i texel"
print(f"✓ set condiviso uniforme in angolo solido, {len(np.unique(rots))} "
      f"rotazioni distinte su {len(probe)} texel")


# ──────────────────────────────────────────────────────────────────────────────
# Ricostruzione dei coni dai raggi condivisi (la stessa che fa il bake)
# ──────────────────────────────────────────────────────────────────────────────

def bin_shared(idx: np.ndarray, envmap: np.ndarray,
               pos_np, nrm_np, t_hit, t_mir, cam_pos):
    """(means (n, K, 3), counts (n, K), R (n, 3)) per i texel `idx`, camera unica.

    Replica il binning di _precompute_spec_cone_shared: i raggi miss prendono
    l'envmap, il livello 0 è il raggio specchio, i campioni oltre il cono più
    largo vengono scartati.
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

    # Livello 0: raggio specchio
    thm = torch.as_tensor(t_mir[idx, 0], device=device)
    mir_ok = (thm == 0.0)                       # miss → envmap; hit → serve il NeRF
    if bool(mir_ok.any()):
        sums[mir_ok, 0] = _sample_envmap_torch(R[mir_ok], env_t, [SKY_W, SKY_H], 0.0)
        counts[mir_ok, 0] = 1.0

    return (sums / torch.clamp(counts, min=1.0)[..., None]).cpu().numpy(), \
           counts.cpu().numpy(), R.cpu().numpy()


def cone_mean(means, counts, k, ring_samples):
    """L(k) come la calcola pbr_solver: media pesata degli anelli 1..k."""
    w = ring_weights_mean(COS_B, k, ring_samples)          # (K-1,)
    wc = w[None, :] * counts[:, 1:]
    num = np.einsum("nk,nkc->nc", wc, means[:, 1:])
    return num / np.maximum(wc.sum(axis=1), 1e-12)[:, None]


RING_NOMINAL = np.asarray(spec_cone_shared_ring_samples(APERTURES, S))

# ──────────────────────────────────────────────────────────────────────────────
# Scena piana per i test analitici
#
# Sul modello reale, con S=8192 direzioni per texel, praticamente ogni texel
# trova qualcosa da colpire: i raggi occlusi richiederebbero il NeRF e non c'è
# più un riferimento in forma chiusa. Un quad planare risolve il problema alla
# radice: le direzioni stanno tutte nell'emisfero sopra la normale, quindi NESSUN
# raggio può colpire il quad stesso e ogni campione vede l'envmap.
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
# camera a distanza finita e fuori asse: R_z varia tra i texel, altrimenti il
# riferimento analitico sarebbe costante e non discriminerebbe nulla
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
print(f"quad planare: {qmask.sum()} texel, di cui {sky_only.size} senza occlusioni")
assert sky_only.size > 500, \
    f"il quad dovrebbe essere privo di auto-occlusioni, trovati {sky_only.size} texel"

# ── Test 3: envmap costante ──────────────────────────────────────────────────
means_c, counts_c, _ = bin_shared(sky_only, make_envmap("constant"),
                                  qpos, qnrm, q_t_hit, q_t_mir, qcam)
has = counts_c > 0
assert np.abs(means_c[has] - 1.0).max() < 1e-3, \
    "media di anello diversa dall'envmap costante"
for k in range(1, K):
    L = cone_mean(means_c, counts_c, k, RING_NOMINAL)
    assert np.abs(L - 1.0).max() < 1e-3, \
        f"L(k={k}) = {L.min():.4f}..{L.max():.4f} su envmap costante"
assert np.abs(means_c[:, 0] - 1.0).max() < 1e-3, "livello specchio ≠ envmap"
print("✓ envmap costante: ogni anello, ogni cono e lo specchio valgono 1.0")

# ── Test 4/5: envmap gradient, riferimento analitico ─────────────────────────
env_grad = make_envmap("gradient")
means_g, counts_g, R_g = bin_shared(sky_only, env_grad,
                                    qpos, qnrm, q_t_hit, q_t_mir, qcam)

# Test 5: livello specchio ≡ envmap lungo R (continua, f(dz) = 0.5 + 0.5·dz)
mirror_ref = 0.5 + 0.5 * R_g[:, 2]
err_mirror = np.abs(means_g[:, 0, 0] - mirror_ref).max()
assert err_mirror < 2e-2, f"specchio ≠ envmap(R): err max {err_mirror:.4f}"
print(f"✓ livello specchio ≡ envmap(reflect(v,n)): err max {err_mirror:.4f} "
      f"(discretizzazione {SKY_W}×{SKY_H})")

# Test 4: L(k) = 0.5 + 0.5·R_z·(1+cos b_k)/2 sui coni non tagliati dall'orizzonte.
# Il taglio va escluso perché la formula chiusa vale sul cono intero; il bake
# reale invece tronca all'orizzonte (ed è corretto che lo faccia).
n_unit_g = qnrm[sky_only] / np.linalg.norm(qnrm[sky_only], axis=-1, keepdims=True)
cos_nr = (n_unit_g * R_g).sum(-1)
theta_R = np.degrees(np.arccos(np.clip(cos_nr, -1, 1)))

checked = 0
widest = None
for k in range(1, K):
    b = APERTURES[k] / 2.0
    unclipped = theta_R + b < 88.0                    # margine sul bordo
    if unclipped.sum() < 20:
        continue
    L = cone_mean(means_g, counts_g, k, RING_NOMINAL)[unclipped, 0]
    ref = 0.5 + 0.5 * R_g[unclipped, 2] * (1.0 + COS_B[k]) / 2.0
    err = np.abs(L - ref).max()
    n_samp = S * (1.0 - COS_B[k])
    assert err < 0.05, (f"L(apertura {APERTURES[k]}°) devia dall'analitico: "
                        f"err max {err:.4f} su {int(unclipped.sum())} texel")
    print(f"    apertura {APERTURES[k]:5.0f}° (~{n_samp:5.0f} campioni): "
          f"err max {err:.4f} su {int(unclipped.sum())} texel")
    checked += 1
    widest = (k, unclipped, err)

assert checked >= 4, f"solo {checked} aperture verificate contro l'analitico"
print(f"✓ riferimento analitico verificato su {checked} aperture")

# Controllo negativo: senza i conteggi nominali nel meta il solver userebbe
# W_i = Ω_i, che con campioni GIÀ proporzionali a Ω_i pesa l'angolo solido due
# volte. Il test analitico deve accorgersene, altrimenti non sta verificando i
# pesi ma solo che la media di qualcosa fa qualcosa. La discriminazione cresce
# con l'apertura, quindi si misura al troncamento più largo raggiunto.
k_w, unclipped_w, err_ok = widest
L_bad = cone_mean(means_g, counts_g, k_w, None)[unclipped_w, 0]
ref_w = 0.5 + 0.5 * R_g[unclipped_w, 2] * (1.0 + COS_B[k_w]) / 2.0
err_bad = np.abs(L_bad - ref_w).max()
assert err_bad > 20.0 * max(err_ok, 1e-6), (
    f"pesi sbagliati indistinguibili a {APERTURES[k_w]}°: "
    f"err corretto {err_ok:.5f} vs err sbagliato {err_bad:.5f}")
print(f"✓ controllo negativo a {APERTURES[k_w]:.0f}°: pesi Ω_i (sbagliati) "
      f"danno err {err_bad:.4f} contro {err_ok:.4f} → il test discrimina "
      f"({err_bad / max(err_ok, 1e-6):.0f}×)")

# ── Test 6: chiusura dei coni e round-trip del formato su disco ──────────────
# Il bake non salva più gli anelli: chiude i coni e scrive un canale RGB per
# candidato. Qui si verifica che quel percorso (somme grezze →
# _cones_from_rings_np → IncrementalExrWriter → read_cones) dia la stessa L che
# la formula pesata di cone_mean, cioè quella validata dai test analitici sopra.
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "cam_000.exr"
    rng = np.random.default_rng(0)
    n_tex = IUM_RES * IUM_RES
    ref_means  = rng.random((n_tex, K, 3), dtype=np.float32).astype(np.float64)
    ref_counts = rng.integers(0, 500, (n_tex, K)).astype(np.float64)
    # il livello 0 è un raggio solo: 0 quando la camera è dietro la superficie
    ref_counts[:, 0] = rng.integers(0, 2, n_tex).astype(np.float64)
    ref_counts[:37] = 0.0                       # texel mai visti dalla camera
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
    # I nomi dei livelli sono ciò che si legge nel viewer: devono portare
    # l'apertura, non contenere punti (nei canali EXR il punto separa il layer
    # dal canale) e stare in ordine alfabetico crescente, così tev elenca i
    # layer in ordine di apertura invece che 10°, 100°, 120°, 15°…
    assert all("." not in n for n in names), f"punto nei nomi dei livelli: {names}"
    assert names == sorted(names), f"ordine alfabetico ≠ ordine angolare: {names}"
    for k in range(1, K):
        assert str(int(APERTURES[k])) in names[k], \
            f"il livello {k} non porta la sua apertura: {names[k]}"
    assert spec_cone_level_name([0.0, 7.5], 1) == "cone_007p5deg", \
        "aperture frazionarie: serve 'p' al posto del punto decimale"
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
        "conteggi validi alterati dal round-trip"
    seen = n_valid > 0
    # la cumsum del bake deve riprodurre la formula pesata, entro il half su disco
    for k in range(1, K):
        ref = cone_mean(ref_means[seen], ref_counts[seen], k, RING_NOMINAL)
        err = np.abs(got_cones[seen][:, k] - ref).max()
        assert err < 2e-3, f"cono {APERTURES[k]:g}°: scarto {err:.2e} da cone_mean"
    # Lo specchio esiste solo dove il raggio è stato lanciato: dove non c'è, il
    # livello 0 resta 0 (stessa convenzione del vecchio bake per anelli, che
    # riempiva le medie solo sui livelli con almeno un campione).
    has_mirror = seen & (ref_counts[:, 0] > 0)
    assert np.abs(got_cones[has_mirror][:, 0] - ref_means[has_mirror][:, 0]).max() < 2e-3, \
        "specchio alterato"
    assert (got_cones[seen & (ref_counts[:, 0] == 0)][:, 0] == 0).all(), \
        "specchio non nullo senza raggio lanciato"
    assert (got_cones[~seen] == 0).all(), "texel senza campioni non azzerati"
print("✓ chiusura dei coni ≡ cone_mean + round-trip IncrementalExrWriter → read_cones")

print("\n✓ tutti i test HemiVis passati")
