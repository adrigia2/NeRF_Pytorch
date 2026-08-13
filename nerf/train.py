from __future__ import annotations

import datetime
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoint import _build_models, _check_ray_convention, save_checkpoint
from .config import NerfConfig
from .csv_logger import CsvLogger
from .dataset import NerfDataset
from .render import render_image, render_rays_depth, render_unified


def _rel_mse(pred, target, eps=1e-3):
    """Relative MSE (variante con eps fuori dal quadrato): divide per pred²+eps.
    Non è la formulazione esatta di RawNeRF — vedi _rel_mse_raw per quella."""
    return (((pred - target) ** 2) / (pred.detach() ** 2 + eps)).mean()

def _rel_mse_raw(pred, target, eps=1e-3):
    """Relative MSE (RawNeRF fedele): pesa ogni pixel per l'inverso del quadrato
    dell'intensità predetta, con eps dentro al quadrato → (pred+eps)².
    Garantisce fedeltà relativa uniforme su tutto il range dinamico.
    Cfr. google-research/multinerf, internal/train_utils.py."""
    return (((pred - target) ** 2) / (pred.detach() + eps) ** 2).mean()

def _mse(pred, target):
    return ((pred - target) ** 2).mean()

def _log_l1(pred, target):
    """L1 in log1p space: compresses highlights, boosts shadow gradients."""
    return F.l1_loss(torch.log1p(pred), torch.log1p(target))

LOSSES = {
    "l1":          F.l1_loss,
    "mse":         _mse,
    "rel_mse":     _rel_mse,      # variante: eps fuori dal quadrato
    "rel_mse_raw": _rel_mse_raw,  # RawNeRF fedele: eps dentro al quadrato
    "log_l1":      _log_l1,
}


# Colonne di <output_dir>/training_metrics.csv, una riga per display block.
#
# Quali confronti sono leciti:
#   - ``loss`` è nelle unità di cfg.loss_type: confrontarla tra run con loss
#     diverse non ha significato, serve a leggere la convergenza dentro un run.
#   - ``mse``/``psnr_db`` sono calcolate sempre, indipendentemente dalla loss
#     usata per il gradiente: sono le uniche colonne confrontabili tra run con
#     loss diverse.
#   - Il dominio è il BATCH di training dell'iterazione, non un'immagine: il
#     batch è una fetta di una permutazione dell'intero pool di raggi, quindi
#     foreground e background sono mescolati nelle proporzioni del dataset. Non è
#     in nessun senso una metrica held-out.
#   - La media è sugli ultimi ``display_every`` iterazioni (la finestra delle
#     deque), NON sull'epoca: per l'aggregato per epoca c'è epoch_metrics.csv.
#   - psnr_db = -10·log10(mse) assume implicitamente MAX_I = 1, ma i target sono
#     HDR (valori > 1): è una rimappatura monotona della MSE, confrontabile tra
#     run sugli stessi dati ma NON con i valori di PSNR della letteratura, e
#     potenzialmente negativa se mse > 1. Con composite_white=False la GT dei
#     raggi di background sono i pixel reali dell'envmap, quindi le zone luminose
#     pesano quadraticamente e dominano la metrica rispetto al range diffuso.
#
# Le ultime quattro colonne sono costanti per run e ridondanti riga per riga, ma
# rendono il CSV auto-descrittivo: sono gli assi dello sweep, quindi concatenare
# N file in un solo dataframe basta a etichettare le curve. La scena non è nota
# qui ma è nel path (<output_root>/<scene>/nerf_train/).
CSV_FIELDS = [
    "iter", "epoch", "loss", "mse", "psnr_db", "lr",
    "iters_per_s", "rays_per_s", "acc_fg",
    "wall_s", "timestamp",
    "loss_type", "rgb_activation", "batch_size", "lr_decay_factor",
]


# Colonne di <output_dir>/epoch_metrics.csv, una riga per epoca completata.
#
# La differenza sostanziale rispetto a CSV_FIELDS è il dominio della media:
# qui ``loss`` e ``mse`` sono medie su TUTTE le iterazioni dell'epoca, cioè su un
# passaggio completo sul dataset, mentre in training_metrics.csv sono medie sugli
# ultimi ``display_every`` iterazioni (con 1266 iter/epoca e display_every=100,
# l'8 % dell'epoca). Sono quindi due granularità diverse e vanno tenute su due
# curve diverse, non concatenate.
#
# ``iters_in_epoch``/``rays_in_epoch`` non sono ridondanti: su un resume a metà
# epoca gli accumulatori partono da dove il segmento è ripreso, quindi la prima
# riga scritta dopo un resume copre solo la coda dell'epoca e ha
# iters_in_epoch < iters_per_epoch. È il comportamento atteso, e queste due
# colonne sono ciò che lo rende leggibile a posteriori.
EPOCH_CSV_FIELDS = [
    "epoch", "iter_end", "iters_in_epoch", "rays_in_epoch",
    "loss", "mse", "psnr_db", "lr",
    "iters_per_s", "rays_per_s",
    "epoch_wall_s", "timestamp",
    "loss_type", "rgb_activation", "batch_size", "lr_decay_factor",
]


def train(transforms_path: str, cfg: NerfConfig, *, ckpt_path: str, output_dir: str,
          num_iters: int, batch_size: int, lr: float, seed: int,
          display_every: int, tb_logger=None) -> float:
    """Train a single-network depth-guided NeRF from transforms_extended.json.

    Uses render_unified with a single batch holding fg+bg together (via in_mask).
    Rays are drawn by epochs: ``dataset.sample_epoch`` walks a fresh permutation
    of the whole ray pool once per epoch, so every ray is seen exactly once per
    epoch and only the last batch of an epoch is shorter than ``batch_size``.
    ``num_iters`` stays the training budget — the epoch is the sampling
    structure, not the unit of the schedule.

    Saves a checkpoint to ckpt_path; resumes if it already exists.  The position
    inside the epoch is derived from the absolute iteration index, so a resume
    needs no extra state in the checkpoint.

    Returns the PSNR (dB) from the last display block, or float('nan') if the
    training ran for fewer than ``display_every`` iterations.
    ``tb_logger`` accepts a monitoring.RunLogger (or None to disable TB logging).

    The display block fires every ``display_every`` iterations, at the end of
    every epoch, and on the last iteration of the call.  Each one appends a row
    to ``<output_dir>/training_metrics.csv`` (columns and their semantics: see
    CSV_FIELDS), and the ones landing on an epoch boundary also append a row to
    ``<output_dir>/epoch_metrics.csv`` (see EPOCH_CSV_FIELDS) — same run, two
    granularities.  Both files are append-only, so resumes and interactive
    continuations accumulate instead of overwriting.
    Note that if the process dies *after* a display block but *before* the next
    checkpoint, the restart rewinds ``iter_start`` and the CSV ends up with
    duplicate/non-monotonic iterations; downstream that is a
    ``drop_duplicates(subset="iter", keep="last")`` after sorting on timestamp.
    Deduplicating at write time would mean re-reading the file on every row.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if cfg.loss_type not in LOSSES:
        raise ValueError(f"Unknown loss_type: {cfg.loss_type!r} "
                         f"(expected one of {sorted(LOSSES)})")
    loss_fn = LOSSES[cfg.loss_type]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  NeRF training on: {device}  "
          f"(activation={cfg.rgb_activation}, loss={cfg.loss_type})")

    dataset = NerfDataset(transforms_path, device, composite_white=False)

    if not dataset.has_depth_split:
        raise RuntimeError(
            "Dataset has no foreground/background depth split. "
            "Ensure transforms_extended.json includes depth_path for every frame."
        )

    # Ordine a epoche: una permutazione del pool intero per epoca, consumata un
    # batch alla volta. Sostituisce il campionamento con reinserimento, che non
    # dava nessuna garanzia di copertura (la frazione di raggi mai visti dopo T
    # iterazioni era exp(-T·batch/n_rays)).
    iters_per_epoch = dataset.configure_epochs(batch_size, seed)
    tail = dataset.n_rays - (iters_per_epoch - 1) * batch_size
    print(f"  Epoche: {dataset.n_rays} raggi / batch {batch_size} = "
          f"{iters_per_epoch} iter/epoca (ultimo batch: {tail} raggi)  —  "
          f"{num_iters} iter ≈ {num_iters / iters_per_epoch:.2f} epoche")
    if iters_per_epoch < display_every:
        print(f"  [warn] iters_per_epoch ({iters_per_epoch}) < display_every "
              f"({display_every}): il display block scatta a ogni fine epoca, "
              f"quindi più spesso di display_every (preview EXR + checkpoint ogni volta)")

    # La sfera di background è ancorata all'ORIGINE del mondo: l'ambiente è una funzione
    # puramente direzionale, quindi il centro è una convenzione, e sceglierlo fisso lo
    # rende riproducibile invece che dipendente dal set di camere. La geometria serve
    # solo a dimensionare il raggio.
    scene_radius, p_min, p_max = dataset.compute_scene_bounds()
    scene_radius = float(scene_radius)
    center = torch.zeros(3, device=device, dtype=torch.float32)
    sphere_radius = float(cfg.bg_radius_mult * scene_radius)
    cfg.far = sphere_radius + cfg.bg_depth_window_end + 1.0

    print(f"  Scene bbox: {[round(v, 3) for v in p_min.tolist()]} .. "
          f"{[round(v, 3) for v in p_max.tolist()]}")
    print(f"  Scene radius (from origin): {scene_radius:.3f}  "
          f"sphere radius: {sphere_radius:.3f}  far: {cfg.far:.3f}")

    # Il guscio deve stare interamente fuori dalla geometria, altrimenti i campioni di
    # background finirebbero dentro l'oggetto, dove vive il campo foreground.
    if sphere_radius <= scene_radius:
        raise RuntimeError(
            f"Background sphere radius {sphere_radius:.3f} is inside the geometry "
            f"(scene radius {scene_radius:.3f}): the shell would intersect the mesh. "
            f"bg_radius_mult is {cfg.bg_radius_mult} and must be > 1."
        )

    # Non è un errore (i raggi bg sono ri-ancorati al centro, quindi la matematica regge
    # comunque), ma un guscio dentro il rig di camere è una condizione che vale la pena
    # vedere: l'ambiente finisce più vicino all'origine di dove stanno gli osservatori.
    cam_max = max(float(np.linalg.norm(m["pose"][:3, 3])) for m in dataset._frames_meta)
    if sphere_radius <= cam_max:
        print(f"  [warn] sphere radius {sphere_radius:.3f} does not enclose all cameras "
              f"(farthest at {cam_max:.3f}); consider bg_radius_mult >= "
              f"{cam_max / max(scene_radius, 1e-9):.2f}")

    model, embed_fn, embeddirs_fn = _build_models(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    iter_start = 0
    ckpt = Path(ckpt_path)
    if ckpt.exists():
        saved = torch.load(str(ckpt), map_location=device)
        # Prima di qualsiasi altra cosa: un checkpoint pre-normalizzazione non è
        # riprendibile. Serve qui e non solo in load_checkpoint() perché questo è
        # l'unico punto che apre il checkpoint senza ricostruire NerfConfig, ed è la
        # strada che percorre resume_skip_step2_if_ckpt.
        _check_ray_convention(saved, str(ckpt))
        model.load_state_dict(saved["model_state"])
        try:
            optimizer.load_state_dict(saved["optimizer"])
        except (ValueError, KeyError):
            print("  [warn] optimizer state skipped (param group mismatch)")
        iter_start = saved["iter_done"]
        if "scene_center" in saved:
            center = torch.tensor(saved["scene_center"], device=device, dtype=torch.float32)
        if "sphere_radius" in saved:
            sphere_radius = float(saved["sphere_radius"])
        print(f"  Ripreso da checkpoint: {ckpt}  (iter {iter_start})")

    model_bundle = (model, embed_fn, embeddirs_fn, device, center, sphere_radius)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV delle metriche, una riga per display block. In append: train() viene
    # rientrata sullo stesso output_dir sia dai resume sia dal loop interattivo
    # di run_pipeline, e i segmenti devono accumularsi.
    csv_logger = CsvLogger(out_dir / "training_metrics.csv", CSV_FIELDS)
    # Secondo file, una riga per epoca completata: stessa politica di append, ma
    # con le medie su un passaggio intero sul dataset invece che sulla finestra.
    epoch_csv_logger = CsvLogger(out_dir / "epoch_metrics.csv", EPOCH_CSV_FIELDS)
    _t_train_start = time.perf_counter()

    # Decay ancorato a un orizzonte FISSO in iterazioni assolute (lr_decay_steps):
    # schedule = funzione pura di i, continuo attraverso i resume. 0 = auto → num_iters.
    decay_steps = cfg.lr_decay_steps if cfg.lr_decay_steps > 0 else num_iters
    loss_window: deque[torch.Tensor] = deque(maxlen=display_every)
    mse_window:  deque[torch.Tensor] = deque(maxlen=display_every)
    # Dimensioni EFFETTIVE dei batch: l'ultimo di ogni epoca è più corto, e
    # rays_per_s calcolato su batch_size nominale sarebbe ottimista.
    rays_window: deque[int] = deque(maxlen=display_every)
    _final_psnr: float = float("nan")  # aggiornato ad ogni display block, ritornato a fine

    # Accumulatori sull'epoca corrente (alimentano solo epoch_metrics.csv: la
    # console e training_metrics.csv restano sulla finestra display_every).
    # Il reset è agganciato al CAMBIO di epoca e non a epoch_iter == 1, altrimenti
    # un resume che riparte a metà epoca non li inizializzerebbe mai.
    _epoch_cur:      int   = -1
    _epoch_loss_sum: torch.Tensor | None = None
    _epoch_mse_sum:  torch.Tensor | None = None
    _epoch_iters:    int   = 0
    _epoch_rays:     int   = 0
    _t_epoch_start:  float = time.perf_counter()

    # ── profiling state ──────────────────────────────────────────────────────
    use_cuda      = device.type == "cuda"
    _prof_active  = cfg.profile_iters > 0
    _prof         = {"sample": 0.0, "render": 0.0, "bwd": 0.0, "opt": 0.0}
    _prof_n       = 0
    _iter_times:  deque[float] = deque(maxlen=display_every)

    if _prof_active:
        dev_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
        print(f"  [prof] GPU: {dev_name}  —  profiling first {cfg.profile_iters} iters")

    def _sync():
        if use_cuda:
            torch.cuda.synchronize()

    model.train()

    for i in range(iter_start, iter_start + num_iters):
        _t_iter = time.perf_counter()
        _profiling = _prof_active and (i - iter_start) < cfg.profile_iters

        epoch, _k = divmod(i, iters_per_epoch)
        epoch_iter = _k + 1                       # 1-based, come iter = i + 1
        is_epoch_end = epoch_iter == iters_per_epoch

        if epoch != _epoch_cur:
            _epoch_cur      = epoch
            _epoch_loss_sum = None
            _epoch_mse_sum  = None
            _epoch_iters    = 0
            _epoch_rays     = 0
            _t_epoch_start  = time.perf_counter()

        # ── LR schedule ─────────────────────────────────────────────────────
        new_lr = lr * (cfg.lr_decay_factor ** min(i / decay_steps, 1.0))
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

        # ── sample (fetta della permutazione dell'epoca; l'ultima è più corta) ─
        if _profiling: _sync(); _t0 = time.perf_counter()
        rays_o, rays_d, rgb_gt, depths, in_mask = dataset.sample_epoch(i)
        if _profiling: _sync(); _prof["sample"] += time.perf_counter() - _t0
        n_rays_batch = rays_o.shape[0]

        # ── render unificato fg+bg ───────────────────────────────────────────
        if _profiling: _sync(); _t0 = time.perf_counter()
        # Unico punto della repo che attiva il rumore sulla densità: è un
        # regolarizzatore, quindi vale solo per il forward su cui si fa backward.
        # Ogni percorso di inferenza usa il default noise_std=0.0 ed è riproducibile.
        rgb_pred = render_unified(
            rays_o, rays_d, depths, in_mask,
            model, embed_fn, embeddirs_fn, cfg, center, sphere_radius,
            noise_std=cfg.raw_noise_std)
        if _profiling: _sync(); _prof["render"] += time.perf_counter() - _t0

        # ── loss ─────────────────────────────────────────────────────────────
        loss = loss_fn(rgb_pred, rgb_gt)

        # ── per-ray MSE per PSNR (no grad) ───────────────────────────────────
        with torch.no_grad():
            mse_val = F.mse_loss(rgb_pred, rgb_gt)

        # ── backward ─────────────────────────────────────────────────────────
        if _profiling: _sync(); _t0 = time.perf_counter()
        optimizer.zero_grad()
        loss.backward()
        if _profiling: _sync(); _prof["bwd"] += time.perf_counter() - _t0

        # ── optimizer step ────────────────────────────────────────────────────
        if _profiling: _sync(); _t0 = time.perf_counter()
        optimizer.step()
        if _profiling: _sync(); _prof["opt"] += time.perf_counter() - _t0

        # ── accumulators ──────────────────────────────────────────────────────
        _loss_d = loss.detach()
        _mse_d  = mse_val.detach()
        loss_window.append(_loss_d)
        mse_window.append(_mse_d)
        rays_window.append(n_rays_batch)

        # Somme sull'epoca: restano tensori sul device, un solo .item() a fine
        # epoca invece di una sincronizzazione per iterazione.
        _epoch_loss_sum = _loss_d if _epoch_loss_sum is None else _epoch_loss_sum + _loss_d
        _epoch_mse_sum  = _mse_d  if _epoch_mse_sum  is None else _epoch_mse_sum  + _mse_d
        _epoch_iters += 1
        _epoch_rays  += n_rays_batch

        _iter_times.append(time.perf_counter() - _t_iter)
        if _profiling:
            _prof_n += 1

        # ── profiling report (printed once, at end of window) ─────────────────
        if _profiling and (i - iter_start) == cfg.profile_iters - 1:
            # list() prima dello slice: deque non è affettabile.
            total_s    = sum(list(_iter_times)[:_prof_n])
            total_rays = sum(list(rays_window)[:_prof_n])
            total_syn  = sum(_prof.values())
            rays_s     = total_rays / max(total_s, 1e-9)
            n_samp     = cfg.depth_window_samples
            pts_s      = total_rays * n_samp / max(total_s, 1e-9)
            if use_cuda:
                mem_mb = torch.cuda.max_memory_allocated(device) / 1024**2
                print(f"  [prof] peak GPU mem: {mem_mb:.0f} MB")
            print(f"  [prof] avg over {_prof_n} iters  "
                  f"iter={1000*total_s/_prof_n:.2f} ms  "
                  f"({1000*total_syn/_prof_n:.2f} ms with sync)")
            print(f"  [prof]  rays/s={rays_s:.0f}  pts/s={pts_s:.0f}  "
                  f"(batch={batch_size} × {n_samp} samples)")
            hdr = f"  [prof]  {'phase':<8}  {'ms/iter':>8}  {'%':>6}"
            print(hdr)
            for k, v in _prof.items():
                ms  = 1000 * v / _prof_n
                pct = 100 * v / max(total_syn, 1e-9)
                print(f"  [prof]  {k:<8}  {ms:>8.3f}  {pct:>5.1f}%")

        # ── display ───────────────────────────────────────────────────────────
        # Scatta ogni display_every, a ogni fine epoca, e sull'ultima iterazione
        # della chiamata. Se un confine di epoca cade su un multiplo di
        # display_every l'or lo rende comunque un blocco solo.
        if (i + 1) % display_every == 0 or is_epoch_end or i == iter_start + num_iters - 1:
            recent_loss = torch.stack(list(loss_window)).mean().item()
            recent_mse  = torch.stack(list(mse_window)).mean().item()
            psnr = -10.0 * np.log10(recent_mse + 1e-10)
            _final_psnr = psnr  # tracciamo il PSNR dell'ultimo block per il ritorno

            win_times   = list(_iter_times)
            win_secs    = max(sum(win_times), 1e-9)
            iters_per_s = len(win_times) / win_secs
            rays_per_s  = sum(rays_window) / win_secs
            print(f"  iter {i + 1}  ep {epoch + 1} ({epoch_iter}/{iters_per_epoch})  "
                  f"{cfg.loss_type}={recent_loss:.4f}  PSNR≈{psnr:.2f} dB  "
                  f"lr={new_lr:.2e}  {iters_per_s:.1f} it/s  {rays_per_s/1e3:.0f}k rays/s")

            # — TensorBoard scalars (no-op se tb_logger è None) —
            if tb_logger is not None:
                tb_logger.log_scalars("nerf", {
                    "loss":        recent_loss,
                    "psnr_db":     psnr,
                    "lr":          new_lr,
                    "iters_per_s": iters_per_s,
                    "epoch":       epoch + 1,
                }, step=i + 1)

            # sample_natural e non sample_epoch: il batch diagnostico deve essere
            # indipendente dall'ordine dell'epoca, altrimenti ne consumerebbe
            # posizioni e i raggi verrebbero visti due volte nello stesso passaggio.
            with torch.no_grad():
                _ro, _rd, _rgb_gt, _dep, _msk = dataset.sample_natural(min(512, batch_size))
                _rgb_pred, _acc = render_unified(
                    _ro, _rd, _dep, _msk,
                    model, embed_fn, embeddirs_fn, cfg, center, sphere_radius,
                    return_acc=True)
                acc_fg = _acc[_msk].mean() if _msk.any() else _acc.mean()
                print(f"    [diag] acc_fg={acc_fg:.4f}  "
                      f"pred=[{_rgb_pred.min():.3f},{_rgb_pred.mean():.3f},{_rgb_pred.max():.3f}]  "
                      f"target=[{_rgb_gt.min():.3f},{_rgb_gt.mean():.3f},{_rgb_gt.max():.3f}]")

            # Riga CSV: scritta qui perché acc_fg nasce nel blocco diag sopra, e
            # prima della preview perché un errore in OpenEXR non deve far perdere
            # una riga già calcolata.
            csv_logger.log({
                "iter":            i + 1,
                "epoch":           epoch + 1,
                "loss":            f"{recent_loss:.6g}",
                "mse":             f"{recent_mse:.6g}",
                "psnr_db":         f"{psnr:.6g}",
                "lr":              f"{new_lr:.6e}",
                "iters_per_s":     f"{iters_per_s:.6g}",
                "rays_per_s":      f"{rays_per_s:.6g}",
                "acc_fg":          f"{float(acc_fg):.6g}",
                # wall_s si azzera a ogni rientro in train(): è il marcatore dei
                # segmenti del loop interattivo.
                "wall_s":          f"{time.perf_counter() - _t_train_start:.3f}",
                "timestamp":       datetime.datetime.now().isoformat(timespec="seconds"),
                "loss_type":       cfg.loss_type,
                "rgb_activation":  cfg.rgb_activation,
                "batch_size":      batch_size,
                "lr_decay_factor": cfg.lr_decay_factor,
            })

            # Riga di epoca: stessa iterazione, dominio diverso — le medie qui
            # sono su tutte le _epoch_iters iterazioni dell'epoca, non sulla
            # finestra. Su un resume a metà epoca _epoch_iters < iters_per_epoch.
            if is_epoch_end and _epoch_iters > 0:
                ep_loss = float(_epoch_loss_sum) / _epoch_iters
                ep_mse  = float(_epoch_mse_sum)  / _epoch_iters
                ep_psnr = -10.0 * np.log10(ep_mse + 1e-10)
                ep_wall = time.perf_counter() - _t_epoch_start
                print(f"    [epoca {epoch + 1}] {_epoch_iters} iter · "
                      f"{_epoch_rays} raggi · {cfg.loss_type}={ep_loss:.4f}  "
                      f"PSNR≈{ep_psnr:.2f} dB  {ep_wall:.1f} s")
                epoch_csv_logger.log({
                    "epoch":           epoch + 1,
                    "iter_end":        i + 1,
                    "iters_in_epoch":  _epoch_iters,
                    "rays_in_epoch":   _epoch_rays,
                    "loss":            f"{ep_loss:.6g}",
                    "mse":             f"{ep_mse:.6g}",
                    "psnr_db":         f"{ep_psnr:.6g}",
                    "lr":              f"{new_lr:.6e}",
                    "iters_per_s":     f"{_epoch_iters / max(ep_wall, 1e-9):.6g}",
                    "rays_per_s":      f"{_epoch_rays / max(ep_wall, 1e-9):.6g}",
                    "epoch_wall_s":    f"{ep_wall:.3f}",
                    "timestamp":       datetime.datetime.now().isoformat(timespec="seconds"),
                    "loss_type":       cfg.loss_type,
                    "rgb_activation":  cfg.rgb_activation,
                    "batch_size":      batch_size,
                    "lr_decay_factor": cfg.lr_decay_factor,
                })
                if tb_logger is not None:
                    tb_logger.log_scalars("nerf/epoch", {
                        "loss":         ep_loss,
                        "mse":          ep_mse,
                        "psnr_db":      ep_psnr,
                        "epoch_wall_s": ep_wall,
                    }, step=i + 1)

            # Preview: _save_preview ora ritorna l'array per il log TB
            preview_img = _save_preview(
                model_bundle, dataset, cfg, out_dir / f"preview_iter_{i+1:06d}.exr"
            )
            if tb_logger is not None and preview_img is not None:
                tb_logger.log_image("nerf/preview", preview_img, step=i + 1, tonemap=True)
                tb_logger.flush()

            # Checkpoint periodico: permette il watch-mode di nerf_viewer e il
            # resume da crash. Scrittura atomica, costo trascurabile vs preview.
            save_checkpoint(ckpt_path, model, optimizer, i + 1, cfg,
                            scene_center=center, sphere_radius=sphere_radius)
            if use_cuda:
                torch.cuda.empty_cache()
            model.train()

    save_checkpoint(ckpt_path, model, optimizer, iter_start + num_iters, cfg,
                    scene_center=center, sphere_radius=sphere_radius)
    print(f"  Checkpoint salvato: {ckpt_path}")
    return _final_psnr


def _save_preview(
    model_bundle, dataset: NerfDataset, cfg: NerfConfig, path: Path
) -> np.ndarray | None:
    """Render a preview frame, save it as EXR, and return the float32 HxWx3 array.

    The returned array can be passed directly to RunLogger.log_image() for
    TensorBoard without triggering a second render.  Returns None on failure.
    """
    import OpenEXR, Imath
    model = model_bundle[0]
    _, _, _, test_pose, test_dep = dataset.get_preview_frame()
    model.eval()
    img = render_image(model_bundle, dataset.H, dataset.W, dataset.focal_x, test_pose, cfg,
                       focal_y=dataset.focal_y, target_depth=test_dep)
    img = np.ascontiguousarray(img.astype(np.float32))
    h, w, _ = img.shape
    pt = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header = OpenEXR.Header(w, h)
    header["channels"] = {"R": pt, "G": pt, "B": pt}
    f = OpenEXR.OutputFile(str(path), header)
    f.writePixels({"R": img[..., 0].tobytes(), "G": img[..., 1].tobytes(), "B": img[..., 2].tobytes()})
    f.close()
    return img
