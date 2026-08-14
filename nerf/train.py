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
    """Relative MSE (variant with eps outside the square): divides by pred²+eps.
    This is not the exact RawNeRF formulation — see _rel_mse_raw for that."""
    return (((pred - target) ** 2) / (pred.detach() ** 2 + eps)).mean()

def _rel_mse_raw(pred, target, eps=1e-3):
    """Relative MSE (faithful RawNeRF): weights every pixel by the inverse square
    of the predicted intensity, with eps inside the square → (pred+eps)².
    Gives uniform relative fidelity across the whole dynamic range.
    Cf. google-research/multinerf, internal/train_utils.py."""
    return (((pred - target) ** 2) / (pred.detach() + eps) ** 2).mean()

def _mse(pred, target):
    return ((pred - target) ** 2).mean()

def _log_l1(pred, target):
    """L1 in log1p space: compresses highlights, boosts shadow gradients."""
    return F.l1_loss(torch.log1p(pred), torch.log1p(target))

LOSSES = {
    "l1":          F.l1_loss,
    "mse":         _mse,
    "rel_mse":     _rel_mse,      # variant: eps outside the square
    "rel_mse_raw": _rel_mse_raw,  # faithful RawNeRF: eps inside the square
    "log_l1":      _log_l1,
}


# Columns of <output_dir>/training_metrics.csv, one row per display block.
#
# Which comparisons are legitimate:
#   - ``loss`` is in the units of cfg.loss_type: comparing it across runs with
#     different losses is meaningless; it is for reading convergence within a run.
#   - ``mse``/``psnr_db`` are always computed, regardless of the loss driving the
#     gradient: they are the only columns comparable across runs that use
#     different losses.
#   - The domain is the training BATCH of the iteration, not an image: the batch
#     is a slice of a permutation of the whole ray pool, so foreground and
#     background are mixed in the dataset's own proportions. It is not a held-out
#     metric in any sense.
#   - The average is over the last ``display_every`` iterations (the deque
#     window), NOT over the epoch: for the per-epoch aggregate see epoch_metrics.csv.
#   - psnr_db = -10·log10(mse) implicitly assumes MAX_I = 1, but the targets are
#     HDR (values > 1): it is a monotone remapping of the MSE, comparable between
#     runs on the same data but NOT with PSNR figures from the literature, and
#     possibly negative when mse > 1. With composite_white=False the GT of the
#     background rays is the real envmap pixels, so the bright regions weigh
#     quadratically and dominate the metric over the diffuse range.
#
# The last four columns are constant per run and redundant row by row, but they
# make the CSV self-describing: they are the axes of the sweep, so concatenating
# N files into one dataframe is enough to label the curves. The scene is not known
# here, but it is in the path (<output_root>/<scene>/nerf_train/).
CSV_FIELDS = [
    "iter", "epoch", "loss", "mse", "psnr_db", "lr",
    "iters_per_s", "rays_per_s", "acc_fg",
    "wall_s", "timestamp",
    "loss_type", "rgb_activation", "batch_size", "lr_decay_factor",
]


# Columns of <output_dir>/epoch_metrics.csv, one row per completed epoch.
#
# The substantive difference from CSV_FIELDS is the domain of the average: here
# ``loss`` and ``mse`` are averaged over ALL the iterations of the epoch, that is
# over a full pass on the dataset, whereas in training_metrics.csv they are
# averaged over the last ``display_every`` iterations (with 1266 iters/epoch and
# display_every=100, 8 % of the epoch). They are two different granularities and
# belong on two different curves, not concatenated.
#
# ``iters_in_epoch``/``rays_in_epoch`` are not redundant: on a resume mid-epoch
# the accumulators start where the segment resumed, so the first row written
# after a resume covers only the tail of the epoch and has
# iters_in_epoch < iters_per_epoch. That is the expected behaviour, and these two
# columns are what makes it readable after the fact.
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

    # Epoch ordering: one permutation of the whole pool per epoch, consumed one
    # batch at a time. This replaces sampling with replacement, which gave no
    # coverage guarantee at all (the fraction of rays never seen after T iterations
    # was exp(-T·batch/n_rays)).
    iters_per_epoch = dataset.configure_epochs(batch_size, seed)
    tail = dataset.n_rays - (iters_per_epoch - 1) * batch_size
    print(f"  Epochs: {dataset.n_rays} rays / batch {batch_size} = "
          f"{iters_per_epoch} iters/epoch (last batch: {tail} rays)  —  "
          f"{num_iters} iters ≈ {num_iters / iters_per_epoch:.2f} epochs")
    if iters_per_epoch < display_every:
        print(f"  [warn] iters_per_epoch ({iters_per_epoch}) < display_every "
              f"({display_every}): the display block fires at every epoch end, "
              f"so more often than display_every (EXR preview + checkpoint each time)")

    # The background sphere is anchored at the world ORIGIN: the environment is a purely
    # directional function, so the centre is a convention, and fixing it makes it
    # reproducible instead of dependent on the camera set. The geometry only serves to
    # size the radius.
    scene_radius, p_min, p_max = dataset.compute_scene_bounds()
    scene_radius = float(scene_radius)
    center = torch.zeros(3, device=device, dtype=torch.float32)
    sphere_radius = float(cfg.bg_radius_mult * scene_radius)
    cfg.far = sphere_radius + cfg.bg_depth_window_end + 1.0

    print(f"  Scene bbox: {[round(v, 3) for v in p_min.tolist()]} .. "
          f"{[round(v, 3) for v in p_max.tolist()]}")
    print(f"  Scene radius (from origin): {scene_radius:.3f}  "
          f"sphere radius: {sphere_radius:.3f}  far: {cfg.far:.3f}")

    # The shell must sit entirely outside the geometry, otherwise the background samples
    # would land inside the object, where the foreground field lives.
    if sphere_radius <= scene_radius:
        raise RuntimeError(
            f"Background sphere radius {sphere_radius:.3f} is inside the geometry "
            f"(scene radius {scene_radius:.3f}): the shell would intersect the mesh. "
            f"bg_radius_mult is {cfg.bg_radius_mult} and must be > 1."
        )

    # Not an error (bg rays are re-anchored at the centre, so the maths still holds), but
    # a shell inside the camera rig is worth seeing: the environment ends up closer to
    # the origin than the observers are.
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
        # Before anything else: a pre-normalization checkpoint is not resumable. It is
        # needed here and not only in load_checkpoint() because this is the only place
        # that opens the checkpoint without rebuilding NerfConfig, and it is the path
        # resume_skip_step2_if_ckpt takes.
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
        print(f"  Resumed from checkpoint: {ckpt}  (iter {iter_start})")

    model_bundle = (model, embed_fn, embeddirs_fn, device, center, sphere_radius)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Metrics CSV, one row per display block. Append mode: train() is re-entered on
    # the same output_dir both by resumes and by the interactive loop of
    # run_pipeline, and the segments have to accumulate.
    csv_logger = CsvLogger(out_dir / "training_metrics.csv", CSV_FIELDS)
    # Second file, one row per completed epoch: same append policy, but with the
    # averages taken over a full pass on the dataset instead of over the window.
    epoch_csv_logger = CsvLogger(out_dir / "epoch_metrics.csv", EPOCH_CSV_FIELDS)
    _t_train_start = time.perf_counter()

    # Decay anchored to a FIXED horizon in absolute iterations (lr_decay_steps):
    # schedule = pure function of i, continuous across resumes. 0 = auto → num_iters.
    decay_steps = cfg.lr_decay_steps if cfg.lr_decay_steps > 0 else num_iters
    loss_window: deque[torch.Tensor] = deque(maxlen=display_every)
    mse_window:  deque[torch.Tensor] = deque(maxlen=display_every)
    # ACTUAL batch sizes: the last one of each epoch is shorter, and rays_per_s
    # computed on the nominal batch_size would be optimistic.
    rays_window: deque[int] = deque(maxlen=display_every)
    _final_psnr: float = float("nan")  # updated at every display block, returned at the end

    # Accumulators over the current epoch (they feed only epoch_metrics.csv: the
    # console and training_metrics.csv stay on the display_every window).
    # The reset is hooked to the epoch CHANGE and not to epoch_iter == 1, otherwise a
    # resume starting mid-epoch would never initialise them.
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
        epoch_iter = _k + 1                       # 1-based, like iter = i + 1
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

        # ── sample (slice of the epoch permutation; the last one is shorter) ──
        if _profiling: _sync(); _t0 = time.perf_counter()
        rays_o, rays_d, rgb_gt, depths, in_mask = dataset.sample_epoch(i)
        if _profiling: _sync(); _prof["sample"] += time.perf_counter() - _t0
        n_rays_batch = rays_o.shape[0]

        # ── unified fg+bg render ────────────────────────────────────────────
        if _profiling: _sync(); _t0 = time.perf_counter()
        # The only place in the repo that enables density noise: it is a regularizer,
        # so it applies only to the forward pass that gets back-propagated. Every
        # inference path uses the default noise_std=0.0 and is reproducible.
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

        # Epoch sums: they stay as device tensors, with a single .item() at the end
        # of the epoch instead of one synchronisation per iteration.
        _epoch_loss_sum = _loss_d if _epoch_loss_sum is None else _epoch_loss_sum + _loss_d
        _epoch_mse_sum  = _mse_d  if _epoch_mse_sum  is None else _epoch_mse_sum  + _mse_d
        _epoch_iters += 1
        _epoch_rays  += n_rays_batch

        _iter_times.append(time.perf_counter() - _t_iter)
        if _profiling:
            _prof_n += 1

        # ── profiling report (printed once, at end of window) ─────────────────
        if _profiling and (i - iter_start) == cfg.profile_iters - 1:
            # list() before slicing: a deque is not sliceable.
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
        # Fires every display_every, at every epoch end, and on the last iteration of
        # the call. If an epoch boundary lands on a multiple of display_every, the or
        # still makes it a single block.
        if (i + 1) % display_every == 0 or is_epoch_end or i == iter_start + num_iters - 1:
            recent_loss = torch.stack(list(loss_window)).mean().item()
            recent_mse  = torch.stack(list(mse_window)).mean().item()
            psnr = -10.0 * np.log10(recent_mse + 1e-10)
            _final_psnr = psnr  # track the last block's PSNR for the return value

            win_times   = list(_iter_times)
            win_secs    = max(sum(win_times), 1e-9)
            iters_per_s = len(win_times) / win_secs
            rays_per_s  = sum(rays_window) / win_secs
            print(f"  iter {i + 1}  ep {epoch + 1} ({epoch_iter}/{iters_per_epoch})  "
                  f"{cfg.loss_type}={recent_loss:.4f}  PSNR≈{psnr:.2f} dB  "
                  f"lr={new_lr:.2e}  {iters_per_s:.1f} it/s  {rays_per_s/1e3:.0f}k rays/s")

            # — TensorBoard scalars (no-op when tb_logger is None) —
            if tb_logger is not None:
                tb_logger.log_scalars("nerf", {
                    "loss":        recent_loss,
                    "psnr_db":     psnr,
                    "lr":          new_lr,
                    "iters_per_s": iters_per_s,
                    "epoch":       epoch + 1,
                }, step=i + 1)

            # sample_natural and not sample_epoch: the diagnostic batch must be
            # independent of the epoch order, otherwise it would consume positions
            # from it and rays would be seen twice in the same pass.
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

            # CSV row: written here because acc_fg is produced by the diag block above,
            # and before the preview because an OpenEXR error must not lose a row that
            # has already been computed.
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
                # wall_s restarts at every re-entry into train(): it is the marker of
                # the interactive loop's segments.
                "wall_s":          f"{time.perf_counter() - _t_train_start:.3f}",
                "timestamp":       datetime.datetime.now().isoformat(timespec="seconds"),
                "loss_type":       cfg.loss_type,
                "rgb_activation":  cfg.rgb_activation,
                "batch_size":      batch_size,
                "lr_decay_factor": cfg.lr_decay_factor,
            })

            # Epoch row: same iteration, different domain — the averages here are over
            # all _epoch_iters iterations of the epoch, not over the window. On a resume
            # mid-epoch, _epoch_iters < iters_per_epoch.
            if is_epoch_end and _epoch_iters > 0:
                ep_loss = float(_epoch_loss_sum) / _epoch_iters
                ep_mse  = float(_epoch_mse_sum)  / _epoch_iters
                ep_psnr = -10.0 * np.log10(ep_mse + 1e-10)
                ep_wall = time.perf_counter() - _t_epoch_start
                print(f"    [epoch {epoch + 1}] {_epoch_iters} iters · "
                      f"{_epoch_rays} rays · {cfg.loss_type}={ep_loss:.4f}  "
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

            # Preview: _save_preview also returns the array, for the TB log
            preview_img = _save_preview(
                model_bundle, dataset, cfg, out_dir / f"preview_iter_{i+1:06d}.exr"
            )
            if tb_logger is not None and preview_img is not None:
                tb_logger.log_image("nerf/preview", preview_img, step=i + 1, tonemap=True)
                tb_logger.flush()

            # Periodic checkpoint: enables nerf_viewer's watch mode and resuming after
            # a crash. Atomic write, negligible cost next to the preview.
            save_checkpoint(ckpt_path, model, optimizer, i + 1, cfg,
                            scene_center=center, sphere_radius=sphere_radius)
            if use_cuda:
                torch.cuda.empty_cache()
            model.train()

    save_checkpoint(ckpt_path, model, optimizer, iter_start + num_iters, cfg,
                    scene_center=center, sphere_radius=sphere_radius)
    print(f"  Checkpoint saved: {ckpt_path}")
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
