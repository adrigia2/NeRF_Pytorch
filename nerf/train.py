from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoint import _build_models, save_checkpoint
from .config import NerfConfig
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


def train(transforms_path: str, cfg: NerfConfig, *, ckpt_path: str, output_dir: str,
          num_iters: int, batch_size: int, lr: float, seed: int,
          display_every: int) -> None:
    """Train a single-network depth-guided NeRF from transforms_extended.json.

    Uses render_unified with a single fixed-shape batch (fg+bg together via in_mask).
    Saves a checkpoint to ckpt_path; resumes if it already exists.
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

    # Compute scene bounds from foreground hit points
    center, max_side = dataset.compute_scene_bounds()
    sphere_radius = float(cfg.bg_radius_mult * max_side)
    cfg.far = sphere_radius + cfg.bg_depth_window_end + 1.0
    print(f"  Scene centre: {center.tolist()}")
    print(f"  Bbox max side: {max_side:.3f}  sphere radius: {sphere_radius:.3f}  far: {cfg.far:.3f}")

    model, embed_fn, embeddirs_fn = _build_models(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    iter_start = 0
    ckpt = Path(ckpt_path)
    if ckpt.exists():
        saved = torch.load(str(ckpt), map_location=device)
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

    # Decay spalmato sull'intero run (incluso l'eventuale resume):
    # lr → 0.2·lr all'ultima iterazione pianificata.
    decay_steps = iter_start + num_iters
    loss_window: deque[torch.Tensor] = deque(maxlen=display_every)
    mse_window:  deque[torch.Tensor] = deque(maxlen=display_every)

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

        # ── LR schedule ─────────────────────────────────────────────────────
        new_lr = lr * (0.2 ** (i / decay_steps))
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

        # ── sample (batch unificato, shape fissa) ───────────────────────────
        if _profiling: _sync(); _t0 = time.perf_counter()
        rays_o, rays_d, rgb_gt, depths, in_mask = dataset.sample_natural(batch_size)
        if _profiling: _sync(); _prof["sample"] += time.perf_counter() - _t0

        # ── render unificato fg+bg ───────────────────────────────────────────
        if _profiling: _sync(); _t0 = time.perf_counter()
        rgb_pred = render_unified(
            rays_o, rays_d, depths, in_mask,
            model, embed_fn, embeddirs_fn, cfg, center, sphere_radius,
            perturb=False)
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
        loss_window.append(loss.detach())
        mse_window.append(mse_val.detach())

        _iter_times.append(time.perf_counter() - _t_iter)
        if _profiling:
            _prof_n += 1

        # ── profiling report (printed once, at end of window) ─────────────────
        if _profiling and (i - iter_start) == cfg.profile_iters - 1:
            total_s   = sum(_iter_times[:_prof_n])
            total_syn = sum(_prof.values())
            rays_s    = batch_size * _prof_n / max(total_s, 1e-9)
            n_samp    = cfg.depth_window_samples
            pts_s     = batch_size * n_samp * _prof_n / max(total_s, 1e-9)
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
        if (i + 1) % display_every == 0 or i == iter_start + num_iters - 1:
            recent_loss = torch.stack(list(loss_window)).mean().item()
            recent_mse  = torch.stack(list(mse_window)).mean().item()
            psnr = -10.0 * np.log10(recent_mse + 1e-10)

            win_times   = list(_iter_times)
            iters_per_s = len(win_times) / max(sum(win_times), 1e-9)
            rays_per_s  = batch_size * iters_per_s
            print(f"  iter {i + 1}  {cfg.loss_type}={recent_loss:.4f}  PSNR≈{psnr:.2f} dB  "
                  f"lr={new_lr:.2e}  {iters_per_s:.1f} it/s  {rays_per_s/1e3:.0f}k rays/s")

            with torch.no_grad():
                _ro, _rd, _rgb_gt, _dep, _msk = dataset.sample_natural(min(512, batch_size))
                _rgb_pred, _acc = render_unified(
                    _ro, _rd, _dep, _msk,
                    model, embed_fn, embeddirs_fn, cfg, center, sphere_radius,
                    perturb=False, return_acc=True)
                acc_fg = _acc[_msk].mean() if _msk.any() else _acc.mean()
                print(f"    [diag] acc_fg={acc_fg:.4f}  "
                      f"pred=[{_rgb_pred.min():.3f},{_rgb_pred.mean():.3f},{_rgb_pred.max():.3f}]  "
                      f"target=[{_rgb_gt.min():.3f},{_rgb_gt.mean():.3f},{_rgb_gt.max():.3f}]")

            _save_preview(model_bundle, dataset, cfg, out_dir / f"preview_iter_{i+1:06d}.exr")
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


def _save_preview(model_bundle, dataset: NerfDataset, cfg: NerfConfig, path: Path):
    import OpenEXR, Imath
    model = model_bundle[0]
    _, _, _, test_pose, test_dep = dataset.get_test_frame()
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
