from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoint import _build_models, save_checkpoint
from .config import NerfConfig
from .dataset import NerfDataset
from .render import render_bg, render_image, render_rays_depth


def _rel_mse(pred, target, eps=1e-3):
    """Relative MSE: weights each pixel by its predicted intensity."""
    return (((pred - target) ** 2) / (pred.detach() ** 2 + eps)).mean()

def _mse(pred, target):
    return ((pred - target) ** 2).mean()


def train(transforms_path: str, cfg: NerfConfig, *, ckpt_path: str, output_dir: str,
          num_iters: int, batch_size: int, lr: float, seed: int,
          display_every: int) -> None:
    """Train a single-network depth-guided NeRF from transforms_extended.json.

    Foreground rays (mesh hit) use render_rays_depth with the mesh window.
    Background rays use render_bg (spherical shell anchored at scene centre).
    Both branches apply an opacity loss to encourage opaque surfaces.
    Saves a checkpoint to ckpt_path; resumes if it already exists.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  NeRF training on: {device}")

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

    decay_steps = cfg.lrate_decay * 1000
    loss_window:        list[torch.Tensor] = []
    mse_window:         list[torch.Tensor] = []
    loss_fg_window:     list[torch.Tensor] = []
    loss_bg_window:     list[torch.Tensor] = []
    loss_opc_fg_window: list[torch.Tensor] = []
    loss_opc_bg_window: list[torch.Tensor] = []

    # ── profiling state ──────────────────────────────────────────────────────
    use_cuda      = device.type == "cuda"
    _prof_active  = cfg.profile_iters > 0
    _prof         = {"sample": 0.0, "fg": 0.0, "bg": 0.0, "bwd": 0.0, "opt": 0.0}
    _prof_n       = 0
    _iter_times:  list[float] = []      # wall time per iter (no extra sync)

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
        new_lr = lr * (0.1 ** (i / decay_steps))
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

        # ── sample ──────────────────────────────────────────────────────────
        if _profiling: _sync(); _t0 = time.perf_counter()
        fg, bg = dataset.sample_natural(batch_size)
        if _profiling: _sync(); _prof["sample"] += time.perf_counter() - _t0

        fg_rays_o, fg_rays_d, fg_rgb, fg_depth = fg
        bg_rays_o, bg_rays_d, bg_rgb = bg
        n_fg = fg_rgb.shape[0]
        n_bg = bg_rgb.shape[0]
        loss = torch.tensor(0.0, device=device)
        loss_fg = loss_bg = loss_opc_fg = loss_opc_bg = None

        # ── fg render + loss ─────────────────────────────────────────────────
        if _profiling: _sync(); _t0 = time.perf_counter()
        if n_fg > 0:
            rgb_fg, acc_fg = render_rays_depth(
                fg_rays_o, fg_rays_d, model, embed_fn, embeddirs_fn, cfg,
                fg_depth, perturb=True, return_acc=True)
            loss_fg     = _rel_mse(rgb_fg, fg_rgb)
            loss_opc_fg = ((1.0 - acc_fg) ** 2).mean()
            loss = loss + loss_fg + cfg.opacity_weight * loss_opc_fg
        if _profiling: _sync(); _prof["fg"] += time.perf_counter() - _t0

        # ── bg render + loss ─────────────────────────────────────────────────
        if _profiling: _sync(); _t0 = time.perf_counter()
        if n_bg > 0:
            rgb_bg, acc_bg = render_bg(
                bg_rays_d, center, sphere_radius,
                model, embed_fn, embeddirs_fn, cfg,
                perturb=True, return_acc=True)
            loss_bg     = _rel_mse(rgb_bg, bg_rgb)
            loss_opc_bg = ((1.0 - acc_bg) ** 2).mean()
            loss = loss + loss_bg + cfg.opacity_weight * loss_opc_bg
        if _profiling: _sync(); _prof["bg"] += time.perf_counter() - _t0

        # ── per-ray MSE for PSNR (no grad) ───────────────────────────────────
        with torch.no_grad():
            mse_num = torch.tensor(0.0, device=device)
            if n_fg > 0:
                mse_num = mse_num + F.mse_loss(rgb_fg, fg_rgb) * n_fg
            if n_bg > 0:
                mse_num = mse_num + F.mse_loss(rgb_bg, bg_rgb) * n_bg
            mse_val = mse_num / max(n_fg + n_bg, 1)

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
        if loss_fg      is not None: loss_fg_window.append(loss_fg.detach())
        if loss_bg      is not None: loss_bg_window.append(loss_bg.detach())
        if loss_opc_fg  is not None: loss_opc_fg_window.append(loss_opc_fg.detach())
        if loss_opc_bg  is not None: loss_opc_bg_window.append(loss_opc_bg.detach())

        _iter_times.append(time.perf_counter() - _t_iter)
        if _profiling:
            _prof_n += 1

        # ── profiling report (printed once, at end of window) ─────────────────
        if _profiling and (i - iter_start) == cfg.profile_iters - 1:
            total_s   = sum(_iter_times[:_prof_n])
            total_syn = sum(_prof.values())
            rays_s    = batch_size * _prof_n / max(total_s, 1e-9)
            fg_samp   = cfg.depth_window_samples
            bg_samp   = cfg.bg_depth_window_samples
            pts_s     = (n_fg * fg_samp + n_bg * bg_samp) * _prof_n / max(total_s, 1e-9)
            if use_cuda:
                mem_mb = torch.cuda.max_memory_allocated(device) / 1024**2
                print(f"  [prof] peak GPU mem: {mem_mb:.0f} MB")
            print(f"  [prof] avg over {_prof_n} iters  "
                  f"iter={1000*total_s/_prof_n:.2f} ms  "
                  f"({1000*total_syn/_prof_n:.2f} ms with sync)")
            print(f"  [prof]  rays/s={rays_s:.0f}  pts/s={pts_s:.0f}  "
                  f"(fg={n_fg} × {fg_samp}  bg={n_bg} × {bg_samp})")
            hdr = f"  [prof]  {'phase':<8}  {'ms/iter':>8}  {'%':>6}"
            print(hdr)
            for k, v in _prof.items():
                ms  = 1000 * v / _prof_n
                pct = 100 * v / max(total_syn, 1e-9)
                print(f"  [prof]  {k:<8}  {ms:>8.3f}  {pct:>5.1f}%")

        # ── display ───────────────────────────────────────────────────────────
        if (i + 1) % display_every == 0 or i == iter_start + num_iters - 1:
            recent_loss = torch.stack(loss_window[-display_every:]).mean().item()
            recent_mse  = torch.stack(mse_window[-display_every:]).mean().item()
            psnr = -10.0 * np.log10(recent_mse + 1e-10)

            win_times   = _iter_times[-display_every:]
            iters_per_s = len(win_times) / max(sum(win_times), 1e-9)
            rays_per_s  = batch_size * iters_per_s
            print(f"  iter {i + 1}  rel_loss={recent_loss:.4f}  PSNR≈{psnr:.2f} dB  "
                  f"lr={new_lr:.2e}  {iters_per_s:.1f} it/s  {rays_per_s/1e3:.0f}k rays/s")

            def _mean(lst): return torch.stack(lst[-display_every:]).mean().item() if lst else float("nan")
            print(f"    [diag] loss_fg={_mean(loss_fg_window):.4f}  "
                  f"opc_fg={_mean(loss_opc_fg_window):.4f}  "
                  f"loss_bg={_mean(loss_bg_window):.4f}  "
                  f"opc_bg={_mean(loss_opc_bg_window):.4f}")

            with torch.no_grad():
                fg_probe, _ = dataset.sample_natural(min(512, batch_size))
                _ro, _rd, _rgb_gt, _dep = fg_probe
                if _rgb_gt.shape[0] > 0:
                    _rgb_pred, _acc = render_rays_depth(
                        _ro, _rd, model, embed_fn, embeddirs_fn, cfg,
                        _dep, perturb=False, return_acc=True)
                    print(f"    [diag] acc_fg={_acc.mean():.4f}  "
                          f"pred=[{_rgb_pred.min():.3f},{_rgb_pred.mean():.3f},{_rgb_pred.max():.3f}]  "
                          f"target=[{_rgb_gt.min():.3f},{_rgb_gt.mean():.3f},{_rgb_gt.max():.3f}]")

            _save_preview(model_bundle, dataset, cfg, out_dir / f"preview_iter_{i+1:06d}.exr")
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
