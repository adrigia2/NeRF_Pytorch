from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoint import _build_models, save_checkpoint
from .config import NerfConfig
from .dataset import NerfDataset
from .render import render_image, render_rays, render_rays_depth


def _rel_mse(pred, target, eps=1e-3):
    """Relative MSE: weights each pixel by its predicted intensity.
    Prevents bright pixels from dominating the gradient in HDR training."""
    return (((pred - target) ** 2) / (pred.detach() ** 2 + eps)).mean()


def train(transforms_path: str, cfg: NerfConfig, *, ckpt_path: str, output_dir: str,
          num_iters: int, batch_size: int, lr: float, seed: int,
          display_every: int) -> None:
    """Train a NeRF model (coarse + fine) from transforms_extended.json.

    Saves a checkpoint to ckpt_path.  If ckpt_path already exists, training
    resumes from that checkpoint for another num_iters steps.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  NeRF training on: {device}")

    dataset = NerfDataset(transforms_path, device)

    coarse, fine, embed_fn, embeddirs_fn = _build_models(cfg, device)
    optimizer = torch.optim.Adam(
        list(coarse.parameters()) + list(fine.parameters()),
        lr=lr, betas=(0.9, 0.999),
    )

    iter_start = 0
    ckpt = Path(ckpt_path)
    if ckpt.exists():
        saved = torch.load(str(ckpt), map_location=device)
        coarse.load_state_dict(saved["coarse_state"])
        fine.load_state_dict(saved["fine_state"])
        optimizer.load_state_dict(saved["optimizer"])
        iter_start = saved["iter_done"]
        print(f"  Ripreso da checkpoint: {ckpt}  (iter {iter_start})")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    decay_steps = cfg.lrate_decay * 1000
    loss_window:     list[torch.Tensor] = []
    mse_window:      list[torch.Tensor] = []
    loss_fg_window:  list[torch.Tensor] = []
    loss_bgf_window: list[torch.Tensor] = []
    loss_bgc_window: list[torch.Tensor] = []
    loss_opc_window: list[torch.Tensor] = []

    coarse.train()
    fine.train()

    for i in range(iter_start, iter_start + num_iters):
        new_lr = lr * (0.1 ** (i / decay_steps))
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

        if cfg.depth_hint_enabled and dataset.has_depth_split:
            fg, bg = dataset.sample_split(batch_size, cfg.foreground_ratio)
            fg_rays_o, fg_rays_d, fg_rgb, fg_depth = fg
            bg_rays_o, bg_rays_d, bg_rgb = bg

            rgb_fg, acc_fg = render_rays_depth(fg_rays_o, fg_rays_d, fine,
                                               embed_fn, embeddirs_fn, cfg, fg_depth,
                                               perturb=True, return_acc=True)
            rgb_bg_f, rgb_bg_c = render_rays(bg_rays_o, bg_rays_d, coarse, fine,
                                             embed_fn, embeddirs_fn, cfg, perturb=True)

            loss_fg      = _rel_mse(rgb_fg,    fg_rgb)
            loss_bg_f    = _rel_mse(rgb_bg_f,  bg_rgb)
            loss_bg_c    = _rel_mse(rgb_bg_c,  bg_rgb)
            loss_opacity = ((1.0 - acc_fg) ** 2).mean()
            loss = loss_fg + loss_bg_f + loss_bg_c + cfg.opacity_weight * loss_opacity

            n_fg, n_bg = fg_rgb.shape[0], bg_rgb.shape[0]
            with torch.no_grad():
                mse_val = (F.mse_loss(rgb_fg, fg_rgb) * n_fg
                           + F.mse_loss(rgb_bg_f, bg_rgb) * n_bg) / (n_fg + n_bg)
        else:
            rays_o, rays_d, target_rgb = dataset.sample_batch(batch_size)
            rgb_fine, rgb_coarse = render_rays(
                rays_o, rays_d, coarse, fine, embed_fn, embeddirs_fn, cfg, perturb=True)
            loss = _rel_mse(rgb_fine, target_rgb) + _rel_mse(rgb_coarse, target_rgb)
            with torch.no_grad():
                mse_val = F.mse_loss(rgb_fine, target_rgb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate detached tensors — no GPU sync per step
        loss_window.append(loss.detach())
        mse_window.append(mse_val.detach())
        if cfg.depth_hint_enabled and dataset.has_depth_split:
            loss_fg_window.append(loss_fg.detach())
            loss_bgf_window.append(loss_bg_f.detach())
            loss_bgc_window.append(loss_bg_c.detach())
            loss_opc_window.append(loss_opacity.detach())

        if (i + 1) % display_every == 0 or i == iter_start + num_iters - 1:
            recent_loss = torch.stack(loss_window[-display_every:]).mean().item()
            recent_mse  = torch.stack(mse_window[-display_every:]).mean().item()
            psnr = -10.0 * np.log10(recent_mse + 1e-10)
            print(f"  iter {i + 1}  rel_loss={recent_loss:.4f}  PSNR≈{psnr:.2f} dB  lr={new_lr:.2e}")

            if cfg.depth_hint_enabled and dataset.has_depth_split and loss_fg_window:
                l_fg  = torch.stack(loss_fg_window[-display_every:]).mean().item()
                l_bgf = torch.stack(loss_bgf_window[-display_every:]).mean().item()
                l_bgc = torch.stack(loss_bgc_window[-display_every:]).mean().item()
                l_opc = torch.stack(loss_opc_window[-display_every:]).mean().item()
                print(f"    [diag] loss_fg={l_fg:.4f}  loss_opacity={l_opc:.4f}  "
                      f"loss_bg_f={l_bgf:.4f}  loss_bg_c={l_bgc:.4f}")
                with torch.no_grad():
                    fg, _ = dataset.sample_split(min(512, batch_size), cfg.foreground_ratio)
                    _ro, _rd, _rgb_gt, _dep = fg
                    _rgb_pred, _acc = render_rays_depth(
                        _ro, _rd, fine, embed_fn, embeddirs_fn, cfg, _dep,
                        perturb=False, return_acc=True)
                    print(f"    [diag] acc_fg={_acc.mean():.4f}  "
                          f"pred=[{_rgb_pred.min():.3f},{_rgb_pred.mean():.3f},{_rgb_pred.max():.3f}]  "
                          f"target=[{_rgb_gt.min():.3f},{_rgb_gt.mean():.3f},{_rgb_gt.max():.3f}]")

            _save_preview(coarse, fine, embed_fn, embeddirs_fn, dataset, cfg, device,
                          out_dir / f"preview_iter_{i+1:06d}.png")
            coarse.train()
            fine.train()

    save_checkpoint(ckpt_path, coarse, fine, optimizer, iter_start + num_iters, cfg)
    print(f"  Checkpoint salvato: {ckpt_path}")


def _save_preview(coarse, fine, embed_fn, embeddirs_fn, dataset, cfg, device, path: Path):
    from PIL import Image
    _, _, _, test_pose, test_dep = dataset.get_test_frame()
    bundle = (coarse, fine, embed_fn, embeddirs_fn, device)
    img = render_image(bundle, dataset.H, dataset.W, dataset.focal_x, test_pose, cfg,
                       focal_y=dataset.focal_y, target_depth=test_dep)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(str(path))
