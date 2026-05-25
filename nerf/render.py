from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .config import NerfConfig
from .rays import get_rays_np, raw2outputs, sample_pdf


def _eval_sky(sky, embeddirs_fn, rays_d: torch.Tensor) -> torch.Tensor:
    """Evaluate the sky MLP for a batch of ray directions. Returns (N, 3) HDR RGB."""
    dirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True).clamp(min=1e-10)
    encoded = embeddirs_fn(dirs)
    return sky(encoded)


def _run_network(pts, viewdirs, model, embed_fn, embeddirs_fn, chunk):
    """Forward model in chunks. pts: (N_rays, N_samples, 3)."""
    N, S, _ = pts.shape
    pts_flat = pts.reshape(-1, 3)

    embedded = embed_fn(pts_flat)
    if embeddirs_fn is not None:
        dirs_flat = viewdirs[:, None, :].expand(N, S, 3).reshape(-1, 3)
        embedded = torch.cat([embedded, embeddirs_fn(dirs_flat)], -1)

    out_chunks = []
    for i in range(0, pts_flat.shape[0], chunk):
        out_chunks.append(model(embedded[i:i + chunk]))
    return torch.cat(out_chunks, 0).reshape(N, S, -1)


def render_rays(rays_o, rays_d, coarse_model, fine_model, embed_fn, embeddirs_fn,
                cfg: NerfConfig, *, near=None, far=None, perturb=True, target_depth=None):
    """Render a batch of rays with coarse + fine hierarchical sampling.

    target_depth: (N_rays,) tensor — only used when cfg.depth_hint_enabled is True.
    Returns: rgb_fine (N, 3), rgb_coarse (N, 3)
    """
    device = rays_o.device
    N = rays_o.shape[0]
    _near = near if near is not None else cfg.near
    _far  = far  if far  is not None else cfg.far

    # ── Coarse: stratified sampling ──────────────────────────────────────────
    t = torch.linspace(0., 1., cfg.N_samples, device=device)
    z_coarse = _near * (1. - t) + _far * t
    z_coarse = z_coarse.expand(N, cfg.N_samples)

    if perturb:
        mids  = 0.5 * (z_coarse[:, 1:] + z_coarse[:, :-1])
        upper = torch.cat([mids, z_coarse[:, -1:]], -1)
        lower = torch.cat([z_coarse[:, :1], mids],  -1)
        z_coarse = lower + (upper - lower) * torch.rand_like(z_coarse)

    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_coarse[:, :, None]
    raw_c = _run_network(pts, rays_d, coarse_model, embed_fn, embeddirs_fn, cfg.chunk)
    rgb_c, _, _, weights, _ = raw2outputs(raw_c, z_coarse, rays_d,
                                          cfg.raw_noise_std, cfg.white_bkgd)

    if cfg.N_importance == 0:
        return rgb_c, rgb_c

    # ── Fine: hierarchical sampling (or depth-hint window) ───────────────────
    z_mid = 0.5 * (z_coarse[:, 1:] + z_coarse[:, :-1])

    if cfg.depth_hint_enabled and target_depth is not None:
        t_hit  = target_depth.to(device)
        t_low  = (t_hit - cfg.depth_window).clamp(min=_near)
        t_high = (t_hit + cfg.depth_window_end).clamp(max=_far)
        u      = torch.rand(N, cfg.N_importance, device=device)
        z_fine = t_low[:, None] + (t_high - t_low)[:, None] * u
    else:
        z_fine = sample_pdf(z_mid, weights[:, 1:-1], cfg.N_importance, det=not perturb)
        z_fine = z_fine.detach()

    z_vals, _ = torch.sort(torch.cat([z_coarse, z_fine], -1), -1)
    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[:, :, None]
    raw_f = _run_network(pts, rays_d, fine_model, embed_fn, embeddirs_fn, cfg.chunk)
    rgb_f, _, _, _, _ = raw2outputs(raw_f, z_vals, rays_d, cfg.raw_noise_std, cfg.white_bkgd)

    return rgb_f, rgb_c


def render_rays_depth(rays_o, rays_d, fine_model, embed_fn, embeddirs_fn, cfg: NerfConfig,
                      t_hit, *, near=None, far=None, perturb=True, return_acc=False,
                      bg_color=None):
    """Single-pass render for foreground rays using OptiX depth as surface prior.

    Skips the coarse network entirely: samples depth_window_samples points in
    [t_hit - depth_window, t_hit + depth_window_end], runs only the fine network.
    bg_color: (N, 3) tensor — composited where acc < 1 instead of white. Ignored when None.
    """
    device = rays_o.device
    N = rays_o.shape[0]
    _near = near if near is not None else cfg.near
    _far  = far  if far  is not None else cfg.far
    n     = cfg.depth_window_samples

    t_low  = (t_hit.to(device) - cfg.depth_window).clamp(min=_near)
    t_high = (t_hit.to(device) + cfg.depth_window_end).clamp(max=_far)

    if perturb:
        u = torch.rand(N, n, device=device)
    else:
        u = torch.linspace(0., 1., n, device=device).expand(N, n)

    z_vals, _ = torch.sort(t_low[:, None] + (t_high - t_low)[:, None] * u, dim=-1)

    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[:, :, None]
    raw = _run_network(pts, rays_d, fine_model, embed_fn, embeddirs_fn, cfg.chunk)
    # use bg_color if provided, else fall back to white_bkgd flag
    _bg = bg_color if bg_color is not None else None
    _white = cfg.white_bkgd if bg_color is None else False
    rgb, _, acc, _, _ = raw2outputs(raw, z_vals, rays_d, cfg.raw_noise_std, _white, bg_color=_bg)
    if return_acc:
        return rgb, acc
    return rgb


def render_image(model_bundle, H: int, W: int, focal_x: float, pose_4x4,
                 cfg: NerfConfig, *,
                 focal_y: float | None = None,
                 cx: float | None = None,
                 cy: float | None = None,
                 target_depth=None) -> np.ndarray:
    """Render a full image. Returns (H, W, 3) float32 numpy array.

    pose_4x4: 4×4 or 3×4 camera-to-world matrix (numpy or tensor).
    target_depth: (H, W) depth map. When cfg.depth_hint_enabled is True and
        target_depth is provided, pixels with depth>0 are rendered via
        render_rays_depth (single-pass, fine network only); background pixels
        (depth==0) use the sky MLP (if cfg.train_background) or the traditional
        coarse+fine path.
    """
    coarse, fine, embed_fn, embeddirs_fn, device, sky = model_bundle

    focal_y = focal_y if focal_y is not None else focal_x
    cx = cx if cx is not None else W / 2.0
    cy = cy if cy is not None else H / 2.0

    K = np.array([[focal_x, 0, cx], [0, focal_y, cy], [0, 0, 1]], dtype=np.float32)

    if isinstance(pose_4x4, torch.Tensor):
        pose_np = pose_4x4.cpu().numpy().astype(np.float32)
    else:
        pose_np = np.array(pose_4x4, dtype=np.float32)

    rays_o_np, rays_d_np = get_rays_np(H, W, K, pose_np)
    rays_o_all = torch.tensor(rays_o_np.reshape(-1, 3), device=device)
    rays_d_all = torch.tensor(rays_d_np.reshape(-1, 3), device=device)
    N_all = rays_o_all.shape[0]

    t_hit_all = None
    fg_mask   = None
    if cfg.depth_hint_enabled and target_depth is not None:
        if isinstance(target_depth, np.ndarray):
            t_hit_all = torch.tensor(target_depth.reshape(-1), device=device, dtype=torch.float32)
        else:
            t_hit_all = target_depth.reshape(-1).to(device)
        fg_mask = t_hit_all > 1e-6  # True = ray hit geometry

    coarse.eval()
    fine.eval()
    if sky is not None:
        sky.eval()
    result_rgb = torch.zeros(N_all, 3, device=device)

    with torch.no_grad():
        if fg_mask is not None and fg_mask.any():
            fg_idx = fg_mask.nonzero(as_tuple=True)[0]
            bg_idx = (~fg_mask).nonzero(as_tuple=True)[0]

            for i in range(0, fg_idx.numel(), cfg.chunk):
                idx = fg_idx[i:i + cfg.chunk]
                bg_col = (_eval_sky(sky, embeddirs_fn, rays_d_all[idx])
                          if sky is not None else None)
                result_rgb[idx] = render_rays_depth(
                    rays_o_all[idx], rays_d_all[idx],
                    fine, embed_fn, embeddirs_fn, cfg,
                    t_hit_all[idx], perturb=False, bg_color=bg_col)

            if sky is not None:
                # background = sky MLP evaluated per direction
                for i in range(0, bg_idx.numel(), cfg.chunk):
                    idx = bg_idx[i:i + cfg.chunk]
                    result_rgb[idx] = _eval_sky(sky, embeddirs_fn, rays_d_all[idx])
            else:
                for i in range(0, bg_idx.numel(), cfg.chunk):
                    idx = bg_idx[i:i + cfg.chunk]
                    rgb, _ = render_rays(rays_o_all[idx], rays_d_all[idx],
                                         coarse, fine, embed_fn, embeddirs_fn,
                                         cfg, perturb=False)
                    result_rgb[idx] = rgb
        else:
            for i in range(0, N_all, cfg.chunk):
                ro = rays_o_all[i:i + cfg.chunk]
                rd = rays_d_all[i:i + cfg.chunk]
                rgb, _ = render_rays(ro, rd, coarse, fine, embed_fn, embeddirs_fn,
                                     cfg, perturb=False)
                result_rgb[i:i + cfg.chunk] = rgb

    return result_rgb.reshape(H, W, 3).cpu().numpy().astype(np.float32)


def query_radiance(model_bundle, origins_np: np.ndarray, dirs_np: np.ndarray,
                   cfg: NerfConfig, *, t_hits_np: np.ndarray | None = None) -> np.ndarray:
    """Query NeRF radiance for secondary rays (indirect irradiance pass).

    origins_np: (N, 3) — ray origins, typically surface point + eps·normal
    dirs_np:    (N, 3) — hemisphere sample directions
    t_hits_np:  (N,)  — OptiX hit distance; used only when cfg.depth_hint_enabled is True
    Returns:    (N, 3) float32 RGB colour per ray
    """
    coarse, fine, embed_fn, embeddirs_fn, device, sky = model_bundle

    rays_o = torch.tensor(origins_np, device=device, dtype=torch.float32)
    rays_d = torch.tensor(dirs_np,    device=device, dtype=torch.float32)

    t_hit = None
    if t_hits_np is not None and cfg.depth_hint_enabled:
        t_hit = torch.tensor(t_hits_np, device=device, dtype=torch.float32)

    # Secondary rays start on the surface — use a small near to not skip nearby geometry
    secondary_near = 0.01

    coarse.eval()
    fine.eval()
    if sky is not None:
        sky.eval()
    results = []
    with torch.no_grad():
        for i in range(0, rays_o.shape[0], cfg.chunk):
            ro = rays_o[i:i + cfg.chunk]
            rd = rays_d[i:i + cfg.chunk]
            td = t_hit[i:i + cfg.chunk] if t_hit is not None else None
            rgb, _ = render_rays(ro, rd, coarse, fine, embed_fn, embeddirs_fn,
                                  cfg, near=secondary_near, perturb=False, target_depth=td)
            results.append(rgb)

    return torch.cat(results, 0).cpu().numpy().astype(np.float32)
