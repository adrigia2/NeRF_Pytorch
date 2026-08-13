from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .config import NerfConfig
from .rays import get_rays_np, raw2outputs


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


def render_rays_depth(rays_o, rays_d, model, embed_fn, embeddirs_fn, cfg: NerfConfig,
                      t_hit, *, near=None, far=None, return_acc=False,
                      bg_color=None, noise_std: float = 0.0,
                      window=None, window_end=None, n_samples=None):
    """Single-pass render for a batch of rays using a known surface distance t_hit.

    Samples n_samples points in [t_hit - window, t_hit + window_end].
    If window/window_end/n_samples are None, the cfg foreground defaults are used.
    bg_color: (N, 3) composited where acc < 1 instead of black. Ignored when None.
    Returns rgb (N, 3), and acc (N,) when return_acc=True.

    rays_d is normalized here: t_hit is a metric distance (OptiX depth, or a sphere
    radius), so direction and distance must share the same parametrization.

    noise_std: rumore gaussiano sulla densità, regolarizzatore di training. Il
    default 0.0 rende deterministico ogni percorso di inferenza; solo il forward
    che produce il gradiente passa cfg.raw_noise_std (train.py).
    """
    device = rays_o.device
    N = rays_o.shape[0]
    rays_d = F.normalize(rays_d, dim=-1)
    _near = near if near is not None else cfg.near
    _far  = far  if far  is not None else cfg.far
    _win      = window    if window    is not None else cfg.depth_window
    _win_end  = window_end if window_end is not None else cfg.depth_window_end
    _n        = n_samples  if n_samples  is not None else cfg.depth_window_samples

    t_low  = (t_hit.to(device) - _win).clamp(min=_near)
    t_high = (t_hit.to(device) + _win_end).clamp(max=_far)

    u = torch.linspace(0., 1., _n, device=device).expand(N, _n)
    z_vals, _ = torch.sort(t_low[:, None] + (t_high - t_low)[:, None] * u, dim=-1)

    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[:, :, None]
    raw = _run_network(pts, rays_d, model, embed_fn, embeddirs_fn, cfg.chunk)
    rgb, _, acc, _, _ = raw2outputs(raw, z_vals, rays_d, noise_std,
                                    bg_color=bg_color,
                                    rgb_activation=cfg.rgb_activation)
    if return_acc:
        return rgb, acc
    return rgb


def render_bg(dirs, center: torch.Tensor, sphere_radius: float,
              model, embed_fn, embeddirs_fn, cfg: NerfConfig,
              *, return_acc=False, noise_std: float = 0.0):
    """Render background rays as a spherical shell centred at `center`.

    Each ray is re-anchored at the scene centre and marched outward:
        p(z) = center + z * normalize(dir),  z around sphere_radius

    This makes the environment a purely directional function (no parallax between
    views), automatically consistent across novel views.

    Returns rgb (N, 3), and acc (N,) when return_acc=True.
    """
    N = dirs.shape[0]
    device = dirs.device

    dirs_n = F.normalize(dirs, dim=-1)
    # Origin = centre broadcast to (N, 3); t_hit = sphere radius for all rays
    origins = center.unsqueeze(0).expand(N, 3)
    t_hit   = torch.full((N,), sphere_radius, device=device, dtype=torch.float32)

    return render_rays_depth(
        origins, dirs_n, model, embed_fn, embeddirs_fn, cfg,
        t_hit,
        near=cfg.near,
        far=sphere_radius + cfg.bg_depth_window_end + 1.0,
        return_acc=return_acc,
        noise_std=noise_std,
        window=cfg.bg_depth_window,
        window_end=cfg.bg_depth_window_end,
        n_samples=cfg.depth_window_samples,
    )


def render_unified(rays_o, rays_d, depths, in_mask, model, embed_fn, embeddirs_fn,
                   cfg: NerfConfig, center: torch.Tensor, sphere_radius: float,
                   *, return_acc=False, noise_std: float = 0.0):
    """Single fixed-shape render for both fg (mesh hit) and bg (sphere shell) rays.

    in_mask=True  → fg: use ray origin + t_hit from depths, mesh depth window.
    in_mask=False → bg: use `center` as origin, t_hit=sphere_radius.
    All intermediate tensors have shape (N, depth_window_samples, ...) — constant across iters.
    Requires cfg.depth_window_samples for both fg and bg (use the same value).

    Directions are normalized for both branches: `depths` is a metric OptiX distance
    and `sphere_radius` a metric radius, so the two branches differ only in origin and
    t_hit — the only difference that is actually meaningful.
    """
    device = rays_o.device
    N = rays_o.shape[0]
    n_s = cfg.depth_window_samples
    sphere_r = torch.full((N,), sphere_radius, device=device, dtype=torch.float32)

    eff_dir    = F.normalize(rays_d, dim=-1)
    eff_origin = torch.where(in_mask[:, None], rays_o, center.unsqueeze(0).expand(N, 3))
    eff_t_hit  = torch.where(in_mask, depths, sphere_r)

    win_lo = torch.where(in_mask,
                         torch.full_like(depths, cfg.depth_window),
                         torch.full_like(depths, cfg.bg_depth_window))
    win_hi = torch.where(in_mask,
                         torch.full_like(depths, cfg.depth_window_end),
                         torch.full_like(depths, cfg.bg_depth_window_end))

    t_low  = (eff_t_hit - win_lo).clamp(min=cfg.near)
    t_high = (eff_t_hit + win_hi).clamp(max=cfg.far)

    u = torch.linspace(0., 1., n_s, device=device).expand(N, n_s)
    z_vals, _ = torch.sort(t_low[:, None] + (t_high - t_low)[:, None] * u, dim=-1)
    pts = eff_origin[:, None, :] + eff_dir[:, None, :] * z_vals[:, :, None]
    raw = _run_network(pts, eff_dir, model, embed_fn, embeddirs_fn, cfg.chunk)
    rgb, _, acc, _, _ = raw2outputs(raw, z_vals, eff_dir, noise_std,
                                    rgb_activation=cfg.rgb_activation)
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
    target_depth: (H, W) depth map from OptiX. Pixels with depth>0 are foreground
        (rendered with mesh window); pixels with depth==0 are background (rendered
        as the spherical shell from the scene centre).
    """
    model, embed_fn, embeddirs_fn, device, center, sphere_radius = model_bundle

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

    if target_depth is None:
        raise ValueError("render_image requires target_depth (OptiX depth map).")

    if isinstance(target_depth, np.ndarray):
        t_hit_all = torch.tensor(target_depth.reshape(-1), device=device, dtype=torch.float32)
    else:
        t_hit_all = target_depth.reshape(-1).to(device)

    fg_mask = t_hit_all > 1e-6

    model.eval()
    result_rgb = torch.zeros(N_all, 3, device=device)

    with torch.no_grad():
        fg_idx = fg_mask.nonzero(as_tuple=True)[0]
        bg_idx = (~fg_mask).nonzero(as_tuple=True)[0]

        for i in range(0, fg_idx.numel(), cfg.chunk):
            idx = fg_idx[i:i + cfg.chunk]
            result_rgb[idx] = render_rays_depth(
                rays_o_all[idx], rays_d_all[idx],
                model, embed_fn, embeddirs_fn, cfg,
                t_hit_all[idx])

        for i in range(0, bg_idx.numel(), cfg.chunk):
            idx = bg_idx[i:i + cfg.chunk]
            result_rgb[idx] = render_bg(
                rays_d_all[idx], center, sphere_radius,
                model, embed_fn, embeddirs_fn, cfg)

    return result_rgb.reshape(H, W, 3).cpu().numpy().astype(np.float32)


def bake_envmap(model_bundle, cfg: NerfConfig, width: int, height: int,
                *, yaw_degrees: float = 0.0) -> np.ndarray:
    """Bake the NeRF background sphere into an equirectangular envmap.

    Inverts the lookup convention of sampleEnvmap in deviceProgramsIrradiance.cu
    (world Z-up, azimuth -atan2(dy, dx) → Blender equirectangular convention, yaw as a
    U-axis shift), so the baked EXR can be fed to IrradianceGenerator.set_inputs
    with the same skybox_yaw_degrees and is pixel-comparable to a real skybox file.

    Returns (height, width, 3) float32 radiance in the NeRF training scale.
    """
    model, embed_fn, embeddirs_fn, device, center, sphere_radius = model_bundle

    yaw_offset_u = (yaw_degrees * np.pi / 180.0) / (2.0 * np.pi)
    yaw_offset_u -= np.floor(yaw_offset_u)  # wrap a [0, 1) come Irradiance_Generator.cpp

    px = (np.arange(width,  dtype=np.float32) + 0.5) / width    # u
    py = (np.arange(height, dtype=np.float32) + 0.5) / height   # v
    u, v = np.meshgrid(px, py, indexing="xy")

    # v = 0.5 - asin(dz)/π  →  dz = sin(π·(0.5 - v))
    elev   = np.pi * (0.5 - v)
    dz     = np.sin(elev)
    cos_el = np.cos(elev)
    # u = 0.5 - atan2(dy, dx)/(2π) + yaw_offset_u  →  atan2(dy,dx) = 2π·(0.5 + yaw_offset_u - u)
    # → dx = cos(φ)·cos_el, dy = sin(φ)·cos_el  with φ = 2π·(0.5 + yaw_offset_u - u)
    phi = 2.0 * np.pi * (0.5 + yaw_offset_u - u)
    dx  =  np.cos(phi) * cos_el
    dy  =  np.sin(phi) * cos_el

    dirs = torch.tensor(np.stack([dx, dy, dz], axis=-1).reshape(-1, 3),
                        device=device, dtype=torch.float32)

    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, dirs.shape[0], cfg.chunk):
            chunks.append(render_bg(dirs[i:i + cfg.chunk], center, sphere_radius,
                                    model, embed_fn, embeddirs_fn, cfg))
    rgb = torch.cat(chunks, 0).cpu().numpy().astype(np.float32)
    return rgb.reshape(height, width, 3)


def query_radiance(model_bundle, origins_np, dirs_np,
                   cfg: NerfConfig, *, t_hits_np=None,
                   return_torch: bool = False):
    """Query NeRF radiance for secondary rays (indirect irradiance pass).

    origins_np: (N, 3) — ray origins (surface point + eps·normal)
    dirs_np:    (N, 3) — hemisphere sample directions
    t_hits_np:  (N,)  — OptiX hit distance; rays with t_hit>0 use the mesh window,
                         rays that miss geometry use the spherical shell from centre.

    Gli input possono essere array NumPy oppure tensori torch già sul device:
    `torch.as_tensor` non copia quando device e dtype coincidono già. Con
    return_torch=True anche l'uscita resta sul device — serve al bake spec_cone
    condiviso, dove le radianze vengono classificate per camera in torch e un
    round-trip GPU→CPU→GPU costerebbe centinaia di MB per tile.

    Returns:    (N, 3) float32 RGB per raggio (NumPy, o tensore se return_torch)
    """
    model, embed_fn, embeddirs_fn, device, center, sphere_radius = model_bundle

    rays_o = torch.as_tensor(origins_np, device=device, dtype=torch.float32)
    rays_d = torch.as_tensor(dirs_np,    device=device, dtype=torch.float32)

    t_hit = None
    if t_hits_np is not None:
        t_hit = torch.as_tensor(t_hits_np, device=device, dtype=torch.float32)

    model.eval()
    results = []
    with torch.no_grad():
        for i in range(0, rays_o.shape[0], cfg.chunk):
            ro  = rays_o[i:i + cfg.chunk]
            rd  = rays_d[i:i + cfg.chunk]
            th  = t_hit[i:i + cfg.chunk] if t_hit is not None else None

            if th is not None:
                fg_mask = th > 1e-6
                chunk_rgb = torch.zeros(ro.shape[0], 3, device=device)

                fg_idx = fg_mask.nonzero(as_tuple=True)[0]
                bg_idx = (~fg_mask).nonzero(as_tuple=True)[0]

                if fg_idx.numel() > 0:
                    chunk_rgb[fg_idx] = render_rays_depth(
                        ro[fg_idx], rd[fg_idx],
                        model, embed_fn, embeddirs_fn, cfg,
                        th[fg_idx], near=0.01)

                if bg_idx.numel() > 0:
                    chunk_rgb[bg_idx] = render_bg(
                        rd[bg_idx], center, sphere_radius,
                        model, embed_fn, embeddirs_fn, cfg)
            else:
                # No hit info: treat all as background
                chunk_rgb = render_bg(
                    rd, center, sphere_radius,
                    model, embed_fn, embeddirs_fn, cfg)

            results.append(chunk_rgb)

    out = torch.cat(results, 0)
    return out if return_torch else out.cpu().numpy().astype(np.float32)
