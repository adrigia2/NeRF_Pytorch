# Vendored from https://github.com/yenchenlin/nerf-pytorch
# Original file: run_nerf_helpers.py
# Commit: 63a5a630c9abd62b0f21c08703d0ac2ea7d4b9dd
# License: MIT
# Modifications: extracted get_rays_np; added raw2outputs (ported from run_nerf.py);
#   removed ndc_rays (unused for object-centric scenes).

import numpy as np
import torch
import torch.nn.functional as F


def get_rays_np(H, W, K, c2w):
    """Generate camera rays for every pixel.

    K: 3×3 intrinsic matrix [[fx,0,cx],[0,fy,cy],[0,0,1]]
    c2w: 3×4 or 4×4 camera-to-world matrix
    Returns rays_o, rays_d each (H, W, 3) float32.
    """
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                        np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([
        (i - K[0][2]) / K[0][0],
        -(j - K[1][2]) / K[1][1],
        -np.ones_like(i),
    ], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o.astype(np.float32), rays_d.astype(np.float32)


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0.0, bg_color=None,
                rgb_activation="softplus"):
    """Convert raw NeRF output to RGB, depth and compositing weights.

    raw: (..., N_samples, 4) — [RGB, sigma] from network
    z_vals: (..., N_samples)
    rays_d: (..., 3)
    rgb_activation: "exp" (HDR) or "softplus".
    Returns: rgb_map, disp_map, acc_map, weights, depth_map
    """
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], -1)
    # convert t-distances to world-space distances
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    if rgb_activation == "exp":
        rgb = torch.exp(raw[..., :3])
    elif rgb_activation == "softplus":
        rgb = F.softplus(raw[..., :3])
    else:
        raise ValueError(f"Unknown rgb_activation: {rgb_activation!r} "
                         f"(expected 'exp' or 'softplus')")

    noise = 0.0
    if raw_noise_std > 0.0:
        noise = torch.randn_like(raw[..., 3]) * raw_noise_std

    alpha = 1.0 - torch.exp(-F.relu(raw[..., 3] + noise) * dists)
    # exclusive cumulative product of (1 - alpha) for transmittance
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], -1), -1
    )[..., :-1]

    rgb_map   = torch.sum(weights[..., None] * rgb, -2)
    depth_map = torch.sum(weights * z_vals, -1)
    acc_map   = weights.sum(-1)
    disp_map  = 1.0 / torch.clamp(
        depth_map / torch.clamp(acc_map, min=1e-10), min=1e-10)

    if bg_color is not None:
        rgb_map = rgb_map + (1.0 - acc_map[..., None]) * bg_color

    return rgb_map, disp_map, acc_map, weights, depth_map
