"""
nerf_module.py — Shared NeRF model, training, and query utilities.

Single source of truth used by tiny_nerf_pytorch.ipynb and images_generator.py.
Checkpoint format is backward-compatible with the existing tinynerf_model_cache.pkl.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NerfConfig:
    # Positional encoding
    num_encoding_functions: int = 10
    num_encoding_functions_views: int = 4
    # Model
    filter_size: int = 256
    num_layers: int = 8
    skips: tuple = (4,)
    use_viewdirs: bool = True
    # Sampling
    near: float = 1.0
    far: float = 20.0
    depth_window: float = 0.5
    depth_window_end: float = 0.0   # if >0, linspace end override (asym front-biased window); 0=use depth_window
    depth_samples_per_ray: int = 64
    chunk_size: int = 16384
    scene_scale: float = 3.0
    # LR schedule: lr decays to 10% after lrate_decay * 1000 steps
    lrate_decay: int = 250
    # HDR mode: predict linear-light radiance (softplus output, Reinhard loss, linear targets)
    hdr_mode: bool = True


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def meshgrid_xy(tensor1: torch.Tensor, tensor2: torch.Tensor):
    """meshgrid with xy indexing: returns (ii, jj) each shape (len(t2), len(t1))."""
    ii, jj = torch.meshgrid(tensor1, tensor2, indexing='ij')
    return ii.transpose(-1, -2), jj.transpose(-1, -2)


def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    cumprod = torch.cumprod(tensor, dim=-1)
    cumprod = torch.roll(cumprod, 1, -1)
    cumprod[..., 0] = 1.0
    return cumprod


def get_minibatches(inputs: torch.Tensor, chunksize: int = 16384):
    return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class TinyNerfModel(nn.Module):
    def __init__(self, filter_size: int = 128, num_encoding_functions: int = 6):
        super().__init__()
        in_dim = 3 + 3 * 2 * num_encoding_functions
        self.layer1 = nn.Linear(in_dim, filter_size)
        self.layer2 = nn.Linear(filter_size, filter_size)
        self.layer3 = nn.Linear(filter_size, 4)
        # Bias density channel toward positive values so weights accumulate from the start.
        # Prevents the "empty scene" sigma-collapse local minimum.
        with torch.no_grad():
            self.layer3.bias[3].fill_(1.0)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


VeryTinyNerfModel = TinyNerfModel  # backward-compatible alias


class NeRFModel(nn.Module):
    """Full NeRF MLP — 8 layers × 256 units, skip connection at layer 4,
    optional view-dependent RGB branch (yenchenlin architecture)."""

    def __init__(
        self,
        D: int = 8,
        W: int = 256,
        skips: tuple = (4,),
        num_encoding_functions: int = 10,
        use_viewdirs: bool = True,
        num_encoding_functions_views: int = 4,
    ):
        super().__init__()
        self.D = D
        self.W = W
        self.skips = set(skips)
        self.use_viewdirs = use_viewdirs

        pt_dim = 3 + 3 * 2 * num_encoding_functions

        # Build point MLP layers, inserting skip concatenation where needed.
        layers = []
        in_dim = pt_dim
        for i in range(D):
            layers.append(nn.Linear(in_dim, W))
            # After layer i, if it's a skip layer the *next* layer receives W + pt_dim.
            in_dim = (W + pt_dim) if i in self.skips else W
        self.pts_linears = nn.ModuleList(layers)

        if use_viewdirs:
            self.alpha_linear   = nn.Linear(W, 1)
            self.feature_linear = nn.Linear(W, W)
            view_dim = 3 + 3 * 2 * num_encoding_functions_views
            self.views_linear = nn.Linear(W + view_dim, W // 2)
            self.rgb_linear   = nn.Linear(W // 2, 3)
            # Positive density bias to prevent empty-scene sigma collapse.
            nn.init.constant_(self.alpha_linear.bias, 1.0)
        else:
            self.output_linear = nn.Linear(W, 4)
            with torch.no_grad():
                self.output_linear.bias[3].fill_(1.0)

    def forward(self, x: torch.Tensor, dirs: torch.Tensor = None) -> torch.Tensor:
        pts_enc = x
        h = x
        for i, layer in enumerate(self.pts_linears):
            h = F.relu(layer(h))
            if i in self.skips:
                h = torch.cat([pts_enc, h], dim=-1)

        if self.use_viewdirs and dirs is not None:
            alpha = self.alpha_linear(h)
            feat  = self.feature_linear(h)
            h     = F.relu(self.views_linear(torch.cat([feat, dirs], dim=-1)))
            rgb   = self.rgb_linear(h)
            return torch.cat([rgb, alpha], dim=-1)   # (..., 4)
        else:
            return self.output_linear(h)


# ──────────────────────────────────────────────────────────────────────────────
# Positional encoding
# ──────────────────────────────────────────────────────────────────────────────

def positional_encoding(
    tensor: torch.Tensor,
    num_encoding_functions: int = 6,
    include_input: bool = True,
    log_sampling: bool = True,
) -> torch.Tensor:
    encoding = [tensor] if include_input else []
    if log_sampling:
        freqs = 2.0 ** torch.linspace(
            0.0, num_encoding_functions - 1, num_encoding_functions,
            dtype=tensor.dtype, device=tensor.device)
    else:
        freqs = torch.linspace(
            2.0 ** 0.0, 2.0 ** (num_encoding_functions - 1), num_encoding_functions,
            dtype=tensor.dtype, device=tensor.device)
    for freq in freqs:
        encoding.append(torch.sin(tensor * freq))
        encoding.append(torch.cos(tensor * freq))
    return encoding[0] if len(encoding) == 1 else torch.cat(encoding, dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# Ray bundle  (returns (H, W, 3) — height-major, matching image layout)
# ──────────────────────────────────────────────────────────────────────────────

def get_ray_bundle(height: int, width: int, focal_length, c2w: torch.Tensor,
                   focal_y=None):
    """Returns (origins, dirs) each (H, W, 3).

    focal_length is the horizontal focal (fl_x).  Pass focal_y (fl_y) when
    pixels are non-square; if omitted, fl_x is used for both axes.
    """
    fx = focal_length.item() if isinstance(focal_length, torch.Tensor) else float(focal_length)
    fy = float(focal_y) if focal_y is not None else fx
    ii, jj = meshgrid_xy(
        torch.arange(width,  dtype=c2w.dtype, device=c2w.device),
        torch.arange(height, dtype=c2w.dtype, device=c2w.device),
    )
    # ii[h,w] = w (column), jj[h,w] = h (row)
    directions = torch.stack(
        [(ii - width * 0.5) / fx, -(jj - height * 0.5) / fy, -torch.ones_like(ii)], dim=-1
    )
    ray_dirs    = torch.sum(directions[..., None, :] * c2w[:3, :3], dim=-1)
    ray_origins = c2w[:3, -1].expand(ray_dirs.shape)
    return ray_origins, ray_dirs


# ──────────────────────────────────────────────────────────────────────────────
# Sampling helpers (kept for bake cell backward compatibility)
# ──────────────────────────────────────────────────────────────────────────────

def compute_query_points_from_rays(
    ray_origins, ray_directions, near_thresh, far_thresh, num_samples, randomize=True
):
    depth_values = torch.linspace(near_thresh, far_thresh, num_samples).to(ray_origins)
    if randomize:
        noise_shape = list(ray_origins.shape[:-1]) + [num_samples]
        depth_values = depth_values + torch.rand(noise_shape).to(ray_origins) * (far_thresh - near_thresh) / num_samples
    query_points = ray_origins[..., None, :] + ray_directions[..., None, :] * depth_values[..., :, None]
    return query_points, depth_values


def compute_query_points_from_depth(
    ray_origins, ray_directions, depth_map, num_samples,
    depth_window=0.15, randomize=True, min_depth=1e-4, depth_is_euclidean=True
):
    if depth_map.dim() == 3 and depth_map.shape[-1] == 1:
        depth_map = depth_map[..., 0]
    if depth_map.shape != ray_origins.shape[:2]:
        if depth_map.t().shape == ray_origins.shape[:2]:
            depth_map = depth_map.t()
        else:
            raise ValueError(
                f"Depth map {tuple(depth_map.shape)} incompatible with rays {tuple(ray_origins.shape[:2])}")
    depth_map = depth_map.to(ray_origins)
    valid = torch.isfinite(depth_map) & (depth_map > min_depth)
    fallback = torch.median(depth_map[valid]) if valid.any() else torch.tensor(1.0).to(ray_origins)
    depth_center = torch.where(valid, depth_map, fallback)
    offsets = torch.linspace(-depth_window, depth_window, num_samples).to(ray_origins)
    if depth_is_euclidean:
        ray_norm = torch.linalg.norm(ray_directions, dim=-1).clamp_min(1e-8)
        depth_t  = depth_center / ray_norm
        depth_values = depth_t[..., None] + offsets[None, None, :] / ray_norm[..., None]
    else:
        depth_values = depth_center[..., None] + offsets
    if randomize and num_samples > 1:
        mids  = 0.5 * (depth_values[..., 1:] + depth_values[..., :-1])
        upper = torch.cat([mids, depth_values[..., -1:]], dim=-1)
        lower = torch.cat([depth_values[..., :1], mids],  dim=-1)
        depth_values = lower + (upper - lower) * torch.rand_like(depth_values)
    depth_values = depth_values.clamp(min=min_depth)
    query_points = ray_origins[..., None, :] + ray_directions[..., None, :] * depth_values[..., :, None]
    return query_points, depth_values


# ──────────────────────────────────────────────────────────────────────────────
# Volume rendering
# ──────────────────────────────────────────────────────────────────────────────

def render_volume_density(radiance_field, ray_origins, depth_values, hdr_mode: bool = False):
    sigma = F.softplus(radiance_field[..., 3])
    if hdr_mode:
        # exp activation: log(pred)=z so gradient of log-MSE loss w.r.t. z never saturates.
        # softplus gradient dies for z<<0 (sigmoid(−10)≈4e-5); exp has no such dead zone.
        rgb = torch.exp(radiance_field[..., :3].clamp(-15.0, 8.0))
    else:
        rgb = torch.sigmoid(radiance_field[..., :3])
    one_e10 = torch.tensor([1e10], dtype=ray_origins.dtype, device=ray_origins.device)
    dists   = torch.cat([depth_values[..., 1:] - depth_values[..., :-1],
                          one_e10.expand(depth_values[..., :1].shape)], dim=-1)
    alpha   = 1.0 - torch.exp(-sigma * dists)
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)
    rgb_map   = (weights[..., None] * rgb).sum(dim=-2)
    depth_map = (weights * depth_values).sum(dim=-1)
    acc_map   = weights.sum(-1)
    return rgb_map, depth_map, acc_map


# ──────────────────────────────────────────────────────────────────────────────
# Core forward pass  (works for any batch shape * before the 3 dims)
# ──────────────────────────────────────────────────────────────────────────────

def _encode_and_run(query_points: torch.Tensor, model, cfg: NerfConfig,
                    dirs: torch.Tensor = None):
    """query_points: (*, S, 3), dirs: (*, 3) optional → radiance_field: (*, S, 4)"""
    shape     = query_points.shape          # (*, S, 3)
    S         = shape[-2]
    batch_shp = shape[:-2]

    flat_pts = query_points.reshape(-1, 3) / cfg.scene_scale    # (N, 3) — not encoded yet
    N = flat_pts.shape[0]

    use_views = (
        dirs is not None
        and getattr(cfg, 'use_viewdirs', False)
        and getattr(model, 'use_viewdirs', False)
    )
    if use_views:
        dirs_norm = F.normalize(dirs, dim=-1)                              # (*, 3)
        dirs_flat = dirs_norm.unsqueeze(-2).expand(*batch_shp, S, 3).reshape(-1, 3)
    else:
        dirs_flat = None

    # Encode and run model per-chunk so peak VRAM is O(chunk_size) not O(N)
    preds = []
    for i in range(0, N, cfg.chunk_size):
        ep = positional_encoding(flat_pts[i:i + cfg.chunk_size], cfg.num_encoding_functions)
        if dirs_flat is not None:
            ed = positional_encoding(dirs_flat[i:i + cfg.chunk_size], cfg.num_encoding_functions_views)
            preds.append(model(ep, ed))
        else:
            preds.append(model(ep))
    return torch.cat(preds, dim=0).reshape(*batch_shp, S, 4)


def run_one_iter(
    origins: torch.Tensor,
    dirs: torch.Tensor,
    model: TinyNerfModel,
    cfg: NerfConfig,
    target_depth: Optional[torch.Tensor] = None,
    randomize: bool = True,
):
    """Forward pass for rays of any shape (*, 3).

    target_depth: (*,) euclidean depth  → depth-guided sampling around t_hit.
      Rays with target_depth <= 1e-4 (BG / sky pixels) automatically fall back
      to stratified sampling over [cfg.near, cfg.far].
    Returns (rgb, depth_map, acc_map) each shape (*,) / (*,3).
    """
    if target_depth is not None:
        valid    = (target_depth > 1e-4)                                   # (*,) bool
        ray_norm = torch.linalg.norm(dirs, dim=-1).clamp_min(1e-8)        # (*,)
        depth_t  = target_depth / ray_norm                                  # (*,)

        S_total  = cfg.depth_samples_per_ray
        S_strat  = S_total // 2
        S_guided = S_total - S_strat

        # Stratified backbone: S_strat samples uniformly over [near, far].
        # These constrain the model in free-space so density outside the surface converges to 0,
        # making stratified-inference at viewer time produce correct results.
        dv_strat = torch.linspace(cfg.near, cfg.far, S_strat,
                                   device=origins.device, dtype=origins.dtype)
        if randomize and S_strat > 1:
            bin_w    = (cfg.far - cfg.near) / S_strat
            dv_strat = dv_strat + (torch.rand_like(dv_strat) - 0.5) * bin_w
        strat_br = dv_strat.expand(list(depth_t.shape) + [S_strat])        # (*,S_strat)

        # Depth-guided fine samples: S_guided samples concentrated near the surface.
        offsets = torch.linspace(-cfg.depth_window, cfg.depth_window_end,
                                  S_guided, device=origins.device, dtype=origins.dtype)
        if randomize and S_guided > 1:
            bin_w_g = (cfg.depth_window_end - (-cfg.depth_window)) / S_guided
            offsets = offsets + (torch.rand_like(offsets) - 0.5) * bin_w_g
        offsets_br = offsets.view(*([1] * depth_t.dim()), S_guided)
        dv_guided  = depth_t.unsqueeze(-1) + offsets_br / ray_norm.unsqueeze(-1)  # (*,S_guided)

        hybrid = torch.cat([strat_br, dv_guided], dim=-1)                  # (*,S_total)

        # BG fallback (valid=False): full stratified over S_total samples
        dv_strat_full = torch.linspace(cfg.near, cfg.far, S_total,
                                        device=origins.device, dtype=origins.dtype)
        if randomize and S_total > 1:
            bin_wf = (cfg.far - cfg.near) / S_total
            dv_strat_full = dv_strat_full + (torch.rand_like(dv_strat_full) - 0.5) * bin_wf

        depth_values = torch.where(valid.unsqueeze(-1), hybrid, dv_strat_full)
        depth_values, _ = torch.sort(depth_values, dim=-1)                 # front-to-back required by compositing
        depth_values = depth_values.clamp(min=1e-4)
    else:
        depth_values = torch.linspace(cfg.near, cfg.far, cfg.depth_samples_per_ray,
                                       device=origins.device, dtype=origins.dtype)
        if randomize:
            noise_shape  = list(origins.shape[:-1]) + [cfg.depth_samples_per_ray]
            depth_values = depth_values + torch.rand(noise_shape,
                                                      device=origins.device,
                                                      dtype=origins.dtype) \
                           * (cfg.far - cfg.near) / cfg.depth_samples_per_ray

    query_points = origins[..., None, :] + dirs[..., None, :] * depth_values[..., :, None]
    rf = _encode_and_run(query_points, model, cfg, dirs=dirs)
    return render_volume_density(rf, origins, depth_values, hdr_mode=getattr(cfg, 'hdr_mode', False))


def render_image(H, W, focal, c2w, model, cfg,
                  target_depth=None, randomize=False, focal_y=None):
    """Render a full image; returns (H, W, 3) rgb, (H, W) depth, (H, W) acc."""
    with torch.no_grad():
        origins, dirs = get_ray_bundle(H, W, focal, c2w, focal_y=focal_y)  # (H, W, 3)
        depth_hw = None
        if target_depth is not None:
            d = target_depth
            if d.shape == (W, H):
                d = d.t()
            if d.shape == (H, W):
                depth_hw = d
        rgb, dep, acc = run_one_iter(origins, dirs, model, cfg,
                                      target_depth=depth_hw, randomize=randomize)
    return rgb, dep, acc  # (H, W, 3), (H, W), (H, W)


# ──────────────────────────────────────────────────────────────────────────────
# Query for indirect irradiance (host-side, called by images_generator.py)
# ──────────────────────────────────────────────────────────────────────────────

def query_radiance(
    model: TinyNerfModel,
    origins_np: np.ndarray,
    dirs_np: np.ndarray,
    t_hits_np: np.ndarray,
    cfg: NerfConfig,
    device=None,
) -> np.ndarray:
    """Query NeRF color for M occluded rays. Returns (M, 3) float32 numpy."""
    M = origins_np.shape[0]
    if M == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if device is None:
        device = next(model.parameters()).device
    origins = torch.from_numpy(origins_np).float().to(device)
    dirs    = torch.from_numpy(dirs_np).float().to(device)
    t_hits  = torch.from_numpy(t_hits_np).float().to(device)
    with torch.no_grad():
        rgb, _, _ = run_one_iter(origins, dirs, model, cfg,
                                  target_depth=t_hits, randomize=False)
    return rgb.cpu().numpy().astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint I/O  (format compatible with existing tinynerf_model_cache.pkl)
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(path: str, model, optimizer,
                    iter_done: int, cfg: NerfConfig, seed: int = 9458):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        'iter_done':                    int(iter_done),
        'model_state_dict':             model.state_dict(),
        'optimizer_state_dict':         optimizer.state_dict(),
        'seed':                         int(seed),
        'model_type':                   'NeRFModel' if isinstance(model, NeRFModel) else 'TinyNerfModel',
        'num_encoding_functions':       int(cfg.num_encoding_functions),
        'num_encoding_functions_views': int(getattr(cfg, 'num_encoding_functions_views', 4)),
        'filter_size':                  int(cfg.filter_size),
        'num_layers':                   int(getattr(cfg, 'num_layers', 3)),
        'skips':                        tuple(getattr(cfg, 'skips', (4,))),
        'use_viewdirs':                 bool(getattr(cfg, 'use_viewdirs', False)),
        # Scene / sampling parameters — needed to reproduce inference conditions
        'near':                         float(cfg.near),
        'far':                          float(cfg.far),
        'scene_scale':                  float(cfg.scene_scale),
        'depth_samples_per_ray':        int(cfg.depth_samples_per_ray),
        'depth_window':                 float(cfg.depth_window),
        'depth_window_end':             float(cfg.depth_window_end),
        'chunk_size':                   int(cfg.chunk_size),
        'hdr_mode':                     bool(getattr(cfg, 'hdr_mode', True)),
    }
    with open(path, 'wb') as f:
        pickle.dump(payload, f)
    print(f"[NeRF] Checkpoint saved: {path}  (iter={iter_done})")


def load_checkpoint(path: str, device=None):
    """Returns (model, optimizer_state_dict | None, iter_done, cfg)."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    num_enc      = payload.get('num_encoding_functions', 6)
    num_enc_view = payload.get('num_encoding_functions_views', 4)
    filter_size  = payload.get('filter_size', 128)
    num_layers   = payload.get('num_layers', 3)
    skips        = payload.get('skips', (4,))
    use_viewdirs = payload.get('use_viewdirs', False)
    model_type   = payload.get('model_type', 'TinyNerfModel')
    hdr_mode     = payload.get('hdr_mode', None)
    if hdr_mode is None:
        print("[NeRF] WARNING: checkpoint has no 'hdr_mode' field (legacy). Defaulting to SDR (sigmoid).")
        hdr_mode = False

    cfg = NerfConfig(
        num_encoding_functions=num_enc,
        num_encoding_functions_views=num_enc_view,
        filter_size=filter_size,
        num_layers=num_layers,
        skips=skips,
        use_viewdirs=use_viewdirs,
        near=payload.get('near', 1.0),
        far=payload.get('far', 20.0),
        scene_scale=payload.get('scene_scale', 3.0),
        depth_samples_per_ray=payload.get('depth_samples_per_ray', 64),
        depth_window=payload.get('depth_window', 0.5),
        depth_window_end=payload.get('depth_window_end', 0.0),
        chunk_size=payload.get('chunk_size', 16384),
        hdr_mode=hdr_mode,
    )
    if model_type == 'NeRFModel':
        model = NeRFModel(
            D=num_layers, W=filter_size, skips=skips,
            num_encoding_functions=num_enc,
            use_viewdirs=use_viewdirs,
            num_encoding_functions_views=num_enc_view,
        ).to(device)
    else:
        model = TinyNerfModel(filter_size, num_enc).to(device)
    model.load_state_dict(payload['model_state_dict'])
    model.eval()
    return model, payload.get('optimizer_state_dict'), payload.get('iter_done', 0), cfg


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

def _load_exr_depth_np(path: str) -> np.ndarray:
    import importlib
    OpenEXR = importlib.import_module('OpenEXR')
    Imath   = importlib.import_module('Imath')
    f  = OpenEXR.InputFile(path)
    dw = f.header()['dataWindow']
    w  = dw.max.x - dw.min.x + 1
    h  = dw.max.y - dw.min.y + 1
    PT = Imath.PixelType(Imath.PixelType.FLOAT)
    chs = list(f.header()['channels'].keys())
    ch  = next((c for c in ('Z', 'R', 'Y', 'X') if c in chs), chs[0])
    return np.frombuffer(f.channel(ch, PT), dtype=np.float32).reshape(h, w)


def _load_exr_mask_np(path: str) -> np.ndarray:
    import importlib
    OpenEXR = importlib.import_module('OpenEXR')
    Imath   = importlib.import_module('Imath')
    f  = OpenEXR.InputFile(path)
    dw = f.header()['dataWindow']
    w  = dw.max.x - dw.min.x + 1
    h  = dw.max.y - dw.min.y + 1
    PT = Imath.PixelType(Imath.PixelType.FLOAT)
    chs = list(f.header()['channels'].keys())
    ch  = chs[0]
    return np.frombuffer(f.channel(ch, PT), dtype=np.float32).reshape(h, w)


def _load_exr_rgb_np(path: str, hdr_mode: bool = False) -> np.ndarray:
    """Load EXR as (H, W, 3) float32.

    hdr_mode=False (SDR): applies gamma 1/2.2 + clip to [0,1] — perceptually uniform
      targets that avoid the "predict-zero" local minimum in MSE training.
    hdr_mode=True  (HDR): returns linear-light radiance (clamped negative artefacts only),
      values can exceed 1.0. Use with Reinhard-mapped loss and softplus output activation.
    """
    import importlib
    OpenEXR = importlib.import_module('OpenEXR')
    Imath   = importlib.import_module('Imath')
    f  = OpenEXR.InputFile(path)
    dw = f.header()['dataWindow']
    w  = dw.max.x - dw.min.x + 1
    h  = dw.max.y - dw.min.y + 1
    PT = Imath.PixelType(Imath.PixelType.FLOAT)
    chs = list(f.header()['channels'].keys())
    def _ch(*names):
        for n in names:
            if n in chs:
                return np.frombuffer(f.channel(n, PT), dtype=np.float32).reshape(h, w)
        return np.zeros((h, w), dtype=np.float32)
    r = _ch('R'); g = _ch('G'); b = _ch('B')
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.maximum(rgb, 0.0)     # clamp EXR negative artefacts
    if not hdr_mode:
        rgb = np.power(rgb, 1.0 / 2.2)
        rgb = np.clip(rgb, 0.0, 1.0)
    return rgb.astype(np.float32)


def _resolve_path(path_str: str, json_dir: str) -> Optional[str]:
    if path_str is None:
        return None
    norm = path_str.replace('\\', os.sep).replace('/', os.sep)
    for candidate in [norm,
                       os.path.join(json_dir, norm),
                       os.path.join(os.getcwd(), norm),
                       os.path.join(os.getcwd(), norm.lstrip('./\\'))]:
        c = os.path.normpath(candidate)
        if os.path.exists(c):
            return c
    return None


class NerfDataset:
    """Loads a transforms_extended.json and pre-computes ray bundles for all frames.

    Requires per-frame depth_path and mask_path (generated by images_generator.py).
    Raises ValueError if those fields are absent.
    """

    def __init__(self, transforms_json_path: str, device=None, test_idx: int = None, hdr_mode: bool = True):
        if device is None:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = torch.device(device)
        self._hdr_mode = hdr_mode

        json_dir = os.path.dirname(os.path.abspath(transforms_json_path))
        with open(transforms_json_path, 'r') as fh:
            data = json.load(fh)

        frames = data.get('frames', [])
        if not frames:
            raise ValueError("transforms_json contains no frames")

        missing_depth = [i for i, fr in enumerate(frames) if 'depth_path' not in fr]
        missing_mask  = [i for i, fr in enumerate(frames) if 'mask_path'  not in fr]
        if missing_depth or missing_mask:
            raise ValueError(
                f"transforms_extended.json must contain 'depth_path' and 'mask_path' per "
                f"frame. Missing depth on frames {missing_depth[:5]}, mask on "
                f"{missing_mask[:5]}. Run images_generator.py first."
            )

        # Intrinsics
        if 'fl_x' in data:
            focal = float(data['fl_x'])
        elif 'camera_angle_x' in data:
            w_px  = int(data['w'])
            focal = 0.5 * w_px / np.tan(0.5 * float(data['camera_angle_x']))
        else:
            raise ValueError("JSON must contain 'fl_x' or 'camera_angle_x'")

        if 'fl_y' in data:
            focal_y = float(data['fl_y'])
        elif 'camera_angle_y' in data:
            h_px    = int(data['h'])
            focal_y = 0.5 * h_px / np.tan(0.5 * float(data['camera_angle_y']))
        else:
            focal_y = focal  # fallback: assume square pixels

        self._H, self._W, self._focal, self._focal_y = int(data['h']), int(data['w']), focal, focal_y

        from PIL import Image as PILImage
        imgs, deps, msks, poses = [], [], [], []
        skipped = 0
        for i, fr in enumerate(frames):
            img_p  = _resolve_path(fr.get('file_path'), json_dir)
            dep_p  = _resolve_path(fr.get('depth_path'), json_dir)
            msk_p  = _resolve_path(fr.get('mask_path'),  json_dir)
            if None in (img_p, dep_p, msk_p):
                print(f"  [NerfDataset] Frame {i}: missing file(s), skipped")
                skipped += 1
                continue

            if img_p.lower().endswith('.exr'):
                img_np = _load_exr_rgb_np(img_p, hdr_mode=self._hdr_mode)
                if img_np.shape[:2] != (self._H, self._W):
                    if self._hdr_mode:
                        # Resize per-channel in float32 mode to preserve HDR range
                        chans = [np.array(
                            PILImage.fromarray(img_np[:, :, c], mode='F').resize(
                                (self._W, self._H), PILImage.LANCZOS),
                            dtype=np.float32) for c in range(3)]
                        img_np = np.stack(chans, axis=-1)
                    else:
                        img_np = np.array(
                            PILImage.fromarray((img_np * 255).astype(np.uint8)).resize(
                                (self._W, self._H), PILImage.LANCZOS),
                            dtype=np.float32) / 255.0
            else:
                im = PILImage.open(img_p).convert('RGB')
                if im.width != self._W or im.height != self._H:
                    im = im.resize((self._W, self._H), PILImage.LANCZOS)
                img_np = np.array(im, dtype=np.float32) / 255.0

            dep_np = _load_exr_depth_np(dep_p)
            if dep_np.shape == (self._W, self._H):
                dep_np = dep_np.T
            if dep_np.shape != (self._H, self._W):
                print(f"  [NerfDataset] Frame {i}: depth shape {dep_np.shape} mismatch, skipped")
                skipped += 1
                continue
            # Clamp OptiX miss sentinel (1e20f) and any non-finite/non-positive value to 0
            dep_np = np.where(np.isfinite(dep_np) & (dep_np > 0) & (dep_np < 1e10), dep_np, 0.0).astype(np.float32)

            if msk_p.lower().endswith('.exr'):
                msk_np = _load_exr_mask_np(msk_p)
            else:
                msk_img = PILImage.open(msk_p).convert('L')
                if msk_img.width != self._W or msk_img.height != self._H:
                    msk_img = msk_img.resize((self._W, self._H), PILImage.NEAREST)
                msk_np = np.array(msk_img, dtype=np.float32) / 255.0
            if msk_np.shape == (self._W, self._H):
                msk_np = msk_np.T
            msk_np = (msk_np > 0.5).astype(np.float32)

            imgs.append(img_np)
            deps.append(dep_np)
            msks.append(msk_np)
            poses.append(np.array(fr['transform_matrix'], dtype=np.float32))

        n = len(imgs)
        if n < 2:
            raise RuntimeError("Need at least 2 valid frames (image + depth + mask)")

        print(f"[NerfDataset] {n} frames loaded ({skipped} skipped) "
              f"[{self._H}×{self._W}] focal={focal:.2f}")

        self._test_idx = test_idx if test_idx is not None else min(n - 1, 5)
        focal_t   = torch.tensor(focal,   dtype=torch.float32)
        focal_y_t = torch.tensor(focal_y, dtype=torch.float32)

        # Pre-compute ray bundles (CPU) and flatten to (N*H*W, *)
        all_o, all_d = [], []
        for i in range(n):
            c2w = torch.from_numpy(poses[i]).float()
            o, d = get_ray_bundle(self._H, self._W, focal_t, c2w, focal_y=focal_y_t)  # (H, W, 3)
            all_o.append(o.reshape(-1, 3))
            all_d.append(d.reshape(-1, 3))

        images_t  = torch.from_numpy(np.stack(imgs).reshape(n, -1, 3)).float()   # (N, H*W, 3)
        depths_t  = torch.from_numpy(np.stack(deps).reshape(n, -1)).float()       # (N, H*W)
        masks_t   = torch.from_numpy(np.stack(msks).reshape(n, -1)).float()       # (N, H*W)
        origins_t = torch.stack(all_o)                                            # (N, H*W, 3)
        dirs_t    = torch.stack(all_d)                                            # (N, H*W, 3)

        self._images  = images_t.reshape(-1, 3)     # (N*H*W, 3)
        self._depths  = depths_t.reshape(-1)         # (N*H*W,)
        self._masks   = masks_t.reshape(-1).bool()   # (N*H*W,)
        self._origins = origins_t.reshape(-1, 3)     # (N*H*W, 3)
        self._dirs    = dirs_t.reshape(-1, 3)        # (N*H*W, 3)

        # Exclude test frame from training index pools
        pix_per_frame = self._H * self._W
        test_start = self._test_idx * pix_per_frame
        test_end   = test_start + pix_per_frame
        all_idx    = torch.arange(n * pix_per_frame)
        non_test   = (all_idx < test_start) | (all_idx >= test_end)

        self._fg_indices = torch.where(self._masks & non_test)[0]
        self._bg_indices = torch.where((~self._masks) & non_test)[0]
        print(f"[NerfDataset] FG rays: {len(self._fg_indices):,} | "
              f"BG rays: {len(self._bg_indices):,}")
        if len(self._fg_indices) == 0:
            raise RuntimeError(
                "[NerfDataset] No foreground rays found — the mask is empty or all-False.\n"
                "Check that Step 1 (Depth_Generator) ran correctly and that the mask EXR "
                "was saved without normalization. Pixel values should be 0.0 or 1.0."
            )

        # Per-frame cache for test/full-image render
        self._imgs_pf  = [torch.from_numpy(imgs[i]).float()  for i in range(n)]
        self._deps_pf  = [torch.from_numpy(deps[i]).float()  for i in range(n)]
        self._msks_pf  = [torch.from_numpy(msks[i]).float()  for i in range(n)]
        self._poses_pf = [torch.from_numpy(poses[i]).float() for i in range(n)]

    # ── properties ──────────────────────────────────────────────────────────

    @property
    def H(self):          return self._H
    @property
    def W(self):          return self._W
    @property
    def focal(self):      return self._focal
    @property
    def focal_y(self):    return self._focal_y
    @property
    def device(self):     return self._device
    @property
    def num_frames(self): return len(self._imgs_pf)
    @property
    def test_idx(self):   return self._test_idx

    # ── data access ─────────────────────────────────────────────────────────

    def get_test_frame(self):
        i = self._test_idx
        return (self._imgs_pf[i].to(self._device),
                self._poses_pf[i].to(self._device),
                self._deps_pf[i].to(self._device),
                self._msks_pf[i].to(self._device))

    def get_frame(self, idx: int):
        return (self._imgs_pf[idx].to(self._device),
                self._poses_pf[idx].to(self._device),
                self._deps_pf[idx].to(self._device),
                self._msks_pf[idx].to(self._device))

    # ── sampling ────────────────────────────────────────────────────────────

    def sample_rays(self, batch_size: int, mask_bias: float = 0.9):
        """Random ray batch with foreground bias.

        Returns (origins, dirs, rgb, depth, is_fg) tensors on self.device.
        depth[i] is valid euclidean depth where is_fg[i] is True (0 elsewhere).
        """
        n_fg = min(int(batch_size * mask_bias), len(self._fg_indices))
        n_bg = batch_size - n_fg

        fg_pick = torch.randint(0, len(self._fg_indices), (n_fg,))
        fg_idx  = self._fg_indices[fg_pick]

        if n_bg > 0 and len(self._bg_indices) > 0:
            bg_pick = torch.randint(0, len(self._bg_indices), (n_bg,))
            bg_idx  = self._bg_indices[bg_pick]
            idx     = torch.cat([fg_idx, bg_idx])
            is_fg   = torch.cat([torch.ones(n_fg, dtype=torch.bool),
                                  torch.zeros(n_bg, dtype=torch.bool)])
        else:
            idx   = fg_idx
            is_fg = torch.ones(n_fg, dtype=torch.bool)

        return (self._origins[idx].to(self._device),
                self._dirs[idx].to(self._device),
                self._images[idx].to(self._device),
                self._depths[idx].to(self._device),
                is_fg.to(self._device))


# ──────────────────────────────────────────────────────────────────────────────
# Monitoring helpers
# ──────────────────────────────────────────────────────────────────────────────

def _save_preview(gt_rgb: torch.Tensor, pred_rgb: torch.Tensor,
                   gt_depth: torch.Tensor, pred_depth: torch.Tensor,
                   output_dir: str, iter_num: int):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def _tonemap(t: torch.Tensor) -> np.ndarray:
        x = t.detach().cpu()
        x = x / (1.0 + x)   # Reinhard — works for both linear HDR and SDR [0,1]
        return x.numpy().clip(0, 1)
    gt_rgb_np   = _tonemap(gt_rgb)
    pred_rgb_np = _tonemap(pred_rgb)
    gt_d_np     = gt_depth.detach().cpu().numpy()
    pred_d_np   = pred_depth.detach().cpu().numpy()

    # Shared depth range from valid GT pixels (excludes background zeros)
    gt_valid = gt_d_np[gt_d_np > 0]
    if gt_valid.size > 0:
        vmin = float(np.percentile(gt_valid, 2))
        vmax = float(np.percentile(gt_valid, 98))
    else:
        vmin, vmax = 0.0, 1.0

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].imshow(gt_rgb_np);                        axes[0, 0].set_title("GT RGB")
    axes[0, 1].imshow(pred_rgb_np);                      axes[0, 1].set_title("Pred RGB")
    im = axes[1, 0].imshow(gt_d_np,   cmap='turbo', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("GT depth")
    axes[1, 1].imshow(pred_d_np, cmap='turbo', vmin=vmin, vmax=vmax)
    axes[1, 1].set_title("Pred depth")
    fig.colorbar(im, ax=axes[1, :], orientation='horizontal', fraction=0.05, pad=0.04,
                 label='depth')
    for ax in axes.flat:
        ax.axis('off')
    fig.suptitle(f"iter {iter_num:06d}")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"preview_iter_{iter_num:06d}.png"), dpi=80)
    plt.close(fig)


def _save_test_comparison(gt: torch.Tensor, pred: torch.Tensor,
                           psnr: float, frame_idx: int, output_dir: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def _tm(t): x = t.detach().cpu(); return (x / (1.0 + x)).numpy().clip(0, 1)
    gt_np   = _tm(gt)
    pred_np = _tm(pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(gt_np);   axes[0].set_title("GT");                    axes[0].axis('off')
    axes[1].imshow(pred_np); axes[1].set_title(f"Pred  PSNR={psnr:.2f}dB"); axes[1].axis('off')
    fig.suptitle(f"Frame {frame_idx}")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"frame_{frame_idx:03d}_psnr{psnr:.2f}.png"), dpi=80)
    plt.close(fig)


def _save_loss_curve(iternums, losses, psnrs, path: str, psnrs_fg=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(iternums, losses); ax1.set_title("Loss");      ax1.set_xlabel("Iter"); ax1.set_yscale('log')
    ax2.plot(iternums, psnrs, label="full");  ax2.set_title("PSNR (dB)"); ax2.set_xlabel("Iter")
    if psnrs_fg:
        ax2.plot(iternums, psnrs_fg, label="FG", linestyle='--')
        ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"[NeRF] Loss curve → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def _make_model(cfg: NerfConfig, device):
    """Instantiate the right model class from a NerfConfig."""
    use_full = getattr(cfg, 'num_layers', 3) > 3 or getattr(cfg, 'use_viewdirs', False)
    if use_full:
        return NeRFModel(
            D=getattr(cfg, 'num_layers', 8),
            W=cfg.filter_size,
            skips=getattr(cfg, 'skips', (4,)),
            num_encoding_functions=cfg.num_encoding_functions,
            use_viewdirs=getattr(cfg, 'use_viewdirs', True),
            num_encoding_functions_views=getattr(cfg, 'num_encoding_functions_views', 4),
        ).to(device)
    return TinyNerfModel(cfg.filter_size, cfg.num_encoding_functions).to(device)


def train(
    dataset: NerfDataset,
    cfg: NerfConfig,
    num_iters:     int      = 10000,
    batch_size:    int      = 4096,
    mask_bias:     float    = 0.9,
    lr:            float    = 5e-4,
    ckpt_path:     str      = None,
    display_every: int      = 100,
    output_dir:    str      = None,
    on_step:       Callable = None,
    seed:          int      = 9458,
):
    """Train NeRF on dataset.

    on_step(iter_num, loss, psnrs_list, losses_list, model) — called every display_every iters.
    Returns trained model.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = dataset.device
    model  = _make_model(cfg, device)
    optim  = torch.optim.Adam(model.parameters(), lr=lr)

    start_iter   = 0
    decay_steps  = getattr(cfg, 'lrate_decay', 250) * 1000

    if ckpt_path and os.path.exists(ckpt_path):
        loaded, opt_state, start_iter, _ = load_checkpoint(ckpt_path, device)
        model.load_state_dict(loaded.state_dict())
        if opt_state:
            optim.load_state_dict(opt_state)
        print(f"[NeRF] Resumed from iter {start_iter}")

    if start_iter >= num_iters:
        print("[NeRF] Training already complete per checkpoint")
        return model

    preview_dir = test_dir = None
    if output_dir:
        preview_dir = os.path.join(output_dir, 'nerf_train')
        test_dir    = os.path.join(output_dir, 'nerf_train', 'test')
        os.makedirs(preview_dir, exist_ok=True)
        os.makedirs(test_dir,    exist_ok=True)

    losses, psnrs, psnrs_fg, iternums = [], [], [], []
    testimg, testpose, testdepth, testmask = dataset.get_test_frame()

    model_name = type(model).__name__
    print(f"[NeRF] {model_name} | iters {start_iter}→{num_iters} | "
          f"batch={batch_size} | fg_bias={mask_bias:.0%} | "
          f"lr={lr} | decay_steps={decay_steps}")

    for i in range(start_iter, num_iters):
        model.train()
        origins, dirs, rgb_tgt, depths, is_fg = dataset.sample_rays(batch_size, mask_bias)

        n_fg = is_fg.sum().item()
        n_bg = (~is_fg).sum().item()

        # Collect sub-predictions then assemble (autograd-safe)
        parts_pred, parts_tgt = [], []

        if n_fg > 0:
            o_fg, d_fg, t_fg = origins[is_fg], dirs[is_fg], depths[is_fg]
            rgb_fg, _, _ = run_one_iter(o_fg, d_fg, model, cfg,
                                         target_depth=t_fg, randomize=True)
            parts_pred.append(rgb_fg)
            parts_tgt.append(rgb_tgt[is_fg])

        if n_bg > 0:
            o_bg, d_bg = origins[~is_fg], dirs[~is_fg]
            rgb_bg, _, _ = run_one_iter(o_bg, d_bg, model, cfg,
                                         target_depth=None, randomize=True)
            parts_pred.append(rgb_bg)
            parts_tgt.append(rgb_tgt[~is_fg])

        all_pred = torch.cat(parts_pred)
        all_tgt  = torch.cat(parts_tgt)
        if getattr(cfg, 'hdr_mode', False):
            # Log-space MSE: uniform gradient across HDR dynamic range (RawNeRF-style).
            # With exp activation, log(pred)=z, so loss ≈ (z_pred − log(tgt))² — no saturation.
            _eps = 1e-3
            loss = ((torch.log(all_pred.clamp_min(0) + _eps)
                     - torch.log(all_tgt.clamp_min(0) + _eps)) ** 2).mean()
        else:
            loss = F.mse_loss(all_pred, all_tgt)

        loss.backward()
        new_lr = lr * (0.1 ** (i / decay_steps))
        for g in optim.param_groups:
            g['lr'] = new_lr
        optim.step()
        optim.zero_grad()

        if i % display_every == 0:
            model.eval()
            rgb_eval, dep_eval, acc_eval = render_image(dataset.H, dataset.W, dataset.focal,
                                                         testpose, model, cfg,
                                                         target_depth=testdepth, randomize=False,
                                                         focal_y=dataset.focal_y)
            hdr = getattr(cfg, 'hdr_mode', False)
            _eps = 1e-3
            if hdr:
                eval_loss = F.mse_loss(
                    torch.log(rgb_eval.clamp_min(0) + _eps),
                    torch.log(testimg .clamp_min(0) + _eps))
            else:
                eval_loss = F.mse_loss(rgb_eval, testimg)
            psnr = -10.0 * torch.log10(eval_loss).item()

            # FG-only PSNR — measures object quality regardless of BG
            fg_mask = testmask.bool()
            if fg_mask.any():
                if hdr:
                    fg_loss = F.mse_loss(
                        torch.log(rgb_eval[fg_mask].clamp_min(0) + _eps),
                        torch.log(testimg [fg_mask].clamp_min(0) + _eps))
                else:
                    fg_loss = F.mse_loss(rgb_eval[fg_mask], testimg[fg_mask])
                psnr_fg = -10.0 * torch.log10(fg_loss).item()
            else:
                psnr_fg = float('nan')

            losses.append(loss.item())
            psnrs.append(psnr)
            psnrs_fg.append(psnr_fg)
            iternums.append(i)

            # Diagnostic: detect sigma/rgb collapse early
            bg_mask = ~fg_mask
            print(f"  [{i:5d}] loss={loss.item():.5f}  "
                  f"PSNR fg={psnr_fg:.2f}dB | full={psnr:.2f}dB  "
                  f"acc[fg={acc_eval[fg_mask].mean():.2f} bg={acc_eval[bg_mask].mean():.2f}]  "
                  f"rgb[fg={rgb_eval[fg_mask].mean():.3f} bg={rgb_eval[bg_mask].mean():.3f}]  "
                  f"tgt[fg={testimg[fg_mask].mean():.3f} bg={testimg[bg_mask].mean():.3f}]")

            if preview_dir:
                _save_preview(testimg, rgb_eval, testdepth, dep_eval, preview_dir, i)

            if on_step:
                on_step(i, loss.item(), psnrs, losses, model)

    if ckpt_path:
        save_checkpoint(ckpt_path, model, optim, num_iters, cfg, seed)

    if output_dir and iternums:
        _save_loss_curve(iternums, losses, psnrs,
                          os.path.join(output_dir, 'nerf_train', 'loss_curve.png'),
                          psnrs_fg=psnrs_fg)

    # Final test loop
    if test_dir:
        print("[NeRF] Test loop...")
        model.eval()
        all_psnr_full, all_psnr_fg = [], []
        hdr = getattr(cfg, 'hdr_mode', False)
        for fi in range(dataset.num_frames):
            gt, pose, dep, msk = dataset.get_frame(fi)
            pred, _, _ = render_image(dataset.H, dataset.W, dataset.focal,
                                       pose, model, cfg,
                                       target_depth=dep, randomize=False,
                                       focal_y=dataset.focal_y)
            _eps = 1e-3
            if hdr:
                psnr_full = -10.0 * torch.log10(F.mse_loss(
                    torch.log(pred.clamp_min(0) + _eps),
                    torch.log(gt  .clamp_min(0) + _eps))).item()
            else:
                psnr_full = -10.0 * torch.log10(F.mse_loss(pred, gt)).item()
            fg_m = msk.bool()
            if fg_m.any():
                if hdr:
                    psnr_fg_v = -10.0 * torch.log10(F.mse_loss(
                        torch.log(pred[fg_m].clamp_min(0) + _eps),
                        torch.log(gt  [fg_m].clamp_min(0) + _eps))).item()
                else:
                    psnr_fg_v = -10.0 * torch.log10(F.mse_loss(pred[fg_m], gt[fg_m])).item()
            else:
                psnr_fg_v = float('nan')
            all_psnr_full.append(psnr_full)
            all_psnr_fg.append(psnr_fg_v)
            _save_test_comparison(gt, pred, psnr_fg_v, fi, test_dir)
        valid_fg = [v for v in all_psnr_fg if not np.isnan(v)]
        print(f"[NeRF] Test PSNR (full)  mean={np.mean(all_psnr_full):.2f}  "
              f"min={min(all_psnr_full):.2f}  max={max(all_psnr_full):.2f} dB")
        if valid_fg:
            print(f"[NeRF] Test PSNR (FG)    mean={np.mean(valid_fg):.2f}  "
                  f"min={min(valid_fg):.2f}  max={max(valid_fg):.2f} dB")

    return model
