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
    num_encoding_functions: int = 6
    filter_size: int = 128
    near: float = 2.0
    far: float = 6.0
    depth_window: float = 0.15
    depth_samples_per_ray: int = 8
    chunk_size: int = 16384


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

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


VeryTinyNerfModel = TinyNerfModel  # backward-compatible alias


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

def render_volume_density(radiance_field, ray_origins, depth_values):
    sigma   = F.relu(radiance_field[..., 3])
    rgb     = torch.sigmoid(radiance_field[..., :3])
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

def _encode_and_run(query_points: torch.Tensor, model: TinyNerfModel, cfg: NerfConfig):
    """query_points: (*, n_samp, 3) → radiance_field: (*, n_samp, 4)"""
    flat_pts  = query_points.reshape(-1, 3)
    encoded   = positional_encoding(flat_pts, cfg.num_encoding_functions)
    preds = []
    for i in range(0, encoded.shape[0], cfg.chunk_size):
        preds.append(model(encoded[i:i + cfg.chunk_size]))
    return torch.cat(preds, dim=0).reshape(*query_points.shape[:-1], 4)


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
    Returns (rgb, depth_map, acc_map) each shape (*,) / (*,3).
    """
    if target_depth is not None:
        ray_norm    = torch.linalg.norm(dirs, dim=-1).clamp_min(1e-8)   # (*,)
        depth_t     = target_depth / ray_norm                             # (*,)
        offsets     = torch.linspace(-cfg.depth_window, cfg.depth_window,
                                      cfg.depth_samples_per_ray,
                                      device=origins.device, dtype=origins.dtype)
        # broadcast offsets over all batch dims
        offsets_br  = offsets.view(*([1] * depth_t.dim()), cfg.depth_samples_per_ray)
        depth_values = depth_t.unsqueeze(-1) + offsets_br / ray_norm.unsqueeze(-1)
        if randomize and cfg.depth_samples_per_ray > 1:
            mids  = 0.5 * (depth_values[..., 1:] + depth_values[..., :-1])
            upper = torch.cat([mids, depth_values[..., -1:]], dim=-1)
            lower = torch.cat([depth_values[..., :1], mids],  dim=-1)
            depth_values = lower + (upper - lower) * torch.rand_like(depth_values)
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
    rf = _encode_and_run(query_points, model, cfg)
    return render_volume_density(rf, origins, depth_values)


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

def save_checkpoint(path: str, model: TinyNerfModel, optimizer,
                    iter_done: int, cfg: NerfConfig, seed: int = 9458):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        'iter_done':            int(iter_done),
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'seed':                 int(seed),
        'num_encoding_functions': int(cfg.num_encoding_functions),
        'filter_size':          int(cfg.filter_size),
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
    num_enc     = payload.get('num_encoding_functions', 6)
    filter_size = payload.get('filter_size', 128)
    cfg   = NerfConfig(num_encoding_functions=num_enc, filter_size=filter_size)
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


def _load_exr_rgb_np(path: str) -> np.ndarray:
    """Load an EXR file as (H, W, 3) float32 in [0, 1] range.  Tone-maps HDR values
    via simple normalization so the result is usable as an RGB training target."""
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
    # Clamp to [0, 1]; HDR images with values > 1 are common but NeRF expects LDR targets
    return np.clip(rgb, 0.0, 1.0)


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

    def __init__(self, transforms_json_path: str, device=None, test_idx: int = None):
        if device is None:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = torch.device(device)

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
                img_np = _load_exr_rgb_np(img_p)
                if img_np.shape[:2] != (self._H, self._W):
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

        # Per-frame cache for test/full-image render
        self._imgs_pf  = [torch.from_numpy(imgs[i]).float()  for i in range(n)]
        self._deps_pf  = [torch.from_numpy(deps[i]).float()  for i in range(n)]
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
                self._deps_pf[i].to(self._device))

    def get_frame(self, idx: int):
        return (self._imgs_pf[idx].to(self._device),
                self._poses_pf[idx].to(self._device),
                self._deps_pf[idx].to(self._device))

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
    gt_rgb_np   = gt_rgb.detach().cpu().numpy().clip(0, 1)
    pred_rgb_np = pred_rgb.detach().cpu().numpy().clip(0, 1)
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
    gt_np   = gt.detach().cpu().numpy().clip(0, 1)
    pred_np = pred.detach().cpu().numpy().clip(0, 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(gt_np);   axes[0].set_title("GT");                    axes[0].axis('off')
    axes[1].imshow(pred_np); axes[1].set_title(f"Pred  PSNR={psnr:.2f}dB"); axes[1].axis('off')
    fig.suptitle(f"Frame {frame_idx}")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"frame_{frame_idx:03d}_psnr{psnr:.2f}.png"), dpi=80)
    plt.close(fig)


def _save_loss_curve(iternums, losses, psnrs, path: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(iternums, losses); ax1.set_title("Loss");      ax1.set_xlabel("Iter"); ax1.set_yscale('log')
    ax2.plot(iternums, psnrs);  ax2.set_title("PSNR (dB)"); ax2.set_xlabel("Iter")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"[NeRF] Loss curve → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(
    dataset: NerfDataset,
    cfg: NerfConfig,
    num_iters:     int      = 10000,
    batch_size:    int      = 4096,
    mask_bias:     float    = 0.9,
    lr:            float    = 5e-3,
    ckpt_path:     str      = None,
    display_every: int      = 100,
    output_dir:    str      = None,
    on_step:       Callable = None,
    seed:          int      = 9458,
) -> TinyNerfModel:
    """Train TinyNeRF on dataset.

    on_step(iter_num, loss, psnrs_list, losses_list, model) — called every display_every iters.
    Returns trained model.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = dataset.device
    model  = TinyNerfModel(cfg.filter_size, cfg.num_encoding_functions).to(device)
    optim  = torch.optim.Adam(model.parameters(), lr=lr)

    start_iter = 0
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

    losses, psnrs, iternums = [], [], []
    testimg, testpose, testdepth = dataset.get_test_frame()

    print(f"[NeRF] Training iters {start_iter}→{num_iters}, "
          f"batch={batch_size}, fg_bias={mask_bias:.0%}, lr={lr}")

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

        loss = F.mse_loss(torch.cat(parts_pred), torch.cat(parts_tgt))
        loss.backward()
        optim.step()
        optim.zero_grad()

        if i % display_every == 0:
            model.eval()
            rgb_eval, dep_eval, _ = render_image(dataset.H, dataset.W, dataset.focal,
                                                   testpose, model, cfg,
                                                   target_depth=testdepth, randomize=False,
                                                   focal_y=dataset.focal_y)
            eval_loss = F.mse_loss(rgb_eval, testimg)
            psnr      = -10.0 * torch.log10(eval_loss).item()
            losses.append(loss.item())
            psnrs.append(psnr)
            iternums.append(i)

            if preview_dir:
                _save_preview(testimg, rgb_eval, testdepth, dep_eval, preview_dir, i)

            if on_step:
                on_step(i, loss.item(), psnrs, losses, model)
            else:
                print(f"  [{i:5d}] loss={loss.item():.5f}  PSNR={psnr:.2f}dB")

    if ckpt_path:
        save_checkpoint(ckpt_path, model, optim, num_iters, cfg, seed)

    if output_dir and iternums:
        _save_loss_curve(iternums, losses, psnrs,
                          os.path.join(output_dir, 'nerf_train', 'loss_curve.png'))

    # Final test loop
    if test_dir:
        print("[NeRF] Test loop...")
        model.eval()
        all_psnr = []
        for fi in range(dataset.num_frames):
            gt, pose, dep = dataset.get_frame(fi)
            pred, _, _ = render_image(dataset.H, dataset.W, dataset.focal,
                                       pose, model, cfg,
                                       target_depth=dep, randomize=False,
                                       focal_y=dataset.focal_y)
            psnr_v = -10.0 * torch.log10(F.mse_loss(pred, gt)).item()
            all_psnr.append(psnr_v)
            _save_test_comparison(gt, pred, psnr_v, fi, test_dir)
        print(f"[NeRF] Test PSNR  mean={np.mean(all_psnr):.2f}  "
              f"min={min(all_psnr):.2f}  max={max(all_psnr):.2f} dB")

    return model
