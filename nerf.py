#!/usr/bin/env python3
"""
NeRF (Mildenhall et al., ECCV 2020) — minimal PyTorch implementation in ONE file.
Features:
- Positional encoding for xyz and view directions
- NeRF MLP (coarse + fine) with skip connection at layer 4
- Stratified sampling + hierarchical importance sampling (PDF)
- Volume rendering (accumulated color, depth, weights)
- Blender synthetic dataset loader (as in the original release)
- Training loop with PSNR, checkpointing, and image rendering

Usage examples:
    python nerf_onefile.py --data_dir ./lego --exp_name lego_nerf --N_iters 200000 \
        --N_rand 4096 --lrate 5e-4 --render_only 0

    # Render a spiral path after training (uses saved ckpt by default)
    python nerf_onefile.py --data_dir ./lego --render_only 1 --ckpt_path ./ckpts/lego_nerf/latest.pt

Expected Blender dataset layout (as per the NeRF authors):
    data_dir/
      ├── transforms_train.json
      ├── transforms_val.json
      ├── transforms_test.json
      ├── r_########.png (or .jpg)
      └── ...

This script aims to be clear and compact rather than hyper-optimized.
"""

import os
import json
import math
import time
import argparse
from dataclasses import dataclass

import numpy as np
import imageio.v2 as imageio
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Utilities
# -----------------------------

def mse2psnr(mse: torch.Tensor) -> torch.Tensor:
    return -10.0 * torch.log10(mse)


def to8b(x):
    x = np.clip(x, 0.0, 1.0)
    return (255 * x).astype(np.uint8)


# -----------------------------
# Positional Encoding
# -----------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs: int, include_input: bool = True, log_sampling: bool = True):
        super().__init__()
        self.include_input = include_input
        if log_sampling:
            freq_bands = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        else:
            freq_bands = torch.linspace(2.0 ** 0.0, 2.0 ** (num_freqs - 1), num_freqs)
        self.register_buffer('freq_bands', freq_bands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., C]
        out = [x] if self.include_input else []
        for f in self.freq_bands:
            out.append(torch.sin(f * math.pi * x))
            out.append(torch.cos(f * math.pi * x))
        return torch.cat(out, dim=-1)

    def out_dim(self, in_dim: int) -> int:
        return (in_dim if self.include_input else 0) + 2 * in_dim * self.freq_bands.shape[0]


# -----------------------------
# NeRF MLP
# -----------------------------

class NeRF(nn.Module):
    def __init__(self, D=8, W=256, in_ch_xyz=63, in_ch_dir=27, skips=(4,)):
        super().__init__()
        self.D = D
        self.W = W
        self.skips = set(skips)

        # xyz stream
        self.pts_linears = nn.ModuleList()
        self.pts_linears.append(nn.Linear(in_ch_xyz, W))
        for i in range(1, D):
            if i in self.skips:
                self.pts_linears.append(nn.Linear(W + in_ch_xyz, W))
            else:
                self.pts_linears.append(nn.Linear(W, W))

        # feature -> sigma/rgb branches
        self.feature_linear = nn.Linear(W, W)
        self.sigma_linear = nn.Linear(W, 1)

        # view dir stream
        self.views_linear = nn.Linear(W + in_ch_dir, W // 2)
        self.rgb_linear = nn.Linear(W // 2, 3)

    def forward(self, x_enc, d_enc):
        h = x_enc
        for i, l in enumerate(self.pts_linears):
            if i in self.skips:
                h = torch.cat([x_enc, h], dim=-1)
            h = F.relu(l(h))
        sigma = F.relu(self.sigma_linear(h))
        feat = self.feature_linear(h)
        h = torch.cat([feat, d_enc], dim=-1)
        h = F.relu(self.views_linear(h))
        rgb = torch.sigmoid(self.rgb_linear(h))
        return torch.cat([rgb, sigma], dim=-1)


# -----------------------------
# Ray helpers & rendering
# -----------------------------

def get_rays(H, W, focal, c2w):
    """Compute rays for all pixels in the image given camera-to-world matrix.
    Returns rays_o, rays_d with shape [H, W, 3].
    """
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=c2w.device),
        torch.arange(H, dtype=torch.float32, device=c2w.device),
        indexing='xy'
    )
    dirs = torch.stack([(i - W * 0.5) / focal,
                        -(j - H * 0.5) / focal,
                        -torch.ones_like(i)], dim=-1)  # camera space
    # Rotate and translate
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o, rays_d


def sample_stratified(N_samples, near, far, rays_o, rays_d, perturb):
    # z_vals: [N_rays, N_samples]
    t_vals = torch.linspace(0.0, 1.0, steps=N_samples, device=rays_o.device)
    z_vals = near * (1.0 - t_vals) + far * t_vals
    z_vals = z_vals.expand(rays_o.shape[0], N_samples)
    if perturb:
        mids = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])
        upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
        lower = torch.cat([z_vals[:, :1], mids], dim=-1)
        t_rand = torch.rand(z_vals.shape, device=rays_o.device)
        z_vals = lower + (upper - lower) * t_rand
    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., None]
    return pts, z_vals


def raw2outputs(raw, z_vals, rays_d, white_bkgd=False):
    """Volume rendering integration.
    raw: [N_rays, N_samples, 4]; z_vals: [N_rays, N_samples]
    returns rgb_map, depth_map, acc_map, weights
    """
    dists = z_vals[:, 1:] - z_vals[:, :-1]
    dists = torch.cat([dists, 1e10 * torch.ones_like(dists[:, :1])], dim=-1)

    rgb = raw[..., :3]  # [N_rays, N_samples, 3]
    sigma = raw[..., 3]  # [N_rays, N_samples]

    # Cosine between ray and normal is ignored here (as in original code)
    alpha = 1.0 - torch.exp(-sigma * dists)

    T = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
    weights = alpha * T  # [N_rays, N_samples]

    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    depth_map = torch.sum(weights * z_vals, dim=-1)
    acc_map = torch.sum(weights, dim=-1)

    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, depth_map, acc_map, weights


def sample_pdf(bins, weights, N_samples, det=False):
    """Hierarchical sampling from piecewise-constant PDF.
    bins: [N_rays, N_bins+1], weights: [N_rays, N_bins]
    returns z_samples: [N_rays, N_samples]
    """
    EPS = 1e-5
    weights = weights + EPS
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)  # [N_rays, N_bins+1]

    if det:
        u = torch.linspace(0.0, 1.0, steps=N_samples, device=bins.device)
        u = u.expand(cdf.shape[0], N_samples)
    else:
        u = torch.rand(cdf.shape[0], N_samples, device=bins.device)

    # Invert CDF
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], dim=-1)  # [N_rays, N_samples, 2]

    cdf_g = torch.gather(cdf.unsqueeze(1).expand(-1, N_samples, -1), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(-1, N_samples, -1), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0]).clamp(min=EPS)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    return samples


@dataclass
class RenderConfig:
    N_samples: int = 64
    N_importance: int = 128
    perturb: bool = True
    white_bkgd: bool = True
    chunk: int = 16384  # rays per chunk for memory control
    near: float = 2.0
    far: float = 6.0


def run_network(pts, dirs, network_fn, pe_xyz, pe_dir):
    # pts: [N_rays, N_samples, 3]; dirs: [N_rays, 3]
    N_rays, N_samples, _ = pts.shape
    pts_flat = pts.reshape(-1, 3)
    dirs_expanded = dirs[:, None, :].expand_as(pts)
    dirs_flat = dirs_expanded.reshape(-1, 3)

    x_enc = pe_xyz(pts_flat)
    d_enc = pe_dir(dirs_flat)

    out = network_fn(x_enc, d_enc)
    out = out.reshape(N_rays, N_samples, 4)
    return out


def render_rays(ray_batch, network_coarse, network_fine, pe_xyz, pe_dir, cfg: RenderConfig):
    rays_o = ray_batch[:, :3]
    rays_d = ray_batch[:, 3:6]
    near = cfg.near * torch.ones_like(rays_o[:, :1])
    far = cfg.far * torch.ones_like(rays_o[:, :1])

    # 1) Stratified sampling (coarse)
    pts, z_vals = sample_stratified(cfg.N_samples, near, far, rays_o, rays_d, cfg.perturb)
    raw = run_network(pts, rays_d, network_coarse, pe_xyz, pe_dir)
    rgb_map_c, depth_map_c, acc_map_c, weights = raw2outputs(raw, z_vals, rays_d, cfg.white_bkgd)

    # 2) Importance sampling (fine)
    if cfg.N_importance > 0:
        z_mids = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])
        z_samples = sample_pdf(z_mids, weights[:, 1:-1], cfg.N_importance, det=not cfg.perturb)
        z_vals_f, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)
        pts_f = rays_o[:, None, :] + rays_d[:, None, :] * z_vals_f[..., None]
        raw_f = run_network(pts_f, rays_d, network_fine, pe_xyz, pe_dir)
        rgb_map_f, depth_map_f, acc_map_f, _ = raw2outputs(raw_f, z_vals_f, rays_d, cfg.white_bkgd)
    else:
        rgb_map_f, depth_map_f, acc_map_f = rgb_map_c, depth_map_c, acc_map_c

    return {
        'rgb_coarse': rgb_map_c,
        'rgb_fine': rgb_map_f,
        'depth_coarse': depth_map_c,
        'depth_fine': depth_map_f,
        'acc_coarse': acc_map_c,
        'acc_fine': acc_map_f,
    }


def render_image(H, W, focal, c2w, network_coarse, network_fine, pe_xyz, pe_dir, cfg: RenderConfig, device):
    with torch.no_grad():
        rays_o, rays_d = get_rays(H, W, focal, c2w)
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)

        all_rgb = []
        for i in range(0, rays_o.shape[0], cfg.chunk):
            chunk_o = rays_o[i:i+cfg.chunk].to(device)
            chunk_d = rays_d[i:i+cfg.chunk].to(device)
            batch = torch.cat([chunk_o, chunk_d], dim=-1)
            out = render_rays(batch, network_coarse, network_fine, pe_xyz, pe_dir, cfg)
            all_rgb.append(out['rgb_fine'].cpu())
        img = torch.cat(all_rgb, dim=0).reshape(H, W, 3).numpy()
        return img


# -----------------------------
# Dataset (Blender synthetic)
# -----------------------------

class BlenderDataset(Dataset):
    def __init__(self, data_dir, split='train', half_res=True, hold_every=8):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.half_res = half_res

        with open(os.path.join(data_dir, f'transforms_{split}.json'), 'r') as f:
            meta = json.load(f)
        self.meta = meta

        self.fnames = []
        self.poses = []
        for frame in meta['frames']:
            fname = os.path.join(data_dir, frame['file_path'] + '.png')
            if not os.path.isfile(fname):
                fname = os.path.join(data_dir, frame['file_path'] + '.jpg')
            self.fnames.append(fname)
            self.poses.append(np.array(frame['transform_matrix'], dtype=np.float32))
        self.poses = np.stack(self.poses, axis=0)

        # Camera params
        H = meta['h'] if 'h' in meta else None
        W = meta['w'] if 'w' in meta else None
        camera_angle_x = meta.get('camera_angle_x', None)

        # Read one image to infer size if missing
        im = imageio.imread(self.fnames[0])
        if im.ndim == 2:
            im = np.repeat(im[..., None], 3, axis=-1)
        if im.shape[-1] == 4:
            im = im[..., :3]
        H0, W0 = im.shape[:2]

        if H is None or W is None:
            H, W = H0, W0
        if self.half_res:
            H //= 2
            W //= 2

        self.H = int(H)
        self.W = int(W)

        if camera_angle_x is not None:
            self.focal = 0.5 * self.W / np.tan(0.5 * camera_angle_x)
        else:
            # Fallback: estimate focal from metadata if provided
            self.focal = meta.get('focal', 0.5 * self.W / np.tan(np.deg2rad(0.5 * 50.0)))

        # Load all images (and resize if needed)
        imgs = []
        for fp in self.fnames:
            img = imageio.imread(fp)
            if img.ndim == 2:
                img = np.repeat(img[..., None], 3, axis=-1)
            if img.shape[-1] == 4:
                img = img[..., :3]
            if (img.shape[1] != self.W) or (img.shape[0] != self.H):
                img = np.array(Image.fromarray(img).resize((self.W, self.H), Image.LANCZOS))
            imgs.append((img / 255.0).astype(np.float32))
        self.imgs = np.stack(imgs, axis=0)  # [N, H, W, 3]

        # Rays/directions are generated on the fly per batch

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        pose = self.poses[idx]
        return img, pose


# -----------------------------
# Training
# -----------------------------

def train_one_epoch(loader, net_c, net_f, pe_xyz, pe_dir, cfg: RenderConfig, optimizer, device, N_rand=4096):
    net_c.train(); net_f.train()
    epoch_loss = 0.0
    for imgs, poses in loader:
        imgs = imgs.to(device)  # [B, H, W, 3]
        poses = poses.to(device)  # [B, 4, 4]
        B, H, W, _ = imgs.shape

        # Randomly sample rays across batch
        i = torch.randint(0, H, (N_rand,), device=device)
        j = torch.randint(0, W, (N_rand,), device=device)
        img_ids = torch.randint(0, B, (N_rand,), device=device)

        target_s = imgs[img_ids, i, j]  # [N_rand, 3]
        c2w = poses[img_ids]  # [N_rand, 4, 4]

        # Per-ray origin/direction
        focals = torch.full((N_rand,), loader.dataset.focal, device=device)
        # Build rays for picked pixels
        # Efficiently compute single-pixel ray for each sample
        x = (j.float() - loader.dataset.W * 0.5) / focals
        y = -(i.float() - loader.dataset.H * 0.5) / focals
        dirs_cam = torch.stack([x, y, -torch.ones_like(x)], dim=-1)  # [N_rand, 3]
        rays_d = torch.sum(dirs_cam[:, None, :] * c2w[:, :3, :3], dim=-1)
        rays_o = c2w[:, :3, 3]

        # Pack and render
        ray_batch = torch.cat([rays_o, rays_d], dim=-1)
        out = render_rays(ray_batch, net_c, net_f, pe_xyz, pe_dir, cfg)
        rgb = out['rgb_fine']

        loss = F.mse_loss(rgb, target_s)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / max(len(loader), 1)


@torch.no_grad()
def validate_once(loader, net_c, net_f, pe_xyz, pe_dir, cfg: RenderConfig, device, max_imgs=1):
    net_c.eval(); net_f.eval()
    psnrs = []
    for imgs, poses in loader:
        imgs = imgs.to(device)
        poses = poses.to(device)
        H, W = loader.dataset.H, loader.dataset.W
        focal = loader.dataset.focal
        for b in range(min(imgs.shape[0], max_imgs)):
            c2w = poses[b]
            img_pred = render_image(H, W, focal, c2w, net_c, net_f, pe_xyz, pe_dir, cfg, device)
            mse = np.mean((img_pred - imgs[b].cpu().numpy()) ** 2)
            psnrs.append(float(-10.0 * np.log10(mse + 1e-10)))
    return float(np.mean(psnrs)) if psnrs else 0.0


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description='NeRF one-file PyTorch')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to Blender dataset folder')
    parser.add_argument('--exp_name', type=str, default='exp_nerf')
    parser.add_argument('--N_rand', type=int, default=4096)
    parser.add_argument('--N_iters', type=int, default=200000)
    parser.add_argument('--lrate', type=float, default=5e-4)
    parser.add_argument('--precise_val_every', type=int, default=10000)
    parser.add_argument('--log_every', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1, help='images per step')
    parser.add_argument('--half_res', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--render_only', type=int, default=0)
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--save_every', type=int, default=10000)
    parser.add_argument('--no_importance', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('ckpts', exist_ok=True)
    ckpt_dir = os.path.join('ckpts', args.exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Data
    train_set = BlenderDataset(args.data_dir, split='train', half_res=bool(args.half_res))
    val_set = BlenderDataset(args.data_dir, split='val', half_res=bool(args.half_res))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.num_workers)

    H, W, focal = train_set.H, train_set.W, train_set.focal

    # Encoders
    pe_xyz = PositionalEncoding(num_freqs=10, include_input=True).to(device)
    pe_dir = PositionalEncoding(num_freqs=4, include_input=True).to(device)
    in_ch_xyz = pe_xyz.out_dim(3)
    in_ch_dir = pe_dir.out_dim(3)

    # Networks
    net_c = NeRF(D=8, W=256, in_ch_xyz=in_ch_xyz, in_ch_dir=in_ch_dir, skips=(4,)).to(device)
    net_f = NeRF(D=8, W=256, in_ch_xyz=in_ch_xyz, in_ch_dir=in_ch_dir, skips=(4,)).to(device)

    print("in_ch_xyz:", in_ch_xyz)
    print("in_ch_dir:", in_ch_dir)


    cfg = RenderConfig(N_samples=64,
                       N_importance=0 if args.no_importance else 128,
                       perturb=True,
                       white_bkgd=True,
                       chunk=16384,
                       near=2.0,
                       far=6.0)

    # Optimizer
    optimizer = torch.optim.Adam(list(net_c.parameters()) + list(net_f.parameters()), lr=args.lrate)

    # (Optional) load checkpoint
    start_iter = 0
    if args.ckpt_path and os.path.isfile(args.ckpt_path):
        print(f"Loading checkpoint: {args.ckpt_path}")
        ckpt = torch.load(args.ckpt_path, map_location=device)
        net_c.load_state_dict(ckpt['net_c'])
        net_f.load_state_dict(ckpt['net_f'])
        optimizer.load_state_dict(ckpt['opt'])
        start_iter = ckpt.get('iter', 0)

    # Render-only mode
    if args.render_only:
        net_c.eval(); net_f.eval()
        out_dir = os.path.join(ckpt_dir, 'renders')
        os.makedirs(out_dir, exist_ok=True)
        for idx in range(len(val_set)):
            img, pose = val_set[idx]
            c2w = torch.from_numpy(pose).to(device)
            img_pred = render_image(H, W, focal, c2w, net_c, net_f, pe_xyz, pe_dir, cfg, device)
            imageio.imwrite(os.path.join(out_dir, f'render_{idx:03d}.png'), to8b(img_pred))
            print(f"Rendered validation view {idx}")
        return

    # Training loop
    t0 = time.time()
    for it in range(start_iter, args.N_iters):
        loss = train_one_epoch(train_loader, net_c, net_f, pe_xyz, pe_dir, cfg, optimizer, device, N_rand=args.N_rand)

        if (it + 1) % args.log_every == 0:
            print(f"Iter {it+1:>7d} | loss {loss:.6f} | time {time.time()-t0:.1f}s")

        if (it + 1) % args.precise_val_every == 0:
            psnr = validate_once(val_loader, net_c, net_f, pe_xyz, pe_dir, cfg, device)
            print(f"[VAL] Iter {it+1} | PSNR {psnr:.2f}dB")

        if (it + 1) % args.save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, 'latest.pt')
            torch.save({'net_c': net_c.state_dict(),
                        'net_f': net_f.state_dict(),
                        'opt': optimizer.state_dict(),
                        'iter': it + 1,
                        'H': H, 'W': W, 'focal': focal}, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    # Final save
    ckpt_path = os.path.join(ckpt_dir, 'final.pt')
    torch.save({'net_c': net_c.state_dict(),
                'net_f': net_f.state_dict(),
                'opt': optimizer.state_dict(),
                'iter': args.N_iters,
                'H': H, 'W': W, 'focal': focal}, ckpt_path)
    print(f"Training complete. Saved final checkpoint to {ckpt_path}")


if __name__ == '__main__':
    main()
