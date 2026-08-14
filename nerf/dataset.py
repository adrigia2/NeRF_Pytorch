from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .rays import get_rays_np


def _load_image_float(path: str) -> np.ndarray:
    """Load image as (H, W, 3) float32 in [0, 1].
    EXR → linear float (assumed already normalised by Step 1).
    LDR (PNG/JPG) → uint8 / 255.
    """
    if path.lower().endswith(".exr"):
        import OpenEXR, Imath
        exr = OpenEXR.InputFile(path)
        dw  = exr.header()["dataWindow"]
        w   = dw.max.x - dw.min.x + 1
        h   = dw.max.y - dw.min.y + 1
        pt  = Imath.PixelType(Imath.PixelType.FLOAT)
        chs = exr.header()["channels"]
        if "R" in chs and "G" in chs and "B" in chs:
            r = np.frombuffer(exr.channel("R", pt), dtype=np.float32).reshape(h, w)
            g = np.frombuffer(exr.channel("G", pt), dtype=np.float32).reshape(h, w)
            b = np.frombuffer(exr.channel("B", pt), dtype=np.float32).reshape(h, w)
        else:
            key = next(iter(chs))
            ch  = np.frombuffer(exr.channel(key, pt), dtype=np.float32).reshape(h, w)
            r = g = b = ch
        return np.stack([r, g, b], axis=-1)
    else:
        from PIL import Image
        return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _load_mask_float(path: str) -> np.ndarray | None:
    """Load OptiX mask as (H, W) float32 in {0, 1}.
    PNG saved by PngWriter is 0 or 255 → divide by 255.
    Returns None on failure.
    """
    try:
        from PIL import Image
        img = Image.open(path).convert("L")  # grayscale
        arr = np.array(img, dtype=np.float32) / 255.0
        return (arr > 0.5).astype(np.float32)
    except Exception:
        return None


def _load_depth_exr(path: str) -> np.ndarray | None:
    """Load single-channel depth EXR as (H, W) float32.
    Clamps OptiX miss-sentinel values (≥ 1e10 → 0).
    """
    try:
        import OpenEXR, Imath
        exr = OpenEXR.InputFile(path)
        dw  = exr.header()["dataWindow"]
        w   = dw.max.x - dw.min.x + 1
        h   = dw.max.y - dw.min.y + 1
        pt  = Imath.PixelType(Imath.PixelType.FLOAT)
        key = next(iter(exr.header()["channels"]))
        ch  = np.frombuffer(exr.channel(key, pt), dtype=np.float32).reshape(h, w)
        return np.where(ch >= 1e10, 0.0, ch).astype(np.float32)
    except Exception:
        return None


class NerfDataset:
    """Training dataset loaded from transforms_extended.json.

    Pre-computes all ray bundles and flattens them into a single array indexed by
    a flat ray id.  Two ways to draw a batch out of it:

    * ``sample_natural`` — uniform *with replacement*, stateless.  Used only for
      the diagnostic batch of the display block.
    * ``configure_epochs`` + ``sample_epoch`` — the training path: one shuffled
      pass over every ray per epoch, so each ray is seen exactly once per epoch
      and the last batch of the epoch is simply shorter.

    One frame is designated as the preview view, accessible via
    get_preview_frame(); with hold_out_preview=True its rays are excluded from the
    training pool, otherwise (the default) it is trained on like any other frame
    and the preview is a debug render of a seen view, not a held-out evaluation.
    """

    def __init__(self, transforms_path: str, device, preview_idx: int = -1,
                 composite_white: bool = True, hold_out_preview: bool = False):
        path = Path(transforms_path)
        base = path.parent

        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)

        self.H       = int(data["h"])
        self.W       = int(data["w"])
        self.focal_x = float(data["fl_x"])
        self.focal_y = float(data.get("fl_y", data["fl_x"]))
        cx = float(data.get("cx", self.W / 2.0))
        cy = float(data.get("cy", self.H / 2.0))

        K = np.array([[self.focal_x, 0, cx],
                      [0, self.focal_y, cy],
                      [0, 0, 1]], dtype=np.float32)

        frames = data["frames"]
        if preview_idx == -1:
            preview_idx = min(len(frames) - 1, 5)
        self._preview_idx = preview_idx

        all_rays_o, all_rays_d, all_rgb, all_depths = [], [], [], []
        self._frames_meta = []

        for i, frame in enumerate(frames):
            img_path = frame["file_path"]
            if not Path(img_path).is_absolute():
                img_path = (base / img_path).resolve().as_posix()

            depth_path = frame.get("depth_path", "")
            if depth_path and not Path(depth_path).is_absolute():
                depth_path = (base / depth_path).resolve().as_posix()

            mask_path = frame.get("mask_path", "")
            if mask_path and not Path(mask_path).is_absolute():
                mask_path = (base / mask_path).resolve().as_posix()

            pose = np.array(frame["transform_matrix"], dtype=np.float32)[:3, :4]
            rays_o, rays_d = get_rays_np(self.H, self.W, K, pose)

            img = np.clip(_load_image_float(img_path), 0.0, None)  # allow HDR values > 1
            dep = _load_depth_exr(depth_path) if depth_path else None

            # When composite_white=True, blend bg pixels to white before storing as GT.
            # When composite_white=False, keep real environment pixels as GT for background rays.
            if mask_path and composite_white:
                mask = _load_mask_float(mask_path)
                if mask is not None:
                    img = img * mask[..., None] + (1.0 - mask[..., None]) * 1.0

            self._frames_meta.append({"img_path": img_path, "pose": pose, "dep": dep})

            if i == preview_idx:
                self._test_rays_o = rays_o.reshape(-1, 3)
                self._test_rays_d = rays_d.reshape(-1, 3)
                self._test_rgb    = img.reshape(-1, 3)
                self._test_pose   = pose
                self._test_dep    = dep
                if hold_out_preview:
                    continue

            all_rays_o.append(rays_o.reshape(-1, 3))
            all_rays_d.append(rays_d.reshape(-1, 3))
            all_rgb.append(img.reshape(-1, 3))
            dep_flat = dep.reshape(-1) if dep is not None else np.zeros(self.H * self.W, dtype=np.float32)
            all_depths.append(dep_flat)

        self.device   = device
        self._rays_o  = torch.tensor(np.concatenate(all_rays_o), device=device, dtype=torch.float32)
        self._rays_d  = torch.tensor(np.concatenate(all_rays_d), device=device, dtype=torch.float32)
        self._rgb     = torch.tensor(np.concatenate(all_rgb),    device=device, dtype=torch.float32)
        self._n_rays  = self._rays_o.shape[0]

        # Per-ray depth and fg/bg index split (populated when depth maps are available)
        depths_np = np.concatenate(all_depths) if all_depths else np.array([], dtype=np.float32)
        if depths_np.size == self._n_rays:
            self._depths = torch.tensor(depths_np, device=device, dtype=torch.float32)
            fg_mask = depths_np > 1e-6
            self._fg_idx = torch.tensor(np.where(fg_mask)[0],  dtype=torch.long, device=device)
            self._bg_idx = torch.tensor(np.where(~fg_mask)[0], dtype=torch.long, device=device)
        else:
            self._depths = None
            self._fg_idx = torch.zeros(0, dtype=torch.long, device=device)
            self._bg_idx = torch.zeros(0, dtype=torch.long, device=device)

        # Epoch-sampling state: inactive until configure_epochs() is called
        # (sample_epoch raises if it was not).
        self._epoch_batch     = 0
        self._epoch_seed      = 0
        self._iters_per_epoch = 0
        self._perm            = None
        self._perm_epoch      = -1

        n_fg = int((self._fg_idx.numel()))
        n_bg = int((self._bg_idx.numel()))
        _preview_regime = "held out" if hold_out_preview else "in training"
        print(f"  Dataset: {len(frames)} frames, {self._n_rays} training rays "
              f"(fg={n_fg}, bg={n_bg}), preview_idx={preview_idx} ({_preview_regime})")
        if n_fg > 0:
            fg_rgb = self._rgb[self._fg_idx]
            print(f"  [diag] fg target HDR: min={fg_rgb.min():.3f}  mean={fg_rgb.mean():.3f}  "
                  f"max={fg_rgb.max():.3f}  (>1: {(fg_rgb > 1).float().mean()*100:.1f}%  "
                  f">10: {(fg_rgb > 10).float().mean()*100:.1f}%)")

    @property
    def num_frames(self) -> int:
        return len(self._frames_meta)

    @property
    def n_rays(self) -> int:
        """Total number of rays in the training pool."""
        return self._n_rays

    @property
    def iters_per_epoch(self) -> int:
        """Iterations per epoch; 0 until configure_epochs() has been called."""
        return self._iters_per_epoch

    @property
    def has_depth_split(self) -> bool:
        """True when per-ray depth data is available and both fg/bg pools are non-empty."""
        return (self._depths is not None
                and self._fg_idx.numel() > 0
                and self._bg_idx.numel() > 0)

    def sample_natural(self, batch_size: int):
        """Uniform sample over all rays, WITH replacement. Fixed-shape batch.

        Returns (rays_o, rays_d, rgb, depths, in_mask) all of shape (batch_size, ...).
        in_mask is True for foreground rays (mesh hit, depth > 0).
        depths is 0 for background rays.

        This is no longer the training sampler (see sample_epoch): it remains for
        the diagnostic batch of the display block, which has to be independent of
        the epoch order so that it does not consume positions from it.
        """
        idxs    = torch.randint(0, self._n_rays, (batch_size,), device=self.device)
        depths  = self._depths[idxs]
        in_mask = depths > 1e-6
        return (self._rays_o[idxs], self._rays_d[idxs],
                self._rgb[idxs], depths, in_mask)

    # ── epoch sampling ────────────────────────────────────────────────────────

    def configure_epochs(self, batch_size: int, seed: int) -> int:
        """Enable epoch ordering for sample_epoch(). Returns iters_per_epoch.

        An epoch is a permutation of the whole ray pool consumed one batch at a
        time: every ray is seen exactly once per epoch, and the last batch is
        shorter whenever batch_size does not divide n_rays.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self._epoch_batch     = int(batch_size)
        self._epoch_seed      = int(seed)
        self._iters_per_epoch = (self._n_rays + batch_size - 1) // batch_size
        self._perm            = None
        self._perm_epoch      = -1
        return self._iters_per_epoch

    def _epoch_indices(self, iteration: int) -> torch.Tensor:
        """Indices (int64, on device) of the batch for the given ABSOLUTE iteration.

        Epoch and offset are derived from `iteration`, so the order is a pure
        function of (seed, batch_size, n_rays) and a resume mid-epoch picks up
        exactly where it left off without storing anything in the checkpoint.
        The permutation comes from a DEDICATED Generator: the diagnostic sampling
        in train() draws on the global RNG and must not be able to shift the epoch
        order (nor the other way round).

        It lives on the CPU as int32: on this scene that is 475 MiB which would
        otherwise add to the ~6 GB of dataset already resident in VRAM, and
        torch.randperm on CUDA would allocate several GB of temporaries at every
        epoch boundary.  The slice actually transferred is a few hundred KB.
        """
        if self._iters_per_epoch == 0:
            raise RuntimeError("configure_epochs() has not been called on this dataset.")

        epoch, k = divmod(int(iteration), self._iters_per_epoch)

        if self._perm_epoch != epoch:
            g = torch.Generator()
            g.manual_seed((self._epoch_seed * 1_000_003 + epoch) % (2 ** 63 - 1))
            # assigned in two steps so the old permutation is released before the
            # new one is allocated
            self._perm = None
            self._perm = torch.randperm(self._n_rays, generator=g, dtype=torch.int32)
            self._perm_epoch = epoch

        lo = k * self._epoch_batch
        hi = min(lo + self._epoch_batch, self._n_rays)
        return self._perm[lo:hi].to(self.device, non_blocking=True).long()

    def sample_epoch(self, iteration: int):
        """Batch for absolute iteration `iteration`, in epoch order.

        Same tuple as sample_natural — (rays_o, rays_d, rgb, depths, in_mask) —
        but the last batch of each epoch has fewer than batch_size elements.
        """
        idxs    = self._epoch_indices(iteration)
        depths  = self._depths[idxs]
        in_mask = depths > 1e-6
        return (self._rays_o[idxs], self._rays_d[idxs],
                self._rgb[idxs], depths, in_mask)

    def get_preview_frame(self):
        """Returns (rays_o_np, rays_d_np, rgb_np, pose_3x4, dep_hw) for the preview view.

        Held out from training only when the dataset was built with
        hold_out_preview=True; otherwise this is a training view.
        """
        return (self._test_rays_o, self._test_rays_d,
                self._test_rgb, self._test_pose, self._test_dep)

    def compute_scene_bounds(self):
        """Measure how far the scene geometry extends from the world origin.

        Hit points are p = rays_o + t_mesh * rays_d for all foreground rays; rays_d is
        a unit vector and t_mesh a metric OptiX distance, so p lies on the mesh surface.

        Returns (scene_radius, p_min, p_max) as CPU float tensors, where scene_radius is
        max |p| — the distance from the world ORIGIN to the farthest surface point.

        No centre is returned: the background sphere is anchored at the world origin by
        design, so the only thing the geometry has to determine is a radius large enough
        for the shell to sit entirely outside it.  max |p| is the measure that actually
        references the origin; the AABB midpoint would not (it is a property of the box,
        and on an off-centre scene a radius derived from the box side could leave the
        shell intersecting the geometry with nothing to flag it).  The AABB is still
        returned because it is useful diagnostics.
        """
        if self._depths is None or self._fg_idx.numel() == 0:
            raise RuntimeError("No foreground rays with depth — cannot compute scene bounds.")
        pts = (self._rays_o[self._fg_idx] +
               self._depths[self._fg_idx, None] * self._rays_d[self._fg_idx])
        scene_radius = pts.norm(dim=1).max()
        return scene_radius, pts.min(0).values, pts.max(0).values

    def get_frame_meta(self, i: int) -> dict:
        return self._frames_meta[i]

    def sample_batch(self, batch_size: int):
        """Sample a random batch of (rays_o, rays_d, rgb) tensors from training set."""
        idxs = torch.randint(0, self._n_rays, (batch_size,), device=self.device)
        return self._rays_o[idxs], self._rays_d[idxs], self._rgb[idxs]
