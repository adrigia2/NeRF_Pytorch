from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NerfConfig:
    # Positional encoding
    multires: int = 10
    multires_views: int = 4

    # Network architecture
    netdepth: int = 8
    netwidth: int = 256
    skips: tuple = (4,)
    use_viewdirs: bool = True

    # Sampling
    near: float = 0.01
    far: float = 20.0          # overwritten at runtime from sphere radius
    perturb: bool = True
    raw_noise_std: float = 0.0

    # Rendering chunking
    chunk: int = 1024 * 32

    # Training optimiser
    lrate: float = 5e-4
    lrate_decay: int = 250    # lr decays to lr*0.1 over lrate_decay*1000 steps

    # Foreground depth-window (mesh surface)
    depth_window: float = 0.5        # samples span [t_hit - window, t_hit + window_end]
    depth_window_end: float = 0.5
    depth_window_samples: int = 32
    opacity_weight: float = 1.0      # weight of opacity loss for both fg and bg

    # Background sphere-shell window (origin = scene centre, t_hit = sphere radius)
    bg_radius_mult: float = 6.0      # sphere radius = bg_radius_mult × max bbox side
    bg_depth_window: float = 2.0     # wider window than mesh (shell is far away)
    bg_depth_window_end: float = 2.0
    bg_depth_window_samples: int = 32

    # Profiling: per-phase synchronized timing for the first N iters (0 = off)
    profile_iters: int = 0
