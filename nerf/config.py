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
    near: float = 1.0
    far: float = 20.0
    N_samples: int = 64       # coarse samples per ray
    N_importance: int = 128   # fine (hierarchical) samples per ray
    perturb: bool = True
    raw_noise_std: float = 0.0
    white_bkgd: bool = True

    # Rendering chunking
    chunk: int = 1024 * 32

    # Training optimiser
    lrate: float = 5e-4
    lrate_decay: int = 250    # lr decays to lr*0.1 over lrate_decay*1000 steps

    # Depth-hint training: single-pass foreground + traditional background
    # depth_hint_enabled=True activates the fg/bg split mode.
    depth_hint_enabled: bool = False
    depth_window: float = 0.5        # samples span [t_hit - window, t_hit + window_end]
    depth_window_end: float = 0.5
    depth_window_samples: int = 32   # number of samples in the depth window (foreground rays)
    foreground_ratio: float = 0.8    # fraction of each training batch drawn from foreground rays
    opacity_weight: float = 1.0      # weight of the foreground opacity loss (acc_fg → 1)
