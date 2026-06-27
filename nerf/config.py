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
    # RGB output activation: "exp" (HDR, mirrors NeILFMLP/pbrnerf) or "softplus".
    # Note: checkpoints saved with one activation are NOT compatible with the
    # other (different weight scales).
    rgb_activation: str = "exp"
    # Training loss: "l1" | "mse" |
    #   "rel_mse"     — variante con eps fuori dal quadrato: / (pred²+eps)
    #   "rel_mse_raw" — RawNeRF fedele: / (pred+eps)² (cfr. multinerf train_utils.py)
    #   "log_l1"      — L1 su log1p (comprime gli highlights)
    # Default qui è "l1"; il default effettivo del training viene da
    # nerf_loss_type in images_generator.py (→ "rel_mse_raw").
    loss_type: str = "l1"
    # Initial bias of rgb_linear when rgb_activation="exp".
    # exp(hdr_init_bias) ≈ rgb output at iter 0; choose ~log(scene_target_mean).
    # -3 → exp(-3)≈0.05; -5 → exp(-5)≈0.007. Required to avoid exp dead-zone collapse.
    hdr_init_bias: float = -3.0

    # Rendering chunking
    chunk: int = 1024 * 32

    # Training optimiser
    # LR schedule: exponential decay to 0.1·lrate spread over the full planned
    # run (iter_start + num_iters) — no separate horizon parameter.
    lrate: float = 5e-4

    # Foreground depth-window (mesh surface)
    depth_window: float = 0.5        # samples span [t_hit - window, t_hit + window_end]
    depth_window_end: float = 0.5
    depth_window_samples: int = 32
    opacity_weight: float = 1.0      # deprecated: no longer used (unified loss, no opacity term)

    # Background sphere-shell window (origin = scene centre, t_hit = sphere radius)
    bg_radius_mult: float = 6.0      # sphere radius = bg_radius_mult × max bbox side
    bg_depth_window: float = 2.0     # wider window than mesh (shell is far away)
    bg_depth_window_end: float = 2.0
    # Profiling: per-phase synchronized timing for the first N iters (0 = off)
    profile_iters: int = 0
