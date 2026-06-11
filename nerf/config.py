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
    # HDR mode: torch.exp activation on RGB instead of softplus.
    # Note: checkpoints saved with use_hdr_activation=False are NOT compatible with True (different weight scales).
    use_hdr_activation: bool = False
    # Initial bias of rgb_linear when use_hdr_activation=True.
    # exp(hdr_init_bias) ≈ mean RGB emitted at iter 0; must be ≈ log(mean_target) to keep exp
    # in its productive gradient zone (too high → explosion, too low → vanishing gradients).
    # After linear normalisation by s, mean_target ≈ mean_raw/s, so the correct value is
    # log(mean_raw) - log(s) — more negative than the un-normalised case.
    # images_generator.py computes and sets this automatically when normalize_images=True;
    # -3 (≈0.05) is a reasonable fallback for un-normalised HDR (mean ~0.05–0.1).
    hdr_init_bias: float = -3.0

    # Loss function used during training.
    # "tonemap_l1"  (default) — L1 on log1p(pred) vs log1p(target): relative error on bright
    #               pixels (correct for albedo = color/irr ratio), absolute on dark ones (stable).
    #               Best choice when training on HDR data.
    # "rel_mse"     — RawNeRF-style relative MSE: (pred-gt)² / (pred²+eps). Good relative
    #               behaviour but eps=1e-3 over-weights dark pixels; eps should be tuned to scale.
    # "l1"          — plain linear L1. Dominated by bright highlights; use only for LDR [0,1] data.
    loss_type: str = "tonemap_l1"

    # Rendering chunking
    chunk: int = 1024 * 32

    # Training optimiser
    lrate: float = 5e-4
    lrate_decay: int = 250    # lr decays to lr*0.1 over lrate_decay*1000 steps

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
