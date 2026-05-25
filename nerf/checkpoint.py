from __future__ import annotations

import dataclasses
from pathlib import Path

import torch

from .config import NerfConfig
from .encoding import get_embedder
from .model import NeRF


def _build_models(cfg: NerfConfig, device):
    """Instantiate coarse + fine NeRF models and their embedders."""
    embed_fn, input_ch = get_embedder(cfg.multires)
    if cfg.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(cfg.multires_views)
    else:
        embeddirs_fn, input_ch_views = None, 0

    coarse = NeRF(D=cfg.netdepth, W=cfg.netwidth, input_ch=input_ch,
                  input_ch_views=input_ch_views, skips=list(cfg.skips),
                  use_viewdirs=cfg.use_viewdirs).to(device)
    fine   = NeRF(D=cfg.netdepth, W=cfg.netwidth, input_ch=input_ch,
                  input_ch_views=input_ch_views, skips=list(cfg.skips),
                  use_viewdirs=cfg.use_viewdirs).to(device)

    return coarse, fine, embed_fn, embeddirs_fn


def save_checkpoint(path: str, coarse_model, fine_model, optimizer,
                    iter_done: int, cfg: NerfConfig) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "coarse_state": coarse_model.state_dict(),
        "fine_state":   fine_model.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "iter_done":    iter_done,
        "config":       {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)},
    }, path)


def load_checkpoint(path: str, device=None):
    """Load a saved NeRF checkpoint.

    Returns:
        model_bundle: (coarse_model, fine_model, embed_fn, embeddirs_fn, device)
        cfg: NerfConfig reconstructed from the saved config dict
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(path, map_location=device)
    cfg_dict = ckpt["config"]
    valid_keys = {f.name for f in dataclasses.fields(NerfConfig)}
    cfg = NerfConfig(**{k: v for k, v in cfg_dict.items() if k in valid_keys})

    coarse, fine, embed_fn, embeddirs_fn = _build_models(cfg, device)
    coarse.load_state_dict(ckpt["coarse_state"])
    fine.load_state_dict(ckpt["fine_state"])
    coarse.eval()
    fine.eval()

    return (coarse, fine, embed_fn, embeddirs_fn, device), cfg
