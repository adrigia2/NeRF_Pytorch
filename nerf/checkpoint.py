from __future__ import annotations

import dataclasses
from pathlib import Path

import torch

from .config import NerfConfig
from .encoding import get_embedder
from .model import NeRF


def _build_models(cfg: NerfConfig, device):
    """Instantiate the single NeRF model and its embedders.

    Returns (model, embed_fn, embeddirs_fn).
    """
    embed_fn, input_ch = get_embedder(cfg.multires)
    if cfg.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(cfg.multires_views)
    else:
        embeddirs_fn, input_ch_views = None, 0

    model = NeRF(D=cfg.netdepth, W=cfg.netwidth, input_ch=input_ch,
                 input_ch_views=input_ch_views, skips=list(cfg.skips),
                 use_viewdirs=cfg.use_viewdirs).to(device)

    if cfg.use_hdr_activation and cfg.use_viewdirs:
        with torch.no_grad():
            torch.nn.init.constant_(model.rgb_linear.bias, cfg.hdr_init_bias)

    return model, embed_fn, embeddirs_fn


def save_checkpoint(path: str, model, optimizer, iter_done: int, cfg: NerfConfig,
                    scene_center=None, sphere_radius: float = 0.0) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    center_list = scene_center.tolist() if scene_center is not None else [0.0, 0.0, 0.0]
    torch.save({
        "model_state":    model.state_dict(),
        "optimizer":      optimizer.state_dict(),
        "iter_done":      iter_done,
        "config":         {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)},
        "scene_center":   center_list,
        "sphere_radius":  float(sphere_radius),
    }, path)


def load_checkpoint(path: str, device=None):
    """Load a saved NeRF checkpoint.

    Returns:
        model_bundle: (model, embed_fn, embeddirs_fn, device, scene_center, sphere_radius)
        cfg: NerfConfig reconstructed from the saved config dict
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(path, map_location=device)
    cfg_dict = ckpt["config"]
    valid_keys = {f.name for f in dataclasses.fields(NerfConfig)}
    cfg = NerfConfig(**{k: v for k, v in cfg_dict.items() if k in valid_keys})

    model, embed_fn, embeddirs_fn = _build_models(cfg, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    center = torch.tensor(ckpt.get("scene_center", [0.0, 0.0, 0.0]),
                          device=device, dtype=torch.float32)
    radius = float(ckpt.get("sphere_radius", 0.0))

    return (model, embed_fn, embeddirs_fn, device, center, radius), cfg
