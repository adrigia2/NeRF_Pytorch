# Vendored from https://github.com/yenchenlin/nerf-pytorch
# Original file: run_nerf_helpers.py
# Commit: 63a5a630c9abd62b0f21c08703d0ac2ea7d4b9dd
# License: MIT
# Modifications: extracted NeRF class; removed load_weights_from_keras (Keras legacy helper).

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkyModel(nn.Module):
    """Directional sky MLP: maps encoded view direction → HDR RGB.

    Replaces the white background constant with a learned environment.
    Input must be pre-encoded with the same embeddirs_fn used by the main NeRF.
    View directions must be normalised before encoding.
    """

    def __init__(self, input_ch_views: int, D: int = 3, W: int = 128):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = input_ch_views
        for _ in range(D):
            layers += [nn.Linear(in_ch, W), nn.ReLU(inplace=True)]
            in_ch = W
        layers.append(nn.Linear(W, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, dirs_encoded):
        return F.softplus(self.net(dirs_encoded))


class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4,
                 skips=(4,), use_viewdirs=False):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W)
                for i in range(D - 1)
            ]
        )
        # Implementation following the official code release
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, _ in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
            for i, _ in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)
            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs
