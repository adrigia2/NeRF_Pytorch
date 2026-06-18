"""Pannello diagnostico dei risultati pbr_solver per ispezione visiva."""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from pbr_solver import _read_exr

out = Path(sys.argv[1] if len(sys.argv) > 1 else "D:/tesi_output/testNewApproach")

metallic = _read_exr(out / "metallic" / "metallic.exr")   # = 1−X
cone_r   = _read_exr(out / "pbr" / "spec_cone_r.exr")
residual = _read_exr(out / "pbr" / "residual.exr")
n_views  = _read_exr(out / "pbr" / "n_views.exr")
albedo   = _read_exr(out / "albedo" / "albedo.exr")
color    = _read_exr(out / "color_texture" / "color_texture.exr")
cmin     = _read_exr(out / "pixel_change" / "color_min.exr")
cvar     = _read_exr(out / "pixel_change" / "color_variance.exr")
L0       = _read_exr(out / "spec_cone" / "cam_000_r00.exr")
L180     = _read_exr(out / "spec_cone" / "cam_000_r06.exr")
C0       = _read_exr(out / "camera_texture" / sorted((out / "camera_texture").glob("*.exr"))[0].name)

def tm(x):  # tonemap HDR per display
    return np.clip(x / max(np.percentile(x[x > 0], 99), 1e-6), 0, 1) ** (1 / 2.2)

fig, axs = plt.subplots(3, 4, figsize=(19, 14))
panels = [
    (metallic, "metallic 1−X (0..1)", dict(cmap="viridis", vmin=0, vmax=1)),
    (cone_r,   "cone r (gradi)",    dict(cmap="magma", vmin=0, vmax=180)),
    (np.log10(residual + 1e-8), "log10 residuo", dict(cmap="inferno")),
    (n_views,  "n_views",           dict(cmap="cividis")),
    (tm(color),  "color_texture (media)", {}),
    (tm(cmin),   "color_min (D)",         {}),
    (tm(cvar),   "color_variance",        {}),
    (tm(albedo), "albedo",                {}),
    (tm(C0),     "camera_texture cam0 (C_0)", {}),
    (tm(L0),     "L_0(r=0) specchio cam0",    {}),
    (tm(L180),   "L_0(r=180°) cam0",          {}),
    (np.log10(cvar.mean(-1) + 1e-9), "log10 varianza media", dict(cmap="inferno")),
]
for ax, (img, title, kw) in zip(axs.flat, panels):
    im = ax.imshow(img, **kw)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    if img.ndim == 2:
        fig.colorbar(im, ax=ax, fraction=0.045)

fig.tight_layout()
fig.savefig(out / "pbr" / "diagnostic_panel.png", dpi=110)
print("salvato:", out / "pbr" / "diagnostic_panel.png")

m = metallic[n_views > 0]
r = cone_r[n_views > 0]
print(f"metallic: media={m.mean():.3f}  mediana={np.median(m):.3f}  "
      f">0.5: {(m > 0.5).mean() * 100:.1f}%")
for lv in [0, 10, 25, 50, 90, 130, 180]:
    print(f"  r={lv:5.1f}°: {(np.abs(r - lv) < 0.5).mean() * 100:5.1f}% dei texel")
print(f"varianza media (texel validi): {cvar.mean(-1)[n_views > 0].mean():.3e}, "
      f"mediana={np.median(cvar.mean(-1)[n_views > 0]):.3e}")
