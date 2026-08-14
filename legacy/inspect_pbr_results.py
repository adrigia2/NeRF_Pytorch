"""Diagnostic panel of the pbr_solver results, for visual inspection.

DEPRECATED: it reads the old un-nested paths (pixel_change/, camera_texture/,
metallic/, ...). Since the sources/{source}/ layout was introduced those paths are no
longer produced by the pipeline. Script not updated/maintained.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from pbr_solver import _read_exr


def _find_variant(folder: Path, stem: str, primary: str = "nerf") -> Path:
    """Look for {stem}_{primary}.exr, then the first {stem}_*.exr variant, then {stem}.exr."""
    p = folder / f"{stem}_{primary}.exr"
    if p.exists():
        return p
    variants = sorted(folder.glob(f"{stem}_*.exr"))
    if variants:
        return variants[0]
    return folder / f"{stem}.exr"   # fallback legacy


out = Path(sys.argv[1] if len(sys.argv) > 1 else "D:/tesi_output/testNewApproach")

metallic = _read_exr(out / "metallic" / "metallic.exr")   # = 1−X
lobe_p   = _read_exr(out / "pbr" / "lobe_param.exr")
residual = _read_exr(out / "pbr" / "residual.exr")
n_views  = _read_exr(out / "pbr" / "n_views.exr")
albedo   = _read_exr(_find_variant(out / "albedo", "albedo"))
color    = _read_exr(_find_variant(out / "color_texture", "color_texture"))
cmin     = _read_exr(out / "pixel_change" / "color_min.exr")
cvar     = _read_exr(out / "pixel_change" / "color_variance.exr")

# Lobe extremes for cam0, reconstructed from the per-ring means (format "rings")
with open(out / "spec_cone" / "spec_cone_meta.json", encoding="utf-8") as fh:
    _meta = json.load(fh)
_K     = _meta["num_levels"]
_rings = np.stack([_read_exr(out / "spec_cone"
                             / _meta["ring_file_pattern"].format(cam=0, level=k))
                   for k in range(_K)], axis=2)                   # (H, W, K, 3)
_cnts  = _read_exr(out / "spec_cone" / _meta["counts_file_pattern"].format(cam=0))
L0 = _rings[:, :, 0]                                              # mirror
_edges = np.asarray(_meta["ring_edges_cos"], dtype=np.float64)
_w  = 2.0 * np.pi * (_edges[:-1] - _edges[1:])  # solid angle per ring: mean over the whole hemisphere
# With non-uniform sampling the weight is \u03a9_i/N_i (see ring_weights_mean in
# pbr_solver.py): without the division, the more heavily sampled rings would weigh more.
_w = _w / np.asarray(_meta.get("ring_samples", 1.0), dtype=np.float64)
_wc = _w[None, None, :] * _cnts[..., 1:]
L180 = (np.einsum("hwk,hwkc->hwc", _wc, _rings[:, :, 1:])
        / np.maximum(_wc.sum(-1), 1e-12)[..., None])

C0       = _read_exr(out / "camera_texture" / sorted((out / "camera_texture").glob("*.exr"))[0].name)

def tm(x):  # HDR tonemap for display
    return np.clip(x / max(np.percentile(x[x > 0], 99), 1e-6), 0, 1) ** (1 / 2.2)

fig, axs = plt.subplots(3, 4, figsize=(19, 14))
panels = [
    (metallic, "metallic 1−X (0..1)", dict(cmap="viridis", vmin=0, vmax=1)),
    (lobe_p,   "lobe param (s; -1=mirror)", dict(cmap="magma")),
    (np.log10(residual + 1e-8), "log10 residual", dict(cmap="inferno")),
    (n_views,  "n_views",           dict(cmap="cividis")),
    (tm(color),  "color_texture (mean)", {}),
    (tm(cmin),   "color_min (D)",         {}),
    (tm(cvar),   "color_variance",        {}),
    (tm(albedo), "albedo",                {}),
    (tm(C0),     "camera_texture cam0 (C_0)", {}),
    (tm(L0),     "L mirror cam0",           {}),
    (tm(L180),   "L lobe s=0 cam0",           {}),
    (np.log10(cvar.mean(-1) + 1e-9), "log10 mean variance", dict(cmap="inferno")),
]
for ax, (img, title, kw) in zip(axs.flat, panels):
    im = ax.imshow(img, **kw)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    if img.ndim == 2:
        fig.colorbar(im, ax=ax, fraction=0.045)

fig.tight_layout()
fig.savefig(out / "pbr" / "diagnostic_panel.png", dpi=110)
print("saved:", out / "pbr" / "diagnostic_panel.png")

m = metallic[n_views > 0]
r = lobe_p[n_views > 0]
print(f"metallic: mean={m.mean():.3f}  median={np.median(m):.3f}  "
      f">0.5: {(m > 0.5).mean() * 100:.1f}%")
for lv in np.unique(r):
    print(f"  lobe param={lv:8.1f}: {(r == lv).mean() * 100:5.1f}% of the texels")
print(f"mean variance (valid texels): {cvar.mean(-1)[n_views > 0].mean():.3e}, "
      f"median={np.median(cvar.mean(-1)[n_views > 0]):.3e}")
