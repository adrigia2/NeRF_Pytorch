"""Close-up view: specularity (1-X), masked cone_r, sanity checks.

DEPRECATED: it reads the old un-nested paths (pixel_change/, camera_texture/,
metallic/, ...). Since the sources/{source}/ layout was introduced those paths are no
longer produced by the pipeline. Script not updated/maintained.
"""
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

X        = _read_exr(out / "pbr" / "diffuse_weight.exr")    # diffuse weight of the fit
cone_r   = _read_exr(out / "pbr" / "spec_cone_r.exr")
residual = _read_exr(out / "pbr" / "residual.exr")
n_views  = _read_exr(out / "pbr" / "n_views.exr")
color    = _read_exr(_find_variant(out / "color_texture", "color_texture"))
cvar     = _read_exr(out / "pixel_change" / "color_variance.exr").mean(-1)

valid = n_views > 0
spec = np.where(valid, 1.0 - X, 0.0)        # specularity = 1 - X

def tm(x):
    return np.clip(x / max(np.percentile(x[x > 0], 99), 1e-6), 0, 1) ** (1 / 2.2)

fig, axs = plt.subplots(2, 3, figsize=(19, 12.5))

im = axs[0, 0].imshow(spec, cmap="inferno", vmin=0, vmax=1)
axs[0, 0].set_title("specularity  1\u2212X  (0=diffuse, 1=fully specular)")
fig.colorbar(im, ax=axs[0, 0], fraction=0.045)

strong = valid & (spec > 0.25)
r_masked = np.where(strong, cone_r, np.nan)
im = axs[0, 1].imshow(r_masked, cmap="magma", vmin=0, vmax=180)
axs[0, 1].set_title("cone r (\u00b0) only where 1\u2212X > 0.25")
fig.colorbar(im, ax=axs[0, 1], fraction=0.045)

axs[0, 2].imshow(tm(color))
axs[0, 2].set_title("color_texture (reference)")

im = axs[1, 0].imshow(np.log10(residual + 1e-8), cmap="inferno")
axs[1, 0].set_title("log10 mean residual per equation")
fig.colorbar(im, ax=axs[1, 0], fraction=0.045)

axs[1, 1].hist(spec[valid].ravel(), bins=80, color="#9467bd")
axs[1, 1].set_title("histogram of 1\u2212X (solved texels)")
axs[1, 1].set_xlabel("1−X")

# chosen r vs specularity: where the fit is nearly diffuse, r should be noise
bins = [0, .05, .1, .2, .4, 1.01]
labels = []
fracs = []
for a, b in zip(bins[:-1], bins[1:]):
    sel = valid & (spec >= a) & (spec < b)
    if sel.sum() == 0:
        continue
    fracs.append([np.mean(np.abs(cone_r[sel] - lv) < .5)
                  for lv in [0, 10, 25, 50, 90, 130, 180]])
    labels.append(f"{a:.2f}–{b:.2f}\n({int(sel.sum())})")
fr = np.array(fracs).T
bot = np.zeros(len(labels))
for i, lv in enumerate([0, 10, 25, 50, 90, 130, 180]):
    axs[1, 2].bar(labels, fr[i], bottom=bot, label=f"r={lv}°")
    bot += fr[i]
axs[1, 2].set_title("distribution of r per specularity band")
axs[1, 2].legend(fontsize=8, ncols=2)

fig.tight_layout()
fig.savefig(out / "pbr" / "diagnostic_panel2.png", dpi=110)
print("saved:", out / "pbr" / "diagnostic_panel2.png")

print(f"texels with 1\u2212X > 0.25: {int(strong.sum())} "
      f"({strong.sum() / valid.sum() * 100:.1f}% of the solved ones)")
for a, b in zip(bins[:-1], bins[1:]):
    sel = valid & (spec >= a) & (spec < b)
    if sel.sum():
        print(f"  1−X in [{a:.2f},{b:.2f}): {int(sel.sum()):7d} texel, "
              f"median residual {np.median(residual[sel]):.2e}, "
              f"median variance {np.median(cvar[sel]):.2e}")
