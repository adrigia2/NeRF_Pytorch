"""Figura riassuntiva delle mappe PBR finali (metallic/roughness)."""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from pbr_solver import _read_exr

out = Path(sys.argv[1] if len(sys.argv) > 1 else "D:/tesi_output/testNewApproach")
met = _read_exr(out / "metallic" / "metallic.exr")
rgh = _read_exr(out / "roughness" / "roughness.exr")
col = _read_exr(out / "color_texture" / "color_texture.exr")


def tm(x):
    return np.clip(x / max(np.percentile(x[x > 0], 99), 1e-6), 0, 1) ** (1 / 2.2)


fig, axs = plt.subplots(1, 3, figsize=(19, 6.5))
im = axs[0].imshow(met, cmap="inferno", vmin=0, vmax=1)
axs[0].set_title("metallic (1−X)")
fig.colorbar(im, ax=axs[0], fraction=0.045)
im = axs[1].imshow(rgh, cmap="viridis", vmin=0, vmax=1)
axs[1].set_title("roughness (r/180; 1 = non vincolato)")
fig.colorbar(im, ax=axs[1], fraction=0.045)
axs[2].imshow(tm(col))
axs[2].set_title("color_texture (riferimento)")
for a in axs:
    a.axis("off")
fig.tight_layout()
fig.savefig(out / "pbr" / "final_maps.png", dpi=110)
print("salvato:", out / "pbr" / "final_maps.png")
