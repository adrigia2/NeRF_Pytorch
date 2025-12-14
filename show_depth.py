import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

bin_path = Path(r"C:\\Users\\adria\\Documents\\GitHub\\OptixProjectCMake\\TestNerf\\NerfResult\\depths\\depth_render_Camera_Front_5.bin")

# 1) leggi il vector<float> (float32)
v = np.fromfile(bin_path, dtype=np.float32)

# 2) scegli la shape (esempio: 1920x1080)
w, h = 1920, 1080
depth = v.reshape(h, w)

# 3) opzionale: maschera valori "invalidi" (spesso 1e20 o simili)
depth_vis = depth.copy()
depth_vis[depth_vis >= 1e19] = np.nan

plt.imshow(depth_vis)   # mostra come immagine
plt.colorbar()
plt.title("Depth map")
plt.show()
