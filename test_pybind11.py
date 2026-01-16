from pathlib import Path
import sys
import os

from matplotlib import pyplot as plt

# ..\OptixProjectCMake
REPO = Path(__file__).resolve().parents[1] / "OptixProjectCMake"
PYD_DIR = (REPO / "out" / "build" / "x64-debug").resolve()

# Windows + Python 3.8+: aiuta per trovare DLL dipendenti
os.add_dll_directory(str(PYD_DIR))

# Per far trovare il .pyd al sistema di import
sys.path.insert(0, str(PYD_DIR))

import depthMapModule as dm
print("Loaded:", dm.__file__)

# Carica il modello
mesh = dm.TriangleMesh()

# resolve full path
# /SwordShield/vase.obj
obj_path = REPO / "Scenes" / "SwordShield" / "Models" / "SwordShield.obj"
print("Obj path:", obj_path.resolve())

transform_json_path = REPO / "Scenes" / "SwordShield" / "Nerf" / "transforms.json"
print("Transform JSON path:", transform_json_path.resolve())

depths_output_path = REPO / "Scenes" / "SwordShield" / "Depth"
print("Depths output path:", depths_output_path.resolve())

mesh.addFromObjFile(str(obj_path.resolve()))

# Crea il renderer
renderer = dm.SampleRenderer(mesh)

# Genera tutte le depth maps
renderer.generateDepthMapsFromTransform(str(transform_json_path.resolve()), str(depths_output_path.resolve()))

import json
from pathlib import Path
import numpy as np

json_path = REPO / "Scenes" / "SwordShield" / "Depth" / "transformDepth.json"
# oppure: json_path = Path(r"C:\percorso\completo\transformDepth.json")

data = json.loads(json_path.read_text(encoding="utf-8"))

# Ottieni le dimensioni dall'oggetto padre
width = data.get('w', 800)  # valori di default se non presenti
height = data.get('h', 600)

frames = data.get("frames", [])
num_frames = len(frames)

if num_frames > 0:
    # Calcola il layout della griglia (righe x colonne)
    # Ogni frame richiede 2 subplots (RGB + Depth)
    total_subplots = num_frames * 2
    cols = int(np.ceil(np.sqrt(total_subplots)))
    rows = int(np.ceil(total_subplots / cols))
    
    # Crea una figura con subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    
    # Flatten axes per facilitare l'iterazione
    if num_frames > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Itera attraverso tutti i frame e visualizza le depth maps affiancate alle immagini RGB
    for i, frame in enumerate(frames):
        depth_file = depths_output_path / frame['depth_path']
        image_file = depths_output_path / frame['file_path']

        print(image_file)
        
        # Leggi i dati binari della depth map (assumendo float32)
        depth_data = np.fromfile(str(depth_file), dtype=np.float32)
        
        # Ridimensiona l'array in una matrice 2D usando le dimensioni dal JSON padre
        depth_map = depth_data.reshape((height, width))
        
        # Leggi l'immagine RGB
        rgb_image = plt.imread(str(image_file))
        
        # Visualizza l'immagine RGB nel subplot corrente (colonna sinistra)
        axes[i*2].imshow(rgb_image)
        axes[i*2].set_title(f"Frame {i+1:03d} - RGB\n{Path(frame['file_path']).name}")
        axes[i*2].axis('off')
        
        # Visualizza la depth map nel subplot successivo (colonna destra)
        im = axes[i*2+1].imshow(depth_map, cmap='viridis')
        axes[i*2+1].set_title(f"Frame {i+1:03d} - Depth\n{Path(frame['depth_path']).name}")
        axes[i*2+1].axis('off')
        plt.colorbar(im, ax=axes[i*2+1], fraction=0.046, pad=0.04)
    
    # Nascondi gli assi non utilizzati
    for j in range(num_frames*2, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()
