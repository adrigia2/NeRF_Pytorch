from pathlib import Path
import sys
import os

REPO = Path(__file__).resolve().parents[1]  # ...\OptixProjectCMake
PYD_DIR = (REPO / "out" / "build" / "x64-debug").resolve()

# Windows + Python 3.8+: aiuta per trovare DLL dipendenti
os.add_dll_directory(str(PYD_DIR))

# Per far trovare il .pyd al sistema di import
sys.path.insert(0, str(PYD_DIR))

import depthMapModule as dm
print("Loaded:", dm.__file__)

# Carica il modello
mesh = dm.TriangleMesh()
mesh.addFromObjFile("C:\\Users\\adria\\Documents\\GitHub\\OptixProjectCMake\\TestNerf\\vase.obj")

# Crea il renderer
renderer = dm.SampleRenderer(mesh)

# Genera tutte le depth maps
renderer.generateDepthMapsFromTransform("C:\\Users\\adria\\Documents\\GitHub\\OptixProjectCMake\\TestNerf\\NerfResult\\transforms.json", "C:\\Users\\adria\\Documents\\GitHub\\OptixProjectCMake\\TestNerf\\NerfResult\\depths")

import json
from pathlib import Path

json_path = Path("transformDepth.json")  # se lo script è nella stessa cartella del json
# oppure: json_path = Path(r"C:\percorso\completo\transformDepth.json")

data = json.loads(json_path.read_text(encoding="utf-8"))

for i, frame in enumerate(data.get("frames", []), start=1):
    print(f"{i:03d}  {frame['depth_path']}")
