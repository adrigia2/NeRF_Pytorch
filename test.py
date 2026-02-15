# %% [markdown]
# # Pipeline OptiX: Depth Maps + IUM + Custom Transforms
# 
# Questo notebook crea una pipeline completa che:
# 1. Genera **Depth Maps** usando OptiX
# 2. Genera **Inverse UV Mapping (IUM)**
# 3. Crea un **transforms.json custom** che include sia i path delle immagini originali che delle depth maps

# %% [markdown]
# ## Setup e Import

# %%
# Import dei moduli necessari
# ..\OptixProjectCMake
from pathlib import Path
import sys
import os
import json

# ..\OptixProjectCMake
REPO = "C:/Users/adria/Documents/GitHub/OptixProjectCMake"
PYD_DIR = "C:/Users/adria/Documents/GitHub/OptixProjectCMake/out/build/vcpkg-x64-debug"

# Windows + Python 3.8+: aiuta per trovare DLL dipendenti
os.add_dll_directory(str(PYD_DIR))

# Per far trovare il .pyd al sistema di import
sys.path.insert(0, str(PYD_DIR))

import depthMapModule as optix
print("Loaded:", optix.__file__)

print("✓ Moduli importati con successo")
print(f"OptiX module location: {PYD_DIR}")

# %% [markdown]
# ## Configurazione Percorsi
# 
# Configura qui i percorsi per:
# - Modello 3D (.obj)
# - transforms.json originale
# - Directory di output per depth maps e IUM

# %%
# === CONFIGURAZIONE PERCORSI ===

# Percorso del modello 3D
MODEL_PATH = "C:/Users/adria/Documents/GitHub/OptixProjectCMake/Scenes/SwordShield/Models/SwordShield.obj"

# Percorso del transforms.json originale
TRANSFORMS_INPUT = "C:/Users/adria/Documents/GitHub/OptixProjectCMake/Scenes/SwordShield/Nerf480p/transforms.json"

# Directory di output
OUTPUT_DIR = "./output"
DEPTH_OUTPUT_DIR = f"{OUTPUT_DIR}/depth_maps/"
IUM_OUTPUT_DIR = f"{OUTPUT_DIR}/ium/"

# Percorso per il transforms.json custom (output)
TRANSFORMS_OUTPUT = f"{OUTPUT_DIR}/transforms_with_depth.json"

# Risoluzione per IUM
IUM_WIDTH = 1024
IUM_HEIGHT = 1024

# Tipo di immagine per depth maps e IUM
IMAGE_TYPE = optix.ImageResultType.OpenEXR  # o ImageResultType.BMP

# Crea le directory se non esistono
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DEPTH_OUTPUT_DIR, exist_ok=True)
os.makedirs(IUM_OUTPUT_DIR, exist_ok=True)

print("✓ Configurazione completata")
print(f"  Modello: {MODEL_PATH}")
print(f"  Input transforms: {TRANSFORMS_INPUT}")
print(f"  Output dir: {OUTPUT_DIR}")

# %% [markdown]
# ## Step 1: Caricamento Modello 3D

# %%
# Inizializza la pipeline OptiX
pipeline = optix.OptiXPipeline()

# Carica il modello 3D
print(f"Caricamento modello: {MODEL_PATH}")
pipeline.load_model(MODEL_PATH)

if pipeline.is_model_loaded():
    print(f"✓ Modello caricato con successo!")
    print(f"  Vertici: {pipeline.get_vertex_count()}")
    print(f"  Triangoli: {pipeline.get_triangle_count()}")
else:
    print("✗ Errore nel caricamento del modello")

# %% [markdown]
# ## Step 2: Generazione Inverse UV Mapping (IUM)

# %%
# Genera l'Inverse UV Mapping
print(f"\nGenerazione IUM ({IUM_WIDTH}x{IUM_HEIGHT})...")

pipeline.generate_ium(
    output_path=IUM_OUTPUT_DIR,
    file_name="inverse_uv_mapping",
    image_type=IMAGE_TYPE,
    width=IUM_WIDTH,
    height=IUM_HEIGHT
)

print("✓ IUM generato con successo!")
print(f"  Salvato in: {IUM_OUTPUT_DIR}")

# %% [markdown]
# ## Step 3: Generazione Depth Maps

# %%
# Genera le depth maps per tutte le camere nel transforms.json
print(f"\nGenerazione depth maps...")
print(f"  Input: {TRANSFORMS_INPUT}")
print(f"  Output: {DEPTH_OUTPUT_DIR}")

pipeline.generate_depth_maps(
    transform_file=TRANSFORMS_INPUT,
    output_dir=DEPTH_OUTPUT_DIR,
    image_type=IMAGE_TYPE
)

print("✓ Depth maps generate con successo!")

# %% [markdown]
# ## Step 4: Creazione Transforms.json Custom
# 
# Legge il transforms.json originale e aggiunge i percorsi delle depth maps generate per ogni frame

# %%
# Carica il transforms.json originale
print(f"\nLettura transforms.json originale...")
with open(TRANSFORMS_INPUT, 'r') as f:
    transforms_data = json.load(f)

print(f"  Frame totali: {len(transforms_data['frames'])}")

# Lista i file depth generati
depth_files = sorted([f for f in os.listdir(DEPTH_OUTPUT_DIR) if f.endswith('.exr')])
print(f"  Depth maps generate: {len(depth_files)}")

# Crea un dizionario per mappare i nomi delle camere ai file depth
# I file depth hanno formato: depth_render_CameraName_N.exr
depth_map = {}
for depth_file in depth_files:
    depth_map[depth_file] = os.path.join(DEPTH_OUTPUT_DIR, depth_file)

print(f"\nMappa depth files creata con {len(depth_map)} entries")

# %%
# Funzione per estrarre il nome della camera dal file_path
def extract_camera_name(file_path):
    """
    Estrae il nome della camera dal path dell'immagine
    Es: 'render_Camera_Back_1.png' -> 'Camera_Back_1'
    """
    filename = os.path.basename(file_path)
    # Rimuove 'render_' dal nome se presente
    if filename.startswith('render_'):
        filename = filename[7:]  # rimuove 'render_'
    # Rimuove l'estensione
    name_without_ext = os.path.splitext(filename)[0]
    return name_without_ext

def find_matching_depth(camera_name, depth_files):
    """
    Trova il file depth corrispondente al nome della camera
    Es: 'Camera_Back_1' -> 'depth_render_Camera_Back_1.exr'
    """
    # Cerca un file depth che contenga il nome della camera
    for depth_file in depth_files:
        # Il formato atteso è: depth_render_CameraName.exr
        if camera_name in depth_file:
            return depth_file
    return None

# Test della funzione
test_path = transforms_data['frames'][0]['file_path']
test_camera = extract_camera_name(test_path)
test_depth = find_matching_depth(test_camera, depth_files)

print(f"Test mapping:")
print(f"  Original path: {test_path}")
print(f"  Camera name: {test_camera}")
print(f"  Depth file: {test_depth}")

# %%
# Aggiungi i percorsi depth a ogni frame
print(f"\nAggiunta percorsi depth ai frames...")

frames_with_depth = 0
frames_without_depth = 0

for frame in transforms_data['frames']:
    # Estrai il nome della camera dall'immagine originale
    camera_name = extract_camera_name(frame['file_path'])
    
    # Trova il file depth corrispondente
    depth_file = find_matching_depth(camera_name, depth_files)
    
    if depth_file:
        # Aggiungi il percorso completo della depth map al frame
        frame['depth_path'] = os.path.join(DEPTH_OUTPUT_DIR, depth_file).replace('\\', '/')
        frames_with_depth += 1
    else:
        # Se non trova la depth, imposta a null
        frame['depth_path'] = None
        frames_without_depth += 1
        print(f"  ⚠ Depth non trovata per: {camera_name}")

print(f"\n✓ Frames processati:")
print(f"  Con depth: {frames_with_depth}")
print(f"  Senza depth: {frames_without_depth}")

# %%
# Salva il transforms.json arricchito
print(f"\nSalvataggio transforms.json custom...")
print(f"  Output: {TRANSFORMS_OUTPUT}")

with open(TRANSFORMS_OUTPUT, 'w') as f:
    json.dump(transforms_data, f, indent=4)

print("✓ transforms.json custom salvato con successo!")

# %% [markdown]
# ## Step 5: Verifica e Visualizza Risultati

# %%
# Mostra un esempio di frame con depth
print("\n=== ESEMPIO FRAME CON DEPTH ===")
example_frame = transforms_data['frames'][0]
print(json.dumps(example_frame, indent=2))

# %%
# Riepilogo finale
print("\n" + "="*60)
print("PIPELINE COMPLETATA CON SUCCESSO!")
print("="*60)
print(f"\n📁 Output generati:")
print(f"  ├─ IUM: {IUM_OUTPUT_DIR}")
print(f"  ├─ Depth Maps: {DEPTH_OUTPUT_DIR}")
print(f"  └─ Transforms custom: {TRANSFORMS_OUTPUT}")
print(f"\n📊 Statistiche:")
print(f"  ├─ Modello: {pipeline.get_vertex_count()} vertici, {pipeline.get_triangle_count()} triangoli")
print(f"  ├─ Depth maps: {len(depth_files)} files")
print(f"  └─ Frames: {len(transforms_data['frames'])} totali")

# %% [markdown]
# ## (Opzionale) Visualizza Depth Map
# 
# Usa questa cella per visualizzare una depth map usando OpenEXR e matplotlib

# %%
# Visualizza una depth map (richiede OpenEXR e matplotlib)
try:
    import OpenEXR
    import Imath
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Carica il primo file depth
    if depth_files:
        depth_path = os.path.join(DEPTH_OUTPUT_DIR, depth_files[0])
        print(f"Visualizzazione: {depth_files[0]}")
        
        # Leggi il file EXR
        exr_file = OpenEXR.InputFile(depth_path)
        header = exr_file.header()
        dw = header['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        
        # Leggi il canale R (la depth)
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        redstr = exr_file.channel('R', FLOAT)
        depth_data = np.frombuffer(redstr, dtype=np.float32)
        depth_data = depth_data.reshape((size[1], size[0]))
        
        # Visualizza
        plt.figure(figsize=(12, 8))
        plt.imshow(depth_data, cmap='viridis')
        plt.colorbar(label='Depth')
        plt.title(f'Depth Map: {depth_files[0]}')
        plt.tight_layout()
        plt.show()
        
        print(f"Depth range: [{depth_data.min():.3f}, {depth_data.max():.3f}]")
    else:
        print("Nessun file depth disponibile")
        
except ImportError:
    print("Per visualizzare le depth maps, installa: pip install OpenEXR matplotlib numpy")
except Exception as e:
    print(f"Errore nella visualizzazione: {e}")


