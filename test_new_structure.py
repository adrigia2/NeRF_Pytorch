
from pathlib import Path
import sys
import os
import json


MODEL_PATH = "C:/Users/adria/Documents/GitHub/OptixProjectCMake/Scenes/SwordShield/Models/SwordShield.obj"

# ..\OptixProjectCMake
REPO = "C:/Users/adria/Documents/GitHub/OptixProjectCMake"
PYD_DIR = "C:/Users/adria/Documents/GitHub/OptixProjectCMake/out/build/x64-debug"

# Windows + Python 3.8+: aiuta per trovare DLL dipendenti
os.add_dll_directory(str(PYD_DIR))

# Per far trovare il .pyd al sistema di import
sys.path.insert(0, str(PYD_DIR))

import depthMapModule as optix

model = optix.TriangleMesh()
model.add_from_obj_file(MODEL_PATH)

print("✓ Modello caricato con successo")

# Create and configure DepthGenerator
depth_gen = optix.DepthGenerator()
depth_gen.set_traversable(model)

camera = optix.Camera()
pos = optix.vec3f(1.0, 2.0, 3.0)
forward = optix.vec3f(0.0, 0.0, -1.0)
up = optix.vec3f(0.0, 1.0, 0)

camera.pos = pos
camera.forward = forward
camera.up = up

depth_gen.set_camera(
    camera,
    fovY=45.0,
    frameSize=(512, 512)
)
depth_gen.need_render_depth(True)
depth_gen.need_render_position(True)
depth_gen.need_render_normal(True)

# Render and get results
depth_gen.render()
depth_result = depth_gen.get_result()

print("✓ Depth rendering completato")
print(f"  - Has depth: {depth_result.has_depth_data()}")
print(f"  - Has positions: {depth_result.has_positional_data()}")
print(f"  - Has normals: {depth_result.has_normal_data()}")

# Access NumPy arrays
if depth_result.has_depth_data():
    depths = depth_result.depths_np
    print(f"  - Depth shape: {depths.shape}")

if depth_result.has_positional_data():
    positions = depth_result.positions_np
    print(f"  - Positions shape: {positions.shape}")

# Create and configure IUMGenerator
ium_gen = optix.IUMGenerator()
ium_gen.set_traversable(model)
ium_gen.set_texture_size(1024)
ium_gen.render()
ium_result = ium_gen.get_result()

print("✓ IUM rendering completato")
if ium_result.has_positions():
    ium_positions = ium_result.positions_np
    print(f"  - IUM Positions shape: {ium_positions.shape}")