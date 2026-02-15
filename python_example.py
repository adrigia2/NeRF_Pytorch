#!/usr/bin/env python3
"""
Esempio di utilizzo del modulo OptiX Python bindings per generare 
Inverse UV Mapping e Depth Maps.

Prerequisiti:
    - depthMapModule compilato e disponibile nel PYTHONPATH
    - File .obj del modello 3D
    - File transforms.json con le pose delle camere (per depth maps)
"""

# ..\OptixProjectCMake
from pathlib import Path
import sys
import os


REPO = Path(__file__).resolve().parents[1] / "OptixProjectCMake"
PYD_DIR = (REPO / "out" / "build" / "vcpkg-x64-debug").resolve()

# Windows + Python 3.8+: aiuta per trovare DLL dipendenti
os.add_dll_directory(str(PYD_DIR))

# Per far trovare il .pyd al sistema di import
sys.path.insert(0, str(PYD_DIR))

import depthMapModule as optix

def example_complete_pipeline():
    """Esempio completo: carica modello, genera IUM e depth maps"""
    
    # Configura i percorsi
    model_path = "C:/Users/adria/Documents/GitHub/OptixProjectCMake/Scenes/SwordShield/Models/SwordShield.obj"
    transforms_file = "C:/Users/adria/Documents/GitHub/OptixProjectCMake/Scenes/SwordShield/Nerf/transforms.json"
    
    # res folder under this repo
    ium_output_path = "./res"
    depth_output_path = "./res"
    
    # Crea le directory di output se non esistono
    os.makedirs(ium_output_path, exist_ok=True)
    os.makedirs(depth_output_path, exist_ok=True)
    
    # Crea la pipeline
    pipeline = optix.OptiXPipeline()
    
    # Esegui tutto il workflow in un colpo solo
    pipeline.process_all(
        model_path=model_path,
        transform_file=transforms_file,
        ium_output_path=ium_output_path,
        depth_output_dir=depth_output_path,
        ium_file_name="ium_result",
        image_type=optix.ImageResultType.OpenEXR,  # o ImageResultType.BMP
        ium_width=1024,
        ium_height=1024
    )
    
    print(f"Pipeline completata!")
    print(f"Vertici: {pipeline.get_vertex_count()}")
    print(f"Triangoli: {pipeline.get_triangle_count()}")


def example_step_by_step():
    """Esempio step-by-step per maggiore controllo"""
    
    # Inizializza
    pipeline = optix.OptiXPipeline()
    
    # Step 1: Carica il modello
    print("Step 1: Caricamento modello...")
    pipeline.load_model("C:/path/to/model.obj")
    
    if pipeline.is_model_loaded():
        print(f"? Modello caricato: {pipeline.get_vertex_count()} vertici")
    
    # Step 2: Genera IUM (Inverse UV Mapping)
    print("\nStep 2: Generazione IUM...")
    pipeline.generate_ium(
        output_path="C:/output/ium/",
        file_name="my_ium",
        image_type=optix.ImageResultType.OpenEXR,
        width=2048,
        height=2048
    )
    print("? IUM generato")
    
    # Step 3: Genera depth maps
    print("\nStep 3: Generazione depth maps...")
    pipeline.generate_depth_maps(
        transform_file="C:/path/to/transforms.json",
        output_dir="C:/output/depth/",
        image_type=optix.ImageResultType.OpenEXR
    )
    print("? Depth maps generate")


def example_only_ium():
    """Genera solo l'Inverse UV Mapping"""
    
    pipeline = optix.OptiXPipeline()
    
    pipeline.load_model("C:/path/to/model.obj")
    
    # Genera IUM ad alta risoluzione
    pipeline.generate_ium(
        output_path="C:/output/",
        file_name="high_res_ium",
        image_type=optix.ImageResultType.OpenEXR,
        width=4096,
        height=4096
    )
    
    print("IUM ad alta risoluzione generato!")


def example_only_depth():
    """Genera solo le depth maps"""
    
    pipeline = optix.OptiXPipeline()
    
    pipeline.load_model("C:/path/to/model.obj")
    
    # Genera solo depth maps in formato BMP
    pipeline.generate_depth_maps(
        transform_file="C:/path/to/transforms.json",
        output_dir="C:/output/depth/",
        image_type=optix.ImageResultType.BMP
    )
    
    print("Depth maps generate!")


def example_batch_processing():
    """Processa pi� modelli in batch"""
    
    models = [
        "model1.obj",
        "model2.obj",
        "model3.obj"
    ]
    
    for model_file in models:
        print(f"\n{'='*60}")
        print(f"Processing: {model_file}")
        print(f"{'='*60}")
        
        pipeline = optix.OptiXPipeline()
        
        try:
            # Carica modello
            pipeline.load_model(f"C:/models/{model_file}")
            
            # Estrai nome base
            base_name = os.path.splitext(model_file)[0]
            
            # Genera IUM
            pipeline.generate_ium(
                output_path=f"C:/output/{base_name}/ium/",
                file_name=f"{base_name}_ium",
                image_type=optix.ImageResultType.OpenEXR
            )
            
            # Genera depth maps
            pipeline.generate_depth_maps(
                transform_file=f"C:/transforms/{base_name}_transforms.json",
                output_dir=f"C:/output/{base_name}/depth/",
                image_type=optix.ImageResultType.OpenEXR
            )
            
            print(f"? {model_file} completato con successo!")
            
        except Exception as e:
            print(f"? Errore durante il processing di {model_file}: {e}")


def example_error_handling():
    """Esempio con gestione degli errori"""
    
    pipeline = optix.OptiXPipeline()
    
    try:
        # Tentativo di generare IUM senza caricare il modello
        pipeline.generate_ium("output/", "test")
        
    except RuntimeError as e:
        print(f"Errore atteso: {e}")
        print("Devo prima caricare un modello!")
    
    # Ora carica il modello e riprova
    try:
        pipeline.load_model("valid_model.obj")
        pipeline.generate_ium("output/", "test")
        print("Successo!")
        
    except Exception as e:
        print(f"Errore: {e}")


if __name__ == "__main__":
    # Configura il livello di log (opzionale)
    optix.set_log_level(2)  # 0=Error, 1=Warning, 2=Info, 3=Debug
    
    # Scegli quale esempio eseguire
    print("OptiX Pipeline - Esempi Python\n")
    
    # Decomenta l'esempio che vuoi eseguire:
    
    example_complete_pipeline()
    # example_step_by_step()
    # example_only_ium()
    # example_only_depth()
    # example_batch_processing()
    # example_error_handling()
    
    print("\nEsempio completato!")
