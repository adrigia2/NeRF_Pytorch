"""
nerf_render_pipeline.py
-----------------------
Legge un file transforms.json (formato NeRF), renderizza per ogni frame
depth / position / normal / mask tramite OptixProgrammablePasses e genera la IUM.
Salva ogni output nel formato scelto (openexr | png); quando il formato
non supporta dati raw (float), normalizza automaticamente i valori in [0,1].
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Protocol

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Enumerazioni
# ──────────────────────────────────────────────────────────────────────────────

class ImageFormat(Enum):
    OPENEXR = "openexr"
    PNG     = "png"

    @property
    def extension(self) -> str:
        return {"openexr": ".exr", "png": ".png"}[self.value]

    @property
    def supports_raw_float(self) -> bool:
        """True se il formato può salvare float a 32 bit senza perdita."""
        return self in {ImageFormat.OPENEXR}


class DataLayer(Enum):
    """Tipo semantico del layer — determina se serve normalizzazione."""
    DEPTH    = auto()   # valori in unità-scena → raw float
    POSITION = auto()   # coordinate world-space → raw float
    NORMAL   = auto()   # vettori [-1,1] → NO raw richiesto ma float ok
    MASK     = auto()   # uint8 0/1 → non raw, nessuna normalizzazione float
    VISIBILITY = auto() # ratio float [0, 1] o bool uint8
    IRRADIANCE          = auto() # energia HDR per texel (RGB float)
    IRRADIANCE_INDIRECT = auto() # contributo indiretto NeRF per texel (RGB float)
    ALBEDO              = auto() # riflettanza HDR per texel (RGB float)


# ──────────────────────────────────────────────────────────────────────────────
# Protocollo writer (estensibile senza modificare il core)
# ──────────────────────────────────────────────────────────────────────────────

class ImageWriter(Protocol):
    """Interfaccia che ogni writer deve implementare."""
    def write(self, array: np.ndarray, path: str) -> None: ...


# ──────────────────────────────────────────────────────────────────────────────
# Writer: OpenEXR
# ──────────────────────────────────────────────────────────────────────────────

class ExrWriter:
    """Scrive array NumPy float32 in un file OpenEXR.

    Shape supportate:
      (H, W)      → canale singolo  'Z'
      (H, W, 3)   → canali RGB      'R','G','B'
      (H, W, 4)   → canali RGBA     'R','G','B','A'
      (H, W, C)   → arbitrary       'Cam0', 'Cam1', ...
    """

    def write(self, array: np.ndarray, path: str) -> None:
        import OpenEXR, Imath  # importazione locale — non obbligatori se non si usa EXR

        array = array.astype(np.float32)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        if array.ndim == 2:
            h, w = array.shape
            header = OpenEXR.Header(w, h)
            header["channels"] = {"Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
            f = OpenEXR.OutputFile(path, header)
            f.writePixels({"Z": array.tobytes()})
        elif array.ndim == 3:
            h, w, c = array.shape
            if c == 1:
                names = ["Z"]
            elif c == 3:
                names = ["R", "G", "B"]
            elif c == 4:
                names = ["R", "G", "B", "A"]
            else:
                names = [f"Cam{i}" for i in range(c)] # N-channel mapping
                
            header = OpenEXR.Header(w, h)
            header["channels"] = {
                n: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)) for n in names
            }
            f = OpenEXR.OutputFile(path, header)
            f.writePixels({n: array[..., i].tobytes() for i, n in enumerate(names)})
        else:
            raise ValueError(f"ExrWriter: ndim={array.ndim} non supportato")

        f.close()


# ──────────────────────────────────────────────────────────────────────────────
# Writer: PNG
# ──────────────────────────────────────────────────────────────────────────────

class PngWriter:
    """Scrive array NumPy in PNG (uint8). I float vengono normalizzati in [0,255]."""

    def write(self, array: np.ndarray, path: str) -> None:
        from PIL import Image

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        uint8 = _to_uint8(array)
        Image.fromarray(uint8).save(path)


# ──────────────────────────────────────────────────────────────────────────────
# Registry writer (aperto all'estensione)
# ──────────────────────────────────────────────────────────────────────────────

_WRITER_REGISTRY: dict[ImageFormat, ImageWriter] = {
    ImageFormat.OPENEXR: ExrWriter(),
    ImageFormat.PNG:     PngWriter(),
}


def register_writer(fmt: ImageFormat, writer: ImageWriter) -> None:
    """Registra un writer per un formato personalizzato."""
    _WRITER_REGISTRY[fmt] = writer


def get_writer(fmt: ImageFormat) -> ImageWriter:
    if fmt not in _WRITER_REGISTRY:
        raise NotImplementedError(f"Nessun writer registrato per il formato: {fmt}")
    return _WRITER_REGISTRY[fmt]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_uint8(array: np.ndarray) -> np.ndarray:
    """Normalizza un array float in [0, 255] uint8 (per PNG, ecc.)."""
    arr = array.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return arr


def _reshape_flat(array: np.ndarray, w: int, h: int) -> np.ndarray:
    """Porta un array flat restituito dalla libreria Optix nella shape (H, W) o (H, W, C).

    La libreria restituisce sempre array flat:
      depths_np    → (N,)    con N = W*H
      positions_np → (N, 3)  con N = W*H
      normals_np   → (N, 3)  con N = W*H
      masks_np     → (N,)    con N = W*H  (uint8)

    Casi gestiti:
      (H, W)      → già nella forma corretta, ritorna invariato
      (H, W, C)   → già nella forma corretta, ritorna invariato
      (N,)        → N == W*H  → reshape a (H, W)
      (N, C)      → N == W*H  → reshape a (H, W, C)
    """
    pixels = w * h

    # Già nella forma corretta
    if array.ndim == 2 and array.shape == (h, w):
        return array
    if array.ndim == 3 and array.shape[:2] == (h, w):
        return array

    # (N,) flat moncanale
    if array.ndim == 1:
        if array.size != pixels:
            raise ValueError(
                f"_reshape_flat: size={array.size} non corrisponde a {w}×{h}={pixels}"
            )
        return array.reshape(h, w)

    # (N, C) flat multicanale  — shape restituita da positions_np e normals_np
    if array.ndim == 2 and array.shape[0] == pixels:
        c = array.shape[1]
        return array.reshape(h, w, c)

    raise ValueError(
        f"_reshape_flat: shape={array.shape} non gestita per dimensioni {w}×{h}"
    )


def _save_layer(
    array: np.ndarray,
    path: str,
    fmt: ImageFormat,
    layer: DataLayer,
) -> None:
    """Salva un layer nel formato richiesto, normalizzando se necessario.

    Logica di normalizzazione:
      - DEPTH / POSITION  → valori raw float; se il formato non supporta float
                            vengono normalizzati in [0, 1] prima di passare al writer.
      - NORMAL            → vettori in [-1, 1]; stessa regola del raw float.
      - MASK              → già uint8 0/1; nessuna normalizzazione applicata.
    """
    needs_raw = layer in {DataLayer.DEPTH, DataLayer.POSITION, DataLayer.NORMAL,
                          DataLayer.IRRADIANCE, DataLayer.IRRADIANCE_INDIRECT, DataLayer.ALBEDO}

    if needs_raw and not fmt.supports_raw_float:
        print(f"    ⚠  {fmt.value} non supporta float raw ({layer.name}) → normalizzo in [0,1]")
        array = array.astype(np.float32)
        mn, mx = array.min(), array.max()
        if mx > mn:
            array = (array - mn) / (mx - mn)

    get_writer(fmt).write(array, path)
    print(f"    ✓ Salvato: {path}  shape={array.shape}")


def _build_output_path(base_dir: str, stem: str, layer_name: str, fmt: ImageFormat) -> str:
    return (Path(base_dir) / layer_name / f"{stem}_{layer_name}{fmt.extension}").resolve().as_posix()


def _as_relative_to(abs_path: str, base_dir: str) -> str:
    """Restituisce abs_path come path relativo rispetto a base_dir in formato posix.
    Se non è sotto base_dir, ritorna il path posix invariato."""
    try:
        return Path(abs_path).relative_to(Path(base_dir)).as_posix()
    except ValueError:
        return Path(abs_path).as_posix()


def _save_debug_comparison(
    src_img_path,
    cam_arr: np.ndarray,   # (H, W, 3) float32 in [0,1], camera_texture
    frame_stem: str,
    out_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    h, w = cam_arr.shape[:2]
    src = _load_image_as_vec3(str(src_img_path), w, h).reshape(h, w, 3)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(np.clip(src, 0, 1))
    axes[0].set_title(f"Camera image\n{frame_stem}")
    axes[0].axis("off")

    axes[1].imshow(np.clip(cam_arr, 0, 1))
    axes[1].set_title(f"Camera texture (UV atlas)\n{frame_stem}")
    axes[1].axis("off")

    fig.tight_layout()
    out_path = out_dir / f"{frame_stem}.png"
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def _save_debug_pixel_change(
    min_arr: np.ndarray,    # (H, W, 3) float32
    max_arr: np.ndarray,    # (H, W, 3) float32
    range_arr: np.ndarray,  # (H, W, 3) float32
    out_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # Normalizza ogni immagine in [0,1] per la visualizzazione
    def _norm(arr: np.ndarray) -> np.ndarray:
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            return (arr - mn) / (mx - mn)
        return np.clip(arr, 0.0, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    axes[0].imshow(_norm(min_arr))
    axes[0].set_title("color_min\n(darkest sample per texel)")
    axes[0].axis("off")

    axes[1].imshow(_norm(max_arr))
    axes[1].set_title("color_max\n(brightest sample per texel)")
    axes[1].axis("off")

    axes[2].imshow(_norm(range_arr))
    axes[2].set_title("color_range  (max − min)\n(variation map)")
    axes[2].axis("off")

    fig.tight_layout()
    out_path = out_dir / "pixel_change_comparison.png"
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"    ✓ Debug pixel_change salvato: {out_path}")


def _compute_peak(image_np: np.ndarray, percentile: float) -> float:
    """Calcola il peak dell'immagine come percentile della luminanza massima per pixel."""
    max_per_pixel = image_np.astype(np.float32).max(axis=-1)  # (H, W)
    return float(np.percentile(max_per_pixel, percentile))


def _load_image_as_vec3(path: str, w: int, h: int) -> np.ndarray:
    """Carica immagine, ridimensiona a (w, h), ritorna float32 (H*W, 3)."""
    if path.lower().endswith(".exr"):
        import OpenEXR, Imath
        from PIL import Image
        exr = OpenEXR.InputFile(path)
        dw = exr.header()["dataWindow"]
        src_w = dw.max.x - dw.min.x + 1
        src_h = dw.max.y - dw.min.y + 1
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        chs = exr.header()["channels"]
        if "R" in chs and "G" in chs and "B" in chs:
            r = np.frombuffer(exr.channel("R", pt), dtype=np.float32).reshape(src_h, src_w)
            g = np.frombuffer(exr.channel("G", pt), dtype=np.float32).reshape(src_h, src_w)
            b = np.frombuffer(exr.channel("B", pt), dtype=np.float32).reshape(src_h, src_w)
        else:
            key = next(iter(chs))
            ch = np.frombuffer(exr.channel(key, pt), dtype=np.float32).reshape(src_h, src_w)
            r = g = b = ch
        arr = np.stack([r, g, b], axis=-1)  # (src_h, src_w, 3) float32 HDR
        if src_w != w or src_h != h:
            def _resize_ch(c):
                return np.array(Image.fromarray(c).resize((w, h), Image.LANCZOS))
            arr = np.stack([_resize_ch(arr[..., i]) for i in range(3)], axis=-1)
        return arr.reshape(-1, 3)
    else:
        from PIL import Image
        img = Image.open(path).convert("RGB").resize((w, h), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr.reshape(-1, 3)


def _load_image_hw3_native(path: str) -> np.ndarray:
    """Carica un'immagine a risoluzione nativa come float32 (H, W, 3).
    EXR → valori HDR raw. LDR (PNG/JPG) → uint8/255 scalato in [0, 1]."""
    if path.lower().endswith(".exr"):
        import OpenEXR, Imath
        exr = OpenEXR.InputFile(path)
        dw = exr.header()["dataWindow"]
        src_w = dw.max.x - dw.min.x + 1
        src_h = dw.max.y - dw.min.y + 1
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        chs = exr.header()["channels"]
        if "R" in chs and "G" in chs and "B" in chs:
            r = np.frombuffer(exr.channel("R", pt), dtype=np.float32).reshape(src_h, src_w)
            g = np.frombuffer(exr.channel("G", pt), dtype=np.float32).reshape(src_h, src_w)
            b = np.frombuffer(exr.channel("B", pt), dtype=np.float32).reshape(src_h, src_w)
        else:
            key = next(iter(chs))
            ch = np.frombuffer(exr.channel(key, pt), dtype=np.float32).reshape(src_h, src_w)
            r = g = b = ch
        return np.stack([r, g, b], axis=-1)
    else:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        return np.array(img, dtype=np.float32) / 255.0


def _load_exr_as_flat(path: str) -> np.ndarray | None:
    """Carica un EXR RGB a dimensione nativa e restituisce (H*W, 3) float32."""
    try:
        import OpenEXR, Imath
        exr = OpenEXR.InputFile(path)
        dw = exr.header()["dataWindow"]
        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        r = np.frombuffer(exr.channel("R", pt), dtype=np.float32).reshape(h, w)
        g = np.frombuffer(exr.channel("G", pt), dtype=np.float32).reshape(h, w)
        b = np.frombuffer(exr.channel("B", pt), dtype=np.float32).reshape(h, w)
        return np.stack([r, g, b], axis=-1).reshape(-1, 3)
    except Exception as e:
        print(f"    ⚠  Impossibile caricare {path}: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Parsing transforms.json
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CameraIntrinsics:
    w: int
    h: int
    fl_x: float
    fl_y: float
    cx: float
    cy: float
    camera_angle_x: float
    camera_angle_y: float


@dataclass
class FrameInfo:
    file_path: str          # path assoluto risolto (per accedere al file su disco)
    file_path_original: str # path così com'è nel JSON originale
    transform_matrix: list[list[float]]
    sharpness: float = 1.0

    @property
    def stem(self) -> str:
        return Path(self.file_path).stem


@dataclass
class TransformsFile:
    intrinsics: CameraIntrinsics
    frames: list[FrameInfo]
    transforms_dir: str     # cartella del transforms.json originale
    scale: float = 1.0
    aabb_scale: int = 16
    raw: dict = field(default_factory=dict)


def load_transforms(path: str) -> TransformsFile:
    transforms_dir = Path(path).parent.resolve()

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    intr = CameraIntrinsics(
        w=data["w"],
        h=data["h"],
        fl_x=data["fl_x"],
        fl_y=data["fl_y"],
        cx=data["cx"],
        cy=data["cy"],
        camera_angle_x=data["camera_angle_x"],
        camera_angle_y=data["camera_angle_y"],
    )

    def _resolve_path(file_path: str) -> str:
        p = Path(file_path)
        if p.is_absolute():
            return p.as_posix()
        return (transforms_dir / p).resolve().as_posix()

    frames = [
        FrameInfo(
            file_path=_resolve_path(f["file_path"]),
            file_path_original=f["file_path"],
            transform_matrix=f["transform_matrix"],
            sharpness=f.get("sharpness", 1.0),
        )
        for f in data["frames"]
    ]
    return TransformsFile(
        intrinsics=intr,
        frames=frames,
        transforms_dir=transforms_dir.as_posix(),
        scale=data.get("scale", 1.0),
        aabb_scale=data.get("aabb_scale", 16),
        raw=data,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Estrazione camera da transform_matrix (convenzione NeRF / OpenCV)
# ──────────────────────────────────────────────────────────────────────────────

def _camera_from_matrix(matrix: list[list[float]], fovy: float, frame_size: list[int], optix_mod) -> object:
    """Ricava posizione, forward e up dalla transform_matrix NeRF 4×4.

    Convenzione colonne della matrice c2w:
      col 0 → right
      col 1 → up
      col 2 → -forward  (NeRF punta la camera lungo -Z)
      col 3 → position
    """
    m = matrix
    pos     = [m[0][3], m[1][3], m[2][3]]
    forward = [-m[0][2], -m[1][2], -m[2][2]]
    up      = [m[0][1],  m[1][1],  m[2][1]]
    return optix_mod.Camera(pos, forward, up, fovy, frame_size)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline principale
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RenderConfig:
    transforms_path: str
    model_path: str
    output_dir: str

    # Normalizzazione HDR delle immagini sorgente (Step 1).
    # Divisore = skybox.max() se skybox_path è impostato, altrimenti max sulle immagini sorgente.
    # Lo skybox normalizzato viene salvato come skybox_normalized.exr e usato in Step 3.
    normalize_images: bool = False

    # Cosa renderizzare
    render_depth:    bool = True
    render_position: bool = True
    render_normal:   bool = True
    render_mask     : bool = True   # mask di validità per ogni frame
    render_ium      : bool = True
    render_visibility: bool = True

    # Formato di salvataggio per ogni layer
    depth_format:    ImageFormat = ImageFormat.OPENEXR
    position_format: ImageFormat = ImageFormat.OPENEXR
    normal_format:   ImageFormat = ImageFormat.OPENEXR
    mask_format:     ImageFormat = ImageFormat.PNG      # uint8 → PNG è naturale
    ium_format:      ImageFormat = ImageFormat.PNG
    visibility_format: ImageFormat = ImageFormat.OPENEXR

    # Dimensione texture IUM [width, height]
    ium_texture_size: list[int] = field(default_factory=lambda: [512, 512])

    # Scala applicata ai depth — deve essere False: scale nei transforms.json
    # è da applicare ANCHE alle traslazioni della camera, e non farlo crea un
    # mismatch (query points NeRF ≠ superficie mesh).  Lasciare False.
    apply_scale: bool = False

    # Color texture
    render_color_texture: bool = False
    color_texture_format: ImageFormat = ImageFormat.OPENEXR
    # Percentile usato per calcolare il peak (default 95° = scarta il top 5% più luminoso)
    color_texture_peak_percentile: float = 100.0

    # Debug output
    debug_camera_texture: bool = False   # salva side-by-side camera image vs camera_texture

    # Pixel change output
    render_pixel_change: bool = False    # salva min/max/range texture in pixel_change/
    debug_pixel_change: bool = False     # salva plot comparativo in debug_pixel_change/

    # Irradiance map (Monte Carlo skybox lighting per-texel)
    render_irradiance: bool = False
    irradiance_format: ImageFormat = ImageFormat.OPENEXR
    skybox_path: str = ""                # path al file EXR equirettangolare
    skybox_size: list[int] = field(default_factory=lambda: [1024, 512])  # resize target
    irradiance_sample_side: int = 16     # N → N×N campioni per emisfero (16 = 256, 256 = 65536)
    skybox_yaw_degrees: float = 0.0      # rotazione yaw skybox; 0° = -Y (Blender fwd) al centro

    # Indirect irradiance via NeRF (precompute once, cache on disk)
    precompute_indirect: bool = False
    indirect_sample_side: int = 64       # N → N×N campioni per texel (separato da irradiance)
    indirect_tile_size: int = 1024       # texel per tile GPU (bilancia memoria/VRAM)
    indirect_nerf_cache_path: str = ""   # path al checkpoint NeRF (default: auto-detect)
    indirect_format: ImageFormat = ImageFormat.OPENEXR
    # Depth hints: se True concentra i sample attorno al t_hit OptiX invece di stratified puro
    indirect_depth_hint_enabled: bool = False
    indirect_depth_window: float = 0.5
    indirect_depth_window_end: float = 0.0

    # Albedo (color_texture / irradiance) — modello Lambertiano ρ = π · L / E
    render_albedo: bool = False
    albedo_format: ImageFormat = ImageFormat.OPENEXR
    albedo_eps: float = 1e-3             # clamp minimo dell'irradiance per evitare /0


@dataclass
class PipelineConfig:
    """Orchestratore a tre step toggle-abili.

    Step 1: genera depth+mask+immagini+transforms_extended.json (minimo per NeRF).
    Step 2: allena il NeRF (nerf/train.py) e salva il checkpoint.
    Step 3: esegue IUM/visibility/color_texture/irradiance/indirect/albedo.
    """
    run_step1: bool = True
    run_step2: bool = True
    run_step3: bool = True

    render: RenderConfig = field(default_factory=RenderConfig)

    # Parametri di nerf/train.py (Step 2)
    nerf_num_iters:        int   = 10000
    nerf_batch_size:       int   = 4096
    nerf_lr:               float = 5e-4
    nerf_display_every:    int   = 100
    nerf_seed:             int   = 9458
    nerf_ckpt_path:        str   = ""  # default: <output_dir>/model/nerf_model_cache.pt
    nerf_train_output_dir: str   = ""  # default: <output_dir>/nerf_train

    # Depth-hint training (Step 2) — attivo solo se run_step1 ha generato depth+mask
    nerf_depth_hint_enabled:   bool  = False
    nerf_foreground_ratio:         float = 0.8   # frazione del batch dai raggi figura
    nerf_foreground_ratio_enabled: bool  = True  # False = ratio naturale del dataset (ignora la percentuale)
    nerf_depth_window_samples: int   = 32    # sample nella finestra per i raggi figura
    nerf_depth_window:         float = 0.5   # [t_hit - window, t_hit + window_end]
    nerf_depth_window_end:     float = 0.5
    nerf_opacity_weight:       float = 1.0   # peso della loss acc_fg→1 (0 = disattiva)
    nerf_raw_noise_std:        float = 0.0   # rumore pre-ReLU sulla densità (1.0 = training robusto)
    nerf_train_background:     bool  = True  # allena sky MLP sui pixel bg; False = bg bianco classico

    # Render dei frame di training col NeRF allenato (post-Step 2)
    enable_nerf_render_train_images: bool = False
    nerf_render_train_images_dir:    str  = ""  # default: <output_dir>/nerf_render_images

    # Se True, chiede all'utente di continuare il training al termine di ogni round
    nerf_interactive_loop: bool = True


def _precompute_indirect_irradiance(
    cfg: RenderConfig,
    ium_res,        # IUM_Generator.Result
    model,          # OptixProgrammablePasses.TriangleMesh
    ium_w: int,
    ium_h: int,
    indirect_path: str,
) -> None:
    """Esegue il pass OptiX tile-by-tile, interroga NeRF per ogni raggio occluso e
    salva irradiance_indirect.exr su disco.  Chiamato solo se precompute_indirect=True.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from nerf import load_checkpoint, query_radiance
    import OptixProgrammablePasses as optix

    # ── Carica il modello NeRF dal cache ──────────────────────────────────────
    cache_path = cfg.indirect_nerf_cache_path
    if not cache_path:
        cache_path = os.path.join(cfg.output_dir, "model", "nerf_model_cache.pt")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"NeRF model cache non trovato: {cache_path}\n"
            "Imposta indirect_nerf_cache_path oppure eseguire prima Step 2."
        )

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_bundle, nerf_cfg = load_checkpoint(cache_path, device)
    if cfg.indirect_depth_hint_enabled:
        nerf_cfg.depth_hint_enabled = True
        nerf_cfg.depth_window       = cfg.indirect_depth_window
        nerf_cfg.depth_window_end   = cfg.indirect_depth_window_end
    print(f"✓ NeRF model caricato da: {cache_path}")

    # ── Pass OptiX tile-by-tile ───────────────────────────────────────────────
    ind_gen = optix.IndirectGenerator()
    ind_gen.set_traversable(model)
    ind_gen.set_inputs(ium_res, cfg.indirect_sample_side, cfg.indirect_tile_size)

    n_tiles = ind_gen.num_tiles()
    num_pix = ind_gen.num_pixels()
    scale   = (2.0 * np.pi) / (cfg.indirect_sample_side ** 2)

    irr_indirect     = np.zeros((num_pix, 3), dtype=np.float64)
    ium_positions_np = ium_res.positions_np.astype(np.float32)
    ium_normals_np   = ium_res.normals_np.astype(np.float32)
    eps              = 1e-4

    print(f"  Indirect precompute: {n_tiles} tile × {cfg.indirect_tile_size} texel, "
          f"N={cfg.indirect_sample_side}")

    for tile_idx in range(n_tiles):
        tile_res = ind_gen.render_tile(tile_idx)
        count    = tile_res.count
        if count == 0:
            continue

        local_idx = tile_res.local_idx_np.copy()
        dirs_np   = tile_res.directions_np.copy()
        cos_np    = tile_res.cos_np.copy()
        t_hit_np  = tile_res.t_hit_np.copy()

        tile_offset = tile_idx * cfg.indirect_tile_size
        global_idx  = tile_offset + local_idx

        origins_np = (ium_positions_np[global_idx]
                      + ium_normals_np[global_idx] * eps)

        colors = query_radiance(model_bundle, origins_np, dirs_np, nerf_cfg, t_hits_np=t_hit_np)

        np.add.at(irr_indirect, global_idx,
                  colors * cos_np[:, None].astype(np.float64))

        if (tile_idx + 1) % max(1, n_tiles // 10) == 0:
            print(f"    tile {tile_idx+1}/{n_tiles}, raggi occlusi: {count}")

    irr_indirect = (irr_indirect * scale).astype(np.float32)

    os.makedirs(os.path.dirname(indirect_path), exist_ok=True)
    irr_indirect_arr = _reshape_flat(irr_indirect, ium_w, ium_h)
    _save_layer(irr_indirect_arr, indirect_path, cfg.indirect_format,
                DataLayer.IRRADIANCE_INDIRECT)
    print(f"✓ irradiance_indirect salvato: {indirect_path}")


def _step1_pretrain_data(cfg: PipelineConfig, optix_mod) -> Path:
    """Renderizza depth + mask per ogni frame, copia le immagini RGB e scrive
    transforms_extended.json con i campi minimi richiesti da NerfDataset.
    """
    rc = cfg.render
    tf = load_transforms(rc.transforms_path)
    intr = tf.intrinsics
    print(f"[Step 1] Trasformazioni caricate: {len(tf.frames)} frame  [{intr.w}×{intr.h}]")

    json_dir = Path(rc.output_dir).resolve()
    json_dir_str = json_dir.as_posix()
    os.makedirs(json_dir, exist_ok=True)

    model = optix_mod.TriangleMesh()
    model.add_from_obj_file(rc.model_path)
    print(f"[Step 1] Modello caricato: {rc.model_path}")

    depth_gen = optix_mod.DepthGenerator()
    depth_gen.set_traversable(model)
    depth_gen.need_render_depth(True)
    depth_gen.need_render_position(False)
    depth_gen.need_render_normal(False)

    images_out_dir = json_dir / "images"
    os.makedirs(images_out_dir, exist_ok=True)

    output_frames = []
    scale = tf.scale if rc.apply_scale else 1.0
    W, H = intr.w, intr.h

    # ── Calcolo divisore di normalizzazione HDR ───────────────────────────────
    norm_divisor: float | None = None
    norm_source: str | None = None
    sky_norm_path: str | None = None  # path allo skybox normalizzato salvato
    if rc.normalize_images:
        if rc.skybox_path:
            sky_raw = _load_image_hw3_native(rc.skybox_path)
            norm_divisor = float(sky_raw.max())
            norm_source = "skybox"
            print(f"[Step 1] Normalizzazione: skybox → max={norm_divisor:.6f}")
            # Salva skybox normalizzato — usato da Step 3 per coerenza radiometrica
            sky_normalized = (sky_raw / norm_divisor).astype(np.float32)
            sky_norm_path = (json_dir / "skybox_normalized.exr").as_posix()
            get_writer(ImageFormat.OPENEXR).write(sky_normalized, sky_norm_path)
            sky_norm_path = _as_relative_to(sky_norm_path, json_dir_str)
            print(f"[Step 1] Skybox normalizzata salvata: {(json_dir / sky_norm_path).resolve()}")
        else:
            running_max = 0.0
            print("[Step 1] Normalizzazione: scansione immagini per trovare il max globale…")
            for frame in tf.frames:
                src = Path(frame.file_path)
                if not src.exists():
                    continue
                arr = _load_image_hw3_native(str(src))
                running_max = max(running_max, float(arr.max()))
            norm_divisor = running_max
            norm_source = "images"
            print(f"[Step 1] Normalizzazione: immagini → max={norm_divisor:.6f}")
        if norm_divisor <= 0:
            print("[Step 1] ⚠  max = 0, normalizzazione disabilitata per questa run.")
            norm_divisor = None

    for idx, frame in enumerate(tf.frames):
        print(f"\n[Step 1] Frame {idx + 1}/{len(tf.frames)}: {frame.stem}")

        src_image = Path(frame.file_path)
        if norm_divisor is not None:
            dst_image = images_out_dir / (src_image.stem + ".exr")
            if src_image.exists():
                arr = _load_image_hw3_native(str(src_image)) / norm_divisor
                get_writer(ImageFormat.OPENEXR).write(arr.astype(np.float32), str(dst_image))
            else:
                print(f"    ⚠  Immagine non trovata, skip: {src_image}")
        else:
            dst_image = images_out_dir / src_image.name
            if src_image.exists():
                shutil.copy2(src_image, dst_image)
            else:
                print(f"    ⚠  Immagine non trovata, skip copia: {src_image}")

        camera = _camera_from_matrix(frame.transform_matrix, intr.camera_angle_y, [W, H], optix_mod)
        depth_gen.set_camera(camera)
        depth_gen.render()
        result = depth_gen.get_result()

        frame_entry: dict = {
            "file_path":        _as_relative_to(dst_image.as_posix(), json_dir_str),
            "sharpness":        frame.sharpness,
            "transform_matrix": frame.transform_matrix,
        }

        if result.has_depth_data():
            depth_arr = _reshape_flat(result.depths_np.astype(np.float32), W, H)
            if rc.apply_scale:
                depth_arr = depth_arr * scale
            out_path = _build_output_path(rc.output_dir, frame.stem, "depth", rc.depth_format)
            _save_layer(depth_arr, out_path, rc.depth_format, DataLayer.DEPTH)
            frame_entry["depth_path"] = _as_relative_to(out_path, json_dir_str)

        mask_arr = _reshape_flat(result.masks_np, W, H)
        out_path = _build_output_path(rc.output_dir, frame.stem, "mask", rc.mask_format)
        _save_layer(mask_arr, out_path, rc.mask_format, DataLayer.MASK)
        frame_entry["mask_path"] = _as_relative_to(out_path, json_dir_str)

        output_frames.append(frame_entry)

    output_json: dict = {**tf.raw}
    output_json["fl_x"] = float(intr.fl_x)
    output_json["fl_y"] = float(intr.fl_y)
    output_json["h"] = H
    output_json["w"] = W
    output_json["frames"] = output_frames

    if norm_divisor is not None:
        output_json["normalization"] = {
            "max": float(norm_divisor),
            "source": norm_source,
            "skybox_path": rc.skybox_path or None,
            "normalized_skybox_path": sky_norm_path,
        }

    out_json_path = json_dir / "transforms_extended.json"
    with open(out_json_path, "w", encoding="utf-8") as fh:
        json.dump(output_json, fh, indent=4)
    print(f"\n[Step 1] JSON minimo salvato in: {out_json_path}")
    return out_json_path


def _step2_train_nerf(cfg: PipelineConfig, transforms_extended_path: Path) -> Path:
    """Allena il NeRF (vanilla bmild coarse+fine) e restituisce il path del checkpoint."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from nerf import NerfConfig, train as nerf_train

    rc   = cfg.render
    ckpt = Path(cfg.nerf_ckpt_path or
                Path(rc.output_dir) / "model" / "nerf_model_cache.pt")
    out_dir = Path(cfg.nerf_train_output_dir or Path(rc.output_dir) / "nerf_train")

    nerf_cfg = NerfConfig(
        depth_hint_enabled        = cfg.nerf_depth_hint_enabled,
        foreground_ratio          = cfg.nerf_foreground_ratio,
        foreground_ratio_enabled  = cfg.nerf_foreground_ratio_enabled,
        depth_window_samples      = cfg.nerf_depth_window_samples,
        depth_window              = cfg.nerf_depth_window,
        depth_window_end          = cfg.nerf_depth_window_end,
        opacity_weight            = cfg.nerf_opacity_weight,
        raw_noise_std             = cfg.nerf_raw_noise_std,
        train_background          = cfg.nerf_train_background,
    )

    print(f"[Step 2] Training NeRF — {cfg.nerf_num_iters} iter, ckpt → {ckpt}")
    if cfg.nerf_depth_hint_enabled:
        ratio_str = (f"fg_ratio={cfg.nerf_foreground_ratio}"
                     if cfg.nerf_foreground_ratio_enabled else "fg_ratio=natural")
        print(f"[Step 2] Depth-hint mode: {ratio_str}, "
              f"window_samples={cfg.nerf_depth_window_samples}, "
              f"window=[t_hit-{cfg.nerf_depth_window}, t_hit+{cfg.nerf_depth_window_end}], "
              f"train_background={cfg.nerf_train_background}")
    nerf_train(
        str(transforms_extended_path), nerf_cfg,
        ckpt_path     = str(ckpt),
        output_dir    = str(out_dir),
        num_iters     = cfg.nerf_num_iters,
        batch_size    = cfg.nerf_batch_size,
        lr            = cfg.nerf_lr,
        seed          = cfg.nerf_seed,
        display_every = cfg.nerf_display_every,
    )
    print(f"[Step 2] Training completato. Checkpoint: {ckpt}")
    return ckpt


def _write_png_float(arr: np.ndarray, path: str) -> None:
    """Salva array float32 [0,1] come PNG uint8."""
    from PIL import Image
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(path)


def _write_sxs_comparison(gt_np: np.ndarray, pred_np: np.ndarray,
                           psnr: float, path: str, label: str = "") -> None:
    """Salva side-by-side GT | Pred con PSNR in testa come PNG."""
    from PIL import Image, ImageDraw
    H, W   = gt_np.shape[:2]
    title_h = 24
    canvas  = np.zeros((H + title_h, W * 2, 3), dtype=np.uint8)
    canvas[title_h:, :W] = (np.clip(gt_np,   0, 1) * 255).astype(np.uint8)
    canvas[title_h:, W:] = (np.clip(pred_np,  0, 1) * 255).astype(np.uint8)
    img  = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    draw.text((W // 4,       4), f"GT  {label}",              fill=(220, 220, 220))
    draw.text((W + W // 4,   4), f"Pred  PSNR={psnr:.2f} dB", fill=(220, 220, 220))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def _load_depth_np(path: str) -> np.ndarray | None:
    """Carica depth EXR come (H, W) float32; ritorna None in caso di errore."""
    try:
        import OpenEXR, Imath
        exr = OpenEXR.InputFile(path)
        dw  = exr.header()["dataWindow"]
        w   = dw.max.x - dw.min.x + 1
        h   = dw.max.y - dw.min.y + 1
        pt  = Imath.PixelType(Imath.PixelType.FLOAT)
        key = next(iter(exr.header()["channels"]))
        ch  = np.frombuffer(exr.channel(key, pt), dtype=np.float32).reshape(h, w)
        return np.where(ch >= 1e10, 0.0, ch).astype(np.float32)
    except Exception as e:
        print(f"    ⚠  Impossibile caricare depth {path}: {e}")
        return None


def _step2b_render_train_images(cfg: PipelineConfig,
                                transforms_extended_path: Path,
                                ckpt_path: Path) -> None:
    """Per ogni frame renderizza col NeRF e salva GT, Pred e GT|Pred sxs come PNG."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from nerf import load_checkpoint, render_image as nerf_render_image

    rc = cfg.render

    model_bundle, nerf_cfg = load_checkpoint(str(ckpt_path))

    tf   = load_transforms(str(transforms_extended_path))
    intr = tf.intrinsics

    with open(transforms_extended_path, encoding="utf-8") as fh:
        raw_frames = json.load(fh)["frames"]

    base = Path(cfg.nerf_render_train_images_dir or
                Path(rc.output_dir) / "nerf_render_images")
    base.mkdir(parents=True, exist_ok=True)

    print(f"\n[Step 2b] Rendering {len(tf.frames)} frame con NeRF...")
    psnrs = []
    for i, (frame, raw_frame) in enumerate(zip(tf.frames, raw_frames)):
        gt_np  = _load_image_hw3_native(frame.file_path)
        pose   = np.array(frame.transform_matrix, dtype=np.float32)

        dep_np = None
        if nerf_cfg.depth_hint_enabled:
            dep_str = raw_frame.get("depth_path", "")
            if dep_str:
                dep_full = (dep_str if Path(dep_str).is_absolute()
                            else (Path(transforms_extended_path).parent / dep_str).resolve().as_posix())
                dep_np = _load_depth_np(dep_full)

        pred_np = nerf_render_image(
            model_bundle, intr.h, intr.w, intr.fl_x, pose, nerf_cfg,
            focal_y=intr.fl_y, cx=intr.cx, cy=intr.cy, target_depth=dep_np,
        )

        psnr = -10.0 * np.log10(np.mean((pred_np - gt_np) ** 2) + 1e-10)
        psnrs.append(psnr)

        stem = f"frame_{i:03d}"
        _write_png_float(gt_np,   (base / f"{stem}_gt.png").as_posix())
        _write_png_float(pred_np, (base / f"{stem}_pred.png").as_posix())
        _write_sxs_comparison(gt_np, pred_np, psnr,
                               (base / f"{stem}_sxs.png").as_posix(), f"frame {i}")

        if (i + 1) % max(1, len(tf.frames) // 5) == 0 or i == len(tf.frames) - 1:
            print(f"  [Step 2b] {i+1}/{len(tf.frames)}  PSNR={psnr:.2f} dB")

    print(f"[Step 2b] Completato — PSNR medio={np.mean(psnrs):.2f} dB → {base}")


def _step3_posttrain_assets(cfg: PipelineConfig,
                             transforms_extended_path: Path,
                             optix_mod) -> dict:
    """Esegue IUM/visibility/color_texture/irradiance/indirect/albedo e aggiorna
    transforms_extended.json in-place con le nuove chiavi.
    """
    rc = cfg.render
    tf = load_transforms(str(transforms_extended_path))
    intr = tf.intrinsics
    print(f"[Step 3] {len(tf.frames)} frame  [{intr.w}×{intr.h}]")

    json_dir = Path(rc.output_dir).resolve()
    json_dir_str = json_dir.as_posix()
    os.makedirs(json_dir, exist_ok=True)

    model = optix_mod.TriangleMesh()
    model.add_from_obj_file(rc.model_path)
    print(f"[Step 3] Modello caricato: {rc.model_path}")

    # Leggi il JSON esistente — lo arricchiremo alla fine
    with open(transforms_extended_path, encoding="utf-8") as fh:
        output_json = json.load(fh)

    ium_result_data: dict = output_json.get("ium", {})

    if rc.render_ium:
        ium_w, ium_h = rc.ium_texture_size[0], rc.ium_texture_size[1]

        ium_gen = optix_mod.IUMGenerator()
        ium_gen.set_traversable(model)
        ium_gen.set_texture_size([ium_w, ium_h])
        ium_gen.render()
        ium_res = ium_gen.get_result()
        print("[Step 3] IUM rendering completato")

        ium_out_dir = json_dir / "ium"
        os.makedirs(ium_out_dir, exist_ok=True)

        if ium_res.has_positions():
            pos_arr = _reshape_flat(ium_res.positions_np.astype(np.float32), ium_w, ium_h)
            ium_pos_path = (ium_out_dir / f"ium_positions{rc.ium_format.extension}").resolve().as_posix()
            _save_layer(pos_arr, ium_pos_path, rc.ium_format, DataLayer.POSITION)
            ium_result_data["ium_positions_path"] = _as_relative_to(ium_pos_path, json_dir_str)

        if ium_res.has_normals():
            norm_arr = _reshape_flat(ium_res.normals_np.astype(np.float32), ium_w, ium_h)
            ium_norm_path = (ium_out_dir / f"ium_normals{rc.ium_format.extension}").resolve().as_posix()
            _save_layer(norm_arr, ium_norm_path, rc.ium_format, DataLayer.NORMAL)
            ium_result_data["ium_normals_path"] = _as_relative_to(ium_norm_path, json_dir_str)

        if ium_res.has_masks():
            mask_arr = _reshape_flat(ium_res.masks_np, ium_w, ium_h)
            ium_mask_path = (ium_out_dir / f"ium_masks{rc.ium_format.extension}").resolve().as_posix()
            _save_layer(mask_arr, ium_mask_path, rc.ium_format, DataLayer.MASK)
            ium_result_data["ium_masks_path"] = _as_relative_to(ium_mask_path, json_dir_str)

        all_cameras = []
        for frame in tf.frames:
            cam = _camera_from_matrix(frame.transform_matrix, intr.camera_angle_y, [intr.w, intr.h], optix_mod)
            all_cameras.append(cam)

        # ── Visibility ───────────────────────────────────────────────────────
        visibility_map = None
        if rc.render_visibility and ium_res.has_positions() and ium_res.has_masks():
            print("[Step 3] Calcolo Visibilità telecamere…")
            vis_gen = optix_mod.VisibilityGenerator()
            vis_gen.set_traversable(model)
            visibility_map = vis_gen.check_visibility(ium_res, ium_w, ium_h, all_cameras)

            vis_out_dir = json_dir / "visibility"
            os.makedirs(vis_out_dir, exist_ok=True)
            vis_path = (vis_out_dir / f"visibility{rc.visibility_format.extension}").resolve().as_posix()

            if rc.visibility_format == ImageFormat.OPENEXR:
                vis_arr = visibility_map.reshape((ium_h, ium_w, len(all_cameras))).astype(np.float32)
                _save_layer(vis_arr, vis_path, rc.visibility_format, DataLayer.VISIBILITY)
            else:
                visible_count = np.sum(visibility_map, axis=1)
                ratio = visible_count.astype(np.float32) / float(len(all_cameras))
                vis_arr = _reshape_flat(ratio, ium_w, ium_h)
                _save_layer(vis_arr, vis_path, rc.visibility_format, DataLayer.VISIBILITY)

            ium_result_data["visibility_path"] = _as_relative_to(vis_path, json_dir_str)

        # ── Color Texture ────────────────────────────────────────────────────
        ct_result = None
        if (rc.render_color_texture and rc.render_ium and rc.render_visibility
                and visibility_map is not None):
            print("[Step 3] Calcolo Color Texture…")

            optix_frames = []
            for i, frame in enumerate(tf.frames):
                cam = all_cameras[i]
                img_path = frame.file_path  # path normalizzato da transforms_extended.json
                img_flat = _load_image_as_vec3(img_path, intr.w, intr.h)
                peak = _compute_peak(img_flat.reshape(intr.h, intr.w, 3),
                                     rc.color_texture_peak_percentile)
                optix_frames.append(optix_mod.Frame(cam, peak, img_flat))

            ct_gen = optix_mod.ColorTexGenerator()
            ct_gen.set_inputs(ium_res, visibility_map, optix_frames)
            ct_gen.render()
            ct_result = ct_gen.get_result()

            ct_out_dir = json_dir / "color_texture"
            os.makedirs(ct_out_dir, exist_ok=True)
            ct_path = (ct_out_dir / f"color_texture{rc.color_texture_format.extension}").resolve().as_posix()
            ct_arr = _reshape_flat(ct_result.colors_np.astype(np.float32), ium_w, ium_h)
            _save_layer(ct_arr, ct_path, rc.color_texture_format, DataLayer.POSITION)
            ium_result_data["color_texture_path"] = _as_relative_to(ct_path, json_dir_str)

            if rc.render_pixel_change:
                pc_dir = json_dir / "pixel_change"
                pc_dir.mkdir(parents=True, exist_ok=True)
                min_arr   = _reshape_flat(ct_result.color_min_np.astype(np.float32), ium_w, ium_h)
                max_arr   = _reshape_flat(ct_result.color_max_np.astype(np.float32), ium_w, ium_h)
                range_arr = np.clip(max_arr - min_arr, 0.0, None)
                var_arr   = _reshape_flat(ct_result.color_variance_np.astype(np.float32), ium_w, ium_h)
                ext = rc.color_texture_format
                _save_layer(min_arr,   (pc_dir / f"color_min{ext.extension}").as_posix(),      ext, DataLayer.POSITION)
                _save_layer(max_arr,   (pc_dir / f"color_max{ext.extension}").as_posix(),      ext, DataLayer.POSITION)
                _save_layer(range_arr, (pc_dir / f"color_range{ext.extension}").as_posix(),    ext, DataLayer.POSITION)
                _save_layer(var_arr,   (pc_dir / f"color_variance{ext.extension}").as_posix(), ext, DataLayer.POSITION)
                if rc.debug_pixel_change:
                    _save_debug_pixel_change(min_arr, max_arr, range_arr, json_dir / "debug_pixel_change")

            cam_tex_dir = json_dir / "camera_texture"
            os.makedirs(cam_tex_dir, exist_ok=True)
            cam_colors = ct_result.camera_colors_np
            for cam_idx, frame in enumerate(tf.frames):
                cam_slice = cam_colors[:, cam_idx, :]
                cam_arr   = _reshape_flat(cam_slice.astype(np.float32), ium_w, ium_h)
                cam_path  = (cam_tex_dir / f"{frame.stem}{rc.color_texture_format.extension}").resolve().as_posix()
                _save_layer(cam_arr, cam_path, rc.color_texture_format, DataLayer.POSITION)
                if rc.debug_camera_texture:
                    src_img_path = images_out_dir_ct / Path(frame.file_path).name
                    _save_debug_comparison(src_img_path, cam_arr, frame.stem,
                                           json_dir / "debug_camera_texture")

        # ── Irradiance (Monte Carlo skybox) ──────────────────────────────────
        irr_res = None
        if (rc.render_irradiance and rc.skybox_path
                and ium_res.has_positions() and ium_res.has_normals()):
            print(f"[Step 3] Calcolo Irradiance "
                  f"({rc.irradiance_sample_side}×{rc.irradiance_sample_side} samples)…")
            sky_w, sky_h = rc.skybox_size[0], rc.skybox_size[1]
            # Usa lo skybox normalizzato dal JSON se disponibile (stessa scala di color+NeRF)
            norm_sky_rel = output_json.get("normalization", {}).get("normalized_skybox_path", "")
            if norm_sky_rel:
                norm_sky_abs = (json_dir / norm_sky_rel).resolve().as_posix()
                skybox_src = norm_sky_abs if Path(norm_sky_abs).exists() else rc.skybox_path
            else:
                skybox_src = rc.skybox_path
            skybox_flat = _load_image_as_vec3(skybox_src, sky_w, sky_h)
            irr_gen = optix_mod.IrradianceGenerator()
            irr_gen.set_traversable(model)
            irr_gen.set_inputs(ium_res, skybox_flat, [sky_w, sky_h],
                               rc.irradiance_sample_side, rc.skybox_yaw_degrees)
            irr_gen.render()
            irr_res = irr_gen.get_result()
            irr_out_dir = json_dir / "irradiance"
            os.makedirs(irr_out_dir, exist_ok=True)
            irr_path = (irr_out_dir / f"irradiance{rc.irradiance_format.extension}").resolve().as_posix()
            irr_arr = _reshape_flat(irr_res.irradiance_np.astype(np.float32), ium_w, ium_h)
            _save_layer(irr_arr, irr_path, rc.irradiance_format, DataLayer.IRRADIANCE)
            ium_result_data["irradiance_path"] = _as_relative_to(irr_path, json_dir_str)

        # ── Indirect Irradiance via NeRF ─────────────────────────────────────
        irr_indirect_flat = None
        if rc.precompute_indirect and ium_res.has_positions() and ium_res.has_normals():
            ind_out_dir = json_dir / "irradiance"
            os.makedirs(ind_out_dir, exist_ok=True)
            ind_path = (ind_out_dir / f"irradiance_indirect{rc.indirect_format.extension}").resolve().as_posix()
            if not os.path.exists(ind_path):
                print(f"[Step 3] Precompute Indirect Irradiance "
                      f"(N={rc.indirect_sample_side}, tile={rc.indirect_tile_size})…")
                _precompute_indirect_irradiance(rc, ium_res, model, ium_w, ium_h, ind_path)
            else:
                print(f"[Step 3] Indirect irradiance trovata su disco: {ind_path}")
            ind_arr = _load_exr_as_flat(ind_path)
            if ind_arr is not None:
                irr_indirect_flat = ind_arr
                ium_result_data["irradiance_indirect_path"] = _as_relative_to(ind_path, json_dir_str)

        # ── Albedo ───────────────────────────────────────────────────────────
        if (rc.render_albedo and ct_result is not None and irr_res is not None):
            print(f"[Step 3] Calcolo Albedo = π · color / max(irradiance, {rc.albedo_eps})…")
            color_flat = ct_result.colors_np.astype(np.float32)
            irr_flat   = irr_res.irradiance_np.astype(np.float32)
            if irr_indirect_flat is not None:
                irr_flat = irr_flat + irr_indirect_flat
            denom = np.maximum(irr_flat, rc.albedo_eps)
            albedo_flat = (np.float32(np.pi) * color_flat) / denom
            if ium_res.has_masks():
                albedo_flat[~ium_res.masks_np.astype(bool)] = 0.0
            albedo_flat = np.clip(albedo_flat, 0.0, 1.0)
            alb_out_dir = json_dir / "albedo"
            os.makedirs(alb_out_dir, exist_ok=True)
            alb_path = (alb_out_dir / f"albedo{rc.albedo_format.extension}").resolve().as_posix()
            alb_arr = _reshape_flat(albedo_flat, ium_w, ium_h)
            _save_layer(alb_arr, alb_path, rc.albedo_format, DataLayer.ALBEDO)
            ium_result_data["albedo_path"] = _as_relative_to(alb_path, json_dir_str)

    # Aggiorna il JSON in-place e riscrivi
    if ium_result_data:
        output_json["ium"] = ium_result_data
    with open(transforms_extended_path, "w", encoding="utf-8") as fh:
        json.dump(output_json, fh, indent=4)
    print(f"\n[Step 3] JSON aggiornato: {transforms_extended_path}")
    return output_json


def run_pipeline(cfg: PipelineConfig) -> dict:
    """Orchestratore a tre step. Ogni step può essere abilitato/disabilitato.

    Step 1 (run_step1): depth+mask per frame + copia immagini + transforms_extended.json minimo.
    Step 2 (run_step2): training NeRF via nerf/train.py, salva checkpoint.
    Step 3 (run_step3): IUM/visibility/color_texture/irradiance/indirect/albedo.
    """
    import OptixProgrammablePasses as optix
    optix.LogManager.set_min_level(optix.LogLevel.Error)
    optix.OptixManager.instance().set_log_level(optix.LogLevel.Disabled)

    transforms_extended = Path(cfg.render.output_dir) / "transforms_extended.json"

    if cfg.run_step1:
        transforms_extended = _step1_pretrain_data(cfg, optix)
    elif not transforms_extended.exists():
        raise FileNotFoundError(
            f"Step 1 disabilitato ma {transforms_extended} non esiste.\n"
            "Attivare run_step1=True oppure eseguire Step 1 manualmente."
        )
    else:
        # Validazione minima del JSON esistente
        with open(transforms_extended, encoding="utf-8") as fh:
            _probe = json.load(fh)
        frames = _probe.get("frames", [])
        if frames and ("depth_path" not in frames[0] or "mask_path" not in frames[0]):
            raise ValueError(
                f"{transforms_extended} non contiene depth_path/mask_path per frame.\n"
                "Attivare run_step1=True per rigenerarlo."
            )

    ckpt_path = Path(cfg.nerf_ckpt_path or
                     Path(cfg.render.output_dir) / "model" / "nerf_model_cache.pt")

    if cfg.run_step2:
        while True:
            ckpt_path = _step2_train_nerf(cfg, transforms_extended)

            if cfg.enable_nerf_render_train_images:
                _step2b_render_train_images(cfg, transforms_extended, ckpt_path)
                render_dir = cfg.nerf_render_train_images_dir or \
                             str(Path(cfg.render.output_dir) / "nerf_render_images")
                print(f"  Anteprime salvate in: {render_dir}")

            if not cfg.nerf_interactive_loop:
                break
            try:
                ans = input("\nContinuare il training? [y/N]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break
            if ans != "y":
                break

    elif cfg.run_step3 and cfg.render.precompute_indirect and not ckpt_path.exists():
        raise FileNotFoundError(
            f"Step 2 disabilitato ma il checkpoint NeRF non esiste: {ckpt_path}\n"
            "Attivare run_step2=True oppure fornire nerf_ckpt_path valido."
        )

    if cfg.run_step3:
        if cfg.render.precompute_indirect and not cfg.render.indirect_nerf_cache_path:
            cfg.render.indirect_nerf_cache_path = str(ckpt_path)

        return _step3_posttrain_assets(cfg, transforms_extended, optix)

    with open(transforms_extended, encoding="utf-8") as fh:
        return json.load(fh)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    REPO = "C:/Users/adria/Documents/GitHub/Tesi/OptixProjectCMake"

    cfg = PipelineConfig(
        run_step1 = True,
        run_step2 = True,
        run_step3 = True,

        render = RenderConfig(
            transforms_path = f"{REPO}/Scenes/SwordShield/NerfOpenEXR/transforms.json",
            model_path      = f"{REPO}/Scenes/SwordShield/Models/SwordShield.obj",
            output_dir      = "output/sworshield_render_nerf_skybox",

            render_depth    = True,
            render_position = False,  # Step 1 produces only depth+mask
            render_normal   = False,
            render_mask     = True,
            render_ium      = True,
            render_color_texture  = True,
            debug_camera_texture  = False,
            render_pixel_change   = True,
            debug_pixel_change    = False,

            render_irradiance      = True,
            irradiance_format      = ImageFormat.OPENEXR,
            skybox_path            = f"{REPO}/Scenes/SwordShield/Blender/assets/hdrs/clouds-sunshine_b963efc0-83f3-4957-8725-34f73b8744ff/clouds-sunshine_2K_09c69240-8e00-4b23-896f-fcde6fd514cc.exr",
            skybox_size            = [1024, 512],
            irradiance_sample_side = 512,

            precompute_indirect           = True,
            indirect_sample_side          = 64,
            indirect_tile_size            = 1024,
            indirect_depth_hint_enabled   = False,

            render_albedo = True,
            albedo_format = ImageFormat.OPENEXR,
            albedo_eps    = 1e-3,

            depth_format         = ImageFormat.OPENEXR,
            mask_format          = ImageFormat.PNG,
            ium_format           = ImageFormat.OPENEXR,
            visibility_format    = ImageFormat.OPENEXR,
            color_texture_format = ImageFormat.OPENEXR,

            ium_texture_size = [512, 512],
            apply_scale      = False,
        ),

        nerf_num_iters     = 10000,
        nerf_batch_size    = 4096,
        nerf_lr            = 5e-4,
        nerf_display_every = 100,
        nerf_seed          = 9458,

        enable_nerf_render_train_images = True,
        nerf_interactive_loop           = True,


        nerf_depth_hint_enabled        = True,
        nerf_foreground_ratio          = 0.8,   # usato solo se nerf_foreground_ratio_enabled=True
        nerf_foreground_ratio_enabled  = False,  # False = ratio naturale del dataset
        nerf_depth_window_samples      = 32,    # sample nella finestra
        nerf_depth_window              = 0.5,
        nerf_depth_window_end          = 0.5,   # finestra simmetrica [t_hit-0.5, t_hit+0.5]
        nerf_opacity_weight            = 1.0,   # supervisione opacità figura (acc_fg → 1)
        nerf_raw_noise_std             = 1.0,   # rumore pre-ReLU per sbloccare la densità
        nerf_train_background          = True,  # sky MLP: apprende l'environment dai pixel bg

    )

    run_pipeline(cfg)
