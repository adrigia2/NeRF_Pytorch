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


def _resolve_external_normal_size(
    rc,
    default_w: int,
    default_h: int,
) -> tuple[int, int, str | None]:
    """Determina larghezza/altezza effettiva per l'IUM quando si usa una normale esterna.

    Se rc.external_normal_path è None restituisce (default_w, default_h, None).

    Se la risoluzione nativa della normale coincide già con default_w×default_h
    restituisce quella stessa risoluzione con mode="match".

    Altrimenti:
      - rc.external_normal_resolution_mode == "resample" → (default_w, default_h, "resample")
      - rc.external_normal_resolution_mode == "adapt"    → (native_w, native_h, "adapt")
      - rc.external_normal_resolution_mode is None       → chiede all'utente a runtime
    """
    if not rc.external_normal_path:
        return default_w, default_h, None

    native = _load_image_hw3_native(rc.external_normal_path)
    native_h, native_w = native.shape[:2]

    if native_w == default_w and native_h == default_h:
        print(f"[IUM] Normale esterna: risoluzione nativa {native_w}×{native_h} "
              f"coincide con ium_texture_size → nessun ricampionamento necessario.")
        return default_w, default_h, "match"

    print(f"[IUM] Normale esterna: risoluzione nativa {native_w}×{native_h}, "
          f"ium_texture_size={default_w}×{default_h} — risoluzioni diverse.")

    mode = rc.external_normal_resolution_mode
    if mode is None:
        while True:
            ans = input(
                f"  Scegli la strategia di risoluzione:\n"
                f"    1 = resample: ridimensiona la normale a {default_w}×{default_h}\n"
                f"    2 = adapt: adatta ium_texture_size a {native_w}×{native_h}\n"
                f"  Scelta [1/2]: "
            ).strip()
            if ans in ("1", "2"):
                mode = "resample" if ans == "1" else "adapt"
                break
            print("  Input non valido, inserisci 1 o 2.")

    if mode == "resample":
        print(f"[IUM] Strategia: resample → la normale verrà ridimensionata a {default_w}×{default_h}.")
        return default_w, default_h, "resample"
    elif mode == "adapt":
        print(f"[IUM] Strategia: adapt → IUM verrà eseguita a {native_w}×{native_h}.")
        return native_w, native_h, "adapt"
    else:
        raise ValueError(
            f"external_normal_resolution_mode non riconosciuto: {mode!r}. "
            "Usa 'resample', 'adapt' o None."
        )


def _apply_external_normal(rc, ium_res, ium_w: int, ium_h: int) -> None:
    """Decodifica la normale esterna da [0,1] a [-1,1] e la inietta nel buffer C++
    di IUM_Generator::Result, sovrascrivendo la face-normal calcolata da OptiX.

    Dopo questa chiamata ium_res.normals_np (e il corrispondente buffer GPU usato
    da IrradianceGenerator / IndirectGenerator) contiene la normale esterna.
    """
    path = rc.external_normal_path
    print(f"[IUM] Carico normale esterna: {path}")

    # Carica e ridimensiona a ium_texture_size (LANCZOS).
    # _load_image_as_vec3 restituisce (N, 3) float32:
    #   - EXR → valori HDR raw (possibilmente già in [-1,1] o [0,1])
    #   - LDR (PNG/JPG) → uint8/255, quindi in [0,1]
    ext = _load_image_as_vec3(path, ium_w, ium_h)  # (N, 3)

    # Decodifica dal range sorgente a [-1,1].
    # Il range va dichiarato esplicitamente in rc.external_normal_range: l'auto-detect
    # su min() globale era fragile perché il ringing LANCZOS (su downscale aggressivi
    # tipo 4096→512) spinge alcuni pixel sotto 0, facendo saltare il decode e lasciando
    # i valori in [0,1] — il bug esatto segnalato.
    if rc.external_normal_range == "0_1":
        n = ext.astype(np.float32) * 2.0 - 1.0
    elif rc.external_normal_range == "-1_1":
        n = ext.astype(np.float32, copy=True)
    else:
        raise ValueError(
            f"external_normal_range non riconosciuto: {rc.external_normal_range!r} "
            f"(attesi '0_1' | '-1_1')."
        )

    if rc.external_normal_flip_green:
        n[:, 1] *= -1.0

    # Renormalizza: LANCZOS/interpolazione può produrre vettori non-unit.
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 1e-8)

    # Azzera i texel fuori dal mesh, coerentemente con la normale generata.
    if ium_res.has_masks():
        m = ium_res.masks_np.astype(bool)
        n[~m] = 0.0

    # Scrive nel buffer C++ (normals_np è una view zero-copy scrivibile).
    ium_res.normals_np[:] = n.astype(np.float32)
    print(f"[IUM] Normale esterna iniettata nel buffer IUM ({ium_w}×{ium_h}, "
          f"{int(np.count_nonzero(m) if ium_res.has_masks() else ium_w * ium_h)} texel validi).")


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

    # Normale IUM fornita dall'esterno (override rispetto a quella calcolata da OptiX).
    # Il file può essere in qualsiasi formato immagine (PNG, JPG, EXR…).
    # Se None (default) viene usata la normale calcolata dall'IUM.
    external_normal_path: str | None = None
    # Strategia da usare se la risoluzione dell'immagine esterna ≠ ium_texture_size.
    #   "resample"  → ricampiona la normale a ium_texture_size (LANCZOS).
    #   "adapt"     → adatta ium_texture_size alla risoluzione nativa della normale
    #                 (positions/mask vengono rigenerati alla stessa risoluzione).
    #   None        → chiede all'utente a runtime.
    external_normal_resolution_mode: str | None = None
    # Range dei valori della normale esterna.
    #   "0_1"  → normal-map standard codificata in [0,1]: applica decode n = ext*2-1.
    #   "-1_1" → EXR già decodificato in [-1,1] (es. un ium_normals ri-caricato): usa as-is.
    # Da impostare esplicitamente: l'auto-detect su min() globale era fragile (il
    # ringing LANCZOS su downscale aggressivi spingeva min sotto 0 saltando il decode).
    external_normal_range: str = "0_1"
    # Inverti il canale verde dopo il decode (utile per baker con convenzione DirectX).
    # Default False = convenzione OpenGL/Blender (Y+ verso l'alto).
    external_normal_flip_green: bool = False

    # Scala applicata ai depth — deve essere False: scale nei transforms.json
    # è da applicare ANCHE alle traslazioni della camera, e non farlo crea un
    # mismatch (query points NeRF ≠ superficie mesh).  Lasciare False.
    apply_scale: bool = False

    # Color texture
    render_color_texture: bool = False
    color_texture_format: ImageFormat = ImageFormat.OPENEXR
    # Percentile usato per calcolare il peak (default 95° = scarta il top 5% più luminoso)
    color_texture_peak_percentile: float = 100.0
    # Sorgente immagine per la color/camera texture (tutte le prospettive).
    # "gt"   → immagini ground-truth (default storico)
    # "nerf" → pred EXR salvati dallo Step 2b in nerf_render_images/iter_*/
    color_texture_image_source: str = "gt"
    # Iterazione Step 2b da cui leggere i pred NeRF. -1 = usa l'ultima disponibile.
    color_texture_nerf_iter: int = -1

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
    # Se True, il pass indiretto usa una finestra di campionamento custom attorno al
    # t_hit OptiX (indirect_depth_window / _end), invece di ereditare quella salvata
    # nel checkpoint di training. Il campionamento è comunque sempre centrato sul t_hit.
    indirect_override_depth_window: bool = False
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

    # Depth-guided training (Step 2) — richiede depth+mask dallo Step 1
    nerf_depth_window_samples: int   = 32    # sample nella finestra mesh per i raggi figura
    nerf_depth_window:         float = 0.5   # [t_hit - window, t_hit + window_end]
    nerf_depth_window_end:     float = 0.5
    nerf_opacity_weight:       float = 1.0   # peso della loss opacità (fg e bg)
    nerf_raw_noise_std:        float = 0.0   # rumore pre-ReLU sulla densità
    nerf_bg_radius_mult:       float = 6.0   # raggio sfera bg = bg_radius_mult × max_side
    nerf_bg_depth_window:      float = 2.0   # finestra bg [R - window, R + window_end]
    nerf_bg_depth_window_end:  float = 2.0
    nerf_profile_iters: int = 0         # per-fase timing sincronizzato per i primi N iter (0=off)

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
    if cfg.indirect_override_depth_window:
        nerf_cfg.depth_window     = cfg.indirect_depth_window
        nerf_cfg.depth_window_end = cfg.indirect_depth_window_end
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


def _find_nerf_pred_dir(base_root: Path, iter_sel: int) -> "Path | None":
    """Restituisce la sottocartella iter_* con l'iterazione richiesta (o la massima)."""
    if not base_root.is_dir():
        return None
    candidates = {}
    for d in base_root.iterdir():
        if d.is_dir() and d.name.startswith("iter_"):
            try:
                candidates[int(d.name[5:])] = d
            except ValueError:
                pass
    if not candidates:
        return None
    if iter_sel == -1:
        return candidates[max(candidates)]
    return candidates.get(iter_sel)


def _nerf_pred_path(pred_dir: Path, idx: int) -> Path:
    return pred_dir / f"frame_{idx:03d}_pred.exr"


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

    hist_dir = images_out_dir / "histograms"
    hist_edges: np.ndarray | None = None
    hist_counts: np.ndarray | None = None   # shape (256, 3)

    for idx, frame in enumerate(tf.frames):
        print(f"\n[Step 1] Frame {idx + 1}/{len(tf.frames)}: {frame.stem}")

        src_image = Path(frame.file_path)
        arr: np.ndarray | None = None
        if norm_divisor is not None:
            dst_image = images_out_dir / (src_image.stem + ".exr")
            if src_image.exists():
                arr = (_load_image_hw3_native(str(src_image)) / norm_divisor).astype(np.float32)
                get_writer(ImageFormat.OPENEXR).write(arr, str(dst_image))
            else:
                print(f"    ⚠  Immagine non trovata, skip: {src_image}")
        else:
            dst_image = images_out_dir / src_image.name
            if src_image.exists():
                shutil.copy2(src_image, dst_image)
                arr = _load_image_hw3_native(str(dst_image)).astype(np.float32)
            else:
                print(f"    ⚠  Immagine non trovata, skip copia: {src_image}")

        if arr is not None:
            _write_rgb_histogram(arr, str(hist_dir / f"{src_image.stem}_hist.png"),
                                 f"frame {idx} — {src_image.stem}")
            if hist_edges is None:
                xmax = max(float(np.percentile(arr, 99.5)), 1e-6)
                hist_edges = np.linspace(0.0, xmax, 257)
                hist_counts = np.zeros((256, 3), dtype=np.int64)
            flat = arr.reshape(-1, 3)
            for ch in range(3):
                c, _ = np.histogram(flat[:, ch], bins=hist_edges)
                hist_counts[:, ch] += c

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

    if hist_counts is not None:
        _write_rgb_histogram_from_counts(hist_counts, hist_edges,
                                         str(hist_dir / "all_frames_hist.png"),
                                         "All frames — RGB histogram")

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
        depth_window_samples      = cfg.nerf_depth_window_samples,
        depth_window              = cfg.nerf_depth_window,
        depth_window_end          = cfg.nerf_depth_window_end,
        opacity_weight            = cfg.nerf_opacity_weight,
        raw_noise_std             = cfg.nerf_raw_noise_std,
        bg_radius_mult            = cfg.nerf_bg_radius_mult,
        bg_depth_window           = cfg.nerf_bg_depth_window,
        bg_depth_window_end       = cfg.nerf_bg_depth_window_end,
        profile_iters             = cfg.nerf_profile_iters,
        use_hdr_activation       = False,
    )

    print(f"[Step 2] Training NeRF (depth-guided) — {cfg.nerf_num_iters} iter, ckpt → {ckpt}")
    print(f"[Step 2] campionamento depth-guided, mesh_window=[t-{cfg.nerf_depth_window}, t+{cfg.nerf_depth_window_end}], "
          f"bg_radius_mult={cfg.nerf_bg_radius_mult}")
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


_RGB_HIST_COLORS = [("R", "#E63946"), ("G", "#2A9D8F"), ("B", "#457B9D")]


def _write_rgb_histogram(arr_hw3: np.ndarray, path: str, title: str) -> None:
    """Save RGB histogram of a (H, W, 3) float32 array as PNG. HDR-aware (no clamp to 1)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("    ⚠  matplotlib non disponibile: istogramma saltato")
        return
    flat = arr_hw3.reshape(-1, 3).astype(np.float32)
    xmax = max(float(np.percentile(flat, 99.5)), 1e-6)
    edges = np.linspace(0.0, xmax, 257)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for ch, (label, color) in enumerate(_RGB_HIST_COLORS):
        counts, _ = np.histogram(flat[:, ch], bins=edges)
        ax.stairs(counts, edges, fill=True, alpha=0.55, color=color, label=label)
    ax.set_xlabel("pixel value (linear HDR)")
    ax.set_ylabel("pixel count")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _write_rgb_histogram_from_counts(counts_n3: np.ndarray, edges: np.ndarray,
                                      path: str, title: str) -> None:
    """Plot RGB histogram from pre-accumulated counts (N_bins, 3) and bin edges (N_bins+1,)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for ch, (label, color) in enumerate(_RGB_HIST_COLORS):
        ax.stairs(counts_n3[:, ch], edges, fill=True, alpha=0.55, color=color, label=label)
    ax.set_xlabel("pixel value (linear HDR)")
    ax.set_ylabel("pixel count")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _write_rgb_hist_comparison(pred_hw3: np.ndarray, gt_hw3: np.ndarray,
                                path: str, stem: str) -> None:
    """Save a two-panel PNG: pred histogram (top) and GT histogram (bottom), shared x axis."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("    ⚠  matplotlib non disponibile: istogramma saltato")
        return
    pred_flat = pred_hw3.reshape(-1, 3).astype(np.float32)
    gt_flat   = gt_hw3.reshape(-1, 3).astype(np.float32)
    xmax = max(float(np.percentile(pred_flat, 99.5)),
               float(np.percentile(gt_flat, 99.5)), 1e-6)
    edges = np.linspace(0.0, xmax, 257)
    fig, (ax_pred, ax_gt) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    for ch, (label, color) in enumerate(_RGB_HIST_COLORS):
        c_pred, _ = np.histogram(pred_flat[:, ch], bins=edges)
        ax_pred.stairs(c_pred, edges, fill=True, alpha=0.55, color=color, label=label)
        c_gt, _ = np.histogram(gt_flat[:, ch], bins=edges)
        ax_gt.stairs(c_gt, edges, fill=True, alpha=0.55, color=color, label=label)
    ax_pred.set_ylabel("pixel count")
    ax_pred.set_title(f"{stem} — Pred")
    ax_pred.legend(loc="upper right")
    ax_gt.set_xlabel("pixel value (linear HDR)")
    ax_gt.set_ylabel("pixel count")
    ax_gt.set_title(f"{stem} — GT")
    ax_gt.legend(loc="upper right")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


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


def _load_mask_bool(path: str) -> np.ndarray | None:
    """Carica una maschera PNG come (H, W) bool (True = figura). None se fallisce."""
    try:
        from PIL import Image
        arr = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        return arr > 0.5
    except Exception as e:
        print(f"    ⚠  Impossibile caricare mask {path}: {e}")
        return None


# Fasce di luminanza GT (lineare HDR) usate per i grafici errore-per-luminanza.
_LUMA_BINS = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.2), (0.2, 1.0), (1.0, 5.0), (5.0, np.inf)]


def _write_error_luminance_plot(gt_np: np.ndarray, pred_np: np.ndarray,
                                 sel: np.ndarray | None, path: str, title: str) -> None:
    """Grafico a barre dell'errore raggruppato per fascia di luminanza GT.

    sel: maschera booleana (H, W) dei pixel da includere; None = tutta l'immagine.
    Mostra, per ogni fascia: |pred-gt| medio assoluto (asse sinistro) ed errore
    relativo medio (asse destro), col numero di pixel per fascia.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("    ⚠  matplotlib non disponibile: grafico errore saltato")
        return

    gl   = gt_np.mean(-1)
    diff = np.abs(pred_np - gt_np).mean(-1)
    if sel is None:
        sel = np.ones(gl.shape, dtype=bool)

    labels, abs_means, rel_means, counts = [], [], [], []
    for lo, hi in _LUMA_BINS:
        b = sel & (gl >= lo) & (gl < hi)
        n = int(b.sum())
        labels.append(f">{lo:g}" if not np.isfinite(hi) else f"{lo:g}–{hi:g}")
        counts.append(n)
        if n == 0:
            abs_means.append(0.0); rel_means.append(0.0)
        else:
            abs_means.append(float(diff[b].mean()))
            rel_means.append(float((diff[b] / (gl[b] + 1e-3)).mean()))

    x = np.arange(len(_LUMA_BINS))
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.bar(x - 0.2, abs_means, width=0.4, color="#4C72B0")
    ax1.set_ylabel("Mean |pred − gt| (absolute)", color="#4C72B0")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")
    ax1.set_xlabel("GT luminance bin (linear HDR)")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=30, ha="right")

    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, rel_means, width=0.4, color="#C44E52")
    ax2.set_ylabel("Mean relative error", color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")

    ymax = max(abs_means) if any(abs_means) else 1.0
    for xi, c in zip(x, counts):
        ax1.text(xi, ymax * 0.01, f"n={c}", ha="center", va="bottom",
                 fontsize=7, rotation=90, color="gray")

    ax1.set_title(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _step2b_render_train_images(cfg: PipelineConfig,
                                transforms_extended_path: Path,
                                ckpt_path: Path) -> None:
    """Per ogni frame renderizza col NeRF e salva GT, Pred e diff come EXR + PNG.

    Gli output vengono scritti in una sottocartella denominata iter_<NNNNNN> all'interno
    di nerf_render_images, così ogni stop del training interattivo produce una directory
    separata senza sovrascrivere gli stop precedenti.
    """
    import sys
    import torch as _torch
    sys.path.insert(0, str(Path(__file__).parent))
    from nerf import load_checkpoint, render_image as nerf_render_image

    rc = cfg.render

    iter_done = int(_torch.load(str(ckpt_path), map_location="cpu")["iter_done"])
    model_bundle, nerf_cfg = load_checkpoint(str(ckpt_path))

    tf   = load_transforms(str(transforms_extended_path))
    intr = tf.intrinsics

    with open(transforms_extended_path, encoding="utf-8") as fh:
        raw_frames = json.load(fh)["frames"]

    base_root = Path(cfg.nerf_render_train_images_dir or
                     Path(rc.output_dir) / "nerf_render_images")
    base = base_root / f"iter_{iter_done:06d}"
    base.mkdir(parents=True, exist_ok=True)

    exr_writer = get_writer(ImageFormat.OPENEXR)

    print(f"\n[Step 2b] Rendering {len(tf.frames)} frame con NeRF (iter={iter_done})...")
    psnrs = []
    for i, (frame, raw_frame) in enumerate(zip(tf.frames, raw_frames)):
        gt_np  = _load_image_hw3_native(frame.file_path)
        pose   = np.array(frame.transform_matrix, dtype=np.float32)

        dep_np = None
        dep_str = raw_frame.get("depth_path", "")
        if dep_str:
            dep_full = (dep_str if Path(dep_str).is_absolute()
                        else (Path(transforms_extended_path).parent / dep_str).resolve().as_posix())
            dep_np = _load_depth_np(dep_full)

        mask_bool = None
        mask_str = raw_frame.get("mask_path", "")
        if mask_str:
            mask_full = (mask_str if Path(mask_str).is_absolute()
                         else (Path(transforms_extended_path).parent / mask_str).resolve().as_posix())
            mask_bool = _load_mask_bool(mask_full)

        pred_np = nerf_render_image(
            model_bundle, intr.h, intr.w, intr.fl_x, pose, nerf_cfg,
            focal_y=intr.fl_y, cx=intr.cx, cy=intr.cy, target_depth=dep_np,
        )

        psnr = -10.0 * np.log10(np.mean((pred_np - gt_np) ** 2) + 1e-10)
        psnrs.append(psnr)

        stem = f"frame_{i:03d}"
        # EXR: HDR float32, no clamp — preserva valori >1 e segno della differenza
        exr_writer.write(gt_np,             (base / f"{stem}_gt.exr").as_posix())
        exr_writer.write(pred_np,           (base / f"{stem}_pred.exr").as_posix())
        exr_writer.write(pred_np - gt_np,   (base / f"{stem}_diff.exr").as_posix())
        # PNG: anteprima visiva clampata a [0,1]
        _write_png_float(gt_np,   (base / f"{stem}_gt.png").as_posix())
        _write_png_float(pred_np, (base / f"{stem}_pred.png").as_posix())
        _write_sxs_comparison(gt_np, pred_np, psnr,
                               (base / f"{stem}_sxs.png").as_posix(), f"frame {i}")

        # RGB histogram comparison: pred (top) vs GT (bottom)
        _write_rgb_hist_comparison(pred_np, gt_np,
                                   (base / f"{stem}_rgb_hist.png").as_posix(), stem)

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
        # Dimensione IUM: potrebbe essere adattata alla risoluzione della normale esterna
        # se external_normal_resolution_mode == "adapt" (o se l'utente lo sceglie a runtime).
        default_ium_w, default_ium_h = rc.ium_texture_size[0], rc.ium_texture_size[1]
        ium_w, ium_h, _ext_norm_mode = _resolve_external_normal_size(
            rc, default_ium_w, default_ium_h
        )

        ium_gen = optix_mod.IUMGenerator()
        ium_gen.set_traversable(model)
        ium_gen.set_texture_size([ium_w, ium_h])
        ium_gen.render()
        ium_res = ium_gen.get_result()
        print("[Step 3] IUM rendering completato")

        # Se è stata fornita una normale esterna, decodificala e iniettala nel buffer
        # C++ di IUM_Generator::Result prima di qualsiasi uso a valle.
        if rc.external_normal_path:
            _apply_external_normal(rc, ium_res, ium_w, ium_h)

        ium_out_dir = json_dir / "ium"
        os.makedirs(ium_out_dir, exist_ok=True)

        if ium_res.has_positions():
            pos_arr = _reshape_flat(ium_res.positions_np.astype(np.float32), ium_w, ium_h)
            ium_pos_path = (ium_out_dir / f"ium_positions{rc.ium_format.extension}").resolve().as_posix()
            _save_layer(pos_arr, ium_pos_path, rc.ium_format, DataLayer.POSITION)
            ium_result_data["ium_positions_path"] = _as_relative_to(ium_pos_path, json_dir_str)

        if ium_res.has_normals():
            norm_arr = _reshape_flat(ium_res.normals_np.astype(np.float32), ium_w, ium_h)
            # Quando si usa la normale esterna forziamo EXR per garantire il salvataggio
            # dei valori raw in [-1, 1] indipendentemente da rc.ium_format.
            norm_save_fmt = (
                ImageFormat.OPENEXR if rc.external_normal_path else rc.ium_format
            )
            ium_norm_path = (ium_out_dir / f"ium_normals{norm_save_fmt.extension}").resolve().as_posix()
            _save_layer(norm_arr, ium_norm_path, norm_save_fmt, DataLayer.NORMAL)
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

            nerf_pred_dir = None
            if rc.color_texture_image_source == "nerf":
                base_root = Path(cfg.nerf_render_train_images_dir or
                                 json_dir / "nerf_render_images")
                nerf_pred_dir = _find_nerf_pred_dir(base_root, rc.color_texture_nerf_iter)
                if nerf_pred_dir is None:
                    print("    ⚠  Nessuna cartella pred NeRF trovata → uso immagini GT")
                else:
                    print(f"[Step 3] Color texture da pred NeRF: {nerf_pred_dir}")

            optix_frames = []
            for i, frame in enumerate(tf.frames):
                cam = all_cameras[i]
                img_path = frame.file_path
                if nerf_pred_dir is not None:
                    pred_path = _nerf_pred_path(nerf_pred_dir, i)
                    if pred_path.exists():
                        img_path = pred_path.as_posix()
                    else:
                        print(f"    ⚠  pred NeRF mancante per frame {i} ({pred_path.name}), uso GT")
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

            _step2b_render_train_images(cfg, transforms_extended, ckpt_path)
            render_dir = cfg.nerf_render_train_images_dir or \
                         str(Path(cfg.render.output_dir) / "nerf_render_images")
            print(f"  EXR/PNG salvati in: {render_dir}")

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

    # Libera la VRAM del training NeRF prima che OptiX allochi i buffer Step 3
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

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
            external_normal_path = f"{REPO}/Scenes/TableAndOther/BlenderBaked/BakedMaterial_normal.exr",
            external_normal_resolution_mode = "resample",  # "adapt" | "resample" | "none"
            transforms_path = f"{REPO}/Scenes/TableAndOther/NerfOpenEXR/transforms.json",
            model_path      = f"{REPO}/Scenes/TableAndOther/Models/Baked.obj",
            output_dir      = "D:/tesi_output/new_scene_nerf_color_mapping_fix_resize_normal_map",

            render_depth    = True,
            render_position = True,  # Step 1 produces only depth+mask
            render_normal   = True,
            render_mask     = True,
            render_ium      = True,
            render_color_texture  = True,
            debug_camera_texture  = False,
            render_pixel_change   = True,
            debug_pixel_change    = False,

            render_irradiance      = True,
            

            skybox_path            = f"{REPO}/Scenes/TableAndOther/Blender/assets/hdri/suburban_garden_4k.exr",
            skybox_size            = [1024, 512],
            irradiance_sample_side = 512,

            precompute_indirect           = True,
            indirect_sample_side          = 64,
            indirect_tile_size            = 1024,
            indirect_override_depth_window = False,

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

            color_texture_image_source = "nerf",  # "nerf" | "gt"
            

        ),

        nerf_num_iters     = 50000,
        nerf_batch_size    = 4096*24,
        nerf_lr            = 5e-4,
        nerf_display_every = 100,
        nerf_seed          = 9458,

        enable_nerf_render_train_images = True,
        nerf_interactive_loop           = False,



        nerf_depth_window_samples      = 4,
        nerf_depth_window              = 0.1,
        nerf_depth_window_end          = 0.1,
        nerf_opacity_weight            = 1.0,
        nerf_raw_noise_std             = 1.0,
        nerf_bg_radius_mult            = 3.0,
        nerf_bg_depth_window           = 0.1,
        nerf_bg_depth_window_end       = 0.1,

        nerf_profile_iters = 0,

    )

    run_pipeline(cfg)
