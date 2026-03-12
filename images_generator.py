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
            channel_names = {1: ["Z"], 3: ["R", "G", "B"], 4: ["R", "G", "B", "A"]}
            names = channel_names.get(c)
            if names is None:
                raise ValueError(f"ExrWriter: numero di canali non supportato: {c}")
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
    needs_raw = layer in {DataLayer.DEPTH, DataLayer.POSITION, DataLayer.NORMAL}

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

    # Cosa renderizzare
    render_depth:    bool = True
    render_position: bool = True
    render_normal:   bool = True
    render_mask     : bool = True   # mask di validità per ogni frame
    render_ium      : bool = True

    # Formato di salvataggio per ogni layer
    depth_format:    ImageFormat = ImageFormat.OPENEXR
    position_format: ImageFormat = ImageFormat.OPENEXR
    normal_format:   ImageFormat = ImageFormat.OPENEXR
    mask_format:     ImageFormat = ImageFormat.PNG      # uint8 → PNG è naturale
    ium_format:      ImageFormat = ImageFormat.PNG

    # Dimensione texture IUM [width, height]
    ium_texture_size: list[int] = field(default_factory=lambda: [512, 512])

    # Scala applicata ai depth (coerente con "scale" in transforms.json)
    apply_scale: bool = True


def run_pipeline(cfg: RenderConfig) -> dict:
    """Esegue l'intera pipeline e restituisce il JSON arricchito.

    Returns:
        Dizionario JSON con le stesse chiavi di transforms.json più i path
        dei layer renderizzati per ogni frame e la sezione "ium".
    """
    import OptixProgrammablePasses as optix  # importato una volta sola

    # Configura il logging
    optix.LogManager.set_min_level(optix.LogLevel.Error)
    optix.OptixManager.instance().set_log_level(optix.LogLevel.Disabled)

    # ── Carica dati di trasformazione ──────────────────────────────────────
    tf = load_transforms(cfg.transforms_path)
    intr = tf.intrinsics
    print(f"✓ Trasformazioni caricate: {len(tf.frames)} frame  [{intr.w}×{intr.h}]")

    # json_dir è la cartella del JSON di output — tutti i path nel JSON
    # saranno relativi a questa cartella, garantendo portabilità
    json_dir = Path(cfg.output_dir).resolve()
    json_dir_str = json_dir.as_posix()
    os.makedirs(json_dir, exist_ok=True)

    # ── Carica modello ─────────────────────────────────────────────────────
    model = optix.TriangleMesh()
    model.add_from_obj_file(cfg.model_path)
    print(f"✓ Modello caricato: {cfg.model_path}")

    # ── DepthGenerator — configurato una volta, riusato per tutti i frame ──
    depth_gen = optix.DepthGenerator()
    depth_gen.set_traversable(model)
    depth_gen.need_render_depth(cfg.render_depth)
    depth_gen.need_render_position(cfg.render_position)
    depth_gen.need_render_normal(cfg.render_normal)
    # La mask di validità viene sempre prodotta dalla libreria; la richiediamo
    # esplicitamente se l'utente la vuole salvare (non costa nulla aggiuntivo).

    # ── IUMGenerator — indipendente dai frame, eseguito una volta sola ─────
    ium_result_data: dict = {}
    if cfg.render_ium:
        ium_w, ium_h = cfg.ium_texture_size[0], cfg.ium_texture_size[1]

        ium_gen = optix.IUMGenerator()
        ium_gen.set_traversable(model)
        ium_gen.set_texture_size([ium_w, ium_h])
        ium_gen.render()
        ium_res = ium_gen.get_result()   # teniamo il riferimento vivo
        print("✓ IUM rendering completato")

        ium_out_dir = json_dir / "ium"
        os.makedirs(ium_out_dir, exist_ok=True)

        # positions_np → (N, 3)  con N = ium_w * ium_h
        if ium_res.has_positions():
            pos_arr = _reshape_flat(
                ium_res.positions_np.astype(np.float32), ium_w, ium_h
            )
            ium_pos_path = (ium_out_dir / f"ium_positions{cfg.ium_format.extension}").resolve().as_posix()
            _save_layer(pos_arr, ium_pos_path, cfg.ium_format, DataLayer.POSITION)
            ium_result_data["ium_positions_path"] = _as_relative_to(ium_pos_path, json_dir_str)

        # masks_np → (N,) uint8 — mappa di validità texel
        if ium_res.has_masks():
            mask_arr = _reshape_flat(ium_res.masks_np, ium_w, ium_h)
            ium_mask_path = (ium_out_dir / f"ium_masks{cfg.ium_format.extension}").resolve().as_posix()
            _save_layer(mask_arr, ium_mask_path, cfg.ium_format, DataLayer.MASK)
            ium_result_data["ium_masks_path"] = _as_relative_to(ium_mask_path, json_dir_str)

    # ── Copia immagini originali in output_dir/images/ ──────────────────────
    images_out_dir = json_dir / "images"
    os.makedirs(images_out_dir, exist_ok=True)

    # ── Loop sui frame ─────────────────────────────────────────────────────
    output_json = {**tf.raw}
    output_frames = []
    scale = tf.scale if cfg.apply_scale else 1.0

    for idx, frame in enumerate(tf.frames):
        print(f"\n── Frame {idx + 1}/{len(tf.frames)}: {frame.stem}")

        # Copia l'immagine originale nella sottocartella images/
        src_image = Path(frame.file_path)
        dst_image = images_out_dir / src_image.name
        if src_image.exists():
            shutil.copy2(src_image, dst_image)
            print(f"    ✓ Immagine copiata: {dst_image.name}")
        else:
            print(f"    ⚠  Immagine non trovata, skip copia: {src_image}")

        camera = _camera_from_matrix(frame.transform_matrix, intr.camera_angle_y, [intr.w, intr.h], optix)
        depth_gen.set_camera(
            camera
        )
        depth_gen.render()
        result = depth_gen.get_result()  # teniamo il riferimento vivo per tutto il frame

        frame_entry: dict = {
            "file_path":        _as_relative_to(dst_image.as_posix(), json_dir_str),
            "sharpness":        frame.sharpness,
            "transform_matrix": frame.transform_matrix,
        }
        stem = frame.stem
        W, H = intr.w, intr.h

        # Depth — (N,) float32 → reshape (H, W) --------------------------------
        if cfg.render_depth and result.has_depth_data():
            depth_arr = _reshape_flat(result.depths_np.astype(np.float32), W, H)
            if cfg.apply_scale:
                depth_arr = depth_arr * scale
            out_path = _build_output_path(cfg.output_dir, stem, "depth", cfg.depth_format)
            _save_layer(depth_arr, out_path, cfg.depth_format, DataLayer.DEPTH)
            frame_entry["depth_path"] = _as_relative_to(out_path, json_dir_str)

        # Position — (N, 3) float32 → reshape (H, W, 3) ------------------------
        if cfg.render_position and result.has_positional_data():
            pos_arr = _reshape_flat(result.positions_np.astype(np.float32), W, H)
            out_path = _build_output_path(cfg.output_dir, stem, "position", cfg.position_format)
            _save_layer(pos_arr, out_path, cfg.position_format, DataLayer.POSITION)
            frame_entry["position_path"] = _as_relative_to(out_path, json_dir_str)

        # Normal — (N, 3) float32 → reshape (H, W, 3) --------------------------
        if cfg.render_normal and result.has_normal_data():
            norm_arr = _reshape_flat(result.normals_np.astype(np.float32), W, H)
            out_path = _build_output_path(cfg.output_dir, stem, "normal", cfg.normal_format)
            _save_layer(norm_arr, out_path, cfg.normal_format, DataLayer.NORMAL)
            frame_entry["normal_path"] = _as_relative_to(out_path, json_dir_str)

        # Mask — (N,) uint8 → reshape (H, W) -----------------------------------
        if cfg.render_mask:
            mask_arr = _reshape_flat(result.masks_np, W, H)
            out_path = _build_output_path(cfg.output_dir, stem, "mask", cfg.mask_format)
            _save_layer(mask_arr, out_path, cfg.mask_format, DataLayer.MASK)
            frame_entry["mask_path"] = _as_relative_to(out_path, json_dir_str)

        output_frames.append(frame_entry)
        # result può uscire di scope in sicurezza solo dopo questo punto

    # ── Salva JSON arricchito ───────────────────────────────────────────────
    output_json["frames"] = output_frames
    if ium_result_data:
        output_json["ium"] = ium_result_data

    out_json_path = (json_dir / "transforms_extended.json").as_posix()
    with open(out_json_path, "w", encoding="utf-8") as fh:
        json.dump(output_json, fh, indent=4)
    print(f"\n✓ JSON arricchito salvato in: {out_json_path}")

    return output_json


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    REPO = "C:/Users/adria/Documents/GitHub/OptixProjectCMake"

    cfg = RenderConfig(
        transforms_path = f"{REPO}/Scenes/SwordShield/NerfRelative/transforms.json",
        model_path      = f"{REPO}/Scenes/SwordShield/Models/SwordShield.obj",
        output_dir      = "output/sworshield_render",

        render_depth    = True,
        render_position = True,
        render_normal   = True,
        render_mask     = True,
        render_ium      = True,

        # Cambia OPENEXR → PNG per normalizzare automaticamente in uint8
        depth_format    = ImageFormat.OPENEXR,
        position_format = ImageFormat.OPENEXR,
        normal_format   = ImageFormat.OPENEXR,
        mask_format     = ImageFormat.OPENEXR,
        ium_format      = ImageFormat.OPENEXR,

        ium_texture_size = [512, 512],
        apply_scale      = True,
    )

    run_pipeline(cfg)
