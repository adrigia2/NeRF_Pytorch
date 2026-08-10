"""
nerf_render_pipeline.py
-----------------------
Legge un file transforms.json (formato NeRF), renderizza per ogni frame
depth / position / normal / mask tramite OptixProgrammablePasses e genera la IUM.
Salva ogni output nel formato scelto (openexr | png); quando il formato
non supporta dati raw (float), normalizza automaticamente i valori in [0,1].
"""

from __future__ import annotations

import contextlib
import copy
import datetime
import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
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
    SPEC_CONE           = auto() # radianza media cono speculare L_j(r) (RGB float HDR)
    METALLIC            = auto() # specularità 1−X del fit PBR, [0,1] (float)
    ROUGHNESS           = auto() # apertura cono / 180 dove attendibile, [0,1] (float)
    SPEC_CONE_R         = auto() # apertura cono best-fit in gradi (float)


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
# Writer: OpenEXR incrementale (per blocchi di scanline)
# ──────────────────────────────────────────────────────────────────────────────

class IncrementalExrWriter:
    """EXR scritto per blocchi di scanline consecutive, senza mai tenere in RAM
    l'immagine intera.

    Serve al bake spec_cone condiviso, dove il loop esterno è sul tile e non sulla
    camera: gli accumulatori di tutte le camere a piena risoluzione non ci starebbero
    (a 4096², K=14, 58 camere sarebbero ~200 GiB), mentre un blocco di scanline per
    camera costa qualche centinaio di KB.

    channels: dict {nome: dtype}, con dtype np.float16 → canale HALF, altrimenti FLOAT.
    Ogni write_block riceve {nome: array (rows, width)} e avanza di `rows` scanline;
    i blocchi vanno scritti in ordine e devono coprire esattamente `height` righe.
    """

    def __init__(self, path: str, width: int, height: int,
                 channels: "dict[str, type]", compression: str = "zip"):
        import OpenEXR, Imath

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path    = path
        self.width   = width
        self.height  = height
        self.channels = dict(channels)
        self.row     = 0

        half  = Imath.PixelType(Imath.PixelType.HALF)
        flt   = Imath.PixelType(Imath.PixelType.FLOAT)
        header = OpenEXR.Header(width, height)
        header["channels"] = {
            name: Imath.Channel(half if dt == np.float16 else flt)
            for name, dt in self.channels.items()
        }
        comp = {"zip":  Imath.Compression.ZIP_COMPRESSION,
                "zips": Imath.Compression.ZIPS_COMPRESSION,
                "none": Imath.Compression.NO_COMPRESSION}[compression]
        header["compression"] = Imath.Compression(comp)
        self._file = OpenEXR.OutputFile(path, header)

    def write_block(self, block: "dict[str, np.ndarray]") -> None:
        rows = None
        payload = {}
        for name, dt in self.channels.items():
            arr = np.ascontiguousarray(block[name], dtype=dt)
            if arr.ndim != 2 or arr.shape[1] != self.width:
                raise ValueError(f"IncrementalExrWriter: canale {name!r} ha shape "
                                 f"{arr.shape}, attese (righe, {self.width})")
            if rows is None:
                rows = arr.shape[0]
            elif arr.shape[0] != rows:
                raise ValueError("IncrementalExrWriter: i canali di un blocco devono "
                                 "avere lo stesso numero di righe")
            payload[name] = arr.tobytes()

        if self.row + rows > self.height:
            raise ValueError(f"IncrementalExrWriter: {self.path} sforerebbe "
                             f"{self.height} righe (riga {self.row} + {rows})")
        self._file.writePixels(payload, rows)
        self.row += rows

    def close(self) -> None:
        if self._file is None:
            return
        if self.row != self.height:
            raise ValueError(f"IncrementalExrWriter: {self.path} chiuso a {self.row} "
                             f"righe su {self.height} (file troncato)")
        self._file.close()
        self._file = None

    def __enter__(self): return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        else:                      # non validare l'altezza se stiamo già fallendo
            self._file = None


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


# ──────────────────────────────────────────────────────────────────────────────
# Logging: tee stdout+stderr su file (diagnostica per run notturni)
# ──────────────────────────────────────────────────────────────────────────────

class _Tee:
    """Scrive su due stream contemporaneamente; flush dopo ogni write."""
    def __init__(self, original, file_stream):
        self._orig = original
        self._file = file_stream

    def write(self, data):
        self._orig.write(data)
        # Il file può essere già chiuso: colorama (init lazy via tqdm) registra un
        # atexit reset legato al _Tee attivo in quel momento, che può scattare dopo
        # l'uscita da _console_to_file.
        if not self._file.closed:
            self._file.write(data)
            self._file.flush()

    def flush(self):
        self._orig.flush()
        if not self._file.closed:
            self._file.flush()

    # Propagate altri attributi al flusso originale (es. encoding, isatty)
    def __getattr__(self, name):
        return getattr(self._orig, name)


@contextlib.contextmanager
def _console_to_file(log_path: str):
    """Context manager: redirige stdout e stderr anche su *log_path* (append, flush per riga)."""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8", buffering=1) as fh:
        orig_out, orig_err = sys.stdout, sys.stderr
        sys.stdout = _Tee(orig_out, fh)
        sys.stderr = _Tee(orig_err, fh)
        try:
            yield
        finally:
            sys.stdout = orig_out
            sys.stderr = orig_err


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
                          DataLayer.IRRADIANCE, DataLayer.IRRADIANCE_INDIRECT, DataLayer.ALBEDO,
                          DataLayer.SPEC_CONE, DataLayer.METALLIC, DataLayer.ROUGHNESS,
                          DataLayer.SPEC_CONE_R}

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


def _save_visibility_map(visibility_map: np.ndarray, vis_path: str,
                         ium_h: int, ium_w: int, n_cams: int,
                         fmt: ImageFormat) -> None:
    """Salva la visibility (num_pix, n_cams) uint8 su disco.

    EXR → multi-canale, un canale 0/1 per camera (formato letto da pbr_solver e
    dagli inspector). Formati non-EXR → frazione di camere visibili per texel.
    """
    if fmt == ImageFormat.OPENEXR:
        vis_arr = visibility_map.reshape((ium_h, ium_w, n_cams)).astype(np.float32)
    else:
        ratio = np.sum(visibility_map, axis=1).astype(np.float32) / float(max(n_cams, 1))
        vis_arr = _reshape_flat(ratio, ium_w, ium_h)
    _save_layer(vis_arr, vis_path, fmt, DataLayer.VISIBILITY)


def _load_camera_masks(mask_dir: Path, stems: "list[str]", num_pix: int) -> "np.ndarray | None":
    """Ricarica le maschere per-camera da <mask_dir>/{stem}.exr → (num_pix, n_cams) uint8.

    Ritorna None se la cartella o anche una sola maschera manca (così il chiamante
    sa che deve ricalcolare color_texture per rigenerarle).
    """
    if not mask_dir.is_dir():
        return None
    masks = np.zeros((num_pix, len(stems)), dtype=np.uint8)
    for j, stem in enumerate(stems):
        p = mask_dir / f"{stem}.exr"
        if not p.exists():
            return None
        masks[:, j] = (_load_image_hw3_native(p.as_posix())[..., 0] > 0.5).reshape(num_pix)
    return masks


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
    # Sorgenti da cui produrre TUTTE le uscite source-dipendenti dello Step 3.
    # Ogni source viene processata per intero e in modo identico, sotto sources/{src}/:
    # color_texture/, camera_texture/, pixel_change/, albedo/, metallic/, roughness/,
    # albedo_pbr/, pbr/. I nomi interni NON sono suffissati (il src è nel nome cartella).
    # I passi source-indipendenti (ium, visibility, irradiance, spec_cone, indirect)
    # restano al livello superiore, condivisi.
    # "gt"   → immagini ground-truth
    # "nerf" → pred EXR salvati dallo Step 2b in nerf_render_images/iter_*/
    color_texture_image_sources: list[str] = field(default_factory=lambda: ["gt"])
    # Iterazione Step 2b da cui leggere i pred NeRF. -1 = usa l'ultima disponibile.
    color_texture_nerf_iter: int = -1
    # Angolo massimo (in gradi) dalla normale del texel oltre il quale il contributo
    # di una camera viene scartato (vista troppo radente → bleed dello sfondo ai bordi).
    # 90.0 = filtro disabilitato; default operativo 75° (scarta entro 15° dalla tangente).
    color_texture_grazing_max_deg: float = 75.0

    # Debug output
    debug_camera_texture: bool = False   # salva side-by-side camera image vs camera_texture

    # Pixel change output
    render_pixel_change: bool = False    # salva min/max/range texture in pixel_change/
    debug_pixel_change: bool = False     # salva plot comparativo in debug_pixel_change/

    # Irradiance map (skybox per-texel, quadratura deterministica su spirale di Fibonacci)
    render_irradiance: bool = False
    irradiance_format: ImageFormat = ImageFormat.OPENEXR
    # Sorgente dell'envmap usato dal pass irradiance:
    #   "file" → EXR equirettangolare letto da skybox_path
    #   "nerf" → bake della bg-sphere del NeRF allenato (checkpoint risolto come per
    #            l'indirect: indirect_nerf_cache_path o <output_dir>/model/...).
    #            skybox_path non è richiesto; la mappa bakata viene salvata come
    #            skybox_nerf_baked.exr per ispezione/confronto con la GT.
    skybox_source: str = "file"
    skybox_path: str = ""                # path al file EXR equirettangolare
    skybox_size: list[int] = field(default_factory=lambda: [1024, 512])  # resize target
    irradiance_sample_side: int = 16     # N → N×N campioni per emisfero (16 = 256, 256 = 65536)
    skybox_yaw_degrees: float = 0.0      # rotazione yaw skybox; 0° = -Y (Blender fwd) al centro
    compare_skybox_to_gt: bool = False   # True → genera skybox_compare/skybox_heatmap.png dopo il bake

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

    # Specular cone pass — precompute L_j(r_k) per il fit PBR C_j = X·D + (1-X)·L_j(r).
    # Campionamento ad anelli concentrici attorno al raggio riflesso: ogni raggio è
    # tracciato/interrogato una volta e i coni si ricostruiscono per cumulativa pesata
    # (vedi _precompute_spec_cone). Richiede render_ium, render_visibility e il
    # checkpoint NeRF (come precompute_indirect); usa la stessa skybox dell'irradiance.
    precompute_spec_cone: bool = False
    # Schema di campionamento:
    #   "per_camera" → anelli concentrici attorno a R_j, rilanciati per ogni camera
    #                  (spec_cone_samples_per_ring / _sample_alloc / _budget / _floor)
    #   "shared"     → un set Fibonacci uniforme sull'emisfero sopra n, tracciato e
    #                  interrogato UNA volta e classificato da ogni camera nel proprio
    #                  anello. La radianza incidente non dipende dalla camera, quindi
    #                  costa S + m raggi/texel invece di m·ΣN_i (m = camere che vedono
    #                  il texel). Le aperture strette costano però risoluzione: un cono
    #                  di apertura a riceve S·(1−cos(a/2)) campioni, quindi sotto ~7°
    #                  (a S=16384) si va sotto i 30 campioni. In compenso raffinare la
    #                  griglia di aperture non costa un raggio in più.
    spec_cone_scheme: str = "per_camera"
    # Campioni condivisi per texel (S), solo per scheme="shared"
    spec_cone_shared_samples: int = 16384
    # Texel per sotto-blocco torch: dirs + radianze costano 24 B/raggio, quindi un
    # tile intero non ci starebbe in VRAM. Non cambia il risultato, solo il picco
    # e l'efficienza: sotto-blocchi piccoli danno kernel torch piccoli e tanto
    # overhead Python nel loop sulle camere.
    spec_cone_chunk_texels: int = 256
    # Raggi per batch nelle query NeRF del bake (override di NerfConfig.chunk, che
    # arriva dal checkpoint e vale 32768). È il vero limite all'occupazione della
    # GPU: il batch della rete è cappato lì, quindi alzare spec_cone_chunk_texels
    # da solo non riempie la scheda. None = usa il valore del checkpoint.
    spec_cone_nerf_chunk: "int | None" = None
    # Aperture TOTALI dei coni in gradi, crescenti, primo elemento = 0 (raggio specchio)
    spec_cone_apertures_deg: list[float] = field(
        default_factory=lambda: [0.0, 10.0, 20.0, 40.0, 60.0, 80.0,
                                 100.0, 120.0, 140.0, 160.0, 180.0])
    # Campioni per anello: int = stesso numero su ogni anello (comportamento
    # storico), oppure list[int] con un valore per anello (len = aperture - 1;
    # il livello 0, raggio specchio, è sempre un raggio solo).
    spec_cone_samples_per_ring: int | list[int] = 32
    # Allocazione automatica quando spec_cone_samples_per_ring è un int:
    #   "uniform"     → stesso numero ovunque
    #   "solid_angle" → N_i ∝ Ω_i, cioè densità angolare uniforme. L'anello
    #                   esterno copre ~45× l'angolo solido del primo, quindi con
    #                   M costante è di gran lunga il più rumoroso; il rumore su
    #                   L attenua β e falsa l'argmin su r (errors-in-variables).
    spec_cone_sample_alloc: str = "uniform"
    # Σ_i N_i target per l'allocazione automatica (None → int × numero di anelli)
    spec_cone_samples_budget: int | None = None
    # Minimo per anello. Serve ai candidati stretti, che usano SOLO gli anelli
    # interni: a 32 nessun candidato riceve meno campioni dell'allocazione
    # uniforme storica, al costo del ~3% di raggi in più.
    spec_cone_samples_floor: int = 32
    # Texel per lancio OptiX: tile grandi = meno overhead di launch/sync e batch
    # NeRF più grandi (query_radiance spezza comunque in cfg.chunk). ~40 MB VRAM.
    # Da ridurre se si alza il budget: la RAM per tile scala con tile × raggi/texel.
    spec_cone_tile_size: int = 8192
    spec_cone_cameras: list[int] | None = None  # indici frame da processare (None = tutti)
    spec_cone_format: ImageFormat = ImageFormat.OPENEXR

    # Mappe PBR finali (pbr_solver) — richiede precompute_spec_cone, color_texture
    # con pixel_change e visibility. Salva metallic/metallic.exr (= 1−X) e
    # roughness/roughness.exr (= r/180 dove attendibile, 1.0 altrove), come l'albedo.
    render_pbr_maps: bool = False
    pbr_min_views: int = 2
    pbr_diffuse_cv_gate: float = 0.05  # std tra camere < gate·luminanza → diffuso (0 = gate disattivato)
    pbr_spec_threshold: float = 0.2    # metallic minimo perché r sia attendibile (0 = nessuna censura)
    # Copia metallic/roughness anche come EXR R/G/B (metallic_rgb.exr,
    # roughness_rgb.exr): il canale singolo 'Z' che scrive ExrWriter non è la
    # convenzione dei bake di Blender, che replicano il grigio su tre canali.
    pbr_write_blender_rgb: bool = True

    # Albedo (color_texture / irradiance) — modello Lambertiano ρ = π · L / E
    render_albedo: bool = False
    albedo_format: ImageFormat = ImageFormat.OPENEXR
    albedo_eps: float = 1e-3             # clamp minimo dell'irradiance per evitare /0


@dataclass
class PipelineConfig:
    """Orchestratore a quattro step toggle-abili.

    Step 1: genera depth+mask+immagini+transforms_extended.json (minimo per NeRF).
    Step 2: allena il NeRF (nerf/train.py) e salva il checkpoint.
    Step 3: esegue IUM/visibility/color_texture/irradiance/indirect/spec_cone (bake).
    Step 4: ricostruzione (fit PBR + albedo) leggendo solo la cache su disco dello
            Step 3 — separato per iterare sulla ricostruzione senza ri-bake dei coni.
    """
    run_step1: bool = True
    run_step2: bool = True
    run_step3: bool = True
    run_step4: bool = True
    # Se True, e se run_step2=True, salta il training NeRF per una scena se il
    # checkpoint <output_dir>/model/nerf_model_cache.pt esiste già (utile per
    # riprendere uno sweep interrotto senza ripetere il training).
    # Caveat: un checkpoint incompleto verrebbe riusato; per forzare il
    # riaddestramento basta cancellare il file .pt di quella config.
    resume_skip_step2_if_ckpt: bool = False

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
    nerf_bg_radius_mult:       float = 6.0   # raggio sfera bg = bg_radius_mult × distanza max dall'origine
    nerf_bg_depth_window:      float = 2.0   # finestra bg [R - window, R + window_end]
    nerf_bg_depth_window_end:  float = 2.0
    nerf_profile_iters: int = 0         # per-fase timing sincronizzato per i primi N iter (0=off)
    nerf_multires:       int   = 10
    nerf_multires_views: int   = 4

    # Attivazione RGB e loss del training (Step 2).
    # nerf_rgb_activation: "exp" (HDR) | "softplus"
    # nerf_loss_type:      "l1" | "mse" | "rel_mse" (eps fuori dal quadrato) |
    #                      "rel_mse_raw" (RawNeRF fedele, eps dentro al quadrato) | "log_l1"
    # N.B. checkpoint salvati con un'attivazione NON sono compatibili con l'altra.
    nerf_rgb_activation: str   = "exp"
    nerf_loss_type:      str   = "rel_mse_raw"

    # Fattore di decay del learning rate: new_lr = lr * (nerf_lr_decay ** min(i/decay_steps, 1.0)).
    # 0.2 → lr decade al 20 % del valore iniziale all'orizzonte nerf_lr_decay_steps;
    # oltre la soglia il LR resta a plateau (lr*factor, non scende più).
    # Sweepabile per confrontare regimi di decadimento: valori < 0.2 più aggressivi,
    # valori > 0.2 più gentili. Propagato a NerfConfig.lr_decay_factor.
    nerf_lr_decay: float = 0.2

    # Orizzonte FISSO (iter assolute) su cui si spalma il decay del LR. 0 = auto → usa
    # nerf_num_iters (run fresh identico a prima). Impostarlo a un valore fisso (es.
    # uguale alla lunghezza pianificata totale) per far sì che riprendere il training
    # continui il decay senza salti. Propagato a NerfConfig.lr_decay_steps.
    nerf_lr_decay_steps: int = 0

    # Render dei frame di training col NeRF allenato (post-Step 2)
    enable_nerf_render_train_images: bool = False
    nerf_render_train_images_dir:    str  = ""  # default: <output_dir>/nerf_render_images

    # Se True, chiede all'utente di continuare il training al termine di ogni round
    nerf_interactive_loop: bool = True



@dataclass
class SceneConfig:
    """Campi che variano per scena, usati da run_pipeline_multi."""
    name: str                               # nome della sottocartella di output (es. "SwordShield")
    transforms_path: str
    model_path: str
    external_normal_path: str | None = None  # sovrascrive RenderConfig solo se non None
    skybox_path: str | None = None           # usato solo se skybox_source == "file"
    note: str = ""                           # nota opzionale specifica della scena


def _resolve_nerf_ckpt_path(cfg: RenderConfig) -> str:
    """Path del checkpoint NeRF usato dai pass di Step 3 (indirect, skybox bake)."""
    path = cfg.indirect_nerf_cache_path
    if not path:
        path = os.path.join(cfg.output_dir, "model", "nerf_model_cache.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"NeRF model cache non trovato: {path}\n"
            "Imposta indirect_nerf_cache_path oppure eseguire prima Step 2."
        )
    return path


def _bake_skybox_from_nerf(cfg: RenderConfig, sky_w: int, sky_h: int,
                           json_dir: Path) -> np.ndarray:
    """Bake della bg-sphere del NeRF in envmap equirettangolare (skybox_source="nerf").

    Salva skybox_nerf_baked.exr in json_dir per ispezione e restituisce (N, 3) float32
    nello stesso layout flat di _load_image_as_vec3.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from nerf import load_checkpoint, bake_envmap
    import torch

    ckpt_path = _resolve_nerf_ckpt_path(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_bundle, nerf_cfg = load_checkpoint(ckpt_path, device)
    print(f"[Step 3] Bake skybox da NeRF ({sky_w}×{sky_h}, "
          f"yaw={cfg.skybox_yaw_degrees}°) — ckpt: {ckpt_path}")
    baked = bake_envmap(model_bundle, nerf_cfg, sky_w, sky_h,
                        yaw_degrees=cfg.skybox_yaw_degrees)
    out_path = (json_dir / "skybox_nerf_baked.exr").resolve().as_posix()
    get_writer(ImageFormat.OPENEXR).write(baked, out_path)
    print(f"[Step 3] Skybox bakata salvata: {out_path}")
    return baked.reshape(-1, 3)


def _resolve_skybox_flat(cfg: RenderConfig, output_json: dict, json_dir: Path,
                         sky_w: int, sky_h: int) -> np.ndarray:
    """Skybox flat (H*W, 3) per i pass Step 3 (irradiance, spec_cone).

    skybox_source="nerf" → bake dalla bg-sphere del NeRF; altrimenti preferisce
    lo skybox normalizzato dal JSON (stessa scala di color+NeRF) se presente.
    """
    if cfg.skybox_source == "nerf":
        baked = json_dir / "skybox_nerf_baked.exr"
        if baked.exists():
            print(f"[Step 3] Skybox NeRF riusata da disco: {baked} "
                  "(cancellare il file per forzare il re-bake)")
            return _load_image_as_vec3(baked.as_posix(), sky_w, sky_h)
        return _bake_skybox_from_nerf(cfg, sky_w, sky_h, json_dir)
    norm_sky_rel = output_json.get("normalization", {}).get("normalized_skybox_path", "")
    if norm_sky_rel:
        norm_sky_abs = (json_dir / norm_sky_rel).resolve().as_posix()
        skybox_src = norm_sky_abs if Path(norm_sky_abs).exists() else cfg.skybox_path
    else:
        skybox_src = cfg.skybox_path
    return _load_image_as_vec3(skybox_src, sky_w, sky_h)


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
    cache_path = _resolve_nerf_ckpt_path(cfg)

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


# ──────────────────────────────────────────────────────────────────────────────
# Set Fibonacci condiviso — ricostruzione lato host delle direzioni del kernel
#
# Queste funzioni replicano BIT PER BIT sharedDirection()/buildONB()/
# rotationFromIndex() di deviceProgramsHemiVis.cu: il kernel restituisce solo i
# t_hit, indicizzati per posizione, quindi una divergenza appaierebbe ogni t_hit
# alla direzione sbagliata senza alcun sintomo visibile se non una L_j errata.
# La parità è verificata da test_hemivis_shared.py.
# ──────────────────────────────────────────────────────────────────────────────

_HEMIVIS_INV_GOLDEN = 0.6180339887498948482   # 1/φ = (√5 − 1)/2
_HEMIVIS_TWO_PI     = 6.283185307179586477


def _hemivis_rotation(global_idx: np.ndarray) -> np.ndarray:
    """Rotazione azimutale in [0, 1) per texel (hash lowbias32 di global_idx).

    Decorrela il pattern QMC tra texel vicini: senza, tutti i texel userebbero le
    stesse direzioni a meno della ONB e il rumore si allineerebbe in bande.
    """
    with np.errstate(over="ignore"):          # l'overflow uint32 È la semantica voluta
        x = np.asarray(global_idx, dtype=np.uint32)
        x = x ^ (x >> np.uint32(16))
        x = x * np.uint32(0x7feb352d)
        x = x ^ (x >> np.uint32(15))
        x = x * np.uint32(0x846ca68b)
        x = x ^ (x >> np.uint32(16))
    return (x >> np.uint32(8)).astype(np.float64) * (1.0 / 16777216.0)


def _hemivis_onb(n):
    """ONB branchless di Frisvad 2012 attorno a n (torch, (..., 3) float32)."""
    import torch
    nz  = n[..., 2]
    sgn = torch.copysign(torch.ones_like(nz), nz)
    a   = -1.0 / (sgn + nz)
    b   = n[..., 0] * n[..., 1] * a
    T = torch.stack([1.0 + sgn * n[..., 0] * n[..., 0] * a, sgn * b, -sgn * n[..., 0]], dim=-1)
    B = torch.stack([b, sgn + n[..., 1] * n[..., 1] * a, -n[..., 1]], dim=-1)
    return T, B


def _hemivis_directions(normals, global_idx: np.ndarray, num_samples: int):
    """Direzioni condivise (n_texel, S, 3) float32 attorno alle normali date.

    Uniformi in angolo solido sull'emisfero sopra n: cosθ_s = 1 − (s + 0.5)/S,
    azimut sulla sequenza aurea con rotazione per texel. L'aritmetica dell'azimut
    è in float64 e ridotta in [0, 2π) PRIMA della trigonometria, come nel kernel:
    in float32 s·goldenAngle arriva a ~4·10⁴ rad, dove un ULP vale già 0.23°.
    """
    import torch
    device = normals.device
    S = int(num_samples)

    n = normals / torch.clamp(torch.linalg.norm(normals, dim=-1, keepdim=True), min=1e-8)
    T, B = _hemivis_onb(n)

    s    = torch.arange(S, device=device, dtype=torch.float64)
    cosT = (1.0 - (s + 0.5) / S).to(torch.float32)                    # (S,)
    sinT = torch.sqrt(torch.clamp(1.0 - cosT * cosT, min=0.0))

    rot = torch.as_tensor(_hemivis_rotation(global_idx),
                          device=device, dtype=torch.float64)          # (n_texel,)
    x   = s[None, :] * _HEMIVIS_INV_GOLDEN + rot[:, None]
    x   = x - torch.floor(x)
    phi = (x * _HEMIVIS_TWO_PI).to(torch.float32)                      # (n_texel, S)

    return (T[:, None, :] * (sinT * torch.cos(phi))[..., None]
            + B[:, None, :] * (sinT * torch.sin(phi))[..., None]
            + n[:, None, :] * cosT[None, :, None])


def _sample_envmap_torch(dirs, envmap, sky_size, yaw_offset_u: float):
    """Lookup equirettangolare (torch), identico a sampleEnvmap() dei kernel CUDA.

    Mondo Z-up, Y-forward (Blender): zenith = +Z, u = 0.5 − atan2(dy,dx)/2π.
    envmap: (H*W, 3) float32 sul device; sky_size = [W, H].
    """
    import torch
    w, h = int(sky_size[0]), int(sky_size[1])
    dz = torch.clamp(dirs[..., 2], -1.0, 1.0)

    u = 0.5 - torch.atan2(dirs[..., 1], dirs[..., 0]) * (1.0 / (2.0 * np.pi)) + yaw_offset_u
    u = u - torch.floor(u)
    v = 0.5 - torch.asin(dz) * (1.0 / np.pi)

    px = torch.clamp((u * w).to(torch.int64), 0, w - 1)   # (int) tronca, u ≥ 0 ⇒ floor
    py = torch.clamp((v * h).to(torch.int64), 0, h - 1)
    return envmap[py * w + px]


def spec_cone_shared_ring_samples(apertures_deg, num_samples: int) -> list[float]:
    """Conteggi NOMINALI per anello del bake condiviso: N_i = S·Ω_i/(2π).

    Con campionamento uniforme in angolo solido il numero atteso di campioni in un
    anello è proporzionale al suo angolo solido, quindi i pesi del solver
    W_i = Ω_i/N_i = 2π/S diventano costanti e `ring_weights_mean` collassa sulla
    media cumulativa semplice L(k) = Σ_{i≤k} somma_i / Σ_{i≤k} conteggio_i.
    Scrivere questi valori nel meta è ciò che rende il solver invariato.
    """
    c = np.cos(np.radians(np.asarray(apertures_deg, dtype=np.float64)) * 0.5)
    return [float(num_samples) * float(c[i] - c[i + 1]) for i in range(c.size - 1)]


def ring_weights_mean(cos_edges, k: int,
                      ring_samples: "np.ndarray | None" = None) -> np.ndarray:
    """Pesi ad angolo solido del cono troncato all'anello k (media pura):
    W_i = Ω_i/N_i con Ω_i = 2π(c_{i-1} − c_i) per i ≤ k, 0 oltre, con
    c = clip(cos b, 0, 1) e N_i = raggi lanciati sull'anello i.
    La normalizzazione per Σ_i W_i·valid_i avviene per-texel al momento
    dell'accumulo (i raggi sotto l'orizzonte escono da numeratore e denominatore).

    ring_samples=None (o uniforme) riproduce ESATTAMENTE il comportamento
    storico: con N costante il fattore 1/N si semplifica in num/den, e saltare
    la divisione evita anche il suo errore di arrotondamento.

    Vive qui e non in pbr_solver perché è matematica del bake: da quando
    spec_cone scrive direttamente i coni, il solver non pesa più nulla. I test
    del kernel la importano da questo modulo.
    """
    c = np.clip(np.asarray(cos_edges, dtype=np.float64), 0.0, 1.0)
    w = 2.0 * np.pi * (c[:-1] - c[1:])
    if ring_samples is not None:
        n = np.asarray(ring_samples, dtype=np.float64)
        if n.shape != w.shape:
            raise ValueError(f"ring_samples: attesi {w.size} valori "
                             f"(un anello ciascuno), ricevuti {n.size}")
        if n.min() <= 0.0:
            raise ValueError("ring_samples: ogni anello richiede N_i > 0")
        if n.max() != n.min():      # uniforme → fattore globale, no-op esatto
            w = w / n
    w[k:] = 0.0
    return w


def spec_cone_ring_samples(apertures_deg, samples_per_ring, alloc="uniform",
                           budget=None, floor=32) -> list[int]:
    """Campioni da LANCIARE sugli anelli 1..K-1 (il livello 0 = raggio specchio
    è sempre un raggio solo, quindi non compare qui).

    Se samples_per_ring è una sequenza viene usata così com'è e `alloc` è
    ignorato. Altrimenti l'allocazione è derivata dagli angoli solidi degli
    anelli, Ω_i = 2π(c_{i-1} − c_i) con c = cos(apertura/2):
      "uniform"     → N_i = samples_per_ring per ogni anello
      "solid_angle" → N_i ∝ Ω_i, normalizzato al budget e con clamp al floor,
                      così ogni raggio copre all'incirca lo stesso angolo solido.
    """
    ap = np.asarray(apertures_deg, dtype=np.float64)
    n_rings = ap.size - 1
    if n_rings < 1:
        raise ValueError("spec_cone_apertures_deg richiede almeno 2 valori")

    if not isinstance(samples_per_ring, (int, np.integer)):
        n = [int(x) for x in samples_per_ring]
        if len(n) != n_rings:
            raise ValueError(f"spec_cone_samples_per_ring: {len(n)} valori, "
                             f"attesi {n_rings} (aperture - 1)")
        if min(n) < 1:
            raise ValueError("spec_cone_samples_per_ring: ogni anello richiede "
                             "almeno 1 campione")
        return n

    m = int(samples_per_ring)
    if m < 1:
        raise ValueError("spec_cone_samples_per_ring deve essere >= 1")
    if alloc == "uniform":
        return [m] * n_rings
    if alloc != "solid_angle":
        raise ValueError(f"spec_cone_sample_alloc sconosciuto: {alloc!r} "
                         "(attesi 'uniform' o 'solid_angle')")

    c = np.cos(np.radians(ap) * 0.5)
    omega = 2.0 * np.pi * (c[:-1] - c[1:])
    total = int(budget) if budget is not None else m * n_rings
    n = np.rint(total * omega / omega.sum())
    return [int(max(x, floor)) for x in n]


def spec_cone_level_name(apertures_deg, k: int) -> str:
    """Nome del livello k: l'APERTURA, non l'indice.

    È il nome che si legge nel viewer (tev raggruppa i canali per prefisso e
    mostra un layer per livello), quindi ci va il dato che serve: `cone_045deg`
    e non `cone06`. Vincoli che spiegano la forma esatta:

    - niente punti: nei nomi dei canali EXR il punto separa layer e canale, e
      `cone_007.5deg.R` verrebbe letto come layer `cone_007`, sublayer `5deg`.
      I gradi frazionari usano quindi `p` come separatore decimale;
    - parte intera a 3 cifre con zero-padding, così l'ordine alfabetico dei
      layer coincide con quello angolare (senza padding `cone_5deg` finirebbe
      dopo `cone_180deg`);
    - il livello 0 è il raggio specchio, una direzione delta e non
      un'integrazione su un cono: si chiama per quello che è.
    """
    if k == 0:
        return "cone_000_mirror"
    a = float(apertures_deg[k])
    if a == int(a):
        return f"cone_{int(a):03d}deg"
    frac = f"{a - int(a):.4f}".split(".")[1].rstrip("0")
    return f"cone_{int(a):03d}p{frac}deg"


def spec_cone_channels(apertures_deg) -> "dict[str, type]":
    """Canali dell'EXR dei coni di una camera: L_j(r) per candidato + validità.

    Un solo file per camera e non uno per apertura: nel bake condiviso il loop
    esterno è sul tile, quindi i writer di tutte le camere restano aperti
    contemporaneamente e K+1 file per camera farebbero 840 handle, oltre il
    limite stdio di MSVC (512). I nomi dei livelli vengono da
    spec_cone_level_name, così scrittura e lettura non possono divergere.

    `valid` è il numero totale di raggi validi del texel (quelli sopra
    l'orizzonte, su tutti i livelli): >0 è la stessa maschera per-camera che il
    bake per anelli scriveva in valid.png.
    """
    ch: "dict[str, type]" = {}
    for k in range(len(apertures_deg)):
        name = spec_cone_level_name(apertures_deg, k)
        for c in "RGB":
            ch[f"{name}.{c}"] = np.float16   # radianza: half basta e dimezza il file
    ch["valid"] = np.float32
    return ch


# L_j(r) per ogni candidato, dalle somme e dai conteggi grezzi per anello:
#     candidato 0 = raggio specchio (livello 0 puro)
#     candidato k = media pura ad angolo solido sul cono troncato all'anello k,
#                   L_k = Σ_{i≤k} W_i·somma_i / Σ_{i≤k} W_i·conteggio_i
# Numeratore e denominatore sono cumulativi negli anelli, quindi tutti i
# candidati escono da una sola cumsum. `weights` sono i W_i = Ω_i/N_i NON
# troncati (ring_weights_mean con k = K-1): il troncamento lo fa la cumsum.
# Si parte dalle somme e non dalle medie per anello perché il vecchio percorso
# (bake → medie in half su disco → solver che ri-mediava) quantizzava un
# passaggio intermedio che qui non esiste.

def _cones_from_rings_np(ring_sum: np.ndarray, ring_valid: np.ndarray,
                         weights: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """(N, K, 3), (N, K), (K-1,) → (N, K, 3). Vedi il commento sopra."""
    num = np.cumsum(ring_sum[:, 1:] * weights[None, :, None], axis=1)
    den = np.cumsum(ring_valid[:, 1:] * weights[None, :], axis=1)
    mirror = ring_sum[:, :1] / np.maximum(ring_valid[:, :1, None], 1.0)
    return np.concatenate([mirror, num / np.maximum(den, eps)[..., None]], axis=1)


def _cones_from_rings_torch(ring_sum, ring_valid, weights, eps: float = 1e-12):
    """(…, K, 3), (…, K), (K-1,) → (…, K, 3), tutto su device. Vedi sopra."""
    import torch
    num = torch.cumsum(ring_sum[..., 1:, :] * weights[:, None], dim=-2)
    den = torch.cumsum(ring_valid[..., 1:] * weights, dim=-1)
    mirror = ring_sum[..., :1, :] / torch.clamp(ring_valid[..., :1, None], min=1.0)
    return torch.cat([mirror, num / torch.clamp(den, min=eps)[..., None]], dim=-2)


def _tile_bar(total: int, desc: str):
    """Barra di avanzamento per i loop sui tile dei bake spec_cone.

    Un bake dura ore e i print periodici non dicono né quanto è passato né
    quanto manca; tqdm dà ETA e percentuale in un solo posto. L'output passa dal
    _Tee di _console_to_file, quindi console.log raccoglie anche i frame
    intermedi: mininterval li tiene a uno ogni due secondi.
    """
    from tqdm import tqdm
    return tqdm(total=total, unit="tile", desc=desc, mininterval=2.0,
                dynamic_ncols=True, smoothing=0.05)


def _tile_bar_step(bar, rays_per_tile: int, n: int = 1) -> None:
    """Avanza la barra di n tile aggiornando il throughput.

    Il throughput è in raggi/s e non in tile/s: un tile vale tile_size × S raggi
    nello schema condiviso e tile_size × (1 + Σ N_i) in quello per-camera, quindi
    i tile/s non sono confrontabili tra configurazioni mentre i raggi/s sì.
    """
    bar.update(n)
    elapsed = bar.format_dict["elapsed"]
    if elapsed > 0:
        bar.set_postfix_str(f"{rays_per_tile * bar.n / elapsed / 1e6:.1f} Mraggi/s",
                            refresh=False)


def _precompute_spec_cone(
    cfg: RenderConfig,
    ium_res,            # IUM_Generator.Result
    model,              # OptixProgrammablePasses.TriangleMesh
    ium_w: int,
    ium_h: int,
    frames,             # tf.frames (per le posizioni camera)
    visibility_map: np.ndarray,   # flat (num_pix * n_cams) uint8
    n_cams: int,
    skybox_flat: "np.ndarray | None",
    sky_size: list[int],
    out_dir: Path,
) -> None:
    """Precompute per-anello per il fit PBR  C_j = (a·x/π)·E + (1-x)·L_j  (pbr_solver.py).

    Campionamento ad anelli concentrici attorno al raggio riflesso
    R_j = reflect(v_j, n) (deviceProgramsSpecCone.cu): ogni raggio è tracciato e
    interrogato sul NeRF una volta sola. Gli anelli restano il modo in cui si
    accumula, ma il bake CHIUDE i coni prima di scrivere: su disco va la media
    pura ad angolo solido L_j(r) di ogni candidato (livello 0 = raggio specchio),
    cioè esattamente la grandezza che il solver mette in regressione.
    Miss → envmap (su GPU), hit → NeRF.

    Output in out_dir: cam_{j:03d}.exr con un canale RGB per livello, chiamato
    con la sua apertura (cone_000_mirror, cone_005deg, …), più valid, e
    spec_cone_meta.json (format "cones", scheme "per_camera").
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from nerf import load_checkpoint, query_radiance
    import OptixProgrammablePasses as optix
    import torch

    cache_path = _resolve_nerf_ckpt_path(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_bundle, nerf_cfg = load_checkpoint(cache_path, device)
    if cfg.indirect_override_depth_window:
        nerf_cfg.depth_window     = cfg.indirect_depth_window
        nerf_cfg.depth_window_end = cfg.indirect_depth_window_end
    print(f"✓ NeRF model caricato da: {cache_path}")

    apertures = [float(a) for a in cfg.spec_cone_apertures_deg]
    K = len(apertures)                  # livelli: 0 = specchio, 1..K-1 = coni

    ring_samples = spec_cone_ring_samples(
        apertures, cfg.spec_cone_samples_per_ring,
        alloc=cfg.spec_cone_sample_alloc,
        budget=cfg.spec_cone_samples_budget,
        floor=cfg.spec_cone_samples_floor)
    rays_per_texel = 1 + sum(ring_samples)
    rays_per_tile  = rays_per_texel * cfg.spec_cone_tile_size
    print(f"    campioni/anello {ring_samples} → {rays_per_texel} raggi/texel, "
          f"{rays_per_tile:,} raggi/tile "
          f"(~{rays_per_tile * 24 / 2**20:.0f} MB device, "
          f"~{rays_per_tile * 84 / 2**20:.0f} MB RAM)")
    if rays_per_tile > 4_000_000:
        suggested = max(256, (4_000_000 // rays_per_texel) // 256 * 256)
        print(f"    ⚠  raggi/tile elevato: valutare spec_cone_tile_size={suggested}")

    gen = optix.SpecConeGenerator()
    gen.set_traversable(model)
    gen.set_inputs(ium_res, apertures, ring_samples, cfg.spec_cone_tile_size)
    if skybox_flat is not None:
        gen.set_envmap(skybox_flat.astype(np.float32), sky_size,
                       cfg.skybox_yaw_degrees)
    else:
        print("    ⚠  spec_cone senza skybox: i raggi miss contribuiscono 0")

    num_pix = gen.num_pixels()
    n_tiles = gen.num_tiles()
    tile_sz = cfg.spec_cone_tile_size

    # Bordi degli anelli: coseni delle semi-aperture (nel meta, per il solver)
    cos_b = np.cos(np.radians(np.asarray(apertures)) * 0.5)

    ium_positions_np = ium_res.positions_np.astype(np.float32)
    ium_normals_np   = ium_res.normals_np.astype(np.float32)
    eps = 1e-4

    vis2d = np.asarray(visibility_map, dtype=np.uint8).reshape(num_pix, n_cams)
    cam_indices = (list(cfg.spec_cone_cameras) if cfg.spec_cone_cameras
                   else list(range(len(frames))))

    os.makedirs(out_dir, exist_ok=True)
    # I coni sono HDR multicanale e vengono scritti con IncrementalExrWriter:
    # spec_cone_format resta solo come estensione dichiarata nel meta.
    fmt = ImageFormat.OPENEXR

    # Le camere già su disco vengono saltate, ma il meta è riscritto sempre:
    # con un campionamento diverso si otterrebbero EXR vecchi descritti da
    # ring_samples nuovi, cioè coni normalizzati per N diversi da quelli con cui
    # sono stati chiusi, senza alcun segnale. Meglio fermarsi.
    meta_path = out_dir / "spec_cone_meta.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as fh:
            old_meta = json.load(fh)
        old_rs = old_meta.get("ring_samples")
        old_ap = old_meta.get("apertures_deg")
        if (old_meta.get("format") != "cones" or old_ap != apertures
                or (old_rs is not None and list(old_rs) != ring_samples)):
            raise RuntimeError(
                f"spec_cone: {out_dir} contiene un bake incompatibile\n"
                f"    su disco:  format={old_meta.get('format')}, "
                f"aperture={old_ap}, ring_samples={old_rs}\n"
                f"    richiesto: format=cones, aperture={apertures}, "
                f"ring_samples={ring_samples}\n"
                f"  Cancellare la cartella spec_cone/ oppure ripristinare la "
                f"configurazione precedente. I bake in formato 'rings'/"
                f"'rings_shared' (medie per anello) non sono più leggibili: da "
                f"quando il bake scrive direttamente i coni il solver non li "
                f"ricostruisce più, vanno rifatti.")

    def _cam_path(j: int) -> Path:
        return out_dir / f"cam_{j:03d}{fmt.extension}"

    # Pesi Ω_i/N_i non troncati: il troncamento a ogni candidato lo fa la cumsum
    # dentro _cones_from_rings_np. Qui gli N_i sono quelli davvero lanciati e in
    # generale non uniformi, quindi il fattore 1/N_i conta.
    cone_w = ring_weights_mean(cos_b, K - 1, np.asarray(ring_samples, dtype=np.float64))

    # Una sola barra su camere × tile: una barra per camera si riaprirebbe 60
    # volte senza mai dare un ETA sull'intero bake. Le camere già su disco
    # restano fuori dal totale, altrimenti l'ETA iniziale conterebbe lavoro che
    # non verrà mai fatto.
    pending = [j for j in cam_indices if not _cam_path(j).exists()]
    bar = _tile_bar(len(pending) * n_tiles, "spec_cone")

    for j in cam_indices:
        cam_path = _cam_path(j)
        if j not in pending:
            print(f"    cam {j}: già su disco, skip")
            continue
        bar.set_description(f"spec_cone cam {j}", refresh=False)

        m = frames[j].transform_matrix
        cam_pos = [float(m[0][3]), float(m[1][3]), float(m[2][3])]
        gen.set_camera(cam_pos, np.ascontiguousarray(vis2d[:, j]))

        ring_sum = np.zeros((num_pix, K, 3), dtype=np.float64)
        valid    = np.zeros((num_pix, K),    dtype=np.int64)

        for tile_idx in range(n_tiles):
            tile_res = gen.render_tile(tile_idx)
            if tile_res.overflow:
                raise RuntimeError(
                    f"spec_cone cam {j} tile {tile_idx}: overflow del buffer "
                    f"compatto ({tile_res.requested} raggi richiesti). La "
                    f"capacità è il worst case esatto, quindi il bake sarebbe "
                    f"incompleto: interrotto invece di salvare medie parziali.")
            off = tile_idx * tile_sz
            tt  = tile_res.tile_texels
            ring_sum[off:off + tt] += tile_res.sky_sum_np.astype(np.float64)
            valid[off:off + tt]    += tile_res.valid_count_np

            if tile_res.count > 0:
                local_idx = tile_res.local_idx_np.copy()
                ring_idx  = tile_res.ring_idx_np.copy()
                dirs_np   = tile_res.directions_np.copy()
                t_hit_np  = tile_res.t_hit_np.copy()

                global_idx = off + local_idx
                origins_np = (ium_positions_np[global_idx]
                              + ium_normals_np[global_idx] * eps)
                colors = query_radiance(model_bundle, origins_np, dirs_np,
                                        nerf_cfg, t_hits_np=t_hit_np)
                colors = np.asarray(colors, dtype=np.float64)
                # accumulo via bincount locale al tile (np.add.at è unbuffered
                # e molto più lento su milioni di indici)
                flat_idx = local_idx.astype(np.int64) * K + ring_idx
                n_bins   = tt * K
                tile_acc = ring_sum[off:off + tt].reshape(n_bins, 3)
                for c in range(3):
                    tile_acc[:, c] += np.bincount(flat_idx, weights=colors[:, c],
                                                  minlength=n_bins)

            _tile_bar_step(bar, rays_per_tile)

        # Chiusura dei coni: L_j(r) per candidato, 0 dove nessun campione
        valid_f = valid.astype(np.float64)
        cones = _cones_from_rings_np(ring_sum, valid_f, cone_w).astype(np.float32)
        n_valid = valid_f.sum(axis=1)
        cones[n_valid <= 0] = 0.0

        with IncrementalExrWriter(cam_path.resolve().as_posix(), ium_w, ium_h,
                                  spec_cone_channels(apertures)) as wr:
            block = {}
            for k in range(K):
                name = spec_cone_level_name(apertures, k)
                for ci, c in enumerate("RGB"):
                    block[f"{name}.{c}"] = _reshape_flat(cones[:, k, ci],
                                                         ium_w, ium_h)
            block["valid"] = _reshape_flat(n_valid.astype(np.float32), ium_w, ium_h)
            wr.write_block(block)
        print(f"    ✓ cam {j}: {K} coni salvati in {cam_path.name}")

    bar.close()

    meta = {
        "format": "cones",
        "scheme": "per_camera",
        "apertures_deg": apertures,
        "ring_edges_cos": [float(c) for c in cos_b],
        # samples_per_ring resta uno scalare informativo per i lettori storici;
        # ring_samples sono gli N_i con cui il bake ha pesato gli anelli (Ω_i/N_i)
        # nel chiudere i coni: documentazione del bake, non un input del solver.
        "samples_per_ring": int(ring_samples[0]) if len(set(ring_samples)) == 1
                            else int(max(ring_samples)),
        "ring_samples": [int(x) for x in ring_samples],
        "samples_total_per_texel": int(rays_per_texel),
        "sample_alloc": cfg.spec_cone_sample_alloc,
        "cameras": [int(j) for j in cam_indices],
        "num_levels": K,
        "cam_file_pattern": "cam_{cam:03d}" + fmt.extension,
        "skybox_yaw_degrees": cfg.skybox_yaw_degrees,
    }
    with open(out_dir / "spec_cone_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"✓ spec_cone meta salvato: {out_dir / 'spec_cone_meta.json'}")


# ──────────────────────────────────────────────────────────────────────────────
# Bake spec_cone a campionamento CONDIVISO tra camere
# ──────────────────────────────────────────────────────────────────────────────

def _precompute_spec_cone_shared(
    cfg: RenderConfig,
    ium_res,            # IUM_Generator.Result
    model,              # OptixProgrammablePasses.TriangleMesh
    ium_w: int,
    ium_h: int,
    frames,             # tf.frames (per le posizioni camera)
    visibility_map: np.ndarray,   # flat (num_pix * n_cams) uint8
    n_cams: int,
    skybox_flat: "np.ndarray | None",
    sky_size: list[int],
    out_dir: Path,
) -> None:
    """Variante di _precompute_spec_cone con i raggi condivisi tra tutte le camere.

    La radianza incidente lungo una direzione non dipende dalla camera, quindi un
    unico set Fibonacci per texel (uniforme in angolo solido sull'emisfero sopra n)
    serve tutte le camere che vedono quel texel: ogni raggio è tracciato e
    interrogato sul NeRF UNA volta, e ogni camera lo classifica nel proprio anello
    in base all'angolo con il suo R_j. Costo per texel `S + m` invece di `m · Σ N_i`
    (m = camere che vedono il texel).

    Il livello 0 (specchio) resta per-camera: è una direzione delta, non
    condivisibile, e si ottiene dalla seconda passata del kernel.

    Uscite in out_dir: cam_{j:03d}.exr con un canale RGB per livello, chiamato
    con la sua apertura (cone_000_mirror, cone_005deg, …), che contiene la
    radianza media sul cono, cioè direttamente la L_j(r) che il solver mette in
    regressione, più valid (raggi validi totali del texel); scritti in streaming
    per blocchi di scanline, e spec_cone_meta.json con format "cones".
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from nerf import load_checkpoint, query_radiance
    import OptixProgrammablePasses as optix
    import torch

    cache_path = _resolve_nerf_ckpt_path(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_bundle, nerf_cfg = load_checkpoint(cache_path, device)
    if cfg.indirect_override_depth_window:
        nerf_cfg.depth_window     = cfg.indirect_depth_window
        nerf_cfg.depth_window_end = cfg.indirect_depth_window_end
    if cfg.spec_cone_nerf_chunk:
        nerf_cfg.chunk = int(cfg.spec_cone_nerf_chunk)
    print(f"✓ NeRF model caricato da: {cache_path} (query chunk {nerf_cfg.chunk})")

    apertures = [float(a) for a in cfg.spec_cone_apertures_deg]
    K = len(apertures)                       # livelli: 0 = specchio, 1..K-1 = anelli
    S = int(cfg.spec_cone_shared_samples)
    cos_b = np.cos(np.radians(np.asarray(apertures)) * 0.5)
    ring_nominal = spec_cone_shared_ring_samples(apertures, S)

    num_pix  = ium_w * ium_h
    tile_sz  = int(cfg.spec_cone_tile_size)
    if tile_sz % ium_w != 0:
        raise ValueError(
            f"spec_cone_tile_size={tile_sz} deve essere multiplo della larghezza IUM "
            f"({ium_w}): il bake condiviso scrive gli EXR in streaming e ogni tile "
            f"deve coprire un numero intero di scanline.")
    chunk_texels = max(1, int(cfg.spec_cone_chunk_texels))

    vis2d = np.asarray(visibility_map, dtype=np.uint8).reshape(num_pix, n_cams)
    cam_indices = (list(cfg.spec_cone_cameras) if cfg.spec_cone_cameras
                   else list(range(len(frames))))
    n_sel = len(cam_indices)

    # ── Diagnostica m: quante camere vedono in media un texel ────────────────
    # È il numero che decide il costo relativo al bake per-camera: quello spende
    # m·Σ N_i raggi per texel, questo S + m.
    ium_mask = np.asarray(ium_res.masks_np).reshape(num_pix) > 0
    if ium_mask.any():
        m_per_texel = vis2d[np.ix_(ium_mask, cam_indices)].sum(axis=1)
        m_mean = float(m_per_texel.mean())
        print(f"    m (camere per texel): media {m_mean:.1f}, mediana "
              f"{np.median(m_per_texel):.0f}, p10 {np.percentile(m_per_texel, 10):.0f}, "
              f"p90 {np.percentile(m_per_texel, 90):.0f}")
        try:
            # confronto informativo col bake per-camera; i suoi parametri possono
            # essere incoerenti con questa griglia (non sono usati da questo schema),
            # e una diagnostica non deve far fallire il bake
            per_cam_rays = 1 + sum(spec_cone_ring_samples(
                apertures, cfg.spec_cone_samples_per_ring,
                alloc=cfg.spec_cone_sample_alloc,
                budget=cfg.spec_cone_samples_budget,
                floor=cfg.spec_cone_samples_floor))
            print(f"    costo atteso vs per-camera: {m_mean * per_cam_rays:.0f} → "
                  f"{S + m_mean:.0f} raggi/texel "
                  f"({m_mean * per_cam_rays / max(S + m_mean, 1.0):.2f}×)")
        except ValueError:
            print(f"    costo atteso: {S + m_mean:.0f} raggi/texel")

    # Campioni attesi per candidato: dice quali aperture sono al limite del rumore
    cum = [f"{apertures[k]:g}°:{S * (1.0 - cos_b[k]):.0f}" for k in range(1, K)]
    print(f"    S={S} raggi/texel condivisi, campioni per candidato → {', '.join(cum)}")

    # ── Guardia sul ri-bake incoerente ───────────────────────────────────────
    # Come nel bake per-camera: il meta viene riscritto sempre, quindi un bake su
    # disco con parametri diversi risulterebbe descritto da un meta nuovo e il
    # solver normalizzerebbe per N sbagliati senza alcun segnale.
    meta_path = out_dir / "spec_cone_meta.json"
    cam_paths = {j: out_dir / f"cam_{j:03d}{ImageFormat.OPENEXR.extension}"
                 for j in cam_indices}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as fh:
            old_meta = json.load(fh)
        same = (old_meta.get("format") == "cones"
                and old_meta.get("scheme") == "shared"
                and old_meta.get("apertures_deg") == apertures
                and old_meta.get("shared_samples") == S)
        if not same:
            raise RuntimeError(
                f"spec_cone: {out_dir} contiene un bake incompatibile\n"
                f"    su disco:  format={old_meta.get('format')}, "
                f"scheme={old_meta.get('scheme')}, "
                f"aperture={old_meta.get('apertures_deg')}, "
                f"S={old_meta.get('shared_samples')}\n"
                f"    richiesto: format=cones, scheme=shared, "
                f"aperture={apertures}, S={S}\n"
                f"  Cancellare la cartella spec_cone/ oppure ripristinare la "
                f"configurazione precedente. I bake in formato 'rings'/"
                f"'rings_shared' (medie per anello) non sono più leggibili: da "
                f"quando il bake scrive direttamente i coni il solver non li "
                f"ricostruisce più, vanno rifatti.")
        if all(p.exists() for p in cam_paths.values()):
            print(f"    tutte le {n_sel} camere già su disco, skip")
            return

    # ── Setup OptiX ──────────────────────────────────────────────────────────
    gen = optix.HemiVisGenerator()
    gen.set_traversable(model)
    gen.set_inputs(ium_res, S, tile_sz)

    cam_pos_list = []
    for j in cam_indices:
        m = frames[j].transform_matrix
        cam_pos_list.append([float(m[0][3]), float(m[1][3]), float(m[2][3])])
    gen.set_cameras(cam_pos_list)

    if gen.num_pixels() != num_pix:
        raise RuntimeError(f"spec_cone: IUM ha {gen.num_pixels()} texel, attesi "
                           f"{num_pix} da ium_texture_size {ium_w}×{ium_h}")

    n_tiles = gen.num_tiles()
    rays_per_tile = tile_sz * S
    print(f"    tile={tile_sz} texel ({tile_sz // ium_w} scanline), {n_tiles} tile, "
          f"{rays_per_tile:,} raggi/tile (~{rays_per_tile * 4 / 2**20:.0f} MB t_hit), "
          f"chunk torch={chunk_texels} texel "
          f"(~{chunk_texels * S * 24 / 2**20:.0f} MB VRAM per dirs+radianze)")

    if skybox_flat is None:
        print("    ⚠  spec_cone senza skybox: i raggi miss contribuiscono 0")
        envmap_t = None
    else:
        envmap_t = torch.as_tensor(np.ascontiguousarray(skybox_flat, dtype=np.float32),
                                   device=device)
    yaw_u = cfg.skybox_yaw_degrees / 360.0

    pos_all = np.asarray(ium_res.positions_np, dtype=np.float32)
    nrm_all = np.asarray(ium_res.normals_np,   dtype=np.float32)
    eps = 1e-4

    cos_edges_t = torch.as_tensor(cos_b, device=device, dtype=torch.float32)
    asc_edges   = -cos_edges_t                       # crescente, per searchsorted
    cam_pos_t   = torch.as_tensor(np.asarray(cam_pos_list, dtype=np.float32), device=device)
    vis_sel     = np.ascontiguousarray(vis2d[:, cam_indices])       # (num_pix, n_sel)

    # Pesi Ω_i/N_i non troncati: il troncamento a ogni candidato lo fa la cumsum
    # dentro _cones_from_rings_torch. Nel bake condiviso gli N_i nominali rendono
    # W_i costante, quindi la formula collassa sulla media cumulativa semplice.
    cone_w_t = torch.as_tensor(
        ring_weights_mean(cos_b, K - 1, np.asarray(ring_nominal)),
        device=device, dtype=torch.float32)

    # ── Writer in streaming, uno per camera ──────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    channels = spec_cone_channels(apertures)
    level_names = [spec_cone_level_name(apertures, k) for k in range(K)]
    writers = {j: IncrementalExrWriter(cam_paths[j].resolve().as_posix(),
                                       ium_w, ium_h, channels)
               for j in cam_indices}

    bar = _tile_bar(n_tiles, "spec_cone shared")
    try:
        for tile_idx in range(n_tiles):
            tile_res = gen.render_tile(tile_idx)
            off = tile_idx * tile_sz
            tt  = tile_res.tile_texels
            t_hit_tile = tile_res.t_hit_np          # (tt, S)
            t_mir_tile = tile_res.t_hit_mirror_np   # (tt, n_sel)

            sums   = torch.zeros((n_sel, tt, K, 3), device=device, dtype=torch.float32)
            counts = torch.zeros((n_sel, tt, K),    device=device, dtype=torch.float32)

            for c0 in range(0, tt, chunk_texels):
                c1 = min(c0 + chunk_texels, tt)
                M  = c1 - c0
                gidx = np.arange(off + c0, off + c1, dtype=np.int64)

                nrm = torch.as_tensor(nrm_all[gidx], device=device)
                pos = torch.as_tensor(pos_all[gidx], device=device)
                nlen = torch.linalg.norm(nrm, dim=-1, keepdim=True)
                texel_ok = (torch.as_tensor(ium_mask[gidx], device=device)
                            & (nlen[:, 0] > 1e-8))
                if not bool(texel_ok.any()):
                    continue
                n_unit = nrm / torch.clamp(nlen, min=1e-8)
                origin = pos + n_unit * eps

                dirs = _hemivis_directions(nrm, gidx, S)                  # (M, S, 3)
                th   = torch.as_tensor(t_hit_tile[c0:c1], device=device)  # (M, S)
                rad  = _shared_ray_radiance(dirs, th, origin, envmap_t, sky_size,
                                            yaw_u, model_bundle, nerf_cfg,
                                            query_radiance)               # (M, S, 3)

                # Raggi specchio: R_j per camera, radianza con la stessa logica
                v = cam_pos_t[None, :, :] - pos[:, None, :]               # (M, n_sel, 3)
                v = v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=1e-8)
                nv = (n_unit[:, None, :] * v).sum(-1, keepdim=True)
                R  = n_unit[:, None, :] * (2.0 * nv) - v                  # (M, n_sel, 3)
                thm = torch.as_tensor(t_mir_tile[c0:c1], device=device)   # (M, n_sel)
                radm = _shared_ray_radiance(R, thm, origin, envmap_t, sky_size,
                                            yaw_u, model_bundle, nerf_cfg,
                                            query_radiance)               # (M, n_sel, 3)

                vis_chunk = torch.as_tensor(vis_sel[gidx], device=device) > 0  # (M, n_sel)

                for jj in range(n_sel):
                    sel_mask = vis_chunk[:, jj] & texel_ok
                    sel = sel_mask.nonzero(as_tuple=True)[0]
                    if sel.numel() == 0:
                        continue

                    # Anelli 1..K-1 dai raggi condivisi. I campioni oltre il cono
                    # più largo finiscono nel bin K, che viene poi scartato: così
                    # si evita una gather booleana su M·S elementi.
                    m_sel  = sel.numel()
                    cosang = (dirs[sel] * R[sel, jj][:, None, :]).sum(-1)      # (m, S)
                    # clamp a [1, K]: 0 è il livello specchio (mai raggiungibile
                    # dai condivisi, ma cosang può sforare 1 per arrotondamento),
                    # K è il bin di scarto per i campioni fuori dal cono più largo.
                    ring = torch.searchsorted(asc_edges, -cosang.contiguous(),
                                              right=True).clamp_(min=1, max=K)
                    flat = (torch.arange(m_sel, device=device)[:, None] * (K + 1)
                            + ring).reshape(-1)

                    acc_s = torch.zeros((m_sel * (K + 1), 3), device=device)
                    acc_c = torch.zeros(m_sel * (K + 1), device=device)
                    acc_s.index_add_(0, flat, rad[sel].reshape(-1, 3))
                    acc_c.index_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))

                    tgt = sel + c0
                    sums[jj].index_add_(0, tgt, acc_s.view(m_sel, K + 1, 3)[:, :K])
                    counts[jj].index_add_(0, tgt, acc_c.view(m_sel, K + 1)[:, :K])

                    # Livello 0: raggio specchio (t_hit < 0 = camera dietro la superficie)
                    mir_ok = sel[thm[sel, jj] >= 0.0]
                    if mir_ok.numel() > 0:
                        lvl0 = torch.zeros((mir_ok.numel(), K, 3), device=device)
                        lvl0[:, 0] = radm[mir_ok, jj]
                        cnt0 = torch.zeros((mir_ok.numel(), K), device=device)
                        cnt0[:, 0] = 1.0
                        sums[jj].index_add_(0, mir_ok + c0, lvl0)
                        counts[jj].index_add_(0, mir_ok + c0, cnt0)

            # ── Scrittura del blocco di scanline, una camera alla volta ──────
            # I coni si chiudono qui, sulla GPU e dalle somme grezze: su disco va
            # già L_j(r), la grandezza che il solver mette in regressione.
            rows_out = tt // ium_w
            for jj, j in enumerate(cam_indices):
                cnt = counts[jj]
                cones = _cones_from_rings_torch(sums[jj], cnt, cone_w_t)
                cones_np = cones.cpu().numpy().reshape(rows_out, ium_w, K, 3)
                valid_np = cnt.sum(dim=-1).cpu().numpy().reshape(rows_out, ium_w)
                block = {}
                for k in range(K):
                    for ci, c in enumerate("RGB"):
                        block[f"{level_names[k]}.{c}"] = cones_np[:, :, k, ci]
                block["valid"] = valid_np
                writers[j].write_block(block)

            _tile_bar_step(bar, rays_per_tile)

        bar.close()
        for wr in writers.values():
            wr.close()
    except BaseException:
        bar.close()
        # Un EXR troncato ha comunque un header valido: se restasse su disco, un
        # rerun con lo stesso meta lo scambierebbe per un bake completo e lo
        # salterebbe. Meglio cancellarlo.
        for wr in writers.values():
            wr._file = None
        for p in cam_paths.values():
            p.unlink(missing_ok=True)
        raise

    meta = {
        "format": "cones",
        "scheme": "shared",
        "apertures_deg": apertures,
        "ring_edges_cos": [float(c) for c in cos_b],
        "shared_samples": S,
        # Conteggi NOMINALI N_i = S·Ω_i/2π usati per i pesi W_i = Ω_i/N_i del
        # bake: qui sono costanti, quindi il cono è la media cumulativa semplice.
        # Ormai è documentazione del bake, non un input del solver.
        "ring_samples": ring_nominal,
        "samples_total_per_texel": int(S + 1),
        "sample_alloc": "shared_fibonacci",
        "cameras": [int(j) for j in cam_indices],
        "num_levels": K,
        "cam_file_pattern": "cam_{cam:03d}" + ImageFormat.OPENEXR.extension,
        "skybox_yaw_degrees": cfg.skybox_yaw_degrees,
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"✓ spec_cone meta salvato: {meta_path}")


def _shared_ray_radiance(dirs, t_hit, origin, envmap_t, sky_size, yaw_u,
                         model_bundle, nerf_cfg, query_radiance):
    """Radianza incidente per raggio: envmap sui miss, NeRF sugli hit.

    dirs (M, R, 3), t_hit (M, R) con >0 hit, =0 miss, <0 raggio non lanciato;
    origin (M, 3) è l'origine comune ai raggi dello stesso texel.
    Restituisce (M, R, 3) sul device (zero dove il raggio non è stato lanciato).
    """
    import torch
    rad = torch.zeros(dirs.shape, device=dirs.device, dtype=torch.float32)

    miss = t_hit == 0.0
    if envmap_t is not None and bool(miss.any()):
        rad[miss] = _sample_envmap_torch(dirs[miss], envmap_t, sky_size, yaw_u)

    hit = t_hit > 0.0
    if bool(hit.any()):
        origins = origin[:, None, :].expand(-1, dirs.shape[1], -1)
        rad[hit] = query_radiance(model_bundle, origins[hit], dirs[hit], nerf_cfg,
                                  t_hits_np=t_hit[hit], return_torch=True)
    return rad


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


def _build_optix_frames_for_source(
    source: str,
    tf,
    all_cameras: list,
    intr,
    rc: "RenderConfig",
    cfg: "PipelineConfig",
    json_dir: Path,
    optix_mod,
) -> list:
    """Costruisce la lista di optix_mod.Frame usando le immagini della sorgente indicata.

    Args:
        source: "gt" (immagini originali) | "nerf" (pred EXR dallo Step 2b).
        tf: oggetto transforms caricato (tf.frames).
        all_cameras: lista di optix_mod.Camera, una per frame.
        intr: intrinseche (intr.w, intr.h).
        rc: RenderConfig corrente.
        cfg: PipelineConfig corrente (per nerf_render_train_images_dir).
        json_dir: cartella base del run (Path).
        optix_mod: modulo OptixProgrammablePasses importato.
    """
    nerf_pred_dir = None
    if source == "nerf":
        base_root = Path(cfg.nerf_render_train_images_dir or
                         json_dir / "nerf_render_images")
        nerf_pred_dir = _find_nerf_pred_dir(base_root, rc.color_texture_nerf_iter)
        if nerf_pred_dir is None:
            print(f"    ⚠  Nessuna cartella pred NeRF trovata (sorgente '{source}') → uso immagini GT")
        else:
            print(f"[Step 3] Color texture da pred NeRF ({source}): {nerf_pred_dir}")

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
    return optix_frames


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
    depth_gen.need_render_depth(rc.render_depth)
    depth_gen.need_render_position(rc.render_position)
    depth_gen.need_render_normal(rc.render_normal)

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

        if rc.render_depth and result.has_depth_data():
            depth_arr = _reshape_flat(result.depths_np.astype(np.float32), W, H)
            if rc.apply_scale:
                depth_arr = depth_arr * scale
            out_path = _build_output_path(rc.output_dir, frame.stem, "depth", rc.depth_format)
            _save_layer(depth_arr, out_path, rc.depth_format, DataLayer.DEPTH)
            frame_entry["depth_path"] = _as_relative_to(out_path, json_dir_str)

        if rc.render_position and result.has_positional_data():
            pos_arr = _reshape_flat(result.positions_np.astype(np.float32), W, H)
            if rc.apply_scale:
                pos_arr = pos_arr * scale
            out_path = _build_output_path(rc.output_dir, frame.stem, "position", rc.position_format)
            _save_layer(pos_arr, out_path, rc.position_format, DataLayer.POSITION)
            frame_entry["position_path"] = _as_relative_to(out_path, json_dir_str)

        if rc.render_normal and result.has_normal_data():
            norm_arr = _reshape_flat(result.normals_np.astype(np.float32), W, H)
            out_path = _build_output_path(rc.output_dir, frame.stem, "normal", rc.normal_format)
            _save_layer(norm_arr, out_path, rc.normal_format, DataLayer.NORMAL)
            frame_entry["normal_path"] = _as_relative_to(out_path, json_dir_str)

        if rc.render_mask:
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


def _step2_train_nerf(
    cfg: PipelineConfig,
    transforms_extended_path: Path,
    tb_logger=None,
) -> tuple[Path, float]:
    """Allena il NeRF e restituisce (ckpt_path, final_psnr_dB).

    ``tb_logger`` è un monitoring.RunLogger (o None per disabilitare il log TB).
    Il PSNR è quello dell'ultimo blocco di display; float('nan') se il training è
    troppo breve per raggiungere il primo blocco.
    """
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
        rgb_activation            = cfg.nerf_rgb_activation,
        loss_type                 = cfg.nerf_loss_type,
        multires                  = cfg.nerf_multires,
        multires_views            = cfg.nerf_multires_views,
        lr_decay_factor           = cfg.nerf_lr_decay,
        lr_decay_steps            = cfg.nerf_lr_decay_steps,
    )

    print(f"[Step 2] Training NeRF (depth-guided) — {cfg.nerf_num_iters} iter, ckpt → {ckpt}")
    print(f"[Step 2] campionamento depth-guided, mesh_window=[t-{cfg.nerf_depth_window}, t+{cfg.nerf_depth_window_end}], "
          f"bg_radius_mult={cfg.nerf_bg_radius_mult}, lr_decay={cfg.nerf_lr_decay}, "
          f"lr_decay_steps={cfg.nerf_lr_decay_steps or cfg.nerf_num_iters} "
          f"({'auto' if cfg.nerf_lr_decay_steps == 0 else 'fisso'})")
    final_psnr = nerf_train(
        str(transforms_extended_path), nerf_cfg,
        ckpt_path     = str(ckpt),
        output_dir    = str(out_dir),
        num_iters     = cfg.nerf_num_iters,
        batch_size    = cfg.nerf_batch_size,
        lr            = cfg.nerf_lr,
        seed          = cfg.nerf_seed,
        display_every = cfg.nerf_display_every,
        tb_logger     = tb_logger,
    )
    print(f"[Step 2] Training completato. Checkpoint: {ckpt}")
    return ckpt, (final_psnr if final_psnr is not None else float("nan"))


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
    """Save a three-panel PNG: pred histogram (top), GT histogram (middle),
    and |Pred − GT| histogram (bottom), shared x axis. Pred and GT share
    the same Y scale; the diff panel uses its own auto scale."""
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
    fig, (ax_pred, ax_gt, ax_diff) = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    ymax = 0
    for ch, (label, color) in enumerate(_RGB_HIST_COLORS):
        c_pred, _ = np.histogram(pred_flat[:, ch], bins=edges)
        ax_pred.stairs(c_pred, edges, fill=True, alpha=0.55, color=color, label=label)
        c_gt, _ = np.histogram(gt_flat[:, ch], bins=edges)
        ax_gt.stairs(c_gt, edges, fill=True, alpha=0.55, color=color, label=label)
        c_diff = np.abs(c_pred.astype(np.int64) - c_gt.astype(np.int64))
        ax_diff.stairs(c_diff, edges, fill=True, alpha=0.55, color=color, label=label)
        ymax = max(ymax, int(c_pred.max()), int(c_gt.max()))
    ax_pred.set_ylim(0, ymax * 1.05)
    ax_gt.set_ylim(0, ymax * 1.05)
    ax_pred.set_ylabel("pixel count")
    ax_pred.set_title(f"{stem} — Pred")
    ax_pred.legend(loc="upper right")
    ax_gt.set_ylabel("pixel count")
    ax_gt.set_title(f"{stem} — GT")
    ax_gt.legend(loc="upper right")
    ax_diff.set_xlabel("pixel value (linear HDR)")
    ax_diff.set_ylabel("pixel count")
    ax_diff.set_title(f"{stem} — |Pred − GT|")
    ax_diff.legend(loc="upper right")
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

    Metriche di bias colore prodotte al termine:
      frame_NNN_bias.png   — scatter densità pred-vs-gt (4 pannelli R/G/B/Luma, log-log)
      bias_scatter_all.png — scatter aggregato su tutti i frame
      metrics_per_frame.csv — PSNR, PSNR-tonemap, errore percentili, residui per frame
      bias_bins.csv         — ratio mediana pred/gt per fascia di luminanza e canale
      metrics_summary.txt   — riepilogo testuale (numeri per la tesi)
    """
    import csv
    import sys
    import torch as _torch
    sys.path.insert(0, str(Path(__file__).parent))
    from nerf import load_checkpoint, render_image as nerf_render_image
    from nerf.metrics import (
        plot_bias_scatter, plot_error_heatmap,
        tonemapped_psnr, highlight_percentile_error,
        signed_residual_stats, binned_median_curve, _LUMA_COEFF,
    )

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

    # ── Accumulatori per scatter aggregato (subsample per contenere la memoria) ──
    _MAX_AGG_PX = 20_000   # pixel massimi prelevati da ogni frame per l'aggregato
    agg_pred_rgb: list[np.ndarray] = []   # (N_i, 3) per frame i
    agg_gt_rgb:   list[np.ndarray] = []

    # ── Metriche per-frame ────────────────────────────────────────────────────
    psnrs: list[float] = []
    metrics_rows: list[dict] = []

    print(f"\n[Step 2b] Rendering {len(tf.frames)} frame con NeRF (iter={iter_done})...")
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

        # ── Metriche di bias colore ───────────────────────────────────────────
        psnr_tm_clip     = tonemapped_psnr(pred_np, gt_np, mask_bool, mode="clip")
        psnr_tm_reinhard = tonemapped_psnr(pred_np, gt_np, mask_bool, mode="reinhard")
        hl_err           = highlight_percentile_error(pred_np, gt_np, mask_bool)
        res_stats        = signed_residual_stats(pred_np, gt_np, mask_bool)

        metrics_rows.append({
            "frame":               i,
            "psnr":                round(psnr, 4),
            "psnr_tonemap_clip":   round(psnr_tm_clip, 4),
            "psnr_tonemap_reinhard": round(psnr_tm_reinhard, 4),
            "rel_err_p99":         round(hl_err.get(99.0,  float("nan")), 5),
            "rel_err_p999":        round(hl_err.get(99.9,  float("nan")), 5),
            "residual_mean":       round(res_stats["mean"],             6),
            "residual_median":     round(res_stats["median"],           6),
            "residual_mean_hl":    round(res_stats["mean_highlight"],   6),
            "residual_median_hl":  round(res_stats["median_highlight"], 6),
        })

        # Scatter bias per-frame (R/G/B/Luma log-log con bisettrice + curva mediana)
        bias_title = (f"{stem}  PSNR={psnr:.2f} dB  "
                      f"tonemap-clip={psnr_tm_clip:.2f} dB  "
                      f"tonemap-Reinhard={psnr_tm_reinhard:.2f} dB")
        plot_bias_scatter(pred_np, gt_np, mask_bool,
                          str(base / f"{stem}_bias.png"), title=bias_title)

        # Heatmap diagnostica per-frame: GT, Pred (clip [0,1]) + ΔR ΔG ΔB + |Δ| luma
        # Nessuna maschera: l'intero frame entra (modello + skybox/background).
        plot_error_heatmap(pred_np, gt_np,
                           str(base / f"{stem}_heatmap.png"), title=bias_title)

        # Accumulo subsample per scatter aggregato (già mascherato)
        p3 = pred_np.astype(np.float32)
        g3 = gt_np.astype(np.float32)
        if mask_bool is not None:
            p3 = p3[mask_bool]
            g3 = g3[mask_bool]
        else:
            p3 = p3.reshape(-1, 3)
            g3 = g3.reshape(-1, 3)
        n_px = p3.shape[0]
        if n_px > _MAX_AGG_PX:
            rng_idx = np.random.choice(n_px, _MAX_AGG_PX, replace=False)
            p3 = p3[rng_idx]
            g3 = g3[rng_idx]
        agg_pred_rgb.append(p3)
        agg_gt_rgb.append(g3)

        if (i + 1) % max(1, len(tf.frames) // 5) == 0 or i == len(tf.frames) - 1:
            print(f"  [Step 2b] {i+1}/{len(tf.frames)}  PSNR={psnr:.2f} dB  "
                  f"tonemap={psnr_tm_clip:.2f} dB")

    # ── metrics_per_frame.csv ─────────────────────────────────────────────────
    _csv_fields = [
        "frame", "psnr", "psnr_tonemap_clip", "psnr_tonemap_reinhard",
        "rel_err_p99", "rel_err_p999",
        "residual_mean", "residual_median", "residual_mean_hl", "residual_median_hl",
    ]
    csv_path = base / "metrics_per_frame.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_csv_fields)
        writer.writeheader()
        writer.writerows(metrics_rows)

    # ── Scatter aggregato + bias_bins.csv ─────────────────────────────────────
    all_pred = np.concatenate(agg_pred_rgb, axis=0)   # (N_total, 3)
    all_gt   = np.concatenate(agg_gt_rgb,   axis=0)
    n_agg    = all_pred.shape[0]

    # Scatter aggregato — reshaping virtuale (1, N, 3) senza maschera (già filtrato)
    agg_pred_hw3 = all_pred.reshape(1, n_agg, 3)
    agg_gt_hw3   = all_gt.reshape(1, n_agg, 3)
    agg_title = (f"Tutti i frame aggregati ({n_agg} pixel campionati, "
                 f"{len(psnrs)} frame, PSNR medio={float(np.mean(psnrs)):.2f} dB)")
    plot_bias_scatter(agg_pred_hw3, agg_gt_hw3, None,
                      str(base / "bias_scatter_all.png"), title=agg_title)

    # Canali per bias_bins.csv (luma ricalcolata dall'RGB aggregato)
    luma_pred_agg = (all_pred * _LUMA_COEFF).sum(-1)
    luma_gt_agg   = (all_gt   * _LUMA_COEFF).sum(-1)
    _channels = [
        ("R",    all_pred[:, 0], all_gt[:, 0]),
        ("G",    all_pred[:, 1], all_gt[:, 1]),
        ("B",    all_pred[:, 2], all_gt[:, 2]),
        ("Luma", luma_pred_agg,  luma_gt_agg),
    ]
    bins_rows: list[dict] = []
    for ch_name, cp, cg in _channels:
        centers, pred_meds, gt_meds, counts = binned_median_curve(cp, cg, n_bins=20)
        for k in range(len(centers)):
            pm = float(pred_meds[k])
            ct = float(centers[k])
            ratio = (pm / ct) if (ct > 1e-6 and not np.isnan(pm)) else float("nan")
            bins_rows.append({
                "channel":    ch_name,
                "bin_idx":    k,
                "center_gt":  f"{ct:.5f}",
                "median_gt":  f"{float(gt_meds[k]):.5f}",
                "median_pred": f"{pm:.5f}" if not np.isnan(pm) else "nan",
                "ratio":      f"{ratio:.4f}" if not np.isnan(ratio) else "nan",
                "count":      int(counts[k]),
            })
    bins_path = base / "bias_bins.csv"
    with open(bins_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["channel", "bin_idx", "center_gt", "median_gt",
                            "median_pred", "ratio", "count"])
        writer.writeheader()
        writer.writerows(bins_rows)

    # ── metrics_summary.txt ───────────────────────────────────────────────────
    def _nanmean(vals):
        arr = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(arr)) if arr else float("nan")

    psnr_tm_clips     = [r["psnr_tonemap_clip"]     for r in metrics_rows]
    psnr_tm_reinhards = [r["psnr_tonemap_reinhard"]  for r in metrics_rows]
    rel_p99s          = [r["rel_err_p99"]            for r in metrics_rows]
    rel_p999s         = [r["rel_err_p999"]           for r in metrics_rows]
    res_means         = [r["residual_mean"]          for r in metrics_rows]
    res_medians       = [r["residual_median"]        for r in metrics_rows]
    res_hl_means      = [r["residual_mean_hl"]       for r in metrics_rows]

    lines = [
        f"=== Analisi bias colore NeRF — iter {iter_done} ===",
        f"Frame valutati          : {len(metrics_rows)}",
        f"Pixel aggregati scatter : {n_agg}",
        "",
        "── PSNR ──────────────────────────────────────────",
        f"  PSNR lineare (HDR)    : {float(np.mean(psnrs)):.3f} dB",
        f"  PSNR tonemap clip     : {_nanmean(psnr_tm_clips):.3f} dB",
        f"  PSNR tonemap Reinhard : {_nanmean(psnr_tm_reinhards):.3f} dB",
        "  (tonemap clip ≈ qualità range diffuso senza coda HDR)",
        "",
        "── Highlight (percentili di luminanza GT) ────────",
        f"  Errore relativo medio p99   : {_nanmean(rel_p99s):.4f}",
        f"  Errore relativo medio p99.9 : {_nanmean(rel_p999s):.4f}",
        "",
        "── Residui con segno (pred − gt) ─────────────────",
        f"  Media globale   : {_nanmean(res_means):.5f}  (>0 sovrastima, <0 sottostima)",
        f"  Mediana globale : {_nanmean(res_medians):.5f}",
        f"  Media highlight (gt>1) : {_nanmean(res_hl_means):.5f}",
        "",
        "── Ratio mediana pred/gt per fascia (canali aggregati) ───",
        "   ratio < 1 = sottostima, ratio > 1 = sovrastima",
        "   (vedi bias_bins.csv per il dettaglio completo)",
    ]
    for ch_name, _cp, _cg in _channels:
        ch_bins = [r for r in bins_rows if r["channel"] == ch_name and r["ratio"] != "nan"]
        hl_bins = [r for r in ch_bins if float(r["center_gt"]) > 1.0]
        diff_bins = [r for r in ch_bins if float(r["center_gt"]) <= 1.0]
        ratio_diff = float(np.mean([float(r["ratio"]) for r in diff_bins])) if diff_bins else float("nan")
        ratio_hl   = float(np.mean([float(r["ratio"]) for r in hl_bins]))   if hl_bins else float("nan")
        lines.append(f"  {ch_name:<5}  range diffuso (gt≤1): {ratio_diff:.3f}  "
                     f"highlight (gt>1): {ratio_hl:.3f}" if not np.isnan(ratio_hl)
                     else f"  {ch_name:<5}  range diffuso (gt≤1): {ratio_diff:.3f}  "
                          f"highlight (gt>1): n/a (nessun campione)")

    summary_path = base / "metrics_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"[Step 2b] Completato — PSNR medio={float(np.mean(psnrs)):.2f} dB  "
          f"tonemap-clip={_nanmean(psnr_tm_clips):.2f} dB → {base}")
    print(f"  Scritti: metrics_per_frame.csv, bias_bins.csv, "
          f"metrics_summary.txt, bias_scatter_all.png")


def _step3_posttrain_assets(
    cfg: PipelineConfig,
    transforms_extended_path: Path,
    optix_mod,
    tb_logger=None,
    timer=None,
) -> dict:
    """Esegue IUM/visibility/color_texture/irradiance/indirect/spec_cone (bake) e
    aggiorna transforms_extended.json in-place con le nuove chiavi.

    Il fit PBR e l'albedo sono stati spostati nello Step 4 (_step4_reconstruction),
    che legge solo la cache su disco prodotta qui.
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
        _t3_ium = time.perf_counter()
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
        if timer is not None:
            timer.record("step3/ium", time.perf_counter() - _t3_ium)

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
        # Il pass calcola l'OCCLUSIONE (shadow ray camera→texel). L'artefatto
        # autoritativo su disco è però la visibility raffinata dalle maschere di
        # color_texture (occlusione∧frustum∧grazing): visibility.exr viene quindi
        # scritto nel blocco color_texture (o come fallback solo-occlusione più
        # sotto, se color_texture non gira). Qui NON si salva.
        visibility_map = None
        visibility_refined = False   # True quando visibility.exr = maschere (frustum+grazing)
        vis_path = None
        if rc.render_visibility and ium_res.has_positions() and ium_res.has_masks():
            _t3_vis = time.perf_counter()
            print("[Step 3] Calcolo Visibilità telecamere…")
            vis_gen = optix_mod.VisibilityGenerator()
            vis_gen.set_traversable(model)
            visibility_map = vis_gen.check_visibility(ium_res, ium_w, ium_h, all_cameras)

            vis_out_dir = json_dir / "visibility"
            os.makedirs(vis_out_dir, exist_ok=True)
            vis_path = (vis_out_dir / f"visibility{rc.visibility_format.extension}").resolve().as_posix()
            ium_result_data["visibility_path"] = _as_relative_to(vis_path, json_dir_str)
            if timer is not None:
                timer.record("step3/visibility", time.perf_counter() - _t3_vis)

        # ── Color Texture ────────────────────────────────────────────────────
        # color_by_source: sorgente → colors_np float32 (copia esplicita: result,
        # generatore e frames vengono rilasciati a fine iterazione, prima di caricare
        # la sorgente successiva). Le texture per-camera vengono scaricate dalla GPU
        # una camera alla volta: il blocco frames×texel non esiste mai in RAM host.
        color_by_source: dict[str, np.ndarray] = {}
        # Le maschere per-camera di color_texture (occlusione∧frustum∧grazing, pre-peak)
        # sono source-indipendenti: le salviamo una sola volta come <out>/camera_mask/
        # {stem}.exr e le riusiamo come visibility condivisa raffinata.
        cam_mask_dir = json_dir / "camera_mask"
        stems = [f.stem for f in tf.frames]
        if (rc.render_color_texture and rc.render_ium and rc.render_visibility
                and visibility_map is not None):
            _t3_ct = time.perf_counter()

            # Ogni source viene processata per intero e in modo identico, sotto sources/{src}/
            ct_sources = rc.color_texture_image_sources
            if not ct_sources:
                raise ValueError("color_texture_image_sources non può essere vuota")

            for src in ct_sources:
                src_dir = json_dir / "sources" / src
                ct_dir = src_dir / "color_texture"
                os.makedirs(ct_dir, exist_ok=True)
                ct_path = (ct_dir / f"color_texture{rc.color_texture_format.extension}").resolve().as_posix()
                cam_tex_dir = src_dir / "camera_texture"

                if os.path.exists(ct_path) and cam_tex_dir.is_dir():
                    # Cache-hit: per non far regredire la visibility servono le maschere
                    # per-camera. Se ci sono su disco le ricarichiamo e raffiniamo; se
                    # mancano NON usiamo la cache (ricalcoliamo per rigenerarle).
                    masks_disk = _load_camera_masks(cam_mask_dir, stems, ium_w * ium_h)
                    if masks_disk is not None:
                        loaded = _load_exr_as_flat(ct_path)
                        if loaded is not None:
                            print(f"[Step 3] Color texture trovata su disco ({src}): {ct_path}")
                            color_by_source[src] = loaded
                            ium_result_data[f"color_texture_path_{src}"] = _as_relative_to(ct_path, json_dir_str)
                            if not visibility_refined and vis_path is not None:
                                visibility_map = masks_disk
                                _save_visibility_map(visibility_map, vis_path, ium_h, ium_w,
                                                     len(all_cameras), rc.visibility_format)
                                visibility_refined = True
                                print("[Step 3] Visibility raffinata dalle maschere per-camera su disco")
                            continue
                    else:
                        print(f"[Step 3] Color texture in cache ({src}) ma maschere per-camera "
                              f"mancanti → ricalcolo per rigenerarle")

                print(f"[Step 3] Calcolo Color Texture (sorgente: {src})…")
                optix_frames = _build_optix_frames_for_source(
                    src, tf, all_cameras, intr, rc, cfg, json_dir, optix_mod)

                ct_gen = optix_mod.ColorTexGenerator()
                ct_gen.set_inputs(ium_res, visibility_map, optix_frames,
                                  grazing_max_deg=rc.color_texture_grazing_max_deg)
                ct_gen.render()
                _ct_res = ct_gen.get_result()

                # Salva color_texture per questa sorgente
                ct_arr = _reshape_flat(_ct_res.colors_np.astype(np.float32), ium_w, ium_h)
                _save_layer(ct_arr, ct_path, rc.color_texture_format, DataLayer.POSITION)
                ium_result_data[f"color_texture_path_{src}"] = _as_relative_to(ct_path, json_dir_str)

                # Copia leggera di colors_np per il calcolo albedo a valle
                color_by_source[src] = np.array(_ct_res.colors_np, dtype=np.float32)

                # Texture per-camera per questa sorgente: sources/{src}/camera_texture/
                # Scarico anche la maschera per-camera (uint8) per raffinare la visibility.
                os.makedirs(cam_tex_dir, exist_ok=True)
                # Le maschere sono source-indipendenti: le salviamo una sola volta,
                # sul primo source che le produce (prima che visibility_refined diventi True).
                save_masks = not visibility_refined
                if save_masks:
                    os.makedirs(cam_mask_dir, exist_ok=True)
                cam_masks = np.zeros((ium_w * ium_h, len(tf.frames)), dtype=np.uint8)
                for cam_idx, frame in enumerate(tf.frames):
                    cam_slice = ct_gen.download_camera_colors(cam_idx)  # (num_pix, 3) float32
                    cam_arr   = _reshape_flat(cam_slice, ium_w, ium_h)
                    cam_path  = (cam_tex_dir / f"{frame.stem}{rc.color_texture_format.extension}").resolve().as_posix()
                    _save_layer(cam_arr, cam_path, rc.color_texture_format, DataLayer.POSITION)
                    cam_masks[:, cam_idx] = ct_gen.download_camera_mask(cam_idx)  # (num_pix,) uint8
                    if save_masks:
                        mask_path = (cam_mask_dir / f"{frame.stem}.exr").resolve().as_posix()
                        _save_layer(_reshape_flat(cam_masks[:, cam_idx], ium_w, ium_h),
                                    mask_path, ImageFormat.OPENEXR, DataLayer.MASK)
                    if rc.debug_camera_texture:
                        src_img_path = images_out_dir_ct / Path(frame.file_path).name
                        _save_debug_comparison(src_img_path, cam_arr, frame.stem,
                                               src_dir / "debug_camera_texture")

                # Raffina la visibility condivisa con la maschera per-camera (una volta):
                # occlusione∧frustum∧grazing, così spec_cone (usa visibility_map in memoria)
                # e pbr_solver (rilegge visibility.exr) non pesano camere che non vedono
                # davvero il texel. Nessuna modifica al solver.
                if not visibility_refined and vis_path is not None:
                    visibility_map = cam_masks
                    _save_visibility_map(visibility_map, vis_path, ium_h, ium_w,
                                         len(all_cameras), rc.visibility_format)
                    visibility_refined = True
                    print("[Step 3] Visibility raffinata con la maschera per-camera di "
                          "color_texture (frustum+grazing) e ri-salvata")

                # pixel_change per ogni sorgente: sources/{src}/pixel_change/
                if rc.render_pixel_change:
                    pc_dir = src_dir / "pixel_change"
                    pc_dir.mkdir(parents=True, exist_ok=True)
                    min_arr   = _reshape_flat(_ct_res.color_min_np.astype(np.float32), ium_w, ium_h)
                    max_arr   = _reshape_flat(_ct_res.color_max_np.astype(np.float32), ium_w, ium_h)
                    range_arr = np.clip(max_arr - min_arr, 0.0, None)
                    var_arr   = _reshape_flat(_ct_res.color_variance_np.astype(np.float32), ium_w, ium_h)
                    ext = rc.color_texture_format
                    _save_layer(min_arr,   (pc_dir / f"color_min{ext.extension}").as_posix(),      ext, DataLayer.POSITION)
                    _save_layer(max_arr,   (pc_dir / f"color_max{ext.extension}").as_posix(),      ext, DataLayer.POSITION)
                    _save_layer(range_arr, (pc_dir / f"color_range{ext.extension}").as_posix(),    ext, DataLayer.POSITION)
                    _save_layer(var_arr,   (pc_dir / f"color_variance{ext.extension}").as_posix(), ext, DataLayer.POSITION)
                    if rc.debug_pixel_change:
                        _save_debug_pixel_change(min_arr, max_arr, range_arr, src_dir / "debug_pixel_change")

                # TensorBoard per questa sorgente
                if tb_logger is not None:
                    tb_logger.log_image(f"texture/color_texture_{src}", ct_arr, step=0, tonemap=True)
                    tb_logger.flush()

                # Rilascio esplicito prima della sorgente successiva: result host
                # (~4 layer da num_pix), generatore (che possiede il buffer VRAM
                # camera_colors ~num_pix×frames e le immagini caricate) e frames.
                del _ct_res, ct_gen, optix_frames

            if timer is not None:
                timer.record("step3/color_texture", time.perf_counter() - _t3_ct)

        # ── Raffinamento visibility da maschere persistite (2-bis) e fallback ─
        # Se color_texture non ha raffinato la visibility (es. render_color_texture
        # disattivo) ma le maschere per-camera esistono su disco, raffina da lì così
        # spec_cone e pbr_solver usano comunque la versione frustum+grazing.
        if not visibility_refined and visibility_map is not None and vis_path is not None:
            masks_disk = _load_camera_masks(cam_mask_dir, stems, ium_w * ium_h)
            if masks_disk is not None:
                visibility_map = masks_disk
                _save_visibility_map(visibility_map, vis_path, ium_h, ium_w,
                                     len(all_cameras), rc.visibility_format)
                visibility_refined = True
                print("[Step 3] Visibility raffinata dalle maschere per-camera su disco (2-bis)")
            else:
                # Fallback: nessuna maschera disponibile → salva la visibility
                # solo-occlusione (frustum/grazing NON applicati).
                _save_visibility_map(visibility_map, vis_path, ium_h, ium_w,
                                     len(all_cameras), rc.visibility_format)
                print("    ⚠  visibility.exr salvata SOLO-OCCLUSIONE (nessuna maschera "
                      "per-camera su disco): frustum/grazing NON applicati. Esegui "
                      "color_texture per raffinarla.")

        # ── Irradiance (skybox, quadratura deterministica) ───────────────────
        irr_res = None
        skybox_flat_step3 = None   # condivisa tra irradiance e spec_cone
        if (rc.render_irradiance
                and (rc.skybox_path or rc.skybox_source == "nerf")
                and ium_res.has_positions() and ium_res.has_normals()):
            _t3_irr = time.perf_counter()
            print(f"[Step 3] Calcolo Irradiance "
                  f"({rc.irradiance_sample_side}×{rc.irradiance_sample_side} samples)…")
            sky_w, sky_h = rc.skybox_size[0], rc.skybox_size[1]
            skybox_flat = _resolve_skybox_flat(rc, output_json, json_dir, sky_w, sky_h)
            skybox_flat_step3 = skybox_flat
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
            if timer is not None:
                timer.record("step3/irradiance", time.perf_counter() - _t3_irr)

        # ── Confronto skybox GT vs NeRF-baked ────────────────────────────────
        # Richiede: compare_skybox_to_gt=True + skybox_path non-vuoto (GT HDR) +
        # skybox_nerf_baked.exr già scritto da _bake_skybox_from_nerf.
        if rc.compare_skybox_to_gt and rc.skybox_path:
            baked_exr = json_dir / "skybox_nerf_baked.exr"
            if baked_exr.exists():
                from nerf.metrics import plot_skybox_compare
                gt_sky    = _load_image_hw3_native(rc.skybox_path)
                baked_sky = _load_image_hw3_native(baked_exr.as_posix())
                sky_cmp_dir = json_dir / "skybox_compare"
                os.makedirs(sky_cmp_dir, exist_ok=True)
                sky_title = (f"Skybox  baked NeRF ({baked_sky.shape[1]}x{baked_sky.shape[0]}) "
                             f"→ GT ({gt_sky.shape[1]}x{gt_sky.shape[0]})")
                plot_skybox_compare(
                    gt_sky, baked_sky,
                    str(sky_cmp_dir / "skybox_heatmap.png"),
                    title=sky_title,
                )
                print(f"[Step 3] Skybox compare salvata: {sky_cmp_dir / 'skybox_heatmap.png'}")
            else:
                print("[Step 3] skybox_compare: skybox_nerf_baked.exr non trovato "
                      "(atteso dopo skybox_source='nerf') — skip")

        # ── Indirect Irradiance via NeRF ─────────────────────────────────────
        irr_indirect_flat = None
        if rc.precompute_indirect and ium_res.has_positions() and ium_res.has_normals():
            _t3_ind = time.perf_counter()
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
            if timer is not None:
                timer.record("step3/indirect", time.perf_counter() - _t3_ind)

        # ── Specular cone L_j(r) via envmap + NeRF ───────────────────────────
        if (rc.precompute_spec_cone and ium_res.has_positions()
                and ium_res.has_normals() and visibility_map is not None):
            _t3_spec = time.perf_counter()
            sky_w, sky_h = rc.skybox_size[0], rc.skybox_size[1]
            if skybox_flat_step3 is None and (rc.skybox_path or rc.skybox_source == "nerf"):
                skybox_flat_step3 = _resolve_skybox_flat(rc, output_json, json_dir, sky_w, sky_h)
            spec_dir = json_dir / "spec_cone"
            if rc.spec_cone_scheme == "shared":
                print(f"[Step 3] Precompute Specular Cone L_j(r), raggi condivisi "
                      f"(aperture={rc.spec_cone_apertures_deg}°, "
                      f"S={rc.spec_cone_shared_samples})…")
                _precompute_spec_cone_shared(
                    rc, ium_res, model, ium_w, ium_h, tf.frames,
                    visibility_map, len(all_cameras),
                    skybox_flat_step3, [sky_w, sky_h], spec_dir)
            elif rc.spec_cone_scheme == "per_camera":
                print(f"[Step 3] Precompute Specular Cone L_j(r) "
                      f"(aperture={rc.spec_cone_apertures_deg}°, "
                      f"alloc={rc.spec_cone_sample_alloc}, "
                      f"budget={rc.spec_cone_samples_budget})…")
                _precompute_spec_cone(rc, ium_res, model, ium_w, ium_h, tf.frames,
                                      visibility_map, len(all_cameras),
                                      skybox_flat_step3, [sky_w, sky_h], spec_dir)
            else:
                raise ValueError(
                    f"spec_cone_scheme sconosciuto: {rc.spec_cone_scheme!r} "
                    "(attesi 'per_camera' o 'shared')")
            ium_result_data["spec_cone_dir"] = _as_relative_to(
                spec_dir.resolve().as_posix(), json_dir_str)
            if timer is not None:
                timer.record("step3/spec_cone", time.perf_counter() - _t3_spec)

    # Aggiorna il JSON in-place e riscrivi
    if ium_result_data:
        output_json["ium"] = ium_result_data
    with open(transforms_extended_path, "w", encoding="utf-8") as fh:
        json.dump(output_json, fh, indent=4)
    print(f"\n[Step 3] JSON aggiornato: {transforms_extended_path}")
    return output_json


def _step4_reconstruction(
    cfg: PipelineConfig,
    transforms_extended_path: Path,
    tb_logger=None,
    timer=None,
) -> dict:
    """Step 4 — ricostruzione (fit PBR + albedo) dalla sola cache su disco dello Step 3.

    Non usa OptiX né il checkpoint NeRF: legge spec_cone/, sources/{src}/color_texture/,
    irradiance/ e ium/ già prodotti dallo Step 3, quindi si può rieseguire da solo
    (run_step1/2/3=False, run_step4=True) per iterare sulla ricostruzione senza ri-bake
    dei coni. Aggiorna transforms_extended.json in-place con le chiavi
    metallic/roughness/albedo_pbr/albedo.
    """
    rc = cfg.render
    json_dir = Path(rc.output_dir).resolve()
    json_dir_str = json_dir.as_posix()

    with open(transforms_extended_path, encoding="utf-8") as fh:
        output_json = json.load(fh)
    ium_result_data: dict = output_json.get("ium", {})

    # ── PBR maps (metallic / roughness dal fit spec-cone) ────────────────────
    # Eseguito per OGNI sorgente in color_texture_image_sources, sotto sources/{src}/.
    # solve_pbr legge tutto da disco (spec_cone/, camera_texture/, pixel_change/).
    if rc.render_pbr_maps:
        _t4_pbr = time.perf_counter()
        spec_meta = json_dir / "spec_cone" / "spec_cone_meta.json"   # source-indipendente
        if spec_meta.exists():
            from pbr_solver import solve_pbr
            for src in rc.color_texture_image_sources:
                cmin_path = json_dir / "sources" / src / "pixel_change" / "color_min.exr"
                if not cmin_path.exists():
                    print(f"    ⚠  render_pbr_maps ({src}): manca {cmin_path} → skip "
                          "(serve render_pixel_change nello Step 3)")
                    continue
                print(f"[Step 4] Fit PBR ({src}) → metallic/roughness "
                      f"(cv_gate={rc.pbr_diffuse_cv_gate}, "
                      f"spec_threshold={rc.pbr_spec_threshold})…")
                pbr_out = solve_pbr(json_dir_str, source=src,
                                    cv_gate=rc.pbr_diffuse_cv_gate,
                                    spec_threshold=rc.pbr_spec_threshold,
                                    min_views=rc.pbr_min_views,
                                    albedo_eps=rc.albedo_eps,
                                    blender_rgb=rc.pbr_write_blender_rgb)
                ium_result_data[f"metallic_path_{src}"] = _as_relative_to(
                    pbr_out["metallic_path"], json_dir_str)
                ium_result_data[f"roughness_path_{src}"] = _as_relative_to(
                    pbr_out["roughness_path"], json_dir_str)
                if pbr_out.get("albedo_pbr_path"):
                    ium_result_data[f"albedo_pbr_path_{src}"] = _as_relative_to(
                        pbr_out["albedo_pbr_path"], json_dir_str)
            if timer is not None:
                timer.record("step4/pbr", time.perf_counter() - _t4_pbr)
        else:
            print("    ⚠  render_pbr_maps: manca spec_cone_meta.json → skip "
                  "(serve precompute_spec_cone nello Step 3)")

    # ── Albedo = π · color / max(irradiance + indiretta, eps) ────────────────
    # Riletta interamente da disco: color_texture per sorgente, irradiance condivisa.
    if rc.render_albedo:
        _t4_alb = time.perf_counter()
        irr_path = json_dir / "irradiance" / f"irradiance{rc.irradiance_format.extension}"
        if not irr_path.exists():
            print(f"    ⚠  render_albedo: manca {irr_path} → skip "
                  "(serve render_irradiance nello Step 3)")
        else:
            print(f"[Step 4] Calcolo Albedo = π · color / max(irradiance, {rc.albedo_eps})…")

            # Denominatore condiviso: irradiance diretta (+ indiretta se presente)
            irr = _load_image_hw3_native(irr_path.as_posix())
            ind_path = json_dir / "irradiance" / f"irradiance_indirect{rc.indirect_format.extension}"
            if ind_path.exists():
                irr = irr + _load_image_hw3_native(ind_path.as_posix())
            denom = np.maximum(irr, rc.albedo_eps)

            # Maschera IUM (canale 0 > 0.5 = texel valido); salvata con rc.ium_format
            mask_path = json_dir / "ium" / f"ium_masks{rc.ium_format.extension}"
            mask = None
            if mask_path.exists():
                mask = _load_image_hw3_native(mask_path.as_posix())[..., 0] > 0.5

            for src in rc.color_texture_image_sources:
                color_path = (json_dir / "sources" / src / "color_texture"
                              / f"color_texture{rc.color_texture_format.extension}")
                if not color_path.exists():
                    print(f"    ⚠  albedo ({src}): manca {color_path} → skip "
                          "(serve render_color_texture nello Step 3)")
                    continue
                color = _load_image_hw3_native(color_path.as_posix())
                albedo = (np.float32(np.pi) * color) / denom
                if mask is not None:
                    albedo[~mask] = 0.0
                albedo = np.clip(albedo, 0.0, 1.0).astype(np.float32)

                alb_dir = json_dir / "sources" / src / "albedo"
                os.makedirs(alb_dir, exist_ok=True)
                alb_path = (alb_dir / f"albedo{rc.albedo_format.extension}").resolve().as_posix()
                _save_layer(albedo, alb_path, rc.albedo_format, DataLayer.ALBEDO)
                ium_result_data[f"albedo_path_{src}"] = _as_relative_to(alb_path, json_dir_str)

                # — TensorBoard: albedo è già in [0,1] → no tonemap —
                if tb_logger is not None:
                    tb_logger.log_image(f"texture/albedo_{src}", albedo, step=0, tonemap=False)
                    tb_logger.flush()

            if timer is not None:
                timer.record("step4/albedo", time.perf_counter() - _t4_alb)

    # Aggiorna il JSON in-place e riscrivi
    if ium_result_data:
        output_json["ium"] = ium_result_data
    with open(transforms_extended_path, "w", encoding="utf-8") as fh:
        json.dump(output_json, fh, indent=4)
    print(f"\n[Step 4] JSON aggiornato: {transforms_extended_path}")
    return output_json


def run_pipeline(
    cfg: PipelineConfig,
    *,
    tb_run_dir: str | None = None,
    tb_enabled: bool = True,
) -> dict:
    """Orchestratore a quattro step. Ogni step può essere abilitato/disabilitato.

    Step 1 (run_step1): depth+mask per frame + copia immagini + transforms_extended.json minimo.
    Step 2 (run_step2): training NeRF via nerf/train.py, salva checkpoint.
    Step 3 (run_step3): IUM/visibility/color_texture/irradiance/indirect/spec_cone (bake).
    Step 4 (run_step4): ricostruzione (fit PBR + albedo) dalla sola cache su disco dello
                        Step 3 — indipendente da OptiX/NeRF, rieseguibile per iterare
                        sulla ricostruzione senza ri-bake dei coni.

    ``tb_run_dir`` è la cartella in cui scrivere gli event file TensorBoard.
    Se None, viene usata <output_dir>/tensorboard.
    ``tb_enabled=False`` disabilita completamente il logging TB (no-op).
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from monitoring import RunLogger, StageTimer, log_timing_breakdown

    import OptixProgrammablePasses as optix
    optix.LogManager.set_min_level(optix.LogLevel.Error)
    optix.OptixManager.instance().set_log_level(optix.LogLevel.Disabled)

    _tb_dir = tb_run_dir or str(Path(cfg.render.output_dir) / "tensorboard")
    logger = RunLogger(_tb_dir, enabled=tb_enabled)
    timer  = StageTimer()

    # Log della config all'inizio del run
    logger.log_text("run/config", json.dumps(asdict(cfg), indent=2, default=str), step=0)

    transforms_extended = Path(cfg.render.output_dir) / "transforms_extended.json"
    final_psnr = float("nan")

    try:
        if cfg.run_step1:
            with timer("step1"):
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
                with timer("step2"):
                    ckpt_path, step2_psnr = _step2_train_nerf(
                        cfg, transforms_extended, tb_logger=logger
                    )
                    if not math.isnan(step2_psnr):
                        final_psnr = step2_psnr

                if cfg.enable_nerf_render_train_images:
                    with timer("step2b"):
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

        elif (cfg.run_step3
              and (cfg.render.precompute_indirect or cfg.render.precompute_spec_cone)
              and not ckpt_path.exists()):
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
            if ((cfg.render.precompute_indirect or cfg.render.precompute_spec_cone)
                    and not cfg.render.indirect_nerf_cache_path):
                cfg.render.indirect_nerf_cache_path = str(ckpt_path)

            with timer("step3"):
                result = _step3_posttrain_assets(
                    cfg, transforms_extended, optix,
                    tb_logger=logger, timer=timer,
                )
        else:
            with open(transforms_extended, encoding="utf-8") as fh:
                result = json.load(fh)

        # Step 4 — ricostruzione (fit PBR + albedo) dalla cache su disco dello Step 3.
        # Indipendente da run_step3: con run_step1/2/3=False si itera sulla
        # ricostruzione senza ri-bake dei coni (non richiede OptiX né il checkpoint).
        if cfg.run_step4:
            with timer("step4"):
                result = _step4_reconstruction(
                    cfg, transforms_extended,
                    tb_logger=logger, timer=timer,
                )

        # ── Timing breakdown e HParams ────────────────────────────────────────
        top_times = {k: v for k, v in timer.timings.items() if "/" not in k}
        total_s = sum(top_times.values())

        log_timing_breakdown(logger, timer.timings, step=0)

        hparams: dict = {
            "loss_type":            cfg.nerf_loss_type,
            "lr":                   cfg.nerf_lr,
            "lr_decay_factor":      cfg.nerf_lr_decay,
            "lr_decay_steps":       cfg.nerf_lr_decay_steps or cfg.nerf_num_iters,
            "num_iters":            cfg.nerf_num_iters,
            "rgb_activation":       cfg.nerf_rgb_activation,
            "batch_size":           cfg.nerf_batch_size,
            "depth_window_samples": cfg.nerf_depth_window_samples,
            "indirect_sample_side": cfg.render.indirect_sample_side
                                    if cfg.render.precompute_indirect else -1,
            "irr_sample_side":      cfg.render.irradiance_sample_side
                                    if cfg.render.render_irradiance else -1,
        }
        metrics: dict = {
            "psnr/final_db": final_psnr,
            "time/total_s":  total_s,
            **{f"time/{k}_s": v for k, v in top_times.items()},
        }
        logger.log_hparams(hparams, metrics)

        return result

    finally:
        logger.close()


# ──────────────────────────────────────────────────────────────────────────────
# Manifest per-run e runner multi-scena
# ──────────────────────────────────────────────────────────────────────────────

def _write_run_manifest(cfg: PipelineConfig, scene: SceneConfig, run_note: str) -> None:
    """Salva run_manifest.json in output_dir con config completa + timestamp + nota."""
    def _enc(o: object) -> str:
        if isinstance(o, Enum):
            return o.name
        if isinstance(o, Path):
            return str(o)
        return str(o)

    manifest = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "run_note": run_note,
        "scene": asdict(scene),
        "config": asdict(cfg),   # include output_dir e tutti i path già risolti per la scena
    }
    out = Path(cfg.render.output_dir) / "run_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False, default=_enc)
    print(f"  manifest salvato → {out}")


def run_pipeline_multi(
    template: PipelineConfig,
    scenes: list[SceneConfig],
    output_root: str,
    run_note: str = "",
    *,
    experiment_tag: str = "",
    tb_log_root: str | None = None,
    tb_enabled: bool = True,
) -> dict:
    """Esegue run_pipeline su ogni scena in sequenza, in sottocartelle distinte.

    Per ogni SceneConfig:
    - clona ``template`` (deep-copy, sicuro con i default_factory mutabili)
    - sovrascrive i path per-scena e ``output_dir = <output_root>/<scene.name>``
    - scrive ``run_manifest.json`` nella sottocartella
    - chiama ``run_pipeline`` con un log_dir TensorBoard isolato

    ``experiment_tag`` raggruppa i run correlati in TensorBoard sotto un unico prefisso.
    Se vuoto, viene derivato dal nome base di ``output_root``.

    ``tb_log_root`` è la radice dei log TensorBoard (deve corrispondere al volume
    montato in docker/tensorboard/docker-compose.yml). Se None, viene letta dalla
    variabile d'ambiente ``TB_LOG_ROOT``; se anche questa manca, il default è
    ``D:/tesi_output/tb_logs``.

    Ogni esecuzione di questa funzione crea sotto:
      ``<tb_log_root>/<experiment_tag>/<scene.name>/<YYYYMMDD-HHMMSS>/``
    in modo da isolare i re-run (niente curve fuse/zig-zag in TensorBoard).

    Se un run fallisce, l'errore viene loggato e si prosegue con il successivo.
    Al termine viene stampato un riepilogo con lo stato di ogni scena.

    ``nerf_ckpt_path`` / ``nerf_train_output_dir`` restano ``""`` nel template e vengono
    derivati automaticamente da ``output_dir`` per scena (un checkpoint per scena).

    Returns:
        dict scena → risultato di run_pipeline (o None se il run è fallito)
    """
    # ── Risolvi tb_log_root ────────────────────────────────────────────────────
    _tb_root = (
        tb_log_root
        or os.environ.get("TB_LOG_ROOT")
        or "D:/tesi_output/tb_logs"
    )
    _tag = experiment_tag or os.path.basename(output_root.rstrip("/\\"))
    _run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    results: dict = {}
    statuses: dict = {}

    for scene in scenes:
        cfg = copy.deepcopy(template)
        cfg.render.transforms_path = scene.transforms_path
        cfg.render.model_path      = scene.model_path
        if scene.external_normal_path is not None:
            cfg.render.external_normal_path = scene.external_normal_path
        if scene.skybox_path is not None:
            cfg.render.skybox_path = scene.skybox_path
        cfg.render.output_dir = os.path.join(output_root, scene.name)
        os.makedirs(cfg.render.output_dir, exist_ok=True)

        full_note = scene.note if scene.note else run_note
        _write_run_manifest(cfg, scene, full_note)

        # Resume: salta il training NeRF se il checkpoint è già presente
        if cfg.resume_skip_step2_if_ckpt and cfg.run_step2:
            _ckpt_resume = Path(cfg.render.output_dir) / "model" / "nerf_model_cache.pt"
            if _ckpt_resume.exists():
                print(f"  ↻ Checkpoint NeRF già presente; salto Step 2 (run_step2=False): {_ckpt_resume}")
                cfg.run_step2 = False

        # Log dir TensorBoard per questa scena e questo run (isolato per run-id)
        tb_run_dir = os.path.join(_tb_root, _tag, scene.name, _run_id)

        _log_path = os.path.join(cfg.render.output_dir, "console.log")
        with _console_to_file(_log_path):
            print(f"\n{'='*70}")
            print(f"  Scena       : {scene.name}")
            print(f"  Output      : {cfg.render.output_dir}")
            print(f"  TB log dir  : {tb_run_dir}")
            print(f"  Console log : {_log_path}")
            print(f"{'='*70}")

            try:
                results[scene.name] = run_pipeline(
                    cfg, tb_run_dir=tb_run_dir, tb_enabled=tb_enabled
                )
                statuses[scene.name] = "ok"
            except Exception as exc:
                print(f"\n  ✗ [{scene.name}] errore: {exc}")
                import traceback
                traceback.print_exc()
                results[scene.name] = None
                statuses[scene.name] = f"error: {exc}"

    # ── Riepilogo ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  Riepilogo run_pipeline_multi:")
    for name, status in statuses.items():
        icon = "✓" if status == "ok" else "✗"
        print(f"    {icon} {name}: {status}")
    print(f"{'='*70}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    REPO = "C:/Users/adria/Documents/GitHub/Tesi/OptixProjectCMake"

    # ── Template condiviso ────────────────────────────────────────────────────
    # I path specifici della scena (transforms_path, model_path, external_normal_path,
    # skybox_path) sono vuoti qui e vengono sovrascritti per ogni SceneConfig.
    # Tutti gli altri parametri di rendering/NeRF sono condivisi tra le scene.
    template = PipelineConfig(
        run_step1 = False,  # output Step 1 già su disco (exp_l1_d02)
        run_step2 = False,  # checkpoint NeRF e render Step 2b già su disco
        run_step3 = True,   # pass texture-space fino al bake dello spec-cone
        run_step4 = True,   # ricostruzione PBR+albedo (metti run_step3=False per iterare solo qui)

        resume_skip_step2_if_ckpt = True,   # salta il training NeRF se il checkpoint esiste già

        render = RenderConfig(
            # path per-scena (sovrascritta da SceneConfig — non modificare qui)
            transforms_path      = "",
            model_path           = "",
            external_normal_path = None,
            output_dir           = "",  # impostato come <output_root>/<scene.name>

            external_normal_resolution_mode = "resample",  # "adapt" | "resample" | "none"

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
            skybox_source          = "nerf",  # "nerf" | "file"

            # skybox_path          = f"{REPO}/Scenes/TableAndOther/Blender/assets/hdri/suburban_garden_4k.exr",
            skybox_size            = [4096, 2048],
            irradiance_sample_side = 512,

            precompute_indirect            = True,
            indirect_sample_side           = 32,
            indirect_tile_size             = 1024,
            indirect_override_depth_window = False,

            render_albedo = True,
            albedo_format = ImageFormat.OPENEXR,
            albedo_eps    = 1e-3,

            depth_format         = ImageFormat.OPENEXR,
            mask_format          = ImageFormat.PNG,
            ium_format           = ImageFormat.OPENEXR,
            visibility_format    = ImageFormat.OPENEXR,
            color_texture_format = ImageFormat.OPENEXR,

            ium_texture_size = [4096, 4096],
            apply_scale      = False,

            color_texture_image_sources = ["gt", "nerf"], # entrambe processate per intero sotto sources/

            precompute_spec_cone = True,
            render_pbr_maps      = True,
            pbr_diffuse_cv_gate  = 0.0,        # gate diffuso disattivato: fit ovunque
            pbr_spec_threshold   = 0.0,        # roughness fittata scritta ovunque
            spec_cone_cameras=None,

            # Raggi condivisi tra le camere: la radianza incidente lungo una
            # direzione non dipende dalla camera, quindi un solo set Fibonacci per
            # texel (tracciato e interrogato sul NeRF una volta) serve tutte le m
            # camere che vedono il texel. A S=16384 il costo pareggia il bake
            # per-camera quando m≈11: il guadagno è reinvestito in risoluzione,
            # non in tempo. La griglia di aperture è raffinata nella zona stretta
            # perché con i raggi condivisi un candidato in più non costa raggi.
            # Il candidato a 5° riceve ~16 campioni, cioè è al limite del rumore:
            # se lobe_param/residual lo mostrano instabile, toglierlo dalla griglia.
            spec_cone_scheme         = "shared",
            spec_cone_shared_samples = 9216, # 96 x 96 Fibonacci samples per texel (shared)

            spec_cone_apertures_deg  = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0,
                                        60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0],
            # deve essere multiplo della larghezza IUM: ogni tile è un blocco di
            # scanline intere, scritte in streaming negli EXR per camera
            spec_cone_tile_size      = 4096,
            # dirs + radianze costano ~0.6 MB per texel a S=16384: 1024 texel sono
            # ~0.6 GB di picco. Alzare finché la VRAM regge — sotto-blocchi grandi
            # riducono l'overhead del loop sulle camere.
            spec_cone_chunk_texels   = 1024,
            # il batch della rete è il collo di bottiglia dell'occupazione GPU
            spec_cone_nerf_chunk     = 4096*24,

            # Usati solo da spec_cone_scheme="per_camera"
            spec_cone_sample_alloc   = "solid_angle",
            spec_cone_samples_budget = 1440,

            color_texture_grazing_max_deg = 75.0,

            # Heatmap diagnostica skybox GT vs NeRF-baked (richiede skybox_path nella SceneConfig)
            compare_skybox_to_gt = True,

        ),

        nerf_num_iters       = 50000,
        nerf_lr_decay_steps  = 100000,  # ancora fissa = lunghezza pianificata; resume corretto
        nerf_batch_size      = 4096*24,
        nerf_lr              = 5e-4,
        nerf_display_every = 100,
        nerf_seed          = 9458,

        enable_nerf_render_train_images = True,
        nerf_interactive_loop           = False,

        nerf_depth_window_samples      = 5,
        nerf_depth_window              = 0.05,
        nerf_depth_window_end          = 0.05,
        nerf_opacity_weight            = 1.0,
        nerf_raw_noise_std             = 1.0,
        # Il raggio della sfera bg è ora bg_radius_mult × (distanza max dall'origine),
        # non più × max_side della bbox: su questa scena la base passa da 3.2 a 2.266,
        # quindi 3.0 darebbe R=6.8 contro i ~12 di prima. 5.0 → R=11.3, che tiene il
        # guscio fuori dal rig di camere (le più lontane stanno a 9.0) e conserva la
        # risoluzione angolare dell'envmap: quanto spazio occupa il guscio rispetto alle
        # frequenze del positional encoding decide quanto è nitido skybox_nerf_baked.exr.
        nerf_bg_radius_mult            = 5.0,
        nerf_bg_depth_window           = 0.05,
        nerf_bg_depth_window_end       = 0.05,
        # nerf_multires = 12,
        # nerf_multires_views = 6,



        nerf_profile_iters = 0,


    )

    # ── Scene ─────────────────────────────────────────────────────────────────
    # Aggiungere/commentare SceneConfig per scegliere quali scene processare.
    # L'output di ogni scena finisce in <output_root>/<scene.name>/.
    SCENES = [
        # SceneConfig(
        #     name             = "TableAndOtherInterior",
        #     transforms_path  = f"{REPO}/Scenes/TableAndOtherInterior/NerfOpenEXR/transforms.json",
        #     model_path       = f"{REPO}/Scenes/TableAndOtherInterior/Models/Baked.obj",
        #     external_normal_path = f"{REPO}/Scenes/TableAndOtherInterior/BlenderBaked/BakedMaterial_normal.exr",
        #     # GT HDR usato solo come riferimento per compare_skybox_to_gt (non per il rendering)
        #     skybox_path      = f"{REPO}/Scenes/TableAndOtherInterior/Blender/assets/hdri/wooden_studio_13_4k.exr",
        # ),
        # SceneConfig(
        #     name             = "TableAndOtherInteriorWithSpecular",
        #     transforms_path  = f"{REPO}/Scenes/TableAndOtherInterior/NerfOpenEXRSmooth/transforms.json",
        #     model_path       = f"{REPO}/Scenes/TableAndOtherInterior/ModelsSmooth/Baked.obj",
        #     external_normal_path = f"{REPO}/Scenes/TableAndOtherInterior/BlenderBakedSmooth/BakedMaterial_normal.exr",
        #     # GT HDR usato solo come riferimento per compare_skybox_to_gt (non per il rendering)
        #     # skybox_path      = f"{REPO}/Scenes/TableAndOtherInterior/Blender/assets/hdri/wooden_studio_13_4k.exr",
        # ),
        #  SceneConfig(
        #     name             = "TableAndOtherInteriorWithSpecularHighDetails",
        #     transforms_path  = f"{REPO}/Scenes/TableAndOtherInterior/NerfOpenEXRHighDetails/transforms.json",
        #     model_path       = f"{REPO}/Scenes/TableAndOtherInterior/ModelsSmooth/Baked.obj",
        #     external_normal_path = f"{REPO}/Scenes/TableAndOtherInterior/BlenderBakedSmooth/BakedMaterial_normal.exr",
        #     # GT HDR usato solo come riferimento per compare_skybox_to_gt (non per il rendering)
        #     # skybox_path      = f"{REPO}/Scenes/TableAndOtherInterior/Blender/assets/hdri/wooden_studio_13_4k.exr",
        # ),
        # SceneConfig(
        #             name             = "TableAndOtherInteriorWithSpecularNight",
        #             transforms_path  = f"{REPO}/Scenes/TableAndOtherInterior/NerfOpenEXRSmoothNight/transforms.json",
        #             model_path       = f"{REPO}/Scenes/TableAndOtherInterior/ModelsSmooth/Baked.obj",
        #             external_normal_path = f"{REPO}/Scenes/TableAndOtherInterior/BlenderBakedSmoothNight/BakedMaterial_normal.exr",
        #             # GT HDR usato solo come riferimento per compare_skybox_to_gt (non per il rendering)
        #             # skybox_path      = f"{REPO}/Scenes/TableAndOtherInterior/Blender/assets/hdri/wooden_studio_13_4k.exr",
        #         )
        # SceneConfig(
        #     name             = "TableAndOtherInterior",
        #     transforms_path  = f"{REPO}/Scenes/TableAndOtherInterior/NerfOpenExrSmoothNoDiffuse/transforms.json",
        #     model_path       = f"{REPO}/Scenes/TableAndOtherInterior/ModelsSmooth/Baked.obj",
        #     external_normal_path = f"{REPO}/Scenes/TableAndOtherInterior/BlenderBakedSmoothNoDiffuse/BakedMaterial_normal.exr",
        #     # GT HDR usato solo come riferimento per compare_skybox_to_gt (non per il rendering)
        #     # skybox_path      = f"{REPO}/Scenes/TableAndOtherInterior/Blender/assets/hdri/wooden_studio_13_4k.exr",
        # )
        # SceneConfig(
        #     name            = "SwordShield",
        #     transforms_path = f"{REPO}/Scenes/SwordShield/NerfOpenEXR/transforms.json",
        #     model_path      = f"{REPO}/Scenes/SwordShield/Models/SwordShield.obj",
        # ),
        # SceneConfig(
        #     name             = "SwordShieldStudio",
        #     transforms_path  = f"{REPO}/Scenes/SwordShield Thesis/NerfStudio/transforms.json",
        #     model_path       = f"{REPO}/Scenes/SwordShield Thesis/BakedStudio/Baked.obj",
        #     external_normal_path = f"{REPO}/Scenes/SwordShield Thesis/BakedStudio/BakedMaterial_normal.exr",
        # ),
        SceneConfig(
            name             = "SwordShieldNight",
            transforms_path  = f"{REPO}/Scenes/SwordShield Thesis/NerfNight/transforms.json",
            model_path       = f"{REPO}/Scenes/SwordShield Thesis/BakedNight/Baked.obj",
            external_normal_path = f"{REPO}/Scenes/SwordShield Thesis/BakedNight/BakedMaterial_normal.exr",
        )

    ]

    # ── Esecuzione ────────────────────────────────────────────────────────────
    # Sweep fattoriale 2×2×2: attivazione × loss × decay — 8 run in totale.
    # Ogni run ottiene il proprio output_root (checkpoint isolati: exp/softplus

    # non sono compatibili tra loro e non devono fare resume incrociato).
    # TB_LOG_ROOT viene letta da docker/tensorboard/.env — non serve cambiarla qui.
    # Per un sweep pulito da zero, cambiare SWEEP_ROOT o svuotare la cartella.

    # (tag_base, rgb_activation, loss_type) — fattoriale 2×2 attivazione × loss
    EXPERIMENTS = [
        # ("exp_relmseraw",      "exp",      "rel_mse_raw"),
        ("softplus_relmseraw", "softplus", "rel_mse_raw"),
        # ("exp_l1",             "exp",      "l1"),
        # ("softplus_l1",        "softplus", "l1"),
        # ("softplus_mse",       "softplus", "mse"),
        # ("exp_mse",            "exp",      "mse")
    ]
    DECAYS     = (0.2,)
    SWEEP_ROOT = "D:/tesi_output/test_sword_shield"

    for name, act, loss in EXPERIMENTS:
        for decay in DECAYS:
            
            cfg = copy.deepcopy(template)
            cfg.nerf_num_iters       = 75000
            cfg.nerf_lr_decay_steps  = 100000  # ancora fissa allineata a num_iters
            cfg.nerf_rgb_activation  = act
            cfg.nerf_loss_type       = loss
            cfg.nerf_lr_decay        = decay
            cfg.render.skybox_source = "nerf"  # il campo vive su RenderConfig, non su PipelineConfig
            tag = f"{name}_d{str(decay).replace('.', '')}"  # es. exp_l1_d01
            run_pipeline_multi(
                cfg, SCENES,
                output_root    = f"{SWEEP_ROOT}/{tag}",
                run_note       = f"{cfg.nerf_num_iters}iter | act={act} | loss={loss} | decay={decay}",
                tb_enabled=False
            )
    