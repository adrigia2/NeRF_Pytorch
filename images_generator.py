"""
images_generator.py
-------------------
Driver of the hybrid rendering pipeline. Reads a transforms.json (NeRF format)
and runs four toggle-able steps: per-frame depth/position/normal/mask via
OptixProgrammablePasses, NeRF training, the texture-space bake (IUM, visibility,
colour texture, irradiance, specular cones) and the PBR reconstruction. It has no
command line: the configuration lives in __main__ at the bottom. See README.md.
"""

from __future__ import annotations

import contextlib
import copy
import datetime
import hashlib
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
# Enumerations
# ──────────────────────────────────────────────────────────────────────────────

class ImageFormat(Enum):
    OPENEXR = "openexr"
    PNG     = "png"

    @property
    def extension(self) -> str:
        return {"openexr": ".exr", "png": ".png"}[self.value]

    @property
    def supports_raw_float(self) -> bool:
        """True when the format can store 32-bit floats without loss."""
        return self in {ImageFormat.OPENEXR}


class DataLayer(Enum):
    """Semantic type of the layer — decides whether normalization is needed."""
    DEPTH    = auto()   # values in scene units → raw float
    POSITION = auto()   # world-space coordinates → raw float
    NORMAL   = auto()   # vectors in [-1,1] → raw not required, but float is fine
    MASK     = auto()   # uint8 0/1 → not raw, no float normalization
    VISIBILITY = auto() # float ratio in [0, 1], or uint8 bool
    IRRADIANCE          = auto() # HDR energy per texel (RGB float)
    IRRADIANCE_INDIRECT = auto() # indirect NeRF contribution per texel (RGB float)
    ALBEDO              = auto() # HDR reflectance per texel (RGB float)
    SPEC_CONE           = auto() # mean specular-cone radiance L_j(r) (RGB float HDR)
    METALLIC            = auto() # specularity 1−X from the PBR fit, [0,1] (float)
    ROUGHNESS           = auto() # cone aperture / 180 where reliable, [0,1] (float)
    SPEC_CONE_R         = auto() # best-fit cone aperture in degrees (float)


# ──────────────────────────────────────────────────────────────────────────────
# Writer protocol (extensible without touching the core)
# ──────────────────────────────────────────────────────────────────────────────

class ImageWriter(Protocol):
    """Interface every writer has to implement."""
    def write(self, array: np.ndarray, path: str) -> None: ...


# ──────────────────────────────────────────────────────────────────────────────
# Writer: OpenEXR
# ──────────────────────────────────────────────────────────────────────────────

class ExrWriter:
    """Write float32 NumPy arrays to an OpenEXR file.

    Supported shapes:
      (H, W)      → single channel  'Z'
      (H, W, 3)   → RGB channels    'R','G','B'
      (H, W, 4)   → RGBA channels   'R','G','B','A'
      (H, W, C)   → arbitrary       'Cam0', 'Cam1', ...
    """

    def write(self, array: np.ndarray, path: str) -> None:
        import OpenEXR, Imath  # local import — not required unless EXR is used

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
            raise ValueError(f"ExrWriter: ndim={array.ndim} not supported")

        f.close()


# ──────────────────────────────────────────────────────────────────────────────
# Writer: incremental OpenEXR (scanline blocks)
# ──────────────────────────────────────────────────────────────────────────────

class IncrementalExrWriter:
    """EXR written in blocks of consecutive scanlines, never holding the whole
    image in RAM.

    This is what the shared spec_cone bake needs, where the outer loop is over
    tiles rather than cameras: full-resolution accumulators for every camera would
    not fit (at 4096², K=14, 58 cameras that is ~200 GiB), whereas one scanline
    block per camera costs a few hundred KB.

    channels: dict {name: dtype}, where dtype np.float16 → HALF channel, else FLOAT.
    Each write_block takes {name: array (rows, width)} and advances by `rows`
    scanlines; blocks must be written in order and cover exactly `height` rows.
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

    HALF_MAX = 65504.0   # largest value representable in float16

    def _check_half_range(self, name: str, arr: np.ndarray) -> None:
        """A value beyond HALF_MAX would become inf in the cast to half, and numpy
        reports that with a RuntimeWarning which the default filter prints ONLY
        ONCE per code location: a whole bake can be corrupted leaving a single
        line in the log. Downstream it is worse than noisy, it is silent — one inf
        in a cone channel makes the solver's centred variance nan, np.argmin
        returns the index of that nan and the texel drops out of the fit entirely.
        """
        a = np.asarray(arr)
        mx = float(np.abs(a).max()) if a.size else 0.0   # nan/inf propagate to the max
        if not np.isfinite(mx) or mx > self.HALF_MAX:
            raise ValueError(
                f"IncrementalExrWriter: {self.path}: channel {name!r} is half but "
                f"holds {mx:.6g} (limit {self.HALF_MAX:g}): the cast would produce "
                f"inf. Switch the channel to np.float32, or reduce the values upstream.")

    def write_block(self, block: "dict[str, np.ndarray]") -> None:
        rows = None
        payload = {}
        for name, dt in self.channels.items():
            if dt == np.float16:
                self._check_half_range(name, block[name])
            arr = np.ascontiguousarray(block[name], dtype=dt)
            if arr.ndim != 2 or arr.shape[1] != self.width:
                raise ValueError(f"IncrementalExrWriter: channel {name!r} has shape "
                                 f"{arr.shape}, expected (rows, {self.width})")
            if rows is None:
                rows = arr.shape[0]
            elif arr.shape[0] != rows:
                raise ValueError("IncrementalExrWriter: the channels of a block must "
                                 "have the same number of rows")
            payload[name] = arr.tobytes()

        if self.row + rows > self.height:
            raise ValueError(f"IncrementalExrWriter: {self.path} would overrun "
                             f"{self.height} rows (row {self.row} + {rows})")
        self._file.writePixels(payload, rows)
        self.row += rows

    def close(self) -> None:
        if self._file is None:
            return
        if self.row != self.height:
            raise ValueError(f"IncrementalExrWriter: {self.path} closed at {self.row} "
                             f"rows out of {self.height} (truncated file)")
        self._file.close()
        self._file = None

    def __enter__(self): return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        else:                      # do not validate the height if we are already failing
            self._file = None


# ──────────────────────────────────────────────────────────────────────────────
# Writer: PNG
# ──────────────────────────────────────────────────────────────────────────────

class PngWriter:
    """Write NumPy arrays as PNG (uint8). Floats are normalised to [0,255]."""

    def write(self, array: np.ndarray, path: str) -> None:
        from PIL import Image

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        uint8 = _to_uint8(array)
        Image.fromarray(uint8).save(path)


# ──────────────────────────────────────────────────────────────────────────────
# Writer registry (open for extension)
# ──────────────────────────────────────────────────────────────────────────────

_WRITER_REGISTRY: dict[ImageFormat, ImageWriter] = {
    ImageFormat.OPENEXR: ExrWriter(),
    ImageFormat.PNG:     PngWriter(),
}


def register_writer(fmt: ImageFormat, writer: ImageWriter) -> None:
    """Register a writer for a custom format."""
    _WRITER_REGISTRY[fmt] = writer


def get_writer(fmt: ImageFormat) -> ImageWriter:
    if fmt not in _WRITER_REGISTRY:
        raise NotImplementedError(f"No writer registered for format: {fmt}")
    return _WRITER_REGISTRY[fmt]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_uint8(array: np.ndarray) -> np.ndarray:
    """Normalise a float array to uint8 in [0, 255] (for PNG and similar)."""
    arr = array.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return arr


# ──────────────────────────────────────────────────────────────────────────────
# Logging: tee stdout+stderr to a file (diagnostics for overnight runs)
# ──────────────────────────────────────────────────────────────────────────────

class _Tee:
    """Write to two streams at once; flush after every write."""
    def __init__(self, original, file_stream):
        self._orig = original
        self._file = file_stream

    def write(self, data):
        self._orig.write(data)
        # The file may already be closed: colorama (lazily initialised via tqdm)
        # registers an atexit reset bound to whichever _Tee was active at the time,
        # and that can fire after _console_to_file has exited.
        if not self._file.closed:
            self._file.write(data)
            self._file.flush()

    def flush(self):
        self._orig.flush()
        if not self._file.closed:
            self._file.flush()

    # Forward every other attribute to the original stream (e.g. encoding, isatty)
    def __getattr__(self, name):
        return getattr(self._orig, name)


@contextlib.contextmanager
def _console_to_file(log_path: str):
    """Context manager: also redirect stdout and stderr to *log_path* (append, line-flushed)."""
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
    """Bring a flat array returned by the OptiX library into shape (H, W) or (H, W, C).

    The library always returns flat arrays:
      depths_np    → (N,)    with N = W*H
      positions_np → (N, 3)  with N = W*H
      normals_np   → (N, 3)  with N = W*H
      masks_np     → (N,)    with N = W*H  (uint8)

    Cases handled:
      (H, W)      → already the right shape, returned unchanged
      (H, W, C)   → already the right shape, returned unchanged
      (N,)        → N == W*H  → reshaped to (H, W)
      (N, C)      → N == W*H  → reshaped to (H, W, C)
    """
    pixels = w * h

    # Already the right shape
    if array.ndim == 2 and array.shape == (h, w):
        return array
    if array.ndim == 3 and array.shape[:2] == (h, w):
        return array

    # (N,) flat single-channel
    if array.ndim == 1:
        if array.size != pixels:
            raise ValueError(
                f"_reshape_flat: size={array.size} does not match {w}×{h}={pixels}"
            )
        return array.reshape(h, w)

    # (N, C) flat multi-channel — the shape returned by positions_np and normals_np
    if array.ndim == 2 and array.shape[0] == pixels:
        c = array.shape[1]
        return array.reshape(h, w, c)

    raise ValueError(
        f"_reshape_flat: shape={array.shape} not handled for dimensions {w}×{h}"
    )


def _save_layer(
    array: np.ndarray,
    path: str,
    fmt: ImageFormat,
    layer: DataLayer,
) -> None:
    """Save a layer in the requested format, normalising when necessary.

    Normalization rules:
      - DEPTH / POSITION  → raw float values; if the format cannot store float,
                            they are normalised to [0, 1] before reaching the writer.
      - NORMAL            → vectors in [-1, 1]; same rule as raw float.
      - MASK              → already uint8 0/1; no normalization applied.
    """
    needs_raw = layer in {DataLayer.DEPTH, DataLayer.POSITION, DataLayer.NORMAL,
                          DataLayer.IRRADIANCE, DataLayer.IRRADIANCE_INDIRECT, DataLayer.ALBEDO,
                          DataLayer.SPEC_CONE, DataLayer.METALLIC, DataLayer.ROUGHNESS,
                          DataLayer.SPEC_CONE_R}

    if needs_raw and not fmt.supports_raw_float:
        print(f"    ⚠  {fmt.value} does not support raw float ({layer.name}) → normalising to [0,1]")
        array = array.astype(np.float32)
        mn, mx = array.min(), array.max()
        if mx > mn:
            array = (array - mn) / (mx - mn)

    get_writer(fmt).write(array, path)
    print(f"    ✓ Saved: {path}  shape={array.shape}")


def _build_output_path(base_dir: str, stem: str, layer_name: str, fmt: ImageFormat) -> str:
    return (Path(base_dir) / layer_name / f"{stem}_{layer_name}{fmt.extension}").resolve().as_posix()


def _as_relative_to(abs_path: str, base_dir: str) -> str:
    """Return abs_path relative to base_dir, in posix form.
    If it is not under base_dir, the posix path is returned unchanged."""
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

    # Normalise each image to [0,1] for display
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
    print(f"    ✓ pixel_change debug saved: {out_path}")


def _compute_peak(image_np: np.ndarray, percentile: float) -> float:
    """Image peak, as a percentile of the per-pixel maximum luminance."""
    max_per_pixel = image_np.astype(np.float32).max(axis=-1)  # (H, W)
    return float(np.percentile(max_per_pixel, percentile))


def _load_image_as_vec3(path: str, w: int, h: int) -> np.ndarray:
    """Load an image, resize it to (w, h), return float32 (H*W, 3)."""
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
    """Load an image at its native resolution as float32 (H, W, 3).
    EXR → raw HDR values. LDR (PNG/JPG) → uint8/255, scaled into [0, 1]."""
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
    """Load an RGB EXR at native size and return (H*W, 3) float32."""
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
        print(f"    ⚠  Cannot load {path}: {e}")
        return None


def _save_visibility_map(visibility_map: np.ndarray, vis_path: str,
                         ium_h: int, ium_w: int, n_cams: int,
                         fmt: ImageFormat) -> None:
    """Write the (num_pix, n_cams) uint8 visibility to disk.

    EXR → multi-channel, one 0/1 channel per camera (the format pbr_solver and the
    inspectors read). Non-EXR formats → fraction of cameras seeing each texel.
    """
    if fmt == ImageFormat.OPENEXR:
        vis_arr = visibility_map.reshape((ium_h, ium_w, n_cams)).astype(np.float32)
    else:
        ratio = np.sum(visibility_map, axis=1).astype(np.float32) / float(max(n_cams, 1))
        vis_arr = _reshape_flat(ratio, ium_w, ium_h)
    _save_layer(vis_arr, vis_path, fmt, DataLayer.VISIBILITY)


def _load_camera_masks(mask_dir: Path, stems: "list[str]", num_pix: int) -> "np.ndarray | None":
    """Reload the per-camera masks from <mask_dir>/{stem}.exr → (num_pix, n_cams) uint8.

    Returns None if the folder or even a single mask is missing, so the caller knows
    it has to recompute color_texture to regenerate them.
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
    """Decide the effective IUM width/height when an external normal map is used.

    When rc.external_normal_path is None it returns (default_w, default_h, None).

    When the normal map's native resolution already matches default_w×default_h it
    returns that same resolution with mode="match".

    Otherwise:
      - rc.external_normal_resolution_mode == "resample" → (default_w, default_h, "resample")
      - rc.external_normal_resolution_mode == "adapt"    → (native_w, native_h, "adapt")
      - rc.external_normal_resolution_mode is None       → asks the user at runtime
    """
    if not rc.external_normal_path:
        return default_w, default_h, None

    native = _load_image_hw3_native(rc.external_normal_path)
    native_h, native_w = native.shape[:2]

    if native_w == default_w and native_h == default_h:
        print(f"[IUM] External normal: native resolution {native_w}×{native_h} "
              f"matches ium_texture_size → no resampling needed.")
        return default_w, default_h, "match"

    print(f"[IUM] External normal: native resolution {native_w}×{native_h}, "
          f"ium_texture_size={default_w}×{default_h} — resolutions differ.")

    mode = rc.external_normal_resolution_mode
    if mode is None:
        while True:
            ans = input(
                f"  Choose the resolution strategy:\n"
                f"    1 = resample: resize the normal map to {default_w}×{default_h}\n"
                f"    2 = adapt: adapt ium_texture_size to {native_w}×{native_h}\n"
                f"  Choice [1/2]: "
            ).strip()
            if ans in ("1", "2"):
                mode = "resample" if ans == "1" else "adapt"
                break
            print("  Invalid input, enter 1 or 2.")

    if mode == "resample":
        print(f"[IUM] Strategy: resample → the normal map will be resized to {default_w}×{default_h}.")
        return default_w, default_h, "resample"
    elif mode == "adapt":
        print(f"[IUM] Strategy: adapt → the IUM will run at {native_w}×{native_h}.")
        return native_w, native_h, "adapt"
    else:
        raise ValueError(
            f"unrecognised external_normal_resolution_mode: {mode!r}. "
            "Use 'resample', 'adapt' or None."
        )


def _apply_external_normal(rc, ium_res, ium_w: int, ium_h: int) -> None:
    """Decode the external normal map from [0,1] to [-1,1] and inject it into the
    C++ buffer of IUM_Generator::Result, overwriting the face normal OptiX computed.

    After this call ium_res.normals_np (and the matching GPU buffer used by
    IrradianceGenerator / IndirectGenerator) holds the external normal.
    """
    path = rc.external_normal_path
    print(f"[IUM] Loading external normal map: {path}")

    # Load and resize to ium_texture_size (LANCZOS).
    # _load_image_as_vec3 returns (N, 3) float32:
    #   - EXR → raw HDR values (possibly already in [-1,1] or [0,1])
    #   - LDR (PNG/JPG) → uint8/255, hence in [0,1]
    ext = _load_image_as_vec3(path, ium_w, ium_h)  # (N, 3)

    # Decode from the source range into [-1,1].
    # The range must be declared explicitly in rc.external_normal_range: auto-detecting
    # it from the global min() was fragile, because LANCZOS ringing (on aggressive
    # downscales such as 4096→512) pushes some pixels below 0, which skipped the decode
    # and left the values in [0,1] — exactly the bug that was reported.
    if rc.external_normal_range == "0_1":
        n = ext.astype(np.float32) * 2.0 - 1.0
    elif rc.external_normal_range == "-1_1":
        n = ext.astype(np.float32, copy=True)
    else:
        raise ValueError(
            f"unrecognised external_normal_range: {rc.external_normal_range!r} "
            f"(expected '0_1' | '-1_1')."
        )

    if rc.external_normal_flip_green:
        n[:, 1] *= -1.0

    # Renormalise: LANCZOS/interpolation can produce non-unit vectors.
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 1e-8)

    # Zero the texels outside the mesh, consistently with the generated normal.
    if ium_res.has_masks():
        m = ium_res.masks_np.astype(bool)
        n[~m] = 0.0

    # Write into the C++ buffer (normals_np is a writable zero-copy view).
    ium_res.normals_np[:] = n.astype(np.float32)
    print(f"[IUM] External normal injected into the IUM buffer ({ium_w}×{ium_h}, "
          f"{int(np.count_nonzero(m) if ium_res.has_masks() else ium_w * ium_h)} valid texels).")


# ──────────────────────────────────────────────────────────────────────────────
# Texture-space test ROI (Step 3 + Step 4)
#
# The ROI is not a parameter threaded through the pipeline: it is one extra factor
# on the IUM mask, applied once right after the IUM render. Every downstream kernel
# already returns early on texels whose mask is zero (deviceProgramsVis.cu:55,
# ColorTex.cu:22, Irradiance.cu:136, Indirect.cu:39, SpecCone.cu:107,
# HemiVis.cu:94) and every generator re-uploads the mask from the host at its own
# set_inputs, so narrowing masks_np is enough to narrow the whole bake.
# The PBR fit and the albedo inherit it by re-reading ium/ium_masks.exr from disk.
# ──────────────────────────────────────────────────────────────────────────────

def _roi_is_active(rc) -> bool:
    return bool(rc.roi_rect) or bool(rc.roi_mask_path)


def _roi_default_tag(rc) -> str:
    """Default sandbox tag: deterministic, readable, filesystem-safe."""
    parts: list[str] = []
    if rc.roi_mask_path:
        stem = Path(rc.roi_mask_path).stem
        parts.append("".join(c if (c.isalnum() or c in "-_") else "_" for c in stem))
    if rc.roi_rect:
        x0, y0, w, h = (int(v) for v in rc.roi_rect)
        rect = f"{x0}_{y0}_{w}x{h}"
        parts.append(rect if parts else f"rect_{rect}")
    return "_".join(parts) if parts else "roi"


def _roi_assets_dir(rc, json_dir: Path) -> "tuple[Path, str | None]":
    """(destination folder for the Step 3/4 outputs, ROI tag).

    Without a ROI this is the run root, exactly as before. With a ROI active it is
    the sandbox <output_dir>/roi/<tag>/, which mirrors the root layout: the
    full-resolution caches are never touched.
    """
    if not _roi_is_active(rc):
        return json_dir, None
    tag = rc.roi_tag or _roi_default_tag(rc)
    return json_dir / "roi" / tag, tag


def _load_roi(rc, ium_w: int, ium_h: int) -> "tuple[np.ndarray | None, dict]":
    """ROI → (flat bool mask (ium_w·ium_h,) | None, fingerprint).

    ROI = AND of the roi_rect rectangle and the roi_mask_path image, each of them
    optional. The fingerprint describes the ROI *resolved* at the IUM resolution,
    which is what the guard compares, so two ROIs written differently but equivalent
    texel by texel count as the same ROI.
    """
    if not _roi_is_active(rc):
        return None, {}

    roi = np.ones((ium_h, ium_w), dtype=bool)

    if rc.roi_rect:
        if len(rc.roi_rect) != 4:
            raise ValueError(
                f"roi_rect must be [x0, y0, w, h], got {rc.roi_rect!r}")
        x0, y0, w, h = (int(v) for v in rc.roi_rect)
        if w <= 0 or h <= 0:
            raise ValueError(f"roi_rect has a non-positive side: {rc.roi_rect!r}")
        x1, y1 = min(x0 + w, ium_w), min(y0 + h, ium_h)
        x0, y0 = max(x0, 0), max(y0, 0)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"roi_rect {rc.roi_rect!r} lies entirely outside the "
                             f"IUM {ium_w}×{ium_h}")
        rect = np.zeros_like(roi)
        rect[y0:y1, x0:x1] = True
        roi &= rect

    if rc.roi_mask_path:
        if not os.path.exists(rc.roi_mask_path):
            raise FileNotFoundError(f"roi_mask_path not found: {rc.roi_mask_path}")
        img = _load_image_hw3_native(rc.roi_mask_path)[..., 0]
        m = img > rc.roi_mask_threshold
        if m.shape != (ium_h, ium_w):
            # NEAREST and not LANCZOS: on a binary mask, interpolation would move the
            # edge by a few texels, and it is the same ringing that made declaring
            # external_normal_range necessary instead of inferring it.
            #
            from PIL import Image
            m = np.array(Image.fromarray(m.astype(np.uint8) * 255)
                         .resize((ium_w, ium_h), Image.NEAREST)) > 127
            print(f"[ROI] mask resampled {img.shape[1]}×{img.shape[0]} → "
                  f"{ium_w}×{ium_h} (NEAREST)")
        roi &= m

    flat = np.ascontiguousarray(roi.reshape(-1))
    if not flat.any():
        raise ValueError("The ROI contains no texels: check roi_rect / "
                         "roi_mask_path (and the roi_mask_threshold).")

    fingerprint = {
        "rect": [int(v) for v in rc.roi_rect] if rc.roi_rect else None,
        "mask_path": rc.roi_mask_path or None,
        "mask_threshold": float(rc.roi_mask_threshold) if rc.roi_mask_path else None,
        "ium_size": [int(ium_w), int(ium_h)],
        "texels": int(flat.sum()),
        "sha1": hashlib.sha1(np.packbits(flat).tobytes()).hexdigest(),
    }
    return flat, fingerprint


def _check_roi_guard(assets_dir: Path, fingerprint: dict) -> None:
    """Write roi.json into the sandbox, or raise if it already describes another ROI.

    The bakes skip work they find on disk (the per-camera files of
    _precompute_spec_cone, irradiance_indirect.exr, the cached colour texture), so
    reusing the same tag with a different ROI would blend two regions into a single
    set of files with no signal at all, exactly as a re-bake with different
    ring_samples would.
    """
    path = assets_dir / "roi.json"
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            old = json.load(fh)
        if old.get("sha1") != fingerprint.get("sha1"):
            raise RuntimeError(
                f"Incompatible ROI in {assets_dir}\n"
                f"    on disk:   {old.get('texels')} texels, rect={old.get('rect')}, "
                f"mask={old.get('mask_path')}, sha1={str(old.get('sha1'))[:12]}\n"
                f"    requested: {fingerprint['texels']} texels, "
                f"rect={fingerprint['rect']}, mask={fingerprint['mask_path']}, "
                f"sha1={fingerprint['sha1'][:12]}\n"
                f"  The outputs already present would be skipped and mixed with the "
                f"new ROI. Delete the folder, or use a different roi_tag.")
        return
    os.makedirs(assets_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(fingerprint, fh, indent=2)


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
    file_path: str          # resolved absolute path (used to reach the file on disk)
    file_path_original: str # path exactly as it appears in the original JSON
    transform_matrix: list[list[float]]
    sharpness: float = 1.0

    @property
    def stem(self) -> str:
        return Path(self.file_path).stem


@dataclass
class TransformsFile:
    intrinsics: CameraIntrinsics
    frames: list[FrameInfo]
    transforms_dir: str     # folder holding the original transforms.json
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
# Camera extraction from transform_matrix (NeRF / OpenCV convention)
# ──────────────────────────────────────────────────────────────────────────────

def _camera_from_matrix(matrix: list[list[float]], fovy: float, frame_size: list[int], optix_mod) -> object:
    """Derive position, forward and up from the 4×4 NeRF transform_matrix.

    Column convention of the c2w matrix:
      col 0 → right
      col 1 → up
      col 2 → -forward  (NeRF points the camera along -Z)
      col 3 → position
    """
    m = matrix
    pos     = [m[0][3], m[1][3], m[2][3]]
    forward = [-m[0][2], -m[1][2], -m[2][2]]
    up      = [m[0][1],  m[1][1],  m[2][1]]
    return optix_mod.Camera(pos, forward, up, fovy, frame_size)


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RenderConfig:
    transforms_path: str
    model_path: str
    output_dir: str

    # HDR normalization of the source images (Step 1).
    # Divisor = skybox.max() when skybox_path is set, otherwise the max over the source images.
    # The normalised skybox is saved as skybox_normalized.exr and used in Step 3.
    normalize_images: bool = False

    # What to render
    render_depth:    bool = True
    render_position: bool = True
    render_normal:   bool = True
    render_mask     : bool = True   # validity mask for every frame
    render_ium      : bool = True
    render_visibility: bool = True

    # Output format for each layer
    depth_format:    ImageFormat = ImageFormat.OPENEXR
    position_format: ImageFormat = ImageFormat.OPENEXR
    normal_format:   ImageFormat = ImageFormat.OPENEXR
    mask_format:     ImageFormat = ImageFormat.PNG      # uint8 → PNG is the natural fit
    ium_format:      ImageFormat = ImageFormat.PNG
    visibility_format: ImageFormat = ImageFormat.OPENEXR

    # IUM texture size [width, height]
    ium_texture_size: list[int] = field(default_factory=lambda: [512, 512])

    # Externally supplied IUM normal (overrides the one OptiX computes).
    # The file may be in any image format (PNG, JPG, EXR…).
    # When None (the default) the normal computed by the IUM pass is used.
    external_normal_path: str | None = None
    # Strategy to use when the external image resolution ≠ ium_texture_size.
    #   "resample"  → resample the normal map to ium_texture_size (LANCZOS).
    #   "adapt"     → adapt ium_texture_size to the normal map's native resolution
    #                 (positions/mask are regenerated at that same resolution).
    #   None        → ask the user at runtime.
    external_normal_resolution_mode: str | None = None
    # Value range of the external normal map.
    #   "0_1"  → standard normal map encoded in [0,1]: applies the decode n = ext*2-1.
    #   "-1_1" → EXR already decoded to [-1,1] (e.g. a reloaded ium_normals): used as is.
    # Set this explicitly: auto-detecting from the global min() was fragile (LANCZOS
    # ringing on aggressive downscales pushed the min below 0 and skipped the decode).
    external_normal_range: str = "0_1"
    # Flip the green channel after the decode (needed for DirectX-convention bakers).
    # Default False = OpenGL/Blender convention (Y+ upwards).
    external_normal_flip_green: bool = False

    # Scale applied to the depths — must stay False: the `scale` in transforms.json
    # has to be applied to the camera translations TOO, and not doing so creates a
    # mismatch (NeRF query points ≠ mesh surface).  Leave it False.
    apply_scale: bool = False

    # Color texture
    render_color_texture: bool = False
    color_texture_format: ImageFormat = ImageFormat.OPENEXR
    # Percentile used to compute the peak (95 would discard the brightest 5 %)
    color_texture_peak_percentile: float = 100.0
    # Sources from which ALL the source-dependent Step 3 outputs are produced.
    # Each source is processed in full and identically, under sources/{src}/:
    # color_texture/, camera_texture/, pixel_change/, albedo/, metallic/, roughness/,
    # albedo_pbr/, pbr/. Inner names are NOT suffixed (the source is the folder name).
    # The source-independent passes (ium, visibility, irradiance, spec_cone, indirect)
    # stay at the top level, shared.
    # "gt"   → ground-truth images
    # "nerf" → predicted EXRs written by Step 2b into nerf_render_images/iter_*/
    color_texture_image_sources: list[str] = field(default_factory=lambda: ["gt"])
    # Step 2b iteration to read the NeRF predictions from. -1 = use the latest available.
    color_texture_nerf_iter: int = -1
    # Maximum angle (degrees) from the texel normal beyond which a camera's
    # contribution is discarded (too grazing a view → background bleed at the edges).
    # 90.0 = filter disabled; operational default 75° (discards within 15° of the tangent).
    color_texture_grazing_max_deg: float = 75.0

    # Debug output
    debug_camera_texture: bool = False   # save a side-by-side of camera image vs camera_texture

    # Pixel change output
    render_pixel_change: bool = False    # save the min/max/range textures in pixel_change/
    debug_pixel_change: bool = False     # save a comparison plot in debug_pixel_change/

    # Irradiance map (per-texel skybox, deterministic quadrature on a Fibonacci spiral)
    render_irradiance: bool = False
    irradiance_format: ImageFormat = ImageFormat.OPENEXR
    # Source of the envmap the irradiance pass uses:
    #   "file" → equirectangular EXR read from skybox_path
    #   "nerf" → bake of the trained NeRF's background sphere (the checkpoint is resolved
    #            as for the indirect pass: indirect_nerf_cache_path or <output_dir>/model/...).
    #            skybox_path is not required; the baked map is saved as
    #            skybox_nerf_baked.exr for inspection and comparison against the GT.
    skybox_source: str = "file"
    skybox_path: str = ""                # path to the equirectangular EXR file
    skybox_size: list[int] = field(default_factory=lambda: [1024, 512])  # resize target
    irradiance_sample_side: int = 16     # N → N×N samples per hemisphere (16 = 256, 256 = 65536)
    skybox_yaw_degrees: float = 0.0      # skybox yaw; 0° puts -Y (Blender forward) at the centre
    compare_skybox_to_gt: bool = False   # True → write skybox_compare/skybox_heatmap.png after the bake

    # Indirect irradiance via NeRF (precompute once, cache on disk)
    precompute_indirect: bool = False
    indirect_sample_side: int = 64       # N → N×N samples per texel (separate from irradiance)
    indirect_tile_size: int = 1024       # texels per GPU tile (trades memory against VRAM)
    indirect_nerf_cache_path: str = ""   # path to the NeRF checkpoint (default: auto-detect)
    indirect_format: ImageFormat = ImageFormat.OPENEXR
    # When True the indirect pass uses a custom sampling window around the OptiX t_hit
    # (indirect_depth_window / _end) instead of inheriting the one stored in the
    # training checkpoint. Sampling is always centred on t_hit either way.
    indirect_override_depth_window: bool = False
    indirect_depth_window: float = 0.5
    indirect_depth_window_end: float = 0.0

    # Specular cone pass — precomputes L_j(r_k) for the PBR fit C_j = X·D + (1-X)·L_j(r).
    # Sampling on concentric rings around the reflected ray: every ray is traced and
    # queried once, and the cones are closed by a weighted cumulative sum
    # (see _precompute_spec_cone). Requires render_ium, render_visibility and the NeRF
    # checkpoint (like precompute_indirect); it reuses the irradiance skybox.
    precompute_spec_cone: bool = False
    # Sampling scheme:
    #   "per_camera" → concentric rings around R_j, relaunched for every camera
    #                  (spec_cone_samples_per_ring / _sample_alloc / _budget / _floor)
    #   "shared"     → one Fibonacci set uniform over the hemisphere above n, traced and
    #                  queried ONCE, then binned by every camera into its own ring.
    #                  Incident radiance does not depend on the camera, so this costs
    #                  S + m rays/texel instead of m·ΣN_i (m = cameras that see the
    #                  texel). Narrow apertures do cost resolution, though: a cone of
    #                  aperture a receives S·(1−cos(a/2)) samples, so below ~7° (at
    #                  S=16384) it drops under 30 samples. In exchange, refining the
    #                  aperture grid does not cost a single extra ray.
    spec_cone_scheme: str = "per_camera"
    # Shared samples per texel (S), only for scheme="shared"
    spec_cone_shared_samples: int = 16384
    # Texels per torch sub-block: dirs + radiances cost 24 B/ray, so a whole tile
    # would not fit in VRAM. It does not change the result, only the peak usage and
    # the efficiency: small sub-blocks mean small torch kernels and a lot of Python
    # overhead in the loop over cameras.
    spec_cone_chunk_texels: int = 256
    # Rays per batch in the bake's NeRF queries (overrides NerfConfig.chunk, which
    # comes from the checkpoint and is 32768). This is the real limit on GPU
    # occupancy: the network batch is capped there, so raising spec_cone_chunk_texels
    # alone will not fill the card. None = use the checkpoint's value.
    spec_cone_nerf_chunk: "int | None" = None
    # TOTAL cone apertures in degrees, increasing, first element = 0 (mirror ray)
    spec_cone_apertures_deg: list[float] = field(
        default_factory=lambda: [0.0, 10.0, 20.0, 40.0, 60.0, 80.0,
                                 100.0, 120.0, 140.0, 160.0, 180.0])
    # Samples per ring: an int = the same number on every ring (the historical
    # behaviour), or a list[int] with one value per ring (len = apertures - 1;
    # level 0, the mirror ray, is always a single ray).
    spec_cone_samples_per_ring: int | list[int] = 32
    # Automatic allocation when spec_cone_samples_per_ring is an int:
    #   "uniform"     → the same number everywhere
    #   "solid_angle" → N_i ∝ Ω_i, i.e. uniform angular density. The outer ring
    #                   covers ~45× the solid angle of the first, so with a constant
    #                   M it is by far the noisiest; noise on L attenuates β and
    #                   biases the argmin over r (errors-in-variables).
    spec_cone_sample_alloc: str = "uniform"
    # Target Σ_i N_i for the automatic allocation (None → int × number of rings)
    spec_cone_samples_budget: int | None = None
    # Per-ring minimum. The narrow candidates need it, since they use ONLY the inner
    # rings: at 32, no candidate receives fewer samples than the historical uniform
    # allocation, at the cost of ~3 % more rays.
    spec_cone_samples_floor: int = 32
    # Texels per OptiX launch: large tiles mean less launch/sync overhead and larger
    # NeRF batches (query_radiance splits by cfg.chunk anyway). ~40 MB of VRAM.
    # Lower it when raising the budget: host RAM per tile scales with tile × rays/texel.
    spec_cone_tile_size: int = 8192
    spec_cone_cameras: list[int] | None = None  # frame indices to process (None = all)
    spec_cone_format: ImageFormat = ImageFormat.OPENEXR

    # Final PBR maps (pbr_solver) — requires precompute_spec_cone, color_texture with
    # pixel_change, and visibility. Writes metallic/metallic.exr (= 1−X) and
    # roughness/roughness.exr (= r/180 where reliable, 1.0 elsewhere), like the albedo.
    render_pbr_maps: bool = False
    pbr_min_views: int = 2
    pbr_spec_threshold: float = 0.2    # minimum metallic for r to be trusted (0 = no censoring)
    # Also copy metallic/roughness as R/G/B EXRs (metallic_rgb.exr,
    # roughness_rgb.exr): the single 'Z' channel ExrWriter writes is not the
    # convention of Blender's bakes, which replicate the grey over three channels.
    pbr_write_blender_rgb: bool = True
    # Texels per solver band: the fit is per-texel, so the texture is partitioned
    # into blocks of whole scanlines and the peak RAM scales with this value
    # (~tile·n_candidates·8B·20, i.e. ~2.5 GiB at 1 M texels and 14 candidates)
    # rather than with the resolution. It does not change the result.
    pbr_tile_texels: int = 1 << 20

    # Albedo (color_texture / irradiance) — Lambertian model ρ = π · L / E
    render_albedo: bool = False
    albedo_format: ImageFormat = ImageFormat.OPENEXR
    albedo_eps: float = 1e-3             # lower clamp on the irradiance, to avoid /0

    # ── Texture-space test ROI (Step 3 + Step 4) ─────────────────────────────
    # Restricts the computation to a portion of the texture, so a bake or fit
    # parameter can be iterated on without paying the 16.7 M texels of a 4096² IUM.
    # It is not an approximation: the ROI is applied as a factor on the IUM mask,
    # and every downstream kernel already returns early on masked texels, so the
    # texels computed get bit-identical values to a full run. The rest stay 0.
    # With a ROI active, ALL the Step 3 and Step 4 outputs go into a sandbox
    # <output_dir>/roi/<tag>/ mirroring the normal layout: the full-resolution
    # caches are never touched, and cleaning up is just deleting the folder.
    # The inputs (images/, nerf_render_images/) and skybox_nerf_baked.exr stay
    # shared with the full run, so the skybox is not re-baked.
    roi_rect: "list[int] | None" = None   # [x0, y0, w, h] in IUM texels
    roi_mask_path: str = ""               # PNG/EXR: channel 0 > threshold = inside the ROI
    roi_mask_threshold: float = 0.5
    # Effective ROI = AND of rectangle and mask (each optional). Both empty means
    # behaviour identical to before (no sandbox).
    roi_tag: str = ""                     # sandbox name ("" = derived from rect/mask)


@dataclass
class PipelineConfig:
    """Four toggle-able steps.

    Step 1: produce depth+mask+images+transforms_extended.json (the NeRF minimum).
    Step 2: train the NeRF (nerf/train.py) and save the checkpoint.
    Step 3: run IUM/visibility/color_texture/irradiance/indirect/spec_cone (the bake).
    Step 4: reconstruction (PBR fit + albedo), reading only the on-disk cache of
            Step 3 — kept separate so the reconstruction can be iterated on
            without re-baking the cones.
    """
    run_step1: bool = True
    run_step2: bool = True
    run_step3: bool = True
    run_step4: bool = True
    # When True, and with run_step2=True, skip the NeRF training for a scene if the
    # checkpoint <output_dir>/model/nerf_model_cache.pt already exists (useful to
    # resume an interrupted sweep without repeating the training).
    # Caveat: an incomplete checkpoint would be reused; to force retraining, delete
    # that configuration's .pt file.
    resume_skip_step2_if_ckpt: bool = False

    render: RenderConfig = field(default_factory=RenderConfig)

    # Parameters of nerf/train.py (Step 2)
    nerf_num_iters:        int   = 10000
    nerf_batch_size:       int   = 4096
    nerf_lr:               float = 5e-4
    nerf_display_every:    int   = 100
    nerf_seed:             int   = 9458
    nerf_ckpt_path:        str   = ""  # default: <output_dir>/model/nerf_model_cache.pt
    nerf_train_output_dir: str   = ""  # default: <output_dir>/nerf_train

    # Depth-guided training (Step 2) — requires depth+mask from Step 1
    nerf_depth_window_samples: int   = 32    # samples in the mesh window for foreground rays
    nerf_depth_window:         float = 0.5   # [t_hit - window, t_hit + window_end]
    nerf_depth_window_end:     float = 0.5
    nerf_opacity_weight:       float = 1.0   # weight of the opacity loss (fg and bg)
    nerf_raw_noise_std:        float = 0.0   # pre-ReLU noise on the density
    nerf_bg_radius_mult:       float = 6.0   # bg sphere radius = bg_radius_mult × max distance from the origin
    nerf_bg_depth_window:      float = 2.0   # bg window [R - window, R + window_end]
    nerf_bg_depth_window_end:  float = 2.0
    nerf_profile_iters: int = 0         # synchronized per-phase timing for the first N iters (0=off)
    nerf_multires:       int   = 10
    nerf_multires_views: int   = 4

    # RGB activation and training loss (Step 2).
    # nerf_rgb_activation: "exp" (HDR) | "softplus"
    # nerf_loss_type:      "l1" | "mse" | "rel_mse" (eps outside the square) |
    #                      "rel_mse_raw" (faithful RawNeRF, eps inside the square) | "log_l1"
    # N.B. checkpoints saved with one activation are NOT compatible with the other.
    nerf_rgb_activation: str   = "exp"
    nerf_loss_type:      str   = "rel_mse_raw"

    # Learning-rate decay factor: new_lr = lr * (nerf_lr_decay ** min(i/decay_steps, 1.0)).
    # 0.2 → the lr decays to 20 % of its initial value at the nerf_lr_decay_steps horizon;
    # past that point the LR plateaus (lr*factor, it does not fall further).
    # Sweepable to compare decay regimes: values < 0.2 are more aggressive, values
    # > 0.2 gentler. Propagated to NerfConfig.lr_decay_factor.
    nerf_lr_decay: float = 0.2

    # FIXED horizon (absolute iterations) the LR decay is spread over. 0 = auto → use
    # nerf_num_iters (a fresh run behaves exactly as before). Setting it to a fixed
    # value (e.g. the total planned length) makes a resumed training continue the
    # decay without a jump. Propagated to NerfConfig.lr_decay_steps.
    nerf_lr_decay_steps: int = 0

    # Render the training frames with the trained NeRF (post-Step 2)
    enable_nerf_render_train_images: bool = False
    nerf_render_train_images_dir:    str  = ""  # default: <output_dir>/nerf_render_images

    # When True, ask the user whether to continue training at the end of each round
    nerf_interactive_loop: bool = True



@dataclass
class SceneConfig:
    """Per-scene fields, used by run_pipeline_multi."""
    name: str                               # name of the output subfolder (e.g. "SwordShield")
    transforms_path: str
    model_path: str
    external_normal_path: str | None = None  # overrides RenderConfig only when not None
    skybox_path: str | None = None           # used only when skybox_source == "file"
    note: str = ""                           # optional scene-specific note


def _resolve_nerf_ckpt_path(cfg: RenderConfig) -> str:
    """Path of the NeRF checkpoint used by the Step 3 passes (indirect, skybox bake)."""
    path = cfg.indirect_nerf_cache_path
    if not path:
        path = os.path.join(cfg.output_dir, "model", "nerf_model_cache.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"NeRF model cache not found: {path}\n"
            "Set indirect_nerf_cache_path, or run Step 2 first."
        )
    return path


def _bake_skybox_from_nerf(cfg: RenderConfig, sky_w: int, sky_h: int,
                           json_dir: Path) -> np.ndarray:
    """Bake the NeRF background sphere into an equirectangular envmap (skybox_source="nerf").

    Saves skybox_nerf_baked.exr in json_dir for inspection and returns (N, 3) float32
    in the same flat layout as _load_image_as_vec3.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from nerf import load_checkpoint, bake_envmap
    import torch

    ckpt_path = _resolve_nerf_ckpt_path(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_bundle, nerf_cfg = load_checkpoint(ckpt_path, device)
    print(f"[Step 3] Baking the skybox from the NeRF ({sky_w}×{sky_h}, "
          f"yaw={cfg.skybox_yaw_degrees}°) — ckpt: {ckpt_path}")
    baked = bake_envmap(model_bundle, nerf_cfg, sky_w, sky_h,
                        yaw_degrees=cfg.skybox_yaw_degrees)
    out_path = (json_dir / "skybox_nerf_baked.exr").resolve().as_posix()
    get_writer(ImageFormat.OPENEXR).write(baked, out_path)
    print(f"[Step 3] Baked skybox saved: {out_path}")
    return baked.reshape(-1, 3)


def _resolve_skybox_flat(cfg: RenderConfig, output_json: dict, json_dir: Path,
                         sky_w: int, sky_h: int) -> np.ndarray:
    """Flat skybox (H*W, 3) for the Step 3 passes (irradiance, spec_cone).

    skybox_source="nerf" → bake from the NeRF background sphere; otherwise prefer
    the normalised skybox from the JSON (same scale as colour+NeRF) when present.
    """
    if cfg.skybox_source == "nerf":
        baked = json_dir / "skybox_nerf_baked.exr"
        if baked.exists():
            print(f"[Step 3] NeRF skybox reused from disk: {baked} "
                  "(delete the file to force a re-bake)")
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
    """Run the OptiX pass tile by tile, query the NeRF for every occluded ray and
    write irradiance_indirect.exr to disk.  Called only when precompute_indirect=True.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from nerf import load_checkpoint, query_radiance
    import OptixProgrammablePasses as optix

    # ── Load the NeRF model from the cache ────────────────────────────────────
    cache_path = _resolve_nerf_ckpt_path(cfg)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_bundle, nerf_cfg = load_checkpoint(cache_path, device)
    if cfg.indirect_override_depth_window:
        nerf_cfg.depth_window     = cfg.indirect_depth_window
        nerf_cfg.depth_window_end = cfg.indirect_depth_window_end
    print(f"✓ NeRF model loaded from: {cache_path}")

    # ── OptiX pass, tile by tile ──────────────────────────────────────────────
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

    # Tiles without a single active texel: the kernel returns early on every thread
    # and the tile would yield count=0. Skipping the launch is therefore equivalent,
    # and enough to make the cost scale with the ROI instead of with the texture
    # resolution. It helps without a ROI too, where empty atlas regions still pay.
    ium_mask_flat = np.asarray(ium_res.masks_np).reshape(-1) > 0
    n_skipped = 0

    print(f"  Indirect precompute: {n_tiles} tiles × {cfg.indirect_tile_size} texels, "
          f"N={cfg.indirect_sample_side}")

    for tile_idx in range(n_tiles):
        off = tile_idx * cfg.indirect_tile_size
        if not ium_mask_flat[off:off + cfg.indirect_tile_size].any():
            n_skipped += 1
            continue

        tile_res = ind_gen.render_tile(tile_idx)
        count    = tile_res.count
        if count == 0:
            continue

        local_idx = tile_res.local_idx_np.copy()
        dirs_np   = tile_res.directions_np.copy()
        cos_np    = tile_res.cos_np.copy()
        t_hit_np  = tile_res.t_hit_np.copy()

        global_idx  = off + local_idx

        origins_np = (ium_positions_np[global_idx]
                      + ium_normals_np[global_idx] * eps)

        colors = query_radiance(model_bundle, origins_np, dirs_np, nerf_cfg, t_hits_np=t_hit_np)

        np.add.at(irr_indirect, global_idx,
                  colors * cos_np[:, None].astype(np.float64))

        if (tile_idx + 1) % max(1, n_tiles // 10) == 0:
            print(f"    tile {tile_idx+1}/{n_tiles}, occluded rays: {count}")

    if n_skipped:
        print(f"    {n_skipped}/{n_tiles} tiles skipped (no active texel)")

    irr_indirect = (irr_indirect * scale).astype(np.float32)

    os.makedirs(os.path.dirname(indirect_path), exist_ok=True)
    irr_indirect_arr = _reshape_flat(irr_indirect, ium_w, ium_h)
    _save_layer(irr_indirect_arr, indirect_path, cfg.indirect_format,
                DataLayer.IRRADIANCE_INDIRECT)
    print(f"✓ irradiance_indirect saved: {indirect_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Shared Fibonacci set — host-side reconstruction of the kernel's directions
#
# These functions replicate sharedDirection()/buildONB()/rotationFromIndex() of
# deviceProgramsHemiVis.cu BIT FOR BIT: the kernel returns only the t_hits, indexed
# by position, so a divergence would pair every t_hit with the wrong direction with
# no visible symptom other than a wrong L_j.
# The parity is verified by scripts/test_hemivis_shared.py.
# ──────────────────────────────────────────────────────────────────────────────

_HEMIVIS_INV_GOLDEN = 0.6180339887498948482   # 1/φ = (√5 − 1)/2
_HEMIVIS_TWO_PI     = 6.283185307179586477


def _hemivis_rotation(global_idx: np.ndarray) -> np.ndarray:
    """Per-texel azimuthal rotation in [0, 1) (lowbias32 hash of global_idx).

    Decorrelates the QMC pattern between neighbouring texels: without it, every texel
    would use the same directions up to the ONB and the noise would align into bands.
    """
    with np.errstate(over="ignore"):          # the uint32 overflow IS the intended semantics
        x = np.asarray(global_idx, dtype=np.uint32)
        x = x ^ (x >> np.uint32(16))
        x = x * np.uint32(0x7feb352d)
        x = x ^ (x >> np.uint32(15))
        x = x * np.uint32(0x846ca68b)
        x = x ^ (x >> np.uint32(16))
    return (x >> np.uint32(8)).astype(np.float64) * (1.0 / 16777216.0)


def _hemivis_onb(n):
    """Frisvad 2012 branchless ONB around n (torch, (..., 3) float32)."""
    import torch
    nz  = n[..., 2]
    sgn = torch.copysign(torch.ones_like(nz), nz)
    a   = -1.0 / (sgn + nz)
    b   = n[..., 0] * n[..., 1] * a
    T = torch.stack([1.0 + sgn * n[..., 0] * n[..., 0] * a, sgn * b, -sgn * n[..., 0]], dim=-1)
    B = torch.stack([b, sgn + n[..., 1] * n[..., 1] * a, -n[..., 1]], dim=-1)
    return T, B


def _hemivis_directions(normals, global_idx: np.ndarray, num_samples: int):
    """Shared directions (n_texel, S, 3) float32 around the given normals.

    Uniform in solid angle over the hemisphere above n: cosθ_s = 1 − (s + 0.5)/S,
    azimuth on the golden sequence with a per-texel rotation. The azimuth arithmetic
    is in float64 and reduced to [0, 2π) BEFORE the trigonometry, as in the kernel:
    in float32 s·goldenAngle reaches ~4·10⁴ rad, where one ULP is already 0.23°.
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
    """Equirectangular lookup (torch), identical to sampleEnvmap() in the CUDA kernels.

    World Z-up, Y-forward (Blender): zenith = +Z, u = 0.5 − atan2(dy,dx)/2π.
    envmap: (H*W, 3) float32 on the device; sky_size = [W, H].
    """
    import torch
    w, h = int(sky_size[0]), int(sky_size[1])
    dz = torch.clamp(dirs[..., 2], -1.0, 1.0)

    u = 0.5 - torch.atan2(dirs[..., 1], dirs[..., 0]) * (1.0 / (2.0 * np.pi)) + yaw_offset_u
    u = u - torch.floor(u)
    v = 0.5 - torch.asin(dz) * (1.0 / np.pi)

    px = torch.clamp((u * w).to(torch.int64), 0, w - 1)   # (int) truncates, u ≥ 0 ⇒ floor
    py = torch.clamp((v * h).to(torch.int64), 0, h - 1)
    return envmap[py * w + px]


def spec_cone_shared_ring_samples(apertures_deg, num_samples: int) -> list[float]:
    """NOMINAL per-ring counts of the shared bake: N_i = S·Ω_i/(2π).

    With uniform sampling in solid angle, the expected number of samples in a ring is
    proportional to its solid angle, so the solver weights W_i = Ω_i/N_i = 2π/S become
    constant and `ring_weights_mean` collapses onto the plain cumulative mean
    L(k) = Σ_{i≤k} sum_i / Σ_{i≤k} count_i.
    Writing these values into the meta is what leaves the solver unchanged.
    """
    c = np.cos(np.radians(np.asarray(apertures_deg, dtype=np.float64)) * 0.5)
    return [float(num_samples) * float(c[i] - c[i + 1]) for i in range(c.size - 1)]


def ring_weights_mean(cos_edges, k: int,
                      ring_samples: "np.ndarray | None" = None) -> np.ndarray:
    """Solid-angle weights of the cone truncated at ring k (pure mean):
    W_i = Ω_i/N_i with Ω_i = 2π(c_{i-1} − c_i) for i ≤ k and 0 beyond, where
    c = clip(cos b, 0, 1) and N_i = rays launched on ring i.
    The normalization by Σ_i W_i·valid_i happens per texel at accumulation time
    (rays below the horizon leave both numerator and denominator).

    ring_samples=None (or a uniform one) reproduces the historical behaviour EXACTLY:
    with a constant N the 1/N factor cancels in num/den, and skipping the division
    also avoids its rounding error.

    It lives here and not in pbr_solver because it is bake mathematics: ever since
    spec_cone writes the cones directly, the solver weights nothing at all. The
    kernel tests import it from this module.
    """
    c = np.clip(np.asarray(cos_edges, dtype=np.float64), 0.0, 1.0)
    w = 2.0 * np.pi * (c[:-1] - c[1:])
    if ring_samples is not None:
        n = np.asarray(ring_samples, dtype=np.float64)
        if n.shape != w.shape:
            raise ValueError(f"ring_samples: expected {w.size} values "
                             f"(one per ring), got {n.size}")
        if n.min() <= 0.0:
            raise ValueError("ring_samples: every ring requires N_i > 0")
        if n.max() != n.min():      # uniform → global factor, an exact no-op
            w = w / n
    w[k:] = 0.0
    return w


def spec_cone_ring_samples(apertures_deg, samples_per_ring, alloc="uniform",
                           budget=None, floor=32) -> list[int]:
    """Samples to LAUNCH on rings 1..K-1 (level 0, the mirror ray, is always a
    single ray, so it does not appear here).

    When samples_per_ring is a sequence it is used as is and `alloc` is ignored.
    Otherwise the allocation is derived from the solid angles of the rings,
    Ω_i = 2π(c_{i-1} − c_i) with c = cos(aperture/2):
      "uniform"     → N_i = samples_per_ring on every ring
      "solid_angle" → N_i ∝ Ω_i, normalised to the budget and clamped to the floor,
                      so every ray covers roughly the same solid angle.
    """
    ap = np.asarray(apertures_deg, dtype=np.float64)
    n_rings = ap.size - 1
    if n_rings < 1:
        raise ValueError("spec_cone_apertures_deg requires at least 2 values")

    if not isinstance(samples_per_ring, (int, np.integer)):
        n = [int(x) for x in samples_per_ring]
        if len(n) != n_rings:
            raise ValueError(f"spec_cone_samples_per_ring: {len(n)} values, "
                             f"expected {n_rings} (apertures - 1)")
        if min(n) < 1:
            raise ValueError("spec_cone_samples_per_ring: every ring requires "
                             "at least 1 sample")
        return n

    m = int(samples_per_ring)
    if m < 1:
        raise ValueError("spec_cone_samples_per_ring must be >= 1")
    if alloc == "uniform":
        return [m] * n_rings
    if alloc != "solid_angle":
        raise ValueError(f"unknown spec_cone_sample_alloc: {alloc!r} "
                         "(expected 'uniform' or 'solid_angle')")

    c = np.cos(np.radians(ap) * 0.5)
    omega = 2.0 * np.pi * (c[:-1] - c[1:])
    total = int(budget) if budget is not None else m * n_rings
    n = np.rint(total * omega / omega.sum())
    return [int(max(x, floor)) for x in n]


def spec_cone_level_name(apertures_deg, k: int) -> str:
    """Name of level k: the APERTURE, not the index.

    This is the name read in the viewer (tev groups channels by prefix and shows one
    layer per level), so it should carry the useful datum: `cone_045deg` and not
    `cone06`. The constraints that explain the exact form:

    - no dots: in EXR channel names the dot separates layer from channel, so
      `cone_007.5deg.R` would be read as layer `cone_007`, sublayer `5deg`.
      Fractional degrees therefore use `p` as the decimal separator;
    - a zero-padded 3-digit integer part, so the alphabetical order of the layers
      matches the angular one (without padding, `cone_5deg` would land after
      `cone_180deg`);
    - level 0 is the mirror ray, a delta direction and not an integral over a cone:
      it is named for what it is.
    """
    if k == 0:
        return "cone_000_mirror"
    a = float(apertures_deg[k])
    if a == int(a):
        return f"cone_{int(a):03d}deg"
    frac = f"{a - int(a):.4f}".split(".")[1].rstrip("0")
    return f"cone_{int(a):03d}p{frac}deg"


def spec_cone_channels(apertures_deg) -> "dict[str, type]":
    """Channels of a camera's cone EXR: L_j(r) per candidate, plus validity.

    One file per camera rather than one per aperture: in the shared bake the outer
    loop is over tiles, so the writers of every camera stay open simultaneously and
    K+1 files per camera would make 840 handles, past the MSVC stdio limit (512).
    Level names come from spec_cone_level_name, so writing and reading cannot
    diverge.

    `valid` is the total number of valid rays of the texel (those above the
    horizon, across all levels): >0 is the same per-camera mask the ring bake used
    to write into valid.png.

    Every channel is float32. The cones were half until 2026-08-10, but half
    saturates at 65504 and the mirror level carries an unaveraged envmap value:
    see the comment in the body. The on-disk format is transparent to the reader
    anyway, because read_cones reads with PixelType.FLOAT.
    """
    ch: "dict[str, type]" = {}
    for k in range(len(apertures_deg)):
        name = spec_cone_level_name(apertures_deg, k)
        for c in "RGB":
            # float32 and not half: level 0 is a SINGLE envmap SAMPLE (the mirror
            # ray, never averaged), and on a night scene the light sources overflow
            # the half range. On the NeRF skybox of SwordShieldNight the R channel
            # of the strongest lamp core is 80609 against a limit of 65504: the cast
            # wrote inf, and one inf in a cone channel drops the texel out of the PBR
            # fit entirely, because np.argmin propagates the resulting nan. The other
            # levels are means over ≥9 rays and had margin, but two formats are not
            # worth the trouble.
            ch[f"{name}.{c}"] = np.float32
    ch["valid"] = np.float32
    return ch


# L_j(r) for every candidate, from the raw per-ring sums and counts:
#     candidate 0 = mirror ray (pure level 0)
#     candidate k = pure solid-angle mean over the cone truncated at ring k,
#                   L_k = Σ_{i≤k} W_i·sum_i / Σ_{i≤k} W_i·count_i
# Numerator and denominator are cumulative over the rings, so every candidate comes
# out of a single cumsum. `weights` are the UNTRUNCATED W_i = Ω_i/N_i
# (ring_weights_mean with k = K-1): the truncation is done by the cumsum.
# It starts from the sums and not from the per-ring means because the old path
# (bake → half means on disk → solver re-averaging) quantized an intermediate step
# that does not exist here.

def _cones_from_rings_np(ring_sum: np.ndarray, ring_valid: np.ndarray,
                         weights: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """(N, K, 3), (N, K), (K-1,) → (N, K, 3). See the comment above."""
    num = np.cumsum(ring_sum[:, 1:] * weights[None, :, None], axis=1)
    den = np.cumsum(ring_valid[:, 1:] * weights[None, :], axis=1)
    mirror = ring_sum[:, :1] / np.maximum(ring_valid[:, :1, None], 1.0)
    return np.concatenate([mirror, num / np.maximum(den, eps)[..., None]], axis=1)


def _cones_from_rings_torch(ring_sum, ring_valid, weights, eps: float = 1e-12):
    """(…, K, 3), (…, K), (K-1,) → (…, K, 3), all on device. See above."""
    import torch
    num = torch.cumsum(ring_sum[..., 1:, :] * weights[:, None], dim=-2)
    den = torch.cumsum(ring_valid[..., 1:] * weights, dim=-1)
    mirror = ring_sum[..., :1, :] / torch.clamp(ring_valid[..., :1, None], min=1.0)
    return torch.cat([mirror, num / torch.clamp(den, min=eps)[..., None]], dim=-2)


def _tile_bar(total: int, desc: str):
    """Progress bar for the tile loops of the spec_cone bakes.

    A bake takes hours, and periodic prints say neither how much has elapsed nor how
    much is left; tqdm gives the ETA and the percentage in one place. The output goes
    through the _Tee of _console_to_file, so console.log collects the intermediate
    frames too: mininterval keeps them to one every two seconds.
    """
    from tqdm import tqdm
    return tqdm(total=total, unit="tile", desc=desc, mininterval=2.0,
                dynamic_ncols=True, smoothing=0.05)


def _tile_bar_step(bar, rays_per_tile: int, n: int = 1) -> None:
    """Advance the bar by n tiles, updating the throughput.

    The throughput is in rays/s and not tiles/s: a tile is tile_size × S rays in the
    shared scheme and tile_size × (1 + Σ N_i) in the per-camera one, so tiles/s are
    not comparable across configurations while rays/s are.
    """
    bar.update(n)
    elapsed = bar.format_dict["elapsed"]
    if elapsed > 0:
        bar.set_postfix_str(f"{rays_per_tile * bar.n / elapsed / 1e6:.1f} Mrays/s",
                            refresh=False)


def _precompute_spec_cone(
    cfg: RenderConfig,
    ium_res,            # IUM_Generator.Result
    model,              # OptixProgrammablePasses.TriangleMesh
    ium_w: int,
    ium_h: int,
    frames,             # tf.frames (for the camera positions)
    visibility_map: np.ndarray,   # flat (num_pix * n_cams) uint8
    n_cams: int,
    skybox_flat: "np.ndarray | None",
    sky_size: list[int],
    out_dir: Path,
) -> None:
    """Per-ring precompute for the PBR fit  C_j = (a·x/π)·E + (1-x)·L_j  (pbr_solver.py).

    Sampling on concentric rings around the reflected ray R_j = reflect(v_j, n)
    (deviceProgramsSpecCone.cu): every ray is traced and queried on the NeRF exactly
    once. The rings remain how the accumulation happens, but the bake CLOSES the
    cones before writing: what goes to disk is the pure solid-angle mean L_j(r) of
    every candidate (level 0 = mirror ray), which is exactly the quantity the solver
    puts into the regression.
    Miss → envmap (on the GPU), hit → NeRF.

    Output in out_dir: cam_{j:03d}.exr with one RGB channel group per level, named
    after its aperture (cone_000_mirror, cone_005deg, …), plus valid, and
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
    print(f"✓ NeRF model loaded from: {cache_path}")

    apertures = [float(a) for a in cfg.spec_cone_apertures_deg]
    K = len(apertures)                  # levels: 0 = mirror, 1..K-1 = cones

    ring_samples = spec_cone_ring_samples(
        apertures, cfg.spec_cone_samples_per_ring,
        alloc=cfg.spec_cone_sample_alloc,
        budget=cfg.spec_cone_samples_budget,
        floor=cfg.spec_cone_samples_floor)
    rays_per_texel = 1 + sum(ring_samples)
    rays_per_tile  = rays_per_texel * cfg.spec_cone_tile_size
    print(f"    samples/ring {ring_samples} → {rays_per_texel} rays/texel, "
          f"{rays_per_tile:,} rays/tile "
          f"(~{rays_per_tile * 24 / 2**20:.0f} MB device, "
          f"~{rays_per_tile * 84 / 2**20:.0f} MB RAM)")
    if rays_per_tile > 4_000_000:
        suggested = max(256, (4_000_000 // rays_per_texel) // 256 * 256)
        print(f"    ⚠  high rays/tile: consider spec_cone_tile_size={suggested}")

    gen = optix.SpecConeGenerator()
    gen.set_traversable(model)
    gen.set_inputs(ium_res, apertures, ring_samples, cfg.spec_cone_tile_size)
    if skybox_flat is not None:
        gen.set_envmap(skybox_flat.astype(np.float32), sky_size,
                       cfg.skybox_yaw_degrees)
    else:
        print("    ⚠  spec_cone without a skybox: missed rays contribute 0")

    num_pix = gen.num_pixels()
    n_tiles = gen.num_tiles()
    tile_sz = cfg.spec_cone_tile_size

    # Ring edges: cosines of the half-apertures (stored in the meta, for the solver)
    cos_b = np.cos(np.radians(np.asarray(apertures)) * 0.5)

    ium_positions_np = ium_res.positions_np.astype(np.float32)
    ium_normals_np   = ium_res.normals_np.astype(np.float32)
    eps = 1e-4

    ium_mask_flat = np.asarray(ium_res.masks_np).reshape(-1) > 0
    vis2d = np.asarray(visibility_map, dtype=np.uint8).reshape(num_pix, n_cams)
    cam_indices = (list(cfg.spec_cone_cameras) if cfg.spec_cone_cameras
                   else list(range(len(frames))))

    os.makedirs(out_dir, exist_ok=True)
    # The cones are multi-channel HDR and are written with IncrementalExrWriter:
    # spec_cone_format survives only as the extension declared in the meta.
    fmt = ImageFormat.OPENEXR

    # Cameras already on disk are skipped, but the meta is always rewritten: with a
    # different sampling one would get old EXRs described by new ring_samples, i.e.
    # cones normalised by different N from the ones they were closed with, with no
    # signal at all. Better to stop.
    meta_path = out_dir / "spec_cone_meta.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as fh:
            old_meta = json.load(fh)
        old_rs = old_meta.get("ring_samples")
        old_ap = old_meta.get("apertures_deg")
        if (old_meta.get("format") != "cones" or old_ap != apertures
                or (old_rs is not None and list(old_rs) != ring_samples)):
            raise RuntimeError(
                f"spec_cone: {out_dir} holds an incompatible bake\n"
                f"    on disk:   format={old_meta.get('format')}, "
                f"apertures={old_ap}, ring_samples={old_rs}\n"
                f"    requested: format=cones, apertures={apertures}, "
                f"ring_samples={ring_samples}\n"
                f"  Delete the spec_cone/ folder, or restore the previous "
                f"configuration. Bakes in the 'rings'/'rings_shared' formats "
                f"(per-ring means) are no longer readable: ever since the bake "
                f"writes the cones directly, the solver no longer reconstructs "
                f"them, so they have to be redone.")

    def _cam_path(j: int) -> Path:
        return out_dir / f"cam_{j:03d}{fmt.extension}"

    # Untruncated Ω_i/N_i weights: the truncation at each candidate is done by the
    # cumsum inside _cones_from_rings_np. Here the N_i are the ones actually launched
    # and generally non-uniform, so the 1/N_i factor matters.
    cone_w = ring_weights_mean(cos_b, K - 1, np.asarray(ring_samples, dtype=np.float64))

    # A single bar over cameras × tiles: one bar per camera would reopen 60 times
    # without ever giving an ETA for the whole bake. Cameras already on disk stay
    # out of the total, otherwise the initial ETA would count work that will never
    # be done.
    pending = [j for j in cam_indices if not _cam_path(j).exists()]
    bar = _tile_bar(len(pending) * n_tiles, "spec_cone")

    for j in cam_indices:
        cam_path = _cam_path(j)
        if j not in pending:
            print(f"    cam {j}: already on disk, skipped")
            continue
        bar.set_description(f"spec_cone cam {j}", refresh=False)

        m = frames[j].transform_matrix
        cam_pos = [float(m[0][3]), float(m[1][3]), float(m[2][3])]
        gen.set_camera(cam_pos, np.ascontiguousarray(vis2d[:, j]))

        ring_sum = np.zeros((num_pix, K, 3), dtype=np.float64)
        valid    = np.zeros((num_pix, K),    dtype=np.int64)

        # Texels this camera produces anything for: the kernel returns early on both
        # the IUM mask and the visibility (deviceProgramsSpecCone.cu:107-109).
        active = ium_mask_flat & (vis2d[:, j] > 0)

        for tile_idx in range(n_tiles):
            off = tile_idx * tile_sz
            # Tiles with no active texel: the kernel returns early on every thread and
            # renderTile zeroes sky_sum/valid_count at every launch
            # (SpecCone_Generator.cpp:336), so leaving ring_sum/valid at zero is
            # identical to launching it.
            if not active[off:off + tile_sz].any():
                _tile_bar_step(bar, rays_per_tile)
                continue

            tile_res = gen.render_tile(tile_idx)
            if tile_res.overflow:
                raise RuntimeError(
                    f"spec_cone cam {j} tile {tile_idx}: compact buffer overflow "
                    f"({tile_res.requested} rays requested). The capacity is the "
                    f"exact worst case, so the bake would be incomplete: aborted "
                    f"instead of saving partial means.")
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
                # accumulate via a tile-local bincount (np.add.at is unbuffered and
                # far slower on millions of indices)
                flat_idx = local_idx.astype(np.int64) * K + ring_idx
                n_bins   = tt * K
                tile_acc = ring_sum[off:off + tt].reshape(n_bins, 3)
                for c in range(3):
                    tile_acc[:, c] += np.bincount(flat_idx, weights=colors[:, c],
                                                  minlength=n_bins)

            _tile_bar_step(bar, rays_per_tile)

        # Close the cones: L_j(r) per candidate, 0 where there was no sample
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
        print(f"    ✓ cam {j}: {K} cones saved to {cam_path.name}")

    bar.close()

    meta = {
        "format": "cones",
        "scheme": "per_camera",
        "apertures_deg": apertures,
        "ring_edges_cos": [float(c) for c in cos_b],
        # samples_per_ring stays an informational scalar for historical readers;
        # ring_samples are the N_i the bake weighted the rings with (Ω_i/N_i) when
        # closing the cones: documentation of the bake, not an input of the solver.
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
    print(f"✓ spec_cone meta saved: {out_dir / 'spec_cone_meta.json'}")


# ──────────────────────────────────────────────────────────────────────────────
# spec_cone bake with sampling SHARED between cameras
# ──────────────────────────────────────────────────────────────────────────────

def _precompute_spec_cone_shared(
    cfg: RenderConfig,
    ium_res,            # IUM_Generator.Result
    model,              # OptixProgrammablePasses.TriangleMesh
    ium_w: int,
    ium_h: int,
    frames,             # tf.frames (for the camera positions)
    visibility_map: np.ndarray,   # flat (num_pix * n_cams) uint8
    n_cams: int,
    skybox_flat: "np.ndarray | None",
    sky_size: list[int],
    out_dir: Path,
) -> None:
    """Variant of _precompute_spec_cone with the rays shared between all cameras.

    The incident radiance along a direction does not depend on the camera, so a
    single Fibonacci set per texel (uniform in solid angle over the hemisphere above
    n) serves every camera that sees that texel: each ray is traced and queried on
    the NeRF ONCE, and every camera bins it into its own ring by the angle to its
    R_j. Cost per texel is `S + m` instead of `m · Σ N_i` (m = cameras that see the
    texel).

    Level 0 (the mirror) stays per camera: it is a delta direction, not shareable,
    and comes from the kernel's second pass.

    Outputs in out_dir: cam_{j:03d}.exr with one RGB channel group per level, named
    after its aperture (cone_000_mirror, cone_005deg, …), holding the mean radiance
    over the cone, i.e. directly the L_j(r) the solver puts into the regression, plus
    valid (total valid rays of the texel); written streaming, in scanline blocks,
    plus spec_cone_meta.json with format "cones".
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
    print(f"✓ NeRF model loaded from: {cache_path} (query chunk {nerf_cfg.chunk})")

    apertures = [float(a) for a in cfg.spec_cone_apertures_deg]
    K = len(apertures)                       # levels: 0 = mirror, 1..K-1 = rings
    S = int(cfg.spec_cone_shared_samples)
    cos_b = np.cos(np.radians(np.asarray(apertures)) * 0.5)
    ring_nominal = spec_cone_shared_ring_samples(apertures, S)

    num_pix  = ium_w * ium_h
    tile_sz  = int(cfg.spec_cone_tile_size)
    if tile_sz % ium_w != 0:
        raise ValueError(
            f"spec_cone_tile_size={tile_sz} must be a multiple of the IUM width "
            f"({ium_w}): the shared bake streams the EXRs, so every tile has to "
            f"cover a whole number of scanlines.")
    chunk_texels = max(1, int(cfg.spec_cone_chunk_texels))

    vis2d = np.asarray(visibility_map, dtype=np.uint8).reshape(num_pix, n_cams)
    cam_indices = (list(cfg.spec_cone_cameras) if cfg.spec_cone_cameras
                   else list(range(len(frames))))
    n_sel = len(cam_indices)

    # ── Diagnostic m: how many cameras see a texel on average ────────────────
    # This is the number that decides the cost relative to the per-camera bake: that
    # one spends m·Σ N_i rays per texel, this one S + m.
    ium_mask = np.asarray(ium_res.masks_np).reshape(num_pix) > 0
    if ium_mask.any():
        m_per_texel = vis2d[np.ix_(ium_mask, cam_indices)].sum(axis=1)
        m_mean = float(m_per_texel.mean())
        print(f"    m (cameras per texel): mean {m_mean:.1f}, median "
              f"{np.median(m_per_texel):.0f}, p10 {np.percentile(m_per_texel, 10):.0f}, "
              f"p90 {np.percentile(m_per_texel, 90):.0f}")
        try:
            # informational comparison with the per-camera bake; its parameters may be
            # inconsistent with this grid (this scheme does not use them), and a
            # diagnostic must not make the bake fail
            per_cam_rays = 1 + sum(spec_cone_ring_samples(
                apertures, cfg.spec_cone_samples_per_ring,
                alloc=cfg.spec_cone_sample_alloc,
                budget=cfg.spec_cone_samples_budget,
                floor=cfg.spec_cone_samples_floor))
            print(f"    expected cost vs per-camera: {m_mean * per_cam_rays:.0f} → "
                  f"{S + m_mean:.0f} rays/texel "
                  f"({m_mean * per_cam_rays / max(S + m_mean, 1.0):.2f}×)")
        except ValueError:
            print(f"    expected cost: {S + m_mean:.0f} rays/texel")

    # Expected samples per candidate: tells which apertures are at the noise limit
    cum = [f"{apertures[k]:g}°:{S * (1.0 - cos_b[k]):.0f}" for k in range(1, K)]
    print(f"    S={S} shared rays/texel, samples per candidate → {', '.join(cum)}")

    # ── Guard against an inconsistent re-bake ────────────────────────────────
    # As in the per-camera bake: the meta is always rewritten, so a bake on disk with
    # different parameters would end up described by a new meta and the solver would
    # normalise by the wrong N with no signal at all.
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
                f"spec_cone: {out_dir} holds an incompatible bake\n"
                f"    on disk:   format={old_meta.get('format')}, "
                f"scheme={old_meta.get('scheme')}, "
                f"apertures={old_meta.get('apertures_deg')}, "
                f"S={old_meta.get('shared_samples')}\n"
                f"    requested: format=cones, scheme=shared, "
                f"apertures={apertures}, S={S}\n"
                f"  Delete the spec_cone/ folder, or restore the previous "
                f"configuration. Bakes in the 'rings'/'rings_shared' formats "
                f"(per-ring means) are no longer readable: ever since the bake "
                f"writes the cones directly, the solver no longer reconstructs "
                f"them, so they have to be redone.")
        if all(p.exists() for p in cam_paths.values()):
            print(f"    all {n_sel} cameras already on disk, skipped")
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
        raise RuntimeError(f"spec_cone: the IUM has {gen.num_pixels()} texels, "
                           f"expected {num_pix} from ium_texture_size {ium_w}×{ium_h}")

    n_tiles = gen.num_tiles()
    rays_per_tile = tile_sz * S
    print(f"    tile={tile_sz} texels ({tile_sz // ium_w} scanlines), {n_tiles} tiles, "
          f"{rays_per_tile:,} rays/tile (~{rays_per_tile * 4 / 2**20:.0f} MB t_hit), "
          f"torch chunk={chunk_texels} texels "
          f"(~{chunk_texels * S * 24 / 2**20:.0f} MB VRAM for dirs+radiances)")

    if skybox_flat is None:
        print("    ⚠  spec_cone without a skybox: missed rays contribute 0")
        envmap_t = None
    else:
        envmap_t = torch.as_tensor(np.ascontiguousarray(skybox_flat, dtype=np.float32),
                                   device=device)
    yaw_u = cfg.skybox_yaw_degrees / 360.0

    pos_all = np.asarray(ium_res.positions_np, dtype=np.float32)
    nrm_all = np.asarray(ium_res.normals_np,   dtype=np.float32)
    eps = 1e-4

    cos_edges_t = torch.as_tensor(cos_b, device=device, dtype=torch.float32)
    asc_edges   = -cos_edges_t                       # increasing, for searchsorted
    cam_pos_t   = torch.as_tensor(np.asarray(cam_pos_list, dtype=np.float32), device=device)
    vis_sel     = np.ascontiguousarray(vis2d[:, cam_indices])       # (num_pix, n_sel)

    # Untruncated Ω_i/N_i weights: the truncation at each candidate is done by the
    # cumsum inside _cones_from_rings_torch. In the shared bake the nominal N_i make
    # W_i constant, so the formula collapses onto the plain cumulative mean.
    cone_w_t = torch.as_tensor(
        ring_weights_mean(cos_b, K - 1, np.asarray(ring_nominal)),
        device=device, dtype=torch.float32)

    # ── Streaming writers, one per camera ────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    channels = spec_cone_channels(apertures)
    level_names = [spec_cone_level_name(apertures, k) for k in range(K)]
    writers = {j: IncrementalExrWriter(cam_paths[j].resolve().as_posix(),
                                       ium_w, ium_h, channels)
               for j in cam_indices}

    bar = _tile_bar(n_tiles, "spec_cone shared")
    n_skipped = 0
    try:
        for tile_idx in range(n_tiles):
            off = tile_idx * tile_sz

            # Tiles with no active texel: no rays to trace or query on the NeRF, but
            # the scanline block still has to be written, because IncrementalExrWriter
            # streams and the rows must stay in order. The zeros are exactly what the
            # full path produces: texel_ok false → sums/counts at zero →
            # _cones_from_rings_torch returns 0, and valid is the sum of the counts.
            #
            if not ium_mask[off:off + tile_sz].any():
                rows_out = min(tile_sz, num_pix - off) // ium_w
                zeros = np.zeros((rows_out, ium_w), dtype=np.float32)
                zero_block = {name: zeros for name in channels}
                for wr in writers.values():
                    wr.write_block(zero_block)
                n_skipped += 1
                _tile_bar_step(bar, rays_per_tile)
                continue

            tile_res = gen.render_tile(tile_idx)
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

                # Mirror rays: R_j per camera, radiance through the same logic
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

                    # Rings 1..K-1 from the shared rays. Samples beyond the widest cone
                    # land in bin K, which is discarded afterwards: that avoids a
                    # boolean gather over M·S elements.
                    m_sel  = sel.numel()
                    cosang = (dirs[sel] * R[sel, jj][:, None, :]).sum(-1)      # (m, S)
                    # clamp to [1, K]: 0 is the mirror level (never reachable from the
                    # shared set, though cosang can exceed 1 by rounding), and K is the
                    # discard bin for samples outside the widest cone.
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

                    # Level 0: mirror ray (t_hit < 0 = camera behind the surface)
                    mir_ok = sel[thm[sel, jj] >= 0.0]
                    if mir_ok.numel() > 0:
                        lvl0 = torch.zeros((mir_ok.numel(), K, 3), device=device)
                        lvl0[:, 0] = radm[mir_ok, jj]
                        cnt0 = torch.zeros((mir_ok.numel(), K), device=device)
                        cnt0[:, 0] = 1.0
                        sums[jj].index_add_(0, mir_ok + c0, lvl0)
                        counts[jj].index_add_(0, mir_ok + c0, cnt0)

            # ── Write the scanline block, one camera at a time ───────────────
            # The cones are closed here, on the GPU and from the raw sums: what goes
            # to disk is already L_j(r), the quantity the solver regresses on.
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
        if n_skipped:
            print(f"    {n_skipped}/{n_tiles} tiles skipped (no active texel), "
                  f"written as zeros")
        for wr in writers.values():
            wr.close()
    except BaseException:
        bar.close()
        # A truncated EXR still has a valid header: were it left on disk, a rerun with
        # the same meta would mistake it for a finished bake and skip it. Better to
        # delete it.
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
        # NOMINAL counts N_i = S·Ω_i/2π used for the bake's W_i = Ω_i/N_i weights:
        # here they are constant, so the cone is the plain cumulative mean.
        # By now this is documentation of the bake, not an input of the solver.
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
    print(f"✓ spec_cone meta saved: {meta_path}")


def _shared_ray_radiance(dirs, t_hit, origin, envmap_t, sky_size, yaw_u,
                         model_bundle, nerf_cfg, query_radiance):
    """Incident radiance per ray: envmap on misses, NeRF on hits.

    dirs (M, R, 3), t_hit (M, R) with >0 hit, =0 miss, <0 ray not launched;
    origin (M, 3) is the origin shared by the rays of the same texel.
    Returns (M, R, 3) on the device (zero where the ray was not launched).
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
    """Return the iter_* subfolder for the requested iteration (or the highest one)."""
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
    """Build the list of optix_mod.Frame using the images of the given source.

    Args:
        source: "gt" (original images) | "nerf" (predicted EXRs from Step 2b).
        tf: the loaded transforms object (tf.frames).
        all_cameras: list of optix_mod.Camera, one per frame.
        intr: intrinsics (intr.w, intr.h).
        rc: the current RenderConfig.
        cfg: the current PipelineConfig (for nerf_render_train_images_dir).
        json_dir: base folder of the run (Path).
        optix_mod: the imported OptixProgrammablePasses module.
    """
    nerf_pred_dir = None
    if source == "nerf":
        base_root = Path(cfg.nerf_render_train_images_dir or
                         json_dir / "nerf_render_images")
        nerf_pred_dir = _find_nerf_pred_dir(base_root, rc.color_texture_nerf_iter)
        if nerf_pred_dir is None:
            print(f"    ⚠  No NeRF prediction folder found (source '{source}') → falling back to GT images")
        else:
            print(f"[Step 3] Colour texture from NeRF predictions ({source}): {nerf_pred_dir}")

    optix_frames = []
    for i, frame in enumerate(tf.frames):
        cam = all_cameras[i]
        img_path = frame.file_path
        if nerf_pred_dir is not None:
            pred_path = _nerf_pred_path(nerf_pred_dir, i)
            if pred_path.exists():
                img_path = pred_path.as_posix()
            else:
                print(f"    ⚠  NeRF prediction missing for frame {i} ({pred_path.name}), using GT")
        img_flat = _load_image_as_vec3(img_path, intr.w, intr.h)
        peak = _compute_peak(img_flat.reshape(intr.h, intr.w, 3),
                             rc.color_texture_peak_percentile)
        optix_frames.append(optix_mod.Frame(cam, peak, img_flat))
    return optix_frames


def _step1_pretrain_data(cfg: PipelineConfig, optix_mod) -> Path:
    """Render depth + mask for every frame, copy the RGB images and write
    transforms_extended.json with the minimum fields NerfDataset requires.
    """
    rc = cfg.render
    tf = load_transforms(rc.transforms_path)
    intr = tf.intrinsics
    print(f"[Step 1] Transforms loaded: {len(tf.frames)} frames  [{intr.w}×{intr.h}]")

    json_dir = Path(rc.output_dir).resolve()
    json_dir_str = json_dir.as_posix()
    os.makedirs(json_dir, exist_ok=True)

    model = optix_mod.TriangleMesh()
    model.add_from_obj_file(rc.model_path)
    print(f"[Step 1] Model loaded: {rc.model_path}")

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

    # ── Compute the HDR normalization divisor ─────────────────────────────────
    norm_divisor: float | None = None
    norm_source: str | None = None
    sky_norm_path: str | None = None  # path of the saved normalised skybox
    if rc.normalize_images:
        if rc.skybox_path:
            sky_raw = _load_image_hw3_native(rc.skybox_path)
            norm_divisor = float(sky_raw.max())
            norm_source = "skybox"
            print(f"[Step 1] Normalization: skybox → max={norm_divisor:.6f}")
            # Save the normalised skybox — Step 3 uses it for radiometric consistency
            sky_normalized = (sky_raw / norm_divisor).astype(np.float32)
            sky_norm_path = (json_dir / "skybox_normalized.exr").as_posix()
            get_writer(ImageFormat.OPENEXR).write(sky_normalized, sky_norm_path)
            sky_norm_path = _as_relative_to(sky_norm_path, json_dir_str)
            print(f"[Step 1] Normalised skybox saved: {(json_dir / sky_norm_path).resolve()}")
        else:
            running_max = 0.0
            print("[Step 1] Normalization: scanning the images for the global max…")
            for frame in tf.frames:
                src = Path(frame.file_path)
                if not src.exists():
                    continue
                arr = _load_image_hw3_native(str(src))
                running_max = max(running_max, float(arr.max()))
            norm_divisor = running_max
            norm_source = "images"
            print(f"[Step 1] Normalization: images → max={norm_divisor:.6f}")
        if norm_divisor <= 0:
            print("[Step 1] ⚠  max = 0, normalization disabled for this run.")
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
                print(f"    ⚠  Image not found, skipped: {src_image}")
        else:
            dst_image = images_out_dir / src_image.name
            if src_image.exists():
                shutil.copy2(src_image, dst_image)
                arr = _load_image_hw3_native(str(dst_image)).astype(np.float32)
            else:
                print(f"    ⚠  Image not found, copy skipped: {src_image}")

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
    print(f"\n[Step 1] Minimal JSON saved to: {out_json_path}")
    return out_json_path


def _step2_train_nerf(
    cfg: PipelineConfig,
    transforms_extended_path: Path,
    tb_logger=None,
) -> tuple[Path, float]:
    """Train the NeRF and return (ckpt_path, final_psnr_dB).

    ``tb_logger`` is a monitoring.RunLogger (or None to disable TB logging).
    The PSNR is the one from the last display block; float('nan') when the training
    is too short to reach the first block.
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
    print(f"[Step 2] depth-guided sampling, mesh_window=[t-{cfg.nerf_depth_window}, t+{cfg.nerf_depth_window_end}], "
          f"bg_radius_mult={cfg.nerf_bg_radius_mult}, lr_decay={cfg.nerf_lr_decay}, "
          f"lr_decay_steps={cfg.nerf_lr_decay_steps or cfg.nerf_num_iters} "
          f"({'auto' if cfg.nerf_lr_decay_steps == 0 else 'fixed'})")
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
    print(f"[Step 2] Training complete. Checkpoint: {ckpt}")
    return ckpt, (final_psnr if final_psnr is not None else float("nan"))


def _write_png_float(arr: np.ndarray, path: str) -> None:
    """Save a float32 [0,1] array as a uint8 PNG."""
    from PIL import Image
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(path)


def _write_sxs_comparison(gt_np: np.ndarray, pred_np: np.ndarray,
                           psnr: float, path: str, label: str = "") -> None:
    """Save a side-by-side GT | Pred PNG with the PSNR in the header."""
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
        print("    ⚠  matplotlib not available: histogram skipped")
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
        print("    ⚠  matplotlib not available: histogram skipped")
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
    """Load a depth EXR as (H, W) float32; returns None on failure."""
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
        print(f"    ⚠  Cannot load depth {path}: {e}")
        return None


def _load_mask_bool(path: str) -> np.ndarray | None:
    """Load a PNG mask as (H, W) bool (True = foreground). None on failure."""
    try:
        from PIL import Image
        arr = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        return arr > 0.5
    except Exception as e:
        print(f"    ⚠  Cannot load mask {path}: {e}")
        return None


# GT luminance bands (linear HDR) used by the error-per-luminance plots.
_LUMA_BINS = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.2), (0.2, 1.0), (1.0, 5.0), (5.0, np.inf)]


def _write_error_luminance_plot(gt_np: np.ndarray, pred_np: np.ndarray,
                                 sel: np.ndarray | None, path: str, title: str) -> None:
    """Bar chart of the error grouped by GT luminance band.

    sel: boolean (H, W) mask of the pixels to include; None = the whole image.
    Shows, per band: the mean absolute |pred-gt| (left axis) and the mean relative
    error (right axis), with the pixel count of each band.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("    ⚠  matplotlib not available: error plot skipped")
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
    """Render every frame with the NeRF and save GT, Pred and diff as EXR + PNG.

    The outputs go into a subfolder named iter_<NNNNNN> inside nerf_render_images, so
    each stop of the interactive training produces a separate directory instead of
    overwriting the previous ones.

    Colour-bias metrics produced at the end:
      frame_NNN_bias.png   — pred-vs-gt density scatter (4 panels R/G/B/Luma, log-log)
      bias_scatter_all.png — scatter aggregated over all frames
      metrics_per_frame.csv — PSNR, tonemapped PSNR, percentile error, per-frame residuals
      bias_bins.csv         — median pred/gt ratio per luminance band and channel
      metrics_summary.txt   — textual summary (the numbers quoted in the thesis)
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

    # ── Accumulators for the aggregate scatter (subsampled to bound memory) ─────
    _MAX_AGG_PX = 20_000   # maximum pixels taken from each frame for the aggregate
    agg_pred_rgb: list[np.ndarray] = []   # (N_i, 3) for frame i
    agg_gt_rgb:   list[np.ndarray] = []

    # ── Per-frame metrics ─────────────────────────────────────────────────────
    psnrs: list[float] = []
    metrics_rows: list[dict] = []

    print(f"\n[Step 2b] Rendering {len(tf.frames)} frames with the NeRF (iter={iter_done})...")
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
        # EXR: HDR float32, no clamp — preserves values >1 and the sign of the difference
        exr_writer.write(gt_np,             (base / f"{stem}_gt.exr").as_posix())
        exr_writer.write(pred_np,           (base / f"{stem}_pred.exr").as_posix())
        exr_writer.write(pred_np - gt_np,   (base / f"{stem}_diff.exr").as_posix())
        # PNG: visual preview, clamped to [0,1]
        _write_png_float(gt_np,   (base / f"{stem}_gt.png").as_posix())
        _write_png_float(pred_np, (base / f"{stem}_pred.png").as_posix())
        _write_sxs_comparison(gt_np, pred_np, psnr,
                               (base / f"{stem}_sxs.png").as_posix(), f"frame {i}")

        # RGB histogram comparison: pred (top) vs GT (bottom)
        _write_rgb_hist_comparison(pred_np, gt_np,
                                   (base / f"{stem}_rgb_hist.png").as_posix(), stem)

        # ── Colour-bias metrics ───────────────────────────────────────────────
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

        # Per-frame bias scatter (R/G/B/Luma log-log, with bisector + median curve)
        bias_title = (f"{stem}  PSNR={psnr:.2f} dB  "
                      f"tonemap-clip={psnr_tm_clip:.2f} dB  "
                      f"tonemap-Reinhard={psnr_tm_reinhard:.2f} dB")
        plot_bias_scatter(pred_np, gt_np, mask_bool,
                          str(base / f"{stem}_bias.png"), title=bias_title)

        # Per-frame diagnostic heatmap: GT, Pred (clip [0,1]) + ΔR ΔG ΔB + |Δ| luma
        # No mask: the whole frame goes in (model + skybox/background).
        plot_error_heatmap(pred_np, gt_np,
                           str(base / f"{stem}_heatmap.png"), title=bias_title)

        # Accumulate a subsample for the aggregate scatter (already masked)
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

    # ── Aggregate scatter + bias_bins.csv ─────────────────────────────────────
    all_pred = np.concatenate(agg_pred_rgb, axis=0)   # (N_total, 3)
    all_gt   = np.concatenate(agg_gt_rgb,   axis=0)
    n_agg    = all_pred.shape[0]

    # Aggregate scatter — virtual reshape to (1, N, 3), no mask (already filtered)
    agg_pred_hw3 = all_pred.reshape(1, n_agg, 3)
    agg_gt_hw3   = all_gt.reshape(1, n_agg, 3)
    agg_title = (f"All frames aggregated ({n_agg} sampled pixels, "
                 f"{len(psnrs)} frames, mean PSNR={float(np.mean(psnrs)):.2f} dB)")
    plot_bias_scatter(agg_pred_hw3, agg_gt_hw3, None,
                      str(base / "bias_scatter_all.png"), title=agg_title)

    # Channels for bias_bins.csv (luma recomputed from the aggregated RGB)
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
        f"=== NeRF colour-bias analysis — iter {iter_done} ===",
        f"Frames evaluated        : {len(metrics_rows)}",
        f"Pixels in the scatter   : {n_agg}",
        "",
        "── PSNR ──────────────────────────────────────────",
        f"  PSNR linear (HDR)     : {float(np.mean(psnrs)):.3f} dB",
        f"  PSNR tonemap clip     : {_nanmean(psnr_tm_clips):.3f} dB",
        f"  PSNR tonemap Reinhard : {_nanmean(psnr_tm_reinhards):.3f} dB",
        "  (tonemap clip ≈ diffuse-range quality without the HDR tail)",
        "",
        "── Highlights (GT luminance percentiles) ─────────",
        f"  Mean relative error p99     : {_nanmean(rel_p99s):.4f}",
        f"  Mean relative error p99.9   : {_nanmean(rel_p999s):.4f}",
        "",
        "── Signed residuals (pred − gt) ──────────────────",
        f"  Global mean     : {_nanmean(res_means):.5f}  (>0 overestimate, <0 underestimate)",
        f"  Global median   : {_nanmean(res_medians):.5f}",
        f"  Highlight mean (gt>1) : {_nanmean(res_hl_means):.5f}",
        "",
        "── Median pred/gt ratio per band (aggregated channels) ───",
        "   ratio < 1 = underestimate, ratio > 1 = overestimate",
        "   (see bias_bins.csv for the full detail)",
    ]
    for ch_name, _cp, _cg in _channels:
        ch_bins = [r for r in bins_rows if r["channel"] == ch_name and r["ratio"] != "nan"]
        hl_bins = [r for r in ch_bins if float(r["center_gt"]) > 1.0]
        diff_bins = [r for r in ch_bins if float(r["center_gt"]) <= 1.0]
        ratio_diff = float(np.mean([float(r["ratio"]) for r in diff_bins])) if diff_bins else float("nan")
        ratio_hl   = float(np.mean([float(r["ratio"]) for r in hl_bins]))   if hl_bins else float("nan")
        lines.append(f"  {ch_name:<5}  diffuse range (gt≤1): {ratio_diff:.3f}  "
                     f"highlights (gt>1): {ratio_hl:.3f}" if not np.isnan(ratio_hl)
                     else f"  {ch_name:<5}  diffuse range (gt≤1): {ratio_diff:.3f}  "
                          f"highlights (gt>1): n/a (no samples)")

    summary_path = base / "metrics_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"[Step 2b] Done — mean PSNR={float(np.mean(psnrs)):.2f} dB  "
          f"tonemap-clip={_nanmean(psnr_tm_clips):.2f} dB → {base}")
    print(f"  Written: metrics_per_frame.csv, bias_bins.csv, "
          f"metrics_summary.txt, bias_scatter_all.png")


def _step3_posttrain_assets(
    cfg: PipelineConfig,
    transforms_extended_path: Path,
    optix_mod,
    tb_logger=None,
    timer=None,
) -> dict:
    """Run IUM/visibility/color_texture/irradiance/indirect/spec_cone (the bake) and
    update transforms_extended.json in place with the new keys.

    The PBR fit and the albedo moved to Step 4 (_step4_reconstruction), which reads
    only the on-disk cache produced here.
    """
    rc = cfg.render
    tf = load_transforms(str(transforms_extended_path))
    intr = tf.intrinsics
    print(f"[Step 3] {len(tf.frames)} frames  [{intr.w}×{intr.h}]")

    json_dir = Path(rc.output_dir).resolve()
    os.makedirs(json_dir, exist_ok=True)

    # With a ROI active, ALL the outputs of this step go into the roi/<tag>/ sandbox,
    # which mirrors the root layout. json_dir remains the source of the shared inputs
    # (images/, nerf_render_images/) and of skybox_nerf_baked.exr, which the ROI run
    # reuses instead of re-baking.
    assets_dir, roi_tag = _roi_assets_dir(rc, json_dir)
    assets_dir_str = assets_dir.as_posix()
    os.makedirs(assets_dir, exist_ok=True)
    roi_fp: dict = {}   # ROI fingerprint, resolved together with the IUM

    model = optix_mod.TriangleMesh()
    model.add_from_obj_file(rc.model_path)
    print(f"[Step 3] Model loaded: {rc.model_path}")

    # Read the existing JSON — it gets enriched at the end
    with open(transforms_extended_path, encoding="utf-8") as fh:
        output_json = json.load(fh)

    ium_result_data: dict = output_json.get("ium", {})

    if rc.render_ium:
        _t3_ium = time.perf_counter()
        # IUM size: it may be adapted to the external normal map's resolution when
        # external_normal_resolution_mode == "adapt" (or the user picks it at runtime).
        default_ium_w, default_ium_h = rc.ium_texture_size[0], rc.ium_texture_size[1]
        ium_w, ium_h, _ext_norm_mode = _resolve_external_normal_size(
            rc, default_ium_w, default_ium_h
        )

        ium_gen = optix_mod.IUMGenerator()
        ium_gen.set_traversable(model)
        ium_gen.set_texture_size([ium_w, ium_h])
        ium_gen.render()
        ium_res = ium_gen.get_result()
        print("[Step 3] IUM rendering complete")
        if timer is not None:
            timer.record("step3/ium", time.perf_counter() - _t3_ium)

        # When an external normal map was supplied, decode it and inject it into the
        # C++ buffer of IUM_Generator::Result before any downstream use.
        if rc.external_normal_path:
            _apply_external_normal(rc, ium_res, ium_w, ium_h)

        # ── ROI: the one place it is applied ─────────────────────────────────
        # masks_np is a writable view on the C++ vector, and every downstream generator
        # re-uploads the mask from the host at its own set_inputs, so from here on
        # visibility, colour texture, irradiance, indirect and spec cone work only on
        # the ROI texels. After _apply_external_normal and not before, so that
        # ium_normals stays the full normal map and only the mask carries the ROI.
        roi_flat, roi_fp = _load_roi(rc, ium_w, ium_h)
        if roi_flat is not None:
            if not ium_res.has_masks():
                raise RuntimeError(
                    "A ROI was requested but the IUM produced no masks: the ROI is "
                    "applied precisely as a factor on ium_masks, and without them "
                    "there is nothing to narrow.")
            _check_roi_guard(assets_dir, roi_fp)
            m = ium_res.masks_np
            n_before = int(np.count_nonzero(m))
            m[~roi_flat] = 0
            n_after = int(np.count_nonzero(m))
            print(f"[ROI] tag '{roi_tag}': {n_after} active texels out of {ium_w * ium_h} "
                  f"({100.0 * n_after / (ium_w * ium_h):.2f} %), "
                  f"{n_before} on the mesh without the ROI → {assets_dir}")

        ium_out_dir = assets_dir / "ium"
        os.makedirs(ium_out_dir, exist_ok=True)

        if ium_res.has_positions():
            pos_arr = _reshape_flat(ium_res.positions_np.astype(np.float32), ium_w, ium_h)
            ium_pos_path = (ium_out_dir / f"ium_positions{rc.ium_format.extension}").resolve().as_posix()
            _save_layer(pos_arr, ium_pos_path, rc.ium_format, DataLayer.POSITION)
            ium_result_data["ium_positions_path"] = _as_relative_to(ium_pos_path, assets_dir_str)

        if ium_res.has_normals():
            norm_arr = _reshape_flat(ium_res.normals_np.astype(np.float32), ium_w, ium_h)
            # With an external normal map we force EXR, to guarantee the raw values in
            # [-1, 1] are stored regardless of rc.ium_format.
            norm_save_fmt = (
                ImageFormat.OPENEXR if rc.external_normal_path else rc.ium_format
            )
            ium_norm_path = (ium_out_dir / f"ium_normals{norm_save_fmt.extension}").resolve().as_posix()
            _save_layer(norm_arr, ium_norm_path, norm_save_fmt, DataLayer.NORMAL)
            ium_result_data["ium_normals_path"] = _as_relative_to(ium_norm_path, assets_dir_str)

        if ium_res.has_masks():
            mask_arr = _reshape_flat(ium_res.masks_np, ium_w, ium_h)
            ium_mask_path = (ium_out_dir / f"ium_masks{rc.ium_format.extension}").resolve().as_posix()
            _save_layer(mask_arr, ium_mask_path, rc.ium_format, DataLayer.MASK)
            ium_result_data["ium_masks_path"] = _as_relative_to(ium_mask_path, assets_dir_str)

        all_cameras = []
        for frame in tf.frames:
            cam = _camera_from_matrix(frame.transform_matrix, intr.camera_angle_y, [intr.w, intr.h], optix_mod)
            all_cameras.append(cam)

        # ── Visibility ───────────────────────────────────────────────────────
        # The pass computes OCCLUSION (a shadow ray camera→texel). The authoritative
        # artefact on disk is however the visibility refined by the colour-texture
        # masks (occlusion∧frustum∧grazing): visibility.exr is therefore written in
        # the colour-texture block (or as the occlusion-only fallback further down,
        # when colour texture does not run). Nothing is saved here.
        visibility_map = None
        visibility_refined = False   # True once visibility.exr = the masks (frustum+grazing)
        vis_path = None
        if rc.render_visibility and ium_res.has_positions() and ium_res.has_masks():
            _t3_vis = time.perf_counter()
            print("[Step 3] Computing camera visibility…")
            vis_gen = optix_mod.VisibilityGenerator()
            vis_gen.set_traversable(model)
            visibility_map = vis_gen.check_visibility(ium_res, ium_w, ium_h, all_cameras)

            vis_out_dir = assets_dir / "visibility"
            os.makedirs(vis_out_dir, exist_ok=True)
            vis_path = (vis_out_dir / f"visibility{rc.visibility_format.extension}").resolve().as_posix()
            ium_result_data["visibility_path"] = _as_relative_to(vis_path, assets_dir_str)
            if timer is not None:
                timer.record("step3/visibility", time.perf_counter() - _t3_vis)

        # ── Color Texture ────────────────────────────────────────────────────
        # color_by_source: source → colors_np float32 (an explicit copy: the result,
        # the generator and the frames are released at the end of the iteration, before
        # the next source is loaded). The per-camera textures are downloaded from the
        # GPU one camera at a time: the frames×texel block never exists in host RAM.
        color_by_source: dict[str, np.ndarray] = {}
        # The per-camera colour-texture masks (occlusion∧frustum∧grazing, pre-peak) are
        # source-independent: they are saved once as <out>/camera_mask/{stem}.exr and
        # reused as the shared refined visibility.
        cam_mask_dir = assets_dir / "camera_mask"
        stems = [f.stem for f in tf.frames]
        if (rc.render_color_texture and rc.render_ium and rc.render_visibility
                and visibility_map is not None):
            _t3_ct = time.perf_counter()

            # Every source is processed in full and identically, under sources/{src}/
            ct_sources = rc.color_texture_image_sources
            if not ct_sources:
                raise ValueError("color_texture_image_sources cannot be empty")

            for src in ct_sources:
                src_dir = assets_dir / "sources" / src
                ct_dir = src_dir / "color_texture"
                os.makedirs(ct_dir, exist_ok=True)
                ct_path = (ct_dir / f"color_texture{rc.color_texture_format.extension}").resolve().as_posix()
                cam_tex_dir = src_dir / "camera_texture"

                if os.path.exists(ct_path) and cam_tex_dir.is_dir():
                    # Cache hit: to avoid regressing the visibility we need the per-camera
                    # masks. If they are on disk we reload them and refine; if they are
                    # missing we do NOT use the cache (we recompute to regenerate them).
                    masks_disk = _load_camera_masks(cam_mask_dir, stems, ium_w * ium_h)
                    if masks_disk is not None:
                        loaded = _load_exr_as_flat(ct_path)
                        if loaded is not None:
                            print(f"[Step 3] Colour texture found on disk ({src}): {ct_path}")
                            color_by_source[src] = loaded
                            ium_result_data[f"color_texture_path_{src}"] = _as_relative_to(ct_path, assets_dir_str)
                            if not visibility_refined and vis_path is not None:
                                visibility_map = masks_disk
                                _save_visibility_map(visibility_map, vis_path, ium_h, ium_w,
                                                     len(all_cameras), rc.visibility_format)
                                visibility_refined = True
                                print("[Step 3] Visibility refined from the per-camera masks on disk")
                            continue
                    else:
                        print(f"[Step 3] Colour texture cached ({src}) but per-camera masks "
                              f"missing → recomputing to regenerate them")

                print(f"[Step 3] Computing the colour texture (source: {src})…")
                optix_frames = _build_optix_frames_for_source(
                    src, tf, all_cameras, intr, rc, cfg, json_dir, optix_mod)

                ct_gen = optix_mod.ColorTexGenerator()
                ct_gen.set_inputs(ium_res, visibility_map, optix_frames,
                                  grazing_max_deg=rc.color_texture_grazing_max_deg)
                ct_gen.render()
                _ct_res = ct_gen.get_result()

                # Save color_texture for this source
                ct_arr = _reshape_flat(_ct_res.colors_np.astype(np.float32), ium_w, ium_h)
                _save_layer(ct_arr, ct_path, rc.color_texture_format, DataLayer.POSITION)
                ium_result_data[f"color_texture_path_{src}"] = _as_relative_to(ct_path, assets_dir_str)

                # Lightweight copy of colors_np, for the downstream albedo computation
                color_by_source[src] = np.array(_ct_res.colors_np, dtype=np.float32)

                # Per-camera textures for this source: sources/{src}/camera_texture/
                # The per-camera mask (uint8) is downloaded too, to refine the visibility.
                os.makedirs(cam_tex_dir, exist_ok=True)
                # The masks are source-independent: they are saved once, on the first
                # source that produces them (before visibility_refined becomes True).
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

                # Refine the shared visibility with the per-camera mask (once):
                # occlusion∧frustum∧grazing, so that spec_cone (which uses visibility_map
                # in memory) and pbr_solver (which re-reads visibility.exr) do not weight
                # cameras that do not really see the texel. No change to the solver.
                if not visibility_refined and vis_path is not None:
                    visibility_map = cam_masks
                    _save_visibility_map(visibility_map, vis_path, ium_h, ium_w,
                                         len(all_cameras), rc.visibility_format)
                    visibility_refined = True
                    print("[Step 3] Visibility refined with the per-camera colour-texture "
                          "mask (frustum+grazing) and re-saved")

                # pixel_change for each source: sources/{src}/pixel_change/
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

                # TensorBoard for this source
                if tb_logger is not None:
                    tb_logger.log_image(f"texture/color_texture_{src}", ct_arr, step=0, tonemap=True)
                    tb_logger.flush()

                # Explicit release before the next source: the host result (~4 layers of
                # num_pix), the generator (which owns the camera_colors VRAM buffer,
                # ~num_pix×frames, and the loaded images) and the frames.
                del _ct_res, ct_gen, optix_frames

            if timer is not None:
                timer.record("step3/color_texture", time.perf_counter() - _t3_ct)

        # ── Visibility refinement from persisted masks (2-bis), and fallback ──
        # If colour texture did not refine the visibility (e.g. render_color_texture
        # off) but the per-camera masks exist on disk, refine from those so spec_cone
        # and pbr_solver still use the frustum+grazing version.
        if not visibility_refined and visibility_map is not None and vis_path is not None:
            masks_disk = _load_camera_masks(cam_mask_dir, stems, ium_w * ium_h)
            if masks_disk is not None:
                visibility_map = masks_disk
                _save_visibility_map(visibility_map, vis_path, ium_h, ium_w,
                                     len(all_cameras), rc.visibility_format)
                visibility_refined = True
                print("[Step 3] Visibility refined from the per-camera masks on disk (2-bis)")
            else:
                # Fallback: no mask available → save the occlusion-only visibility
                # (frustum/grazing NOT applied).
                _save_visibility_map(visibility_map, vis_path, ium_h, ium_w,
                                     len(all_cameras), rc.visibility_format)
                print("    ⚠  visibility.exr saved as OCCLUSION-ONLY (no per-camera mask "
                      "on disk): frustum/grazing NOT applied. Run colour texture to "
                      "refine it.")

        # ── Irradiance (skybox, deterministic quadrature) ────────────────────
        irr_res = None
        skybox_flat_step3 = None   # shared between irradiance and spec_cone
        if (rc.render_irradiance
                and (rc.skybox_path or rc.skybox_source == "nerf")
                and ium_res.has_positions() and ium_res.has_normals()):
            _t3_irr = time.perf_counter()
            print(f"[Step 3] Computing the irradiance "
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
            irr_out_dir = assets_dir / "irradiance"
            os.makedirs(irr_out_dir, exist_ok=True)
            irr_path = (irr_out_dir / f"irradiance{rc.irradiance_format.extension}").resolve().as_posix()
            irr_arr = _reshape_flat(irr_res.irradiance_np.astype(np.float32), ium_w, ium_h)
            _save_layer(irr_arr, irr_path, rc.irradiance_format, DataLayer.IRRADIANCE)
            ium_result_data["irradiance_path"] = _as_relative_to(irr_path, assets_dir_str)
            if timer is not None:
                timer.record("step3/irradiance", time.perf_counter() - _t3_irr)

        # ── GT skybox vs NeRF-baked comparison ───────────────────────────────
        # Requires: compare_skybox_to_gt=True + a non-empty skybox_path (GT HDR) +
        # skybox_nerf_baked.exr already written by _bake_skybox_from_nerf.
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
                print(f"[Step 3] Skybox comparison saved: {sky_cmp_dir / 'skybox_heatmap.png'}")
            else:
                print("[Step 3] skybox_compare: skybox_nerf_baked.exr not found "
                      "(expected after skybox_source='nerf') — skipped")

        # ── Indirect Irradiance via NeRF ─────────────────────────────────────
        irr_indirect_flat = None
        if rc.precompute_indirect and ium_res.has_positions() and ium_res.has_normals():
            _t3_ind = time.perf_counter()
            ind_out_dir = assets_dir / "irradiance"
            os.makedirs(ind_out_dir, exist_ok=True)
            ind_path = (ind_out_dir / f"irradiance_indirect{rc.indirect_format.extension}").resolve().as_posix()
            if not os.path.exists(ind_path):
                print(f"[Step 3] Precompute Indirect Irradiance "
                      f"(N={rc.indirect_sample_side}, tile={rc.indirect_tile_size})…")
                _precompute_indirect_irradiance(rc, ium_res, model, ium_w, ium_h, ind_path)
            else:
                print(f"[Step 3] Indirect irradiance found on disk: {ind_path}")
            ind_arr = _load_exr_as_flat(ind_path)
            if ind_arr is not None:
                irr_indirect_flat = ind_arr
                ium_result_data["irradiance_indirect_path"] = _as_relative_to(ind_path, assets_dir_str)
            if timer is not None:
                timer.record("step3/indirect", time.perf_counter() - _t3_ind)

        # ── Specular cone L_j(r) via envmap + NeRF ───────────────────────────
        if (rc.precompute_spec_cone and ium_res.has_positions()
                and ium_res.has_normals() and visibility_map is not None):
            _t3_spec = time.perf_counter()
            sky_w, sky_h = rc.skybox_size[0], rc.skybox_size[1]
            if skybox_flat_step3 is None and (rc.skybox_path or rc.skybox_source == "nerf"):
                skybox_flat_step3 = _resolve_skybox_flat(rc, output_json, json_dir, sky_w, sky_h)
            spec_dir = assets_dir / "spec_cone"
            if rc.spec_cone_scheme == "shared":
                print(f"[Step 3] Precomputing the specular cones L_j(r), shared rays "
                      f"(apertures={rc.spec_cone_apertures_deg}°, "
                      f"S={rc.spec_cone_shared_samples})…")
                _precompute_spec_cone_shared(
                    rc, ium_res, model, ium_w, ium_h, tf.frames,
                    visibility_map, len(all_cameras),
                    skybox_flat_step3, [sky_w, sky_h], spec_dir)
            elif rc.spec_cone_scheme == "per_camera":
                print(f"[Step 3] Precomputing the specular cones L_j(r) "
                      f"(apertures={rc.spec_cone_apertures_deg}°, "
                      f"alloc={rc.spec_cone_sample_alloc}, "
                      f"budget={rc.spec_cone_samples_budget})…")
                _precompute_spec_cone(rc, ium_res, model, ium_w, ium_h, tf.frames,
                                      visibility_map, len(all_cameras),
                                      skybox_flat_step3, [sky_w, sky_h], spec_dir)
            else:
                raise ValueError(
                    f"unknown spec_cone_scheme: {rc.spec_cone_scheme!r} "
                    "(expected 'per_camera' or 'shared')")
            ium_result_data["spec_cone_dir"] = _as_relative_to(
                spec_dir.resolve().as_posix(), assets_dir_str)
            if timer is not None:
                timer.record("step3/spec_cone", time.perf_counter() - _t3_spec)

    # Update the JSON and rewrite it. With a ROI active the root JSON stays intact and
    # the enriched one goes into the sandbox, with the file_paths rewritten as absolute:
    # in the root they are relative to output_dir, and load_transforms would resolve
    # them against the sandbox. solve_pbr only uses their stems, but this way
    # `python pbr_solver.py <sandbox>` keeps working too.
    if ium_result_data:
        output_json["ium"] = ium_result_data
    out_json_path = transforms_extended_path
    if roi_tag is not None:
        out_json_path = assets_dir / "transforms_extended.json"
        output_json["roi"] = roi_fp
        for entry, frame in zip(output_json.get("frames", []), tf.frames):
            entry["file_path"] = frame.file_path
    with open(out_json_path, "w", encoding="utf-8") as fh:
        json.dump(output_json, fh, indent=4)
    print(f"\n[Step 3] JSON updated: {out_json_path}")
    return output_json


def _step4_reconstruction(
    cfg: PipelineConfig,
    transforms_extended_path: Path,
    tb_logger=None,
    timer=None,
) -> dict:
    """Step 4 — reconstruction (PBR fit + albedo) from the Step 3 on-disk cache alone.

    It uses neither OptiX nor the NeRF checkpoint: it reads spec_cone/,
    sources/{src}/color_texture/, irradiance/ and ium/ already produced by Step 3, so
    it can be re-run on its own (run_step1/2/3=False, run_step4=True) to iterate on the
    reconstruction without re-baking the cones. It updates transforms_extended.json in
    place with the metallic/roughness/albedo_pbr/albedo keys.

    With a ROI active it reads and writes inside the roi/<tag>/ sandbox, where Step 3
    left ium/, spec_cone/ and sources/ already restricted: no ROI logic is needed here,
    the fit and the albedo restrict themselves by re-reading ium_masks.
    """
    rc = cfg.render
    json_dir = Path(rc.output_dir).resolve()
    assets_dir, roi_tag = _roi_assets_dir(rc, json_dir)
    assets_dir_str = assets_dir.as_posix()

    if roi_tag is not None:
        roi_json = assets_dir / "transforms_extended.json"
        if not roi_json.exists():
            raise FileNotFoundError(
                f"Step 4 with ROI '{roi_tag}': the sandbox {assets_dir} does not hold "
                f"the Step 3 outputs. Run run_step3=True with the same ROI "
                f"first.")
        transforms_extended_path = roi_json

    with open(transforms_extended_path, encoding="utf-8") as fh:
        output_json = json.load(fh)
    ium_result_data: dict = output_json.get("ium", {})

    # ── PBR maps (metallic / roughness from the spec-cone fit) ───────────────
    # Run for EVERY source in color_texture_image_sources, under sources/{src}/.
    # solve_pbr reads everything from disk (spec_cone/, camera_texture/, pixel_change/).
    if rc.render_pbr_maps:
        _t4_pbr = time.perf_counter()
        spec_meta = assets_dir / "spec_cone" / "spec_cone_meta.json"   # source-independent
        if spec_meta.exists():
            from pbr_solver import solve_pbr
            for src in rc.color_texture_image_sources:
                print(f"[Step 4] PBR fit ({src}) → metallic/roughness "
                      f"(spec_threshold={rc.pbr_spec_threshold})…")
                pbr_out = solve_pbr(assets_dir_str, source=src,
                                    spec_threshold=rc.pbr_spec_threshold,
                                    min_views=rc.pbr_min_views,
                                    albedo_eps=rc.albedo_eps,
                                    blender_rgb=rc.pbr_write_blender_rgb,
                                    tile_texels=rc.pbr_tile_texels)
                ium_result_data[f"metallic_path_{src}"] = _as_relative_to(
                    pbr_out["metallic_path"], assets_dir_str)
                ium_result_data[f"roughness_path_{src}"] = _as_relative_to(
                    pbr_out["roughness_path"], assets_dir_str)
                if pbr_out.get("albedo_pbr_path"):
                    ium_result_data[f"albedo_pbr_path_{src}"] = _as_relative_to(
                        pbr_out["albedo_pbr_path"], assets_dir_str)
                # The returned dict also carries the full-resolution maps (~900 MiB):
                # only the paths are needed here, and without the del they would stay
                # alive throughout the next source.
                del pbr_out
            if timer is not None:
                timer.record("step4/pbr", time.perf_counter() - _t4_pbr)
        else:
            print("    ⚠  render_pbr_maps: spec_cone_meta.json missing → skipped "
                  "(precompute_spec_cone is needed in Step 3)")

    # ── Albedo = π · color / max(irradiance + indirect, eps) ─────────────────
    # Read entirely from disk: colour texture per source, shared irradiance.
    if rc.render_albedo:
        _t4_alb = time.perf_counter()
        irr_path = assets_dir / "irradiance" / f"irradiance{rc.irradiance_format.extension}"
        if not irr_path.exists():
            print(f"    ⚠  render_albedo: {irr_path} missing → skipped "
                  "(render_irradiance is needed in Step 3)")
        else:
            print(f"[Step 4] Computing albedo = π · color / max(irradiance, {rc.albedo_eps})…")

            # Shared denominator: direct irradiance (+ indirect when present)
            irr = _load_image_hw3_native(irr_path.as_posix())
            ind_path = assets_dir / "irradiance" / f"irradiance_indirect{rc.indirect_format.extension}"
            if ind_path.exists():
                irr = irr + _load_image_hw3_native(ind_path.as_posix())
            denom = np.maximum(irr, rc.albedo_eps)

            # IUM mask (channel 0 > 0.5 = valid texel); saved with rc.ium_format
            mask_path = assets_dir / "ium" / f"ium_masks{rc.ium_format.extension}"
            mask = None
            if mask_path.exists():
                mask = _load_image_hw3_native(mask_path.as_posix())[..., 0] > 0.5

            for src in rc.color_texture_image_sources:
                color_path = (assets_dir / "sources" / src / "color_texture"
                              / f"color_texture{rc.color_texture_format.extension}")
                if not color_path.exists():
                    print(f"    ⚠  albedo ({src}): {color_path} missing → skipped "
                          "(render_color_texture is needed in Step 3)")
                    continue
                color = _load_image_hw3_native(color_path.as_posix())
                albedo = (np.float32(np.pi) * color) / denom
                if mask is not None:
                    albedo[~mask] = 0.0
                albedo = np.clip(albedo, 0.0, 1.0).astype(np.float32)

                alb_dir = assets_dir / "sources" / src / "albedo"
                os.makedirs(alb_dir, exist_ok=True)
                alb_path = (alb_dir / f"albedo{rc.albedo_format.extension}").resolve().as_posix()
                _save_layer(albedo, alb_path, rc.albedo_format, DataLayer.ALBEDO)
                ium_result_data[f"albedo_path_{src}"] = _as_relative_to(alb_path, assets_dir_str)

                # — TensorBoard: the albedo is already in [0,1] → no tonemap —
                if tb_logger is not None:
                    tb_logger.log_image(f"texture/albedo_{src}", albedo, step=0, tonemap=False)
                    tb_logger.flush()

            if timer is not None:
                timer.record("step4/albedo", time.perf_counter() - _t4_alb)

    # Update the JSON in place and rewrite it
    if ium_result_data:
        output_json["ium"] = ium_result_data
    with open(transforms_extended_path, "w", encoding="utf-8") as fh:
        json.dump(output_json, fh, indent=4)
    print(f"\n[Step 4] JSON updated: {transforms_extended_path}")
    return output_json


def run_pipeline(
    cfg: PipelineConfig,
    *,
    tb_run_dir: str | None = None,
    tb_enabled: bool = True,
) -> dict:
    """Four-step orchestrator. Each step can be enabled or disabled independently.

    Step 1 (run_step1): per-frame depth+mask, image copies, minimal transforms_extended.json.
    Step 2 (run_step2): NeRF training via nerf/train.py, saves the checkpoint.
    Step 3 (run_step3): IUM/visibility/color_texture/irradiance/indirect/spec_cone (bake).
    Step 4 (run_step4): reconstruction (PBR fit + albedo) from the Step 3 on-disk cache
                        alone — independent of OptiX/NeRF, re-runnable to iterate on the
                        reconstruction without re-baking the cones.

    ``tb_run_dir`` is the folder the TensorBoard event files are written to.
    When None, <output_dir>/tensorboard is used.
    ``tb_enabled=False`` disables TB logging entirely (a no-op).
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from monitoring import RunLogger, StageTimer, log_timing_breakdown

    import OptixProgrammablePasses as optix
    optix.LogManager.set_min_level(optix.LogLevel.Error)
    optix.OptixManager.instance().set_log_level(optix.LogLevel.Disabled)

    _tb_dir = tb_run_dir or str(Path(cfg.render.output_dir) / "tensorboard")
    logger = RunLogger(_tb_dir, enabled=tb_enabled)
    timer  = StageTimer()

    # Log the config at the start of the run
    logger.log_text("run/config", json.dumps(asdict(cfg), indent=2, default=str), step=0)

    transforms_extended = Path(cfg.render.output_dir) / "transforms_extended.json"
    final_psnr = float("nan")

    try:
        if cfg.run_step1:
            with timer("step1"):
                transforms_extended = _step1_pretrain_data(cfg, optix)
        elif not transforms_extended.exists():
            raise FileNotFoundError(
                f"Step 1 is disabled but {transforms_extended} does not exist.\n"
                "Set run_step1=True, or run Step 1 manually."
            )
        else:
            # Minimal validation of the existing JSON
            with open(transforms_extended, encoding="utf-8") as fh:
                _probe = json.load(fh)
            frames = _probe.get("frames", [])
            if frames and ("depth_path" not in frames[0] or "mask_path" not in frames[0]):
                raise ValueError(
                    f"{transforms_extended} has no depth_path/mask_path per frame.\n"
                    "Set run_step1=True to regenerate it."
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
                    print(f"  EXR/PNG saved to: {render_dir}")

                if not cfg.nerf_interactive_loop:
                    break
                try:
                    ans = input("\nContinue training? [y/N]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    break
                if ans != "y":
                    break

        elif (cfg.run_step3
              and (cfg.render.precompute_indirect or cfg.render.precompute_spec_cone)
              and not ckpt_path.exists()):
            raise FileNotFoundError(
                f"Step 2 is disabled but the NeRF checkpoint does not exist: {ckpt_path}\n"
                "Set run_step2=True, or provide a valid nerf_ckpt_path."
            )

        # Free the NeRF training VRAM before OptiX allocates the Step 3 buffers
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

        # Step 4 — reconstruction (PBR fit + albedo) from the Step 3 on-disk cache.
        # Independent of run_step3: with run_step1/2/3=False the reconstruction can be
        # iterated on without re-baking the cones (no OptiX, no checkpoint needed).
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
# Per-run manifest and multi-scene runner
# ──────────────────────────────────────────────────────────────────────────────

def _write_run_manifest(cfg: PipelineConfig, scene: SceneConfig, run_note: str) -> None:
    """Write run_manifest.json in output_dir: full config + timestamp + note."""
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
        "config": asdict(cfg),   # includes output_dir and every path already resolved for the scene
    }
    out = Path(cfg.render.output_dir) / "run_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False, default=_enc)
    print(f"  manifest saved → {out}")


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
    """Run run_pipeline on each scene in sequence, in separate subfolders.

    For every SceneConfig it:
    - clones ``template`` (a deep copy, safe with the mutable default_factory fields)
    - overrides the per-scene paths and ``output_dir = <output_root>/<scene.name>``
    - writes ``run_manifest.json`` into the subfolder
    - calls ``run_pipeline`` with an isolated TensorBoard log_dir

    ``experiment_tag`` groups related runs in TensorBoard under a single prefix.
    When empty it is derived from the base name of ``output_root``.

    ``tb_log_root`` is the root of the TensorBoard logs (it must match the volume
    mounted in docker/tensorboard/docker-compose.yml). When None it is read from the
    ``TB_LOG_ROOT`` environment variable; if that is missing too, the default is
    ``D:/tesi_output/tb_logs``.

    Every invocation of this function creates:
      ``<tb_log_root>/<experiment_tag>/<scene.name>/<YYYYMMDD-HHMMSS>/``
    so that re-runs stay isolated (no merged, zig-zagging curves in TensorBoard).

    If a run fails, the error is logged and the next one proceeds. A summary with the
    status of every scene is printed at the end.

    ``nerf_ckpt_path`` / ``nerf_train_output_dir`` stay ``""`` in the template and are
    derived automatically from the per-scene ``output_dir`` (one checkpoint per scene).

    Returns:
        dict scene → the run_pipeline result (or None when the run failed)
    """
    # ── Resolve tb_log_root ────────────────────────────────────────────────────
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

        # Resume: skip the NeRF training when the checkpoint is already there
        if cfg.resume_skip_step2_if_ckpt and cfg.run_step2:
            _ckpt_resume = Path(cfg.render.output_dir) / "model" / "nerf_model_cache.pt"
            if _ckpt_resume.exists():
                print(f"  ↻ NeRF checkpoint already present; skipping Step 2 (run_step2=False): {_ckpt_resume}")
                cfg.run_step2 = False

        # TensorBoard log dir for this scene and this run (isolated by run id)
        tb_run_dir = os.path.join(_tb_root, _tag, scene.name, _run_id)

        _log_path = os.path.join(cfg.render.output_dir, "console.log")
        with _console_to_file(_log_path):
            print(f"\n{'='*70}")
            print(f"  Scene       : {scene.name}")
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
                print(f"\n  ✗ [{scene.name}] error: {exc}")
                import traceback
                traceback.print_exc()
                results[scene.name] = None
                statuses[scene.name] = f"error: {exc}"

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  run_pipeline_multi summary:")
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

    # ── Shared template ───────────────────────────────────────────────────────
    # The scene-specific paths (transforms_path, model_path, external_normal_path,
    # skybox_path) are empty here and get overridden for every SceneConfig.
    # Every other rendering/NeRF parameter is shared across the scenes.
    template = PipelineConfig(
        run_step1 = True,   # full run from scratch: geometry passes + NeRF dataset
        run_step2 = True,   # full run from scratch: NeRF training + Step 2b renders
        run_step3 = True,   # texture-space passes up to the spec-cone bake
        run_step4 = True,   # PBR+albedo reconstruction (set run_step3=False to iterate here alone)

        resume_skip_step2_if_ckpt = True,   # skip the NeRF training when the checkpoint already exists

        render = RenderConfig(
            # per-scene paths (overridden by SceneConfig — do not edit them here)
            transforms_path      = "",
            model_path           = "",
            external_normal_path = None,
            output_dir           = "",  # set to <output_root>/<scene.name>

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

            color_texture_image_sources = ["gt"], # each source processed in full under sources/

            precompute_spec_cone = True,
            render_pbr_maps      = True,
            pbr_spec_threshold   = 0.0,        # the fitted roughness is written everywhere
            spec_cone_cameras=None,

            # Rays shared between the cameras: the incident radiance along a direction
            # does not depend on the camera, so a single Fibonacci set per texel (traced
            # and queried on the NeRF once) serves all m cameras that see the texel. At
            # S=16384 the cost breaks even with the per-camera bake when m≈11: the gain
            # is reinvested in resolution, not in time. The aperture grid is refined in
            # the narrow region because with shared rays one more candidate costs no
            # extra rays. The 5° candidate receives ~16 samples, i.e. it sits at the
            # noise limit: if lobe_param/residual show it to be unstable, drop it from
            # the grid.
            spec_cone_scheme         = "shared",
            spec_cone_shared_samples = 9216, # 96 x 96 Fibonacci samples per texel (shared)

            spec_cone_apertures_deg  = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0,
                                        60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0],
            # must be a multiple of the IUM width: every tile is a block of whole
            # scanlines, streamed into the per-camera EXRs
            spec_cone_tile_size      = 4096,
            # dirs + radiances cost ~0.6 MB per texel at S=16384: 1024 texels are ~0.6
            # GB of peak. Raise it as far as the VRAM allows — large sub-blocks reduce
            # the overhead of the loop over cameras.
            spec_cone_chunk_texels   = 1024,
            # the network batch is the bottleneck for GPU occupancy
            spec_cone_nerf_chunk     = 4096*24,

            # Used only by spec_cone_scheme="per_camera"
            spec_cone_sample_alloc   = "solid_angle",
            spec_cone_samples_budget = 1440,

            color_texture_grazing_max_deg = 75.0,

            # Diagnostic GT vs NeRF-baked skybox heatmap (needs skybox_path in the SceneConfig)
            compare_skybox_to_gt = True,

            # roi_rect=[3623,2712, 473,473],
            # roi_tag="top_table_test_irradiance_newcuda"

        ),

        nerf_num_iters       = 50000,
        nerf_lr_decay_steps  = 100000,  # fixed anchor = planned length; makes resumes correct
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
        # The bg sphere radius is now bg_radius_mult × (max distance from the origin),
        # no longer × the max bbox side: on this scene the base goes from 3.2 to 2.266,
        # so 3.0 would give R=6.8 against the ~12 of before. 5.0 → R=11.3, which keeps
        # the shell outside the camera rig (the farthest sit at 9.0) and preserves the
        # angular resolution of the envmap: how much space the shell takes relative to
        # the positional-encoding frequencies decides how sharp skybox_nerf_baked.exr is.
        nerf_bg_radius_mult            = 5.0,
        nerf_bg_depth_window           = 0.05,
        nerf_bg_depth_window_end       = 0.05,
        # nerf_multires = 12,
        # nerf_multires_views = 6,



        nerf_profile_iters = 0,


    )

    # ── Scenes ────────────────────────────────────────────────────────────────
    # Add or comment out SceneConfig entries to choose which scenes to process.
    # The output of each scene lands in <output_root>/<scene.name>/.
    SCENES = [
        # SceneConfig(
        #     name             = "TableAndOtherInterior",
        #     transforms_path  = f"{REPO}/Scenes/TableAndOtherInterior/NerfOpenEXR/transforms.json",
        #     model_path       = f"{REPO}/Scenes/TableAndOtherInterior/Models/Baked.obj",
        #     external_normal_path = f"{REPO}/Scenes/TableAndOtherInterior/BlenderBaked/BakedMaterial_normal.exr",
        #     # GT HDR used only as a reference for compare_skybox_to_gt (not for rendering)
        #     skybox_path      = f"{REPO}/Scenes/TableAndOtherInterior/Blender/assets/hdri/wooden_studio_13_4k.exr",
        # ),
        SceneConfig(
            name             = "TableAndOtherInteriorWithSpecular",
            transforms_path  = f"{REPO}/Scenes/TableAndOtherInterior/NerfOpenEXRSmooth/transforms.json",
            model_path       = f"{REPO}/Scenes/TableAndOtherInterior/ModelsSmooth/Baked.obj",
            external_normal_path = f"{REPO}/Scenes/TableAndOtherInterior/BlenderBakedSmooth/BakedMaterial_normal.exr",
            # GT HDR used only as a reference for compare_skybox_to_gt (not for rendering)
            # skybox_path      = f"{REPO}/Scenes/TableAndOtherInterior/Blender/assets/hdri/wooden_studio_13_4k.exr",
        ),
        #  SceneConfig(
        #     name             = "TableAndOtherInteriorWithSpecularHighDetails",
        #     transforms_path  = f"{REPO}/Scenes/TableAndOtherInterior/NerfOpenEXRHighDetails/transforms.json",
        #     model_path       = f"{REPO}/Scenes/TableAndOtherInterior/ModelsSmooth/Baked.obj",
        #     external_normal_path = f"{REPO}/Scenes/TableAndOtherInterior/BlenderBakedSmooth/BakedMaterial_normal.exr",
        #     # GT HDR used only as a reference for compare_skybox_to_gt (not for rendering)
        #     # skybox_path      = f"{REPO}/Scenes/TableAndOtherInterior/Blender/assets/hdri/wooden_studio_13_4k.exr",
        # ),
        # SceneConfig(
        #             name             = "TableAndOtherInteriorWithSpecularNight",
        #             transforms_path  = f"{REPO}/Scenes/TableAndOtherInterior/NerfOpenEXRSmoothNight/transforms.json",
        #             model_path       = f"{REPO}/Scenes/TableAndOtherInterior/ModelsSmooth/Baked.obj",
        #             external_normal_path = f"{REPO}/Scenes/TableAndOtherInterior/BlenderBakedSmoothNight/BakedMaterial_normal.exr",
        #             # GT HDR used only as a reference for compare_skybox_to_gt (not for rendering)
        #             # skybox_path      = f"{REPO}/Scenes/TableAndOtherInterior/Blender/assets/hdri/wooden_studio_13_4k.exr",
        #         ),
        # SceneConfig(
        #     name             = "TableAndOtherInteriorNoSpecular",
        #     transforms_path  = f"{REPO}/Scenes/TableAndOtherInterior/NerfOpenExrSmoothNoDiffuse/transforms.json",
        #     model_path       = f"{REPO}/Scenes/TableAndOtherInterior/ModelsSmooth/Baked.obj",
        #     external_normal_path = f"{REPO}/Scenes/TableAndOtherInterior/BlenderBakedSmoothNoDiffuse/BakedMaterial_normal.exr",
        #     # GT HDR used only as a reference for compare_skybox_to_gt (not for rendering)
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
        # SceneConfig(
        #     name             = "SwordShieldNight",
        #     transforms_path  = f"{REPO}/Scenes/SwordShield Thesis/NerfNight/transforms.json",
        #     model_path       = f"{REPO}/Scenes/SwordShield Thesis/BakedNight/Baked.obj",
        #     external_normal_path = f"{REPO}/Scenes/SwordShield Thesis/BakedNight/BakedMaterial_normal.exr",
        # )

    ]

    # ── Execution ─────────────────────────────────────────────────────────────
    # Factorial 2×2×2 sweep: activation × loss × decay — 8 runs in total.
    # Each run gets its own output_root (isolated checkpoints: exp and softplus are

    # not compatible with each other and must not resume across configurations).
    # TB_LOG_ROOT is read from docker/tensorboard/.env — no need to change it here.
    # For a clean sweep from scratch, change SWEEP_ROOT or empty the folder.

    # (tag_base, rgb_activation, loss_type) — 2×2 factorial, activation × loss
    EXPERIMENTS = [
        # ("exp_relmseraw",      "exp",      "rel_mse_raw"),
        # ("softplus_relmseraw", "softplus", "rel_mse_raw"),
        ("exp_l1",             "exp",      "l1"),
        # ("softplus_l1",        "softplus", "l1"),
        # ("softplus_mse",       "softplus", "mse"),
        # ("exp_mse",            "exp",      "mse")
    ]
    DECAYS     = (0.2,)
    SWEEP_ROOT = "D:/tesi_output/handoff_check"

    for name, act, loss in EXPERIMENTS:
        for decay in DECAYS:


            
            cfg = copy.deepcopy(template)
            cfg.nerf_num_iters       = 75000
            cfg.nerf_lr_decay_steps  = 100000  # fixed anchor, aligned with num_iters
            cfg.nerf_rgb_activation  = act
            cfg.nerf_loss_type       = loss
            cfg.nerf_lr_decay        = decay
            cfg.render.skybox_source = "nerf"  # the field lives on RenderConfig, not PipelineConfig
            tag = f"{name}_d{str(decay).replace('.', '')}"  # e.g. exp_l1_d01
            run_pipeline_multi(
                cfg, SCENES,
                output_root    = f"{SWEEP_ROOT}/{tag}",
                run_note       = f"{cfg.nerf_num_iters}iter | act={act} | loss={loss} | decay={decay}",
                tb_enabled=False
            )
    