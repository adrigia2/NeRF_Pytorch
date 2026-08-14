"""
exr_to_blender_rgb.py
---------------------
Rewrite the single-channel PBR maps (metallic/roughness) in the convention
Blender's bakes use: three float32 R/G/B channels holding the same replicated
value, plus the header attribute ``colorInteropID = "data"`` that marks the file
as non-colour data for colour management.

This aligns the *container* only: the values are untouched and are not directly
comparable with a Blender bake (this pipeline's roughness is aperture/180, a
cone-width index, not a GGX roughness). The original single-channel `Z` file
stays where it is: the RGB variant is written next to it with the `_rgb` suffix,
so the internal readers (`pbr_solver._read_exr`, `scripts/inspect_final_maps.py`)
keep working.

The write does not go through `ExrWriter`, because the legacy OpenEXR API
silently drops non-standard header attributes (`XXX - unknown attribute`); the
modern `OpenEXR.File` API writes them correctly.

Usage:
    python exr_to_blender_rgb.py <path|dir> [<path|dir> ...]
                                 [--suffix _rgb] [--maps roughness,metallic]
                                 [--force] [--dry-run]

A directory argument is walked recursively looking for files whose name is
exactly `<map>.exr`: one rule covers the three layouts present on disk
(`sources/{source}/roughness/roughness.exr`, the older layout without `sources/`,
and the flat copies next to the Blender bakes) and excludes the already-converted
files and the diagnostics by itself.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from images_generator import _load_image_hw3_native  # noqa: E402

DEFAULT_SUFFIX = "_rgb"
DEFAULT_MAPS = ("roughness", "metallic")

# Attributes replicated from Blender's bakes. `colorInteropID = "data"` tells
# colour management that the file is not a colour and must not be transformed.
BLENDER_ATTRS = {"colorInteropID": "data", "Software": "Tesi pipeline"}

_RGB = ("R", "G", "B")


# ──────────────────────────────────────────────────────────────────────────────
# IO
# ──────────────────────────────────────────────────────────────────────────────

def _exr_channels(path: Path) -> list[str]:
    import OpenEXR
    exr = OpenEXR.InputFile(path.as_posix())
    try:
        return list(exr.header()["channels"].keys())
    finally:
        exr.close()


def _write_rgb_exr(gray: np.ndarray, path: Path,
                   attributes: dict[str, str] | None = None) -> None:
    """Write an (H, W) float32 array to three identical R/G/B channels, ZIP."""
    import OpenEXR

    gray = np.ascontiguousarray(gray, dtype=np.float32)
    header = {
        "compression": OpenEXR.ZIP_COMPRESSION,
        "type": OpenEXR.scanlineimage,
    }
    header.update(attributes or {})
    with OpenEXR.File(header, {c: gray for c in _RGB}) as f:
        f.write(path.as_posix())


# ──────────────────────────────────────────────────────────────────────────────
# Conversion
# ──────────────────────────────────────────────────────────────────────────────

def write_blender_rgb(src: Path | str, dst: Path | str | None = None, *,
                      suffix: str = DEFAULT_SUFFIX, force: bool = False,
                      quiet: bool = False) -> Path | None:
    """Single-channel EXR → replicated R/G/B EXR, next to the original.

    Returns the path written, or None when the conversion was skipped (source
    already RGB, multi-channel source, destination already present).
    """
    src = Path(src)
    dst = Path(dst) if dst is not None else src.with_name(src.stem + suffix + src.suffix)

    def _skip(msg: str) -> None:
        if not quiet:
            print(f"    ⏭  {src.name}: {msg}")

    if not src.exists():
        _skip("not found")
        return None

    chans = _exr_channels(src)
    if set(_RGB).issubset(chans):
        _skip("already in RGB format")
        return None
    if len(chans) != 1:
        _skip(f"{len(chans)} channels ({', '.join(chans)}) → not a scalar map")
        return None
    if dst.exists() and not force:
        _skip(f"{dst.name} already exists (--force to overwrite)")
        return None

    # _load_image_hw3_native replicates the single channel onto r=g=b: taking one
    # of them is enough here, the real replication happens on write.
    gray = _load_image_hw3_native(src.as_posix())[..., 0]
    dst.parent.mkdir(parents=True, exist_ok=True)
    _write_rgb_exr(gray, dst, BLENDER_ATTRS)

    if not quiet:
        h, w = gray.shape
        mb = dst.stat().st_size / (1024 * 1024)
        print(f"    ✓ {dst.name}  ({w}×{h}, channel {chans[0]} → R/G/B, {mb:.1f} MB)")
    return dst


def find_maps(root: Path | str, maps=DEFAULT_MAPS) -> list[Path]:
    """Files to convert under `root` (or [root] when it is already an EXR file)."""
    root = Path(root)
    if root.is_file():
        return [root]
    wanted = set(maps)
    return sorted(p for p in root.rglob("*.exr") if p.stem in wanted)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Single-channel EXR → R/G/B EXR in the Blender convention")
    ap.add_argument("paths", nargs="+",
                    help=".exr files, or directories to walk recursively")
    ap.add_argument("--suffix", default=DEFAULT_SUFFIX,
                    help=f"suffix of the written file (default: {DEFAULT_SUFFIX})")
    ap.add_argument("--maps", default=",".join(DEFAULT_MAPS),
                    help="names of the maps to look for in directories, "
                         f"comma-separated (default: {','.join(DEFAULT_MAPS)})")
    ap.add_argument("--force", action="store_true",
                    help="overwrite destinations that already exist")
    ap.add_argument("--dry-run", action="store_true",
                    help="list the files that would be converted, then stop")
    args = ap.parse_args(argv)

    maps = tuple(m.strip() for m in args.maps.split(",") if m.strip())
    srcs: list[Path] = []
    for p in args.paths:
        found = find_maps(p, maps)
        if not found:
            print(f"⚠  no {maps} map found in {p}")
        srcs += found
    srcs = sorted(dict.fromkeys(srcs))

    if args.dry_run:
        print(f"[dry-run] {len(srcs)} files to convert:")
        for s in srcs:
            print(f"    {s}  →  {s.stem + args.suffix + s.suffix}")
        return 0

    written = 0
    for s in srcs:
        print(f"  {s}")
        if write_blender_rgb(s, suffix=args.suffix, force=args.force) is not None:
            written += 1
    print(f"\n✓ {written}/{len(srcs)} files converted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
