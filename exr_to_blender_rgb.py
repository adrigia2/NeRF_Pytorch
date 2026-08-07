"""
exr_to_blender_rgb.py
---------------------
Riscrive le mappe PBR a canale singolo (metallic/roughness) nella convenzione
usata dai bake di Blender: tre canali R/G/B float32 con lo stesso valore
replicato, più l'attributo di header ``colorInteropID = "data"`` che marca il
file come dato non-colore per il color management.

Serve solo ad allineare il *contenitore*: i valori non vengono toccati e non
sono direttamente confrontabili con un bake di Blender (la roughness della
pipeline è apertura/180, un indice di larghezza del cono, non una roughness
GGX). Il file originale a canale singolo `Z` resta al suo posto: la variante
RGB viene scritta accanto con il suffisso `_rgb`, così i lettori interni
(`pbr_solver._read_exr`, `inspect_final_maps.py`) continuano a funzionare.

La scrittura non passa da `ExrWriter` perché l'API legacy di OpenEXR scarta in
silenzio gli attributi di header non standard (`XXX - unknown attribute`);
l'API moderna `OpenEXR.File` li scrive correttamente.

Uso:
    python exr_to_blender_rgb.py <path|dir> [<path|dir> ...]
                                 [--suffix _rgb] [--maps roughness,metallic]
                                 [--force] [--dry-run]

Se l'argomento è una directory viene percorsa ricorsivamente cercando i file il
cui nome è esattamente `<mappa>.exr`: copre in una sola regola i tre layout
presenti su disco (`sources/{source}/roughness/roughness.exr`, il layout vecchio
senza `sources/`, e le copie piatte accanto ai bake di Blender) ed esclude da
sé i file già convertiti e la diagnostica.
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

# Attributi replicati dai bake di Blender. `colorInteropID = "data"` dice al
# color management che il file non è un colore e non va trasformato.
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
    """Scrive un (H, W) float32 su tre canali R/G/B identici, ZIP."""
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
# Conversione
# ──────────────────────────────────────────────────────────────────────────────

def write_blender_rgb(src: Path | str, dst: Path | str | None = None, *,
                      suffix: str = DEFAULT_SUFFIX, force: bool = False,
                      quiet: bool = False) -> Path | None:
    """EXR a canale singolo → EXR R/G/B replicato, accanto all'originale.

    Ritorna il path scritto, oppure None se la conversione è stata saltata
    (sorgente già RGB, sorgente multi-canale, destinazione già esistente).
    """
    src = Path(src)
    dst = Path(dst) if dst is not None else src.with_name(src.stem + suffix + src.suffix)

    def _skip(msg: str) -> None:
        if not quiet:
            print(f"    ⏭  {src.name}: {msg}")

    if not src.exists():
        _skip("non trovato")
        return None

    chans = _exr_channels(src)
    if set(_RGB).issubset(chans):
        _skip("già in formato RGB")
        return None
    if len(chans) != 1:
        _skip(f"{len(chans)} canali ({', '.join(chans)}) → non è una mappa scalare")
        return None
    if dst.exists() and not force:
        _skip(f"{dst.name} esiste già (--force per sovrascrivere)")
        return None

    # _load_image_hw3_native replica il canale singolo su r=g=b: qui basta
    # riprenderne uno, la replica vera avviene in scrittura.
    gray = _load_image_hw3_native(src.as_posix())[..., 0]
    dst.parent.mkdir(parents=True, exist_ok=True)
    _write_rgb_exr(gray, dst, BLENDER_ATTRS)

    if not quiet:
        h, w = gray.shape
        mb = dst.stat().st_size / (1024 * 1024)
        print(f"    ✓ {dst.name}  ({w}×{h}, canale {chans[0]} → R/G/B, {mb:.1f} MB)")
    return dst


def find_maps(root: Path | str, maps=DEFAULT_MAPS) -> list[Path]:
    """File da convertire sotto `root` (o [root] se è già un file EXR)."""
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
        description="EXR a canale singolo → EXR R/G/B nella convenzione Blender")
    ap.add_argument("paths", nargs="+",
                    help="file .exr oppure directory da percorrere ricorsivamente")
    ap.add_argument("--suffix", default=DEFAULT_SUFFIX,
                    help=f"suffisso del file scritto (default: {DEFAULT_SUFFIX})")
    ap.add_argument("--maps", default=",".join(DEFAULT_MAPS),
                    help="nomi delle mappe da cercare nelle directory, "
                         f"separati da virgola (default: {','.join(DEFAULT_MAPS)})")
    ap.add_argument("--force", action="store_true",
                    help="sovrascrive le destinazioni già esistenti")
    ap.add_argument("--dry-run", action="store_true",
                    help="elenca i file che verrebbero convertiti e si ferma")
    args = ap.parse_args(argv)

    maps = tuple(m.strip() for m in args.maps.split(",") if m.strip())
    srcs: list[Path] = []
    for p in args.paths:
        found = find_maps(p, maps)
        if not found:
            print(f"⚠  nessuna mappa {maps} trovata in {p}")
        srcs += found
    srcs = sorted(dict.fromkeys(srcs))

    if args.dry_run:
        print(f"[dry-run] {len(srcs)} file da convertire:")
        for s in srcs:
            print(f"    {s}  →  {s.stem + args.suffix + s.suffix}")
        return 0

    written = 0
    for s in srcs:
        print(f"  {s}")
        if write_blender_rgb(s, suffix=args.suffix, force=args.force) is not None:
            written += 1
    print(f"\n✓ {written}/{len(srcs)} file convertiti")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
