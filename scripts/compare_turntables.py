#!/usr/bin/env python
"""compare_turntables.py -- one mp4 that puts the turntable videos side by side.

    python compare_turntables.py [--root D:/rerender_video] [--out .../comparison/comparison.mp4]
                                 [--blocks 1,3] [--frames N] [--fps 30] [--short]

Reads the per-scene turntables written by `turntable_video.py` under `D:/rerender_video`
and composes them into a sequence of blocks, one block per comparison, each lasting one
full revolution.  The videos of one block were rendered with the same orbit (pivot,
radius, elevation, start azimuth, direction all come from the camera rig of the run), so
they line up frame by frame and the tiles can simply be pasted next to each other.

-------------------------------------------------------------------------------
The blocks
-------------------------------------------------------------------------------
Two families, TableAndOtherInterior (specular variant) and SwordShield, each in the same
three-block pattern:

  1. studio capture, 2x2:   rows = ground truth / reconstruction,
                            columns = original HDRI / neutral environment;
  2. night capture, 2x2:    same layout;
  3. neutral only, 1 + 2:   the ground truth centred on top (the GT textures do not
                            depend on the capture, so the studio and night GT are the
                            same video up to Monte Carlo noise - measured at a mean
                            difference of 0.5/255), the reconstruction from the studio
                            capture bottom-left and the one from the night capture
                            bottom-right.  Under the neutral environment the render is
                            the albedo itself, so this block shows what each lighting
                            let the pipeline recover, with the lighting taken out.

"Reconstruction" is always `pbr_gt` (the PBR fit).  The NeRF configuration is
`exp_l1_d02` everywhere except `SwordShieldNight`, where exp+L1 collapses to zero
radiance and `softplus_relmseraw_d02` is used instead (`NERF_CFG`).

-------------------------------------------------------------------------------
Frame layout
-------------------------------------------------------------------------------
Canvas 1920x1140: a 60 px title band and two rows of two 960x540 tiles, i.e. every source
frame (1920x1080) scaled by exactly one half.  Tiles are inset by 2 px on each side so a
white 4 px seam separates them.  Each tile carries a small dark pill with a one- or
two-word label in its top-left corner.  Title band and pills are static within a block,
so they are rasterised once per block as an RGBA overlay and alpha-blended on every frame.

-------------------------------------------------------------------------------
Safety
-------------------------------------------------------------------------------
The source tree is read-only for this script: the only output is the mp4 (and its meta
JSON) under `<root>/comparison/`, which is created here.  Before writing anything the
script checks that every source exists, that all tiles of a block agree on frame count,
frame rate and resolution, and that their `turntable_meta.json` orbits are identical.

Encoding goes through `imageio` + `imageio-ffmpeg` (H.264, yuv420p, crf 18), the
combination PowerPoint plays without transcoding.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:    # noqa: BLE001
        pass

# ──────────────────────────────────────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_ROOT = Path("D:/rerender_video")
TILE_W, TILE_H = 960, 540
BAND_H = 60
SEAM = 2                                  # inset per tile side -> 4 px white seam
CANVAS_W, CANVAS_H = 2 * TILE_W, BAND_H + 2 * TILE_H

C_INK = (51, 64, 77)                      # #33404d, the ink of the thesis figures
C_PILL = (30, 36, 42, 190)
C_WHITE = (255, 255, 255)

NERF_CFG_DEFAULT = "exp_l1_d02"
NERF_CFG = {"SwordShieldNight": "softplus_relmseraw_d02"}
RECON = "pbr_gt"

INTERIOR = "TableAndOtherInteriorWithSpecular"
INTERIOR_NIGHT = "TableAndOtherInteriorWithSpecularNight"
SWORD = "SwordShieldStudio"
SWORD_NIGHT = "SwordShieldNight"


@dataclass
class Cell:
    row: int            # 0 = top, 1 = bottom
    x: int              # left edge of the tile on the canvas, in tile units * TILE_W/2
    label: str
    rel: str            # video folder relative to the root, without the `_8s` suffix


@dataclass
class Block:
    title: str
    cells: list


def cfg(scene: str) -> str:
    return NERF_CFG.get(scene, NERF_CFG_DEFAULT)


def gt(scene: str, env: str) -> str:
    return f"gt_control/{scene}/{env}"


def rec(scene: str, env: str) -> str:
    return f"{cfg(scene)}/{scene}/{RECON}/{env}"


def grid_block(title: str, scene: str) -> Block:
    """2x2: rows GT / reconstruction, columns HDRI / neutral."""
    return Block(title, [
        Cell(0, 0, "GT · HDRI", gt(scene, "hdri")),
        Cell(0, TILE_W, "GT · neutral", gt(scene, "neutral")),
        Cell(1, 0, "reconstruction · HDRI", rec(scene, "hdri")),
        Cell(1, TILE_W, "reconstruction · neutral", rec(scene, "neutral")),
    ])


def neutral_block(title: str, gt_scene: str, studio: str, night: str) -> Block:
    """1 + 2: the (shared) GT centred on top, the two reconstructions below."""
    return Block(title, [
        Cell(0, TILE_W // 2, "GT · neutral", gt(gt_scene, "neutral")),
        Cell(1, 0, "reconstruction from studio", rec(studio, "neutral")),
        Cell(1, TILE_W, "reconstruction from night", rec(night, "neutral")),
    ])


BLOCKS = [
    grid_block("Interior · studio", INTERIOR),
    grid_block("Interior · night", INTERIOR_NIGHT),
    neutral_block("Interior · neutral environment", INTERIOR, INTERIOR, INTERIOR_NIGHT),
    grid_block("Sword & shield · studio", SWORD),
    grid_block("Sword & shield · night", SWORD_NIGHT),
    neutral_block("Sword & shield · neutral environment", SWORD, SWORD, SWORD_NIGHT),
]


# ──────────────────────────────────────────────────────────────────────────────
# Sources
# ──────────────────────────────────────────────────────────────────────────────

def _font(size: int) -> ImageFont.FreeTypeFont:
    try:
        from matplotlib import font_manager
        path = font_manager.findfont("DejaVu Sans", fallback_to_default=True)
        return ImageFont.truetype(path, size)
    except Exception:    # noqa: BLE001
        return ImageFont.load_default()


def _probe(video: Path) -> dict:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    info = {"frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "size": (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))}
    cap.release()
    return info


def _orbit(folder: Path) -> dict | None:
    meta = folder / "turntable_meta.json"
    if not meta.exists():
        return None
    o = json.loads(meta.read_text(encoding="utf-8")).get("orbit", {})
    return {k: o.get(k) for k in ("target", "radius", "elevation", "start_azimuth", "direction")}


def resolve(root: Path, blocks: list[Block], suffix: str) -> list[dict]:
    """Check every source of every block and return the per-block source lists."""
    plan = []
    errors = []
    for b in blocks:
        srcs = []
        for c in b.cells:
            folder = root / (c.rel + suffix)
            video = folder / "turntable.mp4"
            if not video.exists():
                errors.append(f"missing {video}")
                continue
            srcs.append({"cell": c, "folder": folder, "video": video,
                         "probe": _probe(video), "orbit": _orbit(folder)})
        if len(srcs) != len(b.cells):
            continue
        probes = {json.dumps(s["probe"], sort_keys=True) for s in srcs}
        if len(probes) != 1:
            errors.append(f"[{b.title}] tiles disagree on frames/fps/size:\n    "
                          + "\n    ".join(f"{s['video']}: {s['probe']}" for s in srcs))
        orbits = {json.dumps(s["orbit"], sort_keys=True) for s in srcs}
        if len(orbits) != 1:
            errors.append(f"[{b.title}] tiles were rendered with different orbits:\n    "
                          + "\n    ".join(f"{s['folder']}: {s['orbit']}" for s in srcs))
        plan.append({"block": b, "sources": srcs, "probe": srcs[0]["probe"]})
    if errors:
        raise SystemExit("cannot build the comparison:\n  " + "\n  ".join(errors))
    return plan


# ──────────────────────────────────────────────────────────────────────────────
# Overlay
# ──────────────────────────────────────────────────────────────────────────────

def _tile_rect(c: Cell) -> tuple[int, int, int, int]:
    x0 = c.x + SEAM
    y0 = BAND_H + c.row * TILE_H + SEAM
    return x0, y0, x0 + TILE_W - 2 * SEAM, y0 + TILE_H - 2 * SEAM


def build_overlay(block: Block) -> tuple[np.ndarray, np.ndarray]:
    """Title band and label pills as (rgb float32, alpha float32) for alpha blending."""
    img = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    title_font = _font(30)
    label_font = _font(22)

    d.rectangle((0, 0, CANVAS_W, BAND_H), fill=C_WHITE + (255,))
    tw = d.textlength(block.title, font=title_font)
    d.text(((CANVAS_W - tw) / 2, BAND_H / 2), block.title, font=title_font,
           fill=C_INK + (255,), anchor="lm")

    for c in block.cells:
        x0, y0, _, _ = _tile_rect(c)
        pad_x, pad_y = 12, 6
        w = d.textlength(c.label, font=label_font)
        box = (x0 + 12, y0 + 12, x0 + 12 + w + 2 * pad_x, y0 + 12 + 22 + 2 * pad_y + 4)
        d.rounded_rectangle(box, radius=8, fill=C_PILL)
        d.text((box[0] + pad_x, (box[1] + box[3]) / 2), c.label, font=label_font,
               fill=C_WHITE + (255,), anchor="lm")

    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr[..., :3], arr[..., 3:4]


# ──────────────────────────────────────────────────────────────────────────────
# Composition
# ──────────────────────────────────────────────────────────────────────────────

def compose_block(entry: dict, writer, max_frames: int | None, step: int) -> int:
    block: Block = entry["block"]
    n_src = entry["probe"]["frames"]
    n = n_src if max_frames is None else min(n_src, max_frames)
    caps = [cv2.VideoCapture(str(s["video"])) for s in entry["sources"]]
    rects = [_tile_rect(s["cell"]) for s in entry["sources"]]
    ov_rgb, ov_a = build_overlay(block)
    base = np.full((CANVAS_H, CANVAS_W, 3), 255, np.uint8)

    written = 0
    t0 = time.time()
    for i in range(n):
        frames = []
        for cap in caps:
            ok, f = cap.read()
            if not ok:
                raise SystemExit(f"[{block.title}] source ended early at frame {i}")
            frames.append(f)
        if i % step:
            continue
        canvas = base.copy()
        for f, (x0, y0, x1, y1) in zip(frames, rects):
            tile = cv2.resize(f, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
            canvas[y0:y1, x0:x1] = tile[..., ::-1]              # BGR -> RGB
        out = canvas.astype(np.float32) / 255.0
        out = out * (1.0 - ov_a) + ov_rgb * ov_a
        writer.append_data((out * 255.0 + 0.5).astype(np.uint8))
        written += 1
    for cap in caps:
        cap.release()
    print(f"  [{block.title}] {written} frames in {time.time() - t0:.1f} s")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help="folder holding gt_control/, exp_l1_d02/, ... (default: %(default)s)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output mp4 (default: <root>/comparison/comparison.mp4)")
    ap.add_argument("--blocks", default=None,
                    help="comma-separated 1-based block indices to render, e.g. 1,3")
    ap.add_argument("--frames", type=int, default=None,
                    help="cap the frames read per block (layout checks)")
    ap.add_argument("--fps", type=int, default=None,
                    help="output frame rate; must divide the source rate (frames are dropped)")
    ap.add_argument("--short", action="store_true",
                    help="use the 4 s turntables instead of the *_8s ones")
    ap.add_argument("--crf", type=int, default=18, help="x264 quality (default: %(default)s)")
    ap.add_argument("--list", action="store_true", help="print the blocks and exit")
    args = ap.parse_args()

    suffix = "" if args.short else "_8s"
    blocks = BLOCKS
    if args.blocks:
        idx = [int(t) for t in args.blocks.split(",")]
        blocks = [BLOCKS[i - 1] for i in idx]
    if args.list:
        for i, b in enumerate(BLOCKS, 1):
            print(f"{i}. {b.title}")
            for c in b.cells:
                print(f"     {c.label:28s} {c.rel}{suffix}")
        return 0

    plan = resolve(args.root, blocks, suffix)
    src_fps = plan[0]["probe"]["fps"]
    for e in plan:
        if e["probe"]["fps"] != src_fps:
            raise SystemExit("blocks have different frame rates")
    out_fps = args.fps or int(round(src_fps))
    step = int(round(src_fps / out_fps))
    if abs(step * out_fps - src_fps) > 1e-6 or step < 1:
        raise SystemExit(f"--fps {out_fps} does not divide the source rate {src_fps}")

    out = args.out or (args.root / "comparison" / "comparison.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    for e in plan:
        for s in e["sources"]:
            if out.resolve().parent == s["folder"].resolve():
                raise SystemExit(f"refusing to write into a source folder: {s['folder']}")

    import imageio.v2 as imageio
    print(f"writing {out}  ({CANVAS_W}x{CANVAS_H} @ {out_fps} fps, crf {args.crf})")
    writer = imageio.get_writer(str(out), format="FFMPEG", mode="I", fps=out_fps,
                                codec="libx264", pixelformat="yuv420p",
                                macro_block_size=1, ffmpeg_params=["-crf", str(args.crf)])
    total = 0
    t0 = time.time()
    try:
        for e in plan:
            total += compose_block(e, writer, args.frames, step)
    finally:
        writer.close()
    print(f"done: {total} frames, {total / out_fps:.1f} s of video, {time.time() - t0:.0f} s wall")

    meta = {
        "output": str(out), "canvas": [CANVAS_W, CANVAS_H], "fps": out_fps,
        "frames": total, "suffix": suffix, "reconstruction": RECON,
        "nerf_cfg": {"default": NERF_CFG_DEFAULT, **NERF_CFG},
        "blocks": [{"title": e["block"].title,
                    "frames": e["probe"]["frames"] if args.frames is None
                    else min(e["probe"]["frames"], args.frames),
                    "cells": [{**asdict(s["cell"]), "video": str(s["video"])}
                              for s in e["sources"]]} for e in plan],
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out.with_name(out.stem + "_meta.json").write_text(json.dumps(meta, indent=2),
                                                      encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
