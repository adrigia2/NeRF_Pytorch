"""nerf_viewer.py — interactive novel-view viewer for the trained NeRF.

Per-frame pipeline:  orbit pose → OptiX DepthGenerator (mesh depth)
                     → nerf.render_image (depth-guided) → tonemap → cv2 window.

Usage:
    python nerf_viewer.py --ckpt <nerf_model_cache.pt> \
                          --transforms <transforms_extended.json> \
                          --obj <model.obj> [--res 256] [--chunk N]

Controls (with the window focused):
    A / D       orbit in azimuth
    W / S       orbit in elevation
    Q / E       zoom out / in
    ← / →       pan the orbit centre: left / right (the camera's right)
    ↑ / ↓       pan the orbit centre: forward / backward (the camera's forward)
    + / -       exposure (±0.5 EV)
    H           render at the dataset resolution (slow, one-off)
    R           reload the checkpoint from disk
    T           toggle watch mode: reload automatically when the ckpt changes
                (useful while training runs, since it saves every display_every)
    X           save the current view (linear EXR + tonemapped PNG)
    ESC         quit
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

import _paths  # noqa: F401

WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # Z-up (Blender)
ELEV_LIMIT = np.deg2rad(85.0)
DEPTH_MISS_SENTINEL = 1e10


def _orbit_c2w(center: np.ndarray, radius: float, az: float, el: float) -> np.ndarray:
    """Posa camera-to-world 4×4 (convenzione NeRF: -Z forward) in orbita attorno a center.

    az=0, el=0 → camera on the -Y side looking towards +Y (Blender's forward).
    """
    ce, se = np.cos(el), np.sin(el)
    offset = np.array([ce * np.sin(az), -ce * np.cos(az), se], dtype=np.float32)
    pos = center + radius * offset
    fwd = -offset
    right = np.cross(fwd, WORLD_UP)
    right /= max(np.linalg.norm(right), 1e-8)
    up_cam = np.cross(right, fwd)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = up_cam
    c2w[:3, 2] = -fwd
    c2w[:3, 3] = pos
    return c2w


def _orbit_from_pose(pose: np.ndarray, center: np.ndarray) -> tuple[float, float, float]:
    """(radius, az, el) of the pose relative to the centre — the inverse of _orbit_c2w."""
    off = pose[:3, 3] - center
    radius = float(np.linalg.norm(off))
    d = off / max(radius, 1e-8)
    el = float(np.arcsin(np.clip(d[2], -1.0, 1.0)))
    az = float(np.arctan2(d[0], -d[1]))
    return radius, az, el


def _tonemap(img: np.ndarray, ev: float) -> np.ndarray:
    """Linear HDR → uint8 BGR for display: exposure, Reinhard, gamma 2.2."""
    x = np.clip(img, 0.0, None) * (2.0 ** ev)
    x = x / (1.0 + x)
    x = np.clip(x, 0.0, 1.0) ** (1.0 / 2.2)
    return (x * 255.0 + 0.5).astype(np.uint8)[..., ::-1]


def _write_exr(img: np.ndarray, path: str) -> None:
    import OpenEXR, Imath
    img = np.ascontiguousarray(img.astype(np.float32))
    h, w, _ = img.shape
    pt = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header = OpenEXR.Header(w, h)
    header["channels"] = {"R": pt, "G": pt, "B": pt}
    f = OpenEXR.OutputFile(path, header)
    f.writePixels({"R": img[..., 0].tobytes(), "G": img[..., 1].tobytes(),
                   "B": img[..., 2].tobytes()})
    f.close()


class NerfViewer:
    def __init__(self, ckpt_path: str, transforms_path: str, obj_path: str,
                 res: int, chunk: int | None):
        import OptixProgrammablePasses as optix
        from nerf import load_checkpoint

        self.optix = optix
        optix.LogManager.set_min_level(optix.LogLevel.Error)
        optix.OptixManager.instance().set_log_level(optix.LogLevel.Disabled)

        self.ckpt_path = ckpt_path
        self._load_ckpt = load_checkpoint
        self.bundle, self.cfg, self.iter_done = load_checkpoint(ckpt_path, return_iter=True)
        if chunk is not None:
            self.cfg.chunk = chunk

        with open(transforms_path, encoding="utf-8") as fh:
            data = json.load(fh)
        self.data_w  = int(data["w"])
        self.data_h  = int(data["h"])
        self.fl_x    = float(data["fl_x"])
        self.fl_y    = float(data.get("fl_y", data["fl_x"]))
        self.fovy    = float(data["camera_angle_y"])  # radians, as in Steps 1 and 3

        # Preview resolution: long side = res, dataset aspect ratio
        scale = res / max(self.data_w, self.data_h)
        self.vw = max(1, round(self.data_w * scale))
        self.vh = max(1, round(self.data_h * scale))

        self.mesh = optix.TriangleMesh()
        self.mesh.add_from_obj_file(obj_path)
        self.depth_gen = optix.DepthGenerator()
        self.depth_gen.set_traversable(self.mesh)
        self.depth_gen.need_render_depth(True)
        self.depth_gen.need_render_position(False)
        self.depth_gen.need_render_normal(False)

        # Initial orbit: the same framing as the first training frame
        center = self.bundle[4].cpu().numpy().astype(np.float32)
        pose0 = np.array(data["frames"][0]["transform_matrix"], dtype=np.float32)
        self.center = center
        self.radius, self.az, self.el = _orbit_from_pose(pose0, center)

        self.ev = 0.0
        self.last_img: np.ndarray | None = None   # ultimo render float (preview o hi-res)
        self.last_was_hires = False
        self.captures_dir = Path(ckpt_path).parent / "viewer_captures"

    # ── rendering ────────────────────────────────────────────────────────────
    def _render_depth(self, c2w: np.ndarray, w: int, h: int) -> np.ndarray:
        pos = c2w[:3, 3].tolist()
        fwd = (-c2w[:3, 2]).tolist()
        up  = c2w[:3, 1].tolist()
        cam = self.optix.Camera(pos, fwd, up, self.fovy, [w, h])
        self.depth_gen.set_camera(cam)
        self.depth_gen.render()
        d = self.depth_gen.get_result().depths_np.astype(np.float32).reshape(h, w)
        return np.where(d >= DEPTH_MISS_SENTINEL, 0.0, d)

    def render(self, hires: bool = False) -> np.ndarray:
        from nerf import render_image
        if hires:
            w, h = self.data_w, self.data_h
            fx, fy = self.fl_x, self.fl_y
        else:
            w, h = self.vw, self.vh
            fx = self.fl_x * w / self.data_w
            fy = self.fl_y * h / self.data_h
        c2w = _orbit_c2w(self.center, self.radius, self.az, self.el)
        depth = self._render_depth(c2w, w, h)
        t0 = time.perf_counter()
        img = render_image(self.bundle, h, w, fx, c2w, self.cfg,
                           focal_y=fy, target_depth=depth)
        dt = time.perf_counter() - t0
        print(f"  render {w}x{h}: {dt:.2f}s  "
              f"(az={np.rad2deg(self.az):.0f}° el={np.rad2deg(self.el):.0f}° r={self.radius:.2f})")
        self.last_img = img
        self.last_was_hires = hires
        return img

    # ── checkpoint reload ────────────────────────────────────────────────────
    def reload_ckpt(self) -> bool:
        try:
            self.bundle, cfg, self.iter_done = self._load_ckpt(self.ckpt_path, return_iter=True)
            cfg.chunk = self.cfg.chunk
            self.cfg = cfg
            print(f"[ok] checkpoint ricaricato (iter {self.iter_done})")
            return True
        except Exception as exc:  # file in scrittura / corrotto: riprova al prossimo giro
            print(f"[warn] reload fallito ({exc}); riprovo piu' tardi")
            return False

    # ── capture ──────────────────────────────────────────────────────────────
    def save_capture(self, disp_bgr: np.ndarray) -> None:
        import cv2
        if self.last_img is None:
            return
        self.captures_dir.mkdir(parents=True, exist_ok=True)
        stem = (f"iter{self.iter_done:06d}_az{np.rad2deg(self.az):+.0f}"
                f"_el{np.rad2deg(self.el):+.0f}_r{self.radius:.2f}")
        exr = self.captures_dir / f"{stem}.exr"
        png = self.captures_dir / f"{stem}.png"
        _write_exr(self.last_img, str(exr))
        cv2.imwrite(str(png), disp_bgr)
        print(f"[ok] vista salvata: {exr}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Viewer interattivo NeRF (OptiX depth + render depth-guided)")
    ap.add_argument("--ckpt",       required=True, help="checkpoint NeRF (nerf_model_cache.pt)")
    ap.add_argument("--transforms", required=True, help="transforms_extended.json dello Step 1")
    ap.add_argument("--obj",        required=True, help="the scene's OBJ mesh")
    ap.add_argument("--res",   type=int, default=256, help="long side of the preview (default 256)")
    ap.add_argument("--chunk", type=int, default=None, help="override cfg.chunk (riduci se VRAM scarsa)")
    args = ap.parse_args()

    import cv2

    viewer = NerfViewer(args.ckpt, args.transforms, args.obj, args.res, args.chunk)
    print(__doc__.split("Comandi")[1].join(["Comandi", ""]))
    print(f"Checkpoint: iter {viewer.iter_done} — preview {viewer.vw}x{viewer.vh}")

    win = "nerf_viewer"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    AZ_STEP, EL_STEP, ZOOM = np.deg2rad(10.0), np.deg2rad(7.5), 1.15
    dirty = True
    auto_reload = False
    last_mtime = os.path.getmtime(args.ckpt)
    last_poll = time.monotonic()
    disp = None

    while True:
        if dirty:
            img = viewer.render()
            disp_scale = max(1, round(640 / max(viewer.vw, viewer.vh)))
            disp = _tonemap(img, viewer.ev)
            if viewer.last_was_hires:
                disp = cv2.resize(disp, (viewer.vw * disp_scale, viewer.vh * disp_scale),
                                  interpolation=cv2.INTER_AREA)
            elif disp_scale > 1:
                disp = cv2.resize(disp, None, fx=disp_scale, fy=disp_scale,
                                  interpolation=cv2.INTER_NEAREST)
            disp = np.ascontiguousarray(disp)
            hud = (f"iter {viewer.iter_done}  ev {viewer.ev:+.1f}  "
                   f"az {np.rad2deg(viewer.az):.0f}  el {np.rad2deg(viewer.el):.0f}"
                   f"{'  [watch]' if auto_reload else ''}")
            cv2.putText(disp, hud, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(disp, hud, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(win, disp)
            dirty = False

        kraw = cv2.waitKeyEx(50)          # waitKeyEx: needed for the extended codes (arrows)
        k = (kraw & 0xFF) if kraw != -1 else 255   # masked value, for the ASCII keys
        if k == 27:                       # ESC
            break
        elif k in (ord("a"), ord("A")):
            viewer.az -= AZ_STEP; dirty = True
        elif k in (ord("d"), ord("D")):
            viewer.az += AZ_STEP; dirty = True
        elif k in (ord("w"), ord("W")):
            viewer.el = min(viewer.el + EL_STEP, ELEV_LIMIT); dirty = True
        elif k in (ord("s"), ord("S")):
            viewer.el = max(viewer.el - EL_STEP, -ELEV_LIMIT); dirty = True
        elif k in (ord("q"), ord("Q")):
            viewer.radius *= ZOOM; dirty = True
        elif k in (ord("e"), ord("E")):
            viewer.radius /= ZOOM; dirty = True
        # arrows: pan the orbit centre relative to the camera orientation (full 3D)
        elif kraw in (65361, 2424832, 65363, 2555904,    # left / right
                      65362, 2490368, 65364, 2621440):   # up / down
            c2w_cur = _orbit_c2w(viewer.center, viewer.radius, viewer.az, viewer.el)
            cam_right = c2w_cur[:3, 0]
            cam_fwd = -c2w_cur[:3, 2]                     # direzione verso cui guarda la camera
            step = viewer.radius * 0.1
            if kraw in (65361, 2424832):      # ←  sinistra
                viewer.center -= cam_right * step
            elif kraw in (65363, 2555904):    # →  destra
                viewer.center += cam_right * step
            elif kraw in (65362, 2490368):    # ↑  forward (along the camera's forward)
                viewer.center += cam_fwd * step
            else:                             # ↓  indietro
                viewer.center -= cam_fwd * step
            dirty = True
        elif k in (ord("+"), ord("=")):
            viewer.ev += 0.5; dirty = True
        elif k == ord("-"):
            viewer.ev -= 0.5; dirty = True
        elif k in (ord("h"), ord("H")):
            print(f"  render hi-res {viewer.data_w}x{viewer.data_h}…")
            viewer.render(hires=True)
            disp_hi = _tonemap(viewer.last_img, viewer.ev)
            cv2.imshow(win, disp_hi)
            disp = disp_hi
        elif k in (ord("r"), ord("R")):
            if viewer.reload_ckpt():
                dirty = True
        elif k in (ord("t"), ord("T")):
            auto_reload = not auto_reload
            print(f"  watch-mode: {'ON' if auto_reload else 'OFF'}")
            dirty = True
        elif k in (ord("x"), ord("X")):
            if disp is not None:
                viewer.save_capture(disp)

        if auto_reload and time.monotonic() - last_poll > 2.0:
            last_poll = time.monotonic()
            try:
                mtime = os.path.getmtime(args.ckpt)
            except OSError:
                mtime = last_mtime
            if mtime != last_mtime:
                last_mtime = mtime
                if viewer.reload_ckpt():
                    dirty = True

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
