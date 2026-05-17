"""
Interactive Pygame viewer for a trained TinyNeRF — progressive-resolution + training-view mode.

The viewer has three modes
--------------------------
  Orbit (default)     drag LMB to orbit around scene centre, scroll to zoom
  Free-fly  [O]       WASD/QE to move, LMB drag to look
  Training views [←/→] snap camera to training-frame poses, split-screen GT ↔ prediction

Progressive rendering
---------------------
  During interaction (drag/key hold/scroll): renders at low resolution for fast feedback.
  After ~300 ms of inactivity: auto-refines to full resolution.

Controls
--------
  Orbit mode
    LMB drag    orbit yaw/pitch around scene centre
    Scroll      zoom in / out
    O           switch to Free-fly

  Free-fly mode
    W / S       forward / backward
    A / D       strafe left / right
    Q / E       move down / up (world Y)
    LMB drag    look (yaw / pitch)
    O           switch back to Orbit

  Training views mode  (enter with ← / →; exit by moving the camera)
    ← / →       previous / next training frame
    Home / End  first / last frame
    C           toggle GT/prediction split-screen on/off

  Common
    SPACE       toggle auto-render on/off
    R           force a full-resolution render right now
    H           reset camera to initial pose
    T           stop viewer, ask for N extra training iterations
    P           stop viewer, proceed to next pipeline step
    ESC         abort
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch


# ─── Public types ─────────────────────────────────────────────────────────────

@dataclass
class ViewerResult:
    action: Literal["continue", "proceed", "abort"]
    extra_iters: int = 0


# ─── Public entry point ───────────────────────────────────────────────────────

def launch_viewer(
    ckpt_path: str,
    transforms_json: str,
    idle_render_size: tuple[int, int] = (160, 120),
    interactive_render_size: tuple[int, int] = (80, 60),
    device: str | None = None,
) -> ViewerResult:
    """Open an interactive NeRF viewer with progressive rendering and training-view mode.

    Parameters
    ----------
    ckpt_path               : Path to .pkl checkpoint (nerf_module.save_checkpoint).
    transforms_json         : Path to transforms_extended.json (intrinsics + poses + GT paths).
    idle_render_size        : (W, H) resolution used when the camera is still.
    interactive_render_size : (W, H) resolution used while the camera is moving.
    device                  : torch device string, or None for auto.
    """
    try:
        import pygame  # noqa: F401
    except ImportError:
        raise ImportError(
            "pygame is required for the NeRF viewer.\n"
            "Install it with:  pip install pygame>=2.5"
        )

    from nerf_module import load_checkpoint, render_image

    torch_dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _, iter_done, nerf_cfg = load_checkpoint(ckpt_path, torch_dev)
    model.eval()

    json_dir = os.path.dirname(os.path.abspath(transforms_json))
    with open(transforms_json, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    src_w = int(data.get("w", idle_render_size[0]))
    src_h = int(data.get("h", idle_render_size[1]))

    if "fl_x" in data:
        src_fx = float(data["fl_x"])
    elif "camera_angle_x" in data:
        src_fx = 0.5 * src_w / math.tan(0.5 * float(data["camera_angle_x"]))
    else:
        raise ValueError("transforms JSON must contain 'fl_x' or 'camera_angle_x'")

    src_fy = float(data.get("fl_y", src_fx))
    if "camera_angle_y" in data:
        src_fy = 0.5 * src_h / math.tan(0.5 * float(data["camera_angle_y"]))

    frames = data.get("frames", [])
    frames_meta = []
    for i, fr in enumerate(frames):
        if "transform_matrix" not in fr:
            continue
        gt_path = _resolve_path(fr.get("file_path"), json_dir)
        frames_meta.append({
            "pose": np.array(fr["transform_matrix"], dtype=np.float32),
            "gt_path": gt_path,
            "name": os.path.basename(fr.get("file_path", f"frame_{i}")),
        })

    if not frames_meta:
        raise ValueError("No frames with 'transform_matrix' found in JSON")

    cam_positions = np.stack([f["pose"][:3, 3] for f in frames_meta])
    pivot = cam_positions.mean(axis=0)
    radii = np.linalg.norm(cam_positions - pivot, axis=1)
    init_radius = max(float(np.median(radii)), 1e-4)

    viewer = _Viewer(
        render_fn=render_image,
        model=model,
        nerf_cfg=nerf_cfg,
        device=torch_dev,
        idle_wh=idle_render_size,
        interactive_wh=interactive_render_size,
        src_w=src_w,
        src_h=src_h,
        src_fx=src_fx,
        src_fy=src_fy,
        pivot=pivot.astype(np.float32),
        init_radius=init_radius,
        init_c2w=frames_meta[0]["pose"],
        iter_done=iter_done,
        frames_meta=frames_meta,
    )
    return viewer.run()


# ─── Camera helpers ───────────────────────────────────────────────────────────

def _lookat_c2w(pos: np.ndarray, forward: np.ndarray) -> np.ndarray:
    """Build a 4×4 c2w matrix (camera looks along local -Z, OpenGL/NeRF convention).

    c2w[:,0]=right  c2w[:,1]=up  c2w[:,2]=-forward  c2w[:,3]=pos
    """
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = up / np.linalg.norm(up)
    c2w[:3, 2] = -forward
    c2w[:3, 3] = pos
    return c2w


def _c2w_to_yaw_pitch(c2w: np.ndarray) -> tuple[float, float]:
    fwd = -c2w[:3, 2]
    pitch = float(np.arcsin(np.clip(fwd[1], -1.0, 1.0)))
    yaw = float(np.arctan2(-fwd[0], -fwd[2]))
    return yaw, pitch


def _c2w_to_orbit(c2w: np.ndarray, pivot: np.ndarray) -> tuple[float, float, float]:
    pos = c2w[:3, 3]
    d = pos - pivot
    radius = float(np.linalg.norm(d))
    if radius < 1e-8:
        return 0.0, 0.0, 1.0
    d_n = d / radius
    pitch = float(np.arcsin(np.clip(d_n[1], -1.0, 1.0)))
    yaw = float(np.arctan2(d_n[0], d_n[2]))
    return yaw, pitch, radius


# ─── Path / image helpers ─────────────────────────────────────────────────────

def _resolve_path(maybe_path: str | None, base_dir: str) -> str | None:
    if not maybe_path:
        return None
    p = maybe_path if os.path.isabs(maybe_path) else os.path.join(base_dir, maybe_path)
    return p if os.path.exists(p) else None


def _load_gt_image(gt_path: str | None, target_wh: tuple[int, int]):
    """Load a GT image and return a pygame.Surface of size target_wh, or None on failure."""
    import pygame
    from PIL import Image

    if gt_path is None:
        return None
    try:
        ext = os.path.splitext(gt_path)[1].lower()
        if ext == ".exr":
            from nerf_module import _load_exr_rgb_np
            arr = _load_exr_rgb_np(gt_path)
            arr = np.clip(arr, 0.0, 1.0)
            rgb_uint8 = (arr * 255).astype(np.uint8)
            im = Image.fromarray(rgb_uint8, mode="RGB")
        else:
            im = Image.open(gt_path).convert("RGB")
        im = im.resize((target_wh[0], target_wh[1]), Image.LANCZOS)
        arr = np.array(im, dtype=np.uint8)  # (H, W, 3)
        return pygame.surfarray.make_surface(arr.transpose(1, 0, 2))  # (W, H, 3)
    except Exception as exc:
        print(f"[Viewer] Could not load GT image {gt_path}: {exc}")
        return None


def _make_placeholder_surface(w: int, h: int, text: str):
    """Return a dark pygame.Surface with centred text."""
    import pygame
    surf = pygame.Surface((w, h))
    surf.fill((40, 40, 40))
    try:
        font = pygame.font.SysFont("Consolas", 13)
    except Exception:
        font = pygame.font.Font(None, 14)
    label = font.render(text, True, (180, 180, 180))
    r = label.get_rect(center=(w // 2, h // 2))
    surf.blit(label, r)
    return surf


# ─── Viewer class ─────────────────────────────────────────────────────────────

class _Viewer:
    _IDLE_DELAY = 0.30   # seconds of stillness before full-res refine

    def __init__(
        self,
        render_fn,
        model,
        nerf_cfg,
        device,
        idle_wh: tuple[int, int],
        interactive_wh: tuple[int, int],
        src_w: int,
        src_h: int,
        src_fx: float,
        src_fy: float,
        pivot: np.ndarray,
        init_radius: float,
        init_c2w: np.ndarray,
        iter_done: int,
        frames_meta: list[dict],
    ):
        self._render_fn = render_fn
        self._model = model
        self._cfg = nerf_cfg
        self._device = device
        self._idle_wh = tuple(idle_wh)
        self._interactive_wh = tuple(interactive_wh)
        self._src_w, self._src_h = src_w, src_h
        self._src_fx, self._src_fy = src_fx, src_fy
        self._pivot = pivot
        self._init_radius = init_radius
        self._init_c2w = init_c2w.copy()
        self._iter_done = iter_done
        self._frames_meta = frames_meta

        # Display window: 4× full-res, min 640×480
        iW, iH = idle_wh
        scale = max(4, min(1280 // iW, 960 // iH))
        self._dW = iW * scale
        self._dH = iH * scale

        # Camera state
        self._mode = "orbit"
        self._orbit_yaw = 0.0
        self._orbit_pitch = 0.0
        self._orbit_radius = init_radius
        self._fly_pos = np.zeros(3, dtype=np.float32)
        self._fly_yaw = 0.0
        self._fly_pitch = 0.0
        self._reset_camera()

        # Progressive render state
        self._auto_render = True
        self._dirty = True
        self._interacting_until = 0.0
        self._last_surf = None
        self._last_render_s = 0.0
        self._last_render_wh: tuple[int, int] | None = None

        # Training-view state
        self._current_frame_idx: int | None = None
        self._compare_mode: bool = False
        self._gt_cache: dict[int, object] = {}   # int → pygame.Surface or None

    # ── Camera ────────────────────────────────────────────────────────────────

    def _reset_camera(self):
        yaw, pitch, radius = _c2w_to_orbit(self._init_c2w, self._pivot)
        self._orbit_yaw = yaw
        self._orbit_pitch = pitch
        self._orbit_radius = radius if radius > 1e-4 else self._init_radius
        self._fly_pos = self._init_c2w[:3, 3].copy()
        self._fly_yaw, self._fly_pitch = _c2w_to_yaw_pitch(self._init_c2w)
        self._dirty = True

    def _current_c2w(self) -> np.ndarray:
        if self._mode == "orbit":
            y, p, r = self._orbit_yaw, self._orbit_pitch, self._orbit_radius
            dfp = np.array(
                [np.sin(y) * np.cos(p), np.sin(p), np.cos(y) * np.cos(p)], dtype=np.float32
            )
            return _lookat_c2w(self._pivot + r * dfp, -dfp)
        else:
            y, p = self._fly_yaw, self._fly_pitch
            fwd = np.array(
                [-np.sin(y) * np.cos(p), np.sin(p), -np.cos(y) * np.cos(p)], dtype=np.float32
            )
            return _lookat_c2w(self._fly_pos, fwd)

    def _mark_interacting(self):
        self._interacting_until = time.time() + self._IDLE_DELAY
        self._dirty = True

    # ── Training-view helpers ─────────────────────────────────────────────────

    def _enter_frame_view(self, idx: int):
        n = len(self._frames_meta)
        idx = max(0, min(n - 1, idx))
        self._current_frame_idx = idx
        self._compare_mode = True
        # Snap camera to frame pose
        c2w = self._frames_meta[idx]["pose"]
        yaw, pitch, radius = _c2w_to_orbit(c2w, self._pivot)
        self._orbit_yaw = yaw
        self._orbit_pitch = pitch
        self._orbit_radius = radius if radius > 1e-4 else self._init_radius
        self._fly_pos = c2w[:3, 3].copy()
        self._fly_yaw, self._fly_pitch = _c2w_to_yaw_pitch(c2w)
        self._dirty = True
        self._interacting_until = 0.0  # idle immediately → full-res render

    def _exit_frame_view(self):
        self._current_frame_idx = None
        self._compare_mode = False

    def _get_gt_surface(self, idx: int):
        """Return a pygame.Surface for the GT image at frame idx (lazy + cached)."""
        if idx not in self._gt_cache:
            gt_path = self._frames_meta[idx].get("gt_path")
            half_w = self._dW // 2
            surf = _load_gt_image(gt_path, (half_w, self._dH))
            if surf is None:
                surf = _make_placeholder_surface(half_w, self._dH, "GT non disponibile")
            self._gt_cache[idx] = surf
        return self._gt_cache[idx]

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _choose_render_wh(self) -> tuple[int, int]:
        if time.time() < self._interacting_until:
            return self._interactive_wh
        return self._idle_wh

    def _do_render(self, W: int, H: int):
        import pygame
        fx = self._src_fx * W / self._src_w
        fy = self._src_fy * H / self._src_h
        c2w = torch.from_numpy(self._current_c2w()).to(self._device)
        t0 = time.time()
        with torch.no_grad():
            rgb, _, _ = self._render_fn(
                H, W, fx, c2w, self._model, self._cfg,
                focal_y=fy, randomize=False,
            )
        self._last_render_s = time.time() - t0
        self._last_render_wh = (W, H)
        rgb_np = (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)  # (H,W,3)
        self._last_surf = pygame.surfarray.make_surface(rgb_np.transpose(1, 0, 2))
        self._dirty = False

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> ViewerResult:
        import pygame

        pygame.init()
        pygame.display.set_caption(f"NeRF Viewer  —  checkpoint iter {self._iter_done}")
        screen = pygame.display.set_mode((self._dW, self._dH))
        clock = pygame.time.Clock()
        try:
            font = pygame.font.SysFont("Consolas", 13)
        except Exception:
            font = pygame.font.Font(None, 14)

        self._print_keybindings()

        move_speed = self._init_radius * 0.04
        pitch_limit = math.pi / 2 - 0.04
        drag_btn: int | None = None

        result = ViewerResult(action="proceed")
        running = True

        while running:
            # ── Continuous fly movement ──────────────────────────────────────
            if self._mode == "fly":
                keys = pygame.key.get_pressed()
                cos_y, sin_y = math.cos(self._fly_yaw), math.sin(self._fly_yaw)
                fwd_xz = np.array([-sin_y, 0.0, -cos_y], dtype=np.float32)
                right   = np.array([ cos_y, 0.0, -sin_y], dtype=np.float32)
                moved = False
                if keys[pygame.K_w]: self._fly_pos += fwd_xz * move_speed; moved = True
                if keys[pygame.K_s]: self._fly_pos -= fwd_xz * move_speed; moved = True
                if keys[pygame.K_a]: self._fly_pos -= right   * move_speed; moved = True
                if keys[pygame.K_d]: self._fly_pos += right   * move_speed; moved = True
                if keys[pygame.K_e]: self._fly_pos[1] += move_speed;        moved = True
                if keys[pygame.K_q]: self._fly_pos[1] -= move_speed;        moved = True
                if moved:
                    self._exit_frame_view()
                    self._mark_interacting()

            # ── Events ───────────────────────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    result = ViewerResult(action="abort")
                    running = False

                elif event.type == pygame.KEYDOWN:
                    k = event.key

                    if k == pygame.K_ESCAPE:
                        result = ViewerResult(action="abort")
                        running = False

                    elif k == pygame.K_p:
                        result = ViewerResult(action="proceed")
                        running = False

                    elif k == pygame.K_t:
                        pygame.display.set_caption("Inserisci N nel terminale…")
                        n = _prompt_int(
                            "\n[NeRF Viewer] Quante iterazioni aggiuntive? (0 = annulla) > "
                        )
                        if n and n > 0:
                            result = ViewerResult(action="continue", extra_iters=n)
                            running = False
                        else:
                            pygame.display.set_caption(
                                f"NeRF Viewer  —  checkpoint iter {self._iter_done}"
                            )

                    elif k == pygame.K_r:
                        self._do_render(*self._idle_wh)

                    elif k == pygame.K_SPACE:
                        self._auto_render = not self._auto_render

                    elif k == pygame.K_h:
                        self._exit_frame_view()
                        self._reset_camera()
                        self._mark_interacting()

                    elif k == pygame.K_o:
                        self._mode = "fly" if self._mode == "orbit" else "orbit"
                        self._mark_interacting()

                    # ── Training-view navigation ─────────────────────────────
                    elif k == pygame.K_LEFT:
                        cur = self._current_frame_idx
                        if cur is None:
                            self._enter_frame_view(len(self._frames_meta) - 1)
                        else:
                            self._enter_frame_view(cur - 1)

                    elif k == pygame.K_RIGHT:
                        cur = self._current_frame_idx
                        if cur is None:
                            self._enter_frame_view(0)
                        else:
                            self._enter_frame_view(cur + 1)

                    elif k == pygame.K_HOME:
                        self._enter_frame_view(0)

                    elif k == pygame.K_END:
                        self._enter_frame_view(len(self._frames_meta) - 1)

                    elif k == pygame.K_c and self._current_frame_idx is not None:
                        self._compare_mode = not self._compare_mode

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    drag_btn = event.button

                elif event.type == pygame.MOUSEBUTTONUP:
                    drag_btn = None

                elif event.type == pygame.MOUSEMOTION and drag_btn == 1:
                    dx, dy = event.rel
                    sens = 0.006
                    self._exit_frame_view()
                    if self._mode == "orbit":
                        self._orbit_yaw += dx * sens
                        self._orbit_pitch = float(np.clip(
                            self._orbit_pitch - dy * sens, -pitch_limit, pitch_limit
                        ))
                    else:
                        self._fly_yaw += dx * sens
                        self._fly_pitch = float(np.clip(
                            self._fly_pitch - dy * sens, -pitch_limit, pitch_limit
                        ))
                    self._mark_interacting()

                elif event.type == pygame.MOUSEWHEEL and self._mode == "orbit":
                    self._exit_frame_view()
                    self._orbit_radius = max(1e-2, self._orbit_radius * (1.0 - event.y * 0.1))
                    self._mark_interacting()

            # ── Auto-render decision ──────────────────────────────────────────
            if self._auto_render:
                if self._dirty:
                    W, H = self._choose_render_wh()
                    self._do_render(W, H)
                elif (
                    self._last_render_wh != self._idle_wh
                    and time.time() >= self._interacting_until
                    and self._last_surf is not None
                ):
                    self._do_render(*self._idle_wh)

            # ── Draw ─────────────────────────────────────────────────────────
            screen.fill((25, 25, 25))

            in_compare = self._compare_mode and self._current_frame_idx is not None

            if self._last_surf is not None:
                if in_compare:
                    half_w = self._dW // 2
                    # Right half: NeRF prediction
                    pred_scaled = pygame.transform.scale(self._last_surf, (half_w, self._dH))
                    screen.blit(pred_scaled, (half_w, 0))
                    # Left half: Ground truth
                    gt_surf = self._get_gt_surface(self._current_frame_idx)
                    screen.blit(gt_surf, (0, 0))
                    # Divider
                    pygame.draw.line(screen, (200, 200, 200), (half_w, 0), (half_w, self._dH), 2)
                    # Labels
                    try:
                        lbl_font = pygame.font.SysFont("Consolas", 16)
                    except Exception:
                        lbl_font = pygame.font.Font(None, 17)
                    screen.blit(lbl_font.render("Ground Truth", True, (255, 255, 100)), (8, self._dH - 24))
                    screen.blit(lbl_font.render("NeRF Prediction", True, (255, 255, 100)), (half_w + 8, self._dH - 24))
                else:
                    scaled = pygame.transform.scale(self._last_surf, (self._dW, self._dH))
                    screen.blit(scaled, (0, 0))
            else:
                _blit_center(screen, font, "Auto-render attivo — in attesa del primo frame…")

            self._draw_overlay(screen, font)
            pygame.display.flip()
            clock.tick(60)

        pygame.quit()
        return result

    # ── Overlay ───────────────────────────────────────────────────────────────

    def _draw_overlay(self, screen, font):
        import pygame

        in_frame = self._current_frame_idx is not None

        if in_frame:
            idx = self._current_frame_idx
            n = len(self._frames_meta)
            name = self._frames_meta[idx].get("name", str(idx))
            compare_hint = "C=nascondi GT" if self._compare_mode else "C=mostra GT"
            mode_line = (
                f"TRAINING VIEW  Frame {idx+1}/{n}  —  {name}"
                f"  [{compare_hint}]  [←/→=prev/next]  [Home/End]"
            )
        else:
            mode_lbl = "ORBIT [O→fly]" if self._mode == "orbit" else "FREE-FLY [O→orbit]"
            ar_lbl = "AUTO ON [SPACE=off]" if self._auto_render else "AUTO OFF [SPACE=on]"
            mode_line = f"Modalità: {mode_lbl}  |  {ar_lbl}"

        res_lbl = ""
        if self._last_render_wh:
            iW, iH = self._last_render_wh
            tag = "full" if self._last_render_wh == self._idle_wh else "low-res"
            res_lbl = f"  |  {iW}×{iH} ({tag}, {self._last_render_s:.2f}s)"

        lines = [
            mode_line + res_lbl,
            f"Iter checkpoint: {self._iter_done}",
        ]
        if not in_frame:
            lines.append("←/→=viste training  R=full render  H=reset  T=train  P=procedi  ESC=annulla")
        else:
            lines.append("Trascina/WASD/scroll per uscire dalla training view e navigare liberamente")

        bg = pygame.Surface((self._dW, len(lines) * 18 + 10), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 150))
        screen.blit(bg, (0, 0))
        for i, line in enumerate(lines):
            surf = font.render(line, True, (230, 230, 80))
            screen.blit(surf, (8, 6 + i * 18))

    def _print_keybindings(self):
        sep = "─" * 62
        iW, iH = self._interactive_wh
        fW, fH = self._idle_wh
        n = len(self._frames_meta)
        print(f"\n{sep}")
        print(f"  NeRF Viewer  —  checkpoint iter {self._iter_done}")
        print(f"  Render: {iW}×{iH} (movimento) → {fW}×{fH} (fermo, auto-refine)")
        print(f"  {n} training frames disponibili")
        print(sep)
        print("  Orbit mode (default):")
        print("    LMB drag    ruota attorno alla scena")
        print("    Scroll      zoom")
        print("  Free-fly mode (O per passare):")
        print("    W/S/A/D     avanti/indietro/strafe  |  Q/E  su/giù")
        print("    LMB drag    guarda")
        print("  Training views (←/→ per entrare):")
        print("    ← / →       frame precedente / successivo")
        print("    Home / End  primo / ultimo frame")
        print("    C           toggle split-screen GT ↔ predizione")
        print("    Muovi camera → esci e torna alla navigazione libera")
        print("  Comune:")
        print("    SPACE       toggle auto-render  |  R  render piena risoluzione")
        print("    H           reset posa  |  T  continua training  |  P  procedi")
        print("    ESC         annulla")
        print(f"{sep}\n")


# ─── Utilities ────────────────────────────────────────────────────────────────

def _prompt_int(prompt: str) -> int | None:
    try:
        return int(input(prompt).strip())
    except (ValueError, EOFError, KeyboardInterrupt):
        return None


def _blit_center(screen, font, text: str):
    import pygame
    surf = font.render(text, True, (180, 180, 180))
    r = surf.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
    screen.blit(surf, r)
