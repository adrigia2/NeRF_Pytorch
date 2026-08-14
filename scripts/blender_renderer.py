"""
blender_renderer.py
-------------------
Renders a 3D scene from all cameras defined in a NeRF-format transforms.json,
using Blender's Cycles or EEVEE engine.

Run via:
    blender --background --python blender_renderer.py -- \
        --model      path/to/model.obj \
        --transforms path/to/transforms.json \
        --output-dir path/to/output \
        [--output-format PNG|EXR|JPEG] \
        [--skybox    path/to/sky.hdr] \
        [--engine    CYCLES|EEVEE] \
        [--samples   128] \
        [--axis-correction] \
        [--mat-color R G B [A]] \
        [--mat-albedo path] [--mat-normal path] \
        [--mat-roughness path|float] [--mat-metallic path|float] \
        [--mat-emission path] [--mat-alpha path]

Output layout:
    output_dir/
        renders/    ← one image per camera frame
        scene/      ← scene.blend with all textures packed in
        textures/   ← copies of every --mat-* texture file
        skybox/     ← copy of the --skybox HDRI
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Capture the launch directory before Blender can change it.
# All relative paths passed via CLI are resolved against this.
_LAUNCH_DIR = Path.cwd()


def _abs(p: str | None) -> str | None:
    """Resolve a path to absolute, anchored at the launch CWD."""
    if p is None:
        return None
    q = Path(p)
    return str(q if q.is_absolute() else (_LAUNCH_DIR / q).resolve())


# ──────────────────────────────────────────────────────────────────────────────
# Camera / transforms dataclasses  (verbatim from images_generator.py:321-394)
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
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    # Blender's own args sit before "--"; everything after is ours.
    argv = sys.argv
    try:
        argv = argv[argv.index("--") + 1:]
    except ValueError:
        argv = []

    p = argparse.ArgumentParser(
        prog="blender_renderer",
        description="Render a scene from NeRF-format cameras using Blender.",
    )
    p.add_argument("--model",          required=True,  help="Path to .obj model file")
    p.add_argument("--transforms",     required=True,  help="Path to transforms.json")
    p.add_argument("--output-dir",     required=True,  dest="output_dir")
    p.add_argument("--output-format",  default="PNG",  dest="output_format",
                   choices=["PNG", "EXR", "JPEG"])
    p.add_argument("--skybox",         default=None,   help="Path to HDRI skybox (.hdr/.exr)")
    p.add_argument("--engine",         default="CYCLES", choices=["CYCLES", "EEVEE"])
    p.add_argument("--samples",        default=128,    type=int)
    p.add_argument("--axis-correction", action="store_true", dest="axis_correction",
                   help="Apply NeRF Y-up → Blender Z-up coordinate change")
    p.add_argument("--device", default="GPU", choices=["GPU", "CPU"],
                   help="Compute device for Cycles rendering (default: GPU)")

    # PBR material override
    p.add_argument("--mat-color",     nargs="+", type=float, dest="mat_color",
                   metavar=("R", "G"),
                   help="Flat base color: R G B [A], values 0–1")
    p.add_argument("--mat-albedo",    default=None, dest="mat_albedo",
                   help="Albedo texture → Base Color")
    p.add_argument("--mat-normal",    default=None, dest="mat_normal",
                   help="Normal map texture → Normal")
    p.add_argument("--mat-roughness", default=None, dest="mat_roughness",
                   help="Roughness texture path or scalar 0–1")
    p.add_argument("--mat-metallic",  default=None, dest="mat_metallic",
                   help="Metallic texture path or scalar 0–1")
    p.add_argument("--mat-emission",  default=None, dest="mat_emission",
                   help="Emission texture → Emission Color")
    p.add_argument("--mat-alpha",     default=None, dest="mat_alpha",
                   help="Alpha/opacity texture → Alpha")

    args = p.parse_args(argv)

    # Resolve all paths to absolute (anchored at launch CWD, not Blender's CWD)
    args.model        = _abs(args.model)
    args.transforms   = _abs(args.transforms)
    args.output_dir   = _abs(args.output_dir)
    args.skybox       = _abs(args.skybox)
    args.mat_albedo   = _abs(args.mat_albedo)
    args.mat_normal   = _abs(args.mat_normal)
    args.mat_emission = _abs(args.mat_emission)
    args.mat_alpha    = _abs(args.mat_alpha)
    # roughness / metallic can be a scalar string — only resolve if it looks like a path
    for attr in ("mat_roughness", "mat_metallic"):
        val = getattr(args, attr)
        if val is not None:
            try:
                float(val)   # it's a scalar — leave it as-is
            except ValueError:
                setattr(args, attr, _abs(val))

    # Validate required paths (after resolution so errors show the absolute path)
    if not Path(args.model).is_file():
        p.error(f"Model not found: {args.model}")
    if not Path(args.transforms).is_file():
        p.error(f"transforms.json not found: {args.transforms}")

    return args


# ──────────────────────────────────────────────────────────────────────────────
# Blender scene utilities
# ──────────────────────────────────────────────────────────────────────────────

def clear_scene() -> None:
    import bpy
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)
    for block in bpy.data.cameras:
        bpy.data.cameras.remove(block)
    for block in bpy.data.lights:
        bpy.data.lights.remove(block)


def import_model(model_path: str) -> None:
    import bpy
    path = str(Path(model_path).resolve())
    ext  = Path(model_path).suffix.lower()
    if ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext in (".obj", ".obj"):
        # Pass use_split_objects=False to avoid creating one object per MTL group.
        # If no .mtl is present alongside the .obj, Blender imports geometry only — no error.
        if bpy.app.version[0] >= 4:
            bpy.ops.wm.obj_import(filepath=path, forward_axis='Y', up_axis='Z')
        else:
            bpy.ops.import_scene.obj(filepath=path, axis_forward='Y', axis_up='Z')
    elif ext in (".gltf", ".glb"):
        bpy.ops.import_scene.gltf(filepath=path)
    else:
        raise ValueError(f"Unsupported model format: {ext}")


def setup_skybox(hdr_path: str) -> None:
    import bpy
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()

    bg  = nt.nodes.new("ShaderNodeBackground")
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    out = nt.nodes.new("ShaderNodeOutputWorld")

    env.image = bpy.data.images.load(str(Path(hdr_path).resolve()))
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])


def setup_black_world() -> None:
    import bpy
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = False
    world.color = (0.0, 0.0, 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# PBR material override
# ──────────────────────────────────────────────────────────────────────────────

def build_material(args: argparse.Namespace):
    """Return a new Principled BSDF material, or None to keep the OBJ's own materials."""
    import bpy

    has_any = any([
        args.mat_color, args.mat_albedo, args.mat_normal,
        args.mat_roughness, args.mat_metallic, args.mat_emission, args.mat_alpha,
    ])
    if not has_any:
        return None

    mat = bpy.data.materials.new("OverrideMaterial")
    mat.use_nodes = True
    nt   = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    def tex(path: str):
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = bpy.data.images.load(str(Path(path).resolve()))
        return n

    # Base Color
    if args.mat_albedo:
        t = tex(args.mat_albedo)
        nt.links.new(t.outputs["Color"], bsdf.inputs["Base Color"])
    elif args.mat_color:
        vals = list(args.mat_color) + [1.0]  # pad alpha if omitted
        r, g, b, a = vals[0], vals[1], vals[2], vals[3]
        bsdf.inputs["Base Color"].default_value = (r, g, b, a)

    # Normal map  (TexImage → NormalMap node → Normal)
    if args.mat_normal:
        t = tex(args.mat_normal)
        t.image.colorspace_settings.name = "Non-Color"
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nt.links.new(t.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])

    # Roughness: scalar or texture
    if args.mat_roughness is not None:
        try:
            bsdf.inputs["Roughness"].default_value = float(args.mat_roughness)
        except ValueError:
            t = tex(args.mat_roughness)
            t.image.colorspace_settings.name = "Non-Color"
            nt.links.new(t.outputs["Color"], bsdf.inputs["Roughness"])

    # Metallic: scalar or texture
    if args.mat_metallic is not None:
        try:
            bsdf.inputs["Metallic"].default_value = float(args.mat_metallic)
        except ValueError:
            t = tex(args.mat_metallic)
            t.image.colorspace_settings.name = "Non-Color"
            nt.links.new(t.outputs["Color"], bsdf.inputs["Metallic"])

    # Emission
    if args.mat_emission:
        t = tex(args.mat_emission)
        nt.links.new(t.outputs["Color"], bsdf.inputs["Emission Color"])

    # Alpha / opacity
    if args.mat_alpha:
        t = tex(args.mat_alpha)
        t.image.colorspace_settings.name = "Non-Color"
        nt.links.new(t.outputs["Color"], bsdf.inputs["Alpha"])
        mat.blend_method = "BLEND"

    return mat


def assign_material(mat) -> None:
    import bpy
    if mat is None:
        return
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            obj.data.materials.clear()
            obj.data.materials.append(mat)


# ──────────────────────────────────────────────────────────────────────────────
# Camera
# ──────────────────────────────────────────────────────────────────────────────

def create_camera(intr: CameraIntrinsics):
    """Create and configure a Blender camera with the given intrinsics."""
    import bpy

    cam_data = bpy.data.cameras.new("RenderCamera")
    cam_obj  = bpy.data.objects.new("RenderCamera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)

    cam_data.type        = "PERSP"
    cam_data.sensor_fit  = "HORIZONTAL"
    cam_data.angle       = intr.camera_angle_x          # exact horizontal FOV
    # Derive sensor_height so that vertical FOV also matches exactly:
    #   tan(fov_y/2) = tan(angle/2) * sensor_h/sensor_w
    #                = (w/(2*fl_x)) * (h*fl_x)/(w*fl_y)
    #                = h/(2*fl_y)  = tan(camera_angle_y/2)  ✓
    cam_data.sensor_width  = 36.0
    cam_data.sensor_height = 36.0 * (intr.h * intr.fl_x) / (intr.w * intr.fl_y)
    cam_data.clip_start  = 0.001
    cam_data.clip_end    = 1000.0
    # Principal-point shift (handles off-center sensors)
    cam_data.shift_x =  (intr.cx - intr.w / 2.0) / intr.w
    cam_data.shift_y = -(intr.cy - intr.h / 2.0) / intr.h  # Y axis flipped in Blender

    return cam_obj


def set_camera_pose(cam_obj, transform_matrix: list[list[float]], apply_axis_correction: bool) -> None:
    """Set camera world matrix from a NeRF-format 4×4 c2w matrix."""
    from mathutils import Matrix

    c2w = Matrix(transform_matrix)

    if apply_axis_correction:
        # Rotate world axes: NeRF Y-up → Blender Z-up
        # Maps NeRF(X, Y, Z) → Blender(X, Z, -Y)
        R = Matrix([
            [1,  0,  0,  0],
            [0,  0, -1,  0],
            [0,  1,  0,  0],
            [0,  0,  0,  1],
        ])
        c2w = R @ c2w

    cam_obj.matrix_world = c2w


# ──────────────────────────────────────────────────────────────────────────────
# GPU setup
# ──────────────────────────────────────────────────────────────────────────────

def enable_gpu_rendering() -> None:
    import bpy
    prefs = bpy.context.preferences.addons["cycles"].preferences
    # Try OPTIX (NVIDIA RT cores) first, then CUDA, HIP (AMD), METAL (Apple)
    for device_type in ("OPTIX", "CUDA", "HIP", "METAL"):
        prefs.compute_device_type = device_type
        prefs.refresh_devices()
        gpu_devices = [d for d in prefs.devices if d.type != "CPU"]
        if gpu_devices:
            for d in prefs.devices:
                d.use = (d.type != "CPU")   # enable all GPU devices, disable CPU
            bpy.context.scene.cycles.device = "GPU"
            print(f"GPU rendering enabled ({device_type}): "
                  f"{', '.join(d.name for d in gpu_devices)}")
            return
    print("WARNING: No GPU device found, rendering on CPU.")


# ──────────────────────────────────────────────────────────────────────────────
# Render configuration
# ──────────────────────────────────────────────────────────────────────────────

def configure_render(scene, intr: CameraIntrinsics, output_format: str,
                     engine: str, samples: int) -> None:
    import bpy

    scene.render.resolution_x          = intr.w
    scene.render.resolution_y          = intr.h
    scene.render.resolution_percentage = 100

    if engine == "CYCLES":
        scene.render.engine  = "CYCLES"
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
    else:
        scene.render.engine = (
            "BLENDER_EEVEE_NEXT" if bpy.app.version[0] >= 4 else "BLENDER_EEVEE"
        )

    fmt = scene.render.image_settings
    if output_format == "PNG":
        fmt.file_format  = "PNG"
        fmt.color_mode   = "RGBA"
        fmt.color_depth  = "8"
    elif output_format == "EXR":
        fmt.file_format  = "OPEN_EXR"
        fmt.color_mode   = "RGBA"
        fmt.color_depth  = "32"
        fmt.exr_codec    = "ZIP"
    elif output_format == "JPEG":
        fmt.file_format  = "JPEG"
        fmt.color_mode   = "RGB"
        fmt.quality      = 95


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import bpy

    args = parse_args()

    # ── Output subdirectories ──────────────────────────────────────────────────
    out       = Path(args.output_dir)
    renders_dir  = out / "renders"
    scene_dir    = out / "scene"
    textures_dir = out / "textures"
    skybox_dir   = out / "skybox"
    for d in (renders_dir, scene_dir, textures_dir, skybox_dir):
        os.makedirs(d, exist_ok=True)

    # ── Load transforms ───────────────────────────────────────────────────────
    tf = load_transforms(args.transforms)
    intr = tf.intrinsics
    print(f"Loaded {len(tf.frames)} frames from {args.transforms}")

    # ── Scene setup ───────────────────────────────────────────────────────────
    clear_scene()
    import_model(args.model)

    mat = build_material(args)
    assign_material(mat)

    if args.skybox:
        setup_skybox(args.skybox)
    else:
        setup_black_world()

    # ── Camera + render settings ──────────────────────────────────────────────
    cam_obj = create_camera(intr)
    bpy.context.scene.camera = cam_obj
    configure_render(bpy.context.scene, intr, args.output_format, args.engine, args.samples)

    if args.engine == "CYCLES" and args.device == "GPU":
        enable_gpu_rendering()

    # ── Render loop ───────────────────────────────────────────────────────────
    for idx, frame in enumerate(tf.frames):
        print(f"[{idx + 1}/{len(tf.frames)}] Rendering {frame.stem} ...")
        set_camera_pose(cam_obj, frame.transform_matrix, args.axis_correction)
        bpy.context.view_layer.update()
        bpy.context.scene.render.filepath = str(renders_dir / frame.stem)
        bpy.ops.render.render(write_still=True)

    print(f"Rendered {len(tf.frames)} frames → {renders_dir}")

    # ── Copy asset files ──────────────────────────────────────────────────────
    tex_attrs = ("mat_albedo", "mat_normal", "mat_roughness",
                 "mat_metallic", "mat_emission", "mat_alpha")
    for attr in tex_attrs:
        src = getattr(args, attr, None)
        if src and Path(src).is_file():
            shutil.copy2(src, textures_dir / Path(src).name)

    if args.skybox and Path(args.skybox).is_file():
        shutil.copy2(args.skybox, skybox_dir / Path(args.skybox).name)

    # ── Save .blend with all textures packed in ───────────────────────────────
    try:
        bpy.ops.file.pack_all()
    except RuntimeError as e:
        # Some images referenced by the model's MTL/material may point to
        # missing paths (e.g. old absolute paths in an imported OBJ's .mtl).
        # pack_all() still packs whatever it can find; log and continue.
        print(f"WARNING: pack_all() reported missing files (non-fatal): {e}")
    blend_path = str(scene_dir / "scene.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path, check_existing=False)
    print(f"Scene saved → {blend_path}")


if __name__ == "__main__":
    main()
