"""
bake_unified_material.py
------------------------
Script to be run INSIDE Blender (Text Editor -> Run Script, Blender 4.2+ LTS)
on the currently open scene.

What it does:
    1. duplicates every MESH object of the scene (the original is left untouched),
    2. joins them into a single mesh with a new UV layer "BakeUV" (Smart UV Project),
    3. bakes every material into a single set of PBR textures
       (base_color / metallic / roughness / normal), with resolution and format
       configurable per channel,
    4. builds a single Principled BSDF material that uses those textures,
    5. assembles everything into a new scene (default: "Baked").

Usage (three equivalent modes, same `BakeConfig` + `run` pipeline):
    A) Panel (recommended): open the file in the Text Editor and press "Run Script".
       A "Bake" panel appears in the sidebar of the 3D Viewport (N key), where every
       parameter can be set, `output_dir` included, and the bake started with the
       "Bake Unified Material" button (the values stay saved in the .blend, so they
       do not have to be set again at every Run Script).
    B) Add-on: installable from Preferences -> Add-ons -> Install... (see `bl_info`).
    C) From code: `baked, new_scene = run(BakeConfig(output_dir=..., ...))`,
       for automated pipelines. To save the .blend as well:
           `_save_blend(cfg, new_scene)`
       (when used from the Text Editor, not nested inside an operator, it is safe to
       do it right after run(); the UI operator instead does it via a deferred timer).
"""

bl_info = {
    "name": "Bake Unified Material",
    "author": "Adriano Cicco",
    "version": (1, 1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar (N) > Bake",
    "description": "Joins the meshes of the scene, bakes the materials into PBR "
                   "textures (base_color/metallic/roughness/normal) and assembles a "
                   "new scene with a single unified material",
    "category": "Material",
}

# NOTE: no `from __future__ import annotations` here. The PropertyGroup/Operator/Panel
# classes below declare their properties as class annotations
# (`name: bpy.props.XProperty(...)`): Blender reads them from `__annotations__` at
# definition time and expects the real `bpy.props` objects, not the "lazy" strings
# produced by PEP 563; with that import the registration would fail. The references to
# `bpy.types.*` in the type hints of the rest of the file stay valid anyway: inside
# Blender they are runtime attributes already available when the script runs (no
# deferred evaluation needed).

import math
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import bpy

# ──────────────────────────────────────────────────────────────────────────────
# Texture formats
# ──────────────────────────────────────────────────────────────────────────────

class TextureFormat(Enum):
    OPEN_EXR = "OPEN_EXR"
    PNG = "PNG"

    @property
    def file_format(self) -> str:
        """Value to assign to `image_settings.file_format` / `image.file_format`."""
        return self.value

    @property
    def extension(self) -> str:
        return {"OPEN_EXR": ".exr", "PNG": ".png"}[self.value]

    @property
    def use_float_buffer(self) -> bool:
        """EXR is a float format: the image has to be created with a 32-bit float buffer."""
        return self is TextureFormat.OPEN_EXR

    def default_color_depth(self) -> str:
        """Default colour depth when not specified in `BakeConfig`."""
        return "16" if self is TextureFormat.OPEN_EXR else "8"


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_CHANNELS = {"base_color": True, "metallic": True, "roughness": True, "normal": True}
_SWIZZLE_AXES = ('POS_X', 'NEG_X', 'POS_Y', 'NEG_Y', 'POS_Z', 'NEG_Z')
_DEFAULT_RESOLUTIONS = {
    "base_color": (4096, 4096),
    "metallic": (2048, 2048),
    "roughness": (2048, 2048),
    "normal": (4096, 4096),
}

# Bake order: base_color/metallic first (emission-swap, more delicate), then the
# native passes.
CHANNEL_ORDER: List[str] = ["base_color", "metallic", "roughness", "normal"]


@dataclass
class BakeConfig:
    output_dir: str

    new_scene_name: str = "Baked"
    merged_object_name: str = "BakedMesh"
    merged_material_name: str = "BakedMaterial"

    # Which channels to bake.
    channels: Dict[str, bool] = field(default_factory=lambda: dict(_DEFAULT_CHANNELS))
    # Resolution (width, height) per channel.
    resolutions: Dict[str, Tuple[int, int]] = field(default_factory=lambda: dict(_DEFAULT_RESOLUTIONS))

    # Coordinate space for the normal map: 'TANGENT' | 'OBJECT'.
    normal_space: str = "TANGENT"
    # Axis swizzle of the normal map (which axis/sign goes into R, G, B).
    # Valid values: 'POS_X' | 'NEG_X' | 'POS_Y' | 'NEG_Y' | 'POS_Z' | 'NEG_Z'.
    normal_r: str = "POS_X"
    normal_g: str = "POS_Y"
    normal_b: str = "POS_Z"

    # Default format/colour depth; optional per-channel overrides.
    tex_format: TextureFormat = TextureFormat.OPEN_EXR
    color_depth: Optional[str] = None  # None -> use the format's default
    format_overrides: Dict[str, TextureFormat] = field(default_factory=dict)
    color_depth_overrides: Dict[str, str] = field(default_factory=dict)

    # base_color: by default the emission-swap of the "Base Color" socket is used
    # (faithful on metals too). If False, the native DIFFUSE pass is used, filtering
    # the colour only (more "physical", but darkened on metals).
    base_color_via_emission: bool = True

    # Bake parameters.
    samples: int = 32
    bake_margin: int = 16

    # Smart UV Project parameters (angle_limit in degrees: converted internally).
    uv_island_margin: float = 0.02
    uv_angle_limit: float = 66.0

    apply_modifiers: bool = True
    # Apply rotation/scale/location to the copies before the join: needed to get
    # object-space normals in the world frame instead of the local one.
    apply_transform: bool = True
    device: str = "GPU"  # 'GPU' | 'CPU'

    # World / environment map.
    copy_world: bool = True  # copy the source scene's World into the new scene

    # Saving of the .blend file at the end of the bake.
    save_blend: bool = True
    blend_filename: Optional[str] = None  # None -> "<new_scene_name>.blend"

    def __post_init__(self) -> None:
        if self.normal_space not in ("TANGENT", "OBJECT"):
            raise ValueError(f"normal_space must be 'TANGENT' or 'OBJECT', not {self.normal_space!r}")
        if self.device not in ("GPU", "CPU"):
            raise ValueError(f"device must be 'GPU' or 'CPU', not {self.device!r}")
        for attr, val in (("normal_r", self.normal_r), ("normal_g", self.normal_g), ("normal_b", self.normal_b)):
            if val not in _SWIZZLE_AXES:
                raise ValueError(f"{attr} must be one of {_SWIZZLE_AXES}, not {val!r}")


def _format_for_channel(cfg: BakeConfig, channel: str) -> TextureFormat:
    return cfg.format_overrides.get(channel, cfg.tex_format)



def _colorspace_for_channel(channel: str, fmt: TextureFormat) -> str:
    """metallic/roughness/normal are non-colour data -> always 'Non-Color'.
    base_color is colour data: 'sRGB' on 8-bit formats (PNG); on EXR
    (float buffer) 'Non-Color' is used as the practical equivalent of "Linear":
    it avoids double gamma conversions and dependencies on the colorspace names of the
    active OCIO config, which vary between Blender versions/themes."""
    if channel == "base_color" and fmt is not TextureFormat.OPEN_EXR:
        return "sRGB"
    return "Non-Color"


# ──────────────────────────────────────────────────────────────────────────────
# Helper: shader nodes
# ──────────────────────────────────────────────────────────────────────────────

def _get_principled(mat: bpy.types.Material) -> Optional[bpy.types.ShaderNodeBsdfPrincipled]:
    """Find a material's Principled BSDF, looking inside node groups too."""
    if mat is None or mat.node_tree is None:
        return None
    return _find_principled_in_tree(mat.node_tree, set())


def _find_principled_in_tree(node_tree: bpy.types.ShaderNodeTree, visited: set) -> Optional[bpy.types.ShaderNodeBsdfPrincipled]:
    if node_tree in visited:
        return None
    visited.add(node_tree)
    for node in node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            return node
    for node in node_tree.nodes:
        if node.type == "GROUP" and node.node_tree is not None:
            found = _find_principled_in_tree(node.node_tree, visited)
            if found is not None:
                return found
    return None


def _get_active_output(node_tree: bpy.types.ShaderNodeTree) -> Optional[bpy.types.ShaderNodeOutputMaterial]:
    fallback = None
    for node in node_tree.nodes:
        if node.type == "OUTPUT_MATERIAL":
            if node.is_active_output:
                return node
            fallback = fallback or node
    return fallback


def _to_rgba(value) -> Tuple[float, float, float, float]:
    """Convert a socket's default_value (float or colour) into an RGBA."""
    if isinstance(value, (int, float)):
        return (float(value),) * 3 + (1.0,)
    seq = tuple(value)
    if len(seq) == 3:
        return seq + (1.0,)
    return seq[:4]


# ──────────────────────────────────────────────────────────────────────────────
# Helper: bake target images
# ──────────────────────────────────────────────────────────────────────────────

def _new_target_image(name: str, width: int, height: int, fmt: TextureFormat, colorspace: str) -> bpy.types.Image:
    image = bpy.data.images.new(
        name=name,
        width=width,
        height=height,
        alpha=False,
        float_buffer=fmt.use_float_buffer,
    )
    try:
        image.colorspace_settings.name = colorspace
    except TypeError:
        # The colorspace name may differ between OCIO versions/configs.
        pass
    return image


def _save_image(image: bpy.types.Image, path: str, fmt: TextureFormat) -> None:
    """Save `image` to disk in the requested format.

    Uses `image.save()` instead of `image.save_render()`: `save_render` applies the
    scene's active view transform (Filmic, AgX, ...) to the baked pixels, corrupting
    the raw data: non-linear tone-mapping curves change the direction of the normal
    vectors and alter the scalar values of roughness/metallic.
    `image.save()` writes the pixels as-is, with no colour transformation at all, and is
    the correct choice for every bake channel (non-display data).

    Note on the bit depth: `image.save()` uses the image's native buffer.
    Images created with `float_buffer=True` (EXR) are saved as 32-bit float; those
    created with `float_buffer=False` (PNG) as 8-bit. The `color_depth` parameter of
    `BakeConfig` has no effect on this saving path.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    image.filepath_raw = path
    image.file_format = fmt.file_format
    image.save()
    # Move from GENERATED to FILE: that way, when the .blend is reopened, Blender
    # loads the image from the file on disk instead of recreating it empty.
    image.source = 'FILE'
    image.reload()


# ──────────────────────────────────────────────────────────────────────────────
# Helper: temporary Image Texture node (bake target) in the materials
# ──────────────────────────────────────────────────────────────────────────────

def _setup_bake_nodes(materials: List[bpy.types.Material], image: bpy.types.Image):
    """Insert into every material an Image Texture node pointing at `image`, select it
    and make it active: it is the node Cycles will bake onto."""
    created = []
    for mat in materials:
        if mat is None or mat.node_tree is None:
            continue
        if not mat.use_nodes:
            mat.use_nodes = True
        nodes = mat.node_tree.nodes
        for node in nodes:
            node.select = False
        tex_node = nodes.new("ShaderNodeTexImage")
        tex_node.image = image
        tex_node.label = "_BAKE_TARGET_"
        tex_node.select = True
        nodes.active = tex_node
        created.append((mat, tex_node))
    return created


def _cleanup_bake_nodes(created) -> None:
    for mat, node in created:
        try:
            mat.node_tree.nodes.remove(node)
        except (ReferenceError, RuntimeError):
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Bake: native passes (DIFFUSE / ROUGHNESS / NORMAL)
# ──────────────────────────────────────────────────────────────────────────────

def _bake_native(bake_type: str, **kwargs) -> bool:
    """Run a native bake pass. Updates the depsgraph before starting the bake and
    returns True if the operator completed successfully."""
    bpy.context.view_layer.update()
    try:
        result = bpy.ops.object.bake(type=bake_type, **kwargs)
        return 'FINISHED' in result
    except RuntimeError as exc:
        print(f"bake_unified_material: WARNING - native bake {bake_type} "
              f"failed: {exc}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Bake: emission-swap (for channels with no native pass, e.g. Metallic, and for
# Base Color when `base_color_via_emission=True`)
# ──────────────────────────────────────────────────────────────────────────────

def _bake_emission(materials: List[bpy.types.Material], socket_name: str,
                   bake_nodes: list) -> bool:
    """Bake the value of the `socket_name` socket of every material's Principled BSDF,
    temporarily rerouting it through an Emission node and using `bake(type='EMIT')`.
    Always restores the original node trees (errors included), thanks to the
    `try/finally`.

    Materials with no Principled BSDF or without the requested socket are skipped:
    their UV islands stay at the clear colour of the target image.

    `bake_nodes` is the (mat, tex_node) list returned by `_setup_bake_nodes`: it is
    used to re-assert the Image Texture node as the active node *after* creating the
    Emission node (which would steal the "active" state otherwise).

    Returns True if the bake completed successfully."""
    setups = []
    bake_success = False
    try:
        for mat in materials:
            principled = _get_principled(mat)
            if principled is None or socket_name not in principled.inputs:
                print(f"bake_unified_material: skip '{mat.name if mat else None}' "
                      f"- no Principled BSDF with socket '{socket_name}'.")
                continue

            node_tree = mat.node_tree
            nodes, links = node_tree.nodes, node_tree.links

            output_node = _get_active_output(node_tree)
            if output_node is None or "Surface" not in output_node.inputs:
                print(f"bake_unified_material: skip '{mat.name}' "
                      f"- no active Output Material node.")
                continue
            surface_input = output_node.inputs["Surface"]

            original_link = (surface_input.links[0].from_socket
                             if surface_input.is_linked else None)

            emission = nodes.new("ShaderNodeEmission")
            emission.label = "_BAKE_EMISSION_"

            socket = principled.inputs[socket_name]
            if socket.is_linked:
                links.new(socket.links[0].from_socket, emission.inputs["Color"])
            else:
                emission.inputs["Color"].default_value = _to_rgba(socket.default_value)

            links.new(emission.outputs["Emission"], surface_input)
            setups.append((node_tree, surface_input, original_link, emission))

        if not setups:
            print(f"bake_unified_material: WARNING - no material can be configured "
                  f"for the emission-swap of '{socket_name}': the channel will be black.")

        # Re-assert the Image Texture node as the active node: nodes.new() above stole
        # the "active" state and gave it to the Emission node; without this step the
        # EMIT bake has no valid target and the channel comes out black.
        for mat, tex_node in bake_nodes:
            if mat is None or mat.node_tree is None:
                continue
            mat.node_tree.nodes.active = tex_node
            tex_node.select = True

        # Sync the depsgraph after the surgery on the nodes.
        bpy.context.view_layer.update()

        try:
            result = bpy.ops.object.bake(type="EMIT")
            bake_success = 'FINISHED' in result
        except RuntimeError as exc:
            print(f"bake_unified_material: WARNING - EMIT bake ('{socket_name}') "
                  f"failed: {exc}")
            bake_success = False
    finally:
        for node_tree, surface_input, original_link, emission in setups:
            links = node_tree.links
            for link in list(surface_input.links):
                links.remove(link)
            if original_link is not None:
                links.new(original_link, surface_input)
            node_tree.nodes.remove(emission)
    return bake_success


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline: setup, duplication, join, UV, bake, material, scene
# ──────────────────────────────────────────────────────────────────────────────

# Preference order for the Cycles GPU device type.
_GPU_DEVICE_TYPE_PRIORITY = ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI")


def _enable_gpu_device() -> bool:
    """Enable the first available GPU device type in the Cycles preferences.

    Blender requires `compute_device_type` to be set and at least one GPU device to be
    enabled (`device.use = True`) before starting a bake on the GPU: without this step
    the *shaded* passes (EMIT/DIFFUSE/ROUGHNESS) can come back zero non-
    deterministically while the NORMAL pass (purely geometric) writes anyway, producing
    the "everything black except the normal" symptom.

    Iterates the types in order of preference (`_GPU_DEVICE_TYPE_PRIORITY`), tries each
    one and stops at the first for which `get_devices()` returns at least one non-CPU
    device. Returns True if at least one GPU device was enabled, False if no GPU device
    is available (the caller then falls back to the CPU)."""
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except (KeyError, AttributeError):
        return False

    for device_type in _GPU_DEVICE_TYPE_PRIORITY:
        try:
            prefs.compute_device_type = device_type
            prefs.get_devices()
        except (TypeError, AttributeError):
            continue
        gpu_devices = [d for d in prefs.devices if d.type != "CPU"]
        if not gpu_devices:
            continue
        for d in gpu_devices:
            d.use = True
        enabled = sum(1 for d in gpu_devices if d.use)
        print(f"bake_unified_material: GPU device enabled "
              f"({device_type}, {enabled} active devices).")
        return True

    return False


def _setup_render_settings(scene: bpy.types.Scene, cfg: BakeConfig) -> None:
    scene.render.engine = "CYCLES"
    scene.cycles.samples = cfg.samples
    scene.render.bake.margin = cfg.bake_margin
    scene.render.bake.use_clear = True
    scene.render.bake.use_selected_to_active = False
    if cfg.device == "GPU":
        if _enable_gpu_device():
            scene.cycles.device = "GPU"
        else:
            print("bake_unified_material: no GPU device available, "
                  "falling back to the CPU.")
            scene.cycles.device = "CPU"
    else:
        scene.cycles.device = "CPU"


def _make_materials_single_user(objects: List[bpy.types.Object]) -> None:
    """Make the duplicates' materials single-user, breaking the sharing with the
    originals.

    `bpy.ops.object.duplicate(linked=False)` copies the object and the mesh data, but
    the materials stay SHARED by default (Blender does not duplicate material
    datablocks). The bake performs surgery on the nodes (adds an Image Texture, swaps
    the Surface socket for an emission): without this function those changes happen on
    the ORIGINALS' materials, even with the restore in `finally`, leaving the node
    trees in a state that crashes at the first re-evaluation of the depsgraph.

    Duplicates that shared the same material get the same copy (dict orig -> copy): the
    per-datablock dedup in `_bake_all` and `_pick_reference_material` stays valid. The
    originals are NEVER touched by the bake."""
    mat_copies: Dict[bpy.types.Material, bpy.types.Material] = {}
    for obj in objects:
        for slot in obj.material_slots:
            orig = slot.material
            if orig is None:
                continue
            if orig not in mat_copies:
                mat_copies[orig] = orig.copy()
            slot.material = mat_copies[orig]
    if mat_copies:
        print(f"bake_unified_material: {len(mat_copies)} material(s) made "
              f"single-user - the original materials will not be modified by the bake.")


def _duplicate_objects(source_objects: List[bpy.types.Object], view_layer: bpy.types.ViewLayer, cfg: BakeConfig) -> List[bpy.types.Object]:
    """Duplicate the source objects (the original scene is left untouched) and, if
    requested, apply the modifiers on the copies via `object.convert`."""
    bpy.ops.object.select_all(action="DESELECT")
    for obj in source_objects:
        obj.select_set(True)
    view_layer.objects.active = source_objects[0]

    bpy.ops.object.duplicate(linked=False)
    duplicates = list(bpy.context.selected_objects)

    if cfg.apply_modifiers:
        bpy.ops.object.select_all(action="DESELECT")
        for obj in duplicates:
            obj.select_set(True)
        view_layer.objects.active = duplicates[0]
        bpy.ops.object.convert(target="MESH", keep_original=False)

    # Break the material sharing with the originals: the bake performs surgery on the
    # COPIES' nodes, the originals stay bit-for-bit intact.
    _make_materials_single_user(duplicates)

    return duplicates


def _apply_transforms(duplicates: List[bpy.types.Object], view_layer: bpy.types.ViewLayer) -> None:
    """Apply rotation/scale/location to every copy.

    Normals in OBJECT space are expressed in the object's local frame: if an object has
    an unapplied rotation (e.g. 180 degrees on Z), the normals come out rotated the
    same way. Applying the transforms aligns object space to the world, so the bake
    returns predictable normals regardless of the pose of the objects in the scene.
    The copies are single-user (`linked=False`), so transform_apply does not touch the
    original meshes."""
    bpy.ops.object.select_all(action="DESELECT")
    for obj in duplicates:
        obj.select_set(True)
    view_layer.objects.active = duplicates[0]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def _join_objects(duplicates: List[bpy.types.Object], view_layer: bpy.types.ViewLayer, cfg: BakeConfig) -> bpy.types.Object:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in duplicates:
        obj.select_set(True)
    merged = duplicates[0]
    view_layer.objects.active = merged

    if len(duplicates) > 1:
        bpy.ops.object.join()
        merged = view_layer.objects.active

    merged.name = cfg.merged_object_name
    merged.data.name = cfg.merged_object_name
    return merged


def _ensure_default_material(obj: bpy.types.Object) -> None:
    """Guarantee every material slot has a material: objects/slots with no material get
    a neutral default Principled BSDF, so their faces also take part in the bake
    correctly."""
    if not obj.data.materials:
        obj.data.materials.append(_get_or_create_default_material())
        return
    for i, mat in enumerate(obj.data.materials):
        if mat is None:
            obj.data.materials[i] = _get_or_create_default_material()


def _get_or_create_default_material() -> bpy.types.Material:
    name = "_BakeDefaultMaterial_"
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
    return mat


UV_LAYER_NAME = "BakeUV"


def _prepare_bake_uv_per_object(obj: bpy.types.Object, view_layer: bpy.types.ViewLayer, cfg: BakeConfig) -> None:
    """Prepare the BakeUV layer on a single duplicated object, before the join.

    - If the object already has UVs: copy the active UV into BakeUV (preserves the
      original projection, no new cuts/islands).
    - If the object has no UVs (a primitive): create BakeUV via Smart UV Project on the
      single object (much more controllable than on the whole joined mesh).

    In both cases BakeUV is set as active and active_render so that, after the join, the
    BakeUV layer of the joined mesh already holds the right data for each object and
    only a final Pack Islands is needed.
    """
    mesh = obj.data

    if mesh.uv_layers:
        # Object with UVs: copy the active UV into BakeUV.
        # active_render stays on the original UV -> the material's textures are sampled
        # with the original UV during the bake.
        # active becomes BakeUV -> the baked pixels are WRITTEN onto BakeUV.
        src = mesh.uv_layers.active
        src.active_render = True  # explicit: keep the original for sampling
        dst = mesh.uv_layers.new(name=UV_LAYER_NAME)
        for s, d in zip(src.data, dst.data):
            d.uv = s.uv[:]
        mesh.uv_layers.active = dst
        # dst.active_render stays False: BakeUV is only the write target
    else:
        # Primitive with no UVs: Smart UV Project on the single object.
        # BakeUV is the only layer available -> it is both active and active_render.
        dst = mesh.uv_layers.new(name=UV_LAYER_NAME)
        mesh.uv_layers.active = dst
        dst.active_render = True

        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        view_layer.objects.active = obj

        bpy.ops.object.mode_set(mode="EDIT")
        try:
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.uv.smart_project(
                island_margin=cfg.uv_island_margin,
                angle_limit=math.radians(cfg.uv_angle_limit),
            )
        finally:
            bpy.ops.object.mode_set(mode="OBJECT")


def _pack_bake_uv(obj: bpy.types.Object, view_layer: bpy.types.ViewLayer, cfg: BakeConfig) -> str:
    """After the join, run Pack Islands on BakeUV so that every island fits in [0,1]
    without overlaps, preserving the shape of each island."""
    mesh = obj.data
    if UV_LAYER_NAME in mesh.uv_layers:
        bake_uv = mesh.uv_layers[UV_LAYER_NAME]
        mesh.uv_layers.active = bake_uv  # the bake's WRITE target
        # active_render has to stay on the source layer set by
        # _prepare_bake_uv_per_object (through src.active_render = True).
        # The fallback is forced only if BakeUV is inadvertently active_render
        # (the join can reset/shuffle its state) or if no other layer is:
        # in both cases the first non-BakeUV layer available is chosen.
        needs_render_uv = bake_uv.active_render or not any(
            uv.active_render for uv in mesh.uv_layers if uv.name != UV_LAYER_NAME
        )
        if needs_render_uv:
            for uv in mesh.uv_layers:
                if uv.name != UV_LAYER_NAME:
                    uv.active_render = True  # exclusive: clears bake_uv automatically
                    break

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    view_layer.objects.active = obj

    bpy.ops.object.mode_set(mode="EDIT")
    try:
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.pack_islands(margin=cfg.uv_island_margin, rotate=False)
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")

    return UV_LAYER_NAME


def _do_bake_with_retry(bake_fn, channel: str) -> bool:
    """Run `bake_fn()` with a single automatic retry if the bake fails.

    The random en-bloc failure (every shaded channel black at once) is typically caused
    by an unsynchronised depsgraph or by a GPU device init not yet complete at the time
    of the first bake. A single retry, after a forced `view_layer.update()`, is enough
    to recover in these cases.

    Returns True if the bake succeeded (on the first attempt or on the retry)."""
    success = bake_fn()
    if not success:
        print(f"bake_unified_material: bake '{channel}' failed, retrying...")
        bpy.context.view_layer.update()
        success = bake_fn()
        if not success:
            print(f"bake_unified_material: WARNING - bake '{channel}' failed "
                  f"even after the retry. The texture may be black.")
    return success


def _bake_all(obj: bpy.types.Object, cfg: BakeConfig) -> Dict[str, Tuple[bpy.types.Image, str]]:
    """Bake every enabled channel and save it to disk.
    Returns {channel: (image, path)}."""
    # Deduplicated: after the join several slots can point at the same material
    # (different objects that shared a material). Processing it twice would create
    # redundant Image Texture/Emission nodes in the same node tree and would break the
    # restore in `_bake_emission`.
    seen = set()
    materials: List[bpy.types.Material] = []
    for slot in obj.material_slots:
        mat = slot.material
        if mat is not None and mat not in seen:
            seen.add(mat)
            materials.append(mat)
    if not materials:
        raise RuntimeError("The joined object has no materials to bake.")

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    baked: Dict[str, Tuple[bpy.types.Image, str]] = {}

    for channel in CHANNEL_ORDER:
        if not cfg.channels.get(channel, False):
            continue

        width, height = cfg.resolutions.get(channel, (2048, 2048))
        fmt = _format_for_channel(cfg, channel)
        colorspace = _colorspace_for_channel(channel, fmt)
        image_name = f"{cfg.merged_material_name}_{channel}"
        image = _new_target_image(image_name, width, height, fmt, colorspace)

        bake_nodes = _setup_bake_nodes(materials, image)
        try:
            if channel == "base_color":
                if cfg.base_color_via_emission:
                    _do_bake_with_retry(
                        lambda: _bake_emission(materials, "Base Color", bake_nodes),
                        channel,
                    )
                else:
                    _do_bake_with_retry(
                        lambda: _bake_native("DIFFUSE", pass_filter={"COLOR"}),
                        channel,
                    )
            elif channel == "roughness":
                _do_bake_with_retry(
                    lambda: _bake_native("ROUGHNESS"),
                    channel,
                )
            elif channel == "metallic":
                _do_bake_with_retry(
                    lambda: _bake_emission(materials, "Metallic", bake_nodes),
                    channel,
                )
            elif channel == "normal":
                # normal_r/g/b have to be passed DIRECTLY to the operator as kwargs:
                # setting them on scene.render.bake is ignored by bpy.ops.object.bake
                # run from a script (the operator uses its own defaults for the missing kwargs).
                _do_bake_with_retry(
                    lambda: _bake_native(
                        "NORMAL",
                        normal_space=cfg.normal_space,
                        normal_r=cfg.normal_r,
                        normal_g=cfg.normal_g,
                        normal_b=cfg.normal_b,
                    ),
                    channel,
                )
        finally:
            _cleanup_bake_nodes(bake_nodes)

        path = os.path.join(cfg.output_dir, f"{image_name}{fmt.extension}")
        _save_image(image, path, fmt)
        # reload() in _save_image reads the colorspace from the EXR file (lin_rec709_scene),
        # overwriting the correct one (Non-Color for roughness/metallic/normal).
        # Restoring it right away means the Image Texture node in the baked material
        # does not apply colour transforms to the data channels.
        try:
            image.colorspace_settings.name = colorspace
        except TypeError:
            # The colorspace name may differ between OCIO versions/configs.
            pass
        baked[channel] = (image, path)

    return baked


_CHANNEL_SOCKET = {"base_color": "Base Color", "metallic": "Metallic", "roughness": "Roughness"}
_CHANNEL_Y = {"base_color": 300, "metallic": 100, "roughness": -100, "normal": -300}

_DEFAULT_MAT_NAME = "_BakeDefaultMaterial_"


def _pick_reference_material(obj: bpy.types.Object) -> Optional[bpy.types.Material]:
    """Return the source material assigned to the largest number of faces of `obj`.

    After the join several slots can point at the same datablock: the count is per
    datablock, not per slot index. The neutral default material
    `_BakeDefaultMaterial_` (inserted by `_ensure_default_material` for the faces
    with no material) is excluded: it carries no meaningful settings.

    Returns `None` if there are no real materials or if the object has no faces.
    """
    slots = obj.material_slots
    if not slots:
        return None

    face_count: Dict[bpy.types.Material, int] = {}
    for poly in obj.data.polygons:
        idx = poly.material_index
        if idx >= len(slots):
            continue
        mat = slots[idx].material
        if mat is None or mat.name == _DEFAULT_MAT_NAME:
            continue
        face_count[mat] = face_count.get(mat, 0) + 1

    if not face_count:
        return None
    return max(face_count, key=lambda m: face_count[m])


def _copy_material_settings(src: bpy.types.Material, dst: bpy.types.Material) -> None:
    """Copy a curated set of datablock settings from `src` to `dst`.

    Uses hasattr + try/except on every attribute so it is robust across Blender
    versions (4.2 EEVEE-Next vs earlier versions, where some fields change name or
    disappear). The chosen fields affect either the Cycles render or the consistency of
    the viewport display:

    Shadows & culling (they affect shadows and shading):
        use_backface_culling, use_backface_culling_shadow, use_transparent_shadow

    Displacement (rendered geometry/bump):
        displacement_method  (Blender 4.1+, top-level)
        cycles.displacement_method  (earlier versions)

    Transparency / render method:
        surface_render_method, use_raytrace_refraction, refraction_depth  (4.2+)
        blend_method, shadow_method, use_screen_refraction, alpha_threshold (pre-4.2)

    Render passes:
        pass_index

    Viewport Display (consistency in the material list / solid mode):
        diffuse_color, metallic, roughness, line_color, line_priority,
        show_transparent_back

    Cycles volume/emission sampling:
        cycles.emission_sampling / cycles.sample_as_light
        cycles.homogeneous_volume, cycles.volume_sampling,
        cycles.volume_interpolation, cycles.volume_step_rate
    """
    # Flat (top-level) attributes
    _FLAT_ATTRS = (
        # shadows & culling
        "use_backface_culling",
        "use_backface_culling_shadow",
        "use_transparent_shadow",
        # displacement (Blender 4.1+)
        "displacement_method",
        # transparency / render method (4.2 EEVEE-Next)
        "surface_render_method",
        "use_raytrace_refraction",
        "refraction_depth",
        # transparency / render method (pre-4.2)
        "blend_method",
        "shadow_method",
        "use_screen_refraction",
        "alpha_threshold",
        # render passes
        "pass_index",
    )
    for attr in _FLAT_ATTRS:
        if hasattr(src, attr) and hasattr(dst, attr):
            try:
                setattr(dst, attr, getattr(src, attr))
            except (AttributeError, TypeError):
                pass

    # Viewport Display (sub-object)
    _DISPLAY_ATTRS = (
        "diffuse_color",
        "metallic",
        "roughness",
        "line_color",
        "line_priority",
        "show_transparent_back",
    )
    src_vd = getattr(src, "diffuse_color", None)  # proxy: if it does not exist, skip everything
    if src_vd is not None:
        for attr in _DISPLAY_ATTRS:
            if hasattr(src, attr) and hasattr(dst, attr):
                try:
                    setattr(dst, attr, getattr(src, attr))
                except (AttributeError, TypeError):
                    pass

    # Cycles (cycles sub-object)
    _CYCLES_ATTRS = (
        # displacement (versions earlier than 4.1)
        "displacement_method",
        # emission sampling
        "emission_sampling",
        "sample_as_light",
        # volume
        "homogeneous_volume",
        "volume_sampling",
        "volume_interpolation",
        "volume_step_rate",
    )
    src_cy = getattr(src, "cycles", None)
    dst_cy = getattr(dst, "cycles", None)
    if src_cy is not None and dst_cy is not None:
        for attr in _CYCLES_ATTRS:
            if hasattr(src_cy, attr) and hasattr(dst_cy, attr):
                try:
                    setattr(dst_cy, attr, getattr(src_cy, attr))
                except (AttributeError, TypeError):
                    pass


def _build_material(cfg: BakeConfig, baked_images: Dict[str, Tuple[bpy.types.Image, str]], ref_material: Optional[bpy.types.Material] = None) -> bpy.types.Material:
    """Create the unified material: a Principled BSDF fed by the textures just baked
    (normal -> Normal Map node with the configured `space`)."""
    mat = bpy.data.materials.new(name=cfg.merged_material_name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (400, 0)
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    principled.location = (100, 0)
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])

    for channel, (image, _path) in baked_images.items():
        tex_node = nodes.new("ShaderNodeTexImage")
        tex_node.image = image
        tex_node.label = channel
        tex_node.location = (-600, _CHANNEL_Y.get(channel, 0))

        if channel == "normal":
            # The Cycles NORMAL pass writes values already encoded in [0,1]
            # (flat tangent = (0.5, 0.5, 1.0)), regardless of the bit depth or of the
            # format (PNG 8-bit, EXR float 32-bit). The x0.5+0.5 encoding happens in the
            # pass itself, NOT in the image's colorspace: behaviour consistent with the
            # Cycles documentation and with the public reference implementations
            # (addon_bake_groups, SimpleBake, ...).
            if cfg.normal_space == "OBJECT":
                # OBJECT space: the baked values are in [0,1] -> decode with
                # x2-1 to bring them back to [-1,1], then normalise.
                # With apply_transform on, object space = world space after the join:
                # do NOT use VectorTransform(Object->World), which would use BakedMesh's
                # matrix at render time (a residual rotation after the join/scene
                # transfer would swap axes). The direct link with decode+normalise is
                # correct and robust.
                normal_map = nodes.new("ShaderNodeNormalMap")
                normal_map.space = 'OBJECT'
                normal_map.location = (-300, _CHANNEL_Y[channel])
                links.new(tex_node.outputs["Color"], normal_map.inputs["Color"])
                links.new(normal_map.outputs["Normal"], principled.inputs["Normal"])
            else:
                # TANGENT space: link directly to the Normal Map node.
                # The node internally applies x2-1 to decode the baked [0,1] values ->
                # no extra remap is needed (adding one would re-encode data already in
                # [0,1], producing wrong normals).
                normal_map = nodes.new("ShaderNodeNormalMap")
                normal_map.space = 'TANGENT'
                normal_map.location = (-300, _CHANNEL_Y[channel])
                links.new(tex_node.outputs["Color"], normal_map.inputs["Color"])
                links.new(normal_map.outputs["Normal"], principled.inputs["Normal"])
        else:
            links.new(tex_node.outputs["Color"], principled.inputs[_CHANNEL_SOCKET[channel]])

    if ref_material is not None:
        _copy_material_settings(ref_material, mat)

    return mat


def _assemble_scene(cfg: BakeConfig, merged_obj: bpy.types.Object, material: bpy.types.Material, uv_layer_name: str, source_scene: bpy.types.Scene) -> bpy.types.Scene:
    """Move the joined object (it was a working duplicate) into a new scene, assign it
    the unified material as its ONLY material and activate the bake UV.
    If `cfg.copy_world` is True, shares the source scene's World (HDRI / environment
    map) with the new scene."""
    # Before the unlink: clear EVERY reference to merged_obj that survives in the
    # session after the migration into the new scene. A dangling pointer to an
    # unlinked object crashes at the first user interaction (click, scene switch,
    # opening the Properties editor, ...).

    # 1. View layer of the source scene: active object.
    for vl in source_scene.view_layers:
        if vl.objects.active is merged_obj:
            vl.objects.active = None
    merged_obj.select_set(False)

    # 2. Every open window: view_layer.active and pin_id in the Properties editors.
    for wm in bpy.data.window_managers:
        for win in wm.windows:
            # Active object for the window's view_layer (can differ from the scene's).
            try:
                if win.view_layer is not None and win.view_layer.objects.active is merged_obj:
                    win.view_layer.objects.active = None
            except (AttributeError, ReferenceError):
                pass
            # Pin in the Properties editors (Space Properties / Space Graph / ...).
            try:
                for area in win.screen.areas:
                    for space in area.spaces:
                        if getattr(space, "pin_id", None) is merged_obj:
                            space.pin_id = None
            except (AttributeError, ReferenceError):
                pass

    for collection in list(merged_obj.users_collection):
        collection.objects.unlink(merged_obj)

    new_scene = bpy.data.scenes.new(cfg.new_scene_name)
    new_scene.collection.objects.link(merged_obj)

    # Diagnostic validation: the Baked scene must contain exactly 1 object.
    scene_objs = list(new_scene.collection.objects)
    if len(scene_objs) == 1 and scene_objs[0].name == cfg.merged_object_name:
        print(f"bake_unified_material: scene '{cfg.new_scene_name}' contains "
              f"1 object: '{cfg.merged_object_name}' \u2713")
    else:
        names = [o.name for o in scene_objs]
        print(f"bake_unified_material: WARNING - scene '{cfg.new_scene_name}' "
              f"contains {len(scene_objs)} objects: {names} "
              f"(expected: ['{cfg.merged_object_name}']) - possible bug in the join.")
    print(f"bake_unified_material: source scene '{source_scene.name}' - "
          f"{len(list(source_scene.objects))} objects (originals intact).")

    if cfg.copy_world and source_scene.world is not None:
        new_scene.world = source_scene.world

    merged_obj.data.materials.clear()
    merged_obj.data.materials.append(material)

    uv_layers = merged_obj.data.uv_layers
    if uv_layer_name in uv_layers:
        bake_uv = uv_layers[uv_layer_name]
        uv_layers.active = bake_uv
        bake_uv.active_render = True  # guarantees the render uses BakeUV

    return new_scene


def _save_blend(cfg: BakeConfig, new_scene: bpy.types.Scene) -> str:
    """Save a COPY of the current .blend to disk, opening on the Baked scene.

    Saving behaviour (NON-DESTRUCTIVE):
    - `save_as_mainfile(copy=True)` does NOT change the session's filepath.
    - It does NOT overwrite or touch the original .blend file.
    - It saves a self-contained copy in `cfg.output_dir` holding the Baked scene, the
      baked textures (generated in RAM, packed into the .blend) and the World/HDRI.

    To make the copy open directly on the Baked scene (instead of on the source scene),
    it performs a temporary switch: sets the window's active scene to `new_scene`, saves
    with copy=True, then restores the previous scene in a `finally`. Safe from a timer:
    the callback runs in a clean context after the operator has already returned, so
    there is no risk of invalidating the depsgraph while the operator is running.
    """
    filename = cfg.blend_filename or f"{cfg.new_scene_name}.blend"
    if not filename.lower().endswith(".blend"):
        filename += ".blend"
    path = os.path.join(cfg.output_dir, filename)

    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    win = bpy.context.window
    prev_scene = win.scene
    try:
        # Temporary switch to the Baked scene: the .blend copy will record Baked as the
        # active scene -> reopening the file shows the joined BakedMesh right away.
        win.scene = new_scene
        bpy.ops.wm.save_as_mainfile(filepath=path, copy=True)
    finally:
        # Guaranteed restore: the live session goes back to the original source scene,
        # exactly as it was before the save, errors included.
        win.scene = prev_scene
    return path


def _log_summary(baked_images: Dict[str, Tuple[bpy.types.Image, str]], blend_path: Optional[str] = None) -> None:
    print("=== bake_unified_material: done ===")
    for channel, (image, path) in baked_images.items():
        print(f"  {channel:11s} {image.size[0]}x{image.size[1]}  ->  {path}")
    if blend_path is not None:
        print(f"  scene saved -> {blend_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def run(cfg: BakeConfig) -> Tuple[Dict[str, Tuple[bpy.types.Image, str]], bpy.types.Scene]:
    """Run the complete bake pipeline.

    Returns `(baked_images, new_scene)`.

    Saving the .blend does NOT happen here, even if `cfg.save_blend` is True:
    `_save_blend` has to be called AFTER the calling operator, if any, has returned
    (via `bpy.app.timers`) to avoid crashes from an inconsistent context.
    The "from code" use (Text Editor -> Run Script, not nested inside an operator) can
    call `_save_blend(cfg, new_scene)` right after `run()`.
    """
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer

    _setup_render_settings(scene, cfg)

    source_objects = [obj for obj in scene.objects if obj.type == "MESH"]
    if not source_objects:
        raise RuntimeError("No MESH object in the current scene.")

    duplicates = _duplicate_objects(source_objects, view_layer, cfg)
    if cfg.apply_transform:
        _apply_transforms(duplicates, view_layer)
    for dup in duplicates:
        _prepare_bake_uv_per_object(dup, view_layer, cfg)
    merged = _join_objects(duplicates, view_layer, cfg)
    _ensure_default_material(merged)
    uv_layer_name = _pack_bake_uv(merged, view_layer, cfg)

    baked_images = _bake_all(merged, cfg)
    # Pick the reference material BEFORE _assemble_scene clears the slots
    # (merged_obj.data.materials.clear()), otherwise the reference is lost.
    ref_mat = _pick_reference_material(merged)
    if ref_mat is not None:
        print(f"bake_unified_material: reference material for Settings -> {ref_mat.name!r}")
    material = _build_material(cfg, baked_images, ref_material=ref_mat)
    new_scene = _assemble_scene(cfg, merged, material, uv_layer_name, scene)

    _log_summary(baked_images)
    return baked_images, new_scene


# ──────────────────────────────────────────────────────────────────────────────
# UI panel (3D Viewport sidebar) - front-end for BakeConfig / run()
# ──────────────────────────────────────────────────────────────────────────────
#
# Exposes every field of `BakeConfig` as a property on Scene (PointerProperty, saved
# in the .blend) and shows them in a "Bake" panel in the sidebar (N key) of the 3D
# Viewport: it allows setting everything visually and starting the bake with a button.
# The pipeline is still `BakeConfig` + `run(cfg)`, unchanged: the panel is only a
# front-end that calls it with the values chosen by the user.
# The "from code" use, with no panel, stays available:
#     run(BakeConfig(output_dir=..., channels={...}, ...))

_NORMAL_SPACE_ITEMS = (
    ('TANGENT', "Tangent", "Tangent space - the standard for baked normal maps"),
    ('OBJECT', "Object", "Object space"),
)
_TEX_FORMAT_ITEMS = (
    ('OPEN_EXR', "OpenEXR", "High-precision float, the default (recommended)"),
    ('PNG', "PNG", "8/16 bit, more compatible"),
)
_COLOR_DEPTH_ITEMS = (
    ('AUTO', "Auto", "Default of the chosen format: 16 bit for EXR, 8 bit for PNG"),
    ('8', "8-bit", "8 bits per channel - PNG only"),
    ('16', "16-bit", "16 bits per channel"),
    ('32', "32-bit Float", "32-bit float per channel - EXR only"),
)
_DEVICE_ITEMS = (
    ('GPU', "GPU", "Bake on the GPU (requires a Cycles device configured in the Preferences)"),
    ('CPU', "CPU", "Bake on the CPU"),
)
_SWIZZLE_ITEMS = (
    ('POS_X', "+X", "Positive X axis"),
    ('NEG_X', "-X", "Negative X axis"),
    ('POS_Y', "+Y", "Positive Y axis"),
    ('NEG_Y', "-Y", "Negative Y axis"),
    ('POS_Z', "+Z", "Positive Z axis"),
    ('NEG_Z', "-Z", "Negative Z axis"),
)


class BakeUnifiedMaterialProperties(bpy.types.PropertyGroup):
    """Mirror of `BakeConfig` on `Scene`: it allows setting the parameters from the
    panel instead of from code. Being a PointerProperty on Scene, the values stay saved
    in the .blend file."""

    output_dir: bpy.props.StringProperty(
        name="Output Dir",
        description="Folder the baked textures are saved into",
        subtype='DIR_PATH',
        default="",
    )
    new_scene_name: bpy.props.StringProperty(name="New Scene", default="Baked")
    merged_object_name: bpy.props.StringProperty(name="Merged Object", default="BakedMesh")
    merged_material_name: bpy.props.StringProperty(name="Merged Material", default="BakedMaterial")

    bake_base_color: bpy.props.BoolProperty(name="Base Color", default=True)
    bake_metallic: bpy.props.BoolProperty(name="Metallic", default=True)
    bake_roughness: bpy.props.BoolProperty(name="Roughness", default=True)
    bake_normal: bpy.props.BoolProperty(name="Normal", default=True)

    res_base_color: bpy.props.IntVectorProperty(name="Resolution", size=2, min=1, default=_DEFAULT_RESOLUTIONS["base_color"])
    res_metallic: bpy.props.IntVectorProperty(name="Resolution", size=2, min=1, default=_DEFAULT_RESOLUTIONS["metallic"])
    res_roughness: bpy.props.IntVectorProperty(name="Resolution", size=2, min=1, default=_DEFAULT_RESOLUTIONS["roughness"])
    res_normal: bpy.props.IntVectorProperty(name="Resolution", size=2, min=1, default=_DEFAULT_RESOLUTIONS["normal"])

    normal_space: bpy.props.EnumProperty(name="Normal Space", items=_NORMAL_SPACE_ITEMS, default='TANGENT')
    normal_r: bpy.props.EnumProperty(name="R", description="Normal-map axis in the red channel", items=_SWIZZLE_ITEMS, default='POS_X')
    normal_g: bpy.props.EnumProperty(name="G", description="Normal-map axis in the green channel", items=_SWIZZLE_ITEMS, default='POS_Y')
    normal_b: bpy.props.EnumProperty(name="B", description="Normal-map axis in the blue channel", items=_SWIZZLE_ITEMS, default='POS_Z')
    tex_format: bpy.props.EnumProperty(name="Format", items=_TEX_FORMAT_ITEMS, default='OPEN_EXR')
    color_depth: bpy.props.EnumProperty(name="Color Depth", items=_COLOR_DEPTH_ITEMS, default='AUTO')

    base_color_via_emission: bpy.props.BoolProperty(
        name="Base Color via Emission",
        description="Bypasses the shading for a Base Color faithful on metals too. "
                    "If disabled, the native DIFFUSE pass is used (more 'physical', but "
                    "darkened on metals)",
        default=True,
    )

    samples: bpy.props.IntProperty(name="Samples", min=1, soft_max=1024, default=32)
    bake_margin: bpy.props.IntProperty(name="Margin (px)", min=0, default=16)

    uv_island_margin: bpy.props.FloatProperty(name="Island Margin", min=0.0, max=1.0, default=0.02)
    uv_angle_limit: bpy.props.FloatProperty(
        name="Angle Limit (°)",
        description="Threshold in degrees for separating the islands in Smart UV Project",
        min=1.0, max=89.0, default=66.0,
    )

    apply_modifiers: bpy.props.BoolProperty(
        name="Apply Modifiers",
        description="Applies the modifiers on the duplicated meshes before the join/bake",
        default=True,
    )
    apply_transform: bpy.props.BoolProperty(
        name="Apply Transform",
        description="Applies rotation/scale/location to the copies before the bake: "
                    "needed for object-space normals aligned to the world "
                    "(avoids swapped axes when objects have unapplied rotations)",
        default=True,
    )
    device: bpy.props.EnumProperty(name="Device", items=_DEVICE_ITEMS, default='GPU')

    copy_world: bpy.props.BoolProperty(
        name="Copy World (env map)",
        description="Brings the current scene's World (HDRI / environment map) "
                    "into the new baked scene",
        default=True,
    )
    save_blend: bpy.props.BoolProperty(
        name="Save .blend",
        description="Saves a copy of the .blend file with the baked scene into the "
                    "output folder at the end of the bake",
        default=True,
    )
    blend_filename: bpy.props.StringProperty(
        name="Blend File",
        description="Name of the .blend file to save (empty -> '<Scene name>.blend')",
        default="",
    )


def _config_from_properties(props: BakeUnifiedMaterialProperties) -> BakeConfig:
    """Build a `BakeConfig` from the values set in the panel."""
    return BakeConfig(
        output_dir=bpy.path.abspath(props.output_dir),
        new_scene_name=props.new_scene_name,
        merged_object_name=props.merged_object_name,
        merged_material_name=props.merged_material_name,
        channels={
            "base_color": props.bake_base_color,
            "metallic": props.bake_metallic,
            "roughness": props.bake_roughness,
            "normal": props.bake_normal,
        },
        resolutions={
            "base_color": tuple(props.res_base_color),
            "metallic": tuple(props.res_metallic),
            "roughness": tuple(props.res_roughness),
            "normal": tuple(props.res_normal),
        },
        normal_space=props.normal_space,
        normal_r=props.normal_r,
        normal_g=props.normal_g,
        normal_b=props.normal_b,
        tex_format=TextureFormat[props.tex_format],
        color_depth=None if props.color_depth == 'AUTO' else props.color_depth,
        base_color_via_emission=props.base_color_via_emission,
        samples=props.samples,
        bake_margin=props.bake_margin,
        uv_island_margin=props.uv_island_margin,
        uv_angle_limit=props.uv_angle_limit,
        apply_modifiers=props.apply_modifiers,
        apply_transform=props.apply_transform,
        device=props.device,
        copy_world=props.copy_world,
        save_blend=props.save_blend,
        blend_filename=props.blend_filename or None,
    )


class OBJECT_OT_bake_unified_material(bpy.types.Operator):
    """Run the whole pipeline (`run`) with the parameters set in the panel"""
    bl_idname = "object.bake_unified_material"
    bl_label = "Bake Unified Material"
    bl_description = ("Joins the meshes of the scene, bakes the materials into a set of "
                      "PBR textures and assembles a new scene with a single material")
    bl_options = {'REGISTER'}  # no UNDO: it creates objects/scenes/images and writes to disk

    def execute(self, context):
        props = context.scene.bake_unified_props

        if not props.output_dir.strip():
            self.report({'ERROR'}, "Set an output folder (Output Dir).")
            return {'CANCELLED'}

        cfg = _config_from_properties(props)
        try:
            baked, new_scene = run(cfg)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, f"Bake failed: {exc}")
            return {'CANCELLED'}

        if cfg.save_blend:
            # Defer the save outside the execute via a timer.
            # Calling save_as_mainfile INSIDE an execute() is a documented cause of
            # deferred crashes (inconsistent context when the operator returns).
            # With first_interval=0.1 the callback runs ~100 ms after the operator has
            # returned, in a clean context.
            # _save_blend switches -> saves (copy=True) -> restores: safe from a timer.
            # Only data by value is captured (strings, config), never direct references
            # to bpy.data (they could become dangling if the scene were deleted before
            # the timer fires).
            scene_name = new_scene.name

            def _deferred_save():
                scn = bpy.data.scenes.get(scene_name)
                if scn is None:
                    print(f"bake_unified_material: scene '{scene_name}' not found, "
                          f"skipping the save.")
                    return None  # one-shot: do not repeat
                try:
                    blend_path = _save_blend(cfg, scn)
                    print(f"bake_unified_material: COPY saved -> {blend_path}")
                    print(f"bake_unified_material: the original file was NOT touched "
                          f"(saved through copy=True into output_dir).")
                except Exception:
                    import traceback
                    traceback.print_exc()
                return None  # one-shot

            bpy.app.timers.register(_deferred_save, first_interval=0.1)

        self.report({'INFO'}, f"Bake done: {len(baked)} channels. "
                    f".blend COPY in: {cfg.output_dir} - original NOT modified.")
        return {'FINISHED'}


class VIEW3D_PT_bake_unified_material(bpy.types.Panel):
    """Panel in the 3D Viewport sidebar (N) with every bake parameter."""
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Bake"
    bl_label = "Bake Unified Material"

    _CHANNEL_ROWS = (
        ("Base Color", "bake_base_color", "res_base_color"),
        ("Metallic", "bake_metallic", "res_metallic"),
        ("Roughness", "bake_roughness", "res_roughness"),
        ("Normal", "bake_normal", "res_normal"),
    )

    def draw(self, context):
        layout = self.layout
        props = context.scene.bake_unified_props

        col = layout.column(align=True)
        col.label(text="Output")
        col.prop(props, "output_dir", text="")
        col.prop(props, "new_scene_name")
        col.prop(props, "merged_object_name")
        col.prop(props, "merged_material_name")
        col.prop(props, "copy_world")
        col.prop(props, "save_blend")
        sub = col.row(align=True)
        sub.enabled = props.save_blend
        sub.prop(props, "blend_filename")

        box = layout.box()
        box.label(text="Channels and resolutions")
        for label, bake_attr, res_attr in self._CHANNEL_ROWS:
            row = box.row(align=True)
            row.prop(props, bake_attr, text=label)
            sub = row.row(align=True)
            sub.enabled = getattr(props, bake_attr)
            sub.prop(props, res_attr, text="")

        box = layout.box()
        box.label(text="Texture format")
        box.prop(props, "tex_format")
        box.prop(props, "color_depth")
        box.prop(props, "base_color_via_emission")

        box = layout.box()
        box.label(text="Bake & UV")
        box.prop(props, "samples")
        box.prop(props, "bake_margin")
        box.prop(props, "uv_island_margin")
        box.prop(props, "uv_angle_limit")

        box = layout.box()
        box.label(text="Advanced")
        box.prop(props, "normal_space")
        sub = box.column(align=True)
        sub.enabled = props.bake_normal
        sub.label(text="Normal swizzle (R  G  B):")
        row = sub.row(align=True)
        row.prop(props, "normal_r", text="")
        row.prop(props, "normal_g", text="")
        row.prop(props, "normal_b", text="")
        box.prop(props, "apply_modifiers")
        box.prop(props, "apply_transform")
        box.prop(props, "device")

        layout.separator()
        layout.operator(OBJECT_OT_bake_unified_material.bl_idname, icon='RENDER_STILL')


_CLASSES = (
    BakeUnifiedMaterialProperties,
    OBJECT_OT_bake_unified_material,
    VIEW3D_PT_bake_unified_material,
)


def register() -> None:
    # Idempotent: a class/property already registered (repeated Run Script or re-enabled
    # add-on) is skipped, so the values already set in the panel are not lost and no
    # "already registered" errors appear.
    for cls in _CLASSES:
        if not hasattr(bpy.types, cls.__name__):
            bpy.utils.register_class(cls)
    if not hasattr(bpy.types.Scene, "bake_unified_props"):
        bpy.types.Scene.bake_unified_props = bpy.props.PointerProperty(type=BakeUnifiedMaterialProperties)


def unregister() -> None:
    if hasattr(bpy.types.Scene, "bake_unified_props"):
        del bpy.types.Scene.bake_unified_props
    for cls in reversed(_CLASSES):
        # Fetch the class *actually registered* from bpy.types (it can be a different
        # object from `cls` if the module was re-run via Run Script): passing the new
        # object to unregister_class would fail.
        existing = getattr(bpy.types, cls.__name__, None)
        if existing is not None:
            try:
                bpy.utils.unregister_class(existing)
            except Exception:
                pass


def _snapshot_scene_props() -> dict:
    """Read the current PropertyGroup values from every scene, so they can be restored
    after an unregister->register cycle (repeated Run Script)."""
    names = list(BakeUnifiedMaterialProperties.__annotations__.keys())
    snap: dict = {}
    for scn in bpy.data.scenes:
        pg = getattr(scn, "bake_unified_props", None)
        if pg is None:
            continue
        snap[scn.name] = {n: getattr(pg, n) for n in names if hasattr(pg, n)}
    return snap


def _restore_scene_props(snap: dict) -> None:
    """Restore the values saved by `_snapshot_scene_props` onto the new PropertyGroup."""
    for scene_name, values in snap.items():
        scn = bpy.data.scenes.get(scene_name)
        pg = getattr(scn, "bake_unified_props", None) if scn else None
        if pg is None:
            continue
        for n, v in values.items():
            try:
                setattr(pg, n, v)
            except Exception:
                pass  # IntVectorProperty and complex types accept the assignment again; the try/except covers the edge cases


if __name__ == "__main__":
    # Force-reload: unregisters the classes possibly already registered by a previous
    # Run Script (fetching them by name from bpy.types, not by object), then registers
    # the new ones. The panel values are saved and restored.
    _saved = _snapshot_scene_props()
    try:
        unregister()
    except Exception:
        pass
    register()
    _restore_scene_props(_saved)
