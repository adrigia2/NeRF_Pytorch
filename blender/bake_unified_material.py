"""
bake_unified_material.py
------------------------
Script da eseguire DENTRO Blender (Text Editor → Run Script, Blender 4.2+ LTS)
sulla scena correntemente aperta.

Cosa fa:
    1. duplica tutti gli oggetti MESH della scena (l'originale resta intatto),
    2. li unisce in un'unica mesh con un nuovo UV layer "BakeUV" (Smart UV Project),
    3. baka tutti i materiali in un singolo set di texture PBR
       (base_color / metallic / roughness / normal), risoluzione e formato
       configurabili per canale,
    4. costruisce un singolo materiale Principled BSDF che usa quelle texture,
    5. assembla tutto in una nuova scena (default: "Baked").

Uso (tre modalità equivalenti — stessa pipeline `BakeConfig` + `run`):
    A) Pannello (consigliato): apri il file nel Text Editor e premi "Run Script".
       Compare un pannello "Bake" nella sidebar del 3D Viewport (tasto N), dove
       impostare tutti i parametri — incluso `output_dir` — e avviare il bake
       col pulsante "Bake Unified Material" (i valori restano salvati nel .blend,
       così non serve reimpostarli ad ogni Run Script).
    B) Add-on: installabile da Preferences → Add-ons → Install... (vedi `bl_info`).
    C) Da codice: `run(BakeConfig(output_dir=..., ...))`, per pipeline
       automatizzate (vedi `BakeConfig` per l'elenco completo dei parametri).
"""

bl_info = {
    "name": "Bake Unified Material",
    "author": "Adriano Cicco",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar (N) > Bake",
    "description": "Unisce le mesh della scena, baka i materiali in texture PBR "
                   "(base_color/metallic/roughness/normal) e assembla una nuova "
                   "scena con un singolo materiale unificato",
    "category": "Material",
}

# NOTA: niente `from __future__ import annotations` qui. Le classi PropertyGroup/
# Operator/Panel più sotto dichiarano le proprietà come annotazioni di classe
# (`nome: bpy.props.XProperty(...)`): Blender le legge da `__annotations__` a
# tempo di definizione e si aspetta gli oggetti `bpy.props` veri e propri, non le
# stringhe "lazy" prodotte da PEP 563 — con quella import la registrazione
# fallirebbe. I riferimenti a `bpy.types.*` nei type hint del resto del file
# restano comunque validi: dentro Blender sono attributi runtime già disponibili
# all'esecuzione dello script (non serve valutazione differita).

import math
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import bpy

# ──────────────────────────────────────────────────────────────────────────────
# Formati texture
# ──────────────────────────────────────────────────────────────────────────────

class TextureFormat(Enum):
    OPEN_EXR = "OPEN_EXR"
    PNG = "PNG"

    @property
    def file_format(self) -> str:
        """Valore da assegnare a `image_settings.file_format` / `image.file_format`."""
        return self.value

    @property
    def extension(self) -> str:
        return {"OPEN_EXR": ".exr", "PNG": ".png"}[self.value]

    @property
    def use_float_buffer(self) -> bool:
        """EXR è un formato float: l'immagine va creata con buffer float a 32 bit."""
        return self is TextureFormat.OPEN_EXR

    def default_color_depth(self) -> str:
        """Color depth di default se non specificata in `BakeConfig`."""
        return "16" if self is TextureFormat.OPEN_EXR else "8"


# ──────────────────────────────────────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_CHANNELS = {"base_color": True, "metallic": True, "roughness": True, "normal": True}
_DEFAULT_RESOLUTIONS = {
    "base_color": (4096, 4096),
    "metallic": (2048, 2048),
    "roughness": (2048, 2048),
    "normal": (4096, 4096),
}

# Ordine di bake: base_color/metallic prima (emission-swap, più delicati),
# poi i pass nativi.
CHANNEL_ORDER: List[str] = ["base_color", "metallic", "roughness", "normal"]


@dataclass
class BakeConfig:
    output_dir: str

    new_scene_name: str = "Baked"
    merged_object_name: str = "BakedMesh"
    merged_material_name: str = "BakedMaterial"

    # Quali canali bakare.
    channels: Dict[str, bool] = field(default_factory=lambda: dict(_DEFAULT_CHANNELS))
    # Risoluzione (larghezza, altezza) per canale.
    resolutions: Dict[str, Tuple[int, int]] = field(default_factory=lambda: dict(_DEFAULT_RESOLUTIONS))

    # Spazio delle coordinate per la normal map: 'TANGENT' | 'OBJECT'.
    normal_space: str = "TANGENT"

    # Formato/colordepth di default; override opzionali per canale.
    tex_format: TextureFormat = TextureFormat.OPEN_EXR
    color_depth: Optional[str] = None  # None → usa il default del formato
    format_overrides: Dict[str, TextureFormat] = field(default_factory=dict)
    color_depth_overrides: Dict[str, str] = field(default_factory=dict)

    # base_color: di default si usa l'emission-swap del socket "Base Color"
    # (fedele anche sui metalli). Se False, si usa il pass nativo DIFFUSE
    # filtrando solo il colore (più "fisico", ma scurito sui metalli).
    base_color_via_emission: bool = True

    # Parametri di bake.
    samples: int = 32
    bake_margin: int = 16

    # Parametri Smart UV Project (angle_limit in gradi: convertito internamente).
    uv_island_margin: float = 0.02
    uv_angle_limit: float = 66.0

    apply_modifiers: bool = True
    device: str = "GPU"  # 'GPU' | 'CPU'

    # World / environment map.
    copy_world: bool = True  # copia il World della scena sorgente nella nuova scena

    # Salvataggio del file .blend a fine bake.
    save_blend: bool = True
    blend_filename: Optional[str] = None  # None → "<new_scene_name>.blend"

    def __post_init__(self) -> None:
        if self.normal_space not in ("TANGENT", "OBJECT"):
            raise ValueError(f"normal_space deve essere 'TANGENT' o 'OBJECT', non {self.normal_space!r}")
        if self.device not in ("GPU", "CPU"):
            raise ValueError(f"device deve essere 'GPU' o 'CPU', non {self.device!r}")


def _format_for_channel(cfg: BakeConfig, channel: str) -> TextureFormat:
    return cfg.format_overrides.get(channel, cfg.tex_format)


def _color_depth_for_channel(cfg: BakeConfig, channel: str) -> str:
    if channel in cfg.color_depth_overrides:
        return cfg.color_depth_overrides[channel]
    if cfg.color_depth is not None:
        return cfg.color_depth
    return _format_for_channel(cfg, channel).default_color_depth()


def _colorspace_for_channel(channel: str, fmt: TextureFormat) -> str:
    """metallic/roughness/normal sono dati non-colore → sempre 'Non-Color'.
    base_color è un dato colore: 'sRGB' su formati a 8 bit (PNG); su EXR
    (buffer float) si usa 'Non-Color' come equivalente pratico di "Linear" —
    evita doppie conversioni gamma e dipendenze dai nomi colorspace dell'OCIO
    config attiva, che variano fra versioni/temi di Blender."""
    if channel == "base_color" and fmt is not TextureFormat.OPEN_EXR:
        return "sRGB"
    return "Non-Color"


# ──────────────────────────────────────────────────────────────────────────────
# Helper: nodi shader
# ──────────────────────────────────────────────────────────────────────────────

def _get_principled(mat: bpy.types.Material) -> Optional[bpy.types.ShaderNodeBsdfPrincipled]:
    """Trova il Principled BSDF di un materiale, cercando anche dentro i node group."""
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
    """Converte il default_value di un socket (float o color) in una RGBA."""
    if isinstance(value, (int, float)):
        return (float(value),) * 3 + (1.0,)
    seq = tuple(value)
    if len(seq) == 3:
        return seq + (1.0,)
    return seq[:4]


# ──────────────────────────────────────────────────────────────────────────────
# Helper: immagini di destinazione del bake
# ──────────────────────────────────────────────────────────────────────────────

def _new_target_image(name: str, width: int, height: int, fmt: TextureFormat, colorspace: str) -> bpy.types.Image:
    image = bpy.data.images.new(
        name=name,
        width=width,
        height=height,
        alpha=False,
        float_buffer=fmt.use_float_buffer,
    )
    image.colorspace_settings.name = colorspace
    return image


def _save_image(image: bpy.types.Image, path: str, fmt: TextureFormat, color_depth: str, scene: bpy.types.Scene) -> None:
    """Salva `image` su disco nel formato/profondità richiesti.

    Si passa per `image.save_render()` con le `image_settings` della scena
    (temporaneamente riconfigurate e ripristinate) perché è l'unico percorso
    dell'API che applica davvero `color_depth` — l'`Image` datablock non lo
    espone direttamente per `Image.save()`.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    settings = scene.render.image_settings
    prev_format, prev_depth = settings.file_format, settings.color_depth
    try:
        settings.file_format = fmt.file_format
        settings.color_depth = color_depth
        image.filepath_raw = path
        image.file_format = fmt.file_format
        image.save_render(path, scene=scene)
    finally:
        settings.file_format, settings.color_depth = prev_format, prev_depth


# ──────────────────────────────────────────────────────────────────────────────
# Helper: nodo Image Texture temporaneo (target del bake) nei materiali
# ──────────────────────────────────────────────────────────────────────────────

def _setup_bake_nodes(materials: List[bpy.types.Material], image: bpy.types.Image):
    """Inserisce in ogni materiale un nodo Image Texture che punta a `image`,
    lo seleziona e lo rende attivo: è il nodo su cui Cycles eseguirà il bake."""
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
# Bake: pass nativi (DIFFUSE / ROUGHNESS / NORMAL)
# ──────────────────────────────────────────────────────────────────────────────

def _bake_native(bake_type: str, **kwargs) -> None:
    bpy.ops.object.bake(type=bake_type, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Bake: emission-swap (per canali senza pass nativo, es. Metallic, e per
# Base Color quando `base_color_via_emission=True`)
# ──────────────────────────────────────────────────────────────────────────────

def _bake_emission(materials: List[bpy.types.Material], socket_name: str) -> None:
    """Baka il valore del socket `socket_name` del Principled BSDF di ogni
    materiale, reindirizzandolo temporaneamente attraverso un nodo Emission
    e usando `bake(type='EMIT')`. Ripristina sempre i node tree originali
    (anche in caso di errore), grazie al `try/finally`.

    Materiali privi di Principled BSDF o del socket richiesto vengono
    semplicemente saltati: le loro isole UV resteranno al colore di clear
    dell'immagine target (nessun valore scritto per quelle facce)."""
    setups = []
    try:
        for mat in materials:
            principled = _get_principled(mat)
            if principled is None or socket_name not in principled.inputs:
                continue

            node_tree = mat.node_tree
            nodes, links = node_tree.nodes, node_tree.links

            output_node = _get_active_output(node_tree)
            if output_node is None or "Surface" not in output_node.inputs:
                continue
            surface_input = output_node.inputs["Surface"]

            original_link = surface_input.links[0].from_socket if surface_input.is_linked else None

            emission = nodes.new("ShaderNodeEmission")
            emission.label = "_BAKE_EMISSION_"

            socket = principled.inputs[socket_name]
            if socket.is_linked:
                links.new(socket.links[0].from_socket, emission.inputs["Color"])
            else:
                emission.inputs["Color"].default_value = _to_rgba(socket.default_value)

            links.new(emission.outputs["Emission"], surface_input)
            setups.append((node_tree, surface_input, original_link, emission))

        bpy.ops.object.bake(type="EMIT")
    finally:
        for node_tree, surface_input, original_link, emission in setups:
            links = node_tree.links
            for link in list(surface_input.links):
                links.remove(link)
            if original_link is not None:
                links.new(original_link, surface_input)
            node_tree.nodes.remove(emission)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline: setup, duplicazione, join, UV, bake, materiale, scena
# ──────────────────────────────────────────────────────────────────────────────

def _setup_render_settings(scene: bpy.types.Scene, cfg: BakeConfig) -> None:
    scene.render.engine = "CYCLES"
    scene.cycles.samples = cfg.samples
    scene.cycles.device = cfg.device
    scene.render.bake.margin = cfg.bake_margin
    scene.render.bake.use_clear = True
    scene.render.bake.use_selected_to_active = False


def _duplicate_objects(source_objects: List[bpy.types.Object], view_layer: bpy.types.ViewLayer, cfg: BakeConfig) -> List[bpy.types.Object]:
    """Duplica gli oggetti sorgente (la scena originale resta intatta) e,
    se richiesto, applica i modificatori sulle copie via `object.convert`."""
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

    return duplicates


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
    """Garantisce che ogni material slot abbia un materiale: agli oggetti/slot
    senza materiale viene assegnato un Principled BSDF neutro di default,
    così anche le loro facce partecipano correttamente al bake."""
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
    """Prepara il layer BakeUV su un singolo oggetto duplicato, prima del join.

    - Se l'oggetto ha già UV: copia la UV attiva in BakeUV (preserva la
      proiezione originale — niente nuovi tagli/isole).
    - Se l'oggetto non ha UV (primitivo): crea BakeUV via Smart UV Project
      sull'oggetto singolo (molto più controllabile che sull'intera mesh joinata).

    In entrambi i casi BakeUV viene impostato come active e active_render così,
    dopo il join, il layer BakeUV della mesh unita conterrà già i dati corretti
    per ciascun oggetto e serve solo un Pack Islands finale.
    """
    mesh = obj.data

    if mesh.uv_layers:
        # Oggetto con UV: copia la UV attiva in BakeUV.
        # active_render resta sull'UV originale → le texture del materiale vengono
        # campionate con la UV originale durante il bake.
        # active diventa BakeUV → i pixel del bake vengono SCRITTI su BakeUV.
        src = mesh.uv_layers.active
        src.active_render = True  # esplicito: mantieni l'originale per il sampling
        dst = mesh.uv_layers.new(name=UV_LAYER_NAME)
        for s, d in zip(src.data, dst.data):
            d.uv = s.uv[:]
        mesh.uv_layers.active = dst
        # dst.active_render rimane False: BakeUV è solo il target di scrittura
    else:
        # Primitivo senza UV: Smart UV Project sul singolo oggetto.
        # BakeUV è l'unico layer disponibile → è sia active che active_render.
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
    """Dopo il join, esegue Pack Islands su BakeUV per far stare tutte le isole
    in [0,1] senza sovrapposizioni, preservando la forma di ogni isola."""
    mesh = obj.data
    if UV_LAYER_NAME in mesh.uv_layers:
        bake_uv = mesh.uv_layers[UV_LAYER_NAME]
        mesh.uv_layers.active = bake_uv  # target di SCRITTURA del bake
        # active_render deve stare sull'UV originale (non BakeUV) così le texture
        # del materiale vengono campionate correttamente durante il bake.
        for uv in mesh.uv_layers:
            if uv.name != UV_LAYER_NAME:
                uv.active_render = True  # esclusivo: azzera active_render sugli altri
                break

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    view_layer.objects.active = obj

    bpy.ops.object.mode_set(mode="EDIT")
    try:
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.pack_islands(margin=cfg.uv_island_margin)
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")

    return UV_LAYER_NAME


def _bake_all(obj: bpy.types.Object, cfg: BakeConfig, scene: bpy.types.Scene) -> Dict[str, Tuple[bpy.types.Image, str]]:
    """Esegue il bake di ogni canale abilitato e lo salva su disco.
    Ritorna {canale: (image, path)}."""
    # Deduplicato: dopo il join più slot possono puntare allo stesso materiale
    # (oggetti diversi che condividevano un materiale). Processarlo due volte
    # creerebbe nodi Image Texture/Emission ridondanti nello stesso node tree
    # e romperebbe il ripristino in `_bake_emission`.
    seen = set()
    materials: List[bpy.types.Material] = []
    for slot in obj.material_slots:
        mat = slot.material
        if mat is not None and mat not in seen:
            seen.add(mat)
            materials.append(mat)
    if not materials:
        raise RuntimeError("L'oggetto unito non ha materiali su cui eseguire il bake.")

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
                    _bake_emission(materials, "Base Color")
                else:
                    _bake_native("DIFFUSE", pass_filter={"COLOR"})
            elif channel == "roughness":
                _bake_native("ROUGHNESS")
            elif channel == "metallic":
                _bake_emission(materials, "Metallic")
            elif channel == "normal":
                _bake_native("NORMAL", normal_space=cfg.normal_space)
        finally:
            _cleanup_bake_nodes(bake_nodes)

        depth = _color_depth_for_channel(cfg, channel)
        path = os.path.join(cfg.output_dir, f"{image_name}{fmt.extension}")
        _save_image(image, path, fmt, depth, scene)
        baked[channel] = (image, path)

    return baked


_CHANNEL_SOCKET = {"base_color": "Base Color", "metallic": "Metallic", "roughness": "Roughness"}
_CHANNEL_Y = {"base_color": 300, "metallic": 100, "roughness": -100, "normal": -300}


def _build_material(cfg: BakeConfig, baked_images: Dict[str, Tuple[bpy.types.Image, str]]) -> bpy.types.Material:
    """Crea il materiale unificato: Principled BSDF alimentato dalle texture
    appena bakate (normal → Normal Map node con lo `space` configurato)."""
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
            normal_map = nodes.new("ShaderNodeNormalMap")
            normal_map.space = cfg.normal_space
            normal_map.location = (-300, _CHANNEL_Y[channel])
            links.new(tex_node.outputs["Color"], normal_map.inputs["Color"])
            links.new(normal_map.outputs["Normal"], principled.inputs["Normal"])
        else:
            links.new(tex_node.outputs["Color"], principled.inputs[_CHANNEL_SOCKET[channel]])

    return mat


def _assemble_scene(cfg: BakeConfig, merged_obj: bpy.types.Object, material: bpy.types.Material, uv_layer_name: str, source_scene: bpy.types.Scene) -> bpy.types.Scene:
    """Sposta l'oggetto unito (era un duplicato di lavoro) in una nuova scena,
    gli assegna come UNICO materiale quello unificato e attiva l'UV di bake.
    Se `cfg.copy_world` è True, condivide il World della scena sorgente (HDRI /
    environment map) con la nuova scena."""
    for collection in list(merged_obj.users_collection):
        collection.objects.unlink(merged_obj)

    new_scene = bpy.data.scenes.new(cfg.new_scene_name)
    new_scene.collection.objects.link(merged_obj)

    if cfg.copy_world and source_scene.world is not None:
        new_scene.world = source_scene.world

    merged_obj.data.materials.clear()
    merged_obj.data.materials.append(material)

    uv_layers = merged_obj.data.uv_layers
    if uv_layer_name in uv_layers:
        bake_uv = uv_layers[uv_layer_name]
        uv_layers.active = bake_uv
        bake_uv.active_render = True  # garantisce che il render usi BakeUV

    return new_scene


def _save_blend(cfg: BakeConfig, new_scene: bpy.types.Scene) -> str:
    """Salva una copia del file .blend corrente (con la scena bakata) su disco.

    Usa `save_as_mainfile(copy=True)` così la sessione corrente non cambia il
    proprio filepath — il file è una copia autonoma che include la scena Baked,
    le texture bakate (generate in RAM, salvate nel .blend) e il World/HDRI
    referenziato dalla scena originale.
    """
    filename = cfg.blend_filename or f"{cfg.new_scene_name}.blend"
    if not filename.lower().endswith(".blend"):
        filename += ".blend"
    path = os.path.join(cfg.output_dir, filename)

    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    # Imposta la scena bakata come attiva nella finestra così il file si apre
    # direttamente su di essa.
    win = bpy.context.window
    if win is not None:
        win.scene = new_scene

    bpy.ops.wm.save_as_mainfile(filepath=path, copy=True)
    return path


def _log_summary(baked_images: Dict[str, Tuple[bpy.types.Image, str]], blend_path: Optional[str] = None) -> None:
    print("=== bake_unified_material: completato ===")
    for channel, (image, path) in baked_images.items():
        print(f"  {channel:11s} {image.size[0]}x{image.size[1]}  ->  {path}")
    if blend_path is not None:
        print(f"  scena salvata -> {blend_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def run(cfg: BakeConfig) -> Dict[str, Tuple[bpy.types.Image, str]]:
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer

    _setup_render_settings(scene, cfg)

    source_objects = [obj for obj in scene.objects if obj.type == "MESH"]
    if not source_objects:
        raise RuntimeError("Nessun oggetto MESH nella scena corrente.")

    duplicates = _duplicate_objects(source_objects, view_layer, cfg)
    for dup in duplicates:
        _prepare_bake_uv_per_object(dup, view_layer, cfg)
    merged = _join_objects(duplicates, view_layer, cfg)
    _ensure_default_material(merged)
    uv_layer_name = _pack_bake_uv(merged, view_layer, cfg)

    baked_images = _bake_all(merged, cfg, scene)
    material = _build_material(cfg, baked_images)
    new_scene = _assemble_scene(cfg, merged, material, uv_layer_name, scene)

    blend_path = _save_blend(cfg, new_scene) if cfg.save_blend else None

    _log_summary(baked_images, blend_path)
    return baked_images


# ──────────────────────────────────────────────────────────────────────────────
# Pannello UI (sidebar 3D Viewport) — front-end per BakeConfig / run()
# ──────────────────────────────────────────────────────────────────────────────
#
# Espone ogni campo di `BakeConfig` come proprietà su Scene (PointerProperty,
# salvata nel .blend) e le mostra in un pannello "Bake" nella sidebar (tasto N)
# del 3D Viewport: permette di impostare tutto visivamente e avviare il bake con
# un pulsante. La pipeline resta sempre `BakeConfig` + `run(cfg)`, invariata —
# il pannello è solo un front-end che la richiama coi valori scelti dall'utente.
# Resta comunque possibile l'uso "da codice", senza pannello:
#     run(BakeConfig(output_dir=..., channels={...}, ...))

_NORMAL_SPACE_ITEMS = (
    ('TANGENT', "Tangent", "Spazio tangente — standard per le normal map da bake"),
    ('OBJECT', "Object", "Spazio oggetto"),
)
_TEX_FORMAT_ITEMS = (
    ('OPEN_EXR', "OpenEXR", "Float ad alta precisione, di default (consigliato)"),
    ('PNG', "PNG", "8/16 bit, più compatibile"),
)
_COLOR_DEPTH_ITEMS = (
    ('AUTO', "Auto", "Default del formato scelto: 16 bit per EXR, 8 bit per PNG"),
    ('8', "8-bit", "8 bit per canale — solo PNG"),
    ('16', "16-bit", "16 bit per canale"),
    ('32', "32-bit Float", "32 bit float per canale — solo EXR"),
)
_DEVICE_ITEMS = (
    ('GPU', "GPU", "Bake sulla GPU (richiede un device Cycles configurato nelle Preferences)"),
    ('CPU', "CPU", "Bake sulla CPU"),
)


class BakeUnifiedMaterialProperties(bpy.types.PropertyGroup):
    """Specchio di `BakeConfig` su `Scene`: permette di impostare i parametri dal
    pannello invece che dal codice. Essendo una PointerProperty su Scene, i
    valori restano salvati nel file .blend."""

    output_dir: bpy.props.StringProperty(
        name="Output Dir",
        description="Cartella in cui salvare le texture bakate",
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
    tex_format: bpy.props.EnumProperty(name="Format", items=_TEX_FORMAT_ITEMS, default='OPEN_EXR')
    color_depth: bpy.props.EnumProperty(name="Color Depth", items=_COLOR_DEPTH_ITEMS, default='AUTO')

    base_color_via_emission: bpy.props.BoolProperty(
        name="Base Color via Emission",
        description="Bypassa lo shading per un Base Color fedele anche sui metalli. "
                    "Se disattivato usa il pass nativo DIFFUSE (più 'fisico', ma "
                    "scurito sui metalli)",
        default=True,
    )

    samples: bpy.props.IntProperty(name="Samples", min=1, soft_max=1024, default=32)
    bake_margin: bpy.props.IntProperty(name="Margin (px)", min=0, default=16)

    uv_island_margin: bpy.props.FloatProperty(name="Island Margin", min=0.0, max=1.0, default=0.02)
    uv_angle_limit: bpy.props.FloatProperty(
        name="Angle Limit (°)",
        description="Soglia in gradi per separare le isole nello Smart UV Project",
        min=1.0, max=89.0, default=66.0,
    )

    apply_modifiers: bpy.props.BoolProperty(
        name="Apply Modifiers",
        description="Applica i modificatori sulle mesh duplicate prima del join/bake",
        default=True,
    )
    device: bpy.props.EnumProperty(name="Device", items=_DEVICE_ITEMS, default='GPU')

    copy_world: bpy.props.BoolProperty(
        name="Copy World (env map)",
        description="Porta il World (HDRI / environment map) della scena corrente "
                    "nella nuova scena bakata",
        default=True,
    )
    save_blend: bpy.props.BoolProperty(
        name="Save .blend",
        description="Salva una copia del file .blend con la scena bakata nella "
                    "cartella di output a fine bake",
        default=True,
    )
    blend_filename: bpy.props.StringProperty(
        name="Blend File",
        description="Nome del file .blend da salvare (vuoto → '<Nome scena>.blend')",
        default="",
    )


def _config_from_properties(props: BakeUnifiedMaterialProperties) -> BakeConfig:
    """Costruisce un `BakeConfig` a partire dai valori impostati nel pannello."""
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
        tex_format=TextureFormat[props.tex_format],
        color_depth=None if props.color_depth == 'AUTO' else props.color_depth,
        base_color_via_emission=props.base_color_via_emission,
        samples=props.samples,
        bake_margin=props.bake_margin,
        uv_island_margin=props.uv_island_margin,
        uv_angle_limit=props.uv_angle_limit,
        apply_modifiers=props.apply_modifiers,
        device=props.device,
        copy_world=props.copy_world,
        save_blend=props.save_blend,
        blend_filename=props.blend_filename or None,
    )


class OBJECT_OT_bake_unified_material(bpy.types.Operator):
    """Esegue l'intera pipeline (`run`) con i parametri impostati nel pannello"""
    bl_idname = "object.bake_unified_material"
    bl_label = "Bake Unified Material"
    bl_description = ("Unisce le mesh della scena, baka i materiali in un set di "
                      "texture PBR e assembla una nuova scena con un materiale unico")
    bl_options = {'REGISTER'}  # niente UNDO: crea oggetti/scene/immagini e scrive su disco

    def execute(self, context):
        props = context.scene.bake_unified_props

        if not props.output_dir.strip():
            self.report({'ERROR'}, "Imposta una cartella di output (Output Dir).")
            return {'CANCELLED'}

        cfg = _config_from_properties(props)
        try:
            baked = run(cfg)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, f"Bake fallito: {exc}")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Bake completato: {len(baked)} canali salvati in {cfg.output_dir}")
        return {'FINISHED'}


class VIEW3D_PT_bake_unified_material(bpy.types.Panel):
    """Pannello nella sidebar (N) del 3D Viewport con tutti i parametri di bake."""
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
        box.label(text="Canali e risoluzioni")
        for label, bake_attr, res_attr in self._CHANNEL_ROWS:
            row = box.row(align=True)
            row.prop(props, bake_attr, text=label)
            sub = row.row(align=True)
            sub.enabled = getattr(props, bake_attr)
            sub.prop(props, res_attr, text="")

        box = layout.box()
        box.label(text="Formato texture")
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
        box.label(text="Avanzate")
        box.prop(props, "normal_space")
        box.prop(props, "apply_modifiers")
        box.prop(props, "device")

        layout.separator()
        layout.operator(OBJECT_OT_bake_unified_material.bl_idname, icon='RENDER_STILL')


_CLASSES = (
    BakeUnifiedMaterialProperties,
    OBJECT_OT_bake_unified_material,
    VIEW3D_PT_bake_unified_material,
)


def register() -> None:
    # Idempotente: se una classe/proprietà è già registrata (Run Script ripetuto
    # o add-on riabilitato) la si salta, così non si perdono i valori già
    # impostati nel pannello e non compaiono errori "already registered".
    for cls in _CLASSES:
        if not hasattr(bpy.types, cls.__name__):
            bpy.utils.register_class(cls)
    if not hasattr(bpy.types.Scene, "bake_unified_props"):
        bpy.types.Scene.bake_unified_props = bpy.props.PointerProperty(type=BakeUnifiedMaterialProperties)


def unregister() -> None:
    if hasattr(bpy.types.Scene, "bake_unified_props"):
        del bpy.types.Scene.bake_unified_props
    for cls in reversed(_CLASSES):
        # Recupera la classe *realmente registrata* da bpy.types (può essere un
        # oggetto diverso da `cls` se il modulo è stato rieseguito via Run Script):
        # passare il nuovo oggetto a unregister_class fallirebbe.
        existing = getattr(bpy.types, cls.__name__, None)
        if existing is not None:
            try:
                bpy.utils.unregister_class(existing)
            except Exception:
                pass


def _snapshot_scene_props() -> dict:
    """Legge i valori attuali del PropertyGroup da tutte le scene, così possono
    essere ripristinati dopo un ciclo unregister→register (Run Script ripetuto)."""
    names = list(BakeUnifiedMaterialProperties.__annotations__.keys())
    snap: dict = {}
    for scn in bpy.data.scenes:
        pg = getattr(scn, "bake_unified_props", None)
        if pg is None:
            continue
        snap[scn.name] = {n: getattr(pg, n) for n in names if hasattr(pg, n)}
    return snap


def _restore_scene_props(snap: dict) -> None:
    """Ripristina i valori salvati da `_snapshot_scene_props` sul nuovo PropertyGroup."""
    for scene_name, values in snap.items():
        scn = bpy.data.scenes.get(scene_name)
        pg = getattr(scn, "bake_unified_props", None) if scn else None
        if pg is None:
            continue
        for n, v in values.items():
            try:
                setattr(pg, n, v)
            except Exception:
                pass  # IntVectorProperty e tipi complessi riaccettano l'assegnazione; il try/except copre i casi limite


if __name__ == "__main__":
    # Force-reload: unregistra le classi eventualmente già registrate da un
    # Run Script precedente (recuperandole per nome da bpy.types, non per oggetto),
    # poi registra le nuove. I valori del pannello vengono salvati e ripristinati.
    _saved = _snapshot_scene_props()
    try:
        unregister()
    except Exception:
        pass
    register()
    _restore_scene_props(_saved)
