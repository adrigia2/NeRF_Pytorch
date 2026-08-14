#!/usr/bin/env python
"""rerender_run.py -- Rirenderizza in Blender una run terminata con le sue texture ricostruite.

    python rerender_run.py <run_dir> --skybox <hdr.exr> [--materials pbr|lambert|gt] ...

`<run_dir>` e' la cartella di scena di una run (quella che contiene `run_manifest.json`),
per esempio:

    D:/tesi_output/test_sword_shield_after_fix_irradiance/exp_l1_d02/TableAndOtherInteriorWithSpecular

Produce le stesse immagini del dataset di training - stessi nomi, stessa risoluzione,
stesso formato - ma con il materiale ricostruito dalla pipeline al posto di quello
originale, cosi' che

    compare_exr.py --original <run_dir>/images --computed <out>/images

le accoppi da solo e ne faccia il confronto pixel a pixel.

Fuori dalla pipeline: come `roi_rerun.py` e `rerun_irradiance.py`, non tocca ne' OptiX ne'
il checkpoint NeRF, legge solo quello che la run ha gia' scritto su disco.

-------------------------------------------------------------------------------
Come gira
-------------------------------------------------------------------------------
Un file solo, in due modi a seconda di dove viene eseguito (`try: import bpy`):

  * con il python normale -> fa da launcher: risolve i path leggendo la run, prepara le
    texture, scrive un job JSON e invoca `blender --background --python <se stesso>`;
  * dentro Blender      -> legge il job, monta la scena e renderizza.

La divisione non e' estetica: le mappe vanno lette e riscritte come EXR, e il Python di
Blender ha numpy ma NON ha OpenEXR, mentre l'env `nerfpytorch` ha entrambi piu' scipy.

-------------------------------------------------------------------------------
Il materiale di riferimento
-------------------------------------------------------------------------------
Il grafo replicato e' quello di `BakedMaterial` nei `Baked.blend` sorgente, che sono i
file che hanno prodotto le immagini di training (il loro `render.filepath` punta
all'ultimo frame del dataset):

    base color -> Base Color
    metallic   -> Metallic
    roughness  -> Roughness
    normal     -> Normal Map (space=OBJECT, strength 1) -> Normal

Color space: il base color e' taggato `Linear Rec.709`, gli altri tre `Non-Color`.  Nel
blend originale anche il base color era `Non-Color`; la differenza non cambia un pixel
(misurato: scarto massimo 0.0, perche' lo spazio di riferimento di scena *e'* Linear
Rec.709 e quindi entrambe le trasformazioni sono l'identita'), ma `Non-Color` significa
"questo non e' un colore, non convertirlo mai" ed e' il tag giusto solo per i dati.  Su un
albedo, in una configurazione con riferimento diverso (ACEScg), passerebbe numeri Rec.709
spacciandoli per ACEScg sbagliando in silenzio.

-------------------------------------------------------------------------------
Tre cose che romperebbero tutto senza dare sintomi
-------------------------------------------------------------------------------
1. `metallic.exr` e `roughness.exr` hanno un solo canale, chiamato `Z`, che Blender non
   legge come immagine: vanno usati i `_rgb` che la pipeline scrive di fianco.  Se mancano
   questo script si ferma e rimanda a `exr_to_blender_rgb.py` invece di caricare un nero.
2. La skybox va passata a mano e sbagliarla e' facile: notturna e studio condividono
   modello, camere e layout, e un render con la skybox sbagliata *sembra* corretto.  Da
   qui la guardia che confronta con l'environment del `Baked.blend` sorgente.
   Le corrispondenze giuste:

     TableAndOtherInteriorWithSpecular / ...NoSpecular
         Scenes/TableAndOtherInterior/Blender/assets/hdri/wooden_studio_13_4k.exr
     TableAndOtherInteriorWithSpecularNight
         Scenes/TableAndOtherInterior/BlenderBakedSmoothNight/cobblestone_street_night_4k.exr
     SwordShieldStudio
         Scenes/SwordShield Thesis/Blender/assets/hdrs/wooden_studio_13_4k.exr
     SwordShieldNight
         Scenes/SwordShield Thesis/Blender/assets/hdrs/cobblestone_street_night_4k.exr
3. L'OBJ e' gia' in frame Blender: `wm.obj_import(forward_axis='Y', up_axis='Z')` da'
   `matrix_world` identita'.  Nessuna correzione d'assi sulle pose delle camere.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:                     # dentro Blender
    import bpy
    IN_BLENDER = True
except ImportError:      # launcher, python normale
    IN_BLENDER = False

# Su Windows stdout arriva in cp1252 e i caratteri usati nei messaggi (⚠, ✗) lo fanno
# esplodere a meta' esecuzione.  Impostare PYTHONIOENCODING dentro lo script sarebbe
# tardi: lo stream esiste gia' quando il modulo viene importato.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:    # noqa: BLE001  -- stream non riconfigurabile: pazienza
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Costanti di render, lette dai cinque Baked.blend sorgente.
#
# Sono identiche in tutti e cinque, per questo si possono fissare qui invece di
# rileggerle scena per scena.  L'unica che varia e' `view_transform` (Raw in alcuni, AgX
# in altri) e non conta: Blender scrive lineare sui formati float, come dimostra il fatto
# che le immagini GT dei blend AgX contengono valori fino a 1.54.  Lo forzo a Raw perche'
# l'output sia inequivocabile.
# ──────────────────────────────────────────────────────────────────────────────

GT_SAMPLES = 4096
GT_ADAPTIVE_THRESHOLD = 0.01
GT_BOUNCES = dict(max_bounces=12, diffuse_bounces=4, glossy_bounces=4,
                  transmission_bounces=12, volume_bounces=0, transparent_max_bounces=8)
GT_CLAMP_DIRECT = 0.0
GT_CLAMP_INDIRECT = 10.0
GT_CLIP_START = 0.1
GT_CLIP_END = 1000.0
GT_BG_STRENGTH = 1.0

DEFAULT_BLENDER = r"C:/Program Files/Blender Foundation/Blender 5.0/blender.exe"
DEFAULT_DILATE = 8          # texel a 4096^2, equivalente al margine 16 del bake GT a 8192^2

CS_COLOR = "Linear Rec.709"
CS_DATA = "Non-Color"


# ══════════════════════════════════════════════════════════════════════════════
# LAUNCHER  (python normale: OpenEXR + scipy disponibili)
# ══════════════════════════════════════════════════════════════════════════════

def _load_exr_hw3(path: Path):
    """(H, W, 3) float32.  Stesso loader di regen_heatmaps: legge R/G/B per nome e
    replica il canale unico se l'EXR ne ha uno solo."""
    import numpy as np
    import OpenEXR
    import Imath

    exr = OpenEXR.InputFile(str(path))
    dw = exr.header()["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    chs = exr.header()["channels"]
    if "R" in chs and "G" in chs and "B" in chs:
        ch = [np.frombuffer(exr.channel(c, pt), np.float32).reshape(h, w) for c in "RGB"]
    else:
        key = next(iter(chs))
        one = np.frombuffer(exr.channel(key, pt), np.float32).reshape(h, w)
        ch = [one, one, one]
    return np.stack(ch, axis=-1)


def _write_exr_rgb(path: Path, img) -> None:
    """Scrive (H, W, 3) float32 su tre canali R/G/B float, compressione ZIP."""
    import numpy as np
    import OpenEXR
    import Imath

    h, w = img.shape[:2]
    header = OpenEXR.Header(w, h)
    ft = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header["channels"] = {c: ft for c in "RGB"}
    header["compression"] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
    out = OpenEXR.OutputFile(str(path), header)
    a = np.ascontiguousarray(img, dtype=np.float32)
    out.writePixels({c: a[..., i].tobytes() for i, c in enumerate("RGB")})
    out.close()


def _nearest_valid_indices(mask):
    """Per ogni texel invalido, l'indice del texel valido piu' vicino, piu' la distanza.

    Calcolato una volta sola e riusato da tutte le mappe della stessa run: la trasformata
    di distanza euclidea su 4096^2 non e' gratis e la maschera e' la stessa per tutte.
    """
    from scipy.ndimage import distance_transform_edt
    dist, idx = distance_transform_edt(~mask, return_indices=True)
    return dist, idx


def _dilate(img, mask, dist, idx, radius: float):
    """Riempie verso l'esterno i texel invalidi entro `radius` con il valore del texel
    valido piu' vicino.

    Serve perche' le mappe ricostruite valgono 0 fuori dalla maschera IUM, bordi delle
    isole UV compresi, e con il filtraggio bilineare Blender pesca quello zero appena
    dentro il bordo.  L'artefatto che ne esce non e' una cucitura scura come verrebbe da
    aspettarsi: sulla roughness lo zero significa SPECCHIO, quindi il bordo diventa una
    riga speculare BRILLANTE (misurata: fino a 0.45 di differenza su una linea larga un
    paio di pixel, 0.24% dei pixel dell'immagine).  Le texture GT non ce l'hanno perche'
    il baker di Blender applica un margine; questo e' l'equivalente.
    """
    out = img.copy()
    fill = (~mask) & (dist <= radius)
    iy, ix = idx[0][fill], idx[1][fill]
    out[fill] = img[iy, ix]
    return out, int(fill.sum())


def _resolve_maps(run_dir: Path, tf_ext: dict, manifest: dict,
                  materials: str, source: str) -> dict:
    """I path assoluti delle mappe, e quali di esse sono ricostruite (cioe' da dilatare).

    In modalita' `lambert` c'e' il solo base color: metallic e roughness diventano scalari,
    fissati da `build_material`.

    Nessun path viene costruito a mano: quelli ricostruiti stanno nel blocco `ium` di
    `transforms_extended.json`, quelli GT si trovano accanto alla normale esterna che il
    manifest registra.
    """
    ium = tf_ext.get("ium", {})
    scene = manifest.get("scene", {})
    ext_normal = scene.get("external_normal_path")
    if not ext_normal:
        raise SystemExit("✗ il manifest non registra `scene.external_normal_path`: "
                         "senza normale non so replicare il materiale di riferimento.")
    normal = Path(ext_normal)

    if materials == "gt":
        # Le mappe originali stanno nella cartella del bake, cioe' quella della normale.
        gt_dir = normal.parent
        maps = {
            "base_color": gt_dir / "BakedMaterial_base_color.exr",
            "metallic": gt_dir / "BakedMaterial_metallic.exr",
            "roughness": gt_dir / "BakedMaterial_roughness.exr",
        }
        reconstructed = []
    else:
        def need(key: str) -> Path:
            rel = ium.get(f"{key}_{source}")
            if not rel:
                raise SystemExit(
                    f"✗ `transforms_extended.json` non ha `{key}_{source}`. "
                    f"Sorgenti disponibili: "
                    f"{sorted({k.rsplit('_', 1)[-1] for k in ium if k.startswith('albedo')})}")
            return run_dir / rel

        if materials == "pbr":
            base = need("albedo_pbr_path")
            # I file a canale singolo (`Z`) non sono leggibili da Blender: servono i _rgb
            # che `exr_to_blender_rgb` scrive di fianco.
            met = need("metallic_path").with_name("metallic_rgb.exr")
            rou = need("roughness_path").with_name("roughness_rgb.exr")
            for p, orig in ((met, "metallic"), (rou, "roughness")):
                if not p.exists():
                    raise SystemExit(
                        f"✗ manca {p}.\n"
                        f"  {orig}.exr ha un solo canale, chiamato `Z`, che Blender non "
                        f"legge come immagine.\n"
                        f"  Genera la copia RGB con:  python exr_to_blender_rgb.py "
                        f"{p.parent}")
            maps = {"base_color": base, "metallic": met, "roughness": rou}
        else:  # lambert
            maps = {"base_color": need("albedo_path")}
        reconstructed = [k for k in maps]

    maps["normal"] = normal
    for k, p in maps.items():
        if not p.exists():
            raise SystemExit(f"✗ mappa `{k}` non trovata: {p}")
    return {"maps": {k: str(v) for k, v in maps.items()},
            "reconstructed": reconstructed}


def _report_material_stats(run_dir: Path, maps: dict) -> None:
    """Statistiche delle tre mappe ricostruite, piu' un avviso sui texel di "metallo nero".

    Non e' cosmetica.  `pbr_solver` scrive albedo = 0 dove il texel risulta interamente
    speculare (x < X_EPS), per convenzione metallica: l'albedo diffuso li' non e' definito.
    Il Principled di Blender pero' interpreta Base Color come il colore di RIFLESSIONE del
    metallo quando Metallic = 1, quindi quegli stessi texel diventano specchi che non
    riflettono nulla, cioe' macchie nere.  E' un disaccordo fra due convenzioni, non un
    errore del render: senza questo conteggio verrebbe scambiato per un bug della pipeline
    di rerender.
    """
    import numpy as np

    mask_path = run_dir / "ium" / "ium_masks.exr"
    if "metallic" not in maps or not mask_path.exists():
        return
    m = _load_exr_hw3(mask_path)[..., 0] > 0.5
    alb = _load_exr_hw3(Path(maps["base_color"]))[m].max(-1)
    met = _load_exr_hw3(Path(maps["metallic"]))[m][..., 0]
    rou = _load_exr_hw3(Path(maps["roughness"]))[m][..., 0]
    print(f"  {m.sum():,} texel validi")
    print(f"  albedo    p50={np.median(alb):.4f}   nero (<1e-3) {100 * (alb < 1e-3).mean():.2f}%")
    print(f"  metallic  p50={np.median(met):.4f}   >0.5 {100 * (met > 0.5).mean():.2f}%")
    print(f"  roughness p50={np.median(rou):.4f}   ==0 (specchio) "
          f"{100 * (rou < 1e-6).mean():.2f}%   ==1 {100 * (rou > 0.999).mean():.2f}%")
    bad = float(((met > 0.5) & (alb < 1e-3)).mean())
    if bad > 1e-4:
        print(f"  ⚠  {100 * bad:.2f}% dei texel ha metallic>0.5 con albedo~0: nel Principled "
              f"sono specchi che non riflettono nulla,\n"
              f"     quindi appariranno NERI.  E' la convenzione metallica di pbr_solver "
              f"(albedo=0 dove x<X_EPS), non un errore del render.")


def _prepare_textures(run_dir: Path, out_dir: Path, maps: dict,
                      reconstructed: list, radius: float) -> dict:
    """Dilata le mappe ricostruite e le riscrive in `<out>/textures/`.

    La normale e le mappe GT non vengono toccate: sono i file originali del bake, che il
    margine ce l'hanno gia'.
    """
    if not reconstructed:
        print("  nessuna mappa ricostruita da preparare (materiali GT, usati i file "
              "originali del bake)")
        return maps
    if radius <= 0:
        print("  dilatazione disattivata (--dilate 0): attese righe speculari "
              "brillanti sui bordi delle isole UV")
        return maps

    mask_path = run_dir / "ium" / "ium_masks.exr"
    if not mask_path.exists():
        print(f"  ⚠  {mask_path} non trovata: salto la dilatazione")
        return maps

    mask = _load_exr_hw3(mask_path)[..., 0] > 0.5
    print(f"  maschera IUM {mask.shape[1]}x{mask.shape[0]}, "
          f"{100.0 * mask.mean():.1f}% dei texel validi")
    dist, idx = _nearest_valid_indices(mask)

    tex_dir = out_dir / "textures"
    tex_dir.mkdir(parents=True, exist_ok=True)
    resolved = dict(maps)
    for key in reconstructed:
        src = Path(maps[key])
        img = _load_exr_hw3(src)
        if img.shape[:2] != mask.shape:
            print(f"  ⚠  {src.name} e' {img.shape[1]}x{img.shape[0]} ma la maschera e' "
                  f"{mask.shape[1]}x{mask.shape[0]}: salto la dilatazione di questa mappa")
            continue
        out, n = _dilate(img, mask, dist, idx, radius)
        dst = tex_dir / src.name
        _write_exr_rgb(dst, out)
        resolved[key] = str(dst)
        print(f"  + textures/{src.name}  ({n} texel riempiti entro {radius:g})")
    del dist, idx, mask
    return resolved


def launcher() -> int:
    import argparse
    import subprocess
    import time

    ap = argparse.ArgumentParser(
        prog="rerender_run",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="cartella di scena della run (contiene run_manifest.json)")
    ap.add_argument("--skybox", required=True,
                    help="HDR originale da usare come environment (NON quello ricostruito)")
    ap.add_argument("--materials", default="pbr", choices=["pbr", "lambert", "gt"],
                    help="pbr: albedo_pbr+metallic+roughness | lambert: albedo diffuso | "
                         "gt: texture originali, controllo del rig (default: pbr)")
    ap.add_argument("--source", default="gt", help="sorgente delle mappe: gt | nerf (default: gt)")
    ap.add_argument("--out", default=None, help="cartella di output (default: <run>/rerender/<modo>)")
    ap.add_argument("--samples", type=int, default=GT_SAMPLES,
                    help=f"sample Cycles (default: {GT_SAMPLES}, come il render GT; "
                         f"abbassarlo e' il modo di fare una prova rapida)")
    ap.add_argument("--frames", nargs="+", default=None,
                    help="stem dei soli frame da renderizzare, es. render_Camera_Shell10_0")
    ap.add_argument("--limit", type=int, default=0, help="renderizza solo i primi N frame")
    ap.add_argument("--force", action="store_true", help="rifai anche i frame gia' su disco")
    ap.add_argument("--dilate", type=float, default=DEFAULT_DILATE,
                    help=f"raggio in texel del riempimento verso l'esterno delle mappe "
                         f"ricostruite (default: {DEFAULT_DILATE}, 0 disattiva)")
    ap.add_argument("--no-normal", action="store_true", dest="no_normal",
                    help="non collegare la normal map")
    ap.add_argument("--device", default="GPU", choices=["GPU", "CPU"])
    ap.add_argument("--blender", default=DEFAULT_BLENDER)
    ap.add_argument("--save-blend", action="store_true", dest="save_blend",
                    help="salva anche scene.blend (senza pack: le texture restano linkate)")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="risolve tutto e stampa il job senza lanciare Blender")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    manifest_path = run_dir / "run_manifest.json"
    tf_path = run_dir / "transforms_extended.json"
    for p in (manifest_path, tf_path):
        if not p.exists():
            raise SystemExit(f"✗ non trovato: {p}\n  `run_dir` deve essere la cartella di "
                             f"scena di una run terminata.")
    skybox = Path(args.skybox).resolve()
    if not skybox.exists():
        raise SystemExit(f"✗ skybox non trovata: {skybox}")
    blender = Path(args.blender)
    if not blender.exists():
        raise SystemExit(f"✗ Blender non trovato: {blender}  (usa --blender)")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tf_ext = json.loads(tf_path.read_text(encoding="utf-8"))
    model = Path(manifest["scene"]["model_path"])
    if not model.exists():
        raise SystemExit(f"✗ modello non trovato: {model}")

    tag = "gt_control" if args.materials == "gt" else f"{args.materials}_{args.source}"
    out_dir = Path(args.out).resolve() if args.out else run_dir / "rerender" / tag
    out_images = out_dir / "images"

    print(f"run       {run_dir}")
    print(f"modello   {model}")
    print(f"skybox    {skybox}")
    print(f"materiali {args.materials}" + ("" if args.materials == "gt" else f" (sorgente {args.source})"))
    print(f"output    {out_dir}")

    resolved = _resolve_maps(run_dir, tf_ext, manifest, args.materials, args.source)
    print("mappe:")
    for k, v in resolved["maps"].items():
        print(f"  {k:11s} {v}")

    # Quali frame mancano davvero.  La decisione sta qui e non dentro Blender, che si
    # limita a renderizzare la lista che riceve.
    stems = [Path(f["file_path"]).stem for f in tf_ext["frames"]]
    if args.frames:
        unknown = sorted(set(args.frames) - set(stems))
        if unknown:
            raise SystemExit(f"✗ stem non presenti in transforms_extended.json: {unknown}")
        stems = [s for s in stems if s in set(args.frames)]
    if args.limit > 0:
        stems = stems[:args.limit]
    todo = stems if args.force else [s for s in stems if not (out_images / f"{s}.exr").exists()]
    skipped = len(stems) - len(todo)
    print(f"frame     {len(todo)} da renderizzare"
          + (f", {skipped} gia' su disco (--force per rifarli)" if skipped else ""))
    if not todo:
        print("niente da fare.")
        return 0

    out_images.mkdir(parents=True, exist_ok=True)
    if args.materials == "pbr":
        print("materiale ricostruito:")
        _report_material_stats(run_dir, resolved["maps"])
    print("texture:")
    maps = _prepare_textures(run_dir, out_dir, resolved["maps"],
                             resolved["reconstructed"], args.dilate)

    job = {
        "run_dir": str(run_dir),
        "transforms": str(tf_path),
        "model": str(model),
        "skybox": str(skybox),
        "materials": args.materials,
        "source": args.source,
        "maps": maps,
        "maps_original": resolved["maps"],
        "dilate": args.dilate,
        "samples": args.samples,
        "device": args.device,
        "no_normal": args.no_normal,
        "save_blend": args.save_blend,
        "out_dir": str(out_dir),
        "frames": todo,
        "script_dir": str(Path(__file__).resolve().parent),
    }
    job_path = out_dir / "rerender_job.json"
    job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")

    if args.dry_run:
        print(f"\n--dry-run: job scritto in {job_path}, Blender non lanciato.")
        return 0

    # --factory-startup: niente addon dell'utente (blenderkit stampa parecchio) e niente
    # preferenze locali, cosi' il render dipende solo da questo script.  I device GPU
    # vengono riabilitati esplicitamente da enable_gpu_rendering().
    cmd = [str(blender), "--background", "--factory-startup",
           "--python", str(Path(__file__).resolve()), "--", str(job_path)]
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd)
    dt = time.time() - t0
    print(f"\nBlender uscito con {rc} dopo {dt / 60.0:.1f} min")
    if rc != 0:
        return rc

    done = sorted(p.name for p in out_images.glob("*.exr"))
    print(f"{len(done)} immagini in {out_images}")
    print(f"\nConfronto:\n  python compare_exr.py --original {run_dir / 'images'} "
          f"--computed {out_images} --output {out_dir / 'compare'}")
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# DENTRO BLENDER
# ══════════════════════════════════════════════════════════════════════════════

def _principled(nt):
    """Il nodo Principled dell'albero, creandolo e collegandolo se manca."""
    bsdf = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)
    if bsdf is not None:
        return bsdf
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    out = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial"), None)
    if out is None:
        out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return bsdf


def build_material(job: dict):
    """Replica il grafo di `BakedMaterial` con le mappe indicate dal job.

    Normal Map in spazio OBJECT: e' cosi' che la normale e' bakeata, ed e' la stessa che
    la pipeline ha consumato in ingresso.  Tenerla identica al GT fa si' che il confronto
    isoli l'errore dei materiali, invece di mescolarlo con la perdita di dettaglio di una
    normale che la pipeline non ricostruisce.
    """
    maps = job["maps"]
    mat = bpy.data.materials.new("RerenderMaterial")
    if mat.node_tree is None:
        mat.use_nodes = True
    nt = mat.node_tree
    bsdf = _principled(nt)

    def tex(path: str, colorspace: str, y: int):
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = bpy.data.images.load(path, check_existing=True)
        n.image.colorspace_settings.name = colorspace
        n.location = (-600, y)
        return n

    # Base color: taggato come colore, non come dato.  Vedi il docstring in testa.
    nt.links.new(tex(maps["base_color"], CS_COLOR, 400).outputs["Color"],
                 bsdf.inputs["Base Color"])

    if "metallic" in maps:
        nt.links.new(tex(maps["metallic"], CS_DATA, 100).outputs["Color"],
                     bsdf.inputs["Metallic"])
        # La roughness ricostruita entra COSI' COM'E'.  E' una scelta, non una
        # derivazione: il file contiene apertura_del_cono/180 del cono vincente, mentre
        # Blender legge questo input come roughness GGX.  Gli estremi coincidono
        # (0 = specchio, 1 = massimamente ruvido), il centro no.  Una eventuale
        # calibrazione apertura -> alpha va innestata qui, fra la texture e l'input.
        nt.links.new(tex(maps["roughness"], CS_DATA, -200).outputs["Color"],
                     bsdf.inputs["Roughness"])
    else:
        # lambert: nessuna componente speculare da ricostruire
        bsdf.inputs["Metallic"].default_value = 0.0
        bsdf.inputs["Roughness"].default_value = 1.0

    if not job["no_normal"]:
        t = tex(maps["normal"], CS_DATA, -500)
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.space = "OBJECT"
        nm.inputs["Strength"].default_value = 1.0
        nm.location = (-300, -500)
        nt.links.new(t.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def setup_world(skybox: str) -> None:
    """World con la sola environment texture, strength 1, nessun nodo di mapping:
    esattamente il World dei blend sorgente."""
    world = bpy.data.worlds.new("RerenderWorld")
    bpy.context.scene.world = world
    if world.node_tree is None:
        world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    bg = nt.nodes.new("ShaderNodeBackground")
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    out = nt.nodes.new("ShaderNodeOutputWorld")
    env.image = bpy.data.images.load(skybox, check_existing=True)
    env.image.colorspace_settings.name = CS_DATA
    bg.inputs["Strength"].default_value = GT_BG_STRENGTH
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])


def check_skybox(job: dict) -> None:
    """Avvisa se la skybox passata non e' quella del `Baked.blend` sorgente.

    Sbagliarla e' il modo piu' facile di invalidare il confronto senza accorgersene:
    notturna e studio condividono modello, camere e layout, e il render *sembra* corretto.
    Confronto solo il basename, perche' nel blend il path e' relativo alla sua cartella.
    Segnala e basta, non sceglie al posto dell'utente.
    """
    from pathlib import Path as P
    cands = [P(job["maps_original"]["normal"]).parent / "Baked.blend",
             P(job["model"]).parent / "Baked.blend"]
    blend = next((c for c in cands if c.exists()), None)
    if blend is None:
        print("[skybox] nessun Baked.blend sorgente trovato: guardia saltata")
        return
    try:
        # Dopo il blocco `with`, dst.worlds contiene i datablock appena caricati (dentro
        # sono ancora nomi).  Vanno letti da li' e non cercati per nome in bpy.data: un
        # World che si chiama gia' "World" verrebbe rinominato in "World.001".
        with bpy.data.libraries.load(str(blend), link=False) as (src, dst):
            dst.worlds = list(src.worlds)[:1]
        loaded = [w for w in dst.worlds if w is not None]
    except Exception as e:                        # noqa: BLE001
        print(f"[skybox] non ho potuto leggere {blend}: {e}")
        return
    found = None
    for world in loaded:
        if world.node_tree is not None:
            for n in world.node_tree.nodes:
                if n.bl_idname == "ShaderNodeTexEnvironment" and n.image:
                    found = P(n.image.filepath.replace("\\", "/")).name
        bpy.data.worlds.remove(world)
    want = P(job["skybox"]).name
    if found is None:
        print(f"[skybox] {blend.name} non ha una environment texture: guardia inefficace")
    elif found.lower() == want.lower():
        print(f"[skybox] ok, coincide con quella di {blend.name}: {found}")
    else:
        print(f"[skybox] ⚠  ATTENZIONE: {blend.name} usa `{found}`, "
              f"tu hai passato `{want}`.\n"
              f"[skybox] ⚠  Se non e' voluto, il confronto con le immagini di training "
              f"non ha significato.")


def configure_render(scene, intr, job: dict) -> None:
    """I parametri dei Baked.blend sorgente, identici in tutte e cinque le scene."""
    r = scene.render
    r.resolution_x = intr.w
    r.resolution_y = intr.h
    r.resolution_percentage = 100
    r.engine = "CYCLES"
    r.film_transparent = False
    r.use_file_extension = True

    c = scene.cycles
    c.samples = job["samples"]
    c.use_adaptive_sampling = True
    c.adaptive_threshold = GT_ADAPTIVE_THRESHOLD
    c.use_denoising = True
    for k, v in GT_BOUNCES.items():
        setattr(c, k, v)
    c.sample_clamp_direct = GT_CLAMP_DIRECT
    c.sample_clamp_indirect = GT_CLAMP_INDIRECT

    fmt = r.image_settings
    fmt.file_format = "OPEN_EXR"
    fmt.color_mode = "RGB"
    fmt.color_depth = "32"
    fmt.exr_codec = "NONE"

    # Irrilevante sui formati float (Blender scrive lineare), ma fissarlo rende l'output
    # inequivocabile: i blend sorgente non concordano fra loro su questo campo.
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0


def blender_main(job_path: str) -> None:
    import time
    from datetime import datetime

    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    sys.path.insert(0, job["script_dir"])
    from blender_renderer import (create_camera, enable_gpu_rendering,  # noqa: E402
                                  import_model, load_transforms, set_camera_pose)

    out_dir = Path(job["out_dir"])
    out_images = out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    tf = load_transforms(job["transforms"])
    intr = tf.intrinsics
    wanted = set(job["frames"])
    frames = [f for f in tf.frames if Path(f.file_path).stem in wanted]
    print(f"[rerender] {len(frames)} frame, {intr.w}x{intr.h}, "
          f"{job['samples']} sample, materiali {job['materials']}")

    check_skybox(job)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    import_model(job["model"])

    # L'import stampa dei WARNING sulle texture del .mtl: sono path assoluti che Blender
    # ri-ancora alla cartella dell'OBJ e quindi non trova.  Innocui, il materiale
    # importato viene comunque buttato via qui sotto.
    mat = build_material(job)
    n_mesh = 0
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            obj.data.materials.clear()
            obj.data.materials.append(mat)
            n_mesh += 1
    print(f"[rerender] materiale applicato a {n_mesh} mesh")

    setup_world(job["skybox"])

    scene = bpy.context.scene
    cam_obj = create_camera(intr)
    # create_camera lascia clip_start a 0.001; i blend sorgente usano 0.1.
    cam_obj.data.clip_start = GT_CLIP_START
    cam_obj.data.clip_end = GT_CLIP_END
    scene.camera = cam_obj
    configure_render(scene, intr, job)
    if job["device"] == "GPU":
        enable_gpu_rendering()
    else:
        scene.cycles.device = "CPU"

    t0 = time.time()
    for i, frame in enumerate(frames, 1):
        stem = Path(frame.file_path).stem
        set_camera_pose(cam_obj, frame.transform_matrix, apply_axis_correction=False)
        bpy.context.view_layer.update()
        scene.render.filepath = str(out_images / stem)
        bpy.ops.render.render(write_still=True)
        el = time.time() - t0
        eta = el / i * (len(frames) - i)
        print(f"[rerender] {i}/{len(frames)} {stem}  "
              f"({el / 60.0:.1f} min, ETA {eta / 60.0:.1f} min)", flush=True)

    meta = dict(job)
    meta.pop("script_dir", None)
    meta.update({
        "blender": bpy.app.version_string,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "n_frames": len(frames),
        "wall_min": round((time.time() - t0) / 60.0, 2),
        "render": {
            "engine": "CYCLES", "samples": job["samples"],
            "adaptive_threshold": GT_ADAPTIVE_THRESHOLD, "denoising": True,
            "bounces": GT_BOUNCES,
            "clamp_direct": GT_CLAMP_DIRECT, "clamp_indirect": GT_CLAMP_INDIRECT,
            "resolution": [intr.w, intr.h], "format": "OPEN_EXR RGB 32 NONE",
            "view_transform": "Raw",
            "colorspace": {"base_color": CS_COLOR, "metallic": CS_DATA,
                           "roughness": CS_DATA, "normal": CS_DATA},
            "normal_map_space": "OBJECT",
        },
    })
    (out_dir / "rerender_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[rerender] meta -> {out_dir / 'rerender_meta.json'}")

    if job["save_blend"]:
        blend = str(out_dir / "scene.blend")
        bpy.ops.wm.save_as_mainfile(filepath=blend, check_existing=False)
        print(f"[rerender] scene.blend -> {blend}")


if __name__ == "__main__":
    if IN_BLENDER:
        argv = sys.argv
        argv = argv[argv.index("--") + 1:] if "--" in argv else []
        if len(argv) != 1:
            raise SystemExit("uso interno: blender --background --python rerender_run.py "
                             "-- <job.json>")
        blender_main(argv[0])
    else:
        raise SystemExit(launcher())
