#!/usr/bin/env python
"""make_ium_normal_figure.py -- Recupera la normale geometrica del pass IUM (figura 3.7b).

    python make_ium_normal_figure.py <run_dir> --out ../Doc/images/ium

Scrive `ium_normal.png`: la normale di faccia che `IUM_Generator` calcola sulla GPU.

**Perche' serve uno script apposta.**  `ium_normals.exr` su disco NON contiene questa
mappa.  Quando la run dichiara una normal map esterna, `_apply_external_normal`
(images_generator.py:638, chiamata a :3098) la sovrascrive host-side *dopo* il render:
il file finale porta la mappa fornita, non quella calcolata dal tracer.  La normale
geometrica esiste quindi solo per il tempo di una chiamata e viene scartata.  Qui si
rifa' il solo pass IUM, senza NeRF e senza mappa esterna, e la si legge prima che
qualcuno la sovrascriva.

Mesh e risoluzione dell'atlante vengono dal `run_manifest.json` della run indicata, non
trascritte a mano: se non coincidessero con quelle della run, il pannello non sarebbe
allineato agli altri due della stessa figura e il confronto sarebbe falso.

Come riconoscere che il risultato e' davvero quello geometrico: il pass calcola normali
di FACCIA, quindi il cubo deve mostrare tinte piatte e nette e la sfera una sfaccettatura
visibile.  La mappa esterna e' bakeata smooth e darebbe gradienti continui ovunque.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from make_depth_figure import normal_rgb, save_png     # noqa: E402
from make_skybox_figure import load_exr                # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="run da cui leggere mesh e risoluzione dell'atlante")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    run, out = Path(args.run_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(run / "run_manifest.json", encoding="utf-8") as fh:
        manifest = json.load(fh)
    model_path = manifest["scene"]["model_path"]
    ium_w, ium_h = manifest["config"]["render"]["ium_texture_size"]
    ext = manifest["scene"].get("external_normal_path")
    print(f"mesh      : {model_path}")
    print(f"atlante   : {ium_w}x{ium_h}")
    print(f"mappa esterna della run (qui NON applicata): {ext}")

    import OptixProgrammablePasses as optix

    model = optix.TriangleMesh()
    model.add_from_obj_file(model_path)

    gen = optix.IUMGenerator()
    gen.set_traversable(model)
    gen.set_texture_size([ium_w, ium_h])
    gen.render()
    res = gen.get_result()          # tenere vivo: le viste *_np sono zero-copy

    nrm = np.array(res.normals_np, dtype=np.float32).reshape(ium_h, ium_w, 3)
    mask = load_exr(run / "ium" / "ium_masks.exr")[..., 0] > 0.5
    print(f"copertura : {100.0 * mask.mean():.2f}% dei texel")

    save_png(normal_rgb(nrm, mask), out / "ium_normal.png")

    # La mappa esterna, cioe' quella che sovrascrive la geometrica e che ogni consumatore
    # legge davvero.  Sul disco e' gia' decodificata (per canale in [-1,1], |n| = 1 sulla
    # maschera), quindi passa per la stessa `normal_rgb` e i due pannelli sono confrontabili.
    # E' anche la versione che la pipeline usa, dopo ricampionamento e conversione di
    # range, non il file sorgente nella cartella della scena.
    disk = load_exr(run / "ium" / "ium_normals.exr")
    save_png(normal_rgb(disk, mask), out / "ium_normal_external.png")

    # Che i due pannelli non siano lo stesso file non e' un dettaglio: confonderli
    # renderebbe la figura una bugia difficile da individuare.
    d = np.abs(disk - nrm)[mask]
    print(f"scarto fra geometrica ed esterna: max {d.max():.4f}, media {d.mean():.4f}  "
          f"→ {'DIVERSE (atteso)' if d.max() > 1e-3 else 'UGUALI (sospetto)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
