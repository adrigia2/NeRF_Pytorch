#!/usr/bin/env python
"""Confronta le curve di training di due sweep che differiscono per un iperparametro.

Nato per rispondere a "il NeRF si allena anche senza il rumore sulla densità?":
appaia `<root>/<config>/<scena>/nerf_train/training_metrics.csv` fra due radici e
sovrappone le curve. `compare_runs.py` confronta run DENTRO un unico sweep (e vuole
gli artefatti dello Step 2b), quindi non copre questo caso.

Il confronto ha senso solo se le due run condividono seed, dati e schedule del LR:
in quel caso le prime N iterazioni della run lunga sono, riga per riga, il braccio
di controllo di quella corta. Lo script lo verifica su `lr` e avvisa se divergono.

Uso:
    python compare_training_curves.py <ref_root> <new_root> [-o OUT] [--iters 10000]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CHECKPOINTS = (100, 1000, 2500, 5000, 10000)
# Righe di CSV su cui mediare attorno a ogni checkpoint. La loss di una singola
# riga è la media di display_every iterazioni su UN batch, e fra righe vicine
# oscilla del 50-120%: confrontare due run su un punto solo misura quell'oscillazione,
# non la differenza fra le run. La mediana su una finestra la toglie di mezzo.
WINDOW = 20


def read_curve(path: Path, max_iter: int) -> "list[dict]":
    """Righe fino a max_iter, solo il primo segmento continuo.

    La CSV è append-only: un resume o una continuazione interattiva riparte con
    wall_s vicino a 0 e reitera gli stessi `iter`. Prendere il primo segmento
    crescente evita di mescolare due tratti di training diversi.
    """
    rows = []
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            it = int(r["iter"])
            if rows and it <= int(rows[-1]["iter"]):
                break
            if it > max_iter:
                break
            rows.append(r)
    return rows


def fnum(row: dict, key: str):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def window_median(rows: "list[dict]", it: int, key: str, window: int = WINDOW) -> float:
    """Mediana di `key` sulle `window` righe che terminano all'iterazione `it`.

    Non il valore alla riga `it`: quello è una media su display_every iterazioni di
    un solo batch e oscilla molto più della differenza fra due run.
    """
    if not rows or int(rows[-1]["iter"]) < it:
        return float("nan")   # la run non è arrivata al checkpoint: niente da riportare
    idx = [i for i, r in enumerate(rows) if int(r["iter"]) <= it]
    if not idx:
        return float("nan")
    sel = rows[max(0, idx[-1] + 1 - window): idx[-1] + 1]
    vals = [v for r in sel if (v := fnum(r, key)) == v]
    if not vals:
        return float("nan")
    vals.sort()
    n = len(vals)
    return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref_root", help="sweep di riferimento (run lunghe)")
    ap.add_argument("new_root", help="sweep nuovo")
    ap.add_argument("-o", "--out", default=None,
                    help="cartella figure (default: <new_root>/curve_confronto)")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--ref-label", default="con rumore")
    ap.add_argument("--new-label", default="senza rumore")
    args = ap.parse_args()

    ref_root, new_root = Path(args.ref_root).resolve(), Path(args.new_root).resolve()
    out = Path(args.out) if args.out else new_root / "curve_confronto"
    out.mkdir(parents=True, exist_ok=True)

    pairs = []
    for csv_new in sorted(new_root.glob("*/*/nerf_train/training_metrics.csv")):
        scene = csv_new.parent.parent.name
        conf = csv_new.parent.parent.parent.name
        csv_ref = ref_root / conf / scene / "nerf_train" / "training_metrics.csv"
        if csv_ref.exists():
            pairs.append((conf, scene, csv_ref, csv_new))
        else:
            print(f"  ⚠  nessun riferimento per {conf}/{scene}")
    if not pairs:
        print("✗ nessuna coppia trovata")
        return 2

    print(f"{len(pairs)} coppie, troncate a {args.iters} iterazioni\n")
    hdr = (f"{'config/scena':52} {'iter':>6} "
           f"{args.ref_label:>13} {args.new_label:>13} {'Δ%':>8}   "
           f"{'psnr rif':>9} {'psnr new':>9} {'acc_fg new':>10}")
    print(hdr)
    print("-" * len(hdr))

    for conf, scene, p_ref, p_new in pairs:
        ref = read_curve(p_ref, args.iters)
        new = read_curve(p_new, args.iters)
        if not ref or not new:
            print(f"{conf}/{scene}: CSV vuota, skip")
            continue

        # Lo schedule del LR deve coincidere, altrimenti le curve non sono confrontabili
        lr_ref = {int(r["iter"]): fnum(r, "lr") for r in ref}
        bad_lr = [i for r in new if (i := int(r["iter"])) in lr_ref
                  and abs(fnum(r, "lr") - lr_ref[i]) > 1e-12]
        if bad_lr:
            print(f"  ⚠  {conf}/{scene}: lr diverso a {len(bad_lr)} iterazioni "
                  f"(prima: {bad_lr[0]}) — curve non direttamente confrontabili")

        label = f"{conf}/{scene}"
        for k, it in enumerate(CHECKPOINTS):
            a = window_median(ref, it, "loss")
            b = window_median(new, it, "loss")
            if a != a or b != b:
                continue
            d = 100.0 * (b - a) / a if a else float("nan")
            print(f"{(label if k == 0 else ''):52} {it:>6} "
                  f"{a:>13.6g} {b:>13.6g} {d:>+7.1f}%   "
                  f"{window_median(ref, it, 'psnr_db'):>9.2f} "
                  f"{window_median(new, it, 'psnr_db'):>9.2f} "
                  f"{window_median(new, it, 'acc_fg'):>10.4f}")

        # ── figura: loss, psnr, acc_fg ───────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        series = [("loss", "loss (%s)" % ref[0].get("loss_type", "?"), True),
                  ("psnr_db", "psnr_db", False),
                  ("acc_fg", "acc_fg (opacità figura)", False)]
        for ax, (col, title, logy) in zip(axes, series):
            for rows, lab, style in ((ref, args.ref_label, "-"),
                                     (new, args.new_label, "--")):
                xs = [int(r["iter"]) for r in rows]
                ys = [fnum(r, col) for r in rows]
                ax.plot(xs, ys, style, label=lab, linewidth=1.2)
            ax.set_title(title)
            ax.set_xlabel("iterazione")
            if logy:
                ax.set_yscale("log")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle(f"{conf} — {scene}   (act={ref[0].get('rgb_activation')}, "
                     f"loss={ref[0].get('loss_type')})")
        fig.tight_layout()
        png = out / f"{conf}__{scene}.png"
        fig.savefig(png, dpi=110)
        plt.close(fig)

    print(f"\n✓ figure in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
