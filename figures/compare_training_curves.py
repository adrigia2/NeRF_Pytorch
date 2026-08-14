#!/usr/bin/env python
"""Compare the training curves of two sweeps that differ by one hyper-parameter.

Written to answer "does the NeRF still train without the density noise?": it pairs
`<root>/<config>/<scene>/nerf_train/training_metrics.csv` across two roots and overlays
the curves. `compare_runs.py` compares runs INSIDE a single sweep (and wants the Step 2b
artefacts), so it does not cover this case.

The comparison only makes sense when the two runs share seed, data and LR schedule: in
that case the first N iterations of the long run are, row by row, the control arm of the
short one. The script checks that on `lr` and warns when they diverge.

Usage:
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
# CSV rows to average around each checkpoint. The loss of a single row is the mean of
# display_every iterations over ONE batch, and it swings by 50-120 % between neighbouring
# rows: comparing two runs at a single point measures that swing, not the difference
# between the runs. A median over a window removes it.
WINDOW = 20


def read_curve(path: Path, max_iter: int) -> "list[dict]":
    """Rows up to max_iter, from the first continuous segment only.

    The CSV is append-only: a resume or an interactive continuation restarts with wall_s
    near 0 and repeats the same `iter` values. Taking the first increasing segment avoids
    mixing two different stretches of training.
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
    """Median of `key` over the `window` rows ending at iteration `it`.

    Not the value at row `it`: that is a mean over display_every iterations of a single
    batch and swings far more than the difference between two runs.
    """
    if not rows or int(rows[-1]["iter"]) < it:
        return float("nan")   # the run never reached the checkpoint: nothing to report
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
    ap.add_argument("ref_root", help="reference sweep (the long runs)")
    ap.add_argument("new_root", help="the new sweep")
    ap.add_argument("-o", "--out", default=None,
                    help="figure folder (default: <new_root>/curve_comparison)")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--ref-label", default="with noise")
    ap.add_argument("--new-label", default="without noise")
    args = ap.parse_args()

    ref_root, new_root = Path(args.ref_root).resolve(), Path(args.new_root).resolve()
    out = Path(args.out) if args.out else new_root / "curve_comparison"
    out.mkdir(parents=True, exist_ok=True)

    pairs = []
    for csv_new in sorted(new_root.glob("*/*/nerf_train/training_metrics.csv")):
        scene = csv_new.parent.parent.name
        conf = csv_new.parent.parent.parent.name
        csv_ref = ref_root / conf / scene / "nerf_train" / "training_metrics.csv"
        if csv_ref.exists():
            pairs.append((conf, scene, csv_ref, csv_new))
        else:
            print(f"  ⚠  no reference for {conf}/{scene}")
    if not pairs:
        print("✗ no pair found")
        return 2

    print(f"{len(pairs)} pairs, truncated at {args.iters} iterations\n")
    hdr = (f"{'config/scene':52} {'iter':>6} "
           f"{args.ref_label:>13} {args.new_label:>13} {'Δ%':>8}   "
           f"{'psnr ref':>9} {'psnr new':>9} {'acc_fg new':>10}")
    print(hdr)
    print("-" * len(hdr))

    for conf, scene, p_ref, p_new in pairs:
        ref = read_curve(p_ref, args.iters)
        new = read_curve(p_new, args.iters)
        if not ref or not new:
            print(f"{conf}/{scene}: empty CSV, skipped")
            continue

        # The LR schedule has to match, otherwise the curves are not comparable
        lr_ref = {int(r["iter"]): fnum(r, "lr") for r in ref}
        bad_lr = [i for r in new if (i := int(r["iter"])) in lr_ref
                  and abs(fnum(r, "lr") - lr_ref[i]) > 1e-12]
        if bad_lr:
            print(f"  ⚠  {conf}/{scene}: lr differs at {len(bad_lr)} iterations "
                  f"(first: {bad_lr[0]}) — curves not directly comparable")

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

        # ── figure: loss, psnr, acc_fg ───────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        series = [("loss", "loss (%s)" % ref[0].get("loss_type", "?"), True),
                  ("psnr_db", "psnr_db", False),
                  ("acc_fg", "acc_fg (foreground opacity)", False)]
        for ax, (col, title, logy) in zip(axes, series):
            for rows, lab, style in ((ref, args.ref_label, "-"),
                                     (new, args.new_label, "--")):
                xs = [int(r["iter"]) for r in rows]
                ys = [fnum(r, col) for r in rows]
                ax.plot(xs, ys, style, label=lab, linewidth=1.2)
            ax.set_title(title)
            ax.set_xlabel("iteration")
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

    print(f"\n✓ figures in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
