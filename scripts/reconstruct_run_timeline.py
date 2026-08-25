#!/usr/bin/env python
"""reconstruct_run_timeline.py -- recover the wall-clock of a run already on disk.

`run_pipeline` measures every stage with a `StageTimer`, but until `run_timings.jsonl`
existed those numbers only reached TensorBoard, and the event files of the runs the
Results chapter is built on are gone. What survives is enough to rebuild the timeline:

  * the mtime of the artefacts each stage writes, which is when that stage finished;
  * `nerf_train/training_metrics.csv`, where the training measured its own `wall_s`;
  * the tqdm bars in `console.log`, which give the elapsed time of the two long bakes
    directly and therefore serve as an independent check on the mtime arithmetic.

The third point is what makes the method defensible rather than plausible, so the
validation is printed with the table and belongs in the caption of any table built from
this output. Use `run_timings.jsonl` for any run made after it existed; this script is
for the ones made before.

    python reconstruct_run_timeline.py <run_dir> [--json OUT.json] [--gap-min 30]

A stage is (name, from, to): `from` and `to` name a sentinel, and a sentinel is a glob
plus which end of its mtime range to take. `minus` subtracts another stage's duration,
for the one window that contains two stages because the run was resumed (the second
invocation overwrote the irradiance of the first, so the window that ends on the indirect
map still contains the first invocation's irradiance).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import re
from pathlib import Path

# (glob, "first" | "last"). "first" is the earliest mtime of the matching files,
# "last" the latest one.
SENTINELS = {
    "manifest":      ("run_manifest.json", "last"),
    "depth":         ("depth/*", "last"),
    "ium":           ("ium/*", "last"),
    "pixel_change":  ("sources/*/pixel_change/*", "last"),
    "ckpt":          ("model/nerf_model_cache.pt", "last"),
    "train_views":   ("nerf_render_images/*/*", "last"),
    "skybox":        ("skybox_nerf_baked.exr", "last"),
    "visibility":    ("visibility/visibility.exr", "last"),
    "irradiance":    ("irradiance/irradiance.exr", "last"),
    "indirect":      ("irradiance/irradiance_indirect.exr", "last"),
    "spec_cone":     ("spec_cone/*", "last"),
    "albedo":        ("sources/*/albedo/*", "last"),
}

# The stage table. Order is the order of the pipeline, not of the clock: a resumed run
# writes the later stages of the first invocation before the earlier stages of the second.
STAGES = [
    ("step1",                 "manifest",    "depth",       None,
     "60 frames: depth, position, normal, mask"),
    ("step2/train",           None,          None,          None,
     "from training_metrics.csv wall_s, self-measured"),
    ("step2b/train_views",    "ckpt",        "train_views", None,
     "re-render of the training views with the trained model"),
    ("step3/ium",             "depth",       "ium",         None,
     "IUM plus the resample of the external normal map"),
    ("step3/visibility+color", "ium",        "pixel_change", None,
     "visibility, colour texture, per-camera masks, pixel change"),
    ("step3/skybox",          "train_views", "skybox",      None,
     "environment map baked from the NeRF background sphere"),
    ("step3/irradiance",      "visibility",  "irradiance",  None,
     "direct irradiance, N x N samples per texel"),
    ("step3/indirect",        "skybox",      "indirect",    "step3/irradiance",
     "indirect irradiance; the window also holds the irradiance of the first invocation"),
    ("step3/spec_cone",       "irradiance",  "spec_cone",   None,
     "shared-ray specular cone bake, all cameras"),
    ("step4",                 "spec_cone",   "albedo",      None,
     "PBR fit plus Lambertian albedo"),
]

# tqdm bars that close a stage, for the independent check. The bar prints the elapsed
# time of the loop, which excludes the set-up but covers almost all of a long bake.
TQDM_CHECK = {
    "step3/spec_cone": "tile",
    "step4": "tile",
}
# tqdm drops the hours field on a bar shorter than an hour, so they are optional.
BAR = re.compile(r"100%\|[^|]*\|\s*(\d+)/(\d+)\s*\[(?:(\d+):)?(\d+):(\d\d)<")


def mtimes(run: Path, glob: str) -> list[float]:
    return sorted(p.stat().st_mtime for p in run.glob(glob) if p.is_file())


def sentinel_time(run: Path, key: str) -> float | None:
    glob, end = SENTINELS[key]
    ts = mtimes(run, glob)
    if not ts:
        return None
    return ts[-1] if end == "last" else ts[0]


def training_wall_s(run: Path) -> tuple[float | None, str]:
    p = run / "nerf_train" / "training_metrics.csv"
    if not p.exists():
        return None, "training_metrics.csv missing"
    last = None
    with open(p, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            last = row
    if last is None or "wall_s" not in last:
        return None, "no wall_s column"
    return float(last["wall_s"]), f"iter {last.get('iter')} at {last.get('timestamp')}"


def tqdm_elapsed(run: Path) -> list[tuple[int, float]]:
    """[(total iterations of the bar, elapsed seconds)] for every completed tqdm bar."""
    p = run / "console.log"
    if not p.exists():
        return []
    out = []
    with open(p, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            for m in BAR.finditer(line):
                hours = int(m.group(3)) if m.group(3) else 0
                out.append((int(m.group(2)),
                            hours * 3600 + int(m.group(4)) * 60 + int(m.group(5))))
    # tqdm rewrites the same bar many times; keep the longest reading per total.
    best: dict[int, float] = {}
    for total, secs in out:
        best[total] = max(best.get(total, 0.0), secs)
    return sorted(best.items())


def invocations(run: Path, gap_min: float) -> list[tuple[float, float]]:
    """Clusters of mtimes separated by more than `gap_min` minutes.

    A run split over several invocations cannot be summed as one window: this is what
    tells the reader which windows belong together.
    """
    ts = sorted({t for g in
                 ("*", "*/*", "*/*/*", "sources/*/*/*") for t in mtimes(run, g)})
    if not ts:
        return []
    out, start, prev = [], ts[0], ts[0]
    for t in ts[1:]:
        if t - prev > gap_min * 60:
            out.append((start, prev))
            start = t
        prev = t
    out.append((start, prev))
    return out


def fmt(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).strftime("%m-%d %H:%M:%S")


def hms(s: float) -> str:
    return f"{int(s) // 3600:d}:{(int(s) % 3600) // 60:02d}:{int(s) % 60:02d}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--json", default="")
    ap.add_argument("--gap-min", type=float, default=30.0)
    a = ap.parse_args()
    run = Path(a.run_dir).resolve()

    print(f"run: {run}")
    inv = invocations(run, a.gap_min)
    print(f"invocations (gaps longer than {a.gap_min:g} min): {len(inv)}")
    for i, (t0, t1) in enumerate(inv, 1):
        print(f"  {i}. {fmt(t0)} -> {fmt(t1)}   ({hms(t1 - t0)})")
    print()

    train_s, train_note = training_wall_s(run)
    durations: dict[str, float] = {}
    rows = []
    for name, src, dst, minus, note in STAGES:
        if src is None:
            secs, span = train_s, train_note
        else:
            t0, t1 = sentinel_time(run, src), sentinel_time(run, dst)
            if t0 is None or t1 is None:
                rows.append((name, None, "sentinel missing", note))
                continue
            secs = t1 - t0
            span = f"{fmt(t0)} -> {fmt(t1)}"
            if minus:
                sub = durations.get(minus)
                if sub is not None:
                    secs -= sub
                    span += f"  minus {minus}"
        if secs is None:
            rows.append((name, None, span, note))
            continue
        durations[name] = secs
        rows.append((name, secs, span, note))

    total = sum(v for v in durations.values())
    print(f"{'stage':<24}{'seconds':>10}{'h:mm:ss':>10}{'share':>8}  window")
    for name, secs, span, _note in rows:
        if secs is None:
            print(f"{name:<24}{'-':>10}{'-':>10}{'-':>8}  {span}")
            continue
        print(f"{name:<24}{secs:>10.1f}{hms(secs):>10}"
              f"{100 * secs / total:>7.1f}%  {span}")
    print(f"{'TOTAL':<24}{total:>10.1f}{hms(total):>10}{100.0:>7.1f}%")
    print()

    bars = tqdm_elapsed(run)
    if bars:
        print("independent check, tqdm bars in console.log (elapsed of the loop):")
        for total_it, secs in bars:
            print(f"  bar over {total_it:>6} iterations: {hms(secs)}  ({secs:.0f} s)")
        print("  compare against the stages that end on a bar "
              f"({', '.join(TQDM_CHECK)}): a mtime window that matches its own bar to "
              "within a few seconds is evidence the arithmetic is right.")
    else:
        print("no completed tqdm bar found in console.log: no independent check.")

    if a.json:
        rec = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "status": "ok",
            "source": "mtime-reconstruction",
            "output_dir": str(run),
            "roi_tag": None,
            "roi_rect": None,
            "total_s": total,
            "steps": {k: v for k, v in durations.items() if "/" not in k},
            "substeps": {k: v for k, v in durations.items() if "/" in k},
            "invocations": [[fmt(t0), fmt(t1)] for t0, t1 in inv],
            "tqdm_bars_s": {str(k): v for k, v in bars},
        }
        Path(a.json).write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print()
        print(f"  -> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
