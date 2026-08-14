#!/usr/bin/env python
"""Compare a run restricted to a ROI with the full run it derives from.

It checks the property that justifies the ROI: **inside** the ROI the texels must receive
the same values as the full run, **outside** everything must stay zero.

The criterion is not uniform, and the distinction is the informative content of the test:

  A  DETERMINISTIC, must be bit-identical.  Per-texel passes with no NeRF query:
     IUM, visibility, color texture, camera_texture, camera_mask, pixel_change,
     irradiance, n_views. A single difference here is a bug in the ROI.
  B  STOCHASTIC by construction.  Everything that goes through query_radiance:
     indirect and spec_cone. raw2outputs adds Gaussian noise to the density and
     (nerf/rays.py:68, `noise = torch.randn_like(...) * raw_noise_std`) e
     raw_noise_std comes from the checkpoint WITHOUT an eval guard, so two identical
     executions of the same bake already give different maps.
  C  like B, propagated: maps derived from the PBR fit.
  D  like B, but passed through an argmin over 14 candidates: roughness and lobe_param
     saltano da un candidato all'altro dove due residui quasi pareggiano.

Only group A gives a pass/fail verdict on the values; for B, C and D the comparison with
the full run has no sensible a-priori threshold. To give it one, pass
`--reference <other-tag>`: two sandboxes generated with the SAME ROI measure the
intrinsic noise, and that is the yardstick for reading the full-vs-ROI column.

The "everything zero outside the ROI" check is deterministic instead, and holds for every
group: a non-zero value there is always a bug.

Uso:
    python compare_roi_run.py <run_dir> [--tag TAG] [--reference full|TAG2]
                              [--cams all|0,12,24] [--source gt]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import _paths  # noqa: F401

from pbr_solver import _ExrBandReader  # noqa: E402

GROUP_DESC = {
    "A": "deterministico, bit-identico",
    "B": "stocastico (raw_noise_std)",
    "C": "derivate dal fit",
    "D": "argmin fra candidati",
}
# Only group A has a threshold on the values: the others are stochastic in themselves.
STRICT = {"A"}

_SAMPLE_PER_BAND = 60_000    # for the percentiles of the relative error, without keeping everything in RAM


class Stats:
    """Band accumulator: inside the ROI the comparison, outside the zero check."""

    def __init__(self, seed: int = 0) -> None:
        self.n = 0
        self.n_exact = 0
        self.max_abs = 0.0
        self.ref_peak = 0.0
        self.max_rel = 0.0
        self.n_outside = 0
        self.max_outside = 0.0
        self._rel: list[np.ndarray] = []
        self._rng = np.random.default_rng(seed)

    def update(self, a: np.ndarray, b: np.ndarray) -> None:
        if a.size == 0:
            return
        both_nan = np.isnan(a) & np.isnan(b)
        exact = (a == b) | both_nan
        self.n += a.size
        self.n_exact += int(exact.sum())
        d = np.abs(a - b)
        d[both_nan] = 0.0
        self.max_abs = max(self.max_abs, float(np.nanmax(d)) if d.size else 0.0)
        peak = float(np.abs(a[~np.isnan(a)]).max()) if (~np.isnan(a)).any() else 0.0
        self.ref_peak = max(self.ref_peak, peak)
        # Element-wise relative error, only where the reference is significantly
        # different from zero: below that threshold the ratio blows up without saying
        # anything, and the absolute error is already reported. The maximum stays
        # dominated by small denominators anyway, so the percentiles are kept too.
        floor = max(peak * 1e-6, 1e-30)
        sig = np.abs(a) > floor
        if sig.any():
            rel = d[sig] / np.abs(a[sig])
            self.max_rel = max(self.max_rel, float(rel.max()))
            k = min(rel.size, _SAMPLE_PER_BAND)
            self._rel.append(rel if k == rel.size
                             else rel[self._rng.integers(0, rel.size, k)])

    def update_outside(self, b: np.ndarray) -> None:
        if b.size == 0:
            return
        self.n_outside += b.size
        self.max_outside = max(self.max_outside, float(np.abs(b).max()))

    @property
    def frac_exact(self) -> float:
        return self.n_exact / self.n if self.n else 1.0

    def percentiles(self) -> "tuple[float, float]":
        """(median, p99) of the relative error, from a bounded sample."""
        if not self._rel:
            return 0.0, 0.0
        s = np.concatenate(self._rel)
        return float(np.median(s)), float(np.percentile(s, 99))


def compare_file(full_p: Path, roi_p: Path, roi_rows: np.ndarray,
                 scope: str, band: int = 128) -> Stats:
    """Compare two EXRs band by band. `roi_rows` is the (H, W) mask of the ROI."""
    st = Stats()
    with _ExrBandReader(full_p) as ra, _ExrBandReader(roi_p) as rb:
        if (ra.height, ra.width) != (rb.height, rb.width):
            raise ValueError(f"{roi_p.name}: dimensioni diverse "
                             f"{ra.width}x{ra.height} vs {rb.width}x{rb.height}")
        if ra.names != rb.names:
            raise ValueError(f"{roi_p.name}: canali diversi\n  full={ra.names}\n  roi ={rb.names}")
        H = ra.height
        for y0 in range(0, H, band):
            r = min(band, H - y0)
            a = ra.read(y0, r)
            b = rb.read(y0, r)
            if a.ndim == 1:
                a, b = a[:, None], b[:, None]
            inside = roi_rows[y0:y0 + r].reshape(-1)
            if scope == "everywhere":
                st.update(a, b)
                continue
            if inside.any():
                st.update(a[inside], b[inside])
            out = ~inside
            if out.any():
                st.update_outside(b[out])
    return st


def build_targets(full: Path, roi: Path, source: str, cams: "list[str] | None") -> list:
    """(relpath, group, scope) of everything that has to be compared."""
    t: list[tuple[str, str, str]] = []
    s = f"sources/{source}"

    # ium_positions/normals are identical over the WHOLE texture: the ROI is applied
    # after the external normal is injected, so only the mask carries it.
    t += [("ium/ium_positions.exr", "A", "everywhere"),
          ("ium/ium_normals.exr",   "A", "everywhere"),
          ("ium/ium_masks.exr",     "A", "roi"),
          ("visibility/visibility.exr", "A", "roi"),
          (f"{s}/color_texture/color_texture.exr", "A", "roi"),
          ("irradiance/irradiance.exr", "A", "roi"),
          (f"{s}/pbr/n_views.exr", "A", "roi")]
    t += [(f"{s}/pixel_change/{n}.exr", "A", "roi")
          for n in ("color_min", "color_max", "color_range", "color_variance")]

    def per_cam(subdir: str, pattern: str, group: str) -> list:
        files = sorted((roi / subdir).glob(pattern))
        if cams is not None:
            files = [p for p in files if any(c in p.stem for c in cams)]
        return [(f"{subdir}/{p.name}", group, "roi") for p in files]

    t += per_cam("camera_mask", "*.exr", "A")
    t += per_cam(f"{s}/camera_texture", "*.exr", "A")

    t += [("irradiance/irradiance_indirect.exr", "B", "roi")]
    t += per_cam("spec_cone", "cam_*.exr", "B")

    t += [(f"{s}/metallic/metallic.exr",     "C", "roi"),
          (f"{s}/albedo_pbr/albedo_pbr.exr", "C", "roi"),
          (f"{s}/albedo/albedo.exr",         "C", "roi"),
          (f"{s}/pbr/diffuse_weight.exr",    "C", "roi"),
          (f"{s}/pbr/diffuse_term.exr",      "C", "roi"),
          (f"{s}/pbr/residual.exr",          "C", "roi")]
    t += [(f"{s}/roughness/roughness.exr", "D", "roi"),
          (f"{s}/pbr/lobe_param.exr",      "D", "roi")]
    return t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="the full reference run (holds roi/<tag>/)")
    ap.add_argument("--tag", default="", help="sandbox da confrontare (default: unica presente)")
    ap.add_argument("--reference", default="full",
                    help="'full' (default) or the tag of a second sandbox with the "
                         "SAME ROI, to measure the intrinsic noise")
    ap.add_argument("--source", default="gt")
    ap.add_argument("--cams", default="all",
                    help="'all' or a list of indices, e.g. 0,12,24 (filters the per-camera files)")
    ap.add_argument("--band", type=int, default=128, help="scanlines per read band")
    args = ap.parse_args()

    full = Path(args.run_dir).resolve()
    roi_root = full / "roi"
    if args.tag:
        roi = roi_root / args.tag
    else:
        tags = sorted(p for p in roi_root.glob("*") if p.is_dir())
        if len(tags) != 1:
            print(f"✗ specificare --tag: sandbox trovate {[p.name for p in tags]}")
            return 2
        roi = tags[0]
    if not roi.is_dir():
        print(f"✗ sandbox not found: {roi}")
        return 2

    with open(roi / "roi.json", encoding="utf-8") as fh:
        fp = json.load(fh)
    W, H = fp["ium_size"]
    mask2d = np.zeros((H, W), dtype=bool)
    if fp.get("rect"):
        x0, y0, w, h = fp["rect"]
        mask2d[max(y0, 0):min(y0 + h, H), max(x0, 0):min(x0 + w, W)] = True
    if fp.get("mask_path"):
        # The resolved ROI cannot be rebuilt from the rect alone: if there was also an
        # image mask, the "outside" comparison would stay correct but the "inside" one
        # would include texels the ROI excluded (and which are 0).
        print(f"  ⚠  the ROI includes the mask {fp['mask_path']}: the comparison "
              f"'dentro' usa il solo rettangolo, quindi è più severo del dovuto")
    if not fp.get("rect") and not fp.get("mask_path"):
        print("✗ roi.json describes no ROI")
        return 2

    cams = None if args.cams == "all" else [c.strip() for c in args.cams.split(",") if c.strip()]

    # The reference is the full run, or a second sandbox with the same ROI
    # (in that case the table measures the intrinsic noise, not the ROI's effect).
    if args.reference == "full":
        ref, ref_label = full, "run piena"
    else:
        ref = roi_root / args.reference
        if not ref.is_dir():
            print(f"✗ reference sandbox not found: {ref}")
            return 2
        with open(ref / "roi.json", encoding="utf-8") as fh:
            ref_fp = json.load(fh)
        if ref_fp["sha1"] != fp["sha1"]:
            print(f"✗ {args.reference} has a different ROI ({ref_fp['texels']} texels "
                  f"vs {fp['texels']}): the comparison would not measure the intrinsic noise")
            return 2
        ref_label = f"sandbox {args.reference} (stessa ROI → rumore intrinseco)"

    print(f"Riferimento : {ref}\n              {ref_label}")
    print(f"Confrontata : {roi}")
    print(f"ROI         : rect={fp.get('rect')}  {fp['texels']} texels  "
          f"su IUM {W}×{H}  (sha1 {fp['sha1'][:12]})")

    manifest = full / "run_manifest.json"
    if manifest.exists():
        with open(manifest, encoding="utf-8") as fh:
            std = json.load(fh)["config"].get("nerf_raw_noise_std")
        if std:
            print(f"\n  ⚠  nerf_raw_noise_std = {std}: raw2outputs aggiunge rumore "
                  f"gaussiano alla densità a OGNI query NeRF, senza guardia di eval\n"
                  f"     (nerf/rays.py:68). I gruppi B/C/D sono quindi stocastici di "
                  f"loro: usare --reference <altro-tag> per il metro.")
    print(f"\n  Verdict on the values only for group A; for every group the "
          f"'fuori dalla ROI deve essere zero'.\n")

    targets = build_targets(full, roi, args.source, cams)
    hdr = (f"{'gr':3} {'file':50} {'texel':>10} {'esatti':>8} "
           f"{'max|Δ|':>10} {'rel p50':>9} {'rel p99':>9} {'rel max':>9} {'fuori':>8}  esito")
    print(hdr)
    print("-" * len(hdr))

    failures: list[str] = []
    missing: list[str] = []
    rows_by_group: dict[str, list[Stats]] = {g: [] for g in GROUP_DESC}

    for rel, group, scope in targets:
        fp_ref, fp_roi = ref / rel, roi / rel
        if not fp_ref.exists() or not fp_roi.exists():
            which = "riferimento" if not fp_ref.exists() else "sandbox"
            missing.append(f"{rel} (manca in {which})")
            continue
        try:
            st = compare_file(fp_ref, fp_roi, mask2d, scope, args.band)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{rel}: {exc}")
            print(f"{group:3} {rel[-50:]:50} {'ERRORE':>10}  {exc}")
            continue

        rows_by_group[group].append(st)
        p50, p99 = st.percentiles()
        ok_out = st.max_outside == 0.0 or scope == "everywhere"
        ok_val = st.n_exact == st.n if group in STRICT else True
        if not ok_out:
            failures.append(f"{rel}: fuori ROI max {st.max_outside:.3e}")
        if not ok_val:
            failures.append(f"{rel}: {st.n - st.n_exact} valori diversi "
                            f"(gruppo deterministico)")

        if group in STRICT:
            verdict = "OK" if (ok_val and ok_out) else "✗"
        elif not ok_out:
            verdict = "✗ outside"
        else:
            verdict = f"info ({100 * (1 - st.frac_exact):.3f}% diversi)"
        print(f"{group:3} {rel[-50:]:50} {st.n:>10,} {100 * st.frac_exact:7.3f}% "
              f"{st.max_abs:10.3e} {p50:9.2e} {p99:9.2e} {st.max_rel:9.2e} "
              f"{st.max_outside:8.1e}  {verdict}")

    print()
    for g, sts in rows_by_group.items():
        if not sts:
            continue
        n_ex = sum(s.n_exact for s in sts)
        n = sum(s.n for s in sts)
        p50 = float(np.median([s.percentiles()[0] for s in sts]))
        p99 = max(s.percentiles()[1] for s in sts)
        print(f"  gruppo {g} ({GROUP_DESC[g]}): {len(sts)} file, "
              f"{100 * n_ex / max(n, 1):.4f}% bit-identici, "
              f"rel p50 mediana {p50:.2e}, rel p99 max {p99:.2e}, "
              f"max fuori ROI {max(s.max_outside for s in sts):.1e}")

    if missing:
        print(f"\n  {len(missing)} files not compared:")
        for m in missing[:10]:
            print(f"    · {m}")
        if len(missing) > 10:
            print(f"    … e altri {len(missing) - 10}")

    if failures:
        print(f"\n✗ {len(failures)} controlli falliti:")
        for f in failures:
            print(f"    {f}")
        return 1
    print("\n✓ checks passed: the deterministic passes are bit-identical inside "
          "the ROI and everything is zero outside.")
    if args.reference == "full":
        print("  Groups B/C/D are informational: weighing them needs two sandboxes "
              "with the same ROI (--reference).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
