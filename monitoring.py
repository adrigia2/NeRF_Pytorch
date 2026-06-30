"""monitoring.py — TensorBoard logging and stage timing for the hybrid pipeline.

All tag names, axis labels, and display strings are in English so they can be
used directly in academic work (thesis figures, paper screenshots, etc.).
Italian stays only in console print() messages and code comments.

Usage
-----
    from monitoring import RunLogger, StageTimer, log_timing_breakdown

    logger = RunLogger("D:/tesi_output/tb_logs/exp1/SwordShield/20260627-120000")
    timer  = StageTimer()

    with timer("step1"):
        ...

    logger.log_scalars("nerf", {"loss": 0.05, "psnr_db": 32.1, "lr": 1e-4}, step=1000)
    logger.flush()

    log_timing_breakdown(logger, timer.timings)
    logger.close()
"""
from __future__ import annotations

import io
import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Tonemap helper
# ──────────────────────────────────────────────────────────────────────────────

def _tonemap_to_uint8(
    arr: np.ndarray,
    exposure: float = 1.0,
    gamma: float = 2.2,
) -> np.ndarray:
    """Convert a linear float32 HxWx3 array to tonemapped uint8 HxWx3.

    Uses Reinhard global tonemapping followed by gamma encoding.  For LDR
    images already in [0, 1] (e.g. color_texture, albedo) the Reinhard
    operator is near-identity, so this works correctly for both HDR NeRF
    previews and LDR textures without needing two separate code paths.
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:                          # (H, W) → (H, W, 1)
        arr = arr[:, :, np.newaxis]
    x = np.clip(arr, 0.0, None) * float(2.0 ** exposure)
    x = x / (1.0 + x)                          # Reinhard
    x = np.clip(x, 0.0, 1.0) ** (1.0 / gamma)
    return (x * 255.0 + 0.5).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# RunLogger
# ──────────────────────────────────────────────────────────────────────────────

class RunLogger:
    """Thin wrapper around torch.utils.tensorboard.SummaryWriter.

    Falls back silently to a no-op when ``tensorboard`` is not installed or
    when ``enabled=False``, so the pipeline never crashes due to missing
    monitoring dependencies.

    Low ``flush_secs`` (default 15 s vs the upstream 120 s) keeps the remote
    TensorBoard view nearly real-time during training.

    All tag names and display strings are in English.
    """

    def __init__(
        self,
        log_dir: str,
        enabled: bool = True,
        flush_secs: int = 15,
    ) -> None:
        self._writer = None
        self._enabled = False
        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter  # type: ignore
                self._writer = SummaryWriter(log_dir=log_dir, flush_secs=flush_secs)
                self._enabled = True
                print(f"  [monitoring] TensorBoard → {log_dir}  (flush ≤{flush_secs}s)")
            except ImportError:
                print("  [monitoring] tensorboard not installed — logging disabled "
                      "(run: pip install tensorboard)")

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ── scalars ───────────────────────────────────────────────────────────────

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Write a single scalar value."""
        if self._writer is not None:
            self._writer.add_scalar(tag, float(value), global_step=step)

    def log_scalars(
        self,
        prefix: str,
        values: dict[str, float],
        step: int,
    ) -> None:
        """Write multiple scalars under a shared prefix (e.g. 'nerf/loss')."""
        if self._writer is not None:
            for k, v in values.items():
                self._writer.add_scalar(f"{prefix}/{k}", float(v), global_step=step)

    # ── images ────────────────────────────────────────────────────────────────

    def log_image(
        self,
        tag: str,
        arr: np.ndarray,
        step: int,
        tonemap: bool = True,
        exposure: float = 1.0,
    ) -> None:
        """Write an HxWx3 float array as a TensorBoard image.

        When ``tonemap=True`` (default), applies Reinhard tonemapping so HDR
        previews are visible.  Set ``tonemap=False`` for arrays already in
        [0, 1] that should not be re-mapped (internally still clips+scales).
        """
        if self._writer is None:
            return
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[2] > 3:
            arr = arr[..., :3]
        if tonemap:
            uint8 = _tonemap_to_uint8(arr, exposure=exposure)
        else:
            uint8 = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
        if uint8.ndim == 2:
            uint8 = uint8[:, :, np.newaxis].repeat(3, axis=2)
        self._writer.add_image(tag, uint8, global_step=step, dataformats="HWC")

    # ── text ──────────────────────────────────────────────────────────────────

    def log_text(self, tag: str, text: str, step: int = 0) -> None:
        """Write a text string (e.g. a JSON config dump) for the Text tab."""
        if self._writer is not None:
            # Wrap in markdown code block so it renders monospaced in the UI.
            self._writer.add_text(tag, f"```\n{text}\n```", global_step=step)

    # ── hparams ───────────────────────────────────────────────────────────────

    def log_hparams(
        self,
        hparams: dict[str, Any],
        metrics: dict[str, float],
    ) -> None:
        """Write hyperparameters + summary metrics for the HParams comparison tab.

        Shows up as a row in TensorBoard's parallel-coordinates / table view —
        useful for comparing many experiments at a glance.
        """
        if self._writer is None:
            return
        # SummaryWriter.add_hparams requires scalar Python types.
        clean_hp: dict[str, int | float | str | bool] = {}
        for k, v in hparams.items():
            if isinstance(v, (int, float, str, bool)):
                clean_hp[k] = v
            elif isinstance(v, (list, tuple)):
                clean_hp[k] = str(v)
            else:
                clean_hp[k] = str(v)
        clean_m = {k: float(v) for k, v in metrics.items()}
        self._writer.add_hparams(clean_hp, clean_m)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def flush(self) -> None:
        """Force-flush pending events to disk (call after important log lines)."""
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        """Close the underlying writer (flushes + releases file handles)."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            self._enabled = False


# ──────────────────────────────────────────────────────────────────────────────
# StageTimer
# ──────────────────────────────────────────────────────────────────────────────

class StageTimer:
    """Accumulate wall-clock durations for named pipeline stages.

    Works as a context manager.  The same stage name can be entered multiple
    times (e.g. across interactive training continuations) — elapsed time is
    accumulated, not overwritten.

    Usage::

        timer = StageTimer()
        with timer("step1"):
            ...
        with timer("step2"):
            ...
        with timer("step3"):
            with timer("step3/ium"):  # sub-stage — also accumulated
                ...
        print(timer.timings)  # OrderedDict preserves insertion order
    """

    def __init__(self) -> None:
        self._timings: OrderedDict[str, float] = OrderedDict()

    @contextmanager
    def __call__(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._timings[name] = (
                self._timings.get(name, 0.0) + (time.perf_counter() - t0)
            )

    def record(self, name: str, elapsed: float) -> None:
        """Manually record an elapsed time (use when context-manager isn't convenient)."""
        self._timings[name] = self._timings.get(name, 0.0) + elapsed

    @property
    def timings(self) -> OrderedDict[str, float]:
        return self._timings


# ──────────────────────────────────────────────────────────────────────────────
# Timing breakdown log
# ──────────────────────────────────────────────────────────────────────────────

def log_timing_breakdown(
    logger: RunLogger,
    timings: dict[str, float],
    step: int = 0,
) -> None:
    """Log per-stage durations as scalars and as a horizontal bar-chart image.

    Scalars (``timing/<stage>``) land in the Scalars tab for cross-run
    comparison.  The bar chart (``timing/breakdown``) gives an at-a-glance
    view of where time was spent in the current run.

    Only top-level stages (no ``/`` in the key) contribute to the total and
    to the bar chart so that sub-stages like ``step3/ium`` don't double-count.
    Sub-stages are still logged as individual scalars.
    """
    if not logger.enabled:
        return

    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend, safe in training scripts
    import matplotlib.pyplot as plt

    # — all stages as individual scalars —
    logger.log_scalars("timing", {k: v for k, v in timings.items()}, step=step)

    # — bar chart of top-level stages —
    top = {k: v for k, v in timings.items() if "/" not in k}
    if not top:
        return

    total = max(sum(top.values()), 1e-9)
    logger.log_scalar("timing/total_s", total, step=step)

    labels = list(top.keys())
    durations = [top[k] for k in labels]
    pcts = [100.0 * d / total for d in durations]

    n = len(labels)
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.45 * n + 1.2)))
    bars = ax.barh(labels[::-1], durations[::-1], color="#4C8BF5", edgecolor="white",
                   height=0.6)
    ax.set_xlabel("Duration (s)")
    ax.set_title("Pipeline stage breakdown")
    ax.spines[["top", "right"]].set_visible(False)

    for bar, sec, pct in zip(bars, durations[::-1], pcts[::-1]):
        ax.text(
            bar.get_width() + total * 0.015,
            bar.get_y() + bar.get_height() / 2.0,
            f"{sec:.1f} s  ({pct:.1f}%)",
            va="center", fontsize=8,
        )
    ax.set_xlim(0, total * 1.30)
    fig.tight_layout()

    # Render figure → numpy via PIL (avoids backend-specific tostring_rgb issues)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    try:
        from PIL import Image
        chart_uint8 = np.array(Image.open(buf).convert("RGB"), dtype=np.uint8)
    except ImportError:
        return  # PIL always present in nerfpytorch env, but guard anyway

    # Use log_image with tonemap=False: chart is already uint8-range
    chart_f32 = chart_uint8.astype(np.float32) / 255.0
    logger.log_image("timing/breakdown", chart_f32, step=step, tonemap=False)
    logger.flush()
