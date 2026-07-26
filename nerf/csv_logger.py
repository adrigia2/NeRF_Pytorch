"""csv_logger.py — append-only CSV logging for training metrics.

Written to be safe across the two ways ``train()`` gets re-entered on the same
output directory: a resume from checkpoint, and the interactive continuation
loop in ``images_generator.run_pipeline`` (which calls ``train()`` again
in-process).  Both cases must *append*, never truncate.

Each ``log()`` call opens, writes and closes the file.  With one row per display
block (~500 rows over 50k iterations) the cost is irrelevant, and in exchange
the logger is stateless and reentrant, and the file on disk is always complete
even if the process is killed mid-training.

Column names and header text are in English so the CSVs can feed thesis figures
directly.  Italian stays only in comments.
"""
from __future__ import annotations

import csv
import datetime
from pathlib import Path


class CsvLogger:
    """Append rows to a CSV, writing the header only when the file is new.

    ``fieldnames`` fixes the schema.  If an existing file on disk was written
    with a different header (e.g. a column was added in a later version of the
    code), the stale file is renamed to ``<stem>_<YYYYMMDD-HHMMSS>.csv`` and a
    fresh one is started.  The mismatch never raises: a logging detail must not
    interrupt a training run that has been going for hours.
    """

    def __init__(self, path: str | Path, fieldnames: list[str]) -> None:
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self._checked = False   # schema verificato una volta per processo

    # ── schema guard ──────────────────────────────────────────────────────────

    def _check_schema(self) -> None:
        """Rotate the file away if its header does not match ``fieldnames``."""
        self._checked = True
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        try:
            with open(self.path, newline="", encoding="utf-8") as fh:
                header = next(csv.reader(fh), None)
        except OSError as exc:
            print(f"  [warn] CSV metriche illeggibile ({exc}); lo sovrascrivo: {self.path}")
            header = None

        if header == self.fieldnames:
            return

        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        rotated = self.path.with_name(f"{self.path.stem}_{stamp}{self.path.suffix}")
        try:
            self.path.rename(rotated)
            print(f"  [warn] header CSV diverso da quello atteso; vecchio file → {rotated.name}")
        except OSError as exc:
            print(f"  [warn] impossibile ruotare il CSV ({exc}); logging disattivato: {self.path}")
            self.fieldnames = []   # disabilita i log successivi senza sollevare

    # ── write ─────────────────────────────────────────────────────────────────

    def log(self, row: dict) -> None:
        """Append one row.  Missing keys are written empty, extra keys ignored."""
        if not self.fieldnames:
            return
        if not self._checked:
            self._check_schema()
            if not self.fieldnames:
                return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not self.path.exists() or self.path.stat().st_size == 0
            # newline="" è obbligatorio col modulo csv su Windows: senza, ogni
            # record viene seguito da una riga vuota.
            with open(self.path, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.fieldnames,
                                        extrasaction="ignore")
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        except OSError as exc:
            print(f"  [warn] scrittura CSV metriche fallita ({exc}): {self.path}")
