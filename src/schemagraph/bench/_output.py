"""Result files shared by the benchmarks: one JSON (summary and rows) and one CSV per run."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any


def blank_if_none(value: object, fmt: str = "{}") -> str:
    """Format a CSV cell, leaving it empty for None."""
    return "" if value is None else fmt.format(value)


def write_outputs(
    out_dir: Path,
    stem: str,
    summary: dict,
    rows: Sequence[Any],
    header: list[str],
    csv_row: Callable[[Any], list],
) -> None:
    """Write ``<stem>.json`` (summary and rows) and ``<stem>.csv`` (one line per row).

    Args:
        out_dir: Folder to write into; created if missing.
        stem: File name without extension.
        summary: The run summary.
        rows: Dataclass rows, dumped with ``asdict`` into the JSON.
        header: CSV column names, in order.
        csv_row: Maps one row to its CSV cells, in ``header`` order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "rows": [asdict(row) for row in rows]}
    (out_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (out_dir / f"{stem}.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in rows:
            writer.writerow(csv_row(row))
