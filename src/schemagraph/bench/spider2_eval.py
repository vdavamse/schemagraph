"""Spider 2.0-Lite execution-match scoring, ported to plain Python (no pandas).

The port follows ``spider2-lite/evaluation_suite/evaluate.py``, so EX here is comparable with the
leaderboard's local subset. What the original does, and this module reproduces:

* the prediction runs on SQLite, is written with ``DataFrame.to_csv`` and read back with
  ``pd.read_csv``; gold results are ``pd.read_csv`` of the shipped CSVs. Both sides therefore go
  through pandas' CSV type inference (a TEXT column of ``"007"`` values becomes the int 7);
* ``.transpose().values.tolist()`` upcasts the whole (selected) frame to one dtype: int columns
  become floats when a float column is present, objects keep their Python types;
* NaN/None -> 0, numbers match within ``abs_tol=1e-2``, everything else by ``==``;
* ``ignore_order`` sorts each column by ``(is None, str(x), is number)``: ``"10" < "9"``, kept
  as is;
* a task scores 1 when every gold column (only ``condition_cols`` when given) equals *some*
  predicted column (extra and reordered predicted columns are fine), for *any* gold variant;
* ``compare_multi_pandas_table``'s quirk: with a single ``_a`` gold file and a flat
  ``condition_cols`` like ``[0, 1]``, gold 0 uses ``condition_cols[0]`` (the int 0, falsy), so every
  column is compared.

Known residual difference: pandas' C float parser can differ from ``float()`` in the last bit,
which only matters for the str-sort of floats with very long representations.
"""

from __future__ import annotations

import csv
import io
import math
import re
from pathlib import Path
from typing import Any

# pandas' default ``na_values``: cells ``read_csv`` reads as NaN.
NA_VALUES = frozenset(
    {
        "",
        "#N/A",
        "#N/A N/A",
        "#NA",
        "-1.#IND",
        "-1.#QNAN",
        "-NaN",
        "-nan",
        "1.#IND",
        "1.#QNAN",
        "<NA>",
        "N/A",
        "NA",
        "NULL",
        "NaN",
        "None",
        "n/a",
        "nan",
        "null",
    }
)
# The official evaluator's ``math.isclose(..., abs_tol=1e-2)``.
TOLERANCE = 1e-2
NA = float("nan")

_INT_PATTERN = re.compile(r"^\s*[+-]?\d+\s*$")
_BOOL_VALUES = {
    "True": True,
    "TRUE": True,
    "true": True,
    "False": False,
    "FALSE": False,
    "false": False,
}
_NO_COLUMNS = "No columns to parse from file"

# One inferred column: (kind, values), kind being "int", "float", "bool", "object" or "na".
Column = tuple[str, list[Any]]


def csv_cell(value: Any) -> str:
    """Write one value of a ``read_sql_query`` frame the way ``DataFrame.to_csv`` does."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _is_float(cell: str) -> bool:
    """Whether pandas would parse ``cell`` as a float."""
    if "_" in cell:  # Python accepts 1_0, pandas does not
        return False
    try:
        float(cell)
    except ValueError:
        return False
    return True


def infer_column(cells: list[str]) -> Column:
    """Infer one column of raw cells the way ``pd.read_csv`` does."""
    values = [None if cell in NA_VALUES else cell for cell in cells]
    present = [cell for cell in values if cell is not None]
    has_na = len(present) < len(values)
    if not present:
        return "na", [NA] * len(values)
    if all(_INT_PATTERN.match(cell) for cell in present):
        if has_na:
            return "float", [NA if cell is None else float(int(cell)) for cell in values]
        return "int", [int(cell) for cell in values]  # type: ignore[arg-type]
    if all(_is_float(cell.strip()) for cell in present):
        return "float", [NA if cell is None else float(cell.strip()) for cell in values]
    if all(cell in _BOOL_VALUES for cell in present):
        kind = "object" if has_na else "bool"
        return kind, [NA if cell is None else _BOOL_VALUES[cell] for cell in values]
    return "object", [NA if cell is None else cell for cell in values]


def _blank_line(cells: list[str]) -> bool:
    """Whether pandas skips this CSV line as blank.

    That is a line of only whitespace: a one-column row whose cell is unquoted spaces (``to_csv``
    never quotes them; an empty cell is written as ``""`` and kept).
    """
    return len(cells) == 1 and cells[0] != "" and not cells[0].strip()


def read_csv_columns(text: str) -> tuple[list[str], list[Column]]:
    """Read CSV text into its header and inferred columns, skipping blank lines like pandas.

    Raises:
        ValueError: The text has no header, as ``pd.read_csv`` does.
    """
    rows = [row for row in csv.reader(io.StringIO(text)) if row and not _blank_line(row)]
    if not rows:
        raise ValueError(_NO_COLUMNS)
    header, body = rows[0], rows[1:]
    width = len(header)
    cells_by_column = [[(row[i] if i < len(row) else "") for row in body] for i in range(width)]
    return header, [infer_column(cells) for cells in cells_by_column]


def pred_columns(columns: list[str], rows: list[list[Any]]) -> list[Column]:
    """Return the prediction as the original sees it after its ``to_csv``/``read_csv`` round trip.

    Raises:
        ValueError: The prediction has no columns, as ``pd.read_csv`` does.
    """
    if not columns:
        raise ValueError(_NO_COLUMNS)
    cell_rows = [[csv_cell(value) for value in row] for row in rows]
    if len(columns) == 1:
        cell_rows = [cells for cells in cell_rows if not _blank_line(cells)]
    return [infer_column([cells[i] for cells in cell_rows]) for i in range(len(columns))]


def frame_vectors(columns: list[Column]) -> list[list[Any]]:
    """Mimic ``frame.transpose().values.tolist()``: one common dtype for the whole frame."""
    kinds = {kind for kind, _ in columns}
    if kinds and kinds <= {"int", "float", "na"} and kinds != {"int"}:
        return [[float(value) for value in values] for _, values in columns]
    return [list(values) for _, values in columns]


def _isna(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _normalize(value: Any) -> Any:
    """Map a missing value to 0, as the original does before comparing."""
    return 0 if _isna(value) else value


def _sort_key(value: Any) -> tuple[bool, str, bool]:
    """The original's ``ignore_order`` sort key; it sorts numbers by their string form."""
    return (value is None, str(value), isinstance(value, (int, float)))


def vectors_match(
    v1: list[Any],
    v2: list[Any],
    ignore_order: bool = False,
    tol: float = TOLERANCE,
) -> bool:
    """Whether two columns are equal: numbers within ``tol``, anything else by ``==``.

    Args:
        v1: One column's values.
        v2: The other column's values.
        ignore_order: Sort both columns with the original's key before comparing.
        tol: Absolute tolerance for numbers.

    Returns:
        True when both have the same length and every pair of values matches.
    """
    left = [_normalize(value) for value in v1]
    right = [_normalize(value) for value in v2]
    if ignore_order:
        left, right = sorted(left, key=_sort_key), sorted(right, key=_sort_key)
    if len(left) != len(right):
        return False
    for a, b in zip(left, right, strict=True):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if not math.isclose(float(a), float(b), abs_tol=tol):
                return False
        elif a != b:
            return False
    return True


def compare_table(
    pred: list[Column],
    gold: list[Column],
    condition_cols: Any = None,
    ignore_order: bool = False,
) -> int:
    """Score a prediction against one gold result, like ``compare_pandas_table``.

    Args:
        pred: The prediction's columns.
        gold: The gold result's columns.
        condition_cols: Gold column indices to compare, a single index, or falsy (``0``
            included, as in the original) to compare every gold column.
        ignore_order: Compare the columns' values as sorted multisets.

    Returns:
        1 when every compared gold column matches some predicted column, else 0.
    """
    if condition_cols:
        if not isinstance(condition_cols, (list, tuple)):
            condition_cols = [condition_cols]
        gold = [gold[i] for i in condition_cols]
    pred_vectors = frame_vectors(pred)
    for gold_vector in frame_vectors(gold):
        if not any(vectors_match(gold_vector, vector, ignore_order) for vector in pred_vectors):
            return 0
    return 1


def compare_multi(
    pred: list[Column],
    golds: list[list[Column]],
    multi_condition_cols: Any = None,
    ignore_order: bool = False,
) -> int:
    """Score a prediction against gold variants, like ``compare_multi_pandas_table``.

    Args:
        pred: The prediction's columns.
        golds: The gold variants' columns, in file order.
        multi_condition_cols: ``condition_cols`` per variant. With several variants a flat list
            applies to each; with one variant, a flat list's first element is used (the
            original's quirk).
        ignore_order: Compare the columns' values as sorted multisets.

    Returns:
        1 when any gold variant matches, else 0.
    """
    if not golds:
        return 0
    if multi_condition_cols in (None, [], [[]], [None]):
        multi_condition_cols = [[] for _ in golds]
    elif len(golds) > 1 and not all(isinstance(cols, list) for cols in multi_condition_cols):
        multi_condition_cols = [multi_condition_cols for _ in golds]
    for i, gold in enumerate(golds):
        if compare_table(pred, gold, multi_condition_cols[i], ignore_order):
            return 1
    return 0


def resolve_gold_paths(instance_id: str, gold_dir: Path) -> tuple[list[Path], bool]:
    """Find a task's gold CSVs.

    Returns:
        ``([<id>.csv], True)`` when a single gold file exists, else the sorted ``<id>_<letter>.csv``
        variants (possibly none) and False.
    """
    single = gold_dir / f"{instance_id}.csv"
    if single.exists():
        return [single], True
    pattern = re.compile(rf"^{re.escape(instance_id)}(_[a-z])?\.csv$")
    names = sorted(path.name for path in gold_dir.iterdir() if pattern.match(path.name))
    return [gold_dir / name for name in names], False


def evaluate_rows(
    instance_id: str,
    columns: list[str],
    rows: list[list[Any]],
    gold_dir: Path,
    standard: dict[str, Any],
) -> tuple[int, str | None]:
    """Score a prediction's full result as the official evaluator would.

    Args:
        instance_id: The task id, which names its gold CSVs.
        columns: The prediction's column names.
        rows: The prediction's rows.
        gold_dir: The directory of gold CSVs.
        standard: The task's eval-standard entry (``condition_cols``, ``ignore_order``).

    Returns:
        ``(score, error)``: 1 and None on a match, else 0 and the original's error text.
    """
    try:
        pred = pred_columns(columns, rows)
    except Exception as error:
        return 0, f"Evaluation Error: {error}"
    paths, single = resolve_gold_paths(instance_id, gold_dir)
    if not paths:
        return 0, "No matching gold file found"
    condition_cols = standard.get("condition_cols")
    ignore_order = standard.get("ignore_order", False)
    try:
        golds = [read_csv_columns(path.read_text(encoding="utf-8"))[1] for path in paths]
        if single:
            score = compare_table(pred, golds[0], condition_cols, ignore_order)
        else:
            score = compare_multi(pred, golds, condition_cols, ignore_order)
    except Exception as error:
        return 0, f"Python Script Error:{error}"
    return score, None if score else "Result Error"
