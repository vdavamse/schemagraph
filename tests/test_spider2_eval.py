"""The plain-Python port of Spider 2.0-Lite's result comparison, including its quirks."""

from __future__ import annotations

import io
import math
import os
import random
import re
from pathlib import Path

import pytest

from schemagraph.bench.spider2_eval import (
    compare_multi,
    compare_table,
    evaluate_rows,
    frame_vectors,
    infer_column,
    pred_columns,
    read_csv_columns,
    resolve_gold_paths,
    vectors_match,
)

PARITY_CASES = 200
_COMPARE_FUNCTION = re.compile(r"^def compare_pandas_table\(.*?(?=^def )", re.S | re.M)


def columns_of(text):
    return read_csv_columns(text)[1]


def test_csv_inference_like_pandas():
    assert infer_column(["1", "2"]) == ("int", [1, 2])
    kind, values = infer_column(["1", ""])
    # An int column with a missing value becomes a float column.
    assert kind == "float" and values[0] == 1.0 and math.isnan(values[1])
    assert infer_column(["1.5", "2"]) == ("float", [1.5, 2.0])
    # A TEXT column of digits is an int column.
    assert infer_column(["007", "8"]) == ("int", [7, 8])
    assert infer_column(["007", "x"]) == ("object", ["007", "x"])
    assert infer_column(["1_0"]) == ("object", ["1_0"])
    assert infer_column([" 12", "3 ", "+4"]) == ("int", [12, 3, 4])
    assert infer_column(["True", "false"]) == ("bool", [True, False])
    assert infer_column(["None", "NULL", "NA"])[0] == "na"


def test_csv_round_trip_keeps_null_rows():
    # A quoted empty cell is a NULL row; a blank line is skipped.
    header, columns = read_csv_columns('a\n""\n1\n\n2\n')
    assert header == ["a"]
    assert columns[0][0] == "float" and len(columns[0][1]) == 3
    assert pred_columns(["x"], [[None], [b"\x00"]])[0][1][1] == "b'\\x00'"
    # A whitespace-only line is blank to pandas, so to_csv + read_csv drops that row.
    assert read_csv_columns("a\n  \n1\n")[1][0] == ("int", [1])
    assert pred_columns(["a"], [["  "], [1]])[0] == ("int", [1])


def test_frame_upcast_and_matching():
    assert frame_vectors([("int", [1, 2]), ("float", [0.5, 1.0])]) == [[1.0, 2.0], [0.5, 1.0]]
    assert frame_vectors([("int", [1]), ("object", ["x"])]) == [[1], ["x"]]
    assert vectors_match([1.0, 2.0], [1.004, 2.0])
    assert not vectors_match([1.0], [1.02])
    assert vectors_match([float("nan")], [0])  # NaN -> 0
    assert vectors_match([10, 9], [9, 10], ignore_order=True)
    assert not vectors_match([1, 2], [2, 1])


def test_compare_table_subset_and_condition_cols():
    gold = columns_of("a,b\n1,x\n2,y\n")
    # Extra and reordered predicted columns are fine.
    assert compare_table(columns_of("q,a,b,extra\nz,1,x,7\nw,2,y,8\n"), gold) == 1
    only_a = columns_of("a\n1\n2\n")
    assert compare_table(only_a, gold) == 0
    assert compare_table(only_a, gold, condition_cols=[0]) == 1
    # 0 is falsy, so every column is compared.
    assert compare_table(only_a, gold, condition_cols=0) == 0


def test_compare_multi_quirks():
    gold = columns_of("a,b\n1,x\n")
    pred_a = columns_of("a\n1\n")
    # One _a file and a flat list: condition_cols[0] == 0, so every column is compared.
    assert compare_multi(pred_a, [gold], [0, 1]) == 0
    assert compare_multi(pred_a, [gold], [[0]]) == 1
    # condition_cols[0] == 1: column 1 only.
    assert compare_multi(columns_of("b\nx\n"), [gold], [1, 2, 3]) == 1
    # More than one gold and a flat list: the list applies to each.
    assert compare_multi(pred_a, [columns_of("a\n9\n"), gold], [0]) == 1
    assert compare_multi(pred_a, []) == 0


def test_evaluate_rows_and_gold_paths(tmp_path):
    (tmp_path / "local001_a.csv").write_text("n\n3\n")
    (tmp_path / "local001_b.csv").write_text("n\n4\n")
    (tmp_path / "local0011.csv").write_text("n\n9\n")  # a different instance
    paths, single = resolve_gold_paths("local001", tmp_path)
    assert [path.name for path in paths] == ["local001_a.csv", "local001_b.csv"]
    assert not single
    assert evaluate_rows("local001", ["count"], [[4]], tmp_path, {"ignore_order": True}) == (1, None)
    assert evaluate_rows("local001", ["count"], [[5]], tmp_path, {}) == (0, "Result Error")
    assert evaluate_rows("local001", [], [], tmp_path, {})[0] == 0
    missing = evaluate_rows("local999", ["x"], [[1]], tmp_path, {})
    assert missing == (0, "No matching gold file found")
    bad_index = evaluate_rows("local001", ["count"], [[4]], tmp_path, {"condition_cols": [7]})
    assert bad_index[1].startswith("Python Script Error")


def _official():
    """Load ``compare_pandas_table`` from the official evaluator in a local Spider2 clone.

    The clone is ``$SPIDER2_ROOT``, ``~/data/Spider2`` or ``~/Spider2``. The function is read from
    it rather than vendored because Spider2 ships no license file.
    """
    pd = pytest.importorskip("pandas")
    roots = [Path(os.environ["SPIDER2_ROOT"])] if os.environ.get("SPIDER2_ROOT") else []
    for root in (*roots, Path.home() / "data/Spider2", Path.home() / "Spider2"):
        source = root / "spider2-lite/evaluation_suite/evaluate.py"
        if source.exists():
            namespace = {"pd": pd, "math": math}
            match = _COMPARE_FUNCTION.search(source.read_text())
            exec(match.group(0), namespace)  # noqa: S102 - the evaluator's own function, for a parity check
            return pd, namespace["compare_pandas_table"]
    pytest.skip("no Spider2 clone with evaluate.py")


def _value_pools(rng):
    """Generators of one column's values, covering the inference and comparison edge cases."""
    return [
        lambda: rng.randint(-5, 50),
        lambda: rng.choice([None, rng.randint(0, 9)]),
        lambda: round(rng.uniform(-3, 3), rng.choice([0, 1, 2, 6])),
        lambda: rng.choice(["a", "b", "007", "10", "9", None, "True"]),
        lambda: rng.choice([1.0, 2.0, None]),
        lambda: rng.choice(["x", 3, 2.5, None]),
    ]


def _to_csv(pd, rows, names):
    buffer = io.StringIO()
    pd.DataFrame(rows, columns=names).to_csv(buffer, index=False)
    return buffer.getvalue()


def test_parity_with_the_official_comparator():
    pd, official = _official()
    rng = random.Random(0)
    pools = _value_pools(rng)
    for _ in range(PARITY_CASES):
        column_count, row_count = rng.randint(1, 3), rng.randint(1, 6)
        generators = [rng.choice(pools) for _ in range(column_count)]
        rows = [[generate() for generate in generators] for _ in range(row_count)]
        names = [f"c{i}" for i in range(column_count)]
        gold_rows = [list(row) for row in rows]
        rng.shuffle(gold_rows)
        if rng.random() < 0.3:
            gold_rows[0][0] = rng.choice([None, 1, "z", 1.004])
        gold_csv = _to_csv(pd, gold_rows, names)
        pred_csv = _to_csv(pd, rows, names)
        condition_cols = rng.choice([None, 0, [0], list(range(column_count))])
        ignore_order = rng.random() < 0.5
        want = official(
            pd.read_csv(io.StringIO(pred_csv)),
            pd.read_csv(io.StringIO(gold_csv)),
            condition_cols,
            ignore_order,
        )
        got = compare_table(pred_columns(names, rows), columns_of(gold_csv), condition_cols, ignore_order)
        assert got == want, (rows, gold_rows, condition_cols, ignore_order)
