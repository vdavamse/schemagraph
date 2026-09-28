"""Pick the held-out tasks: 20 from each third of the local tasks by schema size, none of the 21.

Usage: ``uv run python bench_results/select_heldout.py [/path/to/Spider2]`` (default
``~/data/Spider2``); writes ``bench_results/spider2_exec_heldout60.txt``.

The 21-task subset (``spider2_exec_subset21.txt``) has been tuned on, so its numbers flatter the
agent. The held-out tasks come from the other 114 local tasks, split the same way as the subset:
the 135 local tasks sorted by the number of tables in their SQLite database (then by id) and cut
into thirds of 45 (4-10, 11-17 and 17-38 tables). Within each third, the tasks outside the subset
are taken round-robin over databases (databases by name, each database's tasks by id), so no
database dominates, until 20 are picked.
"""

import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

from schemagraph.bench.spider2_exec import load_tasks, sqlite_path

HERE = Path(__file__).parent
SUBSET = HERE / "spider2_exec_subset21.txt"
OUTPUT = HERE / "spider2_exec_heldout60.txt"
# Held-out tasks per third of the local tasks by schema size.
PER_THIRD = 20


def table_count(database: Path) -> int:
    """Return the number of tables (not views) in a SQLite database."""
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        query = "select count(*) from sqlite_master where type = 'table'"
        return connection.execute(query).fetchone()[0]


def round_robin(task_ids_by_db: dict[str, list[str]], count: int) -> list[str]:
    """Take one task per database in turn (databases by name) until ``count`` are taken."""
    queues = {db: sorted(ids) for db, ids in sorted(task_ids_by_db.items())}
    picked: list[str] = []
    while len(picked) < count and any(queues.values()):
        for queue in queues.values():
            if queue and len(picked) < count:
                picked.append(queue.pop(0))
    return picked


def main(spider2_root: Path) -> None:
    """Write the held-out task list and print how it splits by third."""
    subset = {line.strip() for line in SUBSET.read_text().splitlines() if line.strip()}
    tasks, _ = load_tasks(spider2_root)
    sizes = {task.db: table_count(sqlite_path(spider2_root, task.db)) for task in tasks}
    ranked = sorted(tasks, key=lambda task: (sizes[task.db], task.instance_id))
    third = len(ranked) // 3
    held_out: list[str] = []
    for index in range(3):
        tier = ranked[index * third : (index + 1) * third]
        in_subset = sum(task.instance_id in subset for task in tier)
        candidates: dict[str, list[str]] = defaultdict(list)
        for task in tier:
            if task.instance_id not in subset:
                candidates[task.db].append(task.instance_id)
        picked = round_robin(candidates, PER_THIRD)
        held_out += picked
        tables = [sizes[task.db] for task in tier]
        print(
            f"third {index + 1}: {min(tables)}-{max(tables)} tables, {in_subset} subset tasks, "
            f"{len(picked)} held out from {len(candidates)} databases"
        )
    OUTPUT.write_text("\n".join(held_out) + "\n", encoding="utf-8")
    print(f"{len(held_out)} tasks written to {OUTPUT}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "data/Spider2")
