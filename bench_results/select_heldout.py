"""Pick held-out tasks: N from each third of the local tasks by schema size, none of the 21.

Usage: ``uv run python bench_results/select_heldout.py [/path/to/Spider2] [N]`` (defaults
``~/data/Spider2`` and 20); writes ``bench_results/spider2_exec_heldout{3N}.txt``. Every list
starts the same round-robin, so the 7-per-third list (21 tasks) is part of the 20-per-third
one (60 tasks).

The 21-task subset (``spider2_exec_subset21.txt``) has been tuned on, so its numbers flatter the
agent. The held-out tasks come from the other 114 local tasks, split the same way as the subset:
the 135 local tasks sorted by the number of tables in their SQLite database (then by id) and cut
into thirds of 45 (4-10, 11-17 and 17-38 tables). Within each third, the tasks outside the subset
are taken round-robin over databases (databases by name, each database's tasks by id), so no
database dominates, until N are picked.
"""

import sqlite3
import sys
from collections import defaultdict
from contextlib import closing
from pathlib import Path

from schemagraph.bench.spider2_exec import load_tasks, sqlite_path

HERE = Path(__file__).parent
SUBSET = HERE / "spider2_exec_subset21.txt"
# Held-out tasks per third of the local tasks by schema size, by default.
PER_THIRD = 20


def table_count(database: Path) -> int:
    """Return the number of tables (not views) in a SQLite database."""
    with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as connection:
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


def main(spider2_root: Path, per_third: int) -> None:
    """Write the held-out task list and print how it splits by third."""
    subset = {line.strip() for line in SUBSET.read_text().splitlines() if line.strip()}
    tasks, _ = load_tasks(spider2_root)
    databases = {task.db for task in tasks}
    sizes = {db: table_count(sqlite_path(spider2_root, db)) for db in databases}
    ranked = sorted(tasks, key=lambda task: (sizes[task.db], task.instance_id))
    third, remainder = divmod(len(ranked), 3)
    if remainder:  # the subset's thirds were even; a remainder would silently drop tasks
        raise SystemExit(f"{len(ranked)} local tasks do not split into equal thirds")
    held_out: list[str] = []
    for index in range(3):
        tier = ranked[index * third : (index + 1) * third]
        in_subset = sum(task.instance_id in subset for task in tier)
        candidates: dict[str, list[str]] = defaultdict(list)
        for task in tier:
            if task.instance_id not in subset:
                candidates[task.db].append(task.instance_id)
        picked = round_robin(candidates, per_third)
        held_out += picked
        tables = [sizes[task.db] for task in tier]
        print(
            f"third {index + 1}: {min(tables)}-{max(tables)} tables, {in_subset} subset tasks, "
            f"{len(picked)} held out from {len(candidates)} databases"
        )
    output = HERE / f"spider2_exec_heldout{len(held_out)}.txt"
    output.write_text("\n".join(held_out) + "\n", encoding="utf-8")
    print(f"{len(held_out)} tasks written to {output}")


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "data/Spider2"
    main(root, int(sys.argv[2]) if len(sys.argv) > 2 else PER_THIRD)
