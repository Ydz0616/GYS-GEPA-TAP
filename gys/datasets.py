"""AdvBench task loading and deterministic train/test/heldout splits.

The dataset is the 50-goal AdvBench subset vendored from RICommunity/TAP. A
task is a ``(goal, target_str)`` pair. Splits are deterministic given a seed so
that a heldout generalization run is reproducible.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

from gys.config import DataConfig


@dataclass(frozen=True)
class Task:
    task_id: int
    goal: str
    target_str: str
    category: str
    original_index: int

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "target_str": self.target_str,
            "category": self.category,
            "original_index": self.original_index,
        }


def load_tasks(csv_path: str | Path) -> list[Task]:
    """Load every task from the AdvBench subset CSV, keyed by its row index."""
    tasks: list[Task] = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            tasks.append(
                Task(
                    task_id=int(row[""]),
                    goal=row["goal"].strip(),
                    target_str=row["target"].strip(),
                    category=row["category"].strip(),
                    original_index=int(row["Original index"]),
                )
            )
    return tasks


def split_tasks(
    tasks: list[Task],
    train_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, list[Task]]:
    """Shuffle deterministically and split into train/test/heldout.

    ``heldout`` receives whatever remains after train and test, so the three
    fractions need not sum to exactly 1.0.
    """
    if train_frac < 0 or test_frac < 0 or train_frac + test_frac > 1.0:
        raise ValueError("train_frac and test_frac must be >= 0 and sum to <= 1.0")
    ordered = sorted(tasks, key=lambda task: task.task_id)
    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_frac)
    n_test = int(n * test_frac)
    return {
        "train": shuffled[:n_train],
        "test": shuffled[n_train : n_train + n_test],
        "heldout": shuffled[n_train + n_test :],
    }


def select_tasks(cfg: DataConfig) -> list[Task]:
    """Resolve the tasks named by a ``DataConfig``.

    Explicit ``task_ids`` take precedence and bypass the split. Otherwise the
    requested ``split`` is returned, optionally filtered by ``categories`` and
    truncated to ``limit``.
    """
    tasks = load_tasks(cfg.csv_path)
    by_id = {task.task_id: task for task in tasks}

    if cfg.task_ids:
        selected = [by_id[tid] for tid in cfg.task_ids]
    else:
        split = split_tasks(tasks, cfg.train_frac, cfg.test_frac, cfg.seed)
        if cfg.split not in split:
            raise ValueError(f"unknown split: {cfg.split!r}")
        selected = split[cfg.split]

    if cfg.categories:
        wanted = set(cfg.categories)
        selected = [task for task in selected if task.category in wanted]
    if cfg.limit is not None:
        selected = selected[: cfg.limit]
    return selected
