from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from numbers import Integral
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pyarrow.dataset as pyarrow_dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.causal.data.multi_episode_dataset import validate_causal_manifest


EXPECTED_EPISODE_COUNT = 2499
EXPECTED_EPISODE_IDS = tuple(range(EXPECTED_EPISODE_COUNT))
SPLIT_STRATEGY = "task_stratified_episode_level"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic task-stratified episode manifest for the "
            "complete ArmnetBench SO101 cache. No video or VAE work is performed."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser


def validate_split_ratios(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[float, float, float]:
    ratios = tuple(float(value) for value in (train_ratio, val_ratio, test_ratio))
    if not all(math.isfinite(value) and value > 0.0 for value in ratios):
        raise ValueError("train/val/test ratios must all be finite and positive")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"train/val/test ratios must sum to 1, got {sum(ratios)}")
    return ratios


def _integer_value(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must contain integers, got {value!r}")
    return int(value)


def episode_task_mapping(
    episode_indices: Iterable[object],
    task_indices: Iterable[object],
) -> dict[int, int]:
    """Build a strict one-task-per-episode mapping from parquet columns."""

    episodes = list(episode_indices)
    tasks = list(task_indices)
    if len(episodes) != len(tasks):
        raise ValueError("episode_index/task_index column length mismatch")
    if not episodes:
        raise ValueError("parquet contains no episode/task rows")

    tasks_by_episode: dict[int, set[int]] = defaultdict(set)
    for row, (episode_value, task_value) in enumerate(zip(episodes, tasks)):
        episode_id = _integer_value(episode_value, f"episode_index row {row}")
        task_id = _integer_value(task_value, f"task_index row {row}")
        tasks_by_episode[episode_id].add(task_id)

    ambiguous = {
        episode_id: sorted(task_ids)
        for episode_id, task_ids in tasks_by_episode.items()
        if len(task_ids) != 1
    }
    if ambiguous:
        first_episode = min(ambiguous)
        raise ValueError(
            f"episode {first_episode} maps to multiple task_index values: "
            f"{ambiguous[first_episode]}"
        )
    return {
        episode_id: next(iter(tasks_by_episode[episode_id]))
        for episode_id in sorted(tasks_by_episode)
    }


def read_episode_task_mapping(dataset_root: str | Path) -> dict[int, int]:
    data_root = Path(dataset_root) / "data"
    if not data_root.is_dir():
        raise FileNotFoundError(f"LeRobot parquet directory not found: {data_root}")
    dataset = pyarrow_dataset.dataset(data_root, format="parquet")
    required = {"episode_index", "task_index"}
    missing = required - set(dataset.schema.names)
    if missing:
        raise RuntimeError(f"dataset parquet schema is missing: {sorted(missing)}")
    table = dataset.to_table(columns=["episode_index", "task_index"])
    mapping = episode_task_mapping(
        table.column("episode_index").to_pylist(),
        table.column("task_index").to_pylist(),
    )
    actual_ids = set(mapping)
    expected_ids = set(EXPECTED_EPISODE_IDS)
    if actual_ids != expected_ids:
        raise RuntimeError(
            "dataset episode IDs must be exactly 0..2498: "
            f"missing={sorted(expected_ids - actual_ids)}, "
            f"unexpected={sorted(actual_ids - expected_ids)}"
        )
    if len(mapping) != EXPECTED_EPISODE_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_EPISODE_COUNT} unique episodes, got {len(mapping)}"
        )
    return mapping


def stratified_episode_split(
    episode_to_task: Mapping[int, int],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[dict[str, list[int]], dict[str, dict[str, int]]]:
    train_ratio, val_ratio, test_ratio = validate_split_ratios(
        train_ratio, val_ratio, test_ratio
    )
    if not episode_to_task:
        raise ValueError("episode_to_task must be non-empty")

    episodes_by_task: dict[int, list[int]] = defaultdict(list)
    for episode_id, task_id in episode_to_task.items():
        episodes_by_task[int(task_id)].append(int(episode_id))

    splits = {"train": [], "val": [], "test": []}
    task_counts: dict[str, dict[str, int]] = {}
    for task_offset, task_id in enumerate(sorted(episodes_by_task)):
        episode_ids = np.asarray(sorted(episodes_by_task[task_id]), dtype=np.int64)
        rng = np.random.default_rng(int(seed) + task_offset)
        shuffled = rng.permutation(episode_ids).tolist()
        total = len(shuffled)
        train_count = math.floor(total * train_ratio)
        val_count = math.floor(total * val_ratio)
        test_count = total - train_count - val_count
        task_train = shuffled[:train_count]
        task_val = shuffled[train_count : train_count + val_count]
        task_test = shuffled[train_count + val_count :]
        splits["train"].extend(task_train)
        splits["val"].extend(task_val)
        splits["test"].extend(task_test)
        task_counts[str(task_id)] = {
            "total": total,
            "train": train_count,
            "val": val_count,
            "test": test_count,
        }

    for split in splits:
        splits[split].sort()
        if not splits[split]:
            raise ValueError(
                f"stratified split {split!r} is empty; provide more episodes or ratios"
            )
    return splits, task_counts


def build_manifest(
    *,
    dataset_root: str | Path,
    cache_root: str | Path,
    episode_to_task: Mapping[int, int],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict:
    dataset_root = Path(dataset_root).resolve()
    cache_root = Path(cache_root).resolve()
    expected_ids = set(EXPECTED_EPISODE_IDS)
    actual_ids = set(map(int, episode_to_task))
    if actual_ids != expected_ids or len(episode_to_task) != EXPECTED_EPISODE_COUNT:
        raise ValueError(
            "formal SO101 manifest requires exactly episode IDs 0..2498"
        )
    splits, task_counts = stratified_episode_split(
        episode_to_task,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    episodes = []
    for episode_id in EXPECTED_EPISODE_IDS:
        cache_file = (cache_root / f"episode_{episode_id:03d}.pt").resolve()
        if not cache_file.is_file():
            raise FileNotFoundError(
                f"cache file is missing for episode {episode_id}: {cache_file}"
            )
        episodes.append(
            {
                "episode_index": episode_id,
                "task_index": int(episode_to_task[episode_id]),
                "cache_file": str(cache_file),
            }
        )
    manifest = {
        "manifest_version": 1,
        "dataset": dataset_root.name,
        "dataset_root": str(dataset_root),
        "cache_root": str(cache_root),
        "split_strategy": SPLIT_STRATEGY,
        "split_seed": int(seed),
        "split_ratios": {
            "train": float(train_ratio),
            "val": float(val_ratio),
            "test": float(test_ratio),
        },
        "train_episode_ids": splits["train"],
        "val_episode_ids": splits["val"],
        "test_episode_ids": splits["test"],
        "task_split_counts": task_counts,
        "episodes": episodes,
    }
    validate_formal_manifest(manifest, episode_to_task)
    return manifest


def validate_formal_manifest(
    manifest: dict,
    episode_to_task: Mapping[int, int],
) -> None:
    validate_causal_manifest(manifest)
    entries = manifest["episodes"]
    if len(entries) != EXPECTED_EPISODE_COUNT:
        raise RuntimeError(
            f"formal manifest must contain {EXPECTED_EPISODE_COUNT} episodes"
        )
    entry_ids = [int(entry["episode_index"]) for entry in entries]
    if entry_ids != list(EXPECTED_EPISODE_IDS):
        raise RuntimeError("formal manifest episodes must be sorted IDs 0..2498")
    for split in ("train", "val", "test"):
        ids = manifest[f"{split}_episode_ids"]
        if ids != sorted(ids):
            raise RuntimeError(f"formal manifest {split} episode IDs must be sorted")
    for entry in entries:
        episode_id = int(entry["episode_index"])
        if "task_index" not in entry:
            raise RuntimeError(f"episode {episode_id} entry is missing task_index")
        if int(entry["task_index"]) != int(episode_to_task[episode_id]):
            raise RuntimeError(
                f"episode {episode_id} task_index does not match source parquet"
            )
        if not Path(entry["cache_file"]).is_file():
            raise FileNotFoundError(
                f"manifest cache_file does not exist: {entry['cache_file']}"
            )


def atomic_json_write(value: object, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    args = build_parser().parse_args()
    episode_to_task = read_episode_task_mapping(args.dataset_root)
    manifest = build_manifest(
        dataset_root=args.dataset_root,
        cache_root=args.cache_root,
        episode_to_task=episode_to_task,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    atomic_json_write(manifest, args.output)
    print(f"total episodes: {len(manifest['episodes'])}")
    print(f"task count: {len(manifest['task_split_counts'])}")
    for split in ("train", "val", "test"):
        print(f"{split} episodes: {len(manifest[f'{split}_episode_ids'])}")
    print("per-task split counts:")
    for task_id, counts in manifest["task_split_counts"].items():
        print(f"  task {task_id}: {counts}")
    print(f"manifest output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
