from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


SPLIT_SCHEMA_VERSION = 1
SPLIT_ALGORITHM = "task_stratified_episode_level_floor_80_10_remainder"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def episode_task_mapping(
    episode_indices: Iterable[object], task_indices: Iterable[object]
) -> dict[int, int]:
    grouped: dict[int, set[int]] = defaultdict(set)
    episodes, tasks = list(episode_indices), list(task_indices)
    if not episodes or len(episodes) != len(tasks):
        raise ValueError("episode/task columns must be non-empty with matching lengths")
    for episode, task in zip(episodes, tasks):
        grouped[int(episode)].add(int(task))
    ambiguous = {episode: values for episode, values in grouped.items() if len(values) != 1}
    if ambiguous:
        raise ValueError(f"episodes map to multiple tasks: {ambiguous}")
    return {episode: next(iter(grouped[episode])) for episode in sorted(grouped)}


def read_episode_task_mapping(dataset_root: str | Path) -> dict[int, int]:
    import pyarrow.dataset as ds

    root = Path(dataset_root)
    data_root = root / "data"
    if not data_root.is_dir():
        raise FileNotFoundError(f"LeRobot data directory not found: {data_root}")
    dataset = ds.dataset(data_root, format="parquet")
    required = {"episode_index", "task_index"}
    missing = required - set(dataset.schema.names)
    if missing:
        raise RuntimeError(f"parquet schema is missing columns: {sorted(missing)}")
    table = dataset.to_table(columns=["episode_index", "task_index"])
    return episode_task_mapping(
        table.column("episode_index").to_pylist(),
        table.column("task_index").to_pylist(),
    )


def stratified_episode_split(
    episode_to_task: Mapping[int, int], *, seed: int
) -> tuple[dict[str, list[int]], dict[str, dict[str, int]]]:
    if not episode_to_task:
        raise ValueError("cannot split no episodes")
    by_task: dict[int, list[int]] = defaultdict(list)
    for episode, task in episode_to_task.items():
        by_task[int(task)].append(int(episode))
    splits = {"train": [], "val": [], "test": []}
    task_counts = {}
    for offset, task in enumerate(sorted(by_task)):
        values = np.asarray(sorted(by_task[task]), dtype=np.int64)
        shuffled = np.random.default_rng(int(seed) + offset).permutation(values).tolist()
        train_count = math.floor(0.8 * len(shuffled))
        val_count = math.floor(0.1 * len(shuffled))
        splits["train"].extend(shuffled[:train_count])
        splits["val"].extend(shuffled[train_count : train_count + val_count])
        splits["test"].extend(shuffled[train_count + val_count :])
        task_counts[str(task)] = {
            "total": len(shuffled), "train": train_count, "val": val_count,
            "test": len(shuffled) - train_count - val_count,
        }
    for values in splits.values():
        values.sort()
    validate_split_membership(splits, set(map(int, episode_to_task)))
    return splits, task_counts


def validate_split_membership(
    splits: Mapping[str, Iterable[int]], all_episode_ids: set[int]
) -> None:
    sets = {name: set(map(int, splits[name])) for name in ("train", "val", "test")}
    if any(not values for values in sets.values()):
        raise ValueError("train/val/test splits must all be non-empty")
    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
        raise ValueError("train/val/test episode splits overlap")
    if set().union(*sets.values()) != set(all_episode_ids):
        raise ValueError("split union does not equal all dataset episode IDs")


def build_split_manifest(
    *, dataset_root: str | Path, episode_to_task: Mapping[int, int], seed: int
) -> dict[str, Any]:
    root = Path(dataset_root).resolve()
    splits, task_counts = stratified_episode_split(episode_to_task, seed=seed)
    info_path = root / "meta" / "info.json"
    dataset_info = {}
    if info_path.is_file():
        with info_path.open("r", encoding="utf-8") as handle:
            info = json.load(handle)
        dataset_info = {
            key: info.get(key) for key in ("codebase_version", "fps", "total_episodes")
        }
    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_identity": "armnet/armnetbench_v01_lerobot_so101",
        "dataset_root_at_build": str(root),
        "dataset_info": dataset_info,
        "seed": int(seed),
        "split_algorithm": SPLIT_ALGORITHM,
        "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
        "train_episode_ids": splits["train"],
        "val_episode_ids": splits["val"],
        "test_episode_ids": splits["test"],
        "episode_task_ids": {str(k): int(v) for k, v in episode_to_task.items()},
        "task_counts": task_counts,
    }


def validate_split_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if int(manifest.get("schema_version", -1)) != SPLIT_SCHEMA_VERSION:
        raise RuntimeError("unsupported state split manifest schema")
    if manifest.get("split_algorithm") != SPLIT_ALGORITHM:
        raise RuntimeError("unexpected state split algorithm")
    episode_tasks = {int(k): int(v) for k, v in manifest["episode_task_ids"].items()}
    splits = {
        name: [int(v) for v in manifest[f"{name}_episode_ids"]]
        for name in ("train", "val", "test")
    }
    validate_split_membership(splits, set(episode_tasks))
    return manifest


def load_split_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return validate_split_manifest(json.load(handle))


def episode_ids_for_split(manifest: dict[str, Any], split: str) -> list[int]:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    return [int(v) for v in manifest[f"{split}_episode_ids"]]


def validate_episode_tasks(
    manifest: dict[str, Any], observed_episode_to_task: Mapping[int, int]
) -> None:
    expected = {int(k): int(v) for k, v in manifest["episode_task_ids"].items()}
    unknown = set(map(int, observed_episode_to_task)) - set(expected)
    if unknown:
        raise RuntimeError(f"episodes are absent from state split manifest: {sorted(unknown)}")
    mismatches = {
        int(episode): (expected[int(episode)], int(task))
        for episode, task in observed_episode_to_task.items()
        if expected[int(episode)] != int(task)
    }
    if mismatches:
        raise RuntimeError(f"episode/task mapping differs from state manifest: {mismatches}")


def atomic_write_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
