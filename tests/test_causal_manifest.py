from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import torch

from scripts.build_so101_causal_manifest import (
    EXPECTED_EPISODE_COUNT,
    atomic_json_write,
    build_manifest,
    episode_task_mapping,
    stratified_episode_split,
    validate_split_ratios,
)
from src.causal.data.common import ActionStats
from src.causal.data.multi_episode_dataset import (
    MultiEpisodeCausalDataset,
    load_causal_manifest,
)
from src.causal.datasets import build_dataset, compute_training_action_stats


def _manifest(tmp_path: Path) -> dict:
    return {
        "train_episode_ids": [0],
        "val_episode_ids": [1],
        "test_episode_ids": [2],
        "episodes": [
            {
                "episode_index": episode_id,
                "task_index": episode_id % 2,
                "cache_file": str(tmp_path / f"episode_{episode_id:03d}.pt"),
            }
            for episode_id in range(3)
        ],
    }


def _write_manifest(tmp_path: Path, manifest: dict) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_valid_manifest_passes_and_preserves_task_index(tmp_path: Path):
    loaded = load_causal_manifest(_write_manifest(tmp_path, _manifest(tmp_path)))
    assert loaded["episodes"][1]["task_index"] == 1


def test_duplicate_train_id_fails(tmp_path: Path):
    manifest = _manifest(tmp_path)
    manifest["train_episode_ids"] = [0, 0]
    with pytest.raises(RuntimeError, match="train.*duplicate"):
        load_causal_manifest(_write_manifest(tmp_path, manifest))


def test_duplicate_episode_entry_fails(tmp_path: Path):
    manifest = _manifest(tmp_path)
    manifest["episodes"].append(copy.deepcopy(manifest["episodes"][0]))
    with pytest.raises(RuntimeError, match="duplicate episode entries"):
        load_causal_manifest(_write_manifest(tmp_path, manifest))


@pytest.mark.parametrize(
    ("left", "right"),
    [("train", "val"), ("train", "test"), ("val", "test")],
)
def test_split_overlap_fails(tmp_path: Path, left: str, right: str):
    manifest = _manifest(tmp_path)
    manifest[f"{right}_episode_ids"] = list(manifest[f"{left}_episode_ids"])
    with pytest.raises(RuntimeError, match="overlap"):
        load_causal_manifest(_write_manifest(tmp_path, manifest))


def test_split_union_missing_episode_fails(tmp_path: Path):
    manifest = _manifest(tmp_path)
    manifest["episodes"].append(
        {"episode_index": 3, "cache_file": str(tmp_path / "episode_003.pt")}
    )
    with pytest.raises(RuntimeError, match="split union"):
        load_causal_manifest(_write_manifest(tmp_path, manifest))


def test_split_reference_to_unknown_episode_fails(tmp_path: Path):
    manifest = _manifest(tmp_path)
    manifest["test_episode_ids"] = [99]
    with pytest.raises(RuntimeError, match="unknown episodes.*99"):
        load_causal_manifest(_write_manifest(tmp_path, manifest))


def _synthetic_episode_tasks() -> dict[int, int]:
    return {
        task_id * 20 + offset: task_id
        for task_id in range(3)
        for offset in range(20)
    }


def test_stratified_split_is_deterministic_and_seed_sensitive():
    mapping = _synthetic_episode_tasks()
    first, first_counts = stratified_episode_split(
        mapping, seed=42, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )
    repeated, repeated_counts = stratified_episode_split(
        mapping, seed=42, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )
    changed, _ = stratified_episode_split(
        mapping, seed=43, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )
    assert first == repeated
    assert first_counts == repeated_counts
    assert first != changed


def test_stratified_split_is_complete_disjoint_and_has_expected_task_counts():
    mapping = _synthetic_episode_tasks()
    splits, counts = stratified_episode_split(
        mapping, seed=42, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )
    split_sets = {name: set(ids) for name, ids in splits.items()}
    assert split_sets["train"] | split_sets["val"] | split_sets["test"] == set(mapping)
    assert split_sets["train"].isdisjoint(split_sets["val"])
    assert split_sets["train"].isdisjoint(split_sets["test"])
    assert split_sets["val"].isdisjoint(split_sets["test"])
    for episode_id in mapping:
        assert sum(episode_id in ids for ids in split_sets.values()) == 1
    for task_id in range(3):
        assert counts[str(task_id)] == {
            "total": 20,
            "train": 16,
            "val": 2,
            "test": 2,
        }
        for split, expected in (("train", 16), ("val", 2), ("test", 2)):
            actual = sum(mapping[episode_id] == task_id for episode_id in splits[split])
            assert actual == expected


def test_ratio_sum_invalid_fails():
    with pytest.raises(ValueError, match="sum to 1"):
        validate_split_ratios(0.8, 0.1, 0.2)


def test_missing_cache_file_fails_formal_builder(tmp_path: Path):
    mapping = {
        episode_id: episode_id % 8
        for episode_id in range(EXPECTED_EPISODE_COUNT)
    }
    with pytest.raises(FileNotFoundError, match="episode 0"):
        build_manifest(
            dataset_root=tmp_path / "dataset",
            cache_root=tmp_path / "cache",
            episode_to_task=mapping,
            seed=42,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
        )


def test_episode_with_multiple_task_indices_fails():
    with pytest.raises(ValueError, match="multiple task_index"):
        episode_task_mapping([0, 0, 1], [3, 4, 5])


def test_atomic_manifest_json_round_trip_is_stable(tmp_path: Path):
    manifest = _manifest(tmp_path)
    path = tmp_path / "manifest.json"
    atomic_json_write(manifest, path)
    first_text = path.read_text(encoding="utf-8")
    loaded = json.loads(first_text)
    atomic_json_write(loaded, path)
    assert json.loads(path.read_text(encoding="utf-8")) == manifest
    assert path.read_text(encoding="utf-8") == first_text
    assert not path.with_name(path.name + ".tmp").exists()


def _cache(path: Path, episode_id: int, offset: float) -> None:
    length = 20
    actions = torch.arange(length * 6, dtype=torch.float32).reshape(length, 6)
    torch.save(
        {
            "episode_index": episode_id,
            "latents": torch.full((length, 2, 2, 2), offset),
            "actions": actions + offset,
            "frame_indices": torch.arange(length),
        },
        path,
    )


def _dataset_manifest(tmp_path: Path) -> Path:
    entries = []
    for episode_id in range(3):
        path = tmp_path / f"episode_{episode_id:03d}.pt"
        _cache(path, episode_id, float(episode_id * 1000))
        entries.append({"episode_index": episode_id, "cache_file": str(path)})
    return _write_manifest(
        tmp_path,
        {
            "train_episode_ids": [0],
            "val_episode_ids": [1],
            "test_episode_ids": [2],
            "episodes": entries,
        },
    )


@pytest.mark.parametrize(
    ("stride", "representation", "expected_frames", "expected_actions", "action_dim"),
    [
        (1, "shifted_sampled", [0, 1, 2, 3], [-1, 0, 1, 2], 6),
        (2, "shifted_sampled", [0, 2, 4, 6], [-1, 0, 2, 4], 6),
        (4, "shifted_sampled", [0, 4, 8, 12], [-1, 0, 4, 8], 6),
        (
            2,
            "fast_chunk",
            [0, 2, 4, 6],
            [[-1, -1], [0, 1], [2, 3], [4, 5]],
            12,
        ),
        (
            4,
            "fast_chunk",
            [0, 4, 8, 12],
            [
                [-1, -1, -1, -1],
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
            ],
            24,
        ),
    ],
)
def test_multi_dataset_smoke_preserves_causal_semantics(
    tmp_path: Path,
    stride: int,
    representation: str,
    expected_frames: list[int],
    expected_actions: list,
    action_dim: int,
):
    manifest_path = _dataset_manifest(tmp_path)
    config = {
        "data": {"mode": "multi_episode", "manifest_path": str(manifest_path)},
        "temporal": {"num_frames": 4, "frame_stride": stride},
        "action": {
            "raw_action_dim": 6,
            "representation": representation,
        },
    }
    stats = compute_training_action_stats(config)
    train = build_dataset(config, split="train", action_stats=stats)
    val = build_dataset(config, split="val", action_stats=stats)
    assert isinstance(stats, ActionStats)
    assert isinstance(train, MultiEpisodeCausalDataset)
    assert isinstance(val, MultiEpisodeCausalDataset)
    assert len(train) > 0 and len(val) > 0
    sample = train[0]
    assert int(sample["episode_idx"]) == 0
    assert int(sample["window_start"]) == 0
    assert sample["cache_indices"].tolist() == expected_frames
    assert sample["frame_indices"].tolist() == expected_frames
    assert sample["action_indices"].tolist() == expected_actions
    assert sample["action_valid_mask"].tolist() == [False, True, True, True]
    assert sample["latents"].shape == (4, 2, 2, 2)
    assert sample["action_cond"].shape == (4, action_dim)
    assert torch.equal(sample["action_cond"][0], torch.zeros(action_dim))
    assert math.isfinite(float(sample["action_cond"][1:].sum()))
