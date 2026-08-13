from __future__ import annotations

import json

import pytest
import torch

import src.causal.datasets as dataset_factory
from src.causal.checkpointing import action_stats_from_checkpoint
from src.causal.config import resolve_config
from src.causal.data.common import ActionStats
from src.causal.data.multi_episode_dataset import compute_multi_episode_train_stats
from src.causal.data.single_episode_dataset import compute_single_episode_train_stats
from src.causal.datasets import build_evaluation_dataset, build_training_datasets


def _cache(path, episode_id: int, actions: torch.Tensor):
    length = len(actions)
    torch.save(
        {
            "episode_index": episode_id,
            "latents": torch.randn(length, 2, 2, 2),
            "actions": actions.float(),
            "frame_indices": torch.arange(length),
        },
        path,
    )


def _base_config(data: dict, representation="fast_chunk"):
    return resolve_config(
        {
            "experiment": {"name": "test", "seed": 1},
            "data": data,
            "temporal": {"num_frames": 3, "num_history": 1, "frame_stride": 2},
            "action": {
                "raw_action_dim": 6,
                "alignment": "causal",
                "representation": representation,
                "normalize": True,
                "null_condition": "zero_embedding",
            },
            "model": {
                "in_channels": 2,
                "patch_size": 1,
                "hidden_size": 8,
                "depth": 1,
                "num_heads": 1,
                "mlp_ratio": 2.0,
            },
            "flow_matching": {"future_only_loss": True},
            "train": {"batch_size": 1, "steps": 1, "lr": 1e-4},
            "checkpoint": {
                "output_dir": "unused",
                "save_best": True,
                "save_last": True,
            },
            "latent": {"convention": "posterior_sample_times_scaling_no_shift"},
        }
    )


def test_multi_stats_use_only_train_episode_ids(tmp_path):
    train_path = tmp_path / "train.pt"
    val_path = tmp_path / "val.pt"
    _cache(train_path, 0, torch.arange(60).reshape(10, 6))
    _cache(val_path, 1, torch.full((10, 6), 1_000_000.0))
    manifest = {
        "train_episode_ids": [0],
        "val_episode_ids": [1],
        "episodes": [
            {"episode_index": 0, "cache_file": str(train_path)},
            {"episode_index": 1, "cache_file": str(val_path)},
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    stats = compute_multi_episode_train_stats(manifest_path)
    expected = torch.arange(60, dtype=torch.float32).reshape(10, 6)
    assert torch.equal(stats.mean, expected.mean(0))
    assert torch.equal(stats.std, expected.std(0))


def test_train_and_val_share_exact_same_stats_object(tmp_path):
    train_path = tmp_path / "train.pt"
    val_path = tmp_path / "val.pt"
    _cache(train_path, 0, torch.arange(72).reshape(12, 6))
    _cache(val_path, 1, torch.arange(72, 144).reshape(12, 6))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "train_episode_ids": [0],
                "val_episode_ids": [1],
                "episodes": [
                    {"episode_index": 0, "cache_file": str(train_path)},
                    {"episode_index": 1, "cache_file": str(val_path)},
                ],
            }
        ),
        encoding="utf-8",
    )
    config = _base_config(
        {"mode": "multi_episode", "manifest_path": str(manifest_path)}
    )
    train, val, stats = build_training_datasets(config)
    assert train.action_stats is stats
    assert val.action_stats is stats
    assert torch.equal(train.action_stats.mean, val.action_stats.mean)
    assert torch.equal(train.action_stats.std, val.action_stats.std)


def test_single_episode_stats_use_only_train_temporal_split(tmp_path):
    path = tmp_path / "episode.pt"
    actions = torch.cat((torch.arange(48).reshape(8, 6), torch.full((4, 6), 9999)))
    _cache(path, 0, actions)
    stats = compute_single_episode_train_stats(
        path, {"train": [0, 8], "val": [8, 12]}
    )
    assert torch.equal(stats.mean, actions[:8].float().mean(0))
    assert torch.equal(stats.std, actions[:8].float().std(0))


def test_evaluation_uses_checkpoint_stats_without_recomputation(tmp_path, monkeypatch):
    path = tmp_path / "episode.pt"
    _cache(path, 0, torch.arange(72).reshape(12, 6))
    config = _base_config(
        {
            "mode": "single_episode",
            "cache_path": str(path),
            "episode_id": 0,
            "split_ranges": {"train": [0, 7], "val": [7, 12]},
        },
        representation="shifted_sampled",
    )
    checkpoint_stats = ActionStats(
        torch.full((6,), 123.0), torch.full((6,), 7.0), source="checkpoint"
    )
    checkpoint = {"config": config, "action_stats": checkpoint_stats.to_dict()}

    def forbidden(_config):
        raise AssertionError("evaluation recomputed statistics")

    monkeypatch.setattr(dataset_factory, "compute_training_action_stats", forbidden)
    dataset = build_evaluation_dataset(checkpoint, split="val")
    loaded = action_stats_from_checkpoint(checkpoint)
    assert torch.equal(dataset.action_stats.mean, loaded.mean)
    assert torch.equal(dataset.action_stats.std, loaded.std)
    raw = torch.arange(72).reshape(12, 6).float()[7]
    assert torch.equal(dataset[0]["action_cond"][1], (raw - loaded.mean) / loaded.std)


def test_normalization_happens_exactly_once(tmp_path):
    path = tmp_path / "episode.pt"
    actions = torch.arange(72).reshape(12, 6).float()
    _cache(path, 0, actions)
    stats = ActionStats(torch.full((6,), 10.0), torch.full((6,), 2.0))
    config = _base_config(
        {
            "mode": "single_episode",
            "cache_path": str(path),
            "episode_id": 0,
            "split_ranges": {"train": [0, 7], "val": [7, 12]},
        },
        representation="shifted_sampled",
    )
    checkpoint = {"config": config, "action_stats": stats.to_dict()}
    sample = build_evaluation_dataset(checkpoint, split="val")[0]
    expected = (actions[7] - stats.mean) / stats.std
    assert torch.equal(sample["action_cond"][1], expected)
    assert not torch.equal(sample["action_cond"][1], (expected - stats.mean) / stats.std)


def test_test_dataset_explicitly_uses_checkpoint_stats(tmp_path):
    train_path = tmp_path / "train.pt"
    val_path = tmp_path / "val.pt"
    test_path = tmp_path / "test.pt"
    _cache(train_path, 0, torch.arange(72).reshape(12, 6))
    _cache(val_path, 1, torch.arange(72, 144).reshape(12, 6))
    test_actions = torch.arange(144, 216).reshape(12, 6).float()
    _cache(test_path, 2, test_actions)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "train_episode_ids": [0],
                "val_episode_ids": [1],
                "test_episode_ids": [2],
                "episodes": [
                    {"episode_index": 0, "cache_file": str(train_path)},
                    {"episode_index": 1, "cache_file": str(val_path)},
                    {"episode_index": 2, "cache_file": str(test_path)},
                ],
            }
        ),
        encoding="utf-8",
    )
    config = _base_config(
        {"mode": "multi_episode", "manifest_path": str(manifest_path)},
        representation="shifted_sampled",
    )
    stats = ActionStats(torch.full((6,), 50.0), torch.full((6,), 5.0))
    checkpoint = {"config": config, "action_stats": stats.to_dict()}
    dataset = build_evaluation_dataset(checkpoint, split="test")
    assert dataset.episode_ids == [2]
    assert torch.equal(dataset.action_stats.mean, stats.mean)
    assert torch.equal(dataset.action_stats.std, stats.std)
    assert torch.equal(
        dataset[0]["action_cond"][1],
        (test_actions[0] - stats.mean) / stats.std,
    )


def test_action_stats_are_required_for_evaluation_dataset(tmp_path):
    path = tmp_path / "episode.pt"
    _cache(path, 0, torch.arange(72).reshape(12, 6))
    config = _base_config(
        {
            "mode": "single_episode",
            "cache_path": str(path),
            "episode_id": 0,
            "split_ranges": {"train": [0, 7], "val": [7, 12]},
        }
    )
    with pytest.raises(KeyError):
        build_evaluation_dataset({"config": config}, split="val")
