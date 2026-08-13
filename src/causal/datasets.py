from __future__ import annotations

from pathlib import Path

from .data.common import ActionStats
from .data.multi_episode_dataset import (
    MultiEpisodeCausalDataset,
    compute_multi_episode_train_stats,
    load_causal_manifest,
)
from .data.single_episode_dataset import (
    SingleEpisodeCausalDataset,
    compute_single_episode_train_stats,
)


def _single_cache_path(config: dict) -> Path:
    data = config["data"]
    if "cache_path" in data:
        return Path(data["cache_path"])
    return Path(data["cache_root"]) / f"episode_{int(data['episode_id']):03d}.pt"


def compute_training_action_stats(config: dict) -> ActionStats:
    """The only public path that computes causal normalization statistics."""

    raw_dim = config["action"]["raw_action_dim"]
    if config["data"]["mode"] == "single_episode":
        return compute_single_episode_train_stats(
            _single_cache_path(config),
            config["data"]["split_ranges"],
            raw_dim,
        )
    return compute_multi_episode_train_stats(
        config["data"]["manifest_path"], raw_dim
    )


def build_dataset(
    config: dict,
    *,
    split: str,
    action_stats: ActionStats,
):
    common = dict(
        split=split,
        num_frames=config["temporal"]["num_frames"],
        frame_stride=config["temporal"]["frame_stride"],
        representation=config["action"]["representation"],
        raw_action_dim=config["action"]["raw_action_dim"],
        normalize_actions=True,
        action_stats=action_stats,
    )
    if config["data"]["mode"] == "single_episode":
        return SingleEpisodeCausalDataset(
            _single_cache_path(config),
            split_ranges=config["data"]["split_ranges"],
            **common,
        )
    return MultiEpisodeCausalDataset(
        config["data"]["manifest_path"], **common
    )


def build_training_datasets(config: dict):
    stats = compute_training_action_stats(config)
    train_dataset = build_dataset(config, split="train", action_stats=stats)
    val_dataset = build_dataset(config, split="val", action_stats=stats)
    return train_dataset, val_dataset, stats


def build_evaluation_dataset(
    checkpoint: dict,
    *,
    split: str = "val",
):
    """Build evaluation data only from checkpoint config and checkpoint stats."""

    from .checkpointing import action_stats_from_checkpoint

    config = checkpoint["config"]
    stats = action_stats_from_checkpoint(checkpoint)
    return build_dataset(config, split=split, action_stats=stats)


def data_info(config: dict) -> dict:
    if config["data"]["mode"] == "single_episode":
        episode_id = int(config["data"]["episode_id"])
        return {
            "mode": "single_episode",
            "episode_id": episode_id,
            "train_episode_ids": [episode_id],
            "val_episode_ids": [episode_id],
            "split_ranges": config["data"]["split_ranges"],
        }
    manifest = load_causal_manifest(config["data"]["manifest_path"])
    return {
        "mode": "multi_episode",
        "train_episode_ids": list(map(int, manifest["train_episode_ids"])),
        "val_episode_ids": list(map(int, manifest["val_episode_ids"])),
        "test_episode_ids": list(map(int, manifest.get("test_episode_ids", []))),
        "manifest_path": config["data"]["manifest_path"],
    }
