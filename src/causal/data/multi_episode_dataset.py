from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch
from torch.utils.data import Dataset

from .common import (
    ActionStats,
    build_action_condition,
    build_frame_indices,
    cache_paths_from_manifest,
    compute_action_stats,
    effective_action_dim,
    load_episode_cache,
    valid_window_starts,
)


def _manifest_episode_id(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{context} must be an integer episode ID")
    return int(value)


def validate_causal_manifest(manifest: dict) -> dict:
    """Validate generic train/val/test episode-manifest consistency."""

    if not isinstance(manifest, dict):
        raise RuntimeError("manifest must be a JSON object")
    required = (
        "train_episode_ids",
        "val_episode_ids",
        "test_episode_ids",
        "episodes",
    )
    for key in required:
        if key not in manifest:
            raise RuntimeError(f"manifest is missing {key!r}")

    entries = manifest["episodes"]
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("manifest episodes must be a non-empty list")
    episode_ids = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise RuntimeError(f"episodes[{position}] must be an object")
        for key in ("episode_index", "cache_file"):
            if key not in entry:
                raise RuntimeError(f"episodes[{position}] is missing {key!r}")
        episode_id = _manifest_episode_id(
            entry["episode_index"], f"episodes[{position}].episode_index"
        )
        cache_file = entry["cache_file"]
        if not isinstance(cache_file, str) or not cache_file:
            raise RuntimeError(
                f"episodes[{position}].cache_file must be a non-empty string"
            )
        episode_ids.append(episode_id)
    if len(set(episode_ids)) != len(episode_ids):
        raise RuntimeError("manifest contains duplicate episode entries")
    episode_id_set = set(episode_ids)

    split_sets: dict[str, set[int]] = {}
    for split in ("train", "val", "test"):
        key = f"{split}_episode_ids"
        values = manifest[key]
        if not isinstance(values, list) or not values:
            raise RuntimeError(f"manifest split {split!r} must be a non-empty list")
        ids = [
            _manifest_episode_id(value, f"{key}[{position}]")
            for position, value in enumerate(values)
        ]
        if len(set(ids)) != len(ids):
            raise RuntimeError(f"manifest split {split!r} contains duplicate IDs")
        unknown = set(ids) - episode_id_set
        if unknown:
            raise RuntimeError(
                f"manifest split {split!r} references unknown episodes: "
                f"{sorted(unknown)}"
            )
        split_sets[split] = set(ids)

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_sets[left] & split_sets[right]
        if overlap:
            raise RuntimeError(
                f"manifest splits {left!r}/{right!r} overlap: {sorted(overlap)}"
            )
    split_union = set().union(*split_sets.values())
    if split_union != episode_id_set:
        missing = episode_id_set - split_union
        extra = split_union - episode_id_set
        raise RuntimeError(
            "manifest split union does not equal episode entries: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return manifest


def load_causal_manifest(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return validate_causal_manifest(manifest)


def episode_ids_for_split(manifest: dict, split: str) -> list[int]:
    key = f"{split}_episode_ids"
    if key not in manifest:
        raise ValueError(f"manifest has no {split!r} split")
    ids = [int(value) for value in manifest[key]]
    if not ids:
        raise ValueError(f"manifest split {split!r} is empty")
    return ids


class MultiEpisodeCausalDataset(Dataset):
    """Causal windows from an episode-level split; windows never cross episodes."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str,
        num_frames: int,
        frame_stride: int,
        representation: str,
        raw_action_dim: int = 6,
        normalize_actions: bool = True,
        action_stats: ActionStats | None = None,
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.manifest = load_causal_manifest(self.manifest_path)
        self.split = split
        self.episode_ids = episode_ids_for_split(self.manifest, split)
        self.num_frames = int(num_frames)
        self.frame_stride = int(frame_stride)
        self.representation = representation
        self.raw_action_dim = int(raw_action_dim)
        self.effective_action_dim = effective_action_dim(
            self.raw_action_dim, self.frame_stride, self.representation
        )
        self.normalize_actions = bool(normalize_actions)
        self.action_stats = action_stats
        if self.normalize_actions and self.action_stats is None:
            raise ValueError("MultiEpisodeCausalDataset requires explicit action_stats")
        if not self.normalize_actions and self.action_stats is not None:
            raise ValueError("do not pass action_stats when normalization is disabled")
        if self.action_stats is not None and self.action_stats.raw_dim != self.raw_action_dim:
            raise ValueError("action_stats raw dimension mismatch")

        paths = cache_paths_from_manifest(self.manifest, self.episode_ids)
        self.episodes = {
            episode_id: load_episode_cache(path, self.raw_action_dim)
            for episode_id, path in paths.items()
        }
        self.windows: list[tuple[int, int]] = []
        self.windows_per_episode: dict[int, int] = {}
        for episode_id in self.episode_ids:
            episode = self.episodes[episode_id]
            starts = valid_window_starts(
                0,
                len(episode["latents"]),
                self.num_frames,
                self.frame_stride,
            )
            if not starts:
                raise RuntimeError(f"episode {episode_id} has no valid windows")
            self.windows_per_episode[episode_id] = len(starts)
            self.windows.extend((episode_id, start) for start in starts)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_id, start = self.windows[index]
        episode = self.episodes[episode_id]
        indices = build_frame_indices(start, self.num_frames, self.frame_stride)
        action_cond, valid_mask, action_indices = build_action_condition(
            episode["actions"],
            indices,
            self.frame_stride,
            self.representation,
            self.action_stats,
            self.normalize_actions,
        )
        return {
            "latents": episode["latents"][indices],
            "action_cond": action_cond,
            "action_valid_mask": valid_mask,
            "frame_indices": episode["frame_indices"][indices],
            "cache_indices": indices,
            "action_indices": action_indices,
            "episode_idx": torch.tensor(episode_id, dtype=torch.long),
            "window_start": torch.tensor(start, dtype=torch.long),
        }


def compute_multi_episode_train_stats(
    manifest_path: str | Path,
    raw_action_dim: int = 6,
) -> ActionStats:
    """The sole multi-episode stats path: manifest train episode IDs only."""

    manifest = load_causal_manifest(manifest_path)
    train_ids = episode_ids_for_split(manifest, "train")
    paths = cache_paths_from_manifest(manifest, train_ids)
    actions = [
        load_episode_cache(paths[episode_id], raw_action_dim)["actions"]
        for episode_id in train_ids
    ]
    return compute_action_stats(
        actions,
        raw_action_dim,
        source="multi_episode:train_episode_ids:" + ",".join(map(str, train_ids)),
    )
