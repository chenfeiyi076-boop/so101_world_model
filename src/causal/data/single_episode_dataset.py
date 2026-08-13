from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .common import (
    ActionStats,
    build_action_condition,
    build_frame_indices,
    compute_action_stats,
    effective_action_dim,
    load_episode_cache,
    valid_window_starts,
)


class SingleEpisodeCausalDataset(Dataset):
    """Causal windows from one episode and one explicit temporal split."""

    def __init__(
        self,
        cache_path: str | Path,
        *,
        split: str,
        split_ranges: dict[str, tuple[int, int] | list[int]],
        num_frames: int,
        frame_stride: int,
        representation: str,
        raw_action_dim: int = 6,
        normalize_actions: bool = True,
        action_stats: ActionStats | None = None,
    ) -> None:
        super().__init__()
        if split not in split_ranges:
            raise ValueError(f"split {split!r} is absent from split_ranges")
        self.cache_path = Path(cache_path)
        self.split = split
        self.split_ranges = {
            name: (int(bounds[0]), int(bounds[1]))
            for name, bounds in split_ranges.items()
        }
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
            raise ValueError("SingleEpisodeCausalDataset requires explicit action_stats")
        if not self.normalize_actions and self.action_stats is not None:
            raise ValueError("do not pass action_stats when normalization is disabled")
        if self.action_stats is not None and self.action_stats.raw_dim != self.raw_action_dim:
            raise ValueError("action_stats raw dimension mismatch")

        self.episode = load_episode_cache(self.cache_path, self.raw_action_dim)
        self.episode_id = int(self.episode["episode_index"])
        split_start, split_end = self.split_ranges[self.split]
        if split_end > len(self.episode["latents"]):
            raise ValueError("split range exceeds episode length")
        self.window_starts = valid_window_starts(
            split_start,
            split_end,
            self.num_frames,
            self.frame_stride,
        )
        if not self.window_starts:
            raise RuntimeError(f"split {split!r} has no valid temporal windows")

    def __len__(self) -> int:
        return len(self.window_starts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = self.window_starts[index]
        indices = build_frame_indices(start, self.num_frames, self.frame_stride)
        action_cond, valid_mask, action_indices = build_action_condition(
            self.episode["actions"],
            indices,
            self.frame_stride,
            self.representation,
            self.action_stats,
            self.normalize_actions,
        )
        return {
            "latents": self.episode["latents"][indices],
            "action_cond": action_cond,
            "action_valid_mask": valid_mask,
            "frame_indices": self.episode["frame_indices"][indices],
            "cache_indices": indices,
            "action_indices": action_indices,
            "episode_idx": torch.tensor(self.episode_id, dtype=torch.long),
            "window_start": torch.tensor(start, dtype=torch.long),
        }


def compute_single_episode_train_stats(
    cache_path: str | Path,
    split_ranges: dict[str, tuple[int, int] | list[int]],
    raw_action_dim: int = 6,
) -> ActionStats:
    """The sole single-episode stats path: raw actions from train range only."""

    if "train" not in split_ranges:
        raise ValueError("single-episode normalization requires a train split")
    episode = load_episode_cache(cache_path, raw_action_dim)
    start, end = map(int, split_ranges["train"])
    if start < 0 or end > len(episode["actions"]) or end <= start:
        raise ValueError("invalid single-episode train split")
    return compute_action_stats(
        [episode["actions"][start:end]],
        raw_action_dim,
        source=f"single_episode:{episode['episode_index']}:train_range:{start}:{end}",
    )
