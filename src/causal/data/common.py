from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch


SUPPORTED_REPRESENTATIONS = {"shifted_sampled", "fast_chunk"}
SUPPORTED_FRAME_STRIDES = {1, 2, 4}
NORMALIZATION_METHOD = "per_raw_dimension_zscore_train_only"


@dataclass(frozen=True)
class ActionStats:
    """One fixed set of train-only statistics for raw 6D actions."""

    mean: torch.Tensor
    std: torch.Tensor
    source: str = "training_data"
    method: str = NORMALIZATION_METHOD

    def __post_init__(self) -> None:
        mean = torch.as_tensor(self.mean, dtype=torch.float32).detach().cpu()
        std = torch.as_tensor(self.std, dtype=torch.float32).detach().cpu()
        if mean.ndim != 1 or std.shape != mean.shape:
            raise ValueError("action mean/std must be matching 1D tensors")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("action mean/std must be finite")
        if torch.any(std <= 0):
            raise ValueError("action std must be strictly positive")
        object.__setattr__(self, "mean", mean.contiguous())
        object.__setattr__(self, "std", std.contiguous())

    @property
    def raw_dim(self) -> int:
        return int(self.mean.numel())

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.clone(),
            "std": self.std.clone(),
            "source": self.source,
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "ActionStats":
        return cls(
            mean=value["mean"],
            std=value["std"],
            source=value.get("source", "checkpoint_training_data"),
            method=value.get("method", NORMALIZATION_METHOD),
        )


def effective_action_dim(
    raw_action_dim: int,
    frame_stride: int,
    representation: str,
) -> int:
    _validate_temporal_arguments(1, frame_stride)
    if raw_action_dim <= 0:
        raise ValueError("raw_action_dim must be positive")
    if representation not in SUPPORTED_REPRESENTATIONS:
        raise ValueError(f"unsupported action representation: {representation!r}")
    if representation == "shifted_sampled":
        return int(raw_action_dim)
    return int(raw_action_dim * frame_stride)


def _validate_temporal_arguments(num_frames: int, frame_stride: int) -> None:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if frame_stride not in SUPPORTED_FRAME_STRIDES:
        raise ValueError(
            f"frame_stride must be one of {sorted(SUPPORTED_FRAME_STRIDES)}"
        )


def temporal_span(num_frames: int, frame_stride: int) -> int:
    _validate_temporal_arguments(num_frames, frame_stride)
    return 1 + (num_frames - 1) * frame_stride


def build_frame_indices(
    start: int,
    num_frames: int,
    frame_stride: int,
) -> torch.Tensor:
    _validate_temporal_arguments(num_frames, frame_stride)
    if start < 0:
        raise ValueError("start must be non-negative")
    return start + torch.arange(num_frames, dtype=torch.long) * frame_stride


def build_shifted_sampled_action_indices(
    frame_indices: torch.Tensor,
) -> torch.Tensor:
    """Map [z_s, z_s+k, ...] to [NULL, a_s, a_s+k, ...]."""

    frame_indices = torch.as_tensor(frame_indices, dtype=torch.long)
    if frame_indices.ndim != 1 or frame_indices.numel() == 0:
        raise ValueError("frame_indices must be a non-empty 1D tensor")
    result = torch.full_like(frame_indices, -1)
    result[1:] = frame_indices[:-1]
    return result


def build_fast_chunk_action_indices(
    frame_indices: torch.Tensor,
    frame_stride: int,
) -> torch.Tensor:
    """Return causal high-frequency action intervals, with -1 for NULL."""

    frame_indices = torch.as_tensor(frame_indices, dtype=torch.long)
    _validate_temporal_arguments(int(frame_indices.numel()), frame_stride)
    if frame_indices.ndim != 1 or frame_indices.numel() == 0:
        raise ValueError("frame_indices must be a non-empty 1D tensor")
    if frame_indices.numel() > 1:
        deltas = frame_indices[1:] - frame_indices[:-1]
        if not torch.equal(deltas, torch.full_like(deltas, frame_stride)):
            raise ValueError("frame_indices do not match frame_stride")

    result = torch.full(
        (frame_indices.numel(), frame_stride),
        -1,
        dtype=torch.long,
    )
    offsets = torch.arange(frame_stride, dtype=torch.long)
    if frame_indices.numel() > 1:
        result[1:] = frame_indices[:-1, None] + offsets[None, :]
    return result


def normalize_raw_actions(
    raw_actions: torch.Tensor,
    stats: ActionStats,
) -> torch.Tensor:
    """Normalize raw actions once, before any chunk flattening."""

    raw_actions = torch.as_tensor(raw_actions, dtype=torch.float32)
    if raw_actions.shape[-1] != stats.raw_dim:
        raise ValueError(
            f"expected raw action dim {stats.raw_dim}, got {raw_actions.shape[-1]}"
        )
    return (raw_actions - stats.mean) / stats.std


def compute_action_stats(
    action_tensors: Iterable[torch.Tensor],
    raw_action_dim: int,
    *,
    source: str,
) -> ActionStats:
    """Compute statistics exactly once from a caller-selected training set."""

    collected = []
    for actions in action_tensors:
        actions = torch.as_tensor(actions, dtype=torch.float32)
        if actions.ndim != 2 or actions.shape[1] != raw_action_dim:
            raise ValueError(
                f"expected actions [N,{raw_action_dim}], got {tuple(actions.shape)}"
            )
        collected.append(actions)
    if not collected:
        raise ValueError("cannot compute action statistics from no actions")
    all_actions = torch.cat(collected, dim=0)
    if all_actions.shape[0] < 2:
        raise ValueError("at least two training actions are required")
    return ActionStats(
        mean=all_actions.mean(dim=0),
        std=all_actions.std(dim=0, unbiased=True).clamp_min(1e-6),
        source=source,
    )


def load_episode_cache(path: str | Path, raw_action_dim: int) -> dict:
    path = Path(path)
    cache = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("latents", "actions"):
        if key not in cache:
            raise RuntimeError(f"cache {path} is missing {key!r}")
    latents = torch.as_tensor(cache["latents"]).float().contiguous()
    actions = torch.as_tensor(cache["actions"]).float().contiguous()
    if latents.ndim != 4:
        raise RuntimeError(f"latents must be [N,C,H,W], got {tuple(latents.shape)}")
    if actions.shape != (len(latents), raw_action_dim):
        raise RuntimeError(
            f"actions must be [{len(latents)},{raw_action_dim}], got {tuple(actions.shape)}"
        )
    if not torch.isfinite(latents).all() or not torch.isfinite(actions).all():
        raise RuntimeError(f"non-finite cache values in {path}")
    frame_indices = torch.as_tensor(
        cache.get("frame_indices", torch.arange(len(latents))), dtype=torch.long
    )
    if frame_indices.shape != (len(latents),):
        raise RuntimeError("frame_indices length mismatch")
    return {
        "latents": latents,
        "actions": actions,
        "frame_indices": frame_indices,
        "episode_index": int(cache.get("episode_index", 0)),
        "cache_path": str(path),
    }


def build_action_condition(
    actions: torch.Tensor,
    frame_indices: torch.Tensor,
    frame_stride: int,
    representation: str,
    stats: ActionStats | None,
    normalize: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build condition, validity mask, and auditable action indices."""

    if normalize and stats is None:
        raise ValueError(
            "normalize=True requires explicit fixed action_stats; datasets never compute stats"
        )
    if not normalize and stats is not None:
        raise ValueError("action_stats must be omitted when normalize=False")

    raw_dim = int(actions.shape[-1])
    valid_mask = torch.ones(frame_indices.numel(), dtype=torch.bool)
    valid_mask[0] = False

    if representation == "shifted_sampled":
        action_indices = build_shifted_sampled_action_indices(frame_indices)
        condition = torch.zeros(
            frame_indices.numel(), raw_dim, dtype=torch.float32
        )
        if frame_indices.numel() > 1:
            selected = actions[action_indices[1:]]
            if normalize:
                selected = normalize_raw_actions(selected, stats)
            condition[1:] = selected
    elif representation == "fast_chunk":
        action_indices = build_fast_chunk_action_indices(
            frame_indices, frame_stride
        )
        condition = torch.zeros(
            frame_indices.numel(), frame_stride * raw_dim, dtype=torch.float32
        )
        if frame_indices.numel() > 1:
            selected = actions[action_indices[1:]]
            # Critical: normalize every raw 6D action with the same 6D stats,
            # then flatten. Never derive position-specific chunk statistics.
            if normalize:
                selected = normalize_raw_actions(selected, stats)
            condition[1:] = selected.reshape(frame_indices.numel() - 1, -1)
    else:
        raise ValueError(f"unsupported action representation: {representation!r}")

    return condition, valid_mask, action_indices


def valid_window_starts(
    segment_start: int,
    segment_end: int,
    num_frames: int,
    frame_stride: int,
) -> list[int]:
    span = temporal_span(num_frames, frame_stride)
    if segment_start < 0 or segment_end < segment_start:
        raise ValueError("invalid temporal segment")
    count = segment_end - segment_start - span + 1
    if count <= 0:
        return []
    return list(range(segment_start, segment_start + count))


def cache_paths_from_manifest(
    manifest: dict,
    episode_ids: Sequence[int],
) -> dict[int, Path]:
    mapping = {
        int(item["episode_index"]): Path(item["cache_file"])
        for item in manifest["episodes"]
    }
    missing = set(map(int, episode_ids)) - set(mapping)
    if missing:
        raise RuntimeError(f"episodes missing from cache manifest: {sorted(missing)}")
    return {int(i): mapping[int(i)] for i in episode_ids}
