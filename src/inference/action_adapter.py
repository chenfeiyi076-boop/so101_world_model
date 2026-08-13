from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.causal.data.common import ActionStats


@dataclass
class AdaptedActions:
    condition: torch.Tensor
    valid_mask: torch.Tensor
    action_indices: torch.Tensor
    raw_condition: torch.Tensor


def _rows_for_raw_indices(
    values: torch.Tensor,
    cache_frame_indices: torch.Tensor,
    requested: torch.Tensor,
) -> torch.Tensor:
    values = torch.as_tensor(values, dtype=torch.float32)
    cache_frame_indices = torch.as_tensor(cache_frame_indices, dtype=torch.long)
    requested = torch.as_tensor(requested, dtype=torch.long)
    lookup = {int(raw): row for row, raw in enumerate(cache_frame_indices.tolist())}
    flat = requested.reshape(-1).tolist()
    missing = sorted({int(raw) for raw in flat if int(raw) not in lookup})
    if missing:
        raise IndexError(f"episode cache is missing raw frame indices {missing}")
    rows = torch.tensor([lookup[int(raw)] for raw in flat], dtype=torch.long)
    return values[rows].reshape(*requested.shape, values.shape[-1])


def _raw_control(
    actions: torch.Tensor,
    states: torch.Tensor | None,
    cache_frame_indices: torch.Tensor,
    indices: torch.Tensor,
    control_representation: str,
) -> torch.Tensor:
    commands = _rows_for_raw_indices(actions, cache_frame_indices, indices)
    if control_representation == "absolute":
        return commands
    if states is None:
        raise RuntimeError(
            "legacy delta action requires cache 'states'; it cannot be reconstructed "
            "from actions alone"
        )
    return commands - _rows_for_raw_indices(states, cache_frame_indices, indices)


def _normalize(raw: torch.Tensor, stats: ActionStats) -> torch.Tensor:
    if raw.shape[-1] != stats.raw_dim:
        raise ValueError("raw action dimension does not match checkpoint statistics")
    return (raw - stats.mean) / stats.std


def adapt_legacy_actions(
    *,
    actions: torch.Tensor,
    states: torch.Tensor | None,
    cache_frame_indices: torch.Tensor,
    frame_indices: torch.Tensor,
    config: dict[str, Any],
    stats: ActionStats,
) -> AdaptedActions:
    temporal = config["temporal"]
    action = config["action"]
    frames = torch.as_tensor(frame_indices, dtype=torch.long)
    stride = int(temporal["frame_stride"])
    control_representation = action["control_representation"]

    if action["representation"] == "sampled":
        if action["alignment"] != "synchronized_legacy":
            raise ValueError("legacy sampled adapter requires synchronized_legacy")
        indices = frames.clone()
        raw = _raw_control(
            actions,
            states,
            cache_frame_indices,
            indices,
            control_representation,
        )
        condition = _normalize(raw, stats)
        valid = torch.ones(len(frames), dtype=torch.bool)
        return AdaptedActions(condition, valid, indices, raw)

    if action["alignment"] != "causal_chunk_legacy":
        raise ValueError("legacy chunk adapter requires causal_chunk_legacy")
    indices = torch.empty((len(frames), stride), dtype=torch.long)
    episode_start = int(torch.as_tensor(cache_frame_indices).min())
    offsets = torch.arange(-stride, 0, dtype=torch.long)
    indices[:] = (frames[:, None] + offsets[None, :]).clamp_min(episode_start)
    raw6 = _raw_control(
        actions,
        states,
        cache_frame_indices,
        indices,
        control_representation,
    )
    raw = raw6.reshape(len(frames), -1)
    condition = _normalize(raw6, stats).reshape(len(frames), -1)
    valid = torch.ones(len(frames), dtype=torch.bool)
    return AdaptedActions(condition, valid, indices, raw)


def adapt_causal_actions(
    *,
    actions: torch.Tensor,
    cache_frame_indices: torch.Tensor,
    frame_indices: torch.Tensor,
    config: dict[str, Any],
    stats: ActionStats,
) -> AdaptedActions:
    frames = torch.as_tensor(frame_indices, dtype=torch.long)
    stride = int(config["temporal"]["frame_stride"])
    representation = config["action"]["representation"]
    valid = torch.ones(len(frames), dtype=torch.bool)
    valid[0] = False

    if representation == "shifted_sampled":
        indices = torch.full_like(frames, -1)
        indices[1:] = frames[:-1]
        raw = torch.zeros((len(frames), stats.raw_dim), dtype=torch.float32)
        if len(frames) > 1:
            raw[1:] = _rows_for_raw_indices(
                actions, cache_frame_indices, indices[1:]
            )
        condition = torch.zeros_like(raw)
        condition[1:] = _normalize(raw[1:], stats)
        return AdaptedActions(condition, valid, indices, raw)

    if representation != "fast_chunk":
        raise ValueError(f"unsupported causal action representation {representation!r}")
    indices = torch.full((len(frames), stride), -1, dtype=torch.long)
    offsets = torch.arange(stride, dtype=torch.long)
    if len(frames) > 1:
        indices[1:] = frames[:-1, None] + offsets[None, :]
    raw6 = torch.zeros((len(frames), stride, stats.raw_dim), dtype=torch.float32)
    if len(frames) > 1:
        raw6[1:] = _rows_for_raw_indices(
            actions, cache_frame_indices, indices[1:]
        )
    raw = raw6.reshape(len(frames), -1)
    condition = torch.zeros_like(raw)
    if len(frames) > 1:
        condition[1:] = _normalize(raw6[1:], stats).reshape(len(frames) - 1, -1)
    return AdaptedActions(condition, valid, indices, raw)


def adapt_actions(
    *,
    checkpoint_type: str,
    actions: torch.Tensor,
    states: torch.Tensor | None,
    cache_frame_indices: torch.Tensor,
    frame_indices: torch.Tensor,
    config: dict[str, Any],
    stats: ActionStats,
) -> AdaptedActions:
    if checkpoint_type == "causal_v2":
        return adapt_causal_actions(
            actions=actions,
            cache_frame_indices=cache_frame_indices,
            frame_indices=frame_indices,
            config=config,
            stats=stats,
        )
    if checkpoint_type == "legacy_v1":
        return adapt_legacy_actions(
            actions=actions,
            states=states,
            cache_frame_indices=cache_frame_indices,
            frame_indices=frame_indices,
            config=config,
            stats=stats,
        )
    raise ValueError(f"unknown checkpoint type {checkpoint_type!r}")
