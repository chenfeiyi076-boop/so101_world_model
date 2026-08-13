from __future__ import annotations

import json

import pytest
import torch

from src.causal.data.common import (
    ActionStats,
    build_action_condition,
    build_fast_chunk_action_indices,
    build_frame_indices,
    build_shifted_sampled_action_indices,
    effective_action_dim,
)
from src.causal.data.multi_episode_dataset import MultiEpisodeCausalDataset
from src.causal.model import CausalDiT


@pytest.mark.parametrize(
    ("stride", "frames", "actions"),
    [
        (1, [0, 1, 2, 3], [-1, 0, 1, 2]),
        (2, [0, 2, 4, 6], [-1, 0, 2, 4]),
        (4, [0, 4, 8, 12], [-1, 0, 4, 8]),
    ],
)
def test_shifted_sampled_indices(stride, frames, actions):
    frame_indices = build_frame_indices(0, 4, stride)
    assert frame_indices.tolist() == frames
    assert build_shifted_sampled_action_indices(frame_indices).tolist() == actions


@pytest.mark.parametrize(
    ("stride", "expected"),
    [
        (2, [[-1, -1], [0, 1], [2, 3], [4, 5]]),
        (
            4,
            [
                [-1, -1, -1, -1],
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
            ],
        ),
    ],
)
def test_fast_chunk_indices(stride, expected):
    frames = build_frame_indices(0, 4, stride)
    assert build_fast_chunk_action_indices(frames, stride).tolist() == expected


@pytest.mark.parametrize("stride", [1, 2, 4])
def test_effective_action_dimensions(stride):
    assert effective_action_dim(6, stride, "shifted_sampled") == 6
    assert effective_action_dim(6, stride, "fast_chunk") == 6 * stride


def test_first_slot_is_explicit_null_and_fast_chunk_uses_shared_6d_stats():
    actions = torch.arange(16 * 6, dtype=torch.float32).reshape(16, 6)
    stats = ActionStats(
        mean=torch.arange(6, dtype=torch.float32),
        std=torch.arange(1, 7, dtype=torch.float32),
    )
    frames = build_frame_indices(0, 4, 4)
    condition, mask, indices = build_action_condition(
        actions, frames, 4, "fast_chunk", stats, True
    )
    assert mask.tolist() == [False, True, True, True]
    assert torch.equal(condition[0], torch.zeros(24))
    expected = ((actions[0:4] - stats.mean) / stats.std).reshape(-1)
    assert torch.equal(condition[1], expected)
    assert indices[1].tolist() == [0, 1, 2, 3]


def test_dataset_refuses_implicit_statistics():
    actions = torch.zeros(16, 6)
    frames = build_frame_indices(0, 4, 1)
    with pytest.raises(ValueError, match="explicit fixed action_stats"):
        build_action_condition(
            actions, frames, 1, "shifted_sampled", None, True
        )


def test_null_embedding_is_zero_after_biased_linear():
    model = CausalDiT(
        in_channels=2,
        patch_size=1,
        hidden_size=8,
        depth=1,
        num_heads=1,
        action_dim=6,
    )
    with torch.no_grad():
        model.action_embedder.proj.bias.fill_(7.0)
    actions = torch.randn(1, 4, 6)
    mask = torch.tensor([[False, True, True, True]])
    embedding = model.embed_actions(actions, mask)
    assert torch.equal(embedding[:, 0], torch.zeros_like(embedding[:, 0]))
    assert not torch.equal(embedding[:, 1], torch.zeros_like(embedding[:, 1]))


def _write_cache(path, episode_id: int, length: int, offset: float = 0.0):
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


def test_multi_episode_windows_never_cross_episode(tmp_path):
    paths = []
    for episode_id, offset in ((0, 0.0), (1, 1000.0)):
        path = tmp_path / f"episode_{episode_id:03d}.pt"
        _write_cache(path, episode_id, 12, offset)
        paths.append(path)
    manifest = {
        "train_episode_ids": [0, 1],
        "val_episode_ids": [1],
        "episodes": [
            {"episode_index": i, "cache_file": str(path)}
            for i, path in enumerate(paths)
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    stats = ActionStats(torch.zeros(6), torch.ones(6))
    dataset = MultiEpisodeCausalDataset(
        manifest_path,
        split="train",
        num_frames=4,
        frame_stride=2,
        representation="fast_chunk",
        action_stats=stats,
    )
    for index in range(len(dataset)):
        sample = dataset[index]
        episode_id = int(sample["episode_idx"])
        expected = 0.0 if episode_id == 0 else 1000.0
        assert torch.all(sample["latents"] == expected)
