from __future__ import annotations

import copy

import pytest
import torch

from src.causal.checkpointing import (
    load_checkpoint,
    make_checkpoint,
    save_checkpoint,
)
from src.causal.config import resolve_config, validate_config_compatibility
from src.causal.data.common import ActionStats
from src.causal.runtime import build_model


def config(stride=4, representation="fast_chunk"):
    return resolve_config(
        {
            "experiment": {"name": "checkpoint_test", "seed": 0},
            "data": {
                "mode": "single_episode",
                "episode_id": 0,
                "cache_path": "unused.pt",
                "split_ranges": {"train": [0, 20], "val": [20, 40]},
            },
            "temporal": {
                "num_frames": 4,
                "num_history": 2,
                "frame_stride": stride,
            },
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
                "use_qk_norm": True,
            },
            "flow_matching": {"future_only_loss": True},
            "train": {"batch_size": 1, "steps": 1, "lr": 1e-4},
            "checkpoint": {"output_dir": "unused"},
            "latent": {"convention": "posterior_sample_times_scaling_no_shift"},
        }
    )


def test_checkpoint_round_trip_preserves_resolved_config_and_stats(tmp_path):
    resolved = config()
    model = build_model(resolved)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    stats = ActionStats(torch.arange(6), torch.arange(1, 7))
    checkpoint = make_checkpoint(
        model=model,
        optimizer=optimizer,
        step=9,
        config=resolved,
        action_stats=stats,
        data_info={"mode": "single_episode", "train_episode_ids": [0]},
        best_val_loss=0.5,
    )
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, checkpoint)
    loaded = load_checkpoint(path)
    assert loaded["checkpoint_version"] == 2
    assert loaded["step"] == 9
    assert loaded["config"]["temporal"]["frame_stride"] == 4
    assert loaded["config"]["action"]["raw_action_dim"] == 6
    assert loaded["config"]["action"]["effective_action_dim"] == 24
    assert loaded["config"]["action"]["action_mean"] == stats.mean.tolist()
    assert loaded["config"]["action"]["action_std"] == stats.std.tolist()
    assert torch.equal(loaded["action_stats"]["mean"], stats.mean)
    assert torch.equal(loaded["action_stats"]["std"], stats.std)


def test_config_mismatch_raises():
    checkpoint_config = config(stride=4, representation="fast_chunk")
    requested = config(stride=2, representation="fast_chunk")
    with pytest.raises(ValueError, match="frame_stride"):
        validate_config_compatibility(checkpoint_config, requested)


def test_representation_mismatch_raises():
    checkpoint_config = config(stride=4, representation="fast_chunk")
    requested = config(stride=4, representation="shifted_sampled")
    with pytest.raises(ValueError, match="representation"):
        validate_config_compatibility(checkpoint_config, requested)
