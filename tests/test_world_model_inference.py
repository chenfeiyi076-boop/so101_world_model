from __future__ import annotations

import copy

import pytest
import torch
import yaml

from src.causal.checkpointing import config_with_action_stats
from src.causal.config import resolve_config
from src.causal.data.common import ActionStats
from src.causal.runtime import build_model
from src.inference.action_adapter import adapt_causal_actions, adapt_legacy_actions
from src.inference.checkpoint_loader import LoadedWorldModel, load_world_model
from src.inference.flow_sampler import euler_sample_next, euler_tau_schedule
from src.inference.rollout import autoregressive_rollout
from src.models.dit import DiT


class ConstantVelocity(torch.nn.Module):
    def __init__(self, value: float = 1.0):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.value = value
        self.inputs = []

    def forward(self, latents, tau, actions, action_valid_mask=None):
        self.inputs.append(latents.detach().clone())
        return torch.full_like(latents, self.value) + self.anchor * 0


def _stats() -> ActionStats:
    return ActionStats(torch.arange(6).float(), torch.arange(1, 7).float())


def _causal_config(representation: str = "shifted_sampled", stride: int = 4):
    return resolve_config(
        {
            "experiment": {"name": "inference_test", "seed": 0},
            "data": {
                "mode": "single_episode",
                "episode_id": 0,
                "cache_path": "unused.pt",
                "split_ranges": {"train": [0, 20], "val": [20, 40]},
            },
            "temporal": {
                "num_frames": 10,
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


def _legacy_config(stride: int = 4, hidden_size: int = 8):
    return {
        "model": {
            "in_channels": 2,
            "patch_size": 1,
            "hidden_size": hidden_size,
            "depth": 1,
            "num_heads": 1,
            "action_dim": 6,
            "mlp_ratio": 2.0,
            "use_qk_norm": True,
        },
        "temporal": {"num_frames": 10, "num_history": 2, "frame_stride": stride},
        "action": {
            "alignment": "synchronized_legacy",
            "representation": "sampled",
            "control_representation": "absolute",
            "raw_dim": 6,
        },
        "latent": {"convention": "posterior_sample_times_scaling_no_shift"},
    }


def _legacy_chunk_config():
    config = _legacy_config()
    config["model"]["action_dim"] = 24
    config["action"]["alignment"] = "causal_chunk_legacy"
    config["action"]["representation"] = "chunk"
    return config


def _episode(length: int = 32, future_value: float | None = None):
    latents = torch.zeros(length, 2, 2, 2)
    if future_value is not None:
        latents[8:] = future_value
    return {
        "latents": latents,
        "actions": torch.arange(length * 6).reshape(length, 6).float(),
        "states": torch.zeros(length, 6),
        "frame_indices": torch.arange(length),
    }


def test_euler_schedule_decreases_from_one_to_zero():
    schedule = euler_tau_schedule(4, device=torch.device("cpu"), dtype=torch.float32)
    assert schedule.tolist() == [1.0, 0.75, 0.5, 0.25, 0.0]
    assert torch.all(schedule[1:] < schedule[:-1])


def test_constant_velocity_euler_has_correct_sign():
    model = ConstantVelocity(2.0)
    history = torch.zeros(1, 2, 1, 1, 1)
    actions = torch.zeros(1, 3, 6)
    mask = torch.ones(1, 3, dtype=torch.bool)
    generated, noise = euler_sample_next(
        model=model,
        checkpoint_type="legacy_v1",
        history_latents=history,
        action_cond=actions,
        action_valid_mask=mask,
        num_inference_steps=5,
        initial_noise=torch.full((1, 1, 1, 1, 1), 3.0),
    )
    assert torch.allclose(noise, torch.full_like(noise, 3.0))
    assert torch.allclose(generated, torch.full_like(generated, 1.0))
    assert all(torch.equal(item[:, :2], history) for item in model.inputs)


def test_legacy_stride4_adapter_stays_synchronized():
    actions = torch.arange(16 * 6).reshape(16, 6).float()
    result = adapt_legacy_actions(
        actions=actions,
        states=torch.zeros_like(actions),
        cache_frame_indices=torch.arange(16),
        frame_indices=torch.tensor([0, 4, 8]),
        config=_legacy_config(),
        stats=ActionStats(torch.zeros(6), torch.ones(6)),
    )
    assert result.action_indices.tolist() == [0, 4, 8]
    assert torch.equal(result.condition, actions[[0, 4, 8]])
    assert result.valid_mask.tolist() == [True, True, True]


def test_legacy_chunk_adapter_matches_episode_boundary_padding():
    actions = torch.arange(120 * 6).reshape(120, 6).float()
    result = adapt_legacy_actions(
        actions=actions,
        states=torch.zeros_like(actions),
        cache_frame_indices=torch.arange(120),
        frame_indices=torch.tensor([0, 4, 100]),
        config=_legacy_chunk_config(),
        stats=ActionStats(torch.zeros(6), torch.ones(6)),
    )
    assert result.action_indices.tolist() == [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [96, 97, 98, 99],
    ]


def test_causal_shifted_stride4_adapter():
    actions = torch.arange(16 * 6).reshape(16, 6).float()
    result = adapt_causal_actions(
        actions=actions,
        cache_frame_indices=torch.arange(16),
        frame_indices=torch.tensor([0, 4, 8]),
        config=_causal_config("shifted_sampled"),
        stats=ActionStats(torch.zeros(6), torch.ones(6)),
    )
    assert result.action_indices.tolist() == [-1, 0, 4]
    assert result.valid_mask.tolist() == [False, True, True]
    assert torch.equal(result.condition[0], torch.zeros(6))
    assert torch.equal(result.condition[1:], actions[[0, 4]])


def test_causal_fast_chunk_stride4_adapter_uses_shared_6d_stats():
    actions = torch.arange(16 * 6).reshape(16, 6).float()
    stats = _stats()
    result = adapt_causal_actions(
        actions=actions,
        cache_frame_indices=torch.arange(16),
        frame_indices=torch.tensor([0, 4, 8]),
        config=_causal_config("fast_chunk"),
        stats=stats,
    )
    assert result.action_indices.tolist() == [
        [-1, -1, -1, -1],
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]
    expected = ((actions[:4] - stats.mean) / stats.std).reshape(-1)
    assert torch.equal(result.condition[1], expected)


def test_causal_loader_uses_only_checkpoint_stats(tmp_path):
    config = _causal_config("shifted_sampled")
    stats = _stats()
    model = build_model(config)
    checkpoint = {
        "checkpoint_version": 2,
        "config": config_with_action_stats(config, stats),
        "action_stats": stats.to_dict(),
        "model_state_dict": model.state_dict(),
    }
    path = tmp_path / "causal.pt"
    torch.save(checkpoint, path)
    loaded = load_world_model(path)
    assert loaded.action_stats.source == stats.source
    assert torch.equal(loaded.action_stats.mean, stats.mean)
    assert torch.equal(loaded.action_stats.std, stats.std)


def test_legacy_loader_refuses_missing_stats(tmp_path):
    config = _legacy_config()
    model = DiT(**config["model"])
    checkpoint_path = tmp_path / "legacy.pt"
    config_path = tmp_path / "legacy.yaml"
    torch.save({"model": model.state_dict(), "config": {}}, checkpoint_path)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(RuntimeError, match="no action_mean/action_std"):
        load_world_model(checkpoint_path, legacy_config_path=config_path)


def _dummy_loaded(model, representation="shifted_sampled"):
    return LoadedWorldModel(
        model=model,
        checkpoint={"checkpoint_version": 2},
        checkpoint_type="causal_v2",
        resolved_config=_causal_config(representation),
        action_stats=ActionStats(torch.zeros(6), torch.ones(6)),
        checkpoint_path="dummy.pt",
    )


def test_same_seed_produces_same_rollout():
    episode = _episode()
    first = autoregressive_rollout(
        loaded=_dummy_loaded(ConstantVelocity(0.0)),
        episode_cache=episode,
        start_frame=0,
        rollout_frames=2,
        num_inference_steps=2,
        seed=123,
    )
    second = autoregressive_rollout(
        loaded=_dummy_loaded(ConstantVelocity(0.0)),
        episode_cache=episode,
        start_frame=0,
        rollout_frames=2,
        num_inference_steps=2,
        seed=123,
    )
    assert torch.equal(first["initial_noises"], second["initial_noises"])
    assert torch.equal(first["generated_latents"], second["generated_latents"])


def test_future_ground_truth_never_enters_model_and_generated_is_appended():
    model = ConstantVelocity(0.0)
    artifact = autoregressive_rollout(
        loaded=_dummy_loaded(model),
        episode_cache=_episode(future_value=999.0),
        start_frame=0,
        rollout_frames=2,
        num_inference_steps=1,
        seed=7,
    )
    assert torch.all(artifact["reference_latents"] == 999.0)
    assert all(not torch.any(item == 999.0) for item in model.inputs)
    first_generated = artifact["generated_latents"][0]
    assert torch.equal(model.inputs[1][0, 1], first_generated)
    assert artifact["metadata"]["reference_used_as_model_input"] is False


def test_legacy_checkpoint_config_mismatch_fails(tmp_path):
    config = _legacy_config(stride=4)
    model = DiT(**config["model"])
    checkpoint = {
        "model": model.state_dict(),
        "action_mean": torch.zeros(6),
        "action_std": torch.ones(6),
        "config": {"frame_skip": 2},
    }
    checkpoint_path = tmp_path / "legacy.pt"
    config_path = tmp_path / "legacy.yaml"
    torch.save(checkpoint, checkpoint_path)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="frame_stride"):
        load_world_model(checkpoint_path, legacy_config_path=config_path)


def test_strict_state_dict_loading_rejects_unexpected_key(tmp_path):
    config = _legacy_config()
    model = DiT(**config["model"])
    state = copy.deepcopy(model.state_dict())
    state["unexpected.weight"] = torch.ones(1)
    checkpoint = {
        "model": state,
        "action_mean": torch.zeros(6),
        "action_std": torch.ones(6),
        "config": {},
    }
    checkpoint_path = tmp_path / "legacy.pt"
    config_path = tmp_path / "legacy.yaml"
    torch.save(checkpoint, checkpoint_path)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_world_model(checkpoint_path, legacy_config_path=config_path)
