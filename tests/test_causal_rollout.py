from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from scripts.evaluate_causal import select_evaluation_state_dict
from src.causal.config import resolve_config
from src.causal.data.common import ActionStats, build_action_condition
from src.causal.rollout import (
    aggregate_rollout_metrics,
    autoregressive_causal_rollout,
    build_rollout_catalog,
    deterministic_rollout_noise,
    latent_error_metrics,
    rollout_metric_rows,
    select_rollout_cases,
    threshold_horizon,
)
from src.inference.action_adapter import adapt_causal_actions


class RecordingZeroVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.inputs = []
        self.actions = []
        self.masks = []

    def forward(self, latents, tau, action_cond, action_valid_mask):
        self.inputs.append(latents.detach().cpu().clone())
        self.actions.append(action_cond.detach().cpu().clone())
        self.masks.append(action_valid_mask.detach().cpu().clone())
        return torch.zeros_like(latents) + self.anchor * 0


def _config() -> dict:
    return resolve_config(
        {
            "experiment": {"name": "rollout_test", "seed": 0},
            "data": {"mode": "multi_episode", "manifest_path": "unused.json"},
            "temporal": {
                "num_frames": 10,
                "num_history": 2,
                "frame_stride": 4,
            },
            "action": {
                "raw_action_dim": 6,
                "alignment": "causal",
                "representation": "fast_chunk",
                "normalize": True,
                "null_condition": "zero_embedding",
            },
            "model": {
                "in_channels": 1,
                "patch_size": 1,
                "hidden_size": 8,
                "depth": 1,
                "num_heads": 1,
            },
            "flow_matching": {"future_only_loss": True},
            "train": {
                "batch_size": 1,
                "steps": 1,
                "lr": 1e-4,
                "precision": "fp32",
            },
            "checkpoint": {"output_dir": "unused"},
            "latent": {"convention": "posterior_sample_times_scaling_no_shift"},
        }
    )


def _episode(length: int = 52, future_value: float = 999.0) -> dict:
    latents = torch.zeros(length, 1, 2, 2)
    latents[8:] = future_value
    return {
        "episode_index": 7,
        "latents": latents,
        "actions": torch.arange(length * 6, dtype=torch.float32).reshape(length, 6),
        "frame_indices": torch.arange(length),
    }


def _rollout(rollout_steps: int = 9):
    model = RecordingZeroVelocity()
    result = autoregressive_causal_rollout(
        model=model,
        config=_config(),
        action_stats=ActionStats(torch.zeros(6), torch.ones(6)),
        episode=_episode(),
        episode_id=7,
        start=0,
        rollout_steps=rollout_steps,
        euler_steps=1,
        seed=123,
        noise_draw=0,
        device=torch.device("cpu"),
        precision="fp32",
    )
    return model, result


def test_rollout_never_feeds_gt_future_and_supports_more_than_eight_steps():
    model, result = _rollout(9)
    assert result["gt_future"].shape[0] == 9
    assert torch.all(result["gt_future"] == 999.0)
    assert all(not torch.any(model_input == 999.0) for model_input in model.inputs)
    assert torch.equal(model.inputs[1][0, 2], result["predicted_future"][0])
    assert result["reference_used_as_model_input"] is False


def test_stride4_action_alignment_for_early_and_sliding_predictions():
    _, result = _rollout(9)
    expected = {
        1: [4, 5, 6, 7],
        2: [8, 9, 10, 11],
        8: [32, 33, 34, 35],
        9: [36, 37, 38, 39],
    }
    for step, indices in expected.items():
        assert result["action_indices"][step - 1][-1].tolist() == indices


@pytest.mark.parametrize(
    "cache_row_indices",
    (
        torch.arange(0, 40, 4),
        torch.arange(4, 44, 4),
    ),
    ids=("ordinary", "sliding"),
)
def test_training_and_rollout_fast_chunk_action_conditions_are_equivalent(
    cache_row_indices: torch.Tensor,
):
    length = 48
    actions = torch.arange(length * 6, dtype=torch.float32).reshape(length, 6)
    stats = ActionStats(
        mean=torch.tensor([-2.0, -1.0, 0.5, 1.5, 3.0, 5.0]),
        std=torch.tensor([0.5, 1.5, 2.0, 2.5, 4.0, 8.0]),
    )
    # Training's canonical builder receives cache-row indices. Rollout receives
    # raw frame indices and resolves them back through cache_frame_indices.
    cache_frame_indices = torch.arange(100, 100 + length, dtype=torch.long)
    raw_frame_indices = cache_frame_indices[cache_row_indices]

    adapted = adapt_causal_actions(
        actions=actions,
        cache_frame_indices=cache_frame_indices,
        frame_indices=raw_frame_indices,
        config=_config(),
        stats=stats,
    )
    condition, valid_mask, cache_action_indices = build_action_condition(
        actions,
        cache_row_indices,
        frame_stride=4,
        representation="fast_chunk",
        stats=stats,
        normalize=True,
    )

    canonical_raw_action_indices = torch.full_like(cache_action_indices, -1)
    non_null = cache_action_indices >= 0
    canonical_raw_action_indices[non_null] = cache_frame_indices[
        cache_action_indices[non_null]
    ]
    assert torch.equal(adapted.condition, condition)
    assert torch.equal(adapted.valid_mask, valid_mask)
    assert torch.equal(adapted.action_indices, canonical_raw_action_indices)
    assert torch.equal(adapted.condition[0], torch.zeros(24))
    assert adapted.valid_mask[0].item() is False


def test_sliding_window_lengths_and_first_slot_null_after_slide():
    _, result = _rollout(11)
    assert result["model_input_lengths"] == [3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
    ninth_frames = result["window_frame_indices"][8]
    assert ninth_frames.tolist() == [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
    for condition, mask in zip(
        result["action_conditions"], result["action_valid_masks"]
    ):
        assert mask[0].item() is False
        assert torch.equal(condition[0], torch.zeros_like(condition[0]))


def test_deterministic_noise_repeats_and_longer_rollout_preserves_prefix():
    arguments = {
        "latent_shape": (1, 2, 2),
        "seed": 20260,
        "episode_id": 9,
        "start": 17,
        "noise_draw": 2,
    }
    first = deterministic_rollout_noise(rollout_steps=8, **arguments)
    second = deterministic_rollout_noise(rollout_steps=8, **arguments)
    longer = deterministic_rollout_noise(rollout_steps=16, **arguments)
    assert torch.equal(first, second)
    assert torch.equal(first, longer[:8])


def test_latent_metric_and_aggregation_correctness():
    predicted = torch.tensor([[[[3.0, 4.0]]]])
    target = torch.tensor([[[[0.0, 4.0]]]])
    metrics = latent_error_metrics(predicted, target)
    assert metrics["mse"].item() == pytest.approx(4.5)
    assert metrics["rmse"].item() == pytest.approx(math.sqrt(4.5))
    assert metrics["mae"].item() == pytest.approx(1.5)
    assert metrics["relative_l2"].item() == pytest.approx(0.75)
    assert metrics["cosine_similarity"].item() == pytest.approx(0.8)

    result = {
        "episode_id": 1,
        "start": 0,
        "noise_draw": 0,
        "target_frame_indices": torch.tensor([8]),
        "metrics": metrics,
    }
    rows = rollout_metric_rows(result)
    aggregate = aggregate_rollout_metrics(rows, rollout_steps=1, frame_stride=4)
    assert aggregate[0]["raw_frame_offset"] == 4
    assert aggregate[0]["mse_mean"] == pytest.approx(4.5)
    assert aggregate[0]["mse_std"] == pytest.approx(0.0)


def test_threshold_stops_at_first_crossing_even_if_error_recovers():
    per_step = [
        {"step": step, "mse_mean": value}
        for step, value in enumerate([0.1, 0.2, 0.3, 0.15], start=1)
    ]
    assert threshold_horizon(per_step, threshold=0.25, metric="mse") == (2, 3)
    assert threshold_horizon(per_step, threshold=0.05, metric="mse") == (0, 1)
    assert threshold_horizon(per_step, threshold=1.0, metric="mse") == (4, None)


def test_checkpoint_weight_selection_prefers_ema_but_honors_raw():
    raw = {"weight": torch.tensor(1.0)}
    ema = {"weight": torch.tensor(2.0)}
    checkpoint = {"model_state_dict": raw, "ema_model_state_dict": ema}
    assert select_evaluation_state_dict(checkpoint, "auto") == (ema, "ema")
    assert select_evaluation_state_dict(checkpoint, "raw") == (raw, "raw")


def _write_cache(path: Path, episode_id: int, length: int) -> None:
    torch.save(
        {
            "episode_index": episode_id,
            "latents": torch.zeros(length, 1, 2, 2),
            "actions": torch.zeros(length, 6),
            "frame_indices": torch.arange(length),
        },
        path,
    )


def test_rollout_catalog_cases_stay_inside_their_episode(tmp_path: Path):
    entries = []
    for episode_id in range(4):
        path = tmp_path / f"episode_{episode_id}.pt"
        _write_cache(path, episode_id, 20)
        entries.append({"episode_index": episode_id, "cache_file": str(path)})
    manifest = {
        "train_episode_ids": [0],
        "val_episode_ids": [1],
        "test_episode_ids": [2, 3],
        "episodes": entries,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    catalog = build_rollout_catalog(
        manifest_path,
        split="test",
        num_history=2,
        frame_stride=4,
        rollout_steps=2,
        raw_action_dim=6,
    )
    cases = select_rollout_cases(catalog, max_rollouts=0)
    assert {case.episode_id for case in cases} == {2, 3}
    assert len(cases) == 16
    assert all(case.start + (2 + 2 - 1) * 4 < 20 for case in cases)
