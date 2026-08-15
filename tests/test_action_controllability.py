from __future__ import annotations

import math

import pytest
import torch

from src.causal.action_counterfactual import (
    action_counterfactual_rollouts,
    aggregate_per_case,
    aggregate_per_rollout,
    bootstrap_mean_ci,
    deterministic_derangement,
    future_action_spans,
    paired_action_counterfactual_rollout_batch,
    paired_rollout_step_rows,
    summarize_true_rerun_warnings,
    temporally_shuffled_episode,
    validate_selected_case_split_membership,
)
from src.causal.config import resolve_config
from src.causal.data.common import ActionStats
from src.causal.rollout import (
    RolloutStream,
    autoregressive_causal_rollout,
    deterministic_rollout_noise,
)


class ActionInsensitiveModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, latents, tau, action_cond, action_valid_mask):
        return torch.zeros_like(latents) + self.anchor * 0


class ActionSensitiveModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, latents, tau, action_cond, action_valid_mask):
        signal = action_cond[:, -1].mean(dim=-1).reshape(-1, 1, 1, 1, 1)
        return torch.ones_like(latents) * signal + self.anchor * 0


def _config() -> dict:
    return resolve_config(
        {
            "experiment": {"name": "action_counterfactual_test", "seed": 0},
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
            "latent": {
                "convention": "posterior_sample_times_scaling_no_shift"
            },
        }
    )


def _episode(episode_id: int = 7, length: int = 64) -> dict:
    return {
        "episode_index": episode_id,
        "latents": torch.zeros(length, 1, 2, 2),
        "actions": torch.arange(length * 6, dtype=torch.float32).reshape(length, 6),
        "frame_indices": torch.arange(length),
    }


def _stats() -> ActionStats:
    return ActionStats(torch.zeros(6), torch.ones(6))


def _paired(
    *,
    model: torch.nn.Module | None = None,
    rollout_steps: int = 3,
    episode_id: int = 7,
    start: int = 0,
    noise_draw: int = 0,
):
    return paired_action_counterfactual_rollout_batch(
        model=ActionInsensitiveModel() if model is None else model,
        config=_config(),
        action_stats=_stats(),
        base_streams=[
            RolloutStream(
                episode=_episode(episode_id),
                episode_id=episode_id,
                start=start,
                noise_draw=noise_draw,
            )
        ],
        rollout_steps=rollout_steps,
        euler_steps=1,
        seed=20260,
        shuffle_seed=30360,
        device=torch.device("cpu"),
        precision="fp32",
    )[0]


def test_future_action_span_calculation():
    spans = future_action_spans(
        start=10, num_history=2, frame_stride=4, rollout_steps=3
    )
    assert (spans["history_action_start"], spans["history_action_end"]) == (10, 14)
    assert spans["future_chunk_spans"] == [(14, 18), (18, 22), (22, 26)]
    assert (spans["modified_raw_action_start"], spans["modified_raw_action_end"]) == (
        14,
        26,
    )


def test_history_action_chunk_is_unchanged():
    episode = _episode()
    shuffled, audit = temporally_shuffled_episode(
        episode,
        episode_id=7,
        start=10,
        num_history=2,
        frame_stride=4,
        rollout_steps=3,
        shuffle_seed=30360,
    )
    assert torch.equal(episode["actions"][10:14], shuffled["actions"][10:14])
    assert audit["history_action_unchanged"] is True


def test_shuffle_preserves_each_chunk_internal_order():
    episode = _episode()
    shuffled, audit = temporally_shuffled_episode(
        episode,
        episode_id=7,
        start=10,
        num_history=2,
        frame_stride=4,
        rollout_steps=3,
        shuffle_seed=30360,
    )
    original_chunks = episode["actions"][14:26].reshape(3, 4, 6)
    shuffled_chunks = shuffled["actions"][14:26].reshape(3, 4, 6)
    assert torch.equal(
        shuffled_chunks, original_chunks[torch.tensor(audit["permutation"])]
    )


def test_derangement_has_no_fixed_points_for_r32():
    permutation = deterministic_derangement(
        length=32, shuffle_seed=30360, episode_id=7, start=10
    )
    assert torch.all(permutation != torch.arange(32))
    assert sorted(permutation.tolist()) == list(range(32))


def test_permutation_is_deterministic_and_independent_of_noise_draw():
    first = deterministic_derangement(
        length=32, shuffle_seed=30360, episode_id=7, start=10
    )
    # noise_draw is intentionally absent from the permutation API.
    for _noise_draw in range(4):
        repeated = deterministic_derangement(
            length=32, shuffle_seed=30360, episode_id=7, start=10
        )
        assert torch.equal(first, repeated)


def test_original_episode_is_not_modified():
    episode = _episode()
    before = episode["actions"].clone()
    shuffled, _ = temporally_shuffled_episode(
        episode,
        episode_id=7,
        start=0,
        num_history=2,
        frame_stride=4,
        rollout_steps=3,
        shuffle_seed=30360,
    )
    assert torch.equal(episode["actions"], before)
    assert shuffled["actions"].data_ptr() != episode["actions"].data_ptr()
    assert shuffled["latents"] is episode["latents"]


def test_true_and_shuffle_share_exact_gaussian_noise():
    pair = _paired()
    assert torch.equal(pair["true"]["initial_noises"], pair["shuffle"]["initial_noises"])


def test_true_branch_matches_direct_autoregressive_rollout():
    episode = _episode()
    noises = deterministic_rollout_noise(
        latent_shape=(1, 2, 2),
        rollout_steps=3,
        seed=20260,
        episode_id=7,
        start=0,
        noise_draw=1,
    )
    direct = autoregressive_causal_rollout(
        model=ActionInsensitiveModel(),
        config=_config(),
        action_stats=_stats(),
        episode=episode,
        episode_id=7,
        start=0,
        noise_draw=1,
        rollout_steps=3,
        euler_steps=1,
        seed=20260,
        device=torch.device("cpu"),
        precision="fp32",
        initial_noises=noises,
    )
    pair = paired_action_counterfactual_rollout_batch(
        model=ActionInsensitiveModel(),
        config=_config(),
        action_stats=_stats(),
        base_streams=[RolloutStream(episode, 7, 0, 1, noises)],
        rollout_steps=3,
        euler_steps=1,
        seed=20260,
        shuffle_seed=30360,
        device=torch.device("cpu"),
        precision="fp32",
    )[0]
    true = pair["true"]
    for key in ("target_frame_indices", "initial_noises", "predicted_future"):
        assert torch.equal(true[key], direct[key])
    for metric in direct["metrics"]:
        assert torch.equal(true["metrics"][metric], direct["metrics"][metric])


def test_sliding_after_step_eight_uses_shuffled_action_timeline():
    pair = _paired(rollout_steps=9)
    permutation = pair["audit"]["permutation"]
    original_chunks = _episode()["actions"][4:40].reshape(9, 4, 6)
    expected_step9 = original_chunks[permutation[8]].reshape(-1)
    actual_step9 = pair["shuffle"]["action_conditions"][8][-1]
    assert torch.equal(actual_step9, expected_step9)
    assert pair["shuffle"]["model_input_lengths"][8] == 10


def test_action_insensitive_model_has_zero_divergence_and_delta():
    rows = paired_rollout_step_rows([_paired(model=ActionInsensitiveModel())])
    assert all(row["prediction_divergence_mse"] == 0 for row in rows)
    assert all(row["delta_mse"] == 0 for row in rows)


def test_action_sensitive_model_produces_prediction_divergence():
    rows = paired_rollout_step_rows([_paired(model=ActionSensitiveModel())])
    assert any(row["prediction_divergence_mse"] > 0 for row in rows)


def test_delta_sign_and_true_better_are_not_reversed():
    pair = _paired()
    pair["true"]["metrics"]["mse"] = torch.ones(3)
    pair["shuffle"]["metrics"]["mse"] = torch.full((3,), 2.0)
    rows = paired_rollout_step_rows([pair])
    assert all(row["delta_mse"] == 1.0 for row in rows)
    assert all(row["true_better"] is True for row in rows)


def test_case_aggregation_averages_four_draws_before_case_comparison():
    rollout_rows = []
    for episode_id in (1, 2):
        for draw in range(4):
            true = float(episode_id + draw)
            shuffle = true + (1.0 if episode_id == 1 else -1.0)
            rollout_rows.append(
                {
                    "episode_id": episode_id,
                    "start": 0,
                    "noise_draw": draw,
                    "mean_true_mse": true,
                    "mean_shuffle_mse": shuffle,
                    "mean_delta_mse": shuffle - true,
                    "mean_prediction_divergence_mse": 0.5,
                }
            )
    cases = aggregate_per_case(rollout_rows)
    assert len(cases) == 2
    assert cases[0]["mean_delta_mse"] == pytest.approx(1.0)
    assert cases[0]["true_better_draw_rate"] == pytest.approx(1.0)
    assert cases[1]["mean_delta_mse"] == pytest.approx(-1.0)
    assert cases[1]["true_better_draw_rate"] == pytest.approx(0.0)


def test_bootstrap_ci_is_deterministic_for_same_seed():
    values = [0.1, 0.2, 0.4, 0.8]
    first = bootstrap_mean_ci(values, samples=200, seed=4242)
    second = bootstrap_mean_ci(values, samples=200, seed=4242)
    assert first == second
    assert first[0] == pytest.approx(sum(values) / len(values))


def test_partial_pair_batch_processes_two_two_one():
    base_streams = [
        RolloutStream(_episode(10 + index), 10 + index, 0, 0)
        for index in range(5)
    ]
    results = action_counterfactual_rollouts(
        model=ActionInsensitiveModel(),
        config=_config(),
        action_stats=_stats(),
        base_streams=base_streams,
        pair_batch_size=2,
        rollout_steps=2,
        euler_steps=1,
        seed=20260,
        shuffle_seed=30360,
        device=torch.device("cpu"),
        precision="fp32",
    )
    assert len(results) == 5
    assert [result["true"]["episode_id"] for result in results] == [10, 11, 12, 13, 14]


def test_per_rollout_aggregation_preserves_positive_delta():
    pair = _paired()
    pair["true"]["metrics"]["mse"] = torch.ones(3)
    pair["shuffle"]["metrics"]["mse"] = torch.full((3,), 2.0)
    rollouts = aggregate_per_rollout(paired_rollout_step_rows([pair]))
    assert len(rollouts) == 1
    assert math.isclose(rollouts[0]["mean_delta_mse"], 1.0)


def test_source_case_outside_declared_split_fails():
    selected_cases = [
        {"episode_id": 6, "start": 0},
        {"episode_id": 875, "start": 237},
    ]
    with pytest.raises(RuntimeError, match="outside declared 'val' split"):
        validate_selected_case_split_membership(
            selected_cases,
            split="val",
            split_episode_ids=(6, 7, 8),
        )


def test_rerun_warning_counts_distinguish_metrics_from_streams():
    metric_count, stream_count = summarize_true_rerun_warnings(
        (
            ("step 1 mse", "step 1 relative_l2", "step 2 mse"),
            (),
            ("step 8 cosine_similarity",),
        )
    )
    assert metric_count == 4
    assert stream_count == 2
