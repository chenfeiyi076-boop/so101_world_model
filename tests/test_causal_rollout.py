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
    RolloutStream,
    aggregate_rollout_metrics,
    autoregressive_causal_rollout,
    autoregressive_causal_rollout_batch,
    autoregressive_causal_rollouts,
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


def _constant_episode(
    episode_id: int,
    latent_value: float,
    *,
    action_offset: float = 0.0,
    length: int = 64,
) -> dict:
    return {
        "episode_index": episode_id,
        "latents": torch.full((length, 1, 2, 2), latent_value),
        "actions": (
            torch.arange(length * 6, dtype=torch.float32).reshape(length, 6)
            + action_offset
        ),
        "frame_indices": torch.arange(length),
    }


def _batch_streams(count: int = 4) -> list[RolloutStream]:
    streams = []
    for index in range(count):
        episode_id = 10 + index
        streams.append(
            RolloutStream(
                episode=_constant_episode(
                    episode_id,
                    latent_value=float((index + 1) * 10),
                    action_offset=float(index * 1000),
                ),
                episode_id=episode_id,
                start=index,
                noise_draw=0,
            )
        )
    return streams


def _run_streams(
    streams: list[RolloutStream],
    *,
    batch_size: int,
    rollout_steps: int = 3,
    model: torch.nn.Module | None = None,
):
    model = RecordingZeroVelocity() if model is None else model
    results = autoregressive_causal_rollouts(
        model=model,
        config=_config(),
        action_stats=ActionStats(torch.zeros(6), torch.ones(6)),
        streams=streams,
        batch_size=batch_size,
        rollout_steps=rollout_steps,
        euler_steps=1,
        seed=321,
        device=torch.device("cpu"),
        precision="fp32",
    )
    return model, results


def _assert_rollout_results_equal(left: dict, right: dict) -> None:
    for key in ("episode_id", "start", "noise_draw", "model_input_lengths"):
        assert left[key] == right[key]
    for key in (
        "target_frame_indices",
        "initial_noises",
        "predicted_future",
    ):
        assert torch.equal(left[key], right[key])
    for key in ("action_indices", "action_valid_masks"):
        assert len(left[key]) == len(right[key])
        assert all(torch.equal(a, b) for a, b in zip(left[key], right[key]))
    assert left["metrics"].keys() == right["metrics"].keys()
    for key in left["metrics"]:
        assert torch.equal(left["metrics"][key], right["metrics"][key])


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


def test_batch1_and_batch4_rollouts_are_exactly_equivalent():
    streams = _batch_streams(4)
    _, reference = _run_streams(streams, batch_size=1)
    _, batched = _run_streams(streams, batch_size=4)
    assert len(reference) == len(batched) == 4
    for single, batch in zip(reference, batched):
        _assert_rollout_results_equal(single, batch)


def test_batch_feedback_never_crosses_streams():
    streams = [
        RolloutStream(
            episode=_constant_episode(index, value),
            episode_id=index,
            start=0,
            noise_draw=0,
        )
        for index, value in enumerate((10.0, 100.0, 1000.0), start=1)
    ]
    model = RecordingZeroVelocity()
    _, results = _run_streams(
        streams, batch_size=3, rollout_steps=2, model=model
    )
    assert len(model.inputs) == 2
    for batch_index, expected_history in enumerate((10.0, 100.0, 1000.0)):
        assert torch.all(model.inputs[0][batch_index, :2] == expected_history)
        own_prediction = results[batch_index]["predicted_future"][0]
        assert torch.equal(model.inputs[1][batch_index, 2], own_prediction)
        for other_index, other in enumerate(results):
            if other_index != batch_index:
                assert not torch.equal(
                    model.inputs[1][batch_index, 2], other["predicted_future"][0]
                )


def test_batch_sliding_uses_shared_temporal_length_and_null_first_slot():
    streams = _batch_streams(2)
    model = RecordingZeroVelocity()
    _, results = _run_streams(
        streams, batch_size=2, rollout_steps=11, model=model
    )
    expected_lengths = [3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
    assert [input_.shape[1] for input_ in model.inputs] == expected_lengths
    for result in results:
        assert result["model_input_lengths"] == expected_lengths
        for condition, mask in zip(
            result["action_conditions"], result["action_valid_masks"]
        ):
            assert mask[0].item() is False
            assert torch.equal(condition[0], torch.zeros_like(condition[0]))
    assert torch.equal(model.masks[8][:, 0], torch.zeros(2, dtype=torch.bool))
    assert torch.equal(model.actions[8][:, 0], torch.zeros(2, 24))


def test_batch_action_alignment_is_independent_for_each_episode_and_start():
    streams = _batch_streams(2)
    _, results = _run_streams(streams, batch_size=2, rollout_steps=2)
    for stream, result in zip(streams, results):
        expected_step1 = list(range(stream.start + 4, stream.start + 8))
        expected_step2 = list(range(stream.start + 8, stream.start + 12))
        assert result["action_indices"][0][-1].tolist() == expected_step1
        assert result["action_indices"][1][-1].tolist() == expected_step2
        episode_actions = stream.episode["actions"]
        assert torch.equal(
            result["action_conditions"][0][-1],
            episode_actions[expected_step1].reshape(-1),
        )
    assert not torch.equal(
        results[0]["action_conditions"][0][-1],
        results[1]["action_conditions"][0][-1],
    )


def test_noise_is_independent_of_requested_batch_size():
    streams = _batch_streams(4)
    outputs = {
        batch_size: _run_streams(streams, batch_size=batch_size)[1]
        for batch_size in (1, 2, 4)
    }
    for index in range(len(streams)):
        reference = outputs[1][index]["initial_noises"]
        assert torch.equal(reference, outputs[2][index]["initial_noises"])
        assert torch.equal(reference, outputs[4][index]["initial_noises"])


def test_partial_final_batch_returns_every_stream_in_stable_order():
    streams = _batch_streams(5)
    _, results = _run_streams(streams, batch_size=2)
    assert len(results) == 5
    assert [
        (result["episode_id"], result["start"], result["noise_draw"])
        for result in results
    ] == [(stream.episode_id, stream.start, stream.noise_draw) for stream in streams]


def test_noise_draw_streams_batch_without_identity_or_noise_mixing():
    cases = _batch_streams(2)
    streams = [
        RolloutStream(
            episode=case.episode,
            episode_id=case.episode_id,
            start=case.start,
            noise_draw=noise_draw,
        )
        for case in cases
        for noise_draw in range(2)
    ]
    _, results = _run_streams(streams, batch_size=4)
    assert [
        (result["episode_id"], result["start"], result["noise_draw"])
        for result in results
    ] == [
        (cases[0].episode_id, cases[0].start, 0),
        (cases[0].episode_id, cases[0].start, 1),
        (cases[1].episode_id, cases[1].start, 0),
        (cases[1].episode_id, cases[1].start, 1),
    ]
    assert not torch.equal(results[0]["initial_noises"], results[1]["initial_noises"])
    assert not torch.equal(results[2]["initial_noises"], results[3]["initial_noises"])


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
        {"step": step, "mse_mean": value, "relative_l2_mean": value}
        for step, value in enumerate([0.1, 0.2, 0.3, 0.15], start=1)
    ]
    assert threshold_horizon(per_step, threshold=0.25, metric="mse") == (2, 3)
    assert threshold_horizon(
        per_step, threshold=0.25, metric="relative_l2"
    ) == (2, 3)
    assert threshold_horizon(per_step, threshold=0.05, metric="mse") == (0, 1)
    assert threshold_horizon(per_step, threshold=1.0, metric="mse") == (4, None)


def test_mse_p90_threshold_stops_at_first_crossing():
    per_step = [
        {"step": step, "mse_p90": value}
        for step, value in enumerate([0.08, 0.09, 0.11, 0.07], start=1)
    ]
    assert threshold_horizon(
        per_step, threshold=0.10, metric="mse_p90"
    ) == (2, 3)


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
