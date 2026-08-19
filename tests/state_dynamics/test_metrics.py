from __future__ import annotations

import pytest

from src.state_dynamics.metrics import (
    summarize_rollout_per_episode,
    summarize_rollout_rows,
)


def _row(episode_id: int, step: int, error: float) -> dict[str, object]:
    absolute = [error] * 6
    squared = [error * error] * 6
    return {
        "episode_id": episode_id,
        "rollout_step": step,
        "mae": error,
        "rmse": error,
        "per_joint_absolute_error": absolute,
        "per_joint_squared_error": squared,
        "baseline_per_joint_absolute_error": absolute,
        "baseline_per_joint_squared_error": squared,
    }


def test_rollout_episode_aggregation_and_micro_macro_names_are_unambiguous():
    rows = [_row(0, 1, 1.0), _row(0, 2, 3.0), _row(1, 1, 10.0)]
    micro = summarize_rollout_rows(rows)
    per_episode, macro = summarize_rollout_per_episode(rows)
    assert per_episode == [
        {
            "episode_id": 0,
            "num_rollout_steps": 2,
            "mean_endpoint_mae": 2.0,
            "mean_endpoint_rmse": 2.0,
            "final_endpoint_mae": 3.0,
            "final_endpoint_rmse": 3.0,
        },
        {
            "episode_id": 1,
            "num_rollout_steps": 1,
            "mean_endpoint_mae": 10.0,
            "mean_endpoint_rmse": 10.0,
            "final_endpoint_mae": 10.0,
            "final_endpoint_rmse": 10.0,
        },
    ]
    assert micro["autoregressive_endpoint_micro_mae"] == pytest.approx(14 / 3)
    assert micro["autoregressive_endpoint_micro_rmse"] == pytest.approx(
        (110 / 3) ** 0.5
    )
    assert macro["autoregressive_episode_macro_mean_endpoint_mae"] == 6.0
    assert macro["autoregressive_episode_macro_mean_endpoint_rmse"] == 6.0
    assert macro["autoregressive_episode_macro_final_endpoint_mae"] == 6.5
    assert macro["autoregressive_episode_macro_final_endpoint_rmse"] == 6.5
