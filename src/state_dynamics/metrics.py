from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Sequence

import torch


def raw_error_values(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    prediction = torch.as_tensor(prediction, dtype=torch.float32)
    target = torch.as_tensor(target, dtype=torch.float32)
    if prediction.shape != target.shape or prediction.shape[-1] != 6:
        raise ValueError("prediction/target must have matching [...,6] shapes")
    error = prediction - target
    return {"absolute": error.abs(), "squared": error.square()}


def summarize_teacher_forced(
    predictions: torch.Tensor, targets: torch.Tensor
) -> tuple[dict[str, float], list[dict[str, float]], list[dict[str, float]]]:
    if predictions.ndim != 3 or predictions.shape[1:] != (4, 6):
        raise ValueError("teacher-forced predictions must be [N,4,6]")
    values = raw_error_values(predictions, targets)
    absolute, squared = values["absolute"], values["squared"]
    summary = {
        "overall_mae": float(absolute.mean()),
        "overall_rmse": math.sqrt(float(squared.mean())),
        "num_windows": int(len(predictions)),
    }
    horizons = [
        {
            "horizon": index + 1,
            "horizon_seconds": (index + 1) / 20.0,
            "mae": float(absolute[:, index].mean()),
            "rmse": math.sqrt(float(squared[:, index].mean())),
        }
        for index in range(4)
    ]
    joints = [
        {
            "joint": index,
            "mae": float(absolute[..., index].mean()),
            "rmse": math.sqrt(float(squared[..., index].mean())),
        }
        for index in range(6)
    ]
    return summary, horizons, joints


def summarize_rollout_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize no rollout rows")
    squared_elements = []
    absolute_elements = []
    baseline_squared = []
    baseline_absolute = []
    joint_abs: dict[int, list[float]] = defaultdict(list)
    joint_sq: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        absolute_elements.extend(float(value) for value in row["per_joint_absolute_error"])
        squared_elements.extend(float(value) for value in row["per_joint_squared_error"])
        baseline_absolute.extend(float(value) for value in row["baseline_per_joint_absolute_error"])
        baseline_squared.extend(float(value) for value in row["baseline_per_joint_squared_error"])
        for joint, (absolute, squared) in enumerate(
            zip(row["per_joint_absolute_error"], row["per_joint_squared_error"])
        ):
            joint_abs[joint].append(float(absolute)); joint_sq[joint].append(float(squared))
    endpoint_micro_mae = sum(absolute_elements) / len(absolute_elements)
    endpoint_micro_rmse = math.sqrt(sum(squared_elements) / len(squared_elements))
    command_copy_micro_mae = sum(baseline_absolute) / len(baseline_absolute)
    command_copy_micro_rmse = math.sqrt(sum(baseline_squared) / len(baseline_squared))
    return {
        "num_rollout_endpoints": len(rows),
        "autoregressive_endpoint_micro_mae": endpoint_micro_mae,
        "autoregressive_endpoint_micro_rmse": endpoint_micro_rmse,
        "command_copy_endpoint_micro_mae": command_copy_micro_mae,
        "command_copy_endpoint_micro_rmse": command_copy_micro_rmse,
        # Backward-compatible aliases for state-dynamics v1 outputs.
        "autoregressive_endpoint_mae": endpoint_micro_mae,
        "autoregressive_endpoint_rmse": endpoint_micro_rmse,
        "command_copy_endpoint_mae": command_copy_micro_mae,
        "command_copy_endpoint_rmse": command_copy_micro_rmse,
        "per_joint": [
            {
                "joint": joint,
                "mae": sum(joint_abs[joint]) / len(joint_abs[joint]),
                "rmse": math.sqrt(sum(joint_sq[joint]) / len(joint_sq[joint])),
            }
            for joint in sorted(joint_abs)
        ],
    }


def summarize_rollout_per_episode(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, float | int]], dict[str, float]]:
    if not rows:
        raise ValueError("cannot summarize no rollout rows")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["episode_id"])].append(row)
    episode_rows: list[dict[str, float | int]] = []
    for episode_id, values in sorted(grouped.items()):
        ordered = sorted(values, key=lambda row: int(row["rollout_step"]))
        expected_steps = list(range(1, len(ordered) + 1))
        actual_steps = [int(row["rollout_step"]) for row in ordered]
        if actual_steps != expected_steps:
            raise RuntimeError(
                f"episode {episode_id} rollout steps must be contiguous from 1"
            )
        episode_rows.append({
            "episode_id": episode_id,
            "num_rollout_steps": len(ordered),
            "mean_endpoint_mae": sum(float(row["mae"]) for row in ordered) / len(ordered),
            "mean_endpoint_rmse": sum(float(row["rmse"]) for row in ordered) / len(ordered),
            "final_endpoint_mae": float(ordered[-1]["mae"]),
            "final_endpoint_rmse": float(ordered[-1]["rmse"]),
        })
    count = len(episode_rows)
    macro = {
        "autoregressive_episode_macro_mean_endpoint_mae": sum(
            float(row["mean_endpoint_mae"]) for row in episode_rows
        ) / count,
        "autoregressive_episode_macro_mean_endpoint_rmse": sum(
            float(row["mean_endpoint_rmse"]) for row in episode_rows
        ) / count,
        "autoregressive_episode_macro_final_endpoint_mae": sum(
            float(row["final_endpoint_mae"]) for row in episode_rows
        ) / count,
        "autoregressive_episode_macro_final_endpoint_rmse": sum(
            float(row["final_endpoint_rmse"]) for row in episode_rows
        ) / count,
    }
    return episode_rows, macro
