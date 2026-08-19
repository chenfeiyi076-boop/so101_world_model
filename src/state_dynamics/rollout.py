from __future__ import annotations

from typing import Any

import torch

from .normalization import StateActionStats


@torch.inference_mode()
def autoregressive_state_predictions(
    *,
    model: torch.nn.Module,
    initial_state: torch.Tensor,
    actions: torch.Tensor,
    stats: StateActionStats,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Roll out from state row 0. This function never accepts future GT states."""
    current = torch.as_tensor(initial_state, dtype=torch.float32)
    actions = torch.as_tensor(actions, dtype=torch.float32)
    if current.shape != (6,) or actions.ndim != 2 or actions.shape[1] != 6:
        raise ValueError("initial_state/actions must be [6] and [N,6]")
    steps = max(0, (len(actions) - 1) // 4)
    if steps == 0:
        raise ValueError("episode has no complete 4-action rollout block")
    chunks, endpoints, input_states, action_offsets = [], [], [], []
    model.eval()
    for step in range(steps):
        offset = step * 4
        action_chunk = actions[offset : offset + 4]
        input_states.append(current.clone())
        normalized_state = stats.normalize_states(current).unsqueeze(0).to(device)
        normalized_actions = stats.normalize_actions(action_chunk).unsqueeze(0).to(device)
        predicted_normalized = model(normalized_state, normalized_actions)[0]
        predicted = stats.denormalize_states(predicted_normalized).cpu()
        chunks.append(predicted)
        current = predicted[-1].clone()
        endpoints.append(current)
        action_offsets.append(offset)
    return {
        "predicted_chunks": torch.stack(chunks),
        "predicted_endpoints": torch.stack(endpoints),
        "model_input_states": torch.stack(input_states),
        "action_offsets": torch.tensor(action_offsets, dtype=torch.long),
        "target_state_indices": torch.arange(4, 4 * steps + 1, 4, dtype=torch.long),
        "reference_used_as_model_input": False,
    }


def rollout_metric_rows(
    *, episode_id: int, prediction: dict[str, torch.Tensor],
    states: torch.Tensor, actions: torch.Tensor,
) -> list[dict[str, Any]]:
    rows = []
    for index, target_index in enumerate(prediction["target_state_indices"].tolist()):
        predicted = prediction["predicted_endpoints"][index]
        target = states[target_index]
        baseline = actions[target_index - 1]
        error, baseline_error = predicted - target, baseline - target
        rows.append({
            "episode_id": int(episode_id), "rollout_step": index + 1,
            "horizon_seconds": (index + 1) * 0.2,
            "target_frame": int(target_index),
            "mae": float(error.abs().mean()),
            "rmse": float(error.square().mean().sqrt()),
            "command_copy_endpoint_mae": float(baseline_error.abs().mean()),
            "command_copy_endpoint_rmse": float(baseline_error.square().mean().sqrt()),
            "per_joint_absolute_error": error.abs().tolist(),
            "per_joint_squared_error": error.square().tolist(),
            "baseline_per_joint_absolute_error": baseline_error.abs().tolist(),
            "baseline_per_joint_squared_error": baseline_error.square().tolist(),
        })
    return rows
