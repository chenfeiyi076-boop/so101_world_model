from __future__ import annotations

import torch

from src.state_dynamics.normalization import StateActionStats
from src.state_dynamics.rollout import (
    autoregressive_state_predictions, rollout_metric_rows,
)


class RecordingEndpointModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.inputs = []

    def forward(self, current_state, actions):
        self.inputs.append(current_state.detach().cpu().clone())
        increments = torch.arange(1, 5).view(1, 4, 1).float()
        return current_state[:, None, :] + increments


def unit_stats():
    return StateActionStats(
        torch.zeros(6), torch.ones(6), torch.zeros(6), torch.ones(6)
    )


def test_autoregressive_feedback_uses_previous_predicted_endpoint():
    model = RecordingEndpointModel()
    result = autoregressive_state_predictions(
        model=model, initial_state=torch.zeros(6), actions=torch.zeros(13, 6),
        stats=unit_stats(), device=torch.device("cpu"),
    )
    assert len(model.inputs) == 3
    assert torch.equal(model.inputs[0][0], torch.zeros(6))
    assert torch.equal(model.inputs[1][0], torch.full((6,), 4.0))
    assert torch.equal(model.inputs[2][0], torch.full((6,), 8.0))
    assert result["reference_used_as_model_input"] is False


def test_future_gt_sentinel_cannot_enter_model_and_baseline_alignment_is_t_plus_4():
    model = RecordingEndpointModel()
    actions = torch.arange(13 * 6).reshape(13, 6).float()
    result = autoregressive_state_predictions(
        model=model, initial_state=torch.zeros(6), actions=actions,
        stats=unit_stats(), device=torch.device("cpu"),
    )
    states = torch.full((13, 6), 999999.0); states[0] = 0
    rows = rollout_metric_rows(
        episode_id=7, prediction=result, states=states, actions=actions
    )
    assert all(not torch.any(value == 999999) for value in model.inputs)
    assert rows[0]["target_frame"] == 4
    expected_baseline_error = (actions[3] - states[4]).abs().mean()
    assert rows[0]["command_copy_endpoint_mae"] == float(expected_baseline_error)
