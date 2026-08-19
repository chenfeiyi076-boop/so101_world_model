from __future__ import annotations

import torch
from torch import nn


class StateDynamicsMLP(nn.Module):
    def __init__(
        self,
        *,
        state_dim: int = 6,
        action_dim: int = 6,
        horizon: int = 4,
        hidden_dim: int = 256,
        num_hidden_layers: int = 3,
    ) -> None:
        super().__init__()
        if min(state_dim, action_dim, horizon, hidden_dim, num_hidden_layers) <= 0:
            raise ValueError("all model dimensions must be positive")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        input_dim = self.state_dim + self.horizon * self.action_dim
        output_dim = self.horizon * self.state_dim
        layers: list[nn.Module] = []
        for index in range(int(num_hidden_layers)):
            layers.append(nn.Linear(input_dim if index == 0 else hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, current_state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if current_state.ndim != 2 or current_state.shape[1] != self.state_dim:
            raise ValueError(f"current_state must be [B,{self.state_dim}]")
        expected = (len(current_state), self.horizon, self.action_dim)
        if tuple(actions.shape) != expected:
            raise ValueError(f"actions must be {expected}")
        value = torch.cat((current_state, actions.flatten(1)), dim=1)
        return self.network(value).reshape(-1, self.horizon, self.state_dim)


def parameter_counts(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable
