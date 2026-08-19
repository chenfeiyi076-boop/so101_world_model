from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .dataset import ACTION_DIM, STATE_DIM, EpisodeStateData


@dataclass(frozen=True)
class StateActionStats:
    state_mean: torch.Tensor
    state_std: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor
    source: str = "train_episodes_only"

    def __post_init__(self) -> None:
        values = {
            "state_mean": (self.state_mean, STATE_DIM),
            "state_std": (self.state_std, STATE_DIM),
            "action_mean": (self.action_mean, ACTION_DIM),
            "action_std": (self.action_std, ACTION_DIM),
        }
        for name, (value, dim) in values.items():
            tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu().contiguous()
            if tensor.shape != (dim,) or not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must be finite [{dim}]")
            if name.endswith("std") and torch.any(tensor <= 0):
                raise ValueError(f"{name} must be strictly positive")
            object.__setattr__(self, name, tensor)

    def normalize_states(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.state_mean.to(value.device)) / self.state_std.to(value.device)

    def denormalize_states(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.state_std.to(value.device) + self.state_mean.to(value.device)

    def normalize_actions(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.action_mean.to(value.device)) / self.action_std.to(value.device)

    def to_dict(self) -> dict[str, object]:
        return {
            "state_mean": self.state_mean.clone(), "state_std": self.state_std.clone(),
            "action_mean": self.action_mean.clone(), "action_std": self.action_std.clone(),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "StateActionStats":
        return cls(
            state_mean=value["state_mean"], state_std=value["state_std"],
            action_mean=value["action_mean"], action_std=value["action_std"],
            source=str(value.get("source", "checkpoint_train_episodes_only")),
        )


def compute_train_stats(
    episodes: Iterable[EpisodeStateData], *, epsilon: float = 1e-6
) -> StateActionStats:
    values = list(episodes)
    if not values:
        raise ValueError("cannot compute normalization from no train episodes")
    states = torch.cat([episode.states for episode in values]).float()
    actions = torch.cat([episode.actions for episode in values]).float()
    if len(states) < 2 or not torch.isfinite(states).all() or not torch.isfinite(actions).all():
        raise ValueError("normalization inputs must contain at least two finite rows")
    return StateActionStats(
        state_mean=states.mean(0),
        state_std=states.std(0, unbiased=True).clamp_min(float(epsilon)),
        action_mean=actions.mean(0),
        action_std=actions.std(0, unbiased=True).clamp_min(float(epsilon)),
        source="train_episodes_only",
    )
