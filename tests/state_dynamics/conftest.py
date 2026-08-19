from __future__ import annotations

import torch

from src.state_dynamics.dataset import EpisodeStateData


def make_episode(
    episode_id: int, length: int, *, task_id: int = 0, offset: float = 0.0
) -> EpisodeStateData:
    rows = torch.arange(length, dtype=torch.float32)[:, None]
    dimensions = torch.arange(6, dtype=torch.float32)[None, :]
    return EpisodeStateData(
        episode_id=episode_id, task_id=task_id,
        frame_indices=torch.arange(length),
        actions=(1000 * episode_id + 10 * rows + dimensions + offset).float(),
        states=(100 * episode_id + rows + dimensions / 10 + offset).float(),
    )
