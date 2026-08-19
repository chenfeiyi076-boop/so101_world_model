from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


STATE_DIM = 6
ACTION_DIM = 6


@dataclass(frozen=True)
class EpisodeStateData:
    episode_id: int
    task_id: int
    frame_indices: torch.Tensor
    actions: torch.Tensor
    states: torch.Tensor

    @property
    def length(self) -> int:
        return int(len(self.states))


def _stack_vectors(values: Sequence[object], *, name: str, dim: int) -> torch.Tensor:
    array = np.stack([np.asarray(value, dtype=np.float32).reshape(-1) for value in values])
    tensor = torch.from_numpy(array).float().contiguous()
    if tensor.ndim != 2 or tensor.shape[1] != dim:
        raise RuntimeError(f"{name} must be [N,{dim}], got {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains NaN or Inf")
    return tensor


class SO101StateStore:
    """Read only numerical LeRobot parquet columns; never opens video."""

    def __init__(self, dataset_root: str | Path) -> None:
        import pyarrow.dataset as ds

        self.dataset_root = Path(dataset_root)
        data_root = self.dataset_root / "data"
        if not data_root.is_dir():
            raise FileNotFoundError(f"LeRobot data directory not found: {data_root}")
        self._dataset = ds.dataset(data_root, format="parquet")
        self.required_columns = (
            "episode_index", "frame_index", "task_index", "action", "observation.state"
        )
        missing = set(self.required_columns) - set(self._dataset.schema.names)
        if missing:
            raise RuntimeError(f"parquet schema is missing columns: {sorted(missing)}")

    @property
    def schema_names(self) -> list[str]:
        return list(self._dataset.schema.names)

    def inspect_episode_timestamps(self, episode_id: int) -> dict[str, object]:
        """Read timestamp diagnostics for one episode without making them model inputs."""
        import pyarrow.dataset as ds

        if "timestamp" not in self._dataset.schema.names:
            return {"available": False}
        table = self._dataset.to_table(
            columns=["frame_index", "timestamp"],
            filter=ds.field("episode_index") == int(episode_id),
        )
        frame = table.to_pandas().sort_values("frame_index")
        timestamps = np.asarray(frame["timestamp"], dtype=np.float64)
        if len(timestamps) == 0 or not np.isfinite(timestamps).all():
            raise RuntimeError(f"episode {episode_id} has missing/non-finite timestamps")
        deltas = np.diff(timestamps)
        if len(deltas) and np.any(deltas <= 0):
            raise RuntimeError(f"episode {episode_id} timestamps must be strictly increasing")
        return {
            "available": True,
            "first": float(timestamps[0]),
            "last": float(timestamps[-1]),
            "median_delta_seconds": float(np.median(deltas)) if len(deltas) else None,
        }

    def load_episodes(self, episode_ids: Sequence[int]) -> dict[int, EpisodeStateData]:
        import pyarrow.dataset as ds

        requested = sorted(set(map(int, episode_ids)))
        if not requested:
            raise ValueError("episode_ids must be non-empty")
        table = self._dataset.to_table(
            columns=list(self.required_columns),
            filter=ds.field("episode_index").isin(requested),
        )
        frame = table.to_pandas()
        found = set(map(int, frame["episode_index"].unique())) if len(frame) else set()
        if found != set(requested):
            raise RuntimeError(f"missing requested episodes: {sorted(set(requested) - found)}")
        output = {}
        for episode_id in requested:
            rows = frame[frame["episode_index"] == episode_id].sort_values("frame_index")
            frame_indices = torch.as_tensor(rows["frame_index"].to_numpy(), dtype=torch.long)
            if not torch.equal(frame_indices, torch.arange(len(rows), dtype=torch.long)):
                raise RuntimeError(f"episode {episode_id} frame_index must be contiguous from 0")
            tasks = {int(value) for value in rows["task_index"].unique()}
            if len(tasks) != 1:
                raise RuntimeError(f"episode {episode_id} must map to exactly one task")
            output[episode_id] = EpisodeStateData(
                episode_id=episode_id,
                task_id=next(iter(tasks)),
                frame_indices=frame_indices,
                actions=_stack_vectors(rows["action"].tolist(), name="action", dim=ACTION_DIM),
                states=_stack_vectors(
                    rows["observation.state"].tolist(),
                    name="observation.state", dim=STATE_DIM,
                ),
            )
        return output


class StateDynamicsWindowDataset(Dataset):
    def __init__(
        self,
        episodes: Mapping[int, EpisodeStateData],
        *,
        episode_order: Sequence[int] | None = None,
        horizon: int = 4,
        window_stride: int = 1,
    ) -> None:
        if horizon <= 0 or window_stride <= 0:
            raise ValueError("horizon and window_stride must be positive")
        self.episodes = dict(episodes)
        self.episode_ids = (
            list(map(int, episode_order)) if episode_order is not None else sorted(self.episodes)
        )
        if (
            len(self.episode_ids) != len(self.episodes)
            or set(self.episode_ids) != set(self.episodes)
        ):
            raise ValueError("episode_order must contain each episode exactly once")
        self.horizon = int(horizon)
        self.window_stride = int(window_stride)
        self.starts_by_episode: dict[int, range] = {}
        self.cumulative_ends = []
        running = 0
        for episode_id in self.episode_ids:
            length = self.episodes[episode_id].length
            starts = range(0, max(0, length - self.horizon), self.window_stride)
            if len(starts) == 0:
                raise RuntimeError(f"episode {episode_id} has no legal state windows")
            self.starts_by_episode[episode_id] = starts
            running += len(starts)
            self.cumulative_ends.append(running)

    def __len__(self) -> int:
        return self.cumulative_ends[-1] if self.cumulative_ends else 0

    def locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_position = bisect_right(self.cumulative_ends, index)
        previous = 0 if episode_position == 0 else self.cumulative_ends[episode_position - 1]
        episode_id = self.episode_ids[episode_position]
        local = index - previous
        return episode_id, self.starts_by_episode[episode_id][local]

    def __getitem__(self, index: int) -> dict[str, object]:
        episode_id, start = self.locate(index)
        episode = self.episodes[episode_id]
        end = start + self.horizon
        return {
            "current_state": episode.states[start],
            "actions": episode.actions[start:end],
            "target_states": episode.states[start + 1 : end + 1],
            "episode_id": episode_id,
            "start_frame": start,
            "task_id": episode.task_id,
        }
