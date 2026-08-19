from __future__ import annotations

import pytest
import torch

from src.state_dynamics.dataset import SO101StateStore, StateDynamicsWindowDataset
from .conftest import make_episode


def test_window_count_and_exact_action_state_alignment():
    episode = make_episode(3, 9)
    dataset = StateDynamicsWindowDataset({3: episode}, horizon=4, window_stride=1)
    assert len(dataset) == 9 - 4
    item = dataset[2]
    assert torch.equal(item["current_state"], episode.states[2])
    assert torch.equal(item["actions"][0], episode.actions[2])
    assert torch.equal(item["actions"][3], episode.actions[5])
    assert torch.equal(item["target_states"][0], episode.states[3])
    assert torch.equal(item["target_states"][3], episode.states[6])


def test_windows_never_cross_episode_boundaries():
    episodes = {1: make_episode(1, 6), 2: make_episode(2, 7)}
    dataset = StateDynamicsWindowDataset(episodes, episode_order=[1, 2])
    assert len(dataset) == (6 - 4) + (7 - 4)
    for index in range(len(dataset)):
        item = dataset[index]
        episode = episodes[item["episode_id"]]
        start = item["start_frame"]
        assert torch.equal(item["current_state"], episode.states[start])
        assert torch.equal(item["target_states"][-1], episode.states[start + 4])


def test_mock_parquet_loader_reads_only_numeric_schema(tmp_path):
    pa = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    data_root = tmp_path / "data"; data_root.mkdir()
    rows = []
    for episode_id, length in ((0, 6), (1, 7)):
        episode = make_episode(episode_id, length, task_id=episode_id + 10)
        for frame in range(length):
            rows.append({
                "episode_index": episode_id, "frame_index": frame,
                "task_index": episode.task_id,
                "timestamp": frame / 20.0,
                "action": episode.actions[frame].tolist(),
                "observation.state": episode.states[frame].tolist(),
            })
    parquet.write_table(pa.Table.from_pylist(rows), data_root / "part.parquet")
    store = SO101StateStore(tmp_path)
    loaded = store.load_episodes([0, 1])
    assert loaded[0].actions.shape == (6, 6)
    assert loaded[1].states.shape == (7, 6)
    assert loaded[1].task_id == 11
    timestamp_info = store.inspect_episode_timestamps(1)
    assert timestamp_info["available"] is True
    assert timestamp_info["median_delta_seconds"] == pytest.approx(0.05)
