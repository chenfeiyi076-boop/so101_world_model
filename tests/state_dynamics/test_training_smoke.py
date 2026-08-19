from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.state_dynamics.split import atomic_write_manifest, build_split_manifest

from .conftest import make_episode


def test_synthetic_cli_smoke_runs_one_forward_backward(tmp_path):
    pa = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    data_root = tmp_path / "dataset" / "data"
    data_root.mkdir(parents=True)
    rows = []
    episode_to_task = {episode_id: 0 for episode_id in range(10)}
    manifest = build_split_manifest(
        dataset_root=tmp_path / "dataset",
        episode_to_task=episode_to_task,
        seed=42,
    )
    test_episode_ids = set(manifest["test_episode_ids"])
    for episode_id in range(10):
        # Deliberately omit test data: the training entry point must not read it.
        if episode_id in test_episode_ids:
            continue
        episode = make_episode(episode_id, 6, task_id=0)
        for frame in range(episode.length):
            rows.append({
                "episode_index": episode_id,
                "frame_index": frame,
                "task_index": 0,
                "timestamp": frame / 20.0,
                "action": episode.actions[frame].tolist(),
                "observation.state": episode.states[frame].tolist(),
            })
    parquet.write_table(pa.Table.from_pylist(rows), data_root / "part.parquet")
    manifest_path = tmp_path / "split.json"
    atomic_write_manifest(manifest, manifest_path)
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(
        yaml.safe_dump({
            "model": {"hidden_dim": 8, "num_hidden_layers": 1},
            "train": {
                "batch_size": 8,
                "num_workers": 0,
                "epochs": 1,
                "device": "cpu",
            },
        }),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_so101_state_mlp.py",
            "--config", str(config_path),
            "--dataset-root", str(tmp_path / "dataset"),
            "--split-manifest", str(manifest_path),
            "--output-dir", str(tmp_path / "output"),
            "--smoke-test",
        ],
        cwd=str(Path(__file__).resolve().parents[2]),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SMOKE PASS" in result.stdout
    assert "test episodes (manifest only): 1" in result.stdout
    assert not (tmp_path / "output").exists()
