from __future__ import annotations

from src.state_dynamics.split import (
    build_split_manifest,
    stratified_episode_split,
    validate_split_manifest,
)


def test_split_is_deterministic_disjoint_and_complete(tmp_path):
    mapping = {episode: episode // 20 for episode in range(100)}
    first, counts = stratified_episode_split(mapping, seed=42)
    second, _ = stratified_episode_split(mapping, seed=42)
    assert first == second
    train, val, test = map(set, (first["train"], first["val"], first["test"]))
    assert not (train & val or train & test or val & test)
    assert train | val | test == set(mapping)
    assert (len(train), len(val), len(test)) == (80, 10, 10)
    assert all(value == {"total": 20, "train": 16, "val": 2, "test": 2}
               for value in counts.values())
    manifest = build_split_manifest(
        dataset_root=tmp_path, episode_to_task=mapping, seed=42
    )
    assert validate_split_manifest(manifest) is manifest
