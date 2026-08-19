from __future__ import annotations

import torch

from src.state_dynamics.checkpointing import (
    atomic_save_checkpoint, load_checkpoint, make_checkpoint,
)
from src.state_dynamics.config import resolve_config
from src.state_dynamics.model import StateDynamicsMLP
from src.state_dynamics.normalization import StateActionStats


def test_checkpoint_roundtrip_preserves_exact_output(tmp_path):
    torch.manual_seed(3)
    model = StateDynamicsMLP(); optimizer = torch.optim.AdamW(model.parameters())
    stats = StateActionStats(
        torch.arange(6).float(), torch.arange(1, 7).float(),
        torch.arange(6).float() + 10, torch.arange(1, 7).float() + 2,
    )
    checkpoint = make_checkpoint(
        model=model, optimizer=optimizer, epoch=2, global_step=17,
        config=resolve_config({}), stats=stats,
        split_manifest_path=tmp_path / "split.json",
        split_manifest_sha256="abc", best_val_loss=0.25,
        dataset_identity="synthetic",
    )
    path = tmp_path / "checkpoint.pt"; atomic_save_checkpoint(path, checkpoint)
    loaded = load_checkpoint(path)
    restored = StateDynamicsMLP(); restored.load_state_dict(loaded["model_state_dict"])
    state, actions = torch.randn(3, 6), torch.randn(3, 4, 6)
    assert torch.equal(model(state, actions), restored(state, actions))
    assert loaded["normalization_stats"].source == "train_episodes_only"
