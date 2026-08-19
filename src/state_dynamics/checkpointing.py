from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import torch

from .config import resolve_config
from .normalization import StateActionStats


CHECKPOINT_VERSION = 1


def current_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def make_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
    stats: StateActionStats,
    split_manifest_path: str | Path,
    split_manifest_sha256: str,
    best_val_loss: float,
    dataset_identity: str,
) -> dict[str, Any]:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "config": resolve_config(config),
        "normalization": stats.to_dict(),
        "split_manifest_path": str(Path(split_manifest_path).resolve()),
        "split_manifest_sha256": str(split_manifest_sha256),
        "best_val_loss": float(best_val_loss),
        "random_seed": int(config["experiment"]["seed"]),
        "dataset_identity": str(dataset_identity),
        "git_commit": current_git_commit(),
    }


def atomic_save_checkpoint(path: str | Path, checkpoint: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "checkpoint_version", "model_state_dict", "optimizer_state_dict", "epoch",
        "global_step", "config", "normalization", "split_manifest_path",
        "split_manifest_sha256", "best_val_loss", "random_seed", "dataset_identity",
    }
    missing = required - set(checkpoint)
    if missing:
        raise RuntimeError(f"state dynamics checkpoint is missing: {sorted(missing)}")
    if int(checkpoint["checkpoint_version"]) != CHECKPOINT_VERSION:
        raise RuntimeError("unsupported state dynamics checkpoint version")
    checkpoint["config"] = resolve_config(checkpoint["config"])
    checkpoint["normalization_stats"] = StateActionStats.from_dict(
        checkpoint["normalization"]
    )
    return checkpoint
