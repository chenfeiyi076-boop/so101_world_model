from __future__ import annotations

import copy
import subprocess
from pathlib import Path

import torch

from .config import resolve_config, validate_config_compatibility
from .data.common import ActionStats


CHECKPOINT_VERSION = 2


def current_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def config_with_action_stats(config: dict, stats: ActionStats) -> dict:
    resolved = copy.deepcopy(config)
    resolved["action"]["action_mean"] = stats.mean.tolist()
    resolved["action"]["action_std"] = stats.std.tolist()
    resolved["action"]["normalization_method"] = stats.method
    resolved["action"]["normalization_source"] = stats.source
    return resolved


def make_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict,
    action_stats: ActionStats,
    data_info: dict,
    best_val_loss: float | None,
    scheduler=None,
    ema=None,
) -> dict:
    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "step": int(step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config_with_action_stats(config, action_stats),
        "action_stats": action_stats.to_dict(),
        "data_info": copy.deepcopy(data_info),
        "metrics": {"best_val_loss": best_val_loss},
        "git_commit": current_git_commit(),
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if ema is not None:
        checkpoint["ema_model_state_dict"] = ema.state_dict()
    return checkpoint


def save_checkpoint(path: str | Path, checkpoint: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "checkpoint_version",
        "step",
        "model_state_dict",
        "config",
        "action_stats",
        "data_info",
        "metrics",
    }
    missing = required - checkpoint.keys()
    if missing:
        raise RuntimeError(f"causal checkpoint missing keys: {sorted(missing)}")
    if int(checkpoint["checkpoint_version"]) != CHECKPOINT_VERSION:
        raise RuntimeError("unsupported causal checkpoint version")
    checkpoint["config"] = resolve_config(checkpoint["config"])
    stats = action_stats_from_checkpoint(checkpoint)
    config_mean = torch.tensor(checkpoint["config"]["action"]["action_mean"])
    config_std = torch.tensor(checkpoint["config"]["action"]["action_std"])
    if not torch.equal(config_mean.float(), stats.mean):
        raise RuntimeError("config action_mean differs from checkpoint action_stats")
    if not torch.equal(config_std.float(), stats.std):
        raise RuntimeError("config action_std differs from checkpoint action_stats")
    return checkpoint


def action_stats_from_checkpoint(checkpoint: dict) -> ActionStats:
    """Evaluation's sole source of normalization values."""

    return ActionStats.from_dict(checkpoint["action_stats"])


def validate_requested_config(checkpoint: dict, requested_config: dict) -> None:
    validate_config_compatibility(checkpoint["config"], requested_config)
