from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from src.causal.config import resolve_config
from src.causal.data.common import ActionStats
from src.causal.model import CausalDiT
from src.models.dit import DiT


LATENT_CONVENTION = "posterior_sample_times_scaling_no_shift"


@dataclass
class LoadedWorldModel:
    model: torch.nn.Module
    checkpoint: dict[str, Any]
    checkpoint_type: str
    resolved_config: dict[str, Any]
    action_stats: ActionStats
    checkpoint_path: str


def detect_checkpoint_type(checkpoint: dict[str, Any]) -> str:
    if (
        int(checkpoint.get("checkpoint_version", 0)) >= 2
        and "config" in checkpoint
        and "action_stats" in checkpoint
    ):
        return "causal_v2"
    return "legacy_v1"


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("legacy config must contain a YAML mapping")
    return value


def _require_mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise ValueError(f"legacy config requires mapping {key!r}")
    return item


def resolve_legacy_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate an explicit legacy inference config without guessing semantics."""

    value = copy.deepcopy(config)
    model = _require_mapping(value, "model")
    temporal = _require_mapping(value, "temporal")
    action = _require_mapping(value, "action")
    latent = _require_mapping(value, "latent")

    for key in (
        "in_channels",
        "patch_size",
        "hidden_size",
        "depth",
        "num_heads",
        "action_dim",
    ):
        if key not in model:
            raise ValueError(f"legacy config requires model.{key}")
        model[key] = int(model[key])
    model["mlp_ratio"] = float(model.get("mlp_ratio", 4.0))
    model["use_qk_norm"] = bool(model.get("use_qk_norm", True))

    for key in ("num_frames", "num_history", "frame_stride"):
        if key not in temporal:
            raise ValueError(f"legacy config requires temporal.{key}")
        temporal[key] = int(temporal[key])
    if not 0 < temporal["num_history"] < temporal["num_frames"]:
        raise ValueError("legacy config requires 0 < num_history < num_frames")
    if temporal["frame_stride"] <= 0:
        raise ValueError("legacy frame_stride must be positive")

    if action.get("alignment") not in {
        "synchronized_legacy",
        "causal_chunk_legacy",
    }:
        raise ValueError(
            "legacy action.alignment must be synchronized_legacy or causal_chunk_legacy"
        )
    if action.get("representation") not in {"sampled", "chunk"}:
        raise ValueError("legacy action.representation must be sampled or chunk")
    if (
        action["representation"] == "sampled"
        and action["alignment"] != "synchronized_legacy"
    ):
        raise ValueError("legacy sampled actions require synchronized_legacy alignment")
    if action["representation"] == "chunk" and action["alignment"] != "causal_chunk_legacy":
        raise ValueError("legacy chunk actions require causal_chunk_legacy alignment")

    action["raw_dim"] = int(action.get("raw_dim", 6))
    action["control_representation"] = action.get(
        "control_representation", "absolute"
    )
    if action["control_representation"] not in {"absolute", "delta"}:
        raise ValueError("legacy control_representation must be absolute or delta")
    expected_dim = action["raw_dim"] * (
        temporal["frame_stride"] if action["representation"] == "chunk" else 1
    )
    if model["action_dim"] != expected_dim:
        raise ValueError(
            f"legacy model.action_dim={model['action_dim']} but action semantics require {expected_dim}"
        )
    action["effective_action_dim"] = expected_dim

    if latent.get("convention") != LATENT_CONVENTION:
        raise ValueError(
            f"legacy latent.convention must be {LATENT_CONVENTION!r}"
        )
    return value


def _compare_if_present(
    embedded: dict[str, Any],
    embedded_key: str,
    requested: Any,
    requested_name: str,
) -> None:
    if embedded_key in embedded and embedded[embedded_key] != requested:
        raise ValueError(
            f"legacy checkpoint/config mismatch for {requested_name}: "
            f"checkpoint={embedded[embedded_key]!r}, config={requested!r}"
        )


def validate_legacy_checkpoint_config(
    checkpoint: dict[str, Any], config: dict[str, Any]
) -> None:
    embedded = checkpoint.get("config", {})
    if not isinstance(embedded, dict):
        raise ValueError("legacy checkpoint config must be a mapping when present")
    model = config["model"]
    temporal = config["temporal"]
    action = config["action"]

    for checkpoint_key, requested_key in (
        ("n_frames", "num_frames"),
        ("num_history", "num_history"),
        ("frame_skip", "frame_stride"),
    ):
        _compare_if_present(
            embedded,
            checkpoint_key,
            temporal[requested_key],
            f"temporal.{requested_key}",
        )
    for key in (
        "in_channels",
        "patch_size",
        "hidden_size",
        "depth",
        "num_heads",
        "action_dim",
        "mlp_ratio",
        "use_qk_norm",
    ):
        _compare_if_present(embedded, key, model[key], f"model.{key}")
    _compare_if_present(
        embedded,
        "action_mode",
        action["representation"],
        "action.representation",
    )
    _compare_if_present(
        embedded,
        "action_representation",
        action["control_representation"],
        "action.control_representation",
    )
    if "model" not in checkpoint:
        raise RuntimeError("legacy checkpoint is missing model state_dict key 'model'")

    state_dict = checkpoint["model"]
    action_weight = state_dict.get("action_embedder.proj.weight")
    if action_weight is None:
        raise RuntimeError("legacy checkpoint is missing action_embedder.proj.weight")
    if tuple(action_weight.shape) != (model["hidden_size"], model["action_dim"]):
        raise ValueError(
            "legacy checkpoint action projection shape does not match explicit config"
        )


def _stats_from_legacy(
    checkpoint: dict[str, Any], config: dict[str, Any]
) -> ActionStats:
    if "action_mean" in checkpoint and "action_std" in checkpoint:
        stats = ActionStats(
            checkpoint["action_mean"],
            checkpoint["action_std"],
            source="legacy_checkpoint",
            method="legacy_fixed_zscore",
        )
        action = config["action"]
        if "mean" in action and not torch.equal(
            torch.as_tensor(action["mean"]).float(), stats.mean
        ):
            raise ValueError("legacy config action.mean differs from checkpoint action_mean")
        if "std" in action and not torch.equal(
            torch.as_tensor(action["std"]).float(), stats.std
        ):
            raise ValueError("legacy config action.std differs from checkpoint action_std")
        return stats
    action = config["action"]
    if "mean" in action and "std" in action:
        return ActionStats(
            action["mean"],
            action["std"],
            source="legacy_config_fixed_stats",
            method="legacy_fixed_zscore",
        )
    raise RuntimeError(
        "legacy checkpoint has no action_mean/action_std; provide fixed training "
        "statistics as action.mean/action.std in --legacy-config. Evaluation "
        "episode statistics will not be computed."
    )


def _load_causal(
    checkpoint: dict[str, Any], checkpoint_path: str, device: torch.device
) -> LoadedWorldModel:
    required = {"config", "action_stats", "model_state_dict", "checkpoint_version"}
    missing = required - checkpoint.keys()
    if missing:
        raise RuntimeError(f"causal checkpoint missing keys: {sorted(missing)}")
    config = resolve_config(checkpoint["config"])
    stats = ActionStats.from_dict(checkpoint["action_stats"])
    if stats.raw_dim != int(config["action"]["raw_action_dim"]):
        raise RuntimeError("causal action_stats dimension differs from raw_action_dim")
    config_mean = torch.as_tensor(config["action"].get("action_mean"))
    config_std = torch.as_tensor(config["action"].get("action_std"))
    if not torch.equal(config_mean.float(), stats.mean):
        raise RuntimeError("causal config action_mean differs from checkpoint action_stats")
    if not torch.equal(config_std.float(), stats.std):
        raise RuntimeError("causal config action_std differs from checkpoint action_stats")

    model_cfg = config["model"]
    model = CausalDiT(
        in_channels=model_cfg["in_channels"],
        patch_size=model_cfg["patch_size"],
        hidden_size=model_cfg["hidden_size"],
        depth=model_cfg["depth"],
        num_heads=model_cfg["num_heads"],
        action_dim=config["action"]["effective_action_dim"],
        mlp_ratio=model_cfg["mlp_ratio"],
        use_qk_norm=model_cfg["use_qk_norm"],
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return LoadedWorldModel(
        model=model,
        checkpoint=checkpoint,
        checkpoint_type="causal_v2",
        resolved_config=config,
        action_stats=stats,
        checkpoint_path=checkpoint_path,
    )


def _load_legacy(
    checkpoint: dict[str, Any],
    checkpoint_path: str,
    legacy_config_path: str | Path | None,
    device: torch.device,
) -> LoadedWorldModel:
    if legacy_config_path is None:
        raise ValueError(
            "legacy checkpoint requires --legacy-config; architecture and action "
            "semantics are never inferred silently"
        )
    config = resolve_legacy_config(_read_yaml(legacy_config_path))
    validate_legacy_checkpoint_config(checkpoint, config)
    stats = _stats_from_legacy(checkpoint, config)
    if stats.raw_dim != int(config["action"]["raw_dim"]):
        raise RuntimeError("legacy action statistics dimension differs from action.raw_dim")
    model_cfg = config["model"]
    model = DiT(
        in_channels=model_cfg["in_channels"],
        patch_size=model_cfg["patch_size"],
        hidden_size=model_cfg["hidden_size"],
        depth=model_cfg["depth"],
        num_heads=model_cfg["num_heads"],
        action_dim=model_cfg["action_dim"],
        mlp_ratio=model_cfg["mlp_ratio"],
        use_qk_norm=model_cfg["use_qk_norm"],
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return LoadedWorldModel(
        model=model,
        checkpoint=checkpoint,
        checkpoint_type="legacy_v1",
        resolved_config=config,
        action_stats=stats,
        checkpoint_path=checkpoint_path,
    )


def load_world_model(
    checkpoint_path: str | Path,
    *,
    legacy_config_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> LoadedWorldModel:
    """Load either checkpoint family with strict state_dict semantics."""

    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("world model checkpoint must contain a mapping")
    checkpoint_type = detect_checkpoint_type(checkpoint)
    target_device = torch.device(device)
    if checkpoint_type == "causal_v2":
        if legacy_config_path is not None:
            raise ValueError("--legacy-config cannot override a causal_v2 checkpoint")
        return _load_causal(checkpoint, str(path), target_device)
    return _load_legacy(
        checkpoint, str(path), legacy_config_path, target_device
    )
