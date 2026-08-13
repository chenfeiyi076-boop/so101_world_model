from __future__ import annotations

import copy
from pathlib import Path

import yaml

from .data.common import (
    NORMALIZATION_METHOD,
    SUPPORTED_FRAME_STRIDES,
    SUPPORTED_REPRESENTATIONS,
    effective_action_dim,
)


def load_and_resolve_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("causal config must be a mapping")
    resolved = resolve_config(config)
    resolved["config_path"] = str(path)
    return resolved


def resolve_config(config: dict) -> dict:
    value = copy.deepcopy(config)
    for section in (
        "experiment",
        "data",
        "temporal",
        "action",
        "model",
        "flow_matching",
        "train",
        "checkpoint",
        "latent",
    ):
        if section not in value or not isinstance(value[section], dict):
            raise ValueError(f"missing config section {section!r}")

    data = value["data"]
    temporal = value["temporal"]
    action = value["action"]
    model = value["model"]
    train = value["train"]

    if data.get("mode") not in {"single_episode", "multi_episode"}:
        raise ValueError("data.mode must be single_episode or multi_episode")
    if data["mode"] == "single_episode":
        if "episode_id" not in data or "split_ranges" not in data:
            raise ValueError("single_episode requires episode_id and split_ranges")
        if "train" not in data["split_ranges"] or "val" not in data["split_ranges"]:
            raise ValueError("single_episode requires explicit train and val ranges")
    elif "manifest_path" not in data:
        raise ValueError("multi_episode requires manifest_path")

    num_frames = int(temporal["num_frames"])
    num_history = int(temporal["num_history"])
    frame_stride = int(temporal["frame_stride"])
    if num_frames <= 1 or not 0 < num_history < num_frames:
        raise ValueError("require 0 < num_history < num_frames")
    if frame_stride not in SUPPORTED_FRAME_STRIDES:
        raise ValueError("unsupported frame_stride")
    temporal.update(
        num_frames=num_frames,
        num_history=num_history,
        frame_stride=frame_stride,
    )

    raw_dim = int(action.get("raw_action_dim", action.get("raw_dim", 6)))
    representation = action["representation"]
    if representation not in SUPPORTED_REPRESENTATIONS:
        raise ValueError("unsupported action representation")
    if action.get("alignment", "causal") != "causal":
        raise ValueError("new causal configs require action.alignment=causal")
    if action.get("null_condition", "zero_embedding") != "zero_embedding":
        raise ValueError("NULL condition must use zero_embedding")
    if action.get("normalize", True) is not True:
        raise ValueError("causal training currently requires action normalization")
    action["raw_action_dim"] = raw_dim
    action.pop("raw_dim", None)
    action["alignment"] = "causal"
    action["normalize"] = True
    if action.get("normalization_method", NORMALIZATION_METHOD) != NORMALIZATION_METHOD:
        raise ValueError("unsupported action normalization method")
    action["normalization_method"] = NORMALIZATION_METHOD
    # A YAML config resolves to the policy-level source below. A checkpoint
    # contains the concrete train episode/range provenance and must retain it.
    action["normalization_source"] = action.get(
        "normalization_source", "training_data_only"
    )
    action["null_condition"] = "zero_embedding"
    action["effective_action_dim"] = effective_action_dim(
        raw_dim, frame_stride, representation
    )

    for key in ("in_channels", "patch_size", "hidden_size", "depth", "num_heads"):
        model[key] = int(model[key])
    model["mlp_ratio"] = float(model.get("mlp_ratio", 4.0))
    model["use_qk_norm"] = bool(model.get("use_qk_norm", True))

    train["batch_size"] = int(train["batch_size"])
    train["steps"] = int(train["steps"])
    train["lr"] = float(train["lr"])
    train["weight_decay"] = float(train.get("weight_decay", 0.0))
    train["grad_clip"] = float(train.get("grad_clip", 1.0))
    train["val_every"] = int(train.get("val_every", 250))
    train["val_windows"] = int(train.get("val_windows", 128))
    train["num_workers"] = int(train.get("num_workers", 0))
    value["experiment"]["seed"] = int(value["experiment"].get("seed", 42))

    if value["flow_matching"].get("future_only_loss", True) is not True:
        raise ValueError("flow_matching.future_only_loss must stay true")
    value["flow_matching"]["future_only_loss"] = True
    if value["latent"].get("convention") != "posterior_sample_times_scaling_no_shift":
        raise ValueError("this branch must preserve the existing latent convention")

    value["checkpoint"]["save_best"] = bool(
        value["checkpoint"].get("save_best", True)
    )
    value["checkpoint"]["save_last"] = bool(
        value["checkpoint"].get("save_last", True)
    )
    return value


def checkpoint_compatibility_fields(config: dict) -> dict[str, object]:
    return {
        "data.mode": config["data"]["mode"],
        "temporal.num_frames": config["temporal"]["num_frames"],
        "temporal.num_history": config["temporal"]["num_history"],
        "temporal.frame_stride": config["temporal"]["frame_stride"],
        "action.alignment": config["action"]["alignment"],
        "action.representation": config["action"]["representation"],
        "action.raw_action_dim": config["action"]["raw_action_dim"],
        "action.effective_action_dim": config["action"]["effective_action_dim"],
        "action.normalization_method": config["action"]["normalization_method"],
        "latent.convention": config["latent"]["convention"],
        "model.in_channels": config["model"]["in_channels"],
        "model.patch_size": config["model"]["patch_size"],
        "model.hidden_size": config["model"]["hidden_size"],
        "model.depth": config["model"]["depth"],
        "model.num_heads": config["model"]["num_heads"],
    }


def validate_config_compatibility(
    checkpoint_config: dict,
    requested_config: dict,
) -> None:
    expected = checkpoint_compatibility_fields(checkpoint_config)
    requested = checkpoint_compatibility_fields(requested_config)
    mismatches = [
        f"{key}: checkpoint={expected[key]!r}, requested={requested[key]!r}"
        for key in expected
        if expected[key] != requested[key]
    ]
    if mismatches:
        raise ValueError("causal checkpoint/config mismatch:\n" + "\n".join(mismatches))
