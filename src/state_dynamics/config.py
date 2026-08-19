from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {"name": "so101_state_mlp", "seed": 42},
    "data": {
        "state_dim": 6,
        "action_dim": 6,
        "prediction_horizon": 4,
        "window_stride": 1,
    },
    "model": {"hidden_dim": 256, "num_hidden_layers": 3},
    "train": {
        "epochs": 50,
        "batch_size": 4096,
        "num_workers": 4,
        "lr": 1.0e-3,
        "weight_decay": 1.0e-4,
        "precision": "fp32",
        "early_stopping_patience": 8,
        "device": "auto",
    },
}


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_config(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("state dynamics config must be a mapping")
    config = _merge(DEFAULT_CONFIG, value)
    data, model, train = config["data"], config["model"], config["train"]
    for key in ("state_dim", "action_dim", "prediction_horizon", "window_stride"):
        data[key] = int(data[key])
        if data[key] <= 0:
            raise ValueError(f"data.{key} must be positive")
    if data["state_dim"] != 6 or data["action_dim"] != 6:
        raise ValueError("formal SO101 state/action dimensions must both be 6")
    if data["prediction_horizon"] != 4:
        raise ValueError("formal state dynamics prediction_horizon must be 4")
    model["hidden_dim"] = int(model["hidden_dim"])
    model["num_hidden_layers"] = int(model["num_hidden_layers"])
    if model["hidden_dim"] <= 0 or model["num_hidden_layers"] <= 0:
        raise ValueError("model dimensions/layers must be positive")
    for key in ("epochs", "batch_size", "num_workers", "early_stopping_patience"):
        train[key] = int(train[key])
    if train["epochs"] <= 0 or train["batch_size"] <= 0 or train["num_workers"] < 0:
        raise ValueError("epochs/batch_size must be positive and workers non-negative")
    if train["early_stopping_patience"] <= 0:
        raise ValueError("early_stopping_patience must be positive")
    train["lr"] = float(train["lr"])
    train["weight_decay"] = float(train["weight_decay"])
    if train["lr"] <= 0 or train["weight_decay"] < 0:
        raise ValueError("lr must be positive and weight_decay non-negative")
    if train["precision"] != "fp32":
        raise ValueError("state dynamics v1 only supports fp32")
    if train["device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("train.device must be auto, cpu, or cuda")
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    return resolve_config({} if value is None else value)
