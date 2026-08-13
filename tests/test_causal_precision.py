from __future__ import annotations

from contextlib import nullcontext
from contextlib import contextmanager

import pytest
import torch

import src.causal.runtime as causal_runtime
from scripts.evaluate_causal import evaluation_precision
from src.causal.config import resolve_config
from src.causal.runtime import (
    causal_flow_loss,
    evaluate_flow_loss,
    precision_context,
    validate_precision_device,
)


def _config(precision: str | None = None) -> dict:
    train = {"batch_size": 1, "steps": 1, "lr": 1e-4}
    if precision is not None:
        train["precision"] = precision
    return {
        "experiment": {"name": "precision_test", "seed": 42},
        "data": {
            "mode": "single_episode",
            "episode_id": 0,
            "cache_path": "unused.pt",
            "split_ranges": {"train": [0, 20], "val": [20, 40]},
        },
        "temporal": {"num_frames": 4, "num_history": 2, "frame_stride": 1},
        "action": {
            "raw_action_dim": 6,
            "alignment": "causal",
            "representation": "shifted_sampled",
            "normalize": True,
            "null_condition": "zero_embedding",
        },
        "model": {
            "in_channels": 2,
            "patch_size": 1,
            "hidden_size": 8,
            "depth": 1,
            "num_heads": 1,
        },
        "flow_matching": {"future_only_loss": True},
        "train": train,
        "checkpoint": {"output_dir": "unused"},
        "latent": {"convention": "posterior_sample_times_scaling_no_shift"},
    }


def test_precision_defaults_to_fp32():
    assert resolve_config(_config())["train"]["precision"] == "fp32"


def test_historical_evaluation_config_without_precision_defaults_to_fp32():
    config = _config()
    assert "precision" not in config["train"]
    assert evaluation_precision(config) == "fp32"


def test_explicit_fp32_precision_resolves():
    assert resolve_config(_config("fp32"))["train"]["precision"] == "fp32"


def test_explicit_bf16_precision_resolves():
    assert resolve_config(_config("bf16"))["train"]["precision"] == "bf16"


def test_invalid_precision_fails():
    with pytest.raises(ValueError, match="train.precision"):
        resolve_config(_config("fp16"))


def test_cpu_bf16_is_rejected_instead_of_silently_running():
    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        validate_precision_device("bf16", torch.device("cpu"))
    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        precision_context(torch.device("cpu"), "bf16")


def test_fp32_uses_no_autocast_context():
    with precision_context(torch.device("cpu"), "fp32"):
        value = torch.ones(1) + 1
    assert value.dtype == torch.float32


def test_cuda_bf16_selects_torch_autocast_without_requiring_cuda_ci(monkeypatch):
    calls = []

    def fake_autocast(*, device_type, dtype):
        calls.append((device_type, dtype))
        return nullcontext()

    monkeypatch.setattr(torch, "autocast", fake_autocast)
    with precision_context(torch.device("cuda"), "bf16"):
        pass
    assert calls == [("cuda", torch.bfloat16)]


def test_causal_flow_forward_runs_inside_precision_context(monkeypatch):
    state = {"active": False, "calls": []}

    @contextmanager
    def recording_context(device, precision):
        state["calls"].append((device.type, precision))
        state["active"] = True
        try:
            yield
        finally:
            state["active"] = False

    class RecordingModel(torch.nn.Module):
        def forward(self, latents, tau, action_cond, action_valid_mask):
            assert state["active"]
            return torch.zeros_like(latents)

    monkeypatch.setattr(causal_runtime, "precision_context", recording_context)
    batch = {
        "latents": torch.randn(1, 4, 2, 2, 2),
        "action_cond": torch.randn(1, 4, 6),
        "action_valid_mask": torch.tensor([[False, True, True, True]]),
    }
    loss = causal_flow_loss(
        RecordingModel(),
        batch,
        device=torch.device("cpu"),
        num_history=2,
        precision="fp32",
    )
    assert torch.isfinite(loss)
    assert state["calls"] == [("cpu", "fp32")]


def test_validation_passes_the_requested_precision_to_loss(monkeypatch):
    seen = []

    def fake_loss(model, batch, *, device, num_history, precision):
        seen.append(precision)
        return torch.tensor(2.0)

    monkeypatch.setattr(causal_runtime, "causal_flow_loss", fake_loss)
    model = torch.nn.Linear(1, 1)
    loader = [{"latents": torch.zeros(2, 4, 1, 1, 1)}]
    value = evaluate_flow_loss(
        model,
        loader,
        device=torch.device("cpu"),
        num_history=2,
        seed=42,
        precision="fp32",
    )
    assert value == 2.0
    assert seen == ["fp32"]
