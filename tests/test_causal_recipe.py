from __future__ import annotations

import copy

import pytest
import torch

from scripts.evaluate_causal import select_evaluation_state_dict
from src.causal.checkpointing import (
    load_checkpoint,
    make_checkpoint,
    save_checkpoint,
)
from src.causal.config import resolve_config
from src.causal.data.common import ActionStats
from src.causal.runtime import (
    ExponentialMovingAverage,
    build_optimizer,
    build_scheduler,
    lr_multiplier,
    scheduler_warmup_steps,
    validation_model,
)


def _config(train_updates: dict | None = None) -> dict:
    train = {"batch_size": 1, "steps": 8, "lr": 1e-4}
    if train_updates:
        train.update(copy.deepcopy(train_updates))
    return resolve_config(
        {
            "experiment": {"name": "recipe_test", "seed": 42},
            "data": {
                "mode": "single_episode",
                "episode_id": 0,
                "cache_path": "unused.pt",
                "split_ranges": {"train": [0, 20], "val": [20, 40]},
            },
            "temporal": {
                "num_frames": 4,
                "num_history": 2,
                "frame_stride": 1,
            },
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
            "latent": {
                "convention": "posterior_sample_times_scaling_no_shift"
            },
        }
    )


def test_legacy_recipe_defaults_preserve_old_behavior():
    train = _config()["train"]
    assert train["betas"] == [0.9, 0.999]
    assert train["eps"] == 1e-8
    assert train["scheduler"] == {
        "type": "constant",
        "warmup_ratio": 0.0,
        "min_lr_ratio": 1.0,
    }
    assert train["ema"] == {"enabled": False, "decay": 0.9995}


def test_paper_style_recipe_resolves_without_fixing_total_steps():
    train = _config(
        {
            "steps": 123,
            "precision": "bf16",
            "lr": 1e-4,
            "betas": [0.9, 0.99],
            "eps": 1e-8,
            "weight_decay": 0.002,
            "grad_clip": 1.0,
            "scheduler": {
                "type": "warmup_cosine",
                "warmup_ratio": 0.03,
                "min_lr_ratio": 0.7,
            },
            "ema": {"enabled": True, "decay": 0.9995},
        }
    )["train"]
    assert train["steps"] == 123
    assert train["betas"] == [0.9, 0.99]
    assert train["weight_decay"] == 0.002
    assert train["scheduler"]["warmup_ratio"] == 0.03
    assert train["scheduler"]["min_lr_ratio"] == 0.7
    assert train["ema"] == {"enabled": True, "decay": 0.9995}


@pytest.mark.parametrize("betas", [[0.9], [0.9, 1.0], [-0.1, 0.9]])
def test_invalid_betas_fail(betas):
    with pytest.raises(ValueError, match="beta"):
        _config({"betas": betas})


def test_invalid_eps_fails():
    with pytest.raises(ValueError, match="train.eps"):
        _config({"eps": 0.0})


def test_invalid_scheduler_type_fails():
    with pytest.raises(ValueError, match="scheduler.type"):
        _config({"scheduler": {"type": "linear"}})


@pytest.mark.parametrize("value", [-0.1, 1.0])
def test_invalid_warmup_ratio_fails(value):
    with pytest.raises(ValueError, match="warmup_ratio"):
        _config({"scheduler": {"warmup_ratio": value}})


@pytest.mark.parametrize("value", [0.0, 1.1])
def test_invalid_min_lr_ratio_fails(value):
    with pytest.raises(ValueError, match="min_lr_ratio"):
        _config({"scheduler": {"min_lr_ratio": value}})


@pytest.mark.parametrize("value", [-0.1, 1.0])
def test_invalid_ema_decay_fails(value):
    with pytest.raises(ValueError, match="ema.decay"):
        _config({"ema": {"decay": value}})


def test_build_optimizer_uses_resolved_recipe():
    config = _config(
        {
            "lr": 3e-4,
            "betas": [0.8, 0.95],
            "eps": 2e-7,
            "weight_decay": 0.012,
        }
    )
    model = torch.nn.Linear(2, 1)
    optimizer = build_optimizer(config, model)
    group = optimizer.param_groups[0]
    assert isinstance(optimizer, torch.optim.AdamW)
    assert group["lr"] == 3e-4
    assert group["betas"] == (0.8, 0.95)
    assert group["eps"] == 2e-7
    assert group["weight_decay"] == 0.012


def _used_lrs(config: dict) -> tuple[list[float], object]:
    model = torch.nn.Linear(1, 1)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    used = []
    for _ in range(config["train"]["steps"]):
        used.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return used, scheduler


def test_warmup_cosine_exact_update_semantics():
    config = _config(
        {
            "steps": 8,
            "lr": 1e-4,
            "scheduler": {
                "type": "warmup_cosine",
                "warmup_ratio": 0.25,
                "min_lr_ratio": 0.7,
            },
        }
    )
    used, scheduler = _used_lrs(config)
    base_lr = config["train"]["lr"]
    assert scheduler.warmup_steps == 2
    assert used[0] == pytest.approx(0.5 * base_lr)
    assert used[1] == pytest.approx(base_lr)
    assert used[0] < used[1]
    assert used[2] == pytest.approx(base_lr)
    assert all(left >= right for left, right in zip(used[2:], used[3:]))
    assert used[-1] == pytest.approx(0.7 * base_lr)


def test_constant_scheduler_lr_never_changes():
    config = _config({"steps": 5})
    used, _ = _used_lrs(config)
    assert used == pytest.approx([config["train"]["lr"]] * 5)


def test_tiny_warmup_cosine_has_no_division_by_zero():
    config = _config(
        {
            "steps": 1,
            "scheduler": {
                "type": "warmup_cosine",
                "warmup_ratio": 0.9,
                "min_lr_ratio": 0.7,
            },
        }
    )
    used, scheduler = _used_lrs(config)
    assert scheduler.warmup_steps == 0
    assert used == pytest.approx([0.7 * config["train"]["lr"]])
    assert scheduler_warmup_steps(1, 0.9) == 0
    assert lr_multiplier(0, 1, 0, 0.7) == pytest.approx(0.7)


class _BufferedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1, bias=False)
        self.register_buffer("float_buffer", torch.tensor([1.0]))
        self.register_buffer("int_buffer", torch.tensor([1], dtype=torch.long))


def test_ema_initialization_update_buffers_and_round_trip():
    raw = _BufferedModel()
    with torch.no_grad():
        raw.linear.weight.fill_(1.0)
    ema = ExponentialMovingAverage(raw, decay=0.75)
    assert torch.equal(ema.model.linear.weight, raw.linear.weight)
    assert ema.model.linear.weight.dtype == torch.float32
    assert all(not parameter.requires_grad for parameter in ema.model.parameters())

    with torch.no_grad():
        raw.linear.weight.fill_(5.0)
        raw.float_buffer.fill_(3.0)
        raw.int_buffer.fill_(7)
    ema.update(raw)
    assert torch.equal(ema.model.linear.weight, torch.full_like(raw.linear.weight, 2.0))
    assert torch.equal(ema.model.float_buffer, raw.float_buffer)
    assert torch.equal(ema.model.int_buffer, raw.int_buffer)

    restored = ExponentialMovingAverage(_BufferedModel(), decay=0.75)
    restored.load_state_dict(ema.state_dict())
    for key, value in ema.state_dict().items():
        assert torch.equal(restored.state_dict()[key], value)


def test_validation_model_selects_ema_or_raw():
    raw = torch.nn.Linear(2, 1)
    ema = ExponentialMovingAverage(raw, decay=0.9)
    assert validation_model(raw, ema) is ema.model
    assert validation_model(raw, None) is raw


def _checkpoint_components(config: dict):
    model = torch.nn.Linear(2, 1)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    ema = ExponentialMovingAverage(model, config["train"]["ema"]["decay"])
    stats = ActionStats(torch.zeros(6), torch.ones(6))
    return model, optimizer, scheduler, ema, stats


def test_checkpoint_optionally_saves_scheduler_ema_and_keeps_raw_model():
    config = _config({"ema": {"enabled": True, "decay": 0.9}})
    model, optimizer, scheduler, ema, stats = _checkpoint_components(config)
    checkpoint = make_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        step=1,
        config=config,
        action_stats=stats,
        data_info={"mode": "single_episode"},
        best_val_loss=1.0,
    )
    assert "model_state_dict" in checkpoint
    assert "scheduler_state_dict" in checkpoint
    assert "ema_model_state_dict" in checkpoint


def test_historical_v2_checkpoint_without_optional_recipe_states_loads(tmp_path):
    config = _config()
    model, optimizer, _, _, stats = _checkpoint_components(config)
    checkpoint = make_checkpoint(
        model=model,
        optimizer=optimizer,
        step=1,
        config=config,
        action_stats=stats,
        data_info={"mode": "single_episode"},
        best_val_loss=1.0,
    )
    assert "scheduler_state_dict" not in checkpoint
    assert "ema_model_state_dict" not in checkpoint
    path = tmp_path / "historical_v2.pt"
    save_checkpoint(path, checkpoint)
    loaded = load_checkpoint(path)
    assert loaded["checkpoint_version"] == 2


def test_evaluation_weight_selection_rules():
    raw = {"weight": torch.tensor([1.0])}
    ema = {"weight": torch.tensor([2.0])}
    checkpoint = {"model_state_dict": raw, "ema_model_state_dict": ema}
    assert select_evaluation_state_dict(checkpoint, "auto") == (ema, "ema")
    assert select_evaluation_state_dict(checkpoint, "raw") == (raw, "raw")
    assert select_evaluation_state_dict(checkpoint, "ema") == (ema, "ema")

    historical = {"model_state_dict": raw}
    assert select_evaluation_state_dict(historical, "auto") == (raw, "raw")
    assert select_evaluation_state_dict(historical, "raw") == (raw, "raw")
    with pytest.raises(RuntimeError, match="no EMA weights"):
        select_evaluation_state_dict(historical, "ema")
