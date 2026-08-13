from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from scripts.summarize_budget_pilot import summarize_run
from src.causal.config import load_and_resolve_config
from src.causal.runtime import scheduler_warmup_steps


ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = ROOT / "configs" / "causal" / "pilots"
LAUNCHER = ROOT / "scripts" / "run_budget_pilot_4gpu.sh"
PILOTS = {
    "050k": (50_000, 1_500),
    "100k": (100_000, 3_000),
    "200k": (200_000, 6_000),
    "300k": (300_000, 9_000),
}


def _load(label: str) -> dict:
    return load_and_resolve_config(
        PILOT_DIR / f"budget_stride4_chunk_{label}.yaml"
    )


def _without_allowed_differences(config: dict) -> dict:
    value = copy.deepcopy(config)
    value.pop("config_path", None)
    value["experiment"].pop("name")
    value["train"].pop("steps")
    value["checkpoint"].pop("output_dir")
    return value


def test_budget_pilot_configs_resolve_expected_warmup_and_recipe():
    for label, (steps, expected_warmup) in PILOTS.items():
        config = _load(label)
        train = config["train"]
        assert config["experiment"] == {
            "name": f"budget_stride4_chunk_{label}",
            "seed": 42,
        }
        assert config["data"]["manifest_path"] == (
            "/data/x2227/so101_world_model/data/causal/"
            "armnetbench_so101_seed42_manifest.json"
        )
        assert config["temporal"] == {
            "num_frames": 10,
            "num_history": 2,
            "frame_stride": 4,
        }
        assert config["action"]["representation"] == "fast_chunk"
        assert config["action"]["effective_action_dim"] == 24
        assert train["steps"] == steps
        assert train["batch_size"] == 8
        assert train["precision"] == "bf16"
        assert train["lr"] == 1e-4
        assert train["betas"] == [0.9, 0.99]
        assert train["eps"] == 1e-8
        assert train["weight_decay"] == 0.002
        assert train["grad_clip"] == 1.0
        assert train["scheduler"] == {
            "type": "warmup_cosine",
            "warmup_ratio": 0.03,
            "min_lr_ratio": 0.7,
        }
        assert train["ema"] == {"enabled": True, "decay": 0.9995}
        assert train["val_every"] == 10_000
        assert train["val_windows"] == 256
        assert train["num_workers"] == 0
        assert scheduler_warmup_steps(steps, train["scheduler"]["warmup_ratio"]) == (
            expected_warmup
        )
        assert config["checkpoint"]["output_dir"] == (
            f"/data/x2227/experiments/so101_budget_pilot/{label}"
        )


def test_budget_pilot_configs_differ_only_in_name_steps_and_output_dir():
    configs = [_load(label) for label in PILOTS]
    canonical = _without_allowed_differences(configs[0])
    for config in configs[1:]:
        assert _without_allowed_differences(config) == canonical


def test_budget_pilot_launcher_requires_tmux_and_waits_for_every_process():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert '${TMUX:-}' in launcher
    assert 'wait "${pids[$index]}"' in launcher
    assert 'for index in "${!labels[@]}"' in launcher


def test_budget_summary_reports_missing_without_crashing(tmp_path: Path):
    row = summarize_run(tmp_path, "050k", "budget_stride4_chunk_050k")
    assert row["status"] == "missing"
    assert row["best_step"] == "missing"
    assert row["last_step"] == "missing"
    assert row["wall_hours"] == "missing"


def test_budget_summary_reads_values_from_checkpoint(tmp_path: Path):
    run_dir = tmp_path / "050k"
    run_dir.mkdir()
    name = "budget_stride4_chunk_050k"
    config = _load("050k")
    best = {
        "step": 40_000,
        "config": config,
        "metrics": {"best_val_loss": 0.125},
        "scheduler_state_dict": {"warmup_steps": 1_500},
        "git_commit": "abc123",
    }
    last = {
        "step": 50_000,
        "config": config,
        "metrics": {
            "best_val_loss": 0.125,
            "elapsed_wall_seconds": 7_200.0,
        },
        "scheduler_state_dict": {"warmup_steps": 1_500},
        "git_commit": "abc123",
    }
    torch.save(best, run_dir / f"{name}_best.pt")
    torch.save(last, run_dir / f"{name}_last.pt")
    row = summarize_run(tmp_path, "050k", name)
    assert row == {
        "budget": 50_000,
        "best_step": 40_000,
        "best_val": pytest.approx(0.125),
        "last_step": 50_000,
        "wall_hours": pytest.approx(2.0),
        "precision": "bf16",
        "batch": 8,
        "warmup": 1_500,
        "scheduler": "warmup_cosine",
        "ema_decay": pytest.approx(0.9995),
        "git_commit": "abc123",
        "status": "complete",
    }


def test_budget_summary_reports_missing_wall_hours_for_historical_last(tmp_path: Path):
    run_dir = tmp_path / "050k"
    run_dir.mkdir()
    name = "budget_stride4_chunk_050k"
    checkpoint = {
        "step": 50_000,
        "config": _load("050k"),
        "metrics": {"best_val_loss": 0.125},
        "scheduler_state_dict": {"warmup_steps": 1_500},
    }
    torch.save(checkpoint, run_dir / f"{name}_last.pt")
    row = summarize_run(tmp_path, "050k", name)
    assert row["wall_hours"] == "missing"
