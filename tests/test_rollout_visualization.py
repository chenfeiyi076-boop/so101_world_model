from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.causal.config import resolve_config
from src.causal.data.common import ActionStats
from src.causal.rollout import autoregressive_causal_rollout, rollout_metric_rows
from src.causal.rollout_visualization import (
    CONTACT_HEADER_HEIGHT,
    CONTACT_LEFT_MARGIN,
    REFERENCE_LABEL,
    build_stream_metadata,
    create_contact_sheet,
    decode_cached_latents,
    decoded_tensor_to_uint8,
    future_time_seconds,
    group_stream_rows,
    normalized_display_steps,
    rerun_rollout_stream,
    resolve_vae_directory,
    score_rollout_streams,
    select_quantile_representatives,
    validate_summary_checkpoint_compatibility,
    validate_rerun,
)
from src.so101_cache.vae_cache import LATENT_CONVENTION


class FakeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            scaling_factor=2.0,
            shift_factor=0.0609,
            latent_channels=16,
        )
        self.decode_inputs = []

    def decode(self, value: torch.Tensor):
        self.decode_inputs.append(value.detach().cpu().clone())
        batch = len(value)
        sample = torch.zeros(batch, 3, 2, 2, device=value.device, dtype=value.dtype)
        return SimpleNamespace(sample=sample + self.anchor * 0)


class ZeroVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, latents, tau, action_cond, action_valid_mask):
        return torch.zeros_like(latents) + self.anchor * 0


def _config() -> dict:
    return resolve_config(
        {
            "experiment": {"name": "visualization_test", "seed": 0},
            "data": {"mode": "multi_episode", "manifest_path": "unused.json"},
            "temporal": {
                "num_frames": 10,
                "num_history": 2,
                "frame_stride": 4,
            },
            "action": {
                "raw_action_dim": 6,
                "alignment": "causal",
                "representation": "fast_chunk",
                "normalize": True,
                "null_condition": "zero_embedding",
            },
            "model": {
                "in_channels": 1,
                "patch_size": 1,
                "hidden_size": 8,
                "depth": 1,
                "num_heads": 1,
            },
            "flow_matching": {"future_only_loss": True},
            "train": {
                "batch_size": 1,
                "steps": 1,
                "lr": 1e-4,
                "precision": "fp32",
            },
            "checkpoint": {"output_dir": "unused"},
            "latent": {"convention": LATENT_CONVENTION},
        }
    )


def _episode() -> dict:
    length = 32
    return {
        "episode_index": 6,
        "latents": torch.zeros(length, 1, 2, 2),
        "actions": torch.arange(length * 6, dtype=torch.float32).reshape(length, 6),
        "frame_indices": torch.arange(length),
    }


def _direct_rollout():
    arguments = {
        "model": ZeroVelocity(),
        "config": _config(),
        "action_stats": ActionStats(torch.zeros(6), torch.ones(6)),
        "episode": _episode(),
        "episode_id": 6,
        "start": 0,
        "noise_draw": 1,
        "rollout_steps": 2,
        "euler_steps": 1,
        "seed": 20260,
        "device": torch.device("cpu"),
        "precision": "fp32",
    }
    return arguments, autoregressive_causal_rollout(**arguments)


def _score_rows():
    return [
        {
            "episode_id": index,
            "start": 0,
            "noise_draw": 0,
            "step": 1,
            "target_frame_index": 8,
            "mse": index / 10,
            "rmse": math.sqrt(index / 10),
            "mae": index / 10,
            "relative_l2": index / 20,
            "cosine_similarity": 1.0 - index / 100,
        }
        for index in range(1, 11)
    ]


def _compatibility_summary(config: dict) -> dict:
    return {
        "checkpoint_step": 200000,
        "precision": config["train"]["precision"],
        "weights_used": "ema",
        "num_frames": config["temporal"]["num_frames"],
        "num_history": config["temporal"]["num_history"],
        "frame_stride": config["temporal"]["frame_stride"],
        "action_representation": config["action"]["representation"],
        "effective_action_dim": config["action"]["effective_action_dim"],
    }


def test_decode_divides_scaling_factor_and_never_applies_shift():
    vae = FakeVAE()
    latents = torch.tensor([4.0, 6.0, 8.0]).reshape(3, 1, 1, 1)
    decoded = decode_cached_latents(
        vae, latents, device=torch.device("cpu"), batch_size=2
    )
    assert torch.equal(
        torch.cat(vae.decode_inputs).flatten(), torch.tensor([2.0, 3.0, 4.0])
    )
    assert decoded.shape == (3, 2, 2, 3)


def test_image_conversion_uses_fixed_minus_one_to_one_mapping():
    decoded = torch.tensor([-1.0, 0.0, 1.0]).reshape(1, 3, 1, 1)
    image = decoded_tensor_to_uint8(decoded)
    assert image[0, 0, 0].tolist() == [0, 128, 255]


def test_quantile_representative_selection_is_deterministic_and_unique():
    rows = _score_rows()
    first, distribution = select_quantile_representatives(
        rows, metric="mean_mse", quantiles=(0.1, 0.5, 0.9)
    )
    second, _ = select_quantile_representatives(
        list(reversed(rows)), metric="mean_mse", quantiles=(0.1, 0.5, 0.9)
    )
    assert first == second
    assert [item["label"] for item in first] == ["q10", "q50", "q90"]
    assert [item["episode_id"] for item in first] == [2, 5, 9]
    assert len({(item["episode_id"], item["start"], item["noise_draw"]) for item in first}) == 3
    assert distribution["count"] == 10


def test_duplicate_quantile_output_labels_fail():
    with pytest.raises(ValueError, match="duplicate output labels"):
        select_quantile_representatives(
            _score_rows(), metric="mean_mse", quantiles=(0.101, 0.104)
        )


def test_checkpoint_latent_convention_mismatch_fails():
    config = _config()
    summary = _compatibility_summary(config)
    checkpoint = {"step": 200000, "config": config}
    checkpoint["config"]["latent"]["convention"] = "shifted_latent"
    with pytest.raises(RuntimeError, match="latent convention"):
        validate_summary_checkpoint_compatibility(
            summary, checkpoint, precision="fp32", weights_used="ema"
        )


@pytest.mark.parametrize(
    ("field", "mismatched_value"),
    (
        ("num_frames", 11),
        ("num_history", 3),
        ("frame_stride", 2),
        ("action_representation", "single"),
        ("effective_action_dim", 12),
    ),
)
def test_summary_temporal_and_action_config_mismatch_fails(
    field, mismatched_value
):
    config = _config()
    summary = _compatibility_summary(config)
    summary[field] = mismatched_value
    checkpoint = {"step": 200000, "config": config}
    with pytest.raises(RuntimeError, match=field):
        validate_summary_checkpoint_compatibility(
            summary, checkpoint, precision="fp32", weights_used="ema"
        )


def test_stream_grouping_uses_full_stochastic_identity():
    rows = _score_rows()[:1]
    alternate = dict(rows[0], noise_draw=1, mse=0.9)
    grouped = group_stream_rows(rows + [alternate])
    scores = score_rollout_streams(rows + [alternate], metric="mean_mse")
    assert set(grouped) == {(1, 0, 0), (1, 0, 1)}
    assert scores[(1, 0, 0)] == pytest.approx(0.1)
    assert scores[(1, 0, 1)] == pytest.approx(0.9)


@pytest.mark.parametrize(
    ("rollout_steps", "expected"),
    (
        (32, [1, 4, 8, 16, 32]),
        (20, [1, 4, 8, 16, 20]),
        (8, [1, 4, 8]),
    ),
)
def test_display_steps_are_filtered_sorted_unique_and_include_final(
    rollout_steps, expected
):
    assert normalized_display_steps((16, 1, 4, 8, 32, 4), rollout_steps) == expected


def test_time_labels_are_relative_to_last_history_frame():
    assert future_time_seconds(1, 4, 20) == pytest.approx(0.2)
    assert future_time_seconds(8, 4, 20) == pytest.approx(1.6)
    assert future_time_seconds(32, 4, 20) == pytest.approx(6.4)


def test_visualization_rerun_preserves_identity_targets_noise_and_metrics():
    arguments, direct = _direct_rollout()
    rerun = rerun_rollout_stream(**arguments)
    assert (rerun["episode_id"], rerun["start"], rerun["noise_draw"]) == (6, 0, 1)
    assert torch.equal(rerun["target_frame_indices"], direct["target_frame_indices"])
    assert torch.equal(rerun["initial_noises"], direct["initial_noises"])
    source_rows = rollout_metric_rows(direct)
    assert validate_rerun(rerun, source_rows) == []


def test_metadata_records_frozen_no_shift_reference_semantics(tmp_path: Path):
    _, result = _direct_rollout()
    vae = FakeVAE()
    metadata = build_stream_metadata(
        selection={
            "label": "q50",
            "quantile": 0.5,
            "actual_score": 0.25,
        },
        summary={
            "selection_metric": "mean_mse",
            "checkpoint_path": "checkpoint.pt",
            "checkpoint_step": 200000,
            "weights_used": "ema",
            "rollout_steps": 2,
            "euler_steps": 10,
            "frame_stride": 4,
            "seed": 20260,
        },
        result=result,
        vae_path=tmp_path,
        vae=vae,
        fps=20,
    )
    assert metadata["latent_convention"] == LATENT_CONVENTION
    assert metadata["vae_scaling_factor"] == 2.0
    assert metadata["vae_shift_factor_from_config"] == pytest.approx(0.0609)
    assert metadata["vae_shift_applied"] is False
    assert metadata["reference_label"] == REFERENCE_LABEL


def test_contact_sheet_has_stable_gt_top_prediction_bottom_layout():
    history = torch.full((2, 5, 4, 3), 100, dtype=torch.uint8)
    gt = torch.zeros(8, 5, 4, 3, dtype=torch.uint8)
    gt[..., 0] = 255
    predicted = torch.zeros_like(gt)
    predicted[..., 2] = 255
    sheet = create_contact_sheet(
        history_images=history,
        predicted_images=predicted,
        gt_reconstruction_images=gt,
        display_steps=(1, 4, 8),
        frame_stride=4,
        fps=20,
    )
    assert sheet.mode == "RGB"
    assert sheet.size == (CONTACT_LEFT_MARGIN + 5 * 4, CONTACT_HEADER_HEIGHT + 10)
    first_future_x = CONTACT_LEFT_MARGIN + 2 * 4
    assert sheet.getpixel((first_future_x, CONTACT_HEADER_HEIGHT)) == (255, 0, 0)
    assert sheet.getpixel((first_future_x, CONTACT_HEADER_HEIGHT + 5)) == (0, 0, 255)


def test_contact_sheet_labels_history_as_shared_input(monkeypatch):
    from PIL import ImageDraw

    labels = []
    original_draw = ImageDraw.Draw

    class RecordingDraw:
        def __init__(self, image):
            self.delegate = original_draw(image)

        def text(self, position, value, *args, **kwargs):
            labels.append(value)
            return self.delegate.text(position, value, *args, **kwargs)

    monkeypatch.setattr(ImageDraw, "Draw", RecordingDraw)
    create_contact_sheet(
        history_images=torch.zeros(2, 5, 4, 3, dtype=torch.uint8),
        predicted_images=torch.zeros(1, 5, 4, 3, dtype=torch.uint8),
        gt_reconstruction_images=torch.zeros(1, 5, 4, 3, dtype=torch.uint8),
        display_steps=(1,),
        frame_stride=4,
    )
    assert "Shared History 0" in labels
    assert "Shared History 1" in labels
    assert "Future: Prediction" in labels
    assert "History 0" not in labels
    assert "History 1" not in labels


def test_vae_resolver_supports_full_repo_and_direct_vae_directory(tmp_path: Path):
    full_repo = tmp_path / "full_repo"
    nested_vae = full_repo / "vae"
    nested_vae.mkdir(parents=True)
    (nested_vae / "config.json").write_text("{}", encoding="utf-8")
    assert resolve_vae_directory(full_repo) == nested_vae

    direct_vae = tmp_path / "direct_vae"
    direct_vae.mkdir()
    (direct_vae / "config.json").write_text("{}", encoding="utf-8")
    assert resolve_vae_directory(direct_vae) == direct_vae
