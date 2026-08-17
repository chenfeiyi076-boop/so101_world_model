from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.evaluate_rollout_rgb import write_aggregations
from src.causal.action_visualization import preprocess_original_rgb
from src.causal.config import resolve_config
from src.causal.checkpointing import make_checkpoint, save_checkpoint
from src.causal.data.common import ActionStats
from src.causal.long_rollout_eval import (
    FullEpisodePlan,
    aggregate_draws_to_episode_steps,
    aggregate_episode_steps,
    artifact_path,
    build_rgb_artifact_metadata,
    combine_rgb_metrics,
    deterministic_balanced_shards,
    evaluate_stage_b_episode,
    expected_target_rows,
    fixed_cohort_ids,
    file_sha256,
    full_episode_rollout_steps,
    image_metric_batches,
    make_long_rollout_artifact,
    reader_for_source,
    rgb_resume_decision,
    resume_decision,
    structural_ssim_per_image,
    validate_long_rollout_artifact,
    validate_image_metric_identity,
    validate_rgb_metric_artifact,
    vae_provenance,
    validate_aggregate_only_provenance,
)
from src.causal.rollout import autoregressive_causal_rollout
from src.causal.rollout_visualization import decode_cached_latents
from src.causal.rollout_visualization import decoded_tensor_to_uint8
from src.causal.runtime import build_model
from src.so101_cache.vae_cache import (
    LATENT_CONVENTION,
    atomic_torch_save,
    preprocess_rgb_batch,
)


class ZeroVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.inputs = []

    def forward(self, latents, tau, action_cond, action_valid_mask):
        self.inputs.append(latents.detach().clone())
        return torch.zeros_like(latents) + self.anchor * 0


def config():
    return resolve_config({
        "experiment": {"name": "long", "seed": 0},
        "data": {"mode": "multi_episode", "manifest_path": "unused"},
        "temporal": {"num_frames": 10, "num_history": 2, "frame_stride": 4},
        "action": {
            "raw_action_dim": 6, "alignment": "causal",
            "representation": "fast_chunk", "normalize": True,
            "null_condition": "zero_embedding",
        },
        "model": {
            "in_channels": 1, "patch_size": 1, "hidden_size": 8,
            "depth": 1, "num_heads": 1,
        },
        "flow_matching": {"future_only_loss": True},
        "train": {"batch_size": 1, "steps": 1, "lr": 1e-4, "precision": "fp32"},
        "checkpoint": {"output_dir": "unused"},
        "latent": {"convention": LATENT_CONVENTION},
    })


def episode(length=180):
    latents = torch.zeros(length, 1, 12, 12)
    latents[8:] = 999
    return {
        "latents": latents,
        "actions": torch.arange(length * 6).reshape(length, 6).float(),
        "frame_indices": torch.arange(100, 100 + length),
    }


def rollout(steps):
    model = ZeroVelocity()
    result = autoregressive_causal_rollout(
        model=model, config=config(),
        action_stats=ActionStats(torch.zeros(6), torch.ones(6)),
        episode=episode(), episode_id=7, start=0, rollout_steps=steps,
        euler_steps=1, seed=20260, noise_draw=2,
        device=torch.device("cpu"), precision="fp32",
    )
    return model, result


def metadata(steps=4):
    return {
        "artifact_version": 1, "episode_id": 7, "task_id": 3, "draw_id": 2,
        "number_of_steps": steps, "rollout_seed": 20260,
        "checkpoint_path": str(Path("checkpoint.pt").resolve()),
        "checkpoint_sha256": "abc", "checkpoint_step": 300000,
        "weights_used": "ema", "manifest_path": str(Path("manifest.json").resolve()),
        "manifest_sha256": "def", "cache_path": str(Path("episode.pt").resolve()),
        "frame_stride": 4, "num_history": 2, "euler_steps": 10,
        "action_alignment": "causal", "action_representation": "fast_chunk",
        "effective_action_dim": 24, "latent_convention": LATENT_CONVENTION,
        "vae_identifier": "sd3", "scaling_factor": 1.5305,
        "shift_factor_used": False, "reference_used_as_model_input": False,
    }


def test_full_episode_steps_use_earliest_history_and_episode_end():
    assert full_episode_rollout_steps(
        41, num_history=2, frame_stride=4, max_rollout_steps=None
    ) == 9
    assert full_episode_rollout_steps(
        41, num_history=2, frame_stride=4, max_rollout_steps=4
    ) == 4
    assert expected_target_rows(num_history=2, frame_stride=4, rollout_steps=3).tolist() == [8, 12, 16]


def test_long_rollout_prefix_matches_existing_32_step_semantics():
    short_model, short = rollout(32)
    long_model, long = rollout(40)
    assert torch.equal(short["initial_noises"], long["initial_noises"][:32])
    assert torch.equal(short["predicted_future"], long["predicted_future"][:32])
    for key in short["metrics"]:
        assert torch.equal(short["metrics"][key], long["metrics"][key][:32])
    assert all(not torch.any(value == 999) for value in long_model.inputs)
    assert long["reference_used_as_model_input"] is False


def test_artifact_is_bf16_complete_and_resume_is_strict(tmp_path):
    _, result = rollout(4)
    payload = make_long_rollout_artifact(result, metadata())
    validate_long_rollout_artifact(payload, metadata())
    assert payload["pred_latents"].dtype == torch.bfloat16
    path = artifact_path(tmp_path, 7, 2)
    atomic_torch_save(payload, path)
    assert resume_decision(path, metadata(), overwrite=False) == "skip"
    changed = dict(metadata()); changed["euler_steps"] = 11
    with pytest.raises(RuntimeError, match="metadata mismatch"):
        resume_decision(path, changed, overwrite=False)
    path.with_name(path.name + ".tmp").write_bytes(b"partial")
    assert resume_decision(artifact_path(tmp_path, 8, 0), metadata(), overwrite=False) == "run"


def test_balanced_sharding_has_no_duplicate_or_missing_episodes():
    plans = [FullEpisodePlan(i, None, Path(f"{i}.pt"), steps) for i, steps in enumerate((3, 8, 2, 7, 4, 9, 1))]
    shards = deterministic_balanced_shards(plans, 3)
    assigned = [item.episode_id for shard in shards for item in shard]
    assert sorted(assigned) == list(range(len(plans)))
    assert len(assigned) == len(set(assigned))
    assert deterministic_balanced_shards(plans, 3) == shards


def metric_row(draw, mse, pred_ssim, pred_lpips):
    return {
        "episode_id": 1, "task_id": 0, "draw_id": draw, "step": 1,
        "time_sec": 0.2, "raw_frame_index": 8, "sampled_frame_index": 8,
        "latent_mse": mse, "latent_rmse": mse**0.5, "latent_mae": mse,
        "relative_l2": mse, "cosine": 0.5, "pred_ssim": pred_ssim,
        "recon_ssim": 0.9, "ssim_gap": 0.9 - pred_ssim,
        "pred_lpips": pred_lpips, "recon_lpips": 0.1,
        "lpips_gap": pred_lpips - 0.1,
    }


def test_draws_are_aggregated_as_metrics_not_ensemble_latents():
    # Two predictions -1 and +1 around GT=0 each have MSE 1. Their mean latent
    # has MSE 0, which is deliberately NOT the reported stochastic metric.
    rows = [metric_row(0, 1.0, 0.7, 0.3), metric_row(1, 1.0, 0.5, 0.5)]
    aggregated = aggregate_draws_to_episode_steps(rows, expected_draws=2)[0]
    assert aggregated["mean_latent_mse"] == 1.0
    assert aggregated["mean_pred_ssim"] == pytest.approx(0.6)
    assert aggregated["mean_pred_lpips"] == pytest.approx(0.4)


def test_ssim_lpips_sanity_and_gap_sign():
    image = torch.rand(2, 3, 16, 16)
    assert torch.all(structural_ssim_per_image(image, image) > 0.9999)
    fake_lpips = lambda left, right: (left - right).square().flatten(1).mean(1)
    ssim, lpips = image_metric_batches(
        image, image, lpips_metric=fake_lpips, batch_size=1
    )
    assert torch.all(ssim > 0.9999)
    assert torch.equal(lpips, torch.zeros(2))
    validate_image_metric_identity(fake_lpips)
    base = [{
        "episode_id": 1, "task_id": 0, "draw_id": 0, "step": 1,
        "time_sec": 0.2, "raw_frame_index": 8, "sampled_frame_index": 8,
        "latent_mse": 0, "latent_rmse": 0, "latent_mae": 0,
        "relative_l2": 0, "cosine": 1,
    }]
    combined = combine_rgb_metrics(
        base, pred_ssim=[0.8], recon_ssim=[0.9],
        pred_lpips=[0.3], recon_lpips=[0.1],
    )[0]
    assert combined["ssim_gap"] == pytest.approx(0.1)
    assert combined["lpips_gap"] == pytest.approx(0.2)


def test_available_and_fixed_cohort_aggregation_use_physical_episodes():
    rows = []
    for episode_id, maximum in ((1, 3), (2, 2)):
        for step in range(1, maximum + 1):
            rows.append({
                **aggregate_draws_to_episode_steps(
                    [metric_row(0, float(episode_id), 0.8, 0.2)], expected_draws=1
                )[0],
                "episode_id": episode_id, "step": step, "time_sec": step * 0.2,
            })
    available = aggregate_episode_steps(rows, bootstrap_samples=50, bootstrap_seed=1)
    assert [row["n_episodes"] for row in available] == [2, 2, 1]
    assert fixed_cohort_ids(rows, 3) == {1}
    fixed = aggregate_episode_steps(
        rows, cohort_episode_ids={1}, max_step=3,
        bootstrap_samples=50, bootstrap_seed=1,
    )
    assert [row["n_episodes"] for row in fixed] == [1, 1, 1]


class FakeVAE:
    class Config:
        scaling_factor = 2.0
    config = Config()
    def parameters(self):
        return iter(())
    def decode(self, value):
        class Output: pass
        output = Output(); output.sample = value[:, :3]
        return output


def test_gt_reconstruction_uses_existing_no_shift_decode_convention():
    latents = torch.full((1, 3, 12, 12), 2.0)
    decoded = decode_cached_latents(
        FakeVAE(), latents, device=torch.device("cpu"), batch_size=1
    )
    # z_cache / scaling = 1 -> image conversion maps [-1,1] value 1 to 255.
    assert torch.equal(decoded, torch.full_like(decoded, 255))


def test_raw_and_cached_target_frame_mapping_is_exact():
    cache_frame_indices = torch.arange(100, 141)
    rows = expected_target_rows(num_history=2, frame_stride=4, rollout_steps=4)
    assert rows.tolist() == [8, 12, 16, 20]
    assert cache_frame_indices[rows].tolist() == [108, 112, 116, 120]


def test_rgb_resume_artifact_requires_complete_draw_step_grid():
    expected = {"episode_id": 4, "number_of_steps": 2, "noise_draws": 2}
    rows = []
    for draw in range(2):
        for step in range(1, 3):
            row = {field: 0.0 for field in (
                "time_sec", "latent_mse", "latent_rmse", "latent_mae",
                "relative_l2", "cosine", "pred_ssim", "recon_ssim",
                "ssim_gap", "pred_lpips", "recon_lpips", "lpips_gap",
            )}
            row.update({
                "episode_id": 4, "task_id": 1, "draw_id": draw,
                "step": step, "raw_frame_index": step * 4,
                "sampled_frame_index": step * 4,
            })
            rows.append(row)
    validate_rgb_metric_artifact({"metadata": expected, "rows": rows}, expected)
    with pytest.raises(RuntimeError, match="row count mismatch"):
        validate_rgb_metric_artifact({"metadata": expected, "rows": rows[:-1]}, expected)


def test_stage_a_cli_tiny_smoke_two_episodes_two_draws_four_steps(tmp_path):
    cache_paths = []
    for episode_id in (2, 3):
        path = tmp_path / f"episode_{episode_id}.pt"
        torch.save({
            "latents": torch.zeros(24, 1, 12, 12),
            "actions": torch.arange(24 * 6).reshape(24, 6).float(),
            "frame_indices": torch.arange(24),
            "episode_index": episode_id,
            "metadata": {
                "latent_convention": LATENT_CONVENTION,
                "shift_factor_used": False,
                "vae_identifier": "fake-sd3",
                "scaling_factor": 1.5,
            },
        }, path)
        cache_paths.append(path)
    manifest_path = tmp_path / "manifest.json"
    entries = [
        {"episode_index": 0, "cache_file": str(tmp_path / "unused0.pt"), "task_index": 0},
        {"episode_index": 1, "cache_file": str(tmp_path / "unused1.pt"), "task_index": 0},
        {"episode_index": 2, "cache_file": str(cache_paths[0]), "task_index": 1},
        {"episode_index": 3, "cache_file": str(cache_paths[1]), "task_index": 1},
    ]
    manifest_path.write_text(json.dumps({
        "train_episode_ids": [0], "val_episode_ids": [1],
        "test_episode_ids": [2, 3], "episodes": entries,
        "dataset": "tiny", "dataset_root": str(tmp_path),
    }), encoding="utf-8")
    resolved = config()
    resolved["data"]["manifest_path"] = str(manifest_path)
    stats = ActionStats(torch.zeros(6), torch.ones(6), source="tiny")
    model = build_model(resolved)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    checkpoint = make_checkpoint(
        model=model, optimizer=optimizer, step=1, config=resolved,
        action_stats=stats, data_info={"tiny": True}, best_val_loss=None,
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint_path, checkpoint)
    output = tmp_path / "output"
    subprocess.run([
        sys.executable, "scripts/evaluate_long_rollout.py",
        "--checkpoint", str(checkpoint_path), "--output-dir", str(output),
        "--split", "test", "--noise-draws", "2", "--seed", "17",
        "--euler-steps", "1", "--max-rollout-steps", "4",
        "--max-episodes", "2", "--device", "cpu", "--weights", "raw",
    ], cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True, text=True)
    artifacts = sorted((output / "latents").glob("*.pt"))
    assert len(artifacts) == 4
    with (output / "per_draw_step_latent.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 2 * 2 * 4
    summary = json.loads((output / "summary.json").read_text())
    assert summary["stage_a_completion_status"] == "complete"
    assert summary["actual_total_stochastic_trajectories"] == 4
    assert summary["actual_total_prediction_steps"] == 16


def stage_a_summary_for_rgb():
    return {
        "noise_draws": 2,
        "checkpoint_sha256": "checkpoint-sha",
        "manifest_sha256": "manifest-sha",
        "weights_used": "ema",
        "seed": 20260,
        "euler_steps": 10,
        "frame_stride": 4,
        "num_history": 2,
        "latent_convention": LATENT_CONVENTION,
        "number_of_episodes": 2,
        "actual_total_stochastic_trajectories": 4,
        "actual_total_prediction_steps": 16,
    }


def rgb_rows(episode_id=4, steps=2, draws=2):
    rows = []
    for draw in range(draws):
        for step in range(1, steps + 1):
            rows.append({
                "episode_id": episode_id, "task_id": 1, "draw_id": draw,
                "step": step, "time_sec": step * 0.2,
                "raw_frame_index": step * 4, "sampled_frame_index": step * 4,
                "latent_mse": float(draw + step), "latent_rmse": 1.0,
                "latent_mae": 1.0, "relative_l2": 1.0, "cosine": 0.5,
                "pred_ssim": 0.7, "recon_ssim": 0.9, "ssim_gap": 0.2,
                "pred_lpips": 0.3, "recon_lpips": 0.1, "lpips_gap": 0.2,
            })
    return rows


def test_stage_b_resume_binds_every_stage_a_draw_sha_and_full_provenance(tmp_path):
    draw_paths = []
    for draw, content in enumerate((b"draw-zero", b"draw-one")):
        path = tmp_path / f"draw_{draw}.pt"
        path.write_bytes(content)
        draw_paths.append(path)
    hashes = {str(i): file_sha256(path) for i, path in enumerate(draw_paths)}
    summary = stage_a_summary_for_rgb()
    expected = build_rgb_artifact_metadata(
        episode_id=4, number_of_steps=2, stage_a_summary=summary,
        stage_a_draw_artifact_sha256=hashes, vae_config_sha256="vae-sha",
        vae_weights_sha256="weights-sha",
        vae_weight_files_sha256={"model.safetensors": "file-sha"},
        vae_artifact_sha256="artifact-sha",
        vae_path=tmp_path / "vae", scaling_factor=1.5,
    )
    output = tmp_path / "episode_0004.pt"
    atomic_torch_save({"metadata": expected, "rows": rgb_rows()}, output)
    assert rgb_resume_decision(output, expected, overwrite=False) == "skip"

    for field, value in (("euler_steps", 11), ("weights_used", "raw"), ("seed", 9)):
        changed_summary = dict(summary); changed_summary[field] = value
        changed = build_rgb_artifact_metadata(
            episode_id=4, number_of_steps=2, stage_a_summary=changed_summary,
            stage_a_draw_artifact_sha256=hashes, vae_config_sha256="vae-sha",
            vae_weights_sha256="weights-sha",
            vae_weight_files_sha256={"model.safetensors": "file-sha"},
            vae_artifact_sha256="artifact-sha",
            vae_path=tmp_path / "vae", scaling_factor=1.5,
        )
        with pytest.raises(RuntimeError, match="metadata mismatch"):
            rgb_resume_decision(output, changed, overwrite=False)

    draw_paths[1].write_bytes(b"changed Stage A prediction artifact")
    changed_hashes = {str(i): file_sha256(path) for i, path in enumerate(draw_paths)}
    assert changed_hashes["1"] != hashes["1"]
    changed = build_rgb_artifact_metadata(
        episode_id=4, number_of_steps=2, stage_a_summary=summary,
        stage_a_draw_artifact_sha256=changed_hashes,
        vae_config_sha256="vae-sha", vae_weights_sha256="weights-sha",
        vae_weight_files_sha256={"model.safetensors": "file-sha"},
        vae_artifact_sha256="artifact-sha", vae_path=tmp_path / "vae",
        scaling_factor=1.5,
    )
    with pytest.raises(RuntimeError, match="metadata mismatch"):
        rgb_resume_decision(output, changed, overwrite=False)
    assert rgb_resume_decision(output, changed, overwrite=True) == "run"


def test_aggregate_only_requires_same_stage_a_experiment(tmp_path):
    stage_a = stage_a_summary_for_rgb()
    summary_sha = "stage-a-summary-sha"
    existing = {
        **stage_a,
        "stage_a_long_rollout_dir": str(tmp_path.resolve()),
        "stage_a_summary_sha256": summary_sha,
        "stage_b_completion_status": "complete",
    }
    validate_aggregate_only_provenance(
        stage_a_summary=stage_a, existing_stage_b_summary=existing,
        long_rollout_dir=tmp_path, stage_a_summary_sha256=summary_sha,
    )
    replacements = {
        "checkpoint_sha256": "other-checkpoint",
        "manifest_sha256": "other-manifest",
        "weights_used": "raw",
        "seed": 9,
        "euler_steps": 11,
        "frame_stride": 2,
        "num_history": 3,
        "noise_draws": 4,
    }
    for field, replacement in replacements.items():
        changed = dict(stage_a); changed[field] = replacement
        with pytest.raises(RuntimeError, match="aggregate-only"):
            validate_aggregate_only_provenance(
                stage_a_summary=changed, existing_stage_b_summary=existing,
                long_rollout_dir=tmp_path, stage_a_summary_sha256=summary_sha,
            )
    with pytest.raises(RuntimeError, match="aggregate-only"):
        validate_aggregate_only_provenance(
            stage_a_summary=stage_a, existing_stage_b_summary=existing,
            long_rollout_dir=tmp_path / "different_stage_a",
            stage_a_summary_sha256=summary_sha,
        )
    with pytest.raises(RuntimeError, match="aggregate-only"):
        validate_aggregate_only_provenance(
            stage_a_summary=stage_a, existing_stage_b_summary=existing,
            long_rollout_dir=tmp_path, stage_a_summary_sha256="changed-summary",
        )


def test_vae_provenance_hashes_actual_weights_and_invalidates_resume(tmp_path):
    vae_dir = tmp_path / "vae"
    vae_dir.mkdir()
    (vae_dir / "config.json").write_text('{"scaling_factor": 1.5}', encoding="utf-8")
    weight = vae_dir / "diffusion_pytorch_model.safetensors"
    weight.write_bytes(b"first VAE weights")
    first = vae_provenance(vae_dir)
    assert first["vae_weight_files_sha256"] == {
        "diffusion_pytorch_model.safetensors": file_sha256(weight)
    }
    summary = stage_a_summary_for_rgb()
    draw_hashes = {"0": "draw0", "1": "draw1"}
    first_metadata = build_rgb_artifact_metadata(
        episode_id=4, number_of_steps=2, stage_a_summary=summary,
        stage_a_draw_artifact_sha256=draw_hashes,
        vae_path=vae_dir, scaling_factor=1.5, **first,
    )
    output = tmp_path / "episode.pt"
    atomic_torch_save(
        {"metadata": first_metadata, "rows": rgb_rows()}, output
    )
    assert rgb_resume_decision(output, first_metadata, overwrite=False) == "skip"

    weight.write_bytes(b"different VAE weights")
    second = vae_provenance(vae_dir)
    assert second["vae_weights_sha256"] != first["vae_weights_sha256"]
    assert second["vae_artifact_sha256"] != first["vae_artifact_sha256"]
    second_metadata = build_rgb_artifact_metadata(
        episode_id=4, number_of_steps=2, stage_a_summary=summary,
        stage_a_draw_artifact_sha256=draw_hashes,
        vae_path=vae_dir, scaling_factor=1.5, **second,
    )
    with pytest.raises(RuntimeError, match="metadata mismatch"):
        rgb_resume_decision(output, second_metadata, overwrite=False)


def test_raw_reference_preprocessing_exactly_matches_cache_input_preprocessing():
    height, width = 137, 311
    image = (
        np.arange(height * width * 3, dtype=np.uint32)
        .reshape(height, width, 3)
        .astype(np.uint8)
    )
    stage_b = preprocess_original_rgb([image])
    cache_input = preprocess_rgb_batch([image])
    cache_roundtrip_uint8 = decoded_tensor_to_uint8(cache_input)
    assert stage_b.shape == (1, 256, 256, 3)
    assert stage_b.dtype == torch.uint8
    assert torch.equal(stage_b, cache_roundtrip_uint8)


def test_reader_cache_supports_multiple_dataset_sources_and_reuses_each():
    created = []
    def factory(root, camera):
        value = (root, camera, len(created))
        created.append(value)
        return value
    cache = {}
    first = reader_for_source(
        cache, source_path="dataset_a", camera="front", factory=factory
    )
    second = reader_for_source(
        cache, source_path="dataset_b", camera="front", factory=factory
    )
    first_again = reader_for_source(
        cache, source_path="dataset_a", camera="front", factory=factory
    )
    assert first is first_again
    assert first != second
    assert len(created) == 2


def synthetic_stage_a_payload(draw_id: int, prediction: float, steps: int = 2):
    metadata = {
        "episode_id": 4, "task_id": 1, "draw_id": draw_id,
        "number_of_steps": steps, "frame_stride": 4,
    }
    return {
        "metadata": metadata,
        "pred_latents": torch.full((steps, 1, 2, 2), prediction),
        "sampled_frame_indices": torch.tensor([8, 12]),
        "raw_frame_indices": torch.tensor([108, 112]),
        "metrics": {
            "mse": torch.full((steps,), prediction**2),
            "rmse": torch.full((steps,), abs(prediction)),
            "mae": torch.full((steps,), abs(prediction)),
            "relative_l2": torch.full((steps,), abs(prediction)),
            "cosine_similarity": torch.ones(steps),
        },
    }


def synthetic_decode(_vae, latents, *, device, batch_size):
    del device, batch_size
    values = torch.as_tensor(latents).float().flatten(1).mean(1)
    pixels = values.add(1).mul(127.5).round().clamp(0, 255).to(torch.uint8)
    return pixels[:, None, None, None].expand(-1, 256, 256, 3).contiguous()


def synthetic_lpips(left, right):
    return (left - right).square().flatten(1).mean(1)


def test_stage_b_synthetic_end_to_end_and_streaming_matches_old_batch_math(tmp_path):
    payloads = [
        synthetic_stage_a_payload(0, 0.25),
        synthetic_stage_a_payload(1, 0.50),
    ]
    raw = torch.full((2, 256, 256, 3), 128, dtype=torch.uint8)
    gt_latents = torch.zeros(2, 1, 2, 2)
    rows = evaluate_stage_b_episode(
        draw_payloads=iter(payloads), raw_gt_uint8=raw,
        gt_latents=gt_latents, vae=object(), decode_fn=synthetic_decode,
        lpips_metric=synthetic_lpips, device=torch.device("cpu"),
        decode_batch_size=2, metric_batch_size=1, expected_draws=2,
    )
    assert [(row["draw_id"], row["step"]) for row in rows] == [
        (0, 1), (0, 2), (1, 1), (1, 2)
    ]
    assert all(row["recon_ssim"] == rows[0]["recon_ssim"] for row in rows)
    assert all(row["recon_lpips"] == rows[0]["recon_lpips"] for row in rows)
    assert all(row["ssim_gap"] > 0 for row in rows)
    assert all(row["lpips_gap"] > 0 for row in rows)

    # Reference the previous batched math without using mean latent/RGB.
    raw_float = raw.permute(0, 3, 1, 2).float() / 255
    pred_all = torch.cat([
        synthetic_decode(object(), payload["pred_latents"], device=torch.device("cpu"), batch_size=2)
        for payload in payloads
    ]).permute(0, 3, 1, 2).float() / 255
    old_ssim, old_lpips = image_metric_batches(
        pred_all, raw_float.repeat(2, 1, 1, 1),
        lpips_metric=synthetic_lpips, batch_size=2, device=torch.device("cpu"),
    )
    assert [row["pred_ssim"] for row in rows] == pytest.approx(old_ssim.tolist())
    assert [row["pred_lpips"] for row in rows] == pytest.approx(old_lpips.tolist())

    episode_artifact = tmp_path / "episode_metrics" / "episode_0004.pt"
    expected_metadata = {"episode_id": 4, "number_of_steps": 2, "noise_draws": 2}
    atomic_torch_save({"metadata": expected_metadata, "rows": rows}, episode_artifact)
    validate_rgb_metric_artifact(
        torch.load(episode_artifact, weights_only=False), expected_metadata
    )
    plot_calls = []
    def fake_plots(output_dir, available, fixed):
        plot_calls.append((Path(output_dir), len(available), sorted(fixed)))
    episode_rows, available, fixed, sizes = write_aggregations(
        output_dir=tmp_path, per_draw_rows=rows, noise_draws=2,
        fixed_horizons=[1, 2], bootstrap_samples=20, bootstrap_seed=3,
        plot_fn=fake_plots,
    )
    assert len(episode_rows) == 2
    assert len(available) == 2
    assert sizes == {1: 1, 2: 1}
    assert sorted(fixed) == [1, 2]
    assert plot_calls == [(tmp_path, 2, [1, 2])]
    for name in (
        "per_draw_step.csv", "per_episode_step.csv", "per_step_available.csv",
        "per_step_fixed_H001.csv", "per_step_fixed_H002.csv",
    ):
        assert (tmp_path / name).is_file()


def test_stage_b_shape_alignment_is_hard_requirement():
    payload = synthetic_stage_a_payload(0, 0.25)
    wrong_raw = torch.zeros(2, 255, 256, 3, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="raw GT must be uint8"):
        evaluate_stage_b_episode(
            draw_payloads=[payload], raw_gt_uint8=wrong_raw,
            gt_latents=torch.zeros(2, 1, 2, 2), vae=object(),
            decode_fn=synthetic_decode, lpips_metric=synthetic_lpips,
            device=torch.device("cpu"), decode_batch_size=2,
            metric_batch_size=1, expected_draws=1,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_gpu_ssim_matches_cpu_ssim():
    images = torch.rand(3, 3, 32, 32)
    references = torch.rand(3, 3, 32, 32)
    cpu = structural_ssim_per_image(images, references)
    gpu = structural_ssim_per_image(images.cuda(), references.cuda()).cpu()
    assert torch.allclose(cpu, gpu, rtol=1e-4, atol=1e-5)
