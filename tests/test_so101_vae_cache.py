from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.verify_so101_latent_cache import decode_representative_latents
from src.causal.datasets import _single_cache_path
from src.so101_cache.vae_cache import (
    CAMERA_KEY,
    FPS,
    LATENT_CONVENTION,
    atomic_torch_save,
    episode_cache_path,
    episode_ids_for_shard,
    preprocess_rgb_batch,
    preprocess_rgb_image,
    requested_episode_ids,
    resolve_vae_dtype,
    should_skip_cache,
    temporary_cache_path,
    validate_cache_payload,
    validate_temporal_axis,
    validate_video_table_alignment,
)


def synthetic_cache(num_frames: int = 5) -> dict:
    return {
        "latents": torch.zeros(num_frames, 16, 32, 32, dtype=torch.bfloat16),
        "actions": torch.zeros(num_frames, 6, dtype=torch.float32),
        "states": torch.zeros(num_frames, 6, dtype=torch.float32),
        "timestamps": torch.arange(num_frames, dtype=torch.float64) / FPS,
        "frame_indices": torch.arange(num_frames, dtype=torch.long),
        "episode_index": 7,
        "metadata": {
            "fps": FPS,
            "camera": CAMERA_KEY,
            "latent_convention": LATENT_CONVENTION,
        },
    }


def test_episode_range_is_start_inclusive_end_exclusive():
    assert requested_episode_ids(0, 2) == [0, 1]
    assert requested_episode_ids(7, 10) == [7, 8, 9]


def test_four_way_sharding_is_complete_and_disjoint():
    requested = requested_episode_ids(0, 2499)
    shard_sets = [
        set(episode_ids_for_shard(requested, shard_id, 4))
        for shard_id in range(4)
    ]
    assert set.union(*shard_sets) == set(requested)
    for left in range(4):
        for right in range(left + 1, 4):
            assert shard_sets[left].isdisjoint(shard_sets[right])


def test_cache_filename_matches_causal_v2_single_episode_resolver(tmp_path: Path):
    cache_path = episode_cache_path(tmp_path, 7)
    causal_path = _single_cache_path(
        {
            "data": {
                "cache_root": str(tmp_path),
                "episode_id": 7,
            }
        }
    )
    assert cache_path.name == "episode_007.pt"
    assert cache_path == causal_path
    assert episode_cache_path(tmp_path, 2498).name == "episode_2498.pt"


def test_vae_dtype_policy_uses_fp32_on_cpu_and_bf16_on_cuda():
    assert resolve_vae_dtype(torch.device("cpu")) == torch.float32
    assert resolve_vae_dtype(torch.device("cuda")) == torch.bfloat16


def test_image_preprocess_shape_range_and_dtype():
    image = np.zeros((576, 1024, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    output = preprocess_rgb_batch([image])
    single_output = preprocess_rgb_image(image)
    assert output.shape == (1, 3, 256, 256)
    assert single_output.shape == (3, 256, 256)
    assert torch.equal(single_output, output[0])
    assert output.dtype == torch.float32
    assert float(output.min()) >= -1.0
    assert float(output.max()) <= 1.0
    assert torch.all(output[:, 0] == 1.0)
    assert torch.all(output[:, 1:] == -1.0)


def test_valid_cache_passes_validation():
    summary = validate_cache_payload(synthetic_cache())
    assert summary["episode_index"] == 7
    assert summary["num_frames"] == 5
    assert summary["latent_dtype"] == "torch.bfloat16"


@pytest.mark.parametrize("key", ["actions", "states"])
def test_wrong_action_or_state_shape_fails(key: str):
    cache = synthetic_cache()
    cache[key] = torch.zeros(5, 5)
    with pytest.raises(ValueError, match=key):
        validate_cache_payload(cache)


def test_non_monotonic_timestamp_fails():
    cache = synthetic_cache()
    cache["timestamps"][3] = cache["timestamps"][2]
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_cache_payload(cache)


def test_non_contiguous_frame_indices_fail():
    cache = synthetic_cache()
    cache["frame_indices"][3] = 9
    with pytest.raises(ValueError, match="contiguous"):
        validate_cache_payload(cache)


def test_frame_indices_must_cover_zero_through_n_minus_one():
    timestamps = torch.arange(4, dtype=torch.float64) / FPS
    validate_temporal_axis(timestamps, torch.tensor([0, 1, 2, 3]))
    with pytest.raises(ValueError, match="start at 0"):
        validate_temporal_axis(timestamps, torch.tensor([5, 6, 7, 8]))
    with pytest.raises(ValueError):
        validate_temporal_axis(timestamps, torch.tensor([0, 1, 3, 4]))


def test_video_table_alignment_accepts_different_absolute_origins():
    table = torch.tensor([0.00, 0.05, 0.10, 0.15], dtype=torch.float64)
    video = torch.tensor([12.30, 12.35, 12.40, 12.45], dtype=torch.float64)
    validate_video_table_alignment(table, video, episode_index=7)


def test_video_table_alignment_rejects_relative_timing_drift():
    table = torch.tensor([0.00, 0.05, 0.10, 0.15], dtype=torch.float64)
    video = torch.tensor([12.30, 12.35, 12.41, 12.46], dtype=torch.float64)
    with pytest.raises(
        ValueError,
        match=r"episode 7 .*first bad frame=2.*absolute_error=.*tolerance=",
    ):
        validate_video_table_alignment(table, video, episode_index=7)


def test_video_table_alignment_rejects_non_monotonic_video_time():
    table = torch.tensor([0.00, 0.05, 0.10, 0.15], dtype=torch.float64)
    video = torch.tensor([12.30, 12.35, 12.34, 12.45], dtype=torch.float64)
    with pytest.raises(ValueError, match="not strictly increasing"):
        validate_video_table_alignment(table, video, episode_index=7)


def test_video_table_alignment_rejects_length_mismatch():
    table = torch.tensor([0.00, 0.05, 0.10, 0.15], dtype=torch.float64)
    video = torch.tensor([12.30, 12.35, 12.40], dtype=torch.float64)
    with pytest.raises(ValueError, match="length mismatch"):
        validate_video_table_alignment(table, video, episode_index=7)


def test_video_table_alignment_rejects_non_20hz_video_delta():
    table = torch.tensor([0.00, 0.05, 0.10, 0.15], dtype=torch.float64)
    video = torch.tensor([12.30, 12.36, 12.42, 12.48], dtype=torch.float64)
    with pytest.raises(ValueError, match="video delta is not approximately 20 Hz"):
        validate_video_table_alignment(table, video, episode_index=7)


def test_latent_nan_fails():
    cache = synthetic_cache()
    cache["latents"][2, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="latents contains"):
        validate_cache_payload(cache)


def test_existing_output_without_overwrite_is_skipped(tmp_path: Path):
    output = tmp_path / "episode_000000.pt"
    output.write_bytes(b"already complete")
    assert should_skip_cache(output, overwrite=False)
    assert not should_skip_cache(output, overwrite=True)


def test_atomic_torch_save_replaces_target_and_removes_tmp(tmp_path: Path):
    output = tmp_path / "episode_000007.pt"
    old_payload = {"value": torch.tensor(1)}
    new_payload = {"value": torch.tensor(2)}
    torch.save(old_payload, output)
    atomic_torch_save(new_payload, output)
    loaded = torch.load(output, map_location="cpu", weights_only=False)
    assert int(loaded["value"]) == 2
    assert not temporary_cache_path(output).exists()


def test_synthetic_decode_divides_scaling_without_shift(tmp_path: Path):
    class Config:
        scaling_factor = 2.0
        shift_factor = 0.0609

    class DecodeResult:
        def __init__(self, batch_size: int):
            self.sample = torch.zeros(batch_size, 3, 256, 256)

    class FakeVAE:
        config = Config()

        def __init__(self):
            self.decode_inputs = []

        def decode(self, value: torch.Tensor) -> DecodeResult:
            self.decode_inputs.append(value.detach().cpu())
            return DecodeResult(len(value))

    cache = synthetic_cache()
    cache["latents"][0].fill_(4.0)
    cache["latents"][2].fill_(6.0)
    cache["latents"][4].fill_(8.0)
    vae = FakeVAE()
    decode_representative_latents(cache, vae, torch.device("cpu"), tmp_path)
    assert [float(value.mean()) for value in vae.decode_inputs] == [2.0, 3.0, 4.0]
    assert {path.name for path in tmp_path.glob("*.png")} == {
        "first.png",
        "middle.png",
        "last.png",
    }
