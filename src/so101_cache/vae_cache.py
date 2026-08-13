from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


FPS = 20
CAMERA_KEY = "observation.images.front"
IMAGE_SIZE = 256
LATENT_CHANNELS = 16
LATENT_HEIGHT = 32
LATENT_WIDTH = 32
RAW_ACTION_DIM = 6
VAE_IDENTIFIER = "stabilityai/stable-diffusion-3-medium-diffusers"
LATENT_CONVENTION = "posterior_sample_times_scaling_no_shift"
VIDEO_ALIGNMENT_TOLERANCE_SEC = 5e-3


def requested_episode_ids(episode_start: int, episode_end: int) -> list[int]:
    """Return [episode_start, episode_end); the end is intentionally exclusive."""

    episode_start = int(episode_start)
    episode_end = int(episode_end)
    if episode_start < 0:
        raise ValueError("episode_start must be non-negative")
    if episode_end <= episode_start:
        raise ValueError("episode_end must be greater than episode_start")
    return list(range(episode_start, episode_end))


def episode_ids_for_shard(
    episode_ids: Sequence[int],
    shard_id: int,
    num_shards: int,
) -> list[int]:
    """Assign episodes deterministically with episode_id % num_shards."""

    shard_id = int(shard_id)
    num_shards = int(num_shards)
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("shard_id must satisfy 0 <= shard_id < num_shards")
    return [int(value) for value in episode_ids if int(value) % num_shards == shard_id]


def episode_seed(base_seed: int, episode_index: int) -> int:
    """Derive an order- and shard-independent posterior sampling seed."""

    base_seed = int(base_seed)
    episode_index = int(episode_index)
    if base_seed < 0:
        raise ValueError("seed must be non-negative")
    if episode_index < 0:
        raise ValueError("episode_index must be non-negative")
    return (base_seed + episode_index) % (2**63 - 1)


def episode_cache_path(output_root: str | Path, episode_index: int) -> Path:
    episode_index = int(episode_index)
    if episode_index < 0:
        raise ValueError("episode_index must be non-negative")
    # causal v2's default single-episode resolver uses the same minimum
    # three-digit width.  Values above 999 are not truncated by Python format.
    return Path(output_root) / f"episode_{episode_index:03d}.pt"


def resolve_vae_dtype(device: torch.device | str) -> torch.dtype:
    """Use BF16 for the formal CUDA cache and FP32 for the CPU fallback."""

    return torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32


def should_skip_cache(path: str | Path, overwrite: bool) -> bool:
    return Path(path).is_file() and not bool(overwrite)


def temporary_cache_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".tmp")


def atomic_torch_save(payload: object, path: str | Path) -> None:
    """Write to a sibling temporary file, then atomically replace the target."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_cache_path(path)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def preprocess_rgb_batch(images: Sequence[np.ndarray]) -> torch.Tensor:
    """Apply the frozen SO101 front-camera preprocessing convention.

    RGB uint8 images are resized so their shortest side is 256, center-cropped
    to 256x256, converted to float [0, 1], and finally mapped to [-1, 1].
    """

    if not images:
        raise ValueError("images must be non-empty")
    arrays = [np.asarray(image) for image in images]
    first_shape = arrays[0].shape
    if len(first_shape) != 3 or first_shape[-1] != 3:
        raise ValueError(f"expected RGB HWC images, got {first_shape}")
    for image in arrays:
        if image.shape != first_shape:
            raise ValueError("all images in one batch must have the same shape")
        if image.dtype != np.uint8:
            raise ValueError(f"expected uint8 RGB input, got {image.dtype}")

    x = torch.from_numpy(np.stack(arrays, axis=0)).permute(0, 3, 1, 2).float()
    x = x / 255.0
    _, _, height, width = x.shape
    scale = IMAGE_SIZE / min(height, width)
    resized_height = round(height * scale)
    resized_width = round(width * scale)
    if (resized_height, resized_width) != (height, width):
        x = F.interpolate(
            x,
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        )

    _, _, height, width = x.shape
    top = (height - IMAGE_SIZE) // 2
    left = (width - IMAGE_SIZE) // 2
    x = x[:, :, top : top + IMAGE_SIZE, left : left + IMAGE_SIZE]
    if x.shape[1:] != (3, IMAGE_SIZE, IMAGE_SIZE):
        raise RuntimeError(f"unexpected preprocessed image shape: {tuple(x.shape)}")
    return x.mul(2.0).sub(1.0).contiguous()


def preprocess_rgb_image(image: np.ndarray) -> torch.Tensor:
    """Preprocess one RGB image to [3, 256, 256] in [-1, 1]."""

    return preprocess_rgb_batch([image])[0]


def validate_temporal_axis(
    timestamps: torch.Tensor,
    frame_indices: torch.Tensor,
    *,
    fps: int = FPS,
    timestamp_tolerance: float = 5e-3,
) -> None:
    timestamps = torch.as_tensor(timestamps, dtype=torch.float64)
    frame_indices = torch.as_tensor(frame_indices, dtype=torch.long)
    if timestamps.ndim != 1 or frame_indices.ndim != 1:
        raise ValueError("timestamps and frame_indices must be 1D")
    if timestamps.shape != frame_indices.shape:
        raise ValueError("timestamps/frame_indices length mismatch")
    if not torch.isfinite(timestamps).all():
        raise ValueError("timestamps must be finite")
    if len(timestamps) == 0:
        raise ValueError("timestamps and frame_indices must be non-empty")
    if int(frame_indices[0]) != 0:
        raise ValueError("frame_indices must start at 0")
    if int(frame_indices[-1]) != len(frame_indices) - 1:
        raise ValueError("frame_indices must end at N - 1")
    if len(timestamps) <= 1:
        return

    timestamp_deltas = timestamps[1:] - timestamps[:-1]
    if torch.any(timestamp_deltas <= 0):
        raise ValueError("timestamps must be strictly increasing")
    expected_delta = torch.full_like(timestamp_deltas, 1.0 / float(fps))
    if not torch.allclose(
        timestamp_deltas,
        expected_delta,
        rtol=0.0,
        atol=float(timestamp_tolerance),
    ):
        min_delta = float(timestamp_deltas.min())
        max_delta = float(timestamp_deltas.max())
        raise ValueError(
            f"timestamps are not approximately {fps} Hz: "
            f"delta range=[{min_delta:.6f}, {max_delta:.6f}]"
        )

    frame_deltas = frame_indices[1:] - frame_indices[:-1]
    if not torch.equal(frame_deltas, torch.ones_like(frame_deltas)):
        raise ValueError("frame_indices must be contiguous with step 1")


def validate_video_table_alignment(
    table_timestamps: torch.Tensor,
    video_timestamps: torch.Tensor,
    *,
    episode_index: int | None = None,
    fps: int = FPS,
    tolerance: float = VIDEO_ALIGNMENT_TOLERANCE_SEC,
) -> None:
    """Validate per-frame timing after removing different absolute origins."""

    table = torch.as_tensor(table_timestamps, dtype=torch.float64)
    video = torch.as_tensor(video_timestamps, dtype=torch.float64)
    episode_label = "unknown" if episode_index is None else str(int(episode_index))
    if table.ndim != 1 or video.ndim != 1:
        raise ValueError(
            f"episode {episode_label} video/table timestamps must both be 1D"
        )
    if table.shape != video.shape:
        raise ValueError(
            f"episode {episode_label} video/table timestamp length mismatch: "
            f"table={len(table)}, video={len(video)}"
        )
    if len(table) == 0:
        raise ValueError(f"episode {episode_label} timestamps must be non-empty")
    if not torch.isfinite(table).all() or not torch.isfinite(video).all():
        raise ValueError(
            f"episode {episode_label} video/table timestamps must all be finite"
        )
    if fps <= 0 or tolerance < 0:
        raise ValueError("fps must be positive and tolerance must be non-negative")

    table_relative = table - table[0]
    video_relative = video - video[0]

    def alignment_error(reason: str, frame_index: int) -> ValueError:
        table_value = float(table_relative[frame_index])
        video_value = float(video_relative[frame_index])
        error = abs(table_value - video_value)
        return ValueError(
            f"episode {episode_label} video/table alignment failed ({reason}): "
            f"first bad frame={frame_index}, "
            f"table_relative={table_value:.9f}, "
            f"video_relative={video_value:.9f}, "
            f"absolute_error={error:.9f}, tolerance={float(tolerance):.9f}"
        )

    if len(table) == 1:
        return
    table_deltas = table[1:] - table[:-1]
    video_deltas = video[1:] - video[:-1]
    bad_table_order = torch.nonzero(table_deltas <= 0, as_tuple=False)
    if len(bad_table_order):
        raise alignment_error("table timestamps are not strictly increasing", int(bad_table_order[0]) + 1)
    bad_video_order = torch.nonzero(video_deltas <= 0, as_tuple=False)
    if len(bad_video_order):
        raise alignment_error("video timestamps are not strictly increasing", int(bad_video_order[0]) + 1)

    expected_delta = 1.0 / float(fps)
    bad_table_rate = torch.nonzero(
        torch.abs(table_deltas - expected_delta) > float(tolerance),
        as_tuple=False,
    )
    if len(bad_table_rate):
        raise alignment_error("table delta is not approximately 20 Hz", int(bad_table_rate[0]) + 1)
    bad_video_rate = torch.nonzero(
        torch.abs(video_deltas - expected_delta) > float(tolerance),
        as_tuple=False,
    )
    if len(bad_video_rate):
        raise alignment_error("video delta is not approximately 20 Hz", int(bad_video_rate[0]) + 1)

    relative_errors = torch.abs(table_relative - video_relative)
    bad_alignment = torch.nonzero(
        relative_errors > float(tolerance), as_tuple=False
    )
    if len(bad_alignment):
        raise alignment_error("relative timestamps differ", int(bad_alignment[0]))


def validate_cache_payload(cache: Mapping[str, object]) -> dict[str, object]:
    """Validate one raw-20Hz episode cache and return a compact summary."""

    required = {
        "latents",
        "actions",
        "states",
        "timestamps",
        "frame_indices",
        "episode_index",
        "metadata",
    }
    missing = required - set(cache)
    if missing:
        raise ValueError(f"cache is missing keys: {sorted(missing)}")

    latents = torch.as_tensor(cache["latents"])
    actions = torch.as_tensor(cache["actions"])
    states = torch.as_tensor(cache["states"])
    timestamps = torch.as_tensor(cache["timestamps"])
    frame_indices = torch.as_tensor(cache["frame_indices"])
    episode_index = int(cache["episode_index"])
    metadata = cache["metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")

    if latents.ndim != 4 or tuple(latents.shape[1:]) != (
        LATENT_CHANNELS,
        LATENT_HEIGHT,
        LATENT_WIDTH,
    ):
        raise ValueError(
            "latents must be [N,16,32,32], " f"got {tuple(latents.shape)}"
        )
    if latents.dtype not in {torch.bfloat16, torch.float32}:
        raise ValueError(
            "latents must retain the VAE bfloat16/float32 output dtype; "
            f"got {latents.dtype}"
        )
    frame_count = int(latents.shape[0])
    if frame_count <= 0:
        raise ValueError("cache must contain at least one frame")
    if tuple(actions.shape) != (frame_count, RAW_ACTION_DIM):
        raise ValueError(
            f"actions must be [{frame_count},{RAW_ACTION_DIM}], got {tuple(actions.shape)}"
        )
    if tuple(states.shape) != (frame_count, RAW_ACTION_DIM):
        raise ValueError(
            f"states must be [{frame_count},{RAW_ACTION_DIM}], got {tuple(states.shape)}"
        )
    if actions.dtype != torch.float32:
        raise ValueError(f"actions must be float32, got {actions.dtype}")
    if states.dtype != torch.float32:
        raise ValueError(f"states must be float32, got {states.dtype}")
    if timestamps.dtype != torch.float64:
        raise ValueError(f"timestamps must be float64, got {timestamps.dtype}")
    if frame_indices.dtype != torch.long:
        raise ValueError(f"frame_indices must be int64, got {frame_indices.dtype}")
    if tuple(timestamps.shape) != (frame_count,):
        raise ValueError(f"timestamps must be [{frame_count}]")
    if tuple(frame_indices.shape) != (frame_count,):
        raise ValueError(f"frame_indices must be [{frame_count}]")
    for name, tensor in (
        ("latents", latents),
        ("actions", actions),
        ("states", states),
    ):
        if not torch.isfinite(tensor.float()).all():
            raise ValueError(f"{name} contains NaN or Inf")

    validate_temporal_axis(timestamps, frame_indices, fps=FPS)
    if int(metadata.get("fps", FPS)) != FPS:
        raise ValueError(f"metadata fps must be {FPS}")
    if metadata.get("camera", CAMERA_KEY) != CAMERA_KEY:
        raise ValueError(f"metadata camera must be {CAMERA_KEY!r}")
    if metadata.get("latent_convention", LATENT_CONVENTION) != LATENT_CONVENTION:
        raise ValueError(f"unexpected latent convention: {metadata.get('latent_convention')!r}")
    if metadata.get("input_image_size", [IMAGE_SIZE, IMAGE_SIZE]) != [
        IMAGE_SIZE,
        IMAGE_SIZE,
    ]:
        raise ValueError("metadata input_image_size must be [256, 256]")
    if "episode_index" in metadata and int(metadata["episode_index"]) != episode_index:
        raise ValueError("top-level and metadata episode_index disagree")
    if "latent_dtype" in metadata and metadata["latent_dtype"] != str(latents.dtype):
        raise ValueError("metadata latent_dtype does not match cached latents")
    if "video_timestamps" in cache:
        video_timestamps = torch.as_tensor(cache["video_timestamps"])
        if video_timestamps.shape != (frame_count,):
            raise ValueError(f"video_timestamps must be [{frame_count}]")
        if video_timestamps.dtype != torch.float64:
            raise ValueError("video_timestamps must be float64")
        validate_video_table_alignment(
            timestamps,
            video_timestamps,
            episode_index=episode_index,
            fps=FPS,
            tolerance=VIDEO_ALIGNMENT_TOLERANCE_SEC,
        )
        if metadata.get("video_table_alignment_checked") is not True:
            raise ValueError("metadata must record video_table_alignment_checked=True")
        recorded_tolerance = float(
            metadata.get("video_alignment_tolerance_sec", float("nan"))
        )
        if not np.isclose(
            recorded_tolerance,
            VIDEO_ALIGNMENT_TOLERANCE_SEC,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "metadata video_alignment_tolerance_sec does not match validator"
            )

    return {
        "episode_index": episode_index,
        "num_frames": frame_count,
        "latent_dtype": str(latents.dtype),
        "action_dtype": str(actions.dtype),
        "state_dtype": str(states.dtype),
        "timestamp_dtype": str(timestamps.dtype),
    }
