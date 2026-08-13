from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.so101_cache.so101_reader import SO101LeRobotV3Reader
from src.so101_cache.vae_cache import (
    CAMERA_KEY,
    FPS,
    IMAGE_SIZE,
    LATENT_CHANNELS,
    LATENT_CONVENTION,
    LATENT_HEIGHT,
    LATENT_WIDTH,
    VIDEO_ALIGNMENT_TOLERANCE_SEC,
    VAE_IDENTIFIER,
    atomic_torch_save,
    episode_cache_path,
    episode_ids_for_shard,
    episode_seed,
    preprocess_rgb_batch,
    requested_episode_ids,
    resolve_vae_dtype,
    should_skip_cache,
    validate_cache_payload,
    validate_video_table_alignment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cache raw 20 Hz SO101 front RGB frames as frozen SD3-VAE latents. "
            "No causal action shift, stride, chunking, or normalization is applied."
        ),
        epilog=(
            "Example: --episode-start 0 --episode-end 2 selects episode 0 and "
            "episode 1; --episode-end is exclusive."
        ),
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--vae-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--episode-start",
        type=int,
        required=True,
        help="First episode index to request (inclusive).",
    )
    parser.add_argument(
        "--episode-end",
        type=int,
        required=True,
        help="Stop episode index (exclusive).",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device, for example cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Base seed. Episode E uses seed+E so results do not depend on "
            "process order or sharding."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-encode an episode even if its final .pt file already exists.",
    )
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser


def resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {value!r} was requested but CUDA is unavailable")
    return device


def set_posterior_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def load_vae(vae_root: Path, device: torch.device):
    from diffusers import AutoencoderKL

    # Formal A100 cache dtype policy: bfloat16.  CPU is a float32 fallback for
    # local diagnostics; neither path converts the saved latent to fp16.
    vae_dtype = resolve_vae_dtype(device)
    vae = AutoencoderKL.from_pretrained(
        vae_root,
        subfolder="vae",
        local_files_only=True,
        torch_dtype=vae_dtype,
    )
    vae.eval()
    vae.requires_grad_(False)
    vae.to(device)
    actual_dtype = next(vae.parameters()).dtype
    if actual_dtype != vae_dtype:
        raise RuntimeError(
            f"VAE dtype policy requested {vae_dtype}, but loaded {actual_dtype}"
        )
    latent_channels = int(vae.config.latent_channels)
    if latent_channels != LATENT_CHANNELS:
        raise RuntimeError(
            f"expected SD3 VAE latent_channels={LATENT_CHANNELS}, got {latent_channels}"
        )
    scaling_factor = float(vae.config.scaling_factor)
    if not torch.isfinite(torch.tensor(scaling_factor)) or scaling_factor <= 0:
        raise RuntimeError(f"invalid VAE scaling_factor: {scaling_factor}")
    return vae


@torch.inference_mode()
def encode_episode(
    *,
    reader: SO101LeRobotV3Reader,
    vae,
    device: torch.device,
    dataset_root: Path,
    vae_root: Path,
    output_root: Path,
    episode_index: int,
    batch_size: int,
    base_seed: int,
    overwrite: bool,
) -> dict[str, object]:
    output_path = episode_cache_path(output_root, episode_index)
    if should_skip_cache(output_path, overwrite):
        print(f"SKIP episode {episode_index:06d}: {output_path}")
        return {"episode_index": episode_index, "status": "skipped"}

    print(f"BEGIN episode {episode_index:06d}")
    episode = reader.load_episode(episode_index)
    segment = reader.video_segment(episode_index)
    posterior_seed = episode_seed(base_seed, episode_index)
    set_posterior_seed(posterior_seed, device)

    latent_batches: list[torch.Tensor] = []
    video_timestamps: list[float] = []
    encoded_frames = 0
    vae_dtype = next(vae.parameters()).dtype
    for images, batch_video_timestamps in reader.iter_frame_batches(
        segment,
        expected_frames=episode.num_frames,
        batch_size=batch_size,
    ):
        x = preprocess_rgb_batch(images).to(
            device=device,
            dtype=vae_dtype,
        )
        posterior = vae.encode(x).latent_dist
        # Frozen project convention: sample, multiply by scaling, and NEVER
        # apply SD3 shift_factor.  All future stride experiments reuse this cache.
        z = posterior.sample()
        z = z * vae.config.scaling_factor
        if z.ndim != 4 or tuple(z.shape[1:]) != (
            LATENT_CHANNELS,
            LATENT_HEIGHT,
            LATENT_WIDTH,
        ):
            raise RuntimeError(f"unexpected VAE latent batch shape: {tuple(z.shape)}")
        latent_batches.append(z.detach().cpu().contiguous())
        video_timestamps.extend(float(value) for value in batch_video_timestamps)
        encoded_frames += int(z.shape[0])
        print(
            f"  episode {episode_index:06d}: encoded "
            f"{encoded_frames}/{episode.num_frames}"
        )

    if not latent_batches:
        raise RuntimeError(f"episode {episode_index} produced no latent batches")
    latents = torch.cat(latent_batches, dim=0).contiguous()
    if len(latents) != episode.num_frames or len(video_timestamps) != episode.num_frames:
        raise RuntimeError(
            f"episode {episode_index} final length mismatch: "
            f"latents={len(latents)}, video_pts={len(video_timestamps)}, "
            f"table={episode.num_frames}"
        )

    video_timestamps_tensor = torch.tensor(
        video_timestamps, dtype=torch.float64
    )
    validate_video_table_alignment(
        episode.timestamps,
        video_timestamps_tensor,
        episode_index=episode_index,
        fps=FPS,
        tolerance=VIDEO_ALIGNMENT_TOLERANCE_SEC,
    )

    scaling_factor = float(vae.config.scaling_factor)
    metadata = {
        "episode_index": int(episode_index),
        "fps": FPS,
        "camera": CAMERA_KEY,
        "input_image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "vae_identifier": VAE_IDENTIFIER,
        "vae_root": str(vae_root.resolve()),
        "latent_convention": LATENT_CONVENTION,
        "scaling_factor": scaling_factor,
        "latent_channels": LATENT_CHANNELS,
        "seed": posterior_seed,
        "base_seed": int(base_seed),
        "source_dataset_root": str(dataset_root.resolve()),
        "video_shard": str(segment.path.resolve()),
        "video_from_timestamp": segment.from_timestamp,
        "video_to_timestamp": segment.to_timestamp,
        "video_alignment_tolerance_sec": VIDEO_ALIGNMENT_TOLERANCE_SEC,
        "video_table_alignment_checked": True,
        "vae_dtype": str(vae_dtype),
        "latent_dtype": str(latents.dtype),
        "action_dtype": str(episode.actions.dtype),
        "state_dtype": str(episode.states.dtype),
        "raw_time_axis": True,
        "shift_factor_used": False,
    }
    cache = {
        "latents": latents,
        "actions": episode.actions,
        "states": episode.states,
        "timestamps": episode.timestamps,
        "frame_indices": episode.frame_indices,
        "episode_index": int(episode_index),
        "video_timestamps": video_timestamps_tensor,
        "metadata": metadata,
    }
    summary = validate_cache_payload(cache)
    atomic_torch_save(cache, output_path)
    print(
        f"SAVED episode {episode_index:06d}: {output_path} "
        f"frames={summary['num_frames']} latent_dtype={summary['latent_dtype']}"
    )
    return {
        "episode_index": episode_index,
        "status": "saved",
        "cache_file": str(output_path),
        **summary,
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    requested = requested_episode_ids(args.episode_start, args.episode_end)
    selected = episode_ids_for_shard(requested, args.shard_id, args.num_shards)
    output_paths = [episode_cache_path(args.output_root, value) for value in selected]
    pending = [
        episode_index
        for episode_index, path in zip(selected, output_paths)
        if not should_skip_cache(path, args.overwrite)
    ]

    print("SO101 RAW-20HZ SD3-VAE LATENT CACHE")
    print(f"requested range : [{args.episode_start}, {args.episode_end})")
    print(f"shard           : {args.shard_id}/{args.num_shards}")
    print(f"selected        : {selected}")
    print(f"pending         : {pending}")
    for episode_index, path in zip(selected, output_paths):
        if should_skip_cache(path, args.overwrite):
            print(f"SKIP episode {episode_index:06d}: {path}")
    if not pending:
        print("No episodes require encoding.")
        return

    device = resolve_device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    reader = SO101LeRobotV3Reader(args.dataset_root, camera=CAMERA_KEY)
    vae = load_vae(args.vae_root, device)
    print(f"device          : {device}")
    print(f"vae_dtype       : {next(vae.parameters()).dtype}")
    if device.type == "cuda":
        print(f"gpu             : {torch.cuda.get_device_name(device)}")
    print(f"scaling_factor  : {float(vae.config.scaling_factor)}")
    print(f"shift_factor    : IGNORED ({getattr(vae.config, 'shift_factor', None)})")

    results = []
    for episode_index in pending:
        results.append(
            encode_episode(
                reader=reader,
                vae=vae,
                device=device,
                dataset_root=args.dataset_root,
                vae_root=args.vae_root,
                output_root=args.output_root,
                episode_index=episode_index,
                batch_size=args.batch_size,
                base_seed=args.seed,
                overwrite=args.overwrite,
            )
        )
    print(f"COMPLETE shard {args.shard_id}: saved={len(results)}")


if __name__ == "__main__":
    main()
