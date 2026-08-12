from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL


# ============================================================
# Paths
# ============================================================

REPO_ROOT = Path(
    "data/armnet_so101_sample_tabular"
)

PILOT_MANIFEST = Path(
    "data/multi_episode_pilot/manifest.json"
)

CACHE_ROOT = Path(
    "data/latent_cache/so101_front"
)

CACHE_MANIFEST = (
    CACHE_ROOT / "multi_episode_manifest.json"
)

# ============================================================
# VAE
# ============================================================

MODEL_ID = (
    "stabilityai/"
    "stable-diffusion-3-medium-diffusers"
)

IMAGE_SIZE = 256

BASE_SEED = 1000


# ============================================================
# Load manifest
# ============================================================

def load_manifest() -> dict:

    if not PILOT_MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifest not found: "
            f"{PILOT_MANIFEST}"
        )

    with open(
        PILOT_MANIFEST,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


# ============================================================
# Load one episode's tabular data
# ============================================================

def load_episode_dataframe(
    episode_index: int,
) -> pd.DataFrame:

    parquet_root = (
        REPO_ROOT / "data"
    )

    dataset = ds.dataset(
        parquet_root,
        format="parquet",
    )

    table = dataset.to_table(
        filter=(
            ds.field("episode_index")
            == episode_index
        )
    )

    df = table.to_pandas()

    if len(df) == 0:
        raise RuntimeError(
            f"Episode {episode_index} "
            "not found in parquet."
        )

    if "frame_index" in df.columns:
        df = df.sort_values(
            "frame_index"
        )

    df = df.reset_index(
        drop=True
    )

    return df


# ============================================================
# Video decoding
# ============================================================

def decode_episode_frames(
    video_path: Path,
    from_timestamp: float,
    to_timestamp: float,
    expected_frames: int,
) -> tuple[
    list[np.ndarray],
    list[float],
]:
    """
    从共享 MP4 shard 中精确截取一个 episode。

    返回：

        images:
            list of RGB uint8 [H,W,3]

        timestamps:
            MP4 PTS timestamps
    """

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: "
            f"{video_path}"
        )

    container = av.open(
        str(video_path)
    )

    stream = (
        container.streams.video[0]
    )

    images = []
    timestamps = []

    for frame in container.decode(
        stream
    ):

        if frame.pts is None:
            continue

        timestamp = float(
            frame.pts
            * stream.time_base
        )

        # episode 之前
        if (
            timestamp + 1e-6
            < from_timestamp
        ):
            continue

        # episode 之后
        if timestamp >= to_timestamp:
            break

        image = frame.to_ndarray(
            format="rgb24"
        )

        images.append(
            image
        )

        timestamps.append(
            timestamp
        )

        if (
            len(images)
            >= expected_frames
        ):
            break

    container.close()

    return (
        images,
        timestamps,
    )


# ============================================================
# Image preprocessing
# ============================================================

def preprocess_batch(
    images: list[np.ndarray],
) -> torch.Tensor:
    """
    RGB uint8:
        [B,H,W,3]

    ->
        [B,3,256,256]

    range:
        [-1,1]
    """

    x = np.stack(
        images,
        axis=0,
    )

    x = torch.from_numpy(
        x
    )

    x = x.permute(
        0,
        3,
        1,
        2,
    ).float()

    x = (
        x / 255.0
    )

    _, _, h, w = x.shape

    # ========================================================
    # Resize shortest side -> 256
    # ========================================================

    scale = (
        IMAGE_SIZE
        / min(h, w)
    )

    new_h = round(
        h * scale
    )

    new_w = round(
        w * scale
    )

    if (
        new_h != h
        or new_w != w
    ):

        x = F.interpolate(
            x,
            size=(
                new_h,
                new_w,
            ),
            mode="bilinear",
            align_corners=False,
        )

    # ========================================================
    # Center crop
    # ========================================================

    _, _, h, w = x.shape

    top = (
        h - IMAGE_SIZE
    ) // 2

    left = (
        w - IMAGE_SIZE
    ) // 2

    x = x[
        :,
        :,
        top : top + IMAGE_SIZE,
        left : left + IMAGE_SIZE,
    ]

    if x.shape[-2:] != (
        IMAGE_SIZE,
        IMAGE_SIZE,
    ):
        raise RuntimeError(
            "Unexpected image shape: "
            f"{tuple(x.shape)}"
        )

    # [0,1] -> [-1,1]

    x = (
        x * 2.0
        - 1.0
    )

    return x


# ============================================================
# VAE encode
# ============================================================

@torch.inference_mode()
def encode_episode(
    vae: AutoencoderKL,
    images: list[np.ndarray],
    device: torch.device,
    batch_size: int,
    episode_index: int,
) -> torch.Tensor:

    # --------------------------------------------------------
    # 每个 episode 固定独立 seed。
    #
    # 即使之后改变 episode 处理顺序，
    # posterior.sample() 仍然可复现。
    # --------------------------------------------------------

    seed = (
        BASE_SEED
        + episode_index
    )

    torch.manual_seed(
        seed
    )

    torch.cuda.manual_seed_all(
        seed
    )

    latent_batches = []

    n = len(images)

    for start in range(
        0,
        n,
        batch_size,
    ):

        end = min(
            start + batch_size,
            n,
        )

        x = preprocess_batch(
            images[
                start:end
            ]
        )

        x = x.to(
            device=device,
            dtype=torch.bfloat16,
        )

        posterior = (
            vae.encode(x)
            .latent_dist
        )

        # 与之前 world-model cache 保持一致
        z = posterior.sample()

        z = (
            z
            * vae.config.scaling_factor
        )

        latent_batches.append(
            z.cpu()
        )

        print(
            f"    encoded "
            f"{end:4d}/{n}"
        )

    latents = torch.cat(
        latent_batches,
        dim=0,
    )

    return latents


# ============================================================
# Convert vector column
# ============================================================

def stack_vectors(
    series: pd.Series,
) -> torch.Tensor:

    array = np.stack(
        [
            np.asarray(x)
            .reshape(-1)
            for x in series
        ],
        axis=0,
    )

    return torch.tensor(
        array,
        dtype=torch.float32,
    )


# ============================================================
# Cache one episode
# ============================================================

def cache_episode(
    vae: AutoencoderKL,
    device: torch.device,
    record: dict,
    batch_size: int,
    overwrite: bool,
) -> dict:

    episode_index = int(
        record["episode_index"]
    )

    cache_path = (
        CACHE_ROOT
        / f"episode_{episode_index:03d}.pt"
    )

    print()
    print("=" * 80)
    print(
        f"EPISODE {episode_index}"
    )
    print("=" * 80)

    # ========================================================
    # Existing cache
    # ========================================================

    if (
        cache_path.exists()
        and not overwrite
    ):

        print(
            "cache already exists:"
        )

        print(
            cache_path
        )

        cache = torch.load(
            cache_path,
            map_location="cpu",
        )

        print(
            "latents:",
            tuple(
                cache["latents"].shape
            ),
        )

        return {
            "episode_index": (
                episode_index
            ),
            "cache_file": str(
                cache_path
            ),
            "num_frames": int(
                len(
                    cache["latents"]
                )
            ),
            "status": "existing",
        }

    # ========================================================
    # Tabular data
    # ========================================================

    df = load_episode_dataframe(
        episode_index
    )

    expected_frames = len(
        df
    )

    manifest_length = record.get(
        "length",
        None,
    )

    print(
        "parquet frames:",
        expected_frames,
    )

    if (
        manifest_length is not None
        and int(manifest_length)
        != expected_frames
    ):
        raise RuntimeError(
            "Manifest/parquet length "
            "mismatch: "
            f"{manifest_length} "
            f"vs {expected_frames}"
        )

    # ========================================================
    # Video
    # ========================================================

    video_file = Path(
        record["video_file"]
    )

    video_path = (
        REPO_ROOT
        / video_file
    )

    from_timestamp = record.get(
        "from_timestamp",
        None,
    )

    to_timestamp = record.get(
        "to_timestamp",
        None,
    )

    if (
        from_timestamp is None
        or to_timestamp is None
    ):
        raise RuntimeError(
            f"Episode {episode_index} "
            "has no valid video timestamps."
        )

    print(
        "video:",
        video_path,
    )

    print(
        "from:",
        from_timestamp,
    )

    print(
        "to  :",
        to_timestamp,
    )

    images, video_timestamps = (
        decode_episode_frames(
            video_path=video_path,
            from_timestamp=float(
                from_timestamp
            ),
            to_timestamp=float(
                to_timestamp
            ),
            expected_frames=(
                expected_frames
            ),
        )
    )

    print(
        "decoded frames:",
        len(images),
    )

    if (
        len(images)
        != expected_frames
    ):
        raise RuntimeError(
            "Video/parquet length "
            "mismatch for episode "
            f"{episode_index}: "
            f"video={len(images)}, "
            f"parquet={expected_frames}"
        )

    # ========================================================
    # VAE
    # ========================================================

    latents = encode_episode(
        vae=vae,
        images=images,
        device=device,
        batch_size=batch_size,
        episode_index=episode_index,
    )

    expected_latent_shape = (
        expected_frames,
        16,
        32,
        32,
    )

    print(
        "latent shape:",
        tuple(
            latents.shape
        ),
    )

    if (
        latents.shape
        != expected_latent_shape
    ):
        raise RuntimeError(
            "Unexpected latent shape: "
            f"{tuple(latents.shape)} "
            f"!= "
            f"{expected_latent_shape}"
        )

    if not torch.isfinite(
        latents.float()
    ).all():
        raise RuntimeError(
            "NaN/Inf found in latents."
        )

    # ========================================================
    # Action / state
    # ========================================================

    actions = stack_vectors(
        df["action"]
    )

    states = stack_vectors(
        df["observation.state"]
    )

    if actions.shape != (
        expected_frames,
        6,
    ):
        raise RuntimeError(
            "Unexpected actions shape: "
            f"{tuple(actions.shape)}"
        )

    if states.shape != (
        expected_frames,
        6,
    ):
        raise RuntimeError(
            "Unexpected states shape: "
            f"{tuple(states.shape)}"
        )

    # ========================================================
    # Timestamp
    # ========================================================

    if "timestamp" in df.columns:

        timestamps = torch.tensor(
            df["timestamp"]
            .to_numpy(),
            dtype=torch.float64,
        )

    else:

        timestamps = torch.arange(
            expected_frames,
            dtype=torch.float64,
        ) / 20.0

    # ========================================================
    # Frame index
    # ========================================================

    if "frame_index" in df.columns:

        frame_indices = torch.tensor(
            df["frame_index"]
            .to_numpy(),
            dtype=torch.long,
        )

    else:

        frame_indices = torch.arange(
            expected_frames,
            dtype=torch.long,
        )

    video_timestamps_tensor = (
        torch.tensor(
            video_timestamps,
            dtype=torch.float64,
        )
    )

    # ========================================================
    # Save
    # ========================================================

    cache = {
        "episode_index": (
            episode_index
        ),

        "latents": latents,

        "actions": actions,

        "states": states,

        "timestamps": (
            timestamps
        ),

        "video_timestamps": (
            video_timestamps_tensor
        ),

        "frame_indices": (
            frame_indices
        ),

        "camera": (
            "observation.images.front"
        ),

        "vae_model": (
            MODEL_ID
        ),

        "vae_scaling_factor": float(
            vae.config.scaling_factor
        ),

        "vae_posterior": (
            "sample"
        ),

        "latent_seed": (
            BASE_SEED
            + episode_index
        ),

        "image_size": (
            IMAGE_SIZE
        ),
    }

    torch.save(
        cache,
        cache_path,
    )

    print(
        "saved:",
        cache_path,
    )

    print(
        "latent mean:",
        float(
            latents.float()
            .mean()
        ),
    )

    print(
        "latent std :",
        float(
            latents.float()
            .std()
        ),
    )

    return {
        "episode_index": (
            episode_index
        ),
        "cache_file": str(
            cache_path
        ),
        "num_frames": (
            expected_frames
        ),
        "status": "created",
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    device = torch.device(
        "cuda"
    )

    CACHE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = load_manifest()

    print("=" * 80)
    print("SO101 MULTI-EPISODE LATENT CACHE")
    print("=" * 80)

    print(
        "GPU:",
        torch.cuda.get_device_name(
            0
        ),
    )

    print(
        "episodes:",
        manifest["episode_ids"],
    )

    print(
        "train:",
        manifest[
            "train_episode_ids"
        ],
    )

    print(
        "val:",
        manifest[
            "val_episode_ids"
        ],
    )

    # ========================================================
    # Load VAE ONCE
    # ========================================================

    print()
    print(
        "Loading SD3 VAE once..."
    )

    vae = AutoencoderKL.from_pretrained(
        MODEL_ID,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    )

    vae.eval()
    vae.requires_grad_(
        False
    )

    vae.to(
        device
    )

    print(
        "VAE loaded."
    )

    print(
        "latent channels:",
        vae.config.latent_channels,
    )

    print(
        "scaling factor:",
        vae.config.scaling_factor,
    )

    # ========================================================
    # Cache episodes
    # ========================================================

    results = []

    episode_records = {
        int(x["episode_index"]): x
        for x in manifest["episodes"]
    }

    for episode_index in (
        manifest["episode_ids"]
    ):

        episode_index = int(
            episode_index
        )

        if (
            episode_index
            not in episode_records
        ):
            raise RuntimeError(
                "Episode missing from "
                f"manifest records: "
                f"{episode_index}"
            )

        result = cache_episode(
            vae=vae,
            device=device,
            record=(
                episode_records[
                    episode_index
                ]
            ),
            batch_size=(
                args.batch_size
            ),
            overwrite=(
                args.overwrite
            ),
        )

        results.append(
            result
        )

    # ========================================================
    # Save cache manifest
    # ========================================================

    output_manifest = {
        "source_manifest": str(
            PILOT_MANIFEST
        ),

        "repo_id": (
            manifest["repo_id"]
        ),

        "camera": (
            manifest["camera"]
        ),

        "episode_ids": (
            manifest[
                "episode_ids"
            ]
        ),

        "train_episode_ids": (
            manifest[
                "train_episode_ids"
            ]
        ),

        "val_episode_ids": (
            manifest[
                "val_episode_ids"
            ]
        ),

        "vae_model": (
            MODEL_ID
        ),

        "vae_posterior": (
            "sample"
        ),

        "image_size": (
            IMAGE_SIZE
        ),

        "episodes": (
            results
        ),
    }

    with open(
        CACHE_MANIFEST,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output_manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # Final verification
    # ========================================================

    print()
    print("=" * 80)
    print("CACHE SUMMARY")
    print("=" * 80)

    total_frames = 0

    for x in results:

        print(
            f"episode "
            f"{x['episode_index']:04d}: "
            f"{x['num_frames']} frames  "
            f"[{x['status']}]"
        )

        total_frames += int(
            x["num_frames"]
        )

    print()
    print(
        "total episodes:",
        len(results),
    )

    print(
        "total frames:",
        total_frames,
    )

    print(
        "cache manifest:",
        CACHE_MANIFEST,
    )

    print()
    print(
        "MULTI-EPISODE CACHE COMPLETE"
    )


if __name__ == "__main__":
    main()