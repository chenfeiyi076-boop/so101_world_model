from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from PIL import Image

from inspect_episode0_alignment import (
    REPO_ROOT,
    EPISODE_INDEX,
    VIDEO_KEY,
    load_episode_dataframe,
    load_episode_metadata,
    resolve_video_path,
    get_episode_video_interval,
    decode_episode_frames,
    to_vector,
)


# ============================================================
# Config
# ============================================================

MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"

IMAGE_SIZE = 256
BATCH_SIZE = 8

CACHE_DIR = Path(
    "data/latent_cache/so101_front"
)

CACHE_PATH = (
    CACHE_DIR / "episode_000.pt"
)

RECON_DIR = Path(
    "outputs/episode_000_vae_recon"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Image preprocessing
# ============================================================

def preprocess_batch(
    images: list[np.ndarray],
) -> torch.Tensor:
    """
    输入:
        list of RGB uint8 images
        each: [H, W, 3]

    输出:
        [B, 3, 256, 256]
        range [-1, 1]
    """

    x = np.stack(
        images,
        axis=0,
    )

    x = torch.from_numpy(
        x
    ).permute(
        0,
        3,
        1,
        2,
    ).float()

    x = x / 255.0

    _, _, h, w = x.shape

    # --------------------------------------------------------
    # resize shortest side -> 256
    # --------------------------------------------------------

    if min(h, w) != IMAGE_SIZE:

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

        x = F.interpolate(
            x,
            size=(
                new_h,
                new_w,
            ),
            mode="bilinear",
            align_corners=False,
        )

    # --------------------------------------------------------
    # center crop 256 × 256
    # --------------------------------------------------------

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
            f"Unexpected image shape: {x.shape}"
        )

    # --------------------------------------------------------
    # [0,1] -> [-1,1]
    # --------------------------------------------------------

    x = (
        x * 2.0
        - 1.0
    )

    return x


# ============================================================
# Encode
# ============================================================

@torch.no_grad()
def encode_images(
    vae: AutoencoderKL,
    images: list[np.ndarray],
) -> torch.Tensor:
    """
    第二篇 VAE 路径:

        RGB
        -> [-1,1]
        -> SD3 VAE
        -> posterior.sample()
        -> * scaling_factor

    返回:
        [N,16,32,32]
    """

    all_latents = []

    for start in range(
        0,
        len(images),
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            len(images),
        )

        batch_images = images[
            start:end
        ]

        x = preprocess_batch(
            batch_images
        )

        x = x.to(
            device=DEVICE,
            dtype=torch.bfloat16,
        )

        posterior = (
            vae.encode(x)
            .latent_dist
        )

        # ---------------------------------------------
        # 注意：
        # 第二篇官方代码训练时使用 sample()
        # ---------------------------------------------

        z = posterior.sample()

        z = (
            z
            * vae.config.scaling_factor
        )

        all_latents.append(
            z.cpu()
        )

        print(
            f"encoded "
            f"{end:4d}/{len(images)}"
        )

    return torch.cat(
        all_latents,
        dim=0,
    )


# ============================================================
# Decode representative cached latents
# ============================================================

@torch.no_grad()
def decode_and_save_checks(
    vae: AutoencoderKL,
    latents: torch.Tensor,
) -> None:

    RECON_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    indices = [
        0,
        len(latents) // 2,
        len(latents) - 1,
    ]

    for idx in indices:

        z = latents[
            idx : idx + 1
        ].to(
            DEVICE,
            dtype=torch.bfloat16,
        )

        # 与第二篇官方 VAE decoder 对应
        z = (
            z
            / vae.config.scaling_factor
        )

        reconstruction = vae.decode(
            z,
            return_dict=False,
        )[0]

        reconstruction = (
            reconstruction.float()
            .clamp(-1, 1)
            + 1.0
        ) / 2.0

        image = (
            reconstruction[0]
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        image = (
            image * 255.0
        ).round().clip(
            0,
            255,
        ).astype(
            np.uint8
        )

        Image.fromarray(
            image
        ).save(
            RECON_DIR
            / f"recon_{idx:04d}.png"
        )


# ============================================================
# Main
# ============================================================

def main():

    if DEVICE.type != "cuda":
        raise RuntimeError(
            "This pilot is intended to run on CUDA."
        )

    print("=" * 80)
    print("SO101 EPISODE 0 LATENT CACHE")
    print("=" * 80)

    print(
        "device:",
        DEVICE,
    )

    print(
        "gpu:",
        torch.cuda.get_device_name(0),
    )

    # ========================================================
    # Reproducible posterior sampling
    # ========================================================

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # ========================================================
    # 1. Load tabular episode
    # ========================================================

    df = load_episode_dataframe(
        REPO_ROOT,
        EPISODE_INDEX,
    )

    print(
        "episode rows:",
        len(df),
    )

    # ========================================================
    # 2. Resolve episode video
    # ========================================================

    meta_row = load_episode_metadata(
        REPO_ROOT,
        EPISODE_INDEX,
    )

    video_path = resolve_video_path(
        REPO_ROOT,
        meta_row,
    )

    start_time, end_time = (
        get_episode_video_interval(
            meta_row
        )
    )

    # ========================================================
    # 3. Decode exactly episode 0
    # ========================================================

    video_frames = decode_episode_frames(
        video_path=video_path,
        episode_start_time=start_time,
        episode_end_time=end_time,
        expected_frames=len(df),
    )

    if len(video_frames) != len(df):
        raise RuntimeError(
            "Video/tabular length mismatch: "
            f"video={len(video_frames)}, "
            f"table={len(df)}"
        )

    images = [
        image
        for _, image
        in video_frames
    ]

    video_timestamps = torch.tensor(
        [
            ts
            for ts, _
            in video_frames
        ],
        dtype=torch.float64,
    )

    print(
        "decoded RGB frames:",
        len(images),
    )

    # ========================================================
    # 4. Load frozen SD3 VAE
    # ========================================================

    print()
    print("Loading SD3 VAE...")

    vae = AutoencoderKL.from_pretrained(
        MODEL_ID,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    )

    vae.eval()
    vae.requires_grad_(False)
    vae.to(DEVICE)

    print(
        "latent channels:",
        vae.config.latent_channels,
    )

    print(
        "scaling factor:",
        vae.config.scaling_factor,
    )

    # ========================================================
    # 5. Encode all 510 frames
    # ========================================================

    latents = encode_images(
        vae,
        images,
    )

    print()
    print(
        "latent shape:",
        tuple(latents.shape),
    )

    print(
        "latent dtype:",
        latents.dtype,
    )

    print(
        "latent mean:",
        latents.float().mean().item(),
    )

    print(
        "latent std :",
        latents.float().std().item(),
    )

    if latents.shape != (
        len(df),
        16,
        32,
        32,
    ):
        raise RuntimeError(
            "Unexpected latent shape: "
            f"{tuple(latents.shape)}"
        )

    if not torch.isfinite(
        latents.float()
    ).all():

        raise RuntimeError(
            "NaN/Inf found in latents."
        )

    # ========================================================
    # 6. Action / state
    # ========================================================

    actions = torch.tensor(
        np.stack(
            [
                to_vector(x)
                for x in df["action"]
            ]
        ),
        dtype=torch.float32,
    )

    states = torch.tensor(
        np.stack(
            [
                to_vector(x)
                for x
                in df["observation.state"]
            ]
        ),
        dtype=torch.float32,
    )

    if "timestamp" in df.columns:

        timestamps = torch.tensor(
            df["timestamp"].to_numpy(),
            dtype=torch.float64,
        )

    else:

        timestamps = torch.arange(
            len(df),
            dtype=torch.float64,
        ) / 20.0

    if "frame_index" in df.columns:

        frame_indices = torch.tensor(
            df[
                "frame_index"
            ].to_numpy(),
            dtype=torch.long,
        )

    else:

        frame_indices = torch.arange(
            len(df),
            dtype=torch.long,
        )

    # ========================================================
    # 7. Save cache
    # ========================================================

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache = {
        "episode_index": EPISODE_INDEX,

        "latents": latents,

        "actions": actions,

        "states": states,

        "timestamps": timestamps,

        "video_timestamps": video_timestamps,

        "frame_indices": frame_indices,

        "camera": VIDEO_KEY,

        "vae_model": MODEL_ID,

        "vae_scaling_factor": float(
            vae.config.scaling_factor
        ),

        "vae_posterior": "sample",

        "image_size": IMAGE_SIZE,
    }

    torch.save(
        cache,
        CACHE_PATH,
    )

    print()
    print("=" * 80)
    print("CACHE SAVED")
    print("=" * 80)

    print(
        "path:",
        CACHE_PATH,
    )

    print(
        "latents:",
        tuple(latents.shape),
    )

    print(
        "actions:",
        tuple(actions.shape),
    )

    print(
        "states:",
        tuple(states.shape),
    )

    # ========================================================
    # 8. Decode representative cached latents
    # ========================================================

    decode_and_save_checks(
        vae,
        latents,
    )

    print(
        "reconstruction checks:",
        RECON_DIR,
    )


if __name__ == "__main__":
    main()