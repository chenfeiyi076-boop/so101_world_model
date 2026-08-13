from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.so101_cache.vae_cache import (
    LATENT_CONVENTION,
    resolve_vae_dtype,
    validate_cache_payload,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate one SO101 raw-20Hz latent cache and decode first/middle/last "
            "without applying SD3 shift_factor."
        )
    )
    parser.add_argument("--cache-file", type=Path, required=True)
    parser.add_argument("--vae-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {value!r} was requested but CUDA is unavailable")
    return device


def load_vae(vae_root: Path, device: torch.device):
    from diffusers import AutoencoderKL

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
    return vae


@torch.inference_mode()
def decode_representative_latents(
    cache: dict,
    vae,
    device: torch.device,
    output_dir: Path,
) -> None:
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    latents = torch.as_tensor(cache["latents"])
    actions = torch.as_tensor(cache["actions"])
    states = torch.as_tensor(cache["states"])
    timestamps = torch.as_tensor(cache["timestamps"], dtype=torch.float64)
    frame_indices = torch.as_tensor(cache["frame_indices"], dtype=torch.long)
    episode_index = int(cache["episode_index"])
    selections = {
        "first": 0,
        "middle": len(latents) // 2,
        "last": len(latents) - 1,
    }
    scaling_factor = float(vae.config.scaling_factor)
    vae_dtype = resolve_vae_dtype(device)
    for label, index in selections.items():
        z = latents[index : index + 1].to(device=device, dtype=vae_dtype)
        # Inverse of posterior_sample_times_scaling_no_shift.  shift_factor is
        # deliberately ignored for this project convention.
        z_raw = z / scaling_factor
        reconstruction = vae.decode(z_raw).sample
        image_tensor = (
            reconstruction[0].float().clamp(-1.0, 1.0).add(1.0).div(2.0)
        )
        image_array = (
            image_tensor.permute(1, 2, 0)
            .mul(255.0)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        output_path = output_dir / f"{label}.png"
        Image.fromarray(image_array, mode="RGB").save(output_path)
        print(f"[{label}] episode_index={episode_index}")
        print(f"  cache_index={index}")
        print(f"  frame_index={int(frame_indices[index])}")
        print(f"  timestamp={float(timestamps[index]):.9f}")
        print(f"  action={actions[index].tolist()}")
        print(f"  state={states[index].tolist()}")
        print(f"  png={output_path}")


def main() -> None:
    args = build_parser().parse_args()
    cache = torch.load(args.cache_file, map_location="cpu", weights_only=False)
    summary = validate_cache_payload(cache)
    metadata = cache["metadata"]
    if metadata.get("latent_convention") != LATENT_CONVENTION:
        raise RuntimeError("cache latent convention is incompatible with this verifier")
    print("CACHE VALIDATION PASS")
    for key, value in summary.items():
        print(f"{key}: {value}")

    device = resolve_device(args.device)
    vae = load_vae(args.vae_root, device)
    cached_scaling = float(metadata["scaling_factor"])
    vae_scaling = float(vae.config.scaling_factor)
    if abs(cached_scaling - vae_scaling) > 1e-7:
        raise RuntimeError(
            f"cache/VAE scaling_factor mismatch: {cached_scaling} != {vae_scaling}"
        )
    print(f"VAE scaling_factor: {vae_scaling}")
    print(f"VAE shift_factor: IGNORED ({getattr(vae.config, 'shift_factor', None)})")
    decode_representative_latents(cache, vae, device, args.output_dir)
    print("DECODE VALIDATION COMPLETE")


if __name__ == "__main__":
    main()
