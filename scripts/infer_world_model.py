from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import autoregressive_rollout, load_world_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SO101 future latents with Flow Matching Euler rollout"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--legacy-config")
    parser.add_argument("--episode-cache", required=True)
    parser.add_argument("--start-frame", required=True, type=int)
    parser.add_argument("--rollout-frames", required=True, type=int)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--include-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save future ground-truth latents only as post-generation references",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    loaded = load_world_model(
        args.checkpoint,
        legacy_config_path=args.legacy_config,
        device=args.device,
    )
    cache = torch.load(args.episode_cache, map_location="cpu", weights_only=False)
    artifact = autoregressive_rollout(
        loaded=loaded,
        episode_cache=cache,
        start_frame=args.start_frame,
        rollout_frames=args.rollout_frames,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        include_reference=args.include_reference,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)

    metadata = artifact["metadata"]
    print("checkpoint_type:", artifact["checkpoint_type"])
    print("action_alignment:", metadata["action_alignment"])
    if (
        artifact["checkpoint_type"] == "legacy_v1"
        and metadata["action_alignment"] == "synchronized_legacy"
    ):
        print(
            "warning: synchronized_legacy reproduces the legacy training distribution; "
            "it is not the recommended causal alignment"
        )
    print("action_representation:", metadata["action_representation"])
    print("frame_indices:", artifact["frame_indices"].tolist())
    print("action_indices shape:", tuple(artifact["action_indices"].shape))
    print("generated_latents shape:", tuple(artifact["generated_latents"].shape))
    print("num_inference_steps:", artifact["num_inference_steps"])
    print("output:", output)


if __name__ == "__main__":
    main()
