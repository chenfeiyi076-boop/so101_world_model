from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.causal.checkpointing import load_checkpoint, validate_requested_config
from src.causal.config import load_and_resolve_config
from src.causal.datasets import build_evaluation_dataset
from src.causal.runtime import (
    build_model,
    evaluate_flow_loss,
    validate_precision_device,
)


def evaluation_precision(config: dict) -> str:
    """Treat historical causal checkpoints without precision as FP32."""

    return config["train"].get("precision", "fp32")


def select_evaluation_state_dict(
    checkpoint: dict,
    weights: str,
) -> tuple[dict, str]:
    if weights not in {"auto", "ema", "raw"}:
        raise ValueError("weights must be auto, ema, or raw")
    has_ema = "ema_model_state_dict" in checkpoint
    if weights == "ema" and not has_ema:
        raise RuntimeError("--weights ema requested but checkpoint has no EMA weights")
    if weights == "ema" or (weights == "auto" and has_ema):
        return checkpoint["ema_model_state_dict"], "ema"
    return checkpoint["model_state_dict"], "raw"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a causal checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", help="optional config, used only for strict validation")
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--max-windows", type=int, default=128)
    parser.add_argument("--noise-draws", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--weights", default="auto", choices=("auto", "ema", "raw"))
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    if args.config:
        validate_requested_config(
            checkpoint, load_and_resolve_config(args.config)
        )
    config = checkpoint["config"]  # checkpoint is the source of truth
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    precision = evaluation_precision(config)
    validate_precision_device(precision, device)

    dataset = build_evaluation_dataset(checkpoint, split=args.split)
    if args.max_windows > 0 and args.max_windows < len(dataset):
        positions = (
            torch.linspace(0, len(dataset) - 1, steps=args.max_windows)
            .round()
            .long()
            .unique()
            .tolist()
        )
        dataset = Subset(dataset, positions)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(config).to(device)
    state_dict, selected_weights = select_evaluation_state_dict(
        checkpoint, args.weights
    )
    model.load_state_dict(state_dict)
    values = []
    for draw in range(args.noise_draws):
        values.append(
            evaluate_flow_loss(
                model,
                loader,
                device=device,
                num_history=config["temporal"]["num_history"],
                seed=20260 + draw,
                precision=precision,
            )
        )
    print("checkpoint:", args.checkpoint)
    print("split:", args.split)
    print("frame_stride:", config["temporal"]["frame_stride"])
    print("representation:", config["action"]["representation"])
    print("effective_action_dim:", config["action"]["effective_action_dim"])
    print("precision:", precision)
    print("weights:", selected_weights)
    print("action_mean (checkpoint):", checkpoint["action_stats"]["mean"])
    print("action_std (checkpoint):", checkpoint["action_stats"]["std"])
    print("mean FM loss:", sum(values) / len(values))


if __name__ == "__main__":
    main()
