from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_causal import (
    evaluation_precision,
    select_evaluation_state_dict,
)
from src.causal.checkpointing import action_stats_from_checkpoint, load_checkpoint
from src.causal.data.common import load_episode_cache
from src.causal.rollout import (
    RolloutStream,
    aggregate_rollout_metrics,
    autoregressive_causal_rollout_batch,
    build_rollout_catalog,
    rollout_metric_rows,
    select_rollout_cases,
    threshold_horizon,
)
from src.causal.runtime import build_model, validate_precision_device


PER_ROLLOUT_FIELDS = (
    "episode_id",
    "start",
    "noise_draw",
    "step",
    "target_frame_index",
    "mse",
    "rmse",
    "mae",
    "relative_l2",
    "cosine_similarity",
)

PER_STEP_FIELDS = (
    "step",
    "raw_frame_offset",
    "num_samples",
    "mse_mean",
    "mse_std",
    "mse_median",
    "mse_p90",
    "rmse_mean",
    "mae_mean",
    "relative_l2_mean",
    "cosine_similarity_mean",
    "cumulative_mse_mean",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate causal_v2 autoregressive latent rollout error"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--euler-steps", type=int, default=10)
    parser.add_argument("--max-rollouts", type=int, default=128)
    parser.add_argument("--noise-draws", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--weights", default="auto", choices=("auto", "ema", "raw"))
    parser.add_argument("--error-threshold", type=float)
    parser.add_argument(
        "--threshold-metric",
        default="mse",
        choices=("mse", "mse_p90", "relative_l2"),
    )
    parser.add_argument("--save-latents", action="store_true")
    parser.add_argument("--max-saved-rollouts", type=int, default=4)
    return parser


def _write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save_debug_rollout(output_dir: Path, result: dict, ordinal: int) -> None:
    debug_dir = output_dir / "latents"
    debug_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "episode_id": result["episode_id"],
            "start": result["start"],
            "noise_draw": result["noise_draw"],
            "history_latents": result["history_latents"],
            "predicted_future": result["predicted_future"],
            "gt_future": result["gt_future"],
            "history_frame_indices": result["history_frame_indices"],
            "target_frame_indices": result["target_frame_indices"],
            "window_frame_indices": result["window_frame_indices"],
            "action_indices": result["action_indices"],
            "action_valid_masks": result["action_valid_masks"],
            "initial_noises": result["initial_noises"],
        },
        debug_dir / f"rollout_{ordinal:04d}.pt",
    )


def _print_summary(summary: dict, per_step: list[dict]) -> None:
    print("checkpoint:", summary["checkpoint_path"])
    print("weights:", summary["weights_used"])
    print("split:", summary["split"])
    print("rollout cases:", summary["number_of_rollout_cases"])
    print("rollout steps:", summary["rollout_steps"])
    print("frame stride:", summary["frame_stride"])
    print("Euler steps:", summary["euler_steps"])
    print()
    print("step | mse_mean | mse_p90 | rel_l2 | cosine")
    for row in per_step:
        print(
            f"{row['step']:>4} | {row['mse_mean']:.6g} | "
            f"{row['mse_p90']:.6g} | {row['relative_l2_mean']:.6g} | "
            f"{row['cosine_similarity_mean']:.6g}"
        )
    if summary["threshold_value"] is not None:
        print()
        print("threshold metric:", summary["threshold_metric"])
        print("threshold:", summary["threshold_value"])
        print(
            "max supported horizon:",
            summary["max_supported_horizon"],
            "sampled frames",
        )
        print("first crossing:", summary["first_threshold_crossing_step"])


def main() -> None:
    args = build_parser().parse_args()
    if args.rollout_steps <= 0 or args.euler_steps <= 0:
        raise ValueError("rollout/euler steps must be positive")
    if args.noise_draws <= 0:
        raise ValueError("noise-draws must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.max_saved_rollouts < 0:
        raise ValueError("max-saved-rollouts must be non-negative")

    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    if config["data"]["mode"] != "multi_episode":
        raise ValueError("rollout evaluation requires a multi_episode checkpoint")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    precision = evaluation_precision(config)
    validate_precision_device(precision, device)

    model = build_model(config).to(device)
    state_dict, selected_weights = select_evaluation_state_dict(
        checkpoint, args.weights
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    action_stats = action_stats_from_checkpoint(checkpoint)
    temporal = config["temporal"]
    action = config["action"]
    catalog = build_rollout_catalog(
        config["data"]["manifest_path"],
        split=args.split,
        num_history=temporal["num_history"],
        frame_stride=temporal["frame_stride"],
        rollout_steps=args.rollout_steps,
        raw_action_dim=action["raw_action_dim"],
    )
    cases = select_rollout_cases(catalog, args.max_rollouts)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    saved = 0
    episode_cache = {}
    streams = []
    for case in cases:
        if case.episode_id not in episode_cache:
            episode_cache[case.episode_id] = load_episode_cache(
                case.cache_path, action["raw_action_dim"]
            )
        for noise_draw in range(args.noise_draws):
            streams.append(
                RolloutStream(
                    episode=episode_cache[case.episode_id],
                    episode_id=case.episode_id,
                    start=case.start,
                    noise_draw=noise_draw,
                )
            )

    number_of_batches = (len(streams) + args.batch_size - 1) // args.batch_size
    for batch_index, offset in enumerate(
        range(0, len(streams), args.batch_size), start=1
    ):
        batch_streams = streams[offset : offset + args.batch_size]
        results = autoregressive_causal_rollout_batch(
            model=model,
            config=config,
            action_stats=action_stats,
            streams=batch_streams,
            rollout_steps=args.rollout_steps,
            euler_steps=args.euler_steps,
            seed=args.seed,
            device=device,
            precision=precision,
        )
        for result in results:
            metric_rows.extend(rollout_metric_rows(result))
            if args.save_latents and saved < args.max_saved_rollouts:
                _save_debug_rollout(args.output_dir, result, saved)
                saved += 1
        print(
            f"completed rollout batch {batch_index}/{number_of_batches}: "
            f"{len(batch_streams)} streams",
            flush=True,
        )

    per_step = aggregate_rollout_metrics(
        metric_rows,
        rollout_steps=args.rollout_steps,
        frame_stride=temporal["frame_stride"],
    )
    max_horizon, first_crossing = threshold_horizon(
        per_step,
        threshold=args.error_threshold,
        metric=args.threshold_metric,
    )
    summary = {
        "checkpoint_path": str(Path(args.checkpoint)),
        "checkpoint_step": int(checkpoint["step"]),
        "git_commit": checkpoint.get("git_commit", "unknown"),
        "weights_used": selected_weights,
        "split": args.split,
        "precision": precision,
        "num_frames": int(temporal["num_frames"]),
        "num_history": int(temporal["num_history"]),
        "frame_stride": int(temporal["frame_stride"]),
        "action_representation": action["representation"],
        "effective_action_dim": int(action["effective_action_dim"]),
        "rollout_steps": args.rollout_steps,
        "euler_steps": args.euler_steps,
        "noise_draws": args.noise_draws,
        "batch_size": args.batch_size,
        "number_of_batches": number_of_batches,
        "max_effective_batch_size": min(args.batch_size, len(streams)),
        "seed": args.seed,
        "number_of_rollout_cases": len(cases),
        "number_of_total_stochastic_rollouts": len(cases) * args.noise_draws,
        "threshold_metric": args.threshold_metric,
        "threshold_value": args.error_threshold,
        "max_supported_horizon": max_horizon,
        "first_threshold_crossing_step": first_crossing,
        "selected_rollout_cases": [
            {"episode_id": case.episode_id, "start": case.start}
            for case in cases
        ],
        "reference_used_as_model_input": False,
    }
    _write_csv(args.output_dir / "per_step.csv", per_step, PER_STEP_FIELDS)
    _write_csv(
        args.output_dir / "per_rollout_step.csv",
        metric_rows,
        PER_ROLLOUT_FIELDS,
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _print_summary(summary, per_step)


if __name__ == "__main__":
    main()
