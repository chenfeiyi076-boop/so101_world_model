from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.state_dynamics.checkpointing import load_checkpoint
from src.state_dynamics.dataset import SO101StateStore, StateDynamicsWindowDataset
from src.state_dynamics.metrics import (
    summarize_rollout_per_episode, summarize_rollout_rows, summarize_teacher_forced,
)
from src.state_dynamics.model import StateDynamicsMLP
from src.state_dynamics.rollout import autoregressive_state_predictions, rollout_metric_rows
from src.state_dynamics.split import (
    episode_ids_for_split, file_sha256, load_split_manifest, validate_episode_tasks,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Teacher-forced and 5 Hz autoregressive evaluation of SO101 state MLP"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--max-episodes", type=int, default=0)
    return parser


def write_csv(path: Path, rows, fields) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.max_episodes < 0:
        raise ValueError("invalid batch/workers/max-episodes")
    checkpoint = load_checkpoint(args.checkpoint)
    config, stats = checkpoint["config"], checkpoint["normalization_stats"]
    manifest_path = args.split_manifest or Path(checkpoint["split_manifest_path"])
    if file_sha256(manifest_path) != checkpoint["split_manifest_sha256"]:
        raise RuntimeError("split manifest differs from checkpoint provenance")
    manifest = load_split_manifest(manifest_path)
    if checkpoint["dataset_identity"] != manifest["dataset_identity"]:
        raise RuntimeError("checkpoint and split manifest dataset identities differ")
    ids = episode_ids_for_split(manifest, args.split)
    if args.max_episodes > 0:
        ids = ids[: args.max_episodes]
    store = SO101StateStore(args.dataset_root)
    episodes = store.load_episodes(ids)
    validate_episode_tasks(
        manifest, {episode_id: episode.task_id for episode_id, episode in episodes.items()}
    )
    dataset = StateDynamicsWindowDataset(
        episodes, episode_order=ids, horizon=config["data"]["prediction_horizon"],
        window_stride=config["data"]["window_stride"],
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = StateDynamicsMLP(
        state_dim=config["data"]["state_dim"], action_dim=config["data"]["action_dim"],
        horizon=config["data"]["prediction_horizon"], **config["model"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True); model.eval()
    predictions, targets = [], []
    normalized_squared_sum, normalized_elements = 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            current = stats.normalize_states(batch["current_state"].float()).to(device)
            actions = stats.normalize_actions(batch["actions"].float()).to(device)
            target_normalized = stats.normalize_states(batch["target_states"].float()).to(device)
            predicted_normalized = model(current, actions)
            normalized_squared_sum += float((predicted_normalized - target_normalized).square().sum().cpu())
            normalized_elements += predicted_normalized.numel()
            predictions.append(stats.denormalize_states(predicted_normalized).cpu())
            targets.append(batch["target_states"].float())
    teacher_summary, horizon_rows, joint_rows = summarize_teacher_forced(
        torch.cat(predictions), torch.cat(targets)
    )
    teacher_summary.update({
        "split": args.split,
        "normalized_mse": normalized_squared_sum / normalized_elements,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
    })

    rollout_rows = []
    for episode_id in ids:
        episode = episodes[episode_id]
        prediction = autoregressive_state_predictions(
            model=model, initial_state=episode.states[0], actions=episode.actions,
            stats=stats, device=device,
        )
        if prediction["reference_used_as_model_input"] is not False:
            raise RuntimeError("autoregressive rollout violated GT leakage invariant")
        rollout_rows.extend(rollout_metric_rows(
            episode_id=episode_id, prediction=prediction,
            states=episode.states, actions=episode.actions,
        ))
    rollout_summary = summarize_rollout_rows(rollout_rows)
    per_episode_rows, episode_macro_metrics = summarize_rollout_per_episode(rollout_rows)
    rollout_summary.update(episode_macro_metrics)
    rollout_summary.update({
        "split": args.split, "num_episodes": len(ids),
        "rollout_frequency_hz": 5.0, "seconds_per_step": 0.2,
        "future_gt_used_as_model_input": False,
    })
    public_rollout_rows = [
        {key: row[key] for key in (
            "episode_id", "rollout_step", "horizon_seconds", "target_frame",
            "mae", "rmse", "command_copy_endpoint_mae", "command_copy_endpoint_rmse",
        )} for row in rollout_rows
    ]
    by_step = defaultdict(list)
    for row in public_rollout_rows:
        by_step[int(row["rollout_step"])].append(row)
    time_rows = []
    for step, values in sorted(by_step.items()):
        time_rows.append({
            "rollout_step": step, "horizon_seconds": step * 0.2,
            "n_episodes": len(values),
            "mae_mean": sum(float(v["mae"]) for v in values) / len(values),
            "rmse_mean": sum(float(v["rmse"]) for v in values) / len(values),
            "command_copy_mae_mean": sum(
                float(v["command_copy_endpoint_mae"]) for v in values
            ) / len(values),
            "command_copy_rmse_mean": sum(
                float(v["command_copy_endpoint_rmse"]) for v in values
            ) / len(values),
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "teacher_forced_summary.json", teacher_summary)
    write_csv(args.output_dir / "teacher_forced_per_horizon.csv", horizon_rows,
              ("horizon", "horizon_seconds", "mae", "rmse"))
    write_csv(args.output_dir / "teacher_forced_per_joint.csv", joint_rows,
              ("joint", "mae", "rmse"))
    write_json(args.output_dir / "autoregressive_summary.json", rollout_summary)
    write_csv(args.output_dir / "autoregressive_per_step.csv", public_rollout_rows,
              ("episode_id", "rollout_step", "horizon_seconds", "target_frame", "mae", "rmse",
               "command_copy_endpoint_mae", "command_copy_endpoint_rmse"))
    write_csv(args.output_dir / "autoregressive_per_episode.csv", per_episode_rows,
              ("episode_id", "num_rollout_steps", "mean_endpoint_mae",
               "mean_endpoint_rmse", "final_endpoint_mae", "final_endpoint_rmse"))
    write_csv(args.output_dir / "autoregressive_error_by_time.csv", time_rows,
              ("rollout_step", "horizon_seconds", "n_episodes", "mae_mean", "rmse_mean",
               "command_copy_mae_mean", "command_copy_rmse_mean"))
    write_csv(args.output_dir / "autoregressive_per_joint.csv", rollout_summary["per_joint"],
              ("joint", "mae", "rmse"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        [row["horizon_seconds"] for row in time_rows],
        [row["mae_mean"] for row in time_rows],
        label="MLP autoregressive endpoint MAE",
    )
    axis.plot(
        [row["horizon_seconds"] for row in time_rows],
        [row["command_copy_mae_mean"] for row in time_rows],
        label="Command-copy endpoint MAE",
    )
    axis.set(xlabel="Rollout horizon (seconds)", ylabel="Raw-state MAE")
    axis.grid(alpha=0.3); axis.legend(); figure.tight_layout()
    figure.savefig(args.output_dir / "autoregressive_error_vs_time.png", dpi=160)
    plt.close(figure)
    print("teacher-forced overall MAE:", teacher_summary["overall_mae"])
    print("teacher-forced overall RMSE:", teacher_summary["overall_rmse"])
    print("autoregressive endpoint-micro MAE:", rollout_summary["autoregressive_endpoint_micro_mae"])
    print("autoregressive endpoint-micro RMSE:", rollout_summary["autoregressive_endpoint_micro_rmse"])
    print("episode-macro mean-endpoint MAE:",
          rollout_summary["autoregressive_episode_macro_mean_endpoint_mae"])
    print("episode-macro mean-endpoint RMSE:",
          rollout_summary["autoregressive_episode_macro_mean_endpoint_rmse"])
    print("command-copy endpoint-micro MAE:",
          rollout_summary["command_copy_endpoint_micro_mae"])
    print("command-copy endpoint-micro RMSE:",
          rollout_summary["command_copy_endpoint_micro_rmse"])
    print("output:", args.output_dir)


if __name__ == "__main__":
    main()
