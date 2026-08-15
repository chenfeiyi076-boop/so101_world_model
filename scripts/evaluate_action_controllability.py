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

from scripts.evaluate_causal import evaluation_precision, select_evaluation_state_dict
from src.causal.action_counterfactual import (
    METRIC_INTERPRETATION,
    SHUFFLE_TYPE,
    aggregate_per_case,
    aggregate_per_rollout,
    aggregate_per_step,
    build_summary_metrics,
    paired_action_counterfactual_rollout_batch,
    paired_rollout_step_rows,
    summarize_true_rerun_warnings,
    validate_selected_case_split_membership,
    validate_source_checkpoint_compatibility,
    validate_true_rerun,
)
from src.causal.checkpointing import action_stats_from_checkpoint, load_checkpoint
from src.causal.data.common import cache_paths_from_manifest, load_episode_cache
from src.causal.data.multi_episode_dataset import (
    episode_ids_for_split,
    load_causal_manifest,
)
from src.causal.rollout import RolloutStream
from src.causal.rollout_visualization import (
    group_stream_rows,
    read_rollout_metric_rows,
)
from src.causal.runtime import build_model, validate_precision_device


PER_ROLLOUT_STEP_FIELDS = (
    "episode_id",
    "start",
    "noise_draw",
    "step",
    "target_frame_index",
    "shuffle_source_transition",
    "true_mse",
    "shuffle_mse",
    "delta_mse",
    "true_relative_l2",
    "shuffle_relative_l2",
    "true_cosine_similarity",
    "shuffle_cosine_similarity",
    "prediction_divergence_mse",
    "prediction_divergence_relative_l2",
    "prediction_divergence_cosine",
    "true_better",
)
PER_ROLLOUT_FIELDS = (
    "episode_id",
    "start",
    "noise_draw",
    "mean_true_mse",
    "mean_shuffle_mse",
    "mean_delta_mse",
    "final_true_mse",
    "final_shuffle_mse",
    "final_delta_mse",
    "mean_prediction_divergence_mse",
    "true_better_fraction_steps",
)
PER_STEP_FIELDS = (
    "step",
    "time_sec",
    "true_mse_mean",
    "true_mse_median",
    "true_mse_p90",
    "shuffle_mse_mean",
    "shuffle_mse_median",
    "shuffle_mse_p90",
    "delta_mse_mean",
    "delta_mse_median",
    "delta_mse_p10",
    "delta_mse_p90",
    "true_better_rate",
    "prediction_divergence_mse_mean",
    "prediction_divergence_mse_median",
    "prediction_divergence_mse_p90",
)
PER_CASE_FIELDS = (
    "episode_id",
    "start",
    "num_noise_draws",
    "mean_true_mse",
    "mean_shuffle_mse",
    "mean_delta_mse",
    "mean_prediction_divergence_mse",
    "true_better_draw_rate",
    "fixed_points",
    "permutation",
    "modified_raw_action_start",
    "modified_raw_action_end",
    "history_action_unchanged",
    "normalized_action_chunk_l2_difference_mean",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate factual actions against a temporally shuffled future-action "
            "negative control"
        )
    )
    parser.add_argument("--source-rollout-eval-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shuffle-seed", type=int, default=30360)
    parser.add_argument("--pair-batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=4242)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    return parser


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return value


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            serialized = {
                key: json.dumps(value) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            writer.writerow(serialized)


def _validate_source_streams(
    summary: dict, grouped_rows: dict[tuple[int, int, int], list[dict]]
) -> list[tuple[int, int]]:
    cases = [
        (int(item["episode_id"]), int(item["start"]))
        for item in summary["selected_rollout_cases"]
    ]
    if len(cases) != len(set(cases)):
        raise RuntimeError("source selected_rollout_cases contains duplicates")
    expected = {
        (episode_id, start, noise_draw)
        for episode_id, start in cases
        for noise_draw in range(int(summary["noise_draws"]))
    }
    if set(grouped_rows) != expected:
        missing = sorted(expected - set(grouped_rows))
        extra = sorted(set(grouped_rows) - expected)
        raise RuntimeError(f"source stochastic stream mismatch: missing={missing}, extra={extra}")
    expected_steps = list(range(1, int(summary["rollout_steps"]) + 1))
    for identity, rows in grouped_rows.items():
        steps = [int(row["step"]) for row in rows]
        if steps != expected_steps:
            raise RuntimeError(f"source rollout steps mismatch for {identity}")
    return cases


def main() -> None:
    args = build_parser().parse_args()
    if args.pair_batch_size <= 0:
        raise ValueError("pair-batch-size must be positive")
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")

    source_summary = _load_json(args.source_rollout_eval_dir / "summary.json")
    source_rows = read_rollout_metric_rows(
        args.source_rollout_eval_dir / "per_rollout_step.csv"
    )
    grouped_source = group_stream_rows(source_rows)
    cases = _validate_source_streams(source_summary, grouped_source)
    rollout_steps = int(source_summary["rollout_steps"])
    if rollout_steps < 2:
        raise ValueError("temporal shuffle requires rollout_steps >= 2")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    checkpoint = load_checkpoint(source_summary["checkpoint_path"])
    config = checkpoint["config"]
    precision = evaluation_precision(config)
    state_dict, selected_weights = select_evaluation_state_dict(
        checkpoint, source_summary["weights_used"]
    )
    validate_source_checkpoint_compatibility(
        source_summary,
        checkpoint,
        precision=precision,
        weights_used=selected_weights,
    )
    validate_precision_device(precision, device)

    model = build_model(config).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    action_stats = action_stats_from_checkpoint(checkpoint)
    manifest = load_causal_manifest(config["data"]["manifest_path"])
    split_episode_ids = episode_ids_for_split(manifest, source_summary["split"])
    validate_selected_case_split_membership(
        source_summary["selected_rollout_cases"],
        split=source_summary["split"],
        split_episode_ids=split_episode_ids,
    )
    cache_paths = cache_paths_from_manifest(
        manifest, sorted({episode_id for episode_id, _ in cases})
    )
    episode_cache = {
        episode_id: load_episode_cache(
            cache_paths[episode_id], config["action"]["raw_action_dim"]
        )
        for episode_id in sorted(cache_paths)
    }
    base_streams = [
        RolloutStream(
            episode=episode_cache[episode_id],
            episode_id=episode_id,
            start=start,
            noise_draw=noise_draw,
        )
        for episode_id, start in cases
        for noise_draw in range(int(source_summary["noise_draws"]))
    ]

    step_rows = []
    audit_by_case = {}
    true_rerun_warnings = []
    number_of_batches = (
        len(base_streams) + args.pair_batch_size - 1
    ) // args.pair_batch_size
    for batch_index, offset in enumerate(
        range(0, len(base_streams), args.pair_batch_size), start=1
    ):
        batch = base_streams[offset : offset + args.pair_batch_size]
        batch_results = paired_action_counterfactual_rollout_batch(
            model=model,
            config=config,
            action_stats=action_stats,
            base_streams=batch,
            rollout_steps=rollout_steps,
            euler_steps=int(source_summary["euler_steps"]),
            seed=int(source_summary["seed"]),
            shuffle_seed=args.shuffle_seed,
            device=device,
            precision=precision,
        )
        for pair in batch_results:
            identity = (
                int(pair["true"]["episode_id"]),
                int(pair["true"]["start"]),
                int(pair["true"]["noise_draw"]),
            )
            true_rerun_warnings.append(
                validate_true_rerun(pair["true"], grouped_source[identity])
            )
            audit = pair["audit"]
            case_identity = (int(audit["episode_id"]), int(audit["start"]))
            if (
                case_identity in audit_by_case
                and audit_by_case[case_identity] != audit
            ):
                raise RuntimeError("shuffle audit changed across noise draws")
            audit_by_case[case_identity] = audit
        step_rows.extend(paired_rollout_step_rows(batch_results))
        print(
            f"completed pair batch {batch_index}/{number_of_batches}: "
            f"{len(batch)} base streams, {2 * len(batch)} model variants",
            flush=True,
        )

    rollout_rows = aggregate_per_rollout(step_rows)
    audits = [audit_by_case[key] for key in sorted(audit_by_case)]
    case_rows = aggregate_per_case(rollout_rows, audits=audits)
    per_step = aggregate_per_step(
        step_rows, frame_stride=int(source_summary["frame_stride"])
    )
    summary_metrics = build_summary_metrics(
        rollout_rows,
        case_rows,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    (
        true_rerun_metric_warning_count,
        true_rerun_streams_with_warning,
    ) = summarize_true_rerun_warnings(true_rerun_warnings)
    summary = {
        "source_rollout_eval_dir": str(args.source_rollout_eval_dir),
        "checkpoint_path": source_summary["checkpoint_path"],
        "checkpoint_step": int(source_summary["checkpoint_step"]),
        "weights_used": source_summary["weights_used"],
        "split": source_summary["split"],
        "precision": source_summary["precision"],
        "num_frames": int(source_summary["num_frames"]),
        "num_history": int(source_summary["num_history"]),
        "frame_stride": int(source_summary["frame_stride"]),
        "action_representation": source_summary["action_representation"],
        "effective_action_dim": int(source_summary["effective_action_dim"]),
        "rollout_steps": rollout_steps,
        "noise_draws": int(source_summary["noise_draws"]),
        "euler_steps": int(source_summary["euler_steps"]),
        "seed": int(source_summary["seed"]),
        "shuffle_seed": int(args.shuffle_seed),
        "shuffle_type": SHUFFLE_TYPE,
        "num_cases": len(cases),
        "num_stochastic_rollouts": len(base_streams),
        "pair_batch_size": int(args.pair_batch_size),
        "effective_variant_batch_size_max": min(
            2 * args.pair_batch_size, 2 * len(base_streams)
        ),
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "true_rerun_metric_warning_count": true_rerun_metric_warning_count,
        "true_rerun_streams_with_warning": true_rerun_streams_with_warning,
        "counterfactual_gt_available": False,
        "reference_used_as_model_input": False,
        "metric_interpretation": METRIC_INTERPRETATION,
        "shuffle_audits": audits,
        **summary_metrics,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "per_step.csv", per_step, PER_STEP_FIELDS)
    _write_csv(args.output_dir / "per_rollout.csv", rollout_rows, PER_ROLLOUT_FIELDS)
    _write_csv(
        args.output_dir / "per_rollout_step.csv",
        step_rows,
        PER_ROLLOUT_STEP_FIELDS,
    )
    _write_csv(args.output_dir / "per_case.csv", case_rows, PER_CASE_FIELDS)
    _write_json(args.output_dir / "summary.json", summary)
    print("overall true MSE:", summary["overall_true_mse_mean"])
    print("overall shuffle MSE:", summary["overall_shuffle_mse_mean"])
    print("overall delta MSE:", summary["overall_delta_mse_mean"])
    print(
        "case delta 95% CI:",
        summary["case_delta_ci95_low"],
        summary["case_delta_ci95_high"],
    )
    print("output:", args.output_dir)


if __name__ == "__main__":
    main()
