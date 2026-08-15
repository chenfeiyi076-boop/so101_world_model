from __future__ import annotations

import hashlib
import math
import warnings
from collections import defaultdict
from typing import Any, Sequence

import torch

from src.so101_cache.vae_cache import FPS, LATENT_CONVENTION

from .data.common import ActionStats
from .rollout import (
    RolloutStream,
    autoregressive_causal_rollout_batch,
    deterministic_rollout_noise,
    latent_error_metrics,
)


SHUFFLE_TYPE = "temporal_chunk_derangement"
METRIC_INTERPRETATION = (
    "factual-consistency degradation under temporally shuffled future actions"
)
SOURCE_COMPATIBILITY_FIELDS = (
    "checkpoint_step",
    "precision",
    "weights_used",
    "num_frames",
    "num_history",
    "frame_stride",
    "action_representation",
    "effective_action_dim",
)


def future_action_spans(
    *, start: int, num_history: int, frame_stride: int, rollout_steps: int
) -> dict[str, Any]:
    if start < 0 or num_history <= 0 or frame_stride <= 0:
        raise ValueError("start/history/stride are invalid")
    if rollout_steps < 2:
        raise ValueError("temporal shuffle requires rollout_steps >= 2")
    last_history = start + (num_history - 1) * frame_stride
    return {
        "history_action_start": start,
        "history_action_end": last_history,
        "modified_raw_action_start": last_history,
        "modified_raw_action_end": last_history + rollout_steps * frame_stride,
        "future_chunk_spans": [
            (
                last_history + index * frame_stride,
                last_history + (index + 1) * frame_stride,
            )
            for index in range(rollout_steps)
        ],
    }


def _derangement_seed(shuffle_seed: int, episode_id: int, start: int) -> int:
    payload = f"{shuffle_seed}:{episode_id}:{start}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (
        2**63 - 1
    )


def deterministic_derangement(
    *,
    length: int,
    shuffle_seed: int,
    episode_id: int,
    start: int,
    max_retries: int = 128,
) -> torch.Tensor:
    if length < 2:
        raise ValueError("temporal shuffle requires rollout_steps >= 2")
    if max_retries <= 0:
        raise ValueError("max_retries must be positive")
    seed = _derangement_seed(shuffle_seed, episode_id, start)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    identity = torch.arange(length, dtype=torch.long)
    for _ in range(max_retries):
        permutation = torch.randperm(length, generator=generator)
        if torch.all(permutation != identity):
            return permutation
    # Every non-zero cyclic shift is a derangement. Derive the fallback offset
    # from the same case-level seed so it remains deterministic.
    offset = seed % (length - 1) + 1
    return (identity + int(offset)) % length


def temporally_shuffled_episode(
    episode: dict[str, Any],
    *,
    episode_id: int,
    start: int,
    num_history: int,
    frame_stride: int,
    rollout_steps: int,
    shuffle_seed: int,
    action_stats: ActionStats | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    spans = future_action_spans(
        start=start,
        num_history=num_history,
        frame_stride=frame_stride,
        rollout_steps=rollout_steps,
    )
    actions = torch.as_tensor(episode["actions"], dtype=torch.float32)
    modified_start = int(spans["modified_raw_action_start"])
    modified_end = int(spans["modified_raw_action_end"])
    if modified_end > len(actions):
        raise IndexError("future action shuffle span exceeds episode boundary")
    permutation = deterministic_derangement(
        length=rollout_steps,
        shuffle_seed=shuffle_seed,
        episode_id=episode_id,
        start=start,
    )
    future_chunks = actions[modified_start:modified_end].reshape(
        rollout_steps, frame_stride, actions.shape[-1]
    )
    modified_actions = actions.clone()
    modified_actions[modified_start:modified_end] = future_chunks[
        permutation
    ].reshape(-1, actions.shape[-1])
    counterfactual_episode = dict(episode)
    counterfactual_episode["actions"] = modified_actions

    true_chunks = future_chunks
    shuffled_chunks = future_chunks[permutation]
    if action_stats is not None:
        true_chunks = (true_chunks - action_stats.mean) / action_stats.std
        shuffled_chunks = (shuffled_chunks - action_stats.mean) / action_stats.std
    difference = (true_chunks - shuffled_chunks).flatten(1).norm(dim=1)
    audit = {
        "episode_id": int(episode_id),
        "start": int(start),
        "shuffle_seed": int(shuffle_seed),
        "shuffle_type": SHUFFLE_TYPE,
        "permutation": [int(value) for value in permutation.tolist()],
        "shuffle_source_transitions": [
            int(value) + 1 for value in permutation.tolist()
        ],
        "fixed_points": int((permutation == torch.arange(rollout_steps)).sum()),
        "modified_raw_action_start": modified_start,
        "modified_raw_action_end": modified_end,
        "history_action_start": int(spans["history_action_start"]),
        "history_action_end": int(spans["history_action_end"]),
        "history_action_unchanged": bool(
            torch.equal(
                actions[
                    spans["history_action_start"] : spans["history_action_end"]
                ],
                modified_actions[
                    spans["history_action_start"] : spans["history_action_end"]
                ],
            )
        ),
        "normalized_action_chunk_l2_difference_mean": float(difference.mean()),
    }
    if audit["fixed_points"] != 0 or not audit["history_action_unchanged"]:
        raise RuntimeError("invalid action-shuffle audit")
    return counterfactual_episode, audit


def validate_source_checkpoint_compatibility(
    summary: dict[str, Any],
    checkpoint: dict[str, Any],
    *,
    precision: str,
    weights_used: str,
) -> None:
    config = checkpoint["config"]
    expected = {
        "checkpoint_step": int(checkpoint["step"]),
        "precision": precision,
        "weights_used": weights_used,
        "num_frames": int(config["temporal"]["num_frames"]),
        "num_history": int(config["temporal"]["num_history"]),
        "frame_stride": int(config["temporal"]["frame_stride"]),
        "action_representation": config["action"]["representation"],
        "effective_action_dim": int(config["action"]["effective_action_dim"]),
    }
    for field in SOURCE_COMPATIBILITY_FIELDS:
        if summary.get(field) != expected[field]:
            raise RuntimeError(
                f"source rollout/checkpoint mismatch for {field}: "
                f"source={summary.get(field)!r}, checkpoint={expected[field]!r}"
            )
    if config.get("latent", {}).get("convention") != LATENT_CONVENTION:
        raise RuntimeError("checkpoint latent convention is unsupported")
    if config["action"].get("alignment") != "causal":
        raise RuntimeError("action controllability requires causal alignment")
    if config["action"].get("representation") != "fast_chunk":
        raise RuntimeError("action controllability currently requires fast_chunk")
    if summary.get("reference_used_as_model_input") is not False:
        raise RuntimeError("source must guarantee reference_used_as_model_input=false")


def validate_selected_case_split_membership(
    selected_rollout_cases: Sequence[dict[str, Any]],
    *,
    split: str,
    split_episode_ids: Sequence[int],
) -> None:
    allowed = {int(episode_id) for episode_id in split_episode_ids}
    selected = {int(case["episode_id"]) for case in selected_rollout_cases}
    outside = sorted(selected - allowed)
    if outside:
        raise RuntimeError(
            f"source rollout cases contain episodes outside declared {split!r} "
            f"split: {outside}"
        )


def summarize_true_rerun_warnings(
    warnings_by_stream: Sequence[Sequence[str]],
) -> tuple[int, int]:
    return (
        sum(len(messages) for messages in warnings_by_stream),
        sum(bool(messages) for messages in warnings_by_stream),
    )


def _paired_variant_streams(
    base_streams: Sequence[RolloutStream],
    *,
    config: dict[str, Any],
    action_stats: ActionStats,
    rollout_steps: int,
    seed: int,
    shuffle_seed: int,
) -> tuple[list[RolloutStream], list[dict[str, Any]]]:
    temporal = config["temporal"]
    variants = []
    audits = []
    for stream in base_streams:
        latent_shape = torch.as_tensor(stream.episode["latents"]).shape[1:]
        noises = stream.initial_noises
        if noises is None:
            noises = deterministic_rollout_noise(
                latent_shape=latent_shape,
                rollout_steps=rollout_steps,
                seed=seed,
                episode_id=stream.episode_id,
                start=stream.start,
                noise_draw=stream.noise_draw,
            )
        shuffled_episode, audit = temporally_shuffled_episode(
            stream.episode,
            episode_id=stream.episode_id,
            start=stream.start,
            num_history=int(temporal["num_history"]),
            frame_stride=int(temporal["frame_stride"]),
            rollout_steps=rollout_steps,
            shuffle_seed=shuffle_seed,
            action_stats=action_stats,
        )
        variants.extend(
            (
                RolloutStream(
                    episode=stream.episode,
                    episode_id=stream.episode_id,
                    start=stream.start,
                    noise_draw=stream.noise_draw,
                    initial_noises=torch.as_tensor(noises).clone(),
                ),
                RolloutStream(
                    episode=shuffled_episode,
                    episode_id=stream.episode_id,
                    start=stream.start,
                    noise_draw=stream.noise_draw,
                    initial_noises=torch.as_tensor(noises).clone(),
                ),
            )
        )
        audits.append(audit)
    return variants, audits


def paired_action_counterfactual_rollout_batch(
    *,
    model: torch.nn.Module,
    config: dict[str, Any],
    action_stats: ActionStats,
    base_streams: Sequence[RolloutStream],
    rollout_steps: int,
    euler_steps: int,
    seed: int,
    shuffle_seed: int,
    device: torch.device,
    precision: str,
) -> list[dict[str, Any]]:
    if not base_streams:
        raise ValueError("paired rollout batch must contain a base stream")
    variants, audits = _paired_variant_streams(
        base_streams,
        config=config,
        action_stats=action_stats,
        rollout_steps=rollout_steps,
        seed=seed,
        shuffle_seed=shuffle_seed,
    )
    results = autoregressive_causal_rollout_batch(
        model=model,
        config=config,
        action_stats=action_stats,
        streams=variants,
        rollout_steps=rollout_steps,
        euler_steps=euler_steps,
        seed=seed,
        device=device,
        precision=precision,
    )
    paired = []
    for pair_index, audit in enumerate(audits):
        true_result = results[2 * pair_index]
        shuffle_result = results[2 * pair_index + 1]
        identity_fields = ("episode_id", "start", "noise_draw")
        if any(true_result[key] != shuffle_result[key] for key in identity_fields):
            raise RuntimeError("true/shuffle rollout identity mismatch")
        if not torch.equal(
            true_result["target_frame_indices"],
            shuffle_result["target_frame_indices"],
        ):
            raise RuntimeError("true/shuffle target frame mismatch")
        if not torch.equal(
            true_result["initial_noises"], shuffle_result["initial_noises"]
        ):
            raise RuntimeError("true/shuffle initial Gaussian noise mismatch")
        if true_result["reference_used_as_model_input"] is not False or shuffle_result[
            "reference_used_as_model_input"
        ] is not False:
            raise RuntimeError("future reference was used as model input")
        divergence = latent_error_metrics(
            shuffle_result["predicted_future"], true_result["predicted_future"]
        )
        paired.append(
            {
                "true": true_result,
                "shuffle": shuffle_result,
                "divergence": divergence,
                "audit": audit,
            }
        )
    return paired


def action_counterfactual_rollouts(
    *,
    model: torch.nn.Module,
    config: dict[str, Any],
    action_stats: ActionStats,
    base_streams: Sequence[RolloutStream],
    pair_batch_size: int,
    rollout_steps: int,
    euler_steps: int,
    seed: int,
    shuffle_seed: int,
    device: torch.device,
    precision: str,
) -> list[dict[str, Any]]:
    if pair_batch_size <= 0:
        raise ValueError("pair_batch_size must be positive")
    results = []
    for offset in range(0, len(base_streams), pair_batch_size):
        results.extend(
            paired_action_counterfactual_rollout_batch(
                model=model,
                config=config,
                action_stats=action_stats,
                base_streams=base_streams[offset : offset + pair_batch_size],
                rollout_steps=rollout_steps,
                euler_steps=euler_steps,
                seed=seed,
                shuffle_seed=shuffle_seed,
                device=device,
                precision=precision,
            )
        )
    return results


def validate_true_rerun(
    result: dict[str, Any], source_rows: Sequence[dict[str, Any]]
) -> list[str]:
    identity = (result["episode_id"], result["start"], result["noise_draw"])
    ordered = sorted(source_rows, key=lambda row: int(row["step"]))
    if not ordered or any(
        (int(row["episode_id"]), int(row["start"]), int(row["noise_draw"]))
        != identity
        for row in ordered
    ):
        raise RuntimeError("true rerun/source identity mismatch")
    source_targets = torch.tensor(
        [int(row["target_frame_index"]) for row in ordered], dtype=torch.long
    )
    if not torch.equal(result["target_frame_indices"], source_targets):
        raise RuntimeError("true rerun/source target frame mismatch")
    messages = []
    for index, row in enumerate(ordered):
        for metric in ("mse", "relative_l2", "cosine_similarity"):
            rerun = float(result["metrics"][metric][index])
            source = float(row[metric])
            if not math.isclose(rerun, source, rel_tol=5e-3, abs_tol=1e-5):
                messages.append(
                    f"{identity} step {index + 1} {metric}: "
                    f"rerun={rerun} source={source}"
                )
    if messages:
        warnings.warn("true rollout rerun metric differences: " + "; ".join(messages))
    return messages


def paired_rollout_step_rows(
    paired_results: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for pair in paired_results:
        true = pair["true"]
        shuffled = pair["shuffle"]
        divergence = pair["divergence"]
        sources = pair["audit"]["shuffle_source_transitions"]
        for index, target in enumerate(true["target_frame_indices"].tolist()):
            true_mse = float(true["metrics"]["mse"][index])
            shuffle_mse = float(shuffled["metrics"]["mse"][index])
            rows.append(
                {
                    "episode_id": int(true["episode_id"]),
                    "start": int(true["start"]),
                    "noise_draw": int(true["noise_draw"]),
                    "step": index + 1,
                    "target_frame_index": int(target),
                    "shuffle_source_transition": int(sources[index]),
                    "true_mse": true_mse,
                    "shuffle_mse": shuffle_mse,
                    "delta_mse": shuffle_mse - true_mse,
                    "true_relative_l2": float(
                        true["metrics"]["relative_l2"][index]
                    ),
                    "shuffle_relative_l2": float(
                        shuffled["metrics"]["relative_l2"][index]
                    ),
                    "true_cosine_similarity": float(
                        true["metrics"]["cosine_similarity"][index]
                    ),
                    "shuffle_cosine_similarity": float(
                        shuffled["metrics"]["cosine_similarity"][index]
                    ),
                    "prediction_divergence_mse": float(divergence["mse"][index]),
                    "prediction_divergence_relative_l2": float(
                        divergence["relative_l2"][index]
                    ),
                    "prediction_divergence_cosine": float(
                        divergence["cosine_similarity"][index]
                    ),
                    "true_better": true_mse < shuffle_mse,
                }
            )
    return rows


def _mean(rows: Sequence[dict[str, Any]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows)


def aggregate_per_rollout(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["episode_id"]), int(row["start"]), int(row["noise_draw"]))].append(
            row
        )
    output = []
    for identity, values in sorted(grouped.items()):
        values = sorted(values, key=lambda row: int(row["step"]))
        output.append(
            {
                "episode_id": identity[0],
                "start": identity[1],
                "noise_draw": identity[2],
                "mean_true_mse": _mean(values, "true_mse"),
                "mean_shuffle_mse": _mean(values, "shuffle_mse"),
                "mean_delta_mse": _mean(values, "delta_mse"),
                "final_true_mse": float(values[-1]["true_mse"]),
                "final_shuffle_mse": float(values[-1]["shuffle_mse"]),
                "final_delta_mse": float(values[-1]["delta_mse"]),
                "mean_prediction_divergence_mse": _mean(
                    values, "prediction_divergence_mse"
                ),
                "true_better_fraction_steps": _mean(values, "true_better"),
            }
        )
    return output


def aggregate_per_case(
    rollout_rows: Sequence[dict[str, Any]],
    *,
    audits: Sequence[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rollout_rows:
        grouped[(int(row["episode_id"]), int(row["start"]))].append(row)
    audit_map = {
        (int(item["episode_id"]), int(item["start"])): item
        for item in (audits or [])
    }
    output = []
    for identity, values in sorted(grouped.items()):
        row = {
            "episode_id": identity[0],
            "start": identity[1],
            "num_noise_draws": len(values),
            "mean_true_mse": _mean(values, "mean_true_mse"),
            "mean_shuffle_mse": _mean(values, "mean_shuffle_mse"),
            "mean_delta_mse": _mean(values, "mean_delta_mse"),
            "mean_prediction_divergence_mse": _mean(
                values, "mean_prediction_divergence_mse"
            ),
            "true_better_draw_rate": sum(
                float(item["mean_true_mse"]) < float(item["mean_shuffle_mse"])
                for item in values
            )
            / len(values),
        }
        if identity in audit_map:
            audit = audit_map[identity]
            row.update(
                {
                    "fixed_points": audit["fixed_points"],
                    "permutation": audit["permutation"],
                    "modified_raw_action_start": audit["modified_raw_action_start"],
                    "modified_raw_action_end": audit["modified_raw_action_end"],
                    "history_action_unchanged": audit["history_action_unchanged"],
                    "normalized_action_chunk_l2_difference_mean": audit[
                        "normalized_action_chunk_l2_difference_mean"
                    ],
                }
            )
        output.append(row)
    return output


def _distribution(values: Sequence[float], quantiles: Sequence[float]) -> dict[float, float]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    return {
        float(quantile): float(torch.quantile(tensor, float(quantile)))
        for quantile in quantiles
    }


def aggregate_per_step(
    rows: Sequence[dict[str, Any]], *, frame_stride: int, fps: float = FPS
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["step"])].append(row)
    output = []
    for step, values in sorted(grouped.items()):
        true_values = [float(row["true_mse"]) for row in values]
        shuffle_values = [float(row["shuffle_mse"]) for row in values]
        delta_values = [float(row["delta_mse"]) for row in values]
        divergence_values = [
            float(row["prediction_divergence_mse"]) for row in values
        ]
        true_q = _distribution(true_values, (0.5, 0.9))
        shuffle_q = _distribution(shuffle_values, (0.5, 0.9))
        delta_q = _distribution(delta_values, (0.5, 0.1, 0.9))
        divergence_q = _distribution(divergence_values, (0.5, 0.9))
        output.append(
            {
                "step": step,
                "time_sec": step * frame_stride / float(fps),
                "true_mse_mean": sum(true_values) / len(true_values),
                "true_mse_median": true_q[0.5],
                "true_mse_p90": true_q[0.9],
                "shuffle_mse_mean": sum(shuffle_values) / len(shuffle_values),
                "shuffle_mse_median": shuffle_q[0.5],
                "shuffle_mse_p90": shuffle_q[0.9],
                "delta_mse_mean": sum(delta_values) / len(delta_values),
                "delta_mse_median": delta_q[0.5],
                "delta_mse_p10": delta_q[0.1],
                "delta_mse_p90": delta_q[0.9],
                "true_better_rate": sum(
                    float(row["true_mse"]) < float(row["shuffle_mse"])
                    for row in values
                )
                / len(values),
                "prediction_divergence_mse_mean": sum(divergence_values)
                / len(divergence_values),
                "prediction_divergence_mse_median": divergence_q[0.5],
                "prediction_divergence_mse_p90": divergence_q[0.9],
            }
        )
    return output


def bootstrap_mean_ci(
    values: Sequence[float], *, samples: int, seed: int
) -> tuple[float, float, float]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    if tensor.numel() == 0:
        raise ValueError("bootstrap requires case-level values")
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    indices = torch.randint(
        0, len(tensor), (samples, len(tensor)), generator=generator
    )
    bootstrap_means = tensor[indices].mean(dim=1)
    return (
        float(tensor.mean()),
        float(torch.quantile(bootstrap_means, 0.025)),
        float(torch.quantile(bootstrap_means, 0.975)),
    )


def build_summary_metrics(
    rollout_rows: Sequence[dict[str, Any]],
    case_rows: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    true_mean = _mean(rollout_rows, "mean_true_mse")
    shuffle_mean = _mean(rollout_rows, "mean_shuffle_mse")
    delta_mean = shuffle_mean - true_mean
    case_mean, ci_low, ci_high = bootstrap_mean_ci(
        [float(row["mean_delta_mse"]) for row in case_rows],
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    return {
        "overall_true_mse_mean": true_mean,
        "overall_shuffle_mse_mean": shuffle_mean,
        "overall_delta_mse_mean": delta_mean,
        "overall_relative_degradation": delta_mean / true_mean
        if true_mean != 0
        else None,
        "stochastic_true_better_rate": sum(
            float(row["mean_true_mse"]) < float(row["mean_shuffle_mse"])
            for row in rollout_rows
        )
        / len(rollout_rows),
        "case_true_better_rate": sum(
            float(row["mean_true_mse"]) < float(row["mean_shuffle_mse"])
            for row in case_rows
        )
        / len(case_rows),
        "case_mean_delta_mse": case_mean,
        "case_delta_ci95_low": ci_low,
        "case_delta_ci95_high": ci_high,
        "mean_delta_mse_case_mean": case_mean,
        "mean_delta_mse_ci95_low": ci_low,
        "mean_delta_mse_ci95_high": ci_high,
        "overall_prediction_divergence_mse": _mean(
            rollout_rows, "mean_prediction_divergence_mse"
        ),
    }
