from __future__ import annotations

import hashlib
import math
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from src.inference.action_adapter import adapt_causal_actions
from src.inference.flow_sampler import euler_sample_next

from .data.common import (
    ActionStats,
    cache_paths_from_manifest,
    load_episode_cache,
)
from .data.multi_episode_dataset import (
    episode_ids_for_split,
    load_causal_manifest,
)
from .runtime import precision_context


@dataclass(frozen=True)
class EpisodeRolloutSpan:
    episode_id: int
    cache_path: Path
    num_starts: int


@dataclass(frozen=True)
class RolloutCase:
    episode_id: int
    start: int
    cache_path: Path


def build_rollout_catalog(
    manifest_path: str | Path,
    *,
    split: str,
    num_history: int,
    frame_stride: int,
    rollout_steps: int,
    raw_action_dim: int,
) -> list[EpisodeRolloutSpan]:
    """Describe every legal full-episode rollout without fixed-T windows."""

    if split not in {"val", "test"}:
        raise ValueError("rollout split must be val or test")
    if num_history <= 0 or frame_stride <= 0 or rollout_steps <= 0:
        raise ValueError("history, stride, and rollout_steps must be positive")
    manifest = load_causal_manifest(manifest_path)
    episode_ids = episode_ids_for_split(manifest, split)
    paths = cache_paths_from_manifest(manifest, episode_ids)
    sampled_span = (num_history + rollout_steps - 1) * frame_stride
    catalog = []
    for episode_id in episode_ids:
        path = paths[episode_id]
        episode = load_episode_cache(path, raw_action_dim)
        num_starts = len(episode["latents"]) - sampled_span
        if num_starts > 0:
            catalog.append(EpisodeRolloutSpan(episode_id, path, num_starts))
    return catalog


def select_rollout_cases(
    catalog: Sequence[EpisodeRolloutSpan], max_rollouts: int
) -> list[RolloutCase]:
    """Select deterministic evenly spaced cases across the whole split."""

    if max_rollouts < 0:
        raise ValueError("max_rollouts must be non-negative")
    total = sum(span.num_starts for span in catalog)
    if total == 0:
        raise RuntimeError("split has no legal rollout cases")
    if max_rollouts == 0 or max_rollouts >= total:
        positions = range(total)
    else:
        positions = (
            torch.linspace(0, total - 1, steps=max_rollouts)
            .round()
            .long()
            .unique()
            .tolist()
        )
    ends = []
    running = 0
    for span in catalog:
        running += span.num_starts
        ends.append(running)
    cases = []
    for position in positions:
        span_index = bisect_right(ends, int(position))
        previous_end = 0 if span_index == 0 else ends[span_index - 1]
        span = catalog[span_index]
        cases.append(
            RolloutCase(
                episode_id=span.episode_id,
                start=int(position) - previous_end,
                cache_path=span.cache_path,
            )
        )
    return cases


def _noise_seed(
    seed: int, episode_id: int, start: int, noise_draw: int, step: int
) -> int:
    payload = f"{seed}:{episode_id}:{start}:{noise_draw}:{step}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def deterministic_rollout_noise(
    *,
    latent_shape: Sequence[int],
    rollout_steps: int,
    seed: int,
    episode_id: int,
    start: int,
    noise_draw: int,
) -> torch.Tensor:
    """Draw step-addressed CPU noise so longer rollouts preserve every prefix."""

    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    noises = []
    for step in range(rollout_steps):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            _noise_seed(seed, episode_id, start, noise_draw, step)
        )
        noises.append(
            torch.randn(tuple(latent_shape), generator=generator, dtype=torch.float32)
        )
    return torch.stack(noises)


def _sampled_rows(
    start: int, count: int, frame_stride: int
) -> torch.Tensor:
    return start + torch.arange(count, dtype=torch.long) * frame_stride


@torch.inference_mode()
def autoregressive_causal_rollout(
    *,
    model: torch.nn.Module,
    config: dict[str, Any],
    action_stats: ActionStats,
    episode: dict[str, Any],
    episode_id: int,
    start: int,
    rollout_steps: int,
    euler_steps: int,
    seed: int,
    noise_draw: int,
    device: torch.device,
    precision: str,
    initial_noises: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Generate one-frame-at-a-time futures with only the initial GT history."""

    temporal = config["temporal"]
    max_time = int(temporal["num_frames"])
    num_history = int(temporal["num_history"])
    frame_stride = int(temporal["frame_stride"])
    if config["action"]["alignment"] != "causal":
        raise ValueError("causal rollout requires action.alignment=causal")
    if not 0 < num_history < max_time:
        raise ValueError("rollout requires 0 < num_history < num_frames")
    if rollout_steps <= 0 or euler_steps <= 0:
        raise ValueError("rollout_steps and euler_steps must be positive")

    latents = torch.as_tensor(episode["latents"])
    actions = torch.as_tensor(episode["actions"], dtype=torch.float32)
    cache_frame_indices = torch.as_tensor(
        episode.get("frame_indices", torch.arange(len(latents))), dtype=torch.long
    )
    required_rows = _sampled_rows(
        start, num_history + rollout_steps, frame_stride
    )
    if start < 0 or int(required_rows[-1]) >= len(latents):
        raise IndexError("rollout case exceeds episode boundary")

    history_rows = required_rows[:num_history]
    history_cpu = latents[history_rows].float().contiguous()
    history_raw_indices = cache_frame_indices[history_rows]
    contexts = [item.to(device=device, dtype=torch.float32) for item in history_cpu]
    context_raw_indices = [int(item) for item in history_raw_indices.tolist()]

    if initial_noises is None:
        initial_noises = deterministic_rollout_noise(
            latent_shape=history_cpu.shape[1:],
            rollout_steps=rollout_steps,
            seed=seed,
            episode_id=episode_id,
            start=start,
            noise_draw=noise_draw,
        )
    initial_noises = torch.as_tensor(initial_noises, dtype=torch.float32)
    expected_noise_shape = (rollout_steps, *history_cpu.shape[1:])
    if tuple(initial_noises.shape) != expected_noise_shape:
        raise ValueError(
            f"initial_noises must have shape {expected_noise_shape}, "
            f"got {tuple(initial_noises.shape)}"
        )

    predicted = []
    target_raw_indices = []
    model_input_lengths = []
    window_frame_indices = []
    action_indices = []
    action_conditions = []
    action_valid_masks = []
    max_clean_context = max_time - 1

    for rollout_index in range(rollout_steps):
        target_row = int(required_rows[num_history + rollout_index])
        target_raw_index = int(cache_frame_indices[target_row])
        window_contexts = contexts[-max_clean_context:]
        window_context_indices = context_raw_indices[-max_clean_context:]
        frame_indices = torch.tensor(
            window_context_indices + [target_raw_index], dtype=torch.long
        )
        adapted = adapt_causal_actions(
            actions=actions,
            cache_frame_indices=cache_frame_indices,
            frame_indices=frame_indices,
            config=config,
            stats=action_stats,
        )
        history_batch = torch.stack(window_contexts).unsqueeze(0)
        action_batch = adapted.condition.to(device=device, dtype=torch.float32).unsqueeze(0)
        mask_batch = adapted.valid_mask.to(device=device).unsqueeze(0)
        noise_batch = (
            initial_noises[rollout_index]
            .to(device=device, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        with precision_context(device, precision):
            next_latent, _ = euler_sample_next(
                model=model,
                checkpoint_type="causal_v2",
                history_latents=history_batch,
                action_cond=action_batch,
                action_valid_mask=mask_batch,
                num_inference_steps=euler_steps,
                initial_noise=noise_batch,
            )
        next_item = next_latent[0, 0].float()
        predicted.append(next_item.cpu())
        contexts.append(next_item)
        context_raw_indices.append(target_raw_index)
        target_raw_indices.append(target_raw_index)
        model_input_lengths.append(len(frame_indices))
        window_frame_indices.append(frame_indices)
        action_indices.append(adapted.action_indices)
        action_conditions.append(adapted.condition)
        action_valid_masks.append(adapted.valid_mask)

    predicted_future = torch.stack(predicted).float()
    # Future GT is intentionally first read after the complete autoregressive chain.
    gt_future = latents[required_rows[num_history:]].float().contiguous()
    metrics = latent_error_metrics(predicted_future, gt_future)
    return {
        "episode_id": int(episode_id),
        "start": int(start),
        "noise_draw": int(noise_draw),
        "history_latents": history_cpu,
        "predicted_future": predicted_future,
        "gt_future": gt_future,
        "history_frame_indices": history_raw_indices.clone(),
        "target_frame_indices": torch.tensor(target_raw_indices, dtype=torch.long),
        "initial_noises": initial_noises.clone(),
        "model_input_lengths": model_input_lengths,
        "window_frame_indices": window_frame_indices,
        "action_indices": action_indices,
        "action_conditions": action_conditions,
        "action_valid_masks": action_valid_masks,
        "metrics": metrics,
        "reference_used_as_model_input": False,
    }


def latent_error_metrics(
    predicted: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> dict[str, torch.Tensor]:
    predicted = torch.as_tensor(predicted, dtype=torch.float32)
    target = torch.as_tensor(target, dtype=torch.float32)
    if predicted.shape != target.shape or predicted.ndim < 2:
        raise ValueError("predicted and target must have matching [R,...] shapes")
    difference = predicted - target
    flat_difference = difference.flatten(1)
    flat_predicted = predicted.flatten(1)
    flat_target = target.flatten(1)
    mse = flat_difference.square().mean(dim=1)
    return {
        "mse": mse,
        "rmse": mse.sqrt(),
        "mae": flat_difference.abs().mean(dim=1),
        "relative_l2": flat_difference.norm(dim=1)
        / (flat_target.norm(dim=1) + float(eps)),
        "cosine_similarity": F.cosine_similarity(
            flat_predicted, flat_target, dim=1, eps=float(eps)
        ),
    }


def rollout_metric_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index, target_frame_index in enumerate(
        result["target_frame_indices"].tolist()
    ):
        row = {
            "episode_id": result["episode_id"],
            "start": result["start"],
            "noise_draw": result["noise_draw"],
            "step": index + 1,
            "target_frame_index": target_frame_index,
        }
        for metric, values in result["metrics"].items():
            row[metric] = float(values[index])
        rows.append(row)
    return rows


def aggregate_rollout_metrics(
    rows: Sequence[dict[str, Any]], *, rollout_steps: int, frame_stride: int
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("cannot aggregate no rollout metrics")
    aggregated = []
    cumulative_mse = []
    for step in range(1, rollout_steps + 1):
        step_rows = [row for row in rows if int(row["step"]) == step]
        if not step_rows:
            raise ValueError(f"no metric rows for rollout step {step}")
        values = {
            metric: torch.tensor(
                [float(row[metric]) for row in step_rows], dtype=torch.float64
            )
            for metric in (
                "mse",
                "rmse",
                "mae",
                "relative_l2",
                "cosine_similarity",
            )
        }
        mse = values["mse"]
        cumulative_mse.append(float(mse.mean()))
        aggregated.append(
            {
                "step": step,
                "raw_frame_offset": step * frame_stride,
                "num_samples": len(step_rows),
                "mse_mean": float(mse.mean()),
                "mse_std": float(mse.std(unbiased=False)),
                "mse_median": float(mse.median()),
                "mse_p90": float(torch.quantile(mse, 0.9)),
                "rmse_mean": float(values["rmse"].mean()),
                "mae_mean": float(values["mae"].mean()),
                "relative_l2_mean": float(values["relative_l2"].mean()),
                "cosine_similarity_mean": float(
                    values["cosine_similarity"].mean()
                ),
                "cumulative_mse_mean": sum(cumulative_mse) / len(cumulative_mse),
            }
        )
    return aggregated


def threshold_horizon(
    per_step: Sequence[dict[str, Any]],
    *,
    threshold: float | None,
    metric: str,
) -> tuple[int | None, int | None]:
    if metric not in {"mse", "relative_l2"}:
        raise ValueError("threshold metric must be mse or relative_l2")
    if threshold is None:
        return None, None
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("error threshold must be finite and non-negative")
    key = f"{metric}_mean"
    for row in per_step:
        if float(row[key]) > threshold:
            step = int(row["step"])
            return step - 1, step
    return len(per_step), None
