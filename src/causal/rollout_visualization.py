from __future__ import annotations

import csv
import math
import statistics
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch

from src.so101_cache.vae_cache import (
    FPS,
    LATENT_CHANNELS,
    LATENT_CONVENTION,
    resolve_vae_dtype,
)

from .data.common import ActionStats
from .rollout import autoregressive_causal_rollout


REFERENCE_LABEL = "GT latent reconstruction"
SELECTION_METRICS = {"mean_mse"}
IDENTITY_FIELDS = ("episode_id", "start", "noise_draw")
CONTACT_LEFT_MARGIN = 180
CONTACT_HEADER_HEIGHT = 32
SHARED_HISTORY_PREFIX = "Shared History"


def validate_summary_checkpoint_compatibility(
    summary: dict[str, Any],
    checkpoint: dict[str, Any],
    *,
    precision: str,
    weights_used: str,
) -> None:
    config = checkpoint["config"]
    latent_convention = config.get("latent", {}).get("convention")
    if latent_convention != LATENT_CONVENTION:
        raise RuntimeError(
            "checkpoint latent convention is incompatible with rollout "
            f"visualization: expected {LATENT_CONVENTION!r}, "
            f"got {latent_convention!r}"
        )

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
    for field, checkpoint_value in expected.items():
        summary_value = summary.get(field)
        if summary_value != checkpoint_value:
            raise RuntimeError(
                f"rollout summary/checkpoint mismatch for {field}: "
                f"summary={summary_value!r}, checkpoint={checkpoint_value!r}"
            )


def resolve_vae_directory(path: str | Path) -> Path:
    path = Path(path)
    direct_config = path / "config.json"
    nested = path / "vae"
    if direct_config.is_file():
        return path
    if (nested / "config.json").is_file():
        return nested
    raise FileNotFoundError(
        f"VAE config not found at {direct_config} or {nested / 'config.json'}"
    )


def load_frozen_vae(path: str | Path, device: torch.device):
    from diffusers import AutoencoderKL

    vae_directory = resolve_vae_directory(path)
    dtype = resolve_vae_dtype(device)
    vae = AutoencoderKL.from_pretrained(
        vae_directory,
        local_files_only=True,
        torch_dtype=dtype,
    )
    vae.eval()
    vae.requires_grad_(False)
    vae.to(device)
    actual_dtype = next(vae.parameters()).dtype
    if actual_dtype != dtype:
        raise RuntimeError(f"VAE dtype policy requested {dtype}, got {actual_dtype}")
    if int(vae.config.latent_channels) != LATENT_CHANNELS:
        raise RuntimeError(
            f"expected VAE latent_channels={LATENT_CHANNELS}, "
            f"got {vae.config.latent_channels}"
        )
    scaling = float(vae.config.scaling_factor)
    if not math.isfinite(scaling) or scaling <= 0:
        raise RuntimeError(f"invalid VAE scaling_factor: {scaling}")
    return vae, vae_directory


def decoded_tensor_to_uint8(decoded: torch.Tensor) -> torch.Tensor:
    decoded = torch.as_tensor(decoded, dtype=torch.float32)
    if decoded.ndim != 4 or decoded.shape[1] != 3:
        raise ValueError("decoded images must be [N,3,H,W]")
    return (
        decoded.div(2.0)
        .add(0.5)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .contiguous()
        .cpu()
    )


def _vae_dtype(vae, device: torch.device) -> torch.dtype:
    try:
        return next(vae.parameters()).dtype
    except StopIteration:
        return resolve_vae_dtype(device)


@torch.inference_mode()
def decode_cached_latents(
    vae,
    latents: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Decode z_cache / scaling_factor; project convention never applies shift."""

    if batch_size <= 0:
        raise ValueError("decode batch_size must be positive")
    source = torch.as_tensor(latents, dtype=torch.float32)
    if source.ndim != 4:
        raise ValueError("latents must be [N,C,H,W]")
    scaling = float(vae.config.scaling_factor)
    if not math.isfinite(scaling) or scaling <= 0:
        raise RuntimeError(f"invalid VAE scaling_factor: {scaling}")
    dtype = _vae_dtype(vae, device)
    decoded_batches = []
    for offset in range(0, len(source), batch_size):
        # Exact inverse of posterior_sample_times_scaling_no_shift.
        z_vae = source[offset : offset + batch_size].to(
            device=device, dtype=dtype
        ) / scaling
        decoded_batches.append(vae.decode(z_vae).sample.float().cpu())
    return decoded_tensor_to_uint8(torch.cat(decoded_batches))


def read_rollout_metric_rows(path: str | Path) -> list[dict[str, Any]]:
    integer_fields = {
        "episode_id",
        "start",
        "noise_draw",
        "step",
        "target_frame_index",
    }
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row = {}
            for key, value in raw.items():
                row[key] = int(value) if key in integer_fields else float(value)
            rows.append(row)
    if not rows:
        raise ValueError("per_rollout_step.csv contains no rows")
    return rows


def stream_identity(row: dict[str, Any]) -> tuple[int, int, int]:
    return tuple(int(row[field]) for field in IDENTITY_FIELDS)


def group_stream_rows(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[int, int, int], list[dict[str, Any]]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[stream_identity(row)].append(dict(row))
    return {
        identity: sorted(values, key=lambda item: int(item["step"]))
        for identity, values in sorted(grouped.items())
    }


def score_rollout_streams(
    rows: Sequence[dict[str, Any]], *, metric: str
) -> dict[tuple[int, int, int], float]:
    if metric not in SELECTION_METRICS:
        raise ValueError(f"unsupported selection metric: {metric}")
    return {
        identity: statistics.fmean(float(row["mse"]) for row in stream_rows)
        for identity, stream_rows in group_stream_rows(rows).items()
    }


def _quantile(values: Sequence[float], quantile: float) -> float:
    return float(
        torch.quantile(
            torch.tensor(values, dtype=torch.float64), float(quantile)
        )
    )


def score_distribution(scores: Sequence[float]) -> dict[str, float | int]:
    values = [float(value) for value in scores]
    if not values:
        raise ValueError("cannot summarize no stream scores")
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "median": _quantile(values, 0.5),
        "p10": _quantile(values, 0.1),
        "p50": _quantile(values, 0.5),
        "p90": _quantile(values, 0.9),
        "min": min(values),
        "max": max(values),
    }


def quantile_label(quantile: float) -> str:
    return f"q{round(float(quantile) * 100):02d}"


def select_quantile_representatives(
    rows: Sequence[dict[str, Any]],
    *,
    metric: str,
    quantiles: Sequence[float],
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    if not quantiles:
        raise ValueError("at least one quantile is required")
    if any(not 0.0 <= float(value) <= 1.0 for value in quantiles):
        raise ValueError("quantiles must be within [0,1]")
    labels = [quantile_label(float(value)) for value in quantiles]
    if len(set(labels)) != len(labels):
        raise ValueError(
            "quantiles produce duplicate output labels: " + ", ".join(labels)
        )
    scores = score_rollout_streams(rows, metric=metric)
    ordered = sorted(scores.items())
    used = set()
    selected = []
    for quantile in quantiles:
        target = _quantile(list(scores.values()), float(quantile))
        candidates = sorted(
            ordered,
            key=lambda item: (
                round(abs(item[1] - target), 12),
                item[0],
            ),
        )
        unused = [item for item in candidates if item[0] not in used]
        identity, actual = (unused or candidates)[0]
        used.add(identity)
        selected.append(
            {
                "label": quantile_label(float(quantile)),
                "quantile": float(quantile),
                "target_score": target,
                "actual_score": actual,
                "episode_id": identity[0],
                "start": identity[1],
                "noise_draw": identity[2],
            }
        )
    return selected, score_distribution(list(scores.values()))


def parse_stream_spec(value: str) -> tuple[int, int, int]:
    pieces = value.split(":")
    if len(pieces) != 3:
        raise ValueError("stream must be EPISODE_ID:START:NOISE_DRAW")
    try:
        identity = tuple(int(piece) for piece in pieces)
    except ValueError as error:
        raise ValueError("stream fields must be integers") from error
    if any(item < 0 for item in identity):
        raise ValueError("stream fields must be non-negative")
    return identity


def select_explicit_streams(
    rows: Sequence[dict[str, Any]],
    *,
    metric: str,
    identities: Sequence[tuple[int, int, int]],
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    scores = score_rollout_streams(rows, metric=metric)
    selected = []
    for identity in identities:
        if identity not in scores:
            raise ValueError(f"requested stream is absent from evaluation: {identity}")
        selected.append(
            {
                "label": (
                    f"stream_ep{identity[0]:04d}_s{identity[1]:04d}_d{identity[2]}"
                ),
                "quantile": None,
                "target_score": None,
                "actual_score": scores[identity],
                "episode_id": identity[0],
                "start": identity[1],
                "noise_draw": identity[2],
            }
        )
    return selected, score_distribution(list(scores.values()))


def normalized_display_steps(
    requested: Sequence[int], rollout_steps: int
) -> list[int]:
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    values = {int(step) for step in requested if 0 < int(step) <= rollout_steps}
    values.add(int(rollout_steps))
    return sorted(values)


def future_time_seconds(step: int, frame_stride: int, fps: float = FPS) -> float:
    if step <= 0 or frame_stride <= 0 or fps <= 0:
        raise ValueError("step, frame_stride, and fps must be positive")
    return float(step * frame_stride) / float(fps)


def rerun_rollout_stream(
    *,
    model: torch.nn.Module,
    config: dict[str, Any],
    action_stats: ActionStats,
    episode: dict[str, Any],
    episode_id: int,
    start: int,
    noise_draw: int,
    rollout_steps: int,
    euler_steps: int,
    seed: int,
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    return autoregressive_causal_rollout(
        model=model,
        config=config,
        action_stats=action_stats,
        episode=episode,
        episode_id=episode_id,
        start=start,
        noise_draw=noise_draw,
        rollout_steps=rollout_steps,
        euler_steps=euler_steps,
        seed=seed,
        device=device,
        precision=precision,
    )


def validate_rerun(
    result: dict[str, Any], source_rows: Sequence[dict[str, Any]]
) -> list[str]:
    identity = (result["episode_id"], result["start"], result["noise_draw"])
    if any(stream_identity(row) != identity for row in source_rows):
        raise RuntimeError("rerun/source stochastic stream identity mismatch")
    ordered = sorted(source_rows, key=lambda row: int(row["step"]))
    source_targets = torch.tensor(
        [int(row["target_frame_index"]) for row in ordered], dtype=torch.long
    )
    if not torch.equal(result["target_frame_indices"], source_targets):
        raise RuntimeError("rerun/source target_frame_indices mismatch")
    messages = []
    for index, row in enumerate(ordered):
        for metric in ("mse", "relative_l2", "cosine_similarity"):
            rerun_value = float(result["metrics"][metric][index])
            source_value = float(row[metric])
            if not math.isclose(rerun_value, source_value, rel_tol=5e-3, abs_tol=1e-5):
                messages.append(
                    f"step {index + 1} {metric}: rerun={rerun_value} "
                    f"source={source_value}"
                )
    if messages:
        warnings.warn("rollout rerun metric differences: " + "; ".join(messages))
    return messages


def save_png_sequence(
    images: torch.Tensor,
    directory: Path,
    prefix: str,
    *,
    start_index: int = 0,
) -> None:
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(
        torch.as_tensor(images).cpu().numpy(), start=start_index
    ):
        Image.fromarray(image, mode="RGB").save(
            directory / f"{prefix}_{index:03d}.png"
        )


def create_contact_sheet(
    *,
    history_images: torch.Tensor,
    predicted_images: torch.Tensor,
    gt_reconstruction_images: torch.Tensor,
    display_steps: Sequence[int],
    frame_stride: int,
    fps: float = FPS,
):
    from PIL import Image, ImageDraw

    history = torch.as_tensor(history_images, dtype=torch.uint8)
    predicted = torch.as_tensor(predicted_images, dtype=torch.uint8)
    gt = torch.as_tensor(gt_reconstruction_images, dtype=torch.uint8)
    if history.ndim != 4 or history.shape[-1] != 3 or len(history) == 0:
        raise ValueError("history_images must be non-empty [H,Y,X,3]")
    if predicted.shape != gt.shape or predicted.ndim != 4:
        raise ValueError("predicted/GT images must have matching [R,Y,X,3]")
    steps = normalized_display_steps(display_steps, len(predicted))
    height, width = int(history.shape[1]), int(history.shape[2])
    if tuple(predicted.shape[1:3]) != (height, width):
        raise ValueError("history and future images must share spatial shape")
    columns = len(history) + len(steps)
    sheet = Image.new(
        "RGB",
        (CONTACT_LEFT_MARGIN + columns * width, CONTACT_HEADER_HEIGHT + 2 * height),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (4, CONTACT_HEADER_HEIGHT + height // 2),
        f"Future: {REFERENCE_LABEL}",
        fill=(0, 0, 0),
    )
    draw.text(
        (4, CONTACT_HEADER_HEIGHT + height + height // 2),
        "Future: Prediction",
        fill=(0, 0, 0),
    )
    for index, image in enumerate(history.numpy()):
        x = CONTACT_LEFT_MARGIN + index * width
        item = Image.fromarray(image, mode="RGB")
        sheet.paste(item, (x, CONTACT_HEADER_HEIGHT))
        sheet.paste(item, (x, CONTACT_HEADER_HEIGHT + height))
        draw.text(
            (x + 2, 4),
            f"{SHARED_HISTORY_PREFIX} {index}",
            fill=(0, 0, 0),
        )
    for offset, step in enumerate(steps, start=len(history)):
        x = CONTACT_LEFT_MARGIN + offset * width
        sheet.paste(
            Image.fromarray(gt[step - 1].numpy(), mode="RGB"),
            (x, CONTACT_HEADER_HEIGHT),
        )
        sheet.paste(
            Image.fromarray(predicted[step - 1].numpy(), mode="RGB"),
            (x, CONTACT_HEADER_HEIGHT + height),
        )
        seconds = future_time_seconds(step, frame_stride, fps)
        draw.text((x + 2, 4), f"step {step}  +{seconds:.1f}s", fill=(0, 0, 0))
    return sheet


def build_stream_metadata(
    *,
    selection: dict[str, Any],
    summary: dict[str, Any],
    result: dict[str, Any],
    vae_path: str | Path,
    vae,
    fps: float,
) -> dict[str, Any]:
    metrics = result["metrics"]
    per_step_mse = [float(value) for value in metrics["mse"]]
    return {
        "selection_label": selection["label"],
        "selection_quantile": selection.get("quantile"),
        "selection_metric": summary["selection_metric"],
        "selection_score": float(selection["actual_score"]),
        "episode_id": int(result["episode_id"]),
        "start": int(result["start"]),
        "noise_draw": int(result["noise_draw"]),
        "checkpoint_path": summary["checkpoint_path"],
        "checkpoint_step": int(summary["checkpoint_step"]),
        "weights_used": summary["weights_used"],
        "rollout_steps": int(summary["rollout_steps"]),
        "euler_steps": int(summary["euler_steps"]),
        "frame_stride": int(summary["frame_stride"]),
        "fps": float(fps),
        "seed": int(summary["seed"]),
        "history_frame_indices": result["history_frame_indices"].tolist(),
        "target_frame_indices": result["target_frame_indices"].tolist(),
        "per_step_mse": per_step_mse,
        "per_step_relative_l2": [
            float(value) for value in metrics["relative_l2"]
        ],
        "per_step_cosine_similarity": [
            float(value) for value in metrics["cosine_similarity"]
        ],
        "mean_mse": statistics.fmean(per_step_mse),
        "final_step_mse": per_step_mse[-1],
        "vae_path": str(Path(vae_path)),
        "vae_scaling_factor": float(vae.config.scaling_factor),
        "vae_shift_factor_from_config": getattr(vae.config, "shift_factor", None),
        "vae_shift_applied": False,
        "latent_convention": LATENT_CONVENTION,
        "reference_label": REFERENCE_LABEL,
    }
