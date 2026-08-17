from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch
import torch.nn.functional as F

from src.causal.action_counterfactual import bootstrap_mean_ci
from src.so101_cache.vae_cache import FPS, LATENT_CONVENTION, atomic_torch_save


ARTIFACT_VERSION = 1
RAW_RGB_PREPROCESSING = (
    "shared cache preprocess_rgb_batch: shortest-edge resize 256, center crop "
    "256x256, uint8 -> [0,1] -> [-1,1]; converted back to uint8 for metrics"
)
STAGE_A_FIELDS = (
    "episode_id", "task_id", "draw_id", "step", "time_sec",
    "raw_frame_index", "sampled_frame_index", "latent_mse", "latent_rmse",
    "latent_mae", "relative_l2", "cosine",
)
PER_DRAW_STEP_FIELDS = STAGE_A_FIELDS + (
    "pred_ssim", "recon_ssim", "ssim_gap", "pred_lpips",
    "recon_lpips", "lpips_gap",
)
PER_EPISODE_STEP_FIELDS = (
    "episode_id", "task_id", "step", "time_sec", "num_draws",
    "mean_latent_mse", "mean_relative_l2", "mean_cosine",
    "mean_pred_ssim", "recon_ssim", "mean_ssim_gap",
    "mean_pred_lpips", "recon_lpips", "mean_lpips_gap",
)
AGGREGATE_FIELDS = (
    "step", "time_sec", "n_episodes", "latent_mse_mean",
    "latent_mse_median", "latent_mse_p90", "latent_mse_ci_low",
    "latent_mse_ci_high", "ssim_gap_mean", "ssim_gap_median",
    "ssim_gap_p90", "ssim_gap_ci_low", "ssim_gap_ci_high",
    "lpips_gap_mean", "lpips_gap_median", "lpips_gap_p90",
    "lpips_gap_ci_low", "lpips_gap_ci_high",
)


@dataclass(frozen=True)
class FullEpisodePlan:
    episode_id: int
    task_id: int | None
    cache_path: Path
    rollout_steps: int


def file_sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vae_provenance(path: str | Path) -> dict[str, Any]:
    """Hash the VAE config and every actual weight file, not just config.json."""
    root = Path(path).resolve()
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"VAE config not found: {config_path}")
    weight_suffixes = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}
    weight_paths = sorted(
        file for file in root.rglob("*")
        if file.is_file()
        and (
            file.suffix.lower() in weight_suffixes
            or file.name.lower().endswith(".index.json")
        )
    )
    if not weight_paths:
        raise RuntimeError(f"VAE directory contains no recognized weight files: {root}")
    weight_hashes = {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in weight_paths
    }
    digest = hashlib.sha256()
    for relative_path, value in sorted(weight_hashes.items()):
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    config_sha = file_sha256(config_path)
    weights_sha = digest.hexdigest()
    artifact_digest = hashlib.sha256()
    artifact_digest.update(config_sha.encode("ascii"))
    artifact_digest.update(b"\n")
    artifact_digest.update(weights_sha.encode("ascii"))
    return {
        "vae_config_sha256": config_sha,
        "vae_weights_sha256": weights_sha,
        "vae_weight_files_sha256": weight_hashes,
        "vae_artifact_sha256": artifact_digest.hexdigest(),
    }


def full_episode_rollout_steps(
    num_cache_rows: int,
    *,
    num_history: int,
    frame_stride: int,
    max_rollout_steps: int | None,
) -> int:
    if num_cache_rows <= 0 or num_history <= 0 or frame_stride <= 0:
        raise ValueError("cache rows, history, and stride must be positive")
    # start=0, history rows are 0,d,...,(H-1)d; future begins at H*d.
    available = (num_cache_rows - 1) // frame_stride - (num_history - 1)
    available = max(0, int(available))
    if max_rollout_steps is None:
        return available
    if max_rollout_steps <= 0:
        raise ValueError("max_rollout_steps must be positive when provided")
    return min(available, int(max_rollout_steps))


def deterministic_balanced_shards(
    plans: Sequence[FullEpisodePlan], world_size: int
) -> list[list[FullEpisodePlan]]:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    shards: list[list[FullEpisodePlan]] = [[] for _ in range(world_size)]
    loads = [0] * world_size
    for plan in sorted(plans, key=lambda x: (-x.rollout_steps, x.episode_id)):
        rank = min(range(world_size), key=lambda value: (loads[value], value))
        shards[rank].append(plan)
        loads[rank] += int(plan.rollout_steps)
    for shard in shards:
        shard.sort(key=lambda x: x.episode_id)
    return shards


def artifact_path(root: str | Path, episode_id: int, draw_id: int) -> Path:
    return Path(root) / "latents" / f"episode_{episode_id:04d}_draw_{draw_id}.pt"


def rgb_artifact_path(root: str | Path, episode_id: int) -> Path:
    return Path(root) / "episode_metrics" / f"episode_{episode_id:04d}.pt"


def expected_target_rows(
    *, num_history: int, frame_stride: int, rollout_steps: int
) -> torch.Tensor:
    return torch.arange(
        num_history * frame_stride,
        (num_history + rollout_steps) * frame_stride,
        frame_stride,
        dtype=torch.long,
    )


def build_artifact_metadata(
    *,
    plan: FullEpisodePlan,
    draw_id: int,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    checkpoint_step: int,
    weights_used: str,
    seed: int,
    euler_steps: int,
    config: dict[str, Any],
    manifest_path: str | Path,
    manifest_sha256: str,
    cache_metadata: dict[str, Any],
) -> dict[str, Any]:
    temporal = config["temporal"]
    action = config["action"]
    return {
        "artifact_version": ARTIFACT_VERSION,
        "episode_id": int(plan.episode_id),
        "task_id": plan.task_id,
        "draw_id": int(draw_id),
        "number_of_steps": int(plan.rollout_steps),
        "rollout_seed": int(seed),
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": int(checkpoint_step),
        "weights_used": weights_used,
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_sha256": manifest_sha256,
        "cache_path": str(plan.cache_path.resolve()),
        "frame_stride": int(temporal["frame_stride"]),
        "num_history": int(temporal["num_history"]),
        "euler_steps": int(euler_steps),
        "action_alignment": action["alignment"],
        "action_representation": action["representation"],
        "effective_action_dim": int(action["effective_action_dim"]),
        "latent_convention": config.get("latent", {}).get("convention"),
        "vae_identifier": cache_metadata.get("vae_identifier"),
        "scaling_factor": cache_metadata.get("scaling_factor"),
        "shift_factor_used": bool(cache_metadata.get("shift_factor_used", False)),
        "reference_used_as_model_input": False,
    }


def make_long_rollout_artifact(
    result: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, Any]:
    return {
        "artifact_version": ARTIFACT_VERSION,
        "pred_latents": result["predicted_future"].to(torch.bfloat16).cpu().contiguous(),
        "sampled_frame_indices": expected_target_rows(
            num_history=int(metadata["num_history"]),
            frame_stride=int(metadata["frame_stride"]),
            rollout_steps=int(metadata["number_of_steps"]),
        ),
        "raw_frame_indices": result["target_frame_indices"].long().cpu().contiguous(),
        "metrics": {
            name: torch.as_tensor(values).float().cpu().contiguous()
            for name, values in result["metrics"].items()
        },
        "metadata": metadata,
        "reference_used_as_model_input": bool(result["reference_used_as_model_input"]),
    }


def validate_long_rollout_artifact(
    payload: dict[str, Any], expected_metadata: dict[str, Any]
) -> None:
    if not isinstance(payload, dict) or payload.get("artifact_version") != ARTIFACT_VERSION:
        raise RuntimeError("invalid long-rollout artifact version")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("long-rollout artifact lacks metadata")
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"long-rollout artifact metadata mismatch: {mismatches}")
    steps = int(expected_metadata["number_of_steps"])
    pred = torch.as_tensor(payload.get("pred_latents"))
    if pred.dtype != torch.bfloat16 or pred.ndim != 4 or len(pred) != steps:
        raise RuntimeError("artifact pred_latents must be BF16 [steps,C,H,W]")
    sampled = torch.as_tensor(payload.get("sampled_frame_indices"), dtype=torch.long)
    raw = torch.as_tensor(payload.get("raw_frame_indices"), dtype=torch.long)
    expected_rows = expected_target_rows(
        num_history=int(expected_metadata["num_history"]),
        frame_stride=int(expected_metadata["frame_stride"]),
        rollout_steps=steps,
    )
    if not torch.equal(sampled, expected_rows) or raw.shape != (steps,):
        raise RuntimeError("artifact sampled/raw frame indices are invalid")
    metrics = payload.get("metrics")
    required = {"mse", "rmse", "mae", "relative_l2", "cosine_similarity"}
    if not isinstance(metrics, dict) or not required.issubset(metrics):
        raise RuntimeError("artifact latent metrics are incomplete")
    if any(torch.as_tensor(metrics[name]).shape != (steps,) for name in required):
        raise RuntimeError("artifact latent metric length mismatch")
    if payload.get("reference_used_as_model_input") is not False:
        raise RuntimeError("artifact does not preserve the no-GT-leakage invariant")


def resume_decision(
    path: str | Path,
    expected_metadata: dict[str, Any],
    *,
    overwrite: bool,
) -> str:
    path = Path(path)
    if not path.is_file():
        return "run"
    if overwrite:
        return "run"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_long_rollout_artifact(payload, expected_metadata)
    return "skip"


def atomic_json_save(value: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fields: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def artifact_latent_rows(payload: dict[str, Any], *, fps: float = FPS) -> list[dict[str, Any]]:
    metadata = payload["metadata"]
    metrics = payload["metrics"]
    rows = []
    for index, (sampled, raw) in enumerate(
        zip(payload["sampled_frame_indices"].tolist(), payload["raw_frame_indices"].tolist())
    ):
        rows.append({
            "episode_id": int(metadata["episode_id"]),
            "task_id": metadata.get("task_id"),
            "draw_id": int(metadata["draw_id"]),
            "step": index + 1,
            "time_sec": (index + 1) * int(metadata["frame_stride"]) / float(fps),
            "raw_frame_index": int(raw),
            "sampled_frame_index": int(sampled),
            "latent_mse": float(metrics["mse"][index]),
            "latent_rmse": float(metrics["rmse"][index]),
            "latent_mae": float(metrics["mae"][index]),
            "relative_l2": float(metrics["relative_l2"][index]),
            "cosine": float(metrics["cosine_similarity"][index]),
        })
    return rows


def structural_ssim_per_image(
    images: torch.Tensor, references: torch.Tensor, *, data_range: float = 1.0
) -> torch.Tensor:
    """Per-image RGB SSIM using the standard 11x11 Gaussian window."""
    x = torch.as_tensor(images, dtype=torch.float32)
    y = torch.as_tensor(references, dtype=torch.float32)
    if x.shape != y.shape or x.ndim != 4 or x.shape[1] != 3:
        raise ValueError("SSIM inputs must be matching [N,3,H,W] RGB tensors")
    if x.shape[-2] < 11 or x.shape[-1] < 11 or data_range <= 0:
        raise ValueError("SSIM needs images at least 11x11 and positive data_range")
    coords = torch.arange(11, dtype=torch.float32, device=x.device) - 5
    kernel1d = torch.exp(-(coords.square()) / (2 * 1.5**2))
    kernel1d /= kernel1d.sum()
    kernel = torch.outer(kernel1d, kernel1d).view(1, 1, 11, 11).repeat(3, 1, 1, 1)
    mu_x = F.conv2d(x, kernel, groups=3)
    mu_y = F.conv2d(y, kernel, groups=3)
    mu_x2, mu_y2, mu_xy = mu_x.square(), mu_y.square(), mu_x * mu_y
    sigma_x2 = F.conv2d(x.square(), kernel, groups=3) - mu_x2
    sigma_y2 = F.conv2d(y.square(), kernel, groups=3) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, groups=3) - mu_xy
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return score.flatten(1).mean(1)


class AlexNetLPIPS:
    def __init__(self, device: torch.device) -> None:
        try:
            import lpips
        except ImportError as error:
            raise RuntimeError("LPIPS evaluation requires `pip install lpips`") from error
        self.device = device
        self.model = lpips.LPIPS(net="alex").eval().requires_grad_(False).to(device)

    @torch.inference_mode()
    def __call__(self, images: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(images, dtype=torch.float32)
        y = torch.as_tensor(references, dtype=torch.float32)
        if x.shape != y.shape or x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("LPIPS inputs must be matching [N,3,H,W] RGB tensors")
        if x.device != self.device:
            x = x.to(self.device, non_blocking=True)
        if y.device != self.device:
            y = y.to(self.device, non_blocking=True)
        return self.model(x.mul(2).sub(1), y.mul(2).sub(1)).flatten().float().cpu()


def image_metric_batches(
    images: torch.Tensor,
    references: torch.Tensor,
    *,
    lpips_metric,
    batch_size: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size <= 0:
        raise ValueError("metric batch size must be positive")
    x = torch.as_tensor(images, dtype=torch.float32)
    y = torch.as_tensor(references, dtype=torch.float32)
    target_device = torch.device("cpu") if device is None else torch.device(device)
    ssim, lpips_values = [], []
    for offset in range(0, len(x), batch_size):
        xb = x[offset : offset + batch_size].to(target_device, non_blocking=True)
        yb = y[offset : offset + batch_size].to(target_device, non_blocking=True)
        ssim.append(
            structural_ssim_per_image(xb, yb, data_range=1.0)
            .detach().float().cpu()
        )
        lpips_values.append(
            torch.as_tensor(lpips_metric(xb, yb)).detach().float().cpu()
        )
    return torch.cat(ssim), torch.cat(lpips_values)


def validate_image_metric_identity(lpips_metric) -> None:
    identical = torch.zeros(1, 3, 64, 64)
    ssim = structural_ssim_per_image(identical, identical, data_range=1.0)
    lpips_value = torch.as_tensor(lpips_metric(identical, identical)).float()
    if float(ssim[0]) < 0.9999:
        raise RuntimeError(f"SSIM identity sanity check failed: {float(ssim[0])}")
    if lpips_value.numel() != 1 or abs(float(lpips_value[0])) > 1e-5:
        raise RuntimeError(f"LPIPS identity sanity check failed: {lpips_value.tolist()}")


def build_rgb_artifact_metadata(
    *,
    episode_id: int,
    number_of_steps: int,
    stage_a_summary: dict[str, Any],
    stage_a_draw_artifact_sha256: dict[str, str],
    vae_config_sha256: str,
    vae_weights_sha256: str,
    vae_weight_files_sha256: dict[str, str],
    vae_artifact_sha256: str,
    vae_path: str | Path,
    scaling_factor: float,
) -> dict[str, Any]:
    draws = int(stage_a_summary["noise_draws"])
    expected_draw_keys = {str(value) for value in range(draws)}
    if set(stage_a_draw_artifact_sha256) != expected_draw_keys:
        raise ValueError("Stage A draw SHA map does not cover every configured draw")
    return {
        "episode_id": int(episode_id),
        "number_of_steps": int(number_of_steps),
        "noise_draws": draws,
        "stage_a_checkpoint_sha256": stage_a_summary["checkpoint_sha256"],
        "stage_a_manifest_sha256": stage_a_summary["manifest_sha256"],
        "stage_a_weights_used": stage_a_summary["weights_used"],
        "stage_a_seed": int(stage_a_summary["seed"]),
        "stage_a_euler_steps": int(stage_a_summary["euler_steps"]),
        "stage_a_frame_stride": int(stage_a_summary["frame_stride"]),
        "stage_a_num_history": int(stage_a_summary["num_history"]),
        "stage_a_latent_convention": stage_a_summary["latent_convention"],
        "stage_a_draw_artifact_sha256": dict(sorted(stage_a_draw_artifact_sha256.items())),
        "vae_config_sha256": str(vae_config_sha256),
        "vae_weights_sha256": str(vae_weights_sha256),
        "vae_weight_files_sha256": dict(sorted(vae_weight_files_sha256.items())),
        "vae_artifact_sha256": str(vae_artifact_sha256),
        "vae_path": str(Path(vae_path).resolve()),
        "scaling_factor": float(scaling_factor),
        "shift_applied": False,
        "ssim_implementation": "project Gaussian 11x11 sigma=1.5 per-image RGB",
        "ssim_data_range": 1.0,
        "lpips_implementation": "lpips.LPIPS",
        "lpips_backbone": "alex",
        "raw_rgb_preprocessing": RAW_RGB_PREPROCESSING,
    }


AGGREGATE_ONLY_STAGE_A_FIELDS = (
    "checkpoint_sha256",
    "manifest_sha256",
    "weights_used",
    "seed",
    "euler_steps",
    "frame_stride",
    "num_history",
    "noise_draws",
    "latent_convention",
    "number_of_episodes",
    "actual_total_stochastic_trajectories",
    "actual_total_prediction_steps",
)


def validate_aggregate_only_provenance(
    *,
    stage_a_summary: dict[str, Any],
    existing_stage_b_summary: dict[str, Any],
    long_rollout_dir: str | Path,
    stage_a_summary_sha256: str,
) -> None:
    mismatches = {}
    for field in AGGREGATE_ONLY_STAGE_A_FIELDS:
        if existing_stage_b_summary.get(field) != stage_a_summary.get(field):
            mismatches[field] = (
                existing_stage_b_summary.get(field), stage_a_summary.get(field)
            )
    expected_dir = str(Path(long_rollout_dir).resolve())
    if existing_stage_b_summary.get("stage_a_long_rollout_dir") != expected_dir:
        mismatches["stage_a_long_rollout_dir"] = (
            existing_stage_b_summary.get("stage_a_long_rollout_dir"), expected_dir
        )
    if existing_stage_b_summary.get("stage_a_summary_sha256") != stage_a_summary_sha256:
        mismatches["stage_a_summary_sha256"] = (
            existing_stage_b_summary.get("stage_a_summary_sha256"),
            stage_a_summary_sha256,
        )
    if existing_stage_b_summary.get("stage_b_completion_status") != "complete":
        mismatches["stage_b_completion_status"] = (
            existing_stage_b_summary.get("stage_b_completion_status"), "complete"
        )
    if mismatches:
        raise RuntimeError(
            "aggregate-only Stage A/Stage B provenance mismatch: "
            f"{mismatches}"
        )


def rgb_resume_decision(
    path: str | Path,
    expected_metadata: dict[str, Any],
    *,
    overwrite: bool,
) -> str:
    path = Path(path)
    if not path.is_file() or overwrite:
        return "run"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_rgb_metric_artifact(payload, expected_metadata)
    return "skip"


def reader_for_source(
    cache: dict[tuple[Path, str], Any],
    *,
    source_path: str | Path,
    camera: str,
    factory: Callable[[Path, str], Any],
):
    """Cache readers by dataset root and camera, never by concrete video shard."""
    key = (Path(source_path).resolve(), str(camera))
    if key not in cache:
        cache[key] = factory(key[0], key[1])
    return cache[key]


def validate_metric_rgb_images(
    images: torch.Tensor,
    *,
    expected_frames: int,
    name: str,
    image_size: int = 256,
) -> torch.Tensor:
    value = torch.as_tensor(images)
    expected = (expected_frames, image_size, image_size, 3)
    if value.dtype != torch.uint8 or tuple(value.shape) != expected:
        raise RuntimeError(
            f"{name} must be uint8 {expected}, got dtype={value.dtype} "
            f"shape={tuple(value.shape)}"
        )
    return value


def evaluate_stage_b_episode(
    *,
    draw_payloads: Iterable[dict[str, Any]],
    raw_gt_uint8: torch.Tensor,
    gt_latents: torch.Tensor,
    vae,
    decode_fn: Callable[..., torch.Tensor],
    lpips_metric,
    device: torch.device,
    decode_batch_size: int,
    metric_batch_size: int,
    expected_draws: int,
    image_size: int = 256,
) -> list[dict[str, Any]]:
    """Evaluate one episode while retaining at most one prediction draw RGB."""
    steps = len(gt_latents)
    raw = validate_metric_rgb_images(
        raw_gt_uint8, expected_frames=steps, name="raw GT", image_size=image_size
    )
    gt_recon = decode_fn(
        vae, gt_latents, device=device, batch_size=decode_batch_size
    )
    gt_recon = validate_metric_rgb_images(
        gt_recon, expected_frames=steps, name="GT reconstruction", image_size=image_size
    )

    def to_float(images: torch.Tensor) -> torch.Tensor:
        return images.permute(0, 3, 1, 2).float().div(255.0).contiguous()

    raw_float = to_float(raw)
    recon_float = to_float(gt_recon)
    recon_ssim, recon_lpips = image_metric_batches(
        recon_float, raw_float, lpips_metric=lpips_metric,
        batch_size=metric_batch_size, device=device,
    )
    del gt_recon, recon_float
    rows: list[dict[str, Any]] = []
    seen_draws: set[int] = set()
    reference_raw_indices = None
    reference_sampled_indices = None
    for payload in draw_payloads:
        metadata = payload["metadata"]
        draw_id = int(metadata["draw_id"])
        if draw_id in seen_draws:
            raise RuntimeError(f"duplicate Stage A draw payload: {draw_id}")
        seen_draws.add(draw_id)
        raw_indices = torch.as_tensor(payload["raw_frame_indices"]).long()
        sampled_indices = torch.as_tensor(payload["sampled_frame_indices"]).long()
        if reference_raw_indices is None:
            reference_raw_indices = raw_indices
            reference_sampled_indices = sampled_indices
        elif not torch.equal(raw_indices, reference_raw_indices) or not torch.equal(
            sampled_indices, reference_sampled_indices
        ):
            raise RuntimeError("Stage A draws disagree on target frame identities")
        pred = decode_fn(
            vae, payload["pred_latents"], device=device,
            batch_size=decode_batch_size,
        )
        pred = validate_metric_rgb_images(
            pred, expected_frames=steps, name=f"draw {draw_id} prediction",
            image_size=image_size,
        )
        pred_float = to_float(pred)
        pred_ssim, pred_lpips = image_metric_batches(
            pred_float, raw_float, lpips_metric=lpips_metric,
            batch_size=metric_batch_size, device=device,
        )
        latent_rows = artifact_latent_rows(payload, fps=FPS)
        rows.extend(combine_rgb_metrics(
            latent_rows, pred_ssim=pred_ssim.tolist(),
            recon_ssim=recon_ssim.tolist(), pred_lpips=pred_lpips.tolist(),
            recon_lpips=recon_lpips.tolist(),
        ))
        del pred, pred_float, pred_ssim, pred_lpips
    if seen_draws != set(range(expected_draws)):
        raise RuntimeError("Stage A draw payload coverage mismatch")
    rows.sort(key=lambda row: (int(row["draw_id"]), int(row["step"])))
    return rows


def combine_rgb_metrics(
    latent_rows: Sequence[dict[str, Any]],
    *,
    pred_ssim: Sequence[float],
    recon_ssim: Sequence[float],
    pred_lpips: Sequence[float],
    recon_lpips: Sequence[float],
) -> list[dict[str, Any]]:
    count = len(latent_rows)
    if any(len(values) != count for values in (pred_ssim, recon_ssim, pred_lpips, recon_lpips)):
        raise ValueError("RGB metric lengths must match latent rows")
    output = []
    for row, ps, rs, pl, rl in zip(
        latent_rows, pred_ssim, recon_ssim, pred_lpips, recon_lpips
    ):
        item = dict(row)
        item.update({
            "pred_ssim": float(ps), "recon_ssim": float(rs),
            "ssim_gap": float(rs) - float(ps),
            "pred_lpips": float(pl), "recon_lpips": float(rl),
            "lpips_gap": float(pl) - float(rl),
        })
        output.append(item)
    return output


def validate_rgb_metric_artifact(
    payload: dict[str, Any], expected_metadata: dict[str, Any]
) -> None:
    if not isinstance(payload, dict) or payload.get("metadata") != expected_metadata:
        raise RuntimeError("RGB metric artifact metadata mismatch")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError("RGB metric artifact lacks rows")
    episode_id = int(expected_metadata["episode_id"])
    steps = int(expected_metadata["number_of_steps"])
    draws = int(expected_metadata["noise_draws"])
    if len(rows) != steps * draws:
        raise RuntimeError("RGB metric artifact row count mismatch")
    identities = {(int(r["draw_id"]), int(r["step"])) for r in rows}
    expected = {(draw, step) for draw in range(draws) for step in range(1, steps + 1)}
    if identities != expected:
        raise RuntimeError("RGB metric artifact draw/step coverage mismatch")
    if any(int(row["episode_id"]) != episode_id for row in rows):
        raise RuntimeError("RGB metric artifact episode identity mismatch")
    required = set(PER_DRAW_STEP_FIELDS)
    if any(not required.issubset(row) for row in rows):
        raise RuntimeError("RGB metric artifact has incomplete rows")


def aggregate_draws_to_episode_steps(
    rows: Sequence[dict[str, Any]], *, expected_draws: int | None = None
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["episode_id"]), int(row["step"]))].append(row)
    output = []
    for (episode_id, step), values in sorted(grouped.items()):
        draw_ids = {int(row["draw_id"]) for row in values}
        if len(draw_ids) != len(values):
            raise RuntimeError("duplicate draw at an episode/step identity")
        if expected_draws is not None and draw_ids != set(range(expected_draws)):
            raise RuntimeError(f"episode {episode_id} step {step} draw coverage mismatch")
        def mean(key: str) -> float:
            return sum(float(row[key]) for row in values) / len(values)
        recon_ssim_values = {float(row["recon_ssim"]) for row in values}
        recon_lpips_values = {float(row["recon_lpips"]) for row in values}
        if len(recon_ssim_values) != 1 or len(recon_lpips_values) != 1:
            raise RuntimeError("draw-independent reconstruction baseline changed across draws")
        output.append({
            "episode_id": episode_id, "task_id": values[0].get("task_id"),
            "step": step, "time_sec": float(values[0]["time_sec"]),
            "num_draws": len(values), "mean_latent_mse": mean("latent_mse"),
            "mean_relative_l2": mean("relative_l2"), "mean_cosine": mean("cosine"),
            "mean_pred_ssim": mean("pred_ssim"),
            "recon_ssim": next(iter(recon_ssim_values)),
            "mean_ssim_gap": mean("ssim_gap"),
            "mean_pred_lpips": mean("pred_lpips"),
            "recon_lpips": next(iter(recon_lpips_values)),
            "mean_lpips_gap": mean("lpips_gap"),
        })
    return output


def aggregate_episode_steps(
    rows: Sequence[dict[str, Any]],
    *,
    cohort_episode_ids: set[int] | None = None,
    max_step: int | None = None,
    bootstrap_samples: int = 10000,
    bootstrap_seed: int = 4242,
) -> list[dict[str, Any]]:
    selected = [
        row for row in rows
        if (cohort_episode_ids is None or int(row["episode_id"]) in cohort_episode_ids)
        and (max_step is None or int(row["step"]) <= max_step)
    ]
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[int(row["step"])].append(row)
    output = []
    for step, values in sorted(grouped.items()):
        item: dict[str, Any] = {
            "step": step, "time_sec": float(values[0]["time_sec"]),
            "n_episodes": len(values),
        }
        for prefix, key in (
            ("latent_mse", "mean_latent_mse"),
            ("ssim_gap", "mean_ssim_gap"),
            ("lpips_gap", "mean_lpips_gap"),
        ):
            tensor = torch.tensor([float(row[key]) for row in values], dtype=torch.float64)
            mean, low, high = bootstrap_mean_ci(
                tensor.tolist(), samples=bootstrap_samples, seed=bootstrap_seed + step
            )
            item.update({
                f"{prefix}_mean": mean,
                f"{prefix}_median": float(torch.quantile(tensor, 0.5)),
                f"{prefix}_p90": float(torch.quantile(tensor, 0.9)),
                f"{prefix}_ci_low": low, f"{prefix}_ci_high": high,
            })
        output.append(item)
    return output


def fixed_cohort_ids(
    rows: Sequence[dict[str, Any]], horizon: int
) -> set[int]:
    if horizon <= 0:
        raise ValueError("fixed cohort horizon must be positive")
    maximum: dict[int, int] = defaultdict(int)
    for row in rows:
        episode = int(row["episode_id"])
        maximum[episode] = max(maximum[episode], int(row["step"]))
    return {episode for episode, step in maximum.items() if step >= horizon}


def prediction_latent_disk_bytes(root: str | Path) -> int:
    return sum(path.stat().st_size for path in (Path(root) / "latents").glob("*.pt"))


def generate_aggregate_plots(
    output_dir: str | Path,
    available_rows: Sequence[dict[str, Any]],
    fixed_rows: dict[int, Sequence[dict[str, Any]]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("plot generation requires `pip install matplotlib`") from error
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def line_plot(rows, key, ylabel, filename):
        fig, axis = plt.subplots(figsize=(8, 5))
        axis.plot([r["step"] for r in rows], [r[key] for r in rows])
        axis.set(xlabel="rollout step", ylabel=ylabel)
        axis.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(output_dir / filename, dpi=160); plt.close(fig)

    line_plot(available_rows, "ssim_gap_mean", "mean SSIM gap (higher=worse)", "ssim_gap_available.png")
    line_plot(available_rows, "lpips_gap_mean", "mean LPIPS gap (higher=worse)", "lpips_gap_available.png")
    line_plot(available_rows, "latent_mse_mean", "mean latent MSE", "latent_mse_available.png")
    line_plot(available_rows, "n_episodes", "available physical episodes", "n_episodes_available.png")
    for key, ylabel, filename in (
        ("ssim_gap_mean", "mean SSIM gap (higher=worse)", "ssim_gap_fixed_cohorts.png"),
        ("lpips_gap_mean", "mean LPIPS gap (higher=worse)", "lpips_gap_fixed_cohorts.png"),
    ):
        fig, axis = plt.subplots(figsize=(8, 5))
        for horizon, rows in sorted(fixed_rows.items()):
            n = int(rows[0]["n_episodes"]) if rows else 0
            if rows:
                axis.plot([r["step"] for r in rows], [r[key] for r in rows], label=f"H={horizon}, n={n}")
        axis.set(xlabel="rollout step", ylabel=ylabel); axis.grid(alpha=0.25)
        if axis.lines: axis.legend()
        fig.tight_layout(); fig.savefig(output_dir / filename, dpi=160); plt.close(fig)
