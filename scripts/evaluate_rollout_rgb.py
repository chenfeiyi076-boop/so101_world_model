from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.causal.action_visualization import (
    load_original_rgb_frames,
    resolve_original_rgb_source,
)
from src.causal.data.common import cache_paths_from_manifest, load_episode_cache
from src.causal.data.multi_episode_dataset import episode_ids_for_split, load_causal_manifest
from src.causal.long_rollout_eval import (
    AGGREGATE_FIELDS,
    PER_DRAW_STEP_FIELDS,
    PER_EPISODE_STEP_FIELDS,
    RAW_RGB_PREPROCESSING,
    AlexNetLPIPS,
    FullEpisodePlan,
    aggregate_draws_to_episode_steps,
    aggregate_episode_steps,
    artifact_path,
    atomic_json_save,
    build_rgb_artifact_metadata,
    deterministic_balanced_shards,
    evaluate_stage_b_episode,
    file_sha256,
    fixed_cohort_ids,
    generate_aggregate_plots,
    prediction_latent_disk_bytes,
    read_csv,
    reader_for_source,
    rgb_artifact_path,
    validate_image_metric_identity,
    validate_rgb_metric_artifact,
    rgb_resume_decision,
    vae_provenance,
    validate_aggregate_only_provenance,
    write_csv,
)
from src.causal.rollout_visualization import decode_cached_latents, load_frozen_vae
from src.so101_cache.vae_cache import LATENT_CONVENTION, atomic_torch_save


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline VAE-decoded RGB fidelity evaluation for Stage A artifacts"
    )
    parser.add_argument("--long-rollout-dir", type=Path, required=True)
    parser.add_argument("--vae-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decode-batch-size", type=int, default=8)
    parser.add_argument("--metric-batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=4242)
    parser.add_argument(
        "--fixed-horizons", nargs="+", type=int, default=(32, 50, 100, 150, 200)
    )
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--aggregate-only", action="store_true",
        help="Regenerate CSV aggregations/plots without loading VAE, raw RGB, or world model.",
    )
    return parser


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def distributed_context(requested_device: str) -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1:
        dist.init_process_group(backend="nccl" if requested_device == "cuda" else "gloo")
    return rank, local_rank, world_size, device


def normalize_per_draw_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    integer = {"episode_id", "task_id", "draw_id", "step", "raw_frame_index", "sampled_frame_index"}
    output = []
    for row in rows:
        output.append({
            key: (None if value in (None, "", "None") else int(value))
            if key in integer else float(value)
            for key, value in row.items()
        })
    return output


def write_aggregations(
    *,
    output_dir: Path,
    per_draw_rows: list[dict[str, Any]],
    noise_draws: int,
    fixed_horizons: list[int],
    bootstrap_samples: int,
    bootstrap_seed: int,
    plot_fn: Callable = generate_aggregate_plots,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, list[dict[str, Any]]], dict[int, int]]:
    episode_rows = aggregate_draws_to_episode_steps(
        per_draw_rows, expected_draws=noise_draws
    )
    available = aggregate_episode_steps(
        episode_rows, bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    fixed: dict[int, list[dict[str, Any]]] = {}
    cohort_sizes = {}
    for horizon in sorted(set(fixed_horizons)):
        cohort = fixed_cohort_ids(episode_rows, horizon)
        cohort_sizes[horizon] = len(cohort)
        fixed[horizon] = aggregate_episode_steps(
            episode_rows, cohort_episode_ids=cohort, max_step=horizon,
            bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed,
        ) if cohort else []
        write_csv(
            output_dir / f"per_step_fixed_H{horizon:03d}.csv",
            fixed[horizon], AGGREGATE_FIELDS,
        )
    write_csv(output_dir / "per_draw_step.csv", per_draw_rows, PER_DRAW_STEP_FIELDS)
    write_csv(output_dir / "per_episode_step.csv", episode_rows, PER_EPISODE_STEP_FIELDS)
    write_csv(output_dir / "per_step_available.csv", available, AGGREGATE_FIELDS)
    plot_fn(output_dir, available, fixed)
    return episode_rows, available, fixed, cohort_sizes


def validate_stage_a_payload(payload: dict[str, Any], summary: dict[str, Any]) -> None:
    metadata = payload.get("metadata", {})
    expected = {
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "checkpoint_step": int(summary["checkpoint_step"]),
        "weights_used": summary["weights_used"],
        "manifest_sha256": summary["manifest_sha256"],
        "rollout_seed": int(summary["seed"]),
        "euler_steps": int(summary["euler_steps"]),
        "frame_stride": int(summary["frame_stride"]),
        "num_history": int(summary["num_history"]),
        "latent_convention": LATENT_CONVENTION,
        "vae_identifier": summary["vae_identifier"],
        "scaling_factor": float(summary["scaling_factor"]),
        "shift_factor_used": False,
        "reference_used_as_model_input": False,
    }
    mismatch = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    if mismatch:
        raise RuntimeError(f"Stage A artifact provenance mismatch: {mismatch}")


def inspect_stage_a_draw_artifacts(
    long_rollout_dir: Path,
    plan: FullEpisodePlan,
    summary: dict[str, Any],
) -> tuple[list[Path], dict[str, str], torch.Tensor, torch.Tensor]:
    paths = []
    hashes = {}
    reference_raw = None
    reference_sampled = None
    for draw_id in range(int(summary["noise_draws"])):
        path = artifact_path(long_rollout_dir, plan.episode_id, draw_id)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        validate_stage_a_payload(payload, summary)
        metadata = payload["metadata"]
        if int(metadata["episode_id"]) != plan.episode_id:
            raise RuntimeError("Stage A episode identity mismatch")
        if int(metadata["draw_id"]) != draw_id:
            raise RuntimeError("Stage A draw identity mismatch")
        if int(metadata["number_of_steps"]) != plan.rollout_steps:
            raise RuntimeError("Stage A rollout length mismatch")
        raw = torch.as_tensor(payload["raw_frame_indices"]).long()
        sampled = torch.as_tensor(payload["sampled_frame_indices"]).long()
        if raw.shape != (plan.rollout_steps,) or sampled.shape != (plan.rollout_steps,):
            raise RuntimeError("Stage A target frame index length mismatch")
        if reference_raw is None:
            reference_raw, reference_sampled = raw, sampled
        elif not torch.equal(raw, reference_raw) or not torch.equal(sampled, reference_sampled):
            raise RuntimeError("Stage A draws disagree on physical target frames")
        paths.append(path)
        hashes[str(draw_id)] = file_sha256(path)
        del payload
    return paths, hashes, reference_raw, reference_sampled


def main() -> None:
    args = build_parser().parse_args()
    if args.decode_batch_size <= 0 or args.metric_batch_size <= 0:
        raise ValueError("decode/metric batch sizes must be positive")
    if args.bootstrap_samples <= 0 or any(value <= 0 for value in args.fixed_horizons):
        raise ValueError("bootstrap samples and fixed horizons must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage_a_summary = load_json(args.long_rollout_dir / "summary.json")
    stage_a_summary_sha = file_sha256(args.long_rollout_dir / "summary.json")
    if stage_a_summary.get("stage_a_completion_status") != "complete":
        raise RuntimeError("Stage A is not complete")

    if args.aggregate_only:
        existing = load_json(args.output_dir / "summary.json")
        validate_aggregate_only_provenance(
            stage_a_summary=stage_a_summary,
            existing_stage_b_summary=existing,
            long_rollout_dir=args.long_rollout_dir,
            stage_a_summary_sha256=stage_a_summary_sha,
        )
        rows = normalize_per_draw_rows(read_csv(args.output_dir / "per_draw_step.csv"))
        _, _, _, cohort_sizes = write_aggregations(
            output_dir=args.output_dir, per_draw_rows=rows,
            noise_draws=int(stage_a_summary["noise_draws"]),
            fixed_horizons=list(args.fixed_horizons),
            bootstrap_samples=args.bootstrap_samples, bootstrap_seed=args.bootstrap_seed,
        )
        existing["fixed_horizons"] = sorted(set(args.fixed_horizons))
        existing["fixed_cohort_episode_counts"] = {str(k): v for k, v in cohort_sizes.items()}
        existing["bootstrap_samples"] = args.bootstrap_samples
        existing["bootstrap_seed"] = args.bootstrap_seed
        atomic_json_save(existing, args.output_dir / "summary.json")
        print("offline aggregation/plots regenerated; no DiT or VAE loaded")
        return
    if args.vae_path is None:
        raise ValueError("--vae-path is required unless --aggregate-only is used")

    rank = 0
    try:
        rank, _local_rank, world_size, device = distributed_context(args.device)
        started = time.perf_counter()
        manifest_path = Path(stage_a_summary["manifest_path"])
        if file_sha256(manifest_path) != stage_a_summary["manifest_sha256"]:
            raise RuntimeError("manifest changed after Stage A")
        manifest = load_causal_manifest(manifest_path)
        all_split_ids = episode_ids_for_split(manifest, stage_a_summary["split"])
        paths = cache_paths_from_manifest(manifest, all_split_ids)
        artifact_files = sorted((args.long_rollout_dir / "latents").glob("episode_*_draw_0.pt"))
        episode_ids = []
        plans = []
        for path in artifact_files:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            validate_stage_a_payload(payload, stage_a_summary)
            metadata = payload["metadata"]
            episode_id = int(metadata["episode_id"])
            if episode_id not in paths:
                raise RuntimeError(f"Stage A episode {episode_id} is outside declared split")
            episode_ids.append(episode_id)
            plans.append(FullEpisodePlan(
                episode_id=episode_id, task_id=metadata.get("task_id"),
                cache_path=paths[episode_id], rollout_steps=int(metadata["number_of_steps"]),
            ))
        if len(plans) != int(stage_a_summary["number_of_episodes"]):
            raise RuntimeError("Stage A episode artifact coverage mismatch")
        if len(set(episode_ids)) != len(episode_ids):
            raise RuntimeError("Stage A contains duplicate physical episode identities")
        expected_stage_a_paths = {
            artifact_path(args.long_rollout_dir, plan.episode_id, draw_id).resolve()
            for plan in plans
            for draw_id in range(int(stage_a_summary["noise_draws"]))
        }
        actual_stage_a_paths = {
            path.resolve()
            for path in (args.long_rollout_dir / "latents").glob("*.pt")
        }
        if actual_stage_a_paths != expected_stage_a_paths:
            raise RuntimeError("Stage A draw artifact coverage has missing or extra files")
        shards = deterministic_balanced_shards(plans, world_size)
        assigned = shards[rank]

        # Heavy video dependencies remain outside module import/help paths.
        from src.so101_cache.so101_reader import SO101LeRobotV3Reader

        # resolve_original_rgb_source returns a dataset root + camera, not a
        # concrete shard. The reader resolves each episode's shard internally.
        reader_cache = {}
        reader_factory = lambda root, camera: SO101LeRobotV3Reader(
            root, camera=camera
        )
        vae, vae_directory = load_frozen_vae(args.vae_path, device)
        if not math_isclose(float(vae.config.scaling_factor), float(stage_a_summary["scaling_factor"])):
            raise RuntimeError("VAE scaling_factor differs from Stage A cache provenance")
        lpips_metric = AlexNetLPIPS(device)
        validate_image_metric_identity(lpips_metric)
        if dist.is_initialized():
            shared_vae_identity = [
                vae_provenance(vae_directory) if rank == 0 else None
            ]
            dist.broadcast_object_list(shared_vae_identity, src=0)
            vae_identity = shared_vae_identity[0]
        else:
            vae_identity = vae_provenance(vae_directory)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        generated_frames = 0
        skipped_episodes = 0
        for ordinal, plan in enumerate(assigned, start=1):
            output_path = rgb_artifact_path(args.output_dir, plan.episode_id)
            (
                draw_paths,
                draw_hashes,
                raw_indices,
                sampled_rows,
            ) = inspect_stage_a_draw_artifacts(
                args.long_rollout_dir, plan, stage_a_summary
            )
            expected_metadata = build_rgb_artifact_metadata(
                episode_id=plan.episode_id,
                number_of_steps=plan.rollout_steps,
                stage_a_summary=stage_a_summary,
                stage_a_draw_artifact_sha256=draw_hashes,
                vae_config_sha256=vae_identity["vae_config_sha256"],
                vae_weights_sha256=vae_identity["vae_weights_sha256"],
                vae_weight_files_sha256=vae_identity["vae_weight_files_sha256"],
                vae_artifact_sha256=vae_identity["vae_artifact_sha256"],
                vae_path=vae_directory,
                scaling_factor=float(vae.config.scaling_factor),
            )
            if rgb_resume_decision(
                output_path, expected_metadata, overwrite=args.overwrite
            ) == "skip":
                skipped_episodes += 1
                continue

            source = resolve_original_rgb_source(manifest, plan.cache_path)
            reader = reader_for_source(
                reader_cache, source_path=source[0], camera=source[1],
                factory=reader_factory,
            )
            cache = load_episode_cache(
                plan.cache_path,
                raw_action_dim=int(stage_a_summary["raw_action_dim"]),
            )
            if not torch.equal(cache["frame_indices"][sampled_rows], raw_indices):
                raise RuntimeError("prediction/GT cache physical-frame alignment mismatch")
            raw_uint8 = load_original_rgb_frames(
                reader, episode_id=plan.episode_id,
                requested_frame_indices=raw_indices.tolist(),
            )
            draw_payloads = (
                torch.load(path, map_location="cpu", weights_only=False)
                for path in draw_paths
            )
            rows = evaluate_stage_b_episode(
                draw_payloads=draw_payloads,
                raw_gt_uint8=raw_uint8,
                gt_latents=cache["latents"][sampled_rows],
                vae=vae,
                decode_fn=decode_cached_latents,
                lpips_metric=lpips_metric,
                device=device,
                decode_batch_size=args.decode_batch_size,
                metric_batch_size=args.metric_batch_size,
                expected_draws=int(stage_a_summary["noise_draws"]),
            )
            del raw_uint8, cache, draw_payloads
            saved = {"metadata": expected_metadata, "rows": rows}
            validate_rgb_metric_artifact(saved, expected_metadata)
            atomic_torch_save(saved, output_path)
            generated_frames += plan.rollout_steps * (
                int(stage_a_summary["noise_draws"]) + 1
            )
            print(f"rank {rank}: RGB episode {plan.episode_id} ({ordinal}/{len(assigned)})", flush=True)

        peak_cuda_memory = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" else None
        )
        rank_stats = {
            "rank": rank, "episodes": len(assigned), "decoded_frames": generated_frames,
            "skipped_episodes": skipped_episodes, "wall_seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": peak_cuda_memory,
        }
        gathered = [None] * world_size if rank == 0 else None
        if dist.is_initialized():
            dist.gather_object(rank_stats, gathered, dst=0); dist.barrier()
        else:
            gathered = [rank_stats]
        if rank == 0:
            all_rows = []
            for plan in sorted(plans, key=lambda value: value.episode_id):
                saved = torch.load(rgb_artifact_path(args.output_dir, plan.episode_id), map_location="cpu", weights_only=False)
                if not isinstance(saved.get("rows"), list):
                    raise RuntimeError("RGB metric artifact is incomplete during merge")
                all_rows.extend(saved["rows"])
            all_rows.sort(key=lambda row: (int(row["episode_id"]), int(row["draw_id"]), int(row["step"])))
            episode_rows, available, _fixed, cohort_sizes = write_aggregations(
                output_dir=args.output_dir, per_draw_rows=all_rows,
                noise_draws=int(stage_a_summary["noise_draws"]),
                fixed_horizons=list(args.fixed_horizons),
                bootstrap_samples=args.bootstrap_samples, bootstrap_seed=args.bootstrap_seed,
            )
            wall = max(float(item["wall_seconds"]) for item in gathered)
            frames = sum(int(item["decoded_frames"]) for item in gathered)
            summary = {
                **stage_a_summary,
                "stage_b_completion_status": "complete",
                "stage_a_long_rollout_dir": str(args.long_rollout_dir.resolve()),
                "stage_a_summary_sha256": stage_a_summary_sha,
                "vae_path": str(vae_directory.resolve()),
                **vae_identity,
                "ssim_implementation": "project Gaussian 11x11 sigma=1.5 per-image RGB",
                "ssim_data_range": 1.0,
                "lpips_implementation": "lpips.LPIPS",
                "lpips_backbone": "alex",
                "ssim_gap_definition": "recon_ssim - pred_ssim; higher is worse",
                "lpips_gap_definition": "pred_lpips - recon_lpips; higher is worse",
                "available_case_aggregation_definition": "at step t, average physical episodes with GT available at t after averaging each episode's draw metrics",
                "fixed_cohort_aggregation_definition": "for horizon H, one fixed set of episodes reaching H contributes at every step 1..H",
                "bootstrap_unit": "physical episode",
                "bootstrap_samples": args.bootstrap_samples,
                "bootstrap_seed": args.bootstrap_seed,
                "fixed_horizons": sorted(set(args.fixed_horizons)),
                "fixed_cohort_episode_counts": {str(k): v for k, v in cohort_sizes.items()},
                "stage_b_world_size": world_size,
                "stage_b_gpu_count": world_size if device.type == "cuda" else 0,
                "metric_device": str(device),
                "decode_batch_size": args.decode_batch_size,
                "metric_batch_size": args.metric_batch_size,
                "raw_rgb_preprocessing": RAW_RGB_PREPROCESSING,
                "stage_b_wall_seconds": wall,
                "stage_b_decoded_frames": frames,
                "stage_b_frames_per_second": frames / wall if wall else None,
                "stage_b_per_rank_performance": gathered,
                "stage_b_peak_cuda_memory_bytes_max": max(
                    (
                        int(item["peak_cuda_memory_bytes"])
                        for item in gathered
                        if item["peak_cuda_memory_bytes"] is not None
                    ),
                    default=None,
                ),
                "prediction_latent_disk_bytes": prediction_latent_disk_bytes(args.long_rollout_dir),
                "number_of_per_draw_step_rows": len(all_rows),
                "number_of_per_episode_step_rows": len(episode_rows),
                "number_of_available_steps": len(available),
            }
            atomic_json_save(summary, args.output_dir / "summary.json")
            print("Stage B complete:", args.output_dir, flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def math_isclose(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-8 * max(1.0, abs(left), abs(right))


if __name__ == "__main__":
    main()
