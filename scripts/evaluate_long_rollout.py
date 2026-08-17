from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_causal import evaluation_precision, select_evaluation_state_dict
from src.causal.checkpointing import action_stats_from_checkpoint, load_checkpoint
from src.causal.data.common import cache_paths_from_manifest, load_episode_cache
from src.causal.data.multi_episode_dataset import episode_ids_for_split, load_causal_manifest
from src.causal.long_rollout_eval import (
    ARTIFACT_VERSION,
    STAGE_A_FIELDS,
    FullEpisodePlan,
    artifact_latent_rows,
    artifact_path,
    atomic_json_save,
    build_artifact_metadata,
    deterministic_balanced_shards,
    file_sha256,
    full_episode_rollout_steps,
    make_long_rollout_artifact,
    prediction_latent_disk_bytes,
    resume_decision,
    validate_long_rollout_artifact,
    write_csv,
)
from src.causal.rollout import RolloutStream, autoregressive_causal_rollout_batch
from src.causal.runtime import build_model, validate_precision_device
from src.so101_cache.vae_cache import FPS, LATENT_CONVENTION, atomic_torch_save


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full-episode long-horizon causal latent rollout evaluation"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--noise-draws", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260)
    parser.add_argument("--euler-steps", type=int, default=10)
    parser.add_argument(
        "--max-rollout-steps", type=int,
        help="Optional safety cap; omitted means continue to each episode's GT end.",
    )
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--weights", default="auto", choices=("auto", "ema", "raw"))
    parser.add_argument("--overwrite", action="store_true")
    return parser


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
        backend = "nccl" if requested_device == "cuda" else "gloo"
        dist.init_process_group(backend=backend)
    return rank, local_rank, world_size, device


def cache_metadata(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError(f"cache lacks required metadata: {path}")
    if metadata.get("latent_convention") != LATENT_CONVENTION:
        raise RuntimeError(f"cache latent convention mismatch: {path}")
    if metadata.get("shift_factor_used") is not False:
        raise RuntimeError(f"cache unexpectedly used a VAE shift: {path}")
    return metadata


def build_plans(
    manifest: dict[str, Any],
    *,
    split: str,
    raw_action_dim: int,
    num_history: int,
    frame_stride: int,
    max_rollout_steps: int | None,
    max_episodes: int,
) -> list[FullEpisodePlan]:
    episode_ids = episode_ids_for_split(manifest, split)
    if max_episodes > 0:
        episode_ids = episode_ids[:max_episodes]
    paths = cache_paths_from_manifest(manifest, episode_ids)
    entries = {int(item["episode_index"]): item for item in manifest["episodes"]}
    plans = []
    for episode_id in episode_ids:
        episode = load_episode_cache(paths[episode_id], raw_action_dim)
        steps = full_episode_rollout_steps(
            len(episode["latents"]), num_history=num_history,
            frame_stride=frame_stride, max_rollout_steps=max_rollout_steps,
        )
        if steps <= 0:
            raise RuntimeError(f"episode {episode_id} has no legal future rollout steps")
        task = entries[episode_id].get("task_index")
        plans.append(FullEpisodePlan(
            episode_id=episode_id, task_id=None if task is None else int(task),
            cache_path=paths[episode_id], rollout_steps=steps,
        ))
    return plans


def main() -> None:
    args = build_parser().parse_args()
    if args.noise_draws <= 0 or args.euler_steps <= 0 or args.max_episodes < 0:
        raise ValueError("noise draws/euler steps must be positive; max episodes non-negative")
    rank = 0
    world_size = 1
    initialized = False
    try:
        rank, _local_rank, world_size, device = distributed_context(args.device)
        initialized = dist.is_initialized()
        started = time.perf_counter()
        checkpoint = load_checkpoint(args.checkpoint)
        config = checkpoint["config"]
        if config["data"]["mode"] != "multi_episode":
            raise RuntimeError("full-test long rollout requires a multi_episode checkpoint")
        if config.get("latent", {}).get("convention") != LATENT_CONVENTION:
            raise RuntimeError("checkpoint latent convention is incompatible")
        precision = evaluation_precision(config)
        validate_precision_device(precision, device)
        state_dict, weights_used = select_evaluation_state_dict(checkpoint, args.weights)
        if initialized:
            shared = [file_sha256(args.checkpoint) if rank == 0 else None]
            dist.broadcast_object_list(shared, src=0)
            checkpoint_sha = shared[0]
        else:
            checkpoint_sha = file_sha256(args.checkpoint)
        manifest_path = Path(config["data"]["manifest_path"])
        manifest = load_causal_manifest(manifest_path)
        manifest_sha = file_sha256(manifest_path)
        temporal, action = config["temporal"], config["action"]
        if initialized:
            shared_plans = [build_plans(
                manifest, split=args.split, raw_action_dim=int(action["raw_action_dim"]),
                num_history=int(temporal["num_history"]),
                frame_stride=int(temporal["frame_stride"]),
                max_rollout_steps=args.max_rollout_steps,
                max_episodes=args.max_episodes,
            ) if rank == 0 else None]
            dist.broadcast_object_list(shared_plans, src=0)
            plans = shared_plans[0]
        else:
            plans = build_plans(
                manifest, split=args.split, raw_action_dim=int(action["raw_action_dim"]),
                num_history=int(temporal["num_history"]),
                frame_stride=int(temporal["frame_stride"]),
                max_rollout_steps=args.max_rollout_steps,
                max_episodes=args.max_episodes,
            )
        shards = deterministic_balanced_shards(plans, world_size)
        assigned = shards[rank]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if rank == 0:
            print("full-episode long rollout", flush=True)
            print("checkpoint:", args.checkpoint, flush=True)
            print("weights:", weights_used, flush=True)
            print("episodes:", len(plans), "draws:", args.noise_draws, flush=True)
            print("world size:", world_size, "precision:", precision, flush=True)
            print("per-rank expected steps:", [sum(p.rollout_steps for p in s) for s in shards], flush=True)

        model = build_model(config).to(device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        action_stats = action_stats_from_checkpoint(checkpoint)
        generated_steps = 0
        skipped_trajectories = 0
        for ordinal, plan in enumerate(assigned, start=1):
            episode = load_episode_cache(plan.cache_path, int(action["raw_action_dim"]))
            metadata_from_cache = cache_metadata(plan.cache_path)
            pending_streams, expected_by_draw = [], {}
            for draw_id in range(args.noise_draws):
                expected = build_artifact_metadata(
                    plan=plan, draw_id=draw_id, checkpoint_path=args.checkpoint,
                    checkpoint_sha256=checkpoint_sha, checkpoint_step=int(checkpoint["step"]),
                    weights_used=weights_used, seed=args.seed, euler_steps=args.euler_steps,
                    config=config, manifest_path=manifest_path, manifest_sha256=manifest_sha,
                    cache_metadata=metadata_from_cache,
                )
                expected_by_draw[draw_id] = expected
                path = artifact_path(args.output_dir, plan.episode_id, draw_id)
                if resume_decision(path, expected, overwrite=args.overwrite) == "skip":
                    skipped_trajectories += 1
                else:
                    pending_streams.append(RolloutStream(
                        episode=episode, episode_id=plan.episode_id,
                        start=0, noise_draw=draw_id,
                    ))
            if pending_streams:
                results = autoregressive_causal_rollout_batch(
                    model=model, config=config, action_stats=action_stats,
                    streams=pending_streams, rollout_steps=plan.rollout_steps,
                    euler_steps=args.euler_steps, seed=args.seed,
                    device=device, precision=precision,
                )
                for result in results:
                    draw_id = int(result["noise_draw"])
                    payload = make_long_rollout_artifact(result, expected_by_draw[draw_id])
                    validate_long_rollout_artifact(payload, expected_by_draw[draw_id])
                    atomic_torch_save(payload, artifact_path(
                        args.output_dir, plan.episode_id, draw_id
                    ))
                    generated_steps += plan.rollout_steps
            print(
                f"rank {rank}: episode {plan.episode_id} ({ordinal}/{len(assigned)}) "
                f"steps={plan.rollout_steps} pending_draws={len(pending_streams)}",
                flush=True,
            )
            del episode

        rank_stats = {
            "rank": rank, "episodes": len(assigned),
            "assigned_steps_per_draw": sum(plan.rollout_steps for plan in assigned),
            "generated_steps": generated_steps,
            "skipped_trajectories": skipped_trajectories,
            "wall_seconds": time.perf_counter() - started,
        }
        gathered = [None] * world_size if rank == 0 else None
        if initialized:
            dist.gather_object(rank_stats, gathered, dst=0)
            dist.barrier()
        else:
            gathered = [rank_stats]

        if rank == 0:
            rows = []
            total_steps = 0
            expected_paths = {
                artifact_path(args.output_dir, plan.episode_id, draw_id).resolve()
                for plan in plans for draw_id in range(args.noise_draws)
            }
            actual_paths = {
                path.resolve() for path in (args.output_dir / "latents").glob("*.pt")
            }
            if actual_paths != expected_paths:
                raise RuntimeError(
                    "long-rollout artifact coverage mismatch: "
                    f"missing={sorted(map(str, expected_paths - actual_paths))}, "
                    f"extra={sorted(map(str, actual_paths - expected_paths))}"
                )
            for plan in sorted(plans, key=lambda item: item.episode_id):
                metadata_from_cache = cache_metadata(plan.cache_path)
                for draw_id in range(args.noise_draws):
                    expected = build_artifact_metadata(
                        plan=plan, draw_id=draw_id, checkpoint_path=args.checkpoint,
                        checkpoint_sha256=checkpoint_sha,
                        checkpoint_step=int(checkpoint["step"]), weights_used=weights_used,
                        seed=args.seed, euler_steps=args.euler_steps, config=config,
                        manifest_path=manifest_path, manifest_sha256=manifest_sha,
                        cache_metadata=metadata_from_cache,
                    )
                    payload = torch.load(
                        artifact_path(args.output_dir, plan.episode_id, draw_id),
                        map_location="cpu", weights_only=False,
                    )
                    validate_long_rollout_artifact(payload, expected)
                    rows.extend(artifact_latent_rows(payload, fps=FPS))
                    total_steps += plan.rollout_steps
            rows.sort(key=lambda row: (row["episode_id"], row["draw_id"], row["step"]))
            write_csv(args.output_dir / "per_draw_step_latent.csv", rows, STAGE_A_FIELDS)
            wall = max(float(item["wall_seconds"]) for item in gathered)
            summary = {
                "artifact_version": ARTIFACT_VERSION,
                "checkpoint_path": str(args.checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_step": int(checkpoint["step"]),
                "weights_used": weights_used, "precision": precision,
                "dataset": manifest.get("dataset"),
                "dataset_root": manifest.get("dataset_root"),
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": manifest_sha, "split": args.split,
                "number_of_episodes": len(plans), "noise_draws": args.noise_draws,
                "seed": args.seed, "euler_steps": args.euler_steps,
                "frame_stride": int(temporal["frame_stride"]), "fps": FPS,
                "num_history": int(temporal["num_history"]),
                "raw_action_dim": int(action["raw_action_dim"]),
                "action_alignment": action["alignment"],
                "action_representation": action["representation"],
                "effective_action_dim": int(action["effective_action_dim"]),
                "max_rollout_steps": args.max_rollout_steps,
                "actual_total_stochastic_trajectories": len(plans) * args.noise_draws,
                "actual_total_prediction_steps": total_steps,
                "vae_identifier": rows and cache_metadata(plans[0].cache_path).get("vae_identifier"),
                "latent_convention": LATENT_CONVENTION,
                "scaling_factor": rows and cache_metadata(plans[0].cache_path).get("scaling_factor"),
                "shift_factor_used": False,
                "world_size": world_size,
                "gpu_count": world_size if device.type == "cuda" else 0,
                "stage_a_completion_status": "complete",
                "stage_b_completion_status": "not_started",
                "stage_a_wall_seconds": wall,
                "stage_a_generated_steps": sum(int(item["generated_steps"]) for item in gathered),
                "stage_a_steps_per_second": (
                    sum(int(item["generated_steps"]) for item in gathered) / wall
                    if wall else None
                ),
                "per_rank_performance": gathered,
                "prediction_latent_disk_bytes": prediction_latent_disk_bytes(args.output_dir),
                "reference_used_as_model_input": False,
                "episode_start_definition": "earliest legal sampled history, cache row 0",
                "draw_aggregation_definition": "metrics are computed per stochastic draw; latents are never averaged",
            }
            atomic_json_save(summary, args.output_dir / "summary.json")
            print("Stage A complete:", args.output_dir, flush=True)
            print("prediction latent bytes:", summary["prediction_latent_disk_bytes"], flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
