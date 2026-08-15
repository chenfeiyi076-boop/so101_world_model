from __future__ import annotations

import argparse
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
from src.causal.data.common import cache_paths_from_manifest, load_episode_cache
from src.causal.data.multi_episode_dataset import (
    episode_ids_for_split,
    load_causal_manifest,
)
from src.causal.rollout import RolloutStream, autoregressive_causal_rollout_batch
from src.causal.rollout_visualization import (
    build_stream_metadata,
    create_contact_sheet,
    decode_cached_latents,
    group_stream_rows,
    load_frozen_vae,
    normalized_display_steps,
    parse_stream_spec,
    read_rollout_metric_rows,
    save_png_sequence,
    select_explicit_streams,
    select_quantile_representatives,
    validate_summary_checkpoint_compatibility,
    validate_rerun,
)
from src.causal.runtime import build_model, validate_precision_device
from src.so101_cache.vae_cache import FPS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministically rerun and VAE-decode selected causal rollouts"
    )
    parser.add_argument("--rollout-eval-dir", type=Path, required=True)
    parser.add_argument("--vae-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--selection-metric", default="mean_mse", choices=("mean_mse",))
    parser.add_argument("--quantiles", nargs="+", type=float, default=(0.1, 0.5, 0.9))
    parser.add_argument(
        "--stream",
        action="append",
        default=[],
        help="Explicit EPISODE_ID:START:NOISE_DRAW; repeatable",
    )
    parser.add_argument(
        "--display-steps", nargs="+", type=int, default=(1, 4, 8, 16, 32)
    )
    parser.add_argument("--decode-batch-size", type=int, default=8)
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


def main() -> None:
    args = build_parser().parse_args()
    if args.decode_batch_size <= 0:
        raise ValueError("decode-batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    summary_path = args.rollout_eval_dir / "summary.json"
    rows_path = args.rollout_eval_dir / "per_rollout_step.csv"
    summary = _load_json(summary_path)
    rows = read_rollout_metric_rows(rows_path)
    grouped_rows = group_stream_rows(rows)
    if args.stream:
        identities = [parse_stream_spec(value) for value in args.stream]
        if len(set(identities)) != len(identities):
            raise ValueError("explicit streams must not contain duplicates")
        selections, distribution = select_explicit_streams(
            rows, metric=args.selection_metric, identities=identities
        )
    else:
        selections, distribution = select_quantile_representatives(
            rows,
            metric=args.selection_metric,
            quantiles=args.quantiles,
        )

    checkpoint = load_checkpoint(summary["checkpoint_path"])
    config = checkpoint["config"]
    precision = evaluation_precision(config)
    state_dict, selected_weights = select_evaluation_state_dict(
        checkpoint, summary["weights_used"]
    )
    validate_summary_checkpoint_compatibility(
        summary,
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
    split_ids = episode_ids_for_split(manifest, summary["split"])
    cache_paths = cache_paths_from_manifest(manifest, split_ids)
    episode_cache = {}
    streams = []
    for selection in selections:
        episode_id = int(selection["episode_id"])
        if episode_id not in cache_paths:
            raise RuntimeError(f"selected episode {episode_id} is not in summary split")
        if episode_id not in episode_cache:
            episode_cache[episode_id] = load_episode_cache(
                cache_paths[episode_id], config["action"]["raw_action_dim"]
            )
        streams.append(
            RolloutStream(
                episode=episode_cache[episode_id],
                episode_id=episode_id,
                start=int(selection["start"]),
                noise_draw=int(selection["noise_draw"]),
            )
        )
    results = autoregressive_causal_rollout_batch(
        model=model,
        config=config,
        action_stats=action_stats,
        streams=streams,
        rollout_steps=int(summary["rollout_steps"]),
        euler_steps=int(summary["euler_steps"]),
        seed=int(summary["seed"]),
        device=device,
        precision=precision,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    vae, vae_directory = load_frozen_vae(args.vae_path, device)
    output_dir = args.output_dir or (args.rollout_eval_dir / "visualization")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_summary = dict(summary)
    metadata_summary["selection_metric"] = args.selection_metric
    display_steps = normalized_display_steps(
        args.display_steps, int(summary["rollout_steps"])
    )
    fps = float(FPS)
    for selection, result in zip(selections, results):
        identity = (
            int(selection["episode_id"]),
            int(selection["start"]),
            int(selection["noise_draw"]),
        )
        differences = validate_rerun(result, grouped_rows[identity])
        stream_dir = output_dir / selection["label"]
        stream_dir.mkdir(parents=True, exist_ok=True)
        all_latents = torch.cat(
            (
                result["history_latents"],
                result["predicted_future"],
                result["gt_future"],
            )
        )
        decoded = decode_cached_latents(
            vae,
            all_latents,
            device=device,
            batch_size=args.decode_batch_size,
        )
        history_count = len(result["history_latents"])
        rollout_steps = len(result["predicted_future"])
        history_images = decoded[:history_count]
        predicted_images = decoded[history_count : history_count + rollout_steps]
        gt_images = decoded[history_count + rollout_steps :]
        save_png_sequence(history_images, stream_dir / "history", "history")
        save_png_sequence(
            predicted_images, stream_dir / "pred", "step", start_index=1
        )
        save_png_sequence(
            gt_images, stream_dir / "gt_recon", "step", start_index=1
        )
        contact_sheet = create_contact_sheet(
            history_images=history_images,
            predicted_images=predicted_images,
            gt_reconstruction_images=gt_images,
            display_steps=display_steps,
            frame_stride=int(summary["frame_stride"]),
            fps=fps,
        )
        contact_sheet.save(stream_dir / "contact_sheet.png")
        metadata = build_stream_metadata(
            selection=selection,
            summary=metadata_summary,
            result=result,
            vae_path=vae_directory,
            vae=vae,
            fps=fps,
        )
        metadata["display_steps"] = display_steps
        metadata["rerun_metric_warning_count"] = len(differences)
        _write_json(stream_dir / "metadata.json", metadata)
        print(
            f"visualized {selection['label']}: episode={identity[0]} "
            f"start={identity[1]} draw={identity[2]}",
            flush=True,
        )

    selection_json = {
        "source_rollout_eval_dir": str(args.rollout_eval_dir),
        "selection_metric": args.selection_metric,
        "requested_quantiles": [] if args.stream else list(args.quantiles),
        "selected_streams": selections,
        "score_distribution": distribution,
    }
    _write_json(output_dir / "selection.json", selection_json)
    print("selection:", output_dir / "selection.json")
    print("output:", output_dir)


if __name__ == "__main__":
    main()
