from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_causal import evaluation_precision, select_evaluation_state_dict
from src.causal.action_counterfactual import (
    SHUFFLE_TYPE,
    action_counterfactual_rollouts,
    validate_selected_case_split_membership,
    validate_source_checkpoint_compatibility,
)
from src.causal.action_visualization import (
    attach_representative_draws,
    build_action_metadata,
    build_selection_document,
    create_action_contact_sheet,
    load_original_rgb_frames,
    normalize_display_steps,
    read_action_csv,
    resolve_original_rgb_source,
    select_case_quantiles,
    select_manual_streams,
    validate_action_rerun,
    validate_shuffle_permutation,
    validate_source_result_tables,
)
from src.causal.checkpointing import action_stats_from_checkpoint, load_checkpoint
from src.causal.data.common import cache_paths_from_manifest, load_episode_cache
from src.causal.data.multi_episode_dataset import (
    episode_ids_for_split,
    load_causal_manifest,
)
from src.causal.rollout import RolloutStream
from src.causal.rollout_visualization import (
    decode_cached_latents,
    load_frozen_vae,
    parse_stream_spec,
    save_png_sequence,
)
from src.causal.runtime import build_model, validate_precision_device
SELECTION_TABLE_FIELDS = (
    "label",
    "quantile",
    "target_score",
    "episode_id",
    "start",
    "actual_case_score",
    "selected_noise_draw",
    "selected_draw_score",
    "draw_selection_rule",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize representative TRUE/shuffled-action causal rollouts"
    )
    parser.add_argument("--action-eval-dir", required=True, type=Path)
    parser.add_argument("--vae-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--quantiles", nargs="+", type=float, default=(0.1, 0.5, 0.9))
    parser.add_argument(
        "--display-steps", nargs="+", type=int, default=(1, 4, 8, 16, 24, 32)
    )
    parser.add_argument("--decode-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument(
        "--stream",
        action="append",
        default=[],
        help="Explicit EPISODE_ID:START:NOISE_DRAW; repeatable",
    )
    parser.add_argument("--save-individual-frames", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
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


def _write_selection_table(path: Path, selections: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SELECTION_TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(selections)


def _ensure_output_directory(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"output directory is not empty: {path}; pass --overwrite to reuse it"
        )
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = build_parser().parse_args()
    if args.decode_batch_size <= 0:
        raise ValueError("decode-batch-size must be positive")
    _ensure_output_directory(args.output_dir, args.overwrite)

    summary = _load_json(args.action_eval_dir / "summary.json")
    case_rows = read_action_csv(args.action_eval_dir / "per_case.csv", kind="per_case")
    rollout_rows = read_action_csv(
        args.action_eval_dir / "per_rollout.csv", kind="per_rollout"
    )
    rollout_step_rows = read_action_csv(
        args.action_eval_dir / "per_rollout_step.csv", kind="per_rollout_step"
    )
    validate_source_result_tables(
        summary, case_rows, rollout_rows, rollout_step_rows
    )
    if summary.get("shuffle_type") != SHUFFLE_TYPE:
        raise RuntimeError("unsupported action shuffle type")
    if summary.get("counterfactual_gt_available") is not False:
        raise RuntimeError("action visualization requires counterfactual_gt_available=false")

    if args.stream:
        identities = [parse_stream_spec(value) for value in args.stream]
        selections = select_manual_streams(identities, case_rows, rollout_rows)
    else:
        selections = attach_representative_draws(
            select_case_quantiles(case_rows, args.quantiles), rollout_rows
        )
    display_steps = normalize_display_steps(
        args.display_steps, int(summary["rollout_steps"])
    )
    selection_document = build_selection_document(
        selections,
        action_eval_dir=args.action_eval_dir,
        num_candidate_cases=len(case_rows),
        quantiles=args.quantiles,
        manual=bool(args.stream),
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    checkpoint = load_checkpoint(summary["checkpoint_path"])
    config = checkpoint["config"]
    precision = evaluation_precision(config)
    state_dict, selected_weights = select_evaluation_state_dict(
        checkpoint, summary["weights_used"]
    )
    validate_source_checkpoint_compatibility(
        summary,
        checkpoint,
        precision=precision,
        weights_used=selected_weights,
    )
    validate_precision_device(precision, device)

    manifest = load_causal_manifest(config["data"]["manifest_path"])
    split_episode_ids = episode_ids_for_split(manifest, summary["split"])
    validate_selected_case_split_membership(
        selections,
        split=summary["split"],
        split_episode_ids=split_episode_ids,
    )
    selected_episode_ids = sorted(
        {int(selection["episode_id"]) for selection in selections}
    )
    cache_paths = cache_paths_from_manifest(manifest, selected_episode_ids)
    episode_cache = {
        episode_id: load_episode_cache(
            cache_paths[episode_id], config["action"]["raw_action_dim"]
        )
        for episode_id in selected_episode_ids
    }
    source_details = {
        episode_id: resolve_original_rgb_source(manifest, cache_paths[episode_id])
        for episode_id in selected_episode_ids
    }
    dataset_roots = {details[0].resolve() for details in source_details.values()}
    cameras = {details[1] for details in source_details.values()}
    if len(dataset_roots) != 1 or len(cameras) != 1:
        raise RuntimeError("selected caches disagree on Original RGB source")
    dataset_root = next(iter(dataset_roots))
    camera = next(iter(cameras))

    model = build_model(config).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    action_stats = action_stats_from_checkpoint(checkpoint)
    base_streams = [
        RolloutStream(
            episode=episode_cache[int(selection["episode_id"])],
            episode_id=int(selection["episode_id"]),
            start=int(selection["start"]),
            noise_draw=int(selection["selected_noise_draw"]),
        )
        for selection in selections
    ]
    pairs = action_counterfactual_rollouts(
        model=model,
        config=config,
        action_stats=action_stats,
        base_streams=base_streams,
        pair_batch_size=int(summary["pair_batch_size"]),
        rollout_steps=int(summary["rollout_steps"]),
        euler_steps=int(summary["euler_steps"]),
        seed=int(summary["seed"]),
        shuffle_seed=int(summary["shuffle_seed"]),
        device=device,
        precision=precision,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    source_steps = defaultdict(list)
    for row in rollout_step_rows:
        source_steps[
            (int(row["episode_id"]), int(row["start"]), int(row["noise_draw"]))
        ].append(row)
    rollout_by_identity = {
        (int(row["episode_id"]), int(row["start"]), int(row["noise_draw"])): row
        for row in rollout_rows
    }
    for selection, pair in zip(selections, pairs):
        identity = (
            int(selection["episode_id"]),
            int(selection["start"]),
            int(selection["selected_noise_draw"]),
        )
        if identity not in source_steps:
            raise RuntimeError(f"selected source rollout steps are missing: {identity}")
        validate_shuffle_permutation(pair["audit"], summary["shuffle_audits"])
        (
            true_messages,
            shuffle_messages,
            true_warning_steps,
            shuffle_warning_steps,
        ) = validate_action_rerun(pair, source_steps[identity])
        pair["true_rerun_messages"] = true_messages
        pair["shuffle_rerun_messages"] = shuffle_messages
        pair["true_rerun_warning_steps"] = true_warning_steps
        pair["shuffle_rerun_warning_steps"] = shuffle_warning_steps

    from src.so101_cache.so101_reader import SO101LeRobotV3Reader

    reader = SO101LeRobotV3Reader(dataset_root, camera=camera)
    requested_by_episode = defaultdict(set)
    for pair in pairs:
        episode_id = int(pair["true"]["episode_id"])
        requested_by_episode[episode_id].update(
            int(value) for value in pair["true"]["history_frame_indices"].tolist()
        )
        requested_by_episode[episode_id].update(
            int(value) for value in pair["true"]["target_frame_indices"].tolist()
        )
    original_rgb_by_episode = {}
    for episode_id, requested in requested_by_episode.items():
        ordered = sorted(requested)
        images = load_original_rgb_frames(
            reader,
            episode_id=episode_id,
            requested_frame_indices=ordered,
        )
        original_rgb_by_episode[episode_id] = {
            frame: images[index] for index, frame in enumerate(ordered)
        }

    vae, vae_directory = load_frozen_vae(args.vae_path, device)
    for episode_id, details in source_details.items():
        cache_scaling = float(details[2]["scaling_factor"])
        if not math.isclose(
            cache_scaling, float(vae.config.scaling_factor), rel_tol=0, abs_tol=1e-12
        ):
            raise RuntimeError(f"episode {episode_id} cache/VAE scaling mismatch")

    for selection, pair in zip(selections, pairs):
        true = pair["true"]
        shuffled = pair["shuffle"]
        episode_id = int(true["episode_id"])
        all_latents = torch.cat(
            (true["gt_future"], true["predicted_future"], shuffled["predicted_future"])
        )
        decoded = decode_cached_latents(
            vae,
            all_latents,
            device=device,
            batch_size=args.decode_batch_size,
        )
        rollout_steps = int(summary["rollout_steps"])
        gt_images = decoded[:rollout_steps]
        true_images = decoded[rollout_steps : 2 * rollout_steps]
        shuffled_images = decoded[2 * rollout_steps :]
        rgb_lookup = original_rgb_by_episode[episode_id]
        history_images = torch.stack(
            [rgb_lookup[int(value)] for value in true["history_frame_indices"].tolist()]
        )
        future_images = torch.stack(
            [rgb_lookup[int(value)] for value in true["target_frame_indices"].tolist()]
        )
        if true["target_frame_indices"].tolist() != shuffled[
            "target_frame_indices"
        ].tolist():
            raise RuntimeError("TRUE/shuffle frame indices differ before visualization")

        stream_dir = args.output_dir / selection["label"]
        stream_dir.mkdir(parents=True, exist_ok=True)
        sheet = create_action_contact_sheet(
            history_original=history_images,
            future_original=future_images,
            gt_reconstruction=gt_images,
            true_prediction=true_images,
            shuffled_prediction=shuffled_images,
            display_steps=display_steps,
            frame_stride=int(summary["frame_stride"]),
        )
        sheet.save(stream_dir / "contact_sheet.png")
        if args.save_individual_frames:
            save_png_sequence(
                history_images, stream_dir / "original_rgb", "history"
            )
            save_png_sequence(
                future_images,
                stream_dir / "original_rgb",
                "step",
                start_index=1,
            )
            save_png_sequence(
                gt_images,
                stream_dir / "gt_reconstruction",
                "step",
                start_index=1,
            )
            save_png_sequence(
                true_images,
                stream_dir / "true_prediction",
                "step",
                start_index=1,
            )
            save_png_sequence(
                shuffled_images,
                stream_dir / "shuffled_prediction",
                "step",
                start_index=1,
            )
        identity = (
            episode_id,
            int(true["start"]),
            int(true["noise_draw"]),
        )
        metadata = build_action_metadata(
            selection=selection,
            summary=summary,
            rollout_row=rollout_by_identity[identity],
            pair=pair,
            display_steps=display_steps,
            source_action_eval_dir=args.action_eval_dir,
            camera=camera,
            vae_path=vae_directory,
            vae=vae,
            true_rerun_messages=pair["true_rerun_messages"],
            shuffle_rerun_messages=pair["shuffle_rerun_messages"],
            true_rerun_warning_steps=pair["true_rerun_warning_steps"],
            shuffle_rerun_warning_steps=pair[
                "shuffle_rerun_warning_steps"
            ],
        )
        _write_json(stream_dir / "metadata.json", metadata)
        print(
            f"visualized {selection['label']}: episode={identity[0]} "
            f"start={identity[1]} draw={identity[2]}",
            flush=True,
        )

    _write_json(args.output_dir / "selection.json", selection_document)
    _write_selection_table(args.output_dir / "selection_table.csv", selections)
    print("selection:", args.output_dir / "selection.json")
    print("output:", args.output_dir)


if __name__ == "__main__":
    main()
