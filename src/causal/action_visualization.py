from __future__ import annotations

import csv
import json
import math
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from src.so101_cache.vae_cache import (
    CAMERA_KEY,
    FPS,
    IMAGE_SIZE,
    LATENT_CONVENTION,
    preprocess_rgb_batch,
)

from .rollout_visualization import decoded_tensor_to_uint8, quantile_label


SELECTION_METRIC = "mean_delta_mse"
SELECTION_UNIT = "physical_case"
DRAW_SELECTION_RULE = (
    "closest stochastic rollout delta to physical-case mean delta"
)
REFERENCE_LABELS = (
    "Original RGB",
    "GT latent reconstruction",
    "True Action Prediction",
    "Shuffled Action Prediction",
)
CONTACT_LEFT_MARGIN = 180
CONTACT_HEADER_HEIGHT = 32


def read_action_csv(path: str | Path, *, kind: str) -> list[dict[str, Any]]:
    integer_fields = {
        "episode_id",
        "start",
        "noise_draw",
        "step",
        "target_frame_index",
        "shuffle_source_transition",
        "num_noise_draws",
    }
    boolean_fields = {"true_better", "history_action_unchanged"}
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row = {}
            for key, value in raw.items():
                if key in integer_fields:
                    row[key] = int(value)
                elif key in boolean_fields:
                    row[key] = value.lower() == "true"
                elif key == "permutation":
                    row[key] = json.loads(value)
                else:
                    row[key] = float(value)
            rows.append(row)
    if not rows:
        raise ValueError(f"{kind} CSV contains no rows")
    return rows


def _quantile(values: Sequence[float], quantile: float) -> float:
    return float(
        torch.quantile(torch.tensor(list(values), dtype=torch.float64), quantile)
    )


def select_case_quantiles(
    case_rows: Sequence[dict[str, Any]], quantiles: Sequence[float]
) -> list[dict[str, Any]]:
    if not case_rows:
        raise ValueError("case selection requires physical cases")
    if not quantiles or any(not 0 <= float(value) <= 1 for value in quantiles):
        raise ValueError("quantiles must be non-empty and within [0,1]")
    labels = [quantile_label(float(value)) for value in quantiles]
    if len(set(labels)) != len(labels):
        raise ValueError("quantiles produce duplicate output labels")
    identities = [
        (int(row["episode_id"]), int(row["start"])) for row in case_rows
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("per_case rows contain duplicate physical cases")
    scores = [float(row[SELECTION_METRIC]) for row in case_rows]
    ordered = sorted(zip(identities, scores))
    used = set()
    selected = []
    for quantile, label in zip(quantiles, labels):
        target = _quantile(scores, float(quantile))
        candidates = sorted(
            ordered,
            key=lambda item: (round(abs(item[1] - target), 12), item[0]),
        )
        unused = [item for item in candidates if item[0] not in used]
        identity, score = (unused or candidates)[0]
        used.add(identity)
        selected.append(
            {
                "label": label,
                "quantile": float(quantile),
                "target_score": target,
                "episode_id": identity[0],
                "start": identity[1],
                "actual_case_score": score,
            }
        )
    return selected


def attach_representative_draws(
    selections: Sequence[dict[str, Any]],
    rollout_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for selection in selections:
        candidates = [
            row
            for row in rollout_rows
            if int(row["episode_id"]) == int(selection["episode_id"])
            and int(row["start"]) == int(selection["start"])
        ]
        if not candidates:
            raise ValueError("selected physical case has no stochastic rollouts")
        case_score = float(selection["actual_case_score"])
        chosen = min(
            candidates,
            key=lambda row: (
                round(abs(float(row[SELECTION_METRIC]) - case_score), 12),
                int(row["noise_draw"]),
            ),
        )
        item = dict(selection)
        item.update(
            {
                "selected_noise_draw": int(chosen["noise_draw"]),
                "selected_draw_score": float(chosen[SELECTION_METRIC]),
                "draw_selection_rule": DRAW_SELECTION_RULE,
            }
        )
        output.append(item)
    return output


def select_manual_streams(
    identities: Sequence[tuple[int, int, int]],
    case_rows: Sequence[dict[str, Any]],
    rollout_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(set(identities)) != len(identities):
        raise ValueError("manual streams contain duplicates")
    cases = {
        (int(row["episode_id"]), int(row["start"])): row for row in case_rows
    }
    rollouts = {
        (int(row["episode_id"]), int(row["start"]), int(row["noise_draw"])): row
        for row in rollout_rows
    }
    output = []
    for episode_id, start, noise_draw in identities:
        case_identity = (episode_id, start)
        identity = (episode_id, start, noise_draw)
        if case_identity not in cases or identity not in rollouts:
            raise ValueError(f"manual stream is absent from source results: {identity}")
        output.append(
            {
                "label": f"stream_ep{episode_id:04d}_s{start:04d}_d{noise_draw}",
                "quantile": None,
                "target_score": None,
                "episode_id": episode_id,
                "start": start,
                "actual_case_score": float(cases[case_identity][SELECTION_METRIC]),
                "selected_noise_draw": noise_draw,
                "selected_draw_score": float(rollouts[identity][SELECTION_METRIC]),
                "draw_selection_rule": "explicit manual stream",
            }
        )
    return output


def build_selection_document(
    selections: Sequence[dict[str, Any]],
    *,
    action_eval_dir: str | Path,
    num_candidate_cases: int,
    quantiles: Sequence[float],
    manual: bool,
) -> dict[str, Any]:
    return {
        "source_action_eval_dir": str(Path(action_eval_dir)),
        "selection_metric": SELECTION_METRIC,
        "selection_unit": SELECTION_UNIT,
        "num_candidate_cases": int(num_candidate_cases),
        "quantiles": [] if manual else [float(value) for value in quantiles],
        "selected_streams": [dict(value) for value in selections],
    }


def normalize_display_steps(
    requested: Sequence[int], rollout_steps: int
) -> list[int]:
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    values = sorted({int(step) for step in requested})
    if not values or any(step <= 0 or step > rollout_steps for step in values):
        raise ValueError(f"display steps must be within [1,{rollout_steps}]")
    return values


def display_step_times(
    steps: Sequence[int], *, frame_stride: int, fps: float = FPS
) -> list[float]:
    return [float(step * frame_stride) / float(fps) for step in steps]


def preprocess_original_rgb(images: Sequence[np.ndarray]) -> torch.Tensor:
    preprocessed = preprocess_rgb_batch(images)
    return decoded_tensor_to_uint8(preprocessed)


def resolve_original_rgb_source(
    manifest: dict[str, Any], cache_path: str | Path
) -> tuple[Path, str, dict[str, Any]]:
    dataset_root = manifest.get("dataset_root")
    if not isinstance(dataset_root, str) or not dataset_root:
        raise RuntimeError("manifest does not identify dataset_root for Original RGB")
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("latent cache lacks metadata required for Original RGB")
    camera = metadata.get("camera")
    if camera != CAMERA_KEY:
        raise RuntimeError(
            f"latent cache camera must be {CAMERA_KEY!r}, got {camera!r}"
        )
    if metadata.get("latent_convention") != LATENT_CONVENTION:
        raise RuntimeError("latent cache convention mismatch")
    if metadata.get("video_table_alignment_checked") is not True:
        raise RuntimeError("latent cache lacks verified video/table alignment")
    if metadata.get("input_image_size") != [IMAGE_SIZE, IMAGE_SIZE]:
        raise RuntimeError("latent cache preprocessing metadata mismatch")
    recorded_root = metadata.get("source_dataset_root")
    if not isinstance(recorded_root, str) or not recorded_root:
        raise RuntimeError("latent cache does not identify source_dataset_root")
    if Path(recorded_root).resolve() != Path(dataset_root).resolve():
        raise RuntimeError("manifest/cache source dataset roots disagree")
    return Path(dataset_root), camera, metadata


def load_original_rgb_frames(
    reader,
    *,
    episode_id: int,
    requested_frame_indices: Sequence[int],
) -> torch.Tensor:
    requested = [int(value) for value in requested_frame_indices]
    if len(set(requested)) != len(requested):
        raise ValueError("requested Original RGB frame indices must be unique")
    episode = reader.load_episode(episode_id)
    row_by_frame = {
        int(frame): row for row, frame in enumerate(episode.frame_indices.tolist())
    }
    missing = sorted(set(requested) - set(row_by_frame))
    if missing:
        raise RuntimeError(f"Original RGB source lacks frame indices: {missing}")
    requested_rows = {row_by_frame[frame]: frame for frame in requested}
    images_by_frame = {}
    row = 0
    segment = reader.video_segment(episode_id)
    for images, _timestamps in reader.iter_frame_batches(
        segment, expected_frames=episode.num_frames, batch_size=32
    ):
        for image in images:
            if row in requested_rows:
                images_by_frame[requested_rows[row]] = image
            row += 1
    if row != episode.num_frames or set(images_by_frame) != set(requested):
        raise RuntimeError("Original RGB decoding did not preserve episode frame alignment")
    return preprocess_original_rgb([images_by_frame[frame] for frame in requested])


def validate_shuffle_permutation(
    audit: dict[str, Any], source_audits: Sequence[dict[str, Any]]
) -> None:
    identity = (int(audit["episode_id"]), int(audit["start"]))
    matches = [
        item
        for item in source_audits
        if (int(item["episode_id"]), int(item["start"])) == identity
    ]
    if len(matches) != 1:
        raise RuntimeError(f"source shuffle audit is not unique for case {identity}")
    if [int(value) for value in audit["permutation"]] != [
        int(value) for value in matches[0]["permutation"]
    ]:
        raise RuntimeError(f"shuffle permutation mismatch for case {identity}")


def validate_source_result_tables(
    summary: dict[str, Any],
    case_rows: Sequence[dict[str, Any]],
    rollout_rows: Sequence[dict[str, Any]],
    rollout_step_rows: Sequence[dict[str, Any]],
) -> None:
    case_identities = [
        (int(row["episode_id"]), int(row["start"])) for row in case_rows
    ]
    if len(case_identities) != len(set(case_identities)):
        raise RuntimeError("per_case contains duplicate physical-case identities")
    if len(case_identities) != int(summary["num_cases"]):
        raise RuntimeError("per_case row count does not match summary num_cases")

    rollout_identities = [
        (int(row["episode_id"]), int(row["start"]), int(row["noise_draw"]))
        for row in rollout_rows
    ]
    if len(rollout_identities) != len(set(rollout_identities)):
        raise RuntimeError("per_rollout contains duplicate stochastic identities")
    if len(rollout_identities) != int(summary["num_stochastic_rollouts"]):
        raise RuntimeError(
            "per_rollout row count does not match summary num_stochastic_rollouts"
        )
    expected_rollouts = {
        (episode_id, start, noise_draw)
        for episode_id, start in case_identities
        for noise_draw in range(int(summary["noise_draws"]))
    }
    actual_rollouts = set(rollout_identities)
    if actual_rollouts != expected_rollouts:
        raise RuntimeError(
            "per_rollout stochastic identities mismatch: "
            f"missing={sorted(expected_rollouts - actual_rollouts)}, "
            f"extra={sorted(actual_rollouts - expected_rollouts)}"
        )

    rollout_steps = int(summary["rollout_steps"])
    expected_total = int(summary["num_stochastic_rollouts"]) * rollout_steps
    if len(rollout_step_rows) != expected_total:
        raise RuntimeError(
            "per_rollout_step row count mismatch: "
            f"expected={expected_total}, actual={len(rollout_step_rows)}"
        )
    steps_by_stream: dict[tuple[int, int, int], list[int]] = {}
    for row in rollout_step_rows:
        identity = (
            int(row["episode_id"]),
            int(row["start"]),
            int(row["noise_draw"]),
        )
        steps_by_stream.setdefault(identity, []).append(int(row["step"]))
    if set(steps_by_stream) != expected_rollouts:
        raise RuntimeError("per_rollout_step stochastic identities mismatch")
    expected_steps = list(range(1, rollout_steps + 1))
    for identity, steps in sorted(steps_by_stream.items()):
        if sorted(steps) != expected_steps:
            raise RuntimeError(
                f"per_rollout_step steps mismatch for {identity}: {sorted(steps)}"
            )


def validate_action_rerun(
    pair: dict[str, Any], source_rows: Sequence[dict[str, Any]]
) -> tuple[list[str], list[str], list[int], list[int]]:
    true = pair["true"]
    shuffled = pair["shuffle"]
    identity = (int(true["episode_id"]), int(true["start"]), int(true["noise_draw"]))
    shuffled_identity = (
        int(shuffled["episode_id"]),
        int(shuffled["start"]),
        int(shuffled["noise_draw"]),
    )
    if shuffled_identity != identity:
        raise RuntimeError("TRUE/shuffle rerun identity mismatch")
    ordered = sorted(source_rows, key=lambda row: int(row["step"]))
    if not ordered or any(
        (int(row["episode_id"]), int(row["start"]), int(row["noise_draw"]))
        != identity
        for row in ordered
    ):
        raise RuntimeError("action visualization rerun identity mismatch")
    source_targets = torch.tensor(
        [int(row["target_frame_index"]) for row in ordered], dtype=torch.long
    )
    if not torch.equal(true["target_frame_indices"], source_targets):
        raise RuntimeError("TRUE rerun target frame mismatch")
    if not torch.equal(shuffled["target_frame_indices"], source_targets):
        raise RuntimeError("SHUFFLED rerun target frame mismatch")
    if not torch.equal(true["initial_noises"], shuffled["initial_noises"]):
        raise RuntimeError("TRUE/shuffle Gaussian noise mismatch")
    def compare_branch(branch, prefix: str) -> tuple[list[str], list[int]]:
        messages = []
        warning_steps = set()
        for index, row in enumerate(ordered):
            for metric in ("mse", "relative_l2", "cosine_similarity"):
                rerun_value = float(branch["metrics"][metric][index])
                source_value = float(row[f"{prefix}_{metric}"])
                if not math.isclose(
                    rerun_value, source_value, rel_tol=5e-3, abs_tol=1e-5
                ):
                    messages.append(
                        f"step {index + 1} {metric}: "
                        f"rerun={rerun_value} source={source_value}"
                    )
                    warning_steps.add(index + 1)
        if messages:
            warnings.warn(
                f"{prefix.upper()} action visualization rerun metric differences: "
                + "; ".join(messages)
            )
        return messages, sorted(warning_steps)

    true_messages, true_warning_steps = compare_branch(true, "true")
    shuffle_messages, shuffle_warning_steps = compare_branch(shuffled, "shuffle")
    return (
        true_messages,
        shuffle_messages,
        true_warning_steps,
        shuffle_warning_steps,
    )


def create_action_contact_sheet(
    *,
    history_original: torch.Tensor,
    future_original: torch.Tensor,
    gt_reconstruction: torch.Tensor,
    true_prediction: torch.Tensor,
    shuffled_prediction: torch.Tensor,
    display_steps: Sequence[int],
    frame_stride: int,
    fps: float = FPS,
):
    from PIL import Image, ImageDraw

    history = torch.as_tensor(history_original, dtype=torch.uint8)
    future_rows = [
        torch.as_tensor(value, dtype=torch.uint8)
        for value in (
            future_original,
            gt_reconstruction,
            true_prediction,
            shuffled_prediction,
        )
    ]
    if history.ndim != 4 or history.shape[-1] != 3 or len(history) == 0:
        raise ValueError("history Original RGB must be non-empty [H,Y,X,3]")
    if any(value.shape != future_rows[0].shape for value in future_rows[1:]):
        raise ValueError("all future image rows must have matching shapes")
    if future_rows[0].ndim != 4 or future_rows[0].shape[-1] != 3:
        raise ValueError("future image rows must be [R,Y,X,3]")
    steps = normalize_display_steps(display_steps, len(future_rows[0]))
    height, width = int(history.shape[1]), int(history.shape[2])
    if any(tuple(value.shape[1:3]) != (height, width) for value in future_rows):
        raise ValueError("history and future images must share spatial shape")
    columns = len(history) + len(steps)
    sheet = Image.new(
        "RGB",
        (
            CONTACT_LEFT_MARGIN + columns * width,
            CONTACT_HEADER_HEIGHT + len(REFERENCE_LABELS) * height,
        ),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(sheet)
    for row_index, label in enumerate(REFERENCE_LABELS):
        draw.text(
            (4, CONTACT_HEADER_HEIGHT + row_index * height + height // 2),
            label,
            fill=(0, 0, 0),
        )
    for history_index, image in enumerate(history.numpy()):
        x = CONTACT_LEFT_MARGIN + history_index * width
        item = Image.fromarray(image, mode="RGB")
        for row_index in range(len(REFERENCE_LABELS)):
            sheet.paste(item, (x, CONTACT_HEADER_HEIGHT + row_index * height))
        draw.text(
            (x + 2, 4),
            f"Shared History (Original RGB) {history_index}",
            fill=(0, 0, 0),
        )
    for offset, step in enumerate(steps, start=len(history)):
        x = CONTACT_LEFT_MARGIN + offset * width
        for row_index, values in enumerate(future_rows):
            sheet.paste(
                Image.fromarray(values[step - 1].numpy(), mode="RGB"),
                (x, CONTACT_HEADER_HEIGHT + row_index * height),
            )
        seconds = step * frame_stride / float(fps)
        draw.text((x + 2, 4), f"step {step}  +{seconds:.1f}s", fill=(0, 0, 0))
    return sheet


def build_action_metadata(
    *,
    selection: dict[str, Any],
    summary: dict[str, Any],
    rollout_row: dict[str, Any],
    pair: dict[str, Any],
    display_steps: Sequence[int],
    source_action_eval_dir: str | Path,
    camera: str,
    vae_path: str | Path,
    vae,
    true_rerun_messages: Sequence[str],
    shuffle_rerun_messages: Sequence[str],
    true_rerun_warning_steps: Sequence[int],
    shuffle_rerun_warning_steps: Sequence[int],
) -> dict[str, Any]:
    audit = pair["audit"]
    rerun_true_mse = pair["true"]["metrics"]["mse"].float()
    rerun_shuffle_mse = pair["shuffle"]["metrics"]["mse"].float()
    rerun_delta_mse = rerun_shuffle_mse - rerun_true_mse
    return {
        "source_action_eval_dir": str(Path(source_action_eval_dir)),
        "checkpoint_path": summary["checkpoint_path"],
        "checkpoint_step": int(summary["checkpoint_step"]),
        "weights_used": summary["weights_used"],
        "split": summary["split"],
        "precision": summary["precision"],
        "episode_id": int(selection["episode_id"]),
        "start": int(selection["start"]),
        "noise_draw": int(selection["selected_noise_draw"]),
        "selection_label": selection["label"],
        "selection_metric": SELECTION_METRIC,
        "target_quantile": selection["quantile"],
        "target_score": selection["target_score"],
        "case_score": float(selection["actual_case_score"]),
        "case_mean_delta_mse": float(selection["actual_case_score"]),
        "selected_draw_score": float(selection["selected_draw_score"]),
        "draw_selection_rule": selection["draw_selection_rule"],
        "rollout_steps": int(summary["rollout_steps"]),
        "display_steps": [int(value) for value in display_steps],
        "frame_stride": int(summary["frame_stride"]),
        "fps": FPS,
        "true_mean_mse": float(rollout_row["mean_true_mse"]),
        "shuffle_mean_mse": float(rollout_row["mean_shuffle_mse"]),
        "mean_delta_mse": float(rollout_row["mean_delta_mse"]),
        "selected_draw_mean_true_mse": float(rollout_row["mean_true_mse"]),
        "selected_draw_mean_shuffle_mse": float(rollout_row["mean_shuffle_mse"]),
        "selected_draw_mean_delta_mse": float(rollout_row["mean_delta_mse"]),
        "true_final_mse": float(rollout_row["final_true_mse"]),
        "shuffle_final_mse": float(rollout_row["final_shuffle_mse"]),
        "final_delta_mse": float(rollout_row["final_delta_mse"]),
        "source_true_mean_mse": float(rollout_row["mean_true_mse"]),
        "source_shuffle_mean_mse": float(rollout_row["mean_shuffle_mse"]),
        "source_mean_delta_mse": float(rollout_row["mean_delta_mse"]),
        "source_true_final_mse": float(rollout_row["final_true_mse"]),
        "source_shuffle_final_mse": float(rollout_row["final_shuffle_mse"]),
        "source_final_delta_mse": float(rollout_row["final_delta_mse"]),
        "rerun_true_mean_mse": float(rerun_true_mse.mean()),
        "rerun_shuffle_mean_mse": float(rerun_shuffle_mse.mean()),
        "rerun_mean_delta_mse": float(rerun_delta_mse.mean()),
        "rerun_true_final_mse": float(rerun_true_mse[-1]),
        "rerun_shuffle_final_mse": float(rerun_shuffle_mse[-1]),
        "rerun_final_delta_mse": float(rerun_delta_mse[-1]),
        "shuffle_seed": int(summary["shuffle_seed"]),
        "shuffle_type": summary["shuffle_type"],
        "permutation": audit["permutation"],
        "same_gaussian_noise": torch.equal(
            pair["true"]["initial_noises"], pair["shuffle"]["initial_noises"]
        ),
        "history_action_unchanged": audit["history_action_unchanged"],
        "rerun_metric_warning_count": len(true_rerun_messages)
        + len(shuffle_rerun_messages),
        "true_rerun_metric_warning_count": len(true_rerun_messages),
        "shuffle_rerun_metric_warning_count": len(shuffle_rerun_messages),
        "true_rerun_steps_with_warning": [
            int(value) for value in true_rerun_warning_steps
        ],
        "shuffle_rerun_steps_with_warning": [
            int(value) for value in shuffle_rerun_warning_steps
        ],
        "original_rgb_source_camera": camera,
        "original_rgb_preprocessing": (
            "shortest-side resize to 256 then center crop 256x256"
        ),
        "rgb_frame_indices": pair["true"]["target_frame_indices"].tolist(),
        "latent_frame_indices": pair["true"]["target_frame_indices"].tolist(),
        "vae_path": str(Path(vae_path)),
        "vae_scaling_factor": float(vae.config.scaling_factor),
        "vae_config_shift_factor": getattr(vae.config, "shift_factor", None),
        "vae_shift_applied": False,
        "latent_convention": LATENT_CONVENTION,
        "reference_labels": list(REFERENCE_LABELS),
        "prediction_images_are_decoded_model_latents": True,
    }
