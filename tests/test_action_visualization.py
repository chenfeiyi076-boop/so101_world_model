from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.causal.action_visualization import (
    CONTACT_HEADER_HEIGHT,
    CONTACT_LEFT_MARGIN,
    DRAW_SELECTION_RULE,
    REFERENCE_LABELS,
    attach_representative_draws,
    build_action_metadata,
    build_selection_document,
    create_action_contact_sheet,
    display_step_times,
    load_original_rgb_frames,
    normalize_display_steps,
    preprocess_original_rgb,
    select_case_quantiles,
    select_manual_streams,
    validate_action_rerun,
    validate_shuffle_permutation,
    validate_source_result_tables,
)
from src.causal.rollout_visualization import decode_cached_latents
from src.so101_cache.vae_cache import CAMERA_KEY, LATENT_CONVENTION


class FakeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            scaling_factor=2.0,
            shift_factor=0.0609,
            latent_channels=16,
        )
        self.decode_inputs = []

    def decode(self, value):
        self.decode_inputs.append(value.detach().cpu().clone())
        return SimpleNamespace(
            sample=torch.zeros(len(value), 3, 2, 2, dtype=value.dtype)
            + self.anchor * 0
        )


def _case_rows(count: int = 128):
    return [
        {
            "episode_id": index,
            "start": index * 2,
            "mean_delta_mse": index / 100.0,
        }
        for index in range(count)
    ]


def _rollout_rows(case_rows=None):
    rows = []
    for case in case_rows or _case_rows():
        for draw in range(4):
            delta = float(case["mean_delta_mse"]) + (draw - 1.5) * 0.01
            rows.append(
                {
                    "episode_id": case["episode_id"],
                    "start": case["start"],
                    "noise_draw": draw,
                    "mean_delta_mse": delta,
                    "mean_true_mse": 0.1,
                    "mean_shuffle_mse": 0.1 + delta,
                    "final_true_mse": 0.2,
                    "final_shuffle_mse": 0.2 + delta,
                    "final_delta_mse": delta,
                }
            )
    return rows


def _fake_pair(rollout_steps: int = 2):
    targets = torch.arange(8, 8 + rollout_steps * 4, 4)
    noises = torch.arange(rollout_steps * 4, dtype=torch.float32).reshape(
        rollout_steps, 1, 2, 2
    )
    metrics = {
        "mse": torch.tensor([1.0 + index for index in range(rollout_steps)]),
        "relative_l2": torch.tensor(
            [0.1 + index * 0.1 for index in range(rollout_steps)]
        ),
        "cosine_similarity": torch.tensor(
            [0.9 - index * 0.1 for index in range(rollout_steps)]
        ),
    }
    true = {
        "episode_id": 6,
        "start": 0,
        "noise_draw": 1,
        "history_frame_indices": torch.tensor([0, 4]),
        "target_frame_indices": targets,
        "initial_noises": noises,
        "metrics": metrics,
    }
    shuffled = {
        "episode_id": 6,
        "start": 0,
        "noise_draw": 1,
        "target_frame_indices": targets.clone(),
        "initial_noises": noises.clone(),
        "metrics": {
            "mse": metrics["mse"].clone() + 1.0,
            "relative_l2": metrics["relative_l2"].clone() + 0.1,
            "cosine_similarity": metrics["cosine_similarity"].clone() - 0.1,
        },
    }
    return {
        "true": true,
        "shuffle": shuffled,
        "audit": {
            "episode_id": 6,
            "start": 0,
            "permutation": list(reversed(range(rollout_steps))),
            "history_action_unchanged": True,
        },
    }


def _source_step_rows(pair):
    rows = []
    true = pair["true"]
    for index, target in enumerate(true["target_frame_indices"].tolist()):
        rows.append(
            {
                "episode_id": 6,
                "start": 0,
                "noise_draw": 1,
                "step": index + 1,
                "target_frame_index": target,
                "true_mse": float(true["metrics"]["mse"][index]),
                "true_relative_l2": float(true["metrics"]["relative_l2"][index]),
                "true_cosine_similarity": float(
                    true["metrics"]["cosine_similarity"][index]
                ),
                "shuffle_mse": float(pair["shuffle"]["metrics"]["mse"][index]),
                "shuffle_relative_l2": float(
                    pair["shuffle"]["metrics"]["relative_l2"][index]
                ),
                "shuffle_cosine_similarity": float(
                    pair["shuffle"]["metrics"]["cosine_similarity"][index]
                ),
            }
        )
    return rows


def test_q10_q50_q90_selection_is_deterministic_over_128_cases():
    rows = _case_rows()
    first = select_case_quantiles(rows, (0.1, 0.5, 0.9))
    second = select_case_quantiles(list(reversed(rows)), (0.1, 0.5, 0.9))
    assert first == second
    assert [item["label"] for item in first] == ["q10", "q50", "q90"]
    assert [item["episode_id"] for item in first] == [13, 63, 114]


def test_selection_unit_is_physical_case_not_stochastic_draw():
    selected = select_case_quantiles(_case_rows(), (0.5,))
    assert len(selected) == 1
    assert "noise_draw" not in selected[0]
    document = build_selection_document(
        selected,
        action_eval_dir="action_eval",
        num_candidate_cases=128,
        quantiles=(0.5,),
        manual=False,
    )
    assert document["selection_unit"] == "physical_case"
    assert document["num_candidate_cases"] == 128


def test_representative_draw_is_closest_to_case_mean():
    selection = [
        {
            "label": "q50",
            "quantile": 0.5,
            "target_score": 0.4,
            "episode_id": 6,
            "start": 10,
            "actual_case_score": 0.4,
        }
    ]
    draws = [
        {"episode_id": 6, "start": 10, "noise_draw": 0, "mean_delta_mse": 0.1},
        {"episode_id": 6, "start": 10, "noise_draw": 1, "mean_delta_mse": 0.38},
        {"episode_id": 6, "start": 10, "noise_draw": 2, "mean_delta_mse": 0.8},
        {"episode_id": 6, "start": 10, "noise_draw": 3, "mean_delta_mse": 0.5},
    ]
    result = attach_representative_draws(selection, draws)[0]
    assert result["selected_noise_draw"] == 1
    assert result["selected_draw_score"] == pytest.approx(0.38)
    assert result["draw_selection_rule"] == DRAW_SELECTION_RULE


def test_representative_draw_tie_uses_smallest_noise_draw():
    selection = [
        {
            "label": "q50",
            "quantile": 0.5,
            "target_score": 0.5,
            "episode_id": 6,
            "start": 10,
            "actual_case_score": 0.5,
        }
    ]
    draws = [
        {"episode_id": 6, "start": 10, "noise_draw": 2, "mean_delta_mse": 0.6},
        {"episode_id": 6, "start": 10, "noise_draw": 0, "mean_delta_mse": 0.4},
    ]
    assert attach_representative_draws(selection, draws)[0]["selected_noise_draw"] == 0


def test_manual_stream_selection():
    cases = [{"episode_id": 6, "start": 10, "mean_delta_mse": 0.5}]
    draws = [
        {"episode_id": 6, "start": 10, "noise_draw": 3, "mean_delta_mse": 0.7}
    ]
    selected = select_manual_streams([(6, 10, 3)], cases, draws)[0]
    assert selected["label"] == "stream_ep0006_s0010_d3"
    assert selected["selected_noise_draw"] == 3
    assert selected["draw_selection_rule"] == "explicit manual stream"


def test_r32_display_steps_and_future_offsets():
    steps = normalize_display_steps((1, 4, 8, 16, 24, 32), 32)
    assert steps == [1, 4, 8, 16, 24, 32]
    assert display_step_times(steps, frame_stride=4) == pytest.approx(
        [0.2, 0.8, 1.6, 3.2, 4.8, 6.4]
    )


def test_display_step_above_rollout_horizon_fails():
    with pytest.raises(ValueError, match="within"):
        normalize_display_steps((1, 4, 33), 32)


def test_original_rgb_preprocessing_resize_and_center_crop_shape():
    image = np.zeros((300, 500, 3), dtype=np.uint8)
    image[..., 0] = 255
    result = preprocess_original_rgb([image])
    assert result.shape == (1, 256, 256, 3)
    assert result.dtype == torch.uint8
    assert torch.all(result[..., 0] == 255)
    assert torch.all(result[..., 1:] == 0)


def test_original_rgb_and_gt_reconstruction_labels_are_distinct():
    assert REFERENCE_LABELS[0] == "Original RGB"
    assert REFERENCE_LABELS[1] == "GT latent reconstruction"
    assert REFERENCE_LABELS[0] != REFERENCE_LABELS[1]


def test_vae_decode_reuses_scaling_only_no_shift_convention():
    vae = FakeVAE()
    decode_cached_latents(
        vae,
        torch.tensor([4.0, 6.0]).reshape(2, 1, 1, 1),
        device=torch.device("cpu"),
        batch_size=2,
    )
    assert torch.equal(vae.decode_inputs[0].flatten(), torch.tensor([2.0, 3.0]))


def test_selection_document_schema():
    selection = attach_representative_draws(
        select_case_quantiles(_case_rows(), (0.1,)), _rollout_rows()
    )
    document = build_selection_document(
        selection,
        action_eval_dir=Path("action_eval"),
        num_candidate_cases=128,
        quantiles=(0.1,),
        manual=False,
    )
    assert set(
        (
            "selection_metric",
            "selection_unit",
            "num_candidate_cases",
            "quantiles",
            "selected_streams",
        )
    ).issubset(document)
    selected = document["selected_streams"][0]
    assert set(
        (
            "label",
            "target_score",
            "episode_id",
            "start",
            "actual_case_score",
            "selected_noise_draw",
            "selected_draw_score",
            "draw_selection_rule",
        )
    ).issubset(selected)


def test_metadata_schema_distinguishes_all_four_references(tmp_path):
    pair = _fake_pair()
    selection = {
        "label": "q50",
        "quantile": 0.5,
        "target_score": 0.4,
        "episode_id": 6,
        "start": 0,
        "actual_case_score": 0.41,
        "selected_noise_draw": 1,
        "selected_draw_score": 0.42,
        "draw_selection_rule": DRAW_SELECTION_RULE,
    }
    summary = {
        "checkpoint_path": "checkpoint.pt",
        "checkpoint_step": 300000,
        "weights_used": "ema",
        "split": "val",
        "precision": "bf16",
        "rollout_steps": 2,
        "frame_stride": 4,
        "shuffle_seed": 30360,
        "shuffle_type": "temporal_chunk_derangement",
    }
    rollout = {
        "mean_true_mse": 0.1,
        "mean_shuffle_mse": 0.2,
        "mean_delta_mse": 0.1,
        "final_true_mse": 0.15,
        "final_shuffle_mse": 0.3,
        "final_delta_mse": 0.15,
    }
    metadata = build_action_metadata(
        selection=selection,
        summary=summary,
        rollout_row=rollout,
        pair=pair,
        display_steps=(1, 2),
        source_action_eval_dir=tmp_path,
        camera=CAMERA_KEY,
        vae_path=tmp_path / "vae",
        vae=FakeVAE(),
        true_rerun_messages=("one warning",),
        shuffle_rerun_messages=("shuffle warning 1", "shuffle warning 2"),
        true_rerun_warning_steps=(2,),
        shuffle_rerun_warning_steps=(1, 2),
    )
    assert metadata["reference_labels"] == list(REFERENCE_LABELS)
    assert metadata["original_rgb_source_camera"] == CAMERA_KEY
    assert metadata["vae_shift_applied"] is False
    assert metadata["latent_convention"] == LATENT_CONVENTION
    assert metadata["same_gaussian_noise"] is True
    assert metadata["true_rerun_metric_warning_count"] == 1
    assert metadata["shuffle_rerun_metric_warning_count"] == 2
    assert metadata["source_true_mean_mse"] == pytest.approx(0.1)
    assert metadata["source_shuffle_mean_mse"] == pytest.approx(0.2)
    assert metadata["source_mean_delta_mse"] == pytest.approx(0.1)
    assert metadata["rerun_true_mean_mse"] == pytest.approx(1.5)
    assert metadata["rerun_shuffle_mean_mse"] == pytest.approx(2.5)
    assert metadata["rerun_mean_delta_mse"] == pytest.approx(1.0)


def test_contact_sheet_has_four_stable_rows(monkeypatch):
    from PIL import ImageDraw

    labels = []
    original_draw = ImageDraw.Draw

    class RecordingDraw:
        def __init__(self, image):
            self.delegate = original_draw(image)

        def text(self, position, value, *args, **kwargs):
            labels.append(value)
            return self.delegate.text(position, value, *args, **kwargs)

    monkeypatch.setattr(ImageDraw, "Draw", RecordingDraw)
    history = torch.full((2, 5, 4, 3), 100, dtype=torch.uint8)
    rows = []
    for color in ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)):
        value = torch.zeros(2, 5, 4, 3, dtype=torch.uint8)
        value[:] = torch.tensor(color, dtype=torch.uint8)
        rows.append(value)
    sheet = create_action_contact_sheet(
        history_original=history,
        future_original=rows[0],
        gt_reconstruction=rows[1],
        true_prediction=rows[2],
        shuffled_prediction=rows[3],
        display_steps=(1, 2),
        frame_stride=4,
    )
    assert sheet.size == (
        CONTACT_LEFT_MARGIN + 4 * 4,
        CONTACT_HEADER_HEIGHT + 4 * 5,
    )
    assert all(label in labels for label in REFERENCE_LABELS)
    assert "Shared History (Original RGB) 0" in labels
    assert "Shared History (Original RGB) 1" in labels
    x = CONTACT_LEFT_MARGIN + 2 * 4
    assert sheet.getpixel((x, CONTACT_HEADER_HEIGHT)) == (255, 0, 0)
    assert sheet.getpixel((x, CONTACT_HEADER_HEIGHT + 5)) == (0, 255, 0)
    assert sheet.getpixel((x, CONTACT_HEADER_HEIGHT + 10)) == (0, 0, 255)
    assert sheet.getpixel((x, CONTACT_HEADER_HEIGHT + 15)) == (255, 255, 0)


def test_shuffle_permutation_mismatch_hard_fails():
    pair = _fake_pair(3)
    source = [{"episode_id": 6, "start": 0, "permutation": [1, 2, 0]}]
    with pytest.raises(RuntimeError, match="permutation mismatch"):
        validate_shuffle_permutation(pair["audit"], source)


def test_true_shuffle_gaussian_noise_mismatch_hard_fails():
    pair = _fake_pair()
    pair["shuffle"]["initial_noises"] = pair["shuffle"]["initial_noises"] + 1
    with pytest.raises(RuntimeError, match="Gaussian noise mismatch"):
        validate_action_rerun(pair, _source_step_rows(pair))


@pytest.mark.parametrize("mismatch", ("identity", "target", "noise"))
def test_shuffled_identity_target_and_noise_mismatch_hard_fail(mismatch):
    pair = _fake_pair()
    if mismatch == "identity":
        pair["shuffle"]["noise_draw"] = 3
        expected = "identity mismatch"
    elif mismatch == "target":
        pair["shuffle"]["target_frame_indices"][0] += 1
        expected = "target frame mismatch"
    else:
        pair["shuffle"]["initial_noises"][0] += 1
        expected = "Gaussian noise mismatch"
    with pytest.raises(RuntimeError, match=expected):
        validate_action_rerun(pair, _source_step_rows(_fake_pair()))


def test_shuffled_rerun_numerical_difference_warns():
    pair = _fake_pair()
    source = _source_step_rows(pair)
    source[0]["shuffle_mse"] += 0.1
    with pytest.warns(UserWarning, match="SHUFFLE.*rerun metric differences"):
        true_messages, shuffle_messages, true_steps, shuffle_steps = (
            validate_action_rerun(pair, source)
        )
    assert true_messages == []
    assert true_steps == []
    assert len(shuffle_messages) == 1
    assert shuffle_steps == [1]


def test_rerun_metric_difference_warns_but_target_mismatch_hard_fails():
    pair = _fake_pair()
    source = _source_step_rows(pair)
    source[0]["true_mse"] = source[0]["true_mse"] + 0.1
    with pytest.warns(UserWarning, match="TRUE.*rerun metric differences"):
        true_messages, shuffle_messages, true_steps, shuffle_steps = (
            validate_action_rerun(pair, source)
        )
    assert len(true_messages) == 1
    assert true_steps == [1]
    assert shuffle_messages == []
    assert shuffle_steps == []

    source[0]["target_frame_index"] += 1
    with pytest.raises(RuntimeError, match="target frame mismatch"):
        validate_action_rerun(pair, source)


def _integrity_tables():
    summary = {
        "num_cases": 1,
        "num_stochastic_rollouts": 4,
        "noise_draws": 4,
        "rollout_steps": 2,
    }
    cases = [{"episode_id": 6, "start": 10}]
    rollouts = [
        {"episode_id": 6, "start": 10, "noise_draw": draw}
        for draw in range(4)
    ]
    steps = [
        {"episode_id": 6, "start": 10, "noise_draw": draw, "step": step}
        for draw in range(4)
        for step in (1, 2)
    ]
    return summary, cases, rollouts, steps


def test_source_per_rollout_missing_one_draw_fails():
    summary, cases, rollouts, steps = _integrity_tables()
    with pytest.raises(RuntimeError, match="per_rollout row count"):
        validate_source_result_tables(summary, cases, rollouts[:-1], steps)


def test_source_per_rollout_duplicate_stochastic_identity_fails():
    summary, cases, rollouts, steps = _integrity_tables()
    rollouts[-1] = dict(rollouts[0])
    with pytest.raises(RuntimeError, match="duplicate stochastic"):
        validate_source_result_tables(summary, cases, rollouts, steps)


@pytest.mark.parametrize("kind", ("missing", "extra"))
def test_source_per_rollout_step_missing_or_extra_step_fails(kind):
    summary, cases, rollouts, steps = _integrity_tables()
    if kind == "missing":
        invalid_steps = steps[:-1]
        expected = "row count mismatch"
    else:
        invalid_steps = [dict(row) for row in steps]
        invalid_steps[-1]["step"] = 3
        expected = "steps mismatch"
    with pytest.raises(RuntimeError, match=expected):
        validate_source_result_tables(summary, cases, rollouts, invalid_steps)


def test_fake_original_rgb_reader_maps_exact_frame_indices():
    class FakeReader:
        def load_episode(self, episode_id):
            assert episode_id == 6
            return SimpleNamespace(
                frame_indices=torch.tensor([10, 20, 30]), num_frames=3
            )

        def video_segment(self, episode_id):
            return SimpleNamespace(episode_id=episode_id)

        def iter_frame_batches(self, segment, *, expected_frames, batch_size):
            assert expected_frames == 3
            images = []
            for red in (10, 20, 30):
                image = np.zeros((256, 256, 3), dtype=np.uint8)
                image[..., 0] = red
                images.append(image)
            yield images, [0.0, 0.05, 0.1]

    images = load_original_rgb_frames(
        FakeReader(), episode_id=6, requested_frame_indices=(30, 10)
    )
    assert images.shape == (2, 256, 256, 3)
    assert torch.all(images[0, ..., 0] == 30)
    assert torch.all(images[1, ..., 0] == 10)
