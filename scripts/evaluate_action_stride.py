from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.data.dataset import CachedLatentDataset
from src.models.dit import DiT
from src.diffusion.flow_matching import (
    prepare_flow_matching_batch,
    flow_matching_loss,
)


CACHE_PATH = (
    "data/latent_cache/so101_front/"
    "episode_000.pt"
)

T = 10
NUM_HISTORY = 2


def mean_ci95(
    x: np.ndarray,
):
    mean = float(
        x.mean()
    )

    if len(x) <= 1:
        return (
            mean,
            float("nan"),
            float("nan"),
        )

    std = float(
        x.std(
            ddof=1
        )
    )

    se = (
        std
        / np.sqrt(len(x))
    )

    return (
        mean,
        mean - 1.96 * se,
        mean + 1.96 * se,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--frame-skip",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--noise-draws",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    device = torch.device("cuda")

    checkpoint_path = (
        Path("checkpoints")
        / (
            "episode000_dit_s_fm_"
            f"stride{args.frame_skip}_delta.pt"
        )
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    print("=" * 80)
    print("ACTION CONDITIONING EVALUATION")
    print("=" * 80)

    print(
        "checkpoint:",
        checkpoint_path,
    )

    print(
        "frame_skip:",
        args.frame_skip,
    )

    # ========================================================
    # Dataset
    # ========================================================

    dataset = CachedLatentDataset(
        cache_path=CACHE_PATH,
        n_frames=T,
        frame_skip=args.frame_skip,
        action_mode="sampled",
        action_representation="delta",

        # 必须用 checkpoint 的统计量
        normalize_actions=True,
    )

    action_mean = checkpoint[
        "action_mean"
    ].float()

    action_std = checkpoint[
        "action_std"
    ].float().clamp_min(
        1e-6
    )

    # ========================================================
    # Model
    # ========================================================

    model = DiT(
        in_channels=16,
        patch_size=2,
        hidden_size=384,
        depth=12,
        num_heads=6,
        action_dim=dataset.condition_action_dim,
        mlp_ratio=4.0,
        use_qk_norm=True,
    ).to(device)

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()

    # ========================================================
    # Non-overlapping-ish windows
    #
    # stride1 temporal span = 10 raw frames
    # stride2 temporal span = 19 raw frames
    # stride4 temporal span = 37 raw frames
    # ========================================================

    raw_span = (
        1
        + (T - 1)
        * args.frame_skip
    )

    starts = list(
        range(
            0,
            len(dataset),
            raw_span,
        )
    )

    print(
        "dataset windows:",
        len(dataset),
    )

    print(
        "raw frame span:",
        raw_span,
    )

    print(
        "evaluation windows:",
        len(starts),
    )

    print(
        "noise draws/window:",
        args.noise_draws,
    )

    # 错配到相隔约半条 trajectory 的 action window
    mismatch_offset = (
        len(dataset) // 2
    )

    true_losses = []
    shuffle_losses = []
    zero_losses = []

    mismatch_distances = []

    # ========================================================
    # Evaluation
    # ========================================================

    with torch.inference_mode():

        for eval_i, start in enumerate(
            starts
        ):

            sample = dataset[
                start
            ]

            mismatch_start = (
                start
                + mismatch_offset
            ) % len(dataset)

            mismatch_sample = dataset[
                mismatch_start
            ]

            latents = (
                sample["latents"]
                .unsqueeze(0)
                .to(
                    device=device,
                    dtype=torch.float32,
                )
            )

            # =================================================
            # Normalize with TRAIN statistics
            # =================================================

            true_raw = (
                sample[
                    "actions_raw"
                ].float()
            )

            mismatch_raw = (
                mismatch_sample[
                    "actions_raw"
                ].float()
            )

            true_actions = (
                (
                    true_raw
                    - action_mean
                )
                / action_std
            )

            mismatch_actions = (
                (
                    mismatch_raw
                    - action_mean
                )
                / action_std
            )

            true_actions = (
                sample["actions"]
                .unsqueeze(0)
                .to(device=device,
                    dtype=torch.float32,)
            )

            mismatch_actions = (
                mismatch_sample["actions"]
                .unsqueeze(0)
                .to(device=device,
                    dtype=torch.float32,)
            )

            # =================================================
            # 保留 history action
            # 只 intervention future action
            # =================================================

            shuffled = (
                true_actions.clone()
            )

            shuffled[
                :,
                NUM_HISTORY:,
            ] = mismatch_actions[
                :,
                NUM_HISTORY:,
            ]

            zeroed = (
                true_actions.clone()
            )

            zeroed[
                :,
                NUM_HISTORY:,
            ] = 0.0

            mismatch_rms = torch.sqrt(
                torch.mean(
                    (
                        true_actions[
                            :,
                            NUM_HISTORY:,
                        ]
                        -
                        shuffled[
                            :,
                            NUM_HISTORY:,
                        ]
                    )
                    ** 2
                )
            ).item()

            mismatch_distances.append(
                mismatch_rms
            )

            draw_true = []
            draw_shuffle = []
            draw_zero = []

            for draw in range(
                args.noise_draws
            ):

                # =============================================
                # Deterministic paired random draw
                # =============================================

                seed = (
                    100000
                    + eval_i * 100
                    + draw
                )

                torch.manual_seed(
                    seed
                )

                torch.cuda.manual_seed_all(
                    seed
                )

                # =============================================
                # 一次构造 FM batch
                #
                # true/shuffle/zero 共用：
                #
                # z_tau
                # tau
                # epsilon
                # target
                # =============================================

                fm = prepare_flow_matching_batch(
                    latents=latents,
                    num_history=NUM_HISTORY,
                    history_noise_std=0.0,
                )

                pred_true = model(
                    fm.noisy_latents,
                    fm.tau,
                    true_actions,
                )

                pred_shuffle = model(
                    fm.noisy_latents,
                    fm.tau,
                    shuffled,
                )

                pred_zero = model(
                    fm.noisy_latents,
                    fm.tau,
                    zeroed,
                )

                loss_true = (
                    flow_matching_loss(
                        prediction=pred_true,
                        target_velocity=fm.target_velocity,
                        loss_mask=fm.loss_mask,
                    )
                )

                loss_shuffle = (
                    flow_matching_loss(
                        prediction=pred_shuffle,
                        target_velocity=fm.target_velocity,
                        loss_mask=fm.loss_mask,
                    )
                )

                loss_zero = (
                    flow_matching_loss(
                        prediction=pred_zero,
                        target_velocity=fm.target_velocity,
                        loss_mask=fm.loss_mask,
                    )
                )

                draw_true.append(
                    loss_true.item()
                )

                draw_shuffle.append(
                    loss_shuffle.item()
                )

                draw_zero.append(
                    loss_zero.item()
                )

            true_losses.append(
                np.mean(draw_true)
            )

            shuffle_losses.append(
                np.mean(draw_shuffle)
            )

            zero_losses.append(
                np.mean(draw_zero)
            )

            if (
                eval_i == 0
                or (eval_i + 1) % 5 == 0
                or eval_i == len(starts) - 1
            ):

                print(
                    f"[{eval_i + 1:03d}/"
                    f"{len(starts):03d}] "
                    f"start={start:03d}  "
                    f"true={true_losses[-1]:.4f}  "
                    f"shuffle={shuffle_losses[-1]:.4f}  "
                    f"zero={zero_losses[-1]:.4f}"
                )

    # ========================================================
    # Statistics
    # ========================================================

    true_losses = np.asarray(
        true_losses
    )

    shuffle_losses = np.asarray(
        shuffle_losses
    )

    zero_losses = np.asarray(
        zero_losses
    )

    delta_shuffle = (
        shuffle_losses
        - true_losses
    )

    delta_zero = (
        zero_losses
        - true_losses
    )

    d_true = float(
        true_losses.mean()
    )

    d_shuffle = float(
        shuffle_losses.mean()
    )

    d_zero = float(
        zero_losses.mean()
    )

    gap_shuffle = (
        d_shuffle
        - d_true
    )

    gap_zero = (
        d_zero
        - d_true
    )

    sensitivity = (
        gap_shuffle
        / max(d_true, 1e-8)
        * 100.0
    )

    zero_relative = (
        gap_zero
        / max(d_true, 1e-8)
        * 100.0
    )

    win_shuffle = (
        np.mean(
            true_losses
            < shuffle_losses
        )
        * 100.0
    )

    win_zero = (
        np.mean(
            true_losses
            < zero_losses
        )
        * 100.0
    )

    ds_mean, ds_low, ds_high = (
        mean_ci95(
            delta_shuffle
        )
    )

    dz_mean, dz_low, dz_high = (
        mean_ci95(
            delta_zero
        )
    )

    # ========================================================
    # Result
    # ========================================================

    print()
    print("=" * 80)
    print("RESULT")
    print("=" * 80)

    print(
        f"frame_skip = "
        f"{args.frame_skip}"
    )

    print(
        f"D_true    = {d_true:.6f}"
    )

    print(
        f"D_shuffle = {d_shuffle:.6f}"
    )

    print(
        f"D_zero    = {d_zero:.6f}"
    )

    print()

    print(
        "Delta_shuffle = "
        f"{gap_shuffle:+.6f}"
    )

    print(
        "Delta_zero    = "
        f"{gap_zero:+.6f}"
    )

    print()

    print(
        "S_action = "
        f"{sensitivity:+.2f}%"
    )

    print(
        "zero relative gap = "
        f"{zero_relative:+.2f}%"
    )

    print()

    print(
        "true < shuffle: "
        f"{win_shuffle:.1f}% windows"
    )

    print(
        "true < zero   : "
        f"{win_zero:.1f}% windows"
    )

    print()

    print(
        "paired Delta_shuffle "
        "95% CI: "
        f"[{ds_low:+.6f}, "
        f"{ds_high:+.6f}]"
    )

    print(
        "paired Delta_zero "
        "95% CI: "
        f"[{dz_low:+.6f}, "
        f"{dz_high:+.6f}]"
    )

    print()

    print(
        "mean normalized future-action "
        "mismatch RMS: "
        f"{np.mean(mismatch_distances):.4f}"
    )


if __name__ == "__main__":
    main()