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

CHECKPOINT_PATH = (
    "checkpoints/episode000_dit_s_fm.pt"
)

NUM_HISTORY = 2
T = 10


def mean_ci95(x: np.ndarray):
    """
    简单 paired-window 95% CI。
    """
    mean = float(x.mean())

    if len(x) <= 1:
        return mean, float("nan"), float("nan")

    std = float(
        x.std(ddof=1)
    )

    se = std / np.sqrt(
        len(x)
    )

    return (
        mean,
        mean - 1.96 * se,
        mean + 1.96 * se,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--window-stride",
        type=int,
        default=10,
        help=(
            "Evaluate roughly non-overlapping "
            "T=10 windows."
        ),
    )

    parser.add_argument(
        "--noise-draws",
        type=int,
        default=2,
        help=(
            "Number of independent FM noise/tau "
            "draws per temporal window."
        ),
    )

    args = parser.parse_args()

    device = torch.device("cuda")

    # ========================================================
    # Load checkpoint
    # ========================================================

    print("=" * 80)
    print("LOAD CHECKPOINT")
    print("=" * 80)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    print(
        "training steps:",
        checkpoint.get(
            "steps",
            "unknown",
        ),
    )

    # ========================================================
    # Dataset
    #
    # 这里先关闭 dataset 自己的 normalization，
    # 然后严格使用 checkpoint 保存的 mean/std。
    # ========================================================

    dataset = CachedLatentDataset(
        cache_path=CACHE_PATH,
        n_frames=T,
        frame_skip=1,
        normalize_actions=False,
    )

    print(
        "dataset windows:",
        len(dataset),
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
        action_dim=6,
        mlp_ratio=4.0,
        use_qk_norm=True,
    ).to(device)

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()

    print("model loaded.")

    # ========================================================
    # Evaluation windows
    #
    # stride=10:
    #
    # 0, 10, 20, ...
    #
    # 对于 501 个 window，大约得到 51 个近似不重叠窗口。
    # ========================================================

    starts = list(
        range(
            0,
            len(dataset),
            args.window_stride,
        )
    )

    print(
        "evaluation windows:",
        len(starts),
    )

    # --------------------------------------------------------
    # shuffle 不使用相邻 window。
    #
    # 相邻 20 Hz action 太相似，
    # 所以使用距离约半条 episode 的 action window。
    # --------------------------------------------------------

    mismatch_offset = (
        len(dataset) // 2
    )

    true_window_losses = []
    shuffle_window_losses = []
    zero_window_losses = []

    action_mismatch_distances = []

    # ========================================================
    # Evaluation
    # ========================================================

    print()
    print("=" * 80)
    print("ACTION CONDITIONING EVALUATION")
    print("=" * 80)

    with torch.inference_mode():

        for eval_i, start in enumerate(
            starts
        ):

            sample = dataset[start]

            mismatch_start = (
                start + mismatch_offset
            ) % len(dataset)

            mismatch_sample = dataset[
                mismatch_start
            ]

            # ------------------------------------------------
            # Ground-truth latent
            # ------------------------------------------------

            latents = sample[
                "latents"
            ].unsqueeze(0).to(
                device=device,
                dtype=torch.float32,
            )

            # ------------------------------------------------
            # Normalize actions with TRAIN statistics
            # ------------------------------------------------

            true_raw = sample[
                "actions_raw"
            ].float()

            mismatch_raw = mismatch_sample[
                "actions_raw"
            ].float()

            true_actions = (
                true_raw
                - action_mean
            ) / action_std

            mismatch_actions = (
                mismatch_raw
                - action_mean
            ) / action_std

            true_actions = (
                true_actions
                .unsqueeze(0)
                .to(device)
            )

            mismatch_actions = (
                mismatch_actions
                .unsqueeze(0)
                .to(device)
            )

            # =================================================
            # IMPORTANT
            #
            # history 0,1 的 action 全部保持真实。
            #
            # 只修改 future action 2..9。
            #
            # 这样我们真正测试的是：
            #
            # future prediction 是否依赖 future action。
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

            # ------------------------------------------------
            # 看 true/mismatch action 到底差得够不够大
            # ------------------------------------------------

            action_distance = torch.sqrt(
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

            action_mismatch_distances.append(
                action_distance
            )

            losses_true = []
            losses_shuffle = []
            losses_zero = []

            # =================================================
            # 多个 independent tau/noise draws
            # =================================================

            for draw in range(
                args.noise_draws
            ):

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
                # 只创建一次 FM batch
                #
                # 三种 action 共用完全相同的：
                #
                # tau
                # epsilon
                # noisy latent
                # target velocity
                # =============================================

                fm = prepare_flow_matching_batch(
                    latents=latents,
                    num_history=NUM_HISTORY,
                    history_noise_std=0.0,
                )

                # ---------------- TRUE ----------------

                pred_true = model(
                    fm.noisy_latents,
                    fm.tau,
                    true_actions,
                )

                loss_true = flow_matching_loss(
                    prediction=pred_true,
                    target_velocity=fm.target_velocity,
                    loss_mask=fm.loss_mask,
                )

                # ---------------- SHUFFLE ----------------

                pred_shuffle = model(
                    fm.noisy_latents,
                    fm.tau,
                    shuffled,
                )

                loss_shuffle = flow_matching_loss(
                    prediction=pred_shuffle,
                    target_velocity=fm.target_velocity,
                    loss_mask=fm.loss_mask,
                )

                # ---------------- ZERO ----------------

                pred_zero = model(
                    fm.noisy_latents,
                    fm.tau,
                    zeroed,
                )

                loss_zero = flow_matching_loss(
                    prediction=pred_zero,
                    target_velocity=fm.target_velocity,
                    loss_mask=fm.loss_mask,
                )

                losses_true.append(
                    loss_true.item()
                )

                losses_shuffle.append(
                    loss_shuffle.item()
                )

                losses_zero.append(
                    loss_zero.item()
                )

            # ------------------------------------------------
            # 每个 temporal window 先对 noise draws 求平均。
            #
            # 后面统计单位就是 window，而不是 noise sample。
            # ------------------------------------------------

            true_window_losses.append(
                np.mean(
                    losses_true
                )
            )

            shuffle_window_losses.append(
                np.mean(
                    losses_shuffle
                )
            )

            zero_window_losses.append(
                np.mean(
                    losses_zero
                )
            )

            if (
                eval_i == 0
                or (eval_i + 1) % 10 == 0
                or eval_i == len(starts) - 1
            ):

                print(
                    f"[{eval_i + 1:03d}/{len(starts):03d}] "
                    f"start={start:03d}  "
                    f"true={true_window_losses[-1]:.4f}  "
                    f"shuffle={shuffle_window_losses[-1]:.4f}  "
                    f"zero={zero_window_losses[-1]:.4f}"
                )

    # ========================================================
    # Statistics
    # ========================================================

    true_losses = np.asarray(
        true_window_losses
    )

    shuffle_losses = np.asarray(
        shuffle_window_losses
    )

    zero_losses = np.asarray(
        zero_window_losses
    )

    delta_shuffle = (
        shuffle_losses
        - true_losses
    )

    delta_zero = (
        zero_losses
        - true_losses
    )

    mean_true = float(
        true_losses.mean()
    )

    mean_shuffle = float(
        shuffle_losses.mean()
    )

    mean_zero = float(
        zero_losses.mean()
    )

    shuffle_gap = (
        mean_shuffle
        - mean_true
    )

    zero_gap = (
        mean_zero
        - mean_true
    )

    shuffle_relative = (
        shuffle_gap
        / max(
            mean_true,
            1e-8,
        )
        * 100.0
    )

    zero_relative = (
        zero_gap
        / max(
            mean_true,
            1e-8,
        )
        * 100.0
    )

    shuffle_win_rate = float(
        np.mean(
            true_losses
            < shuffle_losses
        )
        * 100.0
    )

    zero_win_rate = float(
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
    # Output
    # ========================================================

    print()
    print("=" * 80)
    print("RESULT")
    print("=" * 80)

    print(
        f"D_true    = {mean_true:.6f}"
    )

    print(
        f"D_shuffle = {mean_shuffle:.6f}"
    )

    print(
        f"D_zero    = {mean_zero:.6f}"
    )

    print()

    print(
        "Delta_shuffle = "
        f"{shuffle_gap:+.6f} "
        f"({shuffle_relative:+.2f}%)"
    )

    print(
        "Delta_zero    = "
        f"{zero_gap:+.6f} "
        f"({zero_relative:+.2f}%)"
    )

    print()

    print(
        "true < shuffle:"
        f" {shuffle_win_rate:.1f}% windows"
    )

    print(
        "true < zero   :"
        f" {zero_win_rate:.1f}% windows"
    )

    print()

    print(
        "paired Delta_shuffle 95% CI:"
        f" [{ds_low:+.6f}, {ds_high:+.6f}]"
    )

    print(
        "paired Delta_zero 95% CI:"
        f" [{dz_low:+.6f}, {dz_high:+.6f}]"
    )

    print()

    print(
        "mean normalized future-action "
        "mismatch RMS:"
        f" {np.mean(action_mismatch_distances):.4f}"
    )


if __name__ == "__main__":
    main()