from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.multi_episode_dataset import (
    ActionStats,
    MultiEpisodeCachedDataset,
)
from src.models.dit import DiT
from src.diffusion.flow_matching import (
    prepare_flow_matching_batch,
    flow_matching_loss,
)


# ============================================================
# Paths
# ============================================================

MANIFEST_PATH = (
    "data/latent_cache/so101_front/"
    "multi_episode_manifest.json"
)

ACTION_STATS_PATH = (
    "data/multi_episode_pilot/"
    "action_stats_absolute.pt"
)

DEFAULT_CHECKPOINT = (
    "checkpoints/"
    "multi_ep_dit_s_fm_"
    "stride4_abs_sampled_best.pt"
)


# ============================================================
# Temporal setup
# ============================================================

T = 10
NUM_HISTORY = 2

EVAL_SEED = 2026


# ============================================================
# Statistics
# ============================================================

def mean_ci95(
    values: list[float],
) -> tuple[
    float,
    float,
    float,
]:
    """
    Window-level 95% CI:

        mean ± 1.96 * SE

    注意：
        这是 window-level CI，
        不是 episode-level CI。
    """

    x = torch.tensor(
        values,
        dtype=torch.float64,
    )

    mean = float(
        x.mean()
    )

    if len(x) <= 1:
        return (
            mean,
            mean,
            mean,
        )

    std = float(
        x.std(
            unbiased=True
        )
    )

    se = (
        std
        / math.sqrt(
            len(x)
        )
    )

    half = (
        1.96
        * se
    )

    return (
        mean,
        mean - half,
        mean + half,
    )


# ============================================================
# Select evaluation windows
# ============================================================

def select_windows_per_episode(
    dataset: MultiEpisodeCachedDataset,
    max_windows_per_episode: int,
) -> dict[
    int,
    list[int],
]:
    """
    每个 val episode 均匀选窗口。

    返回：
        {
            episode_id:
                [global_dataset_idx, ...]
        }

    max_windows_per_episode <= 0:
        使用该 episode 全部 windows。
    """

    by_episode = defaultdict(
        list
    )

    for global_idx, (
        episode_id,
        start,
    ) in enumerate(
        dataset.windows
    ):

        by_episode[
            int(episode_id)
        ].append(
            global_idx
        )

    selected = {}

    for episode_id in (
        dataset.episode_ids
    ):

        candidates = (
            by_episode[
                int(episode_id)
            ]
        )

        if (
            max_windows_per_episode
            <= 0
            or max_windows_per_episode
            >= len(candidates)
        ):

            chosen = candidates

        else:

            positions = (
                torch.linspace(
                    0,
                    len(candidates) - 1,
                    steps=(
                        max_windows_per_episode
                    ),
                )
                .round()
                .long()
                .unique()
                .tolist()
            )

            chosen = [
                candidates[p]
                for p in positions
            ]

        selected[
            int(episode_id)
        ] = chosen

    return selected


# ============================================================
# Mismatch lookup
# ============================================================

def build_window_lookup(
    dataset: MultiEpisodeCachedDataset,
) -> dict[
    tuple[int, int],
    int,
]:
    """
    (episode_id, local_start)
        ->
    global dataset index
    """

    lookup = {}

    for idx, (
        episode_id,
        start,
    ) in enumerate(
        dataset.windows
    ):

        lookup[
            (
                int(episode_id),
                int(start),
            )
        ] = idx

    return lookup


def get_mismatch_index(
    dataset: MultiEpisodeCachedDataset,
    lookup: dict[
        tuple[int, int],
        int,
    ],
    target_idx: int,
) -> int:
    """
    在 SAME episode 内选择一个远距离 window
    作为 mismatched action source。

    使用半个 episode window-range 的循环偏移。

    这样不会把另一个 episode 的风格/轨迹
    当成 action intervention 的额外变量。
    """

    episode_id, start = (
        dataset.windows[
            target_idx
        ]
    )

    episode_id = int(
        episode_id
    )

    start = int(
        start
    )

    num_windows = int(
        dataset.windows_per_episode[
            episode_id
        ]
    )

    if num_windows < 2:
        raise RuntimeError(
            "Episode has too few windows "
            "for action mismatch."
        )

    shift = max(
        1,
        num_windows // 2,
    )

    mismatch_start = (
        start + shift
    ) % num_windows

    mismatch_idx = lookup[
        (
            episode_id,
            mismatch_start,
        )
    ]

    if mismatch_idx == target_idx:
        raise RuntimeError(
            "Mismatch index equals "
            "target index."
        )

    return mismatch_idx


# ============================================================
# Evaluate one window
# ============================================================

@torch.inference_mode()
def evaluate_window(
    model: DiT,
    dataset: MultiEpisodeCachedDataset,
    target_idx: int,
    mismatch_idx: int,
    device: torch.device,
    noise_draws: int,
) -> dict[str, float]:

    target = dataset[
        target_idx
    ]

    mismatch = dataset[
        mismatch_idx
    ]

    # --------------------------------------------------------
    # Safety: mismatch stays inside same episode
    # --------------------------------------------------------

    target_ep = int(
        target[
            "episode_idx"
        ].item()
    )

    mismatch_ep = int(
        mismatch[
            "episode_idx"
        ].item()
    )

    if target_ep != mismatch_ep:
        raise RuntimeError(
            "Mismatch crossed episodes."
        )

    # --------------------------------------------------------
    # Inputs
    # --------------------------------------------------------

    latents = (
        target[
            "latents"
        ]
        .unsqueeze(0)
        .to(
            device=device,
            dtype=torch.float32,
        )
    )

    true_actions = (
        target[
            "actions"
        ]
        .unsqueeze(0)
        .to(
            device=device,
            dtype=torch.float32,
        )
    )

    mismatch_actions = (
        mismatch[
            "actions"
        ]
        .unsqueeze(0)
        .to(
            device=device,
            dtype=torch.float32,
        )
    )

    # --------------------------------------------------------
    # Intervention
    #
    # History action 保持真实。
    # 只干预 future action slots H:T。
    # --------------------------------------------------------

    shuffled_actions = (
        true_actions.clone()
    )

    shuffled_actions[
        :,
        NUM_HISTORY:,
    ] = (
        mismatch_actions[
            :,
            NUM_HISTORY:,
        ]
    )

    zero_actions = (
        true_actions.clone()
    )

    zero_actions[
        :,
        NUM_HISTORY:,
    ] = 0.0

    # --------------------------------------------------------
    # Action mismatch magnitude
    # --------------------------------------------------------

    action_difference = (
        true_actions[
            :,
            NUM_HISTORY:,
        ]
        - shuffled_actions[
            :,
            NUM_HISTORY:,
        ]
    )

    mismatch_rms = float(
        torch.sqrt(
            torch.mean(
                action_difference
                ** 2
            )
        )
    )

    # --------------------------------------------------------
    # Multiple FM noise draws
    # --------------------------------------------------------

    true_losses = []
    shuffle_losses = []
    zero_losses = []

    for _ in range(
        noise_draws
    ):

        # One FM draw.
        #
        # SAME:
        #   noisy_latents
        #   tau
        #   target_velocity
        #
        # for all three action conditions.

        fm = (
            prepare_flow_matching_batch(
                latents=latents,
                num_history=(
                    NUM_HISTORY
                ),
                history_noise_std=0.0,
            )
        )

        # ====================================================
        # True
        # ====================================================

        pred_true = model(
            fm.noisy_latents,
            fm.tau,
            true_actions,
        )

        loss_true = (
            flow_matching_loss(
                prediction=pred_true,
                target_velocity=(
                    fm.target_velocity
                ),
                loss_mask=(
                    fm.loss_mask
                ),
            )
        )

        # ====================================================
        # Shuffled
        # ====================================================

        pred_shuffle = model(
            fm.noisy_latents,
            fm.tau,
            shuffled_actions,
        )

        loss_shuffle = (
            flow_matching_loss(
                prediction=(
                    pred_shuffle
                ),
                target_velocity=(
                    fm.target_velocity
                ),
                loss_mask=(
                    fm.loss_mask
                ),
            )
        )

        # ====================================================
        # Zero
        # ====================================================

        pred_zero = model(
            fm.noisy_latents,
            fm.tau,
            zero_actions,
        )

        loss_zero = (
            flow_matching_loss(
                prediction=pred_zero,
                target_velocity=(
                    fm.target_velocity
                ),
                loss_mask=(
                    fm.loss_mask
                ),
            )
        )

        true_losses.append(
            float(
                loss_true
            )
        )

        shuffle_losses.append(
            float(
                loss_shuffle
            )
        )

        zero_losses.append(
            float(
                loss_zero
            )
        )

    return {
        "episode": target_ep,

        "target_start": int(
            target[
                "window_start"
            ].item()
        ),

        "mismatch_start": int(
            mismatch[
                "window_start"
            ].item()
        ),

        "true": (
            sum(true_losses)
            / len(true_losses)
        ),

        "shuffle": (
            sum(shuffle_losses)
            / len(shuffle_losses)
        ),

        "zero": (
            sum(zero_losses)
            / len(zero_losses)
        ),

        "mismatch_rms": (
            mismatch_rms
        ),
    }


# ============================================================
# Print report
# ============================================================

def report_results(
    title: str,
    rows: list[dict],
) -> None:

    if not rows:
        raise RuntimeError(
            "No evaluation rows."
        )

    true_values = [
        x["true"]
        for x in rows
    ]

    shuffle_values = [
        x["shuffle"]
        for x in rows
    ]

    zero_values = [
        x["zero"]
        for x in rows
    ]

    delta_shuffle = [
        x["shuffle"]
        - x["true"]
        for x in rows
    ]

    delta_zero = [
        x["zero"]
        - x["true"]
        for x in rows
    ]

    mismatch_values = [
        x["mismatch_rms"]
        for x in rows
    ]

    D_true = (
        sum(true_values)
        / len(true_values)
    )

    D_shuffle = (
        sum(shuffle_values)
        / len(shuffle_values)
    )

    D_zero = (
        sum(zero_values)
        / len(zero_values)
    )

    delta_shuffle_mean = (
        D_shuffle
        - D_true
    )

    delta_zero_mean = (
        D_zero
        - D_true
    )

    S_action = (
        delta_shuffle_mean
        / D_true
        * 100.0
    )

    zero_gap = (
        delta_zero_mean
        / D_true
        * 100.0
    )

    true_lt_shuffle = (
        sum(
            1
            for x in rows
            if (
                x["true"]
                < x["shuffle"]
            )
        )
        / len(rows)
        * 100.0
    )

    true_lt_zero = (
        sum(
            1
            for x in rows
            if (
                x["true"]
                < x["zero"]
            )
        )
        / len(rows)
        * 100.0
    )

    (
        _,
        shuffle_ci_low,
        shuffle_ci_high,
    ) = mean_ci95(
        delta_shuffle
    )

    (
        _,
        zero_ci_low,
        zero_ci_high,
    ) = mean_ci95(
        delta_zero
    )

    mean_mismatch = (
        sum(mismatch_values)
        / len(mismatch_values)
    )

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    print(
        "windows:",
        len(rows),
    )

    print()

    print(
        "D_true    = "
        f"{D_true:.6f}"
    )

    print(
        "D_shuffle = "
        f"{D_shuffle:.6f}"
    )

    print(
        "D_zero    = "
        f"{D_zero:.6f}"
    )

    print()

    print(
        "Delta_shuffle = "
        f"{delta_shuffle_mean:+.6f}"
    )

    print(
        "Delta_zero    = "
        f"{delta_zero_mean:+.6f}"
    )

    print()

    print(
        "S_action = "
        f"{S_action:+.2f}%"
    )

    print(
        "zero relative gap = "
        f"{zero_gap:+.2f}%"
    )

    print()

    print(
        "true < shuffle: "
        f"{true_lt_shuffle:.1f}% "
        "windows"
    )

    print(
        "true < zero   : "
        f"{true_lt_zero:.1f}% "
        "windows"
    )

    print()

    print(
        "paired Delta_shuffle "
        "95% CI: "
        f"[{shuffle_ci_low:+.6f}, "
        f"{shuffle_ci_high:+.6f}]"
    )

    print(
        "paired Delta_zero "
        "95% CI: "
        f"[{zero_ci_low:+.6f}, "
        f"{zero_ci_high:+.6f}]"
    )

    print()

    print(
        "mean normalized "
        "future-action mismatch RMS: "
        f"{mean_mismatch:.4f}"
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            DEFAULT_CHECKPOINT
        ),
    )

    parser.add_argument(
        "--frame-skip",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--noise-draws",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--windows-per-episode",
        type=int,
        default=64,
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    device = torch.device(
        "cuda"
    )

    torch.manual_seed(
        EVAL_SEED
    )

    torch.cuda.manual_seed_all(
        EVAL_SEED
    )

    # ========================================================
    # Load checkpoint
    # ========================================================

    checkpoint_path = Path(
        args.checkpoint
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    print("=" * 80)
    print("MULTI-EPISODE ACTION INTERVENTION")
    print("=" * 80)

    print(
        "checkpoint:",
        checkpoint_path,
    )

    print(
        "checkpoint step:",
        checkpoint.get(
            "steps",
            None,
        ),
    )

    print(
        "checkpoint val loss:",
        checkpoint.get(
            "val_loss",
            None,
        ),
    )

    # ========================================================
    # Stats
    # ========================================================

    stats = ActionStats.load(
        ACTION_STATS_PATH
    )

    # Safety:
    # checkpoint stats must equal
    # current train stats.

    if not torch.allclose(
        checkpoint[
            "action_mean"
        ].float(),
        stats.mean.float(),
    ):
        raise RuntimeError(
            "Checkpoint action_mean "
            "does not match train stats."
        )

    if not torch.allclose(
        checkpoint[
            "action_std"
        ].float(),
        stats.std.float(),
    ):
        raise RuntimeError(
            "Checkpoint action_std "
            "does not match train stats."
        )

    # ========================================================
    # Validation Dataset
    # ========================================================

    dataset = (
        MultiEpisodeCachedDataset(
            manifest_path=(
                MANIFEST_PATH
            ),
            split="val",
            n_frames=T,
            frame_skip=(
                args.frame_skip
            ),
            normalize_actions=True,
            action_mode="sampled",
            action_stats=stats,
        )
    )

    print(
        "val episodes:",
        dataset.episode_ids,
    )

    print(
        "total val windows:",
        len(dataset),
    )

    print(
        "noise draws:",
        args.noise_draws,
    )

    print(
        "windows per episode:",
        args.windows_per_episode,
    )

    # ========================================================
    # Model
    # ========================================================

    action_dim = (
        dataset
        .condition_action_dim
    )

    model = DiT(
        in_channels=16,
        patch_size=2,
        hidden_size=384,
        depth=12,
        num_heads=6,
        action_dim=action_dim,
        mlp_ratio=4.0,
        use_qk_norm=True,
    ).to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "model"
        ]
    )

    model.eval()

    # ========================================================
    # Evaluation window selection
    # ========================================================

    selected = (
        select_windows_per_episode(
            dataset=dataset,
            max_windows_per_episode=(
                args
                .windows_per_episode
            ),
        )
    )

    lookup = (
        build_window_lookup(
            dataset
        )
    )

    all_rows = []

    rows_by_episode = defaultdict(
        list
    )

    total_windows = sum(
        len(v)
        for v in selected.values()
    )

    counter = 0

    # ========================================================
    # Evaluate
    # ========================================================

    for episode_id in (
        dataset.episode_ids
    ):

        indices = selected[
            int(episode_id)
        ]

        for target_idx in indices:

            mismatch_idx = (
                get_mismatch_index(
                    dataset=dataset,
                    lookup=lookup,
                    target_idx=(
                        target_idx
                    ),
                )
            )

            row = evaluate_window(
                model=model,
                dataset=dataset,
                target_idx=(
                    target_idx
                ),
                mismatch_idx=(
                    mismatch_idx
                ),
                device=device,
                noise_draws=(
                    args.noise_draws
                ),
            )

            all_rows.append(
                row
            )

            rows_by_episode[
                int(episode_id)
            ].append(
                row
            )

            counter += 1

            if (
                counter == 1
                or counter % 10 == 0
                or counter == total_windows
            ):

                print(
                    "evaluated "
                    f"{counter}/"
                    f"{total_windows}"
                )

    # ========================================================
    # Per-episode reports
    # ========================================================

    for episode_id in (
        dataset.episode_ids
    ):

        report_results(
            title=(
                "VAL EPISODE "
                f"{episode_id}"
            ),
            rows=(
                rows_by_episode[
                    int(episode_id)
                ]
            ),
        )

    # ========================================================
    # Aggregate report
    # ========================================================

    report_results(
        title=(
            "VAL AGGREGATE"
        ),
        rows=all_rows,
    )

    print()
    print(
        "NOTE: confidence intervals "
        "above are window-level, "
        "not episode-level."
    )


if __name__ == "__main__":
    main()