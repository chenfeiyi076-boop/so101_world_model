from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

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

CHECKPOINT_DIR = Path("checkpoints")


# ============================================================
# World-model temporal setup
# ============================================================

T = 10
NUM_HISTORY = 2

VAL_SEED = 12345


# ============================================================
# Fixed validation subset
# ============================================================

def make_fixed_val_subset(
    dataset,
    max_windows: int,
):
    """
    从整个 val dataset 中均匀选取固定窗口。

    目的：
    1. validation 开销不要太大
    2. 每次验证使用完全相同的 windows
    3. 覆盖整个 val split，而不是只看前几个窗口

    max_windows <= 0:
        使用全部 val windows
    """

    if (
        max_windows <= 0
        or max_windows >= len(dataset)
    ):
        return dataset

    indices = (
        torch.linspace(
            0,
            len(dataset) - 1,
            steps=max_windows,
        )
        .round()
        .long()
        .unique()
        .tolist()
    )

    return Subset(
        dataset,
        indices,
    )


# ============================================================
# Validation
# ============================================================

@torch.inference_mode()
def evaluate_val_loss(
    model: DiT,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """
    固定 validation FM draw。

    非常重要：

    每次 validation 都重新使用同一个 VAL_SEED，
    因此：
        windows 相同
        tau 相同
        epsilon 相同

    同时使用 fork_rng，
    validation 不会改变 training RNG 状态。
    """

    was_training = model.training

    model.eval()

    total_loss = 0.0
    total_samples = 0

    cuda_device_index = (
        torch.cuda.current_device()
    )

    with torch.random.fork_rng(
        devices=[
            cuda_device_index
        ]
    ):
        torch.manual_seed(
            VAL_SEED
        )

        torch.cuda.manual_seed_all(
            VAL_SEED
        )

        for batch in loader:

            latents = batch[
                "latents"
            ].to(
                device=device,
                dtype=torch.float32,
            )

            actions = batch[
                "actions"
            ].to(
                device=device,
                dtype=torch.float32,
            )

            fm = (
                prepare_flow_matching_batch(
                    latents=latents,
                    num_history=NUM_HISTORY,
                    history_noise_std=0.0,
                )
            )

            prediction = model(
                fm.noisy_latents,
                fm.tau,
                actions,
            )

            loss = flow_matching_loss(
                prediction=prediction,
                target_velocity=(
                    fm.target_velocity
                ),
                loss_mask=fm.loss_mask,
            )

            if not torch.isfinite(
                loss
            ):
                raise RuntimeError(
                    "Non-finite validation "
                    "loss."
                )

            batch_size = (
                latents.shape[0]
            )

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += (
                batch_size
            )

    if was_training:
        model.train()

    if total_samples == 0:
        raise RuntimeError(
            "Validation loader "
            "contained zero samples."
        )

    return (
        total_loss
        / total_samples
    )


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    path: Path,
    model: DiT,
    optimizer: torch.optim.Optimizer,
    step: int,
    val_loss: float | None,
    train_dataset: MultiEpisodeCachedDataset,
    val_dataset: MultiEpisodeCachedDataset,
    args,
) -> None:

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model": (
                model.state_dict()
            ),

            "optimizer": (
                optimizer.state_dict()
            ),

            "steps": step,

            # -----------------------------------------------
            # Train-only normalization stats
            # -----------------------------------------------

            "action_mean": (
                train_dataset
                .action_mean
                .cpu()
            ),

            "action_std": (
                train_dataset
                .action_std
                .cpu()
            ),

            # -----------------------------------------------
            # Validation result
            # -----------------------------------------------

            "val_loss": val_loss,

            # -----------------------------------------------
            # Split information
            # -----------------------------------------------

            "train_episode_ids": (
                list(
                    train_dataset
                    .episode_ids
                )
            ),

            "val_episode_ids": (
                list(
                    val_dataset
                    .episode_ids
                )
            ),

            # -----------------------------------------------
            # Configuration
            # -----------------------------------------------

            "config": {
                "n_frames": T,
                "num_history": (
                    NUM_HISTORY
                ),

                "frame_skip": (
                    args.frame_skip
                ),

                "action_mode": (
                    "sampled"
                ),

                "action_representation": (
                    "absolute"
                ),

                "action_dim": (
                    train_dataset
                    .condition_action_dim
                ),

                "in_channels": 16,
                "hidden_size": 384,
                "depth": 12,
                "num_heads": 6,
                "patch_size": 2,
                "mlp_ratio": 4.0,
                "use_qk_norm": True,

                "lr": args.lr,
                "weight_decay": (
                    args.weight_decay
                ),

                "batch_size": (
                    args.batch_size
                ),

                "steps": args.steps,

                "val_every": (
                    args.val_every
                ),

                "val_windows": (
                    args.val_windows
                ),

                "val_seed": (
                    VAL_SEED
                ),

                "manifest_path": (
                    MANIFEST_PATH
                ),

                "action_stats_path": (
                    ACTION_STATS_PATH
                ),
            },
        },
        path,
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--frame-skip",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=4000,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    # 为了与之前 pilot 保持一致，
    # 默认继续使用 weight_decay=0。
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--val-every",
        type=int,
        default=250,
    )

    # 周期验证使用固定 128 windows。
    # 最后还会对全部 val windows
    # 进行一次验证。
    parser.add_argument(
        "--val-windows",
        type=int,
        default=128,
    )

    args = parser.parse_args()

    # ========================================================
    # Device
    # ========================================================

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    device = torch.device(
        "cuda"
    )

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    print("=" * 80)
    print("SO101 MULTI-EPISODE TRAINING")
    print("=" * 80)

    print(
        "GPU             :",
        torch.cuda.get_device_name(
            0
        ),
    )

    print(
        "frame_skip      :",
        args.frame_skip,
    )

    print(
        "sampled frames  :",
        T,
    )

    raw_span = (
        1
        + (T - 1)
        * args.frame_skip
    )

    print(
        "raw frame span  :",
        raw_span,
    )

    print(
        "num_history     :",
        NUM_HISTORY,
    )

    print(
        "action mode     :",
        "sampled",
    )

    print(
        "action repr     :",
        "absolute",
    )

    # ========================================================
    # Train-only action statistics
    # ========================================================

    stats = ActionStats.load(
        ACTION_STATS_PATH
    )

    print()
    print("=" * 80)
    print("ACTION NORMALIZATION")
    print("=" * 80)

    print(
        "stats episodes:",
        stats.episode_ids,
    )

    print(
        "mean:",
        stats.mean,
    )

    print(
        "std :",
        stats.std,
    )

    # ========================================================
    # Train Dataset
    # ========================================================

    train_dataset = (
        MultiEpisodeCachedDataset(
            manifest_path=(
                MANIFEST_PATH
            ),
            split="train",
            n_frames=T,
            frame_skip=(
                args.frame_skip
            ),
            normalize_actions=True,
            action_mode="sampled",
            action_stats=stats,
        )
    )

    # ========================================================
    # Validation Dataset
    #
    # EXACT SAME TRAIN STATS
    # ========================================================

    val_dataset = (
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

    # ========================================================
    # Safety checks
    # ========================================================

    assert set(
        train_dataset.episode_ids
    ).isdisjoint(
        set(
            val_dataset.episode_ids
        )
    )

    assert torch.equal(
        train_dataset.action_mean,
        val_dataset.action_mean,
    )

    assert torch.equal(
        train_dataset.action_std,
        val_dataset.action_std,
    )

    assert (
        train_dataset
        .condition_action_dim
        == 6
    )

    assert (
        val_dataset
        .condition_action_dim
        == 6
    )

    print()
    print("=" * 80)
    print("DATASET")
    print("=" * 80)

    print(
        "train episodes  :",
        train_dataset.episode_ids,
    )

    print(
        "val episodes    :",
        val_dataset.episode_ids,
    )

    print(
        "train windows   :",
        len(train_dataset),
    )

    print(
        "val windows     :",
        len(val_dataset),
    )

    print(
        "action dim      :",
        train_dataset
        .condition_action_dim,
    )

    first = train_dataset[0]

    print(
        "first train ep  :",
        first[
            "episode_idx"
        ].item(),
    )

    print(
        "first indices   :",
        first[
            "indices"
        ].tolist(),
    )

    # ========================================================
    # Train DataLoader
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    # ========================================================
    # Fixed periodic validation subset
    # ========================================================

    val_subset = (
        make_fixed_val_subset(
            val_dataset,
            args.val_windows,
        )
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    # ========================================================
    # Full validation loader
    # ========================================================

    full_val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    print(
        "periodic val windows:",
        len(val_subset),
    )

    # ========================================================
    # DiT-S
    # ========================================================

    action_dim = (
        train_dataset
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

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=(
                args.weight_decay
            ),
        )
    )

    # ========================================================
    # Initial validation
    #
    # 只用于观察初始化水平。
    # 不保存为 best checkpoint。
    # ========================================================

    print()
    print("=" * 80)
    print("INITIAL VALIDATION")
    print("=" * 80)

    initial_val_loss = (
        evaluate_val_loss(
            model=model,
            loader=val_loader,
            device=device,
        )
    )

    print(
        f"step=0000  "
        f"val_fixed="
        f"{initial_val_loss:.6f}"
    )

    # ========================================================
    # Training
    # ========================================================

    model.train()

    step = 0

    recent_losses = []

    best_val_loss = float(
        "inf"
    )

    best_step = None

    torch.cuda.reset_peak_memory_stats()

    # checkpoint filenames

    best_path = (
        CHECKPOINT_DIR
        / (
            "multi_ep_dit_s_fm_"
            f"stride{args.frame_skip}_"
            "abs_sampled_best.pt"
        )
    )

    last_path = (
        CHECKPOINT_DIR
        / (
            "multi_ep_dit_s_fm_"
            f"stride{args.frame_skip}_"
            "abs_sampled_last.pt"
        )
    )

    print()
    print("=" * 80)
    print("TRAINING")
    print("=" * 80)

    while (
        step
        < args.steps
    ):

        for batch in (
            train_loader
        ):

            if (
                step
                >= args.steps
            ):
                break

            latents = batch[
                "latents"
            ].to(
                device=device,
                dtype=torch.float32,
            )

            actions = batch[
                "actions"
            ].to(
                device=device,
                dtype=torch.float32,
            )

            # -----------------------------------------------
            # 每一步重新采样 tau / epsilon
            # -----------------------------------------------

            fm = (
                prepare_flow_matching_batch(
                    latents=latents,
                    num_history=(
                        NUM_HISTORY
                    ),
                    history_noise_std=0.0,
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            prediction = model(
                fm.noisy_latents,
                fm.tau,
                actions,
            )

            loss = (
                flow_matching_loss(
                    prediction=(
                        prediction
                    ),
                    target_velocity=(
                        fm.target_velocity
                    ),
                    loss_mask=(
                        fm.loss_mask
                    ),
                )
            )

            if not torch.isfinite(
                loss
            ):
                raise RuntimeError(
                    "Non-finite loss "
                    f"at step {step}"
                )

            loss.backward()

            grad_norm = (
                torch.nn.utils
                .clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )
            )

            optimizer.step()

            step += 1

            recent_losses.append(
                loss.item()
            )

            if (
                len(recent_losses)
                > 50
            ):
                recent_losses.pop(
                    0
                )

            # -----------------------------------------------
            # Train logging
            # -----------------------------------------------

            if (
                step == 1
                or step % 25 == 0
            ):

                avg50 = (
                    sum(
                        recent_losses
                    )
                    / len(
                        recent_losses
                    )
                )

                print(
                    f"step={step:04d}  "
                    f"train="
                    f"{loss.item():.6f}  "
                    f"avg50="
                    f"{avg50:.6f}  "
                    f"grad="
                    f"{float(grad_norm):.4f}"
                )

            # -----------------------------------------------
            # Fixed validation
            # -----------------------------------------------

            if (
                step
                % args.val_every
                == 0
            ):

                val_loss = (
                    evaluate_val_loss(
                        model=model,
                        loader=val_loader,
                        device=device,
                    )
                )

                print(
                    "-" * 80
                )

                print(
                    f"step={step:04d}  "
                    f"VAL_FIXED="
                    f"{val_loss:.6f}"
                )

                # -------------------------------------------
                # Best checkpoint
                # -------------------------------------------

                if (
                    val_loss
                    < best_val_loss
                ):

                    best_val_loss = (
                        val_loss
                    )

                    best_step = (
                        step
                    )

                    save_checkpoint(
                        path=best_path,
                        model=model,
                        optimizer=(
                            optimizer
                        ),
                        step=step,
                        val_loss=(
                            val_loss
                        ),
                        train_dataset=(
                            train_dataset
                        ),
                        val_dataset=(
                            val_dataset
                        ),
                        args=args,
                    )

                    print(
                        "NEW BEST:"
                    )

                    print(
                        f"  step="
                        f"{best_step}"
                    )

                    print(
                        f"  val="
                        f"{best_val_loss:.6f}"
                    )

                    print(
                        f"  checkpoint="
                        f"{best_path}"
                    )

                print(
                    "-" * 80
                )

    # ========================================================
    # Save LAST
    # ========================================================

    final_fixed_val = (
        evaluate_val_loss(
            model=model,
            loader=val_loader,
            device=device,
        )
    )

    save_checkpoint(
        path=last_path,
        model=model,
        optimizer=optimizer,
        step=step,
        val_loss=(
            final_fixed_val
        ),
        train_dataset=(
            train_dataset
        ),
        val_dataset=(
            val_dataset
        ),
        args=args,
    )

    # ========================================================
    # Full validation for LAST model
    # ========================================================

    print()
    print("=" * 80)
    print("FULL VALIDATION — LAST MODEL")
    print("=" * 80)

    full_val_loss = (
        evaluate_val_loss(
            model=model,
            loader=(
                full_val_loader
            ),
            device=device,
        )
    )

    # ========================================================
    # Summary
    # ========================================================

    peak_gb = (
        torch.cuda
        .max_memory_allocated()
        / 1024**3
    )

    print()
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    print(
        "steps:",
        step,
    )

    print(
        "best step:",
        best_step,
    )

    print(
        "best fixed val:",
        best_val_loss,
    )

    print(
        "last fixed val:",
        final_fixed_val,
    )

    print(
        "last full val:",
        full_val_loss,
    )

    print(
        "best checkpoint:",
        best_path,
    )

    print(
        "last checkpoint:",
        last_path,
    )

    print(
        f"peak CUDA memory: "
        f"{peak_gb:.2f} GB"
    )


if __name__ == "__main__":
    main()