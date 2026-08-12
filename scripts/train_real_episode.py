from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

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

CHECKPOINT_DIR = Path("checkpoints")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--steps",
        type=int,
        default=500,
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

    args = parser.parse_args()

    device = torch.device("cuda")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # ========================================================
    # Dataset
    # ========================================================

    dataset = CachedLatentDataset(
        cache_path=CACHE_PATH,
        n_frames=10,
        frame_skip=1,
        normalize_actions=True,
    )

    print("dataset windows:", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    # ========================================================
    # DiT-S
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.0,
    )

    # ========================================================
    # Training
    # ========================================================

    model.train()

    step = 0
    recent_losses = []

    torch.cuda.reset_peak_memory_stats()

    while step < args.steps:

        for batch in loader:

            if step >= args.steps:
                break

            latents = batch["latents"].to(
                device=device,
                dtype=torch.float32,
            )

            actions = batch["actions"].to(
                device=device,
                dtype=torch.float32,
            )

            # ------------------------------------------------
            # 这里和 R0 最大区别：
            #
            # 每一个 training step 都重新采样
            #
            # tau ~ U(0,1)
            # epsilon ~ N(0,I)
            #
            # 不再固定 FM target。
            # ------------------------------------------------

            fm = prepare_flow_matching_batch(
                latents=latents,
                num_history=2,
                history_noise_std=0.0,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            prediction = model(
                fm.noisy_latents,
                fm.tau,
                actions,
            )

            loss = flow_matching_loss(
                prediction=prediction,
                target_velocity=fm.target_velocity,
                loss_mask=fm.loss_mask,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at step {step}"
                )

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            step += 1

            recent_losses.append(
                loss.item()
            )

            if len(recent_losses) > 50:
                recent_losses.pop(0)

            if (
                step == 1
                or step % 25 == 0
            ):

                avg_loss = sum(
                    recent_losses
                ) / len(recent_losses)

                print(
                    f"step={step:04d}  "
                    f"loss={loss.item():.6f}  "
                    f"avg50={avg_loss:.6f}  "
                    f"grad={float(grad_norm):.4f}"
                )

    # ========================================================
    # Save
    # ========================================================

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        CHECKPOINT_DIR
        / "episode000_dit_s_fm.pt"
    )

    torch.save(
        {
            "model": model.state_dict(),

            "optimizer": optimizer.state_dict(),

            "steps": step,

            "action_mean": dataset.action_mean,

            "action_std": dataset.action_std,

            "config": {
                "n_frames": 10,
                "num_history": 2,
                "frame_skip": 1,
                "hidden_size": 384,
                "depth": 12,
                "num_heads": 6,
                "patch_size": 2,
            },
        },
        checkpoint_path,
    )

    peak_gb = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        "checkpoint:",
        checkpoint_path,
    )

    print(
        f"peak CUDA memory: {peak_gb:.2f} GB"
    )


if __name__ == "__main__":
    main()