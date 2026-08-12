from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset


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

DEVICE = torch.device("cuda")

NUM_HISTORY = 2
T = 10

# 直接选刚才已经肉眼检查过的区域。
WINDOW_START = 250

STEPS = 200
LR = 1e-4


def main():

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # ========================================================
    # Dataset
    # ========================================================

    dataset = CachedLatentDataset(
        cache_path=CACHE_PATH,
        n_frames=T,
        frame_skip=1,
        normalize_actions=True,
    )

    # 只取一个真实 window。
    subset = Subset(
        dataset,
        [WINDOW_START],
    )

    loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    latents = batch["latents"].to(
        DEVICE,
    )

    actions = batch["actions"].to(
        DEVICE,
    )

    print(
        "latents:",
        latents.shape,
    )

    print(
        "actions:",
        actions.shape,
    )

    print(
        "indices:",
        batch["indices"],
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
    ).to(
        DEVICE
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=0.0,
    )

    # ========================================================
    # Fixed FM target
    #
    # R0 是 overfit sanity check，
    # 所以 tau/noise 固定。
    # ========================================================

    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)

    fm = prepare_flow_matching_batch(
        latents=latents,
        num_history=NUM_HISTORY,
        history_noise_std=0.0,
    )

    model.train()

    initial_loss = None

    torch.cuda.reset_peak_memory_stats()

    for step in range(
        1,
        STEPS + 1,
    ):

        optimizer.zero_grad(
            set_to_none=True
        )

        prediction = model(
            fm.noisy_latents,
            fm.tau,
            actions,
        )

        loss = flow_matching_loss(
            prediction,
            fm.target_velocity,
            fm.loss_mask,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at step {step}"
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

        if initial_loss is None:
            initial_loss = loss.item()

        if (
            step == 1
            or step % 10 == 0
        ):
            print(
                f"step={step:04d} "
                f"loss={loss.item():.6f}"
            )

    final_loss = loss.item()

    peak_gb = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    print()
    print(
        f"initial loss: {initial_loss:.6f}"
    )

    print(
        f"final loss  : {final_loss:.6f}"
    )

    print(
        f"ratio       : "
        f"{final_loss / initial_loss:.4f}"
    )

    print(
        f"peak memory : {peak_gb:.2f} GB"
    )


if __name__ == "__main__":
    main()