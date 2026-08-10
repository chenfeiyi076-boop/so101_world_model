from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader


# ============================================================
# Make project root importable
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.data.dataset import DummyWorldModelDataset
from src.models.dit import DiT
from src.diffusion.flow_matching import (
    prepare_flow_matching_batch,
    flow_matching_loss,
)


# ============================================================
# Config
# ============================================================

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError("Config file is empty.")

    return config


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Device
# ============================================================

def get_device(requested: str) -> torch.device:

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False."
        )

    return torch.device(requested)


# ============================================================
# Model
# ============================================================

def build_model(config: dict) -> DiT:

    model_cfg = config["model"]
    data_cfg = config["data"]

    model = DiT(
        in_channels=model_cfg.get(
            "in_channels",
            16,
        ),
        patch_size=model_cfg["patch_size"],
        hidden_size=model_cfg["hidden_size"],
        depth=model_cfg["depth"],
        num_heads=model_cfg["num_heads"],
        action_dim=data_cfg["action_dim"],
        mlp_ratio=model_cfg.get(
            "mlp_ratio",
            4.0,
        ),
        use_qk_norm=model_cfg.get(
            "use_qk_norm",
            True,
        ),
    )

    return model


# ============================================================
# Dataset
# ============================================================

def build_dataloader(config: dict) -> DataLoader:

    data_cfg = config["data"]
    train_cfg = config["training"]

    dataset = DummyWorldModelDataset(
        num_episodes=2,
        episode_length=20,
        n_frames=data_cfg["n_frames"],
        frame_skip=data_cfg["frame_skip"],
        action_dim=data_cfg["action_dim"],

        # CPU debug 时暂时缩小 latent 空间。
        latent_channels=data_cfg.get(
            "latent_channels",
            16,
        ),

        latent_height=data_cfg.get(
            "latent_height",
            8,
        ),

        latent_width=data_cfg.get(
            "latent_width",
            8,
        ),

        windows_per_episode=5,
        seed=train_cfg.get(
            "seed",
            0,
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    return loader


# ============================================================
# Parameter statistics
# ============================================================

def count_parameters(model: torch.nn.Module) -> int:
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================
# Fixed-batch overfit test
# ============================================================

def run_overfit_test(
    model: DiT,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_history: int,
    steps: int,
) -> None:

    print()
    print("=" * 70)
    print("FIXED-BATCH OVERFIT TEST")
    print("=" * 70)

    batch = next(iter(loader))

    latents = batch["latents"].to(
        device=device,
        dtype=torch.float32,
    )

    actions = batch["actions"].to(
        device=device,
        dtype=torch.float32,
    )

    print("latents :", tuple(latents.shape))
    print("actions :", tuple(actions.shape))

    # --------------------------------------------------------
    # VERY IMPORTANT:
    #
    # 这里 FM batch 只创建一次。
    #
    # 因此：
    # tau / noise / target 全部固定。
    #
    # 目的不是模拟真实训练，
    # 而是检查模型有没有能力 overfit 一个固定目标。
    # --------------------------------------------------------

    torch.manual_seed(12345)

    fm = prepare_flow_matching_batch(
        latents=latents,
        num_history=num_history,
        history_noise_std=0.0,
    )

    initial_loss = None

    model.train()

    for step in range(1, steps + 1):

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
                f"Non-finite loss detected at step {step}: "
                f"{loss.item()}"
            )

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        loss_value = loss.item()

        if initial_loss is None:
            initial_loss = loss_value

        if (
            step == 1
            or step % 10 == 0
            or step == steps
        ):
            print(
                f"step={step:4d}  "
                f"loss={loss_value:.6f}  "
                f"grad_norm={float(grad_norm):.6f}"
            )

    final_loss = loss.item()

    print()
    print(
        f"initial loss : {initial_loss:.6f}"
    )

    print(
        f"final loss   : {final_loss:.6f}"
    )

    print(
        f"ratio        : "
        f"{final_loss / initial_loss:.4f}"
    )

    if final_loss < initial_loss:
        print("OVERFIT CHECK: PASS")
    else:
        print("OVERFIT CHECK: WARNING - loss did not decrease")


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/debug.yaml",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    train_cfg = config["training"]
    data_cfg = config["data"]

    seed = train_cfg.get(
        "seed",
        0,
    )

    set_seed(seed)

    device = get_device(
        args.device
    )

    print("=" * 70)
    print("SO101 WORLD MODEL DEBUG TRAIN")
    print("=" * 70)

    print(
        "device:",
        device,
    )

    print(
        "torch version:",
        torch.__version__,
    )

    print(
        "cuda available:",
        torch.cuda.is_available(),
    )

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    loader = build_dataloader(
        config
    )

    model = build_model(
        config
    ).to(device)

    print(
        "trainable parameters:",
        f"{count_parameters(model):,}",
    )

    learning_rate = train_cfg.get(
        "lr",
        1e-4,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )

    run_overfit_test(
        model=model,
        loader=loader,
        optimizer=optimizer,
        device=device,
        num_history=data_cfg["num_history"],
        steps=args.steps,
    )


if __name__ == "__main__":
    main()