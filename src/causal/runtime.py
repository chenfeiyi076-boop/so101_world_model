from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext

import torch

from src.diffusion.flow_matching import (
    flow_matching_loss,
    prepare_flow_matching_batch,
)

from .model import CausalDiT


SUPPORTED_PRECISIONS = {"fp32", "bf16"}


def validate_precision_device(precision: str, device: torch.device) -> None:
    if precision not in SUPPORTED_PRECISIONS:
        raise ValueError(
            f"precision must be one of {sorted(SUPPORTED_PRECISIONS)}, got {precision!r}"
        )
    if precision == "bf16" and device.type != "cuda":
        raise RuntimeError("bf16 causal training/evaluation requires a CUDA device")


def precision_context(device: torch.device, precision: str):
    """Return the shared train/evaluation forward precision context."""

    validate_precision_device(precision, device)
    if precision == "fp32":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def build_model(config: dict) -> CausalDiT:
    model = config["model"]
    return CausalDiT(
        in_channels=model["in_channels"],
        patch_size=model["patch_size"],
        hidden_size=model["hidden_size"],
        depth=model["depth"],
        num_heads=model["num_heads"],
        action_dim=config["action"]["effective_action_dim"],
        mlp_ratio=model["mlp_ratio"],
        use_qk_norm=model["use_qk_norm"],
    )


def causal_flow_loss(
    model: CausalDiT,
    batch: dict[str, torch.Tensor],
    *,
    device: torch.device,
    num_history: int,
    precision: str = "fp32",
) -> torch.Tensor:
    latents = batch["latents"].to(device=device, dtype=torch.float32)
    action_cond = batch["action_cond"].to(device=device, dtype=torch.float32)
    action_valid_mask = batch["action_valid_mask"].to(
        device=device, dtype=torch.bool
    )
    with precision_context(device, precision):
        fm = prepare_flow_matching_batch(
            latents=latents,
            num_history=num_history,
            history_noise_std=0.0,
        )
        prediction = model(
            fm.noisy_latents,
            fm.tau,
            action_cond,
            action_valid_mask,
        )
        return flow_matching_loss(
            prediction=prediction,
            target_velocity=fm.target_velocity,
            loss_mask=fm.loss_mask,
        )


@torch.inference_mode()
def evaluate_flow_loss(
    model: CausalDiT,
    loader: Iterable[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    num_history: int,
    seed: int,
    precision: str = "fp32",
) -> float:
    validate_precision_device(precision, device)
    was_training = model.training
    model.eval()
    total = 0.0
    samples = 0
    devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        for batch in loader:
            loss = causal_flow_loss(
                model,
                batch,
                device=device,
                num_history=num_history,
                precision=precision,
            )
            batch_size = int(batch["latents"].shape[0])
            total += loss.detach().item() * batch_size
            samples += batch_size
    if was_training:
        model.train()
    if samples == 0:
        raise RuntimeError("evaluation loader has no samples")
    return total / samples
