from __future__ import annotations

import copy
import math
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


def build_optimizer(config: dict, model: torch.nn.Module) -> torch.optim.AdamW:
    """Build the resolved causal AdamW recipe without changing parameter groups."""

    train = config["train"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=train["lr"],
        betas=tuple(train["betas"]),
        eps=train["eps"],
        weight_decay=train["weight_decay"],
    )


def scheduler_warmup_steps(total_steps: int, warmup_ratio: float) -> int:
    """Resolve a rounded warmup count while keeping one update after warmup."""

    total_steps = int(total_steps)
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    warmup_steps = round(total_steps * float(warmup_ratio))
    return min(max(int(warmup_steps), 0), total_steps - 1)


def lr_multiplier(
    update_index: int,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    """Return the LR multiplier used by one zero-based optimizer update.

    Warmup updates use (index + 1) / warmup_steps, so the final warmup update
    reaches the base LR. The post-warmup cosine starts at base LR and the final
    optimizer update reaches min_lr_ratio exactly.
    """

    update_index = int(update_index)
    total_steps = int(total_steps)
    warmup_steps = int(warmup_steps)
    min_lr_ratio = float(min_lr_ratio)
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0 <= update_index < total_steps:
        raise ValueError("update_index must satisfy 0 <= value < total_steps")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must satisfy 0 <= value < total_steps")
    if not 0.0 < min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must satisfy 0 < value <= 1")

    if warmup_steps > 0 and update_index < warmup_steps:
        return float(update_index + 1) / float(warmup_steps)

    decay_updates = total_steps - warmup_steps
    decay_index = update_index - warmup_steps
    if decay_updates <= 1:
        return min_lr_ratio
    progress = float(decay_index) / float(decay_updates - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


class UpdateLRScheduler:
    """Set LR for an explicit optimizer update index.

    The constructor sets update 0's LR before the first optimizer.step(). Call
    step() immediately after each optimizer.step() to prepare the next update.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        scheduler_type: str,
        total_steps: int,
        warmup_steps: int,
        min_lr_ratio: float,
    ) -> None:
        if scheduler_type not in {"constant", "warmup_cosine"}:
            raise ValueError("unsupported scheduler type")
        if int(total_steps) <= 0:
            raise ValueError("total_steps must be positive")
        if not 0 <= int(warmup_steps) < int(total_steps):
            raise ValueError("warmup_steps must satisfy 0 <= value < total_steps")
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr_ratio = float(min_lr_ratio)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.update_index = 0
        self._set_update_lr()

    def _multiplier(self) -> float:
        if self.scheduler_type == "constant":
            return 1.0
        return lr_multiplier(
            self.update_index,
            self.total_steps,
            self.warmup_steps,
            self.min_lr_ratio,
        )

    def _set_update_lr(self) -> None:
        multiplier = self._multiplier()
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * multiplier

    def step(self) -> None:
        if self.update_index < self.total_steps - 1:
            self.update_index += 1
        self._set_update_lr()

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        return {
            "scheduler_type": self.scheduler_type,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "base_lrs": list(self.base_lrs),
            "update_index": self.update_index,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        for key in (
            "scheduler_type",
            "total_steps",
            "warmup_steps",
            "min_lr_ratio",
            "base_lrs",
            "update_index",
        ):
            if key not in state_dict:
                raise RuntimeError(f"scheduler state is missing {key!r}")
        if state_dict["scheduler_type"] != self.scheduler_type:
            raise RuntimeError("scheduler type differs from configured scheduler")
        if int(state_dict["total_steps"]) != self.total_steps:
            raise RuntimeError("scheduler total_steps differs from config")
        self.warmup_steps = int(state_dict["warmup_steps"])
        self.min_lr_ratio = float(state_dict["min_lr_ratio"])
        self.base_lrs = [float(value) for value in state_dict["base_lrs"]]
        if len(self.base_lrs) != len(self.optimizer.param_groups):
            raise RuntimeError("scheduler parameter-group count mismatch")
        self.update_index = int(state_dict["update_index"])
        if not 0 <= self.update_index < self.total_steps:
            raise RuntimeError("scheduler update_index is out of range")
        self._set_update_lr()


def build_scheduler(
    config: dict,
    optimizer: torch.optim.Optimizer,
) -> UpdateLRScheduler:
    train = config["train"]
    scheduler = train["scheduler"]
    warmup_steps = scheduler_warmup_steps(
        train["steps"], scheduler["warmup_ratio"]
    )
    return UpdateLRScheduler(
        optimizer,
        scheduler_type=scheduler["type"],
        total_steps=train["steps"],
        warmup_steps=warmup_steps,
        min_lr_ratio=scheduler["min_lr_ratio"],
    )


class ExponentialMovingAverage:
    """FP32 model EMA with copied buffers and no gradient participation."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("EMA decay must satisfy 0 <= value < 1")
        self.model = copy.deepcopy(model).float()
        self.model.eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def update(self, raw_model: torch.nn.Module) -> None:
        raw_parameters = dict(raw_model.named_parameters())
        ema_parameters = dict(self.model.named_parameters())
        if raw_parameters.keys() != ema_parameters.keys():
            raise RuntimeError("raw/EMA parameter names differ")
        for name, ema_parameter in ema_parameters.items():
            raw_parameter = raw_parameters[name].detach().to(
                device=ema_parameter.device, dtype=torch.float32
            )
            ema_parameter.mul_(self.decay).add_(
                raw_parameter, alpha=1.0 - self.decay
            )

        raw_buffers = dict(raw_model.named_buffers())
        ema_buffers = dict(self.model.named_buffers())
        if raw_buffers.keys() != ema_buffers.keys():
            raise RuntimeError("raw/EMA buffer names differ")
        for name, ema_buffer in ema_buffers.items():
            raw_buffer = raw_buffers[name].detach().to(
                device=ema_buffer.device, dtype=ema_buffer.dtype
            )
            ema_buffer.copy_(raw_buffer)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().clone()
            for name, value in self.model.state_dict().items()
        }

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state_dict, strict=True)


def validation_model(
    raw_model: torch.nn.Module,
    ema: ExponentialMovingAverage | None,
) -> torch.nn.Module:
    return ema.model if ema is not None else raw_model


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
