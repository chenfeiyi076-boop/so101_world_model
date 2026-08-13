from __future__ import annotations

import torch


def euler_tau_schedule(
    num_inference_steps: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    return torch.linspace(
        1.0, 0.0, num_inference_steps + 1, device=device, dtype=dtype
    )


def _model_velocity(
    model: torch.nn.Module,
    checkpoint_type: str,
    latents: torch.Tensor,
    tau: torch.Tensor,
    action_cond: torch.Tensor,
    action_valid_mask: torch.Tensor,
) -> torch.Tensor:
    if checkpoint_type == "causal_v2":
        return model(latents, tau, action_cond, action_valid_mask)
    if checkpoint_type == "legacy_v1":
        return model(latents, tau, action_cond)
    raise ValueError(f"unknown checkpoint type {checkpoint_type!r}")


@torch.inference_mode()
def euler_sample_next(
    *,
    model: torch.nn.Module,
    checkpoint_type: str,
    history_latents: torch.Tensor,
    action_cond: torch.Tensor,
    action_valid_mask: torch.Tensor,
    num_inference_steps: int = 10,
    generator: torch.Generator | None = None,
    initial_noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate one future latent by integrating dz/dtau=v from tau=1 to 0."""

    if history_latents.ndim != 5:
        raise ValueError("history_latents must be [B,H,C,Y,X]")
    batch, history, channels, height, width = history_latents.shape
    if action_cond.ndim != 3 or action_cond.shape[:2] != (batch, history + 1):
        raise ValueError("action_cond must be [B,H+1,A]")
    if action_valid_mask.shape != (batch, history + 1):
        raise ValueError("action_valid_mask must be [B,H+1]")

    if initial_noise is None:
        current = torch.randn(
            (batch, 1, channels, height, width),
            device=history_latents.device,
            dtype=history_latents.dtype,
            generator=generator,
        )
    else:
        current = initial_noise.to(
            device=history_latents.device, dtype=history_latents.dtype
        ).clone()
        if current.shape != (batch, 1, channels, height, width):
            raise ValueError("initial_noise must be [B,1,C,Y,X]")
    starting_noise = current.clone()
    schedule = euler_tau_schedule(
        num_inference_steps,
        device=history_latents.device,
        dtype=history_latents.dtype,
    )

    for tau_current, tau_next in zip(schedule[:-1], schedule[1:]):
        model_input = torch.cat((history_latents, current), dim=1)
        tau = torch.zeros(
            (batch, history + 1),
            device=history_latents.device,
            dtype=history_latents.dtype,
        )
        tau[:, -1] = tau_current
        velocity = _model_velocity(
            model,
            checkpoint_type,
            model_input,
            tau,
            action_cond,
            action_valid_mask,
        )
        if velocity.shape != model_input.shape:
            raise RuntimeError("world model velocity shape differs from latent input")
        current = current + (tau_next - tau_current) * velocity[:, -1:]
        if not torch.isfinite(current).all():
            raise RuntimeError("non-finite latent during Euler integration")
    return current, starting_noise
