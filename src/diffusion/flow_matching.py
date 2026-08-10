from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class FlowMatchingBatch:
    """
    OT Flow Matching 构造后的训练 batch。

    noisy_latents:
        [B,T,C,H,W]
        实际送给 DiT 的输入。

    tau:
        [B,T]
        每个 temporal slot 的 flow timestep。

    target_velocity:
        [B,T,C,H,W]
        FM velocity target:
            epsilon - z

    loss_mask:
        [B,T]
        True 表示该 temporal slot 参与训练 loss。

    noise:
        [B,T,C,H,W]
        采样的 Gaussian noise。
    """

    noisy_latents: torch.Tensor
    tau: torch.Tensor
    target_velocity: torch.Tensor
    loss_mask: torch.Tensor
    noise: torch.Tensor


def prepare_flow_matching_batch(
    latents: torch.Tensor,
    num_history: int,
    history_noise_std: float = 0.0,
) -> FlowMatchingBatch:
    """
    构造 action-conditioned world model 的 OT Flow Matching 输入。

    参数
    ----------
    latents:
        clean latent:
            [B,T,C,H,W]

    num_history:
        history frame 数量 H。

        例如:
            T=10
            H=2

        则:
            t=0,1      history
            t=2,...,9  future

    history_noise_std:
        可选的 history 小 Gaussian augmentation。

        第一版建议:
            0.0

        后续如果希望更贴近第二篇实验，可以设为小正数。

    返回
    ----------
    FlowMatchingBatch
    """

    if latents.ndim != 5:
        raise ValueError(
            "latents must have shape [B,T,C,H,W], "
            f"got {tuple(latents.shape)}"
        )

    B, T, C, H, W = latents.shape

    if num_history < 0:
        raise ValueError(
            f"num_history must be >= 0, got {num_history}"
        )

    if num_history >= T:
        raise ValueError(
            "num_history must be smaller than T, "
            f"got num_history={num_history}, T={T}"
        )

    if history_noise_std < 0:
        raise ValueError(
            "history_noise_std must be >= 0, "
            f"got {history_noise_std}"
        )

    # ========================================================
    # 1. Gaussian endpoint epsilon
    # ========================================================

    noise = torch.randn_like(
        latents
    )

    # ========================================================
    # 2. 每个 future temporal slot 独立采样 tau
    #
    # history:
    #     tau = 0
    #
    # future:
    #     tau ~ Uniform(0,1)
    # ========================================================

    tau = torch.zeros(
        B,
        T,
        device=latents.device,
        dtype=latents.dtype,
    )

    tau[:, num_history:] = torch.rand(
        B,
        T - num_history,
        device=latents.device,
        dtype=latents.dtype,
    )

    # [B,T]
    # ->
    # [B,T,1,1,1]

    tau_broadcast = tau[
        :,
        :,
        None,
        None,
        None,
    ]

    # ========================================================
    # 3. OT linear interpolation
    #
    # z_tau = (1 - tau) z + tau epsilon
    # ========================================================

    noisy_latents = (
        (1.0 - tau_broadcast) * latents
        + tau_broadcast * noise
    )

    # ========================================================
    # 4. History 保持 clean
    #
    # num_history=2:
    #
    # z0 z1 | z2 ...
    # clean  | noisy
    # ========================================================

    if num_history > 0:

        noisy_latents[:, :num_history] = (
            latents[:, :num_history]
        )

        # 可选：
        # 第二篇有 history 小 Gaussian augmentation。
        # V0 默认关闭。
        if history_noise_std > 0:

            history_noise = torch.randn_like(
                noisy_latents[:, :num_history]
            )

            noisy_latents[:, :num_history] += (
                history_noise_std
                * history_noise
            )

    # ========================================================
    # 5. Flow velocity target
    #
    # u = epsilon - z
    # ========================================================

    target_velocity = (
        noise - latents
    )

    # ========================================================
    # 6. Future-only loss mask
    # ========================================================

    loss_mask = torch.zeros(
        B,
        T,
        device=latents.device,
        dtype=torch.bool,
    )

    loss_mask[:, num_history:] = True

    return FlowMatchingBatch(
        noisy_latents=noisy_latents,
        tau=tau,
        target_velocity=target_velocity,
        loss_mask=loss_mask,
        noise=noise,
    )


def flow_matching_loss(
    prediction: torch.Tensor,
    target_velocity: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Future-only Flow Matching MSE。

    prediction:
        [B,T,C,H,W]

    target_velocity:
        [B,T,C,H,W]

    loss_mask:
        [B,T]

    返回:
        scalar loss
    """

    if prediction.shape != target_velocity.shape:
        raise ValueError(
            "prediction and target_velocity must have "
            "the same shape, "
            f"got {tuple(prediction.shape)} and "
            f"{tuple(target_velocity.shape)}"
        )

    if prediction.ndim != 5:
        raise ValueError(
            "prediction must have shape [B,T,C,H,W]"
        )

    B, T, _, _, _ = prediction.shape

    if loss_mask.shape != (
        B,
        T,
    ):
        raise ValueError(
            "loss_mask must have shape "
            f"[{B},{T}], "
            f"got {tuple(loss_mask.shape)}"
        )

    if not loss_mask.any():
        raise ValueError(
            "loss_mask contains no valid future frames"
        )

    # --------------------------------------------------------
    # 每个 temporal slot 单独求像素/latent 维 MSE
    #
    # [B,T,C,H,W]
    # ->
    # [B,T]
    # --------------------------------------------------------

    squared_error = (
        prediction - target_velocity
    ).pow(2)

    per_frame_loss = squared_error.mean(
        dim=(2, 3, 4)
    )

    # --------------------------------------------------------
    # 只选择 future
    # --------------------------------------------------------

    loss = per_frame_loss[
        loss_mask
    ].mean()

    return loss