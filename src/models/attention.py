from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Rotary Position Embedding
# ============================================================

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    将最后一维按 pair 做 90 度旋转。

    [..., x1, x2, x3, x4]
    ->
    [..., -x2, x1, -x4, x3]
    """
    original_shape = x.shape

    x = x.reshape(*original_shape[:-1], -1, 2)

    x1 = x[..., 0]
    x2 = x[..., 1]

    x = torch.stack(
        (-x2, x1),
        dim=-1,
    )

    return x.flatten(-2)


def _rope_nd(
    shape: Sequence[int],
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    base: float = 10_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    构造 N-D Rotary Position Embedding。

    temporal:
        shape = (T,)

    spatial:
        shape = (H, W)

    返回:
        cos, sin

    shape:
        [*shape, dim / 2]

    后续会 repeat_interleave 到完整 head_dim。
    """

    ndim = len(shape)

    if ndim <= 0:
        raise ValueError("shape must contain at least one dimension")

    # 每个轴需要 cos/sin pair，因此 head_dim 必须可整除 2 * ndim。
    if dim % (2 * ndim) != 0:
        raise ValueError(
            f"RoPE head dimension must be divisible by {2 * ndim}, "
            f"got dim={dim}, shape={tuple(shape)}"
        )

    dim_per_axis = dim // ndim
    half_dim_per_axis = dim_per_axis // 2

    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(
                half_dim_per_axis,
                device=device,
                dtype=dtype,
            )
            / half_dim_per_axis
        )
    )

    coords = [
        torch.arange(
            size,
            device=device,
            dtype=dtype,
        )
        for size in shape
    ]

    mesh = torch.meshgrid(
        *coords,
        indexing="ij",
    )

    cos_parts = []
    sin_parts = []

    for position in mesh:
        theta = position.unsqueeze(-1) * inv_freq

        cos_parts.append(torch.cos(theta))
        sin_parts.append(torch.sin(theta))

    cos = torch.cat(cos_parts, dim=-1)
    sin = torch.cat(sin_parts, dim=-1)

    return cos, sin


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    sequence_shape: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对 Q/K 使用 N-D RoPE。

    q, k:

        [batch_like, heads, *sequence_shape, head_dim]
    """

    head_dim = q.shape[-1]

    cos, sin = _rope_nd(
        shape=sequence_shape,
        dim=head_dim,
        device=q.device,
        dtype=q.dtype,
    )

    # cos/sin:
    #   [..., head_dim / 2]
    #
    # repeat 后:
    #   [..., head_dim]

    cos = torch.repeat_interleave(
        cos,
        repeats=2,
        dim=-1,
    )

    sin = torch.repeat_interleave(
        sin,
        repeats=2,
        dim=-1,
    )

    # 自动利用 broadcasting 对 batch/head 维展开。
    while cos.ndim < q.ndim:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin

    return q, k


# ============================================================
# Shared attention implementation
# ============================================================

class _FactorizedAttention(nn.Module):
    """
    Spatial / Temporal attention 的共同底层实现。

    输入统一为:

        x: [B, T, H, W, D]

    mode="spatial":
        每个 frame 内对 H*W 个 token 做 attention。

    mode="temporal":
        每个空间位置跨 T 做 causal attention。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mode: str,
        use_qk_norm: bool = True,
    ) -> None:
        super().__init__()

        if dim <= 0:
            raise ValueError(
                f"dim must be > 0, got {dim}"
            )

        if num_heads <= 0:
            raise ValueError(
                f"num_heads must be > 0, got {num_heads}"
            )

        if dim % num_heads != 0:
            raise ValueError(
                f"dim={dim} must be divisible by "
                f"num_heads={num_heads}"
            )

        if mode not in {"spatial", "temporal"}:
            raise ValueError(
                f"mode must be 'spatial' or 'temporal', got {mode}"
            )

        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.mode = mode
        self.use_qk_norm = bool(use_qk_norm)

        # 第二篇当前源码 / WorldGym 都采用一次投影得到 Q K V。
        self.qkv_proj = nn.Linear(
            self.dim,
            self.dim * 3,
            bias=False,
        )

        self.out_proj = nn.Linear(
            self.dim,
            self.dim,
        )

        # 第二篇当前源码使用 per-head RMSNorm 做 QK normalization。
        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(
                self.head_dim,
                elementwise_affine=False,
            )

            self.k_norm = nn.RMSNorm(
                self.head_dim,
                elementwise_affine=False,
            )

    def _check_input(
        self,
        x: torch.Tensor,
    ) -> None:

        if x.ndim != 5:
            raise ValueError(
                "Attention expects x with shape "
                "[B, T, H, W, D], "
                f"got {tuple(x.shape)}"
            )

        if x.shape[-1] != self.dim:
            raise ValueError(
                f"Expected hidden dim={self.dim}, "
                f"got x.shape[-1]={x.shape[-1]}"
            )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        self._check_input(x)

        B, T, H, W, D = x.shape

        # ====================================================
        # 1. Factorize sequence
        # ====================================================

        if self.mode == "spatial":

            # [B,T,H,W,D]
            #
            # ->
            #
            # [B*T,H,W,D]
            #
            # 每张图独立做 spatial attention。

            x_seq = x.reshape(
                B * T,
                H,
                W,
                D,
            )

            sequence_shape = (
                H,
                W,
            )

            is_causal = False

        else:

            # [B,T,H,W,D]
            #
            # ->
            #
            # [B,H,W,T,D]
            #
            # ->
            #
            # [B*H*W,T,D]
            #
            # 同一空间位置跨时间 attention。

            x_seq = (
                x.permute(
                    0,
                    2,
                    3,
                    1,
                    4,
                )
                .contiguous()
                .reshape(
                    B * H * W,
                    T,
                    D,
                )
            )

            sequence_shape = (
                T,
            )

            is_causal = True

        # ====================================================
        # 2. QKV projection
        # ====================================================

        q, k, v = self.qkv_proj(
            x_seq
        ).chunk(
            3,
            dim=-1,
        )

        # spatial:
        #
        # [BT,H,W,D]
        # ->
        # [BT,heads,H,W,head_dim]
        #
        # temporal:
        #
        # [BHW,T,D]
        # ->
        # [BHW,heads,T,head_dim]

        if self.mode == "spatial":

            q = q.reshape(
                B * T,
                H,
                W,
                self.num_heads,
                self.head_dim,
            ).permute(
                0,
                3,
                1,
                2,
                4,
            )

            k = k.reshape(
                B * T,
                H,
                W,
                self.num_heads,
                self.head_dim,
            ).permute(
                0,
                3,
                1,
                2,
                4,
            )

            v = v.reshape(
                B * T,
                H,
                W,
                self.num_heads,
                self.head_dim,
            ).permute(
                0,
                3,
                1,
                2,
                4,
            )

        else:

            q = q.reshape(
                B * H * W,
                T,
                self.num_heads,
                self.head_dim,
            ).permute(
                0,
                2,
                1,
                3,
            )

            k = k.reshape(
                B * H * W,
                T,
                self.num_heads,
                self.head_dim,
            ).permute(
                0,
                2,
                1,
                3,
            )

            v = v.reshape(
                B * H * W,
                T,
                self.num_heads,
                self.head_dim,
            ).permute(
                0,
                2,
                1,
                3,
            )

        # ====================================================
        # 3. QK normalization
        # ====================================================

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # ====================================================
        # 4. Rotary Position Embedding
        # ====================================================

        q, k = _apply_rope(
            q,
            k,
            sequence_shape,
        )

        # ====================================================
        # 5. Flatten spatial sequence
        # ====================================================

        if self.mode == "spatial":

            q = q.flatten(
                start_dim=2,
                end_dim=3,
            )

            k = k.flatten(
                start_dim=2,
                end_dim=3,
            )

            v = v.flatten(
                start_dim=2,
                end_dim=3,
            )

            # [BT,heads,H*W,head_dim]

        # temporal 已经是：
        #
        # [BHW,heads,T,head_dim]

        # ====================================================
        # 6. Scaled Dot Product Attention
        # ====================================================

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=is_causal,
        )

        # ====================================================
        # 7. Merge heads
        # ====================================================

        if self.mode == "spatial":

            # [BT,heads,H*W,head_dim]
            #
            # ->
            #
            # [BT,H*W,D]

            out = (
                out.permute(
                    0,
                    2,
                    1,
                    3,
                )
                .contiguous()
                .reshape(
                    B * T,
                    H * W,
                    D,
                )
            )

        else:

            # [BHW,heads,T,head_dim]
            #
            # ->
            #
            # [BHW,T,D]

            out = (
                out.permute(
                    0,
                    2,
                    1,
                    3,
                )
                .contiguous()
                .reshape(
                    B * H * W,
                    T,
                    D,
                )
            )

        out = self.out_proj(out)

        # ====================================================
        # 8. Restore [B,T,H,W,D]
        # ====================================================

        if self.mode == "spatial":

            out = out.reshape(
                B,
                T,
                H,
                W,
                D,
            )

        else:

            out = (
                out.reshape(
                    B,
                    H,
                    W,
                    T,
                    D,
                )
                .permute(
                    0,
                    3,
                    1,
                    2,
                    4,
                )
                .contiguous()
            )

        return out


# ============================================================
# Public modules
# ============================================================

class SpatialAttention(_FactorizedAttention):
    """
    每一帧内部的空间 self-attention。

    输入/输出:
        [B, T, H, W, D]

    不使用 causal mask。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        use_qk_norm: bool = True,
    ) -> None:

        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mode="spatial",
            use_qk_norm=use_qk_norm,
        )


class CausalTemporalAttention(_FactorizedAttention):
    """
    每一个 spatial token 独立地沿时间方向做 self-attention。

    输入/输出:
        [B, T, H, W, D]

    causal constraint:

        output[t] 只能依赖 input[0:t+1]
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        use_qk_norm: bool = True,
    ) -> None:

        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mode="temporal",
            use_qk_norm=use_qk_norm,
        )