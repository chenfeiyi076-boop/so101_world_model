from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.models.action_embedder import ActionEmbedder
from src.models.attention import (
    SpatialAttention,
    CausalTemporalAttention,
)


# ============================================================
# Timestep embedding
# ============================================================

class TimestepEmbedder(nn.Module):
    """
    将 flow/diffusion timestep tau 映射到 hidden dimension。

    输入:
        tau: [B, T]

    输出:
        [B, T, hidden_size]

    流匹配中 tau 通常是 [0, 1] 之间的连续值。
    """

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size

        self.mlp = nn.Sequential(
            nn.Linear(
                frequency_embedding_size,
                hidden_size,
            ),
            nn.SiLU(),
            nn.Linear(
                hidden_size,
                hidden_size,
            ),
        )

    @staticmethod
    def sinusoidal_embedding(
        t: torch.Tensor,
        dim: int,
        max_period: int = 10_000,
    ) -> torch.Tensor:
        """
        t:
            [N]

        return:
            [N, dim]
        """

        half = dim // 2

        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(
                0,
                half,
                dtype=torch.float32,
                device=t.device,
            )
            / half
        )

        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)

        embedding = torch.cat(
            [
                torch.cos(args),
                torch.sin(args),
            ],
            dim=-1,
        )

        if dim % 2:
            embedding = torch.cat(
                [
                    embedding,
                    torch.zeros_like(
                        embedding[:, :1]
                    ),
                ],
                dim=-1,
            )

        return embedding

    def forward(
        self,
        tau: torch.Tensor,
    ) -> torch.Tensor:

        if tau.ndim != 2:
            raise ValueError(
                "tau must have shape [B, T], "
                f"got {tuple(tau.shape)}"
            )

        B, T = tau.shape

        tau_flat = tau.reshape(B * T)

        freq = self.sinusoidal_embedding(
            tau_flat,
            self.frequency_embedding_size,
        )

        embedding = self.mlp(freq)

        return embedding.reshape(
            B,
            T,
            self.hidden_size,
        )


# ============================================================
# SwiGLU
# ============================================================

class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network.

    第二篇当前实现使用约 2/3 的内部宽度，
    以保持和普通 4D FFN 类似的参数量。
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()

        nominal_hidden = int(
            dim * mlp_ratio
        )

        hidden_dim = int(
            2 * nominal_hidden / 3
        )

        self.w12 = nn.Linear(
            dim,
            2 * hidden_dim,
        )

        self.w3 = nn.Linear(
            hidden_dim,
            dim,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x1, x2 = self.w12(x).chunk(
            2,
            dim=-1,
        )

        return self.w3(
            torch.nn.functional.silu(x1) * x2
        )


# ============================================================
# AdaLN DiT sub-block
# ============================================================

class ConditionedDiTBlock(nn.Module):
    """
    一个带 AdaLN-Zero 条件调制的 DiT 子 block。

    每一个子 block 包含:

        RMSNorm
        -> Attention
        -> gated residual

        RMSNorm
        -> SwiGLU
        -> gated residual

    conditioning:
        c: [B, T, D]

    会广播到所有 spatial tokens。
    """

    def __init__(
        self,
        dim: int,
        attention: nn.Module,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()

        self.dim = dim

        self.norm1 = nn.RMSNorm(
            dim,
            elementwise_affine=False,
            eps=1e-6,
        )

        self.norm2 = nn.RMSNorm(
            dim,
            elementwise_affine=False,
            eps=1e-6,
        )

        self.attn = attention

        self.ffn = SwiGLU(
            dim=dim,
            mlp_ratio=mlp_ratio,
        )

        # ----------------------------------------------------
        # conditioning c 产生:
        #
        # shift_attn
        # scale_attn
        # gate_attn
        # shift_ffn
        # scale_ffn
        # gate_ffn
        #
        # 共 6D
        # ----------------------------------------------------

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                dim,
                6 * dim,
            ),
        )

    @staticmethod
    def modulate(
        x: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        x:
            [B,T,H,W,D]

        shift/scale:
            [B,T,1,1,D]
        """

        return (
            x * (1.0 + scale)
            + shift
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 5:
            raise ValueError(
                "x must have shape [B,T,H,W,D]"
            )

        if c.ndim != 3:
            raise ValueError(
                "c must have shape [B,T,D]"
            )

        if x.shape[0] != c.shape[0]:
            raise ValueError(
                "Batch dimension mismatch "
                "between x and c"
            )

        if x.shape[1] != c.shape[1]:
            raise ValueError(
                "Temporal dimension mismatch "
                "between x and c"
            )

        modulation = self.adaLN_modulation(c)

        (
            shift_attn,
            scale_attn,
            gate_attn,
            shift_ffn,
            scale_ffn,
            gate_ffn,
        ) = modulation.chunk(
            6,
            dim=-1,
        )

        # [B,T,D]
        # ->
        # [B,T,1,1,D]

        shift_attn = shift_attn[:, :, None, None, :]
        scale_attn = scale_attn[:, :, None, None, :]
        gate_attn = gate_attn[:, :, None, None, :]

        shift_ffn = shift_ffn[:, :, None, None, :]
        scale_ffn = scale_ffn[:, :, None, None, :]
        gate_ffn = gate_ffn[:, :, None, None, :]

        # ====================================================
        # Attention branch
        # ====================================================

        h = self.modulate(
            self.norm1(x),
            shift_attn,
            scale_attn,
        )

        h = self.attn(h)

        x = x + gate_attn * h

        # ====================================================
        # Feed-forward branch
        # ====================================================

        h = self.modulate(
            self.norm2(x),
            shift_ffn,
            scale_ffn,
        )

        h = self.ffn(h)

        x = x + gate_ffn * h

        return x


# ============================================================
# Factorized spatial-temporal block
# ============================================================

class FactorizedDiTBlock(nn.Module):
    """
    第二篇论文的 factorized spatial-temporal block。

    一个完整 block:

        Spatial DiTBlock
             ↓
        Temporal DiTBlock

    注意:
        两个子 block 都各自拥有自己的 FFN。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_qk_norm: bool = True,
    ) -> None:
        super().__init__()

        spatial_attention = SpatialAttention(
            dim=dim,
            num_heads=num_heads,
            use_qk_norm=use_qk_norm,
        )

        temporal_attention = CausalTemporalAttention(
            dim=dim,
            num_heads=num_heads,
            use_qk_norm=use_qk_norm,
        )

        self.spatial_block = ConditionedDiTBlock(
            dim=dim,
            attention=spatial_attention,
            mlp_ratio=mlp_ratio,
        )

        self.temporal_block = ConditionedDiTBlock(
            dim=dim,
            attention=temporal_attention,
            mlp_ratio=mlp_ratio,
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:

        x = self.spatial_block(
            x,
            c,
        )

        x = self.temporal_block(
            x,
            c,
        )

        return x


# ============================================================
# Final layer
# ============================================================

class FinalLayer(nn.Module):
    """
    将 DiT hidden token 重新预测成 latent patch。

    hidden:
        [B,T,H',W',D]

    output:
        [B,T,H',W',p*p*C]
    """

    def __init__(
        self,
        dim: int,
        patch_size: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.norm = nn.RMSNorm(
            dim,
            elementwise_affine=False,
            eps=1e-6,
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                dim,
                2 * dim,
            ),
        )

        self.linear = nn.Linear(
            dim,
            patch_size
            * patch_size
            * out_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:

        shift, scale = self.adaLN_modulation(
            c
        ).chunk(
            2,
            dim=-1,
        )

        shift = shift[:, :, None, None, :]
        scale = scale[:, :, None, None, :]

        x = self.norm(x)

        x = (
            x * (1.0 + scale)
            + shift
        )

        return self.linear(x)


# ============================================================
# Complete DiT
# ============================================================

class DiT(nn.Module):
    """
    Action-conditioned factorized spatial-temporal DiT.

    我们项目统一使用 channel-first latent:

        x:
            [B,T,C,H,W]

        tau:
            [B,T]

        actions:
            [B,T,action_dim]

    output:
        [B,T,C,H,W]

    第一版:
        SD3-VAE:
            C = 16

        SO-101:
            action_dim = 6

        patch_size = 2
    """

    def __init__(
        self,
        in_channels: int = 16,
        patch_size: int = 2,
        hidden_size: int = 64,
        depth: int = 2,
        num_heads: int = 2,
        action_dim: int = 6,
        mlp_ratio: float = 4.0,
        use_qk_norm: bool = True,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(
                "in_channels must be > 0"
            )

        if patch_size <= 0:
            raise ValueError(
                "patch_size must be > 0"
            )

        if hidden_size % num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible "
                "by num_heads"
            )

        self.in_channels = int(
            in_channels
        )

        self.patch_size = int(
            patch_size
        )

        self.hidden_size = int(
            hidden_size
        )

        self.depth = int(
            depth
        )

        self.action_dim = int(
            action_dim
        )

        # ====================================================
        # Patch embedding
        #
        # [C,H,W]
        # ->
        # [D,H/p,W/p]
        #
        # 对 SD3:
        #
        # [16,32,32]
        # ->
        # [D,16,16]
        # ====================================================

        self.patch_embed = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # ====================================================
        # Conditioning
        # ====================================================

        self.timestep_embedder = TimestepEmbedder(
            hidden_size=self.hidden_size,
        )

        self.action_embedder = ActionEmbedder(
            action_dim=self.action_dim,
            hidden_size=self.hidden_size,
        )

        # ====================================================
        # DiT backbone
        # ====================================================

        self.blocks = nn.ModuleList(
            [
                FactorizedDiTBlock(
                    dim=self.hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_qk_norm=use_qk_norm,
                )
                for _ in range(self.depth)
            ]
        )

        # ====================================================
        # Output head
        # ====================================================

        self.final_layer = FinalLayer(
            dim=self.hidden_size,
            patch_size=self.patch_size,
            out_channels=self.in_channels,
        )

        self.initialize_weights()

    # ========================================================
    # Initialization
    # ========================================================

    def initialize_weights(
        self,
    ) -> None:

        def basic_init(
            module: nn.Module,
        ) -> None:

            if isinstance(
                module,
                nn.Linear,
            ):
                nn.init.xavier_uniform_(
                    module.weight
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

        self.apply(
            basic_init
        )

        # ----------------------------------------------------
        # Patch Conv 按 Linear 的方式初始化
        # ----------------------------------------------------

        weight = self.patch_embed.weight.data

        nn.init.xavier_uniform_(
            weight.reshape(
                weight.shape[0],
                -1,
            )
        )

        if self.patch_embed.bias is not None:
            nn.init.zeros_(
                self.patch_embed.bias
            )

        # ----------------------------------------------------
        # Timestep MLP initialization
        # ----------------------------------------------------

        nn.init.normal_(
            self.timestep_embedder.mlp[0].weight,
            std=0.02,
        )

        nn.init.normal_(
            self.timestep_embedder.mlp[2].weight,
            std=0.02,
        )

        # ----------------------------------------------------
        # AdaLN-Zero
        #
        # 初始时所有 gated residual ≈ 0
        # ----------------------------------------------------

        for block in self.blocks:

            nn.init.zeros_(
                block.spatial_block
                .adaLN_modulation[-1]
                .weight
            )

            nn.init.zeros_(
                block.spatial_block
                .adaLN_modulation[-1]
                .bias
            )

            nn.init.zeros_(
                block.temporal_block
                .adaLN_modulation[-1]
                .weight
            )

            nn.init.zeros_(
                block.temporal_block
                .adaLN_modulation[-1]
                .bias
            )

        # ----------------------------------------------------
        # Final layer zero initialization
        # ----------------------------------------------------

        nn.init.zeros_(
            self.final_layer
            .adaLN_modulation[-1]
            .weight
        )

        nn.init.zeros_(
            self.final_layer
            .adaLN_modulation[-1]
            .bias
        )

        nn.init.zeros_(
            self.final_layer.linear.weight
        )

        nn.init.zeros_(
            self.final_layer.linear.bias
        )

    # ========================================================
    # Patchify
    # ========================================================

    def patchify(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        x:
            [B,T,C,H,W]

        return:
            [B,T,H/p,W/p,D]
        """

        if x.ndim != 5:
            raise ValueError(
                "x must have shape [B,T,C,H,W], "
                f"got {tuple(x.shape)}"
            )

        B, T, C, H, W = x.shape

        if C != self.in_channels:
            raise ValueError(
                f"Expected C={self.in_channels}, "
                f"got C={C}"
            )

        if H % self.patch_size != 0:
            raise ValueError(
                "Height must be divisible "
                "by patch_size"
            )

        if W % self.patch_size != 0:
            raise ValueError(
                "Width must be divisible "
                "by patch_size"
            )

        x = x.reshape(
            B * T,
            C,
            H,
            W,
        )

        x = self.patch_embed(x)

        _, D, Hp, Wp = x.shape

        x = x.reshape(
            B,
            T,
            D,
            Hp,
            Wp,
        )

        x = x.permute(
            0,
            1,
            3,
            4,
            2,
        ).contiguous()

        return x

    # ========================================================
    # Unpatchify
    # ========================================================

    def unpatchify(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        x:
            [B,T,Hp,Wp,p*p*C]

        return:
            [B,T,C,H,W]
        """

        if x.ndim != 5:
            raise ValueError(
                "x must have shape "
                "[B,T,Hp,Wp,p*p*C]"
            )

        B, T, Hp, Wp, P = x.shape

        p = self.patch_size
        C = self.in_channels

        expected = (
            p * p * C
        )

        if P != expected:
            raise ValueError(
                f"Expected final patch dim={expected}, "
                f"got {P}"
            )

        # ---------------------------------------------
        # [B,T,Hp,Wp,p,p,C]
        # ---------------------------------------------

        x = x.reshape(
            B,
            T,
            Hp,
            Wp,
            p,
            p,
            C,
        )

        # ---------------------------------------------
        # 拼回:
        #
        # Hp × p
        # Wp × p
        # ---------------------------------------------

        x = x.permute(
            0,  # B
            1,  # T
            6,  # C
            2,  # Hp
            4,  # p_h
            3,  # Wp
            5,  # p_w
        ).contiguous()

        x = x.reshape(
            B,
            T,
            C,
            Hp * p,
            Wp * p,
        )

        return x

    # ========================================================
    # Conditioning
    # ========================================================

    def get_condition(
        self,
        tau: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        c_t = timestep_embedding(tau_t)
              + action_embedding(a_t)

        tau:
            [B,T]

        actions:
            [B,T,action_dim]

        return:
            [B,T,D]
        """

        time_condition = (
            self.timestep_embedder(
                tau
            )
        )

        action_condition = (
            self.action_embedder(
                actions
            )
        )

        if (
            time_condition.shape
            != action_condition.shape
        ):
            raise RuntimeError(
                "time/action condition shape mismatch"
            )

        return (
            time_condition
            + action_condition
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        x: torch.Tensor,
        tau: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        x:
            [B,T,C,H,W]

        tau:
            [B,T]

        actions:
            [B,T,action_dim]

        output:
            [B,T,C,H,W]
        """

        B, T, C, H, W = x.shape

        if tau.shape != (
            B,
            T,
        ):
            raise ValueError(
                "tau must have shape "
                f"[{B},{T}], "
                f"got {tuple(tau.shape)}"
            )

        if actions.shape != (
            B,
            T,
            self.action_dim,
        ):
            raise ValueError(
                "actions must have shape "
                f"[{B},{T},{self.action_dim}], "
                f"got {tuple(actions.shape)}"
            )

        # ====================================================
        # VAE latent -> patch tokens
        # ====================================================

        x = self.patchify(x)

        # ====================================================
        # timestep + action
        # ====================================================

        c = self.get_condition(
            tau,
            actions,
        )

        # ====================================================
        # Spatial + Temporal blocks
        # ====================================================

        for block in self.blocks:
            x = block(
                x,
                c,
            )

        # ====================================================
        # hidden D
        # ->
        # p*p*C
        # ====================================================

        x = self.final_layer(
            x,
            c,
        )

        # ====================================================
        # patch grid -> full latent
        # ====================================================

        x = self.unpatchify(x)

        return x