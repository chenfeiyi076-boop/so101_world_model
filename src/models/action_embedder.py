from __future__ import annotations

import torch
import torch.nn as nn


class ActionEmbedder(nn.Module):
    """
    将机器人动作向量映射到 DiT hidden dimension。

    输入:
        actions: [..., action_dim]

    常见情况:
        [B, T, action_dim]

    输出:
        [..., hidden_size]

    常见情况:
        [B, T, hidden_size]

    第一版采用最简单的线性投影：

        R^{action_dim} -> R^{hidden_size}

    对 SO-101:
        action_dim = 6

    Debug DiT:
        hidden_size = 64

    正式 DiT-S:
        hidden_size = 384
    """

    def __init__(
        self,
        action_dim: int,
        hidden_size: int,
    ) -> None:
        super().__init__()

        if action_dim <= 0:
            raise ValueError(
                f"action_dim must be > 0, got {action_dim}"
            )

        if hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be > 0, got {hidden_size}"
            )

        self.action_dim = int(action_dim)
        self.hidden_size = int(hidden_size)

        self.proj = nn.Linear(
            self.action_dim,
            self.hidden_size,
        )

    def forward(
        self,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        参数
        ----
        actions:
            shape [..., action_dim]

            通常:
                [B, T, action_dim]

        返回
        ----
        action_embeddings:
            shape [..., hidden_size]

            通常:
                [B, T, hidden_size]
        """

        if actions.ndim < 2:
            raise ValueError(
                "actions must have at least 2 dimensions, "
                f"got shape={tuple(actions.shape)}"
            )

        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                "Unexpected action dimension: "
                f"expected last dimension={self.action_dim}, "
                f"got shape={tuple(actions.shape)}"
            )

        action_embeddings = self.proj(actions)

        return action_embeddings

    def __repr__(self) -> str:
        return (
            f"ActionEmbedder("
            f"action_dim={self.action_dim}, "
            f"hidden_size={self.hidden_size}"
            f")"
        )