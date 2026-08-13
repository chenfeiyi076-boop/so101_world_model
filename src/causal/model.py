from __future__ import annotations

import torch

from src.models.dit import DiT


class CausalDiT(DiT):
    """Legacy DiT backbone with an explicit, post-embedding NULL mask."""

    def embed_actions(
        self,
        action_cond: torch.Tensor,
        action_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if action_cond.ndim != 3:
            raise ValueError("action_cond must be [B,T,A]")
        if action_valid_mask.shape != action_cond.shape[:2]:
            raise ValueError("action_valid_mask must be [B,T]")
        embedding = self.action_embedder(action_cond)
        return embedding * action_valid_mask.to(
            device=embedding.device, dtype=embedding.dtype
        ).unsqueeze(-1)

    def get_causal_condition(
        self,
        tau: torch.Tensor,
        action_cond: torch.Tensor,
        action_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        time_condition = self.timestep_embedder(tau)
        action_condition = self.embed_actions(action_cond, action_valid_mask)
        if time_condition.shape != action_condition.shape:
            raise RuntimeError("time/action condition shape mismatch")
        return time_condition + action_condition

    def forward(
        self,
        x: torch.Tensor,
        tau: torch.Tensor,
        action_cond: torch.Tensor,
        action_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("x must be [B,T,C,H,W]")
        batch, time, _, _, _ = x.shape
        if tau.shape != (batch, time):
            raise ValueError("tau must be [B,T]")
        if action_cond.shape != (batch, time, self.action_dim):
            raise ValueError(
                f"action_cond must be [{batch},{time},{self.action_dim}]"
            )
        if action_valid_mask.shape != (batch, time):
            raise ValueError("action_valid_mask must be [B,T]")

        x = self.patchify(x)
        condition = self.get_causal_condition(
            tau, action_cond, action_valid_mask
        )
        for block in self.blocks:
            x = block(x, condition)
        x = self.final_layer(x, condition)
        return self.unpatchify(x)
