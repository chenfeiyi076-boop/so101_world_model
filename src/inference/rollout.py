from __future__ import annotations

import copy
from typing import Any

import torch

from .action_adapter import adapt_actions
from .checkpoint_loader import LoadedWorldModel
from .flow_sampler import euler_sample_next


def _cache_rows(
    values: torch.Tensor,
    cache_frame_indices: torch.Tensor,
    requested: torch.Tensor,
) -> torch.Tensor:
    lookup = {int(raw): row for row, raw in enumerate(cache_frame_indices.tolist())}
    missing = sorted({int(raw) for raw in requested.tolist() if int(raw) not in lookup})
    if missing:
        raise IndexError(f"episode cache is missing raw frame indices {missing}")
    rows = torch.tensor([lookup[int(raw)] for raw in requested.tolist()], dtype=torch.long)
    return torch.as_tensor(values)[rows]


def _temporal_config(loaded: LoadedWorldModel) -> tuple[int, int, int]:
    temporal = loaded.resolved_config["temporal"]
    return (
        int(temporal["num_frames"]),
        int(temporal["num_history"]),
        int(temporal["frame_stride"]),
    )


def _action_metadata(config: dict[str, Any], checkpoint_type: str) -> tuple[str, str]:
    action = config["action"]
    if checkpoint_type == "causal_v2":
        return str(action["alignment"]), str(action["representation"])
    return str(action["alignment"]), str(action["representation"])


@torch.inference_mode()
def autoregressive_rollout(
    *,
    loaded: LoadedWorldModel,
    episode_cache: dict[str, Any],
    start_frame: int,
    rollout_frames: int,
    num_inference_steps: int = 10,
    seed: int = 0,
    include_reference: bool = True,
) -> dict[str, Any]:
    """Teacher-forced-action rollout that never feeds future reference latents."""

    if rollout_frames <= 0:
        raise ValueError("rollout_frames must be positive")
    max_frames, num_history, stride = _temporal_config(loaded)
    inference_slots = num_history + 1
    if inference_slots > max_frames:
        raise RuntimeError("H+1 inference context exceeds training num_frames")

    latents = torch.as_tensor(episode_cache["latents"])
    actions = torch.as_tensor(episode_cache["actions"], dtype=torch.float32)
    states_value = episode_cache.get("states")
    states = None if states_value is None else torch.as_tensor(states_value).float()
    cache_frame_indices = torch.as_tensor(
        episode_cache.get("frame_indices", torch.arange(len(latents))), dtype=torch.long
    )
    if len(actions) != len(latents) or len(cache_frame_indices) != len(latents):
        raise RuntimeError("cache latents/actions/frame_indices length mismatch")

    history_indices = start_frame + torch.arange(num_history, dtype=torch.long) * stride
    history_cpu = _cache_rows(latents, cache_frame_indices, history_indices).float()
    parameter = next(loaded.model.parameters())
    device = parameter.device
    dtype = parameter.dtype
    context_latents = [item.to(device=device, dtype=dtype) for item in history_cpu]
    context_indices = [int(value) for value in history_indices.tolist()]
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    generated = []
    initial_noises = []
    action_conditions = []
    raw_conditions = []
    action_masks = []
    action_indices_per_step = []
    generated_indices = []

    for rollout_index in range(rollout_frames):
        target_index = start_frame + (num_history + rollout_index) * stride
        slot_indices = torch.tensor(context_indices + [target_index], dtype=torch.long)
        adapted = adapt_actions(
            checkpoint_type=loaded.checkpoint_type,
            actions=actions,
            states=states,
            cache_frame_indices=cache_frame_indices,
            frame_indices=slot_indices,
            config=loaded.resolved_config,
            stats=loaded.action_stats,
        )
        history_batch = torch.stack(context_latents, dim=0).unsqueeze(0)
        action_batch = adapted.condition.to(device=device, dtype=dtype).unsqueeze(0)
        mask_batch = adapted.valid_mask.to(device=device).unsqueeze(0)
        next_latent, starting_noise = euler_sample_next(
            model=loaded.model,
            checkpoint_type=loaded.checkpoint_type,
            history_latents=history_batch,
            action_cond=action_batch,
            action_valid_mask=mask_batch,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        next_item = next_latent[0, 0]
        generated.append(next_item.cpu())
        initial_noises.append(starting_noise[0, 0].cpu())
        action_conditions.append(adapted.condition)
        raw_conditions.append(adapted.raw_condition)
        action_masks.append(adapted.valid_mask)
        action_indices_per_step.append(adapted.action_indices)
        generated_indices.append(target_index)

        context_latents = (context_latents + [next_item])[-num_history:]
        context_indices = (context_indices + [target_index])[-num_history:]

    generated_index_tensor = torch.tensor(generated_indices, dtype=torch.long)
    reference = None
    if include_reference:
        # Reference access deliberately happens only after generation is complete.
        reference = _cache_rows(
            latents, cache_frame_indices, generated_index_tensor
        ).float()
    alignment, representation = _action_metadata(
        loaded.resolved_config, loaded.checkpoint_type
    )
    checkpoint_version = loaded.checkpoint.get("checkpoint_version", 1)
    all_frame_indices = torch.cat((history_indices, generated_index_tensor))

    return {
        "generated_latents": torch.stack(generated),
        "history_latents": history_cpu,
        "reference_latents": reference,
        "frame_indices": all_frame_indices,
        "history_frame_indices": history_indices,
        "generated_frame_indices": generated_index_tensor,
        "action_indices": torch.stack(action_indices_per_step),
        "raw_actions": torch.stack(raw_conditions),
        "normalized_action_cond": torch.stack(action_conditions),
        "action_valid_mask": torch.stack(action_masks),
        "initial_noises": torch.stack(initial_noises),
        "checkpoint_path": loaded.checkpoint_path,
        "checkpoint_version": int(checkpoint_version),
        "checkpoint_type": loaded.checkpoint_type,
        "resolved_config": copy.deepcopy(loaded.resolved_config),
        "num_inference_steps": int(num_inference_steps),
        "seed": int(seed),
        "metadata": {
            "start_frame_semantics": "raw_episode_frame_index",
            "start_frame": int(start_frame),
            "rollout_frames": int(rollout_frames),
            "frame_stride": stride,
            "num_history": num_history,
            "inference_temporal_slots": inference_slots,
            "training_num_frames": max_frames,
            "action_alignment": alignment,
            "action_representation": representation,
            "normalization_source": loaded.action_stats.source,
            "normalization_method": loaded.action_stats.method,
            "reference_used_as_model_input": False,
        },
    }
