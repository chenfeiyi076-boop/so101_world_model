from __future__ import annotations

import torch

from src.data.multi_episode_dataset import (
    ActionStats,
    MultiEpisodeCachedDataset,
    compute_train_action_stats,
)


MANIFEST = (
    "data/latent_cache/so101_front/"
    "multi_episode_manifest.json"
)

STATS_PATH = (
    "data/multi_episode_pilot/"
    "action_stats_absolute.pt"
)


# ============================================================
# 1. Compute TRAIN stats only
# ============================================================

stats = compute_train_action_stats(
    MANIFEST
)

stats.save(
    STATS_PATH
)

print("=" * 80)
print("TRAIN ACTION STATS")
print("=" * 80)

print(
    "episodes:",
    stats.episode_ids,
)

print(
    "mean:",
    stats.mean,
)

print(
    "std :",
    stats.std,
)


# ============================================================
# 2. Construct train
# ============================================================

train_ds = (
    MultiEpisodeCachedDataset(
        manifest_path=MANIFEST,
        split="train",
        n_frames=10,
        frame_skip=4,
        normalize_actions=True,
        action_mode="sampled",
        action_stats=stats,
    )
)


# ============================================================
# 3. Construct val using EXACT SAME stats
# ============================================================

loaded_stats = (
    ActionStats.load(
        STATS_PATH
    )
)

val_ds = (
    MultiEpisodeCachedDataset(
        manifest_path=MANIFEST,
        split="val",
        n_frames=10,
        frame_skip=4,
        normalize_actions=True,
        action_mode="sampled",
        action_stats=loaded_stats,
    )
)


# ============================================================
# Basic information
# ============================================================

print()
print("=" * 80)
print("DATASET SUMMARY")
print("=" * 80)

print(
    "train episodes:",
    train_ds.episode_ids,
)

print(
    "val episodes:",
    val_ds.episode_ids,
)

print(
    "train windows:",
    len(train_ds),
)

print(
    "val windows:",
    len(val_ds),
)

print(
    "condition dim:",
    train_ds.condition_action_dim,
)

print()
print(
    "train windows per episode:"
)

for ep, n in (
    train_ds
    .windows_per_episode
    .items()
):
    print(
        f"  ep {ep:04d}: {n}"
    )

print()
print(
    "val windows per episode:"
)

for ep, n in (
    val_ds
    .windows_per_episode
    .items()
):
    print(
        f"  ep {ep:04d}: {n}"
    )


# ============================================================
# Check split
# ============================================================

assert set(
    train_ds.episode_ids
).isdisjoint(
    set(
        val_ds.episode_ids
    )
)


# ============================================================
# Check SAME normalization
# ============================================================

assert torch.equal(
    train_ds.action_mean,
    val_ds.action_mean,
)

assert torch.equal(
    train_ds.action_std,
    val_ds.action_std,
)


# ============================================================
# Inspect first samples
# ============================================================

train_x = train_ds[0]
val_x = val_ds[0]

print()
print("=" * 80)
print("SAMPLE SHAPES")
print("=" * 80)

print(
    "train episode:",
    train_x[
        "episode_idx"
    ].item(),
)

print(
    "train indices:",
    train_x["indices"],
)

print(
    "train latents:",
    tuple(
        train_x[
            "latents"
        ].shape
    ),
)

print(
    "train actions:",
    tuple(
        train_x[
            "actions"
        ].shape
    ),
)

print()

print(
    "val episode:",
    val_x[
        "episode_idx"
    ].item(),
)

print(
    "val indices:",
    val_x["indices"],
)

print(
    "val latents:",
    tuple(
        val_x[
            "latents"
        ].shape
    ),
)

print(
    "val actions:",
    tuple(
        val_x[
            "actions"
        ].shape
    ),
)


# ============================================================
# stride=4 expected indices
# ============================================================

assert torch.equal(
    train_x["indices"],
    torch.tensor(
        [
            0,
            4,
            8,
            12,
            16,
            20,
            24,
            28,
            32,
            36,
        ]
    ),
)


# ============================================================
# Verify normalization manually
# ============================================================

expected = (
    train_x[
        "actions_raw"
    ]
    - stats.mean
) / stats.std

assert torch.allclose(
    train_x["actions"],
    expected,
)


# ============================================================
# Critical: verify no window crosses episodes
# ============================================================

for dataset in [
    train_ds,
    val_ds,
]:

    for i in range(
        len(dataset)
    ):

        ep, start = (
            dataset.windows[i]
        )

        N = len(
            dataset.episodes[
                ep
            ]["latents"]
        )

        indices = (
            start
            + torch.arange(
                dataset.n_frames
            )
            * dataset.frame_skip
        )

        assert (
            indices[-1].item()
            < N
        )


print()
print(
    "NO CROSS-EPISODE WINDOWS PASSED"
)

print(
    "TRAIN-ONLY NORMALIZATION PASSED"
)

print(
    "MULTI-EPISODE DATASET PASSED"
)