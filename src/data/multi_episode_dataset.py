from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data.temporal_sampler import TemporalSampler


# ============================================================
# Action statistics
# ============================================================


@dataclass
class ActionStats:
    mean: torch.Tensor
    std: torch.Tensor
    episode_ids: list[int]

    def save(
        self,
        path: str | Path,
    ) -> None:

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            {
                "mean": self.mean.cpu(),
                "std": self.std.cpu(),
                "episode_ids": (
                    list(self.episode_ids)
                ),
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "ActionStats":

        x = torch.load(
            path,
            map_location="cpu",
        )

        return cls(
            mean=x["mean"].float(),
            std=x["std"].float(),
            episode_ids=[
                int(i)
                for i in x["episode_ids"]
            ],
        )


# ============================================================
# Helpers
# ============================================================


def _load_manifest(
    manifest_path: str | Path,
) -> dict:

    manifest_path = Path(
        manifest_path
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            "Manifest not found: "
            f"{manifest_path}"
        )

    with open(
        manifest_path,
        "r",
        encoding="utf-8",
    ) as f:

        manifest = json.load(f)

    required = {
        "train_episode_ids",
        "val_episode_ids",
        "episodes",
    }

    missing = (
        required
        - set(manifest.keys())
    )

    if missing:
        raise RuntimeError(
            "Manifest missing keys: "
            f"{sorted(missing)}"
        )

    return manifest


def _episode_cache_map(
    manifest: dict,
) -> dict[int, Path]:

    result = {}

    for item in manifest["episodes"]:

        episode_index = int(
            item["episode_index"]
        )

        cache_file = Path(
            item["cache_file"]
        )

        result[
            episode_index
        ] = cache_file

    return result


def compute_train_action_stats(
    manifest_path: str | Path,
) -> ActionStats:
    """
    只使用 train episodes 中的原始 absolute actions
    计算全局 mean/std。

    注意：
        这里不是：
            每个 episode 先 normalize

        而是：
            concat 所有 train frames
            -> 一次性计算 mean/std
    """

    manifest = _load_manifest(
        manifest_path
    )

    cache_map = (
        _episode_cache_map(
            manifest
        )
    )

    train_ids = [
        int(x)
        for x in (
            manifest[
                "train_episode_ids"
            ]
        )
    ]

    all_actions = []

    for episode_id in train_ids:

        if (
            episode_id
            not in cache_map
        ):
            raise RuntimeError(
                "Missing train episode "
                f"{episode_id} "
                "from cache manifest."
            )

        cache = torch.load(
            cache_map[episode_id],
            map_location="cpu",
        )

        actions = (
            cache["actions"]
            .float()
        )

        if (
            actions.ndim != 2
            or actions.shape[-1] != 6
        ):
            raise RuntimeError(
                "Expected actions "
                "[N,6], got "
                f"{tuple(actions.shape)} "
                f"for episode "
                f"{episode_id}"
            )

        if not torch.isfinite(
            actions
        ).all():
            raise RuntimeError(
                "NaN/Inf actions in "
                f"episode {episode_id}"
            )

        all_actions.append(
            actions
        )

    actions = torch.cat(
        all_actions,
        dim=0,
    )

    mean = actions.mean(
        dim=0
    )

    std = actions.std(
        dim=0
    )

    std = std.clamp_min(
        1e-6
    )

    return ActionStats(
        mean=mean,
        std=std,
        episode_ids=train_ids,
    )


# ============================================================
# Multi-episode Dataset
# ============================================================


class MultiEpisodeCachedDataset(
    Dataset
):
    """
    多 episode latent dataset。

    正式 baseline 当前只使用：

        absolute commanded joint target

            c_t = a_t

    不使用：

        a_t - q_t

    因为未来 q_t 在真实 rollout 时不可提前获得。

    ----------------------------------------------------------
    split
    ----------------------------------------------------------

    split="train":
        只加载 manifest 中 train_episode_ids

    split="val":
        只加载 manifest 中 val_episode_ids

    ----------------------------------------------------------
    temporal window
    ----------------------------------------------------------

    每个 episode 独立构建 window，因此绝不会跨 episode。

    例如：

        n_frames = 10
        frame_skip = 4

    span:

        1 + (10 - 1) * 4
        = 37 raw frames

    一个 episode 长度 N 时：

        valid starts:
            0 ... N - 37

    ----------------------------------------------------------
    normalization
    ----------------------------------------------------------

    train / val 必须共用：

        train_action_stats

    val Dataset 禁止自行统计 val mean/std。
    """

    SUPPORTED_SPLITS = {
        "train",
        "val",
    }

    SUPPORTED_ACTION_MODES = {
        "sampled",
        "chunk",
    }

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        n_frames: int = 10,
        frame_skip: int = 4,
        normalize_actions: bool = True,
        action_mode: str = "sampled",
        action_stats: ActionStats | None = None,
    ) -> None:

        super().__init__()

        # ====================================================
        # Arguments
        # ====================================================

        if (
            split
            not in self.SUPPORTED_SPLITS
        ):
            raise ValueError(
                "split must be one of "
                f"{sorted(self.SUPPORTED_SPLITS)}, "
                f"got {split!r}"
            )

        if (
            action_mode
            not in self.SUPPORTED_ACTION_MODES
        ):
            raise ValueError(
                "action_mode must be one of "
                f"{sorted(self.SUPPORTED_ACTION_MODES)}, "
                f"got {action_mode!r}"
            )

        if n_frames <= 0:
            raise ValueError(
                "n_frames must be > 0"
            )

        if frame_skip <= 0:
            raise ValueError(
                "frame_skip must be > 0"
            )

        self.manifest_path = Path(
            manifest_path
        )

        self.split = split

        self.n_frames = int(
            n_frames
        )

        self.frame_skip = int(
            frame_skip
        )

        self.normalize_actions = bool(
            normalize_actions
        )

        self.action_mode = (
            action_mode
        )

        self.raw_action_dim = 6

        # ====================================================
        # Temporal sampler
        # ====================================================

        self.temporal_sampler = (
            TemporalSampler(
                n_frames=self.n_frames,
                frame_skip=self.frame_skip,
            )
        )

        # ====================================================
        # Manifest
        # ====================================================

        self.manifest = (
            _load_manifest(
                self.manifest_path
            )
        )

        self.cache_map = (
            _episode_cache_map(
                self.manifest
            )
        )

        if self.split == "train":

            self.episode_ids = [
                int(x)
                for x in self.manifest[
                    "train_episode_ids"
                ]
            ]

        else:

            self.episode_ids = [
                int(x)
                for x in self.manifest[
                    "val_episode_ids"
                ]
            ]

        if not self.episode_ids:
            raise RuntimeError(
                f"No episodes in split "
                f"{self.split}"
            )

        # ====================================================
        # Action statistics
        # ====================================================

        if self.normalize_actions:

            if action_stats is None:

                if self.split == "val":
                    raise ValueError(
                        "Validation dataset "
                        "must receive TRAIN "
                        "action_stats."
                    )

                action_stats = (
                    compute_train_action_stats(
                        self.manifest_path
                    )
                )

            self.action_stats = (
                action_stats
            )

            self.action_mean = (
                action_stats.mean
                .float()
            )

            self.action_std = (
                action_stats.std
                .float()
                .clamp_min(1e-6)
            )

            # 防止意外把 val stats 传进来。
            expected_train_ids = {
                int(x)
                for x in self.manifest[
                    "train_episode_ids"
                ]
            }

            stats_ids = set(
                int(x)
                for x in (
                    action_stats
                    .episode_ids
                )
            )

            if (
                stats_ids
                != expected_train_ids
            ):
                raise RuntimeError(
                    "Action stats were not "
                    "computed from exactly "
                    "the train episodes.\n"
                    f"expected={sorted(expected_train_ids)}\n"
                    f"stats={sorted(stats_ids)}"
                )

        else:

            self.action_stats = None

            self.action_mean = (
                torch.zeros(
                    self.raw_action_dim
                )
            )

            self.action_std = (
                torch.ones(
                    self.raw_action_dim
                )
            )

        # ====================================================
        # Condition dimension
        # ====================================================

        if (
            self.action_mode
            == "sampled"
        ):

            self.condition_action_dim = (
                self.raw_action_dim
            )

        else:

            self.condition_action_dim = (
                self.frame_skip
                * self.raw_action_dim
            )

        # ====================================================
        # Load selected episodes
        #
        # 当前只有 10 个 episode，
        # 全部放 CPU RAM 没问题。
        # ====================================================

        self.episodes = {}

        for episode_id in (
            self.episode_ids
        ):

            if (
                episode_id
                not in self.cache_map
            ):
                raise RuntimeError(
                    "Episode missing from "
                    "cache manifest: "
                    f"{episode_id}"
                )

            cache_path = (
                self.cache_map[
                    episode_id
                ]
            )

            if not cache_path.exists():
                raise FileNotFoundError(
                    f"Missing cache: "
                    f"{cache_path}"
                )

            cache = torch.load(
                cache_path,
                map_location="cpu",
            )

            latents = (
                cache["latents"]
                .float()
                .contiguous()
            )

            actions = (
                cache["actions"]
                .float()
                .contiguous()
            )

            states = (
                cache["states"]
                .float()
                .contiguous()
            )

            N = len(latents)

            if actions.shape != (
                N,
                6,
            ):
                raise RuntimeError(
                    "Bad actions shape "
                    f"for episode "
                    f"{episode_id}: "
                    f"{tuple(actions.shape)}"
                )

            if states.shape != (
                N,
                6,
            ):
                raise RuntimeError(
                    "Bad states shape "
                    f"for episode "
                    f"{episode_id}: "
                    f"{tuple(states.shape)}"
                )

            if (
                latents.ndim != 4
                or latents.shape[1:]
                != (16, 32, 32)
            ):
                raise RuntimeError(
                    "Bad latent shape "
                    f"for episode "
                    f"{episode_id}: "
                    f"{tuple(latents.shape)}"
                )

            timestamps = cache.get(
                "timestamps",
                None,
            )

            if timestamps is not None:
                timestamps = (
                    torch.as_tensor(
                        timestamps
                    )
                )

            frame_indices = cache.get(
                "frame_indices",
                None,
            )

            if frame_indices is None:
                frame_indices = (
                    torch.arange(
                        N,
                        dtype=torch.long,
                    )
                )
            else:
                frame_indices = (
                    torch.as_tensor(
                        frame_indices,
                        dtype=torch.long,
                    )
                )

            self.episodes[
                episode_id
            ] = {
                "latents": latents,
                "actions": actions,
                "states": states,
                "timestamps": (
                    timestamps
                ),
                "frame_indices": (
                    frame_indices
                ),
            }

        # ====================================================
        # Global window index
        #
        # dataset index
        #     ->
        # (episode_id, local_start)
        #
        # 这就是防止跨 episode 的核心。
        # ====================================================

        self.windows: list[
            tuple[int, int]
        ] = []

        self.windows_per_episode = {}

        for episode_id in (
            self.episode_ids
        ):

            N = len(
                self.episodes[
                    episode_id
                ]["latents"]
            )

            num_windows = (
                self.temporal_sampler
                .num_valid_windows(N)
            )

            if num_windows <= 0:
                raise RuntimeError(
                    f"Episode {episode_id} "
                    "is too short: "
                    f"N={N}, "
                    f"span="
                    f"{self.temporal_sampler.span}"
                )

            self.windows_per_episode[
                episode_id
            ] = num_windows

            for start in range(
                num_windows
            ):

                self.windows.append(
                    (
                        episode_id,
                        start,
                    )
                )

        if not self.windows:
            raise RuntimeError(
                "No temporal windows."
            )

    # ========================================================
    # Basic methods
    # ========================================================

    def __len__(
        self,
    ) -> int:

        return len(
            self.windows
        )

    def _normalize_actions(
        self,
        actions: torch.Tensor,
    ) -> torch.Tensor:

        return (
            actions
            - self.action_mean
        ) / self.action_std

    # ========================================================
    # Causal action chunk
    # ========================================================

    def _chunk_indices_for_frame(
        self,
        frame_idx: int,
    ) -> torch.Tensor:
        """
        stride=4:

        target frame 0:
            [0,0,0,0]

        target frame 4:
            [0,1,2,3]

        target frame 8:
            [4,5,6,7]
        """

        start = (
            frame_idx
            - self.frame_skip
        )

        end = frame_idx

        indices = [
            max(0, i)
            for i in range(
                start,
                end,
            )
        ]

        if (
            len(indices)
            != self.frame_skip
        ):
            raise RuntimeError(
                "Unexpected chunk "
                "length."
            )

        return torch.tensor(
            indices,
            dtype=torch.long,
        )

    def _make_action_chunks(
        self,
        actions: torch.Tensor,
        sampled_indices: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        actions:
            本 episode 原始 absolute actions
            [N,6]

        return:
            normalized chunks
            raw chunks

        shape:
            [T, frame_skip*6]
        """

        chunks_raw = []
        chunks_norm = []

        for frame_idx in (
            sampled_indices
            .tolist()
        ):

            chunk_idx = (
                self
                ._chunk_indices_for_frame(
                    frame_idx
                )
            )

            chunk_raw = (
                actions[
                    chunk_idx
                ]
            )

            if (
                self.normalize_actions
            ):

                chunk_norm = (
                    self
                    ._normalize_actions(
                        chunk_raw
                    )
                )

            else:

                chunk_norm = (
                    chunk_raw
                )

            chunks_raw.append(
                chunk_raw.reshape(-1)
            )

            chunks_norm.append(
                chunk_norm.reshape(-1)
            )

        return (
            torch.stack(
                chunks_norm,
                dim=0,
            ),
            torch.stack(
                chunks_raw,
                dim=0,
            ),
        )

    # ========================================================
    # Get item
    # ========================================================

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:

        if (
            index < 0
            or index >= len(self)
        ):
            raise IndexError(
                f"index={index}, "
                f"len={len(self)}"
            )

        episode_id, start = (
            self.windows[
                index
            ]
        )

        episode = (
            self.episodes[
                episode_id
            ]
        )

        N = len(
            episode["latents"]
        )

        window = (
            self.temporal_sampler
            .indices_from_start(
                start_idx=start,
                episode_length=N,
            )
        )

        indices = torch.tensor(
            window.indices,
            dtype=torch.long,
        )

        # ====================================================
        # Visual
        # ====================================================

        latents = (
            episode["latents"][
                indices
            ]
        )

        states = (
            episode["states"][
                indices
            ]
        )

        # ====================================================
        # Absolute commanded actions
        # ====================================================

        sampled_actions_raw = (
            episode["actions"][
                indices
            ]
        )

        if (
            self.action_mode
            == "sampled"
        ):

            actions_raw = (
                sampled_actions_raw
            )

            if (
                self.normalize_actions
            ):

                actions = (
                    self
                    ._normalize_actions(
                        actions_raw
                    )
                )

            else:

                actions = (
                    actions_raw.clone()
                )

        else:

            (
                actions,
                actions_raw,
            ) = (
                self
                ._make_action_chunks(
                    actions=(
                        episode[
                            "actions"
                        ]
                    ),
                    sampled_indices=(
                        indices
                    ),
                )
            )

        # ====================================================
        # Checks
        # ====================================================

        expected_action_shape = (
            self.n_frames,
            self.condition_action_dim,
        )

        if actions.shape != (
            expected_action_shape
        ):
            raise RuntimeError(
                "Unexpected action "
                "shape: "
                f"{tuple(actions.shape)} "
                f"!= "
                f"{expected_action_shape}"
            )

        if latents.shape != (
            self.n_frames,
            16,
            32,
            32,
        ):
            raise RuntimeError(
                "Unexpected latent "
                "shape: "
                f"{tuple(latents.shape)}"
            )

        result = {
            "latents": latents,

            # 模型实际输入
            "actions": actions,

            # 未 normalization
            "actions_raw": (
                actions_raw
            ),

            # sampled absolute command
            "sampled_actions_raw": (
                sampled_actions_raw
            ),

            "states": states,

            # episode 内部 cache row indices
            "indices": indices,

            "frame_indices": (
                episode[
                    "frame_indices"
                ][indices]
            ),

            "episode_idx": (
                torch.tensor(
                    episode_id,
                    dtype=torch.long,
                )
            ),

            "window_start": (
                torch.tensor(
                    start,
                    dtype=torch.long,
                )
            ),
        }

        timestamps = (
            episode[
                "timestamps"
            ]
        )

        if timestamps is not None:

            result[
                "timestamps"
            ] = (
                timestamps[
                    indices
                ]
            )

        return result