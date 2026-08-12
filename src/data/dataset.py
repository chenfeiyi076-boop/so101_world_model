from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from src.data.temporal_sampler import TemporalSampler


@dataclass
class WorldModelSample:
    """
    单个 world-model 时间窗口。

    latents:
        [T, C, H, W]

    actions:
        [T, action_dim]

    indices:
        原始 episode 中对应的时间索引。
    """

    latents: torch.Tensor
    actions: torch.Tensor
    indices: torch.Tensor


# ============================================================
# Dummy Dataset
# ============================================================


class DummyWorldModelDataset(Dataset):
    """
    用随机 latent/action 模拟真实 SO-101 数据。

    当前目的：
    1. 验证 TemporalSampler
    2. 验证 Dataset 输出格式
    3. 验证 DataLoader batch shape
    4. 为后面的 DiT / Flow Matching 提供稳定接口

    目前不读取任何真实图片或 VAE latent。
    """

    def __init__(
        self,
        num_episodes: int = 4,
        episode_length: int = 100,
        n_frames: int = 4,
        frame_skip: int = 1,
        action_dim: int = 6,
        latent_channels: int = 16,
        latent_height: int = 32,
        latent_width: int = 32,
        windows_per_episode: int = 20,
        seed: int = 0,
    ) -> None:

        super().__init__()

        if num_episodes <= 0:
            raise ValueError(
                "num_episodes must be > 0"
            )

        if episode_length <= 0:
            raise ValueError(
                "episode_length must be > 0"
            )

        if n_frames <= 0:
            raise ValueError(
                "n_frames must be > 0"
            )

        if frame_skip <= 0:
            raise ValueError(
                "frame_skip must be > 0"
            )

        if windows_per_episode <= 0:
            raise ValueError(
                "windows_per_episode must be > 0"
            )

        self.num_episodes = num_episodes
        self.episode_length = episode_length

        self.n_frames = n_frames
        self.frame_skip = frame_skip

        self.action_dim = action_dim

        self.latent_channels = latent_channels
        self.latent_height = latent_height
        self.latent_width = latent_width

        self.windows_per_episode = (
            windows_per_episode
        )

        self.temporal_sampler = (
            TemporalSampler(
                n_frames=n_frames,
                frame_skip=frame_skip,
            )
        )

        if (
            episode_length
            < self.temporal_sampler.span
        ):
            raise ValueError(
                "episode_length is too short: "
                f"episode_length={episode_length}, "
                f"required_span="
                f"{self.temporal_sampler.span}"
            )

        # ----------------------------------------------------
        # 固定随机数据，保证 debug 可复现。
        # ----------------------------------------------------

        generator = torch.Generator()
        generator.manual_seed(seed)

        self.episodes = []

        for _ in range(
            num_episodes
        ):

            latents = torch.randn(
                episode_length,
                latent_channels,
                latent_height,
                latent_width,
                generator=generator,
            )

            actions = torch.randn(
                episode_length,
                action_dim,
                generator=generator,
            )

            self.episodes.append(
                {
                    "latents": latents,
                    "actions": actions,
                }
            )

    def __len__(
        self,
    ) -> int:

        return (
            self.num_episodes
            * self.windows_per_episode
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:

        if (
            index < 0
            or index >= len(self)
        ):
            raise IndexError(
                f"index={index} out of range "
                f"for dataset length={len(self)}"
            )

        episode_idx = (
            index
            // self.windows_per_episode
        )

        window_idx = (
            index
            % self.windows_per_episode
        )

        episode = self.episodes[
            episode_idx
        ]

        num_valid_windows = (
            self.temporal_sampler
            .num_valid_windows(
                self.episode_length
            )
        )

        start_idx = (
            window_idx
            % num_valid_windows
        )

        window = (
            self.temporal_sampler
            .indices_from_start(
                start_idx=start_idx,
                episode_length=(
                    self.episode_length
                ),
            )
        )

        indices = torch.tensor(
            window.indices,
            dtype=torch.long,
        )

        latents = (
            episode["latents"][
                indices
            ]
        )

        actions = (
            episode["actions"][
                indices
            ]
        )

        expected_latent_shape = (
            self.n_frames,
            self.latent_channels,
            self.latent_height,
            self.latent_width,
        )

        expected_action_shape = (
            self.n_frames,
            self.action_dim,
        )

        if (
            latents.shape
            != expected_latent_shape
        ):
            raise RuntimeError(
                "Unexpected latent shape: "
                f"{tuple(latents.shape)} "
                f"!= "
                f"{expected_latent_shape}"
            )

        if (
            actions.shape
            != expected_action_shape
        ):
            raise RuntimeError(
                "Unexpected action shape: "
                f"{tuple(actions.shape)} "
                f"!= "
                f"{expected_action_shape}"
            )

        return {
            "latents": latents,
            "actions": actions,
            "indices": indices,
            "episode_idx": torch.tensor(
                episode_idx,
                dtype=torch.long,
            ),
        }


# ============================================================
# Cached Real Dataset
# ============================================================


class CachedLatentDataset(Dataset):
    """
    从离线 VAE latent cache 中读取 temporal windows。

    Cache 格式：

    {
        "latents":       [N, 16, 32, 32],
        "actions":       [N, 6],
        "states":        [N, 6],
        "timestamps":    [N],
        "frame_indices": [N],
        "episode_index": int,
    }

    ----------------------------------------------------------
    Action representation
    ----------------------------------------------------------

    absolute:

        control_t = action_t

    delta:

        control_t = action_t - state_t

    即：

        commanded target position
            -
        current joint position

    ----------------------------------------------------------
    Action mode
    ----------------------------------------------------------

    sampled:

        frame_skip = 4

        latent indices:
            [0, 4, 8, 12, ...]

        condition:
            [c0, c4, c8, c12, ...]

        shape:
            [T, 6]

    chunk:

        frame_skip = 4

        causal chunk:

            slot 0:
                [c0, c0, c0, c0]

            target frame 4:
                [c0, c1, c2, c3]

            target frame 8:
                [c4, c5, c6, c7]

            target frame 12:
                [c8, c9, c10, c11]

        flatten 后：

            [T, 4, 6]
                ->
            [T, 24]

    ----------------------------------------------------------
    Normalization
    ----------------------------------------------------------

    normalization 始终针对当前 action representation。

    absolute:
        使用 absolute action 的 mean/std

    delta:
        使用 (action-state) 的 mean/std

    chunk:
        先逐 timestep 对 6D control normalize，
        再 flatten。

    当前 single-episode pilot 使用本 episode 统计量。

    正式多 episode 实验时必须改为：

        train split statistics
            ->
        train / val / test 共用
    """

    SUPPORTED_ACTION_MODES = {
        "sampled",
        "chunk",
    }

    SUPPORTED_ACTION_REPRESENTATIONS = {
        "absolute",
        "delta",
    }

    def __init__(
        self,
        cache_path: str,
        n_frames: int = 10,
        frame_skip: int = 1,
        normalize_actions: bool = True,
        action_mode: str = "sampled",
        action_representation: str = "absolute",
    ) -> None:

        super().__init__()

        # ====================================================
        # Validate arguments
        # ====================================================

        if n_frames <= 0:
            raise ValueError(
                "n_frames must be > 0"
            )

        if frame_skip <= 0:
            raise ValueError(
                "frame_skip must be > 0"
            )

        if (
            action_mode
            not in self.SUPPORTED_ACTION_MODES
        ):
            raise ValueError(
                f"Unknown action_mode="
                f"{action_mode!r}. "
                f"Expected one of "
                f"{sorted(self.SUPPORTED_ACTION_MODES)}"
            )

        if (
            action_representation
            not in (self.SUPPORTED_ACTION_REPRESENTATIONS
            )
        ):
            valid_representations = sorted(
                self.SUPPORTED_ACTION_REPRESENTATIONS
            )
            raise ValueError(
                "Unknown action_representation={!r}"
                "Expected one of {}".format(
                    action_representation,
                    valid_representations,
                )  
            )

        self.cache_path = cache_path

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

        self.action_representation = (
            action_representation
        )

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
        # Load cache
        # ====================================================

        cache = torch.load(
            cache_path,
            map_location="cpu",
        )

        required_keys = {
            "latents",
            "actions",
            "states",
        }

        missing_keys = (
            required_keys
            - set(cache.keys())
        )

        if missing_keys:
            raise RuntimeError(
                "Cache missing keys: "
                f"{sorted(missing_keys)}"
            )

        # ----------------------------------------------------
        # Visual latent
        # ----------------------------------------------------

        self.latents = (
            cache["latents"]
            .float()
            .contiguous()
        )

        # ----------------------------------------------------
        # 原始 absolute commanded target
        #
        # 永远保留，不覆盖。
        # ----------------------------------------------------

        self.absolute_actions = (
            cache["actions"]
            .float()
            .contiguous()
        )

        # 为兼容之前代码，
        # self.actions 仍指向原始 absolute actions。
        self.actions = (
            self.absolute_actions
        )

        # ----------------------------------------------------
        # Current joint states
        # ----------------------------------------------------

        self.states = (
            cache["states"]
            .float()
            .contiguous()
        )

        # ----------------------------------------------------
        # Delta / target-error control
        #
        # delta_t = action_t - state_t
        # ----------------------------------------------------

        self.delta_actions = (
            self.absolute_actions
            - self.states
        )

        # ====================================================
        # Select actual control representation
        # ====================================================

        if (
            self.action_representation
            == "absolute"
        ):

            self.control_actions = (
                self.absolute_actions
            )

        elif (
            self.action_representation
            == "delta"
        ):

            self.control_actions = (
                self.delta_actions
            )

        else:
            raise AssertionError(
                "Unreachable action representation."
            )

        # ====================================================
        # Optional metadata
        # ====================================================

        self.timestamps = cache.get(
            "timestamps",
            None,
        )

        self.episode_index = int(
            cache.get(
                "episode_index",
                0,
            )
        )

        # ====================================================
        # Basic checks
        # ====================================================

        N = len(
            self.latents
        )

        if N <= 0:
            raise RuntimeError(
                "Cache contains no frames."
            )

        if (
            self.latents.ndim
            != 4
        ):
            raise RuntimeError(
                "latents must have shape "
                "[N,C,H,W], got "
                f"{tuple(self.latents.shape)}"
            )

        if (
            self.absolute_actions.ndim
            != 2
        ):
            action_shape = tuple(self.absolute_actions.shape)
            raise RuntimeError(
                "actions must have shape "
                "[N,action_dim], got ".format(
                    action_shape
                )
                
            )

        if (
            self.states.ndim
            != 2
        ):
            raise RuntimeError(
                "states must have shape "
                "[N,state_dim], got "
                f"{tuple(self.states.shape)}"
            )

        if (
            len(self.absolute_actions)
            != N
        ):
            raise RuntimeError(
                "latents/actions length "
                "mismatch"
            )

        if (
            len(self.states)
            != N
        ):
            raise RuntimeError(
                "latents/states length "
                "mismatch"
            )

        # ----------------------------------------------------
        # SO-101 pilot: 6D
        # ----------------------------------------------------

        if (
            self.absolute_actions.shape[-1]
            != 6
        ):
            raise RuntimeError(
                "Expected 6-D actions, got "
                f"{self.absolute_actions.shape[-1]}"
            )

        if (
            self.states.shape[-1]
            != 6
        ):
            raise RuntimeError(
                "Expected 6-D states, got "
                f"{self.states.shape[-1]}"
            )

        self.raw_action_dim = int(
            self.absolute_actions.shape[-1]
        )

        # ====================================================
        # Finite checks
        # ====================================================

        if not torch.isfinite(
            self.latents
        ).all():
            raise RuntimeError(
                "NaN/Inf found in latents."
            )

        if not torch.isfinite(
            self.absolute_actions
        ).all():
            raise RuntimeError(
                "NaN/Inf found in actions."
            )

        if not torch.isfinite(
            self.states
        ).all():
            raise RuntimeError(
                "NaN/Inf found in states."
            )

        if not torch.isfinite(
            self.delta_actions
        ).all():
            raise RuntimeError(
                "NaN/Inf found in delta actions."
            )

        # ====================================================
        # Timestamp
        # ====================================================

        if (
            self.timestamps
            is not None
        ):

            self.timestamps = (
                torch.as_tensor(
                    self.timestamps
                )
            )

            if (
                len(self.timestamps)
                != N
            ):
                raise RuntimeError(
                    "latents/timestamps "
                    "length mismatch"
                )

        # ====================================================
        # Frame indices
        # ====================================================

        self.frame_indices = (
            cache.get(
                "frame_indices",
                torch.arange(
                    N,
                    dtype=torch.long,
                ),
            )
        )

        self.frame_indices = (
            torch.as_tensor(
                self.frame_indices,
                dtype=torch.long,
            )
        )

        if (
            len(self.frame_indices)
            != N
        ):
            raise RuntimeError(
                "latents/frame_indices "
                "length mismatch"
            )

        # ====================================================
        # Normalization statistics
        #
        # 非常关键：
        #
        # absolute 模式：
        #   对 absolute action 统计
        #
        # delta 模式：
        #   对 action-state 统计
        # ====================================================

        self.action_mean = (
            self.control_actions.mean(
                dim=0
            )
        )

        self.action_std = (
            self.control_actions.std(
                dim=0
            )
        )

        self.action_std = (
            torch.clamp(
                self.action_std,
                min=1e-6,
            )
        )

        # ====================================================
        # DiT action condition dimension
        # ====================================================

        if (
            self.action_mode
            == "sampled"
        ):

            self.condition_action_dim = (
                self.raw_action_dim
            )

        elif (
            self.action_mode
            == "chunk"
        ):

            self.condition_action_dim = (
                self.raw_action_dim
                * self.frame_skip
            )

        else:
            raise AssertionError(
                "Unreachable action mode."
            )

        # ====================================================
        # Number of valid windows
        # ====================================================

        self.num_windows = (
            self.temporal_sampler
            .num_valid_windows(
                N
            )
        )

        if (
            self.num_windows
            <= 0
        ):
            raise RuntimeError(
                f"Episode too short: "
                f"N={N}, "
                f"T={self.n_frames}, "
                f"skip={self.frame_skip}"
            )

    # ========================================================
    # Length
    # ========================================================

    def __len__(
        self,
    ) -> int:

        return self.num_windows

    # ========================================================
    # Normalize
    # ========================================================

    def _normalize_control(
        self,
        control: torch.Tensor,
    ) -> torch.Tensor:
        """
        对当前 action representation 做逐关节 normalization。

        control 最后一维必须为 6。

        支持：

            [T, 6]

        或：

            [chunk_size, 6]
        """

        if (
            control.shape[-1]
            != self.raw_action_dim
        ):
            raise RuntimeError(
                "Expected control last dim "
                f"{self.raw_action_dim}, "
                f"got {control.shape[-1]}"
            )

        return (
            (
                control
                - self.action_mean
            )
            / self.action_std
        )

    # ========================================================
    # Causal chunk indices
    # ========================================================

    def _chunk_indices_for_frame(
        self,
        frame_idx: int,
    ) -> torch.Tensor:
        """
        为 sampled target frame 构造因果 action chunk。

        frame_skip = 4 时：

        target z_0:
            [0, 0, 0, 0]

        target z_4:
            [0, 1, 2, 3]

        target z_8:
            [4, 5, 6, 7]

        target z_12:
            [8, 9, 10, 11]

        即：

            z_t
              -- controls during interval -->
            z_{t+4}

        episode 左边界使用第 0 个 control padding。
        """

        start = (
            frame_idx
            - self.frame_skip
        )

        end = frame_idx

        chunk_indices = [
            max(0, i)
            for i in range(
                start,
                end,
            )
        ]

        if (
            len(chunk_indices)
            != self.frame_skip
        ):
            raise RuntimeError(
                "Unexpected chunk length: "
                f"{len(chunk_indices)} "
                f"!= {self.frame_skip}"
            )

        return torch.tensor(
            chunk_indices,
            dtype=torch.long,
        )

    # ========================================================
    # Make chunks
    # ========================================================

    def _make_action_chunks(
        self,
        indices: torch.Tensor,
        normalize: bool,
    ) -> torch.Tensor:
        """
        根据 sampled latent indices 构造 causal control chunk。

        注意：

        这里使用的是：

            self.control_actions

        因此：

            absolute 模式 -> chunk absolute actions

            delta 模式 -> chunk (action-state)

        输入：
            indices [T]

        输出：
            [T, frame_skip * 6]
        """

        chunks = []

        for frame_idx in (
            indices.tolist()
        ):

            chunk_indices = (
                self
                ._chunk_indices_for_frame(
                    frame_idx
                )
            )

            # -----------------------------------------------
            # 关键：
            # 使用当前 representation。
            #
            # absolute:
            #     a_i
            #
            # delta:
            #     a_i - q_i
            # -----------------------------------------------

            chunk = (
                self.control_actions[
                    chunk_indices
                ]
            )

            # [frame_skip, 6]

            if normalize:

                chunk = (
                    self
                    ._normalize_control(
                        chunk
                    )
                )

            # [frame_skip, 6]
            #       ->
            # [frame_skip * 6]

            chunk = (
                chunk.reshape(-1)
            )

            chunks.append(
                chunk
            )

        actions = torch.stack(
            chunks,
            dim=0,
        )

        expected_shape = (
            self.n_frames,
            self.frame_skip
            * self.raw_action_dim,
        )

        if (
            actions.shape
            != expected_shape
        ):
            raise RuntimeError(
                "Unexpected chunk action "
                "shape: "
                f"{tuple(actions.shape)} "
                f"!= "
                f"{expected_shape}"
            )

        return actions

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
                f"index={index} out of range "
                f"for dataset length={len(self)}"
            )

        # ====================================================
        # Temporal sampling
        # ====================================================

        window = (
            self.temporal_sampler
            .indices_from_start(
                start_idx=index,
                episode_length=(
                    len(self.latents)
                ),
            )
        )

        indices = torch.tensor(
            window.indices,
            dtype=torch.long,
        )

        # ====================================================
        # Visual / state
        # ====================================================

        latents = (
            self.latents[
                indices
            ]
        )

        states = (
            self.states[
                indices
            ]
        )

        # ====================================================
        # Debug representations at sampled frames
        # ====================================================

        sampled_absolute_actions = (
            self.absolute_actions[
                indices
            ]
        )

        sampled_delta_actions = (
            self.delta_actions[
                indices
            ]
        )

        sampled_control_raw = (
            self.control_actions[
                indices
            ]
        )

        # ====================================================
        # Actual condition for DiT
        # ====================================================

        if (
            self.action_mode
            == "sampled"
        ):

            # -----------------------------------------------
            # [T, 6]
            # -----------------------------------------------

            actions_raw = (
                sampled_control_raw
            )

            if (
                self.normalize_actions
            ):

                actions = (
                    self
                    ._normalize_control(
                        actions_raw
                    )
                )

            else:

                actions = (
                    actions_raw.clone()
                )

        elif (
            self.action_mode
            == "chunk"
        ):

            # -----------------------------------------------
            # [T, frame_skip*6]
            # -----------------------------------------------

            actions_raw = (
                self
                ._make_action_chunks(
                    indices=indices,
                    normalize=False,
                )
            )

            if (
                self.normalize_actions
            ):

                actions = (
                    self
                    ._make_action_chunks(
                        indices=indices,
                        normalize=True,
                    )
                )

            else:

                actions = (
                    actions_raw.clone()
                )

        else:
            raise AssertionError(
                "Unreachable action mode."
            )

        # ====================================================
        # Shape checks
        # ====================================================

        if (
            latents.shape[0]
            != self.n_frames
        ):
            raise RuntimeError(
                "Unexpected latent temporal "
                "length: "
                f"{latents.shape[0]}"
            )

        expected_action_shape = (
            self.n_frames,
            self.condition_action_dim,
        )

        if (
            actions.shape
            != expected_action_shape
        ):
            raise RuntimeError(
                "Unexpected action shape: "
                f"{tuple(actions.shape)} "
                f"!= "
                f"{expected_action_shape}"
            )

        if (
            actions_raw.shape
            != expected_action_shape
        ):
            raise RuntimeError(
                "Unexpected raw action shape: "
                f"{tuple(actions_raw.shape)} "
                f"!= "
                f"{expected_action_shape}"
            )

        # ====================================================
        # Finite checks
        # ====================================================

        if not torch.isfinite(
            latents
        ).all():
            raise RuntimeError(
                "NaN/Inf found in latents."
            )

        if not torch.isfinite(
            actions
        ).all():
            raise RuntimeError(
                "NaN/Inf found in actions."
            )

        if not torch.isfinite(
            actions_raw
        ).all():
            raise RuntimeError(
                "NaN/Inf found in raw actions."
            )

        # ====================================================
        # Result
        # ====================================================

        result = {
            # -----------------------------------------------
            # World-model inputs
            # -----------------------------------------------

            "latents": latents,

            # DiT 实际收到的 control condition。
            # 已经根据 action_mode 变成 [T,6] 或 [T,K*6]。
            # normalize_actions=True 时已经 normalization。
            "actions": actions,

            # 与 actions 同 shape，
            # 但没有 normalization。
            "actions_raw": actions_raw,

            # -----------------------------------------------
            # Debug / analysis
            # -----------------------------------------------

            # sampled timestep 的 absolute target:
            # [T,6]
            "absolute_actions": (
                sampled_absolute_actions
            ),

            # sampled timestep 的 target error:
            # [T,6]
            "delta_actions": (
                sampled_delta_actions
            ),

            # 当前 representation 在 sampled timestep 上的
            # 未归一化版本：
            # [T,6]
            "sampled_control_raw": (
                sampled_control_raw
            ),

            # sampled current state:
            # [T,6]
            "states": states,

            # cache row index
            "indices": indices,

            # 原始 frame index
            "frame_indices": (
                self.frame_indices[
                    indices
                ]
            ),

            "episode_idx": (
                torch.tensor(
                    self.episode_index,
                    dtype=torch.long,
                )
            ),
        }

        if (
            self.timestamps
            is not None
        ):

            result[
                "timestamps"
            ] = (
                self.timestamps[
                    indices
                ]
            )

        return result
