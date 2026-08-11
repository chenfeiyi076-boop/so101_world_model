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
            raise ValueError("num_episodes must be > 0")

        if episode_length <= 0:
            raise ValueError("episode_length must be > 0")

        if windows_per_episode <= 0:
            raise ValueError("windows_per_episode must be > 0")

        self.num_episodes = num_episodes
        self.episode_length = episode_length
        self.n_frames = n_frames
        self.frame_skip = frame_skip
        self.action_dim = action_dim

        self.latent_channels = latent_channels
        self.latent_height = latent_height
        self.latent_width = latent_width

        self.windows_per_episode = windows_per_episode

        self.temporal_sampler = TemporalSampler(
            n_frames=n_frames,
            frame_skip=frame_skip,
        )

        if episode_length < self.temporal_sampler.span:
            raise ValueError(
                "episode_length is too short: "
                f"episode_length={episode_length}, "
                f"required_span={self.temporal_sampler.span}"
            )

        # -----------------------------
        # 为了 CPU debug 保证数据确定性，
        # 在初始化时直接创建固定的假 episode。
        # -----------------------------
        generator = torch.Generator()
        generator.manual_seed(seed)

        self.episodes = []

        for episode_idx in range(num_episodes):

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

    def __len__(self) -> int:
        """
        每个 episode 暂时提供固定数量的 temporal windows。
        """

        return self.num_episodes * self.windows_per_episode

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:

        if index < 0 or index >= len(self):
            raise IndexError(
                f"index={index} out of range for dataset "
                f"with length {len(self)}"
            )

        # 当前采用确定性的 episode 映射。
        episode_idx = index // self.windows_per_episode

        window_idx = index % self.windows_per_episode

        episode = self.episodes[episode_idx]

        num_valid_windows = self.temporal_sampler.num_valid_windows(
            self.episode_length
        )

        # 把 dataset index 确定性映射到一个合法 start。
        # 后面真实 Dataset 可以换成随机 window sampling。
        start_idx = window_idx % num_valid_windows

        window = self.temporal_sampler.indices_from_start(
            start_idx=start_idx,
            episode_length=self.episode_length,
        )

        indices = torch.tensor(
            window.indices,
            dtype=torch.long,
        )

        latents = episode["latents"][indices]
        actions = episode["actions"][indices]

        # -----------------------------
        # 强制检查 shape。
        # Debug 阶段宁可早点报错。
        # -----------------------------
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

        if latents.shape != expected_latent_shape:
            raise RuntimeError(
                f"Unexpected latent shape: "
                f"{tuple(latents.shape)} "
                f"!= {expected_latent_shape}"
            )

        if actions.shape != expected_action_shape:
            raise RuntimeError(
                f"Unexpected action shape: "
                f"{tuple(actions.shape)} "
                f"!= {expected_action_shape}"
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