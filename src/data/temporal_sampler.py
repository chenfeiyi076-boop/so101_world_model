from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class TemporalWindow:
    """
    一个从单个 episode 中采样得到的时间窗口。

    例如：
        n_frames = 4
        frame_skip = 1
        start_idx = 3

    则：
        indices = (3, 4, 5, 6)
    """

    start_idx: int
    indices: tuple[int, ...]


class TemporalSampler:
    """
    从单个 episode 中采样固定长度的时间序列。

    参数
    ----
    n_frames:
        最终送入模型的 temporal slots 数量 T。

    frame_skip:
        相邻模型时间位置在原始 episode 中的采样间隔。

        frame_skip = 1:
            [0, 1, 2, 3, ...]

        frame_skip = 2:
            [0, 2, 4, 6, ...]

    注意
    ----
    当前模块只负责“时间索引”。

    它不负责：
    - 图像读取
    - action 读取
    - state 读取
    - action chunk 构造
    - VAE latent 读取

    这些逻辑后面放在 Dataset 中。
    """

    def __init__(
        self,
        n_frames: int,
        frame_skip: int = 1,
    ) -> None:

        if n_frames <= 0:
            raise ValueError(
                f"n_frames must be > 0, got {n_frames}"
            )

        if frame_skip <= 0:
            raise ValueError(
                f"frame_skip must be > 0, got {frame_skip}"
            )

        self.n_frames = int(n_frames)
        self.frame_skip = int(frame_skip)

    @property
    def span(self) -> int:
        """
        一个采样窗口在原始 episode 中覆盖多少个 timestep。

        例如：

        n_frames = 4
        frame_skip = 1

        indices:
            [0, 1, 2, 3]

        span = 4


        n_frames = 4
        frame_skip = 2

        indices:
            [0, 2, 4, 6]

        span = 7
        """

        return 1 + (self.n_frames - 1) * self.frame_skip

    def num_valid_windows(
        self,
        episode_length: int,
    ) -> int:
        """
        计算一个 episode 中共有多少个合法窗口。
        """

        if episode_length < 0:
            raise ValueError(
                f"episode_length must be >= 0, got {episode_length}"
            )

        if episode_length < self.span:
            return 0

        return episode_length - self.span + 1

    def indices_from_start(
        self,
        start_idx: int,
        episode_length: int,
    ) -> TemporalWindow:
        """
        给定起始位置，生成一个确定性的 temporal window。
        """

        if start_idx < 0:
            raise ValueError(
                f"start_idx must be >= 0, got {start_idx}"
            )

        indices = tuple(
            start_idx + i * self.frame_skip
            for i in range(self.n_frames)
        )

        last_idx = indices[-1]

        if last_idx >= episode_length:
            raise ValueError(
                "Temporal window exceeds episode boundary: "
                f"start_idx={start_idx}, "
                f"indices={indices}, "
                f"episode_length={episode_length}"
            )

        return TemporalWindow(
            start_idx=start_idx,
            indices=indices,
        )

    def sample(
        self,
        episode_length: int,
        rng: random.Random | None = None,
    ) -> TemporalWindow:
        """
        从一个 episode 中随机采样一个合法窗口。

        参数
        ----
        episode_length:
            当前 episode 的 timestep 总数。

        rng:
            可选的 random.Random 实例。
            测试时可以指定 seed，保证结果可复现。
        """

        n_windows = self.num_valid_windows(
            episode_length
        )

        if n_windows == 0:
            raise ValueError(
                "Episode is too short for temporal sampling: "
                f"episode_length={episode_length}, "
                f"n_frames={self.n_frames}, "
                f"frame_skip={self.frame_skip}, "
                f"required_span={self.span}"
            )

        if rng is None:
            rng = random

        max_start = n_windows - 1

        start_idx = rng.randint(
            0,
            max_start,
        )

        return self.indices_from_start(
            start_idx=start_idx,
            episode_length=episode_length,
        )

    def __repr__(self) -> str:
        return (
            f"TemporalSampler("
            f"n_frames={self.n_frames}, "
            f"frame_skip={self.frame_skip}, "
            f"span={self.span}"
            f")"
        )