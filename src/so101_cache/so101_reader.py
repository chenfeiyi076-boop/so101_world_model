from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import av
import numpy as np
import pandas as pd
import pyarrow.dataset as pyarrow_dataset
import torch

from .vae_cache import CAMERA_KEY, FPS, RAW_ACTION_DIM, validate_temporal_axis


@dataclass(frozen=True)
class EpisodeTabularData:
    episode_index: int
    actions: torch.Tensor
    states: torch.Tensor
    timestamps: torch.Tensor
    frame_indices: torch.Tensor

    @property
    def num_frames(self) -> int:
        return int(self.actions.shape[0])


@dataclass(frozen=True)
class EpisodeVideoSegment:
    episode_index: int
    path: Path
    from_timestamp: float
    to_timestamp: float


def _find_metadata_column(
    row: pd.Series,
    camera: str,
    suffix: str,
) -> object | None:
    matches = [
        column
        for column in row.index
        if camera in str(column) and str(column).endswith(suffix)
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        return None
    raise RuntimeError(
        f"multiple metadata columns match camera={camera!r}, suffix={suffix!r}: {matches}"
    )


def _stack_vector_column(series: pd.Series, name: str) -> torch.Tensor:
    values = np.stack(
        [np.asarray(value, dtype=np.float32).reshape(-1) for value in series],
        axis=0,
    )
    result = torch.from_numpy(values).to(dtype=torch.float32).contiguous()
    if result.ndim != 2 or result.shape[1] != RAW_ACTION_DIM:
        raise RuntimeError(
            f"{name} must have shape [N,{RAW_ACTION_DIM}], got {tuple(result.shape)}"
        )
    if not torch.isfinite(result).all():
        raise RuntimeError(f"{name} contains NaN or Inf")
    return result


class SO101LeRobotV3Reader:
    """Read raw SO101 episode rows and front frames from LeRobot v3 storage.

    A video file is a shared shard, not an episode.  Episode metadata selects the
    shard with chunk/file indices and the episode interval with PTS timestamps.
    """

    def __init__(self, dataset_root: str | Path, camera: str = CAMERA_KEY) -> None:
        self.dataset_root = Path(dataset_root)
        self.camera = str(camera)
        info_path = self.dataset_root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"LeRobot info.json not found: {info_path}")
        with info_path.open("r", encoding="utf-8") as handle:
            self.info = json.load(handle)

        codebase_version = str(self.info.get("codebase_version", ""))
        if codebase_version and codebase_version != "v3.0":
            raise RuntimeError(
                f"expected LeRobot codebase_version v3.0, got {codebase_version!r}"
            )
        fps = int(self.info.get("fps", FPS))
        if fps != FPS:
            raise RuntimeError(f"expected {FPS} fps dataset, got {fps}")
        if "video_path" not in self.info:
            raise RuntimeError("LeRobot info.json is missing video_path")

        parquet_root = self.dataset_root / "data"
        if not parquet_root.is_dir():
            raise FileNotFoundError(f"LeRobot parquet directory not found: {parquet_root}")
        self._data = pyarrow_dataset.dataset(parquet_root, format="parquet")

        metadata_files = sorted(
            (self.dataset_root / "meta" / "episodes").glob("**/*.parquet")
        )
        if not metadata_files:
            raise FileNotFoundError("no meta/episodes parquet files found")
        metadata_table = pyarrow_dataset.dataset(
            [str(path) for path in metadata_files], format="parquet"
        ).to_table()
        self._metadata = metadata_table.to_pandas()
        if "episode_index" not in self._metadata.columns:
            raise RuntimeError("episode metadata has no episode_index column")
        duplicate_mask = self._metadata["episode_index"].duplicated(keep=False)
        if duplicate_mask.any():
            duplicates = sorted(
                set(int(value) for value in self._metadata.loc[duplicate_mask, "episode_index"])
            )
            raise RuntimeError(f"duplicate episode metadata rows: {duplicates[:10]}")
        self._metadata = self._metadata.set_index("episode_index", drop=False)

    def load_episode(self, episode_index: int) -> EpisodeTabularData:
        episode_index = int(episode_index)
        required_columns = [
            "action",
            "observation.state",
            "timestamp",
            "frame_index",
            "episode_index",
        ]
        missing = set(required_columns) - set(self._data.schema.names)
        if missing:
            raise RuntimeError(f"dataset parquet schema is missing: {sorted(missing)}")
        table = self._data.to_table(
            columns=required_columns,
            filter=pyarrow_dataset.field("episode_index") == episode_index,
        )
        frame = table.to_pandas()
        if frame.empty:
            raise RuntimeError(f"episode {episode_index} has no parquet rows")
        frame = frame.sort_values("frame_index").reset_index(drop=True)
        unique_episode_ids = {int(value) for value in frame["episode_index"].unique()}
        if unique_episode_ids != {episode_index}:
            raise RuntimeError(
                f"episode boundary violation: requested {episode_index}, found {unique_episode_ids}"
            )

        actions = _stack_vector_column(frame["action"], "action")
        states = _stack_vector_column(frame["observation.state"], "observation.state")
        timestamps = torch.as_tensor(
            frame["timestamp"].to_numpy(), dtype=torch.float64
        ).contiguous()
        frame_indices = torch.as_tensor(
            frame["frame_index"].to_numpy(), dtype=torch.long
        ).contiguous()
        if not (len(actions) == len(states) == len(timestamps) == len(frame_indices)):
            raise RuntimeError("episode tabular columns have inconsistent lengths")
        validate_temporal_axis(timestamps, frame_indices, fps=FPS)

        metadata_row = self.metadata_row(episode_index)
        if "length" in metadata_row.index and not pd.isna(metadata_row["length"]):
            metadata_length = int(metadata_row["length"])
            if metadata_length != len(actions):
                raise RuntimeError(
                    f"episode {episode_index} metadata/parquet length mismatch: "
                    f"{metadata_length} != {len(actions)}"
                )
        return EpisodeTabularData(
            episode_index=episode_index,
            actions=actions,
            states=states,
            timestamps=timestamps,
            frame_indices=frame_indices,
        )

    def metadata_row(self, episode_index: int) -> pd.Series:
        episode_index = int(episode_index)
        if episode_index not in self._metadata.index:
            raise RuntimeError(f"episode {episode_index} is absent from metadata")
        row = self._metadata.loc[episode_index]
        if isinstance(row, pd.DataFrame):
            raise RuntimeError(f"episode {episode_index} metadata is not unique")
        return row

    def video_segment(self, episode_index: int) -> EpisodeVideoSegment:
        episode_index = int(episode_index)
        row = self.metadata_row(episode_index)
        chunk_column = _find_metadata_column(row, self.camera, "chunk_index")
        file_column = _find_metadata_column(row, self.camera, "file_index")
        from_column = _find_metadata_column(row, self.camera, "from_timestamp")
        to_column = _find_metadata_column(row, self.camera, "to_timestamp")
        if None in (chunk_column, file_column, from_column, to_column):
            raise RuntimeError(
                f"episode {episode_index} lacks complete front-video shard/interval metadata"
            )

        relative_path = self.info["video_path"].format(
            video_key=self.camera,
            chunk_index=int(row[chunk_column]),
            file_index=int(row[file_column]),
        )
        video_path = self.dataset_root / relative_path
        if not video_path.is_file():
            raise FileNotFoundError(f"video shard not found: {video_path}")
        from_timestamp = float(row[from_column])
        to_timestamp = float(row[to_column])
        if not np.isfinite(from_timestamp) or not np.isfinite(to_timestamp):
            raise RuntimeError("video interval timestamps must be finite")
        if to_timestamp <= from_timestamp:
            raise RuntimeError("video interval must have positive duration")
        return EpisodeVideoSegment(
            episode_index=episode_index,
            path=video_path,
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
        )

    def iter_frame_batches(
        self,
        segment: EpisodeVideoSegment,
        *,
        expected_frames: int,
        batch_size: int,
    ) -> Iterator[tuple[list[np.ndarray], list[float]]]:
        """Decode only one metadata-delimited episode from a shared AV1 shard."""

        expected_frames = int(expected_frames)
        batch_size = int(batch_size)
        if expected_frames <= 0 or batch_size <= 0:
            raise ValueError("expected_frames and batch_size must be positive")

        container = av.open(str(segment.path))
        images: list[np.ndarray] = []
        timestamps: list[float] = []
        decoded_count = 0
        try:
            stream = container.streams.video[0]
            seek_offset = max(0, int(segment.from_timestamp / float(stream.time_base)))
            container.seek(seek_offset, stream=stream, any_frame=False, backward=True)
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                timestamp = float(frame.pts * stream.time_base)
                if timestamp + 1e-6 < segment.from_timestamp:
                    continue
                if timestamp >= segment.to_timestamp:
                    break
                images.append(frame.to_ndarray(format="rgb24"))
                timestamps.append(timestamp)
                decoded_count += 1
                if len(images) == batch_size:
                    yield images, timestamps
                    images, timestamps = [], []
                if decoded_count == expected_frames:
                    break
            if images:
                yield images, timestamps
            if decoded_count != expected_frames:
                raise RuntimeError(
                    f"episode {segment.episode_index} video/parquet length mismatch: "
                    f"decoded={decoded_count}, expected={expected_frames}"
                )
        finally:
            container.close()
