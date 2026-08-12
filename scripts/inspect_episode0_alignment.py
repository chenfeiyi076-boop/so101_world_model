from __future__ import annotations

import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from PIL import Image


REPO_ROOT = Path("data/armnet_so101_sample_tabular")

EPISODE_INDEX = 0
VIDEO_KEY = "observation.images.front"

# 取 episode 中间连续 10 帧
START_FRAME_IN_EPISODE = 250
NUM_FRAMES = 10

OUTPUT_DIR = Path(
    "outputs/inspect_episode_000"
)


def to_vector(x):
    return np.asarray(x).reshape(-1)


def load_episode_dataframe(
    root: Path,
    episode_index: int,
) -> pd.DataFrame:
    """
    只从 parquet 数据集中读取指定 episode，
    避免把整个数据集全部加载到内存。
    """

    parquet_root = root / "data"

    dataset = ds.dataset(
        parquet_root,
        format="parquet",
    )

    table = dataset.to_table(
        filter=(
            ds.field("episode_index")
            == episode_index
        )
    )

    df = table.to_pandas()

    if len(df) == 0:
        raise RuntimeError(
            f"Episode {episode_index} not found."
        )

    if "frame_index" in df.columns:
        df = df.sort_values(
            "frame_index"
        )

    df = df.reset_index(
        drop=True
    )

    return df


def load_episode_metadata(
    root: Path,
    episode_index: int,
) -> pd.Series:

    files = sorted(
        root.glob(
            "meta/episodes/**/*.parquet"
        )
    )

    if not files:
        raise RuntimeError(
            "No episode metadata parquet found."
        )

    df = pd.concat(
        [
            pd.read_parquet(p)
            for p in files
        ],
        ignore_index=True,
    )

    rows = df[
        df["episode_index"]
        == episode_index
    ]

    if len(rows) != 1:
        raise RuntimeError(
            f"Expected 1 metadata row for episode "
            f"{episode_index}, got {len(rows)}"
        )

    return rows.iloc[0]


def find_meta_column(
    row: pd.Series,
    video_key: str,
    suffix: str,
):
    matches = []

    for col in row.index:

        col_str = str(col)

        if (
            video_key in col_str
            and col_str.endswith(suffix)
        ):
            matches.append(col)

    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0:
        return None

    raise RuntimeError(
        f"Multiple metadata columns matched "
        f"{video_key=} {suffix=}: {matches}"
    )


def resolve_video_path(
    root: Path,
    row: pd.Series,
) -> Path:

    chunk_col = find_meta_column(
        row,
        VIDEO_KEY,
        "chunk_index",
    )

    file_col = find_meta_column(
        row,
        VIDEO_KEY,
        "file_index",
    )

    if chunk_col is None or file_col is None:
        raise RuntimeError(
            "Could not resolve front video "
            "chunk_index / file_index."
        )

    chunk_index = int(
        row[chunk_col]
    )

    file_index = int(
        row[file_col]
    )

    with open(
        root / "meta" / "info.json",
        "r",
        encoding="utf-8",
    ) as f:
        info = json.load(f)

    video_template = info["video_path"]

    relative_path = video_template.format(
        video_key=VIDEO_KEY,
        chunk_index=chunk_index,
        file_index=file_index,
    )

    video_path = (
        root / relative_path
    )

    if not video_path.exists():
        raise FileNotFoundError(
            f"Downloaded video not found:\n"
            f"{video_path}"
        )

    return video_path


def get_episode_video_interval(
    row: pd.Series,
):
    from_col = find_meta_column(
        row,
        VIDEO_KEY,
        "from_timestamp",
    )

    to_col = find_meta_column(
        row,
        VIDEO_KEY,
        "to_timestamp",
    )

    if from_col is None:
        raise RuntimeError(
            "front from_timestamp not found "
            "in episode metadata."
        )

    start_time = float(
        row[from_col]
    )

    end_time = (
        float(row[to_col])
        if to_col is not None
        else None
    )

    return start_time, end_time


def decode_episode_frames(
    video_path: Path,
    episode_start_time: float,
    episode_end_time: float | None,
    expected_frames: int,
):
    """
    按 MP4 中每帧的实际 PTS 时间戳，
    截取 episode 对应的视频片段。

    不简单假设 episode 0 就从 MP4 第 0 帧开始。
    """

    container = av.open(
        str(video_path)
    )

    stream = container.streams.video[0]

    frames = []

    for frame in container.decode(
        stream
    ):

        if frame.pts is None:
            continue

        timestamp = float(
            frame.pts
            * stream.time_base
        )

        # episode 之前
        if timestamp + 1e-6 < episode_start_time:
            continue

        # episode 之后
        if (
            episode_end_time is not None
            and timestamp >= episode_end_time
        ):
            break

        image = frame.to_ndarray(
            format="rgb24"
        )

        frames.append(
            (
                timestamp,
                image,
            )
        )

        # metadata 已经告诉我们 episode 长度，
        # 得到足够帧后即可停止。
        if len(frames) >= expected_frames:
            break

    container.close()

    return frames


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("LOAD EPISODE 0 TABULAR DATA")
    print("=" * 80)

    df = load_episode_dataframe(
        REPO_ROOT,
        EPISODE_INDEX,
    )

    print(
        "episode rows:",
        len(df),
    )

    print(
        "columns:",
        list(df.columns),
    )

    if len(df) != 510:
        print(
            "WARNING: expected 510 rows, "
            f"got {len(df)}"
        )

    # ---------------------------------------------------------
    # episode metadata
    # ---------------------------------------------------------

    meta_row = load_episode_metadata(
        REPO_ROOT,
        EPISODE_INDEX,
    )

    video_path = resolve_video_path(
        REPO_ROOT,
        meta_row,
    )

    episode_start_time, episode_end_time = (
        get_episode_video_interval(
            meta_row
        )
    )

    print()
    print("=" * 80)
    print("VIDEO")
    print("=" * 80)

    print(
        "video:",
        video_path,
    )

    print(
        "episode video start:",
        episode_start_time,
    )

    print(
        "episode video end  :",
        episode_end_time,
    )

    # ---------------------------------------------------------
    # decode episode video
    # ---------------------------------------------------------

    print()
    print("Decoding episode frames...")

    video_frames = decode_episode_frames(
        video_path=video_path,
        episode_start_time=episode_start_time,
        episode_end_time=episode_end_time,
        expected_frames=len(df),
    )

    print(
        "decoded frames:",
        len(video_frames),
    )

    # ---------------------------------------------------------
    # 最重要的 sanity check
    # ---------------------------------------------------------

    if len(video_frames) != len(df):
        print()
        print(
            "WARNING:"
        )
        print(
            "video frame count != parquet rows"
        )
        print(
            f"video   : {len(video_frames)}"
        )
        print(
            f"parquet : {len(df)}"
        )
        print(
            "Do not proceed to VAE yet."
        )

    # ---------------------------------------------------------
    # 取连续 10 帧
    # ---------------------------------------------------------

    start = START_FRAME_IN_EPISODE
    end = start + NUM_FRAMES

    if end > min(
        len(df),
        len(video_frames),
    ):
        raise RuntimeError(
            "Requested inspection range "
            "exceeds episode length."
        )

    records = []

    print()
    print("=" * 80)
    print(
        f"INSPECT FRAMES {start} ... {end - 1}"
    )
    print("=" * 80)

    for local_idx in range(
        start,
        end,
    ):

        row = df.iloc[
            local_idx
        ]

        video_timestamp, image = (
            video_frames[
                local_idx
            ]
        )

        action = to_vector(
            row["action"]
        )

        state = to_vector(
            row["observation.state"]
        )

        error = (
            action - state
        )

        frame_index = (
            int(row["frame_index"])
            if "frame_index" in row
            else local_idx
        )

        data_timestamp = (
            float(row["timestamp"])
            if "timestamp" in row
            else float("nan")
        )

        filename = (
            OUTPUT_DIR
            / f"frame_{local_idx:04d}.png"
        )

        Image.fromarray(
            image
        ).save(
            filename
        )

        print()
        print(
            f"episode local index : {local_idx}"
        )

        print(
            f"frame_index         : {frame_index}"
        )

        print(
            f"data timestamp      : {data_timestamp:.6f}"
        )

        print(
            f"video timestamp     : {video_timestamp:.6f}"
        )

        print(
            "action              :",
            np.round(
                action,
                4,
            ),
        )

        print(
            "state               :",
            np.round(
                state,
                4,
            ),
        )

        print(
            "action - state      :",
            np.round(
                error,
                4,
            ),
        )

        records.append(
            {
                "local_index": local_idx,
                "frame_index": frame_index,
                "data_timestamp": data_timestamp,
                "video_timestamp": video_timestamp,
                "image": str(filename),
            }
        )

    # ---------------------------------------------------------
    # 保存一份对应关系 CSV
    # ---------------------------------------------------------

    pd.DataFrame(
        records
    ).to_csv(
        OUTPUT_DIR
        / "alignment.csv",
        index=False,
    )

    print()
    print("=" * 80)
    print("ALIGNMENT INSPECTION COMPLETE")
    print("=" * 80)

    print(
        "images:",
        OUTPUT_DIR,
    )

    print(
        "table :",
        OUTPUT_DIR / "alignment.csv",
    )


if __name__ == "__main__":
    main()