from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


REPO_ID = "armnet/armnetbench_v01_lerobot_so101"
REVISION = "main"

# 你前面下载 meta + data 的目录
ROOT = Path("data/armnet_so101_sample_tabular")

EPISODE_INDEX = 0
VIDEO_KEY = "observation.images.front"


def find_episode_row(root: Path, episode_index: int) -> pd.Series:
    episode_files = sorted(
        root.glob("meta/episodes/**/*.parquet")
    )

    if not episode_files:
        raise RuntimeError(
            f"No episode metadata found under {root / 'meta/episodes'}"
        )

    df = pd.concat(
        [pd.read_parquet(p) for p in episode_files],
        ignore_index=True,
    )

    if "episode_index" not in df.columns:
        raise RuntimeError(
            "episode_index column not found in episode metadata."
        )

    rows = df[
        df["episode_index"] == episode_index
    ]

    if len(rows) != 1:
        raise RuntimeError(
            f"Expected exactly one row for episode {episode_index}, "
            f"got {len(rows)}"
        )

    return rows.iloc[0]


def find_column(row: pd.Series, video_key: str, suffix: str):
    """
    自动寻找类似：

    videos/observation.images.front/chunk_index
    videos/observation.images.front/file_index
    videos/observation.images.front/from_timestamp
    ...

    避免把列名写死。
    """

    candidates = []

    for col in row.index:
        col_str = str(col)

        if (
            video_key in col_str
            and col_str.endswith(suffix)
        ):
            candidates.append(col)

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) == 0:
        return None

    raise RuntimeError(
        f"Multiple columns matched {video_key=} {suffix=}: "
        f"{candidates}"
    )


def main():

    print("=" * 80)
    print("EPISODE 0 FRONT VIDEO DOWNLOAD")
    print("=" * 80)

    # ==========================================================
    # 1. 读取 episode 0 metadata
    # ==========================================================

    row = find_episode_row(
        ROOT,
        EPISODE_INDEX,
    )

    print()
    print("Episode metadata:")
    print("episode_index:", EPISODE_INDEX)

    if "length" in row.index:
        print("length       :", row["length"])

    if "tasks" in row.index:
        print("tasks        :", row["tasks"])

    # ==========================================================
    # 2. 打印所有跟 front/video 有关的 metadata
    # ==========================================================

    print()
    print("Front/video related metadata:")

    for col in row.index:
        col_str = str(col)

        if (
            "front" in col_str.lower()
            or "video" in col_str.lower()
        ):
            print(
                f"  {col_str}: {row[col]}"
            )

    # ==========================================================
    # 3. 找 chunk/file index
    # ==========================================================

    chunk_col = find_column(
        row,
        VIDEO_KEY,
        "chunk_index",
    )

    file_col = find_column(
        row,
        VIDEO_KEY,
        "file_index",
    )

    if chunk_col is None or file_col is None:
        raise RuntimeError(
            "Could not find front video chunk_index/file_index.\n"
            "Look at the printed metadata above and send it to me."
        )

    chunk_index = int(
        row[chunk_col]
    )

    file_index = int(
        row[file_col]
    )

    # ==========================================================
    # 4. 读取 info.json 中官方 video_path template
    # ==========================================================

    info_path = ROOT / "meta" / "info.json"

    with open(
        info_path,
        "r",
        encoding="utf-8",
    ) as f:
        info = json.load(f)

    video_template = info["video_path"]

    filename = video_template.format(
        video_key=VIDEO_KEY,
        chunk_index=chunk_index,
        file_index=file_index,
    )

    print()
    print("Resolved video shard:")
    print(" ", filename)

    # ==========================================================
    # 5. episode 在该 MP4 内部的时间范围
    # ==========================================================

    from_col = find_column(
        row,
        VIDEO_KEY,
        "from_timestamp",
    )

    to_col = find_column(
        row,
        VIDEO_KEY,
        "to_timestamp",
    )

    if from_col is not None:
        print(
            "from_timestamp:",
            row[from_col],
        )

    if to_col is not None:
        print(
            "to_timestamp  :",
            row[to_col],
        )

    # ==========================================================
    # 6. 只下载这个 front-camera shard
    # ==========================================================

    print()
    print("Downloading...")

    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        filename=filename,
        local_dir=ROOT,
    )

    print()
    print("=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)

    print("saved to:")
    print(local_path)


if __name__ == "__main__":
    main()