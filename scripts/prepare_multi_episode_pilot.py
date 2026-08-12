from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


REPO_ID = "armnet/armnetbench_v01_lerobot_so101"
REVISION = "main"

REPO_ROOT = Path(
    "data/armnet_so101_sample_tabular"
)

OUTPUT_ROOT = Path(
    "data/multi_episode_pilot"
)

MANIFEST_PATH = (
    OUTPUT_ROOT / "manifest.json"
)

VIDEO_KEY = "observation.images.front"

REFERENCE_EPISODE = 0
NUM_EPISODES = 10
NUM_TRAIN = 8


def load_episode_metadata(
    root: Path,
) -> pd.DataFrame:

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

    if "episode_index" not in df.columns:
        raise RuntimeError(
            "episode_index missing from metadata."
        )

    return df


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
        "Multiple metadata columns matched: "
        f"{matches}"
    )


def normalize_task_value(
    x,
) -> str:
    """
    metadata 的 tasks 可能是 str/list/ndarray 等。
    pilot 阶段统一转字符串用于筛选。
    """

    if isinstance(x, (list, tuple)):
        return json.dumps(
            list(x),
            ensure_ascii=False,
            sort_keys=True,
        )

    return str(x)


def select_episodes(
    meta: pd.DataFrame,
) -> tuple[pd.DataFrame, str | None]:

    ref_rows = meta[
        meta["episode_index"]
        == REFERENCE_EPISODE
    ]

    if len(ref_rows) != 1:
        raise RuntimeError(
            "Could not uniquely locate "
            f"episode {REFERENCE_EPISODE}."
        )

    ref = ref_rows.iloc[0]

    # --------------------------------------------------------
    # 优先使用 tasks 字段找同任务 episode。
    # --------------------------------------------------------

    task_column = None

    for candidate in [
        "tasks",
        "task",
        "task_index",
    ]:
        if candidate in meta.columns:
            task_column = candidate
            break

    if task_column is None:

        print(
            "WARNING: no task field found; "
            "selecting first episodes."
        )

        selected = (
            meta
            .sort_values(
                "episode_index"
            )
            .head(
                NUM_EPISODES
            )
            .copy()
        )

        return selected, None

    ref_task = normalize_task_value(
        ref[task_column]
    )

    mask = meta[
        task_column
    ].map(
        normalize_task_value
    ) == ref_task

    candidates = (
        meta[mask]
        .sort_values(
            "episode_index"
        )
        .copy()
    )

    print(
        "task column:",
        task_column,
    )

    print(
        "reference task:",
        ref_task,
    )

    print(
        "same-task episodes available:",
        len(candidates),
    )

    if len(candidates) < NUM_EPISODES:
        raise RuntimeError(
            "Not enough same-task episodes: "
            f"need {NUM_EPISODES}, "
            f"found {len(candidates)}."
        )

    # episode 0 保留在 pilot 中
    selected = (
        candidates
        .head(
            NUM_EPISODES
        )
        .copy()
    )

    return selected, ref_task


def load_video_template(
    root: Path,
) -> str:

    info_path = (
        root
        / "meta"
        / "info.json"
    )

    with open(
        info_path,
        "r",
        encoding="utf-8",
    ) as f:
        info = json.load(f)

    return info["video_path"]


def resolve_video_filename(
    row: pd.Series,
    video_template: str,
) -> str:

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

    if (
        chunk_col is None
        or file_col is None
    ):
        raise RuntimeError(
            "Could not find front video "
            "chunk_index/file_index."
        )

    chunk_index = int(
        row[chunk_col]
    )

    file_index = int(
        row[file_col]
    )

    return video_template.format(
        video_key=VIDEO_KEY,
        chunk_index=chunk_index,
        file_index=file_index,
    )


def get_length(
    row: pd.Series,
):

    if "length" in row.index:
        return int(
            row["length"]
        )

    return None


def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("SO101 MULTI-EPISODE PILOT")
    print("=" * 80)

    # ========================================================
    # 1. Load episode metadata
    # ========================================================

    meta = load_episode_metadata(
        REPO_ROOT
    )

    print(
        "metadata episodes:",
        len(meta),
    )

    # ========================================================
    # 2. Select same-task episodes
    # ========================================================

    selected, task_value = (
        select_episodes(
            meta
        )
    )

    episode_ids = [
        int(x)
        for x
        in selected[
            "episode_index"
        ].tolist()
    ]

    if (
        REFERENCE_EPISODE
        not in episode_ids
    ):
        raise RuntimeError(
            "Reference episode 0 "
            "was not selected."
        )

    # --------------------------------------------------------
    # 固定 episode-level split
    #
    # 先简单用前 8 train / 后 2 val。
    # 后面正式实验可以随机 seed split。
    # --------------------------------------------------------

    train_ids = (
        episode_ids[
            :NUM_TRAIN
        ]
    )

    val_ids = (
        episode_ids[
            NUM_TRAIN:
        ]
    )

    print()
    print(
        "selected episodes:",
        episode_ids,
    )

    print(
        "train episodes   :",
        train_ids,
    )

    print(
        "val episodes     :",
        val_ids,
    )

    assert (
        set(train_ids)
        .isdisjoint(
            set(val_ids)
        )
    )

    # ========================================================
    # 3. Resolve front video shards
    # ========================================================

    video_template = (
        load_video_template(
            REPO_ROOT
        )
    )

    episode_records = []

    unique_video_files = set()

    print()
    print("=" * 80)
    print("EPISODE VIDEO MAPPING")
    print("=" * 80)

    for _, row in (
        selected.iterrows()
    ):

        episode_id = int(
            row["episode_index"]
        )

        filename = (
            resolve_video_filename(
                row,
                video_template,
            )
        )

        unique_video_files.add(
            filename
        )

        length = get_length(
            row
        )

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

        from_timestamp = (
            float(row[from_col])
            if from_col is not None
            else None
        )

        to_timestamp = (
            float(row[to_col])
            if to_col is not None
            else None
        )

        print(
            f"episode={episode_id:04d}  "
            f"length={length}  "
            f"video={filename}"
        )

        episode_records.append(
            {
                "episode_index": (
                    episode_id
                ),
                "length": length,
                "video_file": filename,
                "from_timestamp": (
                    from_timestamp
                ),
                "to_timestamp": (
                    to_timestamp
                ),
            }
        )

    # ========================================================
    # 4. Download only unique front shards
    #
    # 多个 episode 可能位于同一个 MP4 shard，
    # 所以一定去重。
    # ========================================================

    print()
    print("=" * 80)
    print("VIDEO DOWNLOAD")
    print("=" * 80)

    print(
        "episodes:",
        len(episode_ids),
    )

    print(
        "unique front shards:",
        len(unique_video_files),
    )

    for i, filename in enumerate(
        sorted(
            unique_video_files
        ),
        start=1,
    ):

        local_file = (
            REPO_ROOT
            / filename
        )

        if local_file.exists():

            print(
                f"[{i:02d}/"
                f"{len(unique_video_files):02d}] "
                "already exists:",
                filename,
            )

            continue

        print(
            f"[{i:02d}/"
            f"{len(unique_video_files):02d}] "
            "downloading:",
            filename,
        )

        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            revision=REVISION,
            filename=filename,
            local_dir=REPO_ROOT,
        )

    # ========================================================
    # 5. Save manifest
    # ========================================================

    manifest = {
        "repo_id": REPO_ID,
        "revision": REVISION,

        "camera": VIDEO_KEY,

        "reference_episode": (
            REFERENCE_EPISODE
        ),

        "task": task_value,

        "episode_ids": (
            episode_ids
        ),

        "train_episode_ids": (
            train_ids
        ),

        "val_episode_ids": (
            val_ids
        ),

        "episodes": (
            episode_records
        ),
    }

    with open(
        MANIFEST_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 80)
    print("MULTI-EPISODE PILOT PREPARED")
    print("=" * 80)

    print(
        "manifest:",
        MANIFEST_PATH,
    )

    print(
        "train:",
        train_ids,
    )

    print(
        "val  :",
        val_ids,
    )


if __name__ == "__main__":
    main()