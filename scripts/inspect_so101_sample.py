from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from huggingface_hub import (
    list_repo_files,
    snapshot_download,
)


REPO_ID = "armnet/armnetbench_v01_lerobot_so101"
REVISION = "main"

ROOT = Path("data/armnet_so101_sample_tabular")


def to_vector(x):
    """
    将 parquet 中的 list / ndarray 等统一转成 1D numpy array。
    """
    if isinstance(x, np.ndarray):
        return x.reshape(-1)

    if isinstance(x, (list, tuple)):
        return np.asarray(x).reshape(-1)

    return np.asarray(x).reshape(-1)


def main():

    # ============================================================
    # 1. 先检查 sample revision 当前有哪些文件
    # ============================================================

    print("=" * 80)
    print("Repository inspection")
    print("=" * 80)

    files = list_repo_files(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=REVISION,
    )

    print(f"repo     : {REPO_ID}")
    print(f"revision : {REVISION}")
    print(f"files    : {len(files)}")
    print()

    print("First files:")
    for f in files[:40]:
        print(" ", f)

    # ============================================================
    # 2. 只下载 meta 和 parquet
    #
    # 不下载 videos/
    # ============================================================

    print()
    print("=" * 80)
    print("Downloading metadata + parquet only")
    print("=" * 80)

    local_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        local_dir=ROOT,
        allow_patterns=[
            "meta/**",
            "data/**",
        ],
    )

    local_dir = Path(local_dir)

    print("local dir:", local_dir)

    # ============================================================
    # 3. 找逐 frame parquet
    # ============================================================

    data_files = sorted(
        local_dir.glob("data/**/*.parquet")
    )

    print()
    print("data parquet files:", len(data_files))

    if not data_files:
        raise RuntimeError(
            "No data parquet files found in sample revision."
        )

    for p in data_files[:10]:
        print(" ", p.relative_to(local_dir))

    # ============================================================
    # 4. 读取第一个 parquet
    # ============================================================

    parquet_path = data_files[0]

    print()
    print("=" * 80)
    print("Inspecting first frame parquet")
    print("=" * 80)

    print("file:", parquet_path)

    df = pd.read_parquet(
        parquet_path
    )

    print("rows   :", len(df))
    print("columns:")

    for col in df.columns:
        print(
            f"  {col:<35} "
            f"dtype={df[col].dtype}"
        )

    # ============================================================
    # 5. 检查 world model 所需字段
    # ============================================================

    required = [
        "action",
        "observation.state",
    ]

    print()
    print("=" * 80)
    print("Required fields")
    print("=" * 80)

    for key in required:

        exists = key in df.columns

        print(
            f"{key:<25}: "
            f"{'FOUND' if exists else 'MISSING'}"
        )

    if "action" not in df.columns:
        raise RuntimeError(
            "'action' is missing."
        )

    if "observation.state" not in df.columns:
        raise RuntimeError(
            "'observation.state' is missing."
        )

    # ============================================================
    # 6. 检查 action/state 维度
    # ============================================================

    action0 = to_vector(
        df.iloc[0]["action"]
    )

    state0 = to_vector(
        df.iloc[0]["observation.state"]
    )

    print()
    print("=" * 80)
    print("Vector dimensions")
    print("=" * 80)

    print(
        "action shape:",
        action0.shape,
    )

    print(
        "state shape :",
        state0.shape,
    )

    print(
        "action[0]:",
        action0,
    )

    print(
        "state[0] :",
        state0,
    )

    if action0.shape != (6,):
        raise RuntimeError(
            f"Expected 6-D action, got {action0.shape}"
        )

    if state0.shape != (6,):
        raise RuntimeError(
            f"Expected 6-D state, got {state0.shape}"
        )

    # ============================================================
    # 7. 找时间 / episode indexing 字段
    # ============================================================

    possible_index_columns = [
        "episode_index",
        "frame_index",
        "index",
        "timestamp",
        "task_index",
        "next.done",
        "next.reward",
    ]

    print()
    print("=" * 80)
    print("Index / time related columns")
    print("=" * 80)

    existing_index_columns = []

    for key in possible_index_columns:

        if key in df.columns:

            existing_index_columns.append(
                key
            )

            print(
                f"{key:<20}: FOUND"
            )

    # ============================================================
    # 8. 打印连续前 10 个 timestep
    # ============================================================

    print()
    print("=" * 80)
    print("First 10 timesteps")
    print("=" * 80)

    n = min(
        10,
        len(df),
    )

    for i in range(n):

        row = df.iloc[i]

        action = to_vector(
            row["action"]
        )

        state = to_vector(
            row["observation.state"]
        )

        error = action - state

        print()
        print(
            f"row {i}"
        )

        for key in existing_index_columns:

            print(
                f"  {key:<15}: "
                f"{row[key]}"
            )

        print(
            "  action:",
            np.round(
                action,
                4,
            ),
        )

        print(
            "  state :",
            np.round(
                state,
                4,
            ),
        )

        print(
            "  a-q   :",
            np.round(
                error,
                4,
            ),
        )

    # ============================================================
    # 9. episode metadata
    # ============================================================

    episode_files = sorted(
        local_dir.glob(
            "meta/episodes/**/*.parquet"
        )
    )

    print()
    print("=" * 80)
    print("Episode metadata")
    print("=" * 80)

    print(
        "episode parquet files:",
        len(episode_files),
    )

    if episode_files:

        ep_df = pd.concat(
            [
                pd.read_parquet(p)
                for p in episode_files
            ],
            ignore_index=True,
        )

        print(
            "episode rows:",
            len(ep_df),
        )

        print(
            "episode columns:",
            list(ep_df.columns),
        )

        useful = [
            c
            for c in [
                "episode_index",
                "tasks",
                "task_index",
                "success",
                "success_class",
                "policy_type",
                "length",
            ]
            if c in ep_df.columns
        ]

        if useful:
            print()
            print(
                ep_df[
                    useful
                ].head(20).to_string(
                    index=False
                )
            )

    print()
    print("=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
