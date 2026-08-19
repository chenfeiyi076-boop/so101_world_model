from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.state_dynamics.dataset import SO101StateStore
from src.state_dynamics.split import (
    atomic_write_manifest,
    build_split_manifest,
    read_episode_task_mapping,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the independent task-stratified SO101 state split"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--causal-manifest", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    mapping = read_episode_task_mapping(args.dataset_root)
    store = SO101StateStore(args.dataset_root)
    first_episode_id = min(mapping)
    first = store.load_episodes([first_episode_id])[first_episode_id]
    print("dataset root:", args.dataset_root.resolve())
    print("parquet schema:", store.schema_names)
    print("episodes:", len(mapping), "tasks:", len(set(mapping.values())))
    print("first episode:", first_episode_id, "frames:", first.length)
    print("action shape:", tuple(first.actions.shape))
    print("observation.state shape:", tuple(first.states.shape))
    print("frame index range:", int(first.frame_indices[0]), int(first.frame_indices[-1]))
    timestamp_info = store.inspect_episode_timestamps(first_episode_id)
    print("timestamp diagnostics:", timestamp_info)
    if args.inspect_only:
        return
    if args.output is None:
        raise ValueError("--output is required unless --inspect-only is used")
    manifest = build_split_manifest(
        dataset_root=args.dataset_root, episode_to_task=mapping, seed=args.seed
    )
    counts = {
        split: len(manifest[f"{split}_episode_ids"])
        for split in ("train", "val", "test")
    }
    if len(mapping) == 2499:
        if set(mapping) != set(range(2499)):
            raise RuntimeError("formal SO101 episode IDs must be exactly 0..2498")
        if counts != {"train": 1999, "val": 248, "test": 252}:
            raise RuntimeError(f"formal SO101 split count mismatch: {counts}")
    if args.causal_manifest:
        with args.causal_manifest.open("r", encoding="utf-8") as handle:
            causal = json.load(handle)
        for split in ("train", "val", "test"):
            if manifest[f"{split}_episode_ids"] != [
                int(v) for v in causal[f"{split}_episode_ids"]
            ]:
                raise RuntimeError(f"state/causal {split} episode IDs differ")
        print("causal manifest compatibility: PASS")
    atomic_write_manifest(manifest, args.output)
    print("split counts:", counts)
    print("output:", args.output)


if __name__ == "__main__":
    main()
