from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.state_dynamics.checkpointing import atomic_save_checkpoint, make_checkpoint
from src.state_dynamics.config import load_config
from src.state_dynamics.dataset import SO101StateStore, StateDynamicsWindowDataset
from src.state_dynamics.model import StateDynamicsMLP, parameter_counts
from src.state_dynamics.normalization import compute_train_stats
from src.state_dynamics.split import (
    episode_ids_for_split, file_sha256, load_split_manifest, validate_episode_tasks,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the independent SO101 state MLP")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def normalized_batch(batch, stats, device):
    current = stats.normalize_states(batch["current_state"].float()).to(device)
    actions = stats.normalize_actions(batch["actions"].float()).to(device)
    target = stats.normalize_states(batch["target_states"].float()).to(device)
    return current, actions, target


@torch.inference_mode()
def validation_loss(model, loader, stats, device) -> float:
    was_training = model.training; model.eval()
    total, elements = 0.0, 0
    for batch in loader:
        current, actions, target = normalized_batch(batch, stats, device)
        squared = (model(current, actions) - target).square()
        total += float(squared.sum().cpu()); elements += squared.numel()
    if was_training:
        model.train()
    if elements == 0:
        raise RuntimeError("validation loader is empty")
    return total / elements


def main() -> None:
    args = build_parser().parse_args()
    if args.max_episodes < 0:
        raise ValueError("max-episodes must be non-negative")
    config = load_config(args.config)
    seed = int(config["experiment"]["seed"]); seed_everything(seed)
    device = resolve_device(config["train"]["device"])
    manifest = load_split_manifest(args.split_manifest)
    store = SO101StateStore(args.dataset_root)
    manifest_ids = {
        split: episode_ids_for_split(manifest, split)
        for split in ("train", "val", "test")
    }
    ids = {split: manifest_ids[split] for split in ("train", "val")}
    limit = 4 if args.smoke_test and args.max_episodes == 0 else args.max_episodes
    if limit > 0:
        ids = {split: values[:limit] for split, values in ids.items()}
    episodes = {split: store.load_episodes(values) for split, values in ids.items()}
    validate_episode_tasks(
        manifest,
        {
            episode_id: episode.task_id
            for split_episodes in episodes.values()
            for episode_id, episode in split_episodes.items()
        },
    )
    stats = compute_train_stats(episodes["train"].values())
    data_cfg = config["data"]
    datasets = {
        split: StateDynamicsWindowDataset(
            episodes[split], episode_order=ids[split],
            horizon=data_cfg["prediction_horizon"],
            window_stride=data_cfg["window_stride"],
        ) for split in ("train", "val")
    }
    batch_size, workers = config["train"]["batch_size"], config["train"]["num_workers"]
    loaders = {
        split: DataLoader(
            datasets[split], batch_size=batch_size, shuffle=(split == "train"),
            num_workers=workers, pin_memory=(device.type == "cuda"), drop_last=False,
        ) for split in ("train", "val")
    }
    model = StateDynamicsMLP(
        state_dim=data_cfg["state_dim"], action_dim=data_cfg["action_dim"],
        horizon=data_cfg["prediction_horizon"], **config["model"],
    ).to(device)
    total_parameters, trainable_parameters = parameter_counts(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )
    print("device:", device, "precision: fp32")
    for split in ("train", "val"):
        print(f"{split} episodes:", len(ids[split]), f"{split} windows:", len(datasets[split]))
    print("test episodes (manifest only):", len(manifest_ids["test"]))
    print("batch size:", batch_size, "steps per epoch:", len(loaders["train"]))
    print("total parameters:", total_parameters, "trainable parameters:", trainable_parameters)
    print("normalization source:", stats.source)
    print("state stats shape:", tuple(stats.state_mean.shape), tuple(stats.state_std.shape))
    print("action stats shape:", tuple(stats.action_mean.shape), tuple(stats.action_std.shape))
    if args.smoke_test:
        batch = next(iter(loaders["train"]))
        current, actions, target = normalized_batch(batch, stats, device)
        loss = F.mse_loss(model(current, actions), target)
        loss.backward(); optimizer.step()
        print("smoke forward/backward loss:", loss.detach().item())
        print("SMOKE PASS")
        return

    existing = [args.output_dir / "best.pt", args.output_dir / "last.pt"]
    if not args.overwrite and any(path.exists() for path in existing):
        raise RuntimeError("output checkpoint exists; use --overwrite explicitly")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_sha = file_sha256(args.split_manifest)
    best_val = math.inf; stale_epochs = 0; global_step = 0
    for epoch in range(1, config["train"]["epochs"] + 1):
        model.train(); loss_sum, batches = 0.0, 0
        for batch in loaders["train"]:
            current, actions, target = normalized_batch(batch, stats, device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(current, actions), target)
            loss.backward(); optimizer.step()
            loss_sum += loss.detach().item(); batches += 1; global_step += 1
        val_loss = validation_loss(model, loaders["val"], stats, device)
        print(
            f"epoch={epoch} train_normalized_mse={loss_sum / batches:.8g} "
            f"val_normalized_mse={val_loss:.8g}", flush=True,
        )
        improved = val_loss < best_val
        if improved:
            best_val = val_loss; stale_epochs = 0
        else:
            stale_epochs += 1
        checkpoint = make_checkpoint(
            model=model, optimizer=optimizer, epoch=epoch, global_step=global_step,
            config=config, stats=stats, split_manifest_path=args.split_manifest,
            split_manifest_sha256=manifest_sha, best_val_loss=best_val,
            dataset_identity=manifest["dataset_identity"],
        )
        atomic_save_checkpoint(args.output_dir / "last.pt", checkpoint)
        if improved:
            atomic_save_checkpoint(args.output_dir / "best.pt", checkpoint)
        if stale_epochs >= config["train"]["early_stopping_patience"]:
            print("early stopping at epoch", epoch); break
    print("best validation normalized MSE:", best_val)
    print("best checkpoint:", args.output_dir / "best.pt")
    print("last checkpoint:", args.output_dir / "last.pt")


if __name__ == "__main__":
    main()
