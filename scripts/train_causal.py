from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.causal.checkpointing import make_checkpoint, save_checkpoint
from src.causal.config import load_and_resolve_config
from src.causal.datasets import build_training_datasets, data_info
from src.causal.runtime import (
    ExponentialMovingAverage,
    build_model,
    build_optimizer,
    build_scheduler,
    causal_flow_loss,
    evaluate_flow_loss,
    validation_model,
    validate_precision_device,
)


def fixed_subset(dataset, maximum: int):
    if maximum <= 0 or maximum >= len(dataset):
        return dataset
    indices = (
        torch.linspace(0, len(dataset) - 1, steps=maximum)
        .round()
        .long()
        .unique()
        .tolist()
    )
    return Subset(dataset, indices)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train causal-action Flow Matching DiT")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()

    config = load_and_resolve_config(args.config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    precision = config["train"]["precision"]
    validate_precision_device(precision, device)

    seed = config["experiment"]["seed"]
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # This is the one and only statistics computation. Both datasets receive
    # the same immutable values; neither dataset can compute its own stats.
    train_dataset, val_dataset, action_stats = build_training_datasets(config)
    train = config["train"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=train["batch_size"],
        shuffle=True,
        num_workers=train["num_workers"],
        drop_last=True,
    )
    val_loader = DataLoader(
        fixed_subset(val_dataset, train["val_windows"]),
        batch_size=train["batch_size"],
        shuffle=False,
        num_workers=train["num_workers"],
        drop_last=False,
    )

    model = build_model(config).to(device)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    ema = (
        ExponentialMovingAverage(model, train["ema"]["decay"])
        if train["ema"]["enabled"]
        else None
    )
    output_dir = Path(config["checkpoint"]["output_dir"])
    name = config["experiment"]["name"]
    best_path = output_dir / f"{name}_best.pt"
    last_path = output_dir / f"{name}_last.pt"
    dataset_info = data_info(config)
    num_history = config["temporal"]["num_history"]

    print("experiment:", name)
    print("mode:", config["data"]["mode"])
    print("representation:", config["action"]["representation"])
    print("frame_stride:", config["temporal"]["frame_stride"])
    print("effective_action_dim:", config["action"]["effective_action_dim"])
    print("precision:", precision)
    print("optimizer betas:", train["betas"])
    print("optimizer eps:", train["eps"])
    print("weight_decay:", train["weight_decay"])
    print("scheduler:", train["scheduler"]["type"])
    print("warmup_ratio:", train["scheduler"]["warmup_ratio"])
    print("warmup_steps:", scheduler.warmup_steps)
    print("min_lr_ratio:", train["scheduler"]["min_lr_ratio"])
    print("ema_enabled:", train["ema"]["enabled"])
    print("ema_decay:", train["ema"]["decay"])
    print("validation_weights:", "ema" if ema is not None else "raw")
    print("action_stats source:", action_stats.source)
    print("action_mean:", action_stats.mean)
    print("action_std:", action_stats.std)
    print("train/val windows:", len(train_dataset), len(val_dataset))

    wall_clock_started_at = time.perf_counter()
    model.train()
    step = 0
    best_val_loss = float("inf")
    while step < train["steps"]:
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = causal_flow_loss(
                model,
                batch,
                device=device,
                num_history=num_history,
                precision=precision,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step + 1}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), train["grad_clip"]
            )
            update_lr = scheduler.get_last_lr()[0]
            optimizer.step()
            if ema is not None:
                ema.update(model)
            scheduler.step()
            step += 1
            if step == 1 or step % 25 == 0:
                loss_value = loss.detach().item()
                grad_norm_value = grad_norm.detach().item()
                elapsed_wall_seconds = time.perf_counter() - wall_clock_started_at
                print(
                    f"step={step:05d} train={loss_value:.6f} "
                    f"grad={grad_norm_value:.4f} lr={update_lr:.8e} "
                    f"elapsed_wall_seconds={elapsed_wall_seconds:.3f}"
                )

            if step % train["val_every"] == 0 or step == train["steps"]:
                val_loss = evaluate_flow_loss(
                    validation_model(model, ema),
                    val_loader,
                    device=device,
                    num_history=num_history,
                    seed=seed + 10000,
                    precision=precision,
                )
                elapsed_wall_seconds = time.perf_counter() - wall_clock_started_at
                print(
                    f"step={step:05d} val={val_loss:.6f} "
                    f"elapsed_wall_seconds={elapsed_wall_seconds:.3f}"
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if config["checkpoint"]["save_best"]:
                        save_checkpoint(
                            best_path,
                            make_checkpoint(
                                model=model,
                                optimizer=optimizer,
                                step=step,
                                config=config,
                                action_stats=action_stats,
                                data_info=dataset_info,
                                best_val_loss=best_val_loss,
                                elapsed_wall_seconds=elapsed_wall_seconds,
                                scheduler=scheduler,
                                ema=ema,
                            ),
                        )
            if step >= train["steps"]:
                break

    elapsed_wall_seconds = time.perf_counter() - wall_clock_started_at
    if config["checkpoint"]["save_last"]:
        save_checkpoint(
            last_path,
            make_checkpoint(
                model=model,
                optimizer=optimizer,
                step=step,
                config=config,
                action_stats=action_stats,
                data_info=dataset_info,
                best_val_loss=best_val_loss,
                elapsed_wall_seconds=elapsed_wall_seconds,
                scheduler=scheduler,
                ema=ema,
            ),
        )
    print("best_val_loss:", best_val_loss)
    print(f"elapsed_wall_seconds: {elapsed_wall_seconds:.3f}")
    print("best checkpoint:", best_path if config["checkpoint"]["save_best"] else None)
    print("last checkpoint:", last_path if config["checkpoint"]["save_last"] else None)


if __name__ == "__main__":
    main()
