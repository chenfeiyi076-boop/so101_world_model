from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_causal import fixed_subset
from src.causal.checkpointing import (
    current_git_commit,
    make_checkpoint,
    save_checkpoint,
)
from src.causal.config import load_and_resolve_config
from src.causal.data.common import ActionStats
from src.causal.datasets import (
    build_dataset,
    compute_training_action_stats,
    data_info,
)
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


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    backend: str

    @property
    def is_main(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class DDPDataLoaderSettings:
    num_workers: int
    torch_num_threads: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int


def global_batch_size(per_gpu_batch_size: int, world_size: int) -> int:
    per_gpu_batch_size = int(per_gpu_batch_size)
    world_size = int(world_size)
    if per_gpu_batch_size <= 0 or world_size <= 0:
        raise ValueError("per-GPU batch size and world size must be positive")
    return per_gpu_batch_size * world_size


def fixed_validation_subset(dataset, maximum: int):
    """Reuse the single-GPU trainer's exact fixed validation subset rule."""

    return fixed_subset(dataset, maximum)


def resolve_dataloader_settings(config: dict[str, Any]) -> DDPDataLoaderSettings:
    train = config["train"]
    distributed = config.get("distributed", {})
    if not isinstance(distributed, dict):
        raise ValueError("distributed config must be a mapping")
    num_workers = int(train["num_workers"])
    torch_num_threads = int(distributed.get("torch_num_threads", 1))
    prefetch_factor = int(distributed.get("prefetch_factor", 2))
    if num_workers < 0:
        raise ValueError("train.num_workers must be non-negative")
    if torch_num_threads <= 0:
        raise ValueError("distributed.torch_num_threads must be positive")
    if prefetch_factor <= 0:
        raise ValueError("distributed.prefetch_factor must be positive")
    return DDPDataLoaderSettings(
        num_workers=num_workers,
        torch_num_threads=torch_num_threads,
        pin_memory=bool(distributed.get("pin_memory", True)),
        persistent_workers=bool(distributed.get("persistent_workers", True)),
        prefetch_factor=prefetch_factor,
    )


def dataloader_worker_kwargs(
    settings: DDPDataLoaderSettings,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "num_workers": settings.num_workers,
        "pin_memory": settings.pin_memory,
    }
    # PyTorch rejects persistent_workers=True and an explicit prefetch_factor
    # when multiprocessing is disabled.
    if settings.num_workers > 0:
        kwargs["persistent_workers"] = settings.persistent_workers
        kwargs["prefetch_factor"] = settings.prefetch_factor
    return kwargs


def build_distributed_train_sampler(
    dataset,
    *,
    world_size: int,
    rank: int,
    seed: int,
) -> DistributedSampler:
    # With drop_last=False DistributedSampler preserves every index, but it
    # pads when the dataset is not divisible by world_size. Refuse that case
    # instead of silently introducing duplicates into an epoch.
    if len(dataset) % int(world_size) != 0:
        raise ValueError(
            "exact no-replacement DDP traversal requires dataset length "
            "to be divisible by world_size"
        )
    return DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=int(seed),
        drop_last=False,
    )


def build_train_loader(
    dataset,
    *,
    sampler: DistributedSampler,
    per_gpu_batch_size: int,
    settings: DDPDataLoaderSettings,
    generator: torch.Generator,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(per_gpu_batch_size),
        sampler=sampler,
        shuffle=False,
        drop_last=False,
        generator=generator,
        **dataloader_worker_kwargs(settings),
    )


def build_validation_loader(
    dataset,
    *,
    per_gpu_batch_size: int,
    settings: DDPDataLoaderSettings,
    generator: torch.Generator,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(per_gpu_batch_size),
        shuffle=False,
        drop_last=False,
        generator=generator,
        **dataloader_worker_kwargs(settings),
    )


def checkpoint_has_ddp_prefix(checkpoint: dict[str, Any]) -> bool:
    for field in ("model_state_dict", "ema_model_state_dict"):
        for key in checkpoint.get(field, {}):
            if key == "module" or key.startswith("module."):
                return True
    return False


def make_ddp_checkpoint(
    *,
    raw_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    ema: ExponentialMovingAverage | None,
    step: int,
    config: dict[str, Any],
    action_stats: ActionStats,
    dataset_info: dict[str, Any],
    best_val_loss: float | None,
    elapsed_wall_seconds: float,
    context: DistributedContext,
    per_gpu_batch_size: int,
    sampler_epoch: int,
) -> dict[str, Any]:
    checkpoint = make_checkpoint(
        model=raw_model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        step=step,
        config=config,
        action_stats=action_stats,
        data_info=dataset_info,
        best_val_loss=best_val_loss,
        elapsed_wall_seconds=elapsed_wall_seconds,
    )
    checkpoint["distributed"] = {
        "enabled": True,
        "world_size": int(context.world_size),
        "backend": context.backend,
        "per_gpu_batch_size": int(per_gpu_batch_size),
        "global_batch_size": global_batch_size(
            per_gpu_batch_size, context.world_size
        ),
        "sampler": "DistributedSampler",
        "sampler_epoch": int(sampler_epoch),
    }
    if checkpoint_has_ddp_prefix(checkpoint):
        raise RuntimeError("DDP checkpoint state_dict contains a module. prefix")
    return checkpoint


def save_checkpoint_on_rank_zero(
    path: str | Path,
    checkpoint: dict[str, Any],
    *,
    rank: int,
    save_fn: Callable[[str | Path, dict[str, Any]], None] = save_checkpoint,
) -> bool:
    if int(rank) != 0:
        return False
    save_fn(path, checkpoint)
    return True


def _integer_environment(name: str) -> int:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(
            f"{name} is missing; launch with torchrun, not plain python"
        )
    try:
        return int(value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer, got {value!r}") from error


def initialize_distributed(backend: str) -> DistributedContext:
    if backend != "nccl":
        raise ValueError("formal causal DDP training requires backend='nccl'")
    rank = _integer_environment("RANK")
    local_rank = _integer_environment("LOCAL_RANK")
    world_size = _integer_environment("WORLD_SIZE")
    if world_size <= 0 or rank < 0 or rank >= world_size or local_rank < 0:
        raise RuntimeError("invalid torchrun rank environment")
    if not torch.cuda.is_available():
        raise RuntimeError("NCCL causal DDP training requires CUDA")
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} CUDA devices are visible"
        )
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")
    return DistributedContext(rank, local_rank, world_size, backend)


def _validate_expected_world_size(
    config: dict[str, Any], context: DistributedContext
) -> None:
    distributed = config.get("distributed", {})
    expected = distributed.get("expected_world_size")
    if expected is not None and int(expected) != context.world_size:
        raise RuntimeError(
            f"config expects world_size={int(expected)}, got {context.world_size}"
        )


def _broadcast_rank_zero_error(error: str | None, context: DistributedContext) -> None:
    payload = [error if context.is_main else None]
    dist.broadcast_object_list(payload, src=0)
    if payload[0] is not None:
        raise RuntimeError(str(payload[0]))


def _require_empty_output_directory(
    output_dir: Path, context: DistributedContext
) -> None:
    error = None
    if context.is_main and output_dir.exists():
        try:
            nonempty = next(output_dir.iterdir(), None) is not None
        except OSError as exception:
            error = f"cannot inspect output directory {output_dir}: {exception}"
        else:
            if nonempty:
                error = f"refusing to overwrite non-empty output directory: {output_dir}"
    _broadcast_rank_zero_error(error, context)


def _compute_and_broadcast_action_stats(
    config: dict[str, Any], context: DistributedContext
) -> ActionStats:
    payload: list[dict[str, Any] | None] = [None]
    if context.is_main:
        try:
            stats = compute_training_action_stats(config)
            payload[0] = {"stats": stats.to_dict()}
        except Exception as exception:  # Propagate instead of leaving peers waiting.
            payload[0] = {
                "error": f"{type(exception).__name__}: {exception}"
            }
    dist.broadcast_object_list(payload, src=0)
    result = payload[0]
    if not isinstance(result, dict):
        raise RuntimeError("rank 0 broadcast invalid action statistics")
    if "error" in result:
        raise RuntimeError(f"rank 0 action-stat computation failed: {result['error']}")
    return ActionStats.from_dict(result["stats"])


def _set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def _distributed_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    result = value.detach().float().clone()
    dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result / float(world_size)


def _run_smoke_test(context: DistributedContext, device: torch.device) -> None:
    value = torch.ones((), device=device, dtype=torch.float32)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    expected = float(context.world_size)
    if value.item() != expected:
        raise RuntimeError(
            f"DDP smoke all-reduce returned {value.item()}, expected {expected}"
        )
    dist.barrier()
    if context.is_main:
        print(
            f"DDP smoke test PASS: backend={context.backend} "
            f"world_size={context.world_size} all_reduce={value.item():.0f}",
            flush=True,
        )


def _print_startup(
    *,
    context: DistributedContext,
    config: dict[str, Any],
    settings: DDPDataLoaderSettings,
    train_windows: int,
    samples_per_rank: int,
    train_loader_steps: int,
    model: torch.nn.Module,
    output_dir: Path,
    scheduler,
) -> None:
    train = config["train"]
    per_gpu_batch = int(train["batch_size"])
    global_batch = global_batch_size(per_gpu_batch, context.world_size)
    full_local_batches = samples_per_rank // per_gpu_batch
    local_remainder = samples_per_rank % per_gpu_batch
    final_local_batch = local_remainder or per_gpu_batch
    final_global_batch = final_local_batch * context.world_size
    print("distributed enabled:", True)
    print("world size:", context.world_size)
    print("backend:", context.backend)
    print("per-GPU batch size:", per_gpu_batch)
    print("global batch size:", global_batch)
    print("train windows:", train_windows)
    print("samples per rank per sampler epoch:", samples_per_rank)
    print("optimizer steps per sampler epoch:", train_loader_steps)
    print("full local batches per rank:", full_local_batches)
    print("final local batch size:", final_local_batch)
    print("nominal global batch size:", global_batch)
    print("final global batch size per full dataset pass:", final_global_batch)
    print("total steps:", train["steps"])
    print("warmup steps:", scheduler.warmup_steps)
    print("learning rate:", train["lr"])
    print("precision:", train["precision"])
    print("workers per rank:", settings.num_workers)
    print("torch CPU threads per rank:", settings.torch_num_threads)
    print("pin memory:", settings.pin_memory)
    print(
        "persistent workers:",
        settings.persistent_workers if settings.num_workers > 0 else False,
    )
    print(
        "prefetch factor:",
        settings.prefetch_factor if settings.num_workers > 0 else None,
    )
    print(
        "model parameters:",
        sum(parameter.numel() for parameter in model.parameters()),
    )
    print("output directory:", output_dir)
    print("git commit:", current_git_commit())
    print("resume supported:", False)


def train(config: dict[str, Any], context: DistributedContext) -> None:
    _validate_expected_world_size(config, context)
    settings = resolve_dataloader_settings(config)
    torch.set_num_threads(settings.torch_num_threads)
    device = torch.device("cuda", context.local_rank)
    precision = config["train"]["precision"]
    validate_precision_device(precision, device)

    train = config["train"]
    seed = int(config["experiment"]["seed"])
    per_gpu_batch = int(train["batch_size"])
    output_dir = Path(config["checkpoint"]["output_dir"])
    _require_empty_output_directory(output_dir, context)

    action_stats = _compute_and_broadcast_action_stats(config, context)
    train_dataset = build_dataset(
        config, split="train", action_stats=action_stats
    )
    val_dataset = (
        build_dataset(config, split="val", action_stats=action_stats)
        if context.is_main
        else None
    )

    train_sampler = build_distributed_train_sampler(
        train_dataset,
        world_size=context.world_size,
        rank=context.rank,
        seed=seed,
    )
    train_generator = torch.Generator(device="cpu")
    train_generator.manual_seed(seed + context.rank)
    train_loader = build_train_loader(
        train_dataset,
        sampler=train_sampler,
        per_gpu_batch_size=per_gpu_batch,
        settings=settings,
        generator=train_generator,
    )
    if len(train_loader) == 0:
        raise RuntimeError("distributed training loader has no full batches")

    val_loader = None
    if context.is_main:
        val_generator = torch.Generator(device="cpu")
        val_generator.manual_seed(seed + 10_000)
        val_loader = build_validation_loader(
            fixed_validation_subset(val_dataset, train["val_windows"]),
            per_gpu_batch_size=per_gpu_batch,
            settings=settings,
            generator=val_generator,
        )

    # Model initialization is deliberately identical on all ranks. DDP then
    # broadcasts rank 0 parameters/buffers before EMA is copied.
    _set_seed(seed)
    raw_model = build_model(config).to(device)
    ddp_model = DistributedDataParallel(
        raw_model,
        device_ids=[context.local_rank],
        output_device=context.local_rank,
        broadcast_buffers=True,
    )
    optimizer = build_optimizer(config, raw_model)
    scheduler = build_scheduler(config, optimizer)
    ema = (
        ExponentialMovingAverage(raw_model, train["ema"]["decay"])
        if train["ema"]["enabled"]
        else None
    )

    # Keep the sampler's common base seed, while giving each rank an independent
    # deterministic Flow Matching RNG stream for its disjoint local samples.
    _set_seed(seed + context.rank)

    name = config["experiment"]["name"]
    best_path = output_dir / f"{name}_best.pt"
    last_path = output_dir / f"{name}_last.pt"
    dataset_info = data_info(config) if context.is_main else None
    num_history = int(config["temporal"]["num_history"])
    global_batch = global_batch_size(per_gpu_batch, context.world_size)

    if context.is_main:
        _print_startup(
            context=context,
            config=config,
            settings=settings,
            train_windows=len(train_dataset),
            samples_per_rank=len(train_sampler),
            train_loader_steps=len(train_loader),
            model=raw_model,
            output_dir=output_dir,
            scheduler=scheduler,
        )

    wall_clock_started_at = time.perf_counter()
    ddp_model.train()
    step = 0
    epoch = 0
    samples_seen = 0
    best_val_loss = float("inf")
    while step < train["steps"]:
        train_sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            loss = causal_flow_loss(
                ddp_model,
                batch,
                device=device,
                num_history=num_history,
                precision=precision,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"rank {context.rank} produced non-finite loss at step {step + 1}"
                )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                raw_model.parameters(), train["grad_clip"]
            )
            update_lr = scheduler.get_last_lr()[0]
            optimizer.step()
            if ema is not None:
                ema.update(raw_model)
            scheduler.step()
            step += 1
            samples_this_step = int(batch["latents"].shape[0]) * context.world_size
            samples_seen += samples_this_step

            should_log = step == 1 or step % 25 == 0
            if should_log:
                global_loss = _distributed_mean(loss, context.world_size)
                if context.is_main:
                    elapsed = time.perf_counter() - wall_clock_started_at
                    equivalent_passes = samples_seen / float(len(train_dataset))
                    epoch_progress = epoch + (batch_index + 1) / len(train_loader)
                    print(
                        f"step={step:06d} train={global_loss.item():.6f} "
                        f"grad={grad_norm.detach().item():.4f} "
                        f"lr={update_lr:.8e} "
                        f"epoch={epoch_progress:.6f} "
                        f"global_samples_this_step={samples_this_step} "
                        f"samples_seen={samples_seen} "
                        f"dataset_passes={equivalent_passes:.6f} "
                        f"elapsed_wall_seconds={elapsed:.3f}",
                        flush=True,
                    )

            should_validate = (
                step % train["val_every"] == 0 or step == train["steps"]
            )
            if should_validate:
                # All ranks finish this optimizer update before rank 0 enters
                # validation; peers wait until rank 0 has also checkpointed.
                dist.barrier()
                if context.is_main:
                    assert val_loader is not None
                    assert dataset_info is not None
                    val_loss = evaluate_flow_loss(
                        validation_model(raw_model, ema),
                        val_loader,
                        device=device,
                        num_history=num_history,
                        seed=seed + 10_000,
                        precision=precision,
                    )
                    elapsed = time.perf_counter() - wall_clock_started_at
                    print(
                        f"step={step:06d} val={val_loss:.6f} "
                        f"elapsed_wall_seconds={elapsed:.3f}",
                        flush=True,
                    )
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        if config["checkpoint"]["save_best"]:
                            checkpoint = make_ddp_checkpoint(
                                raw_model=raw_model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                ema=ema,
                                step=step,
                                config=config,
                                action_stats=action_stats,
                                dataset_info=dataset_info,
                                best_val_loss=best_val_loss,
                                elapsed_wall_seconds=elapsed,
                                context=context,
                                per_gpu_batch_size=per_gpu_batch,
                                sampler_epoch=epoch,
                            )
                            save_checkpoint_on_rank_zero(
                                best_path,
                                checkpoint,
                                rank=context.rank,
                            )
                dist.barrier()

            if step >= train["steps"]:
                break
        epoch += 1

    dist.barrier()
    if context.is_main:
        assert dataset_info is not None
        elapsed = time.perf_counter() - wall_clock_started_at
        if config["checkpoint"]["save_last"]:
            checkpoint = make_ddp_checkpoint(
                raw_model=raw_model,
                optimizer=optimizer,
                scheduler=scheduler,
                ema=ema,
                step=step,
                config=config,
                action_stats=action_stats,
                dataset_info=dataset_info,
                best_val_loss=best_val_loss,
                elapsed_wall_seconds=elapsed,
                context=context,
                per_gpu_batch_size=per_gpu_batch,
                sampler_epoch=max(epoch - 1, 0),
            )
            save_checkpoint_on_rank_zero(
                last_path,
                checkpoint,
                rank=context.rank,
            )
        print("best_val_loss:", best_val_loss)
        print(f"elapsed_wall_seconds: {elapsed:.3f}")
        print(
            "best checkpoint:",
            best_path if config["checkpoint"]["save_best"] else None,
        )
        print(
            "last checkpoint:",
            last_path if config["checkpoint"]["save_last"] else None,
        )
    dist.barrier()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train causal-action Flow Matching DiT with torchrun DDP"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="initialize NCCL and verify one all-reduce without loading data/model",
    )
    args = parser.parse_args()

    config = load_and_resolve_config(args.config)
    distributed = config.get("distributed", {})
    backend = str(distributed.get("backend", "nccl"))
    context: DistributedContext | None = None
    try:
        context = initialize_distributed(backend)
        _validate_expected_world_size(config, context)
        device = torch.device("cuda", context.local_rank)
        if args.smoke_test:
            _run_smoke_test(context, device)
        else:
            train(config, context)
    finally:
        if context is not None and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
