from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from scripts.evaluate_causal import select_evaluation_state_dict
from scripts.train_causal import fixed_subset as single_gpu_fixed_subset
from scripts.train_causal_ddp import (
    DDPDataLoaderSettings,
    DistributedContext,
    build_distributed_train_sampler,
    build_train_loader,
    checkpoint_has_ddp_prefix,
    dataloader_worker_kwargs,
    fixed_validation_subset,
    global_batch_size,
    make_ddp_checkpoint,
    save_checkpoint_on_rank_zero,
)
from src.causal.checkpointing import load_checkpoint, save_checkpoint
from src.causal.config import load_and_resolve_config, resolve_config
from src.causal.data.common import ActionStats
from src.causal.runtime import (
    ExponentialMovingAverage,
    build_model,
    build_optimizer,
    build_scheduler,
    scheduler_warmup_steps,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "causal" / "budget_stride4_chunk_ddp4_150k.yaml"
LAUNCHER_PATH = ROOT / "scripts" / "run_causal_ddp_4gpu.sh"


class IndexDataset(Dataset):
    def __init__(self, size: int):
        self.size = int(size)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        return int(index)


def _tiny_config() -> dict:
    return resolve_config(
        {
            "experiment": {"name": "ddp_checkpoint_test", "seed": 42},
            "data": {
                "mode": "single_episode",
                "episode_id": 0,
                "cache_path": "unused.pt",
                "split_ranges": {"train": [0, 20], "val": [20, 40]},
            },
            "temporal": {
                "num_frames": 4,
                "num_history": 2,
                "frame_stride": 1,
            },
            "action": {
                "raw_action_dim": 6,
                "alignment": "causal",
                "representation": "shifted_sampled",
                "normalize": True,
                "null_condition": "zero_embedding",
            },
            "model": {
                "in_channels": 2,
                "patch_size": 1,
                "hidden_size": 8,
                "depth": 1,
                "num_heads": 1,
                "mlp_ratio": 2.0,
                "use_qk_norm": True,
            },
            "flow_matching": {"future_only_loss": True},
            "train": {
                "batch_size": 2,
                "steps": 10,
                "precision": "fp32",
                "lr": 1e-4,
                "scheduler": {
                    "type": "warmup_cosine",
                    "warmup_ratio": 0.03,
                    "min_lr_ratio": 0.7,
                },
                "ema": {"enabled": True, "decay": 0.9995},
            },
            "distributed": {
                "backend": "nccl",
                "expected_world_size": 4,
            },
            "checkpoint": {"output_dir": "unused"},
            "latent": {
                "convention": "posterior_sample_times_scaling_no_shift"
            },
        }
    )


def _checkpoint_components():
    config = _tiny_config()
    model = build_model(config)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    ema = ExponentialMovingAverage(model, config["train"]["ema"]["decay"])
    stats = ActionStats(torch.zeros(6), torch.ones(6), source="ddp_test")
    context = DistributedContext(rank=0, local_rank=0, world_size=4, backend="nccl")
    checkpoint = make_ddp_checkpoint(
        raw_model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        step=3,
        config=config,
        action_stats=stats,
        dataset_info={"mode": "single_episode", "train_episode_ids": [0]},
        best_val_loss=0.25,
        elapsed_wall_seconds=12.0,
        context=context,
        per_gpu_batch_size=2,
        sampler_epoch=0,
    )
    return config, model, checkpoint


def test_ddp_checkpoint_state_dict_has_no_module_prefix():
    _, model, checkpoint = _checkpoint_components()

    class Wrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

    assert any(key.startswith("module.") for key in Wrapper(model).state_dict())
    assert checkpoint_has_ddp_prefix(checkpoint) is False
    assert not any(
        key.startswith("module.") for key in checkpoint["model_state_dict"]
    )
    assert not any(
        key.startswith("module.") for key in checkpoint["ema_model_state_dict"]
    )


def test_ddp_checkpoint_loads_with_existing_loader_and_evaluation_rules(tmp_path):
    config, _, checkpoint = _checkpoint_components()
    path = tmp_path / "ddp_v2.pt"
    save_checkpoint(path, checkpoint)
    loaded = load_checkpoint(path)
    assert loaded["checkpoint_version"] == 2
    assert loaded["distributed"] == {
        "enabled": True,
        "world_size": 4,
        "backend": "nccl",
        "per_gpu_batch_size": 2,
        "global_batch_size": 8,
        "sampler": "DistributedSampler",
        "sampler_epoch": 0,
    }
    assert loaded["config"]["temporal"] == config["temporal"]
    assert loaded["config"]["action"]["effective_action_dim"] == 6
    restored = build_model(loaded["config"])
    state_dict, selected = select_evaluation_state_dict(loaded, "auto")
    assert selected == "ema"
    restored.load_state_dict(state_dict, strict=True)


def test_global_batch_is_per_gpu_batch_times_world_size():
    assert global_batch_size(8, 4) == 32
    with pytest.raises(ValueError):
        global_batch_size(0, 4)


def _sampler_indices(size: int, *, epoch: int) -> list[list[int]]:
    dataset = IndexDataset(size)
    partitions = []
    for rank in range(4):
        sampler = build_distributed_train_sampler(
            dataset, world_size=4, rank=rank, seed=42
        )
        sampler.set_epoch(epoch)
        partitions.append(list(iter(sampler)))
    return partitions


def test_distributed_sampler_partitions_one_shared_epoch_without_overlap():
    partitions = _sampler_indices(32, epoch=0)
    assert all(len(partition) == 8 for partition in partitions)
    flattened = [index for partition in partitions for index in partition]
    assert len(flattened) == len(set(flattened)) == 32
    assert set(flattened) == set(range(32))
    assert all(set(left).isdisjoint(right) for i, left in enumerate(partitions) for right in partitions[i + 1 :])


def test_nondivisible_dataset_fails_instead_of_padding_duplicate_indices():
    with pytest.raises(ValueError, match="divisible by world_size"):
        build_distributed_train_sampler(
            IndexDataset(35), world_size=4, rank=0, seed=42
        )


def test_formal_dataset_uses_every_window_and_keeps_partial_final_batch():
    train_windows = 831_896
    world_size = 4
    per_gpu_batch = 8
    dataset = IndexDataset(train_windows)
    seen = bytearray(train_windows)
    loaders = []

    settings = DDPDataLoaderSettings(
        num_workers=0,
        torch_num_threads=4,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
    )
    for rank in range(world_size):
        sampler = build_distributed_train_sampler(
            dataset, world_size=world_size, rank=rank, seed=42
        )
        sampler.set_epoch(0)
        assert len(sampler) == 207_974
        for index in sampler:
            assert seen[index] == 0
            seen[index] = 1

        generator = torch.Generator(device="cpu")
        generator.manual_seed(42 + rank)
        loader = build_train_loader(
            dataset,
            sampler=sampler,
            per_gpu_batch_size=per_gpu_batch,
            settings=settings,
            generator=generator,
        )
        assert loader.drop_last is False
        assert len(loader) == 25_997
        loaders.append(loader)

    assert all(seen)
    assert [len(loader) for loader in loaders] == [25_997] * world_size
    batches = iter(loaders[0])
    last_batch = None
    for last_batch in batches:
        pass
    assert last_batch is not None
    assert len(last_batch) == 6
    assert 6 * world_size == 24


def test_sampler_set_epoch_changes_shuffle_deterministically():
    first = _sampler_indices(32, epoch=0)
    repeated = _sampler_indices(32, epoch=0)
    changed = _sampler_indices(32, epoch=1)
    assert first == repeated
    assert first != changed


def test_only_rank_zero_calls_checkpoint_writer(tmp_path):
    calls = []

    def record(path, checkpoint):
        calls.append((Path(path), checkpoint))

    checkpoint = {"value": 1}
    path = tmp_path / "checkpoint.pt"
    assert save_checkpoint_on_rank_zero(
        path, checkpoint, rank=1, save_fn=record
    ) is False
    assert calls == []
    assert save_checkpoint_on_rank_zero(
        path, checkpoint, rank=0, save_fn=record
    ) is True
    assert calls == [(path, checkpoint)]


def test_three_percent_warmup_for_ddp_budgets():
    assert scheduler_warmup_steps(150_000, 0.03) == 4_500
    assert scheduler_warmup_steps(300_000, 0.03) == 9_000


def test_rank_local_ema_is_exact_when_raw_models_and_update_counts_match():
    torch.manual_seed(42)
    left_raw = torch.nn.Linear(3, 2)
    right_raw = copy.deepcopy(left_raw)
    left_ema = ExponentialMovingAverage(left_raw, decay=0.9995)
    right_ema = ExponentialMovingAverage(right_raw, decay=0.9995)

    for update in range(1, 5):
        with torch.no_grad():
            for left_parameter, right_parameter in zip(
                left_raw.parameters(), right_raw.parameters()
            ):
                delta = torch.full_like(left_parameter, update * 0.01)
                left_parameter.add_(delta)
                right_parameter.add_(delta)
        left_ema.update(left_raw)
        right_ema.update(right_raw)

    assert left_ema.state_dict().keys() == right_ema.state_dict().keys()
    for key in left_ema.state_dict():
        assert torch.equal(
            left_ema.state_dict()[key], right_ema.state_dict()[key]
        )


def test_fixed_validation_subset_matches_single_gpu_definition():
    dataset = IndexDataset(1_000)
    ddp_subset = fixed_validation_subset(dataset, 256)
    single_subset = single_gpu_fixed_subset(dataset, 256)
    assert ddp_subset.indices == single_subset.indices
    assert len(ddp_subset) == 256


def test_num_workers_zero_omits_illegal_prefetch_and_persistent_options():
    settings = DDPDataLoaderSettings(
        num_workers=0,
        torch_num_threads=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    kwargs = dataloader_worker_kwargs(settings)
    assert kwargs == {"num_workers": 0, "pin_memory": True}


def test_formal_ddp_recipe_preserves_causal_model_and_training_semantics():
    config = load_and_resolve_config(CONFIG_PATH)
    train = config["train"]
    assert config["experiment"] == {
        "name": "budget_stride4_chunk_ddp4_150k",
        "seed": 42,
    }
    assert config["temporal"] == {
        "num_frames": 10,
        "num_history": 2,
        "frame_stride": 4,
    }
    assert config["action"]["alignment"] == "causal"
    assert config["action"]["representation"] == "fast_chunk"
    assert config["action"]["effective_action_dim"] == 24
    assert config["latent"]["convention"] == (
        "posterior_sample_times_scaling_no_shift"
    )
    assert config["model"] == {
        "in_channels": 16,
        "patch_size": 2,
        "hidden_size": 384,
        "depth": 12,
        "num_heads": 6,
        "mlp_ratio": 4.0,
        "use_qk_norm": True,
    }
    assert train["batch_size"] == 8
    assert train["steps"] == 150_000
    assert train["precision"] == "bf16"
    assert train["lr"] == 1e-4
    assert train["betas"] == [0.9, 0.99]
    assert train["eps"] == 1e-8
    assert train["weight_decay"] == 0.002
    assert train["grad_clip"] == 1.0
    assert train["scheduler"] == {
        "type": "warmup_cosine",
        "warmup_ratio": 0.03,
        "min_lr_ratio": 0.7,
    }
    assert scheduler_warmup_steps(
        train["steps"], train["scheduler"]["warmup_ratio"]
    ) == 4_500
    assert train["ema"] == {"enabled": True, "decay": 0.9995}
    assert train["val_every"] == 2_500
    assert train["val_windows"] == 256
    assert train["num_workers"] == 4
    assert config["distributed"] == {
        "backend": "nccl",
        "expected_world_size": 4,
        "torch_num_threads": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2,
    }


def test_launcher_is_torchrun_four_gpu_and_fail_fast():
    launcher = LAUNCHER_PATH.read_text(encoding="utf-8")
    assert "set -euo pipefail" in launcher
    assert "--nproc_per_node=4" in launcher
    assert "torchrun" in launcher
    assert "CUDA_VISIBLE_DEVICES" in launcher
    assert "refusing to overwrite non-empty output directory" in launcher
    assert "scripts/train_causal_ddp.py" in launcher
    assert 'output_dir="/data/' not in launcher
    assert "load_and_resolve_config" in launcher
    assert 'config["checkpoint"]["output_dir"]' in launcher


def test_yaml_checkpoint_output_dir_is_the_single_source_of_truth():
    config = load_and_resolve_config(CONFIG_PATH)
    assert config["checkpoint"]["output_dir"] == (
        "/data/x2227/experiments/so101_causal_ddp/ddp4_150k"
    )
