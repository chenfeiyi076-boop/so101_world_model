import torch
from torch.utils.data import DataLoader

from src.data.dataset import DummyWorldModelDataset


def test_dataset_single_sample():

    dataset = DummyWorldModelDataset(
        num_episodes=2,
        episode_length=20,
        n_frames=4,
        frame_skip=1,
        action_dim=6,
        windows_per_episode=5,
    )

    sample = dataset[0]

    assert sample["latents"].shape == (
        4,
        16,
        32,
        32,
    )

    assert sample["actions"].shape == (
        4,
        6,
    )

    assert torch.equal(
        sample["indices"],
        torch.tensor([0, 1, 2, 3]),
    )


def test_dataset_frame_skip():

    dataset = DummyWorldModelDataset(
        num_episodes=1,
        episode_length=20,
        n_frames=4,
        frame_skip=2,
        action_dim=6,
        windows_per_episode=5,
    )

    sample = dataset[2]

    assert torch.equal(
        sample["indices"],
        torch.tensor([2, 4, 6, 8]),
    )

    assert sample["latents"].shape == (
        4,
        16,
        32,
        32,
    )


def test_dataloader_batch():

    dataset = DummyWorldModelDataset(
        num_episodes=2,
        episode_length=20,
        n_frames=4,
        frame_skip=1,
        action_dim=6,
        windows_per_episode=5,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )

    batch = next(iter(loader))

    assert batch["latents"].shape == (
        2,
        4,
        16,
        32,
        32,
    )

    assert batch["actions"].shape == (
        2,
        4,
        6,
    )

    assert batch["indices"].shape == (
        2,
        4,
    )