import torch

from src.data.dataset import CachedLatentDataset


CACHE_PATH = (
    "data/latent_cache/so101_front/"
    "episode_000.pt"
)


def test_cached_dataset():

    dataset = CachedLatentDataset(
        cache_path=CACHE_PATH,
        n_frames=10,
        frame_skip=1,
    )

    # 510 - 10 + 1
    assert len(dataset) == 501

    sample = dataset[0]

    assert sample["latents"].shape == (
        10,
        16,
        32,
        32,
    )

    assert sample["actions"].shape == (
        10,
        6,
    )

    assert sample["states"].shape == (
        10,
        6,
    )

    assert torch.equal(
        sample["indices"],
        torch.arange(10),
    )


def test_middle_window():

    dataset = CachedLatentDataset(
        cache_path=CACHE_PATH,
        n_frames=10,
        frame_skip=1,
    )

    sample = dataset[250]

    assert torch.equal(
        sample["indices"],
        torch.arange(
            250,
            260,
        ),
    )


def test_action_is_finite():

    dataset = CachedLatentDataset(
        cache_path=CACHE_PATH,
        n_frames=10,
        frame_skip=1,
    )

    sample = dataset[250]

    assert torch.isfinite(
        sample["actions"]
    ).all()

    assert torch.isfinite(
        sample["latents"]
    ).all()