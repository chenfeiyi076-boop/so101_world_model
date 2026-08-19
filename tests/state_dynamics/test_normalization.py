from __future__ import annotations

import torch

from src.state_dynamics.normalization import compute_train_stats
from .conftest import make_episode


def test_normalization_is_train_only_and_roundtrips():
    train = [make_episode(0, 6), make_episode(1, 7)]
    validation_extreme = make_episode(9, 8, offset=1e9)
    first = compute_train_stats(train)
    second = compute_train_stats(train)
    assert torch.equal(first.state_mean, second.state_mean)
    assert torch.equal(first.action_mean, second.action_mean)
    assert not torch.allclose(first.state_mean, validation_extreme.states.mean(0))
    assert first.state_mean.shape == first.state_std.shape == (6,)
    assert first.action_mean.shape == first.action_std.shape == (6,)
    states = train[0].states[:2]
    actions = train[0].actions[:2]
    assert torch.allclose(
        first.denormalize_states(first.normalize_states(states)), states, atol=1e-5
    )
    assert torch.isfinite(first.normalize_actions(actions)).all()
