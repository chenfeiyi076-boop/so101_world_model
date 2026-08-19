from __future__ import annotations

import pytest
import torch

from src.state_dynamics.model import StateDynamicsMLP, parameter_counts


def test_formal_model_shape_and_parameter_count():
    model = StateDynamicsMLP()
    output = model(torch.randn(5, 6), torch.randn(5, 4, 6))
    assert output.shape == (5, 4, 6)
    assert parameter_counts(model) == (145688, 145688)


def test_model_rejects_wrong_action_shape():
    with pytest.raises(ValueError, match="actions"):
        StateDynamicsMLP()(torch.randn(2, 6), torch.randn(2, 3, 6))
