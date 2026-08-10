import torch

from src.diffusion.flow_matching import (
    prepare_flow_matching_batch,
    flow_matching_loss,
)


def test_flow_matching_shapes():

    torch.manual_seed(0)

    latents = torch.randn(
        2,
        4,
        16,
        8,
        8,
    )

    batch = prepare_flow_matching_batch(
        latents=latents,
        num_history=2,
    )

    assert batch.noisy_latents.shape == latents.shape

    assert batch.target_velocity.shape == latents.shape

    assert batch.noise.shape == latents.shape

    assert batch.tau.shape == (
        2,
        4,
    )

    assert batch.loss_mask.shape == (
        2,
        4,
    )


def test_history_is_clean():

    torch.manual_seed(0)

    latents = torch.randn(
        1,
        4,
        16,
        8,
        8,
    )

    batch = prepare_flow_matching_batch(
        latents=latents,
        num_history=2,
        history_noise_std=0.0,
    )

    assert torch.equal(
        batch.noisy_latents[:, :2],
        latents[:, :2],
    )


def test_history_tau_is_zero():

    torch.manual_seed(0)

    latents = torch.randn(
        2,
        4,
        16,
        8,
        8,
    )

    batch = prepare_flow_matching_batch(
        latents=latents,
        num_history=2,
    )

    assert torch.equal(
        batch.tau[:, :2],
        torch.zeros_like(
            batch.tau[:, :2]
        ),
    )


def test_future_tau_range():

    torch.manual_seed(0)

    latents = torch.randn(
        2,
        10,
        16,
        4,
        4,
    )

    batch = prepare_flow_matching_batch(
        latents=latents,
        num_history=2,
    )

    future_tau = batch.tau[:, 2:]

    assert torch.all(
        future_tau >= 0.0
    )

    assert torch.all(
        future_tau <= 1.0
    )


def test_loss_mask():

    latents = torch.randn(
        1,
        4,
        16,
        4,
        4,
    )

    batch = prepare_flow_matching_batch(
        latents=latents,
        num_history=2,
    )

    expected = torch.tensor(
        [[False, False, True, True]]
    )

    assert torch.equal(
        batch.loss_mask.cpu(),
        expected,
    )


def test_velocity_target():

    torch.manual_seed(0)

    latents = torch.randn(
        1,
        4,
        16,
        4,
        4,
    )

    batch = prepare_flow_matching_batch(
        latents=latents,
        num_history=2,
    )

    expected = (
        batch.noise - latents
    )

    assert torch.allclose(
        batch.target_velocity,
        expected,
    )


def test_perfect_prediction_zero_loss():

    torch.manual_seed(0)

    latents = torch.randn(
        1,
        4,
        16,
        4,
        4,
    )

    batch = prepare_flow_matching_batch(
        latents=latents,
        num_history=2,
    )

    prediction = (
        batch.target_velocity.clone()
    )

    loss = flow_matching_loss(
        prediction=prediction,
        target_velocity=batch.target_velocity,
        loss_mask=batch.loss_mask,
    )

    assert torch.allclose(
        loss,
        torch.tensor(0.0),
        atol=1e-7,
    )


def test_history_does_not_affect_loss():

    torch.manual_seed(0)

    latents = torch.randn(
        1,
        4,
        16,
        4,
        4,
    )

    batch = prepare_flow_matching_batch(
        latents=latents,
        num_history=2,
    )

    prediction1 = torch.zeros_like(
        latents
    )

    prediction2 = prediction1.clone()

    # --------------------------------------------------
    # 只把 history prediction 改成极大的错误值
    # --------------------------------------------------

    prediction2[:, :2] = 1_000_000.0

    loss1 = flow_matching_loss(
        prediction1,
        batch.target_velocity,
        batch.loss_mask,
    )

    loss2 = flow_matching_loss(
        prediction2,
        batch.target_velocity,
        batch.loss_mask,
    )

    # history 不参与 loss，所以两个 loss 必须相同。
    assert torch.allclose(
        loss1,
        loss2,
    )