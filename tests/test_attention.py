import torch

from src.models.attention import (
    SpatialAttention,
    CausalTemporalAttention,
)


def test_spatial_attention_shape():

    torch.manual_seed(0)

    attn = SpatialAttention(
        dim=64,
        num_heads=2,
    )

    x = torch.randn(
        1,   # B
        4,   # T
        4,   # H
        4,   # W
        64,  # D
    )

    y = attn(x)

    assert y.shape == x.shape


def test_temporal_attention_shape():

    torch.manual_seed(0)

    attn = CausalTemporalAttention(
        dim=64,
        num_heads=2,
    )

    x = torch.randn(
        1,
        4,
        4,
        4,
        64,
    )

    y = attn(x)

    assert y.shape == x.shape


def test_temporal_attention_is_causal():

    torch.manual_seed(0)

    attn = CausalTemporalAttention(
        dim=64,
        num_heads=2,
    )

    attn.eval()

    x1 = torch.randn(
        1,
        4,
        2,
        2,
        64,
    )

    # 完全复制一次。
    x2 = x1.clone()

    # 只修改未来 t=2,3。
    x2[:, 2:] = torch.randn_like(
        x2[:, 2:]
    ) * 100.0

    with torch.no_grad():

        y1 = attn(x1)
        y2 = attn(x2)

    # --------------------------------------------------
    # 过去 t=0,1 不允许被未来 t=2,3 影响。
    # --------------------------------------------------

    assert torch.allclose(
        y1[:, :2],
        y2[:, :2],
        atol=1e-5,
        rtol=1e-5,
    )

    # --------------------------------------------------
    # 未来本身应该不同。
    # --------------------------------------------------

    assert not torch.allclose(
        y1[:, 2:],
        y2[:, 2:],
    )


def test_spatial_attention_does_not_mix_frames():

    torch.manual_seed(0)

    attn = SpatialAttention(
        dim=64,
        num_heads=2,
    )

    attn.eval()

    x1 = torch.randn(
        1,
        4,
        2,
        2,
        64,
    )

    x2 = x1.clone()

    # 只修改 frame 3。
    x2[:, 3] = (
        torch.randn_like(
            x2[:, 3]
        )
        * 100.0
    )

    with torch.no_grad():

        y1 = attn(x1)
        y2 = attn(x2)

    # Spatial attention 是逐帧独立的，
    # 所以前三个 frame 不应发生任何变化。

    assert torch.allclose(
        y1[:, :3],
        y2[:, :3],
        atol=1e-5,
        rtol=1e-5,
    )

    # frame 3 应该变化。
    assert not torch.allclose(
        y1[:, 3],
        y2[:, 3],
    )


def test_attention_backward():

    torch.manual_seed(0)

    spatial = SpatialAttention(
        dim=64,
        num_heads=2,
    )

    temporal = CausalTemporalAttention(
        dim=64,
        num_heads=2,
    )

    x = torch.randn(
        1,
        4,
        2,
        2,
        64,
        requires_grad=True,
    )

    y = spatial(x)
    y = temporal(y)

    loss = y.pow(2).mean()

    loss.backward()

    assert x.grad is not None

    assert torch.isfinite(
        x.grad
    ).all()