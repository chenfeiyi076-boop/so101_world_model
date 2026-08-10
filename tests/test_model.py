import torch

from src.models.dit import DiT


model = DiT(
    in_channels=16,
    patch_size=2,
    hidden_size=64,
    depth=2,
    num_heads=2,
    action_dim=6,
)

x = torch.randn(
    1,
    4,
    16,
    32,
    32,
)

tau = torch.rand(
    1,
    4,
)

actions = torch.randn(
    1,
    4,
    6,
)

with torch.no_grad():
    y = model(
        x,
        tau,
        actions,
    )

print("input :", x.shape)
print("output:", y.shape)

print(
    "patch:",
    model.patchify(x).shape,
)