# Causal action experiments

These configs use the independent `src/causal` path and do not change legacy
experiments. Train with:

```bash
python scripts/train_causal.py --config configs/causal/multi_stride4_chunk.yaml
```

Evaluate a checkpoint with checkpoint-owned configuration and normalization:

```bash
python scripts/evaluate_causal.py \
  --checkpoint checkpoints/causal/causal_multi_stride4_fast_chunk_best.pt
```

Passing `--config` to the evaluator only performs a strict compatibility
check. It never overrides the checkpoint configuration or action statistics.

Normalization has one source of truth:

- training computes one raw-6D mean/std from the train split only;
- train and validation datasets receive those exact statistics;
- checkpoints save both their values and provenance;
- evaluation loads them from the checkpoint and never recomputes them;
- `fast_chunk` normalizes each raw 6D action with that same mean/std before
  flattening to 12D or 24D.

## Autoregressive latent rollout evaluation

`evaluate_causal.py` reports noised-ground-truth Flow Matching objective loss
on fixed val/test windows. In contrast, `evaluate_rollout.py` starts each
future latent from pure Gaussian noise and integrates the existing Flow
Matching Euler sampler from tau 1 to 0. Only the initial history latents are
ground truth; every generated future is fed back as context for the next
prediction. Future ground-truth latents are read only after generation for
post-hoc latent-error metrics.

When a rollout grows beyond the training window, the evaluator keeps the most
recent `num_frames - 1` clean context latents and appends one Gaussian target,
so the temporal input never exceeds the checkpoint's trained `num_frames`.
Actions remain the real causal action chunks from the selected val/test
trajectory, and the first slot of every sliding window is explicit NULL.

`--batch-size` controls how many independent `(rollout case, noise draw)`
streams advance through the same future step together on the GPU. For example,
`--batch-size 8` advances eight independent streams in parallel; time steps
within each individual autoregressive rollout remain strictly sequential.

With `--threshold-metric mse_p90`, the supported-horizon threshold is applied
to the 90th percentile of per-rollout latent MSE at each autoregressive step.
As with the mean-error thresholds, the first crossing ends the supported
continuous prefix even if a later step falls below the threshold again.

```bash
python scripts/evaluate_rollout.py \
  --checkpoint /path/to/causal_checkpoint.pt \
  --split test \
  --rollout-steps 32 \
  --output-dir /path/to/rollout_eval
```

The reported MSE, RMSE, MAE, relative L2, and cosine similarity are latent-space
prediction errors. They are not pixel-space or perceptual image-quality metrics.
