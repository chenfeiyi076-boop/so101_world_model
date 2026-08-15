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

## Autoregressive rollout visualization

`visualize_rollout.py` reads an existing rollout evaluation directory and
deterministically reruns only the selected stochastic streams; it does not
rerun the full evaluation or require saved latents. By default it selects
representative q10, q50, and q90 streams by mean per-step rollout MSE. Explicit
`--stream EPISODE_ID:START:NOISE_DRAW` arguments can be used instead.

The prediction shown is the predicted latent decoded through the frozen VAE.
The reference labelled **GT latent reconstruction** is the ground-truth cached
latent decoded through that same VAE. It is not the original RGB ground truth.
Both paths use the cache convention

```text
z_decode = z_cache / scaling_factor
```

and never apply the VAE config's `shift_factor`. Contact-sheet time labels are
future offsets relative to the last ground-truth history frame.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/visualize_rollout.py \
  --rollout-eval-dir /path/to/rollout_eval \
  --vae-path /path/to/sd3-repository-or-vae \
  --output-dir /path/to/rollout_visualization \
  --selection-metric mean_mse \
  --quantiles 0.1 0.5 0.9 \
  --display-steps 1 4 8 16 32 \
  --decode-batch-size 8 \
  --device cuda
```

`--vae-path` accepts either a complete local Diffusers repository containing
`vae/config.json` or the VAE directory itself. Model loading is local-only.

## Action controllability / action-shuffling negative control

`evaluate_action_controllability.py` reuses the checkpoint, cases, noise draws,
seed, Euler steps, and rollout horizon recorded by an existing rollout
evaluation. For each stochastic stream it runs a factual-action branch and a
temporally shuffled-action branch with exactly the same initial Gaussian noise.
Only future `[frame_stride, raw_action_dim]` chunks are permuted; the observed
history transition and the order within every chunk remain unchanged. The
permutation is a deterministic case-level derangement shared by all noise draws
for that `(episode_id, start)` case.

The shuffled branch has no counterfactual ground truth. Comparing its prediction
with the factual future therefore measures factual-consistency degradation under
an action-corruption negative control, not counterfactual prediction accuracy.
Prediction divergence between the two branches measures action sensitivity, but
does not by itself establish causal correctness. Confidence intervals resample
physical `(episode_id, start)` cases after averaging their noise draws; stochastic
draws are not treated as independent physical cases.

The summary separates TRUE-branch rerun tolerance diagnostics:
`true_rerun_metric_warning_count` counts individual metric comparisons above
tolerance, while `true_rerun_streams_with_warning` counts stochastic streams
with at least one such comparison. Identity and target-frame mismatches remain
hard failures.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_action_controllability.py \
  --source-rollout-eval-dir /path/to/300k_val_R32_N128_D4 \
  --output-dir /path/to/300k_val_action_shuffle \
  --shuffle-seed 30360 \
  --pair-batch-size 4 \
  --bootstrap-samples 10000 \
  --bootstrap-seed 4242 \
  --device cuda
```
