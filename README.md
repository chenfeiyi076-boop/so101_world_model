# SO101 causal world model

## Full-test long-horizon rollout and decoded-RGB fidelity

The formal long-horizon evaluator is deliberately split into two stages so the
world model is run only once. Stage A performs one earliest-history rollout per
test episode and stochastic draw, saves every predicted latent as a resumable
BF16 artifact, and records the existing latent MSE/relative-L2/cosine metrics.
Stage B never loads the DiT: it reads those artifacts, decodes prediction and GT
cache latents with the frozen SD3 no-shift convention, reads aligned raw front
RGB, and computes per-draw SSIM and AlexNet LPIPS.

Stage A supports episode/data parallelism rather than DDP model parallelism:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  scripts/evaluate_long_rollout.py \
  --checkpoint /path/to/checkpoint.pt \
  --output-dir /data/x2227/experiments/full_test_long_rollout \
  --split test --noise-draws 4 --seed 20260 --euler-steps 10 \
  --device cuda --weights ema
```

Omitting `--max-rollout-steps` continues each episode until its final available
GT sampled frame. Every `episode x draw` artifact is written through a temporary
file and atomic rename. A complete matching artifact is skipped on resume;
provenance mismatch is a hard error unless `--overwrite` is explicit.

Stage B resume metadata binds every exact Stage A draw file by SHA256 and binds
both the VAE config and all VAE weight files. `--aggregate-only` also verifies
that its Stage A directory, Stage A summary SHA, checkpoint, manifest, weights,
seed, Euler/temporal protocol, and draw counts match the completed Stage B
experiment before reading derived CSVs.

Stage B can also use four GPUs with episode-level sharding:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  scripts/evaluate_rollout_rgb.py \
  --long-rollout-dir /data/x2227/experiments/full_test_long_rollout \
  --vae-path /path/to/stable-diffusion-3-medium-diffusers \
  --output-dir /data/x2227/experiments/full_test_long_rollout_rgb \
  --decode-batch-size 8 --metric-batch-size 16 --device cuda
```

Stage B runtime dependencies are explicit and are never installed by the
evaluation script: `lpips`, `matplotlib`, `torch`, `torchvision`, `diffusers`,
and the existing PyAV/pandas/pyarrow LeRobot reader dependencies must already be
available. Missing dependencies fail fast.

### Output artifacts

Stage A produces the expensive, reusable source artifacts:

```text
full_test_long_rollout/
|-- latents/
|   |-- episode_XXXX_draw_0.pt
|   |-- episode_XXXX_draw_1.pt
|   `-- ...
|-- per_draw_step_latent.csv
`-- summary.json
```

Stage B produces resumable episode metrics and cheap derived tables/plots:

```text
full_test_long_rollout_rgb/
|-- episode_metrics/
|   |-- episode_XXXX.pt
|   `-- ...
|-- per_draw_step.csv
|-- per_episode_step.csv
|-- per_step_available.csv
|-- per_step_fixed_H032.csv
|-- per_step_fixed_H050.csv
|-- per_step_fixed_H100.csv
|-- ...
|-- ssim_gap_available.png
|-- lpips_gap_available.png
|-- latent_mse_available.png
|-- n_episodes_available.png
|-- ssim_gap_fixed_cohorts.png
|-- lpips_gap_fixed_cohorts.png
`-- summary.json
```

Keep `latents/*.pt`: they are the expensive Stage A source of truth. Fixed-H
CSVs and plots are derived artifacts and can be regenerated with
`--aggregate-only`.

Both prediction and GT reconstruction are compared with the same raw RGB:

- `ssim_gap = recon_ssim - pred_ssim` (higher is worse)
- `lpips_gap = pred_lpips - recon_lpips` (higher is worse)

Metrics are computed independently for every stochastic draw and then averaged;
prediction latents or decoded RGB images are never averaged before scoring. The
available-case curves always include `n_episodes(t)`. Fixed-cohort curves use one
unchanging set of physical episodes for every step through horizon `H`. Bootstrap
confidence intervals resample physical episodes, not the four stochastic draws.

Changing fixed horizons or regenerating plots does not run DiT or VAE:

```bash
python scripts/evaluate_rollout_rgb.py \
  --long-rollout-dir /data/x2227/experiments/full_test_long_rollout \
  --output-dir /data/x2227/experiments/full_test_long_rollout_rgb \
  --aggregate-only --fixed-horizons 32 50 100 150 200
```

### Required real-data smoke before full test

Do not start all 252 test episodes first. Run Stage A with two episodes, two
draws, and at most eight future steps:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python scripts/evaluate_long_rollout.py \
  --checkpoint "$CKPT" --output-dir "$LONG_SMOKE" --split test \
  --max-episodes 2 --noise-draws 2 --max-rollout-steps 8 \
  --seed 20260 --euler-steps 10 --device cuda --weights ema
```

Then run the real local SD3 VAE, ArmNetBench raw video, and AlexNet LPIPS:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python scripts/evaluate_rollout_rgb.py \
  --long-rollout-dir "$LONG_SMOKE" --vae-path "$VAE_PATH" \
  --output-dir "$RGB_SMOKE" --decode-batch-size 8 \
  --metric-batch-size 16 --fixed-horizons 4 8 --device cuda
```

Before the full run, inspect `per_draw_step.csv`, `per_episode_step.csv`,
`per_step_available.csv`, `summary.json`, both gap plots, and at least one
aligned Original RGB / GT latent reconstruction / predicted RGB example using
the existing visualization tooling. Confirm that all three refer to the same
physical frame.
