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

## Independent SO101 state-dynamics MLP

The state-dynamics pipeline is separate from the visual causal world model. It
does not read video, VAE latents, or causal cache data and does not import the
causal Dataset. At raw 20 Hz, one sample maps current robot state and four
consecutive actions to four absolute future states:

```text
state[t] [6] + action[t:t+4] [4,6]
    -> MLP(30 -> 256 -> 256 -> 256 -> 24, SiLU)
    -> state[t+1:t+5] [4,6]
```

Windows are built only after an episode-level, task-stratified split. An episode
of length `L` contributes exactly `L - 4` stride-one windows and no window can
cross an episode boundary. The independent seed-42 split uses floor-80%,
floor-10%, and remainder per task; the formal 2,499-episode release must resolve
to 1,999 train, 248 validation, and 252 test episodes. State and action z-score
statistics are computed from train episodes only and stored in each checkpoint.

The formal recipe is FP32 single-GPU AdamW (`lr=1e-3`, `weight_decay=1e-4`,
batch 4096, up to 50 epochs, early-stopping patience 8). Every epoch evaluates
the complete teacher-forced validation set. `best.pt` is selected by validation
normalized absolute-state MSE and `last.pt` is always updated. Training and its
smoke mode read only train/validation parquet rows; test episode IDs may be
reported from the manifest, but test numerical data is loaded only by the
evaluation entry point.

First inspect the real release and build the independent split. The inspect
output reports the parquet schema, episode/task counts, action/state shapes,
frame-index range, and timestamp diagnostics without opening video:

```bash
python scripts/build_so101_state_split.py \
  --dataset-root /data/x2227/datasets/armnetbench_v01_lerobot_so101 \
  --inspect-only

python scripts/build_so101_state_split.py \
  --dataset-root /data/x2227/datasets/armnetbench_v01_lerobot_so101 \
  --output data/state_dynamics/so101_state_seed42_split.json \
  --seed 42 \
  --causal-manifest /data/x2227/so101_world_model/data/causal/armnetbench_so101_seed42_manifest.json
```

For the formal experiment, `--causal-manifest` is mandatory operationally: do
not start training until this command prints
`causal manifest compatibility: PASS`. The state pipeline remains independently
implemented; this check proves exact train/val/test episode-ID compatibility
with the frozen visual-world-model experiment.

Run a bounded real-data forward/backward check before formal training. It reads
at most four episodes from each split and writes no checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_so101_state_mlp.py \
  --config configs/state_dynamics/so101_mlp.yaml \
  --dataset-root /data/x2227/datasets/armnetbench_v01_lerobot_so101 \
  --split-manifest data/state_dynamics/so101_state_seed42_split.json \
  --output-dir /data/x2227/experiments/so101_state_mlp/smoke \
  --smoke-test --max-episodes 4
```

Formal training remains a single-GPU job:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python scripts/train_so101_state_mlp.py \
  --config configs/state_dynamics/so101_mlp.yaml \
  --dataset-root /data/x2227/datasets/armnetbench_v01_lerobot_so101 \
  --split-manifest data/state_dynamics/so101_state_seed42_split.json \
  --output-dir /data/x2227/experiments/so101_state_mlp/mlp256_abs
```

Evaluation reports teacher-forced raw-state MAE/RMSE overall, by 20 Hz horizon,
and by joint. It also performs a 5 Hz autoregressive rollout: only the first GT
state is used, each predicted `t+4` endpoint feeds the next step, and future GT
states are used only after rollout for metrics. GT actions remain inputs. A
command-copy baseline compares `action[t+3]` directly with `state[t+4]`.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_so101_state_mlp.py \
  --checkpoint /data/x2227/experiments/so101_state_mlp/mlp256_abs/best.pt \
  --dataset-root /data/x2227/datasets/armnetbench_v01_lerobot_so101 \
  --split test \
  --output-dir /data/x2227/experiments/so101_state_mlp/mlp256_abs/test
```

Outputs include `teacher_forced_summary.json`, per-horizon/per-joint CSVs,
`autoregressive_summary.json`, `autoregressive_per_episode.csv`, the existing
available-case per-step/per-joint/time CSVs, and
`autoregressive_error_vs_time.png`. The summary labels metrics explicitly as
endpoint-micro or episode-macro; legacy endpoint metric names remain aliases.
Endpoint-micro metrics pool all available endpoint/joint errors, so longer
episodes contribute more endpoints. Per-episode means first average that
episode's available rollout steps, and episode-macro metrics then weight every
physical episode equally.
