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
