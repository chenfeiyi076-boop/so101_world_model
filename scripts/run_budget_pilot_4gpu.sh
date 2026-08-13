#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${TMUX:-}" ]]; then
  echo "ERROR: run this launcher inside tmux." >&2
  echo "Example: tmux new -s so101-budget" >&2
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

experiment_root="/data/x2227/experiments/so101_budget_pilot"
log_root="$experiment_root/logs"

gpus=(0 1 2 3)
labels=(050k 100k 200k 300k)
configs=(
  "configs/causal/pilots/budget_stride4_chunk_050k.yaml"
  "configs/causal/pilots/budget_stride4_chunk_100k.yaml"
  "configs/causal/pilots/budget_stride4_chunk_200k.yaml"
  "configs/causal/pilots/budget_stride4_chunk_300k.yaml"
)
output_dirs=(
  "$experiment_root/050k"
  "$experiment_root/100k"
  "$experiment_root/200k"
  "$experiment_root/300k"
)

# Validate every target before starting any process, preventing partial launch
# when one run would overwrite an existing result.
for index in "${!labels[@]}"; do
  config="${configs[$index]}"
  output_dir="${output_dirs[$index]}"
  log_file="$log_root/${labels[$index]}.log"
  if [[ ! -f "$config" ]]; then
    echo "ERROR: config not found: $config" >&2
    exit 1
  fi
  if [[ -d "$output_dir" ]] && [[ -n "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "ERROR: refusing non-empty output directory: $output_dir" >&2
    exit 1
  fi
  if [[ -e "$log_file" ]]; then
    echo "ERROR: refusing to overwrite existing log: $log_file" >&2
    exit 1
  fi
done

mkdir -p "$log_root"

pids=()
for index in "${!labels[@]}"; do
  gpu="${gpus[$index]}"
  config="${configs[$index]}"
  log_file="$log_root/${labels[$index]}.log"
  nohup env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" \
    python scripts/train_causal.py --config "$config" --device cuda \
    >"$log_file" 2>&1 &
  pids+=("$!")
done

printf '%-5s %-56s %-68s %s\n' "GPU" "CONFIG" "LOG" "PID"
for index in "${!labels[@]}"; do
  printf '%-5s %-56s %-68s %s\n' \
    "${gpus[$index]}" \
    "${configs[$index]}" \
    "$log_root/${labels[$index]}.log" \
    "${pids[$index]}"
done

echo "Waiting for all four training processes inside tmux..."
overall_status=0
for index in "${!labels[@]}"; do
  if wait "${pids[$index]}"; then
    status=0
  else
    status=$?
    overall_status=1
  fi
  printf 'GPU %s (%s) exited with status %s; log: %s\n' \
    "${gpus[$index]}" \
    "${labels[$index]}" \
    "$status" \
    "$log_root/${labels[$index]}.log"
done

if (( overall_status != 0 )); then
  echo "One or more budget-pilot processes failed." >&2
  exit "$overall_status"
fi

echo "All four budget-pilot processes completed successfully."
