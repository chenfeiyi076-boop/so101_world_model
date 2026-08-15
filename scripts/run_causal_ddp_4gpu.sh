#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${TMUX:-}" ]]; then
  echo "ERROR: run the formal DDP launcher inside tmux." >&2
  echo "Example: tmux new -s so101-ddp4" >&2
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config="configs/causal/budget_stride4_chunk_ddp4_150k.yaml"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f "$config" ]]; then
  echo "ERROR: config not found: $config" >&2
  exit 1
fi
if ! command -v torchrun >/dev/null 2>&1; then
  echo "ERROR: torchrun is not available in PATH." >&2
  exit 1
fi
if ! output_dir="$(
  python - "$config" <<'PY'
import sys

from src.causal.config import load_and_resolve_config

config = load_and_resolve_config(sys.argv[1])
output_dir = config["checkpoint"]["output_dir"]
if not isinstance(output_dir, str) or not output_dir.strip():
    raise RuntimeError("checkpoint.output_dir must be a non-empty string")
print(output_dir)
PY
)"; then
  echo "ERROR: failed to read checkpoint.output_dir from config: $config" >&2
  exit 1
fi
if [[ -z "${output_dir//[[:space:]]/}" ]]; then
  echo "ERROR: checkpoint.output_dir resolved to an empty value: $config" >&2
  exit 1
fi
if [[ -d "$output_dir" ]] && [[ -n "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "ERROR: refusing to overwrite non-empty output directory: $output_dir" >&2
  exit 1
fi

command=(
  torchrun
  --standalone
  --nproc_per_node=4
  scripts/train_causal_ddp.py
  --config "$config"
)

echo "config=$config"
echo "output_dir=$output_dir"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
printf 'command:'
printf ' %q' "${command[@]}"
printf '\n'

exec "${command[@]}"
