from __future__ import annotations

import argparse
from pathlib import Path

import torch


RUNS = (
    ("050k", "budget_stride4_chunk_050k"),
    ("100k", "budget_stride4_chunk_100k"),
    ("200k", "budget_stride4_chunk_200k"),
    ("300k", "budget_stride4_chunk_300k"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize SO101 budget-pilot checkpoints without selecting a budget."
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("/data/x2227/experiments/so101_budget_pilot"),
    )
    return parser


def _load_optional(path: Path) -> tuple[dict | None, str | None]:
    if not path.is_file():
        return None, None
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:  # Keep the other pilot rows readable.
        return None, f"{type(error).__name__}: {error}"
    if not isinstance(value, dict):
        return None, "checkpoint is not a mapping"
    return value, None


def summarize_run(experiment_root: Path, label: str, name: str) -> dict[str, object]:
    run_dir = experiment_root / label
    best_path = run_dir / f"{name}_best.pt"
    last_path = run_dir / f"{name}_last.pt"
    best, best_error = _load_optional(best_path)
    last, last_error = _load_optional(last_path)
    source = last or best
    if source is None:
        detail = best_error or last_error or "missing"
        return {
            "budget": label,
            "best_step": "missing",
            "best_val": "missing",
            "last_step": "missing",
            "wall_hours": "missing",
            "precision": "-",
            "batch": "-",
            "warmup": "-",
            "scheduler": "-",
            "ema_decay": "-",
            "git_commit": "-",
            "status": detail,
        }

    config = source.get("config", {})
    train = config.get("train", {})
    scheduler_config = train.get("scheduler", {})
    ema_config = train.get("ema", {})
    budget_steps = train.get("steps", "unknown")
    scheduler_state = source.get("scheduler_state_dict", {})
    warmup_steps = scheduler_state.get("warmup_steps", "missing")
    best_step = best.get("step", "missing") if best else "missing"
    best_val = (
        best.get("metrics", {}).get("best_val_loss", "missing")
        if best
        else "missing"
    )
    last_step = last.get("step", "missing") if last else "missing"
    elapsed_wall_seconds = (
        last.get("metrics", {}).get("elapsed_wall_seconds") if last else None
    )
    wall_hours = (
        elapsed_wall_seconds / 3600
        if isinstance(elapsed_wall_seconds, (int, float))
        else "missing"
    )
    complete = isinstance(last_step, int) and last_step == budget_steps
    errors = [error for error in (best_error, last_error) if error]
    if errors:
        status = "; ".join(errors)
    elif not best:
        status = "best missing"
    elif not last:
        status = "last missing/incomplete"
    elif not complete:
        status = f"incomplete ({last_step}/{budget_steps})"
    else:
        status = "complete"
    return {
        "budget": budget_steps,
        "best_step": best_step,
        "best_val": best_val,
        "last_step": last_step,
        "wall_hours": wall_hours,
        "precision": train.get("precision", "fp32"),
        "batch": train.get("batch_size", "missing"),
        "warmup": warmup_steps,
        "scheduler": scheduler_config.get("type", "constant"),
        "ema_decay": ema_config.get("decay", "disabled"),
        "git_commit": source.get("git_commit", "unknown"),
        "status": status,
    }


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def print_table(rows: list[dict[str, object]]) -> None:
    columns = (
        ("budget", "budget"),
        ("best_step", "best_step"),
        ("best_val", "best_val"),
        ("last_step", "last_step"),
        ("wall_hours", "wall_hours"),
        ("precision", "precision"),
        ("batch", "batch"),
        ("warmup", "warmup"),
        ("scheduler", "scheduler"),
        ("ema_decay", "ema_decay"),
        ("git_commit", "git_commit"),
        ("status", "status"),
    )
    widths = {
        key: max(len(header), *(len(_format_value(row[key])) for row in rows))
        for key, header in columns
    }
    print(" | ".join(header.ljust(widths[key]) for key, header in columns))
    print("-+-".join("-" * widths[key] for key, _ in columns))
    for row in rows:
        print(
            " | ".join(
                _format_value(row[key]).ljust(widths[key]) for key, _ in columns
            )
        )


def main() -> None:
    args = build_parser().parse_args()
    rows = [
        summarize_run(args.experiment_root, label, name)
        for label, name in RUNS
    ]
    print_table(rows)
    print("No budget is selected automatically; compare validation convergence manually.")


if __name__ == "__main__":
    main()
