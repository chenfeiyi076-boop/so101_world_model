from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("script", (
    "scripts/build_so101_state_split.py",
    "scripts/train_so101_state_mlp.py",
    "scripts/evaluate_so101_state_mlp.py",
))
def test_cli_help_without_dataset(script):
    result = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
