# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the jagged-sweep benchmark of the XPU jagged operators."""

import csv
import subprocess  # nosec B404 - runs this repo's own module with fixed args
import sys

import pytest
import torch
from click.testing import CliRunner
from fbgemm_xpu.bench.jagged_sweep import SWEEP_FAMILIES, jagged_sweep

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="requires an XPU device"
)


def test_writes_metadata_and_one_row_per_case(tmp_path):
    output = tmp_path / "sweep.csv"
    result = CliRunner().invoke(
        jagged_sweep,
        [
            "--batch-sizes=2,3",
            "--max-lens=5",
            "--embedding-dim=8",
            "--dtypes=float32,float16",
            "--iters=1",
            f"--output={output}",
        ],
    )
    assert result.exit_code == 0, result.output  # nosec B101

    lines = output.read_text().splitlines()
    metadata = dict(
        line[2:].split(": ", 1) for line in lines if line.startswith("# ")
    )
    assert metadata["batch_sizes"] == "2,3"  # nosec B101
    assert metadata["device"] == torch.xpu.get_device_name()  # nosec B101
    assert "so the GPU does not wait for the host" in metadata["timing"]  # nosec B101

    rows = list(csv.DictReader(line for line in lines if not line.startswith("#")))
    assert len(rows) == 2 * 1 * 2 * len(SWEEP_FAMILIES) * 2  # nosec B101
    assert {r["family"] for r in rows} == set(SWEEP_FAMILIES)  # nosec B101
    assert {r["direction"] for r in rows} == {"fwd", "bwd"}  # nosec B101
    assert {r["dtype"] for r in rows} == {"float32", "float16"}  # nosec B101
    for r in rows:
        assert float(r["time_us"]) > 0  # nosec B101
        assert int(r["bytes"]) > 0  # nosec B101


def test_rejects_unknown_family(tmp_path):
    result = CliRunner().invoke(
        jagged_sweep, ["--families=nope", f"--output={tmp_path / 'x.csv'}"]
    )
    assert result.exit_code != 0  # nosec B101
    assert not (tmp_path / "x.csv").exists()  # nosec B101


def test_runs_as_module(tmp_path):
    output = tmp_path / "smoke.csv"
    result = subprocess.run(  # nosec B603 - sys.executable with a fixed arg list
        [
            sys.executable,
            "-m",
            "fbgemm_xpu.bench.jagged_sweep",
            "--smoke",
            "--families=dense_to_jagged",
            "--dtypes=float32",
            f"--output={output}",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr  # nosec B101
    assert "family=dense_to_jagged" in result.stderr  # nosec B101
    assert output.read_text().startswith(  # nosec B101
        "# command: python -m fbgemm_xpu.bench.jagged_sweep --smoke"
    )
