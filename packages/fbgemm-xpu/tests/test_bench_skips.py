# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the skips used by the patched FBGEMM jagged tensor benchmark."""

import logging

import pytest
from fbgemm_xpu.bench.skips import (
    XPU_SKIPPED_SUB_BENCHMARKS,
    drop_xpu_skipped_sub_benchmarks,
    skipped_on_xpu,
)

_RUNNABLE = ["bench_jagged_2d_to_dense", "bench_dense_to_jagged_2d"]


def _benchmarks():
    names = _RUNNABLE + list(XPU_SKIPPED_SUB_BENCHMARKS)
    return {name: lambda jten: None for name in names}


def test_skipped_on_xpu_logs_instead_of_running(caplog):
    calls = []

    @skipped_on_xpu("no XPU implementation")
    def case(batch_size: int) -> None:
        calls.append(batch_size)

    with caplog.at_level(logging.INFO):
        case(batch_size=4)

    assert calls == []
    assert case.__name__ == "case"
    assert "SKIPPED case on xpu: no XPU implementation" in caplog.text


@pytest.mark.parametrize("only", [None, ""])
def test_drop_without_only_runs_all_remaining(only, caplog):
    benchmarks = _benchmarks()
    with caplog.at_level(logging.INFO):
        assert drop_xpu_skipped_sub_benchmarks(benchmarks, only) is None

    assert list(benchmarks) == _RUNNABLE
    for name in XPU_SKIPPED_SUB_BENCHMARKS:
        assert f"SKIPPED {name} on xpu" in caplog.text


def test_drop_with_only_logs_requested_skips(caplog):
    benchmarks = _benchmarks()
    skipped, other_skipped = list(XPU_SKIPPED_SUB_BENCHMARKS)[:2]
    with caplog.at_level(logging.INFO):
        only = drop_xpu_skipped_sub_benchmarks(
            benchmarks, f"{skipped}, bench_jagged_2d_to_dense,unknown"
        )

    assert only == "bench_jagged_2d_to_dense,unknown"
    assert list(benchmarks) == _RUNNABLE
    assert f"SKIPPED {skipped} on xpu" in caplog.text
    assert other_skipped not in caplog.text


def test_drop_with_only_skipped_returns_empty():
    only = ",".join(XPU_SKIPPED_SUB_BENCHMARKS)
    assert drop_xpu_skipped_sub_benchmarks(_benchmarks(), only) == ""
