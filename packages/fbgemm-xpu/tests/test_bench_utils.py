# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the XPU timing helper used by the patched FBGEMM benchmarks.

The failure this guards against is the one upstream's helper has on XPU: a
wall clock without device synchronisation measures queue submission, and
reports a figure that is wrong but plausible. A copy of known size gives a
physical lower bound on the time - it cannot finish faster than any XPU's
memory bandwidth allows - and submission alone comes in far under it.
"""

import logging
import time

import pytest
import torch
from fbgemm_xpu.bench.bench_utils import benchmark_torch_function

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="requires an XPU device"
)

# Well above the HBM bandwidth of any current Intel GPU (PVC is ~3.3 TB/s).
_MAX_PLAUSIBLE_BYTES_PER_S = 10e12


def test_time_is_bounded_by_device_bandwidth():
    src = torch.empty(256 * 1024 * 1024, dtype=torch.float, device="xpu")
    dst = torch.empty_like(src)
    seconds, _ = benchmark_torch_function(dst.copy_, (src,), iters=5)

    moved = 2 * src.numel() * src.element_size()
    assert seconds > moved / _MAX_PLAUSIBLE_BYTES_PER_S  # nosec B101


def _copy_bytes_per_s(numel, **kwargs):
    # Random, not uninitialised: freshly allocated memory is typically zeroed,
    # and memory compression can then report more than the real bandwidth.
    src = torch.rand(numel, dtype=torch.float, device="xpu")
    dst = torch.empty_like(src)
    seconds, _ = benchmark_torch_function(dst.copy_, (src,), iters=20, **kwargs)
    return 2 * src.numel() * src.element_size() / seconds


def test_default_flush_evicts_last_level_cache():
    # A copy that fits in the last-level cache must not beat a streaming copy
    # far larger than it. Upstream's fixed 40 MB flush fails this on PVC (192
    # MB L2): the operands stay resident and the copy reports ~3x HBM speed.
    llc_floats = torch.xpu.get_device_properties().last_level_cache_size // 4
    resident = _copy_bytes_per_s(llc_floats // 4)
    streaming = _copy_bytes_per_s(8 * llc_floats, flush_gpu_cache_size_mb=0)
    assert resident < 1.25 * streaming  # nosec B101


def test_returns_output_and_passes_kwargs():
    x = torch.arange(8, dtype=torch.float, device="xpu")
    _, out = benchmark_torch_function(torch.add, (x, x), kwargs={"alpha": 2})
    torch.testing.assert_close(out.cpu(), (x + 2 * x).cpu())


def test_host_submission_latency_is_not_timed(caplog):
    # f spends 2 ms on the host before launching a tiny kernel. Without a GPU
    # lead ahead of the start event, the GPU idles through those 2 ms inside
    # the timed window; this is what a slow host does to small shapes on BMG.
    x = torch.zeros(1024, device="xpu")

    def host_heavy():
        time.sleep(2e-3)
        return x.add_(1)

    with caplog.at_level(logging.WARNING):
        seconds, _ = benchmark_torch_function(host_heavy, (), iters=5)
    assert seconds < 0.5e-3  # nosec B101
    assert "waited for the host" not in caplog.text  # nosec B101


def test_warns_when_f_synchronises(caplog):
    x = torch.zeros(1024, device="xpu")

    def synchronising():
        x.add_(1)
        torch.xpu.synchronize()

    with caplog.at_level(logging.WARNING):
        benchmark_torch_function(synchronising, (), iters=2, flush_gpu_cache_size_mb=0)
    assert "waited for the host" in caplog.text  # nosec B101


@pytest.mark.parametrize("device", ["cpu", "mtia"])
def test_rejects_non_xpu_device(device):
    with pytest.raises(ValueError, match="XPU only"):
        benchmark_torch_function(torch.zeros, (1,), device=device)
