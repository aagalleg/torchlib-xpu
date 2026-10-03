# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""XPU timing for the patched upstream FBGEMM benchmark scripts.

The patched scripts import ``benchmark_torch_function`` from here instead of
from ``fbgemm_gpu.bench.bench_utils``. Upstream's helper cannot time XPU: it
calls ``torch.cuda.set_device`` for any device other than cpu or cuda, and its
non-CUDA branch is a wall clock with no device synchronisation, which on XPU
measures queue submission rather than kernel execution.

The call shape and return value match upstream's single-stream path, so the
upstream call sites work unchanged. Upstream call sites never pass ``device``
and rely on its ``"cuda"`` default, so that value is accepted and treated as
XPU; any other non-XPU device is rejected. There is no fallback to wall-clock
timing: without an XPU device this raises.
"""

import functools
import logging
import math
import statistics
import time
from collections.abc import Callable
from typing import Any

import torch

# GPU work queued ahead of each start event must outlast the host's submission
# of the timed iteration; see benchmark_torch_function.
_MIN_LEAD_S = 50e-6
_MAX_LEAD_S = 50e-3
_LEAD_ATTEMPTS = 4
_SLEEP_CALIBRATION_CYCLES = 10_000_000


@functools.cache
def _sleep_supported(device_index: int) -> bool:
    # torch.xpu._sleep needs the ext_oneapi_clock_sub_group extension, which
    # older drivers lack.
    try:
        torch.xpu._sleep(1)
    except (NotImplementedError, RuntimeError):
        return False
    return True


def gpu_lead_method() -> str:
    """The GPU work benchmark_torch_function queues ahead of each start event.

    Describes the current device with a non-zero cache flush.
    """
    if _sleep_supported(torch.xpu.current_device()):
        return "a torch.xpu._sleep spin"
    return "repeated cache flush passes"


def _seconds_per_unit(work: Callable[[int], None], units: int) -> float:
    """GPU time of one unit of ``work``, which queues ``n`` units when called.

    Each length is timed behind a primer, so the GPU is not left waiting for
    the host inside the timed window, and the difference of two lengths
    cancels any fixed per-kernel cost.
    """
    seconds = []
    for n in (units, 2 * units):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        torch.xpu.synchronize()
        work(units)
        start.record()
        work(n)
        end.record()
        end.synchronize()
        seconds.append(start.elapsed_time(end) * 1.0e-3)
    return max(seconds[1] - seconds[0], 1.0e-9) / units


def benchmark_torch_function(
    f: Any,
    args: Any,
    kwargs: dict[str, Any] | None = None,
    flush_gpu_cache_size_mb: int | None = None,
    iters: int = 10,
    num_warmups: int = 2,
    device: str = "xpu",
    name: str = "",
) -> tuple[float, Any]:
    """Time ``f(*args, **kwargs)`` on XPU with device events.

    Returns the median per-iteration time in seconds and the last output.
    Every iteration first overwrites a ``flush_gpu_cache_size_mb`` scratch
    buffer, outside the timed region, so no iteration starts with its operands
    already in cache. The default is twice the device's last-level cache:
    upstream's 40 MB is sized for CUDA parts, and a PVC tile has 192 MB of L2,
    so 40 MB leaves operands resident and inflates bandwidth.

    Events time the GPU timeline, so if the GPU reaches the start event before
    the host has submitted ``f``, the wait for the host is timed as well. Each
    iteration therefore queues GPU work ahead of the start event, lasting twice
    the host submission time seen in the warm-ups. The work is a
    ``torch.xpu._sleep`` spin after the flush where the driver supports it, as
    the spin touches no memory; otherwise the flush buffer is overwritten
    repeatedly, which leaves the cache just as flushed. With neither, that is
    an old driver and a flush size of 0, no work is queued. An iteration whose
    start event completed before the host finished submitting it may include
    such a wait; the measurement is then repeated with more work, and if the
    wait persists, for example because ``f`` synchronises, a warning is logged.
    """
    if device not in ("xpu", "cuda", ""):
        raise ValueError(f"benchmark_torch_function times XPU only, got {device!r}")
    if not torch.xpu.is_available():
        raise RuntimeError("XPU timing requested but no XPU device is available")
    if iters < 1:
        raise ValueError("iters must be >= 1")
    kwargs = kwargs or {}
    if flush_gpu_cache_size_mb is None:
        llc_bytes = torch.xpu.get_device_properties().last_level_cache_size
        flush_gpu_cache_size_mb = 2 * llc_bytes // (1024 * 1024)

    logging.debug(f"Start to benchmark {name}...")
    host_s = 0.0
    output = None
    for _ in range(num_warmups):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        t = time.perf_counter()
        start.record()
        output = f(*args, **kwargs)
        end.record()
        host_s = max(host_s, time.perf_counter() - t)
    lead_s = min(2 * host_s + _MIN_LEAD_S, _MAX_LEAD_S)

    cache = torch.empty(
        int(flush_gpu_cache_size_mb * 1024 * 1024 // 4), dtype=torch.float, device="xpu"
    )

    def flush(passes: int) -> None:
        for _ in range(passes):
            cache.zero_()

    if _sleep_supported(torch.xpu.current_device()):
        sleep_s = _seconds_per_unit(torch.xpu._sleep, _SLEEP_CALIBRATION_CYCLES)

        def queue_lead(seconds: float) -> None:
            flush(1 if flush_gpu_cache_size_mb else 0)
            torch.xpu._sleep(math.ceil(seconds / sleep_s))

    elif flush_gpu_cache_size_mb:
        pass_s = _seconds_per_unit(flush, 2)

        def queue_lead(seconds: float) -> None:
            flush(max(1, math.ceil(seconds / pass_s)))

    else:

        def queue_lead(seconds: float) -> None:
            pass

    for _ in range(_LEAD_ATTEMPTS):
        start = [torch.xpu.Event(enable_timing=True) for _ in range(iters)]
        end = [torch.xpu.Event(enable_timing=True) for _ in range(iters)]
        waited = 0
        torch.xpu.synchronize()
        for i in range(iters):
            queue_lead(lead_s)
            start[i].record()
            output = f(*args, **kwargs)
            end[i].record()
            waited += start[i].query()
        torch.xpu.synchronize()
        if not waited or lead_s >= _MAX_LEAD_S:
            break
        lead_s = min(4 * lead_s, _MAX_LEAD_S)
    if waited:
        logging.warning(
            f"benchmark_torch_function {name}: the GPU may have waited for the "
            f"host in {waited} of {iters} timed iterations, so the time can "
            "include host submission latency."
        )

    elapsed_ms = statistics.median(s.elapsed_time(e) for s, e in zip(start, end))
    return elapsed_ms * 1.0e-3, output
