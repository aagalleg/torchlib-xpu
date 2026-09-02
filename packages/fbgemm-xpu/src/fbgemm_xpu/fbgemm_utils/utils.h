/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <limits>

#include <ATen/xpu/XPUContext.h>

#include "dispatch_macros.h"

namespace fbgemm_xpu {

constexpr size_t kThreadGroupSize = 32;
constexpr size_t kMaxThreads = 1024;

// Not shipped in PyTorch's installed XPU headers; query it directly here.
static inline int64_t syclDeviceMaxWorkGroupSize(
    at::DeviceIndex dev_id = c10::xpu::current_device()) {
      auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
      return dev_prop->max_work_group_size;
}


// ============================================================================
// Block Count Calculation Utilities (from fbgemm_utils.h/sycl)
// ============================================================================

/**
 * @brief Calculate the number of SYCL work-groups (blocks) needed
 * 
 * Base function for calculating block count with overflow protection.
 * 
 * @param num_items Total number of items to process
 * @param threads_per_block Number of work-items per work-group
 * @return Number of work-groups needed (capped at max_blocks)
 */
inline uint32_t xpu_calc_xblock_count_base(int64_t num_items, int64_t threads_per_block) {
  // The number of threads can be as high as 2048 on some newer architectures,
  // but this is not portable.
  TORCH_CHECK(
      threads_per_block <= syclDeviceMaxWorkGroupSize(),
      "Number of threads must be <=1024!");
  constexpr uint64_t max_blocks = 2147483647;
  const auto u_num_items = static_cast<uint64_t>(num_items);
  const auto u_threads = static_cast<uint64_t>(threads_per_block);
  // Overflow safe variant of (a + b - 1) / b
  const uint64_t blocks =
      u_num_items / u_threads + (u_num_items % u_threads != 0);
  return static_cast<uint32_t>(std::min(blocks, max_blocks));
}

/**
 * @brief Calculate the number of SYCL work-groups (blocks) needed
 * 
 * Validates input and calls xpu_calc_xblock_count_base.
 * 
 * @param num_items Total number of items to process (must be >= 0)
 * @param threads_per_block Number of work-items per work-group
 * @return Number of work-groups needed
 */
inline uint32_t xpu_calc_xblock_count(int64_t num_items, int64_t threads_per_block) {
  TORCH_CHECK(
      num_items >= 0,
      "When calculating block counts, the number of items must be positive!");
  return xpu_calc_xblock_count_base(num_items, threads_per_block);
}

/**
 * @brief Cap a work-group count so a launch's flattened work-item count fits
 * in an `int`.
 *
 * DPC++ assumes by default (`-fsycl-id-queries-fit-in-int`) that the total
 * number of work-items in a submission fits in a 32-bit `int`; `queue::submit`
 * throws once that product exceeds INT32_MAX. This mirrors upstream FBGEMM's
 * `utils::cuda::cap_grid_dim_x`, used there to stay under HIP's analogous
 * 2^32 launch-size limit.
 *
 * Callers MUST pair this with a grid-stride loop in the kernel so every
 * logical unit of work in [0, num_groups) is still covered by the
 * (possibly smaller) capped launch.
 *
 * @param num_groups Requested number of work-groups along one dimension
 * @param threads_per_group Total work-items per work-group across all
 *     dimensions of the launch (e.g. local_dim0 * local_dim1)
 * @return Number of work-groups to actually launch (<= num_groups)
 */
inline uint32_t xpu_cap_grid_dim_x(int64_t num_groups, int64_t threads_per_group) {
  TORCH_CHECK(num_groups >= 0, "num_groups must be non-negative");
  TORCH_CHECK(threads_per_group > 0, "threads_per_group must be positive");
  constexpr int64_t kIntMax = std::numeric_limits<int32_t>::max();
  const int64_t max_groups = std::max<int64_t>(1, kIntMax / threads_per_group);
  return static_cast<uint32_t>(std::min(num_groups, max_groups));
}

#define SYCL_DEVICE_GUARD(TENSOR)          \
  c10::OptionalDeviceGuard device_guard;  \
  device_guard.reset_device(TENSOR.device())

} // namespace fbgemm_xpu
