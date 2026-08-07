/*
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * 
 * Original Source: https://github.com/intel/torch-xpu-ops
 */

#pragma once

#include <ATen/xpu/XPUContext.h>

#include <sycl/sycl.hpp>

namespace xpu {
namespace sycl {

template <class KernelClass>
static int64_t syclMaxWorkGroupSize(
    at::DeviceIndex dev_id = at::xpu::current_device()) {
  auto& ctx = c10::xpu::get_device_context();
  auto& dev = c10::xpu::get_raw_device(dev_id);

  auto kid = ::sycl::get_kernel_id<KernelClass>();
  // The kernel won't be built for devices except for the first device.
  // Launching kernel on devices except for the first device will raise
  // runtime error. Here is an alternative as a temporary solution to
  // provide an extra hint to SYCL runtime.
  // https://github.com/intel/llvm/issues/15127
  auto kbundle = ::sycl::get_kernel_bundle<::sycl::bundle_state::executable>(
      ctx, {dev}, {kid});

  ::sycl::kernel k = kbundle.get_kernel(kid);
  return k.get_info<::sycl::info::kernel_device_specific::work_group_size>(dev);
}

static inline int64_t syclDeviceMaxWorkGroupSize(
    at::DeviceIndex dev_id = at::xpu::current_device()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  return dev_prop->max_work_group_size;
}

static inline int64_t syclMaxSubGroupSize(
    at::DeviceIndex dev_id = at::xpu::current_device()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  const auto& subgroup_sizes = dev_prop->sub_group_sizes;
  TORCH_CHECK(
      !subgroup_sizes.empty(),
      "The device subgroup sizes is empty, please check the device status.");
  return *std::max_element(subgroup_sizes.begin(), subgroup_sizes.end());
}

static inline int64_t syclMaxWorkItemsPerTile(
    at::DeviceIndex dev_id = at::xpu::current_device()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  int64_t eu_cnt = dev_prop->gpu_eu_count;
  int64_t simd_width = syclMaxSubGroupSize(dev_id);
  int64_t hw_threads = dev_prop->gpu_hw_threads_per_eu;
  return eu_cnt * simd_width * hw_threads;
}

static inline int64_t syclMaxWorkItemsPerSubSlice(
    at::DeviceIndex dev_id = at::xpu::current_device()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  int64_t simd_width = syclMaxSubGroupSize(dev_id);
  int64_t eu_count = dev_prop->gpu_eu_count_per_subslice;
  return simd_width * eu_count;
}

} // namespace sycl
} // namespace xpu
