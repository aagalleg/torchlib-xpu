/*
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * 
 * Original Source: https://github.com/intel/torch-xpu-ops
 */

#pragma once

#include <c10/xpu/XPUStream.h>

namespace at::xpu {

static inline sycl::queue& getCurrentSYCLQueue() {
  return c10::xpu::getCurrentXPUStream().queue();
}

} // namespace at::xpu
