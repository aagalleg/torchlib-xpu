/*
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * 
 * Original Source: https://github.com/intel/torch-xpu-ops
 */

// NOTE: DO NOT remove file. File required for compilation of fbgemm-xpu.

#pragma once

#include <comm/DeviceProperties.h>
#include <comm/Runtime.h>
#include <comm/SYCLHelpers.h>

using namespace at::xpu;
using namespace xpu::sycl;
