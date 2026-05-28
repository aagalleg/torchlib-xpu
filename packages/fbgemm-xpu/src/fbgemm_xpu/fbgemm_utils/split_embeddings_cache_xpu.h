/*
 * Copyright 2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Portions of this file are derived from FBGEMM
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <ATen/ATen.h>

namespace fbgemm_xpu {

enum uvm_cache_stats_index {
  num_calls = 0,
  num_requested_indices = 1,
  num_unique_indices = 2,
  num_unique_misses = 3,
  num_conflict_unique_misses = 4,
  num_conflict_misses = 5,
};

} // namespace fbgemm_xpu