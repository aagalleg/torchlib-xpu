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

#include <ATen/core/dispatch/Dispatcher.h>

namespace fbgemm_xpu::utils::torch {

inline bool schemaExists(const std::string& qualified_name) {
  return c10::Dispatcher::singleton()
      .findSchema({qualified_name, ""})
      .has_value();
}

} // namespace fbgemm_xpu::utils::torch