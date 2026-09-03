/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */


// ============================================================================
// Type Dispatch Macros for Sparse Operations
// ============================================================================

#define FBGEMM_DISPATCH_FLOATING_TYPES_CASE(...)       \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define FBGEMM_DISPATCH_INTEGRAL_TYPES_CASE(...)     \
  AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__)

#define FBGEMM_DISPATCH_ALL_TYPES(TYPE, NAME, ...)     \
  AT_DISPATCH_SWITCH(                                  \
      TYPE,                                            \
      NAME,                                            \
      FBGEMM_DISPATCH_FLOATING_TYPES_CASE(__VA_ARGS__) \
          FBGEMM_DISPATCH_INTEGRAL_TYPES_CASE(__VA_ARGS__))
