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

/*
 * SYCL/XPU Implementation of jagged_index_select_2d
 *
 * Performs 2D index selection on jagged tensors with offset-based indexing.
 */

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE - JAGGED INDEX SELECT
////////////////////////////////////////////////////////////////////////////////
//
// This file contains SYCL ports of FBGEMM jagged_index_select_2d kernels.
//
// ORIGINAL CUDA SOURCE:
//   File: fbgemm_gpu/src/jagged_tensor_ops/jagged_index_select_2d_forward.cu
//
// KERNEL MAPPING:
//   JaggedIndexSelect2dKernel
//     → jagged_index_select_2d_kernel (CUDA)
//
// HOST FUNCTION MAPPING:
//   jagged_index_select_2d_forward_xpu
//     → jagged_index_select_2d_forward_cuda (CUDA)
//     CUDA File: fbgemm_gpu/src/jagged_tensor_ops/jagged_index_select_2d_forward.cu
//
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <sycl/sycl.hpp>
#include <c10/xpu/XPUStream.h>

#include <ATen/Operators.h>
#include <ATen/Tensor.h>
#include <torch/all.h>
#include <torch/library.h>

namespace fbgemm_xpu {

// Helper for binary search (upper_bound)
// Finds the first index i such that data[i] > target
// range: [0, n)
template <typename T>
inline int binary_search_upper_bound(const T* data, int n, T target);

////////////////////////////////////////////////////////////////////////////////
// JaggedIndexSelect2dKernel - Device Kernel
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Kernel: jagged_index_select_2d_kernel
//   CUDA File: fbgemm_gpu/src/jagged_tensor_ops/jagged_index_select_2d_forward.cu
//
// DESCRIPTION:
//   Copies sequences from input jagged tensor based on indices specified in
//   the indices tensor to an output jagged tensor. Uses binary search to
//   locate source sequences and block-strided copying for columns.
//
////////////////////////////////////////////////////////////////////////////////

template <typename scalar_t, typename index_t, typename offset_t>
class JaggedIndexSelect2dKernel {
public:
    JaggedIndexSelect2dKernel(
        scalar_t* output,
        const scalar_t* values,
        const index_t* indices,
        const offset_t* input_offsets,
        const offset_t* output_offsets,
        int64_t num_dense_output_rows,
        int64_t num_output_rows,
        int64_t num_cols
    ) : output_(output), values_(values), indices_(indices), 
        input_offsets_(input_offsets), output_offsets_(output_offsets),
        num_dense_output_rows_(num_dense_output_rows),
        num_output_rows_(num_output_rows), num_cols_(num_cols) {}

    void operator()(sycl::nd_item<1> item) const;

private:
    scalar_t* output_;
    const scalar_t* values_;
    const index_t* indices_;
    const offset_t* input_offsets_;
    const offset_t* output_offsets_;
    int64_t num_dense_output_rows_;
    int64_t num_output_rows_;
    int64_t num_cols_;
};

////////////////////////////////////////////////////////////////////////////////
// jagged_index_select_2d_forward_xpu - Host Function
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Function: jagged_index_select_2d_forward_cuda
//   CUDA File: fbgemm_gpu/src/jagged_tensor_ops/jagged_index_select_2d_forward.cu
//
// DESCRIPTION:
//   Host function for dispatching jagged_index_select_2d_kernel to XPU.
//   Validates inputs, allocates output tensor, and launches kernel with
//   appropriate work group configuration.
//
////////////////////////////////////////////////////////////////////////////////

at::Tensor jagged_index_select_2d_forward_xpu(
    const at::Tensor& values,
    const at::Tensor& indices,
    const at::Tensor& input_offsets,
    const at::Tensor& output_offsets,
    int64_t num_dense_output_rows);

} // namespace fbgemm_xpu
