/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE - REORDER BATCHED AD OPERATORS
////////////////////////////////////////////////////////////////////////////////
//
// This file contains SYCL ports of FBGEMM batched AD reordering operators.
//
// ORIGINAL CUDA SOURCE:
//   File: fbgemm_gpu/src/sparse_ops/sparse_reorder_batched_ad.cu
//
// KERNEL MAPPING:
//   ReorderBatchedAdLengthsKernel<scalar_t>
//     → reorder_batched_ad_lengths_kernel (CUDA)
//
//   NarrowBroadcastIndicesKernel<scalar_t, index_t>
//     → narrow_broadcast_indices_kernel (CUDA)
//
//   NarrowBatchedBroadcastIndicesKernel<scalar_t, index_t>
//     → narrow_batched_broadcast_indices_kernel (CUDA)
//
//   ReorderBatchedAdIndicesKernel<scalar_t, index_t>
//     → reorder_batched_ad_indices_kernel (CUDA)
//
// HOST FUNCTION MAPPING:
//   reorder_batched_ad_lengths_xpu (SYCL)
//     → reorder_batched_ad_lengths_cuda (CUDA)
//
//   reorder_batched_ad_indices_xpu (SYCL)
//     → reorder_batched_ad_indices_cuda (CUDA)
//
// DESCRIPTION:
//   Reorders batched AD (advertisement) lengths and indices from ragged
//   [B x T x #num_ads_b] layout to [T][B][#num_ads_b] layout for efficient
//   embedding lookups. Supports broadcast modes for lengths and indices.
//
//   The kernels are plain SYCL functors submitted through the PyTorch XPU
//   stream queue, so this file does not depend on the comm/ helper directory.
//
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <sycl/sycl.hpp>

#include <c10/xpu/XPUStream.h>

#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/native/xpu/sycl/KernelUtils.h>
#include <ATen/native/StridedRandomAccessor.h>
#include <torch/library.h>

#include "../fbgemm_utils/utils.h"
#include "../fbgemm_utils/tensor_utils.h"

using at::native::RestrictPtrTraits;

namespace fbgemm_xpu {

// ============================================================================
// SYCL Kernel Functors
// ============================================================================

////////////////////////////////////////////////////////////////////////////////
// ReorderBatchedAdLengthsKernel - Device Kernel
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Kernel: reorder_batched_ad_lengths_kernel
//   CUDA File: fbgemm_gpu/src/sparse_ops/sparse_reorder.cu
//
// DESCRIPTION:
//   Reorders AD lengths from ragged [B x T x #num_ads_b] to [T][B][#num_ads_b].
//   Each warp processes one (batch, table) pair and copies num_ads_b elements.
//   Supports broadcast mode where a single length is replicated for all ads.
//
////////////////////////////////////////////////////////////////////////////////
template <typename scalar_t>
class ReorderBatchedAdLengthsKernel {
public:
    ReorderBatchedAdLengthsKernel(
            at::GenericPackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, int32_t>
                    cat_ad_lengths,
            at::GenericPackedTensorAccessor<int32_t, 1, RestrictPtrTraits, int32_t>
                    batch_offsets,
            at::GenericPackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, int32_t>
                    reordered_cat_ad_lengths,
            int32_t T,
            bool broadcast_lengths)
        : cat_ad_lengths_(cat_ad_lengths),
          batch_offsets_(batch_offsets),
          reordered_cat_ad_lengths_(reordered_cat_ad_lengths),
          T_(T),
          broadcast_lengths_(broadcast_lengths) {}

    void operator()(const sycl::nd_item<2>& item) const;

private:
    at::GenericPackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, int32_t>
            cat_ad_lengths_;
    at::GenericPackedTensorAccessor<int32_t, 1, RestrictPtrTraits, int32_t>
            batch_offsets_;
    // Written by the kernel, so it must stay assignable inside operator() const.
    mutable at::GenericPackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, int32_t>
            reordered_cat_ad_lengths_;
    int32_t T_;
    bool broadcast_lengths_;
};

////////////////////////////////////////////////////////////////////////////////
// NarrowBroadcastIndicesKernel - Device Kernel (B=1 optimization)
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Kernel: narrow_broadcast_indices_kernel
//   CUDA File: fbgemm_gpu/src/sparse_ops/sparse_reorder.cu
//
// DESCRIPTION:
//   Optimized kernel for B=1 broadcast case. Each warp copies one ad segment
//   and broadcasts it to all output positions. Uses table-major iteration.
//
////////////////////////////////////////////////////////////////////////////////
template <typename scalar_t, typename index_t = int32_t>
class NarrowBroadcastIndicesKernel {
public:
    NarrowBroadcastIndicesKernel(
            at::GenericPackedTensorAccessor<index_t, 1, RestrictPtrTraits, int32_t>
                    cat_ad_offsets,
            at::GenericPackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, int32_t>
                    cat_ad_indices,
            at::GenericPackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, int32_t>
                    reordered_cat_ad_indices,
            int num_ads_in_batch,
            int reordered_cat_ad_batches,
            int sub_group_size)
        : cat_ad_offsets_(cat_ad_offsets),
          cat_ad_indices_(cat_ad_indices),
          reordered_cat_ad_indices_(reordered_cat_ad_indices),
          num_ads_in_batch_(num_ads_in_batch),
          reordered_cat_ad_batches_(reordered_cat_ad_batches),
          sub_group_size_(sub_group_size) {}

    void operator()(const sycl::nd_item<1>& item) const;

private:
    at::GenericPackedTensorAccessor<index_t, 1, RestrictPtrTraits, int32_t>
            cat_ad_offsets_;
    at::GenericPackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, int32_t>
            cat_ad_indices_;
    // Written by the kernel, so it must stay assignable inside operator() const.
    mutable at::GenericPackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, int32_t>
            reordered_cat_ad_indices_;
    int num_ads_in_batch_;
    int reordered_cat_ad_batches_;
    int sub_group_size_;
};

////////////////////////////////////////////////////////////////////////////////
// NarrowBatchedBroadcastIndicesKernel - Device Kernel (B>1 optimization)
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Kernel: narrow_batched_broadcast_indices_kernel
//   CUDA File: fbgemm_gpu/src/sparse_ops/sparse_reorder.cu
//
// DESCRIPTION:
//   Optimized kernel for 1 < B < 64 broadcast case. Each warp handles one
//   (table, batch) pair and broadcasts indices from first batch to all ads
//   in that batch. More complex warp assignment than B=1 case.
//
////////////////////////////////////////////////////////////////////////////////
template <typename scalar_t, typename index_t = int32_t>
class NarrowBatchedBroadcastIndicesKernel {
public:
    NarrowBatchedBroadcastIndicesKernel(
            at::GenericPackedTensorAccessor<index_t, 1, RestrictPtrTraits, int32_t>
                    cat_ad_offsets,
            at::GenericPackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, int32_t>
                    cat_ad_indices,
            at::GenericPackedTensorAccessor<index_t, 1, RestrictPtrTraits, int32_t>
                    reordered_cat_ad_offsets,
            at::GenericPackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, int32_t>
                    reordered_cat_ad_indices,
            at::GenericPackedTensorAccessor<int32_t, 1, RestrictPtrTraits, int32_t>
                    batch_offsets,
            int32_t T,
            int sub_group_size)
        : cat_ad_offsets_(cat_ad_offsets),
          cat_ad_indices_(cat_ad_indices),
          reordered_cat_ad_offsets_(reordered_cat_ad_offsets),
          reordered_cat_ad_indices_(reordered_cat_ad_indices),
          batch_offsets_(batch_offsets),
          T_(T),
          sub_group_size_(sub_group_size) {}

    void operator()(const sycl::nd_item<1>& item) const;

private:
    at::GenericPackedTensorAccessor<index_t, 1, RestrictPtrTraits, int32_t>
            cat_ad_offsets_;
    at::GenericPackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, int32_t>
            cat_ad_indices_;
    at::GenericPackedTensorAccessor<index_t, 1, RestrictPtrTraits, int32_t>
            reordered_cat_ad_offsets_;
    // Written by the kernel, so it must stay assignable inside operator() const.
    mutable at::GenericPackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, int32_t>
            reordered_cat_ad_indices_;
    at::GenericPackedTensorAccessor<int32_t, 1, RestrictPtrTraits, int32_t>
            batch_offsets_;
    int32_t T_;
    int sub_group_size_;
};

////////////////////////////////////////////////////////////////////////////////
// ReorderBatchedAdIndicesKernel - Device Kernel (General case)
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Kernel: reorder_batched_ad_indices_kernel
//   CUDA File: fbgemm_gpu/src/sparse_ops/sparse_reorder.cu
//
// DESCRIPTION:
//   General kernel for reordering indices from [B x T x #num_ads_b x L] to
//   [T][B][#num_ads_b][L]. Each warp processes one (batch, table) pair and
//   copies all indices for all ads in that segment. Handles both broadcast
//   and non-broadcast modes.
//
////////////////////////////////////////////////////////////////////////////////
template <typename scalar_t, typename index_t = int32_t>
class ReorderBatchedAdIndicesKernel {
public:
    ReorderBatchedAdIndicesKernel(
            at::GenericPackedTensorAccessor<index_t, 1, RestrictPtrTraits, int32_t>
                    cat_ad_offsets,
            at::GenericPackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, int32_t>
                    cat_ad_indices,
            at::GenericPackedTensorAccessor<index_t, 1, RestrictPtrTraits, int32_t>
                    reordered_cat_ad_offsets,
            at::GenericPackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, int32_t>
                    reordered_cat_ad_indices,
            at::GenericPackedTensorAccessor<int32_t, 1, RestrictPtrTraits, int32_t>
                    batch_offsets,
            int32_t T,
            bool broadcast_indices)
        : cat_ad_offsets_(cat_ad_offsets),
          cat_ad_indices_(cat_ad_indices),
          reordered_cat_ad_offsets_(reordered_cat_ad_offsets),
          reordered_cat_ad_indices_(reordered_cat_ad_indices),
          batch_offsets_(batch_offsets),
          T_(T),
          broadcast_indices_(broadcast_indices) {}

    void operator()(const sycl::nd_item<2>& item) const;

private:
    at::GenericPackedTensorAccessor<index_t, 1, RestrictPtrTraits, int32_t>
            cat_ad_offsets_;
    at::GenericPackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, int32_t>
            cat_ad_indices_;
    at::GenericPackedTensorAccessor<index_t, 1, RestrictPtrTraits, int32_t>
            reordered_cat_ad_offsets_;
    // Written by the kernel, so it must stay assignable inside operator() const.
    mutable at::GenericPackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, int32_t>
            reordered_cat_ad_indices_;
    at::GenericPackedTensorAccessor<int32_t, 1, RestrictPtrTraits, int32_t>
            batch_offsets_;
    int32_t T_;
    bool broadcast_indices_;
};

// ============================================================================
// Kernel Functions
// ============================================================================

/**
 * @brief Reorder batched AD lengths from [B x T x #num_ads_b] to [T][B][#num_ads_b]
 * 
 * Kernel function to reorder advertisement lengths tensor according to a new layout
 * that groups by table first, then batch. Supports broadcast mode where all ads
 * in a batch share the same length.
 * 
 * @param cat_ad_lengths Input lengths tensor [B x T x #num_ads_b] (ragged)
 * @param batch_offsets Batch offset indices [B+1]
 * @param reordered_cat_ad_lengths Output reordered lengths [T x sum(#num_ads_b)]
 * @param T Number of tables/features
 * @param broadcast_lengths If true, broadcast first length to all ads in batch
 * @param grid_size Number of workgroups for kernel launch
 */
void reorder_batched_ad_lengths_kernel_xpu(
    const at::Tensor& cat_ad_lengths,
    const at::Tensor& batch_offsets,
    at::Tensor& reordered_cat_ad_lengths,
    const int32_t T,
    const bool broadcast_lengths,
    const int32_t grid_size);

/**
 * @brief Reorder batched AD indices from [B x T x #num_ads_b x L] to [T][B][#num_ads_b][L]
 * 
 * Kernel function to reorder advertisement indices tensor according to a new layout
 * that groups by table first, then batch. Supports broadcast mode where indices
 * from first batch are replicated across all batches.
 * 
 * @param cat_ad_offsets Input offset indices [B x T x #num_ads_b + 1] (ragged)
 * @param cat_ad_indices Input indices tensor [sum(L)]
 * @param reordered_cat_ad_offsets Output offset indices [T x sum(#num_ads_b) + 1]
 * @param batch_offsets Batch offset indices [B+1]
 * @param reordered_cat_ad_indices Output reordered indices [sum(L)]
 * @param num_ads_in_batch Total number of ads across all batches
 * @param B Batch size
 * @param T Number of tables/features
 * @param broadcast_indices If true, broadcast first batch indices to all batches
 */
void reorder_batched_ad_indices_kernel_xpu(
    const at::Tensor& cat_ad_offsets,
    const at::Tensor& cat_ad_indices,
    const at::Tensor& reordered_cat_ad_offsets,
    const at::Tensor& batch_offsets,
    at::Tensor& reordered_cat_ad_indices,
    const int64_t num_ads_in_batch,
    const int64_t B,
    const int64_t T,
    const bool broadcast_indices);


} // namespace fbgemm_xpu
