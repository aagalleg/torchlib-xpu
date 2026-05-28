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

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE - BACKWARD KERNELS
////////////////////////////////////////////////////////////////////////////////
//
// This file contains SYCL ports of FBGEMM dense embedding backward kernels.
//
// ORIGINAL CUDA SOURCE:
//   File: gen_embedding_backward_dense_split_unweighted_cuda.cu
//   Path: FBGEMM/fbgemm_gpu/_skbuild/linux-x86_64-3.10/cmake-build/ (generated files)
//   Template: training/backward/embedding_backward_split_template.cu
//
// KERNEL MAPPING:
//   SplitEmbeddingNobagBackwardDenseUnweightedKernelWarpPerRow1
//     → split_embedding_backward_..._kernel_warp_kernel (CUDA)
//
//   SplitEmbeddingNobagBackwardDenseUnweightedKernelCtaPerRow
//     → split_embedding_backward_..._kernel_cta_kernel (CUDA)
//
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cassert>
#include <cstdlib>

#include <sycl/sycl.hpp>
#include <c10/xpu/XPUStream.h>

#include <ATen/ATen.h>
#include <ATen/Operators.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/native/StridedRandomAccessor.h>

#include "../fbgemm_utils/backward_utils.h"
#include "../fbgemm_utils/tensor_utils.h"
#include "../fbgemm_utils/utils.h"

using Tensor = at::Tensor;
using at::native::RestrictPtrTraits;

namespace fbgemm_xpu {

    ////////////////////////////////////////////////////////////////////////////////
    // split_embedding_nobag_backward_dense_unweighted_exact_xpu - Host Function
    ////////////////////////////////////////////////////////////////////////////////
    //
    // CUDA SOURCE MAPPING:
    //   CUDA Function: split_embedding_nobag_backward_dense_unweighted_exact_cuda
    //   CUDA File: gen_embedding_backward_dense_split_unweighted_cuda.cu
    //
    // DESCRIPTION:
    //   Host function for no-bag embedding backward pass.
    //   Computes gradients for embedding weights without pooling.
    //
    ////////////////////////////////////////////////////////////////////////////////
    Tensor split_embedding_nobag_backward_dense_unweighted_exact_xpu(
        const Tensor& grad_output,
        const Tensor& dev_weights,
        const Tensor& weights_offsets,
        const c10::SymInt D_,
        const Tensor& hash_size_cumsum,
        const int64_t total_hash_size_bits,
        const Tensor& indices,
        const Tensor& offsets,
        const int64_t unused_,
        const int64_t max_segment_length_per_warp,
        double unused
    );

    ////////////////////////////////////////////////////////////////////////////////
    // SplitEmbeddingNobagBackwardDenseUnweightedKernelWarpPerRow1 - Warp-Level Kernel
    ////////////////////////////////////////////////////////////////////////////////
    //
    // CUDA SOURCE MAPPING:
    //   CUDA Kernel: split_embedding_backward_..._kernel_warp_kernel
    //   CUDA File: gen_embedding_backward_none_split_unweighted_kernel_warp.cu
    //
    // DESCRIPTION:
    //   Warp-level backward kernel for short segments.
    //   One warp processes one embedding row update.
    //   Uses shared memory for gradient accumulation.
    //
    ////////////////////////////////////////////////////////////////////////////////
    template <
        typename emb_t,
        typename grad_t,
        typename cache_t,
        typename index_t,
        int32_t kFixedMaxVecsPerThread,
        int32_t kThreadGroupSize,
        bool kUseVecBlocking>
    class SplitEmbeddingNobagBackwardDenseUnweightedKernelWarpPerRow1 {
    public:
        SplitEmbeddingNobagBackwardDenseUnweightedKernelWarpPerRow1(
            const at::PackedTensorAccessor64<grad_t, 2, RestrictPtrTraits> grad_output,
            at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> dev_weights,
            const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> weights_offsets,
            int64_t D,
            const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> hash_size_cumsum,
            const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> sorted_linear_indices_run,
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_linear_indices_cumulative_run_lengths,
            const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> sorted_infos,
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_linear_indices_num_runs,
            int32_t max_segment_length_per_warp,
            at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> grad_dev_weights,
            const int32_t max_D,
            const int32_t max_vecs_per_thread,
            sycl::local_accessor<cache_t, 1> smem,
            double unused = 0)
        : grad_output_(grad_output),
            dev_weights_(dev_weights),
            weights_offsets_(weights_offsets),
            D_(D),
            hash_size_cumsum_(hash_size_cumsum),
            sorted_linear_indices_run_(sorted_linear_indices_run),
            sorted_linear_indices_cumulative_run_lengths_(sorted_linear_indices_cumulative_run_lengths),
            sorted_infos_(sorted_infos),
            sorted_linear_indices_num_runs_(sorted_linear_indices_num_runs),
            max_segment_length_per_warp_(max_segment_length_per_warp),
            grad_dev_weights_(grad_dev_weights),
            max_D_(max_D),
            max_vecs_per_thread_(max_vecs_per_thread),
            smem_(smem) {}

        void operator()(const sycl::nd_item<2>& item) const;

    private:
        const at::PackedTensorAccessor64<grad_t, 2, RestrictPtrTraits> grad_output_;
        mutable at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> dev_weights_;
        const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> weights_offsets_;
        int64_t D_;
        const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> hash_size_cumsum_;
        const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> sorted_linear_indices_run_;
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_linear_indices_cumulative_run_lengths_;
        const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> sorted_infos_;
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_linear_indices_num_runs_;
        int32_t max_segment_length_per_warp_;
        mutable at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> grad_dev_weights_;
        int32_t max_D_;
        int32_t max_vecs_per_thread_;
        sycl::local_accessor<cache_t, 1> smem_;
    };


    // Run-length encoding kernel
    template <typename index_t>
    class RunLengthEncodeKernel {
    public:
        RunLengthEncodeKernel(
            const index_t* sorted_input,
            index_t* unique_output,
            int32_t* run_lengths,
            int32_t* num_runs,
            int64_t total_elements);

        void operator()(const sycl::nd_item<1>& item) const;

    private:
        const index_t* sorted_input_;
        index_t* unique_output_;
        int32_t* run_lengths_;
        int32_t* num_runs_;
        int64_t total_elements_;
    };

    ////////////////////////////////////////////////////////////////////////////////
    // SplitEmbeddingNobagBackwardDenseUnweightedKernelCtaPerRow - Block-Level Kernel
    ////////////////////////////////////////////////////////////////////////////////
    //
    // CUDA SOURCE MAPPING:
    //   CUDA Kernel: split_embedding_backward_..._kernel_cta_kernel
    //   CUDA File: gen_embedding_backward_none_split_unweighted_kernel_cta.cu
    //
    // DESCRIPTION:
    //   Block-level backward kernel for long segments.
    //   Full CTA (Cooperative Thread Array) processes one embedding row update.
    //   Used when segment length > max_segment_length_per_warp.
    //
    ////////////////////////////////////////////////////////////////////////////////
    template <
        typename emb_t,
        typename grad_t,
        typename cache_t,
        typename index_t,
        int32_t kFixedMaxVecsPerThread,
        int32_t kThreadGroupSize,
        bool kUseVecBlocking>
    class SplitEmbeddingNobagBackwardDenseUnweightedKernelCtaPerRow {
    public:
        SplitEmbeddingNobagBackwardDenseUnweightedKernelCtaPerRow(
        const at::PackedTensorAccessor64<grad_t, 2, RestrictPtrTraits> grad_output,
        at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> dev_weights, // if optimizer != "none"
        const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> weights_offsets,
        int64_t D,
        const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> hash_size_cumsum,
        const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> sorted_linear_indices_run,
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_linear_indices_cumulative_run_lengths,
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> long_run_ids,
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> num_long_run_ids,
        const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> sorted_infos,
        at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> grad_dev_weights, // if not dense and optimizer != "none"
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> long_run_id_to_really_long_run_ids,
        at::PackedTensorAccessor32<at::acc_type<cache_t, true>, 2, RestrictPtrTraits> temp_grad_accum,
        at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> grad_accum_counter,
        const int32_t max_segment_length_per_cta,
        const bool use_deterministic_algorithms,
        const int32_t max_vecs_per_thread,
        sycl::local_accessor<cache_t, 1> smem,
        float unused = 0
        )
        : grad_output_(grad_output),
            dev_weights_(dev_weights),
            weights_offsets_(weights_offsets),
            D_(D),
            hash_size_cumsum_(hash_size_cumsum),
            sorted_linear_indices_run_(sorted_linear_indices_run),
            sorted_linear_indices_cumulative_run_lengths_(sorted_linear_indices_cumulative_run_lengths),
            long_run_ids_(long_run_ids),
            num_long_run_ids_(num_long_run_ids),
            sorted_infos_(sorted_infos),
            grad_dev_weights_(grad_dev_weights),
            long_run_id_to_really_long_run_ids_(long_run_id_to_really_long_run_ids),
            temp_grad_accum_(temp_grad_accum),
            grad_accum_counter_(grad_accum_counter),
            max_segment_length_per_cta_(max_segment_length_per_cta),
            use_deterministic_algorithms_(use_deterministic_algorithms),
            max_vecs_per_thread_(max_vecs_per_thread),
            smem_(smem) {}

        void operator()(const sycl::nd_item<2>& item) const;

    private:
        const at::PackedTensorAccessor64<grad_t, 2, RestrictPtrTraits> grad_output_;
        mutable at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> dev_weights_; // if optimizer != "none"
        const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> weights_offsets_;
        int64_t D_;
        const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> hash_size_cumsum_;
        const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> sorted_linear_indices_run_;
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_linear_indices_cumulative_run_lengths_;
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> long_run_ids_;
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> num_long_run_ids_;
        const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> sorted_infos_;
        mutable at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> grad_dev_weights_; // if not dense and optimizer != "none"
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> long_run_id_to_really_long_run_ids_;
        mutable at::PackedTensorAccessor32<at::acc_type<cache_t, true>, 2, RestrictPtrTraits> temp_grad_accum_;
        mutable at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> grad_accum_counter_;
        int32_t max_segment_length_per_cta_;
        bool use_deterministic_algorithms_;
        int32_t max_vecs_per_thread_;
        sycl::local_accessor<cache_t, 1> smem_;
    };

template <
    typename grad_t,
    typename cache_t,
    int32_t kFixedMaxVecsPerThread,
    int32_t kThreadGroupSize,
    int32_t VEC_WIDTH,
    bool kUseVecBlocking
>
void compute_grad_sum_unweighted_nobag(
    const sycl::nd_item<2>& item,
    Vec4TAcc<cache_t>* grad_sum,
    Vec4TAcc<cache_t>* smem_grad_sum,
    const at::PackedTensorAccessor64<grad_t, 2>& grad_output,
    const int32_t D,
    const int32_t T,
    const at::PackedTensorAccessor32<int64_t, 1>& sorted_infos,
    const int32_t segment_start,
    const int32_t sl_start,
    const int32_t sl_end,
    const int32_t num_vecs
);

template<
    typename emb_t,
    typename cache_t,
    int32_t kFixedMaxVecsPerThread,
    int32_t kThreadGroupSize,
    int32_t VEC_WIDTH,
    bool kUseVecBlocking
>
void store_grad_sum(
    const sycl::nd_item<2>& item,
    at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits>& grad_dev_weights,
    const Vec4TAcc<cache_t>* grad_sum,
    const Vec4TAcc<cache_t>* smem_grad_sum,
    const int32_t D,
    const int64_t weights_offset,
    const int64_t idx,
    const int32_t max_vecs_per_thread
);

} // namespace fbgemm_xpu
