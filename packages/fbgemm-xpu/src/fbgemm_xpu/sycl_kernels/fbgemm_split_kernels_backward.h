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
// This file contains SYCL ports of FBGEMM split embedding backward kernels
// with rowwise Adagrad optimizer.
//
// ORIGINAL CUDA SOURCE:
//   File: fbgemm_gpu/_skbuild/linux-x86_64-3.10/cmake-build/gen_embedding_backward_rowwise_adagrad_split_unweighted_nobag_kernel_cta.cu
//   File: fbgemm_gpu/_skbuild/linux-x86_64-3.10/cmake-build/gen_embedding_backward_rowwise_adagrad_split_unweighted_nobag_kernel_warp.cu
//   Template: fbgemm_gpu/codegen/training/backward/embedding_backward_split_kernel_cta_template.cu
//   Template: fbgemm_gpu/codegen/training/backward/embedding_backward_split_kernel_warp_template.cu
//
// KERNEL MAPPING:
//   SplitEmbeddingBackwardCountUniqueIndicesKernel
//     → split_embedding_backward_count_unique_indices_kernel (CUDA)
//     CUDA File: fbgemm_gpu/codegen/training/backward/embedding_backward_split_grad_template.cu
//
//   SplitEmbeddingNobagBackwardRowwiseAdagradUnweightedKernelCTAPerRow
//     → split_embedding_nobag_backward_codegen_rowwise_adagrad_unweighted_kernel_cta_per_row_1 (CUDA)
//
//   SplitEmbeddingNobagBackwardRowwiseAdagradUnweightedKernelWarpPerRow
//     → split_embedding_nobag_backward_codegen_rowwise_adagrad_unweighted_kernel_warp_per_row_1 (CUDA)
//
// HOST FUNCTION MAPPING:
//   split_embedding_nobag_backward_rowwise_adagrad_unweighted_exact_xpu
//     → split_embedding_nobag_backward_codegen_rowwise_adagrad_unweighted_exact_cuda (CUDA)
//     CUDA File: fbgemm_gpu/_skbuild/linux-x86_64-3.10/cmake-build/gen_embedding_backward_rowwise_adagrad_split_unweighted_nobag_cuda.cu
//     CUDA Template: fbgemm_gpu/codegen/training/backward/embedding_backward_split_template.cu
//
//   split_embedding_nobag_backward_rowwise_adagrad_unweighted_pt2_xpu_wrapper
//     → split_embedding_nobag_backward_codegen_rowwise_adagrad_unweighted_pt2_cuda_wrapper (CUDA)
//     CUDA File: fbgemm_gpu/_skbuild/linux-x86_64-3.10/cmake-build/gen_embedding_backward_split_rowwise_adagrad_pt2_cuda_wrapper.cpp
//     CUDA Template: fbgemm_gpu/codegen/training/pt2/embedding_split_host_pt2_cuda_wrapper_template.cpp
//
// NOTE: The "codegen" string is removed from SYCL implementation names
//       for consistency with the refactored naming convention.
//
////////////////////////////////////////////////////////////////////////////////

/*
 * SYCL/XPU Implementation of split embedding backward kernel headers
 */

#pragma once

#include <sycl/sycl.hpp>
#include <c10/xpu/XPUStream.h>

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/csrc/autograd/record_function_ops.h>
#include <ATen/native/StridedRandomAccessor.h>
#include <ATen/xpu/XPUGeneratorImpl.h>

#include "../fbgemm_utils/utils.h"
#include "../fbgemm_utils/tensor_utils.h"
#include "../fbgemm_utils/split_embeddings_cache_xpu.h"
#include "../fbgemm_utils/backward_utils.h"
#include "../fbgemm_utils/weight_row.h"

using Tensor = at::Tensor;

using namespace fbgemm_xpu::utils;
using at::native::RestrictPtrTraits;
using float4 = sycl::float4;

namespace fbgemm_xpu {
    #define DISPATCH_PLACEHOLDER_TYPES(NAME, ...) \
    return __VA_ARGS__();

    Tensor split_embedding_nobag_backward_rowwise_adagrad_unweighted_pt2_xpu_wrapper(
        const Tensor& grad_output,
        const Tensor& /*host_weights*/,
        const Tensor& dev_weights,
        const Tensor& uvm_weights,
        const Tensor& lxu_cache_weights,
        const Tensor& weights_placements,
        const Tensor& weights_offsets,
        const c10::SymInt D,
        const Tensor& hash_size_cumsum,
        const int64_t total_hash_size_bits,
        const Tensor& indices,
        const Tensor& offsets,
        const Tensor& lxu_cache_locations,
        const int64_t BT_block_size,
        const int64_t max_segment_length_per_warp,
        const bool stochastic_rounding,
        const int64_t info_B_num_bits,
        const int64_t info_B_mask_int64,
        const bool use_uniq_cache_locations,
        const bool use_homogeneous_placements,
        Tensor momentum1_host, 
        Tensor momentum1_dev, 
        Tensor momentum1_uvm, 
        Tensor momentum1_placements, 
        Tensor momentum1_offsets, 
        Tensor learning_rate_tensor, 
        double eps = 0, 
        double weight_decay = 0.0, 
        int64_t weight_decay_mode = 0, 
        double max_norm = 0.0);

    ////////////////////////////////////////////////////////////////////////////////
    // split_embedding_nobag_backward_rowwise_adagrad_unweighted_exact_xpu
    ////////////////////////////////////////////////////////////////////////////////
    //
    // CUDA SOURCE MAPPING:
    //   CUDA Function: split_embedding_nobag_backward_codegen_rowwise_adagrad_unweighted_exact_cuda
//   CUDA File: fbgemm_gpu/_skbuild/linux-x86_64-3.10/cmake-build/gen_embedding_backward_rowwise_adagrad_split_unweighted_nobag_cuda.cu
//   CUDA Template: fbgemm_gpu/codegen/training/backward/embedding_backward_split_template.cu
    //
    // DESCRIPTION:
    //   Host function for no-bag embedding backward pass with rowwise Adagrad.
    //   Computes gradients and updates embedding weights using Adagrad optimizer.
    //   Dispatches to CTA-per-row or warp-per-row kernels based on segment length.
    //
    ////////////////////////////////////////////////////////////////////////////////
    Tensor split_embedding_nobag_backward_rowwise_adagrad_unweighted_exact_xpu(
        const Tensor& grad_output,
        const Tensor& dev_weights,
        const Tensor& uvm_weights,
        const Tensor& lxu_cache_weights,
        const Tensor& weights_placements,
        const Tensor& weights_offsets,
        const c10::SymInt D_,
        const Tensor& hash_size_cumsum,
        const int64_t total_hash_size_bits,
        const Tensor& indices,
        const Tensor& offsets,
        const Tensor& lxu_cache_locations,
        const int64_t unused_,
        const int64_t max_segment_length_per_warp,
        const bool stochastic_rounding,
        const int64_t info_B_num_bits, // int32_t
        const int64_t info_B_mask_int64, // uint32_t
        const bool use_uniq_cache_locations,
        const bool use_homogeneous_placements,
        Tensor momentum1_dev,
        Tensor momentum1_uvm,
        Tensor momentum1_placements,
        Tensor momentum1_offsets,
        Tensor learning_rate_tensor,
        double eps,
        double weight_decay,
        int64_t weight_decay_mode,
        double max_norm
    );

    template <typename info_pta_t, typename info_t, bool nobag>
    class SplitEmbeddingBackwardCountUniqueIndicesKernel {
    public:
        SplitEmbeddingBackwardCountUniqueIndicesKernel(
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
                sorted_linear_indices_num_runs,
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
                sorted_linear_indices_cumulative_run_lengths,
            const at::PackedTensorAccessor32<info_pta_t, 1, RestrictPtrTraits>
                sorted_infos,
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
                weights_placements,
            at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
                dev_or_uvm_unique_indices,
            const int info_B_num_bits
        ) : sorted_linear_indices_num_runs_(sorted_linear_indices_num_runs),
            sorted_linear_indices_cumulative_run_lengths_(sorted_linear_indices_cumulative_run_lengths),
            sorted_infos_(sorted_infos),
            weights_placements_(weights_placements),
            dev_or_uvm_unique_indices_(dev_or_uvm_unique_indices),
            info_B_num_bits_(info_B_num_bits) {};

        void operator()(const sycl::nd_item<1>& item) const;
    
    private:
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_linear_indices_num_runs_;
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_linear_indices_cumulative_run_lengths_;
        const at::PackedTensorAccessor32<info_pta_t, 1, RestrictPtrTraits> sorted_infos_;
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> weights_placements_;
        mutable at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> dev_or_uvm_unique_indices_;
        const int info_B_num_bits_;
    };

    ////////////////////////////////////////////////////////////////////////////////
    // SplitEmbeddingNobagBackwardRowwiseAdagradUnweightedKernelCTAPerRow
    ////////////////////////////////////////////////////////////////////////////////
    //
    // CUDA SOURCE MAPPING:
    //   CUDA Kernel: split_embedding_nobag_backward_codegen_rowwise_adagrad_unweighted_kernel_cta_per_row_1
//   CUDA File: fbgemm_gpu/_skbuild/linux-x86_64-3.10/cmake-build/gen_embedding_backward_rowwise_adagrad_split_unweighted_nobag_kernel_cta.cu
//   CUDA Template: fbgemm_gpu/codegen/training/backward/embedding_backward_split_kernel_cta_template.cu
    //
    // DESCRIPTION:
    //   Backward kernel for long segments (processed by CTA/work-group).
    //   Uses rowwise Adagrad optimizer for gradient updates.
    //   One CTA processes one embedding row for segments exceeding max_segment_length_per_warp.
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
    class SplitEmbeddingNobagBackwardRowwiseAdagradUnweightedKernelCTAPerRow {
    public:        
        SplitEmbeddingNobagBackwardRowwiseAdagradUnweightedKernelCTAPerRow(
            const at::PackedTensorAccessor64<grad_t, 2, RestrictPtrTraits> grad_output,
            at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> dev_weights,
            at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> uvm_weights,
            at::PackedTensorAccessor64<cache_t, 2, RestrictPtrTraits> lxu_cache_weights,
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> weights_placements, // if optimizer != "none"
            const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> weights_offsets,
            int64_t D,
            const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> hash_size_cumsum,
            const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> sorted_linear_indices_run,
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_linear_indices_cumulative_run_lengths,
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> long_run_ids,
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> num_long_run_ids,
            const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> sorted_infos,
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_lxu_cache_locations,
            const bool use_uniq_cache_locations,
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> table_unique_indices_offsets,
            bool stochastic_rounding,
            PhiloxXpuState stochastic_rounding_philox_args, // if not dense and optimizer != "none"
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> long_run_id_to_really_long_run_ids,
            at::PackedTensorAccessor32<at::acc_type<cache_t, true>, 2, RestrictPtrTraits> temp_grad_accum,
            at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> grad_accum_counter,
            const int32_t max_segment_length_per_cta,
            const bool use_deterministic_algorithms,
            const int32_t max_vecs_per_thread,
            at::PackedTensorAccessor64<at::acc_type<cache_t, true>, 1, RestrictPtrTraits> momentum1_dev,
            at::PackedTensorAccessor64<at::acc_type<cache_t, true>, 1, RestrictPtrTraits> momentum1_uvm,
            at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> momentum1_placements,
            at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> momentum1_offsets,
            sycl::local_accessor<cache_t, 1> smem,
            float learning_rate = 0,
            float eps = 0,
            float weight_decay = 0.0,
            int64_t weight_decay_mode = 0,
            float max_norm = 0.0
        ) : grad_output_(grad_output),
            dev_weights_(dev_weights),
            uvm_weights_(uvm_weights),
            lxu_cache_weights_(lxu_cache_weights),
            weights_placements_(weights_placements),
            weights_offsets_(weights_offsets),
            D_(D),
            hash_size_cumsum_(hash_size_cumsum),
            sorted_linear_indices_run_(sorted_linear_indices_run),
            sorted_linear_indices_cumulative_run_lengths_(sorted_linear_indices_cumulative_run_lengths),
            long_run_ids_(long_run_ids),
            num_long_run_ids_(num_long_run_ids),
            sorted_infos_(sorted_infos),
            sorted_lxu_cache_locations_(sorted_lxu_cache_locations),
            use_uniq_cache_locations_(use_uniq_cache_locations),
            table_unique_indices_offsets_(table_unique_indices_offsets),
            stochastic_rounding_(stochastic_rounding),
            stochastic_rounding_philox_args_(stochastic_rounding_philox_args),
            long_run_id_to_really_long_run_ids_(long_run_id_to_really_long_run_ids),
            temp_grad_accum_(temp_grad_accum),
            grad_accum_counter_(grad_accum_counter),
            max_segment_length_per_cta_(max_segment_length_per_cta),
            use_deterministic_algorithms_(use_deterministic_algorithms),
            max_vecs_per_thread_(max_vecs_per_thread),
            momentum1_dev_(momentum1_dev),
            momentum1_uvm_(momentum1_uvm),
            momentum1_placements_(momentum1_placements),
            momentum1_offsets_(momentum1_offsets),
            smem_(smem),
            learning_rate_(learning_rate),
            eps_(eps),
            weight_decay_(weight_decay),
            weight_decay_mode_(weight_decay_mode),
            max_norm_(max_norm) {};

            void operator()(const sycl::nd_item<2>& item) const;

        private:
            const at::PackedTensorAccessor64<grad_t, 2, RestrictPtrTraits> grad_output_;
            mutable at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> dev_weights_;
            mutable at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> uvm_weights_;
            mutable at::PackedTensorAccessor64<cache_t, 2, RestrictPtrTraits> lxu_cache_weights_;
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> weights_placements_;
            const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> weights_offsets_;
            int64_t D_;
            const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> hash_size_cumsum_;
            const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> sorted_linear_indices_run_;
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_linear_indices_cumulative_run_lengths_;
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> long_run_ids_;
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> num_long_run_ids_;
            const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> sorted_infos_;
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_lxu_cache_locations_;
            const bool use_uniq_cache_locations_;
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> table_unique_indices_offsets_;
            bool stochastic_rounding_;
            PhiloxXpuState stochastic_rounding_philox_args_;
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> long_run_id_to_really_long_run_ids_;
            mutable at::PackedTensorAccessor32<at::acc_type<cache_t, true>, 2, RestrictPtrTraits> temp_grad_accum_;
            mutable at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> grad_accum_counter_;
            const int32_t max_segment_length_per_cta_;
            const bool use_deterministic_algorithms_;
            const int32_t max_vecs_per_thread_;
            mutable at::PackedTensorAccessor64<at::acc_type<cache_t, true>, 1, RestrictPtrTraits> momentum1_dev_;
            mutable at::PackedTensorAccessor64<at::acc_type<cache_t, true>, 1, RestrictPtrTraits> momentum1_uvm_;
            mutable at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> momentum1_placements_;
            mutable at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> momentum1_offsets_;
            float learning_rate_;
            float eps_;
            float weight_decay_;
            int64_t weight_decay_mode_;
            float max_norm_;
            sycl::local_accessor<cache_t, 1> smem_;
    };

    ////////////////////////////////////////////////////////////////////////////////
    // SplitEmbeddingNobagBackwardRowwiseAdagradUnweightedKernelWarpPerRow
    ////////////////////////////////////////////////////////////////////////////////
    //
    // CUDA SOURCE MAPPING:
    //   CUDA Kernel: split_embedding_nobag_backward_codegen_rowwise_adagrad_unweighted_kernel_warp_per_row_1
//   CUDA File: fbgemm_gpu/_skbuild/linux-x86_64-3.10/cmake-build/gen_embedding_backward_rowwise_adagrad_split_unweighted_nobag_kernel_warp.cu
//   CUDA Template: fbgemm_gpu/codegen/training/backward/embedding_backward_split_kernel_warp_template.cu
    //
    // DESCRIPTION:
    //   Backward kernel for short segments (processed by warp/sub-group).
    //   Uses rowwise Adagrad optimizer for gradient updates.
    //   One warp processes one embedding row for segments within max_segment_length_per_warp.
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
    class SplitEmbeddingNobagBackwardRowwiseAdagradUnweightedKernelWarpPerRow {
    public:
        SplitEmbeddingNobagBackwardRowwiseAdagradUnweightedKernelWarpPerRow(
            const at::PackedTensorAccessor64<grad_t, 2, RestrictPtrTraits> grad_output,
            at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> dev_weights,
            at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> uvm_weights,
            at::PackedTensorAccessor64<cache_t, 2, RestrictPtrTraits> lxu_cache_weights,
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> weights_placements,
            const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> weights_offsets,
            int64_t D,
            const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> hash_size_cumsum,
            const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> sorted_linear_indices_run,
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_linear_indices_cumulative_run_lengths,
            const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> sorted_infos,
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_lxu_cache_locations,
            const bool use_uniq_cache_locations,
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> table_unique_indices_offsets,
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_linear_indices_num_runs,
            int32_t max_segment_length_per_warp,
            bool stochastic_rounding,
            PhiloxXpuState stochastic_rounding_philox_args, // if not dense and optimizer != "none"
            const int32_t max_D,
            const int32_t max_vecs_per_thread,
            at::PackedTensorAccessor64<at::acc_type<cache_t, true>, 1, RestrictPtrTraits> momentum1_dev,
            at::PackedTensorAccessor64<at::acc_type<cache_t, true>, 1, RestrictPtrTraits> momentum1_uvm,
            at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> momentum1_placements,
            at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> momentum1_offsets,
            sycl::local_accessor<cache_t, 1> smem,
            float learning_rate = 0,
            float eps = 0,
            float weight_decay = 0.0,
            int64_t weight_decay_mode = 0,
            float max_norm = 0.0
        ):  grad_output_(grad_output),
            dev_weights_(dev_weights),
            uvm_weights_(uvm_weights),
            lxu_cache_weights_(lxu_cache_weights),
            weights_placements_(weights_placements),
            weights_offsets_(weights_offsets),
            D_(D),
            hash_size_cumsum_(hash_size_cumsum),
            sorted_linear_indices_run_(sorted_linear_indices_run),
            sorted_linear_indices_cumulative_run_lengths_(sorted_linear_indices_cumulative_run_lengths),
            sorted_infos_(sorted_infos),
            sorted_lxu_cache_locations_(sorted_lxu_cache_locations),
            use_uniq_cache_locations_(use_uniq_cache_locations),
            table_unique_indices_offsets_(table_unique_indices_offsets),
            sorted_linear_indices_num_runs_(sorted_linear_indices_num_runs),
            max_segment_length_per_warp_(max_segment_length_per_warp),
            stochastic_rounding_(stochastic_rounding),
            stochastic_rounding_philox_args_(stochastic_rounding_philox_args),
            max_D_(max_D),
            max_vecs_per_thread_(max_vecs_per_thread),
            momentum1_dev_(momentum1_dev),
            momentum1_uvm_(momentum1_uvm),
            momentum1_placements_(momentum1_placements),
            momentum1_offsets_(momentum1_offsets),
            smem_(smem),
            learning_rate_(learning_rate),
            eps_(eps),
            weight_decay_(weight_decay),
            weight_decay_mode_(weight_decay_mode),
            max_norm_(max_norm) {};

        void operator()(const sycl::nd_item<2>& item) const;

    private:
        const at::PackedTensorAccessor64<grad_t, 2, RestrictPtrTraits> grad_output_;
        mutable at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> dev_weights_;
        mutable at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> uvm_weights_;
        mutable at::PackedTensorAccessor64<cache_t, 2, RestrictPtrTraits> lxu_cache_weights_;
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> weights_placements_;
        const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> weights_offsets_;
        int64_t D_;
        const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> hash_size_cumsum_;
        const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> sorted_linear_indices_run_;
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_linear_indices_cumulative_run_lengths_;
        const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> sorted_infos_;
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_lxu_cache_locations_;
        const bool use_uniq_cache_locations_;
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> table_unique_indices_offsets_;
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_linear_indices_num_runs_;
        int32_t max_segment_length_per_warp_;
        bool stochastic_rounding_;
        PhiloxXpuState stochastic_rounding_philox_args_;
        const int32_t max_D_;
        const int32_t max_vecs_per_thread_;
        mutable at::PackedTensorAccessor64<at::acc_type<cache_t, true>, 1, RestrictPtrTraits> momentum1_dev_;
        mutable at::PackedTensorAccessor64<at::acc_type<cache_t, true>, 1, RestrictPtrTraits> momentum1_uvm_;
        mutable at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> momentum1_placements_;
        mutable at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> momentum1_offsets_;
        sycl::local_accessor<cache_t, 1> smem_;
        float learning_rate_;
        float eps_;
        float weight_decay_;
        int64_t weight_decay_mode_;
        float max_norm_;
    };
} // namespace fbgemm_xpu
