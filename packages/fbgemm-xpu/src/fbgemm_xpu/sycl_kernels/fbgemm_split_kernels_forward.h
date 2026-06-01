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
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE - FORWARD KERNELS
////////////////////////////////////////////////////////////////////////////////
//
// This file contains SYCL ports of FBGEMM split embedding forward kernels.
//
// ORIGINAL CUDA SOURCE:
//   File: fbgemm_gpu/_skbuild/linux-x86_64-3.10/cmake-build/gen_embedding_forward_split_unweighted_nobag_kernel.cu
//   File: fbgemm_gpu/_skbuild/linux-x86_64-3.10/cmake-build/gen_embedding_forward_split_unweighted_nobag_kernel_small.cu
//   Template: fbgemm_gpu/codegen/training/forward/embedding_forward_split_kernel_template.cu
//   Template: fbgemm_gpu/codegen/training/forward/embedding_forward_split_kernel_nobag_small_template.cu
//
// KERNEL MAPPING:
//   SplitEmbeddingNoBagForwardUnweightedKernel
//     → split_embedding_nobag_codegen_forward_unweighted_kernel (CUDA)
//
//   SplitEmbeddingNoBagForwardUnweightedSmallKernel
//     → split_embedding_nobag_codegen_forward_unweighted_small_kernel (CUDA)
//
// HOST FUNCTION MAPPING:
//   split_embedding_nobag_forward_unweighted_xpu
//     → split_embedding_nobag_codegen_forward_unweighted_cuda (CUDA)
//     CUDA File: fbgemm_gpu/_skbuild/linux-x86_64-3.10/cmake-build/gen_embedding_forward_split_unweighted_codegen_cuda.cu
//     CUDA Template: fbgemm_gpu/codegen/training/forward/embedding_forward_split_template.cu
//
//   split_embedding_nobag_forward_unweighted_pt2_xpu_wrapper
//     → split_embedding_nobag_codegen_forward_unweighted_pt2_cuda_wrapper (CUDA)
//     CUDA File: fbgemm_gpu/_skbuild/linux-x86_64-3.10/cmake-build/gen_embedding_forward_split_pt2_cuda_wrapper.cpp
//     CUDA Template: fbgemm_gpu/codegen/training/pt2/embedding_split_host_pt2_cuda_wrapper_template.cpp
//
// NOTE: The "codegen" string is removed from SYCL implementation names
//       for consistency with the refactored naming convention.
//
////////////////////////////////////////////////////////////////////////////////

/*
 * SYCL/XPU Implementation of split embedding forward kernel headers
 */

#pragma once

#include <sycl/sycl.hpp>
#include <c10/xpu/XPUStream.h>

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/csrc/autograd/record_function_ops.h>
#include <ATen/native/StridedRandomAccessor.h>

#include "../fbgemm_utils/utils.h"
#include "../fbgemm_utils/weight_row.h"
#include "../fbgemm_utils/tensor_utils.h"
#include "../fbgemm_utils/split_embeddings_cache_xpu.h"

using Tensor = at::Tensor;

using at::native::RestrictPtrTraits;

namespace fbgemm_xpu {
    #define DISPATCH_KERNEL_FOR_CACHE_CASE(CACHE_CASE_, ...)                       \
    [&] {                                                                        \
        if (CACHE_CASE_ == false) {                                      \
        constexpr auto use_cache_t = false;                            \
        return __VA_ARGS__();                                                    \
        }                                                                          \
        if (CACHE_CASE_ == true) {                                      \
        constexpr auto use_cache_t = true;                            \
        return __VA_ARGS__();                                                    \
        }                                                                          \
        return;                                                                    \
    }()

    ////////////////////////////////////////////////////////////////////////////////
    // SplitEmbeddingNoBagForwardUnweightedKernel - Device Kernel
    ////////////////////////////////////////////////////////////////////////////////
    //
    // CUDA SOURCE MAPPING:
    //   CUDA Kernel: split_embedding_nobag_codegen_forward_unweighted_kernel
    //   CUDA File: fbgemm_gpu/_skbuild/linux-x86_64-3.10/cmake-build/gen_embedding_forward_split_unweighted_nobag_kernel.cu
    //   CUDA Template: fbgemm_gpu/codegen/training/forward/embedding_forward_split_kernel_template.cu
    //
    // DESCRIPTION:
    //   Main forward kernel for no-bag embeddings (sequence embeddings).
    //   Supports cache-aware lookups with LXU cache for UVM-managed tables.
    //
    ////////////////////////////////////////////////////////////////////////////////
    template <
    typename emb_t,
    typename cache_t,
    typename output_t,
    bool use_lxu_cache,
    typename index_t,
    size_t kThreadGroupSize>
    class SplitEmbeddingNoBagForwardUnweightedKernel {
        public:
            SplitEmbeddingNoBagForwardUnweightedKernel(
                const at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> dev_weights,
                const at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> uvm_weights,
                const at::PackedTensorAccessor64<cache_t, 2, RestrictPtrTraits> lxu_cache_weights,
                const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> weights_placements,
                const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> weights_offsets,
                int64_t D,
                FixedDivisor fd_B,
                const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> indices,
                const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> offsets,
                const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> lxu_cache_locations,
                const int32_t* lxu_cache_conflict_misses,
                at::PackedTensorAccessor64<output_t, 2, RestrictPtrTraits> output
            ): dev_weights_(dev_weights),
               uvm_weights_(uvm_weights),
               lxu_cache_weights_(lxu_cache_weights),
               weights_placements_(weights_placements),
               weights_offsets_(weights_offsets),
               D_(D),
               fd_B_(fd_B),
               indices_(indices),
               offsets_(offsets),
               lxu_cache_locations_(lxu_cache_locations),
               lxu_cache_conflict_misses_(lxu_cache_conflict_misses),
               output_(output) {}

            void operator()(const sycl::nd_item<2>& item) const;

        private:
            const at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> dev_weights_;
            const at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> uvm_weights_;
            const at::PackedTensorAccessor64<cache_t, 2, RestrictPtrTraits> lxu_cache_weights_;
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> weights_placements_;
            const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> weights_offsets_;
            const int64_t D_;
            FixedDivisor fd_B_;
            const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> indices_;
            const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> offsets_;
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> lxu_cache_locations_;
            const int32_t* lxu_cache_conflict_misses_;
            mutable at::PackedTensorAccessor64<output_t, 2, RestrictPtrTraits> output_;
    };

    ////////////////////////////////////////////////////////////////////////////////
    // SplitEmbeddingNoBagForwardUnweightedSmallKernel - Device Kernel (Small D)
    ////////////////////////////////////////////////////////////////////////////////
    //
    // CUDA SOURCE MAPPING:
    //   CUDA Kernel: split_embedding_nobag_codegen_forward_unweighted_small_kernel
    //   CUDA File: fbgemm_gpu/_skbuild/linux-x86_64-3.10/cmake-build/gen_embedding_forward_split_unweighted_nobag_kernel_small.cu
    //   CUDA Template: fbgemm_gpu/codegen/training/forward/embedding_forward_split_kernel_nobag_small_template.cu
    //
    // DESCRIPTION:
    //   Optimized forward kernel for small embedding dimensions (D <= 32).
    //   Uses sub-group shuffle operations for efficient small-dimension lookups.
    //
    ////////////////////////////////////////////////////////////////////////////////
    template <
    typename emb_t,
    typename cache_t,
    typename output_t,
    typename index_t,
    size_t kThreadGroupSize>
    class SplitEmbeddingNoBagForwardUnweightedSmallKernel {
        public:
            SplitEmbeddingNoBagForwardUnweightedSmallKernel(
                const at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> dev_weights,
                const at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> uvm_weights,
                const at::PackedTensorAccessor64<cache_t, 2, RestrictPtrTraits> lxu_cache_weights,
                const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> weights_placements,
                const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> weights_offsets,
                int64_t D,
                FixedDivisor fd_B,
                const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> indices,
                const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> offsets,
                const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> lxu_cache_locations,
                at::PackedTensorAccessor64<output_t, 2, RestrictPtrTraits> output
            ): dev_weights_(dev_weights),
               uvm_weights_(uvm_weights),
               lxu_cache_weights_(lxu_cache_weights),
               weights_placements_(weights_placements),
               weights_offsets_(weights_offsets),
               D_(D),
               fd_B_(fd_B),
               indices_(indices),
               offsets_(offsets),
               lxu_cache_locations_(lxu_cache_locations),
               output_(output) {}

            void operator()(const sycl::nd_item<2>& item) const;

        private:
            const at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> dev_weights_;
            const at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> uvm_weights_;
            const at::PackedTensorAccessor64<cache_t, 2, RestrictPtrTraits> lxu_cache_weights_;
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> weights_placements_;
            const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> weights_offsets_;
            const int64_t D_;
            FixedDivisor fd_B_;
            const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> indices_;
            const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> offsets_;
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> lxu_cache_locations_;
            mutable at::PackedTensorAccessor64<output_t, 2, RestrictPtrTraits> output_;
    };

    ////////////////////////////////////////////////////////////////////////////////
    // split_embedding_nobag_forward_unweighted_xpu - Host Function
    ////////////////////////////////////////////////////////////////////////////////
    //
    // CUDA SOURCE MAPPING:
    //   CUDA Function: split_embedding_nobag_codegen_forward_unweighted_cuda
    //   CUDA File: fbgemm_gpu/_skbuild/linux-x86_64-3.10/cmake-build/gen_embedding_forward_split_unweighted_codegen_cuda.cu
    //   CUDA Template: fbgemm_gpu/codegen/training/forward/embedding_forward_split_template.cu
    //
    // DESCRIPTION:
    //   Host function for no-bag embedding forward pass.
    //   Retrieves embedding vectors without pooling (sequence embeddings).
    //   Dispatches to optimized kernel based on embedding dimension size.
    //
    ////////////////////////////////////////////////////////////////////////////////
    Tensor split_embedding_nobag_forward_unweighted_xpu(
            const Tensor& dev_weights,
            const Tensor& uvm_weights,
            const Tensor& lxu_cache_weights,
            const Tensor& weights_placements,
            const Tensor& weights_offsets,
            const c10::SymInt D_,
            const Tensor& indices,
            const Tensor& offsets,
            const Tensor& lxu_cache_locations,
            const Tensor& uvm_cache_stats,
            const int64_t output_dtype,
            const bool is_experimental
        );

    ////////////////////////////////////////////////////////////////////////////////
    // split_embedding_nobag_forward_unweighted_pt2_xpu_wrapper - PT2 Wrapper
    ////////////////////////////////////////////////////////////////////////////////
    //
    // CUDA SOURCE MAPPING:
    //   CUDA Function: split_embedding_nobag_codegen_forward_unweighted_pt2_cuda_wrapper
    //   CUDA File: fbgemm_gpu/_skbuild/linux-x86_64-3.10/cmake-build/gen_embedding_forward_split_pt2_cuda_wrapper.cpp
    //   CUDA Template: fbgemm_gpu/codegen/training/pt2/embedding_split_host_pt2_cuda_wrapper_template.cpp
    //
    // DESCRIPTION:
    //   PT2 (PyTorch 2.0) compilation wrapper for split embedding forward pass.
    //   Dispatches to split_embedding_nobag_forward_unweighted_xpu via PyTorch dispatcher.
    //   Supports SymInt for dynamic shapes and compile-time optimization.
    //
    ////////////////////////////////////////////////////////////////////////////////
    Tensor split_embedding_nobag_forward_unweighted_pt2_xpu_wrapper(
        const Tensor& /*host_weights*/,
        const Tensor& dev_weights,
        const Tensor& uvm_weights,
        const Tensor& lxu_cache_weights,
        const Tensor& weights_placements,
        const Tensor& weights_offsets,
        const c10::SymInt D,
        const Tensor& hash_size_cumsum,
        const Tensor& indices,
        const Tensor& offsets,
        const Tensor& lxu_cache_locations,
        const Tensor& uvm_cache_stats,
        const bool is_experimental,
        const int64_t output_dtype
        );
        
} // namespace fbgemm_xpu