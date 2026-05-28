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
// This file contains SYCL ports of FBGEMM dense embedding forward kernels.
//
// ORIGINAL CUDA SOURCE:
//   File: gen_embedding_forward_dense_unweighted_codegen_cuda.cu
//   Path: FBGEMM/fbgemm_gpu/_skbuild/linux-x86_64-3.10/cmake-build/ (generated files)
//   Template: training/forward/embedding_forward_split_template.cu
//
// KERNEL MAPPING:
//   DenseEmbeddingNobagCodegenForwardUnweightedKernel
//     → dense_embedding_nobag_codegen_forward_unweighted_kernel (CUDA)
//
//   DenseEmbeddingNobagCodegenForwardUnweightedSmallKernel
//     → dense_embedding_nobag_codegen_forward_unweighted_small_kernel (CUDA)
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

#include "../fbgemm_utils/tensor_utils.h"
#include "../fbgemm_utils/utils.h"
#include "../fbgemm_utils/weight_row.h"

using Tensor = at::Tensor;
using at::native::RestrictPtrTraits;

namespace fbgemm_xpu {

    ////////////////////////////////////////////////////////////////////////////////
    // dense_embedding_nobag_forward_unweighted_xpu - Host Function
    ////////////////////////////////////////////////////////////////////////////////
    //
    // CUDA SOURCE MAPPING:
    //   CUDA Function: dense_embedding_nobag_codegen_forward_unweighted_cuda
    //   CUDA File: gen_embedding_forward_dense_unweighted_codegen_cuda.cu
    //
    // DESCRIPTION:
    //   Host function for no-bag embedding forward pass.
    //   Retrieves embedding vectors without pooling (sequence embeddings).
    //
    ////////////////////////////////////////////////////////////////////////////////
    Tensor dense_embedding_nobag_forward_unweighted_xpu(
        const Tensor& dev_weights,
        const Tensor& weights_offsets,
        const c10::SymInt D_,
        const Tensor& indices,
        const Tensor& offsets,
        const int64_t output_dtype,
        const bool is_experimental
    );

    ////////////////////////////////////////////////////////////////////////////////
    // DenseEmbeddingNobagForwardUnweightedKernel - Main Forward Kernel
    ////////////////////////////////////////////////////////////////////////////////
    //
    // CUDA SOURCE MAPPING:
    //   CUDA Kernel: dense_embedding_nobag_codegen_forward_unweighted_kernel
    //   CUDA File: gen_embedding_forward_dense_unweighted_nobag_kernel.cu
    //
    // DESCRIPTION:
    //   Main kernel for no-bag forward pass with general embedding dimensions.
    //   Each thread retrieves one embedding vector and copies to output.
    //
    ////////////////////////////////////////////////////////////////////////////////
    template <
        typename emb_t,
        typename cache_t,
        typename output_t,
        typename index_t,
        size_t kThreadGroupSize>
    class DenseEmbeddingNobagForwardUnweightedKernel {
        public:
            DenseEmbeddingNobagForwardUnweightedKernel(
                const at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> dev_weights, 
                const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> weights_offsets, 
                const int64_t D, 
                FixedDivisor fd_B,
                const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> indices, 
                const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> offsets,
                at::PackedTensorAccessor64<output_t, 2, RestrictPtrTraits> output
            )
            : dev_weights_(dev_weights), weights_offsets_(weights_offsets), D_(D), fd_B_(fd_B),
            indices_(indices), offsets_(offsets), output_(output) {}
            
            void operator()(const sycl::nd_item<2>& item) const;

        private:
            const at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> dev_weights_;
            const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> weights_offsets_;
            const int64_t D_;
            FixedDivisor fd_B_;
            const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> indices_;
            const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> offsets_;
            mutable at::PackedTensorAccessor64<output_t, 2, RestrictPtrTraits> output_;
    };

    ////////////////////////////////////////////////////////////////////////////////
    // DenseEmbeddingNobagForwardUnweightedSmallKernel - Optimized for Small D
    ////////////////////////////////////////////////////////////////////////////////
    //
    // CUDA SOURCE MAPPING:
    //   CUDA Kernel: dense_embedding_nobag_codegen_forward_unweighted_small_kernel
    //   CUDA File: gen_embedding_forward_dense_unweighted_nobag_kernel.cu
    //
    // DESCRIPTION:
    //   Optimized kernel for small embedding dimensions (D <= 32).
    //   Uses fewer threads per workgroup for better occupancy.
    //
    ////////////////////////////////////////////////////////////////////////////////
    template <
        typename emb_t,
        typename cache_t,
        typename output_t,
        typename index_t,
        size_t kThreadGroupSize>
    class DenseEmbeddingNobagForwardUnweightedSmallKernel {
        public:
            DenseEmbeddingNobagForwardUnweightedSmallKernel(
                const at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> dev_weights,
                const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> weights_offsets,
                int64_t D,
                FixedDivisor fd_B,
                const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> indices,
                const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> offsets,
                at::PackedTensorAccessor64<output_t, 2, RestrictPtrTraits> output
            )
            : dev_weights_(dev_weights), weights_offsets_(weights_offsets), D_(D), fd_B_(fd_B),
              indices_(indices), offsets_(offsets), output_(output) {}

            void operator()(const sycl::nd_item<2>& item_ct1) const;

        private:
            const at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> dev_weights_;
            const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> weights_offsets_;
            int64_t D_;
            FixedDivisor fd_B_;
            const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> indices_;
            const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> offsets_;
            mutable at::PackedTensorAccessor64<output_t, 2, RestrictPtrTraits> output_;
    };

} // namespace fbgemm_xpu
