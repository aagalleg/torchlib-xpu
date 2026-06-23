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

{#
// @lint-ignore LINTIGNORE
// @lint-ignore-every CLANGFORMAT
// clang-format off
#}

{%- set mdesc = "dense" if dense else "split" %}

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE - FORWARD KERNELS
////////////////////////////////////////////////////////////////////////////////
//
// This file contains SYCL ports of FBGEMM {{ mdesc }} embedding forward kernels.
//
// ORIGINAL CUDA SOURCE:
//   Template: fbgemm_gpu/codegen/training/forward/embedding_forward_split_kernel_nobag_small_template.cu
//
// KERNEL MAPPING:
//   {{ mdesc | capitalize }}EmbeddingNobagForwardUnweightedSmallKernel
//     → {{ mdesc }}_embedding_nobag_codegen_forward_unweighted_small_kernel (CUDA)
//
////////////////////////////////////////////////////////////////////////////////

/*
 * SYCL/XPU Implementation of {{ mdesc }} embedding forward kernel headers (Small D optimization)
 */

#pragma once

{%- if dense %}
#include <cassert>
#include <cstdlib>
{%- endif %}

#include <sycl/sycl.hpp>
#include <c10/xpu/XPUStream.h>

{%- if dense %}
#include <ATen/ATen.h>
{%- endif %}
#include <ATen/Operators.h>
{%- if dense %}
#include <ATen/core/TensorAccessor.h>
{%- endif %}
#include <torch/all.h>
#include <torch/library.h>
#include <torch/csrc/autograd/record_function_ops.h>
#include <ATen/native/StridedRandomAccessor.h>

{%- if dense %}
#include "../fbgemm_utils/tensor_utils.h"
{%- endif %}
#include "../fbgemm_utils/utils.h"
#include "../fbgemm_utils/weight_row.h"
{%- if not dense %}
#include "../fbgemm_utils/tensor_utils.h"
#include "../fbgemm_utils/split_embeddings_cache_xpu.h"
{%- endif %}

using Tensor = at::Tensor;

using at::native::RestrictPtrTraits;

namespace fbgemm_xpu {

    ////////////////////////////////////////////////////////////////////////////////
    // {{ mdesc | capitalize }}EmbeddingNobagForwardUnweightedSmallKernel - Device Kernel (Small D)
    ////////////////////////////////////////////////////////////////////////////////
    //
    // CUDA SOURCE MAPPING:
    //   CUDA Kernel: {{ mdesc }}_embedding_nobag_codegen_forward_unweighted_small_kernel
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
    class {{ mdesc | capitalize }}EmbeddingNobagForwardUnweightedSmallKernel {
        public:
            {{ mdesc | capitalize }}EmbeddingNobagForwardUnweightedSmallKernel(
                const at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> dev_weights,
                {%- if not dense %}
                const at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> uvm_weights,
                const at::PackedTensorAccessor64<cache_t, 2, RestrictPtrTraits> lxu_cache_weights,
                const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> weights_placements,
                {%- endif %}
                const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> weights_offsets,
                int64_t D,
                FixedDivisor fd_B,
                const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> indices,
                const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> offsets,
                {%- if not dense %}
                const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> lxu_cache_locations,
                {%- endif %}
                at::PackedTensorAccessor64<output_t, 2, RestrictPtrTraits> output
            ): dev_weights_(dev_weights),
               {%- if not dense %}
               uvm_weights_(uvm_weights),
               lxu_cache_weights_(lxu_cache_weights),
               weights_placements_(weights_placements),
               {%- endif %}
               weights_offsets_(weights_offsets),
               D_(D),
               fd_B_(fd_B),
               indices_(indices),
               offsets_(offsets),
               {%- if not dense %}
               lxu_cache_locations_(lxu_cache_locations),
               {%- endif %}
               output_(output) {}

            void operator()(const sycl::nd_item<2>& item) const;

        private:
            const at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> dev_weights_;
            {%- if not dense %}
            const at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> uvm_weights_;
            const at::PackedTensorAccessor64<cache_t, 2, RestrictPtrTraits> lxu_cache_weights_;
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> weights_placements_;
            {%- endif %}
            const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> weights_offsets_;
            const int64_t D_;
            FixedDivisor fd_B_;
            const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> indices_;
            const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> offsets_;
            {%- if not dense %}
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> lxu_cache_locations_;
            {%- endif %}
            mutable at::PackedTensorAccessor64<output_t, 2, RestrictPtrTraits> output_;
    };

} // namespace fbgemm_xpu
