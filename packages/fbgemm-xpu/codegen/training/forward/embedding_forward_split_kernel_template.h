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
// Note: clang-format off doesn't work with this templaterized code,
// so we need to keep lint-ignore-every.
#}

{%- set mdesc = "dense" if dense else ("ssd" if ssd else "split") %}

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE - FORWARD KERNELS
////////////////////////////////////////////////////////////////////////////////
//
// This file contains SYCL ports of FBGEMM {{ mdesc }} embedding forward kernels.
//
// ORIGINAL CUDA SOURCE:
//   File: fbgemm_gpu/_skbuild/linux-x86_64-3.10/cmake-build/gen_embedding_forward_{{ mdesc }}_unweighted_nobag_kernel.cu
//   Template: fbgemm_gpu/codegen/training/forward/embedding_forward_split_kernel_template.cu
//
// KERNEL MAPPING:
//   {{ mdesc | capitalize }}EmbeddingNobagForwardUnweightedKernel
//     → {{ mdesc }}_embedding_nobag_codegen_forward_unweighted_kernel (CUDA)
//
// HOST FUNCTION MAPPING:
//   {{ mdesc }}_embedding_nobag_forward_unweighted_xpu
//     → {{ mdesc }}_embedding_nobag_codegen_forward_unweighted_cuda (CUDA)
//     CUDA File: fbgemm_gpu/_skbuild/linux-x86_64-3.10/cmake-build/gen_embedding_forward_{{ mdesc }}_unweighted_codegen_cuda.cu
//     CUDA Template: fbgemm_gpu/codegen/training/forward/embedding_forward_split_template.cu
//
// NOTE: The "codegen" string is removed from SYCL implementation names
//       for consistency with the refactored naming convention.
//
////////////////////////////////////////////////////////////////////////////////

/*
 * SYCL/XPU Implementation of {{ mdesc }} embedding forward kernel headers
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
#include "gen_embedding_forward_{{ mdesc }}_unweighted_nobag_kernel_small.h"

using Tensor = at::Tensor;

using at::native::RestrictPtrTraits;

namespace fbgemm_xpu {

    ////////////////////////////////////////////////////////////////////////////////
    // {{ mdesc | capitalize }}EmbeddingNobagForwardUnweightedKernel - Device Kernel
    ////////////////////////////////////////////////////////////////////////////////
    //
    // CUDA SOURCE MAPPING:
    //   CUDA Kernel: {{ mdesc }}_embedding_nobag_codegen_forward_unweighted_kernel
    //   CUDA File: fbgemm_gpu/_skbuild/linux-x86_64-3.10/cmake-build/gen_embedding_forward_{{ mdesc }}_unweighted_nobag_kernel.cu
    //   CUDA Template: fbgemm_gpu/codegen/training/forward/embedding_forward_split_kernel_template.cu
    //
    // DESCRIPTION:
    //   Main forward kernel for no-bag embeddings (sequence embeddings).
    {%- if not dense %}
    //   Supports cache-aware lookups with LXU cache for UVM-managed tables.
    {%- else %}
    //   Each thread retrieves one embedding vector and copies to output.
    {%- endif %}
    //
    ////////////////////////////////////////////////////////////////////////////////
    template <
    typename emb_t,
    typename cache_t,
    typename output_t,
    {%- if not dense %}
    bool use_lxu_cache,
    {%- endif %}
    typename index_t,
    size_t kThreadGroupSize>
    class {{ mdesc | capitalize }}EmbeddingNobagForwardUnweightedKernel {
        public:
            {{ mdesc | capitalize }}EmbeddingNobagForwardUnweightedKernel(
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
                const int32_t* lxu_cache_conflict_misses,
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
               lxu_cache_conflict_misses_(lxu_cache_conflict_misses),
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
            const int32_t* lxu_cache_conflict_misses_;
            {%- endif %}
            mutable at::PackedTensorAccessor64<output_t, 2, RestrictPtrTraits> output_;
    };


    ////////////////////////////////////////////////////////////////////////////////
    // {{ mdesc }}_embedding_nobag_forward_unweighted_xpu - Host Function
    ////////////////////////////////////////////////////////////////////////////////
    //
    // CUDA SOURCE MAPPING:
    //   CUDA Function: {{ mdesc }}_embedding_nobag_codegen_forward_unweighted_cuda
    //   CUDA File: fbgemm_gpu/_skbuild/linux-x86_64-3.10/cmake-build/gen_embedding_forward_{{ mdesc }}_unweighted_codegen_cuda.cu
    //   CUDA Template: fbgemm_gpu/codegen/training/forward/embedding_forward_split_template.cu
    //
    // DESCRIPTION:
    //   Host function for no-bag embedding forward pass.
    //   Retrieves embedding vectors without pooling (sequence embeddings).
    //   Dispatches to optimized kernel based on embedding dimension size.
    //
    ////////////////////////////////////////////////////////////////////////////////
    Tensor {{ mdesc }}_embedding_nobag_forward_unweighted_xpu(
        const Tensor& dev_weights,
        {%- if not dense %}
        const Tensor& uvm_weights,
        const Tensor& lxu_cache_weights,
        const Tensor& weights_placements,
        {%- endif %}
        const Tensor& weights_offsets,
        const c10::SymInt D_,
        const Tensor& indices,
        const Tensor& offsets,
        {%- if not dense %}
        const Tensor& lxu_cache_locations,
        const Tensor& uvm_cache_stats,
        {%- endif %}
        const int64_t output_dtype,
        const bool is_experimental
    );

    {%- if not dense %}
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
    {%- endif %}

} // namespace fbgemm_xpu
