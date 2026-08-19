/*
  * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
  * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
  * SPDX-License-Identifier: BSD-3-Clause
  */

#include "reorder_batched_ad.h"

#include <algorithm>

#include <ATen/xpu/XPUContext.h>
#include <c10/xpu/XPUFunctions.h>

namespace fbgemm_xpu {

namespace {

// Largest sub-group size supported by the current XPU device.
int64_t xpu_max_sub_group_size() {
    auto* dev_prop =
            at::xpu::getDeviceProperties(c10::xpu::current_device());
    const auto& sub_group_sizes = dev_prop->sub_group_sizes;
    TORCH_CHECK(
            !sub_group_sizes.empty(),
            "The device subgroup sizes is empty, please check the device status.");
    return *std::max_element(sub_group_sizes.begin(), sub_group_sizes.end());
}

// Largest work-group size supported by the current XPU device.
int64_t xpu_device_max_work_group_size() {
    auto* dev_prop =
            at::xpu::getDeviceProperties(c10::xpu::current_device());
    return dev_prop->max_work_group_size;
}

} // namespace

// ============================================================================
// SYCL Kernel Implementations
// ============================================================================

////////////////////////////////////////////////////////////////////////////////
// ReorderBatchedAdLengthsKernel - Device Kernel
////////////////////////////////////////////////////////////////////////////////
template <typename scalar_t>
void ReorderBatchedAdLengthsKernel<scalar_t>::operator()(
        const sycl::nd_item<2>& item) const {
    const int32_t B = batch_offsets_.size(0) - 1;

    const int32_t num_ads_in_batch = batch_offsets_[B];
    // warp-per-segment.
    const auto b_t =
            item.get_group(0) * item.get_local_range(1) + item.get_local_id(1);
    const int32_t b = b_t % B;
    const int32_t t = b_t / B;
    if (t >= T_) {
        return;
    }

    const int32_t num_ads_b = batch_offsets_[b + 1] - batch_offsets_[b];
    const int32_t input_segment_start =
            broadcast_lengths_ ? T_ * b + t : T_ * batch_offsets_[b] + t * num_ads_b;
    const int32_t output_segment_start = t * num_ads_in_batch + batch_offsets_[b];

    for (auto i = item.get_local_id(0); i < num_ads_b;
              i += item.get_local_range(0)) {
        reordered_cat_ad_lengths_[output_segment_start + i] = broadcast_lengths_
                ? cat_ad_lengths_[input_segment_start]
                : cat_ad_lengths_[input_segment_start + i];
    }
}

////////////////////////////////////////////////////////////////////////////////
// NarrowBroadcastIndicesKernel - Device Kernel (B=1 optimization)
////////////////////////////////////////////////////////////////////////////////
template <typename scalar_t, typename index_t>
void NarrowBroadcastIndicesKernel<scalar_t, index_t>::operator()(
        const sycl::nd_item<1>& item) const {
    const auto lane_id = item.get_local_id(0) % sub_group_size_;
    const auto warp_id =
            (item.get_group(0) * item.get_local_range(0) + item.get_local_id(0)) /
            sub_group_size_;
    const auto table_idx = warp_id / num_ads_in_batch_;
    const auto ads_idx = warp_id % num_ads_in_batch_;
    const auto start_offset = cat_ad_offsets_[table_idx];
    const auto end_offset = cat_ad_offsets_[table_idx + 1];
    const auto num_ads = end_offset - start_offset;
    if (warp_id < reordered_cat_ad_batches_) {
        for (auto i = lane_id; i < num_ads; i += sub_group_size_) {
            reordered_cat_ad_indices_
                    [start_offset * num_ads_in_batch_ + ads_idx * num_ads + i] =
                            cat_ad_indices_[start_offset + i];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// NarrowBatchedBroadcastIndicesKernel - Device Kernel (B>1 optimization)
////////////////////////////////////////////////////////////////////////////////
template <typename scalar_t, typename index_t>
void NarrowBatchedBroadcastIndicesKernel<scalar_t, index_t>::operator()(
        const sycl::nd_item<1>& item) const {
    const auto B = batch_offsets_.size(0) - 1;
    const auto num_ads_in_batch = static_cast<uint32_t>(batch_offsets_[B]);
    // calculate table_id and batch_id for this warp
    const auto warp_id =
            (item.get_group(0) * item.get_local_range(0) + item.get_local_id(0)) /
            static_cast<uint32_t>(sub_group_size_);
    const auto table_id = warp_id / num_ads_in_batch;
    const auto warp_id_in_table = warp_id % num_ads_in_batch;
    // warps in a table equally splited for each B
    const auto num_warp_in_batch = num_ads_in_batch / B;
    const auto batch_id = warp_id_in_table / num_warp_in_batch;
    if (table_id >= T_ || batch_id >= B) {
        return;
    }

    // all table_id and batch_id for this warp is the same
    const auto num_ads_b = batch_offsets_[batch_id + 1] - batch_offsets_[batch_id];
    const auto output_segment_offset_start =
            table_id * num_ads_in_batch + batch_offsets_[batch_id];
    const auto output_segment_start =
            reordered_cat_ad_offsets_[output_segment_offset_start];
    const auto input_segment_offset_start = T_ * batch_id + table_id;
    const auto input_segment_offset_end = input_segment_offset_start + 1;
    const auto input_segment_start = cat_ad_offsets_[input_segment_offset_start];
    const auto input_segment_end = cat_ad_offsets_[input_segment_offset_end];
    const auto num_elements = input_segment_end - input_segment_start;

    const auto warp_id_in_batch = warp_id_in_table % num_warp_in_batch;
    const auto lane_id_in_warp = item.get_local_id(0) % sub_group_size_;
    for (auto i = warp_id_in_batch; i < num_ads_b; i += num_warp_in_batch) {
        for (auto j = lane_id_in_warp; j < num_elements; j += sub_group_size_) {
            reordered_cat_ad_indices_[output_segment_start + i * num_elements + j] =
                    cat_ad_indices_[input_segment_start + j];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// ReorderBatchedAdIndicesKernel - Device Kernel (General case)
////////////////////////////////////////////////////////////////////////////////
template <typename scalar_t, typename index_t>
void ReorderBatchedAdIndicesKernel<scalar_t, index_t>::operator()(
        const sycl::nd_item<2>& item) const {
    const int32_t B = batch_offsets_.size(0) - 1;
    const int32_t num_ads_in_batch = batch_offsets_[B];
    // warp-per-segment.
    const auto b_t =
            item.get_group(0) * item.get_local_range(1) + item.get_local_id(1);
    const int32_t b = b_t % B;
    const int32_t t = b_t / B;
    if (t >= T_) {
        return;
    }

    const auto num_ads_b = batch_offsets_[b + 1] - batch_offsets_[b];
    const auto output_segment_offset_start =
            t * num_ads_in_batch + batch_offsets_[b];
    const auto output_segment_start =
            reordered_cat_ad_offsets_[output_segment_offset_start];
    const int32_t input_segment_offset_start =
            broadcast_indices_ ? T_ * b + t : T_ * batch_offsets_[b] + t * num_ads_b;
    const int32_t input_segment_offset_end = broadcast_indices_
            ? input_segment_offset_start + 1
            : input_segment_offset_start + num_ads_b;
    const auto input_segment_start = cat_ad_offsets_[input_segment_offset_start];
    const auto input_segment_end = cat_ad_offsets_[input_segment_offset_end];
    const auto num_elements = input_segment_end - input_segment_start;

    if (broadcast_indices_) {
        for (auto i = item.get_local_id(0); i < num_ads_b * num_elements;
                  i += item.get_local_range(0)) {
            reordered_cat_ad_indices_[output_segment_start + i] =
                    cat_ad_indices_[input_segment_start + i % num_elements];
        }
    } else {
        // Idea: we want to copy the entire segment of size sum_a(length_{b, t, a})
        // from starting point (given by cat_ad_offsets[b, t])
        // to end point (given by reordered_cat_ad_indices[t][b])
        for (auto i = item.get_local_id(0);
                  i < input_segment_end - input_segment_start;
                  i += item.get_local_range(0)) {
            reordered_cat_ad_indices_[output_segment_start + i] =
                    cat_ad_indices_[input_segment_start + i];
        }
    }
}

// ============================================================================
// Host Kernel Dispatcher Functions
// ============================================================================

void reorder_batched_ad_lengths_kernel_xpu(
        const at::Tensor& cat_ad_lengths,
        const at::Tensor& batch_offsets,
        at::Tensor& reordered_cat_ad_lengths,
        const int32_t T,
        const bool broadcast_lengths,
        const int32_t grid_size) {
    sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();
    FBGEMM_DISPATCH_ALL_TYPES(
            cat_ad_lengths.scalar_type(),
            "reorder_batched_ad_lengths_kernel_xpu",
            [&] {
                queue.submit([&](sycl::handler& cgh) {
                    cgh.parallel_for<ReorderBatchedAdLengthsKernel<scalar_t>>(
                            sycl::nd_range<2>(
                                    sycl::range<2>(32 * grid_size, 32),
                                    sycl::range<2>(32, 32)),
                            ReorderBatchedAdLengthsKernel<scalar_t>(
                                    cat_ad_lengths.packed_accessor32<
                                            scalar_t,
                                            1,
                                            RestrictPtrTraits>(),
                                    batch_offsets.packed_accessor32<
                                            int32_t,
                                            1,
                                            RestrictPtrTraits>(),
                                    reordered_cat_ad_lengths.packed_accessor32<
                                            scalar_t,
                                            1,
                                            RestrictPtrTraits>(),
                                    T,
                                    broadcast_lengths));
                });
            });
}

void reorder_batched_ad_indices_kernel_xpu(
        const at::Tensor& cat_ad_offsets,
        const at::Tensor& cat_ad_indices,
        const at::Tensor& reordered_cat_ad_offsets,
        const at::Tensor& batch_offsets,
        at::Tensor& reordered_cat_ad_indices,
        const int64_t num_ads_in_batch,
        const int64_t B,
        const int64_t T,
        const bool broadcast_indices) {
    sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();
    const int sub_group_size = xpu_max_sub_group_size();
    if (broadcast_indices && T <= 320 && B < 64) {
        TORCH_CHECK(num_ads_in_batch * T == reordered_cat_ad_offsets.numel() - 1);
        if (B == 1) {
            // for B = 1 broadcast case
            constexpr auto kNumWarps = 16;
            const int work_group_size = kNumWarps * sub_group_size;
            const int global_dim =
                    xpu_calc_xblock_count(
                            reordered_cat_ad_offsets.numel() - 1, kNumWarps) *
                    work_group_size;
            FBGEMM_DISPATCH_ALL_TYPES(
                    cat_ad_indices.scalar_type(),
                    "narrow_broadcast_indices_kernel_1",
                    [&] {
                        AT_DISPATCH_INDEX_TYPES(
                                cat_ad_offsets.scalar_type(),
                                "narrow_broadcast_indices_kernel_2",
                                [&] {
                                    queue.submit([&](sycl::handler& cgh) {
                                        cgh.parallel_for<NarrowBroadcastIndicesKernel<
                                                scalar_t,
                                                index_t>>(
                                                sycl::nd_range<1>(
                                                        sycl::range<1>(global_dim),
                                                        sycl::range<1>(work_group_size)),
                                                NarrowBroadcastIndicesKernel<
                                                        scalar_t,
                                                        index_t>(
                                                        cat_ad_offsets.packed_accessor32<
                                                                index_t,
                                                                1,
                                                                RestrictPtrTraits>(),
                                                        cat_ad_indices.packed_accessor32<
                                                                scalar_t,
                                                                1,
                                                                RestrictPtrTraits>(),
                                                        reordered_cat_ad_indices
                                                                .packed_accessor32<
                                                                        scalar_t,
                                                                        1,
                                                                        RestrictPtrTraits>(),
                                                        num_ads_in_batch,
                                                        reordered_cat_ad_offsets.numel() - 1,
                                                        sub_group_size));
                                    });
                                });
                    });
            return;
        } else {
            // for B > 1 and B < 64 broadcast case
            constexpr auto kNumWarps = 16;
            const int work_group_size = kNumWarps * sub_group_size;
            const int global_dim =
                    xpu_calc_xblock_count(T * num_ads_in_batch, kNumWarps) *
                    work_group_size;
            FBGEMM_DISPATCH_ALL_TYPES(
                    cat_ad_indices.scalar_type(),
                    "narrow_batched_broadcast_indices_kernel_1",
                    [&] {
                        AT_DISPATCH_INDEX_TYPES(
                                cat_ad_offsets.scalar_type(),
                                "narrow_batched_broadcast_indices_kernel_2",
                                [&] {
                                    queue.submit([&](sycl::handler& cgh) {
                                        cgh.parallel_for<
                                                NarrowBatchedBroadcastIndicesKernel<
                                                        scalar_t,
                                                        index_t>>(
                                                sycl::nd_range<1>(
                                                        sycl::range<1>(global_dim),
                                                        sycl::range<1>(work_group_size)),
                                                NarrowBatchedBroadcastIndicesKernel<
                                                        scalar_t,
                                                        index_t>(
                                                        cat_ad_offsets.packed_accessor32<
                                                                index_t,
                                                                1,
                                                                RestrictPtrTraits>(),
                                                        cat_ad_indices.packed_accessor32<
                                                                scalar_t,
                                                                1,
                                                                RestrictPtrTraits>(),
                                                        reordered_cat_ad_offsets
                                                                .packed_accessor32<
                                                                        index_t,
                                                                        1,
                                                                        RestrictPtrTraits>(),
                                                        reordered_cat_ad_indices
                                                                .packed_accessor32<
                                                                        scalar_t,
                                                                        1,
                                                                        RestrictPtrTraits>(),
                                                        batch_offsets.packed_accessor32<
                                                                int32_t,
                                                                1,
                                                                RestrictPtrTraits>(),
                                                        T,
                                                        sub_group_size));
                                    });
                                });
                    });
            return;
        }
    }
    FBGEMM_DISPATCH_ALL_TYPES(
            cat_ad_indices.scalar_type(),
            "reorder_batched_ad_indices_kernel_xpu_1",
            [&] {
                AT_DISPATCH_INDEX_TYPES(
                        cat_ad_offsets.scalar_type(),
                        "reorder_batched_ad_indices_kernel_xpu_2",
                        [&] {
                            constexpr auto kNumWarps = 32;
                            const int max_work_group_size =
                                    xpu_device_max_work_group_size();
                            auto max_warp_size = max_work_group_size / kNumWarps;
                            const int global_dim_y =
                                    max_warp_size < sub_group_size ? max_warp_size : sub_group_size;
                            const int global_dim_x =
                                    xpu_calc_xblock_count(B * T, kNumWarps) * kNumWarps;
                            queue.submit([&](sycl::handler& cgh) {
                                cgh.parallel_for<ReorderBatchedAdIndicesKernel<
                                        scalar_t,
                                        index_t>>(
                                        sycl::nd_range<2>(
                                                sycl::range<2>(global_dim_x, global_dim_y),
                                                sycl::range<2>(kNumWarps, global_dim_y)),
                                        ReorderBatchedAdIndicesKernel<scalar_t, index_t>(
                                                cat_ad_offsets.packed_accessor32<
                                                        index_t,
                                                        1,
                                                        RestrictPtrTraits>(),
                                                cat_ad_indices.packed_accessor32<
                                                        scalar_t,
                                                        1,
                                                        RestrictPtrTraits>(),
                                                reordered_cat_ad_offsets.packed_accessor32<
                                                        index_t,
                                                        1,
                                                        RestrictPtrTraits>(),
                                                reordered_cat_ad_indices.packed_accessor32<
                                                        scalar_t,
                                                        1,
                                                        RestrictPtrTraits>(),
                                                batch_offsets.packed_accessor32<
                                                        int32_t,
                                                        1,
                                                        RestrictPtrTraits>(),
                                                T,
                                                broadcast_indices));
                            });
                        });
            });
}

} // namespace fbgemm_xpu
