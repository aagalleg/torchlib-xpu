# Copyright 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Portions of this file are derived from FBGEMM
# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause

# Python wrapper functions for all custom operators under the fbgemm namespace
# This module provides user-friendly interfaces to the C++ operators

from typing import List, Optional, Tuple

import torch
from torch import Tensor

__all__ = [
    "dense_embedding_codegen_lookup_function",
    "split_embedding_codegen_lookup_rowwise_adagrad_function_pt2",
    # "invert_permute",
    # "permute_1D_sparse_data",
    # "jagged_index_select_2d_forward",
    # "asynchronous_complete_cumsum",
    # "dense_to_jagged",
    # "dense_to_jagged_forward",
    # "jagged_to_padded_dense",
    # "jagged_to_padded_dense_forward",
    # "jagged_dense_elementwise_add_jagged_output",
    # "reorder_batched_ad_lengths",
    # "reorder_batched_ad_indices",
    # "permute_2D_sparse_data",
    # "jagged_2d_to_dense",
]


# =============================================================================
# Dense Embedding Operators
# =============================================================================

def dense_embedding_codegen_lookup_function(
    dev_weights: Tensor,
    weights_offsets: Tensor,
    D_offsets: Tensor,
    total_D: int,
    max_D: int,
    hash_size_cumsum: Tensor,
    total_hash_size_bits: int,
    indices: Tensor,
    offsets: Tensor,
    pooling_mode: int,
    indice_weights: Optional[Tensor] = None,
    feature_requires_grad: Optional[Tensor] = None,
    output_dtype: int = 0,
    B_offsets: Optional[Tensor] = None,
    vbe_output_offsets_feature_rank: Optional[Tensor] = None,
    vbe_B_offsets_rank_per_feature: Optional[Tensor] = None,
    max_B: int = -1,
    max_B_feature_rank: int = -1,
    vbe_output_size: int = -1,
    mixed_D: bool = True,
) -> Tensor:

    return torch.ops.fbgemm.dense_embedding_codegen_lookup_function(
        dev_weights,
        weights_offsets,
        D_offsets,
        total_D,
        max_D,
        hash_size_cumsum,
        total_hash_size_bits,
        indices,
        offsets,
        pooling_mode,
        indice_weights,
        feature_requires_grad,
        output_dtype,
        B_offsets,
        vbe_output_offsets_feature_rank,
        vbe_B_offsets_rank_per_feature,
        max_B,
        max_B_feature_rank,
        vbe_output_size,
        mixed_D,
    )


# # =============================================================================
# # Jagged Index Select
# # =============================================================================

# def jagged_index_select_2d_forward(
#     values: Tensor,
#     indices: Tensor,
#     input_offsets: Tensor,
#     output_offsets: Tensor,
#     num_dense_output_rows: int,
# ) -> Tensor:
#     """Selects rows from a jagged tensor based on indices."""
#     return torch.ops.fbgemm.jagged_index_select_2d_forward.default(
#         values, indices, input_offsets, output_offsets, num_dense_output_rows
#     )


# @torch.library.register_fake("fbgemm::jagged_index_select_2d_forward")
# def _(values, indices, input_offsets, output_offsets, num_dense_output_rows):
#     torch._check(values.dim() == 2)
#     torch._check(indices.dim() == 1)
#     torch._check(input_offsets.dim() == 1)
#     torch._check(output_offsets.dim() == 1)
#     num_cols = values.size(1)
#     return torch.empty(
#         (num_dense_output_rows, num_cols), dtype=values.dtype, device=values.device
#     )


# # =============================================================================
# # Invert Permutation
# # =============================================================================

# def invert_permute(permute: Tensor) -> Tensor:
#     """Computes the inverse of a permutation tensor."""
#     return torch.ops.fbgemm.invert_permute.default(permute)


# @torch.library.register_fake("fbgemm::invert_permute")
# def _(permute):
#     torch._check(
#         permute.dim() == 1,
#         lambda: f"invert_permute expects 1D tensor, got {permute.dim()}D",
#     )
#     torch._check(
#         permute.dtype in (torch.int32, torch.int64),
#         lambda: f"invert_permute expects int32 or int64, got {permute.dtype}",
#     )
#     return torch.empty_like(permute)


# # =============================================================================
# # Permute 1D Sparse Data
# # =============================================================================

# def permute_1D_sparse_data(
#     permute: Tensor,
#     lengths: Tensor,
#     indices: Tensor,
#     weights: Optional[Tensor] = None,
#     permuted_lengths_sum: Optional[int] = None,
# ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
#     """Permutes sparse data in jagged/1D format according to permutation indices."""
#     return torch.ops.fbgemm.permute_1D_sparse_data.default(
#         permute, lengths, indices, weights, permuted_lengths_sum
#     )


# @torch.library.register_fake("fbgemm::permute_1D_sparse_data")
# def _(permute, lengths, indices, weights=None, permuted_lengths_sum=None):
#     torch._check(permute.dim() == 1, lambda: f"permute must be 1D, got {permute.dim()}D")
#     torch._check(lengths.dim() == 1, lambda: f"lengths must be 1D, got {lengths.dim()}D")
#     torch._check(indices.dim() == 1, lambda: f"indices must be 1D, got {indices.dim()}D")
#     torch._check(permute.dtype == torch.int32, lambda: f"permute must be int32, got {permute.dtype}")

#     permuted_lengths_size = permute.size(0)
#     permuted_lengths = torch.empty(
#         permuted_lengths_size, dtype=lengths.dtype, device=permute.device
#     )

#     if permuted_lengths_sum is not None:
#         output_size = permuted_lengths_sum
#     else:
#         output_size = indices.size(0)

#     permuted_indices = torch.empty(output_size, dtype=indices.dtype, device=permute.device)

#     if weights is not None:
#         torch._check(weights.dim() == 1, lambda: f"weights must be 1D, got {weights.dim()}D")
#         permuted_weights = torch.empty(output_size, dtype=weights.dtype, device=permute.device)
#     else:
#         permuted_weights = None

#     return permuted_lengths, permuted_indices, permuted_weights

def split_embedding_codegen_lookup_rowwise_adagrad_function_pt2(
    placeholder_autograd_tensor: Tensor,
    weights: Tensor,
    D_offsets: Tensor,
    total_D: int,
    max_D: int,
    hash_size_cumsum: Tensor,
    total_hash_size_bits: int,
    indices: Tensor,
    offsets: Tensor,
    pooling_mode: int,
    aux_int: List[int],
    aux_float: List[float],
    aux_bool: List[bool],
    momentum1: List[Tensor],
    learning_rate_tensor: Tensor,
    output_dtype: int,
    optim_int: List[int],
    optim_float: List[float],
    indice_weights: Optional[Tensor] = None,
    feature_requires_grad: Optional[Tensor] = None,
    aux_tensor: Optional[Tensor] = None,
    max_B: int = -1,
    max_B_feature_rank: int = -1,
    vbe_output_size: int = -1
) -> Tensor:
    return torch.ops.fbgemm.split_embedding_codegen_lookup_rowwise_adagrad_function_pt2(
        placeholder_autograd_tensor,
        weights,
        D_offsets,
        total_D,
        max_D,
        hash_size_cumsum,
        total_hash_size_bits,
        indices,
        offsets,
        pooling_mode,
        indice_weights,
        feature_requires_grad,
        output_dtype,
        aux_tensor,
        aux_int,
        aux_float,
        aux_bool,
        momentum1,
        learning_rate_tensor,
        optim_int,
        optim_float,
        max_B,
        max_B_feature_rank,
        vbe_output_size,
    )
    
# # =============================================================================
# # FBGEMM XPU Operators (temporal)
# # =============================================================================

# def asynchronous_complete_cumsum(t_in: Tensor) -> Tensor:
#     """Computes complete cumulative sum: output[0] = 0, output[i] = sum(t_in[0:i])."""
#     return torch.ops.fbgemm.asynchronous_complete_cumsum.default(t_in)


# @torch.library.register_fake("fbgemm::asynchronous_complete_cumsum")
# def _(t_in):
#     torch._check(t_in.dim() == 1 or t_in.dim() == 2)
#     if t_in.dim() == 1:
#         return torch.empty(t_in.numel() + 1, dtype=t_in.dtype, device=t_in.device)
#     return torch.empty(
#         (t_in.size(0), t_in.size(1) + 1), dtype=t_in.dtype, device=t_in.device
#     )


# @torch.library.impl("fbgemm::asynchronous_complete_cumsum", "CPU")
# def _cumsum_cpu(t_in: Tensor) -> Tensor:
#     if t_in.dim() == 1:
#         return torch.cat([t_in.new_zeros(1), t_in.cumsum(0)])
#     return torch.cat([t_in.new_zeros(t_in.size(0), 1), t_in.cumsum(1)], dim=1)


# def dense_to_jagged(
#     dense: Tensor,
#     x_offsets: List[Tensor],
#     total_L: Optional[int] = None,
# ) -> Tuple[Tensor, List[Tensor]]:
#     """Converts a dense tensor to jagged format using offsets."""
#     return torch.ops.fbgemm.dense_to_jagged.default(dense, x_offsets, total_L)


# @torch.library.register_fake("fbgemm::dense_to_jagged")
# def _(dense, x_offsets, total_L=None):
#     D = dense.size(-1)
#     if total_L is not None:
#         L = total_L
#     else:
#         L = x_offsets[-1].size(0) - 1  # conservative estimate
#     values = torch.empty(L, D, dtype=dense.dtype, device=dense.device)
#     return values, x_offsets


# def dense_to_jagged_forward(
#     dense: Tensor,
#     x_offsets: List[Tensor],
#     total_L: Optional[int] = None,
# ) -> Tensor:
#     """Forward pass of dense-to-jagged conversion."""
#     return torch.ops.fbgemm.dense_to_jagged_forward.default(dense, x_offsets, total_L)


# @torch.library.register_fake("fbgemm::dense_to_jagged_forward")
# def _(dense, x_offsets, total_L=None):
#     D = dense.size(-1)
#     if total_L is not None:
#         L = total_L
#     else:
#         L = x_offsets[-1].size(0) - 1
#     return torch.empty(L, D, dtype=dense.dtype, device=dense.device)


# def jagged_to_padded_dense(
#     values: Tensor,
#     offsets: List[Tensor],
#     max_lengths: List[int],
#     padding_value: float = 0.0,
# ) -> Tensor:
#     """Converts jagged tensor to padded dense tensor."""
#     return torch.ops.fbgemm.jagged_to_padded_dense.default(
#         values, offsets, max_lengths, padding_value
#     )


# @torch.library.register_fake("fbgemm::jagged_to_padded_dense")
# def _(values, offsets, max_lengths, padding_value=0.0):
#     B = offsets[0].size(0) - 1
#     D_folded = values.dim() == 1
#     inner_dense_size = 1 if D_folded else values.size(-1)
#     shape = [B] + list(max_lengths)
#     if not D_folded:
#         shape.append(inner_dense_size)
#     return torch.empty(shape, dtype=values.dtype, device=values.device)


# def jagged_to_padded_dense_forward(
#     values: Tensor,
#     offsets: List[Tensor],
#     max_lengths: List[int],
#     padding_value: float = 0.0,
# ) -> Tensor:
#     """Forward pass of jagged-to-padded-dense conversion."""
#     return torch.ops.fbgemm.jagged_to_padded_dense_forward.default(
#         values, offsets, max_lengths, padding_value
#     )


# @torch.library.register_fake("fbgemm::jagged_to_padded_dense_forward")
# def _(values, offsets, max_lengths, padding_value=0.0):
#     B = offsets[0].size(0) - 1
#     D_folded = values.dim() == 1
#     inner_dense_size = 1 if D_folded else values.size(-1)
#     shape = [B] + list(max_lengths)
#     if not D_folded:
#         shape.append(inner_dense_size)
#     return torch.empty(shape, dtype=values.dtype, device=values.device)


# def jagged_dense_elementwise_add_jagged_output(
#     x_values: Tensor,
#     x_offsets: List[Tensor],
#     y: Tensor,
# ) -> Tuple[Tensor, List[Tensor]]:
#     """Element-wise add of jagged tensor x and dense tensor y, output is jagged."""
#     return torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output.default(
#         x_values, x_offsets, y
#     )


# @torch.library.register_fake("fbgemm::jagged_dense_elementwise_add_jagged_output")
# def _(x_values, x_offsets, y):
#     return torch.empty_like(x_values), x_offsets


# def reorder_batched_ad_lengths(
#     cat_ad_lengths: Tensor,
#     batch_offsets: Tensor,
#     num_ads_in_batch: int,
#     broadcast_lengths: bool,
#     max_batch_size: int = 0,
# ) -> Tensor:
#     """Reorders batched ad lengths according to batch offsets."""
#     return torch.ops.fbgemm.reorder_batched_ad_lengths.default(
#         cat_ad_lengths, batch_offsets, num_ads_in_batch, broadcast_lengths, max_batch_size
#     )


# @torch.library.register_fake("fbgemm::reorder_batched_ad_lengths")
# def _(cat_ad_lengths, batch_offsets, num_ads_in_batch, broadcast_lengths, max_batch_size=0):
#     B = batch_offsets.numel() - 1
#     T = cat_ad_lengths.numel() // (B if broadcast_lengths else num_ads_in_batch)
#     if broadcast_lengths:
#         return torch.empty(T * num_ads_in_batch, dtype=cat_ad_lengths.dtype, device=cat_ad_lengths.device)
#     return torch.empty_like(cat_ad_lengths)


# @torch.library.impl("fbgemm::reorder_batched_ad_lengths", "CPU")
# def _reorder_lengths_cpu(
#     cat_ad_lengths: Tensor,
#     batch_offsets: Tensor,
#     num_ads_in_batch: int,
#     broadcast_lengths: bool = False,
#     max_batch_size: int = 0,
# ) -> Tensor:
#     nB = batch_offsets.numel() - 1
#     if broadcast_lengths:
#         nT = cat_ad_lengths.numel() // nB
#     else:
#         nT = cat_ad_lengths.numel() // num_ads_in_batch
#     output_batch_size = max_batch_size if max_batch_size > 0 else num_ads_in_batch
#     output = torch.zeros(nT * output_batch_size, dtype=cat_ad_lengths.dtype)
#     for b in range(nB):
#         num_ads_b = (batch_offsets[b + 1] - batch_offsets[b]).item()
#         for t in range(nT):
#             out_start = t * output_batch_size + batch_offsets[b].item()
#             if broadcast_lengths:
#                 in_idx = nT * b + t
#                 output[out_start:out_start + num_ads_b] = cat_ad_lengths[in_idx]
#             else:
#                 in_start = nT * batch_offsets[b].item() + t * num_ads_b
#                 output[out_start:out_start + num_ads_b] = cat_ad_lengths[in_start:in_start + num_ads_b]
#     return output


# def reorder_batched_ad_indices(
#     cat_ad_offsets: Tensor,
#     cat_ad_indices: Tensor,
#     reordered_cat_ad_offsets: Tensor,
#     batch_offsets: Tensor,
#     num_ads_in_batch: int,
#     broadcast_indices: bool,
#     num_indices_after_broadcast: int,
# ) -> Tensor:
#     """Reorders batched ad indices according to batch offsets."""
#     return torch.ops.fbgemm.reorder_batched_ad_indices.default(
#         cat_ad_offsets, cat_ad_indices, reordered_cat_ad_offsets,
#         batch_offsets, num_ads_in_batch, broadcast_indices, num_indices_after_broadcast
#     )


# @torch.library.register_fake("fbgemm::reorder_batched_ad_indices")
# def _(cat_ad_offsets, cat_ad_indices, reordered_cat_ad_offsets, batch_offsets,
#       num_ads_in_batch, broadcast_indices, num_indices_after_broadcast):
#     if broadcast_indices:
#         return torch.empty(num_indices_after_broadcast, dtype=cat_ad_indices.dtype, device=cat_ad_indices.device)
#     return torch.empty_like(cat_ad_indices)


# @torch.library.impl("fbgemm::reorder_batched_ad_indices", "CPU")
# def _reorder_indices_cpu(
#     cat_ad_offsets: Tensor,
#     cat_ad_indices: Tensor,
#     reordered_cat_ad_offsets: Tensor,
#     batch_offsets: Tensor,
#     num_ads_in_batch: int,
#     broadcast_indices: bool,
#     num_indices_after_broadcast: int,
# ) -> Tensor:
#     nB = batch_offsets.numel() - 1
#     nT = (reordered_cat_ad_offsets.numel() - 1) // num_ads_in_batch
#     if broadcast_indices:
#         output = torch.empty(num_indices_after_broadcast, dtype=cat_ad_indices.dtype)
#     else:
#         output = torch.empty_like(cat_ad_indices)
#     for b in range(nB):
#         num_ads_b = (batch_offsets[b + 1] - batch_offsets[b]).item()
#         for t in range(nT):
#             out_offset_idx = t * num_ads_in_batch + batch_offsets[b].item()
#             out_start = reordered_cat_ad_offsets[out_offset_idx].item()
#             if broadcast_indices:
#                 in_offset_idx = nT * b + t
#                 in_start = cat_ad_offsets[in_offset_idx].item()
#                 in_end = cat_ad_offsets[in_offset_idx + 1].item()
#                 n = in_end - in_start
#                 for j in range(num_ads_b):
#                     output[out_start + j * n:out_start + (j + 1) * n] = cat_ad_indices[in_start:in_start + n]
#             else:
#                 in_offset_idx = nT * batch_offsets[b].item() + t * num_ads_b
#                 in_start = cat_ad_offsets[in_offset_idx].item()
#                 in_end = cat_ad_offsets[in_offset_idx + num_ads_b].item()
#                 n = in_end - in_start
#                 output[out_start:out_start + n] = cat_ad_indices[in_start:in_start + n]
#     return output


# def permute_2D_sparse_data(
#     permute: Tensor,
#     lengths: Tensor,
#     indices: Tensor,
#     weights: Optional[Tensor] = None,
#     permuted_lengths_sum: Optional[int] = None,
# ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
#     """Permutes 2D sparse data (lengths: [T, B]) according to permutation indices."""
#     return torch.ops.fbgemm.permute_2D_sparse_data.default(
#         permute, lengths, indices, weights, permuted_lengths_sum
#     )


# @torch.library.register_fake("fbgemm::permute_2D_sparse_data")
# def _(permute, lengths, indices, weights=None, permuted_lengths_sum=None):
#     torch._check(lengths.dim() == 2)
#     T = permute.numel()
#     B = lengths.size(1)
#     permuted_lengths = torch.empty((T, B), dtype=lengths.dtype, device=lengths.device)

#     if permuted_lengths_sum is not None:
#         output_size = permuted_lengths_sum
#     else:
#         output_size = indices.size(0)

#     permuted_indices = torch.empty(output_size, dtype=indices.dtype, device=indices.device)
#     permuted_weights = (
#         torch.empty(output_size, dtype=weights.dtype, device=weights.device)
#         if weights is not None else None
#     )
#     return permuted_lengths, permuted_indices, permuted_weights


# # =============================================================================
# # Jagged 2D to Dense
# # =============================================================================

# def jagged_2d_to_dense(
#     values: Tensor,
#     offsets: Tensor,
#     max_sequence_length: int,
# ) -> Tensor:
#     """Converts a jagged 2D tensor to a dense padded tensor."""
#     return torch.ops.fbgemm.jagged_2d_to_dense.default(
#         values, offsets, max_sequence_length
#     )


# @torch.library.register_fake("fbgemm::jagged_2d_to_dense")
# def _(values, offsets, max_sequence_length):
#     B = offsets.numel() - 1
#     if values.dim() == 1:
#         return values.new_empty(B, max_sequence_length)
#     D = values.size(1)
#     return values.new_empty(B, max_sequence_length, D)


# @torch.library.impl("fbgemm::jagged_2d_to_dense", "CPU")
# def _jagged_2d_to_dense_cpu(
#     values: Tensor,
#     offsets: Tensor,
#     max_sequence_length: int,
# ) -> Tensor:
#     B = offsets.numel() - 1
#     if values.dim() == 1:
#         output = values.new_zeros(B, max_sequence_length)
#     else:
#         D = values.size(1)
#         output = values.new_zeros(B, max_sequence_length, D)
#     for i in range(B):
#         start = offsets[i].item()
#         end = offsets[i + 1].item()
#         length = min(end - start, max_sequence_length)
#         if length > 0:
#             output[i, :length] = values[start:start + length]
#     return output
