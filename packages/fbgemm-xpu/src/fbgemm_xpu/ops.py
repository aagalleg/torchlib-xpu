# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Python wrapper functions for all custom operators under the fbgemm namespace
# This module provides user-friendly interfaces to the C++ operators

from typing import List, Optional, Tuple

import torch
from torch import Tensor

__all__ = [
    "dense_embedding_codegen_lookup_function",
    "split_embedding_codegen_lookup_rowwise_adagrad_function_pt2",
    "invert_permute",
    "permute_1D_sparse_data",
    "asynchronous_complete_cumsum",
    "permute_2D_sparse_data",
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


def invert_permute(permute: Tensor) -> Tensor:
    """Computes the inverse of a permutation tensor."""
    return torch.ops.fbgemm.invert_permute.default(permute)


def permute_1D_sparse_data(
    permute: Tensor,
    lengths: Tensor,
    indices: Tensor,
    weights: Optional[Tensor] = None,
    permuted_lengths_sum: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Permutes sparse data in jagged/1D format according to permutation indices."""
    return torch.ops.fbgemm.permute_1D_sparse_data.default(
        permute, lengths, indices, weights, permuted_lengths_sum
    )


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
    vbe_output_size: int = -1,
    vbe_output: Optional[Tensor] = None
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
        vbe_output,
    )


def asynchronous_complete_cumsum(t_in: Tensor) -> Tensor:
    """Computes complete cumulative sum: output[0] = 0, output[i] = sum(t_in[0:i])."""
    return torch.ops.fbgemm.asynchronous_complete_cumsum.default(t_in)


def permute_2D_sparse_data(
    permute: Tensor,
    lengths: Tensor,
    indices: Tensor,
    weights: Optional[Tensor] = None,
    permuted_lengths_sum: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Permutes 2D sparse data (lengths: [T, B]) according to permutation indices."""
    return torch.ops.fbgemm.permute_2D_sparse_data.default(
        permute, lengths, indices, weights, permuted_lengths_sum
    )
