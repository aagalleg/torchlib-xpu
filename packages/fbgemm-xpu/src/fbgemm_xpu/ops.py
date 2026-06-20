# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Python wrapper functions for all custom operators under the fbgemm namespace
# This module provides user-friendly interfaces to the C++ operators

from typing import Optional, Tuple

import torch
from torch import Tensor

__all__ = [
    "invert_permute",
	"permute_1D_sparse_data",
    "asynchronous_complete_cumsum",
    "permute_2D_sparse_data",
]

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
