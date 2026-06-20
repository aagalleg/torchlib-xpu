"""
Test suite for permute_2D_sparse_data operator

Ported from test_fbgemm_ops.py
"""

import random
from itertools import accumulate
from typing import Optional, Tuple

import torch
import unittest
import fbgemm_gpu
import fbgemm_xpu
from torch.testing._internal.common_utils import TestCase, run_tests

DEVICE = "xpu"


# =============================================================================
# Reference / helper functions
# =============================================================================

def permute_indices_ref_(
    lengths: torch.Tensor,
    indices: torch.Tensor,
    weights: Optional[torch.Tensor],
    permute: torch.LongTensor,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    T = lengths.size(0)
    B = lengths.size(1)
    if T == 0 or B == 0:
        return lengths, indices, weights

    permuted_lengths = torch.index_select(lengths.view(T, -1), 0, permute)
    original_segment_lengths = lengths.view(T, -1).sum(dim=1, dtype=torch.int32)
    original_segment_start = [0] + list(accumulate(original_segment_lengths.view(-1)))

    permuted_indices = []
    permuted_weights = []
    for i in range(permute.size(0)):
        start = original_segment_start[permute[i]]
        end = start + original_segment_lengths[permute[i]]
        permuted_indices.append(indices[start:end])
        if weights is not None:
            permuted_weights.append(weights[start:end])

    permuted_indices = torch.cat(permuted_indices, dim=0).flatten()

    if weights is None:
        permuted_weights = None
    else:
        permuted_weights = torch.cat(permuted_weights, dim=0).flatten()

    return permuted_lengths, permuted_indices, permuted_weights


# =============================================================================
# Tests
# =============================================================================

class TestPermute2DSparseData(TestCase):
    def test_basic(self):
        T, B, L = 3, 4, 5
        index_dtype = torch.int32
        lengths = torch.randint(1, L, (T, B), dtype=index_dtype)
        indices = torch.randint(1, int(1e5), (lengths.sum().item(),), dtype=index_dtype)
        weights = torch.rand(lengths.sum().item()).float()

        permute_list = list(range(T))
        random.shuffle(permute_list)
        permute = torch.IntTensor(permute_list)

        permuted_lengths_ref, permuted_indices_ref, permuted_weights_ref = (
            permute_indices_ref_(lengths, indices, weights, permute.long())
        )

        permuted_lengths_xpu, permuted_indices_xpu, permuted_weights_xpu = (
            torch.ops.fbgemm.permute_2D_sparse_data(
                permute.to(DEVICE), lengths.to(DEVICE), indices.to(DEVICE),
                weights.to(DEVICE), None,
            )
        )

        torch.testing.assert_close(permuted_lengths_xpu.cpu(), permuted_lengths_ref)
        torch.testing.assert_close(permuted_indices_xpu.cpu(), permuted_indices_ref)
        torch.testing.assert_close(permuted_weights_xpu.cpu(), permuted_weights_ref)

    def test_no_weights(self):
        T, B, L = 3, 4, 5
        index_dtype = torch.int32
        lengths = torch.randint(1, L, (T, B), dtype=index_dtype)
        indices = torch.randint(1, int(1e5), (lengths.sum().item(),), dtype=index_dtype)

        permute_list = list(range(T))
        random.shuffle(permute_list)
        permute = torch.IntTensor(permute_list)

        permuted_lengths_ref, permuted_indices_ref, _ = (
            permute_indices_ref_(lengths, indices, None, permute.long())
        )

        permuted_lengths_xpu, permuted_indices_xpu, permuted_weights_xpu = (
            torch.ops.fbgemm.permute_2D_sparse_data(
                permute.to(DEVICE), lengths.to(DEVICE), indices.to(DEVICE),
                None, None,
            )
        )

        torch.testing.assert_close(permuted_lengths_xpu.cpu(), permuted_lengths_ref)
        torch.testing.assert_close(permuted_indices_xpu.cpu(), permuted_indices_ref)
        self.assertIsNone(permuted_weights_xpu)

    def test_int64(self):
        T, B, L = 4, 3, 6
        index_dtype = torch.int64
        lengths = torch.randint(1, L, (T, B), dtype=index_dtype)
        indices = torch.randint(1, int(1e5), (lengths.sum().item(),), dtype=index_dtype)

        permute_list = list(range(T))
        random.shuffle(permute_list)
        permute = torch.IntTensor(permute_list)

        permuted_lengths_ref, permuted_indices_ref, _ = (
            permute_indices_ref_(lengths, indices, None, permute.long())
        )

        permuted_lengths_xpu, permuted_indices_xpu, permuted_weights_xpu = (
            torch.ops.fbgemm.permute_2D_sparse_data(
                permute.to(DEVICE), lengths.to(DEVICE), indices.to(DEVICE),
                None, None,
            )
        )

        torch.testing.assert_close(permuted_lengths_xpu.cpu(), permuted_lengths_ref)
        torch.testing.assert_close(permuted_indices_xpu.cpu(), permuted_indices_ref)

    def test_with_repeats(self):
        T, B, L = 4, 3, 5
        index_dtype = torch.int32
        lengths = torch.randint(1, L, (T, B), dtype=index_dtype)
        indices = torch.randint(1, int(1e5), (lengths.sum().item(),), dtype=index_dtype)
        weights = torch.rand(lengths.sum().item()).float()

        permute_list = list(range(T))
        for _ in range(random.randint(0, T)):
            permute_list.append(random.randint(0, T - 1))
        random.shuffle(permute_list)
        permute = torch.IntTensor(permute_list)

        permuted_lengths_ref, permuted_indices_ref, permuted_weights_ref = (
            permute_indices_ref_(lengths, indices, weights, permute.long())
        )

        permuted_lengths_xpu, permuted_indices_xpu, permuted_weights_xpu = (
            torch.ops.fbgemm.permute_2D_sparse_data(
                permute.to(DEVICE), lengths.to(DEVICE), indices.to(DEVICE),
                weights.to(DEVICE), None,
            )
        )

        torch.testing.assert_close(permuted_lengths_xpu.cpu(), permuted_lengths_ref)
        torch.testing.assert_close(permuted_indices_xpu.cpu(), permuted_indices_ref)
        torch.testing.assert_close(permuted_weights_xpu.cpu(), permuted_weights_ref)

    def test_exact_values(self):
        """Test from torch-xpu-ops: known input/output pairs."""
        lengths = torch.tensor(
            [[0, 0, 1], [0, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1]],
            dtype=torch.int32, device=DEVICE,
        )
        indices = torch.tensor([500, 1000, 1999], dtype=torch.int32, device=DEVICE)
        permute = torch.tensor([0, 3, 1, 4, 2, 5], dtype=torch.int32, device=DEVICE)
        weights = torch.rand((3, 64), device=DEVICE)

        lengths_out, values_out, weights_out = torch.ops.fbgemm.permute_2D_sparse_data(
            permute, lengths, indices, weights, indices.numel()
        )

        expected_lengths = torch.tensor(
            [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1]],
            dtype=torch.int32, device=DEVICE,
        )
        self.assertTrue(torch.equal(lengths_out, expected_lengths))
        self.assertTrue(torch.equal(values_out, indices))
        self.assertTrue(torch.equal(weights_out, weights))


if __name__ == "__main__":
    run_tests()
