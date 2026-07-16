# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Test suite for the jagged tensor operators

Covers dense_to_jagged, dense_to_jagged_forward, jagged_to_padded_dense,
jagged_to_padded_dense_forward, jagged_dense_elementwise_add_jagged_output and
jagged_2d_to_dense.

Ported from custom_operator_xpu/test/test_fbgemm_ops.py
"""

import itertools
import random
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
import fbgemm_gpu  # noqa: F401  Provides schemas and fake/meta implementations
import fbgemm_xpu  # noqa: F401
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.optests import opcheck

DEVICE = "xpu"
SEED = 42


# =============================================================================
# Reference / helper functions
# =============================================================================


def generate_jagged_tensor(
    num_jagged_dim: int,
    outer_dense_size: int,
    inner_dense_size: int,
    dtype: torch.dtype,
    device: torch.device,
    fold_inner_dense: bool = False,
) -> Tuple[torch.Tensor, List[torch.LongTensor], npt.NDArray]:
    max_lengths = np.random.randint(low=1, high=10, size=(num_jagged_dim,))
    x_offsets: List[torch.LongTensor] = []
    num_lengths = outer_dense_size
    for d in range(num_jagged_dim):
        lengths = torch.randint(
            low=0,
            high=max_lengths[d] * 2,
            size=(num_lengths,),
            device=device,
        )
        offset = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        x_offsets.append(offset)
        num_lengths = x_offsets[-1][-1].item()

    x_values = torch.rand(
        x_offsets[-1][-1] * inner_dense_size,
        dtype=dtype,
        device=device,
    )
    if inner_dense_size != 1 or not fold_inner_dense:
        x_values = x_values.reshape(x_offsets[-1][-1].item(), inner_dense_size)

    return x_values, x_offsets, max_lengths


def to_padded_dense(
    values: torch.Tensor,
    offsets: List[torch.LongTensor],
    max_lengths: npt.NDArray,
    padding_value: float = 0,
) -> torch.Tensor:
    outer_dense_size = len(offsets[0]) - 1
    inner_dense_size = 1 if values.ndim == 1 else values.size(-1)
    dense = torch.empty(
        (outer_dense_size,) + tuple(max_lengths) + (inner_dense_size,),
        dtype=values.dtype,
        device=values.device,
    )
    for i in range(outer_dense_size):
        for jagged_coord in itertools.product(
            *(list(range(max_l)) for max_l in max_lengths)
        ):
            cur_offset = i
            is_zero = False
            for d in range(len(max_lengths)):
                begin = offsets[d][cur_offset].item()
                end = offsets[d][cur_offset + 1].item()
                if jagged_coord[d] >= end - begin:
                    is_zero = True
                    break
                cur_offset = begin + jagged_coord[d]
            dense[(i,) + jagged_coord] = (
                padding_value if is_zero else values[cur_offset]
            )
    return dense.squeeze(-1) if values.ndim == 1 else dense


def jagged_2d_to_dense_ref(
    values: torch.Tensor,
    offsets: torch.Tensor,
    max_sequence_length: int,
) -> torch.Tensor:
    B = offsets.numel() - 1
    if values.dim() == 1:
        output = values.new_zeros(B, max_sequence_length)
    else:
        D = values.size(1)
        output = values.new_zeros(B, max_sequence_length, D)
    for i in range(B):
        start = int(offsets[i].item())
        end = int(offsets[i + 1].item())
        length = min(end - start, max_sequence_length)
        if length > 0:
            output[i, :length] = values[start : start + length]
    return output


# =============================================================================
# Tests
# =============================================================================


class TestDenseToJagged(TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(SEED)

    def _test_dense_to_jagged(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        dtype: torch.dtype,
        precompute_total_L: bool,
    ):
        device = torch.device(DEVICE)
        values_2d, offsets, max_lengths = generate_jagged_tensor(
            num_jagged_dim, outer_dense_size, inner_dense_size, dtype, device
        )

        dense = torch.ops.fbgemm.jagged_to_padded_dense(values_2d, offsets, max_lengths)

        if precompute_total_L:
            total_L = values_2d.size(0)
            jagged_values, jagged_offsets = torch.ops.fbgemm.dense_to_jagged(
                dense, offsets, total_L
            )
            jagged_values_f = torch.ops.fbgemm.dense_to_jagged_forward(
                dense, offsets, total_L
            )
        else:
            jagged_values, jagged_offsets = torch.ops.fbgemm.dense_to_jagged(
                dense, offsets
            )
            jagged_values_f = torch.ops.fbgemm.dense_to_jagged_forward(dense, offsets)
        torch.testing.assert_close(jagged_values, jagged_values_f)

        # Round trip: jagged -> dense -> jagged -> dense should match
        dense2 = torch.ops.fbgemm.jagged_to_padded_dense(
            jagged_values, jagged_offsets, max_lengths
        )
        torch.testing.assert_close(dense, dense2)

    def test_1d_float32(self):
        self._test_dense_to_jagged(1, 4, 3, torch.float, True)

    def test_1d_float16(self):
        self._test_dense_to_jagged(1, 4, 3, torch.half, True)

    def test_2d_float32(self):
        self._test_dense_to_jagged(2, 3, 4, torch.float, True)

    def test_no_precompute(self):
        self._test_dense_to_jagged(1, 4, 3, torch.float, False)

    def test_random(self):
        for _ in range(5):
            num_jd = random.randint(1, 3)
            ods = random.randint(1, 5)
            ids = random.randint(1, 5)
            dtype = random.choice([torch.float, torch.half, torch.bfloat16])
            precompute = random.choice([True, False])
            self._test_dense_to_jagged(num_jd, ods, ids, dtype, precompute)


class TestJaggedToPaddedDense(TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(SEED)

    def _test_jagged_to_padded_dense(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        padding_value: float,
        dtype: torch.dtype,
        fold_inner_dense: bool = False,
    ):
        device = torch.device(DEVICE)
        x_values, x_offsets, max_lengths = generate_jagged_tensor(
            num_jagged_dim,
            outer_dense_size,
            inner_dense_size,
            torch.float,
            device,
            fold_inner_dense,
        )

        output_ref = to_padded_dense(
            x_values, x_offsets, max_lengths, padding_value=padding_value
        )
        output = torch.ops.fbgemm.jagged_to_padded_dense(
            x_values, x_offsets, max_lengths, padding_value=padding_value
        )
        output_f = torch.ops.fbgemm.jagged_to_padded_dense_forward(
            x_values, x_offsets, max_lengths, padding_value=padding_value
        )

        torch.testing.assert_close(output, output_ref)
        torch.testing.assert_close(output_f, output_ref)

    def test_1d(self):
        self._test_jagged_to_padded_dense(1, 4, 3, 0.0, torch.float)

    def test_2d(self):
        self._test_jagged_to_padded_dense(2, 3, 2, 0.0, torch.float)

    def test_negative_padding(self):
        self._test_jagged_to_padded_dense(1, 4, 3, -1e-8, torch.float)

    def test_fold_inner_dense(self):
        self._test_jagged_to_padded_dense(
            1, 4, 1, 0.0, torch.float, fold_inner_dense=True
        )

    def test_random(self):
        for _ in range(5):
            num_jd = random.randint(1, 3)
            ods = random.randint(1, 5)
            ids = random.randint(1, 5)
            pv = random.choice([0.0, -1e-8])
            self._test_jagged_to_padded_dense(num_jd, ods, ids, pv, torch.float)


class TestJaggedDenseElementwiseAdd(TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(SEED)

    def _test_add(
        self,
        num_jagged_dim: int,
        outer_dense_size: int,
        inner_dense_size: int,
        dtype: torch.dtype,
    ):
        device = torch.device(DEVICE)
        x_values, x_offsets, max_lengths = generate_jagged_tensor(
            num_jagged_dim, outer_dense_size, inner_dense_size, dtype, device
        )

        x_padded = to_padded_dense(x_values, x_offsets, max_lengths)

        y = to_padded_dense(
            torch.rand(
                (
                    max(outer_dense_size * np.prod(max_lengths), x_values.size(0)),
                    inner_dense_size,
                ),
                dtype=dtype,
                device=device,
            ),
            x_offsets,
            max_lengths,
        )
        output_ref = x_padded + y

        output, output_offsets = (
            torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
                x_values, x_offsets, y
            )
        )
        output_dense = to_padded_dense(output, output_offsets, max_lengths)

        torch.testing.assert_close(output_dense, output_ref)

    def test_1d_float32(self):
        self._test_add(1, 4, 3, torch.float)

    def test_2d_float32(self):
        self._test_add(2, 3, 2, torch.float)

    def test_1d_float16(self):
        self._test_add(1, 4, 8, torch.half)

    def test_random(self):
        for _ in range(5):
            num_jd = random.randint(1, 3)
            ods = random.randint(1, 4)
            ids = random.randint(1, 4)
            dtype = random.choice([torch.float, torch.half, torch.bfloat16])
            self._test_add(num_jd, ods, ids, dtype)


class TestJagged2dToDense(TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(SEED)

    def _test_jagged_2d_to_dense(
        self,
        outer_dense_size: int,
        inner_dense_size: int,
        dtype: torch.dtype,
        fold_inner_dense: bool = False,
    ):
        device = torch.device(DEVICE)
        x_values, x_offsets, max_lengths = generate_jagged_tensor(
            1, outer_dense_size, inner_dense_size, dtype, device, fold_inner_dense
        )
        max_sequence_length = int(max_lengths[0])

        output = torch.ops.fbgemm.jagged_2d_to_dense(
            x_values, x_offsets[0], max_sequence_length
        )
        output_ref = jagged_2d_to_dense_ref(
            x_values, x_offsets[0], max_sequence_length
        )
        torch.testing.assert_close(output, output_ref)

    def test_2d(self):
        self._test_jagged_2d_to_dense(4, 3, torch.float)

    def test_fold_inner_dense(self):
        self._test_jagged_2d_to_dense(4, 1, torch.float, fold_inner_dense=True)

    def test_random(self):
        for _ in range(5):
            ods = random.randint(1, 5)
            ids = random.randint(1, 5)
            dtype = random.choice([torch.float, torch.half, torch.bfloat16])
            self._test_jagged_2d_to_dense(ods, ids, dtype)


class TestJaggedOpsOpCheck(TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(SEED)

    def test_opcheck_dense_to_jagged_forward(self):
        device = torch.device(DEVICE)
        values, offsets, max_lengths = generate_jagged_tensor(
            1, 4, 3, torch.float, device
        )
        dense = torch.ops.fbgemm.jagged_to_padded_dense(values, offsets, max_lengths)
        opcheck(
            torch.ops.fbgemm.dense_to_jagged_forward.default,
            (dense, offsets, values.size(0)),
        )

    def test_opcheck_jagged_to_padded_dense_forward(self):
        device = torch.device(DEVICE)
        values, offsets, max_lengths = generate_jagged_tensor(
            1, 4, 3, torch.float, device
        )
        opcheck(
            torch.ops.fbgemm.jagged_to_padded_dense_forward.default,
            (values, offsets, [int(m) for m in max_lengths], 0.0),
        )

    def test_opcheck_jagged_2d_to_dense(self):
        device = torch.device(DEVICE)
        values, offsets, max_lengths = generate_jagged_tensor(
            1, 4, 3, torch.float, device
        )
        opcheck(
            torch.ops.fbgemm.jagged_2d_to_dense.default,
            (values, offsets[0], int(max_lengths[0])),
        )


if __name__ == "__main__":
    run_tests()
