# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Test suite for reorder_batched_ad_lengths and reorder_batched_ad_indices operators

Ported from test_fbgemm_ops.py
"""


import fbgemm_xpu  # noqa: F401
import torch
from torch.testing._internal.common_utils import TestCase, run_tests

DEVICE = "xpu"


class TestReorderBatched(TestCase):
    def test_reorder_ad_lengths_no_broadcast(self):
        B, T, L, A = 3, 2, 5, 2
        cat_ad_lengths = (
            torch.cat(
                [torch.tensor([L for _ in range(T * A)]) for _ in range(B)], 0
            ).to(DEVICE).int()
        )
        batch_offsets = torch.tensor([A * b for b in range(B + 1)]).int().to(DEVICE)
        num_ads_in_batch = B * A

        result = torch.ops.fbgemm.reorder_batched_ad_lengths(
            cat_ad_lengths, batch_offsets, num_ads_in_batch, False
        )
        torch.testing.assert_close(cat_ad_lengths, result)

    def test_reorder_ad_lengths_broadcast(self):
        B, T, L, A = 3, 2, 5, 2
        cat_ad_lengths = (
            torch.cat(
                [torch.tensor([L for _ in range(T)]) for _ in range(B)], 0
            ).to(DEVICE).int()
        )
        cat_ad_lengths_broadcasted = cat_ad_lengths.tile([A])
        batch_offsets = torch.tensor([A * b for b in range(B + 1)]).int().to(DEVICE)
        num_ads_in_batch = B * A

        result = torch.ops.fbgemm.reorder_batched_ad_lengths(
            cat_ad_lengths, batch_offsets, num_ads_in_batch, True
        )
        torch.testing.assert_close(cat_ad_lengths_broadcasted, result)

    def test_reorder_ad_indices(self):
        B, T, L, A = 3, 2, 4, 2
        cat_ad_indices = torch.randint(0, 100, (B * T * A * L,)).int().to(DEVICE)
        cat_ad_lengths = (
            torch.cat(
                [torch.tensor([L for _ in range(T * A)]) for _ in range(B)], 0
            ).int().to(DEVICE)
        )
        batch_offsets = torch.tensor([A * b for b in range(B + 1)]).int().to(DEVICE)
        num_ads_in_batch = B * A

        reordered_cat_ad_lengths = torch.ops.fbgemm.reorder_batched_ad_lengths(
            cat_ad_lengths, batch_offsets, num_ads_in_batch, False
        )
        cat_ad_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            cat_ad_lengths
        ).to(torch.int64)
        reordered_cat_ad_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            reordered_cat_ad_lengths
        ).to(torch.int64)
        reordered_cat_ad_indices = torch.ops.fbgemm.reorder_batched_ad_indices(
            cat_ad_offsets, cat_ad_indices, reordered_cat_ad_offsets,
            batch_offsets, num_ads_in_batch, False, B * T * A * L,
        )
        torch.testing.assert_close(
            reordered_cat_ad_indices.view(T, B, A, L).permute(1, 0, 2, 3),
            cat_ad_indices.view(B, T, A, L),
        )

    def test_reorder_ad_indices_broadcast(self):
        B, T, L, A = 3, 2, 4, 2
        cat_ad_indices = torch.randint(0, 100, (B * T * L,)).int().to(DEVICE)
        cat_ad_lengths = (
            torch.cat(
                [torch.tensor([L for _ in range(T)]) for _ in range(B)], 0
            ).int().to(DEVICE)
        )

        batch_offsets = torch.tensor([A * b for b in range(B + 1)]).int().to(DEVICE)
        num_ads_in_batch = B * A

        reordered_cat_ad_lengths = torch.ops.fbgemm.reorder_batched_ad_lengths(
            cat_ad_lengths, batch_offsets, num_ads_in_batch, True
        )
        cat_ad_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            cat_ad_lengths
        ).to(torch.int64)
        reordered_cat_ad_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            reordered_cat_ad_lengths
        ).to(torch.int64)
        reordered_cat_ad_indices = torch.ops.fbgemm.reorder_batched_ad_indices(
            cat_ad_offsets, cat_ad_indices, reordered_cat_ad_offsets,
            batch_offsets, num_ads_in_batch, True, B * T * A * L,
        )
        torch.testing.assert_close(
            reordered_cat_ad_indices.view(T, B, A, L).permute(1, 0, 2, 3),
            cat_ad_indices.view(B, T, 1, L).tile([1, 1, A, 1]),
        )


if __name__ == "__main__":
    run_tests()
