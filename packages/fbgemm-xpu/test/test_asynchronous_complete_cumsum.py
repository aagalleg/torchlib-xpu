"""
Test suite for asynchronous_complete_cumsum operator

Tests complete cumulative sum computation with leading zero.
For 1D input [a, b, c], returns [0, a, a+b, a+b+c].
For 2D input, cumsum along dimension 1 (columns).
"""

import random

import numpy as np
import torch
import unittest

import fbgemm
from torch.testing._internal.common_utils import TestCase, run_tests

SEED = 42


class TestAsynchronousCompleteCumsum(TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(SEED)

    def test_basic_int32(self):
        if not torch.xpu.is_available():
            self.skipTest("XPU not available")
        
        x = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="xpu")
        result = torch.ops.fbgemm.asynchronous_complete_cumsum(x)
        expected = torch.tensor([0, 1, 3, 6, 10], dtype=torch.int32, device="xpu")
        torch.testing.assert_close(result, expected)

    def test_basic_int64(self):
        if not torch.xpu.is_available():
            self.skipTest("XPU not available")
        
        x = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="xpu")
        result = torch.ops.fbgemm.asynchronous_complete_cumsum(x)
        expected = torch.tensor([0, 1, 3, 6, 10], dtype=torch.int64, device="xpu")
        torch.testing.assert_close(result, expected)

    def test_empty(self):
        if not torch.xpu.is_available():
            self.skipTest("XPU not available")
        
        x = torch.tensor([], dtype=torch.int64, device="xpu")
        result = torch.ops.fbgemm.asynchronous_complete_cumsum(x)
        self.assertEqual(result.numel(), 1)
        self.assertEqual(result[0].item(), 0)

    def test_random(self):
        if not torch.xpu.is_available():
            self.skipTest("XPU not available")
        
        for _ in range(5):
            n = random.randint(0, 50)
            for dtype in [torch.int32, torch.int64]:
                x = torch.randint(0, 100, (n,), dtype=dtype, device="xpu")
                result = torch.ops.fbgemm.asynchronous_complete_cumsum(x)
                expected_np = np.cumsum([0] + x.cpu().numpy().tolist())
                expected = torch.from_numpy(expected_np.astype(
                    np.int32 if dtype == torch.int32 else np.int64
                ))
                torch.testing.assert_close(result.cpu(), expected)


if __name__ == "__main__":
    run_tests()
