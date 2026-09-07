# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import fbgemm_xpu  # noqa: F401
import torch

POOLING_MODE_NONE = 2
SPARSE_TYPE_FP32 = 0
SPARSE_TYPE_FP16 = 1
PLACEMENT_DEVICE = 0
INFO_B_NUM_BITS = 26
INFO_B_MASK = (1 << INFO_B_NUM_BITS) - 1


@unittest.skipUnless(torch.xpu.is_available(), "XPU is required")
class XpuLookupOpsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.accelerator.current_accelerator(check_available=True)
        self.assertIsNotNone(self.device)
        self.assertEqual(self.device.type, "xpu")

    def dense_lookup(
        self,
        dev_weights: torch.Tensor,
        indices: list[int],
        output_dtype: int,
    ) -> torch.Tensor:
        dimension = dev_weights.numel() // 2
        indices_tensor = torch.tensor(
            indices, device=self.device, dtype=torch.int64
        )

        return torch.ops.fbgemm.dense_embedding_codegen_lookup_function(
            dev_weights,
            torch.tensor([0], device=self.device, dtype=torch.int64),
            torch.tensor(
                [0, dimension], device=self.device, dtype=torch.int64
            ),
            dimension,
            dimension,
            torch.tensor([0, 2], device=self.device, dtype=torch.int64),
            2,
            indices_tensor,
            torch.tensor(
                [0, indices_tensor.numel()],
                device=self.device,
                dtype=torch.int64,
            ),
            POOLING_MODE_NONE,
            None,
            None,
            output_dtype,
            None,
            None,
            None,
            -1,
            -1,
            -1,
            False,
        )

    def split_lookup(
        self,
        dev_weights: torch.Tensor,
        indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dimension = dev_weights.numel() // 2
        indices_tensor = torch.tensor(
            indices, device=self.device, dtype=torch.int64
        )
        momentum1_dev = torch.zeros(
            2, device=self.device, dtype=torch.float32
        )

        output = torch.ops.fbgemm.split_embedding_codegen_lookup_rowwise_adagrad_function_pt2(
            torch.zeros(
                (),
                device=self.device,
                dtype=torch.float32,
                requires_grad=True,
            ),
            [
                dev_weights,
                torch.empty(0, device=self.device, dtype=dev_weights.dtype),
                torch.tensor(
                    [PLACEMENT_DEVICE],
                    device=self.device,
                    dtype=torch.int32,
                ),
                torch.tensor([0], device=self.device, dtype=torch.int64),
                torch.empty(
                    (0, dimension),
                    device=self.device,
                    dtype=dev_weights.dtype,
                ),
            ],
            torch.tensor(
                [0, dimension], device=self.device, dtype=torch.int64
            ),
            dimension,
            dimension,
            torch.tensor([0, 2], device=self.device, dtype=torch.int64),
            2,
            indices_tensor,
            torch.tensor(
                [0, indices_tensor.numel()],
                device=self.device,
                dtype=torch.int64,
            ),
            POOLING_MODE_NONE,
            None,
            None,
            SPARSE_TYPE_FP32,
            [
                None,
                None,
                None,
                torch.empty(0, device=self.device, dtype=torch.int32),
                torch.empty(0, device=self.device, dtype=torch.int32),
                None,
                None,
            ],
            [0, INFO_B_NUM_BITS, INFO_B_MASK],
            [0.0, 0.0],
            [False, False, True, False, False, False, False],
            [
                momentum1_dev,
                torch.empty(0, device=self.device, dtype=torch.float32),
                torch.tensor(
                    [PLACEMENT_DEVICE],
                    device=self.device,
                    dtype=torch.int32,
                ),
                torch.tensor([0], device=self.device, dtype=torch.int64),
            ],
            torch.tensor(0.5, dtype=torch.float32),
            [0],
            [0.0, 0.0, 0.0],
            -1,
            -1,
            -1,
            None,
        )
        return output, momentum1_dev

    def test_dense_lookup_forward(self) -> None:
        dev_weights = torch.tensor(
            [0, 1, 2, 3, 10, 11, 12, 13],
            device=self.device,
            dtype=torch.float32,
        )

        output = self.dense_lookup(
            dev_weights,
            indices=[1],
            output_dtype=SPARSE_TYPE_FP16,
        )

        self.assertEqual(output.shape, (1, 4))
        self.assertEqual(output.dtype, torch.float16)
        torch.testing.assert_close(
            output.cpu(),
            torch.tensor([[10, 11, 12, 13]], dtype=torch.float16),
        )

    def test_dense_lookup_backward(self) -> None:
        dev_weights = torch.tensor(
            [0, 1, 2, 3, 10, 11, 12, 13],
            device=self.device,
            dtype=torch.float32,
            requires_grad=True,
        )

        output = self.dense_lookup(
            dev_weights,
            indices=[1],
            output_dtype=SPARSE_TYPE_FP32,
        )
        output.sum().backward()

        gradient = dev_weights.grad
        if gradient is None:
            self.fail("Dense lookup backward did not produce a weight gradient")
        torch.testing.assert_close(
            gradient.cpu(),
            torch.tensor(
                [0, 0, 0, 0, 1, 1, 1, 1],
                dtype=torch.float32,
            ),
        )

    def test_split_rowwise_adagrad_forward_and_update(self) -> None:
        dev_weights = torch.tensor(
            [1, 2, 3, 4, 10, 11, 12, 13],
            device=self.device,
            dtype=torch.float32,
        )

        output, momentum1 = self.split_lookup(dev_weights, indices=[0])

        self.assertEqual(output.shape, (1, 4))
        self.assertEqual(output.dtype, torch.float32)
        torch.testing.assert_close(
            output.cpu(),
            torch.tensor([[1, 2, 3, 4]], dtype=torch.float32),
        )

        output.sum().backward()

        torch.testing.assert_close(
            dev_weights.cpu(),
            torch.tensor(
                [0.5, 1.5, 2.5, 3.5, 10, 11, 12, 13],
                dtype=torch.float32,
            ),
        )
        torch.testing.assert_close(
            momentum1.cpu(),
            torch.tensor([1, 0], dtype=torch.float32),
        )

    def test_dense_general_forward_and_cta_backward_with_fp16_weights(
        self,
    ) -> None:
        dimension = 36
        row = torch.arange(
            1,
            dimension + 1,
            device=self.device,
            dtype=torch.float16,
        )
        dev_weights = torch.cat((torch.zeros_like(row), row)).requires_grad_()

        output = self.dense_lookup(
            dev_weights,
            indices=[1] * 32,
            output_dtype=SPARSE_TYPE_FP32,
        )

        self.assertEqual(output.shape, (32, dimension))
        self.assertEqual(output.dtype, torch.float32)
        torch.testing.assert_close(
            output.cpu(),
            row.float().cpu().repeat(32, 1),
            rtol=0,
            atol=0,
        )

        output.sum().backward()

        gradient = dev_weights.grad
        if gradient is None:
            self.fail("Dense CTA backward did not produce a weight gradient")
        torch.testing.assert_close(
            gradient.cpu(),
            torch.cat(
                (
                    torch.zeros(dimension, dtype=torch.float16),
                    torch.full((dimension,), 32, dtype=torch.float16),
                )
            ),
            rtol=0,
            atol=0,
        )

    def test_split_general_forward_and_cta_update_with_fp16_weights(
        self,
    ) -> None:
        dimension = 36
        row = torch.arange(
            1,
            dimension + 1,
            device=self.device,
            dtype=torch.float16,
        )
        dev_weights = torch.cat((torch.zeros_like(row), row))

        output, momentum1 = self.split_lookup(
            dev_weights,
            indices=[1] * 32,
        )

        self.assertEqual(output.shape, (32, dimension))
        self.assertEqual(output.dtype, torch.float32)
        torch.testing.assert_close(
            output.cpu(),
            row.float().cpu().repeat(32, 1),
            rtol=0,
            atol=0,
        )

        output.sum().backward()

        torch.testing.assert_close(
            dev_weights.cpu(),
            torch.cat(
                (
                    torch.zeros(dimension, dtype=torch.float16),
                    torch.arange(
                        0.5,
                        dimension,
                        dtype=torch.float16,
                    ),
                )
            ),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            momentum1.cpu(),
            torch.tensor([0, 1024], dtype=torch.float32),
            rtol=0,
            atol=0,
        )
