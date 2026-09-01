# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import subprocess
import sys
import textwrap

import pytest
import torch

import fbgemm_xpu


@pytest.mark.parametrize(
    "operator",
    [
        "fbgemm::dense_embedding_codegen_lookup_function",
        "fbgemm::split_embedding_codegen_lookup_rowwise_adagrad_function_pt2",
    ],
)
def test_training_extension_registers_autograd_xpu(operator: str) -> None:
    assert torch._C._dispatch_has_kernel_for_dispatch_key(operator, "AutogradXPU")


def test_native_extensions_load_in_schema_order() -> None:
    loaded_modules = list(sys.modules)

    assert fbgemm_xpu._C is sys.modules["fbgemm_xpu._C"]
    assert fbgemm_xpu._C_training is sys.modules["fbgemm_xpu._C_training"]
    assert loaded_modules.index("fbgemm_xpu._C") < loaded_modules.index(
        "fbgemm_xpu._C_training"
    )


def test_reimport_does_not_reload_native_extensions() -> None:
    core_extension = fbgemm_xpu._C
    training_extension = fbgemm_xpu._C_training

    reloaded_package = importlib.reload(fbgemm_xpu)

    assert reloaded_package._C is core_extension
    assert reloaded_package._C_training is training_extension


@pytest.mark.parametrize(
    "blocked_module",
    ["fbgemm_xpu._C", "fbgemm_xpu._C_training"],
)
def test_missing_native_extension_is_not_silenced(blocked_module: str) -> None:
    script = textwrap.dedent(
        f"""
        import importlib.abc
        import sys

        blocked_module = {blocked_module!r}

        class BlockedExtensionFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == blocked_module:
                    raise ImportError(f"blocked native extension: {{fullname}}")
                return None

        sys.meta_path.insert(0, BlockedExtensionFinder())
        import fbgemm_xpu
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode != 0
    assert f"blocked native extension: {blocked_module}" in completed.stderr
