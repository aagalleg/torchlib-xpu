# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Import torch first to ensure libtorch.so is loaded before our C extension.
# The _C module links against PyTorch libraries, and the dynamic linker needs
# those symbols to be available in memory before loading our extension.
import torch  # noqa: F401

# Import fbgemm_gpu first so that all "fbgemm" operator schemas are registered
# before _C loads the XPU implementations via TORCH_LIBRARY_IMPL.
import fbgemm_gpu  # noqa: F401, E402


# Import the compiled C extension (_C) which contains the registered operators.
# If native dependencies are unavailable, keep import working so metadata like
# __version__ remains accessible.
try:
    from . import _C as _C
except ImportError:
    _C = None

__all__ = ["_C", "__version__"]

try:
    from ._version import __version__
except ModuleNotFoundError:
    try:
        from importlib.metadata import PackageNotFoundError, version
        __version__ = version("fbgemm-xpu")
    except (ImportError, PackageNotFoundError):
        __version__ = "unknown"
