# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
#
# Portions of this file are derived from FBGEMM
# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause

# Main package initialization for the fbgemm module
# This imports the C extension and Python operator wrappers
import torch
from pathlib import Path

# Import the compiled C extension (_C) which contains the registered operators
# Import ops module which provides Python wrapper functions with autograd support
from . import _C, ops

try:
    from ._version import __version__
except ModuleNotFoundError:
    try:
        from importlib.metadata import PackageNotFoundError, version
        __version__ = version("fbgemm-xpu")
    except (ImportError, PackageNotFoundError):
        __version__ = "unknown"
