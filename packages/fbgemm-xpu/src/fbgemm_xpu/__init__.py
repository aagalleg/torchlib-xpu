# Copyright 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
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
