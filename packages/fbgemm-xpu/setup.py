# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

import torch

from setuptools import find_packages, setup

from torch.utils.cpp_extension import BuildExtension, CppExtension, SyclExtension

# Package name for the extension - this becomes the importable module name
library_name = value = os.getenv("EXTENSION_NAME", "fbgemm_xpu") 
print(f"Building extension with library name: {library_name}")
# Determine if we can use Python's stable ABI for wheel compatibility
# py_limited_api allows wheels to work across different Python 3.x versions
# Only available in PyTorch 2.6.0+
if torch.__version__ >= "2.6.0":
    py_limited_api = True
else:
    py_limited_api = False


def _normalize_xpu_arch_list() -> str:
    arch_list = os.getenv("TORCH_XPU_ARCH_LIST", "").strip()
    if not arch_list:
        arch_list = "pvc"
        os.environ["TORCH_XPU_ARCH_LIST"] = arch_list
        print("TORCH_XPU_ARCH_LIST not set; defaulting to pvc")
    return arch_list


def get_extensions():
    """
    Configure and return the list of extensions to build
    
    This function determines whether to compile the primary SYCL extension
    or the CPU fallback extension based on system capabilities, then
    configures the appropriate compilation settings.
    
    Returns:
        List of Extension objects to be built by setuptools
    """
    debug_mode = os.getenv("DEBUG", "0") == "1"
    use_sycl = os.getenv("USE_SYCL", "1") == "1"  # Default enabled
    
    if debug_mode:
        print("Compiling in debug mode")

    # Only enable the SYCL backend if both requested and available.
    use_sycl = use_sycl and torch.xpu.is_available()
    
    if use_sycl:
        extension = SyclExtension
    else:
        extension = CppExtension

    # Link-time optimization and debugging flags
    extra_link_args = []
    
    xpu_arch_list = _normalize_xpu_arch_list() if use_sycl else ""

    extra_compile_args = {
        "cxx": [
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
            "-DPy_LIMITED_API=0x03090000",
            "-Wno-c++11-narrowing",
        ],
        "sycl": [
            "-O3" if not debug_mode else "-O0",
            "-Xs", f"-device {xpu_arch_list}",
            "-fdiagnostics-color=always",
            "-DPy_LIMITED_API=0x03090000",
            "-Wno-c++11-narrowing",
        ],
    }

    if debug_mode:
        extra_compile_args["cxx"].extend([
            "-g",
            "-ggdb",
            "-fno-inline",
            "-fno-inline-functions",
        ])
        extra_compile_args["sycl"].extend([
            "-g", "-ggdb",
            "-fno-inline",
            "-fno-inline-functions",
            "-gline-tables-only",
            "-fdebug-info-for-profiling",
            "-fno-sycl-early-optimizations",
            "-fsycl-device-code-split=per_kernel",
        ])
        extra_link_args.extend([
            "-O0",
            "-g", "-ggdb",
            "-fno-inline",
            "-fno-inline-functions",
            "-fdiagnostics-color=always"
        ])

    project_root = Path(__file__).resolve().parent
    extensions_dir = project_root / "src" / library_name

    # Vendored torch-xpu-ops comm/ headers live under the local SYCL tree.
    vendored_sycl_dir = extensions_dir / "sycl_kernels"
    vendored_comm_dir = vendored_sycl_dir / "comm"
    if use_sycl and vendored_comm_dir.is_dir():
        extra_compile_args["sycl"].append(f"-I{vendored_sycl_dir}")
        extra_compile_args["cxx"].append(f"-I{vendored_sycl_dir}")
        print(f"Using vendored comm headers from: {vendored_comm_dir}")
    elif use_sycl:
        print(
            f"WARNING: vendored comm headers not found at {vendored_comm_dir}.\n"
            f"Builds for vendored FBGEMM XPU operators will fail until they are present."
        )
    
    # Find all C++ source files in the main csrc directory
    sources = [str(p.relative_to(project_root)) for p in extensions_dir.glob("*.cpp")]

    # Sanity check: ensure we found source files
    if not sources:
        raise RuntimeError(f"No C++ sources found in {extensions_dir}")

    # Copy link args for SYCL-specific linking
    sycl_link_args = extra_link_args.copy()
    
    # Add SYCL source files if SYCL backend is enabled
    if use_sycl:
        sycl_dir = extensions_dir / "sycl_kernels"
        fbgemm_utils_dir = extensions_dir / "fbgemm_utils"
        # Collect all SYCL sources recursively so utility kernels are also linked.
        sycl_sources = [str(p.relative_to(project_root)) for p in sycl_dir.rglob("*.sycl")]
        sycl_sources += [str(p.relative_to(project_root)) for p in sycl_dir.rglob("*.cpp")]
        sycl_sources += [str(p.relative_to(project_root)) for p in fbgemm_utils_dir.rglob("*.sycl")]
        sycl_sources += [str(p.relative_to(project_root)) for p in fbgemm_utils_dir.rglob("*.cpp")]
        sources += sycl_sources
        if not sycl_sources:
            print("WARNING: USE_SYCL=1 but no .sycl files found")

    print("Building extension with sources:")
    for s in sources:
        print("  -", s)

    link_args = sycl_link_args if use_sycl else extra_link_args

    return [
        extension(
            "fbgemm._C",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=link_args,
            py_limited_api=py_limited_api,
        )
    ]

# Main setuptools configuration
setup(
    # Package metadata
    name="fbgemm",                           # Package name for pip install
    version="0.1.0",                            # Version number
    package_dir={"fbgemm": f"src/{library_name}"},  # Map fbgemm to actual directory
    packages=["fbgemm"],
    include_package_data=True,
    package_data={"fbgemm": ["src/**/*.cpp", "src/**/*.h", "src/**/*.sycl"]},
    
    # Extension configuration
    ext_modules=get_extensions(),               # C++/SYCL extensions to build
    
    # Dependencies
    install_requires=[],                        # torch is provided by the target xpuTorch environment
    
    # Documentation
    description="FBGEMM XPU operators for Intel GPUs",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    
    # Build system configuration
    cmdclass={"build_ext": BuildExtension},     # Use PyTorch's build system
    
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if py_limited_api else {},
)
