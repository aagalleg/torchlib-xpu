cmake_minimum_required(VERSION 3.18)
project(${SKBUILD_PROJECT_NAME})

set(PYBIND11_FINDPYTHON ON)
set(Python3_FIND_VIRTUALENV FIRST)
if(DEFINED ENV{VIRTUAL_ENV})
    set(Python3_ROOT_DIR "$ENV{VIRTUAL_ENV}")
    set(Python3_EXECUTABLE "$ENV{VIRTUAL_ENV}/bin/python")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

add_subdirectory(src/fbgemm_xpu)
add_subdirectory(src/codegen)
