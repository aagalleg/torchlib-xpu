/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <cstdint>

inline std::optional<int64_t> get_device_index_from_tensor(
    const at::Tensor& ten) {
  return {ten.device().index()};
}

inline bool torch_tensor_on_sycl_xpu_check(const at::Tensor& ten) {
  return ten.is_xpu();
}

inline std::string torch_tensor_device_name(const at::Tensor& ten) {
  return c10::DeviceTypeName(ten.device().type());
}

inline bool torch_tensor_undefined(const at::Tensor& ten) {
  return ten.defined();
}

// Generate constexpr array of variable names to improve diagnostic output and
// raise a message if any non-empty tensor is not on a XPU or not on the same
// XPU as all the other non-empty tensors.
#define TENSORS_ON_SAME_SYCL_XPU_IF_NOT_OPTIONAL(...)                        \
  do {                                                                       \
    const auto tensors_on_same_xpu =                                         \
        tensor_on_same_xpu_if_not_optional_check(#__VA_ARGS__, __VA_ARGS__); \
    TORCH_CHECK(tensors_on_same_xpu.empty(), tensors_on_same_xpu);           \
  } while (false)


template <typename... Tensors>
std::string tensor_on_same_xpu_if_not_optional_check(
    const std::string& var_names_str,
    const Tensors&... tensors) {
  std::optional<int64_t> xpu_index;
  bool on_same_xpu = true;

  // Collect the GPU index of the first non-empty optional tensor and make sure
  // that all tensors are on this same index.
  (
      [&](const auto& tensor) {
        if (!torch_tensor_undefined(tensor)) {
          return;
        }
        if (!torch_tensor_on_sycl_xpu_check(tensor)) {
          on_same_xpu = false;
          return;
        }
        const auto my_xpu_index = get_device_index_from_tensor(tensor);
        if (my_xpu_index) {
          if (!xpu_index) {
            xpu_index = my_xpu_index;
          } else if (*xpu_index != my_xpu_index) {
            on_same_xpu = false;
          }
        }
      }(tensors),
      ...);

  if (on_same_xpu) {
    return "";
  }

  std::vector<std::string> var_names;
  {
    std::string temp;
    for (const auto& x : var_names_str) {
      if (x == ',') {
        var_names.push_back(temp);
        temp = "";
      } else {
        temp.push_back(x);
      }
    }
    var_names.push_back(temp);
  }

  // Not all the tensors on a XPU or on the same XPU, generate a message.
  std::string msg = "Not all tensors were on the same XPU: ";
  size_t current_idx = 0;
  (
      [&](const auto& tensor) {
        if (current_idx > 0) {
          msg.append(", ");
        }
        msg.append(
            var_names.at(current_idx++) + "(" +
            torch_tensor_device_name(tensor));
        const auto xpu_device_index = get_device_index_from_tensor(tensor);
        if (xpu_device_index) {
          msg.append(":" + std::to_string(*xpu_device_index));
        }
        msg.append(")");
      }(tensors),
      ...);

  return msg;
}