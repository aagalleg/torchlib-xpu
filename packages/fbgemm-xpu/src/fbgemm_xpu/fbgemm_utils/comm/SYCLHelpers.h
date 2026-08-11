/*
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * 
 * Original Source: https://github.com/intel/torch-xpu-ops
 */

#pragma once

#include <sycl/sycl.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

struct __SYCL_KER_CONFIG_CONVENTION__ {};

template <typename ker_t, int dim>
static inline typename std::enable_if<
    std::is_base_of_v<__SYCL_KER_CONFIG_CONVENTION__, ker_t>,
    void>::type
sycl_kernel_submit(
    ::sycl::range<dim> global_range,
    ::sycl::range<dim> local_range,
    ::sycl::queue q,
    ker_t ker) {
  auto cgf = [&](::sycl::handler& cgh) {
    ker.sycl_ker_config_convention(cgh);
    cgh.parallel_for<ker_t>(
        ::sycl::nd_range<dim>(global_range, local_range), ker);
  };
  q.submit(cgf);
}

template <typename ker_t, int dim>
static inline typename std::enable_if<
    !std::is_base_of_v<__SYCL_KER_CONFIG_CONVENTION__, ker_t>,
    void>::type
sycl_kernel_submit(
    ::sycl::range<dim> global_range,
    ::sycl::range<dim> local_range,
    ::sycl::queue q,
    ker_t ker) {
  auto cgf = [&](::sycl::handler& cgh) {
    cgh.parallel_for<ker_t>(
        ::sycl::nd_range<dim>(global_range, local_range), ker);
  };
  q.submit(cgf);
}

template <auto* kptr, typename... Kargs>
static inline void sycl_kernel_submit(
    int64_t global_range,
    int64_t local_range,
    ::sycl::queue q,
    int slm_sz,
    Kargs... args) {
  sycl::context ctxt = q.get_context();
  auto exe_bndl =
      syclexp::get_kernel_bundle<kptr, sycl::bundle_state::executable>(ctxt);
  sycl::kernel ker = exe_bndl.template ext_oneapi_get_kernel<kptr>();
  if (slm_sz != 0) {
    syclexp::launch_config cfg{
        ::sycl::nd_range<1>(
            ::sycl::range<1>(global_range), ::sycl::range<1>(local_range)),
        syclexp::properties{syclexp::work_group_scratch_size(slm_sz)}};
    syclexp::nd_launch(q, cfg, ker, args...);
  } else {
    syclexp::launch_config cfg{::sycl::nd_range<1>(
        ::sycl::range<1>(global_range), ::sycl::range<1>(local_range))};
    syclexp::nd_launch(q, cfg, ker, args...);
  }
}

template <auto* kptr, int dim, typename... Kargs>
static inline void sycl_kernel_submit(
    ::sycl::range<dim> global_range,
    ::sycl::range<dim> local_range,
    ::sycl::queue q,
    int slm_sz,
    Kargs... args) {
  sycl::context ctxt = q.get_context();
  auto exe_bndl =
      syclexp::get_kernel_bundle<kptr, sycl::bundle_state::executable>(ctxt);
  sycl::kernel ker = exe_bndl.template ext_oneapi_get_kernel<kptr>();
  if (slm_sz != 0) {
    syclexp::launch_config cfg{
        ::sycl::nd_range<dim>(
            ::sycl::range<dim>(global_range), ::sycl::range<dim>(local_range)),
        syclexp::properties{syclexp::work_group_scratch_size(slm_sz)}};
    syclexp::nd_launch(q, cfg, ker, args...);
  } else {
    syclexp::launch_config cfg{::sycl::nd_range<dim>(
        ::sycl::range<dim>(global_range), ::sycl::range<dim>(local_range))};
    syclexp::nd_launch(q, cfg, ker, args...);
  }
}
